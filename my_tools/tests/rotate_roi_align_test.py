import numpy as np
# import cv2
# import torch

from anchor_generator import generate_anchors, draw_anchors, convert_pts_to_rect, convert_rect_to_pts

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)
YELLOW = (255,255,0)

def convert_region_to_pts(roi):
    cx = roi[0]
    cy = roi[1]
    w = roi[2]
    h = roi[3]
    angle = np.deg2rad(roi[4])

    b = np.cos(angle)*0.5
    a = np.sin(angle)*0.5

    pts = np.zeros(8, dtype=np.float32)
    pts[0] = cx - a*h - b*w
    pts[1] = cy + b*h - a*w
    pts[2] = cx + a*h - b*w
    pts[3] = cy - b*h - a*w
    pts[4] = 2*cx - pts[0]
    pts[5] = 2*cy - pts[1]
    pts[6] = 2*cx - pts[2]
    pts[7] = 2*cy - pts[3]
    return pts


def get_rotated_roi_pooling_pts(rois, pool_dims, spatial_scale=1.0):
    pooled_height, pooled_width = pool_dims

    N = len(rois)
    pooling_pts = np.zeros((N,pooled_height,pooled_width,8), dtype=np.float32)

    for n, bottom_rois in enumerate(rois):
        roi_batch_ind = bottom_rois[0]

        # resize ROIs to spatial scale
        cx = bottom_rois[1] * spatial_scale
        cy = bottom_rois[2] * spatial_scale
        w = bottom_rois[3] * spatial_scale
        h = bottom_rois[4] * spatial_scale

        angle_deg = bottom_rois[5]
        angle = np.deg2rad(angle_deg)

        # compute values used to partition the ROI into the pooling dimensions
        dx = -pooled_width / 2.0
        dy = -pooled_height / 2.0
        Sx = w / pooled_width
        Sy = h / pooled_height
        Alpha = np.cos(angle)
        Beta = -np.sin(angle)
        Dx = cx
        Dy = cy

        M = np.zeros((2,3), dtype=np.float32)
        M[0][0] = Alpha * Sx
        M[0][1] = Beta * Sy
        M[0][2] = Alpha * Sx * dx + Beta * Sy * dy + Dx
        M[1][0] = -Beta * Sx
        M[1][1] = Alpha * Sy
        M[1][2] = -Beta * Sx * dx + Alpha * Sy * dy + Dy

        for ph in range(pooled_height):
            for pw in range(pooled_width):
                # compute the rotated rectangle of the pooling region (4 rectangle points, x y x y format)
                P = pooling_pts[n,ph,pw]
                P[0] = M[0][0] * pw + M[0][1] * (ph + 1) + M[0][2]
                P[1] = M[1][0] * pw + M[1][1] * (ph + 1) + M[1][2]
                P[2] = M[0][0] * pw + M[0][1] * ph + M[0][2]
                P[3] = M[1][0] * pw + M[1][1] * ph + M[1][2]
                P[4] = M[0][0] * (pw + 1) + M[0][1] * ph + M[0][2]
                P[5] = M[1][0] * (pw + 1) + M[1][1] * ph + M[1][2]
                P[6] = M[0][0] * (pw + 1) + M[0][1] * (ph + 1) + M[0][2]
                P[7] = M[1][0] * (pw + 1) + M[1][1] * (ph + 1) + M[1][2]

    return pooling_pts

class RRoiAlignCpu(object):
    def __init__(self, pool_size, spatial_scale=1.0):
        self.spatial_scale = spatial_scale
        self.pooled_height, self.pooled_width = pool_size

        self.bottom_data = None
        self.bottom_rois = None
        self.top_data = None

    def get_bounds_of_rect_pts(self, P, max_height, max_width):
        leftMost = int(max(min(min(P[0], P[2]), min(P[4], P[6])), 0.0))
        topMost = int(max(min(min(P[1], P[3]), min(P[5], P[7])), 0.0))
        rightMost = int(min(max(max(P[0], P[2]), max(P[4], P[6])) + 1, max_width - 1.0))
        bottomMost = int(min(max(max(P[1], P[3]), max(P[5], P[7])) + 1, max_height - 1.0))

        return [leftMost, topMost, rightMost, bottomMost]

    def forward(self, image, rois):
        print("FORWARD")

        pooled_height, pooled_width = self.pooled_height, self.pooled_width
        spatial_scale = self.spatial_scale

        B, C, H, W = image.shape
        N = len(rois)

        top_data = np.zeros((N, C, pooled_height, pooled_width), dtype=image.dtype)

        pooling_pts = get_rotated_roi_pooling_pts(rois, (pooled_height, pooled_width), spatial_scale=spatial_scale)

        for n in range(N):
            roi = rois[n]  # batch_ind, start_x, start_y, end_x, end_y

            batch_ind = int(roi[0])

            for c in range(C):
                input_cn = image[batch_ind, c]

                for ph in range(pooled_height):
                    for pw in range(pooled_width):

                        P = pooling_pts[n, ph, pw]
                        leftMost, topMost, rightMost, bottomMost = self.get_bounds_of_rect_pts(P, H, W)

                        weights = rotated_rect_pixel_interpolation(P, leftMost, rightMost, topMost, bottomMost)

                        val = weights * input_cn[topMost: bottomMost+1, leftMost: rightMost+1]

                        top_data[n, c, ph, pw] = val.sum()

        self.top_data = top_data.copy()
        self.bottom_data = image.copy()
        self.bottom_rois = rois.copy()

        return top_data

    def backward(self):
        print("BACKWARD")
        assert self.top_data is not None and self.bottom_data is not None and self.bottom_rois is not None

        top_diff = np.ones_like(self.top_data) # assume gradient is just 1.0
        bottom_diff = np.zeros_like(self.bottom_data)

        pooled_height, pooled_width = self.pooled_height, self.pooled_width
        spatial_scale = self.spatial_scale

        B, C, H, W = self.bottom_data.shape
        rois = self.bottom_rois
        N = len(rois)

        pooling_pts = get_rotated_roi_pooling_pts(rois, (pooled_height, pooled_width), spatial_scale=spatial_scale)

        for n in range(N):
            roi = rois[n]  # batch_ind, start_x, start_y, end_x, end_y

            batch_ind = int(roi[0])

            for c in range(C):

                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        top_diff_this_bin = top_diff[n, c, ph, pw]

                        P = pooling_pts[n, ph, pw]
                        # get the bounding box of the rotated rect
                        leftMost, topMost, rightMost, bottomMost = self.get_bounds_of_rect_pts(P, H, W)

                        weights = rotated_rect_pixel_interpolation(P, leftMost, rightMost, topMost, bottomMost)

                        weighted_diff = top_diff_this_bin * weights
                        bottom_diff[batch_ind, c, topMost: bottomMost+1, leftMost: rightMost+1] += weighted_diff

        return bottom_diff

def vis_scaled_rois(rois_in, scale=100):
    # rr = rois[-1,1:]
    rois = rois_in.copy()
    rois[:, :-1] *= scale
    img = np.zeros((H*scale,W*scale,3), dtype=np.uint8)
    img[::scale] = WHITE
    img[:, ::scale] = WHITE
    img = draw_anchors(img, rois, [RED])
    cv2.imshow("img", img)
    cv2.waitKey(0)

def rotated_rect_pixel_interpolation(rect_pts, leftMost, rightMost, topMost, bottomMost):
    rect = convert_pts_to_rect(rect_pts)
    r1 = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    roi_area = rect[2] * rect[3]

    weights = np.zeros((bottomMost - topMost + 1, rightMost - leftMost + 1), dtype=np.float32)

    for hh in range(topMost, bottomMost + 1):
        for ww in range(leftMost, rightMost + 1):
            pixel_rect = np.array([ww+0.5, hh+0.5, 1, 1, 0], dtype=np.float32)  # xc,yc,w,h,angle
            r2 = ((pixel_rect[0], pixel_rect[1]), (pixel_rect[2], pixel_rect[3]), pixel_rect[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                inter_area = cv2.contourArea(order_pts)

                # print("ww,hh: (%d,%d), inter_area: %.3f"%(ww,hh,inter_area))

                weight = inter_area / roi_area
                weights[hh-topMost, ww-leftMost] = weight
    return weights

def convert_rect_to_pts2(roi):
    cx = roi[0]
    cy = roi[1]
    w = roi[2]
    h = roi[3]
    angle = np.deg2rad(roi[4])

    b = np.cos(angle)*0.5
    a = np.sin(angle)*0.5

    pts = np.zeros(8, dtype=np.float32)
    pts[0] = cx - a*h - b*w
    pts[1] = cy + b*h - a*w
    pts[2] = cx + a*h - b*w
    pts[3] = cy - b*h - a*w
    pts[4] = 2*cx - pts[0]
    pts[5] = 2*cy - pts[1]
    pts[6] = 2*cx - pts[2]
    pts[7] = 2*cy - pts[3]

    return pts.reshape((4,2))

if __name__ == '__main__':
    import cv2
    import torch
    from layers.rotate_roi_align import RROIAlign

    N = 1
    C = 1
    H = 8
    W = 8

    # ROIs are in original image coordinates
    rois = np.array([
        # [W / 2, H / 2, W / 2, H / 5, -90],  # xc,yc,w,h,angle
        # [W / 2, H / 2, W / 3, H / 5, -90],  # xc,yc,w,h,angle
        # [W / 2, H / 2, W / 4, H / 10, 0],  # xc,yc,w,h,angle
        # [2, 2, W / 5, H / 6, 30],  # xc,yc,w,h,angle
        [0, 3, 3, 3, 3, 30],  # batch_ind,xc,yc,w,h,angle
        [0, 1.5, 1.5, 2, 4, -30],  # batch_ind,xc,yc,w,h,angle
    ], dtype=np.float32)
    batch_inds = rois[:,0].copy()
    rois = rois[:,1:]
    rois = np.array([convert_pts_to_rect(convert_rect_to_pts(roi)) for roi in rois], dtype=np.float32)
    rois = np.concatenate((batch_inds[:,None], rois), axis=1)

    # feature map size
    spatial_scale = 0.8
    pool_dims = (3,2) # ph, pw

    image = np.arange(N*C*H*W, dtype=np.float32).reshape((N,C,H,W))

    rroi_align_cpu = RRoiAlignCpu(pool_dims, spatial_scale)
    out = rroi_align_cpu.forward(image, rois)
    print(out)
    image_grad = rroi_align_cpu.backward()
    print(image_grad)

    # vis_scaled_rois(rois[:, 1:], scale=400//H)

    pooler = RROIAlign(pool_dims, spatial_scale)
    tx = torch.tensor(image, requires_grad=True, device='cuda')
    trois = torch.tensor(rois, device='cuda')
    out = pooler.forward(tx, trois)
    print(out)

    loss = out.sum()
    loss.backward()
    print(tx.grad)

    # rpp = get_rotated_roi_pooling_pts(rois, (1,1), spatial_scale=spatial_scale)
    # rpp2 = convert_region_to_pts(rois[0, 1:])
    # print(rpp.squeeze())
    # print(rpp2)