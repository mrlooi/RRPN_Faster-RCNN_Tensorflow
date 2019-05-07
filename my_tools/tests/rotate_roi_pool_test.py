import numpy as np
import cv2
import torch

from anchor_generator import generate_anchors, draw_anchors, convert_pts_to_rect, convert_rect_to_pts

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)
YELLOW = (255,255,0)

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

def rotate_roi_pool_cpu(image, rois, pool_dims, spatial_scale=1.0):
    """
    image: (H,W,3)  ASSUMES IMAGE IS ALREADY RESIZED TO THE SPATIAL SCALE
    rois: (N, 6) -> batch_ind,xc,yc,w,h,angle_degrees  ORIGINAL IMAGE COORDINATES I.E. NOT RESIZED TO SPATIAL SCALE
    pool_dims: (PH, PW)  pooling height, pooling width
    """

    height, width, channels = image.shape#[:2]

    # pooling size
    pooled_height, pooled_width = pool_dims

    pooling_pts = get_rotated_roi_pooling_pts(rois, pool_dims, spatial_scale=spatial_scale)

    for n, roi in enumerate(rois):
        roi_batch_ind = roi[0]
        scaled_roi = roi.copy()
        scaled_roi[1:5] *= spatial_scale

        # canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image2 = draw_anchors(image, [scaled_roi[1:]], [BLUE])
        cv2.imshow("color", image2)

        canvas = image2.copy()

        P = pooling_pts[n].reshape((pooled_height * pooled_width, 8))

        # draw the pooling pts
        canvas = draw_anchors(canvas, [convert_pts_to_rect(pts) for pts in P], [RED], line_sz=1)

        # for each channel, split the roi into the pooling dimensions, then get the max in each pool
        for cn in range(channels):
            input_cn = image[:,:,cn]

            canvas2 = canvas.copy()

            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    pool_index = ph * pooled_width + pw

                    P = pooling_pts[n,ph,pw]

                    # get the bounding box of the rotated rect
                    leftMost = int(max(round(min(min(P[0], P[2]), min(P[4], P[6]))), 0.0))
                    rightMost = int(min(round(max(max(P[0], P[2]), max(P[4], P[6]))), width - 1.0))
                    topMost = int(max(round(min(min(P[1], P[3]), min(P[5], P[7]))), 0.0))
                    bottomMost = int(min(round(max(max(P[1], P[3]), max(P[5], P[7]))), height - 1.0))

                    # compute key vectors of the rotated rect, used to determine if a point is inside the rotated rect
                    AB = [P[2] - P[0], P[3] - P[1]]
                    ABAB = AB[0] * AB[0] + AB[1] * AB[1]

                    AC = [P[4] - P[0], P[5] - P[1]]
                    ACAC = AC[0] * AC[0] + AC[1] * AC[1]

                    maxval = 0
                    maxidx = -1
                    # bottom_data += (roi_batch_ind * channels + c) * height * width

                    # loop through each pixel in the rect bounding box 
                    for hh in range(topMost, bottomMost + 1):
                        for ww in range(leftMost, rightMost + 1):
                            AP = [ww - P[0], hh - P[1]]
                            ABAP = AB[0] * AP[0] + AB[1] * AP[1]
                            ACAP = AC[0] * AP[0] + AC[1] * AP[1]

                            # if pixel is inside the rotated rect
                            # if ABAB > ABAP and ABAP >= 0 and ACAC > ACAP and ACAP >= 0:
                            if ABAP >= 1e-3 and (ABAB - ABAP) > -1e-3 and ACAP >= 1e-3 and (ACAC - ACAP) > -1e-3:
                                bottom_index = hh * width + ww
                                # canvas2[hh,ww] = [255,0,0]
                                val = input_cn[hh,ww]
                                if val > maxval:
                                    maxval = val
                                    maxidx = bottom_index

                    if maxidx >= 0:
                        hh = maxidx // width
                        ww = maxidx % width
                        cv2.circle(canvas2, (ww,hh), 1, GREEN, -1)

            cv2.imshow("roi max pool", canvas2)
            cv2.waitKey(0)


def gpu_rotate_roi_pool(image, rois, pool_dims, spatial_scale=1.0):
    """
    image: (H,W,3)  ASSUMES IMAGE IS ALREADY RESIZED TO THE SPATIAL SCALE
    rois: (N, 6) -> batch_ind,xc,yc,w,h,angle_degrees  ORIGINAL IMAGE COORDINATES I.E. NOT RESIZED TO SPATIAL SCALE
    pool_dims: (PH, PW)  pooling height, pooling width
    """

    from layers.rotate_roi_pool import RROIPool

    image_tensor = torch.tensor([np.transpose(image, [2,0,1])], device='cuda', requires_grad=True)
    rois_tensor = torch.tensor(rois, device='cuda')

    height, width = image_tensor.shape[2:]

    pooling_pts = get_rotated_roi_pooling_pts(rois, pool_dims, spatial_scale=spatial_scale)

    r_pooler = RROIPool(pool_dims, spatial_scale)

    out, argmax_tensor = r_pooler(image_tensor, rois_tensor)

    # VISUALIZE ARGMAX ACTIVATIONS
    argmax = argmax_tensor.cpu().numpy()
    N, C, PH, PW = argmax.shape
    for n in range(N):
        roi = rois[n]
        scaled_roi = roi.copy()
        scaled_roi[1:5] *= spatial_scale

        canvas_copy = draw_anchors(image, [scaled_roi[1:]], [BLUE]) # canvas.copy()
        # canvas_copy = cv2.resize(canvas_copy, None, fx=spatial_scale, fy=spatial_scale)

        P = pooling_pts[n].reshape((PH*PW,8))

        # draw the rotated rectangle
        canvas_copy = draw_anchors(canvas_copy, [convert_pts_to_rect(pts) for pts in P], [RED], line_sz=1)

        for c in range(C):
            canvas_copy2 = canvas_copy.copy()
            arg_idx = argmax[n,c].flatten()
            # idx = (n * C + c) * height * width
            # print(arg_idx)
            for a in arg_idx:
                hh = a // width # int(round((a // width) / spatial_scale))
                ww = a % width # int(round((a % width) / spatial_scale))
                cv2.circle(canvas_copy2, (ww,hh), 1, GREEN, -1)
            cv2.imshow("canvas", canvas_copy2)
            cv2.waitKey(0)

    # VISUALIZE BACKWARD GRADIENTS
    out.sum().backward()
    gradients = image_tensor.grad.cpu().numpy()
    n_idx,c_idx,row_idx,col_idx = np.where(gradients > 0)

    image_np = image_tensor.detach().cpu().numpy()
    N, C, H, W = image_np.shape
    for n in range(N):
        valid_c_idx = c_idx[n_idx == n]
        valid_row_idx = row_idx[n_idx == n]
        valid_col_idx = col_idx[n_idx == n]
        for c in range(C):
            im = image_np[n,c] #  (H,W)
            rr = valid_row_idx[valid_c_idx==c]
            cc = valid_col_idx[valid_c_idx==c]
            for r,c in zip(rr, cc):
                cv2.circle(im, (c,r), 1, 255, -1)
            cv2.imshow("im", im)
            cv2.waitKey(0)

if __name__ == '__main__':
    # ROIs are in original image coordinates
    rois = np.array([
        [200, 200, 150, 75, 60],  # xc,yc,w,h,angle
        [200, 200, 100, 200, 60],  # xc,yc,w,h,angle
        [200, 200, 240, 60, -45],  # xc,yc,w,h,angle
        [200, 200, 200, 100, -90],  # xc,yc,w,h,angle
    ], dtype=np.float32)
    N = len(rois)
    batch_inds = np.zeros((N, 1), dtype=np.float32)
    rois = np.array([convert_pts_to_rect(convert_rect_to_pts(roi)) for roi in rois], dtype=np.float32)
    print(rois)
    rois = np.concatenate((batch_inds, rois), axis=1)

    imageWidth, imageHeight = (400,400)

    canvas = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
    canvas[::20,:,0] = np.linspace(0, 255, imageWidth)#.reshape((imageWidth, 3))
    canvas[6::20,:,1] = np.linspace(0, 255, imageWidth)#.reshape((imageWidth, 3))
    canvas[12::20,:,2] = np.linspace(0, 255, imageWidth)#.reshape((imageWidth, 3))

    # feature map size
    spatial_scale = 1.0 / 2
    image = cv2.resize(canvas, None, fx=spatial_scale, fy=spatial_scale).astype(np.float32) / 255

    pool_dims = (4, 2) # ph, pw

    rotate_roi_pool_cpu(image, rois, pool_dims, spatial_scale=spatial_scale)
    gpu_rotate_roi_pool(image, rois, pool_dims, spatial_scale=spatial_scale)

