import numpy as np
# import cv2
# import torch

from anchor_generator import generate_anchors, draw_anchors, convert_pts_to_rect, convert_rect_to_pts
from rotate_roi_pool_test import get_rotated_roi_pooling_pts

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)
YELLOW = (255,255,0)


def get_pts_line_params(P):
    line_params = np.zeros(8, np.float32)
    for i in range(4):
        line_params[i * 2] = P[((i + 1) * 2) % 8] - P[i * 2];
        line_params[i * 2 + 1] = P[((i + 1) * 2) % 8 + 1] - P[i * 2 + 1];
    return line_params


def bilinear_interpolate(bottom_data, height, width, y, x):

    #// deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 or y > height or x < -1.0 or x > width):
        #//empty
        return 0;

    if (y <= 0): y = 0;
    if (x <= 0): x = 0;

    y_low = int(y);
    x_low = int(x);
    y_high = 0;
    x_high = 0;

    if (y_low >= height - 1):
        y_high = y_low = height - 1;
        y = float(y_low);
    else:
        y_high = y_low + 1;

    if (x_low >= width - 1):
        x_high = x_low = width - 1;
        x = float(x_low);
    else:
        x_high = x_low + 1;

    ly = y - y_low;
    lx = x - x_low;
    hy = 1. - ly;
    hx = 1. - lx;
    # // do bilinear interpolation
    v1 = bottom_data[y_low * width + x_low];
    v2 = bottom_data[y_low * width + x_high];
    v3 = bottom_data[y_high * width + x_low];
    v4 = bottom_data[y_high * width + x_high];
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx;

    val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;


class RRoiAlignCpu(object):
    def __init__(self, pool_size, spatial_scale=1.0, sampling_ratio=0.0):
        self.spatial_scale = spatial_scale
        self.pooled_height, self.pooled_width = pool_size
        self.sampling_ratio = sampling_ratio

        self.bottom_data = None
        self.bottom_rois = None
        self.top_data = None


    def get_pooling_bi_interp(self, P, roi_h, roi_w, offset_bottom_data, height, width):

        sampling_ratio = self.sampling_ratio

        line_params = get_pts_line_params(P)
        # print(P, line_params)

        roi_bin_grid_h = int(np.ceil(roi_h / self.pooled_height)) if sampling_ratio <= 0 else sampling_ratio
        roi_bin_grid_w = int(np.ceil(roi_w / self.pooled_width)) if sampling_ratio <= 0 else sampling_ratio
        mw = 1.0 / roi_bin_grid_w
        mh = 1.0 / roi_bin_grid_h

        count = roi_bin_grid_h * roi_bin_grid_w

        output_val = 0.0
        for iy in range(roi_bin_grid_h):
            for ix in range(roi_bin_grid_w):
                w = P[0] + (iy + 0.5) * line_params[0] * mh + (ix + 0.5) * line_params[2] * mw
                h = P[1] + (iy + 0.5) * line_params[1] * mh + (ix + 0.5) * line_params[3] * mw

                output_val += bilinear_interpolate(offset_bottom_data, height, width, h, w)

        return output_val / count

    def forward(self, image, rois):
        print("FORWARD")

        pooled_height, pooled_width = self.pooled_height, self.pooled_width
        spatial_scale = self.spatial_scale

        B, C, H, W = image.shape
        N = len(rois)

        top_data = np.zeros((N, C, pooled_height, pooled_width), dtype=image.dtype)

        pooling_pts = get_rotated_roi_pooling_pts(rois, (pooled_height, pooled_width), spatial_scale=spatial_scale)

        for n in range(N):
            roi = rois[n]  # batch_ind, xc, yc, w, h, angle

            batch_ind = int(roi[0])
            roi_w = roi[3] * spatial_scale
            roi_h = roi[4] * spatial_scale

            for c in range(C):
                input_cn = image[batch_ind, c]
                offset_bottom_data = input_cn.flatten()

                for ph in range(pooled_height):
                    for pw in range(pooled_width):

                        P = pooling_pts[n, ph, pw]
                        # P = self.get_dex_pts(roi_pts, ph, pw, pooled_height, pooled_width, spatial_scale)

                        val = self.get_pooling_bi_interp(P, roi_h, roi_w, offset_bottom_data, H, W)

                        top_data[n, c, ph, pw] = val

        self.top_data = top_data.copy()
        self.bottom_data = image.copy()
        self.bottom_rois = rois.copy()

        return top_data

def vis_scaled_rois(rois_in, pool_dims, spatial_scale, scale=100):
    # rr = rois[-1,1:]
    N = len(rois_in)
    PH, PW = pool_dims

    # rois = rois_in.copy()
    # rois[:, :-1] *= scale
    pooling_pts = get_rotated_roi_pooling_pts(rois_in, pool_dims, spatial_scale=spatial_scale)
    pooling_pts *= scale
    pooling_pts_flat = pooling_pts.reshape((N*PH*PW, 8))
    img = np.zeros((H*scale,W*scale,3), dtype=np.uint8)
    img[::scale] = WHITE
    img[:, ::scale] = WHITE
    img = draw_anchors(img, pooling_pts_flat, [RED])

    mean_x = np.mean(pooling_pts_flat[:, ::2], axis=-1)
    mean_y = np.mean(pooling_pts_flat[:, 1::2], axis=-1)
    mid_pts = np.vstack((mean_x, mean_y)).T
    mid_pts = np.round(mid_pts).astype(np.int32)

    sampling_ratio = 0

    img2 = img.copy()
    for n in range(N):
        mpts = mid_pts[n*4:(n+1)*4]  # 4 pts [x,y,...]
        roi = rois_in[n][1:]  # xc yc w h angle
        roi_w, roi_h = roi[2:4] * spatial_scale

        for ix,mpt in enumerate(mpts):
            mpt = tuple(mpt)
            cv2.circle(img, mpt, 2, GREEN)
            cv2.putText(img, "%d"%(ix), tuple(mpt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE)

        for ph in range(PH):
            for pw in range(PW):
                P = pooling_pts[n,ph,pw]
                line_params = get_pts_line_params(P)
                roi_bin_grid_h = int(np.ceil(roi_h / PH)) if sampling_ratio <= 0 else sampling_ratio
                roi_bin_grid_w = int(np.ceil(roi_w / PW)) if sampling_ratio <= 0 else sampling_ratio

                mw = 1.0 / roi_bin_grid_w
                mh = 1.0 / roi_bin_grid_h
                for iy in range(roi_bin_grid_h):
                    for ix in range(roi_bin_grid_w):
                        x = P[0] + (iy + 0.5) * line_params[0] * mh + (ix + 0.5) * line_params[2] * mw
                        y = P[1] + (iy + 0.5) * line_params[1] * mh + (ix + 0.5) * line_params[3] * mw
                        cv2.circle(img2, (int(round(x)), int(round(y))), 3, GREEN, -1)

    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)


def sum_error(x1,x2):
    return np.abs(x1 - x2).sum()

if __name__ == '__main__':
    import cv2
    import time
    # import torch

    B = 1
    C = 1
    H = 5
    W = 5
    PH = 2
    PW = 2

    # ROIs are in original image coordinates
    rois = np.array([
        # [0, 2.5, 2.5, 2, 2, -90],  # xc,yc,w,h,angle
        [0, 1.5, 1.5, 2, 1, -90],  # xc,yc,w,h,angle
        # [W / 2, H / 2, W / 2, H / 5, -45],  # xc,yc,w,h,angle
    ], dtype=np.float32)
    # batch_inds = rois[:,0].copy()
    # rois = rois[:,1:]
    # rois = np.array([convert_pts_to_rect(convert_rect_to_pts(roi)) for roi in rois], dtype=np.float32)
    # rois = np.concatenate((batch_inds[:,None], rois), axis=1)

    # feature map size
    spatial_scale = 1.0
    pool_dims = (PH,PW) # ph, pw

    image = np.arange(B*C*H*W, dtype=np.float32).reshape((B,C,H,W))
    # image = np.random.random(size=(B, C, H, W)).astype(np.float32)

    rroi_align_cpu = RRoiAlignCpu(pool_dims, spatial_scale)
    out = rroi_align_cpu.forward(image, rois)
    print(out)
    # image_grad = rroi_align_cpu.backward()
    # print(image_grad)

    vis_scaled_rois(rois, pool_dims, spatial_scale, scale=400//H)
