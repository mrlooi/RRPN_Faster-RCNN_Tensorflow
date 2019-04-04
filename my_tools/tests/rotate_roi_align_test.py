import numpy as np
# import cv2
# import torch

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
        Alpha = -np.cos(angle)
        Beta = np.sin(angle)
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
                    P[0] = M[0][0] * pw + M[0][1] * ph + M[0][2]
                    P[1] = M[1][0] * pw + M[1][1] * ph + M[1][2]
                    P[2] = M[0][0] * pw + M[0][1] * (ph + 1) + M[0][2]
                    P[3] = M[1][0] * pw + M[1][1] * (ph + 1) + M[1][2]
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
        # roi_batch_ind = roi[0]

        # for each channel, split the roi into the pooling dimensions, then get the max in each pool
        for cn in range(channels):
            input_cn = image[:,:,cn]

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

                            val = input_cn[hh, ww]

                            # if pixel is inside the rotated rect
                            # if ABAB > ABAP and ABAP >= 0 and ACAC > ACAP and ACAP >= 0:
                            # if ABAP >= 1e-3 and (ABAB - ABAP) > -1e-3 and ACAP >= 1e-3 and (ACAC - ACAP) > -1e-3:
                            if ABAP >= 1e-3 and (ABAB - ABAP) > -0.3 and ACAP >= 1e-3 and (ACAC - ACAP) > -0.3:
                                bottom_index = hh * width + ww

                                print(val)
                                if val > maxval:
                                    maxval = val
                                    maxidx = bottom_index

                    if maxidx >= 0:
                        print("Max", maxval)
                        # hh = maxidx // width
                        # ww = maxidx % width

if __name__ == '__main__':
    H = 3
    W = 3

    # ROIs are in original image coordinates
    rois = np.array([
        [1.5, 1.5, 2, 1, -90],  # xc,yc,w,h,angle
        [1.5, 1.5, 3, 1, -90],  # xc,yc,w,h,angle
        [1.5, 1.5, 3, 1, 0],  # xc,yc,w,h,angle
        [1, 1, 2, 1, 0],  # xc,yc,w,h,angle
        [1, 1, 2, 2, 0],  # xc,yc,w,h,angle
        [1.5, 1.5, 2, 1, -45],  # xc,yc,w,h,angle
    ], dtype=np.float32)
    N = len(rois)
    batch_inds = np.zeros((N, 1), dtype=np.float32)
    rois = np.array([convert_pts_to_rect(convert_rect_to_pts(roi)) for roi in rois], dtype=np.float32)
    print(rois)
    rois = np.concatenate((batch_inds, rois), axis=1)

    # feature map size
    spatial_scale = 1.0

    pool_dims = (1, 1) # ph, pw

    # rpp = get_rotated_roi_pooling_pts(rois, pool_dims, spatial_scale)
    C = 1
    image = np.arange(C*H*W, dtype=np.float32).reshape((H,W,C))
    rotate_roi_pool_cpu(image, rois, pool_dims, spatial_scale)
