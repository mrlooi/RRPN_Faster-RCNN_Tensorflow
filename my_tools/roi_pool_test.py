import numpy as np
import cv2

from anchor_generator import generate_anchors, draw_anchors, convert_pts_to_rect, convert_rect_to_pts

if __name__ == '__main__':
    rois = np.array([
        [100,100,100,50,60],  # xc,yc,w,h,angle
        [100, 100, 50, 100, 60],  # xc,yc,w,h,angle
        [100, 100, 120, 30, -45],  # xc,yc,w,h,angle
    ], dtype=np.float32)
    N = len(rois)
    batch_inds = np.zeros((N, 1))
    rois = np.array([convert_pts_to_rect(convert_rect_to_pts(roi)) for roi in rois], dtype=np.float32)
    print(rois)
    rois = np.concatenate((batch_inds, rois), axis=1)

    spatial_scale = 1.0

    imageWidth, imageHeight = (200,200)
    imageWidth = int(imageWidth * spatial_scale)
    imageHeight = int(imageHeight * spatial_scale)

    pooled_width, pooled_height = (4, 2)

    for bottom_rois in rois:
        roi_batch_ind = bottom_rois[0]
        cx = bottom_rois[1]
        cy = bottom_rois[2]
        w = bottom_rois[3]
        h = bottom_rois[4]
        angle = np.deg2rad(bottom_rois[5])

        dx = -pooled_width / 2.0
        dy = -pooled_height / 2.0
        Sx = w * spatial_scale / pooled_width
        Sy = h * spatial_scale / pooled_height
        Alpha = -np.cos(angle)
        Beta = np.sin(angle)
        Dx = cx * spatial_scale
        Dy = cy * spatial_scale

        M = np.zeros((2,3), dtype=np.float32)
        M[0][0] = Alpha * Sx
        M[0][1] = Beta * Sy
        M[0][2] = Alpha * Sx * dx + Beta * Sy * dy + Dx
        M[1][0] = -Beta * Sx
        M[1][1] = Alpha * Sy
        M[1][2] = -Beta * Sx * dx + Alpha * Sy * dy + Dy

        canvas = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
        canvas = draw_anchors(canvas, [bottom_rois[1:]])
        cv2.imshow("img", canvas)

        for pw in range(pooled_width):
            for ph in range(pooled_height):
                P = np.zeros(8, dtype=np.float32)
                P[0] = M[0][0] * pw + M[0][1] * ph + M[0][2]
                P[1] = M[1][0] * pw + M[1][1] * ph + M[1][2]
                P[2] = M[0][0] * pw + M[0][1] * (ph + 1) + M[0][2]
                P[3] = M[1][0] * pw + M[1][1] * (ph + 1) + M[1][2]
                P[4] = M[0][0] * (pw + 1) + M[0][1] * ph + M[0][2]
                P[5] = M[1][0] * (pw + 1) + M[1][1] * ph + M[1][2]
                P[6] = M[0][0] * (pw + 1) + M[0][1] * (ph + 1) + M[0][2]
                P[7] = M[1][0] * (pw + 1) + M[1][1] * (ph + 1) + M[1][2]

                leftMost = int(max(round(min(min(P[0], P[2]), min(P[4], P[6]))), 0.0))
                rightMost = int(min(round(max(max(P[0], P[2]), max(P[4], P[6]))), imageWidth - 1.0))
                topMost = int(max(round(min(min(P[1], P[3]), min(P[5], P[7]))), 0.0))
                bottomMost = int(min(round(max(max(P[1], P[3]), max(P[5], P[7]))), imageHeight - 1.0))

                canvas = draw_anchors(canvas, [convert_pts_to_rect(P)])

                maxval = 0
                maxidx = -1
                # bottom_data += (roi_batch_ind * channels + c) * height * width

                AB = [0, 0]
                AB[0] = P[2] - P[0]
                AB[1] = P[3] - P[1]

                ABAB = AB[0] * AB[0] + AB[1] * AB[1]

                AC = [0,0]
                AC[0] = P[4] - P[0]
                AC[1] = P[5] - P[1]

                ACAC = AC[0] * AC[0] + AC[1] * AC[1]

                for hh in range(topMost, bottomMost + 1):
                    for ww in range(leftMost, rightMost + 1):
                        pass
                        AP = [w - P[0], h - P[1]]
                        ABAP = AB[0] * AP[0] + AB[1] * AP[1]
                        ACAP = AC[0] * AP[0] + AC[1] * AP[1]
                        # if ABAB > ABAP and ABAP >= 0 and ACAC > ACAP and ACAP >= 0:
                        #     bottom_index = hh * width + ww;

        cv2.imshow("roi", canvas)
        cv2.waitKey(0)
