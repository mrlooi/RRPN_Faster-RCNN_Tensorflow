import numpy as np
import cv2
import torch

from anchor_generator import generate_anchors, draw_anchors, convert_pts_to_rect, convert_rect_to_pts

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)
YELLOW = (255,255,0)

def rotate_roi_pool_cpu(input, rois, pool_dims, spatial_scale=1.0):
    height, width, channels = input.shape#[:2]

    # pooling size
    pooled_width, pooled_height = pool_dims

    for bottom_rois in rois:
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

        # canvas = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        input2 = draw_anchors(input, [[cx,cy,w,h,angle_deg]], [BLUE])
        cv2.imshow("color", input2)

        # for each channel, split the roi into the pooling dimensions, then get the max in each pool
        for cn in range(channels):
            input_cn = input[:,:,cn]
            canvas = input2.copy()

            for pw in range(pooled_width):
                for ph in range(pooled_height):
                    pool_index = ph * pooled_width + pw

                    # compute the rotated rectangle of the pooling region (4 rectangle points, x y x y format)
                    P = np.zeros(8, dtype=np.float32)
                    P[0] = M[0][0] * pw + M[0][1] * ph + M[0][2]
                    P[1] = M[1][0] * pw + M[1][1] * ph + M[1][2]
                    P[2] = M[0][0] * pw + M[0][1] * (ph + 1) + M[0][2]
                    P[3] = M[1][0] * pw + M[1][1] * (ph + 1) + M[1][2]
                    P[4] = M[0][0] * (pw + 1) + M[0][1] * ph + M[0][2]
                    P[5] = M[1][0] * (pw + 1) + M[1][1] * ph + M[1][2]
                    P[6] = M[0][0] * (pw + 1) + M[0][1] * (ph + 1) + M[0][2]
                    P[7] = M[1][0] * (pw + 1) + M[1][1] * (ph + 1) + M[1][2]

                    # draw the rotated rectangle
                    canvas = draw_anchors(canvas, [convert_pts_to_rect(P)], [RED], line_sz=1)

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
                            if ABAB > ABAP and ABAP >= 0 and ACAC > ACAP and ACAP >= 0:
                                bottom_index = hh * width + ww
                                # canvas[hh,ww] = [255,0,0]
                                val = input_cn[hh,ww]
                                if val > maxval:
                                    maxval = val
                                    maxidx = bottom_index

                    if maxidx >= 0:
                        hh = maxidx // width
                        ww = maxidx % width
                        cv2.circle(canvas, (ww,hh), 1, GREEN, -1)
                        # canvas[hh,ww] = [0,255,0]

                    cv2.imshow("roi", canvas)
                    cv2.waitKey(0)


if __name__ == '__main__':
    # ROIs are in original image coordinates
    rois = np.array([
        [100,100,100,50,60],  # xc,yc,w,h,angle
        [100, 100, 50, 100, 60],  # xc,yc,w,h,angle
        [100, 100, 120, 30, -45],  # xc,yc,w,h,angle
        [100, 100, 100, 50, -90],  # xc,yc,w,h,angle
    ], dtype=np.float32)
    N = len(rois)
    batch_inds = np.zeros((N, 1), dtype=np.float32)
    rois = np.array([convert_pts_to_rect(convert_rect_to_pts(roi)) for roi in rois], dtype=np.float32)
    print(rois)
    rois = np.concatenate((batch_inds, rois), axis=1)

    imageWidth, imageHeight = (200,200)

    canvas = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
    canvas[::10,:,0] = np.linspace(0, 255, imageWidth)#.reshape((imageWidth, 3))
    canvas[3::10,:,1] = np.linspace(0, 255, imageWidth)#.reshape((imageWidth, 3))
    canvas[6::10,:,2] = np.linspace(0, 255, imageWidth)#.reshape((imageWidth, 3))

    # feature map size
    spatial_scale = 1.0# / 2
    image = cv2.resize(canvas, None, fx=spatial_scale, fy=spatial_scale).astype(np.float32) / 255

    pool_dims = (2, 2)
    # rotate_roi_pool_cpu(image, rois, pool_dims, spatial_scale=spatial_scale)

    image_tensor = torch.tensor([np.transpose(image, [2,0,1])], device='cuda', requires_grad=True)
    rois_tensor = torch.tensor(rois, device='cuda')
    
    from layers.rotate_roi_pool import RROIPool

    width = int(imageWidth * spatial_scale)
    height = int(imageHeight * spatial_scale)

    r_pooler = RROIPool(pool_dims, spatial_scale)
    out, argmax = r_pooler(image_tensor, rois_tensor)
    argmax = argmax.cpu().numpy()
    N, C, PH, PW = argmax.shape
    for n in range(N):
        roi = rois[n][1:]
        canvas_copy = draw_anchors(canvas, [roi], [BLUE]) # canvas.copy()
        canvas_copy = cv2.resize(canvas_copy, None, fx=spatial_scale, fy=spatial_scale)
        for c in range(C):
            canvas_copy2 = canvas_copy.copy()
            arg_idx = argmax[n,c].flatten()
            idx = n * C * width * height + c * width * height
            print(arg_idx)
            for a in arg_idx:
                hh = (a - idx) // width # int(round((a // width) / spatial_scale))
                ww = (a - idx) % width # int(round((a % width) / spatial_scale))
                cv2.circle(canvas_copy2, (ww,hh), 1, GREEN, -1)
            cv2.imshow("canvas", canvas_copy2)
            cv2.waitKey(0)

    # # out.sum().backward()

    # # g = image_tensor.grad.cpu().numpy().squeeze()
    # # rows,cols = np.where(g>0)
