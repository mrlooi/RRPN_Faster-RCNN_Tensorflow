import numpy as np
import cv2
import torch

from anchor_generator import generate_anchors, draw_anchors, convert_pts_to_rect, convert_rect_to_pts

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
WHITE = (255,255,255)
YELLOW = (255,255,0)

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

    cv2.imshow("image", image)
    cv2.waitKey(0)
    