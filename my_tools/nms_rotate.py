# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import _C

def nms_rotate_cpu(boxes, iou_threshold, max_output_size):
    EPSILON = 1e-8

    keep = []

    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for i in range(num):
        if len(keep) >= max_output_size:
            break

        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for j in range(i + 1, num):
            # if suppressed[i] == 1:
            #     continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + EPSILON)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)

class _RotateNMSFunction(Function):
    @staticmethod
    def forward(ctx, r_boxes, nms_threshold, post_nms_top_n):
        # boxes: (N,5)
        N = r_boxes.size(0)
        assert len(r_boxes.shape) == 2 and r_boxes.size(1) == 5

        # keep_inds = torch.zeros(N)
        if r_boxes.is_cuda:
            # keep_inds = keep_inds.type(torch.cuda.IntTensor)
            keep_inds = _C.rotate_nms(r_boxes, nms_threshold, post_nms_top_n)
        else:
            with torch.no_grad():
                keep_inds = nms_rotate_cpu(r_boxes, nms_threshold, post_nms_top_n)
            keep_inds = torch.LongTensor(keep_inds)
            # raise NotImplementedError("Rotate NMS Forward CPU layer not implemented!")

        return keep_inds

class RotateNMS(nn.Module):
    """
    Performs rotated NMS
    DOES NOT PERFORM SORTING ON A 'score/objectness' field. 
    ASSUMES THE INPUT BOXES ARE ALREADY SORTED

    INPUT:
    r_boxes: (N, 5)  [xc,yc,w,h,theta]  # NOTE there's no score field here
    """
    def __init__(self, nms_threshold=0.7, post_nms_top_n=1000):
        super(RotateNMS, self).__init__()

        self.nms_threshold = float(nms_threshold)
        self.post_nms_top_n = int(post_nms_top_n)

    def forward(self, r_boxes):
        """
        r_boxes: (N, 5)  [xc,yc,w,h,theta]  
        """
        return _RotateNMSFunction.apply(r_boxes, self.nms_threshold, self.post_nms_top_n)

    def __repr__(self):
        tmpstr = "%s (nms_thresh=%.2f, post_nms_top_n=%d)"%(self.__class__.__name__,
                    self.nms_threshold, self.post_nms_top_n)
        return tmpstr


def rotate_nms_torch(dets, iou_thresh, device='cpu'):
    nms_rot = RotateNMS(iou_thresh)

    dets_tensor = torch.tensor(dets).to(device)
    keep = nms_rot(dets_tensor)

    return keep.cpu().numpy()


if __name__ == '__main__':
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    from anchor_generator import draw_anchors, convert_anchor_to_rect, get_bounding_box, \
        bb_intersection_over_union, draw_bounding_boxes


    def standard_nms_cpu(dets, iou_thresh):
        N = dets.shape[0]

        # scores = dets[:,-1]
        # sort_idx = np.argsort(scores)[::-1]
        # dets2 = dets[sort_idx]
        # boxes = dets2[:,:-1]
        boxes = dets

        keep = []
        remv = []
        for r in range(N):
            if r in remv:
                continue

            keep.append(r)
            # r1 = convert_anchor_to_rect(boxes[r])
            # b1 = get_bounding_box(r1)
            b1 = boxes[r]

            for c in range(r + 1, N):
                # r2 = convert_anchor_to_rect(boxes[c])
                # b2 = get_bounding_box(r2)
                b2 = boxes[c]

                iou = bb_intersection_over_union(b1, b2)
                if iou >= iou_thresh:
                    remv.append(c)

        return np.array(keep, dtype=np.uint64)


    dets = np.array([
          [50, 50, 100, 100, 0, 0.99],  # xc,yc,w,h,theta (degrees),score
          [60, 60, 100, 100, 0, 0.88],
          [50, 50, 100, 90, 0., 0.66],
          [50, 50, 100, 100, -45., 0.65],
          [50, 50, 90, 50, 45., 0.6],
          [50, 50, 100, 80, -45., 0.55],
          [150, 150, 200, 30, -45., 0.5],
          [160, 155, 200, 30, -45., 0.46],
          [150, 150, 200, 30, 0., 0.45],
          [170, 170, 200, 30, -45., 0.44],
          [170, 170, 160, 40, 45., 0.435],
          [170, 170, 150, 40, 45., 0.434],
          [170, 170, 150, 42, 45., 0.433],
          [170, 170, 200, 30, 45., 0.43],
          [200, 200, 100, 100, 0., 0.42]
    ], dtype=np.float32)

    boxes = dets[:, :-1]
    scores = dets[:, -1]
    rects = np.array([convert_anchor_to_rect(b) for b in boxes])
    bounding_boxes = np.array([get_bounding_box(r) for r in rects])

    # dets = np.hstack((boxes, scores))

    iou_thresh = 0.7
    device_id = 0

    device = 'cuda'
    keep2 = rotate_nms_torch(boxes, iou_thresh, device=device)
    keep = nms_rotate_cpu(boxes, iou_thresh, len(boxes))
    print("CPU keep: ", keep)
    print("GPU keep: ", keep2)
    keep = keep2

    s_keep = standard_nms_cpu(bounding_boxes, iou_thresh)

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # keep = nms_rotate(tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32),
    #                   iou_thresh, 5)
    # with tf.Session() as sess:
    #     keep = sess.run(keep)

    out_boxes = boxes[keep]

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img1 = draw_anchors(img, boxes, RED)
    img2 = draw_anchors(img, out_boxes, GREEN)

    img3 = draw_bounding_boxes(img1, bounding_boxes, BLUE)
    img4 = draw_anchors(img, boxes[s_keep], GREEN)
    cv2.imshow("pre NMS", img3)
    cv2.imshow("post NMS", img4)
    cv2.imshow("pre rotate NMS", img1)
    cv2.imshow("post rotate NMS", img2)

    cv2.waitKey(0)
