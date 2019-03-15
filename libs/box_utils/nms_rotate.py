# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from libs.configs import cfgs
import tensorflow as tf
if cfgs.ROTATE_NMS_USE_GPU:
    from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size,
               use_angle_condition=False, angle_threshold=0, use_gpu=True, gpu_id=0):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """

    if use_gpu:
        keep = nms_rotate_gpu(boxes_list=decode_boxes,
                              scores=scores,
                              iou_threshold=iou_threshold,  # also nms thresh
                              angle_gap_threshold=angle_threshold,
                              use_angle_condition=use_angle_condition,
                              device_id=gpu_id)

        keep = tf.cond(
            tf.greater(tf.shape(keep)[0], max_output_size),
            true_fn=lambda: tf.slice(keep, [0], [max_output_size]),
            false_fn=lambda: keep)

    else:
        keep = tf.py_func(nms_rotate_cpu,
                          inp=[decode_boxes, scores, iou_threshold, max_output_size],
                          Tout=tf.int64)
    return keep


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def nms_rotate_gpu(boxes_list, scores, iou_threshold, use_angle_condition=False, angle_gap_threshold=0, device_id=0):
    if use_angle_condition:
        x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
        boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,  # boxes_list sorted in the .pyx file
                          inp=[det_tensor, iou_threshold, device_id],
                          Tout=tf.int64)
        return keep
    else:
        x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
        boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,
                          inp=[det_tensor, iou_threshold, device_id],
                          Tout=tf.int64)
        keep = tf.reshape(keep, [-1])
        return keep


if __name__ == '__main__':
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)

    from make_rotate_anchors_numpy import draw_anchors, convert_anchor_to_rect, get_bounding_box, bb_intersection_over_union, draw_bounding_boxes

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

                iou = bb_intersection_over_union(b1,b2)
                if iou >= iou_thresh:
                    remv.append(c)

        return np.array(keep, dtype=np.uint64)

    dets = np.array([[50, 50, 100, 100, 0, 0.99],  # xc,yc,w,h,theta (degrees),score
                      [60, 60, 100, 100, 0, 0.88],
                      [50, 50, 100, 90, 0., 0.66],
                      [50, 50, 100, 100, -45., 0.65],
                      [50, 50, 100, 80, -45., 0.55],
                      [150, 150, 200, 30, -45., 0.5],
                      [160, 155, 200, 30, -45., 0.46],
                      [150, 150, 200, 30, 0., 0.45],
                      [170, 170, 200, 30, -45., 0.44],
                      [170, 170, 160, 40, 45., 0.435],
                      [170, 170, 140, 40, 45., 0.434],
                    #   [170, 170, 150, 42, 45., 0.433],
                      [170, 170, 200, 30, 45., 0.43],
                      [200, 200, 100, 100, 0., 0.42]], dtype=np.float32)


    boxes = dets[:,:-1]
    scores = dets[:,-1]
    rects = np.array([convert_anchor_to_rect(b) for b in boxes])
    bounding_boxes = np.array([get_bounding_box(r) for r in rects])

    # dets = np.hstack((boxes, scores))

    iou_thresh = 0.7
    device_id = 0

    s_keep = standard_nms_cpu(bounding_boxes, iou_thresh)

    keep = rotate_gpu_nms(dets, iou_thresh, device_id)

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # keep = nms_rotate(tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32),
    #                   iou_thresh, 5)
    # with tf.Session() as sess:
    #     keep = sess.run(keep)

    out_boxes = boxes[keep]

    img = np.zeros((400,400,3), dtype=np.uint8)
    img1 = draw_anchors(img, boxes, RED)
    img2 = draw_anchors(img, out_boxes, GREEN)
    cv2.imshow("pre rotate NMS", img1)
    cv2.imshow("post rotate NMS", img2)

    img3 = draw_bounding_boxes(img1, bounding_boxes, BLUE)
    img4 = draw_anchors(img, boxes[s_keep], GREEN)
    cv2.imshow("pre NMS", img3)
    cv2.imshow("post NMS", img4)
    cv2.waitKey(0)
