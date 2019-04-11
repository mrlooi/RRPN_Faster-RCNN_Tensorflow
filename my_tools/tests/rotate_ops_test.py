
import numpy as np
import torch

if __name__ == "__main__":

    import time
    import cv2

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # =============================NMS ROTATE TEST===================================== #
    from anchor_generator import draw_anchors, convert_rect_to_pts, get_bounding_box, \
        bb_intersection_over_union, draw_bounding_boxes
    from rotate_ops import RotateNMS, rotate_iou, nms_rotate_cpu

    def rotate_nms_torch(dets, iou_thresh, device='cpu'):
        nms_rot = RotateNMS(iou_thresh)

        dets_tensor = torch.tensor(dets).to(device)
        keep = nms_rot(dets_tensor)

        return keep.cpu().numpy()


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
    # dets = np.array([
    #     [60, 60, 100, 50, -90, 0.9],
    #     [60, 60, 100, 50, -180, 0.8],
    # ], dtype=np.float32)
    # dets[dets[:, -2] < -45, -2] += 180
    # dets[dets[:, -2] > 135, -2] -= 180

    boxes = dets[:, :-1]
    scores = dets[:, -1]
    rects = np.array([convert_rect_to_pts(b) for b in boxes])
    bounding_boxes = np.array([get_bounding_box(r) for r in rects])

    # dets = np.hstack((boxes, scores))

    iou_thresh = 0.7
    device_id = 0

    device = 'cuda'
    keep = rotate_nms_torch(boxes, iou_thresh, device=device)
    keep2 = nms_rotate_cpu(boxes, iou_thresh, len(boxes))
    print("CPU keep: ", keep2)
    print("GPU keep: ", keep)
    # keep = keep2

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


    # =============================IOU ROTATE TEST===================================== #
    
    def iou_rotate_torch(boxes1, boxes2, use_gpu=False):
    
        t_boxes1 = torch.FloatTensor(boxes1)
        t_boxes2 = torch.FloatTensor(boxes2)
        if use_gpu:
            t_boxes1 = t_boxes1.cuda()
            t_boxes2 = t_boxes2.cuda()
    
        iou_matrix = rotate_iou(t_boxes1, t_boxes2)
        iou_matrix = iou_matrix.cpu().numpy()
    
        return iou_matrix
    
    boxes1 = np.array([
        [50, 50, 100, 300, 0],
        [60, 60, 100, 200, 0],
        [200, 200, 100, 200, 80.]
    ], np.float32)
    
    boxes2 = np.array([
        [50, 50, 100, 300, -45.],
        [50, 50, 100, 300, 0.],
        [50, 50, 50, 150, 0.],
        [200, 200, 100, 200, 0.],
        [200, 200, 100, 200, 90.]
    ], np.float32)
    
    start = time.time()
    ious = iou_rotate_torch(boxes1, boxes2, use_gpu=False)
    print(ious)
    print('time: {}s'.format(time.time() - start))
    
    start = time.time()
    ious = iou_rotate_torch(boxes1, boxes2, use_gpu=True)
    print(ious)
    print('time: {}s'.format(time.time() - start))
