import cv2
import numpy as np

def convert_pts_to_rect(pts):
    box = np.int0(pts)
    box = box.reshape([4, 2])
    rect1 = cv2.minAreaRect(box)

    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
    if h > w:
        h, w = w, h
        theta = theta - 90
    return np.array((x, y, w, h, theta), dtype=np.int32)

def convert_rect_to_pts(rect_in):
    x_c, y_c, w, h, theta = rect_in
    rect = ((x_c, y_c), (w, h), theta)
    rect_pts = cv2.boxPoints(rect)
    rect_pts = np.int0(rect_pts)
    return rect_pts


def aa():
    rects = np.array([
        [100, 100, 100, 50, -90],
        [100, 100, 50, 100, 0],
        [100, 100, 50, 100, 90],
        [100, 100, 50, 100, 91],
        [100, 100, 20, 100, -90],
        [100, 100, 70, 50, -30],
    ], dtype=np.float32)

    for r in rects:
        pts = convert_rect_to_pts(r)
        # print(pts)
        print(convert_pts_to_rect(pts))



if __name__ == '__main__':
    import config as cfg
    from anchor_generator import generate_anchors, draw_anchors
    from rotate_ops import rotate_iou, nms_rotate_cpu, iou_rotate_cpu

    # import torch
    # def FCT(x): return torch.cuda.FloatTensor(x)

    # anchor_generator = make_anchor_generator(cfg)
    anchor_sizes = cfg.RPN.ANCHOR_SIZES
    anchor_ratios = [1.3, 2.] # cfg.RPN.ASPECT_RATIOS
    stride = cfg.RPN.ANCHOR_STRIDE[0]
    anchor_angles = cfg.RPN.ANCHOR_ANGLES

    H, W = (224, 224)

    total_anchors = len(anchor_angles) * len(anchor_ratios) * len(anchor_sizes)
    anchors = generate_anchors(anchor_sizes, anchor_ratios, anchor_angles,
                        height=H // stride,
                        width=W // stride,
                        stride=stride)

    rect1 = np.array([W // 2, H // 2, 80, 50, 30], dtype=np.float32)
    rect2 = np.array([W // 2, H // 2, 80, 50, -30], dtype=np.float32)
    rect3 = np.array([W // 2, H // 2, 60, 50, 0], dtype=np.float32)
    gt_anchors = np.array([rect1, rect2, rect3])[2:]

    img = np.zeros((H,W,3), dtype=np.uint8)
    # cv2.imshow("gt", draw_anchors(img, gt_anchors))
    # cv2.waitKey(0)

    iou_matrix = iou_rotate_cpu(gt_anchors, anchors)
    fg_iou_thresh = cfg.RPN.FG_IOU_THRESHOLD
    nms_thresh = cfg.RPN.NMS_THRESH

    gt_ids, a_ids = np.nonzero(iou_matrix > fg_iou_thresh)

    for gt_id, a_id in zip(gt_ids, a_ids):
        gt = gt_anchors[gt_id]
        a = anchors[a_id]
        iou = iou_matrix[gt_id, a_id]
        angle_diff = gt[-1] - a[-1]
        print(gt.astype(np.int32), a.astype(np.int32), angle_diff)
        print(iou)
        # if np.abs(angle_diff) >= 90:
        #     angle_diff = 0
        out = draw_anchors(img, [gt, a])

        gt_mid_pt = tuple(gt[:2])
        out = cv2.putText(out, "%d"%(angle_diff), gt_mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

        a2 = a.copy()
        a2[2:4] = gt[2:4]
        a2[-1] += angle_diff
        cv2.imshow("adjusted", draw_anchors(img, [a2]))
        cv2.imshow("out", out)
        cv2.waitKey(0)
