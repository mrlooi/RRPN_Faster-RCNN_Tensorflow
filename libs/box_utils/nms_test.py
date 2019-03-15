import numpy as np
import cv2

from make_rotate_anchors_numpy import convert_anchor_to_rect

def DIVUP(m, n): 
    return m // n + ((m) % (n) > 0)

EPSILON = 1e-8
threadsPerBlock = 4

def fake_kernel(dev_boxes, diff_thresh):
    N = dev_boxes.shape[0]
    col_blocks = DIVUP(N, threadsPerBlock)
    mask_host = np.zeros((N, col_blocks), dtype=np.int64)
    
    for r in range(N):
        for cb in range(col_blocks):
            for tid in range(threadsPerBlock):
                idx = cb * threadsPerBlock + tid
                c = idx
                if c >= N:
                    break
                if c <= r:
                    continue
                diff = np.abs(dev_boxes[r] - dev_boxes[c])
                if diff < diff_thresh:
                    mask_host[r,cb] |= 1 << tid

        # for c in range(N):
        #     if r == c:
        #         continue
        #     diff = np.abs(dev_boxes[r] - dev_boxes[c])
        #     if diff < diff_thresh:
        #         i = c // threadsPerBlock
        #         j = c % threadsPerBlock
        #         mask_host[r,i] |= 1 << j
    return mask_host

def fake_kernel_test():
    N = 10
    col_blocks = DIVUP(N, threadsPerBlock)

    # x = np.arange(N)
    # thresh = 2
    x = np.random.random(size=N)
    x = np.array([0.22,0.3,0.6,0.1,0.4,0.9,0.8,0.99,0.7,0.01])
    thresh = 0.15

    # x = 
    mask_host = fake_kernel(x, diff_thresh=thresh)
    print(mask_host)

    remv = np.zeros(col_blocks, dtype=np.int64)
    keep_out = np.zeros(N, dtype=np.uint64)

    num_to_keep = 0
    for i in range(N):
        nblock = i // threadsPerBlock
        inblock = i % threadsPerBlock

        if not (remv[nblock] & (1 << inblock)):
            keep_out[num_to_keep] = i
            num_to_keep += 1

            p = mask_host[i]
            for j in range(nblock, col_blocks):
                remv[j] |= p[j]

    keep = keep_out[:num_to_keep]
    print(keep)

def compute_rotate_inter(r1, r2):
    rr1 = ((r1[0],r1[1]),(r1[2],r1[3]),r1[4])
    rr2 = ((r2[0],r2[1]),(r2[2],r2[3]),r2[4])

    int_pts = cv2.rotatedRectangleIntersection(rr1, rr2)[1]
    inter = 0.0
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)

        inter = cv2.contourArea(order_pts)
    return inter

def in_rect(pt_x, pt_y, rect_pts):
    # pt_x = pt[0]
    # pt_y = pt[1]
    pts = rect_pts

    ab = np.zeros(2, dtype=np.float32)
    ad = np.zeros(2, dtype=np.float32)
    ap = np.zeros(2, dtype=np.float32)

    ab[0] = pts[2] - pts[0]
    ab[1] = pts[3] - pts[1]

    ad[0] = pts[6] - pts[0]
    ad[1] = pts[7] - pts[1]

    ap[0] = pt_x - pts[0]
    ap[1] = pt_y - pts[1]

    abab = ab[0] * ab[0] + ab[1] * ab[1]
    abap = ab[0] * ap[0] + ab[1] * ap[1]
    adad = ad[0] * ad[0] + ad[1] * ad[1]
    adap = ad[0] * ap[0] + ad[1] * ap[1]

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0

def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0

def area(int_pts, num_of_inter):
    area = 0.0
    for i in range(0, num_of_inter - 2):
        area += np.abs(trangle_area(int_pts[:2], int_pts[2 * i + 2: 2 * i + 4], int_pts[2 * i + 4: 2 * i + 6]))
    return area

def inter2line(pts1, pts2, i, j):

    temp_pts = np.zeros(2, dtype=np.float32)
    
    a = np.zeros(2, dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    c = np.zeros(2, dtype=np.float32)
    d = np.zeros(2, dtype=np.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if (area_abc * area_abd >= 0):
        return False, temp_pts
    
    area_cda = trangle_area(c, d, a) 
    area_cdb = area_cda + area_abc - area_abd

    if (area_cda * area_cdb >= 0):
        return False, temp_pts
    
    t = area_cda / (area_abd - area_abc)      

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy

    return True, temp_pts

def get_inter_pts(pts1, pts2):
    int_pts = np.zeros(16, dtype=np.float32)
    num_of_inter = 0
    
    for i in range(4):
        if (in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if (in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1

    for i in range(4):
        for j in range(4):
            has_pts, temp_pts = inter2line(pts1, pts2, i, j)
            if (has_pts):
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1
    return int_pts, num_of_inter


def reorder_pts(int_pts, num_of_inter):
    o_int_pts = int_pts.copy()
    if (num_of_inter <= 0):
        return o_int_pts

    # center = np.zeros(2, dtype=np.float32)
    
    # for i in range(num_of_inter):
    #     center[0] += int_pts[2 * i]
    #     center[1] += int_pts[2 * i + 1]

    # center[0] /= num_of_inter
    # center[1] /= num_of_inter
    center = np.mean(int_pts.reshape((-1, 2)), axis=0)

    vs = np.zeros(16, dtype=np.float32)
    v = np.zeros(2, dtype=np.float32)
    for i in range(num_of_inter):
        v[0] = int_pts[2 * i] - center[0]
        v[1] = int_pts[2 * i + 1] - center[1]
        d = np.sqrt(v[0] * v[0] + v[1] * v[1])
        v[0] = v[0] / d
        v[1] = v[1] / d
        if(v[1] < 0):
            v[0] = - 2 - v[0]
        vs[i] = v[0]

    for i in range(1, num_of_inter):
        if (vs[i-1] > vs[i]):
            temp = vs[i]
            tx = o_int_pts[2*i]
            ty = o_int_pts[2*i+1]
            j = i
            while (j > 0 and vs[j-1] > temp):
                vs[j] = vs[j-1]
                o_int_pts[j*2] = o_int_pts[j*2-2]
                o_int_pts[j*2+1] = o_int_pts[j*2-1]
                j -= 1
            vs[j] = temp
            o_int_pts[j*2] = tx
            o_int_pts[j*2+1] = ty

    return o_int_pts

def compute_rotate_inter2(r1, r2):
    rect1 = convert_anchor_to_rect(r1)
    rect2 = convert_anchor_to_rect(r2)
    int_pts, num_of_inter = get_inter_pts(rect1.flatten(), rect2.flatten())
    # print(num_of_inter)
    int_pts = reorder_pts(int_pts, num_of_inter)

    return area(int_pts, num_of_inter)
  


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


def compute_rotate_iou(r1, r2):

    area_r1 = r1[2] * r1[3]
    area_r2 = r2[2] * r2[3]

    int_area = compute_rotate_inter(r1, r2)

    iou = int_area * 1.0 / (area_r1 + area_r2 - int_area + EPSILON)
    # print("area1: %.3f, area2: %.3f, area_inter: %.3f, iou: %.3f"%(area_r1, area_r2, int_area, iou))

    return iou

def rotate_nms_cpu2(dets, iou_thresh):
    N = dets.shape[0]

    boxes = dets[:, :5]

    keep = []
    remv = []
    for r in range(N):
        if r in remv:
            continue
        
        keep.append(r)
        b1 = boxes[r]

        for c in range(r + 1, N):
            b2 = boxes[c]

            iou = compute_rotate_iou(b1,b2)
            if iou >= iou_thresh:
                remv.append(c)

    return np.array(keep, dtype=np.uint64)

if __name__ == "__main__":
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    
    from rotate_polygon_nms import rotate_gpu_nms
    from make_rotate_anchors_numpy import draw_anchors, convert_anchor_to_rect, get_bounding_box, bb_intersection_over_union, draw_bounding_boxes

    dets = np.array([
                      [50, 50, 100, 100, 0, 0.99],  # xc,yc,w,h,theta (degrees),score
                      [60, 60, 100, 100, 0, 0.88],
                      [50, 50, 100, 90, 0., 0.66],
                      [50, 50, 100, 100, -45., 0.65],
                      [50, 50, 90, 50, 45., 0.6],
                      [50, 50, 100, 80, -45., 0.55],
                      [150, 150, 200, 30, -45., 0.5],
                      [160, 155, 200, 30, -45., 0.46],
                      [150, 150, 200, 30, 0., 0.451],
                      [170, 170, 200, 30, -45., 0.45],
                      [170, 170, 150, 42, 45., 0.444],
                      [170, 170, 200, 30, 45., 0.443],
                      [200, 200, 100, 100, 0., 0.442],
                      [170, 170, 160, 40, 45., 0.435],
                      [170, 170, 150, 40, 45., 0.434],
                      [170, 170, 150, 40, 45., 0.433],
                      [170, 170, 170, 40, 45., 0.432],
            ], dtype=np.float32)

    boxes = dets[:,:-1]
    scores = dets[:,-1]
    rects = np.array([convert_anchor_to_rect(b) for b in boxes])
    bounding_boxes = np.array([get_bounding_box(r) for r in rects])

    # dets = np.hstack((boxes, scores))

    iou_thresh = 0.65
    device_id = 0

    s_keep = standard_nms_cpu(bounding_boxes, iou_thresh)


    keep = rotate_nms_cpu2(dets, iou_thresh)
    keep = rotate_gpu_nms(dets, iou_thresh, device_id)
    # keep = nms_rotate_cpu(boxes, scores, iou_thresh, len(boxes))

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

    img3 = draw_bounding_boxes(img1, bounding_boxes, BLUE)
    img4 = draw_anchors(img, boxes[s_keep], GREEN)
    cv2.imshow("pre NMS", img3)
    cv2.imshow("post NMS", img4)
    cv2.imshow("pre rotate NMS", img1)
    cv2.imshow("post rotate NMS", img2)

    cv2.waitKey(0)
