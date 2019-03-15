
import numpy as np
import cv2

def enum_scales2(base_anchor, anchor_scales):
    """
    base_anchor: (4)
    anchor_scales: N
    output: (N,4)

    e.g.
    base_anchors = [0,0,256,256]
    anchor_scales = [0.25,0.5,1]
    output = [[0,0,64,64],[0,0,128,128],[0,0,256,256]]
    """
    output = np.array(base_anchor) * np.array(anchor_scales)[:,None]

    return output

def enum_ratios_and_thetas2(anchors, anchor_ratios, anchor_angles):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    a_ws = anchors[:, 2]  # for base anchor: w == h
    a_hs = anchors[:, 3]
    sqrt_ratios = np.sqrt(anchor_ratios)

    ws = np.reshape(a_ws / sqrt_ratios[:,None], -1)  # flatten (len(anchors)*len(anchor_ratios))
    hs = np.reshape(a_hs * sqrt_ratios[:,None], -1)  # flatten

    ws, _ = np.meshgrid(ws, anchor_angles)
    hs, anchor_angles = np.meshgrid(hs, anchor_angles)

    anchor_angles = np.reshape(anchor_angles, [-1, 1])
    ws = np.reshape(ws, [-1, 1])
    hs = np.reshape(hs, [-1, 1])

    return hs, ws, anchor_angles

def make_anchors2(base_anchor_size, anchor_scales, anchor_ratios, anchor_angles,
                 featuremap_height, featuremap_width, stride):
    """
    returns anchors: (N, 5), each element is [x_center,y_center,width,height,angle]
    N = H * W * (len anchor_scales * len anchor_ratios * len anchor_angles)
    """
    base_anchor = np.array([0, 0, base_anchor_size, base_anchor_size], np.float32)  # [y_center, x_center, h, w]
    ws, hs, angles = enum_ratios_and_thetas2(enum_scales2(base_anchor, anchor_scales),
                                            anchor_ratios, anchor_angles)  # per locations ws and hs and thetas

    x_centers = np.arange(featuremap_width, dtype=np.float32) * stride + stride // 2
    y_centers = np.arange(featuremap_height, dtype=np.float32) * stride + stride // 2

    x_centers, y_centers = np.meshgrid(x_centers, y_centers)

    angles, _ = np.meshgrid(angles, x_centers)
    ws, x_centers = np.meshgrid(ws, x_centers)
    hs, y_centers = np.meshgrid(hs, y_centers)

    anchor_centers = np.stack([x_centers, y_centers], 2)
    anchor_centers = np.reshape(anchor_centers, [-1, 2])

    box_parameters = np.stack([ws, hs, angles], axis=2)
    box_parameters = np.reshape(box_parameters, [-1, 3])
    anchors = np.concatenate([anchor_centers, box_parameters], axis=1)

    return anchors


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def convert_anchor_to_rect(anchor):
    x_c, y_c, w, h, theta = anchor
    rect = ((x_c, y_c), (w, h), theta)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)
    return rect

def get_bounding_box(pts):
    """
    pts: (N, 2) array
    """
    bbox = np.zeros(4, dtype=pts.dtype)
    bbox[:2] = np.min(pts, axis=0)
    bbox[2:] = np.max(pts, axis=0)
    return bbox

def draw_anchors(img, anchors, color=(0,0,255)):
    """
    img: (H,W,3) np.uint8 array
    anchors: (N,5) np.float32 array, where each row is [xc,yc,w,h,angle]
    """
    img_copy = img.copy()
    for anchor in anchors:
        rect = convert_anchor_to_rect(anchor)
        # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        # color = (0,0,255)
        cv2.drawContours(img_copy, [rect], 0, color, 2)
    return img_copy

def draw_bounding_boxes(img, bboxes, color=(0,0,255)):
    """
    img: (H,W,3) np.uint8 array
    bboxes: (N,4) np.float32 array, where each row is [x1,y1,x2,y2]
    """
    img_copy = img.copy()
    for bbox in bboxes.astype(np.int32):
        cv2.rectangle(img_copy, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
    return img_copy


if __name__ == '__main__':

    base_anchor_size = 256
    anchor_scales = [1./8, 1./6, 1./4]  # strides 16,8,4
    anchor_ratios = [0.5, 1.0, 2.0]
    anchor_angles = [-90, -45]#, -60, -45, -30, -15]
    # base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
    base_anchor = np.array([0, 0, base_anchor_size, base_anchor_size], np.float32)

    anchors = enum_scales2(base_anchor, anchor_scales)
    tmp1 = enum_ratios_and_thetas2(anchors, anchor_ratios, anchor_angles[:1])
    total_anchors = len(anchor_angles) * len(anchor_ratios) * len(anchor_scales)

    stride = 16
    W = 800
    H = 800
    anchors = make_anchors2(base_anchor_size,
                           anchor_scales, anchor_ratios, anchor_angles,
                           featuremap_height=H // stride,
                           featuremap_width=W // stride,
                           stride=stride)  # (H*W* (len anchor_ratios * len anchor_angles * len anchor_scales))

    # img = np.zeros([H, W, 3], dtype=np.uint8)
    img_file = "/home/bot/hd/datasets/DOTA/val_clip/images/P0003_0223_0347.png"
    # img_file = "/home/bot/Pictures/boxes_cropped_pc.png"
    img = cv2.imread(img_file)
    img = cv2.resize(img, (W,H))

    batch = total_anchors
    start = 0
    for ix in np.arange(start, len(anchors), batch):
        img1 = draw_anchors(img, anchors[ix:ix+batch])
        cv2.imshow("anchors", img1)
        cv2.waitKey(0)
