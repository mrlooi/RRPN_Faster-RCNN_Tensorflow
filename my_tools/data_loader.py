import torch
import cv2
import numpy as np
import numpy.random as npr

from anchor_generator import convert_pts_to_rect, get_random_color, \
    draw_anchors, make_anchor_generator, generate_anchors, convert_rect_to_pts
from rotate_ops import rotate_iou, nms_rotate_cpu, iou_rotate_cpu


def FT(x): return torch.FloatTensor(x)
def FCT(x): return FT(x).cuda()


class DataLoader(object):
    def __init__(self, img_size=256, min_objects=3, max_objects=10, fill=False):
        self.img_size = img_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.fill = fill

        self.min_height = int(self.img_size / 7)
        self.max_height = int(self.img_size / 3)
        self.min_width = int(self.img_size // 4)
        self.max_width = int(self.img_size // 2.5)
        self.max_area = int(self.img_size * self.img_size / 4)
        self.min_area = int(self.max_area // 3)

        self.is_training = False

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def next_batch(self, batch_sz):
        blank_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        H, W = blank_img.shape[:2]

        image_list = []  # [(H*W*3 nd.array), ...]
        all_rects_list = [] # [[(xc,yc,w,h,theta), ...], ...]

        for i in range(batch_sz):
            num_objects = npr.randint(self.min_objects, self.max_objects + 1)
            # img = blank_img.copy()

            rect_list = []
            color_list = []
            for n in range(num_objects):
                w = npr.randint(self.min_width, self.max_width)
                h = w // (np.random.randint(15,30) / 10.)
                # max_h = min(self.max_area // w, self.max_height)
                # h = npr.randint(self.min_height, max_h) if max_h > self.min_height else max_h
                # h = max(self.min_area // w, h)
                theta = npr.randint(-90, 90)
                # theta = np.random.permutation([-30,0,30])[0] #npr.randint(-90, 90)   # degrees
                rect = np.array([0, 0, w, h, theta], dtype=np.float32)  # center at 0,0

                rect_pts = convert_rect_to_pts(rect)  # (4,2) 4 corners (x,y)
                rect_bb_lt = np.min(rect_pts,axis=0)  # left top point of rect's bounding box. Will be negative since rect is centered at 0,0
                rect_bb_rb = -rect_bb_lt  # right bottom is negative of left top, since rect is symmetric and centered at 0
                x_center, y_center = -rect_bb_lt + 1

                # shift the rect (by some random amount) so that all the rect points are never out of bounds
                x_center = npr.randint(x_center, W - x_center)
                y_center = npr.randint(y_center, H - y_center)
                rect = np.array([x_center,y_center,w,h,theta], dtype=np.float32)

                # # DEBUG ONLY
                # rect = np.array([W // 2, H // 2, 80, 50, theta], dtype=np.float32)
                # color = [0, 255, 0]
                color = get_random_color()
                if self.is_training:
                    # if training, make the rect outputs consistent via minAreaRect
                    rect = np.array(convert_pts_to_rect(convert_rect_to_pts(rect)), dtype=np.float32)

                color_list.append(color)
                rect_list.append(rect)

            img = draw_anchors(blank_img, rect_list, color_list=color_list, fill=self.fill)
            image_list.append(img)

            all_rects_list.append(np.array(rect_list))
            
        return image_list, all_rects_list

    def convert_data_batch_to_tensor(self, data, resize_shape=56, use_cuda=False):
        sz = resize_shape
        image_list, all_rects_list = data
        
        N = len(image_list)
        t_image_list = np.zeros((N, sz, sz, 3), dtype=np.float32)
        
        all_rects_list_resized = []
        for ix, im in enumerate(image_list):
            ori_h, ori_w = im.shape[:2]
            t_im = cv2.resize(im, (sz, sz), interpolation=cv2.INTER_LINEAR)
            rects_list = all_rects_list[ix].copy() # (N,5)

            # shift the centers and rescale the rect
            rects_list[:, 0] *= sz / ori_w # shift center x
            rects_list[:, 1] *= sz / ori_h # shift center y
            rects_list[:, 2] *= sz / ori_w # rescale rect width
            rects_list[:, 3] *= sz / ori_h # rescale rect height

            t_image_list[ix] = t_im.astype(np.float32) / 255  # assumes data is 0-255!

            all_rects_list_resized.append(rects_list)

        t_image_list = np.transpose(t_image_list, [0,3,1,2])  # (N,H,W,3) to (N,3,H,W)
        t_image_tensor = FT(t_image_list)
        if use_cuda:
            t_image_tensor = t_image_tensor.cuda()

        device = t_image_tensor.device
        all_rects_list_resized = [torch.tensor(r, device=device) for r in all_rects_list_resized]

        return t_image_tensor, all_rects_list_resized
    
    def visualize(self, data):
        image_list, all_rects_list = data

        for ix,img in enumerate(image_list):
            # rect_list = all_rects_list[ix]
            # img_rects = draw_anchors(img, rect_list)
            img_rects = img
            cv2.imshow("rects", img_rects)
            cv2.waitKey(0)

def visualize_config_anchors(image, gt_anchors, cfg):

    # anchor_generator = make_anchor_generator(cfg)
    anchor_sizes = cfg.RPN.ANCHOR_SIZES
    anchor_ratios = cfg.RPN.ASPECT_RATIOS
    stride = cfg.RPN.ANCHOR_STRIDE[0]
    anchor_angles = cfg.RPN.ANCHOR_ANGLES
    H, W = image.shape[:2]

    total_anchors = len(anchor_angles) * len(anchor_ratios) * len(anchor_sizes)
    anchors = generate_anchors(anchor_sizes, anchor_ratios, anchor_angles,
                        height=H // stride,
                        width=W // stride,
                        stride=stride)

    iou_matrix = rotate_iou(FCT(gt_anchors), FCT(anchors)).cpu().numpy()
    # sorted_matrix = np.argsort(iou_matrix, axis=1)[:, ::-1]

    fg_iou_thresh = cfg.RPN.FG_IOU_THRESHOLD
    nms_thresh = cfg.RPN.NMS_THRESH

    a,b = np.nonzero(iou_matrix > fg_iou_thresh)
    # gg = iou_matrix > fg_iou_thresh
    # b = np.unique([ix for g in gg for ix,b in enumerate(g) if b!=0])
    # best_anchors = anchors[sorted_matrix[:,0]]
    best_anchors = anchors[b]#[(iou_matrix > 0.8]
    # best_anchors = best_anchors[nms_rotate_cpu(best_anchors, nms_thresh, 1000)]

    img_best_anchors = draw_anchors(image, best_anchors)
    cv2.imshow("img", img)
    cv2.imshow("best_anchors", img_best_anchors)

    batch = total_anchors
    start = 0 # (len(anchors) // total_anchors // 2)
    for ix in np.arange(start, len(anchors), batch):
        stride_anchors = anchors[ix:ix+batch]
        img_stride_anchors = draw_anchors(image, stride_anchors)
        valid_idx = b[np.logical_and(ix <= b, b < ix + batch)]

        # print("Valids: %d"%(len(valid_idx)))
        if len(valid_idx) == 0:
            continue

        valid_anchors = anchors[valid_idx]
        img_valid_stride_anchors = draw_anchors(image, valid_anchors)

        post_nms_anchors = valid_anchors[nms_rotate_cpu(valid_anchors, nms_thresh, 100)]
        img_valid_stride_anchors_post_nms = draw_anchors(image, post_nms_anchors)
        # post_nms_iou_matrix = iou_rotate_cpu(gt_anchors, post_nms_anchors)
        # print(post_nms_iou_matrix)

        cv2.imshow("stride_anchors", img_stride_anchors)
        cv2.imshow("valid_stride_anchors (>%.2f)"%(fg_iou_thresh), img_valid_stride_anchors)
        cv2.imshow("valid_stride_anchors (post NMS)", img_valid_stride_anchors_post_nms)
        cv2.waitKey(0)

    # img_anchors = draw_anchors(image, anchors)
    # cv2.imshow("anchors", img_anchors)
    # cv2.waitKey(0)

if __name__ == '__main__':

    img_size = 256
    min_objects=1
    max_objects=1
    fill = False

    data_loader = DataLoader(img_size, min_objects, max_objects, fill=fill)
    data_loader.train()
    # data = data_loader.next_batch(2)
    # data_loader.visualize(data)
    # img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=128)
    # img_t = [np.transpose(im, [1,2,0]) for im in img_tensor.numpy()]
    # data_loader.visualize([img_t, all_rects_resized])



    import config as cfg
    data = data_loader.next_batch(30)
    images, gt_anchors = data
    for ix, img in enumerate(images):
        visualize_config_anchors(images[ix], gt_anchors[ix], cfg)
