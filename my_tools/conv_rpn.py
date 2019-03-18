import cv2
import numpy as np
import numpy.random as npr
import os
import os.path as osp
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

def FT(x): return torch.FloatTensor(x)
def LT(x): return torch.LongTensor(x)

def convert_anchor_to_rect_points(anchor):
    x_c, y_c, w, h, theta = anchor
    rect = ((x_c, y_c), (w, h), theta)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)
    return rect

def draw_anchors(img, anchors, color=(0,0,255)):
    """
    img: (H,W,3) np.uint8 array
    anchors: (N,5) np.float32 array, where each row is [xc,yc,w,h,angle]
    """
    img_copy = img.copy()
    for anchor in anchors:
        rect = convert_anchor_to_rect_points(anchor)
        # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        # color = (0,0,255)
        cv2.drawContours(img_copy, [rect], 0, color, 2)
    return img_copy

class DataLoader(object):
    def __init__(self, img_size=256, min_objects=3, max_objects=10):
        self.img_size = img_size
        self.min_objects = min_objects
        self.max_objects = max_objects

        self.min_height = int(self.img_size / 16)
        self.max_height = int(self.img_size / 3)
        self.min_width = int(self.img_size / 16)
        self.max_width = int(self.img_size / 3)
        self.max_area = int(self.img_size * self.img_size / 5)

    def next_batch(self, batch_sz):
        blank_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        H, W = blank_img.shape[:2]

        image_list = []  # [(H*W*3 nd.array), ...]
        all_rects_list = [] # [[(xc,yc,w,h,theta), ...], ...]

        for i in range(batch_sz):
            num_objects = np.random.randint(self.min_objects, self.max_objects + 1)
            # img = blank_img.copy()

            rect_list = []
            for n in range(num_objects):
                w = np.random.randint(self.min_width, self.max_width)
                max_h = min(self.max_area // w, self.max_height)
                h = np.random.randint(self.min_height, max_h) if max_h > self.min_height else max_h
                theta = npr.randint(-90, 90)   # degrees
                rect = np.array([0,0,w,h,theta], dtype=np.float32)  # center at 0,0
                rect_pts = convert_anchor_to_rect_points(rect)  # (4,2) 4 corners (x,y)
                rect_bb_lt = np.min(rect_pts,axis=0)  # left top point of rect's bounding box. Will be negative since rect is centered at 0
                rect_bb_rb = -rect_bb_lt  # right bottom is negative of left top, since rect is symmetric and centered at 0
                x_center, y_center = -rect_bb_lt + 1

                # shift the rect (by some random amount) so that all the rect points are never out of bounds
                x_center = np.random.randint(x_center, W - x_center)
                y_center = np.random.randint(y_center, H - y_center)
                rect = np.array([x_center,y_center,w,h,theta], dtype=np.float32)  # center at 0,0

                rect_list.append(rect)

            img = draw_anchors(blank_img, rect_list)
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

        return t_image_tensor, all_rects_list_resized
    
    def visualize(self, data):
        image_list, all_rects_list = data

        for ix,img in enumerate(image_list):
            rect_list = all_rects_list[ix]

            # img_rects = draw_anchors(img, rect_list)
            img_rects = img
            cv2.imshow("rects", img_rects)
            cv2.waitKey(0)

def train(model, dg):
    import torch.optim as optim

    pass

def test(model, dg):
    pass

if __name__ == "__main__":
    img_size = 256
    min_objects=3
    max_objects=10

    data_loader = DataLoader(img_size, min_objects, max_objects)
    data = data_loader.next_batch(1)
    # data_loader.visualize(data)
    # img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=128)
    # img_t = [np.transpose(im, [1,2,0]) for im in img_tensor.numpy()]
    # data_loader.visualize([img_t, all_rects_resized])
