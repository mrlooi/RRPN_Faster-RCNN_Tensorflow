import copy
import os
import os.path as osp

import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor_generator import convert_anchor_to_rect, draw_anchors
from network import DetectionNetwork
from utils import FT, LT

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

class DataLoader(object):
    def __init__(self, img_size=256, min_objects=3, max_objects=10, fill=False):
        self.img_size = img_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.fill = fill

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
                rect_pts = convert_anchor_to_rect(rect)  # (4,2) 4 corners (x,y)
                rect_bb_lt = np.min(rect_pts,axis=0)  # left top point of rect's bounding box. Will be negative since rect is centered at 0,0
                rect_bb_rb = -rect_bb_lt  # right bottom is negative of left top, since rect is symmetric and centered at 0
                x_center, y_center = -rect_bb_lt + 1

                # shift the rect (by some random amount) so that all the rect points are never out of bounds
                x_center = np.random.randint(x_center, W - x_center)
                y_center = np.random.randint(y_center, H - y_center)
                rect = np.array([x_center,y_center,w,h,theta], dtype=np.float32)  

                rect_list.append(rect)

            img = draw_anchors(blank_img, rect_list, fill=self.fill)
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

def train(model, data_loader):
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter()

    import torch.optim as optim
    from collections import defaultdict

    model.train()
    model.cuda()

    n_iters = 100
    lr = 1e-3
    batch_size = 4

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    all_losses_dict = defaultdict(list)
    losses_dict = defaultdict(list)
    for iter in range(1,n_iters+1):
        data = data_loader.next_batch(batch_size)
        img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=256, use_cuda=True)

        box_pred, loss_dict = model.forward(img_tensor, all_rects_resized)

        losses = sum(loss for loss in loss_dict.values())

        losses_dict["total"].append(losses.item())
        all_losses_dict["total"].append(losses.item())
        for k,vv in loss_dict.items():
            v = vv.item()
            losses_dict[k].append(v)
            all_losses_dict[k].append(v)
            # writer.add_scalar(k, v.item(), iter)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if iter % 20 == 0 and iter > 0:
            out_msg = ""
            for k, vv in losses_dict.items():
                v = np.mean(vv)
                out_msg += "%s) %.3f, "%(k, v)
            losses_dict.clear()
            print("iter %d of %d -> %s"%(iter, n_iters, out_msg))

    out_msg = ""
    for k, vv in all_losses_dict.items():
        v = np.mean(vv)
        out_msg += "%s) %.3f, "%(k, v)
    print("END: iter %d of %d -> %s"%(iter, n_iters, out_msg))

    # print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    # writer.close()

def test(model, data_loader, batch_sz=8, use_cuda=False):
    model.eval()
    if use_cuda:
        model.cuda()
    data = data_loader.next_batch(batch_sz)

    img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=256, use_cuda=use_cuda)
    box_preds, _ = model.forward(img_tensor)

    min_score = 0.5

    for ix, img_t in enumerate(img_tensor):
        img = img_t.detach().cpu().numpy()
        img = np.transpose(img, [1,2,0])

        box_pred, score = box_preds[ix]
        box_pred = box_pred.detach().cpu().numpy()
        score = score.detach().cpu().numpy()

        print(score.max())

        valids = score > min_score
        score = score[valids]
        box_pred = box_pred[valids]

        sorted_idx = np.argsort(score)[::-1]
        sorted_score = score[sorted_idx]
        sorted_pred = box_pred[sorted_idx]

        # H,W = img.shape[:2]
        canvas = np.zeros(img.shape, dtype=np.uint8)
        canvas = draw_anchors(canvas, sorted_pred)#, (0,0,255))

        cv2.imshow("img", img)
        cv2.imshow("pred", canvas)
        cv2.waitKey(0)

if __name__ == "__main__":
    img_size = 256
    min_objects=3
    max_objects=5
    fill = False

    data_loader = DataLoader(img_size, min_objects, max_objects, fill=fill)
    data = data_loader.next_batch(4)
    # data_loader.visualize(data)
    # img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=128)
    # img_t = [np.transpose(im, [1,2,0]) for im in img_tensor.numpy()]
    # data_loader.visualize([img_t, all_rects_resized])

    import config as cfg
    model = DetectionNetwork(cfg)
    # model.cuda()

    train(model, data_loader)
    test(model, data_loader, use_cuda=True)

