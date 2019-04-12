import copy
import os
import os.path as osp

import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from modeling.network import DetectionNetwork
from data_loader import DataLoader
# from utils import FT, LT

from anchor_generator import draw_anchors


BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

RESIZE_SHAPE = 256

def train(model, data_loader, batch_size=4):
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter()

    import torch.optim as optim
    from collections import defaultdict

    model.train()
    model.cuda()

    data_loader.train()

    n_iters = 1000
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.999))

    all_losses_dict = defaultdict(list)
    losses_dict = defaultdict(list)

    start_time = time.time()
    for iter in range(1,n_iters+1):
        data = data_loader.next_batch(batch_size)
        img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=RESIZE_SHAPE, use_cuda=True)

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
            print("iter %d of %d -> %s (time taken: %.2f s)"%(iter, n_iters, out_msg, time.time() - start_time))
            start_time = time.time()

    out_msg = ""
    for k, vv in all_losses_dict.items():
        v = np.mean(vv)
        out_msg += "%s) %.3f, "%(k, v)
    print("END: iter %d of %d -> %s"%(iter, n_iters, out_msg))

    # print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    # writer.close()

def custom_model_inference(model, img_tensor):
    x = img_tensor

    features = model.backbone(x)
    rpn_pred, rpn_losses = model.rpn(x, features)
    detector_pred = [[] for p in rpn_pred]
    if model.roi_heads:
        proposals = [pp[:, :5] for pp in rpn_pred]  # last dim (score) not needed
        x, detector_pred, _ = model.roi_heads(features, proposals)
        detector_pred = [p.detach().cpu().numpy() for p in detector_pred]

    # result, _ = model.forward(x)
    rpn_pred = [p.detach().cpu().numpy() for p in rpn_pred]
    return rpn_pred, detector_pred, {}

def visualize_box_preds(img, box_preds, min_score=0.9, color=RED):
    score = box_preds[:, -1]
    box_pred = box_preds[:, :-1]

    valids = score > min_score
    if np.sum(valids) > 0:
        score = score[valids]
        valid_box_pred = box_pred[valids]

        sorted_idx = np.argsort(score)[::-1]
        # sorted_score = score[sorted_idx]
        sorted_pred = valid_box_pred[sorted_idx]

        # H,W = img.shape[:2]
        # canvas = np.zeros(img.shape, dtype=np.uint8)
        canvas = draw_anchors(img, sorted_pred, color)
    else:
        canvas = img.copy()

    return canvas

def test(model, data_loader, batch_sz=8, min_score=0.95, use_cuda=False):
    model.eval()
    if use_cuda:
        model.cuda()

    data_loader.eval()

    data = data_loader.next_batch(batch_sz)

    img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=RESIZE_SHAPE, use_cuda=use_cuda)
    # box_preds, _ = model(img_tensor) # custom_model_inference(model, img_tensor)
    rpn_preds, box_preds, _ = custom_model_inference(model, img_tensor)

    # min_score = 0.95

    for ix, img_t in enumerate(img_tensor):
        img = img_t.detach().cpu().numpy()
        img = np.transpose(img, [1,2,0])

        cv2.imshow("img", img)

        rpn_pred = rpn_preds[ix]  # (N, 6) -> [xc,yc,w,h,angle,score]
        if len(rpn_pred) > 0:
            img_rpn_pred = visualize_box_preds(img, rpn_pred, min_score, BLUE)
            cv2.imshow("rpn pred > %.3f"%(min_score), img_rpn_pred)

        box_pred = box_preds[ix]  # (N, 6) -> [xc,yc,w,h,angle,score]
        if len(box_pred) > 0:
            img_box_pred = visualize_box_preds(img, box_pred, min_score, RED)
            cv2.imshow("box pred > %.3f"%(min_score), img_box_pred)

        cv2.waitKey(0)

if __name__ == "__main__":
    img_size = RESIZE_SHAPE
    min_objects=1
    max_objects=1
    fill = True

    RPN_ONLY = False
    save_path = "model_rpn_0.pth"
    train_batch_sz = 16
    test_batch_sz = 32

    if not RPN_ONLY:
        save_path = "model_detector_0.pth"
        train_batch_sz = 8
        test_batch_sz = 24

    data_loader = DataLoader(img_size, min_objects, max_objects, fill=fill)
    # data = data_loader.next_batch(4)
    # data_loader.visualize(data)
    # img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=128)
    # img_t = [np.transpose(im, [1,2,0]) for im in img_tensor.numpy()]
    # data_loader.visualize([img_t, all_rects_resized])

    import config as cfg
    cfg.RPN_ONLY = RPN_ONLY
    model = DetectionNetwork(cfg)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("Loaded %s"%(save_path))

    # train(model, data_loader, train_batch_sz)
    # torch.save(model.state_dict(), save_path)

    test(model, data_loader, batch_sz=test_batch_sz, use_cuda=True, min_score=0.95)

