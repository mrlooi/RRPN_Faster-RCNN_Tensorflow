import copy
import os
import os.path as osp

import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import DetectionNetwork
from data_loader import DataLoader
# from utils import FT, LT

from anchor_generator import draw_anchors


BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

RESIZE_SHAPE = 224

def train(model, data_loader):
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter()

    import torch.optim as optim
    from collections import defaultdict

    model.train()
    model.cuda()

    data_loader.train()

    n_iters = 1000
    lr = 1e-3
    batch_size = 16

    optimizer = optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.999))

    all_losses_dict = defaultdict(list)
    losses_dict = defaultdict(list)
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
            print("iter %d of %d -> %s"%(iter, n_iters, out_msg))

    out_msg = ""
    for k, vv in all_losses_dict.items():
        v = np.mean(vv)
        out_msg += "%s) %.3f, "%(k, v)
    print("END: iter %d of %d -> %s"%(iter, n_iters, out_msg))

    # print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    # writer.close()

def test(model, data_loader, batch_sz=8, min_score=0.95, use_cuda=False):
    model.eval()
    if use_cuda:
        model.cuda()

    data_loader.eval()

    data = data_loader.next_batch(batch_sz)

    img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=RESIZE_SHAPE, use_cuda=use_cuda)
    box_preds, _ = model.forward(img_tensor)

    # min_score = 0.95

    for ix, img_t in enumerate(img_tensor):
        img = img_t.detach().cpu().numpy()
        img = np.transpose(img, [1,2,0])

        box_pred, score = box_preds[ix]
        box_pred = box_pred.detach().cpu().numpy()
        score = score.detach().cpu().numpy()

        top_k = 3
        best_score_inds = np.argsort(score)[::-1][:top_k]

        cv2.imshow("img", img)
        cv2.imshow("top%d preds" % (top_k), draw_anchors(img, box_pred[best_score_inds], (0, 0, 255)))
        # print(score[best_score_inds])


        valids = score > min_score
        if np.sum(valids) > 0:
            score = score[valids]
            box_pred = box_pred[valids]

            sorted_idx = np.argsort(score)[::-1]
            # sorted_score = score[sorted_idx]
            sorted_pred = box_pred[sorted_idx]

            # H,W = img.shape[:2]
            # canvas = np.zeros(img.shape, dtype=np.uint8)
            canvas = draw_anchors(img, sorted_pred, (0,0,255))
            cv2.imshow("pred > %.3f"%(min_score), canvas)
        cv2.waitKey(0)

if __name__ == "__main__":
    img_size = RESIZE_SHAPE
    min_objects=1
    max_objects=2
    fill = True

    data_loader = DataLoader(img_size, min_objects, max_objects, fill=fill)
    data = data_loader.next_batch(4)
    # data_loader.visualize(data)
    # img_tensor, all_rects_resized = data_loader.convert_data_batch_to_tensor(data, resize_shape=128)
    # img_t = [np.transpose(im, [1,2,0]) for im in img_tensor.numpy()]
    # data_loader.visualize([img_t, all_rects_resized])

    import config as cfg
    model = DetectionNetwork(cfg)

    save_path = "model_rpn_0.pth"
    model.load_state_dict(torch.load(save_path))
    print("Loaded %s"%(save_path))

    # train(model, data_loader)
    # torch.save(model.state_dict(), save_path)

    test(model, data_loader, batch_sz=32, use_cuda=True)

