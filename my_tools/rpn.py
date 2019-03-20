# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from box_coder import BoxCoder
from anchor_generator import make_anchor_generator
from utils import permute_and_flatten
from nms_rotate import RotateNMS

REGRESSION_CN = 5  # 4 for bbox, 5 for rotated bbox (xc,yc,w,h,theta)

class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, in_channels, num_anchors):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * REGRESSION_CN, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # logits = []
        # bbox_reg = []
        # for feature in x:
        #     t = F.relu(self.conv(feature))
        #     logits.append(self.cls_logits(t))
        #     bbox_reg.append(self.bbox_pred(t))
        
        t = F.relu(self.conv(x))
        logits= self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):

        # from .loss import make_rpn_loss_evaluator
        # from .inference import make_rpn_postprocessor

        super(RPNModule, self).__init__()

        self.cfg = cfg

        rpn_box_coder = BoxCoder(weights=cfg.RPN.BOX_REG_WEIGHTS, lib=torch) # weights=(1.0, 1.0, 1.0, 1.0, 1.0))

        self.anchor_generator = make_anchor_generator(cfg)

        self.head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location()[0])

        # raise NotImplementedError

        self.box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        self.box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        # self.loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)


    def forward(self, images, features, targets=None):
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        # if self.cfg.MODEL.RPN_ONLY:
        #     # For end-to-end models, the RPN proposals are an intermediate state
        #     # and don't bother to sort them in decreasing score order. For RPN-only
        #     # models, the proposals are the final output and we return them in
        #     # high-to-low confidence order.
        #     inds = [
        #         box.get_field("objectness").sort(descending=True)[1] for box in boxes
        #     ]
        #     boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.nms_rotate = RotateNMS(nms_threshold=nms_thresh, post_nms_top_n=post_nms_top_n)

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, REGRESSION_CN, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, dtype=torch.int64, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        # image_shapes = [box.size for box in anchors]ra1
        # concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = torch.cat(anchors)
        concat_anchors = concat_anchors.reshape(N, -1, REGRESSION_CN)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, REGRESSION_CN), concat_anchors.view(-1, REGRESSION_CN)
        )

        proposals = proposals.view(N, -1, REGRESSION_CN)

        result = []
        for proposal, score in zip(proposals, objectness):
            # perform NMS
            # proposal: (pre_nms_top_n, REGRESSION_CN), score: (pre_nms_top_n)  
            # ASSUMES proposals are already sorted by score!
            keep = self.nms_rotate(proposal)

            final_proposal = proposal[keep]
            final_score = score[keep]

            result.append([final_proposal, final_score])
        # for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
        #     boxlist = BoxList(proposal, im_shape, mode="xyxy")
        #     boxlist.add_field("objectness", score)
        #     boxlist = boxlist.clip_to_image(remove_empty=False)
        #     boxlist = remove_small_boxes(boxlist, self.min_size)
        #     boxlist = boxlist_nms(
        #         boxlist,
        #         self.nms_thresh,
        #         max_proposals=self.post_nms_top_n,
        #         score_field="objectness",
        #     )
        #     result.append(boxlist)

        return result

    def forward(self, anchors, objectness, box_regression):
        return self.forward_for_single_feature_map(anchors, objectness, box_regression)

def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    # fpn_post_nms_top_n = config.RPN.FPN_POST_NMS_TOP_N_TRAIN
    # if not is_train:
    #     fpn_post_nms_top_n = config.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.RPN.NMS_THRESH
    min_size = config.RPN.MIN_SIZE

    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        # fpn_post_nms_top_n=fpn_post_nms_top_n,
    )
    return box_selector
