# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from box_coder import BoxCoder
from anchor_generator import make_anchor_generator
from utils import permute_and_flatten
from rotate_ops import RotateNMS, rotate_iou
from losses import smooth_l1_loss

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

        self.loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)


    def forward(self, images, features, targets=None):
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            # raise NotImplementedError

            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg, loss_rpn_box_angle = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_rpn_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
            "loss_rpn_box_angle": loss_rpn_box_angle
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

            final_proposal = proposal[keep]  # (N,REGRESSION_CN)
            final_score = score[keep].unsqueeze(1)  # (N,1)

            out = torch.cat((final_proposal, final_score), 1)  # (N, REGRESSION_CN + 1)
            result.append(out)
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

    def forward(self, anchors, objectness, box_regression, targets=None):
        results = self.forward_for_single_feature_map(anchors, objectness, box_regression) # [[final_proposal, final_score],...]

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            results = self.add_gt_proposals(results, targets)

        return results

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].device

        # gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        # for proposal, target in zip(proposals, targets):
        for ix, target in enumerate(targets):
            objectness = torch.ones(len(target), 1, device=device) # (N,1)
            gt_box = torch.cat((target, objectness), 1)  # (N, REGRESSION_CN + 1)
            proposals[ix] = torch.cat((proposals[ix], gt_box), 0)
            # proposal[1] = torch.cat((proposal[1], torch.ones(len(gt_box), device=device)))
            # gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        # proposals = [
        #     cat_boxlist((proposal, gt_box))
        #     for proposal, gt_box in zip(proposals, gt_boxes)
        # ]

        return proposals

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


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        # self.copied_fields = []
        # self.generate_labels_func = generate_labels_func
        # self.discard_cases = ['not_visibility', 'between_thresholds']
        self.discard_cases = ['between_thresholds']

    def match_targets_to_anchors(self, anchor, target):#, copied_fields=[]):
        match_quality_matrix = rotate_iou(target, anchor)

        # CUSTOM LOGIC: SET ALL ANCHORS VS TARGETS WITH ROTATION DIFF > angle threshold TO IOU OF 0 (to prevent rotation ambiguity)
        ANGLE_THRESH = 45 # degrees
        angle_diffs = torch.abs(target[:,-1].unsqueeze(-1) - anchor[:, -1])
        match_quality_matrix[angle_diffs >= ANGLE_THRESH] = 0

        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        # target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor

        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        # matched_targets = target[matched_idxs.clamp(min=0)]
        # matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_idxs

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        matched_target_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_idxs = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image#, self.copied_fields
            )

            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = targets_per_image[matched_idxs.clamp(min=0)]

            # matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # # discard anchors that go out of the boundaries of the image
            # if "not_visibility" in self.discard_cases:
            #     labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets, anchors_per_image
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            # matched_target_idxs.append(matched_idxs.clamp(min=0))

        return labels, regression_targets#, matched_target_idxs


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        # anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        # aaa = [t[:,2] - t[:,3] for t in targets]
        # bbb = torch.sum(torch.cat([a[:, 2] < a[:, 3] for a in anchors]))
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        N, A, H, W = objectness.shape
        N, AxC, H, W = box_regression.shape
        objectness = objectness.permute(0,2,3,1).reshape(-1)  # same shape as labels
        regression = box_regression.permute(0,2,3,1).reshape(-1, AxC // A)  # same shape as regression targets (N,5)

        # objectness, box_regression = \
        #         concat_box_prediction_layers(objectness, box_regression)
        # objectness = objectness.squeeze()

        total_pos = sampled_pos_inds.numel()
        total_neg = sampled_neg_inds.numel()
        total_samples = total_pos + total_neg

        pos_regression = regression[sampled_pos_inds]  # (N, 5) -> xc,yc,w,h,theta
        pos_regression_targets = regression_targets[sampled_pos_inds]  # (N, 5) -> xc,yc,w,h,theta
        pos_angles = pos_regression[:, -1]#.clone()
        pos_angles_targets = pos_regression_targets[:, -1]#.clone()
        box_loss = smooth_l1_loss(
            # pos_regression[:,:-1].clone() - pos_regression_targets[:,:-1].clone(),
            pos_regression[:, :-1] - pos_regression_targets[:, :-1],
            beta=1.0 / 9,
            size_average=False,
        )
        #
        # # # for targets where the height and width are roughly similar, there may be ambiguity in angle regression
        # # # e.g. if height and width are equal, angle regression could be -90 or 0 degrees
        # # # we don't want to penalize this
        # #
        # THRESH = 0.12
        # all_matched_targets = torch.cat([t[mt_idxs] for t, mt_idxs in zip(targets, matched_target_idxs)], dim=0)[sampled_pos_inds]
        # target_w_to_h_ratio = torch.div(all_matched_targets[:, 2], all_matched_targets[:, 3])
        # target_w_to_h_ratio_diff = torch.abs(1.0 - target_w_to_h_ratio)
        # y = target_w_to_h_ratio_diff > THRESH
        # n = target_w_to_h_ratio_diff <= THRESH
        # angle_loss_y = torch.abs(torch.sin(pos_angles[y] - pos_angles_targets[y])).mean()
        # an = pos_angles_targets[n]
        #
        # # cond = n < beta
        # # loss = torch.where(pos_angles_targets[y], 0.5 * n ** 2 / beta, n - 0.5 * beta)
        # angle_loss_n = smooth_l1_loss(torch.sin(pos_angles[n] - pos_angles_targets[n])).mean()

        # angle_loss = (torch.sum(y) * angle_loss_y + torch.sum(n) * angle_loss_n) / total_pos
        angle_loss = torch.abs(torch.sin(pos_angles - pos_angles_targets)).mean()
        # angle_loss = smooth_l1_loss(torch.sin(pos_angles - pos_angles_targets))

        box_loss = box_loss / total_pos  # FOR SOME REASON sampled_inds.numel() WAS DEFAULT

        # objectness_weights = torch.ones_like(labels)
        # objectness_weights[sampled_pos_inds] = float(total_pos) / total_samples
        # objectness_weights[sampled_neg_inds] = float(total_neg) / total_samples

        # criterion = nn.BCELoss(reduce=False)
        # entropy_loss = criterion(objectness[sampled_inds].sigmoid(), labels[sampled_inds])
        # objectness_loss = torch.mul(entropy_loss, objectness_weights[sampled_inds]).mean()
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]#, weight=objectness_weights[sampled_inds]
        )

        return objectness_loss, box_loss, angle_loss

# # This function should be overwritten in RetinaNet
# def generate_rpn_labels(matched_targets):
#     matched_idxs = matched_targets.get_field("matched_idxs")
#     labels_per_image = matched_idxs >= 0
#     return labels_per_image

def make_rpn_loss_evaluator(cfg, box_coder):
    from matcher import Matcher
    from sampler import BalancedPositiveNegativeSampler

    matcher = Matcher(
        cfg.RPN.FG_IOU_THRESHOLD,
        cfg.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.RPN.BATCH_SIZE_PER_IMAGE, cfg.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder
        # generate_rpn_labels
    )
    return loss_evaluator


def build_rpn(cfg, in_channels):    
    # rpn = RPNHead(in_channels, num_anchors)
    rpn = RPNModule(cfg, in_channels)
    return rpn
