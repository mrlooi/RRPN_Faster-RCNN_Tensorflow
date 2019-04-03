import torch
from torch import nn
import torch.nn.functional as F

from pooler import Pooler
from box_coder import BoxCoder
from rotate_ops import RotateNMS, rotate_iou
from losses import smooth_l1_loss
from utils import cat, get_unique_count

REGRESSION_CN = 5  # 4 for bbox, 5 for rotated bbox (xc,yc,w,h,theta)


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, [class_logits, box_regression], result, {}

        loss_classifier, loss_box_reg, loss_box_angle = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            [class_logits, box_regression],
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_box_angle=loss_box_angle),
        )

class ConvROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ConvROIFeatureExtractor, self).__init__()

        resolution = config.ROI_BOX_HEAD.POOLER_RESOLUTION
        # spatial_scale = config.ROI_BOX_HEAD.POOLER_SCALES
        scales = config.ROI_BOX_HEAD.POOLER_SCALES
        # sampling_ratio = config.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales
            # sampling_ratio=sampling_ratio,
        )

        self.pooler = pooler

        c_dims = [128,128]
        self.conv_dim_out = c_dims #,512]
        self.out_channels = self.conv_dim_out[-1]

        inplace = True

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c_dims[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(c_dims[0]),
            nn.ReLU(inplace),
            # nn.Conv2d(64, c_dims[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_dims[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace),
            nn.Conv2d(64, c_dims[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(c_dims[1]),
            nn.ReLU(inplace),
        )

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        x = self.conv1(x)
        x = self.conv2(x)

        return x

class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.ROI_BOX_HEAD.NUM_CLASSES
        # TODO: ADD MULTI CLASSES
        if num_classes > 2:
            raise NotImplementedError

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * REGRESSION_CN)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = rotate_iou(target, proposal)

        # # CUSTOM LOGIC: SET ALL ANCHORS VS TARGETS WITH ROTATION DIFF > angle threshold TO IOU OF 0 (to prevent rotation ambiguity)
        ANGLE_THRESH = 35 # degrees
        angle_diffs = torch.abs(target[:,-1].unsqueeze(-1) - proposal[:, -1])
        match_quality_matrix[angle_diffs >= ANGLE_THRESH] = 0

        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # # Fast RCNN only need "labels" field for selecting the targets
        # target = target.copy_with_fields("labels")
        # # get the targets corresponding GT for each proposal
        # # NB: need to clamp the indices because we can have a single
        # # GT in the image, and matched_idxs can be -2, which goes
        # # out of bounds
        # matched_targets = target[matched_idxs.clamp(min=0)]
        # matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_idxs

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_idxs = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = targets_per_image[matched_idxs.clamp(min=0)]

            # matched_idxs = matched_targets.get_field("matched_idxs")

            # TODO: ADD CLASS LABELS INSTEAD OF CLS AGNOSTIC
            labels_per_image = matched_idxs >= 0  # CLASS AGNOSTIC
            # raise NotImplementedError
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets, proposals_per_image
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # # proposals = list(proposals)
        # # add corresponding label and regression_targets information to the bounding boxes
        # for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
        #     labels, regression_targets, proposals
        # ):
        #     proposals_per_image.add_field("labels", labels_per_image)
        #     proposals_per_image.add_field(
        #         "regression_targets", regression_targets_per_image
        #     )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals[img_idx] = proposals[img_idx][img_sampled_inds]
            labels[img_idx] = labels[img_idx][img_sampled_inds]
            regression_targets[img_idx] = regression_targets[img_idx][img_sampled_inds]

        self._proposals = {"proposals": proposals, "labels": labels, "regression_targets": regression_targets}
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat(proposals["labels"], dim=0)
        regression_targets = cat(proposals["regression_targets"], dim=0)

        # get positive labels
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        total_pos = labels_pos.numel()

        if total_pos == 0:
            return labels_pos.sum(), labels_pos.sum(), labels_pos.sum()  # all 0, sum is convenient to get torch tensor

        # perform weighted classification loss (to prevent class imbalance i.e. too many negative)
        with torch.no_grad():
            num_classes = class_logits.shape[-1]
            label_cnts = torch.stack([(labels == x).sum() for x in range(num_classes)])
            label_weights = 1.0 / label_cnts.to(dtype=torch.float32)
            label_weights /= num_classes   # equal class weighting
        classification_loss = F.cross_entropy(class_logits, labels, weight=label_weights)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.arange(REGRESSION_CN, REGRESSION_CN * 2, device=device)
        else:
            map_inds = REGRESSION_CN * labels_pos[:, None] + torch.arange(REGRESSION_CN, device=device)

        pos_reg_pred = box_regression[sampled_pos_inds_subset[:, None], map_inds]
        pos_reg_targets = regression_targets[sampled_pos_inds_subset]
        box_loss = smooth_l1_loss(
            pos_reg_pred[:, :-1] - pos_reg_targets[:, :-1],
            size_average=False,
            beta=1,
        )

        angle_loss = torch.abs(torch.sin(pos_reg_pred[:, -1] - pos_reg_targets[:, -1])).mean()
        box_loss = 2.0 * box_loss / total_pos #  labels.numel()

        return classification_loss, box_loss, angle_loss

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img

        self.nms_rotate = RotateNMS(nms_threshold=nms, post_nms_top_n=detections_per_img)

        if box_coder is None:
            box_coder = BoxCoder(weights=None, lib=torch)
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        # concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        concat_boxes = torch.cat(boxes, dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -REGRESSION_CN:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for scores, boxes_per_img in zip(
            class_prob, proposals
        ):
            img_results = []
            # Apply threshold on detection probabilities and apply NMS
            # Skip j = 0, because it's the background class
            inds_all = scores > self.score_thresh
            for j in range(1, num_classes):
                inds = inds_all[:, j].nonzero().squeeze(1)
                scores_j = scores[inds, j]
                boxes_j = boxes_per_img[inds, j * REGRESSION_CN: (j + 1) * REGRESSION_CN]

                # sort based on score
                sorted_idx = torch.argsort(scores_j, descending=True)
                scores_j = scores_j[sorted_idx]
                boxes_j = boxes_j[sorted_idx]

                keep = self.nms_rotate(boxes_j)
                boxes_j = boxes_j[keep]
                scores_j = scores_j[keep].unsqueeze(1)

                result = torch.cat((boxes_j, scores_j), 1)
                # TODO: ADD CLS LABELS
                img_results.append(result)

            img_results = cat(img_results)
            number_of_detections = len(img_results)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.detections_per_img > 0:
                img_scores = img_results[:, -1]
                keep = torch.argsort(img_scores, descending=True)[:self.detections_per_img]
                img_results = img_results[keep]

            results.append(img_results)

        return results

def make_roi_box_feature_extractor(cfg, in_channels):
    def init_conv_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    fe = ConvROIFeatureExtractor(cfg, in_channels)
    fe.apply(init_conv_weights)
    return fe

def make_roi_box_predictor(cfg, in_channels):
    return FastRCNNPredictor(cfg, in_channels)

def make_roi_box_loss_evaluator(cfg):
    from matcher import Matcher
    from sampler import BalancedPositiveNegativeSampler

    matcher = Matcher(
        cfg.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights, lib=torch)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator

def make_roi_box_post_processor(cfg):
    bbox_reg_weights = cfg.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights, lib=torch)

    score_thresh = cfg.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.ROI_HEADS.NMS
    detections_per_img = cfg.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.CLS_AGNOSTIC_BBOX_REG

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return postprocessor


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
