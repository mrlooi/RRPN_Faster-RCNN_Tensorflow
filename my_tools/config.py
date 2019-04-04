import numpy as np

# ---------------------------------------------Backbone network config
class BACKBONE: object()

# BACKBONE.KERNEL_SIZES = [7,7,5,5]
# BACKBONE.FILTERS = [64,128,256,256*2]
# BACKBONE.OUT_CHANNELS = BACKBONE.FILTERS[-1]
# assert len(BACKBONE.FILTERS) == len(BACKBONE.STRIDES) == len(BACKBONE.KERNEL_SIZES)
BACKBONE.STRIDES = [2,2,2]  # TODO: MAKE CONFIGURABLE IN NETWORK (CURRENTLY HARDCODED!)
BACKBONE.SPATIAL_STRIDE = np.prod(BACKBONE.STRIDES)


# ---------------------------------------------RPN config
RPN_ONLY = False  #   WARN: RPN_ONLY=False Not Implemented!

class RPN: object()
RPN.BOX_REG_WEIGHTS = None  # None or length 5 list (weights for xc,yc,w,h,theta)
RPN.USE_FPN = False  #   WARN: Not Implemented!

# BASE_ANCHOR_SIZE = 256  # can be modified
# RPN.ANCHOR_SCALES = [0.125, 0.25, 0.5, 1.] 
RPN.ANCHOR_SIZES = [32,60,86]  # np.array(RPN.ANCHOR_SCALES) * BASE_ANCHOR_SIZE

RPN.ANCHOR_STRIDE = [8] # [16]  # can not be modified in most situations, unless using e.g. FPN
# RPN.ASPECT_RATIOS = [0.5, 1., 2.] # [1., 1. / 2, 2., 1. / 4, 4.]#, 1 / 3., 3.]#, 6., 1 / 6.]
# RPN.ANCHOR_ANGLES = [-90,-72,-54,-36,-18] #[-45, -15, 15] #[-90, -75, -60, -45, -30, -15]
RPN.ASPECT_RATIOS = [1.,2.] # [1., 1. / 2, 2., 1. / 4, 4.]#, 1 / 3., 3.]#, 6., 1 / 6.]
RPN.ANCHOR_ANGLES = [-90,-64,-39,-13,13,39,64] #[-45, -15, 15] #[-90, -75, -60, -45, -30, -15]


RPN.FG_IOU_THRESHOLD = 0.6
RPN.BG_IOU_THRESHOLD = 1 - RPN.FG_IOU_THRESHOLD

RPN.BATCH_SIZE_PER_IMAGE = 512
RPN.POSITIVE_FRACTION = 0.5

RPN.PRE_NMS_TOP_N_TRAIN = 5000
RPN.POST_NMS_TOP_N_TRAIN = 1000

RPN.PRE_NMS_TOP_N_TEST = 3000
RPN.POST_NMS_TOP_N_TEST = 300

RPN.NMS_THRESH = 0.6
RPN.MIN_SIZE = 0


# # --------------------------------------------ROI config
# class ROI_HEADS: object()
# ROI_HEADS.BBOX_REG_WEIGHTS = [10., 10., 5.0, 5.0, 5.0]

class ROI_HEADS: object()
ROI_HEADS.USE_FPN = False  #   WARN: Not Implemented!
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
ROI_HEADS.BBOX_REG_WEIGHTS = None  # None or length 5 list (weights for xc,yc,w,h,theta)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
ROI_HEADS.POSITIVE_FRACTION = 0.4

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
ROI_HEADS.SCORE_THRESH = 0.1
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
ROI_HEADS.NMS = 0.3
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
ROI_HEADS.DETECTIONS_PER_IMG = 100


class ROI_BOX_HEAD: object()
ROI_BOX_HEAD.POOLER_RESOLUTION = 4
ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
ROI_BOX_HEAD.POOLER_SCALES = (1.0 / BACKBONE.SPATIAL_STRIDE,)
ROI_BOX_HEAD.NUM_CLASSES = 2  # CURRENTLY ONLY SUPPORTS 2 CLASSES I.E. CLASS AGNOSTIC
# Hidden layer dimension when using an MLP for the RoI box head
ROI_BOX_HEAD.MLP_HEAD_DIM = 512
CLS_AGNOSTIC_BBOX_REG = True
