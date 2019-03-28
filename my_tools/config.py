import numpy as np

# ---------------------------------------------Backbone network config
class BACKBONE: object()

BACKBONE.STRIDES = [2,1,2,2]
BACKBONE.KERNEL_SIZES = [7,7,5,5]
BACKBONE.FILTERS = [64,128,256,256*2]
BACKBONE.OUT_CHANNELS = BACKBONE.FILTERS[-1]
assert len(BACKBONE.FILTERS) == len(BACKBONE.STRIDES) == len(BACKBONE.KERNEL_SIZES)
BACKBONE.SPATIAL_STRIDE = np.prod(BACKBONE.STRIDES)

# ---------------------------------------------RPN config
RPN_ONLY = True  #   WARN: RPN_ONLY=False Not Implemented!
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
RPN.POST_NMS_TOP_N_TRAIN = 2000

RPN.PRE_NMS_TOP_N_TEST = 3000
RPN.POST_NMS_TOP_N_TEST = 100

RPN.NMS_THRESH = 0.5
RPN.MIN_SIZE = 0

# TRAIN_RPN_CLOOBER_POSITIVES = False



# # --------------------------------------------ROI config
# class ROI_HEADS: object()
# ROI_HEADS.BBOX_REG_WEIGHTS = [10., 10., 5.0, 5.0, 5.0]

