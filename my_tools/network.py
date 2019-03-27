import torch
import torch.nn as nn

from rpn import RPNHead, RPNModule


def init_conv_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class DetectionNetwork(nn.Module):
    def __init__(self, cfg, in_channels=3):
        super(DetectionNetwork, self).__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        # self.num_anchors_per_location = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS) * len(cfg.ANCHOR_ANGLES)
        # print("Total anchors: %d"%(self.num_anchors_per_location))
        
        self.backbone, backbone_out_channels = self.build_backbone()
        self.backbone.apply(init_conv_weights)
        self.rpn = self.build_rpn(backbone_out_channels)#, self.num_anchors_per_location)

    def forward(self, x, targets=None):
        features = self.backbone(x)

        rpn_box_pred, rpn_losses = self.rpn(x, features, targets)

        if not self.cfg.RPN_ONLY:
            raise NotImplementedError

        losses = {}
        losses.update(rpn_losses)

        return rpn_box_pred, losses

    # def build_full_network(self):
    #     backbone = self.build_backbone()
    #     return backbone

    def build_rpn(self, in_channels):
        
        # rpn = RPNHead(in_channels, num_anchors)
        rpn = RPNModule(self.cfg, in_channels)
        return rpn

    def build_backbone(self):
        # from layers import conv_transpose2d_by_factor

        backbone = nn.Sequential()

        c = self.cfg.BACKBONE
        cur_filters = self.in_channels
        ix = 1
        for stride, k, filters in zip(c.STRIDES, c.KERNEL_SIZES, c.FILTERS):
            conv = nn.Conv2d(cur_filters, filters, kernel_size=k, stride=stride, padding=k//2)
            backbone.add_module("conv_%d"%(ix), conv)
            backbone.add_module("bn_%d"%(ix), nn.BatchNorm2d(filters))
            backbone.add_module("relu_%d"%(ix), nn.ReLU())

            cur_filters = filters
            ix += 1

        return backbone, cur_filters
