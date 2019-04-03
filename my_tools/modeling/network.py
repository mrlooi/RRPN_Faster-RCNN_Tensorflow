import torch
import torch.nn as nn

from modeling.rpn import build_rpn
from modeling.roi_heads import build_roi_heads

def init_conv_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class VGGX(nn.Module):
    def __init__(self, in_channels=3):
        super(VGGX, self).__init__()

        self.in_channels = in_channels
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv_dim_out = [64,64,128,256]#,512]
        self.conv_spatial_scale = [1.0, 1.0/2, 1.0/4, 1.0/8]#, 1.0/16]
        c_dims = self.conv_dim_out

        inplace = True

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, c_dims[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_dims[1], 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace),
            nn.Conv2d(128, c_dims[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(c_dims[2]),
            nn.ReLU(inplace),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_dims[2], c_dims[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(c_dims[3]),
            nn.ReLU(inplace),
            # nn.Conv2d(128, c_dims[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace),
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(c_dims[3], 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace),
        #     nn.Conv2d(512, c_dims[4], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace),
        # )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.max_pool2d(conv1))
        conv3 = self.conv3(self.max_pool2d(conv2))
        conv4 = self.conv4(self.max_pool2d(conv3))
        # conv5 = self.conv5(self.max_pool2d(conv4))
        return conv4


class DetectionNetwork(nn.Module):
    def __init__(self, cfg, in_channels=3):
        super(DetectionNetwork, self).__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        # self.num_anchors_per_location = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS) * len(cfg.ANCHOR_ANGLES)
        # print("Total anchors: %d"%(self.num_anchors_per_location))
        
        self.backbone = VGGX(in_channels)
        backbone_out_channels = self.backbone.conv_dim_out[-1]
        self.backbone.apply(init_conv_weights)

        self.rpn = build_rpn(self.cfg, backbone_out_channels)#, self.num_anchors_per_location)
        
        self.roi_heads = build_roi_heads(self.cfg, backbone_out_channels)

    def forward(self, x, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        features = self.backbone(x)

        rpn_proposals, rpn_losses = self.rpn(x, features, targets)  # rpn_proposals: (N, 6) -> xc,yc,w,h,angle,score

        if self.roi_heads:
            proposals = [pp[:, :5] for pp in rpn_proposals]   # last dim (score) not needed
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            # x = features
            result = rpn_proposals
            detector_losses = {}

        losses = {}
        if self.training:
            losses.update(rpn_losses)
            losses.update(detector_losses)

        return result, losses
