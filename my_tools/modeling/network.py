import torch
import torch.nn as nn

from modeling.rpn import RPNHead, RPNModule


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
        
        # self.backbone, backbone_out_channels = self.build_backbone()
        self.backbone = VGGX(in_channels)
        backbone_out_channels = self.backbone.conv_dim_out[-1]
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

    # def build_backbone(self):
    #     # from layers import conv_transpose2d_by_factor

    #     backbone = nn.Sequential()

    #     c = self.cfg.BACKBONE
    #     cur_filters = self.in_channels
    #     ix = 1
    #     for stride, k, filters in zip(c.STRIDES, c.KERNEL_SIZES, c.FILTERS):
    #         conv = nn.Conv2d(cur_filters, filters, kernel_size=k, stride=stride, padding=k//2)
    #         backbone.add_module("conv_%d"%(ix), conv)
    #         backbone.add_module("bn_%d"%(ix), nn.BatchNorm2d(filters))
    #         backbone.add_module("relu_%d"%(ix), nn.ReLU())

    #         cur_filters = filters
    #         ix += 1

    #     return backbone, cur_filters
