import torch
from torch import nn

from layers.rotate_roi_pool import RROIPool
from utils import cat

class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio=0):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []

        if len(scales) > 1:
            raise NotImplementedError

        for scale in scales:
            poolers.append(
                RROIPool(
                    output_size, spatial_scale=scale#, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # # get the levels in the feature map by leveraging the fact that the network always
        # # downsamples by a factor of 2 at each level.
        # lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        # lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        # concat_boxes = cat([b.bbox for b in boxes], dim=0)
        concat_boxes = cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            out, argmax = self.poolers[0](x, rois) # TODO: REMOVE ARGMAX (DEBUG?)
            return out
        else:
            raise NotImplementedError

        # levels = self.map_levels(boxes)
        #
        # num_rois = len(rois)
        # num_channels = x[0].shape[1]
        # output_size = self.output_size[0]
        #
        # dtype, device = x[0].dtype, x[0].device
        # result = torch.zeros(
        #     (num_rois, num_channels, output_size, output_size),
        #     dtype=dtype,
        #     device=device,
        # )
        # for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
        #     idx_in_level = torch.nonzero(levels == level).squeeze(1)
        #     rois_per_level = rois[idx_in_level]
        #     result[idx_in_level] = pooler(per_level_feature, rois_per_level)
        #
        # return result
