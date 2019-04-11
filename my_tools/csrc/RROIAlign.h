// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor RROIAlign_forward(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return RROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor RROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return RROIAlign_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

