#pragma once

#include <vector>
#include <torch/extension.h>

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor rotate_nms(
    const at::Tensor& r_boxes, const float nms_threshold
)
{
  if (r_boxes.type().is_cuda())
  {
#ifdef WITH_CUDA
    return rotate_nms_cuda(r_boxes, nms_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return rotate_nms_cpu(r_boxes, nms_threshold);
  }
  // AT_ERROR("Not implemented on the CPU");
}

// Interface for Python
at::Tensor rotate_iou_matrix(
    const at::Tensor& r_boxes1, const at::Tensor& r_boxes2
)
{
  if (r_boxes1.type().is_cuda())
  {
#ifdef WITH_CUDA
    return rotate_iou_matrix_cuda(r_boxes1, r_boxes2);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    return rotate_iou_matrix_cpu(r_boxes1, r_boxes2);
  }
  // AT_ERROR("Not implemented on the CPU");
}