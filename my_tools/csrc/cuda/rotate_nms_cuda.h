#pragma once
#include <torch/extension.h>


at::Tensor rotate_nms_cuda(
    const at::Tensor& r_boxes, const float nms_threshold, const int max_output
);

// std::vector<at::Tensor> hough_voting_backward_cuda(const at::Tensor& grad);