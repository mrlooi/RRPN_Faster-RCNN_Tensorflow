#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <stdio.h>

#include "rotate_rect_ops.h"

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ inline void get_rotated_rect_bounding_box(const T* pts, int& leftMost, int& topMost, 
  int& rightMost, int& bottomMost, const int width, const int height)
{
  leftMost = int(max(min(min(pts[0], pts[2]), min(pts[4], pts[6])), 0.0));
  topMost = int(max(min(min(pts[1], pts[3]), min(pts[5], pts[7])), 0.0));
  rightMost = int(min(max(max(pts[0], pts[2]), max(pts[4], pts[6])) + 1, width - 1.0));
  bottomMost = int(min(max(max(pts[1], pts[3]), max(pts[5], pts[7])) + 1, height - 1.0));
}

template <typename T>
__global__ void RRoIAlignFForward(const int nthreads, const T* bottom_data,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* top_data)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;  // batch_ind, xc, yc, w, h, angle
    int roi_batch_ind = offset_bottom_rois[0];

    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);
    T P_area = offset_bottom_rois[3] * spatial_scale / pooled_width * offset_bottom_rois[4] * spatial_scale / pooled_height;  // area = w * h

    int leftMost, topMost, rightMost, bottomMost;
    get_rotated_rect_bounding_box(P, leftMost, topMost, rightMost, bottomMost, width, height);
    
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    T output_val = 0.0;
    for (int hh = topMost; hh < bottomMost+1; ++hh) {
      for (int ww = leftMost; ww < rightMost+1; ++ww) {
        // T pixel_rect[5] = {ww+0.5f,hh+0.5f,1,1,0};
        T pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};
        
        T inter_area = computeRectInterArea(P, pixel_rect_vertices);
        T px_weight = inter_area / P_area;
        output_val += px_weight * offset_bottom_data[hh * width + ww];
      }
    }
    top_data[index] = output_val;
  }
}

template <typename T>
__global__ void RRoIAlignBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois, const float spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    T* bottom_diff,
    const T* bottom_rois) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;  // batch_ind, xc, yc, w, h, angle
    int roi_batch_ind = offset_bottom_rois[0];

    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);
    T P_area = offset_bottom_rois[3] * spatial_scale / pooled_width * offset_bottom_rois[4] * spatial_scale / pooled_height;  // area = w * h
    
    int leftMost, topMost, rightMost, bottomMost;
    get_rotated_rect_bounding_box(P, leftMost, topMost, rightMost, bottomMost, width, height);
    
    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    for (int hh = topMost; hh < bottomMost+1; ++hh) {
      for (int ww = leftMost; ww < rightMost+1; ++ww) {
        // T pixel_rect[5] = {ww+0.5f,hh+0.5f,1,1,0};
        T pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};
        
        T inter_area = computeRectInterArea(P, pixel_rect_vertices);
        T px_weight = inter_area / P_area;
        atomicAdd(offset_bottom_diff + hh * width + ww, static_cast<T>(px_weight * top_diff_this_bin));
      }
    }

  } // CUDA_1D_KERNEL_LOOP
} // RRoIAlignBackward



at::Tensor RROIAlign_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width)
{
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
//  auto argmax = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kInt));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "RROIAlign_forward", [&] {
    RRoIAlignFForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>()
     );
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

at::Tensor RROIAlign_backward_cuda(const at::Tensor& grad,
                      const at::Tensor& rois,
                      const float spatial_scale,
                      const int pooled_height,
                      const int pooled_width,
                      const int batch_size,
                      const int channels,
                      const int height,
                      const int width)
{
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "RROIAlign_backward", [&] {
    RRoIAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
        grad.numel(),
        grad.contiguous().data<scalar_t>(),
        num_rois,
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        grad_input.data<scalar_t>(),
        rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
