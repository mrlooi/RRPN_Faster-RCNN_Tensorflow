#include "rotate_nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "RROIPool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("rotate_nms", &rotate_nms, "rotate_nms");
  m.def("rotate_iou_matrix", &rotate_iou_matrix, "rotate_iou_matrix");

  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");

  // rotated ROI implementations
  m.def("rotate_roi_pool_forward", &RROIPool_forward, "RROIPool_forward");
  m.def("rotate_roi_pool_backward", &RROIPool_backward, "RROIPool_backward");


}
