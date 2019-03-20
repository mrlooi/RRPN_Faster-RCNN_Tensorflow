#include "rotate_nms.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("rotate_nms", &rotate_nms, "rotate_nms");
}
