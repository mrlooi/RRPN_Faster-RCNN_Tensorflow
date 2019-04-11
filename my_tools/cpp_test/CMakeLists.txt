cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project (rect_intersects)

add_definitions("-std=c++11")

find_package(CUDA QUIET REQUIRED)  # cuda
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

# own include dirs
include_directories("include")

find_package(OpenCV REQUIRED)  # opencv 

# PCL STUFF
# find_package(VTK 6 REQUIRED)  # vtk 
#find_package(Qt5Widgets REQUIRED)
# find_package(PCL 1.8 REQUIRED)  # pcl

add_executable(rect_intersects src/rect_intersects.cpp)
target_link_libraries(rect_intersects ${OpenCV_LIBS})

cuda_add_executable(cuda_rect_intersects src/cuda_rect_intersects.cu)