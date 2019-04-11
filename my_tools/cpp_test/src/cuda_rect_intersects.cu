#include <iostream>
#include <vector>
#include <time.h>
#include <cuda_runtime.h>
#include <ctime>
#include <omp.h>

#include "rotate_rect_ops.h"

#define PRINT(a) std::cout << #a << ": " << a << std::endl;

__device__ const int BLOCKSIZE = 512;
// __device__ const int BLOCKSIZE_2D = 32;

static inline void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

template <typename T>
__global__ void rotated_rect_pixel_interpolation_kernel(const T *gpu_roi, int height, int width, T* out)
{
    T rect_vertices[8];
    convert_region_to_pts(gpu_roi, rect_vertices);

    T* P = rect_vertices;

    int leftMost = int(max(min(min(P[0], P[2]), min(P[4], P[6])), 0.0f));
    int topMost = int(max(min(min(P[1], P[3]), min(P[5], P[7])), 0.0f));
    int rightMost = int(min(max(max(P[0], P[2]), max(P[4], P[6])) + 1, width - 1.0f));
    int bottomMost = int(min(max(max(P[1], P[3]), max(P[5], P[7])) + 1, height - 1.0f));

    T roi_area = gpu_roi[2] * gpu_roi[3];
    for(int hh = topMost; hh < bottomMost + 1; hh++)
    {
        for(int ww = leftMost; ww < rightMost + 1; ww++)
        {
            // T pixel_rect[5] = {ww+0.5f,hh+0.5f,1,1,0};
            T pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};
            
            float interArea = computeRectInterArea(rect_vertices, pixel_rect_vertices);
            out[hh * width + ww] = interArea;
        }
    }
}

template <typename T>
__global__ void rotated_rect_IoU_lernel(const T *roi1, const T *roi2, T* out)
{
    out[0] = computeRectIoU(roi1, roi2);
}

void run_rotated_rect_pixel_interpolation()
{
    int H = 10;
    int W = 10;

    // int PH = 2;
    // int PW = 2;
    // std::vector<int> pool_dims {PH, PW}; 
    
    float roi_xc = 3;
    float roi_yc = 3;
    float roi_w = 3;
    float roi_h = 3;
    float roi_angle = 30;

    float roi_f[5] = {roi_xc,roi_yc,roi_w,roi_h,roi_angle};
    int numBytes = 5 * sizeof(float);

    std::vector<float> cpu_output(H*W);

    // convert to gpu 
    float *gpu_roi;
    float *gpu_output;
    HANDLE_ERROR(cudaMalloc((void**)&gpu_roi, numBytes));
    HANDLE_ERROR(cudaMalloc((void**)&gpu_output, H*W*sizeof(float)));
    HANDLE_ERROR(cudaMemset(gpu_output, 0, H*W*sizeof(float)));

    // copy cpu values to gpu
    HANDLE_ERROR(cudaMemcpy(gpu_roi, roi_f, numBytes, cudaMemcpyHostToDevice));

    int N = 1;
    int maxThreadsPerBlock = BLOCKSIZE;
    dim3 threadsPerBlock(maxThreadsPerBlock, 1, 1);
    dim3 blocksPerGrid((N + maxThreadsPerBlock - 1) / maxThreadsPerBlock, 1, 1);
    rotated_rect_pixel_interpolation_kernel<<<blocksPerGrid,threadsPerBlock>>>(gpu_roi, H, W, gpu_output);
    HANDLE_ERROR(cudaDeviceSynchronize());
    CHECK_LAUNCH_ERROR();

    // retrieve the results
    HANDLE_ERROR(cudaMemcpy(&cpu_output[0], gpu_output, H*W*sizeof(float), cudaMemcpyDeviceToHost));

    // deallocate memory
    HANDLE_ERROR(cudaFree(gpu_roi)); 
    HANDLE_ERROR(cudaFree(gpu_output)); 

    for(int hh = 0; hh < H; hh++)
    {
        for(int ww = 0; ww < W; ww++)
        {
            float interArea = cpu_output[hh*W + ww];
            if (interArea > 0)
            {
                printf("ww,hh: (%d,%d), inter_area: %.3f\n", ww,hh, interArea);
            }
        }
    }
}

void run_rotated_rect_iou()
{
    float roi1[5] = {50, 50, 100, 300, 0.};
    float roi2[5] = {50, 50, 100, 300, 0.};
    int numBytes = 5 * sizeof(float);
    
    // convert to gpu 
    float *gpu_roi1;
    float *gpu_roi2;
    float *gpu_output;
    HANDLE_ERROR(cudaMalloc((void**)&gpu_roi1, numBytes));
    HANDLE_ERROR(cudaMalloc((void**)&gpu_roi2, numBytes));
    HANDLE_ERROR(cudaMalloc((void**)&gpu_output, sizeof(float)));

    // copy cpu values to gpu
    HANDLE_ERROR(cudaMemcpy(gpu_roi1, roi1, numBytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_roi2, roi2, numBytes, cudaMemcpyHostToDevice));

    rotated_rect_IoU_lernel<<<1,1>>>(gpu_roi1, gpu_roi2, gpu_output);
    HANDLE_ERROR(cudaDeviceSynchronize());
    CHECK_LAUNCH_ERROR();

    // retrieve the results
    std::vector<float> cpu_output(1);
    HANDLE_ERROR(cudaMemcpy(&cpu_output[0], gpu_output, sizeof(float), cudaMemcpyDeviceToHost));

    printf("IOU: %.3f\n", cpu_output[0]);

    // deallocate memory
    HANDLE_ERROR(cudaFree(gpu_roi1)); 
    HANDLE_ERROR(cudaFree(gpu_roi2));     
}

int main(int argc, char *argv[])
{
    run_rotated_rect_pixel_interpolation();
    run_rotated_rect_iou();
	return 0;
}