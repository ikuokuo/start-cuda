#include "common/gpu_array.hpp"

__global__ void kernel_grayscale(
    uint8_t * const rgb8_pixels,
    uint8_t * const gray8_pixels,
    const int size) {
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
  const unsigned int tid = ((gridDim.x * blockDim.x) * idy) + idx;
  if (tid < size) {
    // Y = 0.299*R + 0.587*G + 0.114*B
    const unsigned int rbg8_n = 3 * tid;
    gray8_pixels[tid] = 0.299*rgb8_pixels[rbg8_n] +
      0.587*rgb8_pixels[rbg8_n+1] +
      0.114*rgb8_pixels[rbg8_n+2];
  }
}

void gpu_grayscale(uint8_t * const rgb8_pixels, int rgb8_size,
                   uint8_t * const gray8_pixels, int gray8_size) {
  // rgb8: 7538136=1084x2318x3, gray8: 2512712=1084x2318
  GpuArray<uint8_t> gpu_rgb8_pixels(rgb8_size);
  GpuArray<uint8_t> gpu_gray8_pixels(gray8_size);

  gpu_rgb8_pixels.Set(rgb8_pixels, rgb8_size);
  gpu_gray8_pixels.Set(gray8_pixels, gray8_size);

  kernel_grayscale<<<gray8_size/256+1, 256>>>(
    gpu_rgb8_pixels.GetData(),
    gpu_gray8_pixels.GetData(), gray8_size);

  gpu_rgb8_pixels.Get(rgb8_pixels, rgb8_size);
  gpu_gray8_pixels.Get(gray8_pixels, gray8_size);
}
