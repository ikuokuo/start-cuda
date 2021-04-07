#include <cuda_runtime.h>

#include <algorithm>
#include <iterator>
#include <iostream>

// #include "common/gpu_array.hpp"

#define ARRAY_SIZE 128

__global__ void kernel_multipy(int * const a,
    const int * const b,
    const int * const c) {
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  a[thread_idx] = b[thread_idx] * c[thread_idx];
}

int main(int argc, char const *argv[]) {
  int a[ARRAY_SIZE];
  int b[ARRAY_SIZE];
  int c[ARRAY_SIZE];

  // std::fill(b, b+ARRAY_SIZE, 2);
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    b[i] = i;
  }
  std::fill(c, c+ARRAY_SIZE, 3);

  {
    size_t bytes = sizeof(int) * ARRAY_SIZE;

    int * gpu_a;
    int * gpu_b;
    int * gpu_c;

    cudaMalloc((void **)&gpu_a, bytes);
    cudaMalloc((void **)&gpu_b, bytes);
    cudaMalloc((void **)&gpu_c, bytes);

    cudaMemcpy(gpu_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c, c, bytes, cudaMemcpyHostToDevice);

    kernel_multipy<<<2, 64>>>(gpu_a, gpu_b, gpu_c);

    cudaMemcpy(a, gpu_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, gpu_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, gpu_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
  }
  /*
  {
    GpuArray<int> gpu_a(ARRAY_SIZE);
    GpuArray<int> gpu_b(ARRAY_SIZE);
    GpuArray<int> gpu_c(ARRAY_SIZE);

    gpu_a.Set(a, ARRAY_SIZE);
    gpu_b.Set(b, ARRAY_SIZE);
    gpu_c.Set(c, ARRAY_SIZE);

    kernel_multipy<<<2, 64>>>(gpu_a.GetData(), gpu_b.GetData(), gpu_c.GetData());

    gpu_a.Get(a, ARRAY_SIZE);
    gpu_b.Get(b, ARRAY_SIZE);
    gpu_c.Get(c, ARRAY_SIZE);
  }
  */

  std::copy(a, a+ARRAY_SIZE, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}
