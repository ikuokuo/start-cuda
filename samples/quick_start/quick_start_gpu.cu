#include <algorithm>
#include <iostream>
#include <iterator>

#include "common/time_cost.hpp"

#define ARRAY_SIZE 256

__global__ void kernel_array_add(
    const int * const lhs,
    const int * const rhs,
    int * const sum,
    size_t n) {
  const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  sum[i] = lhs[i] + rhs[i];
}

int main(int argc, char const *argv[]) {
  int lhs[ARRAY_SIZE];
  int rhs[ARRAY_SIZE];
  int sum[ARRAY_SIZE];

  // fill values
  std::fill(lhs, lhs+ARRAY_SIZE, 2);
  std::fill(rhs, rhs+ARRAY_SIZE, 3);

  {
    size_t bytes = sizeof(int) * ARRAY_SIZE;

    int *gpu_lhs;
    int *gpu_rhs;
    int *gpu_sum;

    cudaMalloc((void**)&gpu_lhs, bytes);
    cudaMalloc((void**)&gpu_rhs, bytes);
    cudaMalloc((void**)&gpu_sum, bytes);

    // host to device
    cudaMemcpy(gpu_lhs, lhs, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rhs, rhs, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sum, sum, bytes, cudaMemcpyHostToDevice);

    TIME_BEG("kernel_array_add");
    kernel_array_add<<<ARRAY_SIZE/64, 64>>>(
        gpu_lhs, gpu_rhs, gpu_sum, ARRAY_SIZE);
    TIME_END("kernel_array_add");

    // device to host
    cudaMemcpy(lhs, gpu_lhs, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(rhs, gpu_rhs, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sum, gpu_sum, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_lhs);
    cudaFree(gpu_rhs);
    cudaFree(gpu_sum);
  }

  // print sum
  std::cout << "sum[" << ARRAY_SIZE << "]:" << std::endl;
  std::copy(sum, sum+ARRAY_SIZE, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}
