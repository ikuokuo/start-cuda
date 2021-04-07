#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>

#define CUDA_CALL(func) do { \
  const cudaError_t a = (func); \
  if (a != cudaSuccess) { \
    printf("\nCUDA Error: %s (err_num=%d)\n", cudaGetErrorString(a), a); \
    cudaDeviceReset(); \
    assert(0); \
  } \
} while (0)

#define CUDA_ERROR_CHECK \
__host__ void cuda_error_check(const char *prefix, const char *postfix) { \
  if (cudaPeekAtLastError() != cudaSuccess) { \
    printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), \
      postfix); \
    cudaDeviceReset(); \
    /*wait_exit();*/ \
    assert(0); \
  } \
}
