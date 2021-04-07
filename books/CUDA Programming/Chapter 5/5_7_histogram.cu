#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

#include "common/time_cost.hpp"

#ifndef ARRAY_SIZE
#  define ARRAY_SIZE(x) (sizeof((x)) / sizeof((x)[0]))
#endif

using CalcHistFunc = std::function<void(uint32_t * const, size_t,
                                        uint32_t * const, size_t)>;

void print_data(const uint32_t * const data, size_t size, bool use_hex = false,
    std::function<void(size_t)> pre_func = nullptr) {
  using namespace std;
  // copy(data, data + size, ostream_iterator<int>(cout, " "));
  auto first = data;
  auto last = data + size;
  uint32_t line_nums = 8;
  uint32_t page_nums = 1024;
  uint32_t n = 0;
  while (first != last) {
    if (pre_func) pre_func(n);
    if (use_hex) {
      cout << "0x" << hex << setfill('0');
    } else {
      cout << dec << setfill(' ');
    }
    cout << setw(8) << *first++;
    ++n;
    if (n % line_nums == 0) {
      cout << endl;
    } else {
      cout << ' ';
    }
    if (n % page_nums == 0) cout << endl;
  }
  if (n % line_nums != 0) cout << endl;
}

__shared__ uint32_t d_bin_data_shared[256];

// Each read is 4 bytes, not one, 32 x 4 = 128 byte reads
// Accumulate into shared memory N times
__global__ void histogram256Kernel(
    uint32_t * const d_hist_data,
    uint32_t * const d_bin_data,
    const uint32_t N) {
  // Work out our thread id
  const uint32_t idx = (blockIdx.x * (blockDim.x*N) + threadIdx.x);
  const uint32_t idy = (blockIdx.y * blockDim.y) + threadIdx.y;
  const uint32_t tid = idx + idy * (blockDim.x*N) * (gridDim.x);

  // Clear shared memory
  d_bin_data_shared[threadIdx.x] = 0;

  // Wait for all threads to update shared memory
  __syncthreads();

  for (uint32_t i = 0, tid_offset = 0; i < N; i++, tid_offset+=256) {
    const uint32_t value_u32 = d_hist_data[tid+tid_offset];

    atomicAdd(&(d_bin_data_shared[(value_u32 & 0x000000FF)]), 1);
    atomicAdd(&(d_bin_data_shared[(value_u32 & 0x0000FF00) >> 8]), 1);
    atomicAdd(&(d_bin_data_shared[(value_u32 & 0x00FF0000) >> 16]), 1);
    atomicAdd(&(d_bin_data_shared[(value_u32 & 0xFF000000) >> 24]), 1);
  }

  // Wait for all threads to update shared memory
  __syncthreads();

  // The write the accumulated data back to global memory in blocks, not scattered
  atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}


void gpu_hist(uint32_t * const hist_data, size_t hist_size,
              uint32_t * const bin_data, size_t bin_size) {
  uint32_t *gpu_hist;
  uint32_t *gpu_bin;

  size_t n = sizeof(uint32_t);
  cudaMalloc((void **)&gpu_hist, hist_size*n);
  cudaMalloc((void **)&gpu_bin, bin_size*n);

  cudaMemcpy(gpu_hist, hist_data, hist_size*n, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_bin, bin_data, bin_size*n, cudaMemcpyHostToDevice);

  const uint32_t N = 2;
  histogram256Kernel<<<hist_size/256/N, 256>>>(gpu_hist, gpu_bin, N);

  cudaMemcpy(hist_data, gpu_hist, hist_size*n, cudaMemcpyDeviceToHost);
  cudaMemcpy(bin_data, gpu_bin, bin_size*n, cudaMemcpyDeviceToHost);

  cudaFree(gpu_hist);
  cudaFree(gpu_bin);
}

void cpu_hist(uint32_t * const hist_data, size_t hist_size,
              uint32_t * const bin_data, size_t bin_size) {
  for (size_t i = 0; i < hist_size; ++i) {
    const uint32_t &data = *(hist_data + i);
    for (size_t n = 0; n < 4; ++n) {
      ++bin_data[(data >> (8 * n)) & 0xff];
    }
  }
}

float test_hist(CalcHistFunc calc_hist,
                uint32_t * const hist_data, size_t hist_size,
                uint32_t * const bin_data, size_t bin_size) {
  using namespace std;
  using namespace times;

  fill(bin_data, bin_data + bin_size, 0);

  auto time_beg = now();
  calc_hist(hist_data, hist_size, bin_data, bin_size);
  auto time_end = now();
  float ms = count<std::chrono::microseconds>(time_end-time_beg) * 0.001f;

  cout << "BEG: " << to_local_string(time_beg) << endl
    << "END: " << to_local_string(time_end) << endl
    << "COST: " << ms << " ms" << endl;
  print_data(bin_data, bin_size, false, [](size_t n) {
    cout << '[' << hex << setfill('0') << setw(2) << n << ']';
  });
  if (accumulate(bin_data, bin_data + bin_size, 0) != (hist_size*4)) {
    cout << "WARN: total hist bytes are incorrect" << endl;
  }
  return ms;
}

uint32_t *big_rand_data(size_t bytes) {
  if (bytes % 4 != 0)
    throw std::length_error("Data size must be divisible by 4");
  size_t n = bytes / 4;
  std::random_device rd;
  std::uniform_int_distribution<uint32_t> dist(0, 255);
  uint32_t *data = new uint32_t[n];
  for (size_t i = 0; i < n; ++i) {
    data[i] = (dist(rd) & 0xff) +
              ((dist(rd) & 0xff) << 8) +
              ((dist(rd) & 0xff) << 16) +
              ((dist(rd) & 0xff) << 24);
  }
  return data;
}

int main(int argc, char const *argv[]) {
  using namespace std;

  size_t hist_size = 1024;
  // size_t hist_size = 1024 * 1024 * 1;
  cout << "Generating histogram data ..." << endl;
  uint32_t *hist_data = big_rand_data(hist_size*4);
  // print_data(hist_data, hist_size, true);

  size_t bin_size = 256;
  uint32_t bin_data[bin_size];

  cout << endl << "### Test CPU Histogram ###" << endl;
  float cpu_cost = test_hist(cpu_hist, hist_data, hist_size, bin_data,
    bin_size);

  cout << endl << "### Test GPU Histogram ###" << endl;
  float gpu_cost = test_hist(gpu_hist, hist_data, hist_size, bin_data,
    bin_size);

  cout << endl << "### CPU ";
  if (cpu_cost < gpu_cost) {
    cout << '<';
  } else if (cpu_cost > gpu_cost) {
    cout << '>';
  } else {
    cout << '=';
  }
  cout << " GPU ###" << endl;

  delete[] hist_data;
  return 0;
}
