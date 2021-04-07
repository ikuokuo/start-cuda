#include <iostream>
#include <random>

#include "common/gpu_array.hpp"
#include "common/time_cost.hpp"

#define u32 uint32_t

#define NUM_ELEM 1024
#define MAX_NUM_LISTS 1024

// 基数排序 CPU

__host__ void cpu_sort(u32 * const data, const u32 num_elements) {
  // 只需一个列表：将 bit=0 的从列表头开始存放， bit=1 的从列表尾开始存放。
  /*static*/ u32 cpu_tmp_0[num_elements];
  /*static*/ u32 cpu_tmp_1[num_elements];

  for (u32 bit = 0; bit < 32; bit++) {
    u32 base_cnt_0 = 0;
    u32 base_cnt_1 = 0;

    const u32 bit_mask = (1 << bit);
    for (u32 i = 0; i < num_elements; i++) {
      const u32 d = data[i];
      if ((d & bit_mask) > 0) {
        cpu_tmp_1[base_cnt_1] = d;
        base_cnt_1++;
      } else {
        cpu_tmp_0[base_cnt_0] = d;
        base_cnt_0++;
      }
    }

    // Copy data back to source - first the zero list
    for (u32 i = 0; i < base_cnt_0; i++) {
      data[i] = cpu_tmp_0[i];
    }

    // Copy data back to source - then the one list
    for (u32 i = 0; i < base_cnt_1; i++) {
      data[base_cnt_0 + i] = cpu_tmp_1[i];
    }
  }
}

// 基数排序 GPU

__device__ void copy_data_to_shared(const u32 * const data,
    u32 * const sort_tmp,
    const u32 num_lists,
    const u32 num_elements,
    const u32 tid) {
  // Copy data into temp source
  for (u32 i = 0; i < num_elements; i += num_lists) {
    sort_tmp[i + tid] = data[i + tid];
  }
  __syncthreads();
}

__device__ void radix_sort(u32 * const sort_tmp,
    const u32 num_lists,
    const u32 num_elements,
    const u32 tid,
    u32 * const sort_tmp_1) {
  // Sort into num_list, lists
  // Apply radix sort on 32 bits of data
  for (u32 bit = 0; bit < 32; bit++) {
    u32 base_cnt_0 = 0;
    u32 base_cnt_1 = 0;

    const u32 bit_mask = (1 << bit);
    for (u32 i = 0; i < num_elements; i += num_lists) {
      const u32 elem = sort_tmp[i + tid];
      if ((elem & bit_mask) > 0) {
        sort_tmp_1[base_cnt_1 + tid] = elem;
        base_cnt_1 += num_lists;
      } else {
        sort_tmp[base_cnt_0 + tid] = elem;
        base_cnt_0 += num_lists;
      }
    }

    // Copy data back to source - then the one list
    for (u32 i = 0; i < base_cnt_1; i += num_lists) {
      sort_tmp[base_cnt_0 + i + tid] = sort_tmp_1[i + tid];
    }
  }

  __syncthreads();
}

__device__ void merge_array1(const u32 * const src_array,
    u32 * const dest_array,
    const u32 num_lists,
    const u32 num_elements,
    const u32 tid);

__global__ void radix_sort(u32 * const data,
    const u32 num_lists,
    const u32 num_elements) {
  const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  __shared__ u32 sort_tmp[NUM_ELEM];
  __shared__ u32 sort_tmp_1[NUM_ELEM];

  copy_data_to_shared(data, sort_tmp, num_lists, num_elements, tid);

  radix_sort(sort_tmp, num_lists, num_elements, tid, sort_tmp_1);

  merge_array1(sort_tmp, data, num_lists, num_elements, tid);
}

__host__ void gpu_sort(u32 * const data, const u32 num_elements) {
  size_t bytes = sizeof(u32) * num_elements;
  //std::cout << "bytes: " << bytes << std::endl;

  u32 *gpu_data;

  cudaMalloc((void **)&gpu_data, bytes);

  cudaMemcpy(gpu_data, data, bytes, cudaMemcpyHostToDevice);

  // num_lists: 基数排序所产生的独立列表的数目，它应该等于内核函数每个线程块启动的线程数目
  radix_sort<<<num_elements/256, 256>>>(gpu_data, 256, num_elements);

  cudaMemcpy(data, gpu_data, bytes, cudaMemcpyDeviceToHost);

  cudaFree(gpu_data);
}

// 辅助函数

template<class IntType = int>
IntType *rand_data(size_t n,
    IntType a = 0,
    IntType b = std::numeric_limits<IntType>::max()) {
  std::random_device rd;
  std::uniform_int_distribution<IntType> dist(a, b);

  IntType *data = new IntType[n];
  for (size_t i = 0; i < n; ++i) {
    data[i] = dist(rd);
  }
  return data;
}

template<class IntType = int>
IntType *copy_data(const IntType * const data, size_t n) {
  IntType *data_new = new IntType[n];
  std::copy(data, data + n, data_new);
  return data_new;
}

template<class IntType = int>
std::ostream &print_data(const IntType * const data, size_t n,
    const char *delim = " ", std::ostream &os = std::cout) {
  std::copy(data, data + n, std::ostream_iterator<IntType>(os, delim));
  return os;
}

float time_cost(std::function<void()> func) {
  using namespace std;
  using namespace times;
  auto time_beg = now();
  func();
  auto time_end = now();
  float ms = count<std::chrono::microseconds>(time_end-time_beg) * 0.001f;
  cout << "BEG: " << to_local_string(time_beg) << endl
    << "END: " << to_local_string(time_end) << endl
    << "COST: " << ms << " ms" << endl;
  return ms;
}

int main(int argc, char const *argv[]) {
  size_t n = NUM_ELEM;
  u32 *data = rand_data<u32>(n, 0, 999);
  u32 *data2 = copy_data<u32>(data, n);

  std::cout << "### Data ###" << std::endl;
  print_data<u32>(data, n) << std::endl;

  std::cout << "\n### Data Sorted with CPU ###" << std::endl;
  time_cost([&data, &n]() {
    cpu_sort(data, n);
  });
  print_data<u32>(data, n) << std::endl;

  std::cout << "\n### Data Sorted with GPU ###" << std::endl;
  time_cost([&data2, &n]() {
    gpu_sort(data2, n);
  });
  print_data<u32>(data2, n) << std::endl;

  delete[] data;
  delete[] data2;
  return 0;
}

__device__ void merge_array1(const u32 * const src_array,
    u32 * const dest_array,
    const u32 num_lists,
    const u32 num_elements,
    const u32 tid) {
  __shared__ u32 list_indexes[MAX_NUM_LISTS];

  // Multiple threads
  list_indexes[tid] = 0;
  __syncthreads();

  // Single threaded
  if (tid == 0) {
    const u32 num_elements_pre_list = (num_elements / num_lists);

    for (u32 i = 0; i < num_elements; i++) {
      u32 min_val = 0xFFFFFFFF;
      u32 min_idx = 0;

      // Iterate over each of the lists
      for (u32 list = 0; list < num_lists; list++) {
        // If the current list has already been emptied then ignore it
        if (list_indexes[list] < num_elements_pre_list) {
          const u32 src_idx = list + (list_indexes[list] * num_lists);

          const u32 data = src_array[src_idx];
          if (data <= min_val) {
            min_val = data;
            min_idx = list;
          }
        }
      }

      list_indexes[min_idx]++;
      dest_array[i] == min_val;
    }
  }
}
