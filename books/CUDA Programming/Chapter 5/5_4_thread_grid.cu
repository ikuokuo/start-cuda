#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void what_is_my_id_2d_A(
    unsigned int * const block_x,
    unsigned int * const block_y,
    unsigned int * const thread,
    unsigned int * const calc_thread,
    unsigned int * const x_thread,
    unsigned int * const y_thread,
    unsigned int * const grid_dimx,
    unsigned int * const block_dimx,
    unsigned int * const grid_dimy,
    unsigned int * const block_dimy) {
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
  const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

  block_x[thread_idx]     = blockIdx.x;   // 线程网格X维度上的线程块索引
  block_y[thread_idx]     = blockIdx.y;   // 线程网格Y维度上的线程块索引
  thread[thread_idx]      = threadIdx.x;  // 线程块X维度上的线程索引
  calc_thread[thread_idx] = thread_idx;
  x_thread[thread_idx]    = idx;
  y_thread[thread_idx]    = idy;
  grid_dimx[thread_idx]   = gridDim.x;    // 线程网格X维度上线程块的数量
  block_dimx[thread_idx]  = blockDim.x;   // 一个线程块X维度上线程数量
  grid_dimy[thread_idx]   = gridDim.y;    // 线程网格Y维度上线程块的数量
  block_dimy[thread_idx]  = blockDim.y;   // 一个线程块Y维度上线程数量
}

#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16

#define ARRAY_SIZE_IN_BYTES (ARRAY_SIZE_X * ARRAY_SIZE_Y * sizeof(unsigned int))

// Declare statically six arrays of ARRAY_SIZE each
unsigned int cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_xthread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_ythread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

int main(int argc, char const *argv[]) {
  // 线程块，条纹布局
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  // ┃ 线程0~15，块0    ┆ 线程16~31，块0   ┃
  // ┃ 线程32~47，块0   ┆ 线程48~63，块0   ┃
  // ┃ 线程64~79，块0   ┆ 线程80~95，块0   ┃
  // ┃ 线程96~111，块0  ┆ 线程112~127，块0 ┃
  // ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
  // ┃ 线程0~15，块1    ┆ 线程16~31，块1   ┃
  // ┃ 线程32~47，块1   ┆ 线程48~63，块1   ┃
  // ┃ 线程64~79，块1   ┆ 线程80~95，块1   ┃
  // ┃ 线程96~111，块1  ┆ 线程112~127，块1 ┃
  // ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
  // ┃ 线程0~15，块2    ┆ 线程16~31，块2   ┃
  // ┃ 线程32~47，块2   ┆ 线程48~63，块2   ┃
  // ┃ 线程64~79，块2   ┆ 线程80~95，块2   ┃
  // ┃ 线程96~111，块2  ┆ 线程112~127，块2 ┃
  // ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
  // ┃ 线程0~15，块4    ┆ 线程16~31，块4   ┃
  // ┃ 线程32~47，块4   ┆ 线程48~63，块4   ┃
  // ┃ 线程64~79，块4   ┆ 线程80~95，块4   ┃
  // ┃ 线程96~111，块4  ┆ 线程112~127，块4 ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  // Total thread count = 32 * 4 = 128
  const dim3 threads_rect(32, 4);
  const dim3 blocks_rect(1, 4);

  // 线程块，方块布局
  // ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  // ┃ 线程0~15，块0    ┃ 线程0~15，块1    ┃
  // ┃ 线程16~31，块0   ┃ 线程16~31，块1   ┃
  // ┃ 线程32~47，块0   ┃ 线程32~47，块1   ┃
  // ┃ 线程48~63，块0   ┃ 线程48~63，块1   ┃
  // ┃ 线程64~79，块0   ┃ 线程64~79，块1   ┃
  // ┃ 线程80~95，块0   ┃ 线程80~95，块1   ┃
  // ┃ 线程96~111，块0  ┃ 线程96~111，块1  ┃
  // ┃ 线程112~127，块0 ┃ 线程112~127，块1 ┃
  // ┣━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━┫
  // ┃ 线程0~15，块2    ┃ 线程0~15，块4    ┃
  // ┃ 线程16~31，块2   ┃ 线程16~31，块4   ┃
  // ┃ 线程32~47，块2   ┃ 线程32~47，块4   ┃
  // ┃ 线程48~63，块2   ┃ 线程48~63，块4   ┃
  // ┃ 线程64~79，块2   ┃ 线程64~79，块4   ┃
  // ┃ 线程80~95，块2   ┃ 线程80~95，块4   ┃
  // ┃ 线程96~111，块2  ┃ 线程96~111，块4  ┃
  // ┃ 线程112~127，块2 ┃ 线程112~127，块4 ┃
  // ┗━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━┛
  // Total thread count = 16 * 8 = 128
  const dim3 threads_square(16, 8);
  const dim3 blocks_square(2, 2);

  // Declare pointers for GPU based params
  unsigned int * gpu_block_x;
  unsigned int * gpu_block_y;
  unsigned int * gpu_thread;
  unsigned int * gpu_calc_thread;
  unsigned int * gpu_xthread;
  unsigned int * gpu_ythread;
  unsigned int * gpu_grid_dimx;
  unsigned int * gpu_block_dimx;
  unsigned int * gpu_grid_dimy;
  unsigned int * gpu_block_dimy;

  // Allocate four arrays on the GPU
  cudaMalloc((void **)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_xthread, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_ythread, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_block_dimy, ARRAY_SIZE_IN_BYTES);

  for (int kernel = 0; kernel < 2; kernel++) {
    switch (kernel) {
      case 0: {
        what_is_my_id_2d_A<<<blocks_rect, threads_rect>>>(
          gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread,
          gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
          gpu_grid_dimy, gpu_block_dimy);
      } break;
      case 1: {
        what_is_my_id_2d_A<<<blocks_square, threads_square>>>(
          gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread,
          gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
          gpu_grid_dimy, gpu_block_dimy);
      } break;
      default: exit(1); break;
    }

    // Copy back gpu results to the CPU
    cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_xthread, gpu_xthread, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_ythread, gpu_ythread, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES,
               cudaMemcpyDeviceToHost);
    printf("\nKernel %d\n", kernel);
    // Iterate through the arrays and print
    for (int y = 0; y < ARRAY_SIZE_Y; y++) {
      for (int x = 0; x < ARRAY_SIZE_X; x++) {
        printf("CT %3u BKX %1u BKY %1u TID %2u YTID %2u XTID %2u "
          "GDX %1u BDX %1u GDY %1u BDY %1u\n",
          cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x],
          cpu_thread[y][x], cpu_ythread[y][x], cpu_xthread[y][x],
          cpu_grid_dimx[y][x], cpu_block_dimx[y][x],
          cpu_grid_dimy[y][x], cpu_block_dimy[y][x]);
      }
    }
  }

  // Free the arrays on the GPU as now we're done with them
  cudaFree(gpu_block_x);
  cudaFree(gpu_block_y);
  cudaFree(gpu_thread);
  cudaFree(gpu_calc_thread);
  cudaFree(gpu_xthread);
  cudaFree(gpu_ythread);
  cudaFree(gpu_grid_dimx);
  cudaFree(gpu_block_dimx);
  cudaFree(gpu_grid_dimy);
  cudaFree(gpu_block_dimy);

  return 0;
}
