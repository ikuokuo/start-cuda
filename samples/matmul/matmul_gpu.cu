#include <cuda_runtime.h>

#include <string>

#define TIME_PRECISION 6
#include "common/time_cost.hpp"
#include "mat.hpp"

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const MatrixC &A, const MatrixC &B, MatrixC *C) {
  TIME_BEG_FUNC2;
  // Load A and B to device memory
  Matrix d_A;
  d_A.width = A.width; d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
  Matrix d_B;
  d_B.width = B.width; d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

  // Allocate C in device memory
  Matrix d_C;
  d_C.width = C->width; d_C.height = C->height;
  size = C->width * C->height * sizeof(float);
  cudaMalloc(&d_C.elements, size);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  TIME_BEG_FUNC("MatMulKernel");
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  TIME_END_FUNC("MatMulKernel");

  // Read C from device memory
  cudaMemcpy(C->elements, d_C.elements, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
  TIME_END_FUNC2;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
  // Each thread computes one element of C
  // by accumulating results into Cvalue
  float Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e]
            * B.elements[e * B.width + col];
  C.elements[row * C.width + col] = Cvalue;
}

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  std::string dashes(60, '-');

  // Matrix on CPU

  int m_w = 1024, m_h = 1024;

  std::cout << "A" << dashes << std::endl;
  MatrixC A(m_w, m_h);
  A.Fill([](int r, int c) {
    if (r == 0) return c;
    if (c == 0) return r;
    return 0;
  }).Print(10, 10, MatrixC::PrintFormat(4, 0));

  std::cout << std::endl << "B" << dashes << std::endl;
  MatrixC B(A);
  B.Print(10, 10, MatrixC::PrintFormat(4, 0));

  // MatMul

  std::cout << std::endl << "C=AB" << dashes << std::endl;
  MatrixC C(m_w, m_h);
  MatMul(A, B, &C);
  C.Print(10, 10, MatrixC::PrintFormat(9, 0));

  return 0;
}
