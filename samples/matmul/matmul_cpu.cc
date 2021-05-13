#include <string>

#include "common/time_cost.hpp"
#include "mat.hpp"

void MatMul(const MatrixC &A, const MatrixC &B, MatrixC *C) {
  TIME_BEG_FUNC2;
  for (int r = 0; r < C->height; r++) {
    for (int c = 0; c < C->width; c++) {
      MatrixC::value_t val = 0;
      for (int i = 0; i < C->width; i++) {
        val += *A[{r, i}] + *B[{i, c}];
      }
      *((*C)[{r, c}]) = val;
    }
  }
  TIME_END_FUNC2;
}

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  std::string dashes(60, '-');

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

  std::cout << std::endl << "C=AB" << dashes << std::endl;
  MatrixC C(m_w, m_h);
  MatMul(A, B, &C);
  C.Print(10, 10, MatrixC::PrintFormat(9, 0));

  return 0;
}
