#include <algorithm>
#include <iostream>
#include <iterator>

#include "common/time_cost.hpp"

#define ARRAY_SIZE 256

void array_add(const int * const lhs,
               const int * const rhs,
               int * const sum,
               size_t n) {
  TIME_BEG(__func__);
  for (size_t i = 0; i < n; ++i) {
    sum[i] = lhs[i] + rhs[i];
  }
  TIME_END(__func__);
}

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  int lhs[ARRAY_SIZE];
  int rhs[ARRAY_SIZE];
  int sum[ARRAY_SIZE];

  // fill values
  std::fill(lhs, lhs+ARRAY_SIZE, 2);
  std::fill(rhs, rhs+ARRAY_SIZE, 3);

  // sum = lhs + rhs
  array_add(lhs, rhs, sum, ARRAY_SIZE);

  // print sum
  std::cout << "sum[" << ARRAY_SIZE << "]:" << std::endl;
  std::copy(sum, sum+ARRAY_SIZE, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}
