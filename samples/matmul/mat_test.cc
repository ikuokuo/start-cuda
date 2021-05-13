#include <string>
#include <tuple>
#include <vector>

#include "mat.hpp"

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;
  auto h1 = std::string(80, '=');
  auto h2 = std::string(60, '-');
  {
    MatrixC m(10, 10);
    m.Fill([](int r, int c) { return r*10 + c; });
    std::vector<std::tuple<int, int>> limits{
      {-1, -1}, {0, 0}, {1, 1}, {2, 2}, {5, 5}, {10, 10}, {11, 11},
      {0, 5}, {5, 0}, {1, 5}, {5, 1},
    };
    int lim_w, lim_h;
    for (auto &&lim : limits) {
      std::tie(lim_w, lim_h) = lim;
      std::cout << "M.Print HxW=" << lim_h << "x" << lim_w << h2 << std::endl;
      m.Print(lim_w, lim_h);
      std::cout << std::endl;
    }
  }
  std::cout << h1 << std::endl;
  {
    MatrixC m(1024, 1024);
    m.Fill([](int r, int /*c*/) { return r; });
    m.Print(10, 10, MatrixC::PrintFormat(8, 2));

    MatrixC a(m);  // copy
    MatrixC b(std::move(m));  // move

    std::cout << std::endl << "m.Print " << h2 << std::endl;
    m.Print(10, 10, MatrixC::PrintFormat(8, 2));
    std::cout << std::endl << "a.Print " << h2 << std::endl;
    a.Print(10, 10, MatrixC::PrintFormat(8, 2));
    std::cout << std::endl << "b.Print " << h2 << std::endl;
    b.Print(10, 10, MatrixC::PrintFormat(8, 2));
  }
  return 0;
}
