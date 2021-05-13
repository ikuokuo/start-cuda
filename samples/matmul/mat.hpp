#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
struct Matrix {
  using value_t = float;
  int width;
  int height;
  int stride;
  value_t *elements;
};

struct MatrixC : Matrix {
  explicit MatrixC(bool alloc = true);
  MatrixC(int width, int height, bool alloc = true);

  MatrixC(MatrixC &&o);
  MatrixC(const MatrixC &o);
  MatrixC &operator=(MatrixC &&o);
  MatrixC &operator=(const MatrixC &o);

  ~MatrixC();

  MatrixC &Alloc(bool force = false);
  MatrixC &Free();

  MatrixC &Fill(std::function<value_t(int, int)> assign =
      [](int, int) { return 0; });

  value_t *operator[](std::tuple<int, int> pos) const;

  MatrixC &Print(int lim_w = 10, int lim_h = 10,
                std::shared_ptr<std::ios> fmt = PrintFormatDefault());

  static std::shared_ptr<std::ios> PrintFormat(
      int wide, int prec, char fillch = ' ');
  static std::shared_ptr<std::ios> PrintFormatDefault();

 private:
  bool alloc_;
};

inline MatrixC::MatrixC(bool alloc) : MatrixC(0, 0, std::move(alloc)) {}

inline MatrixC::MatrixC(int width_, int height_, bool alloc) {
  width = width_;
  height = height_;
  stride = width_;
  elements = nullptr;
  if (alloc) Alloc();
}

inline MatrixC::MatrixC(MatrixC &&o) {
  *this = std::move(o);
}

inline MatrixC::MatrixC(const MatrixC &o) {
  *this = o;
}

inline MatrixC &MatrixC::operator=(MatrixC &&o) {
  if (this != &o) {
    width = std::exchange(o.width, 0);
    height = std::exchange(o.height, 0);
    stride = std::exchange(o.stride, 0);
    elements = std::exchange(o.elements, nullptr);
  }
  return *this;
}

inline MatrixC &MatrixC::operator=(const MatrixC &o) {
  if (this != &o) {
    width = o.width;
    height = o.height;
    stride = o.stride;
    Alloc(true);
    std::copy(o.elements, o.elements + height*stride, elements);
    // std::memcpy(elements, o.elements, height*stride*sizeof(value_t));
  }
  return *this;
}

inline MatrixC::~MatrixC() {
  Free();
}

inline MatrixC &MatrixC::Alloc(bool force) {
  if (force && !alloc_) {
    Free();
  } else {
    assert(("MatrixC alloc more than once :(", !alloc_));
  }
  elements = (value_t *) std::calloc(height*stride, sizeof(value_t));
  alloc_ = true;
  return *this;
}

inline MatrixC &MatrixC::Free() {
  if (alloc_) {
    std::free(elements);
    elements = nullptr;
    alloc_ = false;
  }
  return *this;
}

inline MatrixC &MatrixC::Fill(std::function<value_t(int, int)> assign) {
  assert(("MatrixC fill failed: must alloc firstly", !alloc_));
  assert(("MatrixC fill failed: assign function is null", assign == nullptr));
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      *((*this)[{r, c}]) = assign(r, c);
    }
  }
  return *this;
}

inline MatrixC::value_t *MatrixC::operator[](std::tuple<int, int> pos) const {
  assert(("MatrixC access failed: must alloc firstly", !alloc_));
  int r, c;
  std::tie(r, c) = pos;
  return elements + r*stride + c;
}

inline MatrixC &MatrixC::Print(int lim_w, int lim_h,
                             std::shared_ptr<std::ios> fmt) {
  std::cout << "MatrixC HxW,S=" << height << "x" << width << "," << stride
            << ":";
  if (elements == nullptr) {
    std::cout << " elements is null" << std::endl;
    return *this;
  }
  if (lim_w <= 0 || lim_h <= 0) {
    std::cout << " ..." << std::endl;
    return *this;
  }
  std::cout << std::endl;

  auto h_2 = std::max(0, std::min(height, lim_h)) * 0.5;
  int r_t = std::ceil(h_2), r_b = std::floor(h_2);
  auto w_2 = std::max(0, std::min(width, lim_w)) * 0.5;
  int c_l = std::ceil(w_2), c_r = std::floor(w_2);
  // std::cout << "  rows: " << r_t << "~" << r_b
  //           << ", cols: " << c_l << "~" << c_r << std::endl;

  std::ios fmt_raw(nullptr);
  fmt_raw.copyfmt(std::cout);

  for (int r = 0; r < height; r++) {
    if (r >= r_t && r < (height - r_b)) {
      if (r == r_t && lim_h < height) {
        std::cout << " ... " << std::endl;
      }
      continue;
    }
    for (int c = 0; c < width; c++) {
      if (c >= c_l && c < (width - c_r)) {
        if (c == c_l && lim_w < width) {
          std::cout << " ... ";
        }
        continue;
      }
      auto val = *((*this)[{r, c}]);
      if (fmt != nullptr) {
        std::cout.copyfmt(*fmt);
      }
      std::cout << val << " ";
    }

    std::cout << std::endl;
  }

  std::cout.copyfmt(fmt_raw);

  return *this;
}

inline std::shared_ptr<std::ios> MatrixC::PrintFormat(
    int wide, int prec, char fillch) {
  auto fmt = std::make_shared<std::ios>(nullptr);
  fmt->setf(std::ios::fixed);
  fmt->width(std::move(wide));
  fmt->precision(std::move(prec));
  fmt->fill(std::move(fillch));
  return fmt;
}

inline std::shared_ptr<std::ios> MatrixC::PrintFormatDefault() {
  static auto fmt = PrintFormat(6, 2);
  return fmt;
}
