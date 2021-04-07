#include <algorithm>
#include <functional>
#include <iterator>
#include <iostream>
#include <vector>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>

#include "common/times.hpp"

namespace {

float time_cost(std::function<void()> func) {
  using namespace std;  // NOLINT
  using namespace times;  // NOLINT
  auto time_beg = now();
  func();
  auto time_end = now();
  float ms = count<std::chrono::microseconds>(time_end-time_beg) * 0.001f;
  cout << "BEG: " << to_local_string(time_beg) << endl
    << "END: " << to_local_string(time_end) << endl
    << "COST: " << ms << " ms" << endl;
  return ms;
}

// http://www.boost.org/doc/libs/1_63_0/libs/gil/test/
// http://stackoverflow.com/questions/8300555/using-boost-gil-to-convert-an-image-into-raw-bytes

struct Rgb8PixelReader {
  uint8_t *rgb8_pixels;
  explicit Rgb8PixelReader(uint8_t *pixels) : rgb8_pixels(pixels) {}
  void operator()(boost::gil::rgb8_pixel_t p) {
    using boost::gil::at_c;
    *rgb8_pixels++ = at_c<0>(p);
    *rgb8_pixels++ = at_c<1>(p);
    *rgb8_pixels++ = at_c<2>(p);
  }
};

struct Gray8PixelWriter {
  const uint8_t *gray_pixels;
  explicit Gray8PixelWriter(const uint8_t *pixels) : gray_pixels(pixels) {}
  void operator()(boost::gil::gray8_pixel_t &p) {
    using boost::gil::at_c;
    at_c<0>(p) = *gray_pixels++;
  }
};

}  // namespace

void cpu_grayscale(uint8_t * const rgb8_pixels, int rgb8_size,
                   uint8_t * const gray8_pixels, int gray8_size) {
  int rbg8_i = 0;
  for (int gray8_i = 0; gray8_i < gray8_size; ++gray8_i) {
    rbg8_i = 3 * gray8_i;
    gray8_pixels[gray8_i] = 0.299*rgb8_pixels[rbg8_i] +
      0.587*rgb8_pixels[rbg8_i+1] +
      0.114*rgb8_pixels[rbg8_i+2];
  }
}

void gpu_grayscale(uint8_t * const rgb8_pixels, int rgb8_size,
                   uint8_t * const gray8_pixels, int gray8_size);

// http://stackoverflow.com/questions/2410976/how-to-define-a-string-literal-in-gcc-command-line
#ifndef STR
  #define STR(x) #x
#endif
#ifndef STRINGIFY
  #define STRINGIFY(x) STR(x)
#endif

#ifndef MY_DATA_DIR
  #define MY_DATA_DIR STRINGIFY(MY_SAMPLES_DIR)"/data"
#endif
#ifndef MY_OUTPUT_DIR
  #define MY_OUTPUT_DIR STRINGIFY(MY_SAMPLES_DIR)"/_output"
#endif

int main(int argc, char const *argv[]) {
  using namespace std;  // NOLINT
  using namespace boost::gil;  // NOLINT

  cout << "Read: " << MY_DATA_DIR"/lena.jpg" << endl;
  rgb8_image_t img_lena;
  jpeg_read_image(MY_DATA_DIR"/lena.jpg", img_lena);
  const int w = img_lena.width(), h = img_lena.height();
  cout << "Size: " << w << "x" << h << endl;

  cout << "### Test Boost with Write ###" << endl;
  time_cost([&img_lena]() {
    jpeg_write_view(MY_OUTPUT_DIR"/lena_gray_boost.jpg",
      color_converted_view<gray8_pixel_t>(const_view(img_lena)));
  });

  {
    const int size = w * h;
    const int rgb8_size = size * num_channels<rgb8_image_t>();
    const int gray8_size = size * num_channels<gray8_image_t>();
    // cout << "rgb8: " << rgb8_size << ", gray8: " << gray8_size << endl;

    // rgb8 pixels
    uint8_t *rgb8_pixels = new uint8_t[rgb8_size];
    // read rgb8 pixels
    for_each_pixel(const_view(img_lena), Rgb8PixelReader(rgb8_pixels));

    // gray8 pixels
    uint8_t *gray8_pixels = new uint8_t[gray8_size];
    // gray image
    gray8_image_t img_gray(w, h);

    // grayscale pixels with gpu
    cout << "### Test GPU without Write ###" << endl;
    time_cost([&rgb8_pixels, &rgb8_size, &gray8_pixels, &gray8_size]() {
      gpu_grayscale(rgb8_pixels, rgb8_size, gray8_pixels, gray8_size);
    });
    // write gray8 pixels
    for_each_pixel(view(img_gray), Gray8PixelWriter(gray8_pixels));
    jpeg_write_view(MY_OUTPUT_DIR"/lena_gray_gpu.jpg",
      color_converted_view<rgb8_pixel_t>(view(img_gray)));

    // grayscale pixels with cpu
    cout << "### Test CPU without Write ###" << endl;
    time_cost([&rgb8_pixels, &rgb8_size, &gray8_pixels, &gray8_size]() {
      cpu_grayscale(rgb8_pixels, rgb8_size, gray8_pixels, gray8_size);
    });
    // write gray8 pixels
    for_each_pixel(view(img_gray), Gray8PixelWriter(gray8_pixels));
    jpeg_write_view(MY_OUTPUT_DIR"/lena_gray_cpu.jpg",
      color_converted_view<rgb8_pixel_t>(view(img_gray)));

    delete[] rgb8_pixels;
    delete[] gray8_pixels;
  }

  return 0;
}
