#include "decoder.h"

#include <fstream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "common/timing.hpp"
#include "../samples.hpp"

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  auto timing = times::Timing::Create("Read");
  auto img_path = MY_DATA_DIR"/lena.jpg";
  // auto img = cv::imread(img_path, cv::IMREAD_COLOR);
  std::vector<char> raw_data;
  size_t raw_len;
  {
    // Read an image from disk.
    std::ifstream input(img_path,
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
      std::cerr << "Cannot open image: " << img_path << std::endl;
      return EXIT_FAILURE;
    }

    // Get the size
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    // resize if buffer is too small
    if (raw_data.size() < (size_t)file_size) {
      raw_data.resize(file_size);
    }
    if (!input.read(raw_data.data(), file_size)) {
      std::cerr << "Cannot read from file: " << img_path << std::endl;
      return EXIT_FAILURE;
    }
    raw_len = file_size;
  }
  std::cout << "Read: " << img_path << std::endl;
  timing->DumpToLog();

  timing->Reset("NvJpegDecoder");
  // auto out_fmt = NVJPEG_OUTPUT_BGR;
  auto out_fmt = NVJPEG_OUTPUT_YUV;
  int out_w, out_h;
  NvJpegDecoder nvjpeg_decoder{};
  // decode more times, as cudaMalloc cost time for the first time
  for (int i = 0; i < 3; i++) {
    auto decode_ok = nvjpeg_decoder(
        (const unsigned char *)raw_data.data(), raw_len,
        out_fmt, &out_w, &out_h);
    if (decode_ok) {
      std::cerr << "nvjpeg decode ok: " << out_w << "x" << out_h << std::endl;
    } else {
      std::cerr << "nvjpeg decode fail" << std::endl;
    }
    timing->AddSplit("decode " + std::to_string(i));
  }
  auto out_i = nvjpeg_decoder.out_buf();

  cv::Mat bgr;
  if (out_fmt == NVJPEG_OUTPUT_BGR) {
    auto b = cv::Mat(out_h, out_w, CV_8UC1);
    auto g = cv::Mat(out_h, out_w, CV_8UC1);
    auto r = cv::Mat(out_h, out_w, CV_8UC1);
    int n = out_w * out_h;
    cudaMemcpy(b.data, out_i->channel[0], n, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data, out_i->channel[1], n, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.data, out_i->channel[2], n, cudaMemcpyDeviceToHost);
    cv::merge(std::vector<cv::Mat>{b, g, r}, bgr);
  } else if (out_fmt == NVJPEG_OUTPUT_YUV) {  // yuv422
    auto y = cv::Mat(out_h, out_w, CV_8UC1);
    auto uv = cv::Mat(out_h, out_w, CV_8UC1);
    int n = out_w * out_h, n_2 = n / 2;
    cudaMemcpy(y.data, out_i->channel[0], n, cudaMemcpyDeviceToHost);
    cudaMemcpy(uv.data, out_i->channel[1], n_2, cudaMemcpyDeviceToHost);
    cudaMemcpy(uv.data+n_2, out_i->channel[2], n_2, cudaMemcpyDeviceToHost);
    auto yuyv = cv::Mat(out_h, out_w, CV_8UC2);
    for (int i = 0; i < n; i++) {
      *(yuyv.data + 2*i) = *(y.data + i);           // y
    }
    for (int i = 0; i < n_2; i++) {
      *(yuyv.data + 4*i + 1) = *(uv.data + i);      // u
      *(yuyv.data + 4*i + 3) = *(uv.data+n_2 + i);  // v
    }
    cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
  }
  timing->AddSplit("convert");
  timing->DumpToLog();

  constexpr auto win_name = "lena";
  cv::namedWindow(win_name);
  cv::imshow(win_name, bgr);
  cv::waitKey();

  return 0;
}
