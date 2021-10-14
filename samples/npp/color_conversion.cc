#include <npp.h>
#include <nppi.h>

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "common/timing.hpp"
#include "../samples.hpp"

namespace {

void PrintfNPPinfo() {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);
}

void PutText(const cv::Mat &img, const std::string &text) {
  int baseline = 0;
  int margin = 10;
  cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN,
    1, 1, &baseline);
  cv::putText(img, text, cv::Point(margin, margin + size.height),
    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255));
}

void ShowImage(const std::string &name,
    const cv::Mat &img,
    const std::string &text,
    bool clone = true,
    int delay = 500) {
  cv::Mat mat = clone ? img.clone() : img;
  PutText(mat, text);
  cv::imshow(name, mat);
  if (delay >= 0) {
    cv::waitKey(delay);
  }
}

cv::Mat BGR2YUV_NV12(const cv::Mat &src) {
  auto src_h = src.rows, src_w = src.cols;
  cv::Mat dst(src_h * 1.5, src_w, CV_8UC1);
  cv::cvtColor(src, dst, cv::COLOR_BGR2YUV_I420);
  auto n_y = src_h * src_w;
  auto n_uv = n_y / 2, n_u = n_y / 4;
  std::vector<uint8_t> uv(n_uv);
  std::copy(dst.data+n_y, dst.data+n_y+n_uv, uv.data());
  for (auto i = 0; i < n_u; i++) {
    dst.data[n_y + 2*i] = uv[i];            // U
    dst.data[n_y + 2*i + 1] = uv[n_u + i];  // V
  }
  return dst;
}

}  // namespace

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  auto timing = times::Timing::Create("Read");
  auto img_path = MY_DATA_DIR"/lena.jpg";
  auto img = cv::imread(img_path, cv::IMREAD_COLOR);
  std::cout << "Read: " << img_path << std::endl;
  timing->DumpToLog();

  constexpr auto win_name = "lena";
  cv::namedWindow(win_name);
  ShowImage(win_name, img, "RAW");

  // OpenCV imgproc
  std::cout << std::endl << "OpenCV imgproc ---------------------" << std::endl;
  {
    cv::Mat bgr = img;
    cv::Mat yuv;

    timing->Reset("BGR > YUV, cvtColor");
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);
    timing->DumpToLog();
    std::cout << "  YUV: " << yuv.cols << "x" << yuv.rows << " "
      << yuv.channels() << std::endl;
    ShowImage(win_name, yuv, "BGR > YUV, cv::COLOR_BGR2YUV");

    timing->Reset("YUV > BGR, cvtColor");
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR);
    timing->DumpToLog();
    std::cout << "  BGR: " << bgr.cols << "x" << bgr.rows << " "
      << bgr.channels() << std::endl;
    ShowImage(win_name, bgr, "YUV > BGR, cv::COLOR_YUV2BGR");
  }

  // npp color conversion
  std::cout << std::endl << "NPP color conversion ---------------" << std::endl;
  PrintfNPPinfo();

  auto img_h = img.rows, img_w = img.cols, img_c = img.channels();
  auto img_n = img_h * img_w * img_c;
  auto img_bytes = img.total() * img.elemSize();

  cv::Mat dst(img_h, img_w, CV_8UC3);
  auto dst_bytes = dst.total() * dst.elemSize();

  // NppStatus nppiBGRToYUV_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);  NOLINT
  timing->Reset("BGR > YUV, npp");
  {
    Npp8u *pSrc;
    cudaMalloc((void **)&pSrc, img_bytes);

    Npp8u *pDst;
    cudaMalloc((void **)&pDst, img_n * sizeof(Npp8u));

    timing->AddSplit("cudaMalloc");

    cudaMemcpy(pSrc, img.data, img_bytes, cudaMemcpyHostToDevice);

    timing->AddSplit("cudaMemcpyHostToDevice");

    auto nSrcStep = img_w * img_c, nDstStep = img_w * img_c;
    NppiSize oSizeROI{img_w, img_h};
    NppStatus ret = nppiBGRToYUV_8u_C3R(pSrc, nSrcStep, pDst, nDstStep,
                                        oSizeROI);
    if (ret != NPP_SUCCESS) {
      std::cerr << "nppiBGRToYUV_8u_C3R failed: " << ret << std::endl;
    }

    timing->AddSplit("nppiBGRToYUV_8u_C3R");

    cudaMemcpy(dst.data, pDst, dst_bytes, cudaMemcpyDeviceToHost);

    timing->AddSplit("cudaMemcpyDeviceToHost");

    cudaFree(pSrc);
    cudaFree(pDst);

    timing->AddSplit("cudaFree");
  }
  timing->DumpToLog();
  ShowImage(win_name, dst, "BGR > YUV, nppiBGRToYUV_8u_C3R");

  // NppStatus nppiYUVToBGR_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);  NOLINT
  timing->Reset("YUV > BGR, npp");
  {
    auto yuv = dst;
    auto yuv_bytes = yuv.total() * yuv.elemSize();

    Npp8u *pSrc;
    cudaMalloc((void **)&pSrc, yuv_bytes);

    Npp8u *pDst;
    cudaMalloc((void **)&pDst, img_n * sizeof(Npp8u));

    timing->AddSplit("cudaMalloc");

    cudaMemcpy(pSrc, yuv.data, yuv_bytes, cudaMemcpyHostToDevice);

    timing->AddSplit("cudaMemcpyHostToDevice");

    auto nSrcStep = img_w * img_c, nDstStep = img_w * img_c;
    NppiSize oSizeROI{img_w, img_h};
    NppStatus ret = nppiYUVToBGR_8u_C3R(pSrc, nSrcStep, pDst, nDstStep,
                                        oSizeROI);
    if (ret != NPP_SUCCESS) {
      std::cerr << "nppiYUVToBGR_8u_C3R failed: " << ret << std::endl;
    }

    timing->AddSplit("nppiYUVToBGR_8u_C3R");

    cudaMemcpy(dst.data, pDst, dst_bytes, cudaMemcpyDeviceToHost);

    timing->AddSplit("cudaMemcpyDeviceToHost");

    cudaFree(pSrc);
    cudaFree(pDst);

    timing->AddSplit("cudaFree");
  }
  timing->DumpToLog();
  ShowImage(win_name, dst, "YUV > BGR, nppiYUVToBGR_8u_C3R");

  std::cout << std::endl << "NV12 <> BGR ------------------------" << std::endl;

  timing->Reset("BGR > NV12, custom");
  auto nv12 = BGR2YUV_NV12(img);
  timing->DumpToLog();
  ShowImage(win_name, nv12, "BGR > NV12, BGR2YUV_NV12");

  timing->Reset("NV12 > BGR, cvtColor");
  {
    cv::cvtColor(nv12, dst, cv::COLOR_YUV2BGR_NV12);
  }
  timing->DumpToLog();
  ShowImage(win_name, dst, "NV12 > BGR, cv::COLOR_YUV2BGR_NV12");

  // NppStatus nppiNV12ToBGR_8u_P2C3R(const Npp8u * const pSrc[2], int rSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);  NOLINT
  timing->Reset("NV12 > BGR, npp");
  {
    auto nv12_bytes = nv12.total() * nv12.elemSize();

    Npp8u *d_nv12;
    cudaMalloc((void **)&d_nv12, nv12_bytes);

    Npp8u *d_bgr;
    cudaMalloc((void **)&d_bgr, img_n * sizeof(Npp8u));

    timing->AddSplit("cudaMalloc");

    cudaMemcpy(d_nv12, nv12.data, nv12_bytes, cudaMemcpyHostToDevice);

    timing->AddSplit("cudaMemcpyHostToDevice");

    const Npp8u * const pSrc[2] {
      d_nv12,
      d_nv12 + (img_w * img_h),
    };
    Npp8u *pDst = d_bgr;
    auto nSrcStep = img_w, nDstStep = img_w * img_c;
    NppiSize oSizeROI{img_w, img_h};
    NppStatus ret = nppiNV12ToBGR_8u_P2C3R(pSrc, nSrcStep, pDst, nDstStep,
                                           oSizeROI);
    if (ret != NPP_SUCCESS) {
      std::cerr << "nppiNV12ToBGR_8u_P2C3R failed: " << ret << std::endl;
    }

    timing->AddSplit("nppiNV12ToBGR_8u_P2C3R");

    cudaMemcpy(dst.data, pDst, dst_bytes, cudaMemcpyDeviceToHost);

    timing->AddSplit("cudaMemcpyDeviceToHost");

    cudaFree(d_nv12);
    cudaFree(d_bgr);

    timing->AddSplit("cudaFree");
  }
  timing->DumpToLog();
  ShowImage(win_name, dst, "NV12 > BGR, nppiNV12ToBGR_8u_P2C3R");

  cv::destroyAllWindows();
  return 0;
}
