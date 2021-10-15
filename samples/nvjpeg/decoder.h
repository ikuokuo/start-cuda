#pragma once

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <iostream>

#define CHECK_CUDA(call)                                                       \
{                                                                              \
  cudaError_t _e = (call);                                                     \
  if (_e != cudaSuccess) {                                                     \
    std::cerr << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__      \
        << ":" << __LINE__ << std::endl;                                       \
    exit(1);                                                                   \
  }                                                                            \
}

#define CHECK_NVJPEG(call)                                                     \
{                                                                              \
  nvjpegStatus_t _e = (call);                                                  \
  if (_e != NVJPEG_STATUS_SUCCESS) {                                           \
    std::cerr << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__            \
        << ":" << __LINE__ << std::endl;                                       \
    exit(1);                                                                   \
  }                                                                            \
}

inline int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
inline int dev_free(void *p) { return (int)cudaFree(p); }
inline int host_malloc(void **p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }  // NOLINT
inline int host_free(void *p) { return (int)cudaFreeHost(p); }

class NvJpegDecoder {
 public:
  NvJpegDecoder() {
    dev_allocator_ = {&dev_malloc, &dev_free};
    pinned_allocator_ = {&host_malloc, &host_free};

    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT,
        &dev_allocator_, &pinned_allocator_, NVJPEG_FLAGS_DEFAULT,
        &nvjpeg_handle_));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    // initialize buffer
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      out_buf_.channel[c] = NULL;
      out_buf_.pitch[c] = 0;
      out_buf_sz_.pitch[c] = 0;
    }
  }

  ~NvJpegDecoder() {
    CHECK_CUDA(cudaStreamDestroy(stream_));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state_));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle_));
    // release buffer
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
      if (out_buf_.channel[c]) CHECK_CUDA(cudaFree(out_buf_.channel[c]));
  }

  bool operator()(const unsigned char *data, size_t length,
      nvjpegOutputFormat_t fmt, int *out_w, int *out_h,
      nvjpegImage_t *out_i = nullptr) {
    if (PrepareBuffer(data, length, fmt, out_w, out_h)) {
      return false;
    }
    if (DecodeImage(data, length, fmt)) {
      return false;
    }
    if (out_i) {
      for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
        out_i->channel[c] = out_buf_.channel[c];
        out_i->pitch[c] = out_buf_.pitch[c];
      }
    }
    return true;
  }

  nvjpegImage_t *out_buf() { return &out_buf_; }

 private:
  int PrepareBuffer(const unsigned char *data, size_t length,
      nvjpegOutputFormat_t fmt, int *out_w, int *out_h) {
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;

    CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle_, data, length,
        &channels, &subsampling, widths, heights));
    *out_w = widths[0];
    *out_h = heights[0];

    {  // print image info below
      std::cout << "Image is " << channels << " channels." << std::endl;
      for (int c = 0; c < channels; c++) {
        std::cout << "Channel #" << c << " size: " << widths[c] << " x "
                  << heights[c] << std::endl;
      }

      switch (subsampling) {
        case NVJPEG_CSS_444:
          std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
          break;
        case NVJPEG_CSS_440:
          std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
          break;
        case NVJPEG_CSS_422:
          std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
          break;
        case NVJPEG_CSS_420:
          std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
          break;
        case NVJPEG_CSS_411:
          std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
          break;
        case NVJPEG_CSS_410:
          std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
          break;
        case NVJPEG_CSS_GRAY:
          std::cout << "Grayscale JPEG " << std::endl;
          break;
        case NVJPEG_CSS_UNKNOWN:
          std::cout << "Unknown chroma subsampling" << std::endl;
          return EXIT_FAILURE;
      }
    }

    int mul = 1;
    if (fmt == NVJPEG_OUTPUT_RGBI || fmt == NVJPEG_OUTPUT_BGRI) {
      // in the case of interleaved RGB output, write only to single channel,
      // but 3 samples at once
      channels = 1;
      mul = 3;
    } else if (fmt == NVJPEG_OUTPUT_RGB || fmt == NVJPEG_OUTPUT_BGR) {
      // in the case of rgb create 3 buffers with sizes of original image
      channels = 3;
      widths[1] = widths[2] = widths[0];
      heights[1] = heights[2] = heights[0];
    }

    // realloc output buffer if required
    for (int c = 0; c < channels; c++) {
      int aw = mul * widths[c];
      int ah = heights[c];
      size_t sz = aw * ah;
      out_buf_.pitch[c] = aw;
      if (sz > out_buf_sz_.pitch[c]) {
        if (out_buf_.channel[c]) {
          CHECK_CUDA(cudaFree(out_buf_.channel[c]));
        }
        CHECK_CUDA(cudaMalloc((void **)&out_buf_.channel[c], sz));
        out_buf_sz_.pitch[c] = sz;
      }
    }

    return EXIT_SUCCESS;
  }

  int DecodeImage(const unsigned char *data, size_t length,
      nvjpegOutputFormat_t fmt) {
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float time = 0;
    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    CHECK_CUDA(cudaEventRecord(startEvent, stream_));
    {
      CHECK_NVJPEG(nvjpegDecode(nvjpeg_handle_, nvjpeg_state_, data, length,
          fmt, &out_buf_, stream_));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, stream_));

    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&time, startEvent, stopEvent));
    std::cout << "DecodeImage cuda cost: " << time << " ms" << std::endl;
    return EXIT_SUCCESS;
  }

  nvjpegDevAllocator_t dev_allocator_;
  nvjpegPinnedAllocator_t pinned_allocator_;
  nvjpegJpegState_t nvjpeg_state_;
  nvjpegHandle_t nvjpeg_handle_;
  cudaStream_t stream_;

  // output buffer
  nvjpegImage_t out_buf_;
  // output buffer size, for convenience
  nvjpegImage_t out_buf_sz_;
};

// nvJPEG Documentation
//  https://docs.nvidia.com/cuda/nvjpeg/index.html
// nvJPEG Examples
//  https://github.com/NVIDIA/CUDALibrarySamples/tree/master/nvJPEG
