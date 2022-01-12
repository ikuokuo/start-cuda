#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kWARNING)
    : reportable_severity_(severity) {
  }

  void log(Severity severity, char const* msg) noexcept override {
    if (severity > reportable_severity_) return;
    switch (severity) {
    case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR"; break;
    case Severity::kERROR:          std::cerr << "ERROR"; break;
    case Severity::kWARNING:        std::cerr << "WARNING"; break;
    case Severity::kINFO:           std::cerr << "INFO"; break;
    case Severity::kVERBOSE:        std::cerr << "VERBOSE"; break;
    default:                        std::cerr << "UNKNOWN"; break;
    }
    std::cerr << ": " << msg << std::endl;
  }

 private:
  Severity reportable_severity_;
};

class RVM {
 public:
  RVM();
  bool Load(const std::string &engine_filename);
  bool InferFP32(const std::string &input_filename,
      const std::string& output_filename);
  bool InferFP16(const std::string &input_filename,
      const std::string& output_filename);
  void PrintEngineInfo();

 private:
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};

RVM::RVM() {
}

bool RVM::Load(const std::string &engine_filename) {
  std::ifstream engine_file(engine_filename, std::ios::binary);
  if (engine_file.fail()) {
    std::cerr << "ERROR: engine file load failed" << std::endl;
    return false;
  }

  engine_file.seekg(0, std::ifstream::end);
  auto fsize = engine_file.tellg();
  engine_file.seekg(0, std::ifstream::beg);

  std::vector<char> engine_data(fsize);
  engine_file.read(engine_data.data(), fsize);

  static Logger logger{Logger::Severity::kINFO};
  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger));
  engine_.reset(
      runtime->deserializeCudaEngine(engine_data.data(), fsize, nullptr));
  if (engine_ == nullptr) {
    std::cerr << "ERROR: engine data deserialize failed" << std::endl;
    return false;
  }
  return true;
}

bool RVM::InferFP32(const std::string &input_filename,
    const std::string& output_filename) {
  assert(engine_ != nullptr);
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  if (!context) {
    return false;
  }

  // Bindings
  //  Input[0] name=src dims=[1,3,1080,1920] datatype=FLOAT
  //  Input[1] name=r1i dims=[1,1,1,1] datatype=FLOAT
  //  Input[2] name=r2i dims=[1,1,1,1] datatype=FLOAT
  //  Input[3] name=r3i dims=[1,1,1,1] datatype=FLOAT
  //  Input[4] name=r4i dims=[1,1,1,1] datatype=FLOAT
  //  Output[5] name=r4o dims=[1,64,18,32] datatype=FLOAT
  //  Output[6] name=r3o dims=[1,40,36,64] datatype=FLOAT
  //  Output[7] name=r2o dims=[1,20,72,128] datatype=FLOAT
  //  Output[8] name=r1o dims=[1,16,144,256] datatype=FLOAT
  //  Output[9] name=fgr dims=[1,3,1080,1920] datatype=FLOAT
  //  Output[10] name=pha dims=[1,1,1080,1920] datatype=FLOAT

  auto GetMemorySize = [](const nvinfer1::Dims &dims,
                          const int32_t elem_size) -> int32_t {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1,
        std::multiplies<int64_t>()) * elem_size;
  };

  auto nb = engine_->getNbBindings();

  // Allocate CUDA memory for all bindings
  std::vector<void *> bindings(nb, nullptr);
  std::vector<int32_t> bindings_size(nb, 0);
  for (int32_t i = 0; i < nb; i++) {
    assert(engine_->getBindingDataType(i) == nvinfer1::DataType::kFLOAT);  // fp32 NOLINT

    auto dims = engine_->getBindingDimensions(i);
    auto size = GetMemorySize(dims, sizeof(float));
    if (cudaMalloc(&bindings[i], size) != cudaSuccess) {
      std::cerr << "ERROR: cuda memory allocation failed, size = " << size
          << " bytes" << std::endl;
      return false;
    }
    bindings_size[i] = size;
  }

  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::cerr << "ERROR: cuda stream creation failed" << std::endl;
    return false;
  }

  auto src_dims = engine_->getBindingDimensions(0);
  auto src_h = src_dims.d[2], src_w = src_dims.d[3];
  auto src_n = src_h * src_w;
  auto dst_h = 0, dst_w = 0;

  // Copy data to input binding memory
  {
    // img: HWC BGR [0,255] u8
    auto img = cv::imread(input_filename, cv::IMREAD_COLOR);
    assert(img.type() == CV_8UC3);
    dst_h = img.rows;
    dst_w = img.cols;

    if (src_h != img.rows || src_w != img.cols) {
      cv::resize(img, img, cv::Size(src_w, src_h));
    }

    // src: BCHW RGB [0,1] fp32
    auto src = cv::Mat(img.rows, img.cols, CV_32FC3);
    assert(src.total()*src.elemSize() == bindings_size[0]);
    {
      auto src_data = (float*)(src.data);
      for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
          auto &&bgr = img.at<cv::Vec3b>(y, x);
          /*r*/ *(src_data + y*src_w + x) = bgr[2] / 255.;
          /*g*/ *(src_data + src_n + y*src_w + x) = bgr[1] / 255.;
          /*b*/ *(src_data + src_n*2 + y*src_w + x) = bgr[0] / 255.;
        }
      }
    }
    if (cudaMemcpyAsync(bindings[0], src.data, bindings_size[0],
        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      std::cerr << "ERROR: CUDA memory copy of src failed, size = "
          << bindings_size[0] << " bytes" << std::endl;
      return false;
    }

    // r1i, r2i, r3i, r4i
    cv::Mat ret = cv::Mat::zeros(4, 4, CV_32FC1);
    for (int32_t i = 1; i < 5; i++) {
      if (cudaMemcpyAsync(bindings[i], ret.data+((i-1)*16), bindings_size[i],
          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "ERROR: CUDA memory copy of r" << i << "i failed, size = "
            << bindings_size[i] << " bytes" << std::endl;
        return false;
      }
    }
  }

  // Run TensorRT inference
  bool status = context->enqueueV2(bindings.data(), stream, nullptr);
  if (!status) {
    std::cout << "ERROR: TensorRT inference failed" << std::endl;
    return false;
  }

  // Copy data from output binding memory
  auto fgr = cv::Mat(src_h, src_w, CV_32FC3);  // BCHW RGB [0,1] fp32
  if (cudaMemcpyAsync(fgr.data, bindings[9], bindings_size[9],
      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    std::cerr << "ERROR: CUDA memory copy of output failed, size = "
        << bindings_size[9] << " bytes" << std::endl;
    return false;
  }
  auto pha = cv::Mat(src_h, src_w, CV_32FC1);  // BCHW A [0,1] fp32
  if (cudaMemcpyAsync(pha.data, bindings[10], bindings_size[10],
      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    std::cerr << "ERROR: CUDA memory copy of output failed, size = "
        << bindings_size[10] << " bytes" << std::endl;
    return false;
  }
  cudaStreamSynchronize(stream);

  // Compose `fgr` and `pha`
  assert(fgr.type() == CV_32FC3);
  assert(pha.type() == CV_32FC1);
  auto com = cv::Mat(src_h, src_w, CV_8UC4);  // HWC BGRA [0,255] u8
  {
    auto fgr_data = (float*)(fgr.data);
    auto pha_data = (float*)(pha.data);
    for (int y = 0; y < com.rows; ++y) {
      for (int x = 0; x < com.cols; ++x) {
        auto &&elem = com.at<cv::Vec4b>(y, x);
        auto alpha = *(pha_data + y*src_w + x);
        if (alpha > 0) {
          /*r*/ elem[2] = *(fgr_data + y*src_w + x) * 255;
          /*g*/ elem[1] = *(fgr_data + src_n + y*src_w + x) * 255;
          /*b*/ elem[0] = *(fgr_data + src_n*2 + y*src_w + x) * 255;
        } else {
          /*r*/ elem[2] = 0;
          /*g*/ elem[1] = 0;
          /*b*/ elem[0] = 0;
        }
        /*a*/ elem[3] = alpha * 255;
      }
    }
  }
  if (dst_h != com.rows || dst_w != com.cols) {
    cv::resize(com, com, cv::Size(dst_w, dst_h));
  }
  if (!output_filename.empty()) {
    cv::imwrite(output_filename, com);
    std::cout << "save to " << output_filename << std::endl;
  } else {
    auto win_name = "rvm_infer";
    cv::namedWindow(win_name);
    cv::imshow(win_name, com);
    cv::waitKey();
    cv::destroyWindow(win_name);
  }

  // Free CUDA resources
  for (int32_t i = 0; i < nb; i++) {
    cudaFree(bindings[i]);
  }
  return true;
}

bool RVM::InferFP16(const std::string &input_filename,
    const std::string& output_filename) {
  assert(engine_ != nullptr);
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  if (!context) {
    return false;
  }

  // Bindings
  //  Input[0] name=src dims=[1,3,1080,1920] datatype=HALF
  //  Input[1] name=r1i dims=[1,1,1,1] datatype=HALF
  //  Input[2] name=r2i dims=[1,1,1,1] datatype=HALF
  //  Input[3] name=r3i dims=[1,1,1,1] datatype=HALF
  //  Input[4] name=r4i dims=[1,1,1,1] datatype=HALF
  //  Output[5] name=r4o dims=[1,64,18,32] datatype=HALF
  //  Output[6] name=r3o dims=[1,40,36,64] datatype=HALF
  //  Output[7] name=r2o dims=[1,20,72,128] datatype=HALF
  //  Output[8] name=r1o dims=[1,16,144,256] datatype=HALF
  //  Output[9] name=fgr dims=[1,3,1080,1920] datatype=HALF
  //  Output[10] name=pha dims=[1,1,1080,1920] datatype=HALF

  auto GetMemorySize = [](const nvinfer1::Dims &dims,
                          const int32_t elem_size) -> int32_t {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1,
        std::multiplies<int64_t>()) * elem_size;
  };

  auto f2w = [](float f) -> ushort {
    return cv::float16_t(f).bits();
  };
  auto w2f = [](ushort w) -> float {
    return float(cv::float16_t::fromBits(w));
  };

  auto nb = engine_->getNbBindings();

  // Allocate CUDA memory for all bindings
  std::vector<void *> bindings(nb, nullptr);
  std::vector<int32_t> bindings_size(nb, 0);
  for (int32_t i = 0; i < nb; i++) {
    assert(engine_->getBindingDataType(i) == nvinfer1::DataType::kHALF);  // fp16 NOLINT

    auto dims = engine_->getBindingDimensions(i);
    auto size = GetMemorySize(dims, sizeof(float) / 2);
    if (cudaMalloc(&bindings[i], size) != cudaSuccess) {
      std::cerr << "ERROR: cuda memory allocation failed, size = " << size
          << " bytes" << std::endl;
      return false;
    }
    bindings_size[i] = size;
  }

  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::cerr << "ERROR: cuda stream creation failed" << std::endl;
    return false;
  }

  auto src_dims = engine_->getBindingDimensions(0);
  auto src_h = src_dims.d[2], src_w = src_dims.d[3];
  auto src_n = src_h * src_w;
  auto dst_h = 0, dst_w = 0;

  // Copy data to input binding memory
  {
    // img: HWC BGR [0,255] u8
    auto img = cv::imread(input_filename, cv::IMREAD_COLOR);
    assert(img.type() == CV_8UC3);
    dst_h = img.rows;
    dst_w = img.cols;

    if (src_h != img.rows || src_w != img.cols) {
      cv::resize(img, img, cv::Size(src_w, src_h));
    }

    // src: BCHW RGB [0,1] fp16
    auto src = cv::Mat(img.rows, img.cols, CV_16FC3);
    assert(src.total()*src.elemSize() == bindings_size[0]);
    {
      auto src_data = (ushort*)(src.data);
      for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
          auto &&bgr = img.at<cv::Vec3b>(y, x);
          /*r*/ *(src_data + y*src_w + x) = f2w(bgr[2]/255.);
          /*g*/ *(src_data + src_n + y*src_w + x) = f2w(bgr[1]/255.);
          /*b*/ *(src_data + src_n*2 + y*src_w + x) = f2w(bgr[0]/255.);
        }
      }
    }
    if (cudaMemcpyAsync(bindings[0], src.data, bindings_size[0],
        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      std::cerr << "ERROR: CUDA memory copy of src failed, size = "
          << bindings_size[0] << " bytes" << std::endl;
      return false;
    }

    // r1i, r2i, r3i, r4i
    cv::Mat ret = cv::Mat::zeros(4, 4, CV_16FC1);
    for (int32_t i = 1; i < 5; i++) {
      if (cudaMemcpyAsync(bindings[i], ret.data+((i-1)*8), bindings_size[i],
          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "ERROR: CUDA memory copy of r" << i << "i failed, size = "
            << bindings_size[i] << " bytes" << std::endl;
        return false;
      }
    }
  }

  // Run TensorRT inference
  bool status = context->enqueueV2(bindings.data(), stream, nullptr);
  if (!status) {
    std::cout << "ERROR: TensorRT inference failed" << std::endl;
    return false;
  }

  // Copy data from output binding memory
  auto fgr = cv::Mat(src_h, src_w, CV_16FC3);  // BCHW RGB [0,1] fp16
  if (cudaMemcpyAsync(fgr.data, bindings[9], bindings_size[9],
      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    std::cerr << "ERROR: CUDA memory copy of output failed, size = "
        << bindings_size[9] << " bytes" << std::endl;
    return false;
  }
  auto pha = cv::Mat(src_h, src_w, CV_16FC1);  // BCHW A [0,1] fp16
  if (cudaMemcpyAsync(pha.data, bindings[10], bindings_size[10],
      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    std::cerr << "ERROR: CUDA memory copy of output failed, size = "
        << bindings_size[10] << " bytes" << std::endl;
    return false;
  }
  cudaStreamSynchronize(stream);

  // Compose `fgr` and `pha`
  assert(fgr.type() == CV_16FC3);
  assert(pha.type() == CV_16FC1);
  auto com = cv::Mat(src_h, src_w, CV_8UC4);  // HWC BGRA [0,255] u8
  {
    auto fgr_data = (ushort*)(fgr.data);
    auto pha_data = (ushort*)(pha.data);
    for (int y = 0; y < com.rows; ++y) {
      for (int x = 0; x < com.cols; ++x) {
        auto &&elem = com.at<cv::Vec4b>(y, x);
        auto alpha = w2f(*(pha_data + y*src_w + x));
        if (alpha > 0) {
          /*r*/ elem[2] = w2f(*(fgr_data + y*src_w + x)) * 255;
          /*g*/ elem[1] = w2f(*(fgr_data + src_n + y*src_w + x)) * 255;
          /*b*/ elem[0] = w2f(*(fgr_data + src_n*2 + y*src_w + x)) * 255;
        } else {
          /*r*/ elem[2] = 0;
          /*g*/ elem[1] = 0;
          /*b*/ elem[0] = 0;
        }
        /*a*/ elem[3] = alpha * 255;
      }
    }
  }
  if (dst_h != com.rows || dst_w != com.cols) {
    cv::resize(com, com, cv::Size(dst_w, dst_h));
  }
  if (!output_filename.empty()) {
    cv::imwrite(output_filename, com);
    std::cout << "save to " << output_filename << std::endl;
  } else {
    auto win_name = "rvm_infer";
    cv::namedWindow(win_name);
    cv::imshow(win_name, com);
    cv::waitKey();
    cv::destroyWindow(win_name);
  }

  // Free CUDA resources
  for (int32_t i = 0; i < nb; i++) {
    cudaFree(bindings[i]);
  }
  return true;
}

void RVM::PrintEngineInfo() {
  assert(engine_ != nullptr);

  std::cout << "Engine" << std::endl;
  std::cout << " Name=" << engine_->getName() << std::endl;
  std::cout << " DeviceMemorySize=" << engine_->getDeviceMemorySize() / (1<< 20)
      << " MiB" << std::endl;
  std::cout << " MaxBatchSize=" << engine_->getMaxBatchSize() << std::endl;

  std::cout << "Bindings" << std::endl;
  auto nb = engine_->getNbBindings();
  for (int32_t i = 0; i < nb; i++) {
    auto is_input = engine_->bindingIsInput(i);
    auto name = engine_->getBindingName(i);
    auto dims = engine_->getBindingDimensions(i);
    auto datatype = engine_->getBindingDataType(i);
    static auto datatype_names = std::map<nvinfer1::DataType, std::string>{
      {nvinfer1::DataType::kFLOAT, "FLOAT"},
      {nvinfer1::DataType::kHALF, "HALF"},
      {nvinfer1::DataType::kINT8, "INT8"},
      {nvinfer1::DataType::kINT32, "INT32"},
      {nvinfer1::DataType::kBOOL, "BOOL"},
    };

    std::cout << " " << (is_input ? "Input[" : "Output[") << i << "]"
              << " name=" << name << " dims=[";
    for (int32_t j = 0; j < dims.nbDims; j++) {
      std::cout << dims.d[j];
      if (j < dims.nbDims-1) std::cout << ",";
    }
    std::cout << "] datatype=" << datatype_names[datatype] << std::endl;
  }
}

int main(int argc, char const *argv[]) {
  std::string engine_filename{"rvm_mobilenetv3_fp32_sim_modified.engine"};
  std::string input_filename{"input.jpg"};
  std::string output_filename{"output.png"};

  if (argc > 1) engine_filename = argv[1];
  if (argc > 2) input_filename = argv[2];
  if (argc > 3) output_filename = argv[3];

  std::cout << "engine_filename: " << engine_filename << std::endl;
  std::cout << "input_filename: " << input_filename << std::endl;
  std::cout << "output_filename: " << output_filename << std::endl;

  RVM rvm;
  if (!rvm.Load(engine_filename)) {
    return EXIT_FAILURE;
  }
  rvm.PrintEngineInfo();
  if (!rvm.InferFP32(input_filename, output_filename)) {
    return EXIT_FAILURE;
  }
  // if (!rvm.InferFP16(input_filename, output_filename)) {
  //   return EXIT_FAILURE;
  // }

  return EXIT_SUCCESS;
}
