# summary

status("")
status("Platform:")
status("  HOST_OS: ${HOST_OS}")
status("  HOST_ARCH: ${HOST_ARCH}")
status("  HOST_COMPILER: ${CMAKE_CXX_COMPILER_ID}")
status("    COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

status("")
status("Package:")
status("  CUDAToolkit: " IF CUDAToolkit_FOUND "${CUDAToolkit_VERSION}" ELSE "NO")
status("    CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
status("    CUDNN: " IF CUDNN_FOUND "${CUDNN_VERSION}" ELSE "NO")
status("  OpenCV: " IF OpenCV_FOUND "${OpenCV_VERSION}" ELSE "NO")
status("  TensorRT: " IF TensorRT_FOUND "${TensorRT_VERSION_STRING}" ELSE "NO")

status("")
status("Build:")
status("  Samples: " IF BUILD_SAMPLES "YES" ELSE "NO")

status("")
