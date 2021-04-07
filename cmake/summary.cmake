# summary

status("")
status("Platform:")
status("  HOST_OS: ${HOST_OS}")
status("  HOST_ARCH: ${HOST_ARCH}")
status("  HOST_COMPILER: ${CMAKE_CXX_COMPILER_ID}")
status("    COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

status("")
status("Build:")
status("  Samples: " IF BUILD_SAMPLES "YES" ELSE "NO")

status("")
status("CUDA:")
status("  ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

status("")
