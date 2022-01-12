cmake_minimum_required(VERSION 3.20)
project(start_cuda VERSION 1.0.0 LANGUAGES C CXX CUDA)

include(cmake/global.cmake)

set(MY_ROOT ${PROJECT_SOURCE_DIR})

set(MY_VERSION ${PROJECT_VERSION})
set_version_values(MY_VERSION)

# cmake options

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()
if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# CUDA GPUs
#  https://developer.nvidia.com/cuda-gpus
# Compute Capabilities
#  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)

# flags

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wformat")
if(NOT HOST_OS STREQUAL "Win")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
status("CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# packages

LIST(APPEND CMAKE_MODULE_PATH "${MY_ROOT}/cmake/modules")

## threads

find_package(Threads REQUIRED)

## cuda

find_package(CUDAToolkit REQUIRED)
status_var(CUDAToolkit_INCLUDE_DIRS)
status_var(CUDAToolkit_LIBRARY_DIR)

## cudnn

find_package(CUDNN)
if(CUDNN_FOUND)
  status_var(CUDNN_INCLUDE_DIRS)
  status_var(CUDNN_LIBRARIES)
endif()

## tensorrt

find_package(TensorRT)
if(TensorRT_FOUND)
  status_var(TensorRT_INCLUDE_DIRS)
  status_var(TensorRT_LIBRARIES)
endif()

## opencv

find_package(OpenCV 4 REQUIRED)
status_var(OpenCV_INCLUDE_DIRS)
status_var(OpenCV_LIBS)

# configs

add_definitions(-DTIME_COST)

# targets

include_directories(
  include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDNN_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(MY_OUTPUT "${MY_ROOT}/_output")
set_outdir(
  ARCHIVE "${MY_OUTPUT}/lib"
  LIBRARY "${MY_OUTPUT}/lib"
  RUNTIME "${MY_OUTPUT}/bin"
)

# add_sample(NAME SRCS srcs LIBS libs OUTDIR outdir)
macro(add_sample NAME)
  set(options)
  set(oneValueArgs OUTDIR)
  set(multiValueArgs SRCS LIBS)
  cmake_parse_arguments(THIS "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  add_executable(${NAME} ${THIS_SRCS})
  if(THIS_LIBS)
    target_link_libraries(${NAME} ${THIS_LIBS})
  endif()
  set(__outdir ${THIS_OUTDIR})
  if(NOT __outdir AND _outdir)
    set(__outdir ${_outdir})
  endif()
  if(__outdir)
    set_target_outdir(${NAME}
      ARCHIVE "${MY_OUTPUT}/lib/${__outdir}"
      LIBRARY "${MY_OUTPUT}/lib/${__outdir}"
      RUNTIME "${MY_OUTPUT}/bin/${__outdir}"
    )
  endif()
  unset(__outdir)
endmacro()

if(BUILD_BOOKS)
  add_subdirectory(books)
endif()

if(BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

# install

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
#status_var(CMAKE_INSTALL_PREFIX)
#status_var(CMAKE_INSTALL_INCLUDEDIR)
#status_var(CMAKE_INSTALL_BINDIR)
#status_var(CMAKE_INSTALL_LIBDIR)
#status_var(CMAKE_INSTALL_CMAKEDIR)

set(CMAKE_MACOSX_RPATH 1)
#set(CMAKE_INSTALL_RPATH "${MY_CMAKE_RPATH}")

# others

include(cmake/summary.cmake)
