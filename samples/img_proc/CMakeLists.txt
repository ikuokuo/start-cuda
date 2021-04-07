set(MY_CURR ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(MY_SAMPLES_DIR ${MY_CURR} DIRECTORY)

find_package(Boost REQUIRED)

# nvcc fatal : Stray '"' character
#add_definitions(-DMY_SAMPLES_DIR=\"${MY_SAMPLES_DIR}\")

## grayscale

add_sample(grayscale
  SRCS
    grayscale.cc
    grayscale_gpu.cu
  LIBS jpeg
)
target_compile_definitions(grayscale
  PUBLIC
    MY_DATA_DIR="${MY_SAMPLES_DIR}/data"
    MY_OUTPUT_DIR="${MY_OUTPUT}"
)
target_compile_options(grayscale
  PRIVATE -Wno-unused-parameter
)
