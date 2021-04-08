set(MY_CURR ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(MY_SAMPLES_DIR ${MY_CURR} DIRECTORY)

## all

file(GLOB samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc *.cu)

foreach(sample ${samples})
  get_filename_component(name ${sample} NAME_WE)
  add_sample(${name} SRCS ${sample} LIBS ${OpenCV_LIBS}
    CUDA::cudart CUDA::nppc CUDA::nppicc
  )
  target_compile_definitions(${name} PUBLIC
    MY_DATA_DIR="${MY_SAMPLES_DIR}/data"
    MY_OUTPUT_DIR="${MY_OUTPUT}"
  )
endforeach()
