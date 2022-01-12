file(GLOB samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc *.cu)

foreach(sample ${samples})
  get_filename_component(name ${sample} NAME_WE)
  add_sample(${name} SRCS ${sample} LIBS ${OpenCV_LIBS}
    CUDA::cudart CUDA::cublas ${CUDNN_LIBRARIES} ${TensorRT_LIBRARIES}
  )
  target_compile_options(${name}
    PRIVATE -Wno-deprecated-declarations
  )
endforeach()
