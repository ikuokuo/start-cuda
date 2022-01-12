set(_outdir samples)

add_subdirectory(quick_start)
add_subdirectory(matmul)
#add_subdirectory(img_proc)
add_subdirectory(npp)
add_subdirectory(nvjpeg)

if(TensorRT_FOUND)
  add_subdirectory(tensorrt)
endif()

unset(_outdir)
