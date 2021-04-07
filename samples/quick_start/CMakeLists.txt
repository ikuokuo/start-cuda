file(GLOB samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc *.cu)

foreach(sample ${samples})
  get_filename_component(name ${sample} NAME_WE)
  add_sample(${name} SRCS ${sample})
endforeach()
