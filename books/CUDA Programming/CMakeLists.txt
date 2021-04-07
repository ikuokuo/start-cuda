set(MY_CURR ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(MY_BOOK_NAME ${MY_CURR} NAME_WE)

set(_outdir books/${MY_BOOK_NAME})

file(GLOB_RECURSE samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc *.cu)

foreach(sample ${samples})
  get_filename_component(name ${sample} NAME_WE)
  add_sample(${name} SRCS ${sample})
endforeach()

unset(_outdir)
