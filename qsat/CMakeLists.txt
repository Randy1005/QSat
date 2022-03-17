add_library(qsat qsat.cpp)
target_include_directories(qsat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(qsat PUBLIC utils)
