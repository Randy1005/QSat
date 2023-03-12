add_library(qsat qsat.cpp)
message(STATUS "$ENV{DPCPP_ROOT}")

# include headers from sycl
target_include_directories(qsat PUBLIC 
  $ENV{DPCPP_ROOT}/llvm/build/include/
)


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(qsat PUBLIC Threads::Threads)
set_property(TARGET qsat PROPERTY CXX_STANDARD 17)
target_compile_options(qsat INTERFACE -Wall -Wextra -Wfatal-errors)

find_package(breakid REQUIRED)

if (breakid_FOUND)
  message(STATUS "breakid found")
  target_link_libraries(qsat LINK_PUBLIC breakid)
  target_include_directories(qsat PUBLIC 
    ${BREAKID_INCLUDE_DIRS}  
  )
  message(STATUS "BREAKID_INCLUDE_DIRS=${BREAKID_INCLUDE_DIRS}")
endif()
