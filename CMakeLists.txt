cmake_minimum_required(VERSION 3.26.0)

# this is hardcoded, can I configure cmake to auto locate
# nvcc for me?
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)


project(QSat VERSION 1.0 LANGUAGES C CXX)


# set everything up for c++20 features
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 75)


# set default compiler to be gcc g++ (DISCOURAGED)
# set(CMAKE_C_COMPILER $ENV{DPCPP_COMPILER})
# set(CMAKE_CXX_COMPILER $ENV{DPCPP_COMPILER})
# set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda")

# test that filesystem header actually is there and works

#------------------------------------------------------------------------------
# default release build
#------------------------------------------------------------------------------



# link libraries
# find sycl library
# find_library(SYCL_LIB NAMES sycl libsycl HINTS $ENV{DPCPP_ROOT}/llvm/build/lib/ REQUIRED)





# message(STATUS "SYCL lib path=" ${SYCL_LIB})

# set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release ")
  set(
    CMAKE_BUILD_TYPE Release
    CACHE
    STRING "Choose the type of build."
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )

endif()

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -Xcompiler -g -lineinfo --expt-extended-lambda")


# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)


# Library-specific variable
set(QSAT_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

# Benchmark directory variable
set(QSAT_BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/benchmarks)



# Binary folder
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

message("PROJECT_NAME: " ${PROJECT_NAME})
message("QSAT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
message("CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "QSAT_BENCHMARK_DIR: " ${QSAT_BENCHMARK_DIR})
message(STATUS "BIN_DIR: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})





option(QSAT_BUILD_CUDA "Enables builds of cuda code" ON)





# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------
include(CheckLanguage)


# Enable test
include(CTest)
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")




# -----------------------------------------------------------------------------
# find nvcc
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------
if(QSAT_BUILD_CUDA)
  message(STATUS "Configuring CUDA ...")
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "\nNo CUDA compiler found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_definitions(-DUSE_CUDA)
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\nQSat requires CUDA at least v11")
  endif()
endif()  

# -----------------------------------------------------------------------------
# find thrust
# -----------------------------------------------------------------------------
#set(Thrust_DIR ${PROJECT_SOURCE_DIR}/thrust/thrust/cmake/)
#message(STATUS "Thrust_DIR=" ${Thrust_DIR})
#find_package(Thrust REQUIRED CONFIG)
#thrust_create_target(Thrust)


# -----------------------------------------------------------------------------
# include sub cmake list
# -----------------------------------------------------------------------------

# add sub directories
add_subdirectory(qsat)
add_subdirectory(${QSAT_3RD_PARTY_DIR}/PEGTL)
add_subdirectory(${QSAT_3RD_PARTY_DIR}/minisat)
add_subdirectory(${QSAT_3RD_PARTY_DIR}/breakid)
add_subdirectory(main)
add_subdirectory(unittests)
add_subdirectory(inttests)

# -----------------------------------------------------------------------------
# benchmark directories
# -----------------------------------------------------------------------------

# SC-2022 benchmarks : SAT Competition 2022
add_subdirectory(${QSAT_BENCHMARK_DIR}/SC-2022)


# parafrost benchmarks
add_subdirectory(${QSAT_BENCHMARK_DIR}/parafrost-benchmarks)


# uniform random 3-SAT 
add_subdirectory(${QSAT_BENCHMARK_DIR}/uf50-218)
add_subdirectory(${QSAT_BENCHMARK_DIR}/uuf50-218)
add_subdirectory(${QSAT_BENCHMARK_DIR}/uf75-325)
add_subdirectory(${QSAT_BENCHMARK_DIR}/uuf75-325)
add_subdirectory(${QSAT_BENCHMARK_DIR}/uf100-430)
add_subdirectory(${QSAT_BENCHMARK_DIR}/uuf100-430)
add_subdirectory(${QSAT_BENCHMARK_DIR}/uf125-538)
add_subdirectory(${QSAT_BENCHMARK_DIR}/uuf125-538)



# random 3-SAT with controlled backbone size
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m403_b10)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m403_b30)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m403_b50)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m403_b70)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m403_b90)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m411_b10)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m411_b30)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m411_b50)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m411_b70)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m411_b90)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m449_b10)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m449_b30)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m449_b50)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m449_b70)
add_subdirectory(${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m449_b90)

# AIM benchmarks
add_subdirectory(${QSAT_BENCHMARK_DIR}/aim)

# Bejing competition benchmarks
add_subdirectory(${QSAT_BENCHMARK_DIR}/Bejing)

# JNH benmarks : random SAT instances with variable length clauses
add_subdirectory(${QSAT_BENCHMARK_DIR}/jnh)

# DUBOIS benchmarks
add_subdirectory(${QSAT_BENCHMARK_DIR}/dubois)

# LRAN benchmarks
add_subdirectory(${QSAT_BENCHMARK_DIR}/lran)

# flat graph coloring benchmarks
add_subdirectory(${QSAT_BENCHMARK_DIR}/flat50-115)

# PRET benchmarks : encoded 2-colouring forced to be unsatisfiable
add_subdirectory(${QSAT_BENCHMARK_DIR}/pret)

# SSA benchmarks : circuit fault analysis: single-stuck-at fault
add_subdirectory(${QSAT_BENCHMARK_DIR}/ssa)

# IBM bounded model checking benchmarks
add_subdirectory(${QSAT_BENCHMARK_DIR}/ibm-bmc)


