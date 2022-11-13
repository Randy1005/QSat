cmake_minimum_required(VERSION 3.18.2)

project(QSat VERSION 1.0)

# set everything up for c++ 17 features
set(CMAKE_CXX_STANDARD 17)


# test that filesystem header actually is there and works


#------------------------------------------------------------------------------
# default release build
#------------------------------------------------------------------------------



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




# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)


# Library-specific variable
set(QSAT_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

# Benchmark directory variable
set(QSAT_BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/benchmarks)


message("PROJECT_NAME: " ${PROJECT_NAME})
message("QSAT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
message("CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "QSAT_BENCHMARK_DIR: " ${QSAT_BENCHMARK_DIR})



# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------

# Enable test
include(CTest)
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")

# -----------------------------------------------------------------------------
# include sub cmake list
# -----------------------------------------------------------------------------

# add sub directories
add_subdirectory(qsat)
add_subdirectory(${QSAT_3RD_PARTY_DIR}/PEGTL)
add_subdirectory(${QSAT_3RD_PARTY_DIR}/minisat)
add_subdirectory(main)
add_subdirectory(unittests)
add_subdirectory(inttests)

# -----------------------------------------------------------------------------
# benchmark directories
# -----------------------------------------------------------------------------

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





