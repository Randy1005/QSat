cmake_minimum_required(VERSION 3.18.2)

project(QSat VERSION 1.0)

message("PROJECT_NAME:" ${PROJECT_NAME})
message("PROJECT_SOURCE_DIR:" ${PROJECT_SOURCE_DIR})
message("CMAKE_BUILD_TYPE:" ${CMAKE_BUILD_TYPE})

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# Library-specific variable
set(QSAT_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------

# Enable test
include(CTest)

# -----------------------------------------------------------------------------
# include sub cmake list
# -----------------------------------------------------------------------------

# add sub directories
add_subdirectory(qsat)
add_subdirectory(main)
add_subdirectory(unittests)



