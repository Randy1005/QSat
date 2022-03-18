cmake_minimum_required(VERSION 3.18.2)

project(QSat VERSION 1.0)

message("PROJECT_NAME:" ${PROJECT_NAME})
message("PROJECT_SOURCE_DIR:" ${PROJECT_SOURCE_DIR})

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# add sub directories
add_subdirectory(qsat)
add_subdirectory(main)



