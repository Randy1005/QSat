cmake_minimum_required(VERSION 3.18.2)

project(QSat VERSION 1.0)

# add sub directories
add_subdirectory(qsat)
add_subdirectory(src)
add_subdirectory(utils)

target_link_libraries(${PROJECT_NAME} PUBLIC qsat)


