add_library(qsat qsat.cpp)

target_link_libraries(qsat mtl)
set_property(TARGET qsat PROPERTY CXX_STANDARD 17)
target_compile_options(qsat INTERFACE -Wall -Wextra -Wfatal-errors)
