add_library(qsat qsat.cpp)

#target_compile_features(qsat INTERFACE cxx_std_17)
set_property(TARGET qsat PROPERTY CXX_STANDARD 17)

target_link_options(qsat INTERFACE -lstdc++fs)
target_compile_options(qsat INTERFACE -Wall -Wextra -Wfatal-errors)
