add_library(qsat qsat.cpp)
add_library(cnfgraph cnfgraph.cpp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(qsat PRIVATE Threads::Threads)
set_property(TARGET qsat PROPERTY CXX_STANDARD 17)
target_compile_options(qsat INTERFACE -Wall -Wextra -Wfatal-errors)

target_link_libraries(cnfgraph PRIVATE Threads::Threads)
set_property(TARGET cnfgraph PROPERTY CXX_STANDARD 17)
target_compile_options(cnfgraph INTERFACE -Wall -Wextra -Wfatal-errors)

target_link_libraries(qsat LINK_PUBLIC cnfgraph)

