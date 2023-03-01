add_executable(QSat main.cpp)

# options/flags pass to the compiler
target_compile_features(qsat INTERFACE cxx_std_17)
target_compile_options(qsat INTERFACE -Wall -Wextra -Wfatal-errors)

# target_compile_features(cnfgraph INTERFACE cxx_std_17)
# target_compile_options(cnfgraph INTERFACE -Wall -Wextra -Wfatal-errors)


# include and libary linking
# target_link_libraries(qsat LINK_PUBLIC ${PROJECT_SOURCE_DIR}/libmetis/libmetis.a
#                           LINK_PUBLIC ${PROJECT_SOURCE_DIR}/libmetis/libGKlib.a
#                           LINK_PUBLIC ${PROJECT_SOURCE_DIR}/libbliss/libbliss_static.a)

target_link_libraries(QSat LINK_PUBLIC qsat 
                           LINK_PUBLIC stdc++fs)
target_include_directories(QSat PRIVATE ${PROJECT_SOURCE_DIR})

# create a variable for this executable
set(QSAT_EXEC ${PROJECT_SOURCE_DIR}/build/main/QSat)

