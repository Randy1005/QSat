message(STATUS "Adding integration tests from CBS_k3_n100_m449_b90 (1000 instances)")

# IMPORTANT: must have this package to run my own python script
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# configure checker / bad input remover variable paths
set(QSAT_BAD_INPUT_REMOVER ${QSAT_3RD_PARTY_DIR}/remove_bad_lines_from_cnf.py)
set(QSAT_CHECKER ${PROJECT_SOURCE_DIR}/inttests/shell.py)


file(GLOB files "*.cnf")
foreach(file ${files})
	string(REPLACE "${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m449_b90/" "" CNF_NAME "${file}")
	string(REGEX REPLACE "\\.[^.]*$" "" CNF_WITHOUT_EXT ${CNF_NAME})
	# message("CNF_NAME = " ${CNF_WITHOUT_EXT})
	set(IT_TEST_NAME "it.${CNF_WITHOUT_EXT}")

	# message("Removing bad cnf characters ... for " ${CNF_NAME})
	# message("bad input remover = " ${QSAT_BAD_INPUT_REMOVER})
	# execute_process(COMMAND ${Python3_EXECUTABLE} ${QSAT_BAD_INPUT_REMOVER} ${QSAT_BENCHMARK_DIR}/CBS_k3_n100_m449_b90/${CNF_NAME})

	# message("IT_TEST_NAME = " ${IT_TEST_NAME})
	add_test(${IT_TEST_NAME} ${QSAT_CHECKER} "CBS_k3_n100_m449_b90/${CNF_NAME}")
	set_tests_properties(${IT_TEST_NAME} PROPERTIES LABELS "integration")
endforeach()
