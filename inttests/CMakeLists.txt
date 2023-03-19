#TODO: probably should refactor these to a "miscellaneous" directory

# Integration test on benchmarks
message(STATUS "Building integration tests ...")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# execute_process(COMMAND ${Python3_EXECUTABLE} my_script.py my_arg)

set(QSAT_CHECKER ${PROJECT_SOURCE_DIR}/inttests/shell.py)
set(QSAT_BAD_INPUT_REMOVER ${QSAT_3RD_PARTY_DIR}/remove_bad_lines_from_cnf.py)


# harder problems
add_test(it.s3-3-3-10 ${QSAT_CHECKER} "s3-3-3-10.cnf")
add_test(it.s3-3-3-1 ${QSAT_CHECKER} "s3-3-3-1.cnf")
add_test(it.s3-3-3-3 ${QSAT_CHECKER} "s3-3-3-3.cnf")
add_test(it.s3-3-3-4 ${QSAT_CHECKER} "s3-3-3-4.cnf")
add_test(it.s3-3-3-8 ${QSAT_CHECKER} "s3-3-3-8.cnf")
add_test(it.fpga10_8_sat ${QSAT_CHECKER} "fpga10_8_sat.cnf")
add_test(it.fpga10_9_sat ${QSAT_CHECKER} "fpga10_9_sat.cnf")
add_test(it.fpga12_8_sat ${QSAT_CHECKER} "fpga12_8_sat.cnf")
add_test(it.fpga12_9_sat ${QSAT_CHECKER} "fpga12_9_sat.cnf")
add_test(it.fpga12_11_sat ${QSAT_CHECKER} "fpga12_11_sat.cnf")
add_test(it.fpga12_12_sat ${QSAT_CHECKER} "fpga12_12_sat.cnf")
add_test(it.fpga13_9_sat ${QSAT_CHECKER} "fpga13_9_sat.cnf")
add_test(it.fpga13_10_sat ${QSAT_CHECKER} "fpga13_10_sat.cnf")
add_test(it.fpga13_12_sat ${QSAT_CHECKER} "fpga13_12_sat.cnf")
add_test(it.chnl10_11 ${QSAT_CHECKER} "chnl10_11.cnf")
add_test(it.chnl10_12 ${QSAT_CHECKER} "chnl10_12.cnf")
add_test(it.chnl10_13 ${QSAT_CHECKER} "chnl10_13.cnf")
add_test(it.chnl11_12 ${QSAT_CHECKER} "chnl11_12.cnf")
add_test(it.chnl11_13 ${QSAT_CHECKER} "chnl11_13.cnf")
add_test(it.chnl11_20 ${QSAT_CHECKER} "chnl11_20.cnf")
add_test(it.hole7 ${QSAT_CHECKER} "hole7.cnf")
add_test(it.hole8 ${QSAT_CHECKER} "hole8.cnf")
add_test(it.hole9 ${QSAT_CHECKER} "hole9.cnf")
add_test(it.hole11 ${QSAT_CHECKER} "hole11.cnf")
add_test(it.hole12 ${QSAT_CHECKER} "hole12.cnf")
add_test(it.Urq3_5 ${QSAT_CHECKER} "Urq3_5.cnf")
add_test(it.Urq4_5 ${QSAT_CHECKER} "Urq4_5.cnf")
add_test(it.Urq5_5 ${QSAT_CHECKER} "Urq5_5.cnf")
add_test(it.Urq6_5 ${QSAT_CHECKER} "Urq6_5.cnf")
add_test(it.Urq7_5 ${QSAT_CHECKER} "Urq7_5.cnf")
add_test(it.Urq8_5 ${QSAT_CHECKER} "Urq8_5.cnf")

set_tests_properties(
	it.s3-3-3-10
	it.s3-3-3-1
	it.s3-3-3-3
	it.s3-3-3-4
	it.s3-3-3-8
	it.fpga10_8_sat
	it.fpga10_9_sat
	it.fpga12_8_sat
	it.fpga12_9_sat
	it.fpga12_11_sat
	it.fpga12_12_sat
	it.fpga13_9_sat
	it.fpga13_10_sat
	it.fpga13_12_sat
	it.chnl10_11
	it.chnl10_12
	it.chnl10_13
	it.chnl11_12
	it.chnl11_13
	it.chnl11_20
	it.hole7
	it.hole8
	it.hole9
	it.hole11
	it.hole12
	it.Urq3_5 
	it.Urq4_5 
	it.Urq5_5 
	it.Urq6_5 
	it.Urq7_5 
	it.Urq8_5 
	PROPERTIES LABELS "integration")


