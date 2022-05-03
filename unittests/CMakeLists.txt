enable_testing()

include(${QSAT_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND QSAT_UNITTESTS
  basics
	dimacs_v16_c18
  dimacs_v20_c91
  dimacs_v20_c91_2
  dimacs_v20_c91_3
  dimacs_v20_c91_4
  dimacs_v42_c133
)

foreach(unittest IN LISTS QSAT_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_include_directories(${unittest} PRIVATE ${PROJECT_SOURCE_DIR} ${QSAT_3RD_PARTY_DIR}/doctest)
  target_link_libraries(${unittest} LINK_PUBLIC qsat)
  doctest_discover_tests(${unittest})
endforeach()

