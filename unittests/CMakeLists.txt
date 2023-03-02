enable_testing()

include(${QSAT_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND QSAT_UNITTESTS
  verify_basics
  verify_heap
  verify_enqueue
	verify_watches
	verify_detach_clauses
)

foreach(unittest IN LISTS QSAT_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
	add_test(NAME ${unittest} COMMAND ${unittest})	
	set_property(TEST ${unittest} PROPERTY LABELS "unit")
  target_include_directories(${unittest} PRIVATE ${PROJECT_SOURCE_DIR} ${QSAT_3RD_PARTY_DIR}/doctest)
  target_link_libraries(${unittest} LINK_PUBLIC 
                        qsat 
                        stdc++fs)
  doctest_discover_tests(${unittest})
endforeach()

