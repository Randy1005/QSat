add_test( Statistics /home/randy/QSat/build-debug/bin/verify_basics [==[--test-case=Statistics]==])
set_tests_properties( Statistics PROPERTIES WORKING_DIRECTORY /home/randy/QSat/build-debug/unittests)
add_test( [==[Literal Operators + Evaluation]==] /home/randy/QSat/build-debug/bin/verify_basics [==[--test-case=Literal Operators + Evaluation]==])
set_tests_properties( [==[Literal Operators + Evaluation]==] PROPERTIES WORKING_DIRECTORY /home/randy/QSat/build-debug/unittests)
set( verify_basics_TESTS Statistics [==[Literal Operators + Evaluation]==])
