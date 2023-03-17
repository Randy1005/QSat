#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

// Unittest: basic interface usage
TEST_CASE("Clause Signatures" * doctest::timeout(300)) {

  qsat::Literal a(1), b(-7), c(9);
  qsat::Solver solver;

  std::cout << "hi\n";  

}


