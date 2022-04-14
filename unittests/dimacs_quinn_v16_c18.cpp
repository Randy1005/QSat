
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

const std::string v16_c18 = "\
p cnf 16 18\
  1    2  0\
 -2   -4  0\
  3    4  0\
 -4   -5  0\
  5   -6  0\
  6   -7  0\
  6    7  0\
  7  -16  0\
  8   -9  0\
 -8  -14  0\
  9   10  0\
  9  -10  0\
-10  -11  0\
 10   12  0\
 11   12  0\
 13   14  0\
 14  -15  0\
 15   16  0";


TEST_CASE("DIMACS.v16_c18.sat") {
  qsat::Solver solver;

  std::istringstream iss(v16_c18);

  solver.read_dimacs(iss);

  REQUIRE(solver.num_variables() == 16);
  REQUIRE(solver.num_clauses() == 18);
  REQUIRE(solver.solve() == true); 

}



