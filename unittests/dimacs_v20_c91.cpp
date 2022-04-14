#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

const std::string v20_c91 = "\
  p cnf 20 91\
  8 -9 7 0\
  -7 9 -15 0\
  9 8 1 0\
  5 -19 2 0\
  -1 -9 -8 0\
  4 19 16 0\
  9 -2 -12 0\
  1 20 5 0\
  -15 -6 8 0\
  -20 -16 -15 0\
  -11 7 18 0\
  -20 11 -18 0\
  1 2 11 0\
  10 -13 -16 0\
  -1 15 10 0\
  3 -12 -5 0\
  -20 -10 9 0\
  20 -11 -8 0\
  15 5 4 0\
  -5 -7 19 0\
  1 3 -15 0\
  17 3 -4 0\
  -11 -1 -5 0\
  -10 -5 -14 0\
  -19 -18 -15 0\
  -14 20 12 0\
  16 5 10 0\
  12 -8 9 0\
  -19 -3 13 0\
  9 16 -20 0\
  -1 2 -16 0\
  19 -17 3 0\
  -12 -9 -17 0\
  7 -17 20 0\
  -17 -8 -2 0\
  -1 -18 -14 0\
  -11 -15 14 0\
  10 -2 -20 0\
  17 11 -16 0\
  2 11 -19 0\
  -2 4 10 0\
  5 18 -14 0\
  13 -11 16 0\
  -18 -7 19 0\
  10 -14 -13 0\
  -4 16 3 0\
  14 -15 -2 0\
  -3 -10 19 0\
  9 19 6 0\
  -12 -3 -20 0\
  -3 14 1 0\
  -10 -17 -7 0\
  -19 -4 -1 0\
  4 12 -10 0\
  -18 4 -17 0\
  -20 -15 16 0\
  14 7 -1 0\
  12 -6 -17 0\
  -10 -16 6 0\
  9 15 -17 0\
  4 -6 -11 0\
  9 -8 3 0\
  -4 -6 -16 0\
  -5 19 -4 0\
  14 17 1 0\
  9 -18 -12 0\
  -16 17 -8 0\
  -1 11 16 0\
  7 10 9 0\
  4 -2 11 0\
  -17 -20 -3 0\
  -4 -11 -5 0\
  -11 -7 20 0\
  -5 2 -18 0\
  16 5 -3 0\
  14 7 -17 0\
  8 19 -1 0\
  13 16 -12 0\
  -10 3 8 0\
  18 -5 9 0\
  -14 12 -19 0\
  5 19 -16 0\
  -20 -5 -7 0\
  -7 -2 -15 0\
  17 -12 -1 0\
  7 13 10 0\
  -20 14 -9 0\
  -6 -20 -2 0\
  -5 -18 15 0\
  -16 -8 -20 0\
  14 19 -10 0";

TEST_CASE("DIMACS.v20_c91.sat") {
  qsat::Solver solver;

  std::istringstream iss(v20_c91);

  solver.read_dimacs(iss);

  REQUIRE(solver.num_variables() == 20);
  REQUIRE(solver.num_clauses() == 91);
  REQUIRE(solver.solve() == true); 

}


