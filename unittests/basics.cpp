#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

TEST_CASE("Statistics" * doctest::timeout(300)) {

  qsat::Literal a(1), b(-7), c(9);
  qsat::Solver solver;
  
  REQUIRE(solver.num_clauses() == 0);
  REQUIRE(solver.num_variables() == 0);
  
  solver.add_clause({a});
  REQUIRE(solver.num_clauses() == 1);
  REQUIRE(solver.num_variables() == 1);

  solver.add_clause({b, c});
  REQUIRE(solver.num_clauses() == 2);
  REQUIRE(solver.num_variables() == 9);

  solver.reset();
  REQUIRE(solver.num_clauses() == 0);
  REQUIRE(solver.num_variables() == 0);
}

// Unittest: a => a must be true
TEST_CASE("CNF.1v.1c.sat" * doctest::timeout(300)) {
  qsat::Literal a(1);
  qsat::Solver solver;
  solver.add_clause({a});
  REQUIRE(solver.solve() == true);
  // TODO
  //REQUIRE(solver.assignment_of(1) == qsat::Assignment::TRUE);
}

// Unittest (a + b')(a'+ b) 
TEST_CASE("CNF.2v.2c.sat" * doctest::timeout(300)) {
  qsat::Solver solver;
  solver.add_clause({ qsat::Literal(1),  qsat::Literal(-2) });
  solver.add_clause({ qsat::Literal(-1), qsat::Literal(2) });
  REQUIRE(solver.num_clauses() == 2);
  REQUIRE(solver.num_variables() == 2);
  REQUIRE(solver.solve() == true);

}

// Unittest (a)(a') 
TEST_CASE("CNF.1v.2c.unsat" * doctest::timeout(300)) {
  qsat::Solver solver;
  solver.add_clause({qsat::Literal(1)});
  solver.add_clause({qsat::Literal(-1)});
  REQUIRE(solver.num_clauses() == 2);
  REQUIRE(solver.num_variables() == 1);
  REQUIRE(solver.solve() == false);
}

// TODO: 



