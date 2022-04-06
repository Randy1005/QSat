#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>
#include <filesystem>

// Unittest: basic interface usage
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

// Unittest (a + b)(b + a')(a + b')(a' + b')
TEST_CASE("CNF.2v.4c.unsat" * doctest::timeout(300)) {
  qsat::Solver solver;
  solver.add_clause({qsat::Literal(1), qsat::Literal(2)});
  solver.add_clause({qsat::Literal(2), qsat::Literal(-1)});
  solver.add_clause({qsat::Literal(1), qsat::Literal(-2)});
  solver.add_clause({qsat::Literal(-1), qsat::Literal(-2)});

  REQUIRE(solver.num_clauses() == 4);
  REQUIRE(solver.num_variables() == 2);
  REQUIRE(solver.solve() == false);
}

// Unittest
// (a + b + c)(a + b + c')(a + b' + c)(a + b' + c')(a' + b + c)(a' + b + c')
// (a' + b' + c)(a' + b' + c')
TEST_CASE("CNF.3v.8c.unsat" * doctest::timeout(300)) {
  qsat::Solver solver;
  solver.add_clause({qsat::Literal(1), qsat::Literal(2), qsat::Literal(3)});
  solver.add_clause({qsat::Literal(1), qsat::Literal(2), qsat::Literal(-3)});
  solver.add_clause({qsat::Literal(1), qsat::Literal(-2), qsat::Literal(3)});
  solver.add_clause({qsat::Literal(1), qsat::Literal(-2), qsat::Literal(-3)});
  solver.add_clause({qsat::Literal(-1), qsat::Literal(2), qsat::Literal(3)});
  solver.add_clause({qsat::Literal(-1), qsat::Literal(2), qsat::Literal(-3)});
  solver.add_clause({qsat::Literal(-1), qsat::Literal(-2), qsat::Literal(3)});
  solver.add_clause({qsat::Literal(-1), qsat::Literal(-2), qsat::Literal(-3)});

  REQUIRE(solver.num_clauses() == 8);
  REQUIRE(solver.num_variables() == 3);
  REQUIRE(solver.solve() == false);
}


// Unittest (a + b')(a' + b)(a + b) => unique solution (a = True, b = True)
TEST_CASE("CNF.2v.3c.sat.unique" * doctest::timeout(300)) {
  qsat::Solver solver;
  solver.add_clause({qsat::Literal(1), qsat::Literal(-2)});
  solver.add_clause({qsat::Literal(-1), qsat::Literal(2)});
  solver.add_clause({qsat::Literal(1), qsat::Literal(2)});
  
  REQUIRE(solver.num_clauses() == 3);
  REQUIRE(solver.num_variables() == 2);
  REQUIRE(solver.solve() == true);
  REQUIRE(solver.assignment_of(1) == qsat::Assignment::TRUE);
  REQUIRE(solver.assignment_of(2) == qsat::Assignment::TRUE);
}

// TODO: unittest typically doesn't do file
TEST_CASE("ReadCNF.20v.91c.sat.0" * doctest::timeout(300)) {
  qsat::Solver solver;
  solver.read_dimacs("../../dimacs_test_files/sat_v20_c91.cnf");

  REQUIRE(solver.num_clauses() == 91);
  REQUIRE(solver.num_variables() == 20);
  REQUIRE(solver.solve() == true);
}

TEST_CASE("ReadCNF.20v.91c.sat.1" * doctest::timeout(300)) {
  qsat::Solver solver;
  solver.read_dimacs("../../dimacs_test_files/sat_v20_c91_2.cnf");

  REQUIRE(solver.num_clauses() == 91);
  REQUIRE(solver.num_variables() == 20);
  REQUIRE(solver.solve() == true);
}




