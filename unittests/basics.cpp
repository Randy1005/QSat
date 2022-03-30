#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

TEST_CASE("Clauses" * doctest::timeout(300)) {

  qsat::Solver solver;


  REQUIRE(solver.clauses().size() == 0);
  
  qsat::Clause c;

}

TEST_CASE("Literals" * doctest::timeout(300)) {

  qsat::Solver solver;


  REQUIRE(solver.clauses().size() == 0);
  
  qsat::Clause c;

}

// Unittest: a => a must be true
TEST_CASE("CNF.1l.1c" * doctest::timeout(300)) {

  qsat::Solver solver;


  REQUIRE(solver.clauses().size() == 0);
  
  qsat::Clause c;
}

// Unittest (a + b')(a'+ b) 
TEST_CASE("CNF.2c.2l" * doctest::timeout(300)) {

  qsat::Solver solver;


  REQUIRE(solver.clauses().size() == 0);
  
  qsat::Clause c;
}




