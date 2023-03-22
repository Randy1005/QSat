#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

// Unittest: basic interface usage
TEST_CASE("Statistics" * doctest::timeout(300)) {

  qsat::Literal a(1), b(-7), c(9);
  qsat::Solver solver;
 

  REQUIRE(solver.num_clauses() == 0);
  REQUIRE(solver.num_variables() == 0);
  
	// NOTE: we don't store unit clauses
	// the unit literal is enqueued directly
	solver.add_clause({a});
  REQUIRE(solver.num_clauses() == 0);
  REQUIRE(solver.num_variables() == 1);

  solver.add_clause({b, c});
  REQUIRE(solver.num_clauses() == 1);
  REQUIRE(solver.num_variables() == 9);

  solver.reset();
  REQUIRE(solver.num_clauses() == 0);
  REQUIRE(solver.num_variables() == 0);
}

// Unittest: literal operators / literal evaluation
TEST_CASE("Literal Operators + Evaluation" * doctest::timeout(300)) {
  qsat::Literal a(-1), b(2);
  qsat::Solver s;

  s.add_clause({a, b});

  // value check: a, b are evaluated to UNDEF now
  REQUIRE(s.value(a) == qsat::Status::UNDEFINED);
  REQUIRE(s.value(b) == qsat::Status::UNDEFINED);

  REQUIRE(s.value(var(a)) == qsat::Status::UNDEFINED);
  REQUIRE(s.value(var(b)) == qsat::Status::UNDEFINED);
  

  // sign check: sign(a) == 1, sign(b) == 0
  REQUIRE(sign(a) == 1);
  REQUIRE(sign(b) == 0);

  // negation
  REQUIRE(sign(~a) == 0);
  REQUIRE(sign(~b) == 1);

  s.assign(var(a), 1);
  s.assign(var(b), 0);

  REQUIRE(s.value(a) == qsat::Status::FALSE);
  REQUIRE(s.value(b) == qsat::Status::FALSE);

	s.assign(var(a), 0);
	REQUIRE(s.value(a) == qsat::Status::TRUE);

	s.assign(var(b), 1);
	REQUIRE(s.value(b) == qsat::Status::TRUE);

  // negate a, b
  qsat::Literal a_bar = ~a;
  qsat::Literal b_bar = ~b;
  
  // sign check: sign(a') == 1, sign(b') == 0
  REQUIRE(sign(a_bar) == 0);
  REQUIRE(sign(b_bar) == 1);

  // negation
  REQUIRE(sign(~a_bar) == 1);
  REQUIRE(sign(~b_bar) == 0);

  REQUIRE(s.value(a_bar) == qsat::Status::FALSE);
  REQUIRE(s.value(b_bar) == qsat::Status::FALSE);

}


