#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

// Unittest: basic interface usage
TEST_CASE("Solver Enqueue Functionality" * doctest::timeout(300)) {

  qsat::Literal a(1), b(-7), c(9), d(-4);
  qsat::Solver s;

  qsat::Clause c0({a, d, c});
  qsat::Clause c1({b, c});
  qsat::Clause c2({d});

  // NOTE:
  // we don't usually call add_clause like this
  // solely for the purpose of unit testing
  s.add_clause(c0.literals);
  s.add_clause(c1.literals);
  s.add_clause(c2.literals);

  // pre-condition:
  // num_assigns = 1, value(a, b, c) = undef
	// d is unit clause, gets enqueued during add_clause
  REQUIRE(s.num_assigns() == 1);
  REQUIRE(s.value(a) == qsat::Status::UNDEFINED);
  REQUIRE(s.value(b) == qsat::Status::UNDEFINED);
  REQUIRE(s.value(c) == qsat::Status::UNDEFINED);
  REQUIRE(s.value(d) == qsat::Status::TRUE);
 
  // invoke enqueue
  // e.g. enq a, with c0 as reason clause
  //      enq b, with c1 as reason clause
  //      enq c, no reason clause
  //      enq d, with c2 as reason clause
  bool res_a = s.enqueue(a, 0);
  
  // should be successful, because a has no initial assignment 
  REQUIRE(res_a == true);
  REQUIRE(s.value(a) == qsat::Status::TRUE);
  
  // case: b is already assigned true
  // which makes its value false
  // then enqueueing b would cause conflict
  s.assign(var(b), true);
  bool res_b = s.enqueue(b, 1);
  REQUIRE(res_b == false);
  REQUIRE(s.value(b) == qsat::Status::FALSE);


}


