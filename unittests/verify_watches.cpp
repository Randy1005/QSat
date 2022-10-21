#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

// Unittest: basic interface usage
TEST_CASE("Watches Data Structure Correctness" * doctest::timeout(300)) {

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

 	// watches (upon completing add clauses)
	
	
	// a: lit 0, b: lit 13
	// c: lit 16, d: lit 7
	// ~a, ~d should be watched in c0
	// ~b, ~c should be watched in c1
	// c2 is unit clause, nothing to be watched
	REQUIRE(s.watches[1].size() == 1);
	REQUIRE(s.watches[6].size() == 1);
	REQUIRE(s.watches[12].size() == 1);
	REQUIRE(s.watches[17].size() == 1);
	// other literals should have no watchers
	REQUIRE(s.watches[0].size() == 0);
	REQUIRE(s.watches[10].size() == 0);
	
	// watches[~a]'s 0th watcher clause is c0
	// watches[~a]'s 0th blocker is d (7)
	REQUIRE(s.watches[1][0].cref == 0);
	REQUIRE(s.watches[1][0].blocker.id == 7);
	// watches[~d]'s 0th watcher clause is c0
	// watches[~d]'s 0th blocker is a (0)
	REQUIRE(s.watches[6][0].cref == 0);
	REQUIRE(s.watches[6][0].blocker.id == 0);
	// watches[~b]'s 0th watcher clause is c1
	// watches[~b]'s 0th blocker is c (16)
	REQUIRE(s.watches[12][0].cref == 1);
	REQUIRE(s.watches[12][0].blocker.id == 16);
	// watches[~c]'s 0th watcher clause is c1
	// watches[~c]'s 0th blocker is b (13)
	REQUIRE(s.watches[17][0].cref == 1);
	REQUIRE(s.watches[17][0].blocker.id == 13);
	


}


