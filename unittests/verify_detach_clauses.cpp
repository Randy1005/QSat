#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

TEST_CASE("Clause Detachment Correctness" * doctest::timeout(300)) {

  qsat::Literal a(1), b(-7), c(9), d(-4), e(2);
  qsat::Solver s;

  qsat::Clause c0({a, d, c});
  qsat::Clause c1({b, c});
  qsat::Clause c2({d, e});

  // NOTE:
  // we don't usually call add_clause like this
  // solely for the purpose of unit testing
  s.add_clause(c0.literals);
  s.add_clause(c1.literals);
  s.add_clause(c2.literals);

 	// watches (upon completing add clauses)
	
	
	// a: lit 0, b: lit 13
	// c: lit 16, d: lit 7
	// e: lit 2
	// ~a, ~d should be watched in c0
	// ~b, ~c should be watched in c1
	// ~d, ~e should be watched in c2
	REQUIRE(s.watches[1].size() == 1);
	REQUIRE(s.watches[12].size() == 1);
	REQUIRE(s.watches[17].size() == 1);
	REQUIRE(s.watches[6].size() == 2);
	REQUIRE(s.watches[3].size() == 1);
	// other literals should have no watchers
	REQUIRE(s.watches[0].size() == 0);
	REQUIRE(s.watches[10].size() == 0);
	
	// watches[~a]'s 0th watcher clause is c0
	// watches[~a]'s 0th blocker is d (7)
	REQUIRE(s.watches[1][0].cref == 0);
	REQUIRE(s.watches[1][0].blocker.id == 7);
	// watches[~d]'s 0th watcher clause is c0
	// watches[~d]'s 0th blocker is a (0)
	// watches[~d]'s 1th watcher clause is c2
	// watches[~d]'s 0th blocker is e (2)
	REQUIRE(s.watches[6][0].cref == 0);
	REQUIRE(s.watches[6][0].blocker.id == 0);
	REQUIRE(s.watches[6][1].cref == 2);
	REQUIRE(s.watches[6][1].blocker.id == 2);
	// watches[~b]'s 0th watcher clause is c1
	// watches[~b]'s 0th blocker is c (16)
	REQUIRE(s.watches[12][0].cref == 1);
	REQUIRE(s.watches[12][0].blocker.id == 16);
	// watches[~c]'s 0th watcher clause is c1
	// watches[~c]'s 0th blocker is b (13)
	REQUIRE(s.watches[17][0].cref == 1);
	REQUIRE(s.watches[17][0].blocker.id == 13);
	// watches[~e]'s 0th watcher clause is c2
	// watches[~e]'s 0th blocker is d (7)
	REQUIRE(s.watches[3][0].cref == 2);
	REQUIRE(s.watches[3][0].blocker.id == 7);

	
	// detach c0
	// watches[~a] (watches[1]) would become empty
	// watches[~d] (watches[6]) would have a size of 1
	//		- the one watcher left is [c2, 2]
	s.remove_clause(0);
	REQUIRE(s.watches[1].size() == 0);
	REQUIRE(s.watches[6].size() == 1);
	REQUIRE(s.watches[6][0].cref == 2);	
	REQUIRE(s.watches[6][0].blocker.id == 2);	


	// add two clauses
	// s.t. ~b (12) has a total of 3 watchers
	// [c1, 16], [c3, 0], [c4, 17]
	s.add_clause({b, ~a, e});
	s.add_clause({b, ~c, d});
	REQUIRE(s.watches[12].size() == 3);

	// detach c3
	// watches[~b] would become:
	// [c1, 16], [c4, 17]
	s.remove_clause(3);
	REQUIRE(s.watches[12].size() == 2);
	REQUIRE(s.watches[12][0].cref == 1);
	REQUIRE(s.watches[12][0].blocker.id == 16);
	REQUIRE(s.watches[12][1].cref == 4);
	REQUIRE(s.watches[12][1].blocker.id == 17);


	// detach c1
	// watches[~b] (12) becomes:
	// [c4, 17]
	// watches[~c] (17) becomes empty
	s.remove_clause(1);
	REQUIRE(s.watches[12].size() == 1);
	REQUIRE(s.watches[12][0].cref == 4);
	REQUIRE(s.watches[12][0].blocker.id == 17);
	REQUIRE(s.watches[17].size() == 0);


}


