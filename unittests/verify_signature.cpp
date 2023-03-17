#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

// Unittest: basic interface usage
TEST_CASE("Clause Signatures" * doctest::timeout(300)) {
  qsat::Solver s;

  auto& cs = s.clauses();

  // clause 0 has lits: 0, 2 (vars 0, 1)
  // sig(0) = 0 & 31 = 0
  // sig(2) = 1 & 31 = 1
  // sig(clause 0) = 2^0 | 2^1 = 110000....0 (only bit 0 and bit 1 is set)
  s.add_clause({ qsat::Literal{1}, qsat::Literal{2} });
  cs[0].calc_signature();

  REQUIRE(cs[0].signature != 0);
  REQUIRE(cs[0].signature == 3);
  
  // clause 1 has lits: 0, 2, 5 (vars 0, 1, 2)
  // sig(0) = 0 & 31 = 0
  // sig(2) = 1 & 31 = 1
  // sig(5) = 2 & 31 = 2
  // sig(clause 1) = 2^0 | 2^1| 2^2 = 111000....0 (only bit 0, bit 1, bit 2 is set)
  s.add_clause({ qsat::Literal{1}, qsat::Literal{2}, qsat::Literal{-3} });
  cs[1].calc_signature();
  
  REQUIRE(cs[1].signature != 0);
  REQUIRE(cs[1].signature == 7);
  

  // clause 2 has lits: 128, 70, 56 (vars 64, 35, 28)
  // sig(64) = 64 & 31 = 0
  // sig(35) = 35 & 31 = 3
  // sig(28) = 28 & 31 = 28
  s.add_clause({ qsat::Literal{65}, qsat::Literal{36}, qsat::Literal{29} });
  cs[2].calc_signature();
  uint32_t sig2 = (1 | 1<<3 | 1<<28); 
  REQUIRE(cs[2].signature != 0);
  REQUIRE(cs[2].signature == sig2);
}


