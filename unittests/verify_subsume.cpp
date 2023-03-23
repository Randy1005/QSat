#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

// Unittest: subset check 
TEST_CASE("Subset Check" * doctest::timeout(300)) {
  qsat::Solver s;

  uint32_t a[] = {0, 2, 3, 4};
  uint32_t b[] = {0, 2, 3, 4, 6};
  
  REQUIRE(s.is_subset(b, 5, a, 4) == true);


  uint32_t c[] = {0, 1, 3, 5};
  uint32_t d[] = {0, 1, 2, 5, 7};
  REQUIRE(s.is_subset(d, 5, c, 4) == false);
  

  REQUIRE(s.is_subset(d, 5, c, 2) == true);
  REQUIRE(s.is_subset(d, 5, c, 3) == false);


}

