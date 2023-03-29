#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

TEST_CASE("Resize Device Literals" * doctest::timeout(300)) {
  
  qsat::CuMM cumm;
  
  auto res = cumm.has_unified_mem(100, "Sample");
  
  // no allocations in the beginning
  REQUIRE(res == false);


  size_t free, penalty;

  auto gpu_cnt = qsat::get_gpu_info(free, penalty);
  REQUIRE(gpu_cnt >= 0);

  // initialize cuda memory manager
  cumm.init(free, penalty);


  res = cumm.has_unified_mem(100, "Sample2");
  REQUIRE(res == true);
}


