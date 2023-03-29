#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

TEST_CASE("Device/Unified Memory Check" * doctest::timeout(300)) {
  
  qsat::CuMM cumm;
  
  auto res = cumm.has_unified_mem(3*MBYTE, "Sample");
  
  // no allocations in the beginning
  REQUIRE(res == false);


  size_t free, penalty;

  auto gpu_cnt = qsat::get_gpu_info(free, penalty);
  REQUIRE(gpu_cnt >= 0);

  // initialize cuda memory manager
  cumm.init(free, penalty);

  // check if memory manager has unified memory
  res = cumm.has_unified_mem(3*MBYTE, "Sample Allocate USM");
  REQUIRE(res == true);

  // check if memory manager has device memory
  res = cumm.has_device_mem(3*MBYTE, "Sample Allocate Device");
  REQUIRE(res == true);
}

TEST_CASE("Resize Device Literals" * doctest::timeout(300)) {
  
  qsat::CuMM cumm;
  

  size_t free, penalty;

  auto gpu_cnt = qsat::get_gpu_info(free, penalty);
  REQUIRE(gpu_cnt >= 0);

  // initialize cuda memory manager
  cumm.init(free, penalty);

  // allocate space of 1000 literlas
  auto* lits_pool_mem = cumm.resize_lits(1000);
  
} 


