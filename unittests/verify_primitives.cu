#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/primitives.cuh>

TEST_CASE("Warp Aggregated Atomics" * doctest::timeout(300)) {

  cudaDeviceSynchronize();
}


