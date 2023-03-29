#ifndef __GL_ATOMIC_
#define __GL_ATOMIC_

#include "definitions.cuh"
#include "datatypes.hpp"
#include <cooperative_groups.h>

namespace coopg = cooperative_groups;

namespace qsat {
// warp aggregated atomic add
// see: 
// https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
template<class T>
_QSAT_D_ T atomic_agg_inc(T* counter) {
	// get a group of currently coalesced threads
	auto g = coopg::coalesced_threads();	

	printf("coop group size=%lu", g.size());

}	














}

#endif
