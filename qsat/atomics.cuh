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
	T warp_res;
	
	// leader thread of this group 
	// performs the addition
	if (g.thread_rank() == 0) {
		warp_res = atomicAdd(counter, g.size());
	}

	return g.shfl(warp_res, 0) + g.thread_rank();
}	













}

#endif
