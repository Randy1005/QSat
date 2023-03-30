#ifndef __GRID_CONFIG
#define __GRID_CONFIG


#include "definitions.cuh"
#include "datatypes.hpp"

namespace qsat {

typedef uint32 Gridtype;

// linearized thread id x / block id x
#define global_bx (Gridtype)(blockDim.x * blockIdx.x)
#define global_bx_off (Gridtype)((blockDim.x << 1) * blockIdx.x)
#define global_tx (Gridtype)(global_bx + threadIdx.x)

#define global_tx_off	(Gridtype)(global_bx_off + threadIdx.x)
#define stride_x (Gridtype)(blockDim.x * gridDim.x)
#define stride_x_off (Gridtype)((blockDim.x << 1) * gridDim.x)
// linearized thread id y / block id y
#define global_by	(Gridtype)(blockDim.y * blockIdx.y)
#define global_ty	(Gridtype)(global_by + threadIdx.y)
#define stride_y	(Gridtype)(blockDim.y * gridDim.y)


// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS)							     \
		(((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

#define OPTIMIZEBLOCKS(DATALEN, NTHREADS)                                \
		assert(DATALEN);                                                 \
		assert(NTHREADS);                                                \
		assert(max_gpu_threads);                                           \
		const Gridtype REALBLOCKS = ROUNDUPBLOCKS(DATALEN, NTHREADS);    \
		const Gridtype MAXBLOCKS = max_gpu_threads / NTHREADS;             \
		const Gridtype nBlocks = MIN(REALBLOCKS, MAXBLOCKS);             \



}

#endif
