#ifndef _CU_DEFS
#define _CU_DEFS

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "logging.hpp"

namespace qsat {

extern cudaDeviceProp dev_prop;
extern uint32 max_gpu_threads;
extern size_t max_gpu_shared_mem;

// global variables
cudaDeviceProp dev_prop;
uint32 max_gpu_threads;
size_t max_gpu_shared_mem;

bool quiet_en = false;
int verbose = 2;



__forceinline__ void CUDA_CHECK(cudaError_t returncode)
{
#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
	if (returncode != cudaSuccess) {
		QLOGEN("CUDA runtime failure due to %s", cudaGetErrorString(returncode));
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}
#endif
}

__forceinline__ void sync(const cudaStream_t& stream = 0) 
{
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

#define sync_all() CUDA_CHECK(cudaDeviceSynchronize());






} // end of namespace -----------------------------------------------



#endif
