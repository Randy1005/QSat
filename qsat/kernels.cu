/**
 * some more headers here:
 * bounded.cuh
 * subsume.cuh
 * blocked.cuh
 * redundancy.cuh
 * simplify.cuh
 */



#include "cub/device/device_scan.cuh"

namespace qsat {


__device__ uint32 gcounter;

__global__ void reset_counter() {
	gcounter = 0;
}







__global__ void copy_if_k(uint32* __restrict__ dst, CNF* __restrict__ src) {



}







}
