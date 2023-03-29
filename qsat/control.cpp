#include "definitions.cuh"
#include "control.hpp"

namespace qsat {

inline int SM2Cores(int major, int minor) {
	typedef struct { int SM; int cores; } SM;

	SM n_cores[] = {
		{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
		{0x50, 128}, {0x52, 128}, {0x53, 128},
		{0x60,  64}, {0x61, 128}, {0x62, 128},
		{0x70,  64}, {0x72,  64}, {0x75,  64},
		{0x80,  64}, {0x86, 128}, {0x87, 128},
		{-1, -1}
	};

	int index = 0;
	while (n_cores[index].SM != -1) {
		if (n_cores[index].SM == ((major << 4) + minor)) {
			return n_cores[index].cores;
		}
		index++;
	}
	QLOGW("cannot map to cores/SM due to unknown SM");
	return -1;
}



int get_gpu_info(size_t& free, size_t& penalty) {
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  if (!dev_count) return 0;

  // 0 represents MASTER_GPU
  CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));
  assert(dev_prop.totalGlobalMem);
  if (dev_prop.warpSize != 32) QLOGEN("GPU warp size not supported");
 
  // NOTE: not sure what penalty is?
  penalty = 220 * MBYTE; 
  
  // assign total global memory as free space
  free = dev_prop.totalGlobalMem;
 
  size_t shared_penalty = 512;

  max_gpu_threads = 
    dev_prop.multiProcessorCount * dev_prop.maxThreadsPerMultiProcessor;
  max_gpu_shared_mem = dev_prop.sharedMemPerBlock - shared_penalty;

  if (!quiet_en) {
		QLOG1(" Available GPU: %d x %s%s @ %.2fGHz%s (compute cap: %d.%d)", 
			dev_count, CREPORTVAL, dev_prop.name, 
      ratio((double)dev_prop.clockRate, 1e6), 
      CNORMAL, dev_prop.major, dev_prop.minor);

		const int cores = SM2Cores(dev_prop.major, dev_prop.minor);
		QLOG1(" Available GPU Multiprocessors: %d MPs (%s cores/MP)", 
      dev_prop.multiProcessorCount, (cores < 0 ? "unknown" : std::to_string(cores).c_str()));
		QLOG1(" Available Global memory: %zd GB", free / GBYTE);
		QLOG1(" Max GPU threads: %zu", max_gpu_threads);
  }

  return dev_count;
}













} // end of namespace -------------------------------
