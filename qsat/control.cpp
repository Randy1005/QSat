#include "control.hpp"

namespace qsat {

int get_gpu_info(size_t& free) {
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  if (!dev_count) return 0;

  // 0 represents MASTER_GPU
  CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));
  assert(dev_prop.totalGlobalMem);
  if (dev_prop.warpSize != 32) QLOGEN("GPU warp size not supported");
  
  std::cout << "total global mem=" << dev_prop.totalGlobalMem << '\n';

}













} // end of namespace -------------------------------
