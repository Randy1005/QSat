#include <thrust/device_ptr.h>
#include "constants.hpp"
#include "definitions.cuh"
#include "definitions.hpp"
#include "sclause.hpp"
#include "clause.hpp"

namespace qsat {

typedef thrust::device_ptr<uint32_t> t_iptr;

struct CuPool {
	addr_t mem;
	size_t cap;
};

struct CuLits {
	// NOTE: let's attempt using
	// cub's functionality

	uint32* mem;
	size_t size, cap;
};

struct CuHist {

	// NOTE: not sure what this is ...
	S_Ref* d_segs;

	uint32* d_hist, *h_hist;
	
	// NOTE: not sure what this is ...
	uint32* d_vorg;

	t_iptr thrust_hist;

	CuHist() :
		d_segs(nullptr),
		d_hist(nullptr),
		h_hist(nullptr),
		d_vorg(nullptr)
	{}

	~CuHist() {
		h_hist = nullptr;
		d_hist = nullptr;
		d_segs = nullptr;
		d_vorg = nullptr;
	}


	inline uint32	operator[] (const uint32& i) const { 
		assert(h_hist && i < cnf_info.n_dual_vars); 
		return h_hist[i]; 
	}
	
	inline uint32& operator[]	(const uint32& i) { 
		assert(h_hist && i < cnf_info.n_dual_vars); 
		return h_hist[i]; 
	}
	inline void	cache_hist	(const cudaStream_t& s = 0) {
		CUDA_CHECK(
			cudaMemcpyAsync(h_hist, 
				d_hist, 
				cnf_info.n_dual_vars * sizeof(uint32), 
				cudaMemcpyDeviceToHost, 
				s)
		);
	}
	inline void	fetch_vars	(const uint32* vorg, const cudaStream_t& s = 0) {
		CUDA_CHECK(
			cudaMemcpyAsync(d_vorg, 
				vorg, 
				(cnf_info.max_var + 1) * sizeof(uint32), 
				cudaMemcpyHostToDevice, 
				s)
		);
	}

};






} // end of name space ----------------------------------
