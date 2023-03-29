#ifndef __CU_MEMORY_
#define __CU_MEMORY_

#include "simptypes.cuh"
namespace qsat {

/**
@ class CUDA memory manager
*/
class CuMM {
public:
	CuMM() {
		assert(this);
		memset(this, 0, sizeof(*this));
	}

	template<typename Pool>
	inline void free(Pool& pool) {
		if (pool.mem) {
			assert(pool.cap);
			CUDA_CHECK(cudaFree(pool.mem));
			pool.mem = nullptr;
			_cap -= pool.cap;
			assert(_cap >= 0);
			_free_mem += pool.cap;
			pool.cap = 0;
		}
	}


	template<typename Pool>
	inline void dfree(Pool& pool) {
		if (pool.mem) {
			assert(pool.cap);
			CUDA_CHECK(cudaFree(pool.mem));
			pool.mem = nullptr;
			_dcap -= pool.cap;
			assert(_dcap >= 0);
			_free_mem += pool.cap;
			pool.cap = 0;
		}
	}
	
	inline bool has_unified_mem(
		const size_t& min_cap, 
		const char* action) {
		// calculate how much memory we need for 
		// this allocation
		const int64 used = _cap + min_cap; 
		QLOG2(2, "Allocating GPU unified memory for %s (used/free = %.3f/%lld MB)", 
			action, (double)used / MBYTE, _free_mem / MBYTE);
		
		// if not enough free memory
		// skip simplify phase
		if (used >= _free_mem) {
			QLOGW("Not enough free memory for %s (curr used = %lld MB) \
				-> skip GPU simplifier", 
				action, used / MBYTE);
			
			return false;
		}

		// if we have enough memory
		// set mem capacity to current allocation size
		_cap = used;
		_free_mem -= used;

		assert(_free_mem >= 0);
		assert(_tot_mem >= _free_mem);
		assert(_tot_mem >= _cap);
		return true;
	}

	inline bool has_device_mem(
		const size_t& min_cap, 
		const char* action) {
		// calculate how much memory we need for 
		// this allocation
		const int64 used = _cap + min_cap; 
		QLOG2(2, "Allocating GPU device-only memory for %s (used/free = %.3f/%lld MB)", 
			action, (double)used / MBYTE, _free_mem / MBYTE);
		
		// if not enough free memory
		// skip simplify phase
		if (used >= _free_mem) {
			QLOGW("Not enough free memory for %s (curr used = %lld MB) \
				-> skip GPU simplifier", 
				action, used / MBYTE);
			
			return false;
		}

		// if we have enough memory
		// set mem capacity to current allocation size
		_cap = used;
		_free_mem -= used;

		assert(_free_mem >= 0);
		assert(_tot_mem >= _free_mem);
		assert(_tot_mem >= _cap);
		return true;
	}

	void init(const int64 total, const int64 penalty) {
		_penalty = penalty;
		_tot_mem = total;
		_free_mem = total - penalty;
	}

	bool alloc_hist(CuHist& cuhist);

	uint32* resize_lits(const size_t& min_lits);

private:
	CuPool _hist_pool;

	CuLits _lits_pool;

	// --------------------------
	// memory statistics trackers:
	// cap:		unified memory capacity
	// dcap:	device memory capacity
	// --------------------------
	int64 _cap, _dcap, _maxcap;
	int64 _free_mem, _tot_mem, _penalty;


};














} // end of namespace -----------------------------



#endif
