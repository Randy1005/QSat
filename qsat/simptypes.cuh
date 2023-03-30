#ifndef __SIMP_TYPES_
#define __SIMP_TYPES_

#include <thrust/device_ptr.h>
#include "constants.hpp"
#include "definitions.cuh"
#include "definitions.hpp"
#include "sclause.hpp"
#include "clause.hpp"
#include "vector.cuh"

namespace qsat {


// -----------------------
// Global Simplifier Types
// -----------------------

typedef thrust::device_ptr<uint32_t> t_iptr;
typedef CuVec<uint32> CuVecU;
typedef CuVec<Byte> CuVecB;
typedef CuVec<SRef> CuSRef;




struct CuPool {
	addr_t mem;
	size_t cap;
};

struct CuCNF {
	uint32* mem;
	SRef size, cap;
};

struct CuLits {
	// NOTE: let's attempt using
	// cub's functionality

	uint32* mem;
	size_t size, cap;
};

struct CuHist {

	// NOTE: not sure what this is ...
	SRef* d_segs;

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
	inline void	cache_hist(const cudaStream_t& s = 0) {
		CUDA_CHECK(
			cudaMemcpyAsync(h_hist, 
				d_hist, 
				cnf_info.n_dual_vars * sizeof(uint32), 
				cudaMemcpyDeviceToHost, 
				s)
		);
	}
	inline void	fetch_vars(const uint32* vorg, const cudaStream_t& s = 0) {
		CUDA_CHECK(
			cudaMemcpyAsync(d_vorg, 
				vorg, 
				(cnf_info.max_var + 1) * sizeof(uint32), 
				cudaMemcpyHostToDevice, 
				s)
		);
	}

};

// ---------------------------
// Simplifier CNF
// ---------------------------
class CNF {

public:
	_QSAT_H_D_ CNF(const SRef& data_cap, const uint32& cs_cap):
		_bucket((Byte)sizeof(uint32))
	{
		assert(_bucket == sizeof(uint32));
		assert(data_cap);
		assert(cs_cap);
		SRef* cs_mem = (SRef*)(this + 1);
		_refs.alloc(cs_mem, cs_cap);	
		_data.mem = (uint32*)(cs_mem + cs_cap);
		_data.cap = data_cap;
		_data.size = 0;
	}

	_QSAT_H_D_ SRef* refs_data(const uint32& i = 0) {
		return _refs + i;
	}

	_QSAT_H_D_ CuCNF& data() {
		return _data;
	}
	
	_QSAT_H_D_ uint32 size() {
		return _refs.size();
	}

	_QSAT_H_D_ SRef ref(const uint32& i) {
		assert(i < _refs.size());
		return _refs[i];
	}

	_QSAT_H_D_ const SRef& ref(const uint32& i) const {
		assert(i < _refs.size());
		return _refs[i];
	}

	// calculates how many bytes n literals occupy
	_QSAT_H_D_ size_t calc_bytes(const int& nlits) {
		return sizeof(SClause) + nlits * sizeof(uint32);
	}

	// function to construct sclause
	// from clause on host
	_QSAT_H_ void new_clause(Clause& src) {
		assert(src.literals.size() > 1);	
		size_t c_bytes = calc_bytes(src.literals.size());
		assert(_data.size < _data.cap);
	}


private:
	CuCNF _data;
	CuSRef _refs;
	Byte _bucket;



};






} // end of name space ----------------------------------
#endif
