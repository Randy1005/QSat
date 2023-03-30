#include "memory.cuh"
#include "primitives.cuh"
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/cub.cuh>


namespace qsat {

//=============================//
//    CUDA memory management   //
//=============================//
const size_t hc_srsize = sizeof(S_Ref);
const size_t hc_scsize = sizeof(SClause);
//const size_t hc_otsize = sizeof(OT);
//const size_t hc_olsize = sizeof(OL);
//const size_t hc_cnfsize = sizeof(CNF);
const size_t hc_varsize = sizeof(uint32);
//const size_t hc_cuvecsize = sizeof(cuVecU);



uint32* CuMM::resize_lits(const size_t& min_lits) {
	assert(min_lits);

	// we need minimum capacity of numlits * sizeof(uint32)
	const size_t min_cap = min_lits * hc_varsize;
	
	// not enough memory to store literals
	if (_lits_pool.cap < min_cap) {
		dfree(_lits_pool);
		assert(_lits_pool.mem == nullptr);
		if (!has_device_mem(min_cap, "Literals")) {
			return nullptr;
		}
		CUDA_CHECK(cudaMalloc((void**)&_lits_pool.mem, min_cap));
		_lits_pool.cap = min_cap;
		_lits_pool.size = min_lits;

	}
		
	return _lits_pool.mem;
}


bool CuMM::alloc_hist(CuHist& cuhist) {
	assert(cnf_info.n_dual_vars == V2L(cnf_info.max_var+1ULL));
	

}















} // end of namespace


