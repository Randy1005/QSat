#include "memory.cuh"
#include "primitives.cuh"
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/cub.cuh>


namespace qsat {

//=============================//
//    CUDA memory management   //
//=============================//
const size_t hc_srsize = sizeof(SRef);
const size_t hc_scsize = sizeof(SClause);
//const size_t hc_otsize = sizeof(OT);
//const size_t hc_olsize = sizeof(OL);
const size_t hc_cnfsize = sizeof(CNF);
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

bool CuMM::resize_cnf(CNF*& cnf, const size_t& cls_cap, 
	const size_t& lits_cap) {

	assert(cls_cap && cls_cap <= UINT32_MAX);
	assert(lits_cap && lits_cap <= UINT32_MAX);
	
	const size_t cs_bytes = cls_cap * hc_srsize;
	const size_t data_bytes = cls_cap * hc_scsize + lits_cap * sizeof(uint32);

	assert(data_bytes % sizeof(uint32) == 0);

	const size_t min_cap = hc_cnfsize + data_bytes + cs_bytes;

	assert(min_cap);

	if (_cnf_pool.cap == 0) {
		assert(cnf == nullptr);
		assert(_cnf_pool.mem == nullptr);
	
		if (!has_unified_mem(min_cap, "CNF")) {
			return false;
		}
		CUDA_CHECK(cudaMallocManaged((void**)&_cnf_pool.mem, min_cap));

		if (_is_mem_advise_safe) {
			QLOGN2(2, "Advising GPU driver to favor global over system memory in %s call", 
				__func__);
			CUDA_CHECK(
				cudaMemAdvise(_cnf_pool.mem, min_cap, 
					cudaMemAdviseSetPreferredLocation, 0)
			);
			QLDONE(2, 5);
		}
		cnf = (CNF*)_cnf_pool.mem;
		const SRef data_cap = SRef(data_bytes / sizeof(uint32));
		new (cnf) CNF(data_cap, uint32(cls_cap));
		_d_cnf_mem = cnf->data().mem;
		_d_refs_mem = cnf->refs_data();	
		_cnf_pool.cap = min_cap;
	}

	return true;
}















} // end of namespace


