#include "sycl_memory_manager.hpp"

namespace qsat {

/**
 * @brief initialize device database
 *  initialize database informations:
 *  1. shared cnf formula
 *  2. shared clause indices
 */
void SyclMM::init_device_db(Solver& s) {
  
  // initialize memory for CNF on device
  assert(s.num_clauses() != 0);
  auto& cs = s.clauses();   
  
  // -----------------------------------------
  // TODO: first, parallel construct occurrence table on device 
  // ----------------------------------------
  
  
  
  
  
  
  std::vector<uint32_t> lits, indices;
  std::vector<ClauseInfo> cl_infos;
  
  indices.emplace_back(0);
  for (size_t i = 0; i < s.num_clauses(); i++) {
    const auto& ls = cs[i].literals;
    
    // calculate signature for clause
    cs[i].calc_signature();
    cl_infos.emplace_back(0, 0, 0, 0, 
                          ls.size()*sizeof(uint32_t),
                          0, cs[i].signature); 

    for (auto l : ls) {
      lits.emplace_back(static_cast<uint32_t>(l.id)); 
    }

    if (i >= 1) {
      indices.emplace_back(ls.size()+indices[i-1]);
    }
  }
 
  assert(lits.size() != 0);
  assert(indices.size() != 0);


  sh_cnf = sycl::mallocshared<uint32_t>(2*lits.size(), queue); 
  sh_idxs = sycl::mallocshared<uint32_t>(2*indices.size(), queue); 
  assert(sh_cnf); 
  assert(sh_idxs); 

  queue.memcpy(sh_cnf, lits.data(), sizeof(uint32_t)*lits.size());
  queue.memcpy(sh_idxs, indices.data(), sizeof(uint32_t)*indices.size());
  

  queue.parallel_for(sycl::range<1>(lits.size()), [this](sycl::id<1> i) {
    sh_cnf[i] *= 100;    
  }).wait();

}






} // -------------------------------- end of namespace
