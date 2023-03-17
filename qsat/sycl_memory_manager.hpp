#include "taskflow/taskflow.hpp"
#include "taskflow/sycl/syclflow.hpp"
#include <vector>

namespace qsat {

class Solver;
struct Literal;
struct Clause;

// @brief shared clause info
// state: ORIGINAL, LEARNT, DELETED
// added: is resolvent?
// flag:  contributes to gate extraction?
// lbd:   literal block distance (look up glucose)
// size:  clause size in bytes
// sig:   clause signature (hash value of 32 bits)
// used:  how long a LEARNT clause should be used 
//        before deleted by database reduction
struct ClauseInfo {
  ClauseInfo(const char state, 
      const char added, 
      const char flag,
      char used,
      int size,
      int lbd,
      uint32_t sig) :
    state(state),
    added(added),
    flag(flag),
    used(used),
    size(size),
    lbd(lbd),
    sig(sig)
  {
  }

  

  char state;
  char added, flag;
  char used;
  int size, lbd;
  uint32_t sig;
};




struct SyclMM {

  SyclMM() = default;

  void init_device_db();

  // @brief sycl task queue
  sycl::queue queue;

  // @brief task flow object
  tf::Taskflow tf;

  // @brief taskflow executor
  tf::Executor executor;
};


struct DeviceData {
  // @brief shared cnf
  // literals stored in shared space
  // between host and device
  uint32_t* sh_cnf; 
    
  // @brief shared clause indices
  // indices to record clause c starts
  // on nth literal
  //
  // indices[c] -> n
  uint32_t* sh_idxs;

 

}



} // -------------------------- end of namespace
