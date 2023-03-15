#include <vector>


namespace qsat {

class SClause {
  
public:  
  SClause() = default;
  
  // state: ORIGINAL, LEARNT, DELETED 
  char state;

  // flag: contributes to gate extraction?
  // added: is this a resolvent?
  // used: how long this clause should be used before
  //       clause database reduction
  char flag, added, used;
  
  // lbd: literal blocking distance
  //      (i.e. the number of decision levels
  //      contributing to a conflict)
  int size, lbd;

  // clause signature: hash its literals to a 32-bit value
  uint32_t signature;

  // NOTE: the implementation in ParaFrost is
  // a union of array<uint32_t>
  // with a total size of 20 bytes
  //
  // using a vector would consume 40 bytes
  // -> an empty vector is already 24 bytes
  std::vector<uint32_t> literals;

};


class CNF {
public:
  struct {
    SClause* mem;
    // size is always <= capacity
    uint64_t size, cap;
  } clauses;
  
  struct {
    uint64_t* mem;
    uint32_t size, cap;
  } references;

};






} // end of namespace ------------------------------------------
