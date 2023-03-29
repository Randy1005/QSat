#pragma once
#include "clause.hpp"

namespace qsat {

class SClause {
public:
  SClause() :
    _state(ORIGINAL),
    _flag(0),
    _added(0),
    _used(0),
    _lbd(0),
    _sig(0),
    _sz(0)
  {}

  SClause(const Clause& c) {
    init(c); 
  }

  SClause(const std::vector<Literal>& lits) {
    init(lits);
  }

  inline void init(const Clause& c) {
    _state = c.learnt;
    _sz = c.literals.size();
    _sig = 0;
    _flag = 0;
    _added = 0;
    assert(original() == !c.learnt);
    assert(!c.mark);
    // TODO: Consider LBD later
    //if (learnt()) {
    //  _lbd = src.lbd() & MAX_LBD_M;
    //  _u = src.usage();
    //}
    //else { _lbd = 0, _u = 0; }
    copy_from(c.literals);
  }

  inline void init(const std::vector<Literal>& lits) {
   	_state = ORIGINAL;
    _sz = lits.size();
    _lbd = 0;
    _sig = 0;
    _flag = 0;
    _added = 0;
    _used = 0;
    copy_from(lits);
  }



  template<typename Src>
  inline void copy_from(const Src& from) {
    assert(_sz);
    for (int i = 0; i < _sz; i++) {
      _lits[i] = from[i].id;  
    }
  }

  inline bool	added() const { 
    return _added; 
  }
  inline bool	empty() const { 
    return !_sz; 
  }
  inline bool	original() const { 
    return !_state; 
  }
  inline bool	deleted() const { 
    return _state & DELETED; 
  }
  inline bool	learnt() const { 
    return _state & LEARNT; 
  }

  inline void	calc_sig(const uint32& init_sig = 0) {
    _sig = init_sig;
    for (int i = 0; i < _sz; i++)
      _sig |= MAPHASH(_lits[i]);
  }


private:
  unsigned _state, _flag, _added, _used;
  unsigned _lbd;
  uint32 _sig;
  uint32 _lits[1]; 
  int _sz;

};

typedef SClause* S_Ref;


} // end of namespace ------------------------------
