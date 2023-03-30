#ifndef __SCLAUSE_
#define __SCLAUSE_

#include "definitions.cuh"
#include "clause.hpp"

namespace qsat {

// -------------------------------------
// Abstract Clause for GPU Inprocessing
// -------------------------------------

class SClause {
public:
  // _QSAT_H_D_ SRef		blockSize      () const { assert(_sz); return _sz + (sizeof(*this) / sizeof(uint32)); }
  _QSAT_H_D_ size_t		capacity       () const { assert(_sz); return size_t(_sz) * sizeof(uint32) + sizeof(*this); }
  _QSAT_H_D_			SClause        () :
    _st(ORIGINAL)
    , _f(0)
    , _a(0)
    , _u(0)
    , _lbd(0)
    , _sig(0)
    , _sz(0)
  {}
  _QSAT_H_D_			SClause        (uint32* lits, const int& size) :
    _st(ORIGINAL)
    , _f(0)
    , _a(0)
    , _u(0)
    , _lbd(0)
    , _sig(0)
    , _sz(size)
  {
    copyLitsFrom(lits);
  }

  _QSAT_H_D_			SClause        (SClause& src) :
    _st(src.status())
    , _f(src.molten())
    , _a(src.added())
    , _u(src.usage())
    , _lbd(src.lbd())
    , _sig(src.sig())
    , _sz(src.size())
  {
    assert(_lbd <= MAX_LBD);
    assert(!src.deleted());
    copyLitsFrom(src);
  }
  _QSAT_H_D_ void		copyLitsFrom   (uint32* src) {
    assert(_sz);
    for (int k = 0; k < _sz; k++) {
      assert(src[k] > 1);
      _lits[k] = src[k];
    }
  }
  _QSAT_H_D_ void		resize         (const int& size) { _sz = size; }
  _QSAT_H_D_ void		push           (const uint32& lit) { _lits[_sz++] = lit; }
  _QSAT_H_D_ void		set_lbd        (const unsigned& lbd) { assert(_lbd < MAX_LBD); _lbd = lbd; }
  _QSAT_H_D_ void		set_sig        (const uint32& sig) { _sig = sig; }
  _QSAT_H_D_ void		set_usage      (const CL_ST& usage) { _u = usage; }
  _QSAT_H_D_ void		set_status     (const CL_ST& status) { _st = status; }
  _QSAT_H_D_ uint32&    operator[]	   (const int& i) { assert(i < _sz); return _lits[i]; }
  _QSAT_H_D_ uint32		operator[]	   (const int& i) const { assert(i < _sz); return _lits[i]; }
  _QSAT_H_D_ uint32*    data           (const int& i = 0) { assert(i < _sz); return _lits + i; }
  _QSAT_H_D_ uint32*    end            () { return _lits + _sz; }
  _QSAT_H_D_ uint32		back           () { assert(_sz); return _lits[_sz - 1]; }
  _QSAT_H_D_ uint32		back           () const { assert(_sz); return _lits[_sz - 1]; }
  _QSAT_H_D_ operator   uint32*        () { assert(_sz); return _lits; }
  _QSAT_H_D_ void		pop            () { _sz--; }
  _QSAT_H_D_ void		clear          () { _sz = 0; }
  _QSAT_H_D_ void		freeze         () { _f = 0; }
  _QSAT_H_D_ void		melt           () { _f = 1; }
  _QSAT_H_D_ void		markAdded      () { _a = 1; }
  _QSAT_H_D_ void		markDeleted    () { _st = DELETED; }
  _QSAT_H_D_ CL_ST		usage          () const { return _u; }
  _QSAT_H_D_ bool		molten         () const { return _f; }
  _QSAT_H_D_ bool		added          () const { return _a; }
  _QSAT_H_D_ bool		empty          () const { return !_sz; }
  _QSAT_H_D_ bool		original       () const { return !_st; }
  _QSAT_H_D_ bool		deleted        () const { return _st & DELETED; }
  _QSAT_H_D_ bool		learnt         () const { return _st & LEARNT; }
  _QSAT_H_D_ CL_ST		status         () const { return _st; }
  _QSAT_H_D_ int		size           () const { return _sz; }
  _QSAT_H_D_ unsigned	lbd            () const { return _lbd; }
  _QSAT_H_D_ uint32		sig            () const { return _sig; }
  _QSAT_H_D_ void		shareTo        (uint32* dest) {
    assert(_sz > 1);
    for (int k = 0; k < _sz; k++) {
      assert(_lits[k] > 1);
      dest[k] = _lits[k];
    }
  }
  _QSAT_H_D_ bool		has            (const uint32& lit) const { // binary search
    assert(_sz);
    if (_sz == 2) {
      if (_lits[0] == lit || _lits[1] == lit) return true;
      else return false;
    }
    else {
      assert(isSorted());
      int low = 0, high = _sz - 1, mid;
      uint32 first = _lits[low], last = _lits[high];
      while (first <= lit && last >= lit) {
        mid = ((low + high) >> 1);
        uint32 m = _lits[mid];
        if (m < lit) first = _lits[low = mid + 1];
        else if (m > lit) last = _lits[high = mid - 1];
        else return true; // found
      }
      if (_lits[low] == lit) return true; // found
      else return false; // Not found
    }
  }
  _QSAT_H_D_ bool		isSorted       () const {
    for (int i = 0; i < _sz; i++)
      if (i > 0 && _lits[i] < _lits[i - 1])
        return false;
    return true;
  }
  _QSAT_H_D_ int		hasZero        () const {
    for (int i = 0; i < _sz; i++)
      if (!_lits[i])
        return i;
    return -1;
  }
  _QSAT_H_D_ void		print          () const {
    printf("(");
    for (int l = 0; l < _sz; l++) {
      int lit = int(ABS(_lits[l]));
      lit = (SIGN(_lits[l])) ? -lit : lit;
      printf("%4d ", lit);
    }
    char st = 'U';
    if (deleted()) st = 'X';
    else if (added()) st = 'A';
    else if (original()) st = 'O';
    else if (learnt()) st = 'L';
    printf(") %c:%d, used=%d, lbd=%d, s=0x%X\n", st, molten(), usage(), _lbd, _sig);
  }

private:
  unsigned _st : 2, _f : 1, _a : 1, _u : 2;
  unsigned _lbd : 26;
  uint32 _sig;
  int _sz;
  uint32 _lits[];

};

typedef SClause* SRef;


} // end of namespace ------------------------------

#endif
