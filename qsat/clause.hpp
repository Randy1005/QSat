#pragma once
#include <vector>
#include "datatypes.hpp"

namespace qsat {


/**
@struct Literal
@brief struct to create a literal

A literal is created from a given integer variable based on the following
encoding method:

v is positive => id = 2|v| - 2 => assignment id/2
v is negative => id = 2|v| - 1 => assignment id/2

var => id => assignment
 1  => 0  => 0/2 = 0
-1  => 1  => 1/2 = 0
 2  => 2  => 2/2 = 1
-3  => 5  => 5/2 = 2
...
*/
struct Literal {
  friend struct Clause;
  friend class Solver;
	
	Literal() = default;
  Literal(const Literal& p) = default;

  /**
  @brief constructs a literal with a given variable
  */
  Literal(int var);

  bool operator == (const Literal& p) const {
    return id == p.id;
  }
  
  bool operator != (const Literal& p) const {
    return id != p.id;
  }

  int id = -1;
};

// constant for representing undefined literal
const Literal LIT_UNDEF;

// constant for representing undefined variable
const int VAR_UNDEF = -1;

/**
 * utility inline methods
 * var(lit), ~lit, signed(lit), etc.
 */
inline Literal operator ~(const Literal& p) {
  Literal q = LIT_UNDEF;
  q.id = p.id ^ 1;
  return q;
}

inline int var(const Literal& p) {
  return p.id >> 1;
}

inline bool sign(const Literal& p) {
  return p.id & 1;
}





/**
@struct Clause
@brief struct to create a clause
*/
struct Clause {
  
  Clause() = default;
  Clause(const Clause&) = default;
  Clause(Clause&&) = default;

  /**
  @brief constructs a clause with given literals using copy semantics
  */
  Clause(const std::vector<Literal>& lits, bool is_learnt = false);

  Clause(std::vector<Literal>&& lits, bool is_learnt = false);

  /**
  @brief default copy assignment operator
  */
  Clause& operator=(const Clause& rhs) = default;

  /**
  @brief default move assignment operator
  */
  Clause& operator=(Clause&& rhs) = default;


  void calc_signature();

  std::vector<Literal> literals;

	// a learnt clause or not
	bool learnt = false;

	// clause activity
	// (for reducing clause database)
	double activity = 0;

	// marking this clause as deleted (mark = 1)
	// or not (mark = 0)
	unsigned int mark = 0;

	// relocation index
	// when we reduce database
	// some clauses will be moved around
	// we record where they will be moved to
	// -1 means not relocated
	int reloc = -1;

  // signature (hashed to 32-bit)
  uint32_t signature;
};

// constant:
// an undefined/empty clause id
const int CREF_UNDEF = -1; 


} // end of namespace ---------------------------------------
