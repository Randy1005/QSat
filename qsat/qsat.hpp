#pragma once
#include <iostream>
#include <fstream>
#include <vector>
// #include <map>
#include <stack>
#include <algorithm>
#include <string>
#include <chrono>
// #include <unordered_map>
#include <filesystem>
#include "heap.hpp"
#include "intel_task_grammar.hpp"

namespace qsat {

struct Clause;
struct Literal;
struct VarInfo;

enum class Status {
  FALSE = 0,
  TRUE  = 1,
  UNDEFINED
};


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
-2  => 3  => 3/2 = 1
 3  => 4  => 4/2 = 2
-3  => 5  => 5/2 = 2
...
*/
struct Literal {
  friend struct Clause;
  friend class Solver;

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

  int id;
};

// constant for representing undefined literal
const Literal lit_undef = {-1};


/**
 * utility inline methods
 * var(lit), ~lit, signed(lit), etc.
 */
inline Literal operator ~(const Literal& p) {
  Literal q(p.id);
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
  Clause(const std::vector<Literal>& lits, bool undef = false);

  Clause(std::vector<Literal>&& lits, bool undef = false);

  /**
  @brief default copy assignment operator
  */
  Clause& operator=(const Clause& rhs) = default;

  /**
  @brief default move assignment operator
  */
  Clause& operator=(Clause&& rhs) = default;

  // TODO: implement == operator?
  bool is_undef = false;

  std::vector<Literal> literals;
};

// constant:
// an undefined/empty clause
const Clause cla_undef({}, true);

/**
 * @struct VarInfo
 * @brief stores a variable's reason clause
 * and its decision level
 */
struct VarInfo {
  VarInfo() = default;

  VarInfo(const Clause& c, int lvl) :
    reason(c),
    decision_level(lvl)
  {
	}
  
	Clause reason;
  int decision_level;
};


/**
 * @struct Watcher
 * @brief stores a clause watching a specified literal
 * and a blocker literal that gets affected by the 
 * specified literal (used in the lit-vec<Watcher> mapping 'watches')
 */
struct Watcher {
	Clause& cref;
	Literal blocker;

	Watcher(Clause& cr, Literal p) :
		cref(cr),
		blocker(p)
	{
	}
	
};


/**
@class Solver
@brief a class to create a SAT solver object
*/
class Solver {
public: 
  /**
  @brief constructs a solver object
  */
  Solver();

  /**
  @brief reads in dimacs cnf file, and store the literals and clauses
  @param inputFileName the dimacs cnf file name
  */
  void read_dimacs(const std::string& inputFileName);
  
  /**
  @brief dumps the solver info via std::ostream
  @code{.cpp}
  solver.dump(std::cout); // dumps the solver info to standard output

  std::ofstream ofs("info.dmp"); 
  solver.dump(ofs); // dumps the solver info to the file "info.dmp"
  @endcode

  @param os output stream to dump info to
  */
  void dump(std::ostream& os) const;

  /**
  @brief solves the given cnf expression

  Determine whether the given cnf is satisfiable or not.
  If satisfiable, also construct a solution for the user
  @returns true if the input cnf is satisfiable, otherwise return false
  */
  bool solve();

  /**
  @brief a getter method for the stored clauses
  */
  const std::vector<Clause>& clauses() const; 
  
  /**
  @brief adds a clause given a vector of literals (using move semantics)
  */
  void add_clause(std::vector<Literal>&& lits);

  /**
  @brief adds a clause given a vector of literals (using copy semantics)
  */
  void add_clause(const std::vector<Literal>& lits);


	// TODO:
	// how do I implement FAST detaching?
	// Clauses' indices are not fixed
	// they might get moved around during solving
	// should I still use std::vector for storing clauses?

	/**
	 * @brief attach clause
	 * initialize the watched literals for
	 * newly added clauses
	 */
	void attach_clause(const Clause& c);
	
	/**
	 * @brief detach clause
	 * inverse action of attach, remove the watchers
	 */
	void detach_clause(const Clause& c);


  size_t num_clauses() const { 
    return _clauses.size(); 
  }
  size_t num_variables() const {
    return _assigns.size(); 
  }
  size_t num_assigns() const { 
    return _trail.size(); 
  }
  
  size_t decision_level() const {
    return _trail_lim.size();
  }

  // TODO: this shouldn't be a public interface
  // but I need this to unit test literal op functionalities
  void assign(int v, bool val) {
    _assigns[v] = val ? Status::TRUE : Status::FALSE;
  }

  /**
   * @brief value
   * @in v the variable id
   * returns the evaluated value of a variable
   */
  inline Status value(int v) const {
    return _assigns[v];
  }
  
  /**
   * @brief value
   * @in p the literal id
   * returns the evaluated value of a literal
   */
  inline Status value(const Literal& p) const {
    if (_assigns[var(p)] == Status::UNDEFINED) {
      return Status::UNDEFINED;
    }
    else {
      return static_cast<int>(_assigns[var(p)]) ^ sign(p) ? 
        Status::TRUE : 
        Status::FALSE;
    }
  }
  

  inline bool unchecked_enqueue(const Literal &p, const Clause& from) {
    assert(value(p) == Status::UNDEFINED);

    // make the assignment, so this literal
    // evaluates to true
    _assigns[var(p)] = static_cast<Status>(!sign(p)); 
    
    // store decision level and reason clause
    _var_info[var(p)] = VarInfo{from, static_cast<int>(decision_level())};
 
    // push this literal into trail
    _trail.push_back(p);
    
    return true;
  }
  
  /**
   * @brief enqueue
   * if value(p) is evaluated, check for conflict
   * else store this new fact, update assignment, trail, etc.
   */
  inline bool enqueue(const Literal& p, const Clause& from = cla_undef) {
    return value(p) != Status::UNDEFINED ? 
      value(p) != Status::FALSE : 
      unchecked_enqueue(p, from); 
  }


  void reset();
  void read_dimacs(std::istream&);


  bool transpile_task_to_z3(const std::string& task_file_name);
  bool transpile_task_to_dimacs(const std::string& task_file_name);

private:

  /**
  @brief utility method that reads in a parsed symbol, encode into a literal and store it
  @param[in] in the parsed symbol
  @param[out] lits the literal vector to store all the encoded literals (for each clause) 
  */
  void _read_clause(int symbol, std::vector<Literal>& lits);
  void _init();
  void _print_assigns();

  /**
   * @brief insert variable order:
   * inserts a variable into the heap
   * if it's not already in there
   * @param v the variable to insert
   */
  // TODO: minisat has a "decision" vector to 
  // mark if a var can be used as decision
  // but seems like in its implementation
  // every var can be used as decision
  // ignore "decision" for now, but keep in mind
  // it has this unused feature
  void _insert_var_order(int v);
 

  /**
   * @brief new variable:
   * constructs a new SAT variable
   * also updates related data structures
   * e.g. activities, order_heap ...
   * P.S. invoked during parsing
   */
  void _new_var(int new_v);


  std::vector<Clause> _clauses; 
  
  // assignment vector 
  std::vector<Status> _assigns;
  
  // heuristic activities for variables
  std::vector<double> _activities;
 
  // var info vector (reason, decision level)
  std::vector<VarInfo> _var_info;


  // priority queue 
  // for selecting var with max activity
  Heap _order_heap;
  
  // trail 
  // keeps track of the literals we made decisions on
  // (all decision levels)
  std::vector<Literal> _trail;

  // trail_lim
  // keeps track of trail sizes for each decision level
  // _trail is 1-D, but with _trail_lim we know how many 
  // decisions are in a single decision level
  std::vector<int> _trail_lim;


	// watches
	// watches[lit] maps to a list of watchers 
	// watching 'lit'
	std::vector<std::vector<Watcher>> watches;

  // output file stream to write to z3py
  std::ofstream _z3_ofs;
};


}  // end of namespace --------------------------------------------------------





