#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <string>
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
	
	Literal() = default;

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
  Clause(const std::vector<Literal>& lits, bool is_learnt  = false);

  Clause(std::vector<Literal>&& lits, bool is_learnt = false);

  /**
  @brief default copy assignment operator
  */
  Clause& operator=(const Clause& rhs) = default;

  /**
  @brief default move assignment operator
  */
  Clause& operator=(Clause&& rhs) = default;

  std::vector<Literal> literals;

	// is this a learnt clause?
	bool learnt = false;
};

// constant:
// an undefined/empty clause id
const int CREF_UNDEF = -1; 

/**
 * @struct VarInfo
 * @brief stores a variable's reason clause
 * and its decision level
 */
struct VarInfo {
  VarInfo() = default;

  VarInfo(int cref, int lvl) :
    reason_cla(cref),
    decision_level(lvl)
  {
	}
  
	int reason_cla;
  int decision_level;
};


/**
 * @struct Watcher
 * @brief stores a clause watching a specified literal
 * and a blocker literal that gets affected by the 
 * specified literal (used in the lit-vec<Watcher> mapping 'watches')
 */
struct Watcher {
	Watcher() = default;

	Watcher(int cr, Literal p) :
		cref(cr),
		blocker(p)
	{
	}
	
	Watcher& operator=(const Watcher&) = default;
	Watcher(const Watcher&) = default;
	Watcher(Watcher&&) = default;

	// clause reference id
	int cref;
	Literal blocker;
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
  @param input_file the dimacs cnf file name
  */
  void read_dimacs(const std::string& input_file);
  
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

  void print_assigns();
  
	// TODO: this shouldn't be a public interface
  // but I need this to unit test literal op functionalities
  void assign(int v, bool val) {
    _assigns[v] = val ? Status::TRUE : Status::FALSE;
  }
	
	
	const Clause& clause(int i) const {
		return _clauses[i];
	}

  /**
   * @brief value
   * @in v the variable id
   * returns the evaluated value of a variable
   */
  Status value(int v) const;
  
  /**
   * @brief value
   * @in p the literal id
   * returns the evaluated value of a literal
   */
  Status value(const Literal& p) const;  

  bool unchecked_enqueue(const Literal &p, const int from_cla);
  
  /**
   * @brief enqueue
   * if value(p) is evaluated, check for conflict
   * else store this new fact, update assignment, trail, etc.
   */
  bool enqueue(const Literal& p, const int from_cla = CREF_UNDEF);

	void var_bump_activity(int v);
	void var_bump_activity(int v, int var_inc);
	void var_decay_activity();
	void cla_bump_activity(Clause& c);
	void cla_decay_activity();

	int level(int v) const;
	int reason(int v) const;

	/**
	 * @brief propagate
	 * carries out boolean constraint propagation (BCP)
	 * and propagates all enqueued facts
	 * if a conflict is encountered, return the conflict clause id
	 * else return an undefined clause id
	 */
	// TODO: define as public for now
	// for the purpose of functionality testing
	int propagate();

	/**
	 * @brief search
	 * main search loop that runs BCP
	 * and resolves conflict
	 */
	bool search(/* TODO: search parameters go here */);

	/**
	 * @brief analyze
	 * analyze a given conflict clause, and produce a reason clause
	 * and also a backtrack level the solver should revert its state to
	 * Pre-condition:
	 * + out_learnt is cleared
	 * + current decision level must be greater than root level
	 * Post-condition:
	 * + out_learnt[0] is the asserting literal at 'out_btlevel'
	 * + if out_learnt.size > 1, then out_learnt[1] has the greatest
	 *   decision level of the rest of the literals
	 */
	void analyze(int confl_cref, std::vector<Literal>& out_learnt, int& out_btlevel);


  void reset();
  void read_dimacs(std::istream&);


  bool transpile_task_to_z3(const std::string& task_file_name);
  bool transpile_task_to_dimacs(const std::string& task_file_name);


	// watches
	// watches[lit] maps to a list of watchers watching 'lit'
	std::vector<std::vector<Watcher>> watches;

	// statistic variables
	uint64_t propagations = 0;

	// user-configurable variables
	double var_inc;
	double cla_inc;
private:

  /**
  @brief utility method that reads in a parsed symbol, encode into a literal and store it
  @param[in] in the parsed symbol
  @param[out] lits the literal vector to store all the encoded literals (for each clause) 
  */
  void _read_clause(int symbol, std::vector<Literal>& lits);
  void _init();

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
	
	/**
	 * @brief attach clause
	 * initialize the watched literals for
	 * newly added clauses
	 */
	void _attach_clause(const int c_id);
	
	/**
	 * @brief detach clause
	 * inverse action of attach, remove the watchers
	 */
	void _detach_clause(const int c_id);


	/**
	 * @brief new decision level
	 * begin a new decision level
	 * increase the size of trail_lim by
	 * pushing the the accumulated length of previous trail
	 */
	void _new_decision_level();



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


	// qhead
	// an index to track
	// which literal we're propagating in the trail
	// NOTE: no more explicit propagation queue defined
	int _qhead;


	/**
	 * some temp data structures to prevent
	 * allocation overhead
	 */
	
	// seen 
	// a list which records whether a variable is examined
	// (may be used in multiple methods)
	// values are just 0 or 1
	std::vector<char> _seen;

  // output file stream to write to z3py
  std::ofstream _z3_ofs;
};



/**
 * inline method implementations 
 */
inline Status Solver::value(int v) const {
	return _assigns[v];
}

inline Status Solver::value(const Literal& p) const {
	if (_assigns[var(p)] == Status::UNDEFINED) {
		return Status::UNDEFINED;
	}
	else {
		return static_cast<int>(_assigns[var(p)]) ^ sign(p) ? 
			Status::TRUE : 
			Status::FALSE;
	}
}

inline bool Solver::unchecked_enqueue(const Literal &p, const int from_cla) {
	assert(value(p) == Status::UNDEFINED);

	// make the assignment, such that this literal
	// evaluates to true
	_assigns[var(p)] = static_cast<Status>(!sign(p)); 
	
	// store decision level and reason clause
	_var_info[var(p)] = VarInfo{from_cla, static_cast<int>(decision_level())};

	// push this literal into trail
	_trail.push_back(p);
	
	return true;
}

inline bool Solver::enqueue(const Literal& p, const int from_cla) {
	return value(p) != Status::UNDEFINED ? 
		value(p) != Status::FALSE : 
		unchecked_enqueue(p, from_cla); 
}


inline void Solver::var_bump_activity(int v) {
	var_bump_activity(v, var_inc);
}
inline void Solver::var_bump_activity(int v, int var_inc) {

}

inline void Solver::var_decay_activity() {

}

// TODO: implementation
inline void Solver::cla_bump_activity(Clause& c) {

} 

inline void Solver::cla_decay_activity() {

}


// variable information
inline int Solver::level(int v) const {
	return _var_info[v].decision_level;
}

inline int Solver::reason(int v) const {
	return _var_info[v].reason_cla;
}

inline void Solver::_new_decision_level() {
	_trail_lim.push_back(_trail.size());
}













}  // end of namespace --------------------------------------------------------





