#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <string>
#include <filesystem>
#include <random>
#include <cmath>
#include "heap.hpp"
#include "taskflow/taskflow.hpp"
#include "taskflow/sycl/syclflow.hpp"
#include <breakid.hpp>
// #include "intel_task_grammar.hpp"

namespace qsat {

struct Clause;
struct Literal;
struct VarInfo;
struct ClauseInfo;

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

	bool operator != (const Watcher& rhs) {
		return cref != rhs.cref;
	}

	bool operator == (const Watcher& rhs) {
		return cref == rhs.cref;
	}
		
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
  @brief reads in dimacs cnf file, and store the literals and clauses
  in the BreakID data structures
  @param input_file the dimacs cnf file name
  */
  void read_dimacs_bid(const std::string& input_file);

  /**
  @brief build constraint graph using the bliss library
  */
  void build_graph();

  /**
  @brief add symmetry breaking clauses to the original CNF
  */
  void add_symm_brk_cls();

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
  Status solve();

  /**
  @brief solve with a set of assumptions
  assumptions are a set of literals that the user
  forces to be evaluated to true
  */
  Status solve(const std::vector<Literal>& assumptions);

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
		// NOTE: _clauses.size is actually orig_clauses + learnt_clauses
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



  void dump_assigns(std::ostream& os) const;
  
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

	void remove_clause(const int cref);

	void var_bump_activity(int v);
	void var_bump_activity(int v, double inc);
	void var_decay_activity();
	void cla_bump_activity(Clause& c);
	void cla_decay_activity();
	int level(int v) const;
	int reason(int v) const;
	
	/**
	 * @ is removed
	 * returns true if a clause is marked to be 
	 * deleted
	 */
	bool is_removed(int cref) const;
	
	/**
	 * @ watcher deleted
	 * returns true if the cref in this watcher has been
	 * marked to delete
	 */
	bool watcher_deleted(Watcher& w) const;


	/**
	 * @brief locked
	 * a clause is locked when it's a reason of a current assignment
	 * do not remove it during database reduction
	 */
	bool locked(const int cref) const;

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
	Status search(int nof_conflicts);

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

	/**
	 * @brief reduce_db
	 * remove half of the learnt clauses, exclude the ones locked by 
	 * current assignments. Also don't remove binary clauses.
	 */
	void reduce_db();

  void reset();
  void read_dimacs(std::istream&);

	/**
	 * @ clean all watches
	 * a wrapper utility that invokes _clean_watch
	 * on all watches
	 */
	void clean_all_watches();

	/**
	 * @ relocate all
	 * relocate all crefs that's referenced in
	 * watches, _var_info, etc.
	 */
	void relocate_all();


  /**
   * @brief sycl_check_subsumptions
   * if a clause A is a subset of clause B
   * clause B can be removed, because A is more
   * restrictive than B regarding satisfiability
   */
  void sycl_check_subsumptions();

  void init_device_db();

	/**
	 * intel task file transpiling
	 */
  // bool transpile_task_to_z3(const std::string& task_file_name);
  // bool transpile_task_to_dimacs(const std::string& task_file_name);


	// watches
	// watches[lit] maps to a list of watchers watching 'lit'
	std::vector<std::vector<Watcher>> watches;


	// statistic variables
	uint64_t propagations = 0;
	uint64_t conflicts = 0;
	uint64_t decisions = 0;
	uint64_t num_learnts = 0;
	uint64_t starts = 0;
	int num_orig_clauses;
  int num_orig_vars;

	// user-configurable variables
	double var_inc;
	double cla_inc;
	double var_decay;
	double cla_decay;
  int bid_verbosity;
  int64_t bid_steps_lim;

	int restart_first; // the initial restart limit
	double restart_inc; // the factor which restart limit is multiplied at each restart

	bool enable_reduce_db;
	bool enable_rnd_pol;
	bool enable_luby;

	double learnt_size_factor;
	double max_learnts;
	
	// configurable phase-saving [0: no, 1: limited, 2: full]
	int phase_saving;


  // the BreakID instance
  // for symmetry detection and breaking
  BID::BreakID breakid;
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
	void _attach_clause(const int cref);
	
	/**
	 * @brief detach clause
	 * inverse action of attach, remove the watchers
	 */
	void _detach_clause(const int cref);


	/**
	 * @brief new decision level
	 * begin a new decision level
	 * increase the size of trail_lim by
	 * pushing the the accumulated length of previous trail
	 */
	void _new_decision_level();

	/**
	 * @brief pick branch literal
	 * based on variable activities (order_heap)
	 * prioritize which literal to propagate first
	 */
	Literal _pick_branch_lit();


	/**
	 * @ cancel until
	 * revert the state of the solver to a certain level
	 * (keeping all assignments at 'level' but not beyond)
	 */
	void _cancel_until(int level);

	/**
	 * @ smudge watch
	 * set the dirty bit of the specified watches[p]
	 * notifying the solver that some of the watchers in
	 * watches[p] has a cref that's marked as removed
	 */
	void _smudge_watch(int p);

	/**
	 * @ clean watch
	 * from specified watches[p], iterate through the watchers,
	 * if a watcher has a cref that's marked as removed, remove that
	 * watcher
	 */
	void _clean_watch(int p);

	/**
	 * @ luby
	 * calculates the finite subsequence
	 * of the luby sequence
	 *
	 * 0 : 1
	 * 1 : 1 1 2
	 * 2 : 1 1 2 1 1 2 4
	 * 3 : 1 1 2 1 1 2 4 1 1 2 1 1 2 4 8
	 */
	double _luby(double y, int x);


  /**
   * @luby_mis
   * maximum independent set algorithm
   */
  void _luby_mis();

  std::vector<Clause> _clauses; 

  std::vector<std::vector<BID::BLit>> _bid_clauses;

	// learnt clauses
	// stores the index to the learnt clauses
	// we still store the actual clause objects in _clauses
	std::vector<int> _learnts;


  // assignment vector 
  std::vector<Status> _assigns;

  // user assumptions
  std::vector<Literal> _assumptions;

	// solution model (if SAT)
	std::vector<Status> _model;

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

	// watches dirty bits
	std::vector<bool> _watches_dirty;

	// watches dirties
	// stores the dirty indices
	std::vector<int> _watches_dirties;


	// solver search status
	// a status variable to record result of solving: SAT/UNSAT/UNDEF
	Status _solver_search_status = Status::UNDEFINED;

	// random device to seed the random number generator
	std::random_device _rd;	
	
	// mersenne twister random number generator
	// WARNING: this rng costs a lot of memory according to some developers
	std::mt19937 _mtrng;

	// distributions
	std::uniform_int_distribution<int> _uni_int_dist;
	std::uniform_real_distribution<double> _uni_real_dist;



	/**
	 * some temp data structures to prevent
	 * allocation overhead
	 */
	
	// seen 
	// a list which records whether a variable is examined
	// (may be used in multiple methods)
	// values are just 0 or 1
	std::vector<bool> _seen;

  // output file stream to write to z3py
  std::ofstream _z3_ofs;

  // sycl task queue
  sycl::queue _queue;

  // task flow object
  tf::Taskflow _tf;
  
  // taskflow executor
  tf::Executor _executor;

  // @brief shared cnf
  // literals stored in shared space
  // between host and device
  uint32_t* _sh_cnf; 
    
  // @brief shared indices
  // indices to record clause c starts
  // on nth literal
  //
  // indices[c] -> n
  uint32_t* _sh_idxs;

 
   
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

inline void Solver::var_bump_activity(int v, double inc) {
	// rescale if var activities exceed a certain value
	if ((_activities[v] += inc) >= 1e20) {
		for (size_t i = 0; i < num_variables(); i++) {
			_activities[i] *= 1e-20;
		}
		var_inc *= 1e-20;
	}

	// update variable order heap
	// (bubble up this variable in the heap)
	if (_order_heap.in_heap(v)) {
		_order_heap.decrease(v);
	}
}

inline void Solver::var_decay_activity() {
	var_inc *= (1 / var_decay); 
}

inline void Solver::cla_bump_activity(Clause& c) {
	// rescale clause activities if exceed a
	// certain large value
	if ((c.activity += cla_inc) >= 1e20) {
		for (const auto& l : _learnts) {
			_clauses[l].activity *= 1e-20;
		}
		cla_inc *= 1e-20;
	}
} 

inline void Solver::cla_decay_activity() {
	cla_inc *= (1 / cla_decay);
}

inline int Solver::level(int v) const {
	return _var_info[v].decision_level;
}

inline int Solver::reason(int v) const {
	return _var_info[v].reason_cla;
}

inline void Solver::_new_decision_level() {
	_trail_lim.push_back(_trail.size());
}

inline bool Solver::locked(const int cref) const {
	const Clause& c = _clauses[cref];

	return value(c.literals[0]) == Status::TRUE &&
		reason(var(c.literals[0])) != CREF_UNDEF &&
		reason(var(c.literals[0])) == cref;
}

inline bool Solver::is_removed(int cref) const {
	return _clauses[cref].mark == 1;
}

inline bool Solver::watcher_deleted(Watcher& w) const {
	return _clauses[w.cref].mark == 1;
}

}  // end of namespace --------------------------------------------------------





