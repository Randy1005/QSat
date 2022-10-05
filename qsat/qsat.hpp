#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <stack>
#include <algorithm>
#include <string>
#include <chrono>
#include <unordered_map>
#include <filesystem>
// TODO: this is probably not the right way to include
// but cmake is acting weird with header-only libraries
#include "../mtl/heap.hpp"
#include "intel_task_grammar.hpp"


namespace qsat {

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

/**
 * utility inline methods
 * var(lit), ~lit, signed(lit), etc.
 */
inline Literal operator ~(const Literal& p) {
  Literal q(0);
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
  Clause(const std::vector<Literal>& lits);

  Clause(std::vector<Literal>&& lits);

  /**
  @brief default copy assignment operator
  */
  Clause& operator=(const Clause& rhs) = delete;

  /**
  @brief default move assignment operator
  */
  Clause& operator=(Clause&& rhs) = default;

  std::vector<Literal> literals;
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

  size_t num_clauses() const { 
    return _clauses.size(); 
  }
  size_t num_variables() const {
    return _assigns.size(); 
  }
  size_t n_assigns() const { 
    return _trail.size(); 
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
  
  /**
   * @brief enqueue
   * if value(p) is evaluated, check for conflict
   * else store this new fact, update assignment, trail, etc.
   */
  inline bool enqueue(Literal& p, Clause& from) {

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

  // output file stream to write to z3py
  std::ofstream _z3_ofs;
};


}  // end of namespace --------------------------------------------------------





