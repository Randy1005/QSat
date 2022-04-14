#pragma once
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <chrono>
#include <unordered_map>

namespace qsat {

enum class Status {
  FALSE = 0,
  TRUE  = 1,
  UNDEFINED
};

struct ClauseSatisfiability {
  int clause_id;
  bool is_modified;
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
class Literal {
  
  friend struct Clause;
  friend class Solver;

  public:
    // TODO: friend class Clause
    /**
    @brief constructs a literal with a given variable
    */
    Literal(int var);

  private:
    size_t _id;

};

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

  size_t num_clauses() const   { return _clauses.size(); }
  size_t num_variables() const { return _assignments.size(); }

  // v is positive => id = 2|v| - 2 => assignment id/2
  // v is negative => id = 2|v| - 1 => assignment id/2
  Status assignment_of(int variable) const {
    return _assignments[variable > 0 ? variable - 1 : -variable - 1];
  }

  void reset();
  void read_dimacs(std::istream&);

private:

  /**
  @brief utility method that reads in a parsed symbol, encode into a literal and store it
  @param[in] in the parsed symbol
  @param[out] lits the literal vector to store all the encoded literals (for each clause) 
  */
  void _read_clause(int symbol, std::vector<Literal>& lits);
  
  /*
  bool _dpll(std::vector<Clause>& clauses);
  void _unit_propagate(std::vector<Clause>& clauses);
  bool _has_unit_clause(std::vector<Clause>& clauses, size_t& unitClauseIndex);
  void _determine_literal(std::vector<Clause>& clauses, int new_lit_id);
  */

  bool _backtrack(int decision_depth, std::vector<Status>& assignments);
  bool _evaluate_clauses(const std::vector<Status>& assignments) ;
  size_t _propagate_constraint(int decision_depth, const std::vector<Status>& assignments);
  void _init();
  void _print_assignments();


  

  std::vector<Clause> _clauses; 
  std::vector<Status> _assignments;

  // mapping: assignments (variable) -> clauses' id
  std::unordered_map<int, std::vector<ClauseSatisfiability>> _var_to_clauses;

  // counter for currently satisfied clauses
  size_t _num_sat_clauses = 0;

  // lookup for the status of each clause
  std::vector<Status> _clauses_status;
};



}  // end of namespace --------------------------------------------------------





