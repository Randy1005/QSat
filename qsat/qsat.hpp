// TODO: for each hpp you should always start with pragma once
//       to make the compiler include/compile it only once
#pragma once
#include <vector>
#include <algorithm>
#include <string>

namespace qsat {

using VariableType = int; 

/**
@struct Literal
@brief struct to create a literal

A literal is created from a given integer variable based on the following
encoding method:

l  = 2*v
l' = 2*v+1
*/
struct Literal {
  /**
  @brief constructs a literal with a given variable
  */
  Literal(VariableType var, bool isSigned = false);
  int id;
};

/**
@struct Clause
@brief struct to create a clause
*/
struct Clause {
  
  Clause() = default;
  /**
  @brief constructs a clause with given literals
  */
  Clause(const std::vector<Literal>& lits);
 

  /**
  @brief move constructor
  */
  Clause(const Clause&& rhs);

  /**
  @brief copy assignment operator
  */
  Clause& operator=(const Clause& rhs);

  std::vector<Literal> literals;
};

/**
@class Solver
@brief a class to create a SAT solver object
*/
class Solver {
public: 
  // TODO: may change to include something in the future
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

  // TODO: implement a solve function
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

private:
  /**
  @brief utility method that reads in a parsed symbol, encode into a literal and store it
  @param[in] in the parsed symbol
  @param[out] lits the literal vector to store all the encoded literals (for each clause) 
  */
  void _read_clause(int symbol, std::vector<Literal>& lits);
  
  /**
  @brief pushes a vector of literals into the clauses vector
  @param lits the vector of literals to store as a clause
  @returns true if the clause was pushed successfully, otherwise false
  */
  bool _add_clause(const std::vector<Literal>& lits);
  
  bool _dpll(std::vector<Clause>& clauses, std::vector<int>& assignments);

  void _unit_propagate(std::vector<Clause>& clauses, std::vector<int>& assignments);

  bool _has_unit_clause(std::vector<Clause>& clauses, size_t& unitClauseIndex);

  std::vector<Clause>& _determine_literal(std::vector<Clause>& clauses, std::vector<int>& assignments, int new_lit_id);

  std::vector<Clause> _clauses; 
  std::vector<int> _assignments;
  size_t _num_variables;
  size_t _num_clauses;
};



}  // end of namespace --------------------------------------------------------





