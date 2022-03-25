// TODO: for each hpp you should always start with pragma once
//       to make the compiler include/compile it only once
#pragma once
#include <vector>
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
  /**
  @brief constructs a clause with given literals
  */
  Clause(const std::vector<Literal>& lits);
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
  Solver() = default;

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
  bool _add_clause(std::vector<Literal>& lits);

  std::vector<Clause> _clauses; 
};



}  // end of namespace --------------------------------------------------------





