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

A leteral is created from a given integer variable based on the following
encoding method:

l  = 2*v
l' = 2*v+1
*/
struct Literal {

  Literal(VariableType var, bool isSigned = false);

  // TODO: id
  int id;
};

// TODO: document it
// struct: Clause
struct Clause {
  Clause(const std::vector<Literal>& lits);

  std::vector<Literal> literals;
};

class Solver {
public: 

  // TODO: may change to include something in the future
  Solver() = default;

  void read_dimacs(const std::string& inputFileName);
  void dump(std::ostream& os) const;

  // TODO: implement a solve function
 
   // TODO: finish below
  const std::vector<Clause>& clauses() const; 

private:

  void _read_clause(std::istringstream& in, std::vector<Literal>& lits);
  bool _add_clause(std::vector<Literal>& lits);

  std::vector<Clause> _clauses; 
};



}  // end of namespace --------------------------------------------------------





