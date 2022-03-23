// TODO: for each hpp you should always start with pragma once
//       to make the compiler include/compile it only once
#pragma once

#include <vector>
#include <string>

namespace qsat {

// TODO: typedef is C-style... use "using" instead for c++
using VariableType = int; 

// class: Literal
// TODO: if everything is public -> just use struct
//       can we do structure Literal...?
struct Literal {

  Literal(VariableType var, bool isSigned = false);

  // TODO: id
  // variable naming rule is the same as function => int id;
  int id;
};



// TODO: struct is enough I think
struct Clause {
  // TODO: const std::vector<Literal>& is better
  Clause(const std::vector<Literal>& lits);

  std::vector<Literal> literals;
};

class Solver {
public: 
  // TODO: if you don't have anything (empty) => use default
  Solver() = default;

  // TODO: change the naming rules
  // use read_dimacs
  void read_dimacs(const std::string& inputFileName);

  // TODO:
  // use dump
  void dump(std::ostream& os) const;

private:

  // TODO: private should alwasy prefix at '_'
  // _read_clause, _add_clause, _clauses
  void _read_clause(std::istringstream& in, std::vector<Literal>& lits);
  bool _add_clause(std::vector<Literal>& lits);
  std::vector<Clause> _clauses; 
};



}  // end of namespace --------------------------------------------------------





