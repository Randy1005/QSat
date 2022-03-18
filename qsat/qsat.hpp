// TODO: for each hpp you should always start with pragma once
//       to make the compiler include/compile it only once
#pragma once

#include <vector>
#include <string>

namespace qsat {

// TODO: typedef is C-style... use "using" instead for c++
typedef int Variable;
//using VariableType = int; (Type means it is an alias)

// class: Literal
// TODO: if everything is public -> just use struct
//       can we do structure Literal...?
struct Literal {

  Literal(Variable var, bool sign = false);
    
  // TODO: id
  // variable naming rule is the same as function => int id;
  int literalId;
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
    void ParseDimacs(const std::string& inputFileName);

    // TODO:
    // use dump
    void Dump(std::ostream& os) const;

  private:

    // TODO: private should alwasy prefix at '_'
    // _read_clause, _add_clause, _clauses
    void ReadClause(std::istringstream& in, std::vector<Literal>& lits);
    bool AddClause(std::vector<Literal>& lits);
    std::vector<Clause> clauses; 
};



}  // end of namespace --------------------------------------------------------





