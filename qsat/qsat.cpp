#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include "qsat.hpp"

namespace qsat {

Literal::Literal(VariableType var, bool isSigned) :
  id(var + var + (int)isSigned)
{

}

Clause::Clause(const std::vector<Literal>& lits) :
  literals(lits) 
{

}

Clause::Clause(std::vector<Literal>&& lits) :
  literals(std::move(lits)) 
{

}

// we may implement something in the constructor in the future, we don't know yet
//Solver::Solver() {
//
//}

bool Solver::solve() {

}

void Solver::read_dimacs(const std::string& inputFileName) {
  std::ifstream ifs;
  ifs.open(inputFileName);

  if(!ifs) {
    throw std::runtime_error("failed to open a file");
  }

  int symbol = -1;
  std::string buf;
  std::vector<Literal> literals;
  size_t num_variables = 0;
  size_t num_clauses = 0;    

  while (true) {

    ifs >> buf;

    if (ifs.eof()) {
      break;
    }

    if (buf == "c") {
      std::getline(ifs, buf);
    }
    else if (buf == "p") {
      ifs >> buf >> num_variables >> num_clauses;
    }
    else {
      symbol = std::stoi(buf);
      while (symbol != 0) { _read_clause(symbol, literals); ifs >> symbol; }
      // TODO: should std::move(literals)
      _add_clause(literals);
      literals.clear();
    }
  }
}

void Solver::_read_clause(int symbol, std::vector<Literal>& lits) { 
  int variable = (std::abs(symbol) - 1);
  lits.push_back((symbol > 0) ? Literal(variable, false) : Literal(variable, true));
}


bool Solver::_add_clause(std::vector<Literal>& lits) {
  // TODO: figure out how to optimize this 
  // 1. explain to me how many copies that happened over here
  // 2. understand the move semantics in C++11
  // 3. use move semantics to optimize the code
  _clauses.push_back(Clause(lits));
  return true;
}

void Solver::dump(std::ostream& os) const {
  os << "Dump Clauses:\n";
  for (const auto& clause : _clauses) {
    for (const auto& lit : clause.literals) {
      os << lit.id << " ";
    }
    os << '\n';
  }
}

const std::vector<Clause>& Solver::clauses() const {
  return _clauses;
}

}  // end of namespace qsat ---------------------------------------------------









