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

// we may implement something in the constructor in the future, we don't know yet
//Solver::Solver() {
//
//}

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

  // TODO: finish the code below
  while(true) {
    ifs >> buf;

    if(ifs.eof()) {
      break;
    }

    if(buf == "c") {
      std::getline(ifs, buf);
      std::cout << "comment: " << buf << '\n';
    }
    else if(buf == "p") {
      ifs >> buf >> num_variables >> num_clauses;
      std::cout << "p " << num_variables << ' ' << num_clauses << '\n';
    }
    else {
      symbol = std::stoi(buf);
      std::cout << symbol << ' ';
      while(symbol != 0) {
        ifs >> symbol;
        std::cout << symbol << ' ';
      }
      std::cout << '\n';
    }
  }

  //while (std::getline(ifs, buf)) {
  //
  //  std::istringstream iss(buf);

  //  if (buf[0] == 'c') continue;
  //  else if (buf[0] == 'p') {
  //    std::string dummyStr;
  //    iss >> dummyStr >> dummyStr >> num_variables >> num_clauses;
  //  }
  //  else {
  //    _read_clause(iss, literals);
  //    _add_clause(literals);
  //  }

  //}
  
}

void Solver::_read_clause(std::istringstream& iss, std::vector<Literal>& lits) { 
  int parsedLiteral, variable;

  lits.clear();
  for (;;) {
    iss >> parsedLiteral;
    if (parsedLiteral == 0) break;
    variable = abs(parsedLiteral) - 1;
    lits.push_back((parsedLiteral > 0) ? Literal(variable, false) : Literal(variable, true));
  }
}

/**
 * @brief should do some preprocessing at this point
 * but just pushes a clause into the solver for now
 * and always successful
 *
 * @param lits the literals to form a clause and add to the solver
 * @returns true if successfully added a clause to the solver, otherwise false
 */
bool Solver::_add_clause(std::vector<Literal>& lits) {
  _clauses.push_back(Clause(lits));
  return true;
}


/**
 * @brief dumps solver data structures through std::ostream (currently dumps stored literals)
 *
 * @param os the output stream target to dump to 
 */
void Solver::dump(std::ostream& os) const {
  os << "Dump Clauses:" << std::endl;
  for (Clause clause : _clauses) {
    for (Literal lit : clause.literals) {
      os << lit.id << " ";
    }
    os << std::endl;
  }
}

}  // end of namespace qsat ---------------------------------------------------


