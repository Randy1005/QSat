#ifndef QSAT_HPP
#define QSAT_HPP

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include "qsat.hpp"


qsat::Literal::Literal(VariableType var, bool isSigned) :
  id(var + var + (int)isSigned)
{

}

qsat::Clause::Clause(const std::vector<Literal>& lits) :
  literals(lits) 
{

}

// we may implement something in the constructor in the future, we don't know yet
//qsat::Solver::Solver() {
//
//}

void qsat::Solver::read_dimacs(const std::string& inputFileName) {
  std::ifstream ifs;
  ifs.exceptions(std::ifstream::badbit);

  // TODO: doe not throw file invalid exception
  try {
    ifs.open(inputFileName);

    std::string lineBuffer;
    std::vector<Literal> literals;
    // TODO: variable naming 
    // counter uses size_t
    size_t numVariables = 0;
    size_t  numClauses = 0;    

    while (std::getline(ifs, lineBuffer)) {
    
      std::istringstream iss(lineBuffer);

      if (lineBuffer[0] == 'c') continue;
      else if (lineBuffer[0] == 'p') {
        std::string dummyStr;
        iss >> dummyStr >> dummyStr >> numVariables >> numClauses;
      }
      else {
        _read_clause(iss, literals);
        _add_clause(literals);
      }

    }

  }
  catch (const std::ifstream::failure& fail) {
    throw std::runtime_error(fail.what());
  }
  
}

void qsat::Solver::_read_clause(std::istringstream& iss, std::vector<Literal>& lits) { 
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
bool qsat::Solver::_add_clause(std::vector<Literal>& lits) {
  _clauses.push_back(Clause(lits));
  return true;
}


/**
 * @brief dumps solver data structures through std::ostream (currently dumps stored literals)
 *
 * @param os the output stream target to dump to 
 */
void qsat::Solver::dump(std::ostream& os) const {
  os << "Dump Clauses:" << std::endl;
  for (Clause clause : _clauses) {
    for (Literal lit : clause.literals) {
      os << lit.id << " ";
    }
    os << std::endl;
  }
}


#endif


