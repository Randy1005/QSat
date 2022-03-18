#ifndef QSAT_HPP
#define QSAT_HPP

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include "qsat.hpp"


// Literal Class constructor
qsat::Literal::Literal(Variable var, bool sign) :
  literalId(var + var + (int)sign)
{

}

// Clause Class constructor
qsat::Clause::Clause(const std::vector<Literal>& lits) :
  literals(lits) 
{

}

//// Solver Class constructor
//qsat::Solver::Solver() {
//
//}

void qsat::Solver::ParseDimacs(const std::string& inputFileName) {


  // TODO: create an input file stream
  std::ifstream ifs(inputFileName, std::ifstream::in);

  if(!ifs) {
    std::cerr << "file " << inputFileName << " is invalid";
    std::exit(EXIT_FAILURE);

    // TODO: throw an std::runtime_error 
  }

  std::string lineBuffer;


  std::vector<Literal> literals;
  // TODO: variable naming 
  // counter uses size_t
  int numVariables = 0;
  int numClauses = 0;    

  while (std::getline(ifs, lineBuffer)) {
    
    std::istringstream iss(lineBuffer);

    if (lineBuffer[0] == 'c') continue;
    else if (lineBuffer[0] == 'p') {
      // TODO: there's gotta be a better way to do it
      std::string dummyStr;
      is >> dummyStr >> dummyStr >> numVariables >> numClauses;
      // TODO: I don't think this is invalid ...
      if (numVariables == 0 || numClauses == 0) {
        std::cerr << "Parsing error: Invalid number of variables or clauses" << std::endl;
      }
    }
    else {
      ReadClause(iss, literals);
      AddClause(literals);
    }

  } 


}

void qsat::Solver::ReadClause(std::istringstream& iss, std::vector<Literal>& lits) { 
  int parsedLiteral, variable;

  lits.clear();
  for (;;) {
    iss >> parsedLiteral;
    if (parsedLiteral == 0) break;
    variable = abs(parsedLiteral) - 1;
    lits.push_back((parsedLiteral > 0) ? Literal(variable, true) : Literal(variable, false));
  }
}

/**
 * Should do some preprocessing at this point
 * but just pushes a clause into the solver for now
 * and always successful
 *
 * @param
 * @returns
 */
bool qsat::Solver::AddClause(std::vector<Literal>& lits) {
  clauses.push_back(Clause(lits));
  return true;
}


/**
 * TODO: Write a summary for Dump, w/ example usages
 */
void qsat::Solver::Dump(std::ostream& os) const {
  os << "Dump Clauses:" << std::endl;
  for (Clause clause : clauses) {
    for (Literal lit : clause.literals) {
      os << lit.literalId << " ";
    }
    os << std::endl;
  }
}




#endif


