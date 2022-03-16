#ifndef QSAT_HPP
#define QSAT_HPP
#include <qsat.hpp>


// Declaration: Literal Class constructor
qsat::Literal::Literal(Variable var, bool sign) :
  literalId(var + var + (int)sign)
{

}

// Declaration: Clause Class constructor
qsat::Clause::Clause(std::vector<Literal>& lits) :
  literals(lits) 
{

}

#endif


