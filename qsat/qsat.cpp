#include <iostream>
#include <cmath>
#include <fstream>
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
Solver::Solver() {
  _assignments.clear();
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

  while (true) {

    ifs >> buf;

    if (ifs.eof()) {
      break;
    }
    if (buf == "c") {
      std::getline(ifs, buf);
    }
    else if (buf == "p") {
      ifs >> buf >> _num_variables >> _num_clauses;
    }
    else {
      symbol = std::stoi(buf);
      while (symbol != 0) { 
        _read_clause(symbol, literals); 
        ifs >> symbol; 
      }
      add_clause(std::move(literals));
      literals.clear();
    }
  }
}

void Solver::_read_clause(int symbol, std::vector<Literal>& lits) { 
  int variable = (std::abs(symbol) - 1);
  lits.push_back((symbol > 0) ? Literal(variable, false) : Literal(variable, true));

}

// TODO: 
void Solver::add_clause(std::vector<Literal>&& lits) {
  _clauses.push_back(Clause(std::move(lits)));
}

void Solver::add_clause(const std::vector<Literal>& lits) {
  _clauses.push_back(Clause(lits));
}

bool Solver::_dpll(std::vector<Clause>& clauses, std::vector<int>& assignments) {
  // base case: if the entire cnf expression is empty 
  // (not necessarily empty from the beginning, we may remove clauses during the pro)
  if (clauses.size() == 0) {
    return true;
  }
  
  // base case: if the cnf expression contains an empty clause
  // TODO: Is there a more efficient way to do this?
  for (const Clause& c : clauses) {
    if (c.literals.empty()) {
      return false;
    }
  }

  // unit propagation
  _unit_propagate(clauses, assignments);

  // pick a literal (now it's always the first literal in the first clause)
  // can apply variable reordering later
  int new_lit_id = clauses[0].literals[0].id;
  int neg_new_lit_id = (new_lit_id % 2 == 0) ? new_lit_id + 1 : new_lit_id - 1;

  // recurse into 2 branches, one branch picks the chosen literal, 
  // the other branch picks the negated chosen literal 
  return _dpll(_determine_literal(clauses, assignments, new_lit_id), assignments) || 
         _dpll(_determine_literal(clauses, assignments, neg_new_lit_id), assignments); 

}
  

void Solver::_unit_propagate(std::vector<Clause>& clauses, std::vector<int>& assignments) { 
  size_t unit_clause_index;
  while (_has_unit_clause(clauses, unit_clause_index)) {
    int lit_id = clauses[unit_clause_index].literals[0].id; 
    
    _determine_literal(clauses, assignments, lit_id);
    
    // remove this clause
    clauses.erase(clauses.begin() + unit_clause_index);

  }

}

bool Solver::_has_unit_clause(std::vector<Clause>& clauses, size_t& unitClauseIndex) {
  for (size_t i = 0; i < clauses.size(); i++) {
    if (clauses[i].literals.size() == 1) {
      unitClauseIndex = i ;
      return true;
    }
  }
  return false;
}

// TODO: no need to return a reference if it is a private member
std::vector<Clause>& Solver::_determine_literal(std::vector<Clause>& clauses, std::vector<int>& assignments, int new_lit_id) {
  // assign value to this unassigned literal
  assignments.push_back(new_lit_id % 2 == 0 ? (new_lit_id / 2) + 1 : ((new_lit_id + 1) / 2) * (-1));
  int neg_new_lit_id = new_lit_id % 2 == 0 ? new_lit_id + 1 : new_lit_id - 1;
  
  // 2. remove the negated literal from all clauses that contains it 
  // (it evaluates to false, we still need to determine other literals in this clause)
  // 1. remove all clauses that contains this literal (it evaluates to true, don't care anymore)
  for (size_t i = 0; i < clauses.size(); i++) {
    
    // TODO: use auto
    auto neg_lit_itr = std::find_if(
      clauses[i].literals.begin(), clauses[i].literals.end(), 
      [&](const Literal& lit) {
        return lit.id == neg_new_lit_id;
      }
    );

    // found case 1
    if (neg_lit_itr != clauses[i].literals.end()) {
      clauses.erase(clauses.begin() + i);
    }
    
    auto lit_itr = std::find_if(
      clauses[i].literals.begin(), clauses[i].literals.end(), 
      [&](const Literal& lit) {
        return lit.id == new_lit_id;
      }
    ); 
   
    // found case 2
    if (lit_itr != clauses[i].literals.end()) {
      clauses[i].literals.erase(lit_itr);
    }
  }     

  return clauses;
}

void Solver::dump(std::ostream& os) const {
  os << "Dump Clauses:\n";
  // TODO
  for (const auto& clause : _clauses) {
    for (const auto& lit : clause.literals) {
      os << lit.id << " ";
    }
    os << "\n";
  }

  os << "Solution:\n";
  for (size_t i = 0; i < _assignments.size(); i++) {
    os << _assignments[i] << " ";
  }
  os << "\n";
}

bool Solver::solve() {
  if (_dpll(_clauses, _assignments)) {
    std::cout << "SAT\n";
    return true;
  }
  else {
    std::cout << "UNSAT\n";
    return false;
  }
}

const std::vector<Clause>& Solver::clauses() const {
  return _clauses;
}

}  // end of namespace qsat ---------------------------------------------------









