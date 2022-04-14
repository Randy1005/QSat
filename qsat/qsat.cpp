#include <iostream>
#include <cmath>
#include <fstream>
#include "qsat.hpp"

namespace qsat {

Literal::Literal(int var) {
  if (var == 0) {
    throw std::runtime_error("variable cannot be zero");
  }
 _id = (var > 0) ? 2 * var - 2 : 2 * -var - 1;
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

}

void Solver::read_dimacs(const std::string& inputFileName) {
  std::ifstream ifs;
  ifs.open(inputFileName);

  if (!ifs) {
    throw std::runtime_error("failed to open a file");
  }

  int variable = -1;
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
      ifs >> buf >> buf >> buf;
    }
    else {
      variable = std::stoi(buf);
      while (variable != 0) { 
        _read_clause(variable, literals); 
        ifs >> variable; 
      }
      add_clause(std::move(literals));
      literals.clear();
    }
  }
}

void Solver::_read_clause(int variable, std::vector<Literal>& lits) { 
  lits.push_back(Literal(variable));
}

void Solver::add_clause(std::vector<Literal>&& lits) {
  
  // resize the assignment vector to the current largest variable
  size_t max = 0;

  for (const auto& l : lits) {
    max = std::max(max, l._id / 2);
  }

  for (const auto& l : lits) {
    // TODO: does this work?
    _var_to_clauses[l._id / 2].push_back(_clauses.size());
  }


  if (max >= _assignments.size()) {
    _assignments.resize(max + 1);
  }

  _clauses.push_back(Clause(std::move(lits)));

}

void Solver::add_clause(const std::vector<Literal>& lits) {
 
  // resize the assignment vector to the current largest variable
  size_t max = 0;

  for (const auto& l : lits) {
    max = std::max(max, l._id / 2);
  }

  for (const auto& l : lits) {
    // TODO: does this work?
    _var_to_clauses[l._id / 2].push_back(_clauses.size());
  }

  if (max >= _assignments.size()) {
    _assignments.resize(max + 1);
  }

  _clauses.push_back(Clause(lits));
  
}

bool Solver::_backtrack(int decision_depth, std::vector<Status>& assignments) {
  // base case: we exceeded the maximum decision depth
  // and still didn't find satisfiable assignments
  if (decision_depth >= num_variables()) {
    // std::cout << "reached max depth\n";
    return false;
  }

  for (int val = 0; val <= 1; val++) {
    assignments[decision_depth] = static_cast<Status>(val);
    
    // _print_assignments();    
    // if all of the clauses evaluates to true, then we have a solution
    if (_evaluate_clauses(assignments)) {
      return true;
    }
    
    if (_backtrack(decision_depth + 1, assignments)) {
      return true;
    }

    // if backtrack returns failure, clear out the previous assignment
    // 0 -> assign false, 1 -> assign true, 2 -> unassigned
    assignments[decision_depth] = Status::UNDEFINED;

  }

  // searched the whole tree, and didn't find a solution
  // std::cout << "searched the whole tree, returning\n"; 
  return false;
}

/**
@brief this method checks if all clauses evaluate to true, if so return true
       if any one of the clauses evaluates to false, then return false
*/
// ?? How do we add in constraint propagation?
// It needs to maintain the state of clauses before propagation
// so we can reset if we made a wrong decision
bool Solver::_evaluate_clauses(const std::vector<Status>& assignments) {
  for (const auto& c : _clauses) {

    bool clause_is_sat = false;
    for (const auto& lit : c.literals) {
      // assignment[lit / 2] to get the corresponding variable's assignment
      // and xor with the rightmost bit of lit (lit & 1) 
      // (equals to checking if the lit is even)
      if (assignments[lit._id / 2] != Status::UNDEFINED && 
          static_cast<int>(assignments[lit._id / 2]) ^ (lit._id & 1)) 
      {
        clause_is_sat = true;
        break;
      }
    }

    if (!clause_is_sat) {
      return false;
    }
  }

  return true;
}


void Solver::_print_assignments() {
  for (size_t i = 0; i < _assignments.size(); i++) {
    std::cout << static_cast<int>(_assignments[i]) << " ";
  }
  std::cout << "\n";
}


/*
bool Solver::_dpll(std::vector<Clause>& clauses) {
  // base case: if the entire cnf expression is empty 
  // not necessarily empty from the beginning
  // we may remove clauses during the process
  if (clauses.size() == 0) {
    return true;
  }
  
  // base case: if the cnf expression contains an empty clause
  // TODO: Is there a more efficient way to do this?
  for (const auto& c : clauses) {
    if (c.literals.empty()) {
      return false;
    }
  }

  // unit propagation
  _unit_propagate(clauses);

  // pick a literal (now it's always the first literal in the first clause)
  // can apply variable reordering later
  int new_lit_id = clauses[0].literals[0].id;
  int neg_new_lit_id = (new_lit_id % 2 == 0) ? new_lit_id + 1 : new_lit_id - 1;
 
  // I think I have to make a copy here? probably a design flaw
  // in order to recurse with 2 different clauses
  std::vector<Clause> c_assigned_neg_lit = clauses;

  _determine_literal(clauses, new_lit_id);
  _determine_literal(c_assigned_neg_lit, neg_new_lit_id);

  // recurse into 2 branches, one branch picks the chosen literal, 
  // the other branch picks the negated chosen literal 
  return _dpll(clauses) || _dpll(c_assigned_neg_lit);  
}
  

void Solver::_unit_propagate(std::vector<Clause>& clauses) { 
  size_t unit_clause_index;
  while (_has_unit_clause(clauses, unit_clause_index)) {
    int lit_id = clauses[unit_clause_index].literals[0].id;
    
    // remove this clause
    clauses.erase(clauses.begin() + unit_clause_index);
    _assignments.push_back(lit_id % 2 == 0 ? (lit_id / 2) + 1 : ((lit_id + 1) / 2) * (-1));
    _determine_literal(clauses, lit_id);
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

void Solver::_determine_literal(std::vector<Clause>& clauses, int new_lit_id) {
  int neg_new_lit_id = new_lit_id % 2 == 0 ? new_lit_id + 1 : new_lit_id - 1;
  
  // 1. remove all clauses that contains this literal 
  // (it evaluates to true, don't care anymore)
  // 2. remove the negated literal from all clauses that contains it 
  // (it evaluates to false, we still need to determine other literals in this clause)
  for (size_t i = 0; i < clauses.size(); i++) {
    auto lit_itr = std::find_if(
      clauses[i].literals.begin(), clauses[i].literals.end(), 
      [&](const Literal& lit) {
        return lit.id == new_lit_id;
      }
    );

    auto neg_lit_itr = std::find_if(
      clauses[i].literals.begin(), clauses[i].literals.end(), 
      [&](const Literal& lit) {
        return lit.id == neg_new_lit_id;
      }
    );

    // found case 1
    if (lit_itr != clauses[i].literals.end()) {
      clauses.erase(clauses.begin() + i);
    }
    // found case 2
    else if (neg_lit_itr != clauses[i].literals.end()) {
      clauses[i].literals.erase(neg_lit_itr);
    }
  }     
}
*/

void Solver::dump(std::ostream& os) const {

  os << num_variables() << " variables\n";
  os << num_clauses() << " clauses\n";

  os << "Assignments:\n";
  for (size_t i = 0; i < _assignments.size(); i++) {
    os << static_cast<int>(_assignments[i]) << " ";
  }


  os << "\nVar to Clause Mapping:\n";
  for (const auto& [key, value] : _var_to_clauses) {
    os << "Var: " << key << " -> Clauses: ";
    for (const auto& v : value) {
      os << v << " ";
    }
    os << "\n";
  }

  os << "\n";
}

void Solver::_init() {
  // initialize assignments
  _assignments.resize(num_variables());
  for (size_t i = 0; i < num_variables(); i++) {
    _assignments[i] = Status::UNDEFINED; // unassigned state
  }

  // initialize clauses status
  _clauses_status.resize(num_clauses());
  for (size_t i = 0; i < num_clauses(); i++) {
    _clauses_status[i] = Status::UNDEFINED;
  }

}

void Solver::reset() {
  _assignments.clear();
  _clauses.clear();
  _clauses_status.clear();
  _var_to_clauses.clear();
  _num_sat_clauses = 0;
}

bool Solver::solve() {
  _init();
  int decision_depth = 0;
  return _backtrack(decision_depth, _assignments);
}

const std::vector<Clause>& Solver::clauses() const {
  return _clauses;
}

}  // end of namespace qsat ---------------------------------------------------









