#include <cmath>
#include <iomanip>
#include <sstream>
#include "qsat.hpp"

namespace qsat {

Literal::Literal(int var) {
  if (var == 0) {
    throw std::runtime_error("variable cannot be zero");
  }
 id = (var > 0) ? 2 * var - 2 : 2 * -var - 1;
}

Clause::Clause(const std::vector<Literal>& lits, bool undef) :
  literals(lits),
  is_undef(undef)
{

}

Clause::Clause(std::vector<Literal>&& lits, bool undef) :
  literals(std::move(lits)),
  is_undef(undef)
{

}

// we may implement something in the constructor in the future, we don't know yet
Solver::Solver() :
  _order_heap(VarOrderLt(_activities))
{
}

void Solver::read_dimacs(const std::string& inputFileName) {
  std::ifstream ifs;
  ifs.open(inputFileName);

  if (!ifs) {
    throw std::runtime_error("failed to open a file");
  }

  read_dimacs(ifs);
}

void Solver::read_dimacs(std::istream& is) {
  int variable = -1;
  std::string buf;
  std::vector<Literal> literals;

  while (true) {
    is >> buf;

    if (is.eof()) {
      break;
    }
    if (buf == "c") {
      std::getline(is, buf);
    }
    else if (buf == "p") {
      is >> buf >> buf >> buf;
    }
    else {
      variable = std::stoi(buf);
      while (variable != 0) { 
        _read_clause(variable, literals); 
        is >> variable; 
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
  int max = 0;

  for (const auto& l : lits) {
    max = std::max(max, var(l));
  }

  if (max >= _assigns.size()) {
    _assigns.resize(max + 1);
  }
  
  for (const auto& l : lits) {
    // add new var
    _new_var(var(l));
  }

  _clauses.push_back(Clause(std::move(lits)));

}

void Solver::add_clause(const std::vector<Literal>& lits) {
  // resize the assignment vector to the current largest variable
  int max = 0;

  for (const auto& l : lits) {
    max = std::max(max, var(l));
  }

  if (max >= _assigns.size()) {
    _assigns.resize(max + 1);
  }
  
  for (const auto& l : lits) {
    // add new var
    _new_var(var(l));
  }

  _clauses.push_back(Clause(lits));
}

/*
bool Solver::_backtrack(int decision_depth, std::vector<Status>& assignments) {
  // base case: we exceeded the maximum decision depth
  // and still didn't find satisfiable assignments
  if (decision_depth >= num_variables()) {
    // std::cout << "reached max depth\n";
    return false;
  }

  for (int val = 0; val <= 1; val++) {
    assignments[decision_depth] = static_cast<Status>(val);
    
    // propagate constraints and update satisfiability of the clauses corresponding to
    // current deciding assignment (variable)
    // size_t added_sat_clauses_cnt = _propagate_constraint(decision_depth, assignments);

    // check if all clauses are sat
    // _num_sat_clauses += added_sat_clauses_cnt;
    
    if (_num_sat_clauses == num_clauses()) {
      return true;
    }

    if (_evaluate_clauses(assignments)) {
      return true;
    }


    if (_backtrack(decision_depth + 1, assignments)) {
      return true;
    }
    


    // if backtrack returns failure, clear out the previous assignment
    // 0 -> assign false, 1 -> assign true, 2 -> unassigned
    // num_sat_clauses minus the newly added sat clauses
    // reset clause satisfiability
    assignments[decision_depth] = Status::UNDEFINED;
  }

  // searched the whole tree, and didn't find a solution
  // std::cout << "searched the whole tree, returning\n"; 
  return false;
}
*/


/**
@brief this method checks if all clauses evaluate to true, if so return true
       if any one of the clauses evaluates to false, then return false
*/
/*
bool Solver::_evaluate_clauses(const std::vector<Status>& assignments) {
	for (const auto& c : _clauses) {
 		bool clause_is_sat = false;
		for (const auto& lit : c.literals) {
			// assignment[lit / 2] to get the corresponding variable's assignment
			// and xor with the rightmost bit of lit (lit & 1) 
			// (equals to checking if the lit is even)
			if (assignments[lit.id / 2] != Status::UNDEFINED && 
      stic_cast<int>(assignments[lit.id / 2]) ^ (lit.id & 1)) 
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
*/


void Solver::_print_assigns() {
  for (size_t i = 0; i < _assigns.size(); i++) {
    std::cout << static_cast<int>(_assigns[i]) << " ";
  }
  std::cout << "\n";
}


void Solver::dump(std::ostream& os) const {

  os << num_variables() << " variables\n";
  os << num_clauses() << " clauses\n";

  os << "\n";
}

void Solver::_init() {
  // initialize assignments
  _assigns.resize(num_variables());
  for (size_t i = 0; i < num_variables(); i++) {
    _assigns[i] = Status::UNDEFINED; // unassigned state
  }

}

void Solver::_insert_var_order(int v) {
  if (!_order_heap.in_heap(v)) {
    _order_heap.insert(v);
  }
}

void Solver::_new_var(int v) {
  _activities.resize(num_variables());
  // initialize activities[v] to 0.0
  _activities[v] = 0;

  // TODO: should consider placing _assigns.resize
  // here too, for consistency
  // initialize _assigns[v] to undefined
  _assigns[v] = Status::UNDEFINED;

  // initialize var info
  _var_info.resize(num_variables());
  
  // insert this var into order heap
  _insert_var_order(v);
}

void Solver::reset() {
  _assigns.clear();
  _clauses.clear();
}

bool Solver::solve() {

}

const std::vector<Clause>& Solver::clauses() const {
  return _clauses;
}

bool Solver::transpile_task_to_z3(const std::string& task_file_name) { 	 
	std::ifstream ifs;
  ifs.open(task_file_name);


  if (!ifs) {
    throw std::runtime_error("failed to open task file."); 
  }

  // open z3py file to write to
  _z3_ofs = std::ofstream("../intel_task_files/_gen_z3.py", std::ios::trunc);
  _z3_ofs << "from z3 import *\n";
  _z3_ofs << "from time import process_time\n"; 
  _z3_ofs << "s = Solver()\n";

  // parse task file
  std::string line_buf;
  std::stringstream write_ss;
  var_state v_state;
  constraint_state c_state;
  long recog_cnt = 0;
  long unrecog_cnt = 0;
  
  // predefine packed var
  // a PackedVar object has:
  // 1. an enum member
  // 2. a common value member
  // P.S. if some variables only have a common value specified
  // we can set the enum member as None
  _z3_ofs << "class PackedVar:\n"
          << "  def __init__(self,enum,common_val):\n"
          << "    self.enum = enum\n"
          << "    self.common_val = common_val\n\n\n";


  // predefine common values enum sort in z3
  // we only need to define this once
  // so do it here
  // probabaly only need up to ss_3?
  _z3_ofs << "E_SS, (ss_0,ss_1,ss_2,ss_3,ss_4,ss_5) "
          << "= EnumSort(\'E_SS\', "
          << "['ss_0','ss_1','ss_2','ss_3','ss_4','ss_5'])\n";
  
  while (true) {
    if (ifs.eof()) {
      break;
    }

    std::getline(ifs, line_buf);

    line_buf += '@';
    pegtl::string_input in(line_buf, "task_file");
    
    try {
      if (pegtl::parse<var_table_grammar, action>(in, v_state, write_ss) ||
          pegtl::parse<constraint_table_grammar, action>
                      (in, c_state, write_ss)) {
        recog_cnt++; 
        _z3_ofs << write_ss.str();

      } else {
        unrecog_cnt++;
      }

      // clear out the stringstream buffer
      write_ss.str("");

    }
    catch(const pegtl::parse_error& e) {
      const auto p = e.positions().front();
      std::cerr << e.what() << "\n"
                << in.line_at(p) << "\n"
                << std::setw(p.column) << '^' << std::endl;
      throw std::runtime_error("error parsing string.");
    }
    
  }

  // finished file reading
  // write time measurement code to z3 here
  _z3_ofs << "end_time = 0.0\n";
  _z3_ofs << "start_time = process_time()\n";
  _z3_ofs << "print(s.check())\n";
  _z3_ofs << "end_time = process_time()\n";
  _z3_ofs << "print(\"CPU time: \" + str((end_time - start_time) * 1000.0) + \" ms.\")\n";

  
  std::cout << "recog grammar: " << recog_cnt << std::endl;
  std::cout << "unrecog grammar: " << unrecog_cnt << std::endl;


  return true;

}

bool Solver::transpile_task_to_dimacs(const std::string& task_file_name) {
  return true;
}

}  // end of namespace qsat ---------------------------------------------------









