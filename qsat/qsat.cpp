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

Clause::Clause(const std::vector<Literal>& lits, bool is_learnt) :
  literals(lits),
	learnt(is_learnt)
{
}

Clause::Clause(std::vector<Literal>&& lits, bool is_learnt) :
  literals(std::move(lits)),
  learnt(is_learnt)
{
}



Solver::Solver() :
  _order_heap(VarOrderLt(_activities)),
	_qhead(0),

	var_inc(1),
	cla_inc(1),
	var_decay(0.95),

	_mtrng(_rd())
{

}

void Solver::read_dimacs(const std::string& input_file) {
  std::ifstream ifs;
  ifs.open(input_file);

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
	

	if (lits.size() == 0) {
		// empty clause
		// TODO: should handle this
	}
	else if (lits.size() == 1) {
		// unit clause
		// don't store it
		// instead, enqueue the only literal
		enqueue(lits[0]);	
	}
	else {
		// initialize watcher literals for this clause
		_clauses.push_back(Clause(std::move(lits)));
		_attach_clause(num_clauses() - 1);
	}

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

	
	if (lits.size() == 0) {
		// empty clause
		// TODO: should handle this
	}
	else if (lits.size() == 1) {
		// unit clause
		// don't store it
		// instead, enqueue the only literal
		enqueue(lits[0]);	
	}
	else {
		// initialize watcher literals for this clause
		_clauses.push_back(Clause(lits));
		_attach_clause(num_clauses() - 1);
	}
	
}


int Solver::propagate() {
	int confl_cref = CREF_UNDEF;
	int num_props = 0;

	while (_qhead < _trail.size()) {
		// move qhead forward, and get an enqueued face p
		// p is the literal we're propagating
		Literal p = _trail[_qhead++];
		
		std::cout << "propagating lit " << p.id << "...\n";
		num_props++;	
		// obtain the watchers of this literal
		// std::vector<Watcher> ws = std::move(watches[p.id]);
		std::vector<Watcher>& ws = watches[p.id];

		size_t i, j;
		bool next_watcher;
		for (i = j = 0; i < ws.size(); )	{
			next_watcher = false;
			// no need to inspect clause if
			// the blocker literal is already satisfied
			// and copy this watcher to the front
			Literal blocker = ws[i].blocker;
			if (value(blocker) == Status::TRUE) {
				ws[j++] = ws[i++];
				continue;
			}


			// make sure the false literal is data[1]
			int cr = ws[i].cref;
			Clause& c = _clauses[cr];
			Literal false_lit = ~p;
			if (c.literals[0] == false_lit) {
				c.literals[0] = c.literals[1];
				c.literals[1] = false_lit;
			}
			assert(c.literals[1] == false_lit);
			i++;
			
			// if 0th watch is true, then this
			// clause is already satisfied
			Literal first = c.literals[0];
			Watcher w = Watcher(cr, first);
			if (first != blocker && value(first) == Status::TRUE) {
				// copy this new watcher (c, blocker = c[0])
				// to the front and move on to the
				// next watcher clause
				ws[j++] = w;
				continue;
			}

			// then we look for an unwatched && non-falsified
			// literal to use as a new watch
			// (start from the 3rd literal in clause)
			for (size_t k = 2; k < c.literals.size(); k++) {
				if (value(c.literals[k]) != Status::FALSE) {
					c.literals[1] = c.literals[k];
					c.literals[k] = false_lit;
					// and update watchers
					watches[(~c.literals[1]).id].push_back(w);
					next_watcher = true;
					break;
				}
			}
			

			if (next_watcher) {
				continue;
			}


			// if we reached here:
			// no new watch is found
			// this clause is unit, we try to propagate
			// (we check if c[0] conflicts with current assignment)
			ws[j++] = w;
			if (value(first) == Status::FALSE) {
				// conflict!
				confl_cref = cr;
				// propagation ends, move qhead to end of trail
				_qhead = _trail.size();
				// and copy the remaining watches
				// they still need to be examined in the next propagation
				while (i < c.literals.size()) {
					ws[j++] = ws[i++];
				}
			}
			else {
				// valid propagation!
				// place this new fact on propagation queue
				// and record the reason clause
				unchecked_enqueue(first, cr);
			}
		}
		// shrink the watcher list for this literal
		// keep the first c.size - (i-j) watchers
		if (i - j != 0) {
			ws.resize(ws.size() - i + j);
		}	
	}
	
	
	propagations += num_props;
	return confl_cref;
}


bool Solver::search() {
	
	std::vector<Literal> learnt_clause;
	int backtrack_level;
	
	for (;;) {
		int confl_cref = propagate();
		if (confl_cref != CREF_UNDEF) {
			// conflict encountered!
			std::cout << "conflict!\n";
			
			std::cout << "CONFLICT clause:\n";
			Clause& c = _clauses[confl_cref];
			for (int i = 0; i < c.literals.size(); i++) {
				std::cout << c.literals[i].id << ", ";
			}
			std::cout << "\n";

			if (decision_level() == 0) {
				// top level conflict : UNSAT
				return false;
			}
			
			learnt_clause.clear();
			analyze(confl_cref, learnt_clause, backtrack_level);	
		
			std::cout << "learnt clause:\n";
			for (int i = 0; i < learnt_clause.size(); i++) {
				std::cout << learnt_clause[i].id << ", ";
			}
			std::cout << "\n";
			std::cout << "bt_lvl: " << backtrack_level << "\n";
			
			return false;
		}
		else {
			// no conflict, we can continue making decisions

			Literal next_lit = _pick_branch_lit();
			

			std::cout << "#DECISION lit: " << next_lit.id << "\n";

			if (next_lit == LIT_UNDEF) {
				std::cout << "next_lit undef, found a solution!\n";
				print_assigns();
				return true;
			}

			// begin new decision level
			_new_decision_level();
			unchecked_enqueue(next_lit, CREF_UNDEF);

		}
	
	}


}



void Solver::analyze(int confl_cref, 
		std::vector<Literal>& out_learnt, 
		int& out_btlevel) {
	
	int path_cnt = 0;
	Literal p = LIT_UNDEF;
	
	// leave room for the asserting literal
	out_learnt.push_back(LIT_UNDEF);

	// traverse from the tail of the trail
	int index = _trail.size() - 1;

	do {
		assert(confl_cref != CREF_UNDEF);
		Clause& confl_c = _clauses[confl_cref];
		// if this conflict clause is already learnt
		// i.e. a previously learnt clause causing conflict
		// this clause should be considered more 'active'
		if (confl_c.learnt) {
			cla_bump_activity(confl_c);
		}

		// if p == lit_undef, that means we're looking at the 
		// conflict point (the one 'propagation()' produced)
		// so every literal of confl_c has to be considered
		//
		// if not, then we're in the process of traversing the
		// implication graph, confl_c[0] is the asserting literal
		// (meaning we exclude this lit when we produce a learnt clause)
		for (size_t j = (p == LIT_UNDEF) ? 0 : 1; j < confl_c.literals.size(); j++) {
			Literal q = confl_c.literals[j];
			
			// if we haven't checked this variable yet
			// and its decision level > 0
			// that means this variable is a contribution to
			// conflicts, so bump its activity
			if (!_seen[var(q)] && level(var(q)) > 0) {
				var_bump_activity(var(q));
				_seen[var(q)] = 1;
			
				// if level(q) > current decision level
				// that means we're have not yet reached the
				// UIP (unique implication point)
				// so increment the path counter
				// 
				// NOTE: path_cnt represents the branches originating
				// from the conflict point
				if (level(var(q)) >= decision_level()) {
					path_cnt++;
				}
				// otherwise, we have found a literal at UIP
				else {
					out_learnt.push_back(q);
				}
			}
		}

		// select the next literal to look at
		// skip the ones that are already examined
		while (!_seen[var(_trail[index--])]);
		p = _trail[index + 1];
		// use the reason clause of the new p as conflict clause
		confl_cref = reason(var(p));
		_seen[var(p)] = 0;
		path_cnt--;

	} while (path_cnt > 0);

	// add the asserting literal
	out_learnt[0] = ~p;


	// TODO: we could implement clause simplification
	// before calculating the backtrack level


	// find the correct backtrack level
	if (out_learnt.size() == 1) {
		// meaning it's a top level conflict
		out_btlevel = 0;
	}			
	else {
		int max = 1;
		// find the literal with second-highest level
		// and swap it out with out_learnt[1]
		// so that we have the first and second highest level
		// at [0] and [1]
		for (int i = 2; i < out_learnt.size(); i++) {
			if (level(var(out_learnt[i])) > level(var(out_learnt[max]))) {
				max = i;
			}
		}

		// swap in this literal at index 1
		Literal p = out_learnt[max];
		out_learnt[max] = out_learnt[1];
		out_learnt[1] = p;
		
		out_btlevel = level(var(p));
	}

	// clear the seen list
	_seen.clear();
}


void Solver::_attach_clause(const int c_id) {
	std::vector<Literal>& lits = _clauses[c_id].literals;
	watches[(~lits[0]).id].push_back(Watcher(c_id, lits[1]));
	watches[(~lits[1]).id].push_back(Watcher(c_id, lits[0]));
}

// TODO: will be needed when we reduce learnt clauses
void Solver::_detach_clause(const int c_id) {
}


void Solver::print_assigns() {
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

Literal Solver::_pick_branch_lit() {
	// NOTE: minisat uses a mixed strategy
	// combining random selection + activity-based
	// and then choose a polarity for that var
	// we use only activity-based for now
	
	int next = VAR_UNDEF;
	
	// TODO: make a random decision first?

	// activity-based decision
	while (next == VAR_UNDEF || value(next) != Status::UNDEFINED) {
		
		if (_order_heap.empty()) {
			next = VAR_UNDEF;
			break;
		}
		else {
			next = _order_heap.remove_max();
		}
	}


	// choose polarity for this variable (making a literal)
	// for now, use a random polarity
	if (next == VAR_UNDEF) {
		return LIT_UNDEF;
	}
	else {
		// WARNING:
		// variable stored in heap are indexed from 0
		// but our interface requires it to index from 1
		// BE VERY CAUTIOUS in the future
		Literal p(next + 1);
		
		return (_uint_dist(_mtrng) % 2) ? p : ~p;	
	}

}

void Solver::_insert_var_order(int v) {
	// TODO: somehow not inserting all the variables??? 
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

	// resize watches
	watches.resize(2 * num_variables());

	// resize the seen list
	_seen.resize(num_variables());
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









