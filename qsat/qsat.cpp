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
	
	// TODO: consider making these into a cli option
	var_inc(1),
	cla_inc(1),
	var_decay(0.95),
	cla_decay(0.999),
	phase_saving(0),

	enable_reduce_db(false),
	enable_rnd_pol(false),
	learnt_size_factor(0.333),
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
      is >> buf >> _num_orig_clauses >> buf;
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
		
		// move qhead forward, and get an enqueued fact p
		// p is the literal we're propagating
		Literal p = _trail[_qhead++];
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

			/*
			std::cout << "c" << cr << " learnt? " << c.learnt << "\n";
			for (auto& l : c.literals) {
				std::cout << l.id << ", ";
			}
			std::cout << "\nfalse_lit = " << false_lit.id << "\n";
			*/
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
				while (i < ws.size()) {
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
		// keep the first ws.size - (i-j) watchers
		ws.resize(ws.size() - i + j);
	}
	
	propagations += num_props;
	return confl_cref;
}


Status Solver::search() {
	std::vector<Literal> learnt_clause;
	int backtrack_level = 0;

	for (;;) {
		int confl_cref = propagate();
		if (confl_cref != CREF_UNDEF) {
			// conflict encountered!
			conflicts++;

			if (decision_level() == 0) {
				// top level conflict : UNSAT
				return Status::FALSE;
			}
			
			learnt_clause.clear();
			analyze(confl_cref, learnt_clause, backtrack_level);	
			
			// undo everything until the backtrack level
			_cancel_until(backtrack_level);
		
			
			if (learnt_clause.size() == 1) {
				// immediately enqueue the only literal
				unchecked_enqueue(learnt_clause[0], CREF_UNDEF);
			}
			else {
				// store the learnt clause
				int learnt_cref = _clauses.size();
				_learnts.push_back(learnt_cref);
				_clauses.push_back(Clause(learnt_clause, true));
					
				// initialize watches for this clause
				_attach_clause(learnt_cref);

				// bump this learnt clauses' activity
				cla_bump_activity(_clauses[learnt_cref]);
					
				// learnt[0] immediately becomes the next
				// candidate to propagate
				unchecked_enqueue(learnt_clause[0], learnt_cref);
			}

			var_decay_activity();
			cla_decay_activity();
		}
		else {
			// no conflict, we can continue making decisions
			
			// TODO: exceed conflict budget, should restart

			// exceeded max_learnt, should reduce clause database
			if (static_cast<double>(_learnts.size()) - num_assigns() >= max_learnts &&
					enable_reduce_db) {
				reduce_db();	
			}

			decisions++;
			Literal next_lit = _pick_branch_lit();
			if (next_lit == LIT_UNDEF) {
				return Status::TRUE;
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
	out_learnt.push_back(p);


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
				_seen[var(q)] = true;
			
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
	// before calculating the backtrack level here


	// find the correct backtrack level
	if (out_learnt.size() == 1) {
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
		Literal r = out_learnt[max];
		out_learnt[max] = out_learnt[1];
		out_learnt[1] = r;
		out_btlevel = level(var(r));
	}

	// reset the seen list
	for (int i = 0; i < _seen.size(); i++) {
		_seen[i] = false;
	}
}

void Solver::remove_clause(const int cref) {
	Clause& c = _clauses[cref];
	_detach_clause(cref);

	if (locked(cref)) {
		_var_info[var(c.literals[0])].reason_cla = CREF_UNDEF;
	}

	// free up memory in this clause 
	c.literals.erase(c.literals.begin(), c.literals.end());
	c.literals.clear();
}


void Solver::_attach_clause(const int cref) {
	const std::vector<Literal>& lits = _clauses[cref].literals;
	assert(lits.size() > 1);

	watches[(~lits[0]).id].push_back(Watcher(cref, lits[1]));
	watches[(~lits[1]).id].push_back(Watcher(cref, lits[0]));

	if (_clauses[cref].learnt) {
		num_learnts++;		
	}

}

void Solver::_detach_clause(const int cref) {
	const std::vector<Literal>& lits = _clauses[cref].literals;
	assert(lits.size() > 1);
	
	// NOTE:
	// minisat has strict/lazy clause detachment
	// we only implement strict detachment for now
	std::vector<Watcher>& ws0 = watches[(~lits[0]).id];
	std::vector<Watcher>& ws1 = watches[(~lits[1]).id];
	
	int i = 0, j = 0;	
	Watcher to_detach0(cref, lits[1]);
	Watcher to_detach1(cref, lits[0]);

	for (; i < ws0.size() && ws0[i] != to_detach0; i++); 
	for (; j < ws1.size() && ws1[j] != to_detach1; j++); 
		
	assert(i < ws0.size());
	assert(j < ws1.size());
	
	// swap the watcher to remove with
	// the last element, and pop it
	// TODO: more efficient way to remove?
	ws0[i] = ws0[ws0.size() - 1];
	ws0.pop_back();

	ws1[j] = ws1[ws1.size() - 1];
	ws1.pop_back();
	
	if (_clauses[cref].learnt) {
		num_learnts--;	
	}
}

struct reduce_db_lt {
	std::vector<Clause>& clauses;
	reduce_db_lt(std::vector<Clause>& cs) :
		clauses(cs)
	{
	}

	bool operator () (int x, int y) {
		return clauses[x].literals.size() > 2 &&
			(clauses[y].literals.size() == 2 || 
			 clauses[x].activity < clauses[y].activity);
	}
};



void Solver::reduce_db() {
	int i, j;

	// remove any clause below this activity
	double extra_limit = cla_inc / _learnts.size();

	std::sort(_learnts.begin(), _learnts.end(), reduce_db_lt(_clauses));
	for (i = j = 0; i < _learnts.size(); i++) {
		Clause& c = _clauses[_learnts[i]];
		if (c.literals.size() > 2 && !locked(_learnts[i]) &&
				(i < _learnts.size() / 2 || c.activity < extra_limit)) {
			remove_clause(_learnts[i]);
		}
		else {
			_learnts[j++] = _learnts[i];	
		}
	}


	int shrink_size = i - j;
	_learnts.resize(_learnts.size() - shrink_size);
	
	// TODO: figure out a way to correctly shrink the _clauses 
	// coerce them together, now _clauses is fragmented
	// TODO: maybe try clearing all the learnt in _clauses and attach them again
	// using the new, sorted, reduced _learnts
	/*
	for (int k = 0; k < _learnts.size(); k++) {
		int cr = _clauses[_learnts[k]];

	}

	std::cout << "after reduce:\n";
	std::cout << "_clauses.size = " << _clauses.size() << "\n";
	std::cout << "_learnts.size = " << _learnts.size() << "\n";
	*/
}


void Solver::dump_assigns(std::ostream& os) const {
  for (size_t i = 0; i < _assigns.size(); i++) {
		if (_assigns[i] == Status::TRUE || _assigns[i] == Status::UNDEFINED) {
			os << i + 1;
		}
		else {
			os << "-" << (i + 1);
		}
		os << " ";
	}
	// add a 0 denoting the end of solution
  os << "0\n";
}


void Solver::dump(std::ostream& os) const {
	switch(_solver_search_status) {
		case Status::TRUE:
			os << "SAT\n";
			dump_assigns(os);
			break;
		case Status::FALSE:
			os << "UNSAT\n";
			break;
		case Status::UNDEFINED:
			os << "UNDET\n";
			break;

		default:
			break;
	}
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
		// but our lit(var) interface requires it to index from 1
		// BE VERY CAUTIOUS in the future

		// TODO:
		// random polarity performs "sometimes" better than
		// fixed polarity, in terms of number of conflicts
		// come up with a better polarity mode in the future
		
		Literal p(next + 1);
		int rnd = static_cast<int>(_uni_int_dist(_mtrng)) % 2;
		
		if (enable_rnd_pol) {
			return rnd ? p : ~p;
		}
		else {
			return p;
		}
	}

}

void Solver::_cancel_until(int level) {
	if (decision_level() > level) {
		for (int u = _trail.size() - 1; u >= _trail_lim[level]; u--) {
			int v = var(_trail[u]);
			
			// undo assignment
			_assigns[v] = Status::UNDEFINED;

			// TODO: implement phase saving
			if (phase_saving > 1 || (phase_saving == 1 && u > _trail_lim.back())) {
			
			}

			// re-insert variables into heap
			// they were removed when they were selected for propagtion
			_insert_var_order(v);
		}

		// point qhead to 'level'
		_qhead = _trail_lim[level];
		
		// shrink trail
		_trail.resize(_trail_lim[level]);

		// revert trail_lim
		_trail_lim.resize(level);
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

	// resize watches
	watches.resize(2 * num_variables());

	// resize the seen list
	_seen.resize(num_variables(), false);
}

void Solver::reset() {
  _assigns.clear();
  _clauses.clear();
}

Status Solver::solve() {
	_model.clear();

	// initialize max learnt clause database size
	max_learnts = _num_orig_clauses * learnt_size_factor;

	// TODO: restart configurations can be implemented
	// TODO: budget can be defined too, conflict budget and propagtion budget
	while (_solver_search_status == Status::UNDEFINED) {
		_solver_search_status = search();	
	}

	if (_solver_search_status == Status::TRUE) {
		// extend and copy model
		_model.resize(num_variables(), Status::UNDEFINED);
		for (int i = 0; i < num_variables(); i++) {
			_model[i] = value(i);
		}
	}
	
	// revert all the way back to level 0
	// (if we're doing incremental solving)
	// _cancel_until(0);
	return _solver_search_status;
}

const std::vector<Clause>& Solver::clauses() const {
  return _clauses;
}


/*
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

*/

}  // end of namespace qsat ---------------------------------------------------









