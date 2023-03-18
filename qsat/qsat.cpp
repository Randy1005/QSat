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

void Clause::calc_signature() {
  uint32_t sig = 0;

  // each literal signature is by hashing to 32-bit value
  // by ANDing var(lit) with 31 (bitwise 111110000....0)
  //                         ^            ^
  //                         bit0         bit31
  //
  // then the clause signature is calculated by ORing all 
  // the 2^(lit sig) together
  //
  //
  // e.g. 
  // c0.lits = {0, 2, 4}
  // var(lits) = {0, 1, 2}
  // -- sig(0) = 0 & 31 = 0
  // -- sig(1) = 1 & 31 = 1
  // -- sig(2) = 2 & 31 = 2
  // sig(c0) = 2^0 | 2^1 | 2^2 = 2 = (111000.....0)
  //
  // essentially saying that c0 contains var 0, 1, 2
  // but we don't know which literals they represent
  // lits could be {0, 2, 4}, {1, 3, 5}, {0, 1, 2, 3, 4, 5} etc.
  //
  // for the subset test pruning
  // say C = {0, 2, 4}, C' = {1, 3, 5}
  // sig(C)   = 111000....0
  // sig(C')  = 111000....0
  // sig(C) & ~sig(C') = 0 ---> signature hit
  // meaning all the variable abstracts in C matches up with
  // all the variable abstracts in C'
  // for these "signature hits" we need to do a full subset test
  //
  // otherwise if the var abstracts don't match, we know subsumptions
  // doesn't exist
  
  for (size_t i = 0; i < literals.size(); i++) {
    sig |= 1 << (var(literals[i]) & 31);
  }
  signature = std::move(sig);
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
	restart_first(100), // set to -1 to disable
	restart_inc(1.1),

	enable_reduce_db(true),
	enable_rnd_pol(true),
	enable_luby(false),
	learnt_size_factor(0.333),
	
  bid_verbosity(1),
  bid_steps_lim(10000),

  _mtrng(_rd()),
	_uni_real_dist(std::uniform_real_distribution(0.0, 1.0))
{
  breakid.set_verbosity(bid_verbosity);
  bid_steps_lim *= 1000LL;
  breakid.set_steps_lim(bid_steps_lim);
  breakid.set_useMatrixDetection(true);
  
}

void Solver::read_dimacs(const std::string& input_file) {
  std::ifstream ifs;
  ifs.open(input_file);
  if (!ifs) {
    throw std::runtime_error("failed to open a file");
  }
  read_dimacs(ifs);
}

void Solver::read_dimacs_bid(const std::string& input_file) {
  std::ifstream file(input_file);
  if (!file) {
    std::cerr << "BreakID Error: No CNF file found.\n";
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  std::vector<BID::BLit> inclause;
  while (std::getline(file, line)) {
    if (line.size() == 0 || line.front() == 'c') {
      // do nothing, this is a comment line
    } else if (line.front() == 'p') {
      std::string info = line.substr(6);
      std::istringstream iss(info);
      uint32_t nbVars;
      iss >> nbVars;
      uint32_t nbClauses;
      iss >> nbClauses;
      _bid_clauses.reserve(nbClauses);
    } else {
      //  parse the clauses, removing tautologies and double lits
      std::istringstream iss(line);
      int l;
      while (iss >> l) {
        if (l == 0) {
          if (inclause.size() == 0) {
            std::cerr << "BreakID Error: Theory cannot contain empty clause.\n";
            std::exit(EXIT_FAILURE);
          }
          _bid_clauses.push_back(inclause);
          inclause.clear();
        } else {
          inclause.push_back(BID::BLit(abs(l)-1, l < 0));
        }
      }
    }
  }
  
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
      is >> buf >> num_orig_vars >> num_orig_clauses;
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

void Solver::build_graph() {
  breakid.start_dynamic_cnf(num_variables());

  for (auto cl : _bid_clauses) {
    breakid.add_clause(cl.data(), cl.size()); 
  }

  breakid.end_dynamic_cnf();
}

void Solver::add_symm_brk_cls() {
  const auto brk_cls = breakid.get_brk_cls();
  for (auto cl : brk_cls) {
    std::vector<Literal> lits;
    for (auto l : cl) {
      Literal lit = LIT_UNDEF;
      lit.id = l.toInt();
      lits.push_back(std::move(lit)); 
    }
    add_clause(std::move(lits));
  }

  num_orig_clauses = _clauses.size();

}

void Solver::_read_clause(int variable, std::vector<Literal>& lits) { 
  lits.emplace_back(variable);
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
		_clauses.emplace_back(std::move(lits));
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
		_clauses.emplace_back(lits);
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

		for (i = j = 0; i < ws.size(); ) {
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
			assert(cr < _clauses.size());
			
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


Status Solver::search(int nof_conflicts) {
	std::vector<Literal> learnt_clause;
	int backtrack_level = 0;
	int conflict_c = 0;
	starts++;

	for (;;) {
		int confl_cref = propagate();
    
    if (confl_cref != CREF_UNDEF) {
			// conflict encountered!
			conflicts++;

			// conflict counter for one search
			// to trigger restart;
			conflict_c++;

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
				_clauses.emplace_back(learnt_clause, true);
					
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
			
			// exceeded restart interval, should restart
			if (nof_conflicts >= 0 && conflict_c >= nof_conflicts) {
				_cancel_until(0);
        return Status::UNDEFINED;
			}

			// exceeded max_learnt, should reduce clause database
			if (static_cast<double>(_learnts.size()) - num_assigns() >= max_learnts &&
					enable_reduce_db) {
				reduce_db();
			}

			Literal next_lit = LIT_UNDEF;

      // perform user provided assumptions
      while (decision_level() < _assumptions.size()) {
        Literal p = _assumptions[decision_level()];
        if (value(p) == Status::TRUE) {
          // dummy decision level
          _new_decision_level();
        }
        else if (value(p) == Status::FALSE) {
          // TODO:
          // What should we do here?
          // This is a final conflict 
          // in terms of assumptions
          // We wanna reason the set of assumptions
          // that led to the assignment of 'p'
          return Status::FALSE;
        }
        else {
          next_lit = p;
          break;
        }
      }


			if (next_lit == LIT_UNDEF) {
        // new variable decision
        decisions++;
        next_lit = _pick_branch_lit();
        if (next_lit == LIT_UNDEF) {
				  return Status::TRUE;
        }
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


void Solver::sycl_check_subsumptions() {
}


void Solver::remove_clause(const int cref) {
	Clause& c = _clauses[cref];
	_detach_clause(cref);

	if (locked(cref)) {
		_var_info[var(c.literals[0])].reason_cla = CREF_UNDEF;
	}	

	// mark this clause as to be removed
	c.mark = 1;	
}


void Solver::_attach_clause(const int cref) {
	const std::vector<Literal>& lits = _clauses[cref].literals;
	assert(lits.size() > 1);

	watches[(~lits[0]).id].emplace_back(cref, lits[1]);
	watches[(~lits[1]).id].emplace_back(cref, lits[0]);

	if (_clauses[cref].learnt) {
		num_learnts++;
	}

}

void Solver::_detach_clause(const int cref) {
	const std::vector<Literal>& lits = _clauses[cref].literals;
	assert(lits.size() > 1);
	
	// TODO:
	// minisat has strict/lazy clause detachment
	// we only implement lazy detachment for now
	_smudge_watch((~lits[0]).id);	
	_smudge_watch((~lits[1]).id);	
	
	assert(_watches_dirty[(~lits[0]).id]);
	assert(_watches_dirty[(~lits[1]).id]);

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

	bool operator () (const int x, const int y) const {
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



	_learnts.resize(_learnts.size() - i + j);
	
	for (int i = 0; i < _learnts.size(); i++) {
		assert(!is_removed(_learnts[i]));
		// set relocation crefs
		_clauses[_learnts[i]].reloc = num_orig_clauses + i; 
		
		// update learnt clauses indices to be in order again
		_learnts[i] = num_orig_clauses + i;
	}

	
	relocate_all();
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
		// NOTE:
		// variable stored in heap are indexed from 0
		// but our lit(var) interface requires it to index from 1
		// BE VERY CAUTIOUS in the future

		// TODO:
		// random polarity performs "sometimes" better than
		// fixed polarity, in terms of number of conflicts
		// come up with a better polarity mode in the future
		
		Literal p(next + 1);
	
		// if random polarity mode is enabled
		// there's a 5% of chance of picking the
		// positive polarity
		if (enable_rnd_pol) {
		  double prob = _uni_real_dist(_mtrng);
			return prob > 0.98 ? p : ~p;
		}
		else {
			return ~p;
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

	// resize watches dirty bits
	_watches_dirty.resize(2 * num_variables(), false);

	// resize the seen list
	_seen.resize(num_variables(), false);
}

void Solver::_smudge_watch(int p) {
	if (_watches_dirty[p] == false) {
		_watches_dirty[p] = true;
		_watches_dirties.push_back(p);
	}
}

void Solver::_clean_watch(int p) {
	std::vector<Watcher>& ws = watches[p];
	int i, j;
	for (i = j = 0; i < ws.size(); i++) {
		if (!watcher_deleted(ws[i])) {
			// keep this watcher by copying it to the front
			ws[j++] = ws[i];
		}
	}

	ws.resize(ws.size() - i + j);
	// and clear the dirty bit for this watches index
	_watches_dirty[p] = false;

}

double Solver::_luby(double y, int x) {
	int size, seq;
	for (size = 1, seq = 0; size < x + 1; seq++, size = 2 * size + 1);

	while (size - 1 != x) {
		size = (size - 1) >> 1;
		seq--;
		x = x % size;
	}
	return std::pow(y, seq);

}

void Solver::_luby_mis() {
  std::vector<bool> visited(num_variables(), false);
  std::vector<int> mis;

  for (int v = 0; v < num_variables(); v++) {
    if (visited[v]) {
      continue;
    }

    mis.push_back(v);

    // traverse v's watch list
    Literal p(v + 1);
    for (auto& w : watches[p.id]) {
      for (auto& l : _clauses[w.cref].literals) {
        if (!visited[var(l)] && var(l) != v) {
          visited[var(l)] = true;
        } 
      } 
    }

    for (auto& w : watches[(~p).id]) {
      for (auto& l : _clauses[w.cref].literals) {
        if (!visited[var(l)] && var(l) != v) {
          visited[var(l)] = true;
        } 
      } 
    }

  }


  // print out the maximal independent set
  /*
  std::cout << "mis:\n";
  for (int i = 0; i < mis.size(); i++) {
    std::cout << mis[i] << " ";
  }
  std::cout << "\n";
  */
  std::cout << "mis.size = " << mis.size() << "\n";

}

void Solver::clean_all_watches() {
	for (int i = 0; i < _watches_dirties.size(); i++) {
		// dirties might contain duplicates
		// so check first if these watchers are already cleaned
		if (_watches_dirty[_watches_dirties[i]]) {
			_clean_watch(_watches_dirties[i]);
			assert(!_watches_dirty[_watches_dirties[i]]);
		}
	}
	_watches_dirties.clear();
}

void Solver::relocate_all() {
	clean_all_watches();

	// update relocated crefs in watches
	for (int p = 0; p < watches.size(); p++) {
		std::vector<Watcher>& ws = watches[p];
		for (int i = 0; i < ws.size(); i++) {
			Clause& c = _clauses[ws[i].cref];
			if (c.reloc != -1) {
				ws[i].cref = c.reloc;
			}
		}
	}

	// update relocated crefs for reasons
	for (int p = 0; p < _trail.size(); p++) {
		int v = var(_trail[p]);
		// it's not safe to call 'locked' on relocated clauses
		// we'd rather leave these clauses dangling
		if (reason(v) != CREF_UNDEF && 
				(_clauses[reason(v)].reloc != -1 || locked(reason(v)))) {
			assert(!is_removed(reason(v)));
			if (_clauses[reason(v)].reloc != -1) {
				_var_info[v].reason_cla = _clauses[reason(v)].reloc;
			}
		}
	}


	std::vector<Clause> new_clauses(_clauses.size());

	for (int i = 0; i < _clauses.size(); i++) {
		const Clause& c = _clauses[i];
		if (!is_removed(i)) {
			if (c.reloc != -1) {
				new_clauses[c.reloc] = _clauses[i];
				new_clauses[c.reloc].reloc = -1;
			}
			else {
				// not relocated clause, assign it to the same index
				new_clauses[i] = _clauses[i];
			}
		}
		
	}

	new_clauses.resize(num_orig_clauses + _learnts.size());
	_clauses.resize(num_orig_clauses + _learnts.size());

	// update the actual clause database		
	_clauses = std::move(new_clauses);
}


void Solver::reset() {
  _assigns.clear();
  _clauses.clear();
}

Status Solver::solve() {
  // initialize inprocessor database
  init_device_db();

 	_model.clear();

	// initialize max learnt clause database size
	max_learnts = num_orig_clauses * learnt_size_factor;

	// TODO: budget can be defined too, conflict budget and propagtion budget
	
	int curr_restarts = 0;

  while (_solver_search_status == Status::UNDEFINED) {
		// calculate restart base with luby sequence
    // or geometric sequence
		double restart_base = enable_luby ? 
			_luby(restart_inc, curr_restarts) : 
			std::pow(restart_inc, curr_restarts);

		_solver_search_status = search(restart_base * restart_first);

		curr_restarts++;
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

Status Solver::solve(const std::vector<Literal>& assumps) { 
  _assumptions = std::move(assumps);
  return solve();
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


void Solver::init_device_db() {
  
  // initialize memory for CNF on device
  assert(num_clauses() != 0);
  
  // -----------------------------------------
  // TODO: first, parallel construct occurrence table on device 
  // ----------------------------------------
 

  // construct literal sequence
  // and clause lookup index list
  //
  // e.g.
  // C0 = {0, 2, 8}, C1 = {1, 4, 5}
  //
  // lit seq =      {0 2 8 1 4 5}
  //                 ^     ^
  // lookup index = {0     3    }    
  std::vector<uint32_t> lits, indices;
  std::vector<ClauseInfo> cl_infos;
  
  indices.emplace_back(0);
  for (size_t i = 0; i < num_clauses(); i++) {
    const auto& ls = _clauses[i].literals;
    
    // calculate signature for clause
    _clauses[i].calc_signature();
    cl_infos.emplace_back(0, 0, 0, 0, 
                          ls.size()*sizeof(uint32_t),
                          0, _clauses[i].signature); 

    for (auto l : ls) {
      lits.emplace_back(static_cast<uint32_t>(l.id)); 
    }

    if (i >= 1) {
      indices.emplace_back(ls.size()+indices[i-1]);
    }
  }
 
  assert(lits.size() != 0);
  assert(indices.size() != 0);

  inprocess = InprocessCnf(
                lits.size(), 
                indices.size(),
                &sycl_q,
                this);

  /*
  d_data.sh_cnf = sycl::malloc_device<uint32_t>(2*lits.size(), sycl_q); 
  d_data.sh_idxs = sycl::malloc_device<uint32_t>(2*indices.size(), sycl_q); 
  assert(d_data.sh_cnf); 
  assert(d_data.sh_idxs); 

  sycl_q.memcpy(d_data.sh_cnf, lits.data(), sizeof(uint32_t)*lits.size());
  sycl_q.memcpy(d_data.sh_idxs, indices.data(), sizeof(uint32_t)*indices.size());
  */

}

InprocessCnf::InprocessCnf(
    int n_lits, 
    int n_indices, 
    sycl::queue* q,
    Solver *s) :
  _q(q),
  _s(s)
{

  alloc();

}

void InprocessCnf::alloc() {

  
}






}  // end of namespace ---------------------------------------------------









