

// main model search loop
Status Solver::search(int max_conflicts, int max_learnts, SearchParam param) {
  // search parameter setup
  conflicts = 0;
  var_decay = 1 / param.var_decay;
  cla_decay = 1 / param.cla_decay;

  // solution model
  model.clear();

  // learnt clauses
  learnts.clear();

  // backtrack level
  bt_level = 0;

  while (true) {
    Clause confl = propagate();
    
    // conflict!
    if (confl != C_UNDEF) {
      // top level conflict -> cnf is UNSAT
      if (decision_level() == ROOT) {
        return Status::False;
      }

      // conflict analysis
      Clause learnt_clause;
      analyze(confl, learnt_clause, bt_level);

      // undo everything until the output bt_level
      cancel_until(bt_level);

      // add learnt clause into database
      record(learnt_clause);

      // decay var / clause activities
      decay_activities();

    }
    // no conflict, propagation successful
    else {
      // top level:
      // some preprocessing can be done
      // according to minisat
      if (decision_level() == 0) {
        simplify_db();
      }

      // if learnt clauses exceed max_learnts
      // reduce database
      if (learnts.size() - nAssigns() >= max_learnts) {
        reduce_db();
      }

      if (nAssigns() == nVars()) {
        // model found
        // populate model vector
        return Status::True;
      }
      else if (conflicts >= max_conflicts) {
        // force a restart
        cancel_until(ROOT);
        return Status::UNDEF;
      }
      else {
        // pick a new decision var
        // select_candidate can be a mix strategy of 
        // activity-based or random selection
        Lit next_p = select_candidate();
        
        // put this new decision on the propagation queue
        // and update its assignment, level, etc.
        enqueue(next_p);
      }
    }

  }
}


// top-level solve method
bool solve() {
  // example search parameter settings
  SearchParam param(var_decay = .95, cla_decay = .99);
  double max_conflicts = 100;
  double max_learnts = nClauses() / 3;
  Status stat = UNDEF;

  while (stat == UNDEF) {
    stat = search((int)max_conflicts, (int)max_learnts, param);
    // example max conflict, learnts scaling factors
    max_conflicts *= 1.5;
    max_learnts *= 1.1;
  }

  // for incremental solver purposes
  cancel_until(ROOT);

  return stat;
}

