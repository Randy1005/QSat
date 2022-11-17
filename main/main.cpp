#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
	if (argc < 2) {
    std::cerr << "Usage: ./QSat cnf_file [optional: out_file]]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  qsat::Solver s;
	s.read_dimacs(argv[1]);
  
	std::chrono::steady_clock::time_point start_time, end_time; 
  start_time = std::chrono::steady_clock::now(); 
	qsat::Status res = s.solve();
	end_time = std::chrono::steady_clock::now(); 

	std::cout << "================ QSat Statisitics ================\n";
	std::cout << "num variables:\t" << s.num_variables() << "\n";
	std::cout << "num clauses:\t" << s.num_orig_clauses << "\n";
	std::cout << "conflicts:\t" << s.conflicts << "\n";
	std::cout << "propagations:\t" << s.propagations << "\n";
	std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "run time: " 
            << elapsed_time.count() / 1000.0 
            << " s\n";
 
	if (res == qsat::Status::TRUE) {
		std::cout << "+++ SAT +++\n";
	}
	else if (res == qsat::Status::UNDEFINED) {
		std::cout << "*** UNDET ***\n";
	}
	else {
		std::cout << "--- UNSAT ---\n";
	}
	
	if (argc == 3) {
		std::ofstream os(argv[2]);
		s.dump(os);
	}

	std::cout << "==================================================\n";

	/*
	qsat::Literal a(1), b(-7), c(9), d(-4), e(2);
  qsat::Solver s;

  qsat::Clause c0({a, d, c});
  qsat::Clause c1({b, c});
  qsat::Clause c2({d, e});

  // NOTE:
  // we don't usually call add_clause like this
  // solely for the purpose of unit testing
  s.add_clause(c0.literals);
  s.add_clause(c1.literals);
  s.add_clause(c2.literals);
	// add two clauses
	// s.t. ~b (12) has a total of 3 watchers
	// [c1, 16], [c3, 0], [c4, 17]
	s.add_clause({b, ~a, e});
	s.add_clause({b, ~c, d});

 	// watches (upon completing add clauses)
	
	
	// a: lit 0, b: lit 13
	// c: lit 16, d: lit 7
	// e: lit 2
	// ~a, ~d should be watched in c0
	// ~b, ~c should be watched in c1
	// ~d, ~e should be watched in c2
	
	// watches[~a]'s 0th watcher clause is c0
	// watches[~a]'s 0th blocker is d (7)
	// watches[~d]'s 0th watcher clause is c0
	// watches[~d]'s 0th blocker is a (0)
	// watches[~d]'s 1th watcher clause is c2
	// watches[~d]'s 0th blocker is e (2)
	// watches[~b]'s 0th blocker is c (16)
	// watches[~c]'s 0th watcher clause is c1
	// watches[~c]'s 0th blocker is b (13)
	// watches[~e]'s 0th watcher clause is c2
	// watches[~e]'s 0th blocker is d (7)

	// detach c0
	// watches[~a] (watches[1]) would become empty
	// watches[~d] (watches[6]) would have a size of 1
	//		- the one watcher left is [c2, 2]
	s.remove_clause(0);



	// detach c3
	// watches[~b] would become:
	// [c1, 16], [c4, 17]
	s.remove_clause(3);


	// detach c1
	// watches[~b] (12) becomes:
	// [c4, 17]
	// watches[~c] (17) becomes empty
	s.remove_clause(1);
	
	s.clean_all_watches();
	*/

  return 0;
}
