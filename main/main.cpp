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
	std::cout << "conflicts:\t" << s.conflicts << "\n";
	std::cout << "propagations:\t" << s.propagations << "\n";
	std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
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
	qsat::Solver s;
	qsat::Literal a(1), b(-7), c(9), d(-4), e(2);

  qsat::Clause c0({a, d, c});
  qsat::Clause c1({b, c});
  qsat::Clause c2({d, e});

  // NOTE:
  // we don't usually call add_clause like this
  // solely for the purpose of unit testing
  s.add_clause(c0.literals);
  s.add_clause(c1.literals);
  s.add_clause(c2.literals);

 	// watches (upon completing add clauses)
	
	// a: lit 0, b: lit 13
	// c: lit 16, d: lit 7
	// e: lit 2
	// ~a, ~d should be watched in c0
	// ~b, ~c should be watched in c1
	// ~d, ~e should be watched in c2
	

	// add two clauses
	// s.t. ~b (12) has a total of 3 watchers
	// [c1, 16], [c3, 0], [c4, 17]
	s.add_clause({b, ~a, e});
	s.add_clause({b, ~c, d});

	for (int i = 0; i < s.num_variables() * 2; i++) {
		std::cout << "watches[" << i << "]:\n";
		for (auto& w : s.watches[i]) {
			std::cout << "cref = " << w.cref << " | blocker = " << w.blocker.id << "\n";
		}
		std::cout << "\n";
	}

	// std::cout << "solve.result = " << static_cast<int>(res) << "\n";


	s.reduce_test();
	for (int i = 0; i < s.num_variables() * 2; i++) {
		std::cout << "watches[" << i << "]:\n";
		for (auto& w : s.watches[i]) {
			std::cout << "cref = " << w.cref << " | blocker = " << w.blocker.id << "\n";
		}
		std::cout << "\n";
	}

	
	std::cout << "solve.result = " << static_cast<int>(s.solve()) << "\n";
	*/




  return 0;
}
