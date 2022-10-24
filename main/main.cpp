#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./QSat cnf_file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
 
  qsat::Solver solver;
  
  std::chrono::steady_clock::time_point start_time, end_time; 

  start_time = std::chrono::steady_clock::now(); 
  
  qsat::Solver s;

	s.read_dimacs(argv[1]);
	

	bool res = s.search();

	// a lucky example with no conflicts
	/*
	// watches (upon completing add clauses)
	for (size_t p = 0; p < 2 * s.num_variables(); p++) {

		std::cout << "watches[" << p << "] = \n";
		std::cout << ";;;;;;;;;;;;;;;;;;\n";
		for (auto& w : s.watches[p]) {
			for (auto& p : s.clause(w.cref).literals) {
				std::cout << p.id << " ";
			}
			std::cout << "\n";
			std::cout << "blocker: " << w.blocker.id << "\n";
			std::cout << "====================\n";
		}
	}

	s.enqueue(qsat::Literal(-1));
	std::cout << "assigns:\n";
	s.print_assigns();

	std::cout << "prop result = " << s.propagate() << "\n";

	// watches (after 1 propagation)
	for (size_t p = 0; p < 2 * s.num_variables(); p++) {

		std::cout << "watches[" << p << "] = \n";
		std::cout << ";;;;;;;;;;;;;;;;;;\n";
		for (auto& w : s.watches[p]) {
			for (auto& p : s.clause(w.cref).literals) {
				std::cout << p.id << " ";
			}
			std::cout << "\n";
			std::cout << "blocker: " << w.blocker.id << "\n";
			std::cout << "====================\n";
		}
	}

	s.enqueue(qsat::Literal(2));
	std::cout << "assigns:\n";
	s.print_assigns();


	std::cout << "prop result = " << s.propagate() << "\n";
	
	// watches (after 2 propagation)
	for (size_t p = 0; p < 2 * s.num_variables(); p++) {

		std::cout << "watches[" << p << "] = \n";
		std::cout << ";;;;;;;;;;;;;;;;;;\n";
		for (auto& w : s.watches[p]) {
			for (auto& p : s.clause(w.cref).literals) {
				std::cout << p.id << " ";
			}
			std::cout << "\n";
			std::cout << "blocker: " << w.blocker.id << "\n";
			std::cout << "====================\n";
		}
	}
	
	
	std::cout << "assigns:\n";
	s.print_assigns();

	s.enqueue(qsat::Literal(4));
	std::cout << "prop result = " << s.propagate() << "\n";
	std::cout << "assigns:\n";
	s.print_assigns();
	*/


  end_time = std::chrono::steady_clock::now(); 
	



  std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
            << elapsed_time.count()
            << " ms\n";
  

  return 0;
}
