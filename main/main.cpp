#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
	if (argc < 2) {
    std::cerr << "Usage: ./QSat cnf_file [optional: out_file]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

	// test metis
	idx_t n = 6;

	qsat::Solver s;
	s.read_dimacs(argv[1]);

  
  auto start_t = std::chrono::steady_clock::now(); 
	qsat::Status res = s.solve();
	auto end_t = std::chrono::steady_clock::now(); 
  std::cout << "run time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(
								end_t - start_t
							 ).count() / 1000.0
            << " s\n";

	std::cout << "================ QSat Statisitics ================\n";
	std::cout << "num variables:\t" << s.num_variables() << "\n";
	std::cout << "num clauses:\t" << s.num_orig_clauses << "\n";
	std::cout << "restarts:\t" << s.starts << "\n";
	std::cout << "conflicts:\t" << s.conflicts << "\n";
	std::cout << "propagations:\t" << s.propagations << "\n";
 
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
	return 0;
}
