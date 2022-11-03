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
  
	std::cout << "conflicts:\t" << s.conflicts << "\n";
	std::cout << "propagations:\t" << s.propagations << "\n";
	std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
            << elapsed_time.count()
            << " ms\n";
 
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

  return 0;
}
