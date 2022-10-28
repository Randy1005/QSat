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
	
	qsat::Status res = s.solve();
	
		
  end_time = std::chrono::steady_clock::now(); 
  std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
            << elapsed_time.count()
            << " ms\n";
 
	std::cout << "conflicts: " << s.conflicts << "\n";
	if (res == qsat::Status::TRUE) {
		std::cout << "+++ SAT +++\n";
	}
	else if (res == qsat::Status::UNDEFINED) {
		std::cout << "--- ran out of budget? ---\n";
	}
	else {
		std::cout << "--- UNSAT ---\n";
	}


  return 0;
}
