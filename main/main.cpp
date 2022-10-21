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


	// watches (upon completing add clauses)
	int p = 12;
	std::cout << "watches[" << p << "] = \n";
	for (auto& w : s.watches[p]) {
		for (auto& p : s.clause(w.cref).literals) {
			std::cout << p.id << " ";
		}
		std::cout << "\n";
		std::cout << "blocker: " << w.blocker.id << "\n";
		std::cout << "====================\n";
	}
	
	


  end_time = std::chrono::steady_clock::now(); 


  std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
            << elapsed_time.count()
            << " ms\n";
  

  return 0;
}
