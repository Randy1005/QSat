#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./QSat input_dimacs_file.cnf" << std::endl;
    std::exit(EXIT_FAILURE);
  }
 
  qsat::Solver solver;
  
  std::chrono::steady_clock::time_point start_time, end_time; 
  solver.read_dimacs(argv[1]);

  start_time = std::chrono::steady_clock::now(); 
  if (solver.solve()) {
    end_time = std::chrono::steady_clock::now();
    solver.dump(std::cout);
  } else {
    end_time = std::chrono::steady_clock::now();
    std::cout << "UNSAT\n";
  }

  std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
            << elapsed_time.count()
            << " ms\n";


  return 0;
}
