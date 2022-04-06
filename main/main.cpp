#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./QSat input_dimacs_file.cnf" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  qsat::Solver solver;
  solver.read_dimacs(argv[1]);

  std::chrono::steady_clock::time_point start_time, end_time; 
  start_time = std::chrono::steady_clock::now(); 
  if (solver.solve()) {
    end_time = std::chrono::steady_clock::now();
    solver.dump(std::cout);
  }

  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Run time: " 
            <<  elapsed_time.count() * 1000.0f
            << " ms\n";


  return 0;
}
