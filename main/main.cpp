#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./QSat task_file.task" << std::endl;
    std::exit(EXIT_FAILURE);
  }
 
  qsat::Solver solver;
  
  std::chrono::steady_clock::time_point start_time, end_time; 

  start_time = std::chrono::steady_clock::now(); 

  bool res = solver.transpile_task_to_z3(argv[1]);  

  end_time = std::chrono::steady_clock::now(); 


   
  std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
            << elapsed_time.count()
            << " ms\n";
  

  return 0;
}
