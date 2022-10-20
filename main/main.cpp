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
  
  qsat::Literal a(1), b(-7), c(9), d(-4);
  qsat::Solver s;

  qsat::Clause c0({a, d, c});
  qsat::Clause c1({b, c});
  qsat::Clause c2({d});

  // NOTE:
  // we don't usually call add_clause like this
  // solely for the purpose of unit testing
  s.add_clause(c0.literals);
  s.add_clause(c1.literals);
  s.add_clause(c2.literals);


  // invoke enqueue
  // e.g. enq a, with c0 as reason clause
  //      enq b, with c1 as reason clause
  //      enq c, no reason clause
  //      enq d, with c2 as reason clause
  bool res_a = s.enqueue(a, c0);
  
  end_time = std::chrono::steady_clock::now(); 


  std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "Run time: " 
            << elapsed_time.count()
            << " ms\n";
  

  return 0;
}
