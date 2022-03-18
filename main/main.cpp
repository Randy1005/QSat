#include <iostream>
#include <qsat/qsat.hpp>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "Usage: ./QSat input_dimacs_file.cnf" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  qsat::Solver solver;

  // TODO: function is alwasy little-case separated by _
  //       => parse_dimacs
  solver.ParseDimacs(argv[1]);
  
  // TODO: dump
  solver.Dump(std::cout);
}
