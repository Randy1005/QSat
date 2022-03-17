#include <iostream>
#include <qsat.hpp>

int main(int argc, char* argv[]) {
	if(argc != 2) {
    std::cerr << "Usage: ./QSat input_dimacs_file.cnf" << std::endl;
    std::exit(EXIT_FAILURE);
  }
	
	qsat::Solver solver;
	solver.ParseDimacs(argv[1]);

  solver.Dump(std::cout);
}
