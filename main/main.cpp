#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
	if (argc < 2) {
    std::cerr << "Usage: ./QSat cnf_file [optional: out_file]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  qsat::Solver s;
	qsat::Status res;
	s.read_dimacs(argv[1]);
  
	std::chrono::steady_clock::time_point start_time, end_time; 
  start_time = std::chrono::steady_clock::now(); 
	
	
	tf::Taskflow taskflow;
	tf::Executor executor;
	auto [example, solve] = taskflow.emplace(
			[]() { std::cout << "another task\n"; },
			[&res, &s]() { res = s.solve(); }
	);


	executor.run(taskflow).wait();
	
	end_time = std::chrono::steady_clock::now(); 

	std::cout << "================ QSat Statisitics ================\n";
	std::cout << "num variables:\t" << s.num_variables() << "\n";
	std::cout << "num clauses:\t" << s.num_orig_clauses << "\n";
	std::cout << "restarts:\t" << s.starts << "\n";
	std::cout << "conflicts:\t" << s.conflicts << "\n";
	std::cout << "propagations:\t" << s.propagations << "\n";
	std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "run time: " 
            << elapsed_time.count() / 1000.0 
            << " s\n";
 
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
