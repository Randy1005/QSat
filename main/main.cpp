#include <iostream>
#include <qsat/qsat.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
	if (argc < 2) {
    std::cerr << "Usage: ./QSat cnf_file [optional: out_file]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  qsat::Solver s0, s1, s2, s3;
	qsat::Status res0, res1, res2, res3;
	s0.read_dimacs(argv[1]);
	s1.read_dimacs(argv[1]);
	s2.read_dimacs(argv[1]);
	s3.read_dimacs(argv[1]);
  
	std::chrono::steady_clock::time_point start_time, end_time; 
  start_time = std::chrono::steady_clock::now(); 
	
	
	tf::Taskflow taskflow;
	tf::Executor executor;

	auto [t0, t1, t2, t3] = taskflow.emplace(
			[&s0, &res0]() { 
				qsat::Literal a{1}, b{2};
				res0 = s0.solve({a, b});
				std::cout << "s0 result = " << static_cast<int>(res0) << "\n";
			},
			[&s1, &res1]() { 
				qsat::Literal a{1}, b{2};
				res1 = s1.solve({~a, b});
				std::cout << "s1 result = " << static_cast<int>(res1) << "\n";
			},
			[&s2, &res2]() { 
				qsat::Literal a{1}, b{2};
				res2 = s2.solve({a, ~b});
				std::cout << "s2 result = " << static_cast<int>(res2) << "\n";
			},
			[&s3, &res3]() { 
				qsat::Literal a{1}, b{2};
				res3 = s3.solve({~a, ~b});
				std::cout << "s3 result = " << static_cast<int>(res3) << "\n";
			}
	);


	executor.run(taskflow).wait();
	
	end_time = std::chrono::steady_clock::now(); 
	std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;  
  std::cout << "run time: " 
            << elapsed_time.count() / 1000.0 
            << " s\n";
	/*
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
	*/
  return 0;
}
