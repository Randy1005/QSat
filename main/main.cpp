#include <iostream>
#include <chrono>
#include <qsat/qsat.hpp>

//int main(int argc, char* argv[]) {
//	if (argc < 2) {
//    std::cerr << "Usage: ./QSat cnf_file [optional: out_file]\n";
//    std::exit(EXIT_FAILURE);
//  }
//
//	tf::Taskflow taskflow;
//	tf::Executor executor;
//
//	qsat::Solver s;
//
//	auto [readcnf, 
//			  readcnf_bid, 
//	      break_sym,
//				solve_bid] = 
//	taskflow.emplace(
//		[&]() {
//			s.read_dimacs(argv[1]);
//		},
//		[&]() {
//			s.read_dimacs_bid(argv[1]);
//		},
//		[&]() {
//			s.build_graph();
//			s.breakid.detect_subgroups();
//			s.breakid.break_symm();
//			if (s.bid_verbosity) {
//				s.breakid.print_symm_break_stats();
//			}
//			std::cout << "Num sym breaking clauses: " << 
//									 s.breakid.get_num_break_cls() << "\n";
//			s.add_symm_brk_cls();
//			std::cout << "Num total clauses: " <<
//									 s.num_clauses() << "\n";
//		
//		},
//		[&]() {
//			auto start_t = std::chrono::steady_clock::now(); 
//			qsat::Status res = s.solve();
//			auto end_t = std::chrono::steady_clock::now(); 
//			std::cout << "================ QSat Statisitics ================\n";
//			std::cout << "run time: " 
//								<< std::chrono::duration_cast<std::chrono::milliseconds>(
//										end_t - start_t
//									 ).count() / 1000.0
//								<< " s\n";
//			std::cout << "num variables:\t" << s.num_variables() << "\n";
//			std::cout << "num clauses:\t" << s.num_orig_clauses << "\n";
//			std::cout << "restarts:\t" << s.starts << "\n";
//			std::cout << "conflicts:\t" << s.conflicts << "\n";
//			std::cout << "propagations:\t" << s.propagations << "\n";
//		 
//			if (res == qsat::Status::TRUE) {
//				std::cout << "+++ SAT +++\n";
//			}
//			else if (res == qsat::Status::UNDEFINED) {
//				std::cout << "*** UNDET ***\n";
//			}
//			else {
//				std::cout << "--- UNSAT ---\n";
//			}
//			
//			if (argc == 3) {
//				std::ofstream os(argv[2]);
//				s.dump(os);
//			}
//			std::cout << "==================================================\n";
//		}
//	);
//
//	readcnf.name("readcnf");
//	readcnf_bid.name("readcnf breakid");
//	break_sym.name("break symm");
//	solve_bid.name("solve breakid");
//
//	readcnf.precede(break_sym);
//	readcnf_bid.precede(break_sym);
//	break_sym.precede(solve_bid);
//	executor.run(taskflow).wait();
//	
//	return 0;
//}
//
//

int main(int argc, char* argv[]) {
	if (argc < 2) {
    std::cerr << "Usage: ./QSat cnf_file [optional: out_file]\n";
    std::exit(EXIT_FAILURE);
  }

	qsat::Solver s;
	s.read_dimacs(argv[1]);
	

	auto start_t = std::chrono::steady_clock::now(); 
	qsat::Status res = s.solve();
	auto end_t = std::chrono::steady_clock::now(); 
	std::cout << "================ QSat Statisitics ================\n";
	std::cout << "run time: " 
						<< std::chrono::duration_cast<std::chrono::milliseconds>(
								end_t - start_t
							 ).count() / 1000.0
						<< " s\n";
	std::cout << "num variables:\t" << s.num_variables() << "\n";
	std::cout << "num clauses:\t" << s.num_orig_clauses << "\n";
	std::cout << "restarts:\t" << s.starts << "\n";
	std::cout << "conflicts:\t" << s.conflicts << "\n";
	std::cout << "propagations:\t" << s.propagations << "\n";
 
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

