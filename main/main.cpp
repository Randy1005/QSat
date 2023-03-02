#include <iostream>
#include <chrono>
#include <qsat/qsat.hpp>

int main(int argc, char* argv[]) {

	if (argc < 2) {
    std::cerr << "Usage: ./QSat cnf_file [optional: out_file]\n";
    std::exit(EXIT_FAILURE);
  }

	tf::Taskflow taskflow;
	tf::Taskflow sycltf;
	sycl::queue queue;	
	tf::Executor executor;

	


	qsat::Solver s0;
	qsat::Solver s1;
	
	auto [readcnf, 
			  readcnf_bid, 
	      break_sym,
				solve,
				solve_bid] = 
	taskflow.emplace(
		[&]() {
			s0.read_dimacs(argv[1]);
		},
		[&]() {
			s1.read_dimacs_bid(argv[1]);
			s1.read_dimacs(argv[1]);
		},
		[&]() {
			s1.build_graph();
			s1.breakid.detect_subgroups();
			s1.breakid.break_symm();
			if (s1.bid_verbosity) {
				s1.breakid.print_symm_break_stats();
			}
			std::cout << "Num sym breaking clauses: " << 
									 s1.breakid.get_num_break_cls() << "\n";
			s1.add_symm_brk_cls();
			std::cout << "Num total clauses: " <<
									 s1.num_clauses() << "\n";
						
		},
		[&]() {
			auto start_t = std::chrono::steady_clock::now(); 
			qsat::Status res = s0.solve();
			auto end_t = std::chrono::steady_clock::now(); 
			std::cout << "================ QSat Statisitics ================\n";
			std::cout << "run time: " 
								<< std::chrono::duration_cast<std::chrono::milliseconds>(
										end_t - start_t
									 ).count() / 1000.0
								<< " s\n";
			std::cout << "num variables:\t" << s0.num_variables() << "\n";
			std::cout << "num clauses:\t" << s0.num_orig_clauses << "\n";
			std::cout << "restarts:\t" << s0.starts << "\n";
			std::cout << "conflicts:\t" << s0.conflicts << "\n";
			std::cout << "propagations:\t" << s0.propagations << "\n";
		 
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
				s0.dump(os);
			}
			std::cout << "==================================================\n";
		},
		[&]() {
			auto start_t = std::chrono::steady_clock::now(); 
			qsat::Status res = s1.solve();
			auto end_t = std::chrono::steady_clock::now(); 
			std::cout << "================ QSat Statisitics ================\n";
			std::cout << "run time: " 
								<< std::chrono::duration_cast<std::chrono::milliseconds>(
										end_t - start_t
									 ).count() / 1000.0
								<< " s\n";
			std::cout << "num variables:\t" << s1.num_variables() << "\n";
			std::cout << "num clauses:\t" << s1.num_orig_clauses << "\n";
			std::cout << "restarts:\t" << s1.starts << "\n";
			std::cout << "conflicts:\t" << s1.conflicts << "\n";
			std::cout << "propagations:\t" << s1.propagations << "\n";
		 
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
				s1.dump(os);
			}
			std::cout << "==================================================\n";
			
		}

	);

	solve.name("solve");
	readcnf.name("readcnf");
	readcnf_bid.name("readcnf breakid");
	break_sym.name("break symm");
	solve_bid.name("solve breakid");

	readcnf.precede(solve);
	readcnf_bid.precede(break_sym);
	break_sym.precede(solve_bid);

	// taskflow.dump(std::cout);
	executor.run(taskflow).wait();

	
	return 0;
}

/*
int main(int argc, char* argv[]) {

  qsat::GraphManager gm;
  tf::Taskflow taskflow; 
  tf::Executor executor;

	gm.dimacs_graph2csr(argv[1]);	
  // initialize number of partitions
  gm.n_parts = 2; 
 
  int errorCode = METIS_PartGraphKway(&(gm.n_verts), &(gm.n_weights), 
                                      gm.xadj.data(), gm.adjncy.data(),
                                      NULL, NULL, NULL, &(gm.n_parts), 
                                      NULL, NULL, NULL, 
                                      &(gm.objval), gm.partitions.data());

  // Print out the partitioning result
  if (errorCode == METIS_OK) {
      std::cout << "Partitioning successful!\n";
  } else {
      std::cout << "Partitioning failed with error code " << errorCode << "\n";
  }

  gm.build_part2verts();
  for (size_t i = 0; i < gm.n_parts; i++) {
    taskflow.emplace([&gm, i]() {
      gm.construct_bliss_graph(i);
    });
  }

  executor.run(taskflow);

  return 0;
}
*/
