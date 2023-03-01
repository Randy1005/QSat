#include <iostream>
#include <chrono>
#include <qsat/qsat.hpp>

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
  std::cout << "run time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(
								end_t - start_t
							 ).count() / 1000.0
            << " s\n";
	std::cout << "================ QSat Statisitics ================\n";
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
