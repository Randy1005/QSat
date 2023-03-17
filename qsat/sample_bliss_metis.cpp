
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
