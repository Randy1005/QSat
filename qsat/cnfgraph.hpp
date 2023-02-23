#include <iostream>
#include <fstream>
#include "metis/metis.h"
#include "bliss/graph.hh"
#include "bliss/utils.hh"

namespace qsat {
struct GraphManager {
  GraphManager() = default;

  /* 
   * @brief dimacs_graph2csr
   * converts the dimacs graph format to csr format
   * which can be processed by the METIS library
   */
  void dimacs_graph2csr(const std::string& g_file);
  
  /* @brief construct_bliss_graph
   * constructs a graph with bliss's internal
   * graph structure
   */
  bliss::Graph construct_bliss_graph(const size_t part_id);

  void build_part2verts();

  std::vector<idx_t> xadj, adjncy;
  idx_t n_verts, n_edges;

  std::vector<std::vector<idx_t>> adj_list;

  idx_t n_weights  = 1;
  idx_t n_parts;

  idx_t objval;
  std::vector<idx_t> partitions;
  std::vector<idx_t> vwgts;

  // the mapping from partitions to vertices
  std::vector<std::vector<idx_t>> part2verts;

};

}

