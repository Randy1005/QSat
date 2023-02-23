#include "cnfgraph.hpp"

namespace qsat {


void GraphManager::dimacs_graph2csr(const std::string& g_file) {
    
  std::ifstream dimacsFile(g_file);
  if (!dimacsFile) {
    std::cerr << "failed to open graph dimacs file\n";
  }


  std::string line;
  while (std::getline(dimacsFile, line)) {
      // Ignore comments and empty lines
      if (line.empty() || line[0] == 'c') {
          continue;
      }

      // Read the number of nodes and edges
      if (line[0] == 'p') {
          std::sscanf(line.c_str(), "p edge %d %d", &n_verts, &n_edges);
          break;
      }
  }

  // Resize vectors for the CSR format
  xadj.resize(n_verts + 1, 0);
  adjncy.reserve(2 * n_edges);
  adj_list.resize(n_verts);
  partitions.resize(n_verts);

  // weights of vertices
  vwgts.resize(n_verts * n_weights, 0);

  // Read the edges and build adjacency list 
  while (std::getline(dimacsFile, line)) {
    if (line[0] == 'e') {
        // Read an edge (undirected)
        int src, dst;
        std::sscanf(line.c_str(), "e %d %d", &src, &dst);
        adj_list[src-1].emplace_back(static_cast<idx_t>(dst-1));
        adj_list[dst-1].emplace_back(static_cast<idx_t>(src-1));
        xadj[src]++;
        xadj[dst]++;
    }
  }
  
  // iterate through adjacency list
  // and build the csr adjacency array
  for (const auto& l : adj_list) {
    for (const auto& n : l) {
      adjncy.emplace_back(n);
    }
  }

  // Compute the prefix sum of out_xadj to obtain the row pointers (xadj)
  for (size_t i = 1; i < xadj.size(); i++) {
    xadj[i] += xadj[i-1];
  }

}

void GraphManager::build_part2verts() {
  part2verts.resize(n_parts);
  for (size_t i = 0; i < n_verts; i++) {
    part2verts[partitions[i]].emplace_back(i);
  }
}

bliss::Graph GraphManager::construct_bliss_graph(const size_t part_id) {
  bliss::Graph g;
  const auto& verts = part2verts[part_id];
  
  // mapping that maps global vertices to partition (local) vertices
  std::unordered_map<size_t, size_t> gv2lv;


  // add the vertices to the graph
  // first, or else bliss would throw
  // an out of bound vertex exception
  // 
  // TODO: for now, I'm adding all
  // vertices with the default color
  // not sure how coloring would affect
  // symmetry detection
  for (size_t i = 0; i < verts.size(); i++) {
    gv2lv.insert( {verts[i], i} );
    g.add_vertex();
  }

  // NOTE:
  // bliss graph vertices are labeled starting
  // from 0, but the vertices we stored are labeled w.r.t
  // the orignal larger graph
  // e.g.
  // say for this subgraph there are 2 vertices: 0, 1
  // but these 2 vertices in the 
  // original graph might be
  // 32, 90 
  // then bliss would throw an out of bound exception
  //
  // NOTE:
  // a temporary solution:
  // map the original label to 0, 1, 2 ... etc.
  // and now the orders in part2verts[N] matter

  size_t nof_edges = 0;

  for (const auto v : verts) {
    // get the adjacent vertices of v
    // and add an edge to each of these vertices 
    // NOTE:
    // this would introduce duplicate edges eventually
    // which is not encouraged by bliss
    // TODO: figure out some way to add only distinct edges
    const auto& adj_vs = adj_list[v];
    for (const auto& a : adj_vs) {
      // in this subgraph
      // we probably only want edges connecting the vertices
      // in the same partition
      if (partitions[v] == partitions[a]) {
        nof_edges++;
        g.add_edge(static_cast<unsigned int>(gv2lv[v]), 
                   static_cast<unsigned int>(gv2lv[a]));
      }
    }
  }
  
  // g.write_dot("graph.dot");

  // find_automorphism results
  bliss::Stats stats;
  auto report_aut = [&](unsigned int n, const unsigned int* aut) {
    std::cout << "Generators: ";
    bliss::print_permutation(stdout, n, aut, 1);
    std::cout << "\n";
  };


  g.find_automorphisms(stats, report_aut);
  // std::cout << "n_generators = " << stats.get_nof_generators() << "\n";

  return g;
}








}

