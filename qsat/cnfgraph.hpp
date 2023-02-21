#include "metis.h"
#include <iostream>
#include <fstream>

namespace qsat {
  void dimacs_graph2csr(const std::string& g_file, 
                        std::vector<idx_t>& out_xadj,
                        std::vector<idx_t>& out_adjncy,
                        idx_t& n_verts,
                        idx_t& n_edges) {
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

    // Allocate vectors for the CSR format
    out_xadj.resize(n_verts + 1, 0);
    out_adjncy.reserve(2 * n_edges);
    std::vector<std::vector<idx_t>> adj_list(n_verts);


    // Read the edges and build adjacency list 
    while (std::getline(dimacsFile, line)) {
      if (line[0] == 'e') {
          // Read an edge (undirected)
          int src, dst;
          std::sscanf(line.c_str(), "e %d %d", &src, &dst);
          adj_list[src-1].push_back(static_cast<idx_t>(dst-1));
          adj_list[dst-1].push_back(static_cast<idx_t>(src-1));
          out_xadj[src]++;
          out_xadj[dst]++;
      }
    }

    for (auto& l : adj_list) {
      for (auto& n : l) {
        out_adjncy.emplace_back(n);
      }
    }

    // Compute the prefix sum of out_xadj to obtain the row pointers
    for (size_t i = 1; i < out_xadj.size(); i++) {
        out_xadj[i] += out_xadj[i-1];
    }

    // Print the CSR arrays
    /*
    std::cout << "xadj: ";
    for (auto& x : out_xadj) {
      std::cout << x << " ";
    } 
    std::cout << "\n";
    
    std::cout << "adj: ";
    for (auto& a : out_adjncy) {
      std::cout << a << " ";
    } 
    std::cout << "\n";
    */
  }







}
