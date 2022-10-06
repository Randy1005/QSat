#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <cassert>

namespace qsat {

struct VarOrderLt {
  const std::vector<double>& activities;
  VarOrderLt(const std::vector<double>& act)
    : activities(act)
  {
  }

  bool operator()(int a, int b) const {
    return activities[a] > activities[b];
  }
};

/**
 * @brief Heap class for quick selecting
 * the variable with maximum activity to decide next
 */
class Heap {
  // heap of variables
  std::vector<int> heap;
  
  // each variables' index in the heap
  std::vector<int> indices;
  
  // "less than" comparator
  VarOrderLt lt;

  /**
   * @brief index traversal methods
   * 0 -> root
   * for node n:
   * left child = 2*n+1
   * right child = (n+1)*2
   * parent = (n-1)/2
   */
  int left(int i) {
    return i * 2 + 1;
  }

  int right(int i) {
    return (i + 1) * 2;
  }

  int parent(int i) {
    return (i - 1) >> 1;
  }

  /**
   * @brief percolate up method: 
   * heapifies the tree "upwards" starting from the specified node
   * @param i the node index as the starting point to heapify upwards
   */
  void percolate_up(int i) {
    // var at heap index i
    int var = heap[i];
    
    // index of i's parent
    int p = parent(i);

    while (i != 0 && lt(var, heap[p])) {
      // parent now becomes left child
      heap[i] = heap[p];

      // update indices
      indices[heap[p]] = i;

      // traverse to i's parent
      // p should become parent of p
      i = p;
      p = parent(p);
    }

    // do the final swap
    heap[i] = var;
    indices[var] = i;
  }

  /**
   * @brief percolate down method
   * heapifies the tree "downwards" starting from the specified node
   * @param i the node index as the starting point to heapify downwards 
   */
  void percolate_down(int i) {
    int var = heap[i];
    while (left(i) < static_cast<int>(heap.size())) {
      // pick the child with larger activity
      int child = right(i) < static_cast<int>(heap.size()) && lt(heap[right(i)], heap[left(i)])
        ? right(i) : left(i);
      // if the chosen child has activity less than current var
      // then break
      if (!lt(heap[child], var)) {
        break;
      }
      
      // parent now becomes child
      heap[i] = heap[child];
      // update indices
      indices[heap[i]] = i;
      i = child; 
    }

    // do the final swap
    heap[i] = var;
    indices[var] = i;
  }

public:
  Heap(const VarOrderLt& c) : 
    lt(c)
  {
  }
  
  size_t size() const {
    return heap.size();
  }

  bool empty() const {
    return heap.size() == 0;
  }

  bool in_heap(int v) {
    // indices is basically a map<var, index>
    // to see if a var exists in indices
    // just check indices[var]
    if (empty()) {
      return false;
    }
    
    return indices[v] >= 0;
  }

  int operator[](int i) {
    assert(i < static_cast<int>(heap.size()));
    return heap[i];
  }

  /**
   * @brief decrease:
   * given a variable id, percolate this variable upwards
   * if it has larger activity than its parent
   * @param v the variable id to percolate
   */
  void decrease(int v) {
    assert(in_heap(v));
    percolate_up(indices[v]);
  }

  /**
   * @brief increase:
   * given a variable id, percolate this variable downwards
   * if it has smaller activity than its child
   * @param v the variable id to percolate
   */
  void increase(int v) {
    assert(in_heap(v));
    percolate_down(indices[v]);
  }

  /**
   * @brief insert:
   * insert a variable at the bottom of heap
   * and heapify upwards starting from that
   * newly inserted node
   */
  void insert(int v) {
    if (v + 1 > static_cast<int>(indices.size())) {
      indices.resize(v + 1);
    }
    indices[v] = -1;

    // pre-condition:
    // this variable must NOT exist in heap
    assert(!in_heap(v));
    heap.push_back(v);
    
    // place this variable at the bottom of heap
    indices[v] = heap.size() - 1;
    
    // and heapify upwards
    percolate_up(indices[v]);
  }

  /**
   * @brief remove_max_act:
   * get the root node (max activity)
   * replace the root node with the last node
   * and heapify downwards
   */
  int remove_max() {
    int var = heap[0];
    // replace root with last element in heap
    heap[0] = heap.back();
    indices[heap[0]] = 0;
    indices[var] = -1;
    // pop the last element
    heap.pop_back();
    
    if (heap.size() > 0) {
      percolate_down(0);
    }

    return var;
  }






};

}
