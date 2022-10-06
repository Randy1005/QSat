#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>

// TODO: add extreme cases
// + heap with one element
// + randomized elements (sort them to verify)


// Unittest: heap functionaliy
TEST_CASE("Order Heap" * doctest::timeout(300)) {
  std::vector<double> activities{20.0, 14.2, 4.5, 7.5, 8.4, 17.5, 0.3};
  qsat::Heap heap{qsat::VarOrderLt(activities)};
   
  REQUIRE(heap.empty());

  heap.insert(0);
  heap.insert(6);
  heap.insert(1);
  heap.insert(2);
  
  REQUIRE(heap[0] == 0);
  REQUIRE(heap[1] == 2);
  REQUIRE(heap[2] == 1); 
  REQUIRE(heap[3] == 6);

  heap.insert(4);
  
  REQUIRE(heap[0] == 0);
  REQUIRE(heap[1] == 4);
  REQUIRE(heap[2] == 1); 
  REQUIRE(heap[3] == 6);
  REQUIRE(heap[4] == 2);

  heap.insert(3);
  heap.insert(5);

  
  // heap {0, 4, 5, 6, 2, 3, 1}
  REQUIRE(heap[0] == 0);
  REQUIRE(heap[1] == 4);
  REQUIRE(heap[2] == 5); 
  REQUIRE(heap[3] == 6);
  REQUIRE(heap[4] == 2);
  REQUIRE(heap[5] == 3);
  REQUIRE(heap[6] == 1);
  REQUIRE(heap.size() == 7);

  // pop max = var 0
  int max = heap.remove_max();
  REQUIRE(max == 0);


  // heap {5, 4, 1, 6, 2, 3}
  REQUIRE(heap[0] == 5);
  REQUIRE(heap[1] == 4);
  REQUIRE(heap[2] == 1);
  REQUIRE(heap[3] == 6);
  REQUIRE(heap[4] == 2);
  REQUIRE(heap[5] == 3);
  REQUIRE(heap.size() == 6); 


  // modify var 2's activity to be 9.6
  // and invoke decrease on var 2
  activities[2] = 9.6;
  heap.decrease(2);

  // heap becomes {5, 2, 1, 6, 4, 3}
  // the right subtree (w.r.t. root)
  // does not get percolated
  REQUIRE(heap[0] == 5);
  REQUIRE(heap[1] == 2);
  REQUIRE(heap[2] == 1);
  REQUIRE(heap[3] == 6);
  REQUIRE(heap[4] == 4);
  REQUIRE(heap[5] == 3);

  // modify var 5's activity to be 0.2
  // and invoke increase on var 5
  activities[5] = 0.2;
  heap.increase(5);

  // heap becomes {1, 2, 3, 6, 4, 5}
  // left subtree (w.r.t. root)
  // does not get percolated
  REQUIRE(heap[0] == 1);
  REQUIRE(heap[1] == 2);
  REQUIRE(heap[2] == 3);
  REQUIRE(heap[3] == 6);
  REQUIRE(heap[4] == 4);
  REQUIRE(heap[5] == 5);
}


