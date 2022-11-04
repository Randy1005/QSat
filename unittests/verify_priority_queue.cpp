#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <qsat/qsat.hpp>
#include <queue>
#include <vector>

// TODO: add extreme cases to uniittest
// + heap with one element
// + randomized elements (sort them to verify)

struct VarOrder {
	int var;
	const double activity;

	friend bool operator < (const VarOrder& lhs, const VarOrder& rhs) {
		return lhs.activity < rhs.activity;
	}
};


// Unittest: heap functionaliy
TEST_CASE("Priority Queue" * doctest::timeout(300)) {
  std::vector<double> activities{20.0, 14.2, 234.2, 52.3, 89.2, 4.5, 7.5, 8.4, 17.5, 0.3};
 
	std::priority_queue<VarOrder> pq;
	
	for (int i = 0; i < static_cast<int>(activities.size()); i++) {
	}

}


