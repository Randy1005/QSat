#ifndef __STATS_
#define __STATS_

#include "datatypes.hpp"
#include <cassert>
#include <cstring>



namespace qsat {


struct Stats {


  struct {
    uint64 original, learnt;
  } literals; 



};



}




#endif
