
#ifndef __GL_DEFS_
#define __GL_DEFS_

#include <iostream>
#include <algorithm>
#include <cstring>
#include <locale>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <climits>
#include <cstdlib>
#include <csignal>
#include "logging.hpp"
#include "datatypes.hpp"
#include "constants.hpp"

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cpuid.h>
#elif defined(__CYGWIN__)
#include </usr/include/sys/resource.h>
#include </usr/include/sys/mman.h>
#include </usr/include/sys/sysinfo.h>
#include </usr/include/sys/unistd.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <intrin.h>
#include <Winnt.h>
#include <io.h>
#endif
#undef ERROR
#undef hyper 
#undef SET_BOUNDS


namespace qsat {
struct Occur {
  uint32 ps, ns;  
};

struct CNFInfo {
  uint32 org_vars, max_var, n_dual_vars;
  uint32 n_org_cls, n_clauses, n_literals;

  CNFInfo() {
    // reset struct
    assert(this);
    memset(this, 0, sizeof(*this));
  }

};

extern CNFInfo cnf_info;







inline double ratio(const double& x, const double& y) {
	return y ? (x/y) : 0;
}
inline uint64 ratio(const uint64& x, const uint64& y) {
	return y ? (x/y) : 0;
}








} // end of namespace ---------------------------


#endif

