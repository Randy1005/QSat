#ifndef __LOGGING_
#define __LOGGING_

#include <cstdio>
#include <cstring>
#include "color.hpp"



#define QLOGEN(FORMAT, ...) \
  do { \
     SETCOLOR(CERROR, stderr); \
     fprintf(stderr, "ERROR - "); \
     fprintf(stderr, FORMAT, ## __VA_ARGS__); \
     putc('\n', stderr); \
     SETCOLOR(CNORMAL, stderr); \
  } while (0)












#endif
