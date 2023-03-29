#ifndef DATATYPES_H
#define DATATYPES_H

#include <cstddef>

namespace qsat {

// primitive types
typedef const char* arg_t;
typedef unsigned char Byte;
typedef Byte* addr_t;
typedef signed char CL_ST;
typedef signed char CNF_ST;
typedef signed char LIT_ST;
typedef unsigned int uint32;
typedef signed long long int int64;
typedef unsigned long long int uint64;
typedef size_t C_REF;
typedef void* G_REF;


} // end of namespace ------------------------------

#endif
