#ifndef __GPU_PRIMITIVE_
#define __GPU_PRIMITIVE_

#include "atomics.cuh"
#include "simptypes.cuh"

namespace qsat {

template<typename T>
_QSAT_IN_D_ void CuVec<T>::insert(const T& val)
{
	const uint32 idx = atomicInc(&_sz, _cap);
	assert(_check_atomic_bound(idx, _cap));
	_mem[idx] = val;
}

template<typename T>
_QSAT_IN_D_ void CuVec<T>::push(const T& val) 
{
	const uint32 idx = atomic_agg_inc(&_sz);
	assert(_check_atomic_bound(idx, _cap));
	_mem[idx] = val;
}

template<typename T>
_QSAT_IN_D_ T* CuVec<T>::jump(const uint32& n) 
{
	const uint32 idx = atomicAdd(&_sz, n);
	assert(_check_atomic_bound(idx, _cap));
	return _mem + idx;
}












}












#endif
