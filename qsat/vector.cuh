#ifndef __CU_VECTOR_
#define __CU_VECTOR_

#include "definitions.cuh"
#include "constants.hpp"


namespace qsat {


template<typename T>
class CuVec {
public:
	_QSAT_H_D_ CuVec():
		_mem(nullptr),
		_sz(0),
		_cap(0)
	{}

	_QSAT_H_D_ ~CuVec() {
	
	
	}

	_QSAT_H_D_ void	alloc(T* head) { 
		_mem = head; 
	}

	_QSAT_H_D_ void	alloc(T* head, const uint32& cap) { 
		_mem = head; 
		this->_cap = cap; 
	}

	_QSAT_H_D_ void	alloc(const uint32& cap) { 
		_mem = (T*)(this + 1); 
		this->_cap = cap; 
	}

	_QSAT_IN_D_ T* jump(const uint32&);
	_QSAT_IN_D_ void insert(const T&);
	_QSAT_IN_D_ void push(const T&);
	
	_QSAT_H_D_ void	_pop() { 
		_sz--; 
	}

	_QSAT_H_D_ void	_shrink(const uint32& n) { 
		_sz -= n; 
	}

	_QSAT_H_D_ void	_push(const T& val) { 
		_mem[_sz++] = val; 
	}

	_QSAT_H_D_ CuVec<T>&  operator=(CuVec<T>& rhs) { 
		return *this; 
	}

	_QSAT_H_D_ const T& operator[](const uint32& idx) const { 
		assert(idx < _sz); 
		return _mem[idx]; 
	}

	_QSAT_H_D_ T& operator[](const uint32& idx)	{
		assert(idx < _sz); 
		return _mem[idx]; 
	}

	_QSAT_H_D_ T& at(const uint32& idx) { 
		return _mem[idx]; 
	}

	_QSAT_H_D_ const T& at(const uint32& idx) const { 
		return _mem[idx]; 
	}

	_QSAT_H_D_ operator T* () { 
		return _mem; 
	}

	_QSAT_H_D_ T* data() { 
		return _mem; 
	}

	_QSAT_H_D_ T* end() { 
		return _mem + _sz; 
	}

	_QSAT_H_D_ T& back() { 
		assert(_sz); 
		return _mem[_sz - 1]; 
	}

	_QSAT_H_D_ uint32* sizeptr() {
		return (uint32*) &_sz; 
	}

	_QSAT_H_D_ uint32	size() const { 
		return _sz; 
	}

	_QSAT_H_D_ bool	empty() const { 
		return !_sz; 
	}

	_QSAT_H_D_ uint32	capacity() const { 
		return _cap; 
	}

	_QSAT_H_D_ void	resize(const uint32& n) { 
		assert(n <= _cap); 
		_sz = n; 
	}

	_QSAT_H_D_ void	share_to(T* dest) {
		assert(_sz && _mem);
		T* s = _mem, * e = s + _sz;
		while (s != e) *dest++ = *s++;
	}
	_QSAT_H_D_ void	copy_from(T* src) {
		assert(_sz);
		T* d = _mem, * e = d + _sz;
		while (d != e) *d++ = *src++;
	}
	_QSAT_H_D_ void	copy_from(T* src, const uint32& n) {
		assert(n <= _sz);
		T* d = _mem, * e = d + n;
		while (d != e) *d++ = *src++;
	}
	_QSAT_H_D_ void	clear(const bool& free = false) {
		if (free) {
			_mem = nullptr; 
		}			 
		_cap = 0;
		_sz = 0;
	}
	_QSAT_H_D_ void	print(const bool& lit_type = false, 
		const bool& clause_type = false) {
		printf("->(size = %d)[", _sz);
		if (lit_type) {
			for (uint32 i = 0; i < _sz; i++) {
				printf("%2d  ", 
					SIGN(_mem[i]) ? 
					-int(ABS(_mem[i])) : 
					int(ABS(_mem[i]))
				);

				if (clause_type && _mem[i] < 2) { 
					printf("\nc \t\t");
				}
				else if (!clause_type && i && i < _sz - 1 && i % 10 == 0) { 
					printf("\nc \t\t");
				}
			}
		}
		else {
			for (uint32 i = 0; i < _sz; i++) {
				printf("%2lld  ", uint64(_mem[i]));
				if (i && i < _sz - 1 && i % 10 == 0) {
					printf("\nc \t\t");
				}
			}
		}
		printf("]\n");
	}



private:
	T* _mem;
	uint32 _sz, _cap;

	_QSAT_D_ bool _check_atomic_bound(const uint32 idx, const uint32 cap) {
		if (idx >= cap) {
			printf("c ERR: vector atomic returned index [%d] exceeding\ 
				allocated capacity [%d]\n", idx, cap);
			return false;
		}	
		return true;
	}




};




} // end of namespace -------------------------



#endif
