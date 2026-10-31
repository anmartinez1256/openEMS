/*
 * Copyright (C) 2024 Yifeng Li <tomli@tomli.me>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ARRAYLIB_ALLOCATOR_H
#define ARRAYLIB_ALLOCATOR_H

#include <cstddef>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

namespace ArrayLib
{
	template <typename T>
	class SimpleAllocator;

	template <typename T>
	class AlignedAllocator;
};

template <typename T>
class ArrayLib::SimpleAllocator
{
public:
	static T* alloc(size_t numelem)
	{
		T* ptr = new T[numelem]();
		return ptr;
	}

	static void free(T* ptr, size_t /*numelem*/)
	{
		delete[] ptr;
	}
};

#ifdef _WIN32
#include <malloc.h>
#endif

#include <atomic>
#include <cstdlib>

namespace ArrayLib
{
	// Large arrays come page-aligned from the system, so two arrays of the same size (the
	// voltages and the currents) put matching cells into the same cache sets: on Apple M5 a
	// 0.2-0.3 Mcell model then runs 3-4x slower than a bigger or smaller one. Offsetting each
	// large allocation by a different sub-page amount removes the aliasing. The base pointer
	// is kept in the word before the returned block. OPENEMS_NO_STAGGER=1 turns it off.
	static const size_t STAGGER_MIN_BYTES = 256*1024;
	static const size_t STAGGER_SPAN = 16384;   // L1 way size on Apple cores (also covers 4 KB)
	static const size_t STAGGER_STEP = 2560;    // 256 B granularity, cycles through the span
	inline bool StaggerApplies(size_t bytes, size_t alignment)
	{
		static const bool disabled = (std::getenv("OPENEMS_NO_STAGGER") != NULL);
		return !disabled && (bytes >= STAGGER_MIN_BYTES) && (alignment <= 256);
	}
	inline size_t StaggerOffset(size_t bytes, size_t alignment)
	{
		static std::atomic<unsigned> counter{0};
		if (!StaggerApplies(bytes, alignment))
			return 0;
		const unsigned k = counter.fetch_add(1);
		return 256 + (size_t)((k * STAGGER_STEP) % (STAGGER_SPAN - 256));
	}
}

template <typename T>
class ArrayLib::AlignedAllocator
{
public:
	static T* alloc(size_t numelem)
	{
		size_t alignment;
		// User-defined SIMD variables can be larger than the built-in C
		// types, so they need custom alignments to their own size.
		if (sizeof(T) <= sizeof(void*))
			// POSIX says alignment >= sizeof(void*)
			alignment = sizeof(void *);
		else
			// POSIX says alignment must be a power of 2.
			// TODO: replace with std::bit_ceil(sizeof(T)) when upgrading to C++20.
			alignment = 1 << (size_t) std::ceil(std::log2(sizeof(T)));

		const size_t bytes = numelem * sizeof(T);
		const size_t offset = StaggerOffset(bytes, alignment);
		char* base;
#ifdef _WIN32
		base = (char*) _mm_malloc(bytes + offset, alignment);
		if (base == NULL)
		{
			std::cerr << "Failed to allocate aligned memory" << std::endl;
			throw std::bad_alloc();
		}
#else
		int retval = posix_memalign((void**) &base, alignment, bytes + offset);
		if (retval != 0)
		{
			std::cerr << "Failed to allocate aligned memory" << std::endl;
			throw std::bad_alloc();
		}
#endif
		T* buf = (T*) (base + offset);
		if (offset)
			((void**) buf)[-1] = base;
		std::uninitialized_fill(buf, buf + numelem, T());
		return buf;
	}

	static void free(T* ptr, size_t numelem)
	{
		if (ptr)
		{
			for (size_t i = 0; i < numelem; i++)
				(&ptr[i])->~T();
			void* base = ptr;
			const size_t alignment = (sizeof(T) <= sizeof(void*)) ? sizeof(void*) : (size_t)1 << (size_t) std::ceil(std::log2(sizeof(T)));
			if (StaggerApplies(numelem * sizeof(T), alignment))   // staggered allocation: base pointer stored before it
				base = ((void**) ptr)[-1];
#ifdef _WIN32
			_mm_free(base);
#else
			std::free(base);
#endif
		}

	}
};

#endif // ARRAYLIB_ALLOCATOR_H
