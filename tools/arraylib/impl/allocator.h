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
	// Two large arrays of the same size, the voltages and the currents, can end up at a distance that
	// slows the sweep down badly: on one thread of an Apple M5 under macOS 27, models whose field arrays
	// are 2-4 MiB each (about 0.175-0.35 Mcells, 12 B per cell and array) ran 3.5-3.6x slower than bigger
	// or smaller ones. libmalloc places two such blocks exactly 4 MiB + 16 KiB apart; moving one array
	// by 512 B or more (16 or 32 KiB included) removes the slowdown. The hardware mechanism was not
	// identified. Offsetting each large allocation by a different amount avoids such distances whatever
	// the allocator does. The base pointer is kept in the word before the returned block.
	// OPENEMS_NO_STAGGER=1 turns it off (an empty value or 0 is the same as unset).
	static const size_t STAGGER_MIN_BYTES = 256*1024;
	static const size_t STAGGER_SPAN = 16384;   // offsets from 256 B to just under 16 KiB
	static const size_t STAGGER_STEP = 2560;    // consecutive large arrays move 2560 B against each other (mod the span)
	inline bool StaggerApplies(size_t bytes, size_t alignment)
	{
		static const bool disabled = []() {
			const char* v = std::getenv("OPENEMS_NO_STAGGER");
			return (v!=NULL) && (*v!='\0') && (std::strcmp(v, "0")!=0);
		}();
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
