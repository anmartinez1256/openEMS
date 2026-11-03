#ifndef DENORMAL_H
#define DENORMAL_H

#include <boost/predef.h>

#if BOOST_ARCH_X86
#include <xmmintrin.h>
#endif
#if defined(__aarch64__)
#include <cstdint>
#include "useful.h"
#endif

// Disable denormal (subnormal) floating point numbers. These exceedingly
// small numbers may create a substantial overhead depending on the CPU
// (microcode assists are required os x86).
//
// Flushing to zero also discards the lowest bits of a decaying field, so two
// builds can only be compared bit by bit with it turned off. Configure with
// -DENABLE_FLUSH_TO_ZERO=OFF to do that; it is slow and meant for debugging
// only. Note the macro is the negative, so that a build system which does not
// know about it at all still gets the fast default.
//
// TODO: Only implemented on x86 (arm64 flushes only on request, OPENEMS_FTZ
// below). Do other CPUs like POWER, ARM have denormal overheads? If so,
// implement them too.

namespace Denormal
{
	inline void Disable();

	//! The calling thread's flush-to-zero state before Disable(), and putting it back: Engine_sse saves it
	//! on the thread that creates the engine and restores it when the engine is destroyed there. Only
	//! arm64 with OPENEMS_FTZ saves and restores anything; elsewhere both do nothing, so on x86 the
	//! creating thread keeps MXCSR FZ|DAZ after the run, as in stock.
	inline unsigned long long Save();
	inline void Restore(unsigned long long state);
};

#if defined(__aarch64__)
namespace Denormal
{
	// Apple cores have no subnormal penalty, so the default keeps them (bit-identical to upstream).
	// OPENEMS_FTZ=1 sets FPCR.FZ wherever Disable() runs: on the thread that creates an engine of the
	// Engine_sse family (sse, sse-compressed, multithreaded, cylindrical) and in the multithreaded
	// workers. These engines then match a GPU that flushes subnormals; the basic engine never flushes.
	inline bool FlushRequested()
	{
		static const bool on = EnvFlag("OPENEMS_FTZ");
		return on;
	}
}
#endif

inline void Denormal::Disable()
{
#if BOOST_ARCH_X86 && !defined(OPENEMS_NO_FLUSH_TO_ZERO)
	// read the old MXCSR setting
	unsigned int oldMXCSR = _mm_getcsr();

	// set DAZ and FZ bits (flush to zero)
	unsigned int newMXCSR = oldMXCSR | 0x8040;

	// write the new MXCSR setting to the MXCSR
	_mm_setcsr( newMXCSR );
#elif defined(__aarch64__)
	if (FlushRequested())
	{
		uint64_t fpcr;
		__asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
		fpcr |= (1ull<<24);
		__asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
	}
#endif
}

inline unsigned long long Denormal::Save()
{
#if defined(__aarch64__)
	if (FlushRequested())
	{
		uint64_t fpcr;
		__asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
		return fpcr;
	}
#endif
	return 0;
}

inline void Denormal::Restore(unsigned long long state)
{
#if defined(__aarch64__)
	if (FlushRequested())
	{
		// only the FZ bit that Disable() sets; the other FPCR bits stay as they are now
		uint64_t fpcr;
		__asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
		fpcr = (fpcr & ~(1ull<<24)) | (state & (1ull<<24));
		__asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
	}
#else
	(void)state;
#endif
}

#endif // DENORMAL_H
