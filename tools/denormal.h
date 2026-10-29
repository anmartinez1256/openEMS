#ifndef DENORMAL_H
#define DENORMAL_H

#include <boost/predef.h>

#if BOOST_ARCH_X86
#include <xmmintrin.h>
#endif
#if defined(__aarch64__)
#include <cstdint>
#include <cstdlib>
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
// TODO: Only implemented on x86. Do other CPUs like POWER, ARM have
// denormal overheads? If so, implement them too.

namespace Denormal
{
	inline void Disable();
};

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
	// Apple cores have no subnormal penalty, so the default keeps them (bit-identical to upstream).
	// OPENEMS_FTZ=1 sets FPCR.FZ, which makes the CPU engines match a GPU that flushes subnormals.
	static const bool on = getenv("OPENEMS_FTZ")!=NULL;
	if (on)
	{
		uint64_t fpcr;
		__asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
		fpcr |= (1ull<<24);
		__asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
	}
#endif
}

#endif // DENORMAL_H
