/*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SPIN_BARRIER_H
#define SPIN_BARRIER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

//! Reusable barrier that spins for a short time before it parks the thread.
/*!
  A boost::barrier (mutex + condition variable) costs 5-20 us per wait on 4-10 threads, which
  is a large part of a timestep on small models. Here the waiting threads spin on a generation
  counter for up to SpinTime() (default 50 us, env OPENEMS_BARRIER_SPIN_US) and only then park on
  a condition variable, so a barrier between threads that arrive close together costs < 1 us.
  The last thread to arrive can increment a counter before it releases the others.
  */
class SpinBarrier
{
public:
	explicit SpinBarrier(unsigned int numThreads) : m_numThreads(numThreads), m_count(0), m_generation(0), m_sleepers(0) {}

	//! Wait until all threads have arrived. The last one increments *inc (if given) before releasing the others.
	void wait(volatile unsigned int* inc=NULL)
	{
		const unsigned int gen = m_generation.load(std::memory_order_acquire);
		if (m_count.fetch_add(1, std::memory_order_acq_rel) + 1 == m_numThreads)
		{
			if (inc)
				++(*inc);
			m_count.store(0, std::memory_order_relaxed);
			bool notify;
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_generation.store(gen+1, std::memory_order_release);
				notify = m_sleepers>0;
			}
			if (notify)
				m_cond.notify_all();
			return;
		}

		const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now() + SpinTime();
		for (unsigned int n=1;;++n)
		{
			if (m_generation.load(std::memory_order_acquire) != gen)
				return;
			CpuRelax();
			if ((n%128)==0 && std::chrono::steady_clock::now()>stop)
				break;
		}

		std::unique_lock<std::mutex> lock(m_mutex);
		++m_sleepers;
		while (m_generation.load(std::memory_order_acquire) == gen)
			m_cond.wait(lock);
		--m_sleepers;
	}

	static std::chrono::microseconds SpinTime()
	{
		static const std::chrono::microseconds t(ReadSpinTime());
		return t;
	}

protected:
	static long ReadSpinTime()
	{
		const char* env = getenv("OPENEMS_BARRIER_SPIN_US");
		if (env)
			return atol(env);
		return 50;
	}

	static inline void CpuRelax()
	{
#if defined(__aarch64__) || defined(__arm__)
		__asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
		__builtin_ia32_pause();
#else
		std::this_thread::yield();
#endif
	}

	const unsigned int m_numThreads;
	std::atomic<unsigned int> m_count;
	std::atomic<unsigned int> m_generation;
	unsigned int m_sleepers; //!< guarded by m_mutex
	std::mutex m_mutex;
	std::condition_variable m_cond;
};

#endif // SPIN_BARRIER_H
