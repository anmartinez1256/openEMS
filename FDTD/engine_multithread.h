/*
*	Copyright (C) 2010 Sebastian Held (sebastian.held@gmx.de)
*
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

#ifndef ENGINE_MULTITHREAD_H
#define ENGINE_MULTITHREAD_H

#include "operator_multithread.h"
#include "engine_sse_compressed.h"
#include "tools/spin_barrier.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include <boost/thread.hpp>
#include <boost/fusion/include/list.hpp>
#include <boost/fusion/container/list/list_fwd.hpp>
#include <boost/fusion/include/list_fwd.hpp>

#include "tools/useful.h"
#if defined(_WIN32) && !defined(__GNUC__)
#include <Winsock2.h> // for struct timeval
#else
#include <sys/time.h>
#endif


class Engine_Multithread;

namespace NS_Engine_Multithread
{

class DBG   // debug
{
public:
	DBG() {}
	~DBG() { std::cout << os.str();}
	std::ostringstream& cout() {return os;}
protected:
	std::ostringstream os;
};

class Timer   //debug
{
public:
	Timer() {gettimeofday(&t1,NULL);}
	double elapsed() {gettimeofday(&t2,NULL); return (t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)*1e-6;}
protected:
	timeval t1,t2;
};

class thread
{
public:
	thread( Engine_Multithread* ptr, unsigned int start, unsigned int stop, unsigned int stop_h, unsigned int threadID );
	void operator()();

protected:
	unsigned int m_start, m_stop, m_stop_h, m_threadID;
	Engine_Multithread *m_enginePtr;
};
} // namespace


class Engine_Multithread : public Engine_SSE_Compressed
{
	friend class NS_Engine_Multithread::thread;
	friend class Engine_CylinderMultiGrid;
public:
	static Engine_Multithread* New(const Operator_Multithread* op, unsigned int numThreads = 0);
	virtual ~Engine_Multithread();

	virtual void setNumThreads( unsigned int numThreads );
	virtual void Init();
	virtual void Reset();
	virtual void NextInterval(float curr_speed);

	//! Iterate \a iterTS number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	virtual void DoPreVoltageUpdates()  {throw std::runtime_error("Engine_Multithread::DoPreVoltageUpdates without thread ID is invalid");}
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {throw std::runtime_error("Engine_Multithread::DoPostVoltageUpdates without thread ID is invalid");}
	virtual void DoPostVoltageUpdates(int threadID);
	virtual void Apply2Voltages()       {throw std::runtime_error("Engine_Multithread::Apply2Voltages without thread ID is invalid");}
	virtual void Apply2Voltages(int threadID);

	virtual void DoPreCurrentUpdates()  {throw std::runtime_error("Engine_Multithread::DoPreCurrentUpdates without thread ID is invalid");}
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() {throw std::runtime_error("Engine_Multithread::DoPostCurrentUpdates without thread ID is invalid");}
	virtual void DoPostCurrentUpdates(int threadID);
	virtual void Apply2Current()        {throw std::runtime_error("Engine_Multithread::Apply2Current without thread ID is invalid");}
	virtual void Apply2Current(int threadID);

	//! On an engine thread of a fused sweep with neighbour flags: that thread's timestep (the threads run apart)
	virtual unsigned int GetNumberOfTimesteps();

protected:
	Engine_Multithread(const Operator_Multithread* op);
	void changeNumThreads(unsigned int numThreads);
	const Operator_Multithread* m_Op_MT;
	boost::thread_group *m_thread_group;
	boost::barrier *m_startBarrier, *m_stopBarrier;
	SpinBarrier *m_IterateBarrier;

	//! One extension hook call in a timestep, optionally followed by a barrier
	struct HookStep
	{
		Engine_Extension* ext;
		bool barrier;   //!< wait on m_IterateBarrier after the call
		bool inc_numTS; //!< this is the last barrier of the timestep: it increments numTS
	};
	//! Per hook group (pre/post/apply voltages, pre/post/apply currents): the calls and barriers, see BuildHookPlan()
	std::vector<HookStep> m_HookPlan[6];
	std::vector<Engine_Extension*> m_HookPlanExts; //!< the extensions the plan was built for (the list can change after Init)
	bool m_IncNumTS_AtCurrBarrier; //!< no current hooks run after the current update: its barrier increments numTS
	void BuildHookPlan();
	void RunHooks(int group, int threadID);

	//! Fused sweep: every thread runs the voltage update, the voltage hooks, and the current update with
	//! its hooks x-line by x-line over its own slab, so a line is still in the cache when its currents are
	//! updated (one pass over the fields per timestep instead of two) and a timestep needs two barriers.
	//! It needs UseIJKN() (the IJKN field layout and the packed or geometry coefficient tables) and extensions that
	//! all implement the line hooks (LINE_HOOKS).
	bool m_Fused;
	std::vector<unsigned int> m_Blocks;     //!< x-line blocks [m_Blocks[b], m_Blocks[b+1]) of the fused sweep
	std::vector<unsigned int> m_SlabBlocks; //!< thread t sweeps the blocks [m_SlabBlocks[t], m_SlabBlocks[t+1])
	std::vector<Engine_Extension*> m_LineHooks[6]; //!< per hook group: the extensions to call, in order
	void SetupFused();
	std::string m_FusedMsg; //!< the last fused-sweep line printed (verbose>0), see ReportFused()

	//! The tiled sweep (OPENEMS_TBLOCK=k, with the fused sweep's conditions): temporal blocking with ONE tile of
	//! x-lines for all threads, as upstream master's cpu engine does it. A block of k timesteps on a tile [a, b):
	//! the core runs the voltages of step t on [a+t, b-t) and its currents on [a+t, b-t-1) for t = 0 .. k-1 (no
	//! shrinking at a domain edge), then the wedge at every tile boundary c the voltages on [c-t, c+t) and the
	//! currents on [c-t-1, c+t). Each range is a task: every thread updates its y lines of it, then all meet at
	//! the barrier. Between two updates a hook stage runs the Post and Apply hooks of the task before and the Pre
	//! hooks of the task after: the x-lines of both are cut once into one share per thread, so all hooks of an
	//! x-line (and of a coupled pair) run on one thread in the order of a one-thread sweep, and the hooks of other
	//! lines are independent of them (the line hook contract). Every cell gets the operands of the unfused update
	//! (bit-identical), and the fields of a tile stay in the cache for k timesteps.
	struct TileTask
	{
		bool volt;               //!< voltages (and their hooks), or currents
		unsigned int x0, x1;     //!< the x-lines [x0, x1)
		unsigned int ts;         //!< the timestep within the block
	};
	bool m_Tiled;
	unsigned int m_TileK, m_TileW;
	std::vector<std::vector<TileTask> > m_TilePlan; //!< [depth]: the tasks of a block of depth timesteps, 1 .. m_TileK
	std::string m_TiledMsg;
	//! The threads' shares in the tiled sweep, cumulative in 1/65536 ([0] = 0, [threads] = 65536): the y lines of
	//! an update and the x-lines of a hook stage are cut at them. Equal at first; BalanceTiled() moves them to the
	//! measured speed of every thread between two IterateTS() calls (a thread on an efficiency core is slower, and
	//! every task ends at a barrier, so the slowest share sets the pace). OPENEMS_BALANCE=0 keeps them equal.
	std::vector<unsigned int> m_TileCum;
	bool m_TileHooksAll; //!< the hook stages on all threads; OPENEMS_TBLOCK_HOOKS=0: on thread 0 alone
	void SetupTiled(const std::vector<unsigned int>& coupled);
	void TiledRun(int threadID, unsigned int iterTS);
	//! The x-lines of one thread in a hook stage over the lines [p0, p1) and [q0, q1) (either may be empty):
	//! [s[0], s[1]) and [s[2], s[3]), the second one only when the two ranges have a gap between them
	void TileHookShare(int threadID, unsigned int p0, unsigned int p1, unsigned int q0, unsigned int q1, unsigned int s[4]) const;
	void BalanceTiled();
	void ReportFused(const std::string& msg);
	void FusedTimestep(int threadID);

	//! Neighbour flags (the fused sweep when every extension is TS_LOCAL; OPENEMS_FLAGS=0 keeps the barriers): a thread
	//! waits for its neighbours only, instead of the two barriers of every timestep. A timestep count one thread
	//! publishes and its neighbour waits for (spins for m_FlagSpin, then sleeps); 256 B apart, so that two of them
	//! never share a cache line (C++11: new does not align).
	struct StepFlag
	{
		std::atomic<unsigned int> value;    //!< timesteps done in this IterateTS() call
		std::atomic<unsigned int> sleepers; //!< the neighbour sleeps on cond
		std::mutex mutex;
		std::condition_variable cond;
		char pad[256-2*sizeof(std::atomic<unsigned int>)-sizeof(std::mutex)-sizeof(std::condition_variable)];
		void Publish(unsigned int v);
		//! spins for \a spin, then sleeps until the count is at least \a v; returns the time it waited (0 without a wait)
		std::chrono::steady_clock::duration WaitFor(unsigned int v, std::chrono::microseconds spin);
	};
	bool m_UseFlags;
	std::chrono::microseconds m_FlagSpin; //!< spin time of the flag waits, from the thread count (SetupFused, FlagSpinTime)
	StepFlag* m_SweepDone;    //!< per thread: timesteps of which its sweep (all voltages of its slab) is done
	StepFlag* m_BoundaryDone; //!< per thread: timesteps of which the currents of the line before its slab are done
	void FusedTimestepFlags(int threadID, unsigned int step);
	//! time and x-lines of each thread's sweep, to size the slabs (threads on efficiency cores are slower, PML lines cost more)
	struct FusedStat
	{
		double busy;
		double lines;
		// OPENEMS_FUSED_STAT=1: totals since the slabs were set up (the sweep, the waits at the two barriers, the
		// currents of the line before the slab), printed when the threads change or stop (PrintFusedStat)
		double sweep, wait1, bound, wait2, steps;
		char pad[64-7*sizeof(double)]; // 64 B each (C++11: new does not align to 64; the Apple M5's cache line is 128 B)
	};
	FusedStat* m_FusedStat;
	unsigned int m_FusedStatThreads; //!< entries of m_FusedStat
	void PrintFusedStat() const;
	bool m_Balance;
	std::vector<double> m_LineCost; //!< smoothed time per x-line (s), from the threads' measurements
	//! Between two IterateTS() calls: move the slab boundaries so that every thread gets the same estimated time
	void BalanceSlabs();
	void RunLineHooks(int group, unsigned int x0, unsigned int x1);
	//! hooks and update of the currents of the x-lines [x0, x1)
	void FusedCurrents(unsigned int x0, unsigned int x1);

	//! x-lines of one thread for the voltage or current update; idle threads take lines from the others
	struct WorkRange
	{
		std::atomic<int> next; //!< next x-line to update
		int start, stop;       //!< first and last x-line of this range
		char pad[64-sizeof(std::atomic<int>)-2*sizeof(int)]; // 64 B per range: one per cache line on 64-B-line CPUs, two per line on the Apple M5 (128 B)
	};
	WorkRange* m_VoltRange;
	WorkRange* m_CurrRange;
	//! Update all lines of the own range in order, then help with the ranges of the other threads.
	void UpdateStealing(WorkRange* ranges, int threadID, bool voltages);
	void ResetRanges(WorkRange* ranges);
	volatile unsigned int m_iterTS;
	unsigned int m_numThreads; //!< number of worker threads
	unsigned int m_max_numThreads; //!< max. number of worker threads
	volatile bool m_stopThreads;
	bool m_opt_speed;
	float m_last_speed;

	//! numThreads==0: IterateTS() times the engine itself and picks the thread count, see TuneThreads()
	bool m_auto_threads;
	struct ThreadTuner
	{
		unsigned int start;  //!< first thread count tried
		unsigned int best;   //!< fastest thread count so far
		double best_speed;   //!< its speed (timesteps/s)
		std::vector<unsigned int> todo;   //!< thread counts still to time, in this order
		std::vector<double> speed;        //!< timesteps/s by thread count (0: not timed)
		bool refined;        //!< the neighbours of the best coarse count are queued
		double settle;       //!< time left before the timing of the current thread count starts (s)
		unsigned int ts;     //!< timesteps timed with the current thread count
		double time;         //!< and their time (s)
		double window;       //!< time to measure each thread count (s)
		double ts_per_s;     //!< last speed, for the length of the pieces of IterateTS()
		unsigned int maxUse; //!< most threads that get x-lines (CalcStartStopLines): every candidate is clamped to it
	} m_Tuner;
	//! Run \a n timesteps on the threads, returns the time (s)
	double RunTimesteps(unsigned int n);
	//! Times P (the performance-core count on Apple, half the hardware threads elsewhere; see StartThreads),
	//! then P+2, P+4, .. up to the maximum (until one is clearly slower than the best) and P-2, then the two
	//! neighbours of the fastest; keeps the fastest count. Every candidate is clamped to m_Tuner.maxUse.
	//! (Coarse steps: the speed is not monotonic, e.g. P+1 threads can be slower than P and P+2.)
	void TuneThreads(unsigned int iterTS, double seconds);

#ifdef ENABLE_DEBUG_TIME
	std::map<boost::thread::id, std::vector<double> > m_timer_list;
#endif
};

#endif // ENGINE_MULTITHREAD_H
