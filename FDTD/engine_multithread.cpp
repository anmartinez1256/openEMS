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

//#define ENABLE_DEBUG_TIME

#ifdef ENABLE_DEBUG_TIME
#define DEBUG_TIME(x) x;
#else
#define DEBUG_TIME(x) ;
#endif



#include "engine_multithread.h"
#include "extensions/engine_extension.h"
#include "tools/denormal.h"
#include "tools/useful.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time/gregorian/gregorian.hpp"
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;

//! \brief construct an Engine_Multithread instance
//! it's the responsibility of the caller to free the returned pointer
Engine_Multithread* Engine_Multithread::New(const Operator_Multithread* op, unsigned int numThreads)
{
	cout << "Create FDTD engine (compressed SSE + multi-threading)" << endl;
	Engine_Multithread* e = new Engine_Multithread(op);
	e->setNumThreads( numThreads );
	e->Init();
	return e;
}

Engine_Multithread::Engine_Multithread(const Operator_Multithread* op) : Engine_SSE_Compressed(op)
{
	m_Op_MT = op;
	m_type = SSE;
	m_IterateBarrier = 0;
	m_VoltRange = 0;
	m_CurrRange = 0;
	m_startBarrier = 0;
	m_stopBarrier = 0;
	m_thread_group = 0;
	m_max_numThreads = AvailableThreads();
	m_numThreads = 0;
	m_last_speed = 0;
	m_opt_speed = false;
	m_auto_threads = false;
	m_Fused = false;
	m_FusedStat = 0;
	m_Balance = false;
	m_stopThreads = true;
}

//! first thread count of the automatic search: the performance cores (Apple) or half the hardware threads
static unsigned int StartThreads(unsigned int maxThreads)
{
	unsigned int n = 0;
#ifdef __APPLE__
	int pcores = 0;
	size_t len = sizeof(pcores);
	if ((sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &len, NULL, 0)==0) && (pcores>0))
		n = pcores;
#endif
	if (n==0)
		n = (maxThreads+1)/2;
	return std::max(1u, std::min(n, maxThreads));
}

Engine_Multithread::~Engine_Multithread()
{
#ifdef ENABLE_DEBUG_TIME
	NS_Engine_Multithread::DBG().cout() << "Engine_Multithread::~Engine_Multithread()" << endl;
	std::map<boost::thread::id, std::vector<double> >::iterator it;
	for (it=m_timer_list.begin(); it!=m_timer_list.end(); it++)
	{
		NS_Engine_Multithread::DBG().cout() << "*** DEBUG Thread: " << it->first << std::endl;
		std::vector<double>::iterator it2;
		for (it2=it->second.begin(); it2<it->second.end();)
		{
			NS_Engine_Multithread::DBG().cout() << "after voltage update, before barrier1: " << fixed << setprecision(6) << *(it2++) << std::endl;
			NS_Engine_Multithread::DBG().cout() << "after barrier1, before barrier2: "       << fixed << setprecision(6) << *(it2++) << std::endl;
			NS_Engine_Multithread::DBG().cout() << "after barrier2, before current update: " << fixed << setprecision(6) << *(it2++) << std::endl;
			NS_Engine_Multithread::DBG().cout() << "after current update, before barrier3: " << fixed << setprecision(6) << *(it2++) << std::endl;
			NS_Engine_Multithread::DBG().cout() << "after barrier3: "                        << fixed << setprecision(6) << *(it2++) << std::endl;
		}
	}
#endif

	Reset();
}

void Engine_Multithread::setNumThreads( unsigned int numThreads )
{
	m_numThreads = numThreads;
}

void Engine_Multithread::Init()
{
	m_stopThreads = true;
	m_opt_speed = false;
	m_auto_threads = false;
	Engine_SSE_Compressed::Init();

	// initialize threads
	m_stopThreads = false;
	if (m_numThreads == 0)
	{
		m_auto_threads = true;
		m_numThreads = StartThreads(m_max_numThreads);
		m_Tuner.start = m_Tuner.best = m_numThreads;
		m_Tuner.best_speed = 0;
		m_Tuner.todo.clear();
		for (unsigned int n=m_numThreads+2; n<m_max_numThreads+2; n+=2)
			m_Tuner.todo.push_back(std::min(n, m_max_numThreads));
		if (m_numThreads>2)
			m_Tuner.todo.push_back(m_numThreads-2);
		m_Tuner.speed.assign(m_max_numThreads+1, 0);
		m_Tuner.refined = false;
		m_Tuner.ts = 0;
		m_Tuner.time = 0;
		const char* env = getenv("OPENEMS_TUNE_WINDOW_MS");
		m_Tuner.window = (env ? atof(env) : 150)*1e-3;
		m_Tuner.settle = 0.4*m_Tuner.window;
		m_Tuner.ts_per_s = 0;
	}
	else if (m_numThreads > m_max_numThreads)
		m_numThreads = m_max_numThreads;

	this->changeNumThreads(m_numThreads);
}

void Engine_Multithread::Reset()
{
	if (m_thread_group!=0) // prevent multiple invocations
	{
		ClearExtensions(); //prevent extensions from interfering with thread reset...

		// stop the threads
		//NS_Engine_Multithread::DBG().cout() << "stopping all threads" << endl;
		m_thread_group->interrupt_all();
		m_thread_group->join_all(); // wait for termination
		delete m_IterateBarrier;
		m_IterateBarrier = 0;
		delete[] m_VoltRange;
		m_VoltRange = 0;
		delete[] m_CurrRange;
		m_CurrRange = 0;
		delete[] m_FusedStat;
		m_FusedStat = 0;
		delete m_startBarrier;
		m_startBarrier = 0;
		delete m_stopBarrier;
		m_stopBarrier = 0;
		delete m_thread_group;
		m_thread_group = 0;
	}

	Engine_SSE_Compressed::Reset();
}

void Engine_Multithread::changeNumThreads(unsigned int numThreads)
{
	if (m_thread_group!=0)
	{
		m_thread_group->interrupt_all();
		m_thread_group->join_all(); // wait for termination
		delete m_thread_group;
		m_thread_group = 0;
		//m_stopThreads = false;
	}

	m_numThreads = numThreads;

	if (g_settings.GetVerboseLevel()>0)
		cout << "Multithreaded engine using " << m_numThreads << " threads. Utilization: (";

	std::vector<unsigned int> m_Start_Lines;
	std::vector<unsigned int> m_Stop_Lines;
	m_Op_MT->CalcStartStopLines( m_numThreads, m_Start_Lines, m_Stop_Lines );

	if (m_IterateBarrier!=0)
		delete m_IterateBarrier;
	// make sure all threads are waiting
	m_IterateBarrier = new SpinBarrier(m_numThreads); // numThread workers

	if (m_startBarrier!=0)
		delete m_startBarrier;
	m_startBarrier = new boost::barrier(m_numThreads+1); // numThread workers + 1 controller

	if (m_stopBarrier!=0)
		delete m_stopBarrier;
	m_stopBarrier = new boost::barrier(m_numThreads+1); // numThread workers + 1 controller

	delete[] m_VoltRange;
	delete[] m_CurrRange;
	m_VoltRange = new WorkRange[m_numThreads];
	m_CurrRange = new WorkRange[m_numThreads];
	for (unsigned int n=0; n<m_numThreads; n++)
	{
		m_VoltRange[n].start = m_Start_Lines.at(n);
		m_VoltRange[n].stop  = m_Stop_Lines.at(n);
		m_CurrRange[n].start = m_Start_Lines.at(n);
		// the last x-line has no current update
		m_CurrRange[n].stop  = (n==m_numThreads-1) ? (int)m_Stop_Lines.at(n)-1 : m_Stop_Lines.at(n);
	}
	ResetRanges(m_VoltRange);
	ResetRanges(m_CurrRange);

	m_thread_group = new boost::thread_group();
	for (unsigned int n=0; n<m_numThreads; n++)
	{
		unsigned int start = m_Start_Lines.at(n);
		unsigned int stop = m_Stop_Lines.at(n);
		unsigned int stop_h = stop;
		if (n == m_numThreads-1)
		{
			// last thread
			stop_h = stop-1;
			if (g_settings.GetVerboseLevel()>0)
				cout << stop-start+1 << ")" << endl;
		}
		else
			if (g_settings.GetVerboseLevel()>0)
				cout << stop-start+1 << ";";
		boost::thread *t = new boost::thread( NS_Engine_Multithread::thread(this,start,stop,stop_h,n) );
		m_thread_group->add_thread( t );
	}

	for (size_t n=0; n<m_Eng_exts.size(); ++n)
		m_Eng_exts.at(n)->SetNumberOfThreads(m_numThreads);
	// after SetNumberOfThreads: the extensions pick their fast paths there (the threads wait at the start barrier)
	BuildHookPlan();
}

bool Engine_Multithread::IterateTS(unsigned int iterTS)
{
	// e.g. the cylindrical multi-grid adds and sorts extensions after Init (the threads wait at the start barrier)
	if (m_HookPlanExts!=m_Eng_exts)
		BuildHookPlan();

	// While the thread count is tuned, the call runs in pieces of 1/8 of the measurement window (a long
	// call, e.g. between the probes of a large model, would otherwise time a single thread count),
	// with the slabs balanced after every piece, as they will be for the rest of the run.
	while (m_auto_threads && (iterTS>0))
	{
		unsigned int n = std::min(iterTS, 4u);   // the first piece estimates the speed
		if (m_Tuner.ts_per_s>0)
			n = (unsigned int)std::min<double>(iterTS, std::max(1.0, m_Tuner.window/8*m_Tuner.ts_per_s));
		const double seconds = RunTimesteps(n);
		if (seconds>0)
			m_Tuner.ts_per_s = n/seconds;
		iterTS -= n;
		if (m_Fused && m_Balance)
			BalanceSlabs();
		TuneThreads(n, seconds);
	}
	if (iterTS>0)
	{
		RunTimesteps(iterTS);
		if (m_Fused && m_Balance)
			BalanceSlabs();
	}
	return true;
}

double Engine_Multithread::RunTimesteps(unsigned int n)
{
	const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	m_iterTS = n;
	m_startBarrier->wait(); // start the threads
	m_stopBarrier->wait(); // wait for the threads to finish n time steps
	return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}

void Engine_Multithread::BalanceSlabs()
{
	// wait for ~20 ms of sweeps per thread, the measurement is noisy below
	double total = 0;
	for (unsigned int t=0; t<m_numThreads; ++t)
		total += m_FusedStat[t].busy;
	if (total<0.02*m_numThreads)
		return;

	// time per line of each thread, spread over its lines, smoothed over the calls
	const unsigned int NX = numLines[0];
	const bool first = m_LineCost.empty();
	m_LineCost.resize(NX, 0);
	for (unsigned int t=0; t<m_numThreads; ++t)
	{
		FusedStat& stat = m_FusedStat[t];
		const double cost = (stat.lines>0) ? stat.busy/stat.lines : 0;
		for (unsigned int x=m_Blocks[m_SlabBlocks[t]]; x<m_Blocks[m_SlabBlocks[t+1]]; ++x)
			m_LineCost[x] = first ? cost : 0.5*(m_LineCost[x]+cost);
		stat.busy = stat.lines = 0;
	}

	// new boundaries at equal shares of the estimated time, on block boundaries
	std::vector<double> cum(NX+1, 0);
	for (unsigned int x=0; x<NX; ++x)
		cum[x+1] = cum[x] + m_LineCost[x];
	const unsigned int nb = m_Blocks.size()-1;
	unsigned int b = 0;
	for (unsigned int t=1; t<m_numThreads; ++t)
	{
		const double target = cum[NX]*t/m_numThreads;
		while ((b<nb) && (cum[m_Blocks[b+1]]<=target))
			++b;
		// the nearer one of the boundaries around the target
		if ((b<nb) && (target-cum[m_Blocks[b]] > cum[m_Blocks[b+1]]-target))
			++b;
		b = std::max(b, m_SlabBlocks[t-1]+1);
		b = std::min(b, nb-(m_numThreads-t));
		m_SlabBlocks[t] = b;
	}
}

void Engine_Multithread::TuneThreads(unsigned int iterTS, double seconds)
{
	ThreadTuner& t = m_Tuner;
	// after a thread change: new caches, the slabs are balanced
	if (t.settle>0)
	{
		t.settle -= seconds;
		return;
	}
	t.ts += iterTS;
	t.time += seconds;
	if ((t.time<t.window) || (t.ts<2))
		return;

	const double speed = t.ts/t.time;
	t.speed[m_numThreads] = speed;
	if (speed>t.best_speed)
	{
		t.best = m_numThreads;
		t.best_speed = speed;
	}
	// clearly slower with more threads: skip the larger counts
	if ((m_numThreads>t.start) && (speed<0.85*t.best_speed))
		t.todo.erase(std::remove_if(t.todo.begin(), t.todo.end(), [&](unsigned int n) {return n>m_numThreads;}), t.todo.end());
	while (true)
	{
		while (!t.todo.empty() && (t.speed[t.todo.front()]>0))
			t.todo.erase(t.todo.begin());
		if (!t.todo.empty() || t.refined)
			break;
		// the coarse counts are done: the neighbours of the fastest one
		t.refined = true;
		if (t.best>1)
			t.todo.push_back(t.best-1);
		if (t.best<m_max_numThreads)
			t.todo.push_back(t.best+1);
	}
	if (t.todo.empty())
	{
		m_auto_threads = false;
		if (t.best!=m_numThreads)
			changeNumThreads(t.best);
		cout << "Multithreaded Engine: Best performance found using " << m_numThreads << " threads." << std::endl;
		return;
	}
	const unsigned int next = t.todo.front();
	t.todo.erase(t.todo.begin());
	changeNumThreads(next);
	t.ts = 0;
	t.time = 0;
	t.settle = 0.4*t.window;
}

void Engine_Multithread::NextInterval(float curr_speed)
{
	Engine_SSE_Compressed::NextInterval(curr_speed);
	if (!m_opt_speed) return;
	if (curr_speed<m_last_speed)
	{
		this->changeNumThreads(m_numThreads-1);
		cout << "Multithreaded Engine: Best performance found using " << m_numThreads << " threads." << std::endl;
		m_opt_speed = false;
	}
	else if (m_numThreads<m_max_numThreads)
	{
		m_last_speed = curr_speed;
		this->changeNumThreads(m_numThreads+1);
	}
}

void Engine_Multithread::ResetRanges(WorkRange* ranges)
{
	for (unsigned int n=0; n<m_numThreads; ++n)
		ranges[n].next.store(ranges[n].start, std::memory_order_relaxed);
}

void Engine_Multithread::UpdateStealing(WorkRange* ranges, int threadID, bool voltages)
{
	// The order of the x-lines does not matter within one update (each cell only reads the other
	// field), so the result is the same whichever thread updates a line. One line per grab: the
	// atomic costs ~10-50 ns, a line is >1 us of work.
	for (unsigned int k=0; k<m_numThreads; ++k)
	{
		WorkRange& r = ranges[(threadID+k)%m_numThreads];
		for (;;)
		{
			const int x = r.next.fetch_add(1, std::memory_order_relaxed);
			if (x>r.stop)
				break;
			if (voltages)
				UpdateVoltages(x, 1);
			else
				UpdateCurrents(x, 1);
		}
	}
}

void Engine_Multithread::BuildHookPlan()
{
	const unsigned int flags[6] = {Engine_Extension::PRE_VOLT, Engine_Extension::POST_VOLT, Engine_Extension::APPLY_VOLT,
								   Engine_Extension::PRE_CURR, Engine_Extension::POST_CURR, Engine_Extension::APPLY_CURR};
	for (int g=0; g<6; ++g)
	{
		std::vector<HookStep>& plan = m_HookPlan[g];
		plan.clear();
		// pre-updates run in reverse order -> highest priority gets access to the fields last
		const bool reverse = (g==0) || (g==3);
		// apply-hooks write engine fields: keep the priority order with one barrier per extension
		const bool apply = (g==2) || (g==5);
		const int N = m_Eng_exts.size();
		for (int k=0; k<N; ++k)
		{
			Engine_Extension* ext = m_Eng_exts.at(reverse ? N-1-k : k);
			const unsigned int used = ext->GetUsedHooks();
			if (used==Engine_Extension::LEGACY_HOOKS)
			{
				// as before: a barrier before and after its (possibly empty) hook
				if (!plan.empty())
					plan.back().barrier = true;
				HookStep step = {ext, true, false};
				plan.push_back(step);
			}
			else if (used & flags[g])
			{
				// ordered: keep a barrier before and after the hook
				const bool ordered = apply || (used & Engine_Extension::PREPOST_WRITES_FIELDS);
				if (ordered && !plan.empty())
					plan.back().barrier = true;
				HookStep step = {ext, ordered, false};
				plan.push_back(step);
			}
		}
		// pre/post hooks of all extensions share one barrier at the end of the group
		if (!plan.empty())
			plan.back().barrier = true;
	}
	m_HookPlanExts = m_Eng_exts;
	// the last barrier of a timestep increments numTS before it releases the threads
	m_IncNumTS_AtCurrBarrier = false;
	if (!m_HookPlan[5].empty())
		m_HookPlan[5].back().inc_numTS = true;
	else if (!m_HookPlan[4].empty())
		m_HookPlan[4].back().inc_numTS = true;
	else
		m_IncNumTS_AtCurrBarrier = true;

	SetupFused();
}

void Engine_Multithread::SetupFused()
{
	m_Fused = false;
	for (int g=0; g<6; ++g)
		m_LineHooks[g].clear();
	const char* env = getenv("OPENEMS_FUSED");
	if ((env && (atoi(env)==0)) || !UseIJKN())
		return;

	const unsigned int flags[6] = {Engine_Extension::PRE_VOLT, Engine_Extension::POST_VOLT, Engine_Extension::APPLY_VOLT,
								   Engine_Extension::PRE_CURR, Engine_Extension::POST_CURR, Engine_Extension::APPLY_CURR};
	const int N = m_Eng_exts.size();
	std::vector<unsigned int> coupled;
	for (int k=0; k<N; ++k)
	{
		const unsigned int used = m_Eng_exts.at(k)->GetUsedHooks();
		if ((used==Engine_Extension::LEGACY_HOOKS) || !(used & Engine_Extension::LINE_HOOKS))
		{
			if (g_settings.GetVerboseLevel()>0)
				cout << "Multithreaded engine: no fused sweep, the extension \"" << m_Eng_exts.at(k)->GetExtensionName() << "\" has no line hooks" << endl;
			return;
		}
		m_Eng_exts.at(k)->GetCoupledLines(coupled);
	}
	for (int g=0; g<6; ++g)
	{
		// same extension order as the hook plan: pre-updates in reverse order
		const bool reverse = (g==0) || (g==3);
		for (int k=0; k<N; ++k)
		{
			Engine_Extension* ext = m_Eng_exts.at(reverse ? N-1-k : k);
			if (ext->GetUsedHooks() & flags[g])
				m_LineHooks[g].push_back(ext);
		}
	}

	// blocks: single x-lines, except that a coupled line x is swept together with x+1
	const unsigned int NX = numLines[0];
	std::vector<bool> joined(NX, false);
	for (size_t n=0; n<coupled.size(); ++n)
		if (coupled[n]+1<NX)
			joined[coupled[n]] = true;
	m_Blocks.clear();
	for (unsigned int x=0; x<NX; ++x)
		if ((x==0) || !joined[x-1])
			m_Blocks.push_back(x);
	m_Blocks.push_back(NX);
	const unsigned int nb = m_Blocks.size()-1;
	if (nb<m_numThreads)
		return;

	// slabs of about the same number of lines, on block boundaries
	m_SlabBlocks.assign(m_numThreads+1, nb);
	m_SlabBlocks[0] = 0;
	unsigned int b = 0;
	for (unsigned int t=1; t<m_numThreads; ++t)
	{
		const unsigned int target = (unsigned int)((uint64_t)NX*t/m_numThreads);
		while ((b<nb) && (m_Blocks[b]<target))
			++b;
		b = std::max(b, m_SlabBlocks[t-1]+1);
		b = std::min(b, nb-(m_numThreads-t));
		m_SlabBlocks[t] = b;
	}
	delete[] m_FusedStat;
	m_FusedStat = new FusedStat[m_numThreads];
	for (unsigned int t=0; t<m_numThreads; ++t)
		m_FusedStat[t].busy = m_FusedStat[t].lines = 0;
	m_LineCost.clear();
	const char* bal = getenv("OPENEMS_BALANCE");
	m_Balance = (m_numThreads>1) && !(bal && (atoi(bal)==0));
	m_Fused = true;
	if (g_settings.GetVerboseLevel()>0)
		cout << "Multithreaded engine: fused sweep, " << nb << " blocks of x-lines" << endl;
}

void Engine_Multithread::RunLineHooks(int group, unsigned int x0, unsigned int x1)
{
	const std::vector<Engine_Extension*>& exts = m_LineHooks[group];
	for (size_t n=0; n<exts.size(); ++n)
	{
		switch (group)
		{
		case 0: exts[n]->DoPreVoltageUpdatesLines(x0, x1); break;
		case 1: exts[n]->DoPostVoltageUpdatesLines(x0, x1); break;
		case 2: exts[n]->Apply2VoltagesLines(x0, x1); break;
		case 3: exts[n]->DoPreCurrentUpdatesLines(x0, x1); break;
		case 4: exts[n]->DoPostCurrentUpdatesLines(x0, x1); break;
		case 5: exts[n]->Apply2CurrentLines(x0, x1); break;
		}
	}
}

void Engine_Multithread::FusedCurrents(unsigned int x0, unsigned int x1)
{
	if (x0>=x1)
		return;
	RunLineHooks(3, x0, x1);
	UpdateCurrents(x0, x1-x0);
	RunLineHooks(4, x0, x1);
	RunLineHooks(5, x0, x1);
}

void Engine_Multithread::FusedTimestep(int threadID)
{
	// The voltages of a block need the old currents of the block and of the line before it; the
	// currents of line x need the final voltages of x and x+1. So after the voltages (and their hooks)
	// of a block are done, the currents of the line before the block and of the block but its last line
	// are updated: every cell sees the same operands as in the unfused update (bit-identical).
	// The line before the slab belongs to the previous thread's slab: it waits for the first barrier.
	const unsigned int b0 = m_SlabBlocks[threadID], b1 = m_SlabBlocks[threadID+1];
	const unsigned int start = m_Blocks[b0];
	const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (unsigned int b=b0; b<b1; ++b)
	{
		const unsigned int x0 = m_Blocks[b], x1 = m_Blocks[b+1];
		RunLineHooks(0, x0, x1);
		UpdateVoltages(x0, x1-x0);
		RunLineHooks(1, x0, x1);
		RunLineHooks(2, x0, x1);
		FusedCurrents((x0==start) ? x0 : x0-1, x1-1);
	}
	FusedStat& stat = m_FusedStat[threadID];
	stat.busy += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
	stat.lines += m_Blocks[b1]-start;
	m_IterateBarrier->wait();
	if (start>0)
		FusedCurrents(start-1, start);
	m_IterateBarrier->wait(&numTS);
}

void Engine_Multithread::RunHooks(int group, int threadID)
{
	const std::vector<HookStep>& plan = m_HookPlan[group];
	for (size_t n=0; n<plan.size(); ++n)
	{
		Engine_Extension* ext = plan[n].ext;
		switch (group)
		{
		case 0: ext->DoPreVoltageUpdates(threadID); break;
		case 1: ext->DoPostVoltageUpdates(threadID); break;
		case 2: ext->Apply2Voltages(threadID); break;
		case 3: ext->DoPreCurrentUpdates(threadID); break;
		case 4: ext->DoPostCurrentUpdates(threadID); break;
		case 5: ext->Apply2Current(threadID); break;
		}
		if (plan[n].barrier)
			m_IterateBarrier->wait(plan[n].inc_numTS ? &numTS : NULL);
	}
}

void Engine_Multithread::DoPreVoltageUpdates(int threadID)
{
	RunHooks(0, threadID);
}

void Engine_Multithread::DoPostVoltageUpdates(int threadID)
{
	RunHooks(1, threadID);
}

void Engine_Multithread::Apply2Voltages(int threadID)
{
	RunHooks(2, threadID);
}

void Engine_Multithread::DoPreCurrentUpdates(int threadID)
{
	RunHooks(3, threadID);
}

void Engine_Multithread::DoPostCurrentUpdates(int threadID)
{
	RunHooks(4, threadID);
}

void Engine_Multithread::Apply2Current(int threadID)
{
	RunHooks(5, threadID);
}

//
// *************************************************************************************************************************
//
namespace NS_Engine_Multithread
{

thread::thread( Engine_Multithread* ptr, unsigned int start, unsigned int stop, unsigned int stop_h, unsigned int threadID )
{
	m_enginePtr = ptr;
	m_start = start;
	m_stop = stop;
	m_stop_h = stop_h;
	m_threadID = threadID;
}

void thread::operator()()
{
	//std::cout << "thread::operator() Parameters: " << m_start << " " << m_stop << std::endl;
	//DBG().cout() << "Thread " << m_threadID << " (" << boost::this_thread::get_id() << ") started." << endl;

	// speed up the calculation of denormal floating point values (flush-to-zero)
	Denormal::Disable();

	while (!m_enginePtr->m_stopThreads)
	{
		// wait for start
		//DBG().cout() << "Thread " << m_threadID << " (" << boost::this_thread::get_id() << ") waiting..." << endl;
		m_enginePtr->m_startBarrier->wait();
		//cout << "Thread " << boost::this_thread::get_id() << " waiting... started." << endl;

		if (m_enginePtr->m_stopThreads)
		{
			//DBG().cout() << "Thread " << m_threadID << " (" << boost::this_thread::get_id() << ") stop!." << endl;
			return;
		}

		DEBUG_TIME( Timer timer1 );

		for (unsigned int iter=0; iter<m_enginePtr->m_iterTS; ++iter)
		{
			if (m_enginePtr->m_Fused)
			{
				m_enginePtr->FusedTimestep(m_threadID);
				continue;
			}

			// pre voltage stuff...
			m_enginePtr->DoPreVoltageUpdates(m_threadID);

			//voltage updates
			m_enginePtr->UpdateStealing(m_enginePtr->m_VoltRange, m_threadID, true);

			// record time
			DEBUG_TIME( m_enginePtr->m_timer_list[boost::this_thread::get_id()].push_back( timer1.elapsed() ); )

			//cout << "Thread " << boost::this_thread::get_id() << " m_barrier1 waiting..." << endl;
			m_enginePtr->m_IterateBarrier->wait();

			// all voltage lines are done: rearm them for the next timestep (seen by all after the next barrier)
			if (m_threadID == 0)
				m_enginePtr->ResetRanges(m_enginePtr->m_VoltRange);

			// record time
			DEBUG_TIME( m_enginePtr->m_timer_list[boost::this_thread::get_id()].push_back( timer1.elapsed() ); )

			//post voltage stuff...
			m_enginePtr->DoPostVoltageUpdates(m_threadID);
			m_enginePtr->Apply2Voltages(m_threadID);

			// record time
			DEBUG_TIME( m_enginePtr->m_timer_list[boost::this_thread::get_id()].push_back( timer1.elapsed() ); )

			//pre current stuff
			m_enginePtr->DoPreCurrentUpdates(m_threadID);

			//current updates
			m_enginePtr->UpdateStealing(m_enginePtr->m_CurrRange, m_threadID, false);

			// record time
			DEBUG_TIME( m_enginePtr->m_timer_list[boost::this_thread::get_id()].push_back( timer1.elapsed() ); )
			// the last barrier of the timestep increments numTS (here, or after the last current hook)
			m_enginePtr->m_IterateBarrier->wait(m_enginePtr->m_IncNumTS_AtCurrBarrier ? &m_enginePtr->numTS : NULL);

			if (m_threadID == 0)
				m_enginePtr->ResetRanges(m_enginePtr->m_CurrRange);

			// record time
			DEBUG_TIME( m_enginePtr->m_timer_list[boost::this_thread::get_id()].push_back( timer1.elapsed() ); )

			//post current stuff
			m_enginePtr->DoPostCurrentUpdates(m_threadID);
			m_enginePtr->Apply2Current(m_threadID);
		}

		m_enginePtr->m_stopBarrier->wait();
	}

	//DBG().cout() << "Thread " << m_threadID << " (" << boost::this_thread::get_id() << ") finished." << endl;
}

} // namespace

