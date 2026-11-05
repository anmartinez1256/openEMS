/*
*	Copyright (C) 2015 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_STEADYSTATE_H
#define ENGINE_EXT_STEADYSTATE_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "tools/array_ops.h"
#include "tools/arraylib/array_e.h"
#include <atomic>
#include <mutex>

class Operator_Ext_SteadyState;
class Engine_Interface_FDTD;

class Engine_Ext_SteadyState : public Engine_Extension
{
	friend class HIP_Ext_SteadyState;
	friend class Metal_Ext_SteadyState;
public:
	Engine_Ext_SteadyState(Operator_Ext_SteadyState* op_ext);
	virtual ~Engine_Ext_SteadyState();

	virtual void SetNumberOfThreads(int nrThread);
	//! TS_LOCAL while a period is longer than twice the thread count: the threads are then less than a period apart (the
	//! records keep three periods, the copies and the count of copied lines are not touched again before the next
	//! evaluation step), and every evaluation sees the records and copies of its timestep
	virtual unsigned int GetUsedHooks() const;

	virtual void Apply2Voltages();
	virtual void Apply2Current();
	virtual void Apply2VoltagesLines(unsigned int x0, unsigned int x1);

	void SetEngineInterface(Engine_Interface_FDTD* eng_if) {m_Eng_Interface=eng_if;}
	double GetLastDiff() {return m_last_max_diff;}

protected:
	//! the E-field records of timestep TS, the probes on the x-lines [x0, x1) only
	void RecordProbes(unsigned int TS, unsigned int x0, unsigned int x1);
	//! timestep TS ends a period (from the third period on): compare the last two periods
	bool IsEvaluationStep(unsigned int TS) const;
	void EvaluatePeriod(unsigned int TS, double curr_total_energy);

	//! Line hooks for the fused sweep of the SSE engines (fields in the I-J-K-N layout). The unfused hook sees all fields
	//! at one point of the timestep: the voltages after their update, the post hooks and the apply hooks before this one
	//! (the extension comes first in that order), the currents before their update. The fused sweep has that state line
	//! by line, at this hook: the hook records the probes of its lines, and on an evaluation step copies the voltages and
	//! currents of its lines; the thread that copies the last line evaluates the period from the copies with the same
	//! energy sum (Engine_sse::FastEnergy). Same values, same operations (bit-identical). The copies (24 B per cell) are
	//! allocated at the first evaluation step that runs the fused sweep.
	bool m_Lines;
	bool m_LinesChecked;
	void SetupLines();
	ArrayLib::ArrayENG<f4vector> m_SnapVolt, m_SnapCurr;
	std::once_flag m_SnapOnce;
	std::atomic<unsigned int> m_SnapLines; //!< x-lines copied in this evaluation step
	unsigned int m_NX;

	Operator_Ext_SteadyState* m_Op_SS;
	double m_last_max_diff;
	std::vector<double*> m_E_records;
	std::vector<double*> m_H_records;

	double last_total_energy;
	Engine_Interface_FDTD* m_Eng_Interface;
};


#endif // ENGINE_EXT_STEADYSTATE_H
