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

#include "engine_ext_steadystate.h"
#include "operator_ext_steadystate.h"
#include "FDTD/engine_sse.h"
#include "FDTD/engine_interface_fdtd.h"
#include <cstring>
#include <limits>

Engine_Ext_SteadyState::Engine_Ext_SteadyState(Operator_Ext_SteadyState* op_ext): Engine_Extension(op_ext)
{
	m_Op_SS = op_ext;
	m_Priority = ENG_EXT_PRIO_STEADYSTATE;

	// three periods of records (see RecordProbes)
	for (size_t n=0;n<m_Op_SS->m_E_probe_dir.size();++n)
	{
		double* rec = new double[m_Op_SS->m_TS_period*3];
		m_E_records.push_back(rec);
	}
	m_last_max_diff = 1;
	last_total_energy = 0;
	m_Eng_Interface = NULL;
	m_Lines = false;
	m_LinesChecked = false;
	m_SnapLines = 0;
	m_NX = 0;
}

Engine_Ext_SteadyState::~Engine_Ext_SteadyState()
{
	for (size_t n=0;n<m_E_records.size();++n)
	{
		delete[] m_E_records.at(n);
		m_E_records.at(n) = NULL;
	}
	m_E_records.clear();
	delete m_Eng_Interface;
	m_Eng_Interface = NULL;
}

void Engine_Ext_SteadyState::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);
	if (!m_LinesChecked && m_Eng)
		SetupLines();
}

void Engine_Ext_SteadyState::SetupLines()
{
	m_LinesChecked = true;
	m_Lines = false;
	if (m_Eng->GetType()!=Engine::SSE)
		return;
	const ArrayLib::ArrayENG<f4vector>* V = ((Engine_sse*)m_Eng)->f4_volt_ptr;
	const ArrayLib::ArrayENG<f4vector>* I = ((Engine_sse*)m_Eng)->f4_curr_ptr;
	if ((V==NULL) || (I==NULL))
		return;
	// the I-J-K-N layout of the fused sweep: every x-line is one block of the arrays (one memcpy per call)
	for (const ArrayLib::ArrayENG<f4vector>* A : {V, I})
		if ((A->stride(0)!=1) || (A->stride(3)!=3) || (A->stride(2)!=3*A->extent(3)) || (A->stride(1)!=A->stride(2)*A->extent(2)))
			return;
	m_NX = V->extent(1);
	m_Lines = true;
}

unsigned int Engine_Ext_SteadyState::GetUsedHooks() const
{
	OPENEMS_HOOKS_DECLARED_BY(Engine_Ext_SteadyState);
	return APPLY_VOLT | (m_Lines ? LINE_HOOKS | ((m_Op_SS->m_TS_period>2u*m_NrThreads) ? TS_LOCAL : 0) : 0);
}

bool Engine_Ext_SteadyState::IsEvaluationStep(unsigned int TS) const
{
	const unsigned int p = m_Op_SS->m_TS_period;
	return (TS%p==0) && (TS>=2*p);
}

void Engine_Ext_SteadyState::RecordProbes(unsigned int TS, unsigned int x0, unsigned int x1)
{
	// the record of timestep TS at TS mod 3 periods: an evaluation step reads the last two periods, while a thread up to a
	// period ahead (neighbour flags, see GetUsedHooks) writes the third
	const unsigned int rel_pos = TS%(3*m_Op_SS->m_TS_period);
	for (size_t n=0;n<m_E_records.size();++n)
	{
		const unsigned int x = m_Op_SS->m_E_probe_pos[0].at(n);
		if ((x>=x0) && (x<x1))
			m_E_records.at(n)[rel_pos] = m_Eng->GetVolt(m_Op_SS->m_E_probe_dir.at(n), x, m_Op_SS->m_E_probe_pos[1].at(n), m_Op_SS->m_E_probe_pos[2].at(n));
	}
}

void Engine_Ext_SteadyState::Apply2Voltages()
{
	const unsigned int TS = m_Eng->GetNumberOfTimesteps();
	RecordProbes(TS, 0, std::numeric_limits<unsigned int>::max());
	if (IsEvaluationStep(TS))
		EvaluatePeriod(TS, m_Eng_Interface->CalcFastEnergy());
}

void Engine_Ext_SteadyState::Apply2VoltagesLines(unsigned int x0, unsigned int x1)
{
	const unsigned int TS = m_Eng->GetNumberOfTimesteps();
	RecordProbes(TS, x0, x1);
	if (!IsEvaluationStep(TS))
		return;
	const ArrayLib::ArrayENG<f4vector>& V = *((Engine_sse*)m_Eng)->f4_volt_ptr;
	const ArrayLib::ArrayENG<f4vector>& I = *((Engine_sse*)m_Eng)->f4_curr_ptr;
	std::call_once(m_SnapOnce, [&]()
	{
		m_SnapVolt.Init("SteadyState_volt", {V.extent(1), V.extent(2), V.extent(3)});
		m_SnapCurr.Init("SteadyState_curr", {I.extent(1), I.extent(2), I.extent(3)});
	});
	const size_t line = V.stride(1); // f4vectors per x-line, all three components
	memcpy(m_SnapVolt.data()+x0*line, V.data()+x0*line, (x1-x0)*line*sizeof(f4vector));
	memcpy(m_SnapCurr.data()+x0*line, I.data()+x0*line, (x1-x0)*line*sizeof(f4vector));
	// the thread that copies the last lines evaluates (the others' copies and records are complete then)
	if (m_SnapLines.fetch_add(x1-x0, std::memory_order_acq_rel)+(x1-x0)!=m_NX)
		return;
	m_SnapLines.store(0, std::memory_order_relaxed);
	unsigned int numNodes[3];
	m_Eng_Interface->GetFastEnergyNodes(numNodes);
	EvaluatePeriod(TS, Engine_sse::FastEnergy(m_SnapVolt, m_SnapCurr, numNodes, V.extent(3)));
}

void Engine_Ext_SteadyState::EvaluatePeriod(unsigned int TS, double curr_total_energy)
{
	const unsigned int p = m_Op_SS->m_TS_period;
	const unsigned int rel_pos = TS%(2*p);
	bool no_valid = true;
	m_last_max_diff = 0;
	if (last_total_energy>0)
	{
		m_last_max_diff = abs(curr_total_energy-last_total_energy)/last_total_energy;
		no_valid = false;
	}
	//cerr << curr_total_energy << "/" << last_total_energy << "=" << abs(curr_total_energy-last_total_energy)/last_total_energy << endl;
	last_total_energy = curr_total_energy;
	unsigned int old_pos = 0;
	unsigned int new_pos = p;
	if (rel_pos<=p)
	{
		new_pos = 0;
		old_pos = p;
	}
	//cerr << TS << "/" << rel_pos << ": one period complete, new_pos" << new_pos << " old pos: " << old_pos << endl;
	// slot k of the two-period ring (TS mod 2p) is the last timestep <= TS with that slot; it is kept at its timestep mod 3p
	std::vector<unsigned int> slot(2*p);
	for (unsigned int k=0;k<2*p;++k)
		slot[k] = (TS - (TS+2*p-k)%(2*p)) % (3*p);
	double *curr_pow = new double[m_E_records.size()];
	double *diff_pow = new double[m_E_records.size()];
	double max_pow = 0;
	for (size_t n=0;n<m_E_records.size();++n)
	{
		double *buf = m_E_records.at(n);
		curr_pow[n] = 0;
		diff_pow[n] = 0;
		for (unsigned int nt=0;nt<p;++nt)
		{
			const double vn = buf[slot[nt+new_pos]], vo = buf[slot[nt+old_pos]];
			curr_pow[n] += vn*vn;
			diff_pow[n] += (vo-vn)*(vo-vn);
		}
		max_pow = std::max(max_pow, curr_pow[n]);
	}
	for (size_t n=0;n<m_E_records.size();++n)
	{
		//cerr << "curr_pow: " << curr_pow[n] <<  " diff_pow: " << diff_pow[n] << " diff: " << diff_pow[n]/curr_pow[n] << endl;
		if (curr_pow[n]>max_pow*1e-2)
		{
			m_last_max_diff = std::max(m_last_max_diff, diff_pow[n]/curr_pow[n]);
			//cerr << m_last_max_diff << endl;
			no_valid = false;
		}
	}
	if ((no_valid) || (m_last_max_diff>1))
		m_last_max_diff = 1;
	delete[] curr_pow; curr_pow = NULL;
	delete[] diff_pow; diff_pow = NULL;
	//cerr << m_last_max_diff << endl;
}

void Engine_Ext_SteadyState::Apply2Current()
{

}
