/*
*	Copyright (C) 2011 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "engine_ext_excitation.h"
#include "operator_ext_excitation.h"
#include "FDTD/engine_sse.h"
#include <algorithm>

Engine_Ext_Excitation::Engine_Ext_Excitation(Operator_Ext_Excitation* op_ext) : Engine_Extension(op_ext)
{
	m_Op_Exc = op_ext;
	m_Priority = ENG_EXT_PRIO_EXCITATION;
	SortByLine(m_Op_Exc->Volt_index[0], m_Op_Exc->Volt_Count, m_VoltOrder, m_VoltX);
	SortByLine(m_Op_Exc->Curr_index[0], m_Op_Exc->Curr_Count, m_CurrOrder, m_CurrX);
}

void Engine_Ext_Excitation::SortByLine(const unsigned int* x, unsigned int count, std::vector<unsigned int>& order, std::vector<unsigned int>& sorted_x)
{
	order.resize(count);
	for (unsigned int n=0; n<count; ++n)
		order[n] = n;
	std::stable_sort(order.begin(), order.end(), [x](unsigned int a, unsigned int b) {return x[a]<x[b];});
	sorted_x.resize(count);
	for (unsigned int n=0; n<count; ++n)
		sorted_x[n] = x[order[n]];
}

Engine_Ext_Excitation::~Engine_Ext_Excitation()
{

}

template <typename EngType>
void Engine_Ext_Excitation::Apply2VoltagesImpl(EngType* eng, const unsigned int* order, unsigned int n0, unsigned int n1)
{
	//soft voltage excitation here (E-field excite)
	int exc_pos;
	unsigned int ny;
	unsigned int pos[3];
	int numTS = m_Eng->GetNumberOfTimesteps();
	unsigned int length = m_Op_Exc->m_Exc->GetLength();
	FDTD_FLOAT* exc_volt =  m_Op_Exc->m_Exc->GetVoltageSignal();

	int p = numTS+1;
	if (m_Op_Exc->m_Exc->GetSignalPeriod()>0)
		p = int(m_Op_Exc->m_Exc->GetSignalPeriod()/m_Op_Exc->m_Exc->GetTimestep());

	for (unsigned int k=n0; k<n1; ++k)
	{
		const unsigned int n = order ? order[k] : k;
		exc_pos = numTS - (int)m_Op_Exc->Volt_delay[n];
		exc_pos *= (exc_pos>0);
		exc_pos %= p;
		exc_pos *= (exc_pos<(int)length);
		ny = m_Op_Exc->Volt_dir[n];
		pos[0]=m_Op_Exc->Volt_index[0][n];
		pos[1]=m_Op_Exc->Volt_index[1][n];
		pos[2]=m_Op_Exc->Volt_index[2][n];
		eng->EngType::SetVolt(ny,pos, eng->EngType::GetVolt(ny,pos) + m_Op_Exc->Volt_amp[n]*exc_volt[exc_pos]);
	}
}

void Engine_Ext_Excitation::Apply2Voltages()
{
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, (const unsigned int*)NULL, 0u, m_Op_Exc->Volt_Count);
}

void Engine_Ext_Excitation::Apply2VoltagesLines(unsigned int x0, unsigned int x1)
{
	// the fused sweep runs on the SSE engines only
	const unsigned int n0 = std::lower_bound(m_VoltX.begin(), m_VoltX.end(), x0) - m_VoltX.begin();
	const unsigned int n1 = std::lower_bound(m_VoltX.begin(), m_VoltX.end(), x1) - m_VoltX.begin();
	if (n0<n1)
		Apply2VoltagesImpl<Engine_sse>((Engine_sse*)m_Eng, m_VoltOrder.data(), n0, n1);
}

template <typename EngType>
void Engine_Ext_Excitation::Apply2CurrentImpl(EngType* eng, const unsigned int* order, unsigned int n0, unsigned int n1)
{
	//soft current excitation here (H-field excite)
	int exc_pos;
	unsigned int ny;
	unsigned int pos[3];
	int numTS = m_Eng->GetNumberOfTimesteps();
	unsigned int length = m_Op_Exc->m_Exc->GetLength();
	FDTD_FLOAT* exc_curr =  m_Op_Exc->m_Exc->GetCurrentSignal();

	int p = numTS+1;
	if (m_Op_Exc->m_Exc->GetSignalPeriod()>0)
		p = int(m_Op_Exc->m_Exc->GetSignalPeriod()/m_Op_Exc->m_Exc->GetTimestep());

	for (unsigned int k=n0; k<n1; ++k)
	{
		const unsigned int n = order ? order[k] : k;
		exc_pos = numTS - (int)m_Op_Exc->Curr_delay[n];
		exc_pos *= (exc_pos>0);
		exc_pos %= p;
		exc_pos *= (exc_pos<(int)length);
		ny = m_Op_Exc->Curr_dir[n];
		pos[0]=m_Op_Exc->Curr_index[0][n];
		pos[1]=m_Op_Exc->Curr_index[1][n];
		pos[2]=m_Op_Exc->Curr_index[2][n];
		eng->EngType::SetCurr(ny,pos, eng->EngType::GetCurr(ny,pos) + m_Op_Exc->Curr_amp[n]*exc_curr[exc_pos]);
	}
}

void Engine_Ext_Excitation::Apply2Current()
{
	ENG_DISPATCH_ARGS(Apply2CurrentImpl, (const unsigned int*)NULL, 0u, m_Op_Exc->Curr_Count);
}

void Engine_Ext_Excitation::Apply2CurrentLines(unsigned int x0, unsigned int x1)
{
	const unsigned int n0 = std::lower_bound(m_CurrX.begin(), m_CurrX.end(), x0) - m_CurrX.begin();
	const unsigned int n1 = std::lower_bound(m_CurrX.begin(), m_CurrX.end(), x1) - m_CurrX.begin();
	if (n0<n1)
		Apply2CurrentImpl<Engine_sse>((Engine_sse*)m_Eng, m_CurrOrder.data(), n0, n1);
}
