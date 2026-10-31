/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "engine_ext_dispersive.h"
#include "operator_ext_dispersive.h"
#include "FDTD/engine_sse.h"
#include <algorithm>

Engine_Ext_Dispersive::Engine_Ext_Dispersive(Operator_Ext_Dispersive* op_ext_disp) : Engine_Extension(op_ext_disp)
{
	m_Op_Ext_Disp = op_ext_disp;
	m_LinesReady = false;
	m_LinesChecked = false;
	int order = m_Op_Ext_Disp->m_Order;
	curr_ADE = new FDTD_FLOAT**[order];
	volt_ADE = new FDTD_FLOAT**[order];
	for (int o=0;o<order;++o)
	{
		curr_ADE[o] = new FDTD_FLOAT*[3];
		volt_ADE[o] = new FDTD_FLOAT*[3];
		for (int n=0; n<3; ++n)
		{
			if (m_Op_Ext_Disp->m_curr_ADE_On[o]==true)
			{
				curr_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Disp->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count[o]; ++i)
					curr_ADE[o][n][i]=0.0;
			}
			else
				curr_ADE[o][n] = NULL;
			if (m_Op_Ext_Disp->m_volt_ADE_On[o]==true)
			{
				volt_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Disp->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count[o]; ++i)
					volt_ADE[o][n][i]=0.0;
			}
			else
				volt_ADE[o][n] = NULL;
		}
	}
}

Engine_Ext_Dispersive::~Engine_Ext_Dispersive()
{
	if (curr_ADE==NULL && volt_ADE==NULL)
		return;

	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		for (int n=0; n<3; ++n)
		{
			delete[] curr_ADE[o][n];
			delete[] volt_ADE[o][n];
		}
		delete[] curr_ADE[o];
		delete[] volt_ADE[o];
	}
	delete[] curr_ADE;
	curr_ADE=NULL;

	delete[] volt_ADE;
	volt_ADE=NULL;
}

bool Engine_Ext_Dispersive::AnyVoltADE() const
{
	for (int o=0; o<m_Op_Ext_Disp->m_Order; ++o)
		if (m_Op_Ext_Disp->m_volt_ADE_On[o])
			return true;
	return false;
}

bool Engine_Ext_Dispersive::AnyCurrADE() const
{
	for (int o=0; o<m_Op_Ext_Disp->m_Order; ++o)
		if (m_Op_Ext_Disp->m_curr_ADE_On[o])
			return true;
	return false;
}

unsigned int Engine_Ext_Dispersive::GetUsedHooks() const
{
	return (AnyVoltADE() ? APPLY_VOLT : 0) | (AnyCurrADE() ? APPLY_CURR : 0) | (m_LinesReady ? LINE_HOOKS : 0);
}

void Engine_Ext_Dispersive::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);
	if (!m_LinesChecked && m_Eng)
		SetupLines();
}

void Engine_Ext_Dispersive::SetupLines()
{
	m_LinesChecked = true;
	m_LinesReady = false;
	if (m_Eng->GetType()!=Engine::SSE)
		return;
	Engine_sse* eng = (Engine_sse*)m_Eng;
	ArrayLib::ArrayENG<f4vector>* V = eng->f4_volt_ptr;
	ArrayLib::ArrayENG<f4vector>* I = eng->f4_curr_ptr;
	if ((V->stride(0)!=I->stride(0)) || (V->stride(1)!=I->stride(1)) || (V->stride(2)!=I->stride(2)) ||
		(V->stride(3)!=I->stride(3)) || ((uint64_t)V->size()*4 >= (1ull<<32)))
		return;
	const unsigned int nvec = V->extent(3);
	const int order = m_Op_Ext_Disp->m_Order;
	m_SortedIdx.assign(order, std::vector<unsigned int>());
	m_SortedX.assign(order, std::vector<unsigned int>());
	m_FloatIdx.assign(order, std::vector<unsigned int>());
	for (int o=0; o<order; ++o)
	{
		const unsigned int count = m_Op_Ext_Disp->m_LM_Count.at(o);
		unsigned int** pos = m_Op_Ext_Disp->m_LM_pos[o];
		std::vector<unsigned int>& idx = m_SortedIdx[o];
		idx.resize(count);
		for (unsigned int i=0; i<count; ++i)
			idx[i] = i;
		const unsigned int* x = pos[0];
		std::stable_sort(idx.begin(), idx.end(), [x](unsigned int a, unsigned int b) {return x[a]<x[b];});
		m_SortedX[o].resize(count);
		m_FloatIdx[o].resize((size_t)3*count);
		for (unsigned int k=0; k<count; ++k)
		{
			const unsigned int i = idx[k];
			m_SortedX[o][k] = x[i];
			for (unsigned int n=0; n<3; ++n)
				m_FloatIdx[o][3*(size_t)k+n] = (unsigned int)(4*(uint64_t)V->linearIndex({n, pos[0][i], pos[1][i], pos[2][i]%nvec}) + pos[2][i]/nvec);
		}
	}
	m_LinesReady = true;
}

void Engine_Ext_Dispersive::LineRange(int o, unsigned int x0, unsigned int x1, unsigned int& k0, unsigned int& k1) const
{
	const std::vector<unsigned int>& X = m_SortedX[o];
	k0 = std::lower_bound(X.begin(), X.end(), x0) - X.begin();
	k1 = std::lower_bound(X.begin(), X.end(), x1) - X.begin();
}

void Engine_Ext_Dispersive::Apply2VoltagesLines(unsigned int x0, unsigned int x1)
{
	float* __restrict volt = (float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data();
	for (int o=0; o<m_Op_Ext_Disp->m_Order; ++o)
	{
		if (m_Op_Ext_Disp->m_volt_ADE_On[o]==false) continue;
		unsigned int k0, k1;
		LineRange(o, x0, x1, k0, k1);
		const unsigned int* __restrict idx = m_SortedIdx[o].data();
		const unsigned int* __restrict fi = m_FloatIdx[o].data();
		const FDTD_FLOAT* __restrict ade0 = volt_ADE[o][0];
		const FDTD_FLOAT* __restrict ade1 = volt_ADE[o][1];
		const FDTD_FLOAT* __restrict ade2 = volt_ADE[o][2];
		for (unsigned int k=k0; k<k1; ++k)
		{
			const unsigned int i = idx[k];
			volt[fi[3*k+0]] = volt[fi[3*k+0]] - ade0[i];
			volt[fi[3*k+1]] = volt[fi[3*k+1]] - ade1[i];
			volt[fi[3*k+2]] = volt[fi[3*k+2]] - ade2[i];
		}
	}
}

void Engine_Ext_Dispersive::Apply2CurrentLines(unsigned int x0, unsigned int x1)
{
	float* __restrict curr = (float*)((Engine_sse*)m_Eng)->f4_curr_ptr->data();
	for (int o=0; o<m_Op_Ext_Disp->m_Order; ++o)
	{
		if (m_Op_Ext_Disp->m_curr_ADE_On[o]==false) continue;
		unsigned int k0, k1;
		LineRange(o, x0, x1, k0, k1);
		const unsigned int* __restrict idx = m_SortedIdx[o].data();
		const unsigned int* __restrict fi = m_FloatIdx[o].data();
		const FDTD_FLOAT* __restrict ade0 = curr_ADE[o][0];
		const FDTD_FLOAT* __restrict ade1 = curr_ADE[o][1];
		const FDTD_FLOAT* __restrict ade2 = curr_ADE[o][2];
		for (unsigned int k=k0; k<k1; ++k)
		{
			const unsigned int i = idx[k];
			curr[fi[3*k+0]] = curr[fi[3*k+0]] - ade0[i];
			curr[fi[3*k+1]] = curr[fi[3*k+1]] - ade1[i];
			curr[fi[3*k+2]] = curr[fi[3*k+2]] - ade2[i];
		}
	}
}

template <typename EngType>
void Engine_Ext_Dispersive::Apply2VoltagesImpl(EngType* eng)
{
	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		if (m_Op_Ext_Disp->m_volt_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];

		for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count.at(o); ++i)
		{
			eng->EngType::SetVolt(0,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVolt(0,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][0][i]
			);
			eng->EngType::SetVolt(1,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVolt(1,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][1][i]
			);
			eng->EngType::SetVolt(2,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVolt(2,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][2][i]
			);
		}
	}
}

void Engine_Ext_Dispersive::Apply2Voltages()
{
	ENG_DISPATCH(Apply2VoltagesImpl);
}

template <typename EngType>
void Engine_Ext_Dispersive::Apply2CurrentImpl(EngType* eng)
{
	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		if (m_Op_Ext_Disp->m_curr_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];

		for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count.at(o); ++i)
		{
			eng->EngType::SetCurr(0,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurr(0,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][0][i]
			);
			eng->EngType::SetCurr(1,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurr(1,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][1][i]
			);
			eng->EngType::SetCurr(2,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurr(2,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][2][i]
			);
		}
	}
}

void Engine_Ext_Dispersive::Apply2Current()
{
	ENG_DISPATCH(Apply2CurrentImpl);
}
