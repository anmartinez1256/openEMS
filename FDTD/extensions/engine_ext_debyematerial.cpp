/*
*	Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "engine_ext_debyematerial.h"
#include "operator_ext_debyematerial.h"
#include "FDTD/engine_sse.h"
#include "tools/global.h"
#include <cstdlib>
#include <iostream>
#include <new>

Engine_Ext_DebyeMaterial::Engine_Ext_DebyeMaterial(Operator_Ext_DebyeMaterial* op_ext_debye) : Engine_Ext_Dispersive(op_ext_debye)
{
	m_Op_Ext_Deb = op_ext_debye;
	m_Packed = false;
	m_BlockStride = 0;
	// the base allocated volt_ADE[0], which carries the joint correction
	unsigned int count = m_Op_Ext_Deb->m_LM_Count.at(0);

	volt_pole_ADE = new FDTD_FLOAT**[m_Op_Ext_Deb->m_PoleCount];
	for (int o=0; o<m_Op_Ext_Deb->m_PoleCount; ++o)
	{
		volt_pole_ADE[o] = new FDTD_FLOAT*[3];
		for (int n=0; n<3; ++n)
		{
			volt_pole_ADE[o][n] = new FDTD_FLOAT[count];
			for (unsigned int i=0; i<count; ++i)
				volt_pole_ADE[o][n][i] = 0.0;
		}
	}

	volt_pre_ADE = new FDTD_FLOAT*[3];
	for (int n=0; n<3; ++n)
	{
		volt_pre_ADE[n] = new FDTD_FLOAT[count];
		for (unsigned int i=0; i<count; ++i)
			volt_pre_ADE[n][i] = 0.0;
	}
}

Engine_Ext_DebyeMaterial::~Engine_Ext_DebyeMaterial()
{
	if (volt_pole_ADE!=NULL)
	{
		for (int o=0; o<m_Op_Ext_Deb->m_PoleCount; ++o)
		{
			for (int n=0; n<3; ++n)
				delete[] volt_pole_ADE[o][n];
			delete[] volt_pole_ADE[o];
		}
		delete[] volt_pole_ADE;
		volt_pole_ADE = NULL;
	}

	if (volt_pre_ADE!=NULL)
		for (int n=0; n<3; ++n)
			delete[] volt_pre_ADE[n];
	delete[] volt_pre_ADE;
	volt_pre_ADE = NULL;
}

template <typename EngType>
void Engine_Ext_DebyeMaterial::DoPreVoltageUpdatesImpl(EngType* eng)
{
	unsigned int **pos = m_Op_Ext_Deb->m_LM_pos[0];
	int poles = m_Op_Ext_Deb->m_PoleCount;

	for (unsigned int i=0; i<m_Op_Ext_Deb->m_LM_Count.at(0); ++i)
	{
		for (int n=0; n<3; ++n)
		{
			FDTD_FLOAT volt = eng->EngType::GetVolt(n,pos[0][i],pos[1][i],pos[2][i]);
			FDTD_FLOAT pre = 0.0;
			for (int o=0; o<poles; ++o)
			{
				// as much of the pole's new state as V^n already fixes
				FDTD_FLOAT W = m_Op_Ext_Deb->v_relax_ADE[o][n][i]*volt_pole_ADE[o][n][i]
				             + m_Op_Ext_Deb->v_drive_ADE[o][n][i]*volt;
				pre += W - volt_pole_ADE[o][n][i];
				volt_pole_ADE[o][n][i] = W;
			}
			volt_pre_ADE[n][i] = pre;
		}
	}
}

void Engine_Ext_DebyeMaterial::DoPreVoltageUpdates()
{
	if (m_Packed)
		return PrePacked(0, ~0u);
	ENG_DISPATCH(DoPreVoltageUpdatesImpl);
}

template <typename EngType>
void Engine_Ext_DebyeMaterial::DoPostVoltageUpdatesImpl(EngType* eng)
{
	unsigned int **pos = m_Op_Ext_Deb->m_LM_pos[0];
	int poles = m_Op_Ext_Deb->m_PoleCount;

	for (unsigned int i=0; i<m_Op_Ext_Deb->m_LM_Count.at(0); ++i)
	{
		for (int n=0; n<3; ++n)
		{
			FDTD_FLOAT v_raw = eng->EngType::GetVolt(n,pos[0][i],pos[1][i],pos[2][i]);
			// solved for the V^(n+1) the poles themselves depend on
			FDTD_FLOAT volt = (v_raw - volt_pre_ADE[n][i]) * m_Op_Ext_Deb->v_solve_ADE[n][i];
			volt_ADE[0][n][i] = v_raw - volt;   // what Apply2Voltages() subtracts
			// V^(n+1) is now known, so the pole states can be finished
			for (int o=0; o<poles; ++o)
				volt_pole_ADE[o][n][i] += m_Op_Ext_Deb->v_drive_ADE[o][n][i]*volt;
		}
	}
}

void Engine_Ext_DebyeMaterial::DoPostVoltageUpdates()
{
	if (m_Packed)
		return PostPacked(0, ~0u);
	ENG_DISPATCH(DoPostVoltageUpdatesImpl);
}

void Engine_Ext_DebyeMaterial::Apply2Voltages()
{
	if (m_Packed)
		return ApplyPacked(0, ~0u);
	Engine_Ext_Dispersive::Apply2Voltages();
}

void Engine_Ext_DebyeMaterial::Apply2VoltagesLines(unsigned int x0, unsigned int x1)
{
	if (m_Packed)
		return ApplyPacked(x0, x1);
	Engine_Ext_Dispersive::Apply2VoltagesLines(x0, x1);
}

void Engine_Ext_DebyeMaterial::SetupPacked()
{
	m_Packed = false;
	std::vector<float>().swap(m_Block);
	const char* env = getenv("OPENEMS_ADE_PACKED");
	if (env && *env && (atoi(env)==0))
		return;
	const unsigned int P = m_Op_Ext_Deb->m_PoleCount;
	// Debye has voltage ADE of order 0 only; the packed hooks replace exactly those
	if ((P<1) || (m_Op_Ext_Deb->m_Order!=1) || !m_Op_Ext_Deb->m_volt_ADE_On[0] || m_Op_Ext_Deb->m_curr_ADE_On[0])
		return;
	if (!SetupGroups())
		return;
	const size_t groups = m_GroupX.size();
	m_BlockStride = 3*(3*P+3)*4;
	if ((uint64_t)groups*m_BlockStride*sizeof(float) > 2*FieldBytes())
	{
		ClearGroups();
		return;
	}
	try
	{
		m_Block.assign((size_t)groups*m_BlockStride, 0.0f);
	}
	catch (std::bad_alloc&)
	{
		ClearGroups();
		return;
	}
	// coefficients, and the current state (zero before the first step) of every used lane
	for (size_t g=0; g<groups; ++g)
		for (unsigned int l=0; l<m_GroupLanes[g]; ++l)
		{
			const unsigned int i = m_GroupK[g]+l;   // = the original position index (SetupGroups: in x order)
			for (unsigned int n=0; n<3; ++n)
			{
				float* c = &m_Block[g*m_BlockStride + (size_t)n*(3*P+3)*4];
				for (unsigned int o=0; o<P; ++o)
				{
					c[4*o+l] = volt_pole_ADE[o][n][i];
					c[4*(P+2+o)+l] = m_Op_Ext_Deb->v_relax_ADE[o][n][i];
					c[4*(2*P+2+o)+l] = m_Op_Ext_Deb->v_drive_ADE[o][n][i];
				}
				c[4*P+l] = volt_pre_ADE[n][i];
				c[4*(P+1)+l] = volt_ADE[0][n][i];
				c[4*(3*P+2)+l] = m_Op_Ext_Deb->v_solve_ADE[n][i];
			}
		}
	m_Packed = true;
	if (g_settings.GetVerboseLevel()>1)
		std::cout << GetExtensionName() << ": packed hooks, " << groups << " groups of " << m_Op_Ext_Deb->m_LM_Count.at(0)
				  << " positions, " << m_Block.size()*sizeof(float)/1048576.0 << " MiB" << std::endl;
}

// the operations of DoPreVoltageUpdatesLines() per lane: W = v_relax*W + v_drive*V, pre += W - W_old
void Engine_Ext_DebyeMaterial::PrePacked(unsigned int x0, unsigned int x1)
{
	const float* __restrict volt = (const float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data();
	unsigned int g0, g1;
	GroupRange(x0, x1, g0, g1);
	const unsigned int P = m_Op_Ext_Deb->m_PoleCount;
	const unsigned int cs = (3*P+3)*4;
	for (unsigned int g=g0; g<g1; ++g)
	{
		float* __restrict b = &m_Block[(size_t)g*m_BlockStride];
		const unsigned int* __restrict f = &m_GroupFi[12*(size_t)g];
		for (unsigned int n=0; n<3; ++n)
		{
			float* __restrict c = b + n*cs;
			const unsigned int* fn = f + 4*n;
			const v4sf v = {volt[fn[0]], volt[fn[1]], volt[fn[2]], volt[fn[3]]};
			v4sf pre = {0.0f, 0.0f, 0.0f, 0.0f};
			for (unsigned int o=0; o<P; ++o)
			{
				const v4sf w = Load4(c+4*o);
				const v4sf W = Load4(c+4*(P+2+o))*w + Load4(c+4*(2*P+2+o))*v;
				pre += W - w;
				Store4(c+4*o, W);
			}
			Store4(c+4*P, pre);
		}
	}
}

// the operations of DoPostVoltageUpdatesLines() per lane: V = (V_raw - pre)*v_solve, correction = V_raw - V, W += v_drive*V
void Engine_Ext_DebyeMaterial::PostPacked(unsigned int x0, unsigned int x1)
{
	const float* __restrict volt = (const float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data();
	unsigned int g0, g1;
	GroupRange(x0, x1, g0, g1);
	const unsigned int P = m_Op_Ext_Deb->m_PoleCount;
	const unsigned int cs = (3*P+3)*4;
	for (unsigned int g=g0; g<g1; ++g)
	{
		float* __restrict b = &m_Block[(size_t)g*m_BlockStride];
		const unsigned int* __restrict f = &m_GroupFi[12*(size_t)g];
		for (unsigned int n=0; n<3; ++n)
		{
			float* __restrict c = b + n*cs;
			const unsigned int* fn = f + 4*n;
			const v4sf v_raw = {volt[fn[0]], volt[fn[1]], volt[fn[2]], volt[fn[3]]};
			const v4sf v_n = (v_raw - Load4(c+4*P)) * Load4(c+4*(3*P+2));
			Store4(c+4*(P+1), v_raw - v_n);
			for (unsigned int o=0; o<P; ++o)
				Store4(c+4*o, Load4(c+4*o) + Load4(c+4*(2*P+2+o))*v_n);
		}
	}
}

// the operation of Apply2VoltagesLines() per lane: V -= correction (only the used lanes are written)
void Engine_Ext_DebyeMaterial::ApplyPacked(unsigned int x0, unsigned int x1)
{
	float* __restrict volt = (float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data();
	unsigned int g0, g1;
	GroupRange(x0, x1, g0, g1);
	const unsigned int P = m_Op_Ext_Deb->m_PoleCount;
	const unsigned int cs = (3*P+3)*4;
	for (unsigned int g=g0; g<g1; ++g)
	{
		const float* __restrict b = &m_Block[(size_t)g*m_BlockStride];
		const unsigned int* __restrict f = &m_GroupFi[12*(size_t)g];
		const unsigned int lanes = m_GroupLanes[g];
		for (unsigned int n=0; n<3; ++n)
		{
			const float* c = b + n*cs + 4*(P+1);
			const unsigned int* fn = f + 4*n;
			if (lanes==4)
			{
				v4sf v = {volt[fn[0]], volt[fn[1]], volt[fn[2]], volt[fn[3]]};
				v = v - Load4(c);
				volt[fn[0]] = v[0];
				volt[fn[1]] = v[1];
				volt[fn[2]] = v[2];
				volt[fn[3]] = v[3];
			}
			else
				for (unsigned int l=0; l<lanes; ++l)
					volt[fn[l]] = volt[fn[l]] - c[l];
		}
	}
}

void Engine_Ext_DebyeMaterial::DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	if (m_Packed)
		return PrePacked(x0, x1);
	const float* __restrict volt = (const float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data();
	unsigned int k0, k1;
	LineRange(0, x0, x1, k0, k1);
	const unsigned int* __restrict idx = m_SortedIdx[0].data();
	const unsigned int* __restrict fi = m_FloatIdx[0].data();
	int poles = m_Op_Ext_Deb->m_PoleCount;

	for (unsigned int k=k0; k<k1; ++k)
	{
		const unsigned int i = idx[k];
		for (int n=0; n<3; ++n)
		{
			FDTD_FLOAT volt_n = volt[fi[3*(size_t)k+n]];
			FDTD_FLOAT pre = 0.0;
			for (int o=0; o<poles; ++o)
			{
				FDTD_FLOAT W = m_Op_Ext_Deb->v_relax_ADE[o][n][i]*volt_pole_ADE[o][n][i]
				             + m_Op_Ext_Deb->v_drive_ADE[o][n][i]*volt_n;
				pre += W - volt_pole_ADE[o][n][i];
				volt_pole_ADE[o][n][i] = W;
			}
			volt_pre_ADE[n][i] = pre;
		}
	}
}

void Engine_Ext_DebyeMaterial::DoPostVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	if (m_Packed)
		return PostPacked(x0, x1);
	const float* __restrict volt = (const float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data();
	unsigned int k0, k1;
	LineRange(0, x0, x1, k0, k1);
	const unsigned int* __restrict idx = m_SortedIdx[0].data();
	const unsigned int* __restrict fi = m_FloatIdx[0].data();
	int poles = m_Op_Ext_Deb->m_PoleCount;

	for (unsigned int k=k0; k<k1; ++k)
	{
		const unsigned int i = idx[k];
		for (int n=0; n<3; ++n)
		{
			FDTD_FLOAT v_raw = volt[fi[3*(size_t)k+n]];
			FDTD_FLOAT volt_n = (v_raw - volt_pre_ADE[n][i]) * m_Op_Ext_Deb->v_solve_ADE[n][i];
			volt_ADE[0][n][i] = v_raw - volt_n;
			for (int o=0; o<poles; ++o)
				volt_pole_ADE[o][n][i] += m_Op_Ext_Deb->v_drive_ADE[o][n][i]*volt_n;
		}
	}
}

// This used to inherit Engine_Ext_Dispersive's declaration (APPLY_VOLT | LINE_HOOKS), so the
// multithreaded engine never called the Pre/Post voltage hooks above and Debye cells ran as their base material.
// The Pre/Post hooks only read engine fields and write the extension's own state; the poles are applied by the
// inherited Apply2Voltages() (and its line hook).
unsigned int Engine_Ext_DebyeMaterial::GetUsedHooks() const
{
	OPENEMS_HOOKS_DECLARED_BY(Engine_Ext_DebyeMaterial);
	return DispersiveHooks() | PRE_VOLT | POST_VOLT;
}
