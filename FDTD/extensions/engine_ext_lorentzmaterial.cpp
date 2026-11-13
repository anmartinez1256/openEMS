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

#include "engine_ext_lorentzmaterial.h"
#include "operator_ext_lorentzmaterial.h"
#include "FDTD/engine_sse.h"
#include "tools/global.h"
#include <cstdlib>
#include <iostream>
#include <new>

Engine_Ext_LorentzMaterial::Engine_Ext_LorentzMaterial(Operator_Ext_LorentzMaterial* op_ext_lorentz) : Engine_Ext_Dispersive(op_ext_lorentz)
{
	m_Op_Ext_Lor = op_ext_lorentz;
	m_Order = m_Op_Ext_Lor->GetDispersionOrder();
	int order = m_Op_Ext_Lor->m_Order;

	curr_Lor_ADE = new FDTD_FLOAT**[order];
	volt_Lor_ADE = new FDTD_FLOAT**[order];
	for (int o=0;o<order;++o)
	{
		curr_Lor_ADE[o] = new FDTD_FLOAT*[3];
		volt_Lor_ADE[o] = new FDTD_FLOAT*[3];
		for (int n=0; n<3; ++n)
		{
			if (m_Op_Ext_Lor->m_curr_Lor_ADE_On[o]==true)
			{
				curr_Lor_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Lor->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count[o]; ++i)
					curr_Lor_ADE[o][n][i]=0.0;
			}
			else
				curr_Lor_ADE[o][n] = NULL;

			if (m_Op_Ext_Lor->m_volt_Lor_ADE_On[o]==true)
			{
				volt_Lor_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Lor->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count[o]; ++i)
					volt_Lor_ADE[o][n][i]=0.0;
			}
			else
				volt_Lor_ADE[o][n] = NULL;
		}
	}
}

Engine_Ext_LorentzMaterial::~Engine_Ext_LorentzMaterial()
{
	if (curr_Lor_ADE==NULL && volt_Lor_ADE==NULL)
		return;

	for (int o=0;o<m_Op_Ext_Lor->m_Order;++o)
	{
		for (int n=0; n<3; ++n)
		{
			delete[] curr_Lor_ADE[o][n];
			delete[] volt_Lor_ADE[o][n];
		}
		delete[] curr_Lor_ADE[o];
		delete[] volt_Lor_ADE[o];
	}
	delete[] curr_Lor_ADE;
	curr_Lor_ADE=NULL;

	delete[] volt_Lor_ADE;
	volt_Lor_ADE=NULL;
}

void Engine_Ext_LorentzMaterial::SetupPacked()
{
	m_PV = PackedSide();
	m_PI = PackedSide();
#if defined(__GNUC__) || defined(__clang__)
	if (!m_VoltMerged && !m_CurrMerged)
		return;
	if (!SetupGroups())
		return;
	Operator_Ext_LorentzMaterial* op = m_Op_Ext_Lor;
	PackRefusal rv, ri;
	uint64_t nv, ni, uv, ui;
	const bool v = SetupSide(m_PV, m_VoltMerged, m_VoltOrders, volt_ADE, volt_Lor_ADE, op->v_int_ADE, op->v_ext_ADE,
							 op->v_Lor_ADE, op->m_volt_Lor_ADE_On, rv, nv, uv);
	const bool i = SetupSide(m_PI, m_CurrMerged, m_CurrOrders, curr_ADE, curr_Lor_ADE, op->i_int_ADE, op->i_ext_ADE,
							 op->i_Lor_ADE, op->m_curr_Lor_ADE_On, ri, ni, ui);
	// one line for the extension: a side refused by the memory limit or a failed allocation, else OPENEMS_ADE_PACKED=0
	{
		const bool mv = (rv==PACK_LIMIT) || (rv==PACK_ALLOC);
		const bool mi = (ri==PACK_LIMIT) || (ri==PACK_ALLOC);
		if (mv || mi)
			ReportPackRefusal(mv ? rv : ri, mv && mi ? " for voltages and currents" : (mv ? " for voltages" : " for currents"),
							  (mv ? nv : 0) + (mi ? ni : 0), mv ? uv : ui);
		else if ((rv==PACK_OFF) || (ri==PACK_OFF))
			ReportPackRefusal(PACK_OFF, "", 0, 0);
	}
	if (!v && !i)
	{
		ClearGroups();
		return;
	}
	if (g_settings.GetVerboseLevel()>1)
		std::cout << GetExtensionName() << ": packed hooks, " << m_GroupX.size() << " groups of " << m_Op_Ext_Lor->m_LM_Count.at(0)
				  << " positions, voltages " << (v ? m_PV.orders.size() : 0) << " orders, currents " << (i ? m_PI.orders.size() : 0)
				  << " orders, " << (m_PV.block.size()+m_PI.block.size())*sizeof(float)/1048576.0 << " MiB" << std::endl;
#endif
}

bool Engine_Ext_LorentzMaterial::SetupSide(PackedSide& s, bool merged, const std::vector<int>& orders, FDTD_FLOAT*** ade,
	FDTD_FLOAT*** lor, FDTD_FLOAT*** c_int, FDTD_FLOAT*** c_ext, FDTD_FLOAT*** c_lor, const bool* lor_on,
	PackRefusal& r, uint64_t& need, uint64_t& used)
{
	s = PackedSide();
	r = PACK_NOT_ELIGIBLE;
	need = 0;
	used = 0;
	// the groups are built on the positions of order 0; MergeOrders() made sure every merged order has the same ones
	if (!merged || orders.empty() || (m_FloatIdx[orders[0]]!=m_FloatIdx[0]))
		return false;
	if (PackedOff())
	{
		r = PACK_OFF;
		return false;
	}
	const size_t groups = m_GroupX.size();
	const unsigned int no = orders.size();
	const unsigned int stride = 3*no*20;
	// the limit counts every packed block of this engine: the other extensions' and this one's other side (s is empty)
	need = (uint64_t)groups*stride*sizeof(float);
	used = PackedBytesOfOthers() + PackedBytes();
	if (used+need > 2*FieldBytes())
	{
		r = PACK_LIMIT;
		return false;
	}
	try
	{
		s.block.assign(groups*stride, 0.0f);
	}
	catch (std::bad_alloc&)
	{
		s = PackedSide();
		r = PACK_ALLOC;
		return false;
	}
	r = PACK_OK;
	s.stride = stride;
	s.orders = orders;
	for (int o : orders)
		s.lor.push_back(lor_on[o] ? 1 : 0);
	// coefficients, and the current state (zero before the first step) of every used lane
	for (size_t g=0; g<groups; ++g)
		for (unsigned int l=0; l<m_GroupLanes[g]; ++l)
		{
			const unsigned int k = m_GroupK[g]+l;   // the position index of every merged order (in x order)
			for (unsigned int n=0; n<3; ++n)
				for (unsigned int m=0; m<no; ++m)
				{
					const int o = orders[m];
					float* c = &s.block[g*stride + (size_t)(n*no+m)*20];
					c[l] = ade[o][n][k];
					c[8+l] = c_int[o][n][k];
					c[12+l] = c_ext[o][n][k];
					if (lor_on[o])
					{
						c[4+l] = lor[o][n][k];
						c[16+l] = c_lor[o][n][k];
					}
				}
		}
	s.on = true;
	return true;
}

#if defined(__GNUC__) || defined(__clang__)
// the operations of PreMerged() per lane
void Engine_Ext_LorentzMaterial::PrePacked(const float* __restrict field, PackedSide& s, unsigned int x0, unsigned int x1)
{
	unsigned int g0, g1;
	GroupRange(x0, x1, g0, g1);
	const unsigned int no = s.orders.size();
	for (unsigned int g=g0; g<g1; ++g)
	{
		float* __restrict b = &s.block[(size_t)g*s.stride];
		const unsigned int* __restrict f = &m_GroupFi[12*(size_t)g];
		for (unsigned int n=0; n<3; ++n)
		{
			const unsigned int* fn = f + 4*n;
			const v4sf v = {field[fn[0]], field[fn[1]], field[fn[2]], field[fn[3]]};
			for (unsigned int m=0; m<no; ++m)
			{
				float* __restrict c = b + (n*no+m)*20;
				v4sf a = Load4(c);
				if (s.lor[m])
				{
					v4sf l = Load4(c+4);
					l += Load4(c+16)*a;
					a *= Load4(c+8);
					a += Load4(c+12)*(v-l);
					Store4(c+4, l);
				}
				else
				{
					a *= Load4(c+8);
					a += Load4(c+12)*v;
				}
				Store4(c, a);
			}
		}
	}
}

// the operations of ApplyMerged() per lane: field -= ADE of every merged order, in order (only the used lanes are written)
void Engine_Ext_LorentzMaterial::ApplyPacked(float* __restrict field, const PackedSide& s, unsigned int x0, unsigned int x1)
{
	unsigned int g0, g1;
	GroupRange(x0, x1, g0, g1);
	const unsigned int no = s.orders.size();
	for (unsigned int g=g0; g<g1; ++g)
	{
		const float* __restrict b = &s.block[(size_t)g*s.stride];
		const unsigned int* __restrict f = &m_GroupFi[12*(size_t)g];
		const unsigned int lanes = m_GroupLanes[g];
		for (unsigned int n=0; n<3; ++n)
		{
			const unsigned int* fn = f + 4*n;
			const float* c = b + n*no*20;
			if (lanes==4)
			{
				v4sf v = {field[fn[0]], field[fn[1]], field[fn[2]], field[fn[3]]};
				for (unsigned int m=0; m<no; ++m)
					v = v - Load4(c + m*20);
				field[fn[0]] = v[0];
				field[fn[1]] = v[1];
				field[fn[2]] = v[2];
				field[fn[3]] = v[3];
			}
			else
				for (unsigned int l=0; l<lanes; ++l)
				{
					float v = field[fn[l]];
					for (unsigned int m=0; m<no; ++m)
						v = v - c[m*20+l];
					field[fn[l]] = v;
				}
		}
	}
}
#else
// without the vector extensions SetupPacked() never packs (m_PV.on, m_PI.on stay false), so these are never called
void Engine_Ext_LorentzMaterial::PrePacked(const float* field, PackedSide& s, unsigned int x0, unsigned int x1)
{UNUSED(field); UNUSED(s); UNUSED(x0); UNUSED(x1);}
void Engine_Ext_LorentzMaterial::ApplyPacked(float* field, const PackedSide& s, unsigned int x0, unsigned int x1)
{UNUSED(field); UNUSED(s); UNUSED(x0); UNUSED(x1);}
#endif

void Engine_Ext_LorentzMaterial::Apply2Voltages()
{
	if (m_PV.on)
		return ApplyPacked((float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data(), m_PV, 0, ~0u);
	Engine_Ext_Dispersive::Apply2Voltages();
}

void Engine_Ext_LorentzMaterial::Apply2Current()
{
	if (m_PI.on)
		return ApplyPacked((float*)((Engine_sse*)m_Eng)->f4_curr_ptr->data(), m_PI, 0, ~0u);
	Engine_Ext_Dispersive::Apply2Current();
}

void Engine_Ext_LorentzMaterial::Apply2VoltagesLines(unsigned int x0, unsigned int x1)
{
	if (m_PV.on)
		return ApplyPacked((float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data(), m_PV, x0, x1);
	Engine_Ext_Dispersive::Apply2VoltagesLines(x0, x1);
}

void Engine_Ext_LorentzMaterial::Apply2CurrentLines(unsigned int x0, unsigned int x1)
{
	if (m_PI.on)
		return ApplyPacked((float*)((Engine_sse*)m_Eng)->f4_curr_ptr->data(), m_PI, x0, x1);
	Engine_Ext_Dispersive::Apply2CurrentLines(x0, x1);
}

unsigned int Engine_Ext_LorentzMaterial::GetUsedHooks() const
{
	OPENEMS_HOOKS_DECLARED_BY(Engine_Ext_LorentzMaterial);
	return DispersiveHooks() | (AnyVoltADE() ? PRE_VOLT : 0) | (AnyCurrADE() ? PRE_CURR : 0);
}

void Engine_Ext_LorentzMaterial::PreMerged(const float* __restrict field, FDTD_FLOAT*** ade, FDTD_FLOAT*** lor,
	FDTD_FLOAT*** c_int, FDTD_FLOAT*** c_ext, FDTD_FLOAT*** c_lor, const bool* lor_on, const std::vector<int>& orders,
	unsigned int x0, unsigned int x1)
{
	unsigned int k0, k1;
	LineRange(orders[0], x0, x1, k0, k1);
	const unsigned int* __restrict fi = m_FloatIdx[orders[0]].data();
	const size_t no = orders.size();
	unsigned int k = k0;
#if defined(__GNUC__) || defined(__clang__)
	for (; k+4<=k1; k+=4)
		for (unsigned int n=0; n<3; ++n)
		{
			const unsigned int* f = fi + 3*(size_t)k + n;
			const v4sf v = {field[f[0]], field[f[3]], field[f[6]], field[f[9]]};
			for (size_t m=0; m<no; ++m)
			{
				const int o = orders[m];
				v4sf a = Load4(ade[o][n]+k);
				if (lor_on[o])
				{
					v4sf l = Load4(lor[o][n]+k);
					l += Load4(c_lor[o][n]+k)*a;
					a *= Load4(c_int[o][n]+k);
					a += Load4(c_ext[o][n]+k)*(v-l);
					Store4(lor[o][n]+k, l);
				}
				else
				{
					a *= Load4(c_int[o][n]+k);
					a += Load4(c_ext[o][n]+k)*v;
				}
				Store4(ade[o][n]+k, a);
			}
		}
#endif
	for (; k<k1; ++k)
		for (unsigned int n=0; n<3; ++n)
		{
			const float v = field[fi[3*k+n]];
			for (size_t m=0; m<no; ++m)
			{
				const int o = orders[m];
				if (lor_on[o])
				{
					lor[o][n][k]+=c_lor[o][n][k]*ade[o][n][k];
					ade[o][n][k] *= c_int[o][n][k];
					ade[o][n][k] += c_ext[o][n][k] * (v-lor[o][n][k]);
				}
				else
				{
					ade[o][n][k] *= c_int[o][n][k];
					ade[o][n][k] += c_ext[o][n][k] * v;
				}
			}
		}
}

// the same per-position operations as DoPreVoltageUpdatesImpl, for the positions on x-lines [x0,x1)
void Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	const float* __restrict volt = (const float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data();
	if (m_PV.on)
		return PrePacked(volt, m_PV, x0, x1);
	if (m_VoltMerged)
		return PreMerged(volt, volt_ADE, volt_Lor_ADE, m_Op_Ext_Lor->v_int_ADE, m_Op_Ext_Lor->v_ext_ADE,
						 m_Op_Ext_Lor->v_Lor_ADE, m_Op_Ext_Lor->m_volt_Lor_ADE_On, m_VoltOrders, x0, x1);
	for (int o=0;o<m_Order;++o)
	{
		if (m_Op_Ext_Lor->m_volt_ADE_On[o]==false) continue;
		unsigned int k0, k1;
		LineRange(o, x0, x1, k0, k1);
		const unsigned int* __restrict idx = m_SortedIdx[o].data();
		const unsigned int* __restrict fi = m_FloatIdx[o].data();
		FDTD_FLOAT* const* ade = volt_ADE[o];
		FDTD_FLOAT* const* v_int = m_Op_Ext_Lor->v_int_ADE[o];
		FDTD_FLOAT* const* v_ext = m_Op_Ext_Lor->v_ext_ADE[o];
		if (m_Op_Ext_Lor->m_volt_Lor_ADE_On[o])
		{
			FDTD_FLOAT* const* lor = volt_Lor_ADE[o];
			FDTD_FLOAT* const* v_lor = m_Op_Ext_Lor->v_Lor_ADE[o];
			for (unsigned int k=k0; k<k1; ++k)
			{
				const unsigned int i = idx[k];
				for (unsigned int n=0; n<3; ++n)
				{
					lor[n][i]+=v_lor[n][i]*ade[n][i];
					ade[n][i] *= v_int[n][i];
					ade[n][i] += v_ext[n][i] * (volt[fi[3*k+n]]-lor[n][i]);
				}
			}
		}
		else
		{
			for (unsigned int k=k0; k<k1; ++k)
			{
				const unsigned int i = idx[k];
				for (unsigned int n=0; n<3; ++n)
				{
					ade[n][i] *= v_int[n][i];
					ade[n][i] += v_ext[n][i] * volt[fi[3*k+n]];
				}
			}
		}
	}
}

void Engine_Ext_LorentzMaterial::DoPreCurrentUpdatesLines(unsigned int x0, unsigned int x1)
{
	const float* __restrict curr = (const float*)((Engine_sse*)m_Eng)->f4_curr_ptr->data();
	if (m_PI.on)
		return PrePacked(curr, m_PI, x0, x1);
	if (m_CurrMerged)
		return PreMerged(curr, curr_ADE, curr_Lor_ADE, m_Op_Ext_Lor->i_int_ADE, m_Op_Ext_Lor->i_ext_ADE,
						 m_Op_Ext_Lor->i_Lor_ADE, m_Op_Ext_Lor->m_curr_Lor_ADE_On, m_CurrOrders, x0, x1);
	for (int o=0;o<m_Order;++o)
	{
		if (m_Op_Ext_Lor->m_curr_ADE_On[o]==false) continue;
		unsigned int k0, k1;
		LineRange(o, x0, x1, k0, k1);
		const unsigned int* __restrict idx = m_SortedIdx[o].data();
		const unsigned int* __restrict fi = m_FloatIdx[o].data();
		FDTD_FLOAT* const* ade = curr_ADE[o];
		FDTD_FLOAT* const* i_int = m_Op_Ext_Lor->i_int_ADE[o];
		FDTD_FLOAT* const* i_ext = m_Op_Ext_Lor->i_ext_ADE[o];
		if (m_Op_Ext_Lor->m_curr_Lor_ADE_On[o])
		{
			FDTD_FLOAT* const* lor = curr_Lor_ADE[o];
			FDTD_FLOAT* const* i_lor = m_Op_Ext_Lor->i_Lor_ADE[o];
			for (unsigned int k=k0; k<k1; ++k)
			{
				const unsigned int i = idx[k];
				for (unsigned int n=0; n<3; ++n)
				{
					lor[n][i]+=i_lor[n][i]*ade[n][i];
					ade[n][i] *= i_int[n][i];
					ade[n][i] += i_ext[n][i] * (curr[fi[3*k+n]]-lor[n][i]);
				}
			}
		}
		else
		{
			for (unsigned int k=k0; k<k1; ++k)
			{
				const unsigned int i = idx[k];
				for (unsigned int n=0; n<3; ++n)
				{
					ade[n][i] *= i_int[n][i];
					ade[n][i] += i_ext[n][i] * curr[fi[3*k+n]];
				}
			}
		}
	}
}

template <typename EngType>
void Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesImpl(EngType* eng)
{
	for (int o=0;o<m_Order;++o)
	{
		if (m_Op_Ext_Lor->m_volt_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Lor->m_LM_pos[o];

		if (m_Op_Ext_Lor->m_volt_Lor_ADE_On[o])
		{
			for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count.at(o); ++i)
			{
				volt_Lor_ADE[o][0][i]+=m_Op_Ext_Lor->v_Lor_ADE[o][0][i]*volt_ADE[o][0][i];
				volt_ADE[o][0][i] *= m_Op_Ext_Lor->v_int_ADE[o][0][i];
				volt_ADE[o][0][i] += m_Op_Ext_Lor->v_ext_ADE[o][0][i] * (eng->EngType::GetVolt(0,pos[0][i],pos[1][i],pos[2][i])-volt_Lor_ADE[o][0][i]);

				volt_Lor_ADE[o][1][i]+=m_Op_Ext_Lor->v_Lor_ADE[o][1][i]*volt_ADE[o][1][i];
				volt_ADE[o][1][i] *= m_Op_Ext_Lor->v_int_ADE[o][1][i];
				volt_ADE[o][1][i] += m_Op_Ext_Lor->v_ext_ADE[o][1][i] * (eng->EngType::GetVolt(1,pos[0][i],pos[1][i],pos[2][i])-volt_Lor_ADE[o][1][i]);

				volt_Lor_ADE[o][2][i]+=m_Op_Ext_Lor->v_Lor_ADE[o][2][i]*volt_ADE[o][2][i];
				volt_ADE[o][2][i] *= m_Op_Ext_Lor->v_int_ADE[o][2][i];
				volt_ADE[o][2][i] += m_Op_Ext_Lor->v_ext_ADE[o][2][i] * (eng->EngType::GetVolt(2,pos[0][i],pos[1][i],pos[2][i])-volt_Lor_ADE[o][2][i]);
			}
		}
		else
		{
			for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count.at(o); ++i)
			{
				volt_ADE[o][0][i] *= m_Op_Ext_Lor->v_int_ADE[o][0][i];
				volt_ADE[o][0][i] += m_Op_Ext_Lor->v_ext_ADE[o][0][i] * eng->EngType::GetVolt(0,pos[0][i],pos[1][i],pos[2][i]);

				volt_ADE[o][1][i] *= m_Op_Ext_Lor->v_int_ADE[o][1][i];
				volt_ADE[o][1][i] += m_Op_Ext_Lor->v_ext_ADE[o][1][i] * eng->EngType::GetVolt(1,pos[0][i],pos[1][i],pos[2][i]);

				volt_ADE[o][2][i] *= m_Op_Ext_Lor->v_int_ADE[o][2][i];
				volt_ADE[o][2][i] += m_Op_Ext_Lor->v_ext_ADE[o][2][i] * eng->EngType::GetVolt(2,pos[0][i],pos[1][i],pos[2][i]);
			}
		}
	}
}

void Engine_Ext_LorentzMaterial::DoPreVoltageUpdates()
{
	if (m_PV.on)
		return PrePacked((const float*)((Engine_sse*)m_Eng)->f4_volt_ptr->data(), m_PV, 0, ~0u);
	ENG_DISPATCH(DoPreVoltageUpdatesImpl);
}

template <typename EngType>
void Engine_Ext_LorentzMaterial::DoPreCurrentUpdatesImpl(EngType* eng)
{
	for (int o=0;o<m_Order;++o)
	{
		if (m_Op_Ext_Lor->m_curr_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Lor->m_LM_pos[o];

		if (m_Op_Ext_Lor->m_curr_Lor_ADE_On[o])
		{
			for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count.at(o); ++i)
			{
				curr_Lor_ADE[o][0][i]+=m_Op_Ext_Lor->i_Lor_ADE[o][0][i]*curr_ADE[o][0][i];
				curr_ADE[o][0][i] *= m_Op_Ext_Lor->i_int_ADE[o][0][i];
				curr_ADE[o][0][i] += m_Op_Ext_Lor->i_ext_ADE[o][0][i] * (eng->EngType::GetCurr(0,pos[0][i],pos[1][i],pos[2][i])-curr_Lor_ADE[o][0][i]);

				curr_Lor_ADE[o][1][i]+=m_Op_Ext_Lor->i_Lor_ADE[o][1][i]*curr_ADE[o][1][i];
				curr_ADE[o][1][i] *= m_Op_Ext_Lor->i_int_ADE[o][1][i];
				curr_ADE[o][1][i] += m_Op_Ext_Lor->i_ext_ADE[o][1][i] * (eng->EngType::GetCurr(1,pos[0][i],pos[1][i],pos[2][i])-curr_Lor_ADE[o][1][i]);

				curr_Lor_ADE[o][2][i]+=m_Op_Ext_Lor->i_Lor_ADE[o][2][i]*curr_ADE[o][2][i];
				curr_ADE[o][2][i] *= m_Op_Ext_Lor->i_int_ADE[o][2][i];
				curr_ADE[o][2][i] += m_Op_Ext_Lor->i_ext_ADE[o][2][i] * (eng->EngType::GetCurr(2,pos[0][i],pos[1][i],pos[2][i])-curr_Lor_ADE[o][2][i]);
			}
		}
		else
		{
			for (unsigned int i=0; i<m_Op_Ext_Lor->m_LM_Count.at(o); ++i)
			{
				curr_ADE[o][0][i] *= m_Op_Ext_Lor->i_int_ADE[o][0][i];
				curr_ADE[o][0][i] += m_Op_Ext_Lor->i_ext_ADE[o][0][i] * eng->EngType::GetCurr(0,pos[0][i],pos[1][i],pos[2][i]);

				curr_ADE[o][1][i] *= m_Op_Ext_Lor->i_int_ADE[o][1][i];
				curr_ADE[o][1][i] += m_Op_Ext_Lor->i_ext_ADE[o][1][i] * eng->EngType::GetCurr(1,pos[0][i],pos[1][i],pos[2][i]);

				curr_ADE[o][2][i] *= m_Op_Ext_Lor->i_int_ADE[o][2][i];
				curr_ADE[o][2][i] += m_Op_Ext_Lor->i_ext_ADE[o][2][i] * eng->EngType::GetCurr(2,pos[0][i],pos[1][i],pos[2][i]);
			}
		}
	}
}

void Engine_Ext_LorentzMaterial::DoPreCurrentUpdates()
{
	if (m_PI.on)
		return PrePacked((const float*)((Engine_sse*)m_Eng)->f4_curr_ptr->data(), m_PI, 0, ~0u);
	ENG_DISPATCH(DoPreCurrentUpdatesImpl);
}
