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

#include "engine_ext_upml.h"
#include "operator_ext_upml.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "tools/useful.h"
#include <cstring>
#include <algorithm>

Engine_Ext_UPML::Engine_Ext_UPML(Operator_Ext_UPML* op_ext) : Engine_Extension(op_ext)
{
	m_Op_UPML = op_ext;

	//this ABC extension should be executed first!
	m_Priority = ENG_EXT_PRIO_UPML;

	volt_flux.Init("volt_flux", m_Op_UPML->m_numLines);
	curr_flux.Init("curr_flux", m_Op_UPML->m_numLines);

	m_Vec = false;
	m_VecChecked = false;
	m_nvec = 0;

	SetNumberOfThreads(1);
}

Engine_Ext_UPML::~Engine_Ext_UPML()
{
}

void Engine_Ext_UPML::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);

	m_numX = AssignJobs2Threads(m_Op_UPML->m_numLines[0],m_NrThreads,false);
	m_start.resize(m_NrThreads,0);
	m_start.at(0)=0;
	for (size_t n=1; n<m_numX.size(); ++n)
		m_start.at(n) = m_start.at(n-1) + m_numX.at(n-1);

	if (!m_VecChecked && m_Eng)
		SetupVectorized();
	if (m_Vec)
	{
		// split the (x,y)-rows instead of the few x-lines of a PML slab
		m_RowNum = AssignJobs2Threads(m_Op_UPML->m_numLines[0]*m_Op_UPML->m_numLines[1], m_NrThreads, false);
		m_RowStart.assign(m_NrThreads, 0);
		for (size_t n=1; n<m_RowNum.size(); ++n)
			m_RowStart.at(n) = m_RowStart.at(n-1) + m_RowNum.at(n-1);
		m_RowNum.resize(m_NrThreads, 0);
	}
}

#if defined(__GNUC__) || defined(__clang__)
#define UPML_F4VECTOR 1
typedef int UPML_v4si __attribute__ ((vector_size (F4VECTOR_SIZE)));
//! lanes of a where the mask is set, lanes of b elsewhere
static inline v4sf UPML_Blend(const v4sf m, const v4sf a, const v4sf b)
{
	return (v4sf)(((UPML_v4si)a & (UPML_v4si)m) | ((UPML_v4si)b & ~(UPML_v4si)m));
}
#endif

void Engine_Ext_UPML::SetupVectorized()
{
	m_VecChecked = true;
	m_Vec = false;
#ifdef UPML_F4VECTOR
	if (m_Eng->GetType()!=Engine::SSE)
		return;
	Engine_sse* eng = (Engine_sse*)m_Eng;
	ArrayLib::ArrayENG<f4vector>* V = eng->f4_volt_ptr;
	ArrayLib::ArrayENG<f4vector>* I = eng->f4_curr_ptr;
	if ((V->stride(0)!=1) || (V->stride(3)!=3) || (I->stride(0)!=1) || (I->stride(3)!=3) ||
		(V->stride(1)!=I->stride(1)) || (V->stride(2)!=I->stride(2)))
		return;
	const unsigned int nvec = V->extent(3);
	const unsigned int Nz = m_Op_UPML->m_Op->GetNumberOfLines(2);
	if ((4*nvec<Nz) || (4*(nvec-1)>=Nz))
		return;
	// the f4vector lanes hold z = lane*nvec + v: only a PML over the full z range maps onto whole vectors
	if ((m_Op_UPML->m_StartPos[2]!=0) || (m_Op_UPML->m_numLines[2]!=Nz))
		return;

	m_nvec = nvec;
	const unsigned int NX = m_Op_UPML->m_numLines[0];
	const unsigned int NY = m_Op_UPML->m_numLines[1];
	const size_t size = (size_t)NX*NY*nvec*3;
	f4vector zero;
	zero.f[0] = zero.f[1] = zero.f[2] = zero.f[3] = 0;
	m_f4_vv.assign(size, zero);   m_f4_vvfo.assign(size, zero); m_f4_vvfn.assign(size, zero);
	m_f4_ii.assign(size, zero);   m_f4_iifo.assign(size, zero); m_f4_iifn.assign(size, zero);
	m_f4_volt_flux.assign(size, zero); m_f4_curr_flux.assign(size, zero);
	for (unsigned int x=0; x<NX; ++x)
		for (unsigned int y=0; y<NY; ++y)
			for (unsigned int v=0; v<nvec; ++v)
				for (unsigned int n=0; n<3; ++n)
				{
					const size_t k = (((size_t)x*NY+y)*nvec+v)*3+n;
					for (unsigned int l=0; l<4; ++l)
					{
						const unsigned int z = l*nvec+v;
						if (z>=Nz)
							continue;
						m_f4_vv[k].f[l]   = m_Op_UPML->vv(n, x, y, z);
						m_f4_vvfo[k].f[l] = m_Op_UPML->vvfo(n, x, y, z);
						m_f4_vvfn[k].f[l] = m_Op_UPML->vvfn(n, x, y, z);
						m_f4_ii[k].f[l]   = m_Op_UPML->ii(n, x, y, z);
						m_f4_iifo[k].f[l] = m_Op_UPML->iifo(n, x, y, z);
						m_f4_iifn[k].f[l] = m_Op_UPML->iifn(n, x, y, z);
						m_f4_volt_flux[k].f[l] = volt_flux(n, x, y, z);
						m_f4_curr_flux[k].f[l] = curr_flux(n, x, y, z);
					}
				}
	m_f4_LaneMask.assign(nvec, zero);
	for (unsigned int v=0; v<nvec; ++v)
		for (unsigned int l=0; l<4; ++l)
		{
			const unsigned int bits = (l*nvec+v<Nz) ? 0xFFFFFFFFu : 0u;
			memcpy(&m_f4_LaneMask[v].f[l], &bits, sizeof(bits));
		}
	m_Vec = true;
#endif
}

// The f4vector hooks below do the same arithmetic as the scalar ones, lane by lane, in the
// same order and with the same contractions (bit-identical); lanes outside the mesh are left as they are.

void Engine_Ext_UPML::DoPreVoltageUpdatesRows(unsigned int r0, unsigned int r1)
{
#ifdef UPML_F4VECTOR
	Engine_sse* eng = (Engine_sse*)m_Eng;
	f4vector* __restrict field = eng->f4_volt_ptr->data();
	const ptrdiff_t sx = eng->f4_volt_ptr->stride(1), sy = eng->f4_volt_ptr->stride(2);
	const unsigned int NY = m_Op_UPML->m_numLines[1], nvec = m_nvec;
	for (unsigned int r=r0; r<r1; ++r)
	{
		f4vector* __restrict e = field + (r/NY + m_Op_UPML->m_StartPos[0])*sx + (r%NY + m_Op_UPML->m_StartPos[1])*sy;
		const size_t b = (size_t)r*nvec*3;
		const f4vector* __restrict cv  = &m_f4_vv[b];
		const f4vector* __restrict cfo = &m_f4_vvfo[b];
		f4vector* __restrict flux = &m_f4_volt_flux[b];
		for (unsigned int k=0; k<3*nvec; ++k)
		{
			const v4sf E = e[k].v, F = flux[k].v;
			const v4sf p = -(cfo[k].v * F);
			const v4sf f = cv[k].v * E + p;
			e[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, F, E);
			flux[k].v = f;
		}
	}
#else
	UNUSED(r0);
	UNUSED(r1);
#endif
}

void Engine_Ext_UPML::DoPostVoltageUpdatesRows(unsigned int r0, unsigned int r1)
{
#ifdef UPML_F4VECTOR
	Engine_sse* eng = (Engine_sse*)m_Eng;
	f4vector* __restrict field = eng->f4_volt_ptr->data();
	const ptrdiff_t sx = eng->f4_volt_ptr->stride(1), sy = eng->f4_volt_ptr->stride(2);
	const unsigned int NY = m_Op_UPML->m_numLines[1], nvec = m_nvec;
	for (unsigned int r=r0; r<r1; ++r)
	{
		f4vector* __restrict e = field + (r/NY + m_Op_UPML->m_StartPos[0])*sx + (r%NY + m_Op_UPML->m_StartPos[1])*sy;
		const size_t b = (size_t)r*nvec*3;
		const f4vector* __restrict cfn = &m_f4_vvfn[b];
		f4vector* __restrict flux = &m_f4_volt_flux[b];
		for (unsigned int k=0; k<3*nvec; ++k)
		{
			const v4sf E = e[k].v, f = flux[k].v;
			const v4sf En = f + cfn[k].v * E;
			flux[k].v = E;
			e[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, En, E);
		}
	}
#else
	UNUSED(r0);
	UNUSED(r1);
#endif
}

void Engine_Ext_UPML::DoPreCurrentUpdatesRows(unsigned int r0, unsigned int r1)
{
#ifdef UPML_F4VECTOR
	Engine_sse* eng = (Engine_sse*)m_Eng;
	f4vector* __restrict field = eng->f4_curr_ptr->data();
	const ptrdiff_t sx = eng->f4_curr_ptr->stride(1), sy = eng->f4_curr_ptr->stride(2);
	const unsigned int NY = m_Op_UPML->m_numLines[1], nvec = m_nvec;
	for (unsigned int r=r0; r<r1; ++r)
	{
		f4vector* __restrict h = field + (r/NY + m_Op_UPML->m_StartPos[0])*sx + (r%NY + m_Op_UPML->m_StartPos[1])*sy;
		const size_t b = (size_t)r*nvec*3;
		const f4vector* __restrict ci  = &m_f4_ii[b];
		const f4vector* __restrict cfo = &m_f4_iifo[b];
		f4vector* __restrict flux = &m_f4_curr_flux[b];
		for (unsigned int k=0; k<3*nvec; ++k)
		{
			const v4sf H = h[k].v, F = flux[k].v;
			const v4sf p = -(cfo[k].v * F);
			const v4sf f = ci[k].v * H + p;
			h[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, F, H);
			flux[k].v = f;
		}
	}
#else
	UNUSED(r0);
	UNUSED(r1);
#endif
}

void Engine_Ext_UPML::DoPostCurrentUpdatesRows(unsigned int r0, unsigned int r1)
{
#ifdef UPML_F4VECTOR
	Engine_sse* eng = (Engine_sse*)m_Eng;
	f4vector* __restrict field = eng->f4_curr_ptr->data();
	const ptrdiff_t sx = eng->f4_curr_ptr->stride(1), sy = eng->f4_curr_ptr->stride(2);
	const unsigned int NY = m_Op_UPML->m_numLines[1], nvec = m_nvec;
	for (unsigned int r=r0; r<r1; ++r)
	{
		f4vector* __restrict h = field + (r/NY + m_Op_UPML->m_StartPos[0])*sx + (r%NY + m_Op_UPML->m_StartPos[1])*sy;
		const size_t b = (size_t)r*nvec*3;
		const f4vector* __restrict cfn = &m_f4_iifn[b];
		f4vector* __restrict flux = &m_f4_curr_flux[b];
		for (unsigned int k=0; k<3*nvec; ++k)
		{
			const v4sf H = h[k].v, f = flux[k].v;
			const v4sf Hn = f + cfn[k].v * H;
			flux[k].v = H;
			h[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, Hn, H);
		}
	}
#else
	UNUSED(r0);
	UNUSED(r1);
#endif
}

bool Engine_Ext_UPML::LineRows(unsigned int x0, unsigned int x1, unsigned int& r0, unsigned int& r1) const
{
	const unsigned int start = m_Op_UPML->m_StartPos[0], NX = m_Op_UPML->m_numLines[0], NY = m_Op_UPML->m_numLines[1];
	x0 = std::max(x0, start);
	x1 = std::min(x1, start+NX);
	if (x0>=x1)
		return false;
	r0 = (x0-start)*NY;
	r1 = (x1-start)*NY;
	return true;
}

void Engine_Ext_UPML::DoPreVoltageUpdatesVec(int threadID)
{
	if (threadID<m_NrThreads)
		DoPreVoltageUpdatesRows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
}

void Engine_Ext_UPML::DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	unsigned int r0, r1;
	if (LineRows(x0, x1, r0, r1))
		DoPreVoltageUpdatesRows(r0, r1);
}

void Engine_Ext_UPML::DoPostVoltageUpdatesVec(int threadID)
{
	if (threadID<m_NrThreads)
		DoPostVoltageUpdatesRows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
}

void Engine_Ext_UPML::DoPostVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	unsigned int r0, r1;
	if (LineRows(x0, x1, r0, r1))
		DoPostVoltageUpdatesRows(r0, r1);
}

void Engine_Ext_UPML::DoPreCurrentUpdatesVec(int threadID)
{
	if (threadID<m_NrThreads)
		DoPreCurrentUpdatesRows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
}

void Engine_Ext_UPML::DoPreCurrentUpdatesLines(unsigned int x0, unsigned int x1)
{
	unsigned int r0, r1;
	if (LineRows(x0, x1, r0, r1))
		DoPreCurrentUpdatesRows(r0, r1);
}

void Engine_Ext_UPML::DoPostCurrentUpdatesVec(int threadID)
{
	if (threadID<m_NrThreads)
		DoPostCurrentUpdatesRows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
}

void Engine_Ext_UPML::DoPostCurrentUpdatesLines(unsigned int x0, unsigned int x1)
{
	unsigned int r0, r1;
	if (LineRows(x0, x1, r0, r1))
		DoPostCurrentUpdatesRows(r0, r1);
}

template <typename EngType>
void Engine_Ext_UPML::DoPreVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;

	if (threadID>=m_NrThreads)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (unsigned int lineX=0; lineX<m_numX.at(threadID); ++lineX)
	{
		loc_pos[0]=lineX+m_start.at(threadID);
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = m_Op_UPML->vv(0, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetVolt(0,pos)
						 - m_Op_UPML->vvfo(0, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetVolt(0,pos, volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));
				volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->vv(1, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetVolt(1,pos)
						 - m_Op_UPML->vvfo(1, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetVolt(1,pos, volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));
				volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->vv(2, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetVolt(2,pos)
						 - m_Op_UPML->vvfo(2, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetVolt(2,pos, volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
				volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;
			}
		}
	}
}

void Engine_Ext_UPML::DoPreVoltageUpdates(int threadID)
{
	if (m_Vec)
		return DoPreVoltageUpdatesVec(threadID);
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_UPML::DoPostVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;
	if (threadID>=m_NrThreads)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (unsigned int lineX=0; lineX<m_numX.at(threadID); ++lineX)
	{
		loc_pos[0]=lineX+m_start.at(threadID);
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetVolt(0,pos);
				eng->EngType::SetVolt(0,pos, f_help + m_Op_UPML->vvfn(0, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetVolt(1,pos);
				eng->EngType::SetVolt(1,pos, f_help + m_Op_UPML->vvfn(1, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetVolt(2,pos);
				eng->EngType::SetVolt(2,pos, f_help + m_Op_UPML->vvfn(2, loc_pos[0], loc_pos[1], loc_pos[2]) * volt_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
			}
		}
	}
}

void Engine_Ext_UPML::DoPostVoltageUpdates(int threadID)
{
	if (m_Vec)
		return DoPostVoltageUpdatesVec(threadID);
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_UPML::DoPreCurrentUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;
	if (threadID>=m_NrThreads)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (unsigned int lineX=0; lineX<m_numX.at(threadID); ++lineX)
	{
		loc_pos[0]=lineX+m_start.at(threadID);
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = m_Op_UPML->ii(0, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetCurr(0,pos)
						 - m_Op_UPML->iifo(0, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetCurr(0,pos, curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));
				curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->ii(1, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetCurr(1,pos)
						 - m_Op_UPML->iifo(1, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetCurr(1,pos, curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));
				curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

				f_help = m_Op_UPML->ii(2, loc_pos[0], loc_pos[1], loc_pos[2])   * eng->EngType::GetCurr(2,pos)
						 - m_Op_UPML->iifo(2, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				eng->EngType::SetCurr(2,pos, curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
				curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = f_help;

			}
		}
	}
}

void Engine_Ext_UPML::DoPreCurrentUpdates(int threadID)
{
	if (m_Vec)
		return DoPreCurrentUpdatesVec(threadID);
	ENG_DISPATCH_ARGS(DoPreCurrentUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_UPML::DoPostCurrentUpdatesImpl(EngType* eng, int threadID)
{
	if (m_Eng==NULL)
		return;
	if (threadID>=m_NrThreads)
		return;

	unsigned int pos[3];
	unsigned int loc_pos[3];
	FDTD_FLOAT f_help;

	for (unsigned int lineX=0; lineX<m_numX.at(threadID); ++lineX)
	{
		loc_pos[0]=lineX+m_start.at(threadID);
		pos[0] = loc_pos[0] + m_Op_UPML->m_StartPos[0];
		for (loc_pos[1]=0; loc_pos[1]<m_Op_UPML->m_numLines[1]; ++loc_pos[1])
		{
			pos[1] = loc_pos[1] + m_Op_UPML->m_StartPos[1];
			for (loc_pos[2]=0; loc_pos[2]<m_Op_UPML->m_numLines[2]; ++loc_pos[2])
			{
				pos[2] = loc_pos[2] + m_Op_UPML->m_StartPos[2];

				f_help = curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]);
				curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetCurr(0,pos);
				eng->EngType::SetCurr(0,pos, f_help + m_Op_UPML->iifn(0, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(0, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]);
				curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetCurr(1,pos);
				eng->EngType::SetCurr(1,pos, f_help + m_Op_UPML->iifn(1, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(1, loc_pos[0], loc_pos[1], loc_pos[2]));

				f_help = curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]);
				curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]) = eng->EngType::GetCurr(2,pos);
				eng->EngType::SetCurr(2,pos, f_help + m_Op_UPML->iifn(2, loc_pos[0], loc_pos[1], loc_pos[2]) * curr_flux(2, loc_pos[0], loc_pos[1], loc_pos[2]));
			}
		}
	}
}

void Engine_Ext_UPML::DoPostCurrentUpdates(int threadID)
{
	if (m_Vec)
		return DoPostCurrentUpdatesVec(threadID);
	ENG_DISPATCH_ARGS(DoPostCurrentUpdatesImpl, threadID);
}
