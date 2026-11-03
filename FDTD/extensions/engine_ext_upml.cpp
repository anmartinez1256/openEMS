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
#include <cstdlib>
#include <map>
#include "operator_ext_upml.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "tools/useful.h"
#include "tools/global.h"
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
	m_Sse = false;
	m_CoefTable = false;
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
	{
		SetupVectorized();
		if (!m_Vec)
			SetupScalarSSE();
	}
	if (m_Vec || m_Sse)
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
	SetupCoefTable4();
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
		f4vector* __restrict flux = &m_f4_volt_flux[b];
		if (m_CoefTable)
		{
			const Coef4* __restrict tab = m_CoefTab4.data();
			const unsigned int* __restrict ix = m_CoefIdx4.data() + b;
			for (unsigned int k=0; k<3*nvec; ++k)
			{
				const Coef4& c = tab[ix[k]];
				const v4sf E = e[k].v, F = flux[k].v;
				const v4sf p = -(c.vvfo.v * F);
				const v4sf f = c.vv.v * E + p;
				e[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, F, E);
				flux[k].v = f;
			}
			continue;
		}
		const f4vector* __restrict cv  = &m_f4_vv[b];
		const f4vector* __restrict cfo = &m_f4_vvfo[b];
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
		f4vector* __restrict flux = &m_f4_volt_flux[b];
		if (m_CoefTable)
		{
			const Coef4* __restrict tab = m_CoefTab4.data();
			const unsigned int* __restrict ix = m_CoefIdx4.data() + b;
			for (unsigned int k=0; k<3*nvec; ++k)
			{
				const v4sf E = e[k].v, f = flux[k].v;
				const v4sf En = f + tab[ix[k]].vvfn.v * E;
				flux[k].v = E;
				e[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, En, E);
			}
			continue;
		}
		const f4vector* __restrict cfn = &m_f4_vvfn[b];
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
		f4vector* __restrict flux = &m_f4_curr_flux[b];
		if (m_CoefTable)
		{
			const Coef4* __restrict tab = m_CoefTab4.data();
			const unsigned int* __restrict ix = m_CoefIdx4.data() + b;
			for (unsigned int k=0; k<3*nvec; ++k)
			{
				const Coef4& c = tab[ix[k]];
				const v4sf H = h[k].v, F = flux[k].v;
				const v4sf p = -(c.iifo.v * F);
				const v4sf f = c.ii.v * H + p;
				h[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, F, H);
				flux[k].v = f;
			}
			continue;
		}
		const f4vector* __restrict ci  = &m_f4_ii[b];
		const f4vector* __restrict cfo = &m_f4_iifo[b];
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
		f4vector* __restrict flux = &m_f4_curr_flux[b];
		if (m_CoefTable)
		{
			const Coef4* __restrict tab = m_CoefTab4.data();
			const unsigned int* __restrict ix = m_CoefIdx4.data() + b;
			for (unsigned int k=0; k<3*nvec; ++k)
			{
				const v4sf H = h[k].v, f = flux[k].v;
				const v4sf Hn = f + tab[ix[k]].iifn.v * H;
				flux[k].v = H;
				h[k].v = UPML_Blend(m_f4_LaneMask[k/3].v, Hn, H);
			}
			continue;
		}
		const f4vector* __restrict cfn = &m_f4_iifn[b];
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
	if (!LineRows(x0, x1, r0, r1))
		return;
	if (m_Vec)
		DoPreVoltageUpdatesRows(r0, r1);
	else
		DoPreVoltageUpdatesSSERows(r0, r1);
}

void Engine_Ext_UPML::DoPostVoltageUpdatesVec(int threadID)
{
	if (threadID<m_NrThreads)
		DoPostVoltageUpdatesRows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
}

void Engine_Ext_UPML::DoPostVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	unsigned int r0, r1;
	if (!LineRows(x0, x1, r0, r1))
		return;
	if (m_Vec)
		DoPostVoltageUpdatesRows(r0, r1);
	else
		DoPostVoltageUpdatesSSERows(r0, r1);
}

void Engine_Ext_UPML::DoPreCurrentUpdatesVec(int threadID)
{
	if (threadID<m_NrThreads)
		DoPreCurrentUpdatesRows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
}

void Engine_Ext_UPML::DoPreCurrentUpdatesLines(unsigned int x0, unsigned int x1)
{
	unsigned int r0, r1;
	if (!LineRows(x0, x1, r0, r1))
		return;
	if (m_Vec)
		DoPreCurrentUpdatesRows(r0, r1);
	else
		DoPreCurrentUpdatesSSERows(r0, r1);
}

void Engine_Ext_UPML::DoPostCurrentUpdatesVec(int threadID)
{
	if (threadID<m_NrThreads)
		DoPostCurrentUpdatesRows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
}

void Engine_Ext_UPML::DoPostCurrentUpdatesLines(unsigned int x0, unsigned int x1)
{
	unsigned int r0, r1;
	if (!LineRows(x0, x1, r0, r1))
		return;
	if (m_Vec)
		DoPostCurrentUpdatesRows(r0, r1);
	else
		DoPostCurrentUpdatesSSERows(r0, r1);
}

void Engine_Ext_UPML::SetupScalarSSE()
{
	m_Sse = false;
	if (m_Eng->GetType()!=Engine::SSE)
		return;
	Engine_sse* eng = (Engine_sse*)m_Eng;
	ArrayLib::ArrayENG<f4vector>* V = eng->f4_volt_ptr;
	ArrayLib::ArrayENG<f4vector>* I = eng->f4_curr_ptr;
	if ((V->stride(0)!=1) || (V->stride(3)!=3) || (I->stride(0)!=1) || (I->stride(3)!=3) ||
		(V->stride(1)!=I->stride(1)) || (V->stride(2)!=I->stride(2)))
		return;
	const unsigned int nvec = V->extent(3);
	m_zoff.resize(m_Op_UPML->m_numLines[2]);
	for (unsigned int lz=0; lz<m_Op_UPML->m_numLines[2]; ++lz)
	{
		const unsigned int z = lz + m_Op_UPML->m_StartPos[2];
		m_zoff[lz] = 12*(z%nvec) + z/nvec; // f4vector (z%nvec) of the row is 3 vectors = 12 floats further, lane z/nvec
	}
	m_Sse = true;
	SetupCoefTable();
}

void Engine_Ext_UPML::SetupCoefTable4()
{
	m_CoefTable = false;
	m_CoefTab4.clear();
	m_CoefIdx4.clear();
	if (EnvFlag("OPENEMS_NO_PML_TABLE"))
		return;
	struct Key
	{
		f4vector c[6];
		bool operator<(const Key& o) const { return memcmp(c, o.c, sizeof(c))<0; }
	};
	std::map<Key, unsigned int> lut;
	const size_t size = m_f4_vv.size();
	m_CoefIdx4.resize(size);
	for (size_t k=0; k<size; ++k)
	{
		Key key;
		key.c[0] = m_f4_vv[k]; key.c[1] = m_f4_vvfo[k]; key.c[2] = m_f4_vvfn[k];
		key.c[3] = m_f4_ii[k]; key.c[4] = m_f4_iifo[k]; key.c[5] = m_f4_iifn[k];
		std::map<Key, unsigned int>::iterator it = lut.find(key);
		if (it==lut.end())
		{
			it = lut.insert(std::make_pair(key, (unsigned int)m_CoefTab4.size())).first;
			Coef4 cf = {key.c[0], key.c[1], key.c[2], key.c[3], key.c[4], key.c[5]};
			m_CoefTab4.push_back(cf);
		}
		m_CoefIdx4[k] = it->second;
	}
	// the six per-cell arrays are not read any more
	std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_f4_vv);   std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_f4_vvfo);
	std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_f4_vvfn); std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_f4_ii);
	std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_f4_iifo); std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_f4_iifn);
	m_CoefTable = true;
	if (g_settings.GetVerboseLevel()>1)
		std::cout << "UPML (f4vector): " << m_CoefTab4.size() << " distinct coefficient sets for " << size/3 << " z-vectors" << std::endl;
}

void Engine_Ext_UPML::SetupCoefTable()
{
	m_CoefTable = false;
	m_CoefTab.clear();
	m_CoefIdx.clear();
	if (EnvFlag("OPENEMS_NO_PML_TABLE"))
		return;
	const unsigned int NX = m_Op_UPML->m_numLines[0], NY = m_Op_UPML->m_numLines[1], NZ = m_Op_UPML->m_numLines[2];
	struct Key
	{
		FDTD_FLOAT c[6];
		bool operator<(const Key& o) const { return memcmp(c, o.c, sizeof(c))<0; }
	};
	std::map<Key, unsigned int> lut;
	m_CoefIdx.resize((size_t)NX*NY*3*NZ);
	for (unsigned int lx=0; lx<NX; ++lx)
		for (unsigned int ly=0; ly<NY; ++ly)
			for (unsigned int n=0; n<3; ++n)
			{
				unsigned int* __restrict ix = m_CoefIdx.data() + (((size_t)lx*NY+ly)*3+n)*NZ;
				for (unsigned int lz=0; lz<NZ; ++lz)
				{
					Key k;
					k.c[0] = m_Op_UPML->vv(n, lx, ly, lz);   k.c[1] = m_Op_UPML->vvfo(n, lx, ly, lz); k.c[2] = m_Op_UPML->vvfn(n, lx, ly, lz);
					k.c[3] = m_Op_UPML->ii(n, lx, ly, lz);   k.c[4] = m_Op_UPML->iifo(n, lx, ly, lz); k.c[5] = m_Op_UPML->iifn(n, lx, ly, lz);
					std::map<Key, unsigned int>::iterator it = lut.find(k);
					if (it==lut.end())
					{
						it = lut.insert(std::make_pair(k, (unsigned int)m_CoefTab.size())).first;
						Coef cf = {k.c[0], k.c[1], k.c[2], k.c[3], k.c[4], k.c[5]};
						m_CoefTab.push_back(cf);
					}
					ix[lz] = it->second;
				}
			}
	m_CoefTable = true;
	if (g_settings.GetVerboseLevel()>1)
		std::cout << "UPML: " << m_CoefTab.size() << " distinct coefficient sets for " << (size_t)NX*NY*NZ << " cells" << std::endl;
}

// row r = (x,y) of the PML box: pointers to the floats of component n in the engine field (lane offsets in m_zoff)
// and to the coefficient / flux rows (contiguous in z)
#define UPML_SSE_ROW_SETUP(FIELD) \
	const unsigned int NY = m_Op_UPML->m_numLines[1], NZ = m_Op_UPML->m_numLines[2]; \
	const unsigned int* __restrict zo = m_zoff.data(); \
	Engine_sse* eng = (Engine_sse*)m_Eng; \
	float* __restrict field = (float*)eng->FIELD->data(); \
	const size_t sx = 4*eng->FIELD->stride(1), sy = 4*eng->FIELD->stride(2);

void Engine_Ext_UPML::DoPreVoltageUpdatesSSERows(unsigned int r0, unsigned int r1)
{
	UPML_SSE_ROW_SETUP(f4_volt_ptr)
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>& cvv = m_Op_UPML->vv;
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>& cfo = m_Op_UPML->vvfo;
	for (unsigned int r=r0; r<r1; ++r)
	{
		const unsigned int lx = r/NY, ly = r%NY;
		float* __restrict e = field + (lx + m_Op_UPML->m_StartPos[0])*sx + (ly + m_Op_UPML->m_StartPos[1])*sy;
		if (m_CoefTable)
		{
			const Coef* __restrict tab = m_CoefTab.data();
			for (unsigned int n=0; n<3; ++n)
			{
				const unsigned int* __restrict ix = m_CoefIdx.data() + ((size_t)r*3+n)*NZ;
				FDTD_FLOAT* __restrict flux = volt_flux.data() + n*volt_flux.stride(0) + lx*volt_flux.stride(1) + ly*volt_flux.stride(2);
				float* __restrict en = e + 4*n;
				for (unsigned int lz=0; lz<NZ; ++lz)
				{
					float& V = en[zo[lz]];
					const Coef& c = tab[ix[lz]];
					const FDTD_FLOAT f_help = c.vv * V - c.vvfo * flux[lz];
					V = flux[lz];
					flux[lz] = f_help;
				}
			}
			continue;
		}
		for (unsigned int n=0; n<3; ++n)
		{
			const size_t b = n*cvv.stride(0) + lx*cvv.stride(1) + ly*cvv.stride(2);
			const FDTD_FLOAT* __restrict vv = cvv.data() + b;
			const FDTD_FLOAT* __restrict vvfo = cfo.data() + b;
			FDTD_FLOAT* __restrict flux = volt_flux.data() + n*volt_flux.stride(0) + lx*volt_flux.stride(1) + ly*volt_flux.stride(2);
			float* __restrict en = e + 4*n;
			for (unsigned int lz=0; lz<NZ; ++lz)
			{
				float& V = en[zo[lz]];
				const FDTD_FLOAT f_help = vv[lz] * V - vvfo[lz] * flux[lz];
				V = flux[lz];
				flux[lz] = f_help;
			}
		}
	}
}

void Engine_Ext_UPML::DoPostVoltageUpdatesSSERows(unsigned int r0, unsigned int r1)
{
	UPML_SSE_ROW_SETUP(f4_volt_ptr)
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>& cfn = m_Op_UPML->vvfn;
	for (unsigned int r=r0; r<r1; ++r)
	{
		const unsigned int lx = r/NY, ly = r%NY;
		float* __restrict e = field + (lx + m_Op_UPML->m_StartPos[0])*sx + (ly + m_Op_UPML->m_StartPos[1])*sy;
		if (m_CoefTable)
		{
			const Coef* __restrict tab = m_CoefTab.data();
			for (unsigned int n=0; n<3; ++n)
			{
				const unsigned int* __restrict ix = m_CoefIdx.data() + ((size_t)r*3+n)*NZ;
				FDTD_FLOAT* __restrict flux = volt_flux.data() + n*volt_flux.stride(0) + lx*volt_flux.stride(1) + ly*volt_flux.stride(2);
				float* __restrict en = e + 4*n;
				for (unsigned int lz=0; lz<NZ; ++lz)
				{
					float& V = en[zo[lz]];
					const FDTD_FLOAT f_help = flux[lz];
					flux[lz] = V;
					V = f_help + tab[ix[lz]].vvfn * flux[lz];
				}
			}
			continue;
		}
		for (unsigned int n=0; n<3; ++n)
		{
			const FDTD_FLOAT* __restrict vvfn = cfn.data() + n*cfn.stride(0) + lx*cfn.stride(1) + ly*cfn.stride(2);
			FDTD_FLOAT* __restrict flux = volt_flux.data() + n*volt_flux.stride(0) + lx*volt_flux.stride(1) + ly*volt_flux.stride(2);
			float* __restrict en = e + 4*n;
			for (unsigned int lz=0; lz<NZ; ++lz)
			{
				float& V = en[zo[lz]];
				const FDTD_FLOAT f_help = flux[lz];
				flux[lz] = V;
				V = f_help + vvfn[lz] * flux[lz];
			}
		}
	}
}

void Engine_Ext_UPML::DoPreCurrentUpdatesSSERows(unsigned int r0, unsigned int r1)
{
	UPML_SSE_ROW_SETUP(f4_curr_ptr)
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>& cii = m_Op_UPML->ii;
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>& cfo = m_Op_UPML->iifo;
	for (unsigned int r=r0; r<r1; ++r)
	{
		const unsigned int lx = r/NY, ly = r%NY;
		float* __restrict h = field + (lx + m_Op_UPML->m_StartPos[0])*sx + (ly + m_Op_UPML->m_StartPos[1])*sy;
		if (m_CoefTable)
		{
			const Coef* __restrict tab = m_CoefTab.data();
			for (unsigned int n=0; n<3; ++n)
			{
				const unsigned int* __restrict ix = m_CoefIdx.data() + ((size_t)r*3+n)*NZ;
				FDTD_FLOAT* __restrict flux = curr_flux.data() + n*curr_flux.stride(0) + lx*curr_flux.stride(1) + ly*curr_flux.stride(2);
				float* __restrict hn = h + 4*n;
				for (unsigned int lz=0; lz<NZ; ++lz)
				{
					float& C = hn[zo[lz]];
					const Coef& c = tab[ix[lz]];
					const FDTD_FLOAT f_help = c.ii * C - c.iifo * flux[lz];
					C = flux[lz];
					flux[lz] = f_help;
				}
			}
			continue;
		}
		for (unsigned int n=0; n<3; ++n)
		{
			const size_t b = n*cii.stride(0) + lx*cii.stride(1) + ly*cii.stride(2);
			const FDTD_FLOAT* __restrict ii = cii.data() + b;
			const FDTD_FLOAT* __restrict iifo = cfo.data() + b;
			FDTD_FLOAT* __restrict flux = curr_flux.data() + n*curr_flux.stride(0) + lx*curr_flux.stride(1) + ly*curr_flux.stride(2);
			float* __restrict hn = h + 4*n;
			for (unsigned int lz=0; lz<NZ; ++lz)
			{
				float& C = hn[zo[lz]];
				const FDTD_FLOAT f_help = ii[lz] * C - iifo[lz] * flux[lz];
				C = flux[lz];
				flux[lz] = f_help;
			}
		}
	}
}

void Engine_Ext_UPML::DoPostCurrentUpdatesSSERows(unsigned int r0, unsigned int r1)
{
	UPML_SSE_ROW_SETUP(f4_curr_ptr)
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>& cfn = m_Op_UPML->iifn;
	for (unsigned int r=r0; r<r1; ++r)
	{
		const unsigned int lx = r/NY, ly = r%NY;
		float* __restrict h = field + (lx + m_Op_UPML->m_StartPos[0])*sx + (ly + m_Op_UPML->m_StartPos[1])*sy;
		if (m_CoefTable)
		{
			const Coef* __restrict tab = m_CoefTab.data();
			for (unsigned int n=0; n<3; ++n)
			{
				const unsigned int* __restrict ix = m_CoefIdx.data() + ((size_t)r*3+n)*NZ;
				FDTD_FLOAT* __restrict flux = curr_flux.data() + n*curr_flux.stride(0) + lx*curr_flux.stride(1) + ly*curr_flux.stride(2);
				float* __restrict hn = h + 4*n;
				for (unsigned int lz=0; lz<NZ; ++lz)
				{
					float& C = hn[zo[lz]];
					const FDTD_FLOAT f_help = flux[lz];
					flux[lz] = C;
					C = f_help + tab[ix[lz]].iifn * flux[lz];
				}
			}
			continue;
		}
		for (unsigned int n=0; n<3; ++n)
		{
			const FDTD_FLOAT* __restrict iifn = cfn.data() + n*cfn.stride(0) + lx*cfn.stride(1) + ly*cfn.stride(2);
			FDTD_FLOAT* __restrict flux = curr_flux.data() + n*curr_flux.stride(0) + lx*curr_flux.stride(1) + ly*curr_flux.stride(2);
			float* __restrict hn = h + 4*n;
			for (unsigned int lz=0; lz<NZ; ++lz)
			{
				float& C = hn[zo[lz]];
				const FDTD_FLOAT f_help = flux[lz];
				flux[lz] = C;
				C = f_help + iifn[lz] * flux[lz];
			}
		}
	}
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
	if (m_Sse)
	{
		if (threadID<m_NrThreads)
			DoPreVoltageUpdatesSSERows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
		return;
	}
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
	if (m_Sse)
	{
		if (threadID<m_NrThreads)
			DoPostVoltageUpdatesSSERows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
		return;
	}
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
	if (m_Sse)
	{
		if (threadID<m_NrThreads)
			DoPreCurrentUpdatesSSERows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
		return;
	}
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
	if (m_Sse)
	{
		if (threadID<m_NrThreads)
			DoPostCurrentUpdatesSSERows(m_RowStart[threadID], m_RowStart[threadID]+m_RowNum[threadID]);
		return;
	}
	ENG_DISPATCH_ARGS(DoPostCurrentUpdatesImpl, threadID);
}
