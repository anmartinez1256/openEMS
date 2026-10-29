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

#include "engine_ext_mur_abc.h"
#include "operator_ext_mur_abc.h"
#include "FDTD/engine.h"
#include "FDTD/engine_sse.h"
#include "tools/useful.h"
#include "operator_ext_excitation.h"
#include <algorithm>

using std::cerr;
using std::endl;

Engine_Ext_Mur_ABC::Engine_Ext_Mur_ABC(Operator_Ext_Mur_ABC* op_ext) :
	Engine_Extension(op_ext),
	m_Mur_Coeff_nyP (op_ext->m_Mur_Coeff_nyP),
	m_Mur_Coeff_nyPP(op_ext->m_Mur_Coeff_nyPP),
	m_volt_nyP ("volt_nyP",  op_ext->m_numLines),
	m_volt_nyPP("volt_nyPP", op_ext->m_numLines)
{
	m_Op_mur = op_ext;
	m_numLines[0] = m_Op_mur->m_numLines[0];
	m_numLines[1] = m_Op_mur->m_numLines[1];
	m_ny = m_Op_mur->m_ny;
	m_nyP = m_Op_mur->m_nyP;
	m_nyPP = m_Op_mur->m_nyPP;
	m_LineNr = m_Op_mur->m_LineNr;
	m_LineNr_Shift = m_Op_mur->m_LineNr_Shift;

	//find if some excitation is on this mur-abc and find the max length of this excite, so that the abc can start after the excitation is done...
	int maxDelay=-1;
	Operator_Ext_Excitation* Exc_ext = m_Op_mur->m_Op->GetExcitationExtension();
	if (Exc_ext) //an unused excitation extension is dropped by the operator
	{
		for (unsigned int n=0; n<Exc_ext->GetVoltCount(); ++n)
		{
			if ( ((Exc_ext->Volt_dir[n]==m_nyP) || (Exc_ext->Volt_dir[n]==m_nyPP)) && (Exc_ext->Volt_index[m_ny][n]==m_LineNr) )
			{
				if ((int)Exc_ext->Volt_delay[n]>maxDelay)
					maxDelay = (int)Exc_ext->Volt_delay[n];
			}
		}
	}
	m_start_TS = 0;
	if (maxDelay>=0)
	{
		m_start_TS = maxDelay + m_Op_mur->m_Op->GetExcitationSignal()->GetLength() + 10; //give it some extra timesteps, for the excitation to travel at least one cell away
		cerr << "Engine_Ext_Mur_ABC::Engine_Ext_Mur_ABC: Warning: Excitation inside the Mur-ABC #" <<  m_ny << "-" << (int)(m_LineNr>0) << " found!!!!  Mur-ABC will be switched on after excitation is done at " << m_start_TS << " timesteps!!! " << endl;
	}

	m_Fast = false;
	m_FastChecked = false;
	m_OffsetNyP = m_OffsetNyPP = m_OffsetShift = 0;

	SetNumberOfThreads(1);
}

Engine_Ext_Mur_ABC::~Engine_Ext_Mur_ABC()
{
}


void Engine_Ext_Mur_ABC::SetNumberOfThreads(int nrThread)
{
	Engine_Extension::SetNumberOfThreads(nrThread);

	m_numX = AssignJobs2Threads(m_numLines[0],m_NrThreads,false);
	m_start.resize(m_NrThreads,0);
	m_start.at(0)=0;
	for (size_t n=1; n<m_numX.size(); ++n)
		m_start.at(n) = m_start.at(n-1) + m_numX.at(n-1);

	if (!m_FastChecked && m_Eng)
		SetupFast();
}

void Engine_Ext_Mur_ABC::SetupFast()
{
	m_FastChecked = true;
	m_Fast = false;
	if (m_Eng->GetType()!=Engine::SSE)
		return;
	ArrayLib::ArrayENG<f4vector>* V = ((Engine_sse*)m_Eng)->f4_volt_ptr;
	const unsigned int nvec = V->extent(3);
	if ((uint64_t)V->size()*4 >= (1ull<<32))
		return;
	// float index of component n at pos: 4*linearIndex(n, x, y, z%nvec) + z/nvec
	struct FloatIndex
	{
		ArrayLib::ArrayENG<f4vector>* V; unsigned int nvec;
		int64_t operator()(unsigned int n, const unsigned int pos[3]) const
		{return 4*(int64_t)V->linearIndex({n, pos[0], pos[1], pos[2]%nvec}) + pos[2]/nvec;}
	} idx = {V, nvec};

	m_FloatIndex.resize((size_t)m_numLines[0]*m_numLines[1]);
	unsigned int pos[] = {0,0,0};
	unsigned int pos_shift[] = {0,0,0};
	pos[m_ny] = m_LineNr;
	pos_shift[m_ny] = m_LineNr_Shift;
	for (unsigned int i=0; i<m_numLines[0]; ++i)
		for (unsigned int j=0; j<m_numLines[1]; ++j)
		{
			pos[m_nyP] = pos_shift[m_nyP] = i;
			pos[m_nyPP] = pos_shift[m_nyPP] = j;
			const int64_t base = idx(0, pos);
			const int64_t oP = idx(m_nyP, pos) - base, oPP = idx(m_nyPP, pos) - base;
			const int64_t oS = idx(m_nyP, pos_shift) - idx(m_nyP, pos);
			if ((i==0) && (j==0))
			{
				m_OffsetNyP = oP;
				m_OffsetNyPP = oPP;
				m_OffsetShift = oS;
			}
			// the offsets must be the same for all cells of the face
			if ((oP!=m_OffsetNyP) || (oPP!=m_OffsetNyPP) || (oS!=m_OffsetShift) ||
				(idx(m_nyPP, pos_shift)-idx(m_nyPP, pos)!=oS))
				return;
			m_FloatIndex[(size_t)i*m_numLines[1]+j] = (unsigned int)base;
		}

	// face cells by x-line in memory order, for the line hooks
	const unsigned int NX = V->extent(1);
	std::vector<std::vector<unsigned int> > line(NX);
	for (unsigned int i=0; i<m_numLines[0]; ++i)
		for (unsigned int j=0; j<m_numLines[1]; ++j)
		{
			pos[m_nyP] = i;
			pos[m_nyPP] = j;
			line[pos[0]].push_back(i*m_numLines[1]+j);
		}
	const bool aligned = (m_OffsetNyP%4==0) && (m_OffsetNyPP%4==0) && (m_OffsetShift%4==0);
	const FDTD_FLOAT* cP = m_Mur_Coeff_nyP.data();
	const FDTD_FLOAT* cPP = m_Mur_Coeff_nyPP.data();
	m_VecStart.assign(1, 0);
	m_ScStart.assign(1, 0);
	m_VecIndex.clear(); m_VecCoeffP.clear(); m_VecCoeffPP.clear();
	m_ScIndex.clear(); m_ScCoeffP.clear(); m_ScCoeffPP.clear();
	for (unsigned int x=0; x<NX; ++x)
	{
		std::vector<unsigned int>& cells = line[x];
		std::sort(cells.begin(), cells.end(), [&](unsigned int a, unsigned int b) {return m_FloatIndex[a]<m_FloatIndex[b];});
		for (size_t n=0; n<cells.size();)
		{
			const unsigned int f = m_FloatIndex[cells[n]];
			if (aligned && (f%4==0) && (n+3<cells.size()) && (m_FloatIndex[cells[n+3]]==f+3))
			{
				// sorted and distinct: the four floats of one f4vector
				f4vector p, pp;
				for (int l=0; l<4; ++l)
				{
					p.f[l] = cP[cells[n+l]];
					pp.f[l] = cPP[cells[n+l]];
				}
				m_VecIndex.push_back(f/4);
				m_VecCoeffP.push_back(p);
				m_VecCoeffPP.push_back(pp);
				n += 4;
				continue;
			}
			m_ScIndex.push_back(f);
			m_ScCoeffP.push_back(cP[cells[n]]);
			m_ScCoeffPP.push_back(cPP[cells[n]]);
			++n;
		}
		m_VecStart.push_back(m_VecIndex.size());
		m_ScStart.push_back(m_ScIndex.size());
	}
	m_VecP.resize(m_VecIndex.size());
	m_VecPP.resize(m_VecIndex.size());
	m_ScP.resize(m_ScIndex.size());
	m_ScPP.resize(m_ScIndex.size());
	m_Fast = true;
}

namespace
{
//! The face cell updates of the fast path, same arithmetic in the same order as the templated hooks (bit-identical).
struct MurFast
{
	FDTD_FLOAT* E;
	const FDTD_FLOAT* cP;
	const FDTD_FLOAT* cPP;
	FDTD_FLOAT* vP;
	FDTD_FLOAT* vPP;
	const unsigned int* fi;
	ptrdiff_t oP, oPP, oS;

	inline void Pre(size_t k) const
	{
		const FDTD_FLOAT* e = E + fi[k];
		vP[k]  = e[oP+oS]  - cP[k]  * e[oP];
		vPP[k] = e[oPP+oS] - cPP[k] * e[oPP];
	}
	inline void Post(size_t k) const
	{
		const FDTD_FLOAT* e = E + fi[k];
		vP[k]  += cP[k]  * e[oP+oS];
		vPP[k] += cPP[k] * e[oPP+oS];
	}
	inline void Apply(size_t k) const
	{
		FDTD_FLOAT* e = E + fi[k];
		e[oP]  = vP[k];
		e[oPP] = vPP[k];
	}
};
}

#define MUR_FAST_DATA \
	const MurFast d = {(FDTD_FLOAT*)((Engine_sse*)m_Eng)->f4_volt_ptr->data(), m_Mur_Coeff_nyP.data(), m_Mur_Coeff_nyPP.data(), \
					   m_volt_nyP.data(), m_volt_nyPP.data(), m_FloatIndex.data(), m_OffsetNyP, m_OffsetNyPP, m_OffsetShift};

void Engine_Ext_Mur_ABC::DoPreVoltageUpdatesFast(int threadID)
{
	if (IsActive()==false) return;
	if (threadID>=m_NrThreads)
		return;
	MUR_FAST_DATA
	const size_t k0 = (size_t)m_start.at(threadID)*m_numLines[1];
	const size_t k1 = k0 + (size_t)m_numX.at(threadID)*m_numLines[1];
	for (size_t k=k0; k<k1; ++k)
		d.Pre(k);
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesFast(int threadID)
{
	if (IsActive()==false) return;
	if (threadID>=m_NrThreads)
		return;
	MUR_FAST_DATA
	const size_t k0 = (size_t)m_start.at(threadID)*m_numLines[1];
	const size_t k1 = k0 + (size_t)m_numX.at(threadID)*m_numLines[1];
	for (size_t k=k0; k<k1; ++k)
		d.Post(k);
}

void Engine_Ext_Mur_ABC::Apply2VoltagesFast(int threadID)
{
	if (IsActive()==false) return;
	if (threadID>=m_NrThreads)
		return;
	MUR_FAST_DATA
	const size_t k0 = (size_t)m_start.at(threadID)*m_numLines[1];
	const size_t k1 = k0 + (size_t)m_numX.at(threadID)*m_numLines[1];
	for (size_t k=k0; k<k1; ++k)
		d.Apply(k);
}

// The line hooks: the vectors with the same arithmetic per lane as MurFast (bit-identical), then the single cells
#define MUR_LINE_DATA \
	f4vector* E4 = ((Engine_sse*)m_Eng)->f4_volt_ptr->data(); \
	const ptrdiff_t oP = m_OffsetNyP/4, oPP = m_OffsetNyPP/4, oS = m_OffsetShift/4; \
	const MurFast d = {(FDTD_FLOAT*)E4, m_ScCoeffP.data(), m_ScCoeffPP.data(), m_ScP.data(), m_ScPP.data(), m_ScIndex.data(), \
					   m_OffsetNyP, m_OffsetNyPP, m_OffsetShift};

void Engine_Ext_Mur_ABC::DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	if (IsActive()==false) return;
	MUR_LINE_DATA
	for (unsigned int g=m_VecStart[x0]; g<m_VecStart[x1]; ++g)
	{
		const f4vector* e = E4 + m_VecIndex[g];
		m_VecP[g].v  = e[oP+oS].v  - m_VecCoeffP[g].v  * e[oP].v;
		m_VecPP[g].v = e[oPP+oS].v - m_VecCoeffPP[g].v * e[oPP].v;
	}
	for (unsigned int k=m_ScStart[x0]; k<m_ScStart[x1]; ++k)
		d.Pre(k);
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	if (IsActive()==false) return;
	MUR_LINE_DATA
	for (unsigned int g=m_VecStart[x0]; g<m_VecStart[x1]; ++g)
	{
		const f4vector* e = E4 + m_VecIndex[g];
		m_VecP[g].v  += m_VecCoeffP[g].v  * e[oP+oS].v;
		m_VecPP[g].v += m_VecCoeffPP[g].v * e[oPP+oS].v;
	}
	for (unsigned int k=m_ScStart[x0]; k<m_ScStart[x1]; ++k)
		d.Post(k);
}

void Engine_Ext_Mur_ABC::Apply2VoltagesLines(unsigned int x0, unsigned int x1)
{
	if (IsActive()==false) return;
	MUR_LINE_DATA
	for (unsigned int g=m_VecStart[x0]; g<m_VecStart[x1]; ++g)
	{
		f4vector* e = E4 + m_VecIndex[g];
		e[oP].v  = m_VecP[g].v;
		e[oPP].v = m_VecPP[g].v;
	}
	for (unsigned int k=m_ScStart[x0]; k<m_ScStart[x1]; ++k)
		d.Apply(k);
}

void Engine_Ext_Mur_ABC::GetCoupledLines(std::vector<unsigned int>& lines) const
{
	// on an x-face the hooks read the line next to the face
	if (m_ny==0)
		lines.push_back(std::min<unsigned int>(m_LineNr, m_LineNr_Shift));
}

template <typename EngType>
void Engine_Ext_Mur_ABC::DoPreVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (IsActive()==false) return;
	if (m_Eng==NULL) return;
	if (threadID>=m_NrThreads)
		return;

	// See detailed comments in operator_ext_mur_abc.h, not repeated here.
	unsigned int pos[] = {0,0,0};
	unsigned int pos_shift[] = {0,0,0};
	pos[m_ny] = m_LineNr;
	pos_shift[m_ny] = m_LineNr_Shift;

	for (unsigned int i = m_start.at(threadID); i < m_start.at(threadID) + m_numX.at(threadID); i++)
	{
		pos[m_nyP] = i;
		pos_shift[m_nyP] = i;

		for (unsigned int j = 0; j < m_numLines[1]; j++)
		{
			pos[m_nyPP] = j;
			pos_shift[m_nyPP] = j;

			m_volt_nyP(i,j) = eng->EngType::GetVolt(m_nyP, pos_shift) -
				m_Op_mur->m_Mur_Coeff_nyP(i,j) * eng->EngType::GetVolt(m_nyP, pos);

			m_volt_nyPP(i,j) = eng->EngType::GetVolt(m_nyPP, pos_shift) -
				m_Op_mur->m_Mur_Coeff_nyPP(i,j) * eng->EngType::GetVolt(m_nyPP, pos);
		}
	}
}

void Engine_Ext_Mur_ABC::DoPreVoltageUpdates(int threadID)
{
	if (m_Fast)
		return DoPreVoltageUpdatesFast(threadID);
	ENG_DISPATCH_ARGS(DoPreVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_Mur_ABC::DoPostVoltageUpdatesImpl(EngType* eng, int threadID)
{
	if (IsActive()==false) return;
	if (m_Eng==NULL) return;
	if (threadID>=m_NrThreads)
		return;
	unsigned int pos_shift[] = {0,0,0};
	pos_shift[m_ny] = m_LineNr_Shift;

	for (unsigned int i = m_start.at(threadID); i < m_start.at(threadID) + m_numX.at(threadID); i++)
	{
		pos_shift[m_nyP] = i;

		for (unsigned int j = 0; j < m_numLines[1]; j++)
		{
			pos_shift[m_nyPP] = j;

			m_volt_nyP(i,j) +=
				m_Op_mur->m_Mur_Coeff_nyP(i,j) * eng->EngType::GetVolt(m_nyP, pos_shift);
			m_volt_nyPP(i,j) +=
				m_Op_mur->m_Mur_Coeff_nyPP(i,j) * eng->EngType::GetVolt(m_nyPP, pos_shift);
		}
	}
}

void Engine_Ext_Mur_ABC::DoPostVoltageUpdates(int threadID)
{
	if (m_Fast)
		return DoPostVoltageUpdatesFast(threadID);
	ENG_DISPATCH_ARGS(DoPostVoltageUpdatesImpl, threadID);
}

template <typename EngType>
void Engine_Ext_Mur_ABC::Apply2VoltagesImpl(EngType* eng, int threadID)
{
	if (IsActive()==false) return;
	if (threadID>=m_NrThreads)
		return;
	if (m_Eng==NULL) return;
	unsigned int pos[] = {0,0,0};
	pos[m_ny] = m_LineNr;

	for (unsigned int i = m_start.at(threadID); i < m_start.at(threadID) + m_numX.at(threadID); i++)
	{
		pos[m_nyP] = i;

		for (unsigned int j = 0; j < m_numLines[1]; j++)
		{
			pos[m_nyPP] = j;

			eng->EngType::SetVolt(m_nyP , pos, m_volt_nyP(i,j));
			eng->EngType::SetVolt(m_nyPP, pos, m_volt_nyPP(i,j));
		}
	}
}

void Engine_Ext_Mur_ABC::Apply2Voltages(int threadID)
{
	if (m_Fast)
		return Apply2VoltagesFast(threadID);
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, threadID);
}
