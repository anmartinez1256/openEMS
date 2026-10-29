/*
*	Additional
*	Copyright (C) 2023 Gadi Lahav (gadi@rfwithcare.com)
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

#include "engine_ext_lumpedRLC.h"
#include "operator_ext_lumpedRLC.h"

#include "FDTD/engine_sse.h"
#include <algorithm>

Engine_Ext_LumpedRLC::Engine_Ext_LumpedRLC(Operator_Ext_LumpedRLC* op_ext_RLC) : Engine_Extension(op_ext_RLC)
{
	// Local pointer of the operator.
	m_Op_Ext_RLC = op_ext_RLC;

	v_Vdn		= new FDTD_FLOAT*[3];
	v_Jn		= new FDTD_FLOAT*[3];
	v_Il		= NULL;
	m_RotatedVd = -1;
	m_RotatedJ = -1;

	// No additional allocations are required if there are no actual lumped elements.
	if (!(m_Op_Ext_RLC->RLC_count))
		return;

	m_Order.resize(m_Op_Ext_RLC->RLC_count);
	for (unsigned int n=0; n<m_Order.size(); ++n)
		m_Order[n] = n;
	const unsigned int* x = m_Op_Ext_RLC->v_RLC_pos[0];
	std::stable_sort(m_Order.begin(), m_Order.end(), [x](unsigned int a, unsigned int b) {return x[a]<x[b];});
	m_X.resize(m_Order.size());
	for (unsigned int n=0; n<m_Order.size(); ++n)
		m_X[n] = x[m_Order[n]];

	// Initialize ADE containers for currents and voltages
	v_Il		= new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];

	for (unsigned int posIdx = 0 ; posIdx < m_Op_Ext_RLC->RLC_count ; ++posIdx)
		v_Il[posIdx] 	= 0.0;

	for (unsigned int k = 0 ; k < 3 ; k++)
	{
		v_Vdn[k] = new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];
		v_Jn[k] = new FDTD_FLOAT[m_Op_Ext_RLC->RLC_count];

		for (unsigned int posIdx = 0 ; posIdx < m_Op_Ext_RLC->RLC_count ; ++posIdx)
		{
			v_Jn[k][posIdx] = 0.0;
			v_Vdn[k][posIdx] = 0.0;
		}
	}

}

Engine_Ext_LumpedRLC::~Engine_Ext_LumpedRLC()
{
	// Only delete if values were allocated in the first place
	if (m_Op_Ext_RLC->RLC_count)
	{
		delete[] v_Il;

		for (unsigned int k = 0 ; k < 3 ; k++)
		{
			delete[] v_Vdn[k];
			delete[] v_Jn[k];
		}
	}

	delete[] v_Vdn;
	delete[] v_Jn;

	v_Il	= NULL;

	v_Vdn	= NULL;
	v_Jn	= NULL;

	m_Op_Ext_RLC = NULL;


}

void Engine_Ext_LumpedRLC::Rotate(FDTD_FLOAT** c)
{
	FDTD_FLOAT	*v_temp;
	v_temp = c[2];
	c[2] = c[1];
	c[1] = c[0];
	c[0] = v_temp;
}

void Engine_Ext_LumpedRLC::RotateOnce(std::atomic<int>& stamp, FDTD_FLOAT** c)
{
	const int ts = m_Eng->GetNumberOfTimesteps();
	if (stamp.load(std::memory_order_acquire)==ts)
		return;
	std::lock_guard<std::mutex> lock(m_RotateMutex);
	if (stamp.load(std::memory_order_relaxed)==ts)
		return;
	Rotate(c);
	stamp.store(ts, std::memory_order_release);
}

void Engine_Ext_LumpedRLC::LineRange(unsigned int x0, unsigned int x1, unsigned int& n0, unsigned int& n1) const
{
	n0 = std::lower_bound(m_X.begin(), m_X.end(), x0) - m_X.begin();
	n1 = std::lower_bound(m_X.begin(), m_X.end(), x1) - m_X.begin();
}

void Engine_Ext_LumpedRLC::UpdateInductorCurrents(const unsigned int* order, unsigned int n0, unsigned int n1)
{
	// In pre-process, only update the parallel inductor current:
	for (unsigned int k = n0 ; k < n1 ; k++)
	{
		const unsigned int pIdx = order ? order[k] : k;
		v_Il[pIdx] += (m_Op_Ext_RLC->v_RLC_i2v[pIdx])*(m_Op_Ext_RLC->v_RLC_ilv[pIdx])*v_Vdn[1][pIdx];
	}
}

void Engine_Ext_LumpedRLC::DoPreVoltageUpdates()
{
	if (!m_Op_Ext_RLC->RLC_count)
		return;

	// Iterate Vd containers
	Rotate(v_Vdn);
	UpdateInductorCurrents(NULL, 0, m_Op_Ext_RLC->RLC_count);
}

void Engine_Ext_LumpedRLC::DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1)
{
	if (!m_Op_Ext_RLC->RLC_count)
		return;
	RotateOnce(m_RotatedVd, v_Vdn);
	unsigned int n0, n1;
	LineRange(x0, x1, n0, n1);
	UpdateInductorCurrents(m_Order.data(), n0, n1);
}

template <typename EngType>
void Engine_Ext_LumpedRLC::Apply2VoltagesImpl(EngType* eng, const unsigned int* order, unsigned int n0, unsigned int n1)
{
	unsigned int **pos = m_Op_Ext_RLC->v_RLC_pos;
	int *dir = m_Op_Ext_RLC->v_RLC_dir;

	// Read engine calculated node voltage
	for (unsigned int k = n0 ; k < n1 ; k++)
	{
		const unsigned int pIdx = order ? order[k] : k;
		v_Vdn[0][pIdx] = eng->EngType::GetVolt(dir[pIdx],pos[0][pIdx],pos[1][pIdx],pos[2][pIdx]);
	}

	// Post process: Calculate node voltage with respect to the lumped RLC auxilliary quantity, J
	for (unsigned int k = n0 ; k < n1 ; k++)
	{
		const unsigned int pIdx = order ? order[k] : k;
		// Calculate updated node voltage, with series and parallel additions
		v_Vdn[0][pIdx] = 	(m_Op_Ext_RLC->v_RLC_vvd[pIdx])*(
							v_Vdn[0][pIdx] - v_Il[pIdx]						// Addition for Parallel inductor
							+
							(m_Op_Ext_RLC->v_RLC_vv2[pIdx])*v_Vdn[2][pIdx]	// Vd[n-2] addition
							+
							(m_Op_Ext_RLC->v_RLC_vj1[pIdx])*v_Jn[1][pIdx]	// J[n-1] addition
							+
							(m_Op_Ext_RLC->v_RLC_vj2[pIdx])*v_Jn[2][pIdx]);	// J[n-2] addition

		// Update J[0]
		v_Jn[0][pIdx] =	(m_Op_Ext_RLC->v_RLC_ib0[pIdx])*(v_Vdn[0][pIdx] - v_Vdn[2][pIdx])
						-
						((m_Op_Ext_RLC->v_RLC_b1[pIdx])*(m_Op_Ext_RLC->v_RLC_ib0[pIdx]))*v_Jn[1][pIdx]
						-
						((m_Op_Ext_RLC->v_RLC_b2[pIdx])*(m_Op_Ext_RLC->v_RLC_ib0[pIdx]))*v_Jn[2][pIdx];
	}


	// Update node voltage
	for (unsigned int k = n0 ; k < n1 ; k++)
	{
		const unsigned int pIdx = order ? order[k] : k;
		eng->EngType::SetVolt(dir[pIdx],pos[0][pIdx],pos[1][pIdx],pos[2][pIdx],v_Vdn[0][pIdx]);
	}
}

void Engine_Ext_LumpedRLC::Apply2Voltages()
{
	if (!m_Op_Ext_RLC->RLC_count)
		return;

	// Iterate J containers
	Rotate(v_Jn);
	ENG_DISPATCH_ARGS(Apply2VoltagesImpl, (const unsigned int*)NULL, 0u, m_Op_Ext_RLC->RLC_count);
}

void Engine_Ext_LumpedRLC::Apply2VoltagesLines(unsigned int x0, unsigned int x1)
{
	if (!m_Op_Ext_RLC->RLC_count)
		return;
	RotateOnce(m_RotatedJ, v_Jn);
	unsigned int n0, n1;
	LineRange(x0, x1, n0, n1);
	// the fused sweep runs on the SSE engines only
	if (n0<n1)
		Apply2VoltagesImpl<Engine_sse>((Engine_sse*)m_Eng, m_Order.data(), n0, n1);
}
