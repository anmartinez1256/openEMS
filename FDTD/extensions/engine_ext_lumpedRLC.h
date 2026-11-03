/*
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

#ifndef ENGINE_EXT_LUMPEDRLC_H
#define ENGINE_EXT_LUMPEDRLC_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"
#include <atomic>
#include <mutex>
#include <vector>

class Operator_Ext_LumpedRLC;

class Engine_Ext_LumpedRLC : public Engine_Extension
{
	friend class Operator_Ext_LumpedRLC;
	friend class Operator;
	friend class ContinuousStructure;

public:

	Engine_Ext_LumpedRLC(Operator_Ext_LumpedRLC *op_ext_RLC);
	virtual ~Engine_Ext_LumpedRLC();

	virtual unsigned int GetUsedHooks() const {OPENEMS_HOOKS_DECLARED_BY(Engine_Ext_LumpedRLC); return PRE_VOLT | APPLY_VOLT | LINE_HOOKS;}

	virtual void DoPreVoltageUpdates();
	virtual void Apply2Voltages();

	virtual void DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1);
	virtual void Apply2VoltagesLines(unsigned int x0, unsigned int x1);

protected:
	//! update the elements order[n0..n1) (the elements n0..n1 if order is NULL)
	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng, const unsigned int* order, unsigned int n0, unsigned int n1);
	void UpdateInductorCurrents(const unsigned int* order, unsigned int n0, unsigned int n1);

	//! the [n],[n-1],[n-2] containers move on once per timestep
	static void Rotate(FDTD_FLOAT** c);
	//! line hooks: the first call of a timestep rotates (the other threads wait for it)
	void RotateOnce(std::atomic<int>& stamp, FDTD_FLOAT** c);
	std::mutex m_RotateMutex;
	std::atomic<int> m_RotatedVd, m_RotatedJ; //!< timestep of the last rotation by a line hook

	//! the elements sorted by x-line, for the line hooks
	std::vector<unsigned int> m_Order, m_X;
	//! elements [n0, n1) of m_Order are on the x-lines [x0, x1)
	void LineRange(unsigned int x0, unsigned int x1, unsigned int& n0, unsigned int& n1) const;

	Operator_Ext_LumpedRLC* m_Op_Ext_RLC;

	// Auxilliary containers

	// Array setup: volt_C_ADE[mesh_pos]
	FDTD_FLOAT *v_Il;		// Container for current on inductor- Parallel RLC

	FDTD_FLOAT **v_Vdn;		// Container for nodal vd at [n],[n-1],[n-2]
	FDTD_FLOAT **v_Jn;		// Container for nodal J at [n],[n-1],[n-2]

};

#endif // ENGINE_EXT_LUMPEDRLC_H
