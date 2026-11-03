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

#ifndef ENGINE_EXT_MUR_ABC_H
#define ENGINE_EXT_MUR_ABC_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"
#include "tools/arraylib/array_ij.h"

class Operator_Ext_Mur_ABC;

class Engine_Ext_Mur_ABC : public Engine_Extension
{
public:
	Engine_Ext_Mur_ABC(Operator_Ext_Mur_ABC* op_ext);
	virtual ~Engine_Ext_Mur_ABC();

	virtual void SetNumberOfThreads(int nrThread);
	virtual unsigned int GetUsedHooks() const {OPENEMS_HOOKS_DECLARED_BY(Engine_Ext_Mur_ABC); return PRE_VOLT | POST_VOLT | APPLY_VOLT | (m_Fast ? LINE_HOOKS : 0);}

	virtual void DoPreVoltageUpdates() {Engine_Ext_Mur_ABC::DoPreVoltageUpdates(0);}
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {Engine_Ext_Mur_ABC::DoPostVoltageUpdates(0);}
	virtual void DoPostVoltageUpdates(int threadID);
	virtual void Apply2Voltages() {Engine_Ext_Mur_ABC::Apply2Voltages(0);}
	virtual void Apply2Voltages(int threadID);

	//! The ABC is switched off until this timestep, if an excitation lies on its plane
	unsigned int GetStartTimestep() const {return m_start_TS;}

	virtual void DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1);
	virtual void DoPostVoltageUpdatesLines(unsigned int x0, unsigned int x1);
	virtual void Apply2VoltagesLines(unsigned int x0, unsigned int x1);
	virtual void GetCoupledLines(std::vector<unsigned int>& lines) const;

protected:
	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPostVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng, int threadID);

	Operator_Ext_Mur_ABC* m_Op_mur;

	inline bool IsActive() {if (m_Eng->GetNumberOfTimesteps()<m_start_TS) return false; return true;}
	unsigned int m_start_TS;

	// See detailed comments in operator_ext_mur_abc.h, not repeated here.
	int m_ny, m_nyP, m_nyPP;
	unsigned int m_LineNr;
	int m_LineNr_Shift;
	unsigned int m_numLines[2];

	std::vector<unsigned int> m_start;
	std::vector<unsigned int> m_numX;

	ArrayLib::ArrayIJ<FDTD_FLOAT>& m_Mur_Coeff_nyP;
	ArrayLib::ArrayIJ<FDTD_FLOAT>& m_Mur_Coeff_nyPP;
	ArrayLib::ArrayIJ<FDTD_FLOAT> m_volt_nyP; //n+1 direction
	ArrayLib::ArrayIJ<FDTD_FLOAT> m_volt_nyPP; //n+2 direction

	//! Fast path for the SSE engines: the float index of every face cell in the f4vector field array
	//! (component 0), so the hooks need no z%numVectors, z/numVectors per cell. The other component
	//! and the shifted line are constant offsets from it.
	bool m_Fast;
	bool m_FastChecked;
	std::vector<unsigned int> m_FloatIndex;
	ptrdiff_t m_OffsetNyP, m_OffsetNyPP, m_OffsetShift;
	//! The face cells of the line hooks by x-line, in memory order: runs of four cells that fill one
	//! f4vector (its four lanes, cells z, z+n_v, z+2n_v, z+3n_v of a z-line of an x- or y-face) are
	//! updated as vectors, the other cells one by one.
	//! Line x: vectors m_VecStart[x] .. m_VecStart[x+1], cells m_ScStart[x] .. m_ScStart[x+1].
	//! Own coefficient and save arrays in that order (the threaded hooks keep m_volt_nyP/nyPP).
	std::vector<unsigned int> m_VecStart, m_VecIndex;   //!< f4vector index of component 0
	std::vector<f4vector> m_VecCoeffP, m_VecCoeffPP, m_VecP, m_VecPP;
	std::vector<unsigned int> m_ScStart, m_ScIndex;     //!< float index of component 0
	std::vector<FDTD_FLOAT> m_ScCoeffP, m_ScCoeffPP, m_ScP, m_ScPP;
	void SetupFast();
	void DoPreVoltageUpdatesFast(int threadID);
	void DoPostVoltageUpdatesFast(int threadID);
	void Apply2VoltagesFast(int threadID);
};

#endif // ENGINE_EXT_MUR_ABC_H
