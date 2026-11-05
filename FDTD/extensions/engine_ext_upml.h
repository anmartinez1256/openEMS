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

#ifndef ENGINE_EXT_UPML_H
#define ENGINE_EXT_UPML_H

#include "engine_extension.h"
#include "tools/global.h"
#include "tools/aligned_allocator.h"
#include "tools/array_ops.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_UPML;

class Engine_Ext_UPML : public Engine_Extension
{
	friend class Engine_Metal_Diamond;
public:
	Engine_Ext_UPML(Operator_Ext_UPML* op_ext);
	virtual ~Engine_Ext_UPML();

	virtual void SetNumberOfThreads(int nrThread);
	virtual unsigned int GetUsedHooks() const {OPENEMS_HOOKS_DECLARED_BY(Engine_Ext_UPML); return PRE_VOLT | POST_VOLT | PRE_CURR | POST_CURR | PREPOST_WRITES_FIELDS | ((m_Vec || m_Sse) ? LINE_HOOKS | TS_LOCAL : 0);}

	virtual void DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1);
	virtual void DoPostVoltageUpdatesLines(unsigned int x0, unsigned int x1);
	virtual void DoPreCurrentUpdatesLines(unsigned int x0, unsigned int x1);
	virtual void DoPostCurrentUpdatesLines(unsigned int x0, unsigned int x1);

	virtual void DoPreVoltageUpdates() {Engine_Ext_UPML::DoPreVoltageUpdates(0);};
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {Engine_Ext_UPML::DoPostVoltageUpdates(0);};
	virtual void DoPostVoltageUpdates(int threadID);

	virtual void DoPreCurrentUpdates() {Engine_Ext_UPML::DoPreCurrentUpdates(0);};
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() {Engine_Ext_UPML::DoPostCurrentUpdates(0);};
	virtual void DoPostCurrentUpdates(int threadID);

protected:
	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPostVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPostCurrentUpdatesImpl(EngType* eng, int threadID);

	Operator_Ext_UPML* m_Op_UPML;

	std::vector<unsigned int> m_start;
	std::vector<unsigned int> m_numX;

	//! f4vector path for the SSE engines when the PML covers the full z range (see SetupVectorized())
	bool m_Vec;
	bool m_VecChecked;
	unsigned int m_nvec;
	std::vector<unsigned int> m_RowStart, m_RowNum; //!< (x,y)-rows of the PML per thread
	//! coefficients and fluxes in the engine order: index ((x*Ny+y)*nvec+v)*3+n, lane l = z/nvec
	std::vector<f4vector, aligned_allocator<f4vector> > m_f4_vv, m_f4_vvfo, m_f4_vvfn, m_f4_ii, m_f4_iifo, m_f4_iifn;
	std::vector<f4vector, aligned_allocator<f4vector> > m_f4_volt_flux, m_f4_curr_flux;
	std::vector<f4vector, aligned_allocator<f4vector> > m_f4_LaneMask; //!< per z-vector: all bits set in lanes inside the mesh
	void SetupVectorized();
	//! scalar path for the SSE engines when the PML is not a full-z slab: the template above with the
	//! f4vector lane offsets and the coefficient rows precomputed (same arithmetic, same order)
	bool m_Sse;
	std::vector<unsigned int> m_zoff; //!< float offset of local z inside an (n,x,y) row of f4vectors
	//! The six coefficients of a PML cell component take only a few hundred distinct values (the
	//! profile along the PML direction times the materials), so the scalar SSE path reads them through
	//! a per-(row, n, z) index into a small table (24 B per entry) instead of six per-cell arrays
	//! (72 B per cell). Same values, same arithmetic. OPENEMS_NO_PML_TABLE=1 keeps the arrays.
	struct Coef { FDTD_FLOAT vv, vvfo, vvfn, ii, iifo, iifn; };
	std::vector<Coef> m_CoefTab;
	std::vector<unsigned int> m_CoefIdx; //!< index ((r*3+n)*NZ + lz), r = lx*NY+ly
	bool m_CoefTable;
	void SetupCoefTable();
	//! the same for the f4vector path: one entry (96 B) per distinct set, index per (x,y,z-vector,n)
	struct Coef4 { f4vector vv, vvfo, vvfn, ii, iifo, iifn; };
	std::vector<Coef4, aligned_allocator<Coef4> > m_CoefTab4;
	std::vector<unsigned int> m_CoefIdx4;
	void SetupCoefTable4();
	//! y-vector form of the scalar SSE path (with the table): four consecutive y-rows of an x-line are one group, lane
	//! l = row 4g+l, with the fluxes and the coefficient sets stored per group, so a group takes one vector operation
	//! per step of the update; the field values are gathered and scattered lane by lane. Same operations per cell as
	//! the scalar path. The last group of an x-line, when it has fewer than four rows, runs lane by lane on the same
	//! arrays; the rows are always handed over in whole groups. OPENEMS_PML_YVEC=0: the scalar rows.
	bool m_YVec;
	unsigned int m_NG; //!< groups per x-line: ceil(NY/4)
	std::vector<f4vector, aligned_allocator<f4vector> > m_y_volt_flux, m_y_curr_flux; //!< index ((lx*m_NG+g)*3+n)*NZ+lz
	std::vector<Coef4, aligned_allocator<Coef4> > m_CoefTabY; //!< distinct coefficient sets of a group
	std::vector<unsigned int> m_CoefIdxY; //!< same index as the fluxes
	void SetupYVec();
	//! HOOK 0: pre-voltage, 1: post-voltage, 2: pre-current, 3: post-current update of the rows [r0, r1)
	template <int HOOK> void YVecRows(unsigned int r0, unsigned int r1);
	void SetupScalarSSE();
	void DoPreVoltageUpdatesSSERows(unsigned int r0, unsigned int r1);
	void DoPostVoltageUpdatesSSERows(unsigned int r0, unsigned int r1);
	void DoPreCurrentUpdatesSSERows(unsigned int r0, unsigned int r1);
	void DoPostCurrentUpdatesSSERows(unsigned int r0, unsigned int r1);
	void DoPreVoltageUpdatesVec(int threadID);
	void DoPostVoltageUpdatesVec(int threadID);
	void DoPreCurrentUpdatesVec(int threadID);
	void DoPostCurrentUpdatesVec(int threadID);
	//! the f4vector hooks for the PML (x,y)-rows [r0, r1)
	void DoPreVoltageUpdatesRows(unsigned int r0, unsigned int r1);
	void DoPostVoltageUpdatesRows(unsigned int r0, unsigned int r1);
	void DoPreCurrentUpdatesRows(unsigned int r0, unsigned int r1);
	void DoPostCurrentUpdatesRows(unsigned int r0, unsigned int r1);
	//! the PML rows of the x-lines [x0, x1), false if there are none
	bool LineRows(unsigned int x0, unsigned int x1, unsigned int& r0, unsigned int& r1) const;

	ArrayLib::ArrayNIJK<FDTD_FLOAT> volt_flux;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> curr_flux;
};

#endif // ENGINE_EXT_UPML_H
