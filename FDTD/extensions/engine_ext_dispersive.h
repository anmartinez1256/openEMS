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

#ifndef ENGINE_EXT_DISPERSIVE_H
#define ENGINE_EXT_DISPERSIVE_H

#include "engine_extension.h"
#include <vector>
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"
#include "tools/array_ops.h"
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
//! four consecutive floats at any float address, for the merged ADE line hooks
inline v4sf Load4(const float* p) {v4sf v; memcpy(&v, p, sizeof(v)); return v;}
inline void Store4(float* p, v4sf v) {memcpy(p, &v, sizeof(v));}
#endif

class Operator_Ext_Dispersive;

class Engine_Ext_Dispersive : public Engine_Extension
{
	friend class Engine_Metal_Diamond;
public:
	Engine_Ext_Dispersive(Operator_Ext_Dispersive* op_ext_disp);
	virtual ~Engine_Ext_Dispersive();

	virtual void Apply2Voltages();
	virtual void Apply2Current();

	virtual void SetNumberOfThreads(int nrThread);
	virtual unsigned int GetUsedHooks() const;
	//! the hooks of the dispersive base (for subclasses that add their own)
	unsigned int DispersiveHooks() const;
	virtual void Apply2VoltagesLines(unsigned int x0, unsigned int x1);
	virtual void Apply2CurrentLines(unsigned int x0, unsigned int x1);

protected:
	//! Line hooks for the SSE engines: the positions of every order sorted by x, with the float index of each
	//! component in the f4vector field arrays precomputed (same per-position arithmetic as the templates).
	bool m_LinesReady;
	bool m_LinesChecked;
	std::vector<std::vector<unsigned int> > m_SortedIdx; //!< [order][k]: original position index, sorted by x
	std::vector<std::vector<unsigned int> > m_SortedX;   //!< [order][k]: its x
	std::vector<std::vector<unsigned int> > m_FloatIdx;  //!< [order][3*k+n]: float index of component n
	//! All orders with voltage (current) ADE have the same positions, already in x order (e.g. the two orders of a
	//! conducting sheet): the line hooks then run these orders back to back per position, four positions at a time,
	//! with the same operations per float in the same order (OPENEMS_ADE_MERGED=0: one order after the other)
	bool m_VoltMerged, m_CurrMerged;
	std::vector<int> m_VoltOrders, m_CurrOrders; //!< those orders
	bool MergeOrders(const bool* on, std::vector<int>& orders) const;
	//! the merged Apply2*Lines(): field[float index] -= ADE of every order, in order
	void ApplyMerged(float* field, FDTD_FLOAT*** ade, const std::vector<int>& orders, unsigned int x0, unsigned int x1);
	void SetupLines();
	//! called at the end of SetupLines(): subclasses build their packed groups here
	virtual void SetupPacked() {}
	void LineRange(int o, unsigned int x0, unsigned int x1, unsigned int& k0, unsigned int& k1) const;

	//! Groups for packed hooks: the positions of order 0 (in x order, each listed once) in groups of up to four
	//! positions of ONE x-line, so a group is never shared by two threads; built by SetupGroups(), false (and empty)
	//! when the positions do not allow it
	bool SetupGroups();
	void ClearGroups();
	//! the groups of the x-lines [x0, x1)
	void GroupRange(unsigned int x0, unsigned int x1, unsigned int& g0, unsigned int& g1) const;
	std::vector<unsigned int> m_GroupX;      //!< [g]: x-line of group g (ascending)
	std::vector<unsigned int> m_GroupK;      //!< [g]: first position (index k of order 0) of group g
	std::vector<unsigned char> m_GroupLanes; //!< [g]: positions in group g, 1..4
	std::vector<unsigned int> m_GroupFi;     //!< [12*g+4*n+l]: float index of component n of lane l (unused lanes: lane 0's)
	//! the bytes of the engine's voltage and current arrays (the limit for packed copies: twice this)
	uint64_t FieldBytes() const;
	bool AnyVoltADE() const;
	bool AnyCurrADE() const;

	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng);

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng);

	Operator_Ext_Dispersive* m_Op_Ext_Disp;

	//! Dispersive order
	int m_Order;

	//! ADE currents
	// Array setup: curr_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***curr_ADE;

	//! ADE voltages
	// Array setup: volt_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***volt_ADE;
};

#endif // ENGINE_EXT_DISPERSIVE_H
