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
	void SetupLines();
	void LineRange(int o, unsigned int x0, unsigned int x1, unsigned int& k0, unsigned int& k1) const;
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
