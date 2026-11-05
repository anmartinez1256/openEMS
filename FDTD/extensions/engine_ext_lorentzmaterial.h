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

#ifndef ENGINE_EXT_LORENTZMATERIAL_H
#define ENGINE_EXT_LORENTZMATERIAL_H

#include "engine_ext_dispersive.h"

class Operator_Ext_LorentzMaterial;

class Engine_Ext_LorentzMaterial : public Engine_Ext_Dispersive
{
	friend class Engine_Metal_Diamond;
public:
	Engine_Ext_LorentzMaterial(Operator_Ext_LorentzMaterial* op_ext_lorentz);
	virtual ~Engine_Ext_LorentzMaterial();

	virtual void DoPreVoltageUpdates();

	virtual void DoPreCurrentUpdates();

	virtual unsigned int GetUsedHooks() const;
	virtual void DoPreVoltageUpdatesLines(unsigned int x0, unsigned int x1);
	virtual void DoPreCurrentUpdatesLines(unsigned int x0, unsigned int x1);

protected:
	//! the merged DoPre*UpdatesLines() (see m_VoltMerged): per position the ADE update of every order
	void PreMerged(const float* field, FDTD_FLOAT*** ade, FDTD_FLOAT*** lor, FDTD_FLOAT*** c_int, FDTD_FLOAT*** c_ext,
				   FDTD_FLOAT*** c_lor, const bool* lor_on, const std::vector<int>& orders, unsigned int x0, unsigned int x1);

	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng);

	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng);

	Operator_Ext_LorentzMaterial* m_Op_Ext_Lor;

	//! ADE Lorentz voltages
	// Array setup: volt_Lor_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***volt_Lor_ADE;

	//! ADE Lorentz currents
	// Array setup: curr_Lor_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***curr_Lor_ADE;

};

#endif // ENGINE_EXT_LORENTZMATERIAL_H
