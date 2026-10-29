/*
*	Copyright (C) 2011 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_EXCITATION_H
#define ENGINE_EXT_EXCITATION_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_Excitation;

class Engine_Ext_Excitation : public Engine_Extension
{
public:
	Engine_Ext_Excitation(Operator_Ext_Excitation* op_ext);
	virtual ~Engine_Ext_Excitation();

	virtual unsigned int GetUsedHooks() const {return APPLY_VOLT | APPLY_CURR | LINE_HOOKS;}

	virtual void Apply2Voltages();
	virtual void Apply2Current();

	virtual void Apply2VoltagesLines(unsigned int x0, unsigned int x1);
	virtual void Apply2CurrentLines(unsigned int x0, unsigned int x1);

protected:
	//! excite the points order[n0..n1) (the points n0..n1 if order is NULL)
	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng, const unsigned int* order, unsigned int n0, unsigned int n1);

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng, const unsigned int* order, unsigned int n0, unsigned int n1);

	Operator_Ext_Excitation* m_Op_Exc;

	//! the excitation points sorted by x-line (stable, so points on the same cell keep their order), for the line hooks
	std::vector<unsigned int> m_VoltOrder, m_VoltX;
	std::vector<unsigned int> m_CurrOrder, m_CurrX;
	static void SortByLine(const unsigned int* x, unsigned int count, std::vector<unsigned int>& order, std::vector<unsigned int>& sorted_x);
};

#endif // ENGINE_EXT_EXCITATION_H
