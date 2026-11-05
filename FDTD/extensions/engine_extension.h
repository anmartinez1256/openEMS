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

#ifndef ENGINE_EXTENSION_H
#define ENGINE_EXTENSION_H

#include <typeinfo>

#define ENG_EXT_PRIO_DEFAULT 0 //default engine extension priority

// priority definitions for some important extensions
#define ENG_EXT_PRIO_STEADYSTATE		+2e6  //steady state extension priority
#define ENG_EXT_PRIO_UPML				+1e6  //unaxial pml extension priority
#define ENG_EXT_PRIO_CYLINDER			+1e5  //cylindrial extension priority
#define ENG_EXT_PRIO_TFSF				+5e4  //total-field/scattered-field extension priority
#define ENG_EXT_PRIO_EXCITATION			-1000 //excitation priority
#define ENG_EXT_PRIO_CYLINDERMULTIGRID	-3000 //cylindrial multi-grid extension priority

#include <string>
#include <vector>

class Operator_Extension;
class Engine;

//! Abstract base-class for all engine extensions
//! Put this first in every GetUsedHooks() override: a subclass that does not override GetUsedHooks() itself then
//! gets LEGACY_HOOKS (every hook called), so it can never inherit a declaration that misses the hooks it adds
//! (Engine_Ext_DebyeMaterial once inherited Engine_Ext_Dispersive's and lost its Pre/Post voltage hooks).
#define OPENEMS_HOOKS_DECLARED_BY(CLASS) do { if (typeid(*this) != typeid(CLASS)) return Engine_Extension::LEGACY_HOOKS; } while (0)

class Engine_Extension
{
public:
	virtual ~Engine_Extension();

	virtual void SetNumberOfThreads(int nrThread);

	//! Hook flags for GetUsedHooks()
	enum Hooks {PRE_VOLT=1, POST_VOLT=2, APPLY_VOLT=4, PRE_CURR=8, POST_CURR=16, APPLY_CURR=32,
				PREPOST_WRITES_FIELDS=64, //!< its Pre/Post hooks do change engine fields (e.g. the UPML)
				LINE_HOOKS=128, //!< implements the *Lines() hooks below, for the fused sweep
				TS_LOCAL=256,   //!< its line hooks also work while the threads are at different timesteps, see below
				LEGACY_HOOKS=~0u};
	//! Which of the six update hooks this extension implements (or'ed Hooks).
	/*!
	  The multithreaded engine skips the hooks an extension does not implement, and runs the
	  Pre/Post hooks of all extensions that declare them before a single barrier, as long as they
	  only read engine fields (as documented below). An extension whose Pre/Post hooks write engine
	  fields must add PREPOST_WRITES_FIELDS: it then gets a barrier before and after each of these
	  hooks. Apply2* hooks keep one barrier per extension, because they write engine fields in
	  priority order.
	  The default LEGACY_HOOKS keeps the old behaviour: every hook is called, followed by a barrier.
	  Every override must start with OPENEMS_HOOKS_DECLARED_BY(ThisClass) and every subclass that adds a hook
	  must declare its own set: a declaration is never inherited.
	  */
	virtual unsigned int GetUsedHooks() const {return LEGACY_HOOKS;}

	//! This method will be called __before__ the main engine does the usual voltage updates. This method may __not__ change the engine voltages!!!
	virtual void DoPreVoltageUpdates() {}
	virtual void DoPreVoltageUpdates(int threadID);
	//! This method will be called __after__ the main engine does the usual voltage updates. This method may __not__ change the engine voltages!!!
	virtual void DoPostVoltageUpdates() {}
	virtual void DoPostVoltageUpdates(int threadID);
	//! This method will be called __after__ all updates to the voltages and extensions and may add/set its results to the engine voltages, but may __not__ rely on the current value of the engine voltages!!!
	virtual void Apply2Voltages() {}
	virtual void Apply2Voltages(int threadID);

	//! This method will be called __before__ the main engine does the usual current updates. This method may __not__ change the engine current!!!
	virtual void DoPreCurrentUpdates() {}
	virtual void DoPreCurrentUpdates(int threadID);
	//! This method will be called __after__ the main engine does the usual current updates. This method may __not__ change the engine current!!!
	virtual void DoPostCurrentUpdates() {}
	virtual void DoPostCurrentUpdates(int threadID);
	//! This method will be called __after__ all updates to the current and extensions and may add/set its results to the engine current, but may __not__ rely on the current value of the engine current!!!
	virtual void Apply2Current() {}
	virtual void Apply2Current(int threadID);

	//! Hooks for the fused sweep of the multithreaded engine (extensions with LINE_HOOKS).
	/*!
	  The fused sweep runs the voltage update, its hooks and the current update x-line by x-line
	  and calls these instead of the hooks above: the same work, for the x-lines [x0, x1) only, with
	  the same order of hooks and extensions for every cell. A line hook may touch the fields of its
	  own lines only (and of the other line of a pair (x, x+1) from GetCoupledLines()). Different threads
	  call them for disjoint line ranges at the same time.
	  Invariant: the engine never updates the currents of the last x-line (NX-1, they stay 0) and no
	  excitation entry can sit on it, so the fused sweep never calls the *CurrentLines() hooks for it;
	  a current hook of an extension must therefore be a no-op on NX-1 (true for every extension in tree).
	  TS_LOCAL: the line hooks keep no state shared by x-lines that changes from one timestep to the next, and
	  read the timestep through GetNumberOfTimesteps() only. When every extension declares it, the threads of the
	  fused sweep wait for their neighbours only, so they can be timesteps apart from each other (each one counts
	  its own timesteps, which GetNumberOfTimesteps() returns on that thread).
	  */
	virtual void DoPreVoltageUpdatesLines(unsigned int /*x0*/, unsigned int /*x1*/) {}
	virtual void DoPostVoltageUpdatesLines(unsigned int /*x0*/, unsigned int /*x1*/) {}
	virtual void Apply2VoltagesLines(unsigned int /*x0*/, unsigned int /*x1*/) {}
	virtual void DoPreCurrentUpdatesLines(unsigned int /*x0*/, unsigned int /*x1*/) {}
	virtual void DoPostCurrentUpdatesLines(unsigned int /*x0*/, unsigned int /*x1*/) {}
	virtual void Apply2CurrentLines(unsigned int /*x0*/, unsigned int /*x1*/) {}
	//! The lower line x of every pair (x, x+1) that one of the voltage line hooks couples: a hook on x that reads x+1,
	//! or a hook on x+1 that reads x (e.g. a Mur ABC on an x-face). Both lines are then swept in one block by one thread.
	virtual void GetCoupledLines(std::vector<unsigned int>& /*lines*/) const {}

	//! Set the Engine to this extension. This will usually done automatically by Engine::AddExtension
	virtual void SetEngine(Engine* eng) {m_Eng=eng;}

	//! Get the operator extension this engine extension was created by
	Operator_Extension* GetOperatorExtension() const {return m_Op_ext;}

	//! Get the priority for this extension
	virtual int GetPriority() const {return m_Priority;}

	//! Set the priority for this extension
	virtual void SetPriority(int val) {m_Priority=val;}

	virtual bool operator< (const Engine_Extension& other);

	virtual std::string GetExtensionName() const;

protected:
	Engine_Extension(Operator_Extension* op_ext);

	Operator_Extension* m_Op_ext;
	Engine* m_Eng;

	int m_Priority;

	int m_NrThreads;
};

#endif // ENGINE_EXTENSION_H
