/*
*	Copyright (C) 2026 the emsKit author
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

#ifndef ENGINE_METAL_DIAMOND_H
#define ENGINE_METAL_DIAMOND_H

#include "engine_sse_compressed.h"
#include "operator_multithread.h"

//! Diamond-blocked Metal engine (--engine=metal-diamond, macOS with ENABLE_METAL).
/*!
  Runs the update equations of Engine_SSE_Compressed on the GPU, K timesteps per tile: the x and y axes are split into
  mountain ranges, which shrink by one cell per half-step, and valley ranges between them, which grow into what the
  mountains left (schedule after luikore's Metal engine, openEMS pull request #228, macos/doc/metal-engine.rst, GPL).
  The fields are the engine's own f4vector arrays, kept in Metal buffers in shared memory, so probes, dumps and the
  extensions read and write the same memory. Every cell sees the operands of the CPU engines in the same order, with
  fma() where clang fuses on arm64 and Metal fast math off, so the fields are bit-identical to the CPU engines when
  those flush subnormals as the GPU does (OPENEMS_FTZ=1).
  The arrays use the N-I-J-K order (components apart), not the I-J-K-N order of the CPU engines.
  Extensions done on the GPU: excitation (voltage and current), Mur ABC, lumped RLC, UPML, Drude/Lorentz materials
  (conducting sheets), Debye materials. With any other extension each timestep runs the (generic) CPU loops of
  Engine_SSE_Compressed on the same memory.
  Knobs: OPENEMS_DIAMOND_K (timesteps per tile, default 4), OPENEMS_DIAMOND_TILE=M,V or M:V (mountain and valley widths in
  cells, default 2K,2K), OPENEMS_DIAMOND_TG (threads per threadgroup, default from the tile size, 256-1024),
  OPENEMS_DIAMOND_BATCH (timesteps per command buffer, default 64), OPENEMS_DIAMOND_TIMING=1 (setup, IterateTS and GPU
  times at the end; 2 also splits the GPU time by phase, one command buffer per dispatch).
*/
class Engine_Metal_Diamond : public Engine_SSE_Compressed
{
public:
	static Engine_Metal_Diamond* New(const Operator_SSE_Compressed* op);
	virtual ~Engine_Metal_Diamond();

	virtual bool IterateTS(unsigned int iterTS);

protected:
	Engine_Metal_Diamond(const Operator_SSE_Compressed* op);
	virtual void Init();
	virtual void Reset();

	//! build the GPU tables and the extension data on the first IterateTS (the extensions are complete then)
	bool SetupGPU();

	struct State;
	State* m_md;
};

//! The compressed SSE operator, creating Engine_Metal_Diamond. It is built as Operator_Multithread builds it (Calc_EC,
//! CalcPEC, the timestep search and Calc_ECOperatorPos on every core; the same operator as the single-threaded setup,
//! which took 2-7x longer on the six benchmark models); the engine itself runs on the GPU.
class Operator_Metal_Diamond : public Operator_Multithread
{
public:
	static Operator_Metal_Diamond* New();
	virtual Engine* CreateEngine();

protected:
	Operator_Metal_Diamond() : Operator_Multithread() {}
};

#endif // ENGINE_METAL_DIAMOND_H
