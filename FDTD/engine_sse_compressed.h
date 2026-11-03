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

#ifndef ENGINE_SSE_COMPRESSED_H
#define ENGINE_SSE_COMPRESSED_H

#include "engine_sse.h"
#include "operator_sse_compressed.h"

class Engine_SSE_Compressed : public Engine_sse
{
public:
	static Engine_SSE_Compressed* New(const Operator_SSE_Compressed* op);
	virtual ~Engine_SSE_Compressed();

protected:
	Engine_SSE_Compressed(const Operator_SSE_Compressed* op);
	const Operator_SSE_Compressed* Op;

	virtual void UpdateVoltages(unsigned int startX, unsigned int numX);
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX);

	//! The original updates, for any array layout.
	void UpdateVoltages_Generic(unsigned int startX, unsigned int numX);
	void UpdateCurrents_Generic(unsigned int startX, unsigned int numX);
	//! Same arithmetic in the same order (bit-identical), for the default I-J-K-N layout with
	//! loop invariants in local restrict pointers (union stores otherwise make the compiler reload them).
	void UpdateVoltages_IJKN(unsigned int startX, unsigned int numX);
	void UpdateCurrents_IJKN(unsigned int startX, unsigned int numX);
	//! true if both field arrays use the I-J-K-N layout (component stride 1, z stride 3) and the packed
	//! coefficient tables below exist (not with OPENEMS_NO_PACKED_COEFF or after a failed allocation);
	//! otherwise the generic loops run and Engine_Multithread does not fuse its sweep
	bool UseIJKN() const;

	//! The six coefficient vectors one operator needs for a voltage (or current) update, packed in
	//! access order (vv0 vi0 vv1 vi1 vv2 vi2 / ii0 iv0 ii1 iv1 ii2 iv2): the operator's six separate
	//! tables cost six cache lines per operator, one packed entry sits in 1-2. Plain copies of the
	//! operator's values, so the arithmetic is unchanged. OPENEMS_NO_PACKED_COEFF=1 keeps the tables,
	//! and with them the generic loops and the unfused sweep.
	struct PackedCoeff { f4vector c[6]; };
	std::vector<PackedCoeff, aligned_allocator<PackedCoeff> > m_PackedV, m_PackedI;
	virtual void Init();
	virtual void Reset();
	void BuildPackedCoeffs();
};

#endif // ENGINE_SSE_COMPRESSED_H
