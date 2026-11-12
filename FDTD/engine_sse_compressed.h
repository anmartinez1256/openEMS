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
#include <cstdint>

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
	//! The loops above, with the coefficients from the packed or the geometry tables (CoefPacked, CoefGeom in the .cpp)
	template<class C> void UpdateVoltages_IJKN_T(unsigned int startX, unsigned int numX, const C& coef, unsigned int y0, unsigned int y1);
	template<class C> void UpdateCurrents_IJKN_T(unsigned int startX, unsigned int numX, const C& coef, unsigned int y0, unsigned int y1);
	//! The same updates for the y lines [y0, y1) of the x lines only, for the tiled sweep of the multithreaded engine,
	//! whose threads share a block of x lines; only with UseIJKN()
	void UpdateVoltagesY(unsigned int startX, unsigned int numX, unsigned int y0, unsigned int y1);
	void UpdateCurrentsY(unsigned int startX, unsigned int numX, unsigned int y0, unsigned int y1);
	//! true if both field arrays use the I-J-K-N layout (component stride 1, z stride 3) and the packed or
	//! the geometry coefficient tables below exist (not with OPENEMS_NO_PACKED_COEFF or after a failed allocation);
	//! otherwise the generic loops run and Engine_Multithread does not fuse its sweep
	bool UseIJKN() const;
	bool HasCoeffTables() const {return !m_PackedV.empty() || !m_GeomV.empty();}

	//! The six coefficient vectors one operator needs for a voltage (or current) update, packed in
	//! access order (vv0 vi0 vv1 vi1 vv2 vi2 / ii0 iv0 ii1 iv1 ii2 iv2): the operator's six separate
	//! tables cost six cache lines per operator, one packed entry sits in 1-2. Plain copies of the
	//! operator's values, so the arithmetic is unchanged. OPENEMS_NO_PACKED_COEFF=1 keeps the tables,
	//! and with them the generic loops and the unfused sweep.
	struct PackedCoeff { f4vector c[6]; };
	std::vector<PackedCoeff, aligned_allocator<PackedCoeff> > m_PackedV, m_PackedI;
	//! Geometry tables, instead of the packed ones when those would exceed 4 MB (a graded mesh with many operators, the
	//! SMA launch: 64410 operators, 12.4 MB packed); OPENEMS_COEF=packed or geom chooses one. Per operator only vi0 vi1 vi2
	//! in m_GeomV (iv0 iv1 iv2 in m_GeomI), 48 B in the order of the operators, and the 16-bit index of its vv0 vv1 vv2
	//! (ii0 ii1 ii2) in a small table of the distinct triplets, m_MatTabV (m_MatTabI): 100 B per operator for both
	//! updates instead of 192. Same values, same arithmetic.
	std::vector<f4vector, aligned_allocator<f4vector> > m_GeomV, m_GeomI, m_MatTabV, m_MatTabI;
	std::vector<uint16_t> m_MatV, m_MatI;
	virtual void Init();
	virtual void Reset();
	void BuildPackedCoeffs();
	bool BuildGeomCoeffs();
	void FreeCoeffTables();
};

#endif // ENGINE_SSE_COMPRESSED_H
