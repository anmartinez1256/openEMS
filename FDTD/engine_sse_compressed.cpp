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

#include "engine_sse_compressed.h"
#include "tools/global.h"
#include "tools/useful.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

// enable AltiVec mode on ppc64le in GCC's emmintrin.h
#include <boost/predef.h>
#if BOOST_ARCH_PPC_64 && BOOST_ENDIAN_LITTLE_BYTE
#define NO_WARN_X86_INTRINSICS
#include <emmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

using std::cout;
using std::endl;

Engine_SSE_Compressed* Engine_SSE_Compressed::New(const Operator_SSE_Compressed* op)
{
	cout << "Create FDTD engine (compressed SSE)" << endl;
	Engine_SSE_Compressed* e = new Engine_SSE_Compressed(op);
	e->Init();
	return e;
}

Engine_SSE_Compressed::Engine_SSE_Compressed(const Operator_SSE_Compressed* op) : Engine_sse(op)
{
	Op = op;
}

Engine_SSE_Compressed::~Engine_SSE_Compressed()
{
}

void Engine_SSE_Compressed::Init()
{
	Engine_sse::Init();
	BuildPackedCoeffs();
}

void Engine_SSE_Compressed::Reset()
{
	FreeCoeffTables();
	Engine_sse::Reset();
}

void Engine_SSE_Compressed::FreeCoeffTables()
{
	std::vector<PackedCoeff, aligned_allocator<PackedCoeff> >().swap(m_PackedV);
	std::vector<PackedCoeff, aligned_allocator<PackedCoeff> >().swap(m_PackedI);
	std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_GeomV);
	std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_GeomI);
	std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_MatTabV);
	std::vector<f4vector, aligned_allocator<f4vector> >().swap(m_MatTabI);
	std::vector<uint16_t>().swap(m_MatV);
	std::vector<uint16_t>().swap(m_MatI);
}

void Engine_SSE_Compressed::BuildPackedCoeffs()
{
	FreeCoeffTables();
	if (EnvFlag("OPENEMS_NO_PACKED_COEFF"))
		return;
	const size_t numOps = Op->f4_vv_Compressed[0].size();
	for (int n=0; n<3; ++n)
		if ((Op->f4_vv_Compressed[n].size()!=numOps) || (Op->f4_vi_Compressed[n].size()!=numOps) ||
		    (Op->f4_ii_Compressed[n].size()!=numOps) || (Op->f4_iv_Compressed[n].size()!=numOps))
			return;
	// the layout of the coefficients: the geometry tables when the packed ones would exceed 4 MB (on the Apple M5 the
	// SMA launch's 12.4 MB of packed tables run 3-7 % faster as geometry tables at 8 threads and with the automatic
	// thread count, within the noise at 1 and 4; models with packed tables of 1.6 MB and less lose 2-5 % at 4 threads),
	// else packed; OPENEMS_COEF=packed or geom chooses one
	std::string layout = (192*numOps > ((size_t)4<<20)) ? "geom" : "packed";
	if (const char* c = getenv("OPENEMS_COEF"))
		if (*c)
			layout = c;
	if ((layout=="geom") && BuildGeomCoeffs())
		return;
	// 192 B per operator, at most one operator per z-vector: never more than twice the field memory.
	// Only a failed allocation skips the tables (both are freed then).
	try
	{
		m_PackedV.resize(numOps);
		m_PackedI.resize(numOps);
	}
	catch (const std::bad_alloc&)
	{
		FreeCoeffTables();
		std::cerr << "Engine_SSE_Compressed: out of memory for the packed coefficient tables, not packed: "
				  << "the generic update loops run (and the multithreaded engine does not fuse its sweep)" << std::endl;
		return;
	}
	for (size_t i=0; i<numOps; ++i)
		for (int n=0; n<3; ++n)
		{
			m_PackedV[i].c[2*n]   = Op->f4_vv_Compressed[n][i];
			m_PackedV[i].c[2*n+1] = Op->f4_vi_Compressed[n][i];
			m_PackedI[i].c[2*n]   = Op->f4_ii_Compressed[n][i];
			m_PackedI[i].c[2*n+1] = Op->f4_iv_Compressed[n][i];
		}
}

namespace
{
// three f4vectors by their bits (one operator's vv0 vv1 vv2 or ii0 ii1 ii2)
struct F12Key
{
	uint32_t w[12];
	bool operator==(const F12Key& o) const {return memcmp(w, o.w, sizeof(w))==0;}
};
struct F12KeyHash
{
	size_t operator()(const F12Key& k) const
	{
		uint64_t h = 1469598103934665603ull;
		for (int i=0; i<12; ++i)
			h = (h ^ k.w[i]) * 1099511628211ull;
		return (size_t)(h ^ (h>>32));
	}
};
typedef std::vector<f4vector, aligned_allocator<f4vector> > F4Table;
// the index of operator k's triplet c[0..2][k] in tab (appended when new); false when tab is full (65535 triplets)
bool TripletIndex(std::unordered_map<F12Key, uint16_t, F12KeyHash>& lut, F4Table& tab, const F4Table* c, size_t k, uint16_t& id)
{
	F12Key key;
	for (int n=0; n<3; ++n)
		memcpy(key.w+4*n, &c[n][k], 4*sizeof(uint32_t));
	auto it = lut.find(key);
	if (it!=lut.end())
	{
		id = it->second;
		return true;
	}
	if (lut.size()>=65535)
		return false;
	id = (uint16_t)lut.size();
	lut.emplace(key, id);
	for (int n=0; n<3; ++n)
		tab.push_back(c[n][k]);
	return true;
}
}

bool Engine_SSE_Compressed::BuildGeomCoeffs()
{
	const size_t numOps = Op->f4_vv_Compressed[0].size();
	try
	{
		std::unordered_map<F12Key, uint16_t, F12KeyHash> lutV, lutI;
		m_GeomV.resize(3*numOps);
		m_GeomI.resize(3*numOps);
		m_MatV.resize(numOps);
		m_MatI.resize(numOps);
		for (size_t k=0; k<numOps; ++k)
		{
			for (int n=0; n<3; ++n)
			{
				m_GeomV[3*k+n] = Op->f4_vi_Compressed[n][k];
				m_GeomI[3*k+n] = Op->f4_iv_Compressed[n][k];
			}
			if (!TripletIndex(lutV, m_MatTabV, Op->f4_vv_Compressed, k, m_MatV[k]) ||
				!TripletIndex(lutI, m_MatTabI, Op->f4_ii_Compressed, k, m_MatI[k]))
			{
				FreeCoeffTables();
				return false;
			}
		}
	}
	catch (const std::bad_alloc&)
	{
		FreeCoeffTables();
		return false;
	}
	if (g_settings.GetVerboseLevel()>0)
		cout << "Engine_SSE_Compressed: geometry coefficient tables, " << numOps << " operators, " << m_MatTabV.size()/3 << " + "
			 << m_MatTabI.size()/3 << " distinct vv / ii triplets (" << (96*numOps + 4*numOps + 48*(m_MatTabV.size()/3+m_MatTabI.size()/3))/1024
			 << " KB instead of " << 192*numOps/1024 << " KB packed)" << endl;
	return true;
}

void Engine_SSE_Compressed::UpdateVoltages_Generic(unsigned int startX, unsigned int numX)
{
	f4vector* f4_volt = f4_volt_ptr->data();
	f4vector* f4_curr = f4_curr_ptr->data();

	unsigned int pos[3];
	unsigned int v_pos;
	unsigned int i_pos;
	unsigned i_pos_z_start, i_pos_z_end;
	int i_shift[3]; // shift of current index by x/y/z
	f4vector temp;

	// array index shift for x/y/z field component
	int v_N_shift  = f4_volt_ptr->stride(0);
	int i_N_shift  = f4_curr_ptr->stride(0);

	// index shift for a z-position
	i_shift[2] = f4_curr_ptr->stride(3);

	pos[0] = startX;
	unsigned int index=0;
	for (unsigned int posX=0; posX<numX; ++posX)
	{
		// index shift for a x-position
		i_shift[0] = (pos[0]>0)*f4_curr_ptr->stride(1);
		for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
		{
			// index shift for a y-position
			i_shift[1] = (pos[1]>0)*f4_curr_ptr->stride(2);
			for (pos[2]=1; pos[2]<numVectors; ++pos[2])
			{
				index = Op->m_Op_index(pos[0], pos[1], pos[2]);
				i_pos = f4_curr_ptr->linearIndex({0, pos[0], pos[1], pos[2]});
				v_pos = f4_volt_ptr->linearIndex({0, pos[0], pos[1], pos[2]});

				// x-polarization
				f4_volt[v_pos].v *= Op->f4_vv_Compressed[0][index].v;
				f4_volt[v_pos].v +=
				    Op->f4_vi_Compressed[0][index].v * (
				        f4_curr[i_pos + 2*i_N_shift             ].v -
				        f4_curr[i_pos + 2*i_N_shift - i_shift[1]].v -
				        f4_curr[i_pos +   i_N_shift             ].v +
				        f4_curr[i_pos +   i_N_shift - i_shift[2]].v
				    );

				// y-polarization
				v_pos += v_N_shift; // go to next component
				f4_volt[v_pos].v *= Op->f4_vv_Compressed[1][index].v;
				f4_volt[v_pos].v +=
				    Op->f4_vi_Compressed[1][index].v * (
				        f4_curr[i_pos                           ].v -
				        f4_curr[i_pos               - i_shift[2]].v -
				        f4_curr[i_pos + 2*i_N_shift             ].v +
				        f4_curr[i_pos + 2*i_N_shift - i_shift[0]].v
				    );

				// z-polarization
				v_pos += v_N_shift; // go to next component
				f4_volt[v_pos].v *= Op->f4_vv_Compressed[2][index].v;
				f4_volt[v_pos].v +=
				    Op->f4_vi_Compressed[2][index].v * (
				        f4_curr[i_pos + i_N_shift             ].v -
				        f4_curr[i_pos + i_N_shift - i_shift[0]].v -
				        f4_curr[i_pos                         ].v +
				        f4_curr[i_pos             - i_shift[1]].v
				    );
			}

			i_pos_z_start = f4_curr_ptr->linearIndex({0, pos[0], pos[1], 0});
			i_pos_z_end   = f4_curr_ptr->linearIndex({0, pos[0], pos[1], numVectors-1});

			// for pos[2] = 0
			// x-polarization
			index = Op->m_Op_index(pos[0], pos[1], 0);
#ifdef __SSE2__
			temp.v = (__m128)_mm_slli_si128(
			             (__m128i)f4_curr[i_pos_z_end + i_N_shift].v, 4
			         );
#else
			temp.f[0] = 0;
			temp.f[1] = f4_curr[i_pos_z_end + i_N_shift].f[0];
			temp.f[2] = f4_curr[i_pos_z_end + i_N_shift].f[1];
			temp.f[3] = f4_curr[i_pos_z_end + i_N_shift].f[2];
#endif
			v_pos = f4_volt_ptr->linearIndex({0, pos[0], pos[1], 0});
			f4_volt[v_pos].v *= Op->f4_vv_Compressed[0][index].v;
			f4_volt[v_pos].v +=
			    Op->f4_vi_Compressed[0][index].v * (
			        f4_curr[i_pos_z_start + i_N_shift*2             ].v -
			        f4_curr[i_pos_z_start + i_N_shift*2 - i_shift[1]].v -
			        f4_curr[i_pos_z_start + i_N_shift  ].v +
			        temp.v
			    );

			// y-polarization
#ifdef __SSE2__
			temp.v = (__m128)_mm_slli_si128(
			             (__m128i)f4_curr[i_pos_z_end].v, 4
			         );
#else
			temp.f[0] = 0;
			temp.f[1] = f4_curr[i_pos_z_end].f[0];
			temp.f[2] = f4_curr[i_pos_z_end].f[1];
			temp.f[3] = f4_curr[i_pos_z_end].f[2];
#endif
			v_pos += v_N_shift; // go to next component
			f4_volt[v_pos].v *= Op->f4_vv_Compressed[1][index].v;
			f4_volt[v_pos].v +=
			    Op->f4_vi_Compressed[1][index].v * (
			        f4_curr[i_pos_z_start].v -
			        temp.v -
			        f4_curr[i_pos_z_start + i_N_shift*2             ].v +
			        f4_curr[i_pos_z_start + i_N_shift*2 - i_shift[0]].v
			    );

			// z-polarization
			v_pos += v_N_shift; // go to next component
			f4_volt[v_pos].v *= Op->f4_vv_Compressed[2][index].v;
			f4_volt[v_pos].v +=
			    Op->f4_vi_Compressed[2][index].v * (
			        f4_curr[i_pos_z_start + i_N_shift             ].v -
			        f4_curr[i_pos_z_start + i_N_shift - i_shift[0]].v -
			        f4_curr[i_pos_z_start                         ].v +
			        f4_curr[i_pos_z_start             - i_shift[1]].v
			    );

		}
		++pos[0];
	}
}

void Engine_SSE_Compressed::UpdateCurrents_Generic(unsigned int startX, unsigned int numX)
{
	f4vector* f4_volt = f4_volt_ptr->data();
	f4vector* f4_curr = f4_curr_ptr->data();

	unsigned int v_pos;
	unsigned int i_pos;
	int v_shift[3]; // shift of voltages index by x/y/z
	unsigned v_pos_z_start, v_pos_z_end;

	unsigned int pos[3];
	f4vector temp;

	// array index shift for x/y/z field component
	int v_N_shift  = f4_volt_ptr->stride(0);
	int i_N_shift  = f4_curr_ptr->stride(0);

	// index shift for a x-, y- and z-position
	for (unsigned int n=0;n<3;++n)
		v_shift[n]= f4_volt_ptr->stride(n+1);

	pos[0] = startX;
	unsigned int index;
	for (unsigned int posX=0; posX<numX; ++posX)
	{
		for (pos[1]=0; pos[1]<numLines[1]-1; ++pos[1])
		{
			for (pos[2]=0; pos[2]<numVectors-1; ++pos[2])
			{
				index = Op->m_Op_index(pos[0], pos[1], pos[2]);
				i_pos = f4_curr_ptr->linearIndex({0, pos[0], pos[1], pos[2]});
				v_pos = f4_volt_ptr->linearIndex({0, pos[0], pos[1], pos[2]});

				// x-pol
				f4_curr[i_pos].v *= Op->f4_ii_Compressed[0][index].v;
				f4_curr[i_pos].v +=
				    Op->f4_iv_Compressed[0][index].v * (
				        f4_volt[v_pos + 2*v_N_shift             ].v -
				        f4_volt[v_pos + 2*v_N_shift + v_shift[1]].v -
				        f4_volt[v_pos +   v_N_shift             ].v +
				        f4_volt[v_pos +   v_N_shift + v_shift[2]].v
				    );

				// y-pol
				i_pos += i_N_shift;
				f4_curr[i_pos].v *= Op->f4_ii_Compressed[1][index].v;
				f4_curr[i_pos].v +=
				    Op->f4_iv_Compressed[1][index].v * (
				        f4_volt[v_pos                           ].v -
				        f4_volt[v_pos +               v_shift[2]].v -
				        f4_volt[v_pos + 2*v_N_shift             ].v +
				        f4_volt[v_pos + 2*v_N_shift + v_shift[0]].v
				    );

				// z-pol
				i_pos += i_N_shift;
				f4_curr[i_pos].v *= Op->f4_ii_Compressed[2][index].v;
				f4_curr[i_pos].v +=
				    Op->f4_iv_Compressed[2][index].v * (
				        f4_volt[v_pos +   v_N_shift             ].v -
				        f4_volt[v_pos +   v_N_shift + v_shift[0]].v -
				        f4_volt[v_pos                           ].v +
				        f4_volt[v_pos +               v_shift[1]].v
				    );
			}

			v_pos_z_start = f4_volt_ptr->linearIndex({0, pos[0], pos[1], 0});
			v_pos_z_end   = f4_volt_ptr->linearIndex({0, pos[0], pos[1], numVectors-1});

			index = Op->m_Op_index(pos[0], pos[1], numVectors-1);
			// for pos[2] = numVectors-1
			// x-pol
#ifdef __SSE2__
			temp.v = (__m128)_mm_srli_si128(
			             (__m128i)f4_volt[v_pos_z_start + v_N_shift].v, 4
			         );
#else
			temp.f[0] = f4_volt[v_pos_z_start + v_N_shift].f[1];
			temp.f[1] = f4_volt[v_pos_z_start + v_N_shift].f[2];
			temp.f[2] = f4_volt[v_pos_z_start + v_N_shift].f[3];
			temp.f[3] = 0;
#endif
			i_pos = f4_curr_ptr->linearIndex({0, pos[0], pos[1], numVectors-1});
			f4_curr[i_pos].v *=
			    Op->f4_ii_Compressed[0][index].v;
			f4_curr[i_pos].v +=
			    Op->f4_iv_Compressed[0][index].v * (
			        f4_volt[v_pos_z_end + 2*v_N_shift             ].v -
			        f4_volt[v_pos_z_end + 2*v_N_shift + v_shift[1]].v -
			        f4_volt[v_pos_z_end + v_N_shift               ].v +
			        temp.v
			    );

			// y-pol
#ifdef __SSE2__
			temp.v = (__m128)_mm_srli_si128(
			             (__m128i)f4_volt[v_pos_z_start].v, 4
			         );
#else
			temp.f[0] = f4_volt[v_pos_z_start].f[1];
			temp.f[1] = f4_volt[v_pos_z_start].f[2];
			temp.f[2] = f4_volt[v_pos_z_start].f[3];
			temp.f[3] = 0;
#endif
			i_pos += i_N_shift;
			f4_curr[i_pos].v *=
			    Op->f4_ii_Compressed[1][index].v;
			f4_curr[i_pos].v +=
			    Op->f4_iv_Compressed[1][index].v * (
			        f4_volt[v_pos_z_end].v -
			        temp.v -
			        f4_volt[v_pos_z_end + 2*v_N_shift             ].v +
			        f4_volt[v_pos_z_end + 2*v_N_shift + v_shift[0]].v
			    );

			// z-pol
			i_pos += i_N_shift;
			f4_curr[i_pos].v *=
			    Op->f4_ii_Compressed[2][index].v;
			f4_curr[i_pos].v +=
			    Op->f4_iv_Compressed[2][index].v * (
			        f4_volt[v_pos_z_end + v_N_shift             ].v -
			        f4_volt[v_pos_z_end + v_N_shift + v_shift[0]].v -
			        f4_volt[v_pos_z_end                         ].v +
			        f4_volt[v_pos_z_end             + v_shift[1]].v
			    );
		}
		++pos[0];
	}
}

bool Engine_SSE_Compressed::UseIJKN() const
{
	return (f4_volt_ptr->stride(0)==1) && (f4_volt_ptr->stride(3)==3) &&
		   (f4_curr_ptr->stride(0)==1) && (f4_curr_ptr->stride(3)==3) &&
		   (f4_volt_ptr->stride(1)==f4_curr_ptr->stride(1)) && (f4_volt_ptr->stride(2)==f4_curr_ptr->stride(2)) &&
		   (Op->m_Op_index.stride(2)==1) && (Op->f4_vv_Compressed[0].size()>0) &&
		   ((m_PackedV.size()==Op->f4_vv_Compressed[0].size()) || (m_MatV.size()==Op->f4_vv_Compressed[0].size()));
}

void Engine_SSE_Compressed::UpdateVoltages(unsigned int startX, unsigned int numX)
{
	if (UseIJKN())
		UpdateVoltages_IJKN(startX, numX);
	else
		UpdateVoltages_Generic(startX, numX);
}

void Engine_SSE_Compressed::UpdateCurrents(unsigned int startX, unsigned int numX)
{
	if (UseIJKN())
		UpdateCurrents_IJKN(startX, numX);
	else
		UpdateCurrents_Generic(startX, numX);
}

typedef decltype(f4vector::v) f4vector_t; // v4sf, or __m128 with MSVC

// {0, a0, a1, a2}: the z-1 neighbours of z-vector 0 are the lanes below in the last z-vector
static inline f4vector ShiftLanesUp(const f4vector& a)
{
	f4vector r;
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12)
	const v4sf zero = {0,0,0,0};
	r.v = __builtin_shufflevector(zero, a.v, 0, 4, 5, 6);
#else
	r.f[0] = 0;
	r.f[1] = a.f[0];
	r.f[2] = a.f[1];
	r.f[3] = a.f[2];
#endif
	return r;
}

// {a1, a2, a3, 0}: the z+1 neighbours of the last z-vector are the lanes above in z-vector 0
static inline f4vector ShiftLanesDown(const f4vector& a)
{
	f4vector r;
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12)
	const v4sf zero = {0,0,0,0};
	r.v = __builtin_shufflevector(a.v, zero, 1, 2, 3, 4);
#else
	r.f[0] = a.f[1];
	r.f[1] = a.f[2];
	r.f[2] = a.f[3];
	r.f[3] = 0;
#endif
	return r;
}

// The six coefficient vectors of one operator in access order (vv0 vi0 vv1 vi1 vv2 vi2 for the voltages, ii0 iv0 ii1 iv1
// ii2 iv2 for the currents), from the packed table (six per operator) or from the geometry tables (vi or iv per
// operator, vv or ii from the table of distinct triplets). The same values either way.
namespace
{
struct CoefPacked
{
	const f4vector* __restrict pk;
	inline void Get(unsigned int op, const f4vector* (&k)[6]) const
	{
		const f4vector* p = pk + 6*(size_t)op;
		for (int i=0; i<6; ++i)
			k[i] = p + i;
	}
};

struct CoefGeom
{
	const f4vector* __restrict geom;   // per operator: vi0 vi1 vi2 (iv0 iv1 iv2)
	const f4vector* __restrict mat;    // per triplet: vv0 vv1 vv2 (ii0 ii1 ii2)
	const uint16_t* __restrict id;     // per operator: its triplet
	inline void Get(unsigned int op, const f4vector* (&k)[6]) const
	{
		const f4vector* g = geom + 3*(size_t)op;
		const f4vector* m = mat + 3*(size_t)id[op];
		k[0] = m;
		k[1] = g;
		k[2] = m+1;
		k[3] = g+1;
		k[4] = m+2;
		k[5] = g+2;
	}
};
}

template<class C> void Engine_SSE_Compressed::UpdateVoltages_IJKN_T(unsigned int startX, unsigned int numX, const C& coef)
{
	// (x,y,z-vector) holds the three components at [0..2], z-vectors are 3 apart
	f4vector* __restrict volt = f4_volt_ptr->data();
	const f4vector* __restrict curr = f4_curr_ptr->data();
	const unsigned int* __restrict op_idx = Op->m_Op_index.data();
	const ptrdiff_t sx = f4_curr_ptr->stride(1);
	const ptrdiff_t sy = f4_curr_ptr->stride(2);
	const ptrdiff_t isx = Op->m_Op_index.stride(0);
	const ptrdiff_t isy = Op->m_Op_index.stride(1);
	const unsigned int ny = numLines[1];
	const unsigned int nvec = numVectors;

	for (unsigned int x=startX; x<startX+numX; ++x)
	{
		const ptrdiff_t shx = (x>0) ? sx : 0;
		for (unsigned int y=0; y<ny; ++y)
		{
			const ptrdiff_t shy = (y>0) ? sy : 0;
			f4vector* __restrict v = volt + x*sx + y*sy;
			const f4vector* __restrict c = curr + x*sx + y*sy;
			const unsigned int* __restrict ip = op_idx + x*isx + y*isy;
			f4vector_t a, t;
			for (unsigned int z=1; z<nvec; ++z)
			{
				const f4vector* k[6];
				coef.Get(ip[z], k);
				f4vector* __restrict vz = v + 3*z;
				const f4vector* __restrict cz = c + 3*z;

				// x-polarization
				t = cz[2].v - cz[2-shy].v - cz[1].v + cz[1-3].v;
				a = vz[0].v * k[0]->v;
				vz[0].v = a + k[1]->v * t;

				// y-polarization
				t = cz[0].v - cz[0-3].v - cz[2].v + cz[2-shx].v;
				a = vz[1].v * k[2]->v;
				vz[1].v = a + k[3]->v * t;

				// z-polarization
				t = cz[1].v - cz[1-shx].v - cz[0].v + cz[0-shy].v;
				a = vz[2].v * k[4]->v;
				vz[2].v = a + k[5]->v * t;
			}

			// z-vector 0
			const f4vector* k[6];
			coef.Get(ip[0], k);
			const f4vector* __restrict ce = c + 3*(nvec-1);

			t = c[2].v - c[2-shy].v - c[1].v + ShiftLanesUp(ce[1]).v;
			a = v[0].v * k[0]->v;
			v[0].v = a + k[1]->v * t;

			t = c[0].v - ShiftLanesUp(ce[0]).v - c[2].v + c[2-shx].v;
			a = v[1].v * k[2]->v;
			v[1].v = a + k[3]->v * t;

			t = c[1].v - c[1-shx].v - c[0].v + c[0-shy].v;
			a = v[2].v * k[4]->v;
			v[2].v = a + k[5]->v * t;
		}
	}
}

template<class C> void Engine_SSE_Compressed::UpdateCurrents_IJKN_T(unsigned int startX, unsigned int numX, const C& coef)
{
	const f4vector* __restrict volt = f4_volt_ptr->data();
	f4vector* __restrict curr = f4_curr_ptr->data();
	const unsigned int* __restrict op_idx = Op->m_Op_index.data();
	const ptrdiff_t sx = f4_volt_ptr->stride(1);
	const ptrdiff_t sy = f4_volt_ptr->stride(2);
	const ptrdiff_t isx = Op->m_Op_index.stride(0);
	const ptrdiff_t isy = Op->m_Op_index.stride(1);
	const unsigned int ny = numLines[1];
	const unsigned int nvec = numVectors;

	for (unsigned int x=startX; x<startX+numX; ++x)
	{
		for (unsigned int y=0; y+1<ny; ++y)
		{
			const f4vector* __restrict v = volt + x*sx + y*sy;
			f4vector* __restrict c = curr + x*sx + y*sy;
			const unsigned int* __restrict ip = op_idx + x*isx + y*isy;
			f4vector_t a, t;
			for (unsigned int z=0; z+1<nvec; ++z)
			{
				const f4vector* k[6];
				coef.Get(ip[z], k);
				const f4vector* __restrict vz = v + 3*z;
				f4vector* __restrict cz = c + 3*z;

				// x-pol
				t = vz[2].v - vz[2+sy].v - vz[1].v + vz[1+3].v;
				a = cz[0].v * k[0]->v;
				cz[0].v = a + k[1]->v * t;

				// y-pol
				t = vz[0].v - vz[0+3].v - vz[2].v + vz[2+sx].v;
				a = cz[1].v * k[2]->v;
				cz[1].v = a + k[3]->v * t;

				// z-pol
				t = vz[1].v - vz[1+sx].v - vz[0].v + vz[0+sy].v;
				a = cz[2].v * k[4]->v;
				cz[2].v = a + k[5]->v * t;
			}

			// last z-vector
			const f4vector* k[6];
			coef.Get(ip[nvec-1], k);
			const f4vector* __restrict ve = v + 3*(nvec-1);
			f4vector* __restrict ce = c + 3*(nvec-1);

			t = ve[2].v - ve[2+sy].v - ve[1].v + ShiftLanesDown(v[1]).v;
			a = ce[0].v * k[0]->v;
			ce[0].v = a + k[1]->v * t;

			t = ve[0].v - ShiftLanesDown(v[0]).v - ve[2].v + ve[2+sx].v;
			a = ce[1].v * k[2]->v;
			ce[1].v = a + k[3]->v * t;

			t = ve[1].v - ve[1+sx].v - ve[0].v + ve[0+sy].v;
			a = ce[2].v * k[4]->v;
			ce[2].v = a + k[5]->v * t;
		}
	}
}

void Engine_SSE_Compressed::UpdateVoltages_IJKN(unsigned int startX, unsigned int numX)
{
	if (!m_MatV.empty())
		UpdateVoltages_IJKN_T(startX, numX, CoefGeom{m_GeomV.data(), m_MatTabV.data(), m_MatV.data()});
	else
		UpdateVoltages_IJKN_T(startX, numX, CoefPacked{m_PackedV.data()->c});
}

void Engine_SSE_Compressed::UpdateCurrents_IJKN(unsigned int startX, unsigned int numX)
{
	if (!m_MatI.empty())
		UpdateCurrents_IJKN_T(startX, numX, CoefGeom{m_GeomI.data(), m_MatTabI.data(), m_MatI.data()});
	else
		UpdateCurrents_IJKN_T(startX, numX, CoefPacked{m_PackedI.data()->c});
}
