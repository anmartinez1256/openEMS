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

// Diamond-blocked Metal engine, see engine_metal_diamond.h. The tile schedule follows luikore's Metal engine (openEMS
// pull request #228, macos/doc/metal-engine.rst, GPL); the kernels are this fork's own (emsKit swift_engine/fdtd.swift
// moved to the engine's f4vector arrays in N-I-J-K order, plus the current excitation, Mur ABC, lumped RLC and UPML).

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "FDTD/engine_metal_diamond.h"
#include "FDTD/extensions/engine_ext_excitation.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/engine_ext_mur_abc.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/engine_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/engine_ext_upml.h"
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/engine_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/engine_ext_debyematerial.h"
#include "FDTD/extensions/operator_ext_debyematerial.h"
#include "FDTD/excitation.h"
#include "tools/useful.h"
#include "tools/global.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <vector>

using std::cout;
using std::cerr;
using std::endl;

namespace
{

const char* DIAMOND_SOURCE = R"MSL(
#include <metal_stdlib>
using namespace metal;
// the CPU engines' arithmetic: no contraction beyond the explicit fma() (clang fuses "a += b*c" on arm64)
#pragma METAL fp contract(off)

struct Dims { uint nx, ny, nz, nv; };
// the grid as compile-time constants (the kernel is built per model): the divisions by nv and nz become multiplications
constant Dims dims = {DIAMOND_NX, DIAMOND_NY, DIAMOND_NZ, DIAMOND_NV};
// One dispatch advances the tiles of one phase, or of two phases whose tiles do not depend on each other (mountain x
// valley and valley x mountain: tile set tg.z). types: x type | y type << 1 of set 0, the same << 2 of set 1 (0 = mountain,
// 1 = valley); per set the base of its tiles' entry lists and its tiles in x and y.
struct Phase { uint depth, ts0, types, lbase0, ntx0, nty0, lbase1, ntx1, nty1; };
struct Ext { uint vcount, ccount, length, period0, numMur, numApply, rlcSerial, numUpml, numAde; }; // period0: 0 = no signal period
struct MurFace { uint ny, nyP, nyPP, line, shift, n1, startTS, coef, size; };
struct Upml { uint x0, y0, z0, nx, ny, nz, cidx, flux, rcpZ, okZ; };   // box start and lines, coefficient index (U), flux (St); ceil(2^32 / nz) and whether i / nz may use it
// where the extension data sit in the read-only words (U), the read-only floats (Fc) and the state (St)
struct Ofs { uint xrun, xro, xele, mf, apply, rlcI, roff, rlist, upml, arun, aro, aele;  // U
             uint vsig, csig, mcoef, rlcC, ucoef, adeC;                                // Fc
             uint mscr, rlcS, adeS; };                                                 // St

// the f4vector of component n at (x, y, zv) in the N-I-J-K layout, and the float of cell z (z = lane*nv + zv)
inline uint I(uint n, uint x, uint y, uint zv, constant Dims& d) { return ((n*d.nx + x)*d.ny + y)*d.nv + zv; }
inline uint F(uint n, uint x, uint y, uint z, constant Dims& d) { return I(n, x, y, z % d.nv, d)*4 + z / d.nv; }

// Engine_SSE_Compressed::UpdateVoltages_IJKN / UpdateCurrents_IJKN for one f4vector, same operands in the same
// order; packed coefficients per operator: vv0 vi0 vv1 vi1 vv2 vi2 (ii0 iv0 ii1 iv1 ii2 iv2 for the currents); split
// (DIAMOND_SPLIT): per operator the indices of vi0 vi1 vi2 and vv0 | vv1 << 10 | vv2 << 20 in the table pc
inline void cellV(device float4* volt, device const float4* curr, device const float4* pv, device const float4* pc,
                  device const uint* opi, constant Dims& d, uint x, uint y, uint zv)
{
#if DIAMOND_SPLIT
    const uint4 q = ((device const uint4*)pv)[opi[(x*d.ny + y)*d.nv + zv]];
#else
    device const float4* c = pv + 6*opi[(x*d.ny + y)*d.nv + zv];
#endif
    const uint xm = x > 0 ? x-1 : x, ym = y > 0 ? y-1 : y;
    const float4 c0 = curr[I(0,x,y,zv,d)], c1 = curr[I(1,x,y,zv,d)], c2 = curr[I(2,x,y,zv,d)];
    float4 c1zm, c0zm;
#if DIAMOND_FASTIDX
    {   // the z neighbour below: the vector before, or for the first one the last vector moved up a lane (one load each way)
        const uint zq = zv > 0 ? zv-1 : d.nv-1;
        const float4 a = curr[I(1,x,y,zq,d)], b = curr[I(0,x,y,zq,d)];
        c1zm = zv > 0 ? a : float4(0.0f, a.x, a.y, a.z);
        c0zm = zv > 0 ? b : float4(0.0f, b.x, b.y, b.z);
    }
#else
    if (zv > 0) { c1zm = curr[I(1,x,y,zv-1,d)]; c0zm = curr[I(0,x,y,zv-1,d)]; }
    else {
        const float4 a = curr[I(1,x,y,d.nv-1,d)]; c1zm = float4(0.0f, a.x, a.y, a.z);
        const float4 b = curr[I(0,x,y,d.nv-1,d)]; c0zm = float4(0.0f, b.x, b.y, b.z);
    }
#endif
    const float4 c2ym = curr[I(2,x,ym,zv,d)], c2xm = curr[I(2,xm,y,zv,d)];
    const float4 c1xm = curr[I(1,xm,y,zv,d)], c0ym = curr[I(0,x,ym,zv,d)];
    float4 v;
#if DIAMOND_SPLIT
    v = volt[I(0,x,y,zv,d)]; v = v * pc[q.w & 1023u]; v = fma(pc[q.x], ((c2 - c2ym) - c1) + c1zm, v); volt[I(0,x,y,zv,d)] = v;
    v = volt[I(1,x,y,zv,d)]; v = v * pc[(q.w >> 10) & 1023u]; v = fma(pc[q.y], ((c0 - c0zm) - c2) + c2xm, v); volt[I(1,x,y,zv,d)] = v;
    v = volt[I(2,x,y,zv,d)]; v = v * pc[q.w >> 20]; v = fma(pc[q.z], ((c1 - c1xm) - c0) + c0ym, v); volt[I(2,x,y,zv,d)] = v;
#else
    v = volt[I(0,x,y,zv,d)]; v = v * c[0]; v = fma(c[1], ((c2 - c2ym) - c1) + c1zm, v); volt[I(0,x,y,zv,d)] = v;
    v = volt[I(1,x,y,zv,d)]; v = v * c[2]; v = fma(c[3], ((c0 - c0zm) - c2) + c2xm, v); volt[I(1,x,y,zv,d)] = v;
    v = volt[I(2,x,y,zv,d)]; v = v * c[4]; v = fma(c[5], ((c1 - c1xm) - c0) + c0ym, v); volt[I(2,x,y,zv,d)] = v;
#endif
}

inline void cellI(device float4* curr, device const float4* volt, device const float4* pi, device const float4* pc,
                  device const uint* opi, constant Dims& d, uint x, uint y, uint zv)
{
#if DIAMOND_SPLIT
    const uint4 q = ((device const uint4*)pi)[opi[(x*d.ny + y)*d.nv + zv]];
#else
    device const float4* c = pi + 6*opi[(x*d.ny + y)*d.nv + zv];
#endif
    const float4 v0 = volt[I(0,x,y,zv,d)], v1 = volt[I(1,x,y,zv,d)], v2 = volt[I(2,x,y,zv,d)];
    float4 v1zp, v0zp;
#if DIAMOND_FASTIDX
    {
        const bool in = zv+1 < d.nv;
        const uint zq = in ? zv+1 : 0u;
        const float4 a = volt[I(1,x,y,zq,d)], b = volt[I(0,x,y,zq,d)];
        v1zp = in ? a : float4(a.y, a.z, a.w, 0.0f);
        v0zp = in ? b : float4(b.y, b.z, b.w, 0.0f);
    }
#else
    if (zv+1 < d.nv) { v1zp = volt[I(1,x,y,zv+1,d)]; v0zp = volt[I(0,x,y,zv+1,d)]; }
    else {
        const float4 a = volt[I(1,x,y,0,d)]; v1zp = float4(a.y, a.z, a.w, 0.0f);
        const float4 b = volt[I(0,x,y,0,d)]; v0zp = float4(b.y, b.z, b.w, 0.0f);
    }
#endif
    const float4 v2yp = volt[I(2,x,y+1,zv,d)], v0yp = volt[I(0,x,y+1,zv,d)];
    const float4 v2xp = volt[I(2,x+1,y,zv,d)], v1xp = volt[I(1,x+1,y,zv,d)];
    float4 i;
#if DIAMOND_SPLIT
    i = curr[I(0,x,y,zv,d)]; i = i * pc[q.w & 1023u]; i = fma(pc[q.x], ((v2 - v2yp) - v1) + v1zp, i); curr[I(0,x,y,zv,d)] = i;
    i = curr[I(1,x,y,zv,d)]; i = i * pc[(q.w >> 10) & 1023u]; i = fma(pc[q.y], ((v0 - v0zp) - v2) + v2xp, i); curr[I(1,x,y,zv,d)] = i;
    i = curr[I(2,x,y,zv,d)]; i = i * pc[q.w >> 20]; i = fma(pc[q.z], ((v1 - v1xp) - v0) + v0yp, i); curr[I(2,x,y,zv,d)] = i;
#else
    i = curr[I(0,x,y,zv,d)]; i = i * c[0]; i = fma(c[1], ((v2 - v2yp) - v1) + v1zp, i); curr[I(0,x,y,zv,d)] = i;
    i = curr[I(1,x,y,zv,d)]; i = i * c[2]; i = fma(c[3], ((v0 - v0zp) - v2) + v2xp, i); curr[I(1,x,y,zv,d)] = i;
    i = curr[I(2,x,y,zv,d)]; i = i * c[4]; i = fma(c[5], ((v1 - v1xp) - v0) + v0yp, i); curr[I(2,x,y,zv,d)] = i;
#endif
}

// r / w in the loops over a tile's cells, w = a width that is only known at run time (a tile's width in y changes with
// the half-step), so the compiler cannot turn the division into a multiplication: with rcp = ceil(2^32 / w), the high
// word of r * rcp is floor(r / w) while r * (w - 1) < 2^32. The reciprocals come from the host: RCPW[w] for the tile
// widths 2 .. 64 (okW: such a width, and r < 2^26), a field of the record for a UPML box's lines. They are not computed
// here: on the M5 (macOS 27.0.1) the GPU's 0xffffffffu / w returned 3579139454 for w = 3 (1431655765 is right).
constant uint RCPW[65] = {   // ceil(2^32 / w), w = 2 .. 64
    0u, 0u, 2147483648u, 1431655766u, 1073741824u, 858993460u, 715827883u, 613566757u,
    536870912u, 477218589u, 429496730u, 390451573u, 357913942u, 330382100u, 306783379u, 286331154u,
    268435456u, 252645136u, 238609295u, 226050911u, 214748365u, 204522253u, 195225787u, 186737709u,
    178956971u, 171798692u, 165191050u, 159072863u, 153391690u, 148102321u, 143165577u, 138547333u,
    134217728u, 130150525u, 126322568u, 122713352u, 119304648u, 116080198u, 113025456u, 110127367u,
    107374183u, 104755300u, 102261127u, 99882961u, 97612894u, 95443718u, 93368855u, 91382283u,
    89478486u, 87652394u, 85899346u, 84215046u, 82595525u, 81037119u, 79536432u, 78090315u,
    76695845u, 75350304u, 74051161u, 72796056u, 71582789u, 70409300u, 69273667u, 68174085u,
    67108864u };
inline uint rcpW(uint w) { return RCPW[min(w, 64u)]; }
inline bool okW(uint count, uint w) { return w >= 2u && w <= 64u && count < (1u << 26); }
inline uint divW(uint r, uint w, uint rcp, bool ok)
{
#if DIAMOND_FASTIDX
    if (ok) return mulhi(r, rcp);
#endif
    return r / w;
}

// The range of a tile at half-step s: a mountain [lo, hi) is [lo + s/2, hi - (s+1)/2), a valley between two
// mountains (lo = the left mountain's hi, hi = the right mountain's lo) is [lo - (s+1)/2, hi + s/2); clipped to [0, n).
inline int2 span(int2 r, uint type, uint s, int n)
{
    const int fl = int(s / 2), ce = int((s + 1) / 2);
    int a = type == 0 ? r.x + fl : r.x - ce, b = type == 0 ? r.y - ce : r.y + fl;
    return int2(max(a, 0), min(b, n));
}

// Engine_Ext_Excitation: index into the signal for timestep ts (0 <= pos <= ts: without a period, the CPU's pos % (ts+1)
// is pos)
inline int excPos(uint ts, uint delay, constant Ext& e)
{
    int pos = int(ts) - int(delay);
    pos *= (pos > 0);
    if (e.period0) pos %= int(e.period0);
    pos *= (pos < int(e.length));
    return pos;
}

// The Mur face cells inside the tile: the face line lies in the tile's x (y) range for an x (y) face, the shift line
// too (the domain edges sit inside the edge valleys), a z face is complete in every tile
inline uint murCount(thread const MurFace& m, int2 sx, int2 sy, constant Dims& d)
{
    const uint wx = uint(sx.y - sx.x), wy = uint(sy.y - sy.x);
    return m.ny == 0 ? wy*d.nz : (m.ny == 1 ? wx*d.nz : wx*wy);
}
inline void murPos(thread const MurFace& m, uint i, int2 sx, int2 sy, constant Dims& d, thread uint* pos, uint rcp, bool ok)
{
    const uint wy = uint(sy.y - sy.x);
    if (m.ny == 0)      { pos[0] = m.line; pos[1] = uint(sy.x) + i / d.nz; pos[2] = i % d.nz; }
    else if (m.ny == 1) { pos[0] = uint(sx.x) + i / d.nz; pos[1] = m.line; pos[2] = i % d.nz; }
    else                { const uint q = divW(i, wy, rcp, ok); pos[0] = uint(sx.x) + q; pos[1] = uint(sy.x) + (i - q*wy); pos[2] = m.line; }
}

// stage 0: Pre (before the E update), 1: Post (after it), 2: Apply, 3: Post and Apply; Engine_Ext_Mur_ABC's MurFast arithmetic
// (vP = E_shift - c*E_face is one fmsub on arm64, vP += c*E_shift one fmadd)
inline void murStage(uint stage, const MurFace m, device float* E, device const float* mc, device float* ms,
                     int2 sx, int2 sy, constant Dims& d, uint tid, uint tgs)
{
    const uint count = murCount(m, sx, sy, d);
    const uint rcp = rcpW(uint(sy.y - sy.x));
    const bool ok = okW(count, uint(sy.y - sy.x));
    for (uint i = tid; i < count; i += tgs)
    {
        uint pos[3];
        murPos(m, i, sx, sy, d, pos, rcp, ok);
        const uint k = pos[m.nyP]*m.n1 + pos[m.nyPP];
        const uint fP = F(m.nyP, pos[0], pos[1], pos[2], d), fPP = F(m.nyPP, pos[0], pos[1], pos[2], d);
        if (stage == 2) { E[fP] = ms[m.coef + k]; E[fPP] = ms[m.coef + m.size + k]; continue; }
        const float cP = mc[m.coef + k], cPP = mc[m.coef + m.size + k];
        pos[m.ny] = m.shift;
        const uint sP = F(m.nyP, pos[0], pos[1], pos[2], d), sPP = F(m.nyPP, pos[0], pos[1], pos[2], d);
        if (stage == 0)
        {
            ms[m.coef + k]          = fma(-cP,  E[fP],  E[sP]);
            ms[m.coef + m.size + k] = fma(-cPP, E[fPP], E[sPP]);
        }
        else if (stage == 1)
        {
            ms[m.coef + k]          = fma(cP,  E[sP],  ms[m.coef + k]);
            ms[m.coef + m.size + k] = fma(cPP, E[sPP], ms[m.coef + m.size + k]);
        }
        else
        {
            // stage 3: Post and Apply of this element in one go (the same values: the element's own scratch and face
            // floats, the shift line's field, which no face of the same axis writes)
            const float vP = fma(cP, E[sP], ms[m.coef + k]), vPP = fma(cPP, E[sPP], ms[m.coef + m.size + k]);
            ms[m.coef + k] = vP;
            ms[m.coef + m.size + k] = vPP;
            E[fP] = vP;
            E[fPP] = vPP;
        }
    }
}

// Engine_Ext_LumpedRLC for the elements of extension ext in this tile (the tile's list: the elements its ranges
// reach). Stage 0: the inductor current (DoPreVoltageUpdates, independent of the fields), the node voltage from the
// updated field and J, the state moves on one timestep; stage 1: the voltage back into the field (all reads before
// all writes, as on the CPU; one thread in element order when two elements share a cell). Clang's contraction of the
// CPU expressions: Il += (i2v*ilv)*Vd1 one fmadd; vvd*((Vd0 - Il) + vv2*Vd2 + vj1*J1 + vj2*J2) three fmadds then a
// multiply; ib0*(Vd0 - Vd2) - (b1*ib0)*J1 - (b2*ib0)*J2 the first product fused with -(b1*ib0)*J1, then one fmsub.
// Element: x, y, z, direction, extension; coefficients: i2v*ilv, vvd, vv2, vj1, vj2, ib0, b1*ib0, b2*ib0;
// state: Il, Vd[n-1], Vd[n-2], J[n-1], J[n-2].
inline void rlcStage(uint stage, uint ext, device float* E, device const uint* ri, device const float* rc,
                     device float* rs, device const uint* off, device const uint* list, uint li, int2 sx, int2 sy,
                     constant Dims& d, uint tid, uint tgs, bool serial)
{
    if (stage == 1 && serial)
    {
        if (tid != 0) return;
        tgs = 1;
    }
    for (uint l = off[li] + tid; l < off[li+1]; l += tgs)
    {
        const uint k = list[l];
        device const uint* q = ri + 5*k;
        if (q[4] != ext) continue;
        const int x = int(q[0]), y = int(q[1]);
        if (x < sx.x || x >= sx.y || y < sy.x || y >= sy.y) continue;
        const uint fi = F(q[3], uint(x), uint(y), q[2], d);
        device float* st = rs + 5*k;
        if (stage == 1) { E[fi] = st[1]; continue; }
        device const float* c = rc + 8*k;
        const float il = fma(c[0], st[1], st[0]);
        const float vd = c[1] * fma(c[4], st[4], fma(c[3], st[3], fma(c[2], st[2], E[fi] - il)));
        const float j = fma(-c[7], st[4], fma(c[5], vd - st[2], -(c[6] * st[3])));
        st[0] = il; st[2] = st[1]; st[1] = vd; st[4] = st[3]; st[3] = j;
    }
}

// Engine_Ext_UPML for the cells of box u inside the tile (the three components, z complete in the box); stage 0
// before the update, 1 after it; the voltages in the E step, the currents in the H step. Clang's contraction of the
// CPU expressions: vv*E - vvfo*flux is fma(vv, E, -(vvfo*flux)), flux_old + vvfn*E one fmadd. Coefficient sets:
// vv vvfo vvfn ii iifo iifn, one index per component and box cell; the flux: voltages, then currents.
inline void upmlStage(uint stage, bool eStep, const Upml u, device float* Fd, device const uint* U,
                      device const float* tab, device float* St, int2 sx, int2 sy, constant Dims& d, uint tid, uint tgs)
{
    const int x0 = max(sx.x, int(u.x0)), x1 = min(sx.y, int(u.x0 + u.nx));
    const int y0 = max(sy.x, int(u.y0)), y1 = min(sy.y, int(u.y0 + u.ny));
    if (x1 <= x0 || y1 <= y0) return;
    const uint wy = uint(y1 - y0), N = u.nx*u.ny*u.nz;
    const uint count = uint(x1 - x0) * wy * 3 * u.nz;
    device float* flux = St + u.flux + (eStep ? 0 : 3*N);
    const uint co = eStep ? 0 : 3;
    const uint rcpY = rcpW(wy);
    const bool okZ = u.okZ != 0u, okY = okW(count, wy);
    for (uint i = tid; i < count; i += tgs)
    {
        uint r = divW(i, u.nz, u.rcpZ, okZ);
        const uint lz = i - r*u.nz;
        const uint n = r % 3;
        r /= 3;
        const uint qx = divW(r, wy, rcpY, okY);
        const uint y = uint(y0) + (r - qx*wy), x = uint(x0) + qx;
        const uint c = ((n*u.nx + (x - u.x0))*u.ny + (y - u.y0))*u.nz + lz;
        device const float* t = tab + 6*U[u.cidx + c] + co;
        const uint fi = F(n, x, y, u.z0 + lz, d);
        if (stage == 0) { const float h = fma(t[0], Fd[fi], -(t[1]*flux[c])); Fd[fi] = flux[c]; flux[c] = h; }
        else            { const float h = flux[c]; flux[c] = Fd[fi]; Fd[fi] = fma(t[2], flux[c], h); }
    }
}

// The dispersive materials for the cells in this tile: Engine_Ext_LorentzMaterial (Drude/Lorentz, conducting sheets) and
// Engine_Ext_DebyeMaterial, as runs: the elements of one extension at one cell for this step type, in the CPU's order (the
// orders one after the other, the list order within an order). Run: float index of component 0 (component n at + n*cs),
// x << 16 | y, first and end element; element: coefficient offset, state offset (the H step's parts included), flags
// (1 voltage ADE, 2 current ADE, 4 voltage Lorentz term, 8 current Lorentz term, 16 Debye, poles << 8).
// Lorentz coefficients v_int, v_ext, v_Lor, i_int, i_ext, i_Lor [3] each; state: volt ADE, volt Lorentz, curr ADE,
// curr Lorentz [3]. Pre (DoPre*UpdatesLines, before the update; reads the field only): Lor += v_Lor*ADE one fmadd,
// ADE *= v_int, then ADE += v_ext*(f - Lor) one fmadd (two statements: not contracted across them).
// Debye coefficients (v_relax, v_drive)[3] per pole, then v_solve[3]; state: the correction (volt ADE)[3], pre[3], W[pole][3].
// Pre: W = v_relax*W_old + v_drive*V^n (clang: fma(v_relax, W_old, v_drive*V)), pre += W - W_old; Post (after the update):
// V = (V_raw - pre)*v_solve, correction = V_raw - V, W += v_drive*V one fmadd.
inline bool inSpan(uint xy, int2 sx, int2 sy)
{
    const int x = int(xy >> 16), y = int(xy & 0xffffu);
    return x >= sx.x && x < sx.y && y >= sy.x && y < sy.y;
}

inline void adePre(bool eStep, device const float* Fd, device const uint4* ar, device const uint4* ae, device const float* ac,
                   device float* as, uint r0, uint r1, int2 sx, int2 sy, uint cs, uint tid, uint tgs)
{
    const uint cnt = 3*(r1 - r0);
    for (uint i = tid; i < cnt; i += tgs)
    {
        const uint4 R = ar[r0 + i/3];
        if (!inSpan(R.y, sx, sy)) continue;
        const uint n = i % 3;
        const float f = Fd[R.x + n*cs];
        for (uint l = R.z; l < R.w; ++l)
        {
            const uint4 q = ae[l];
            device const float* c = ac + q.x;
            device float* st = as + q.y;
            if (q.z & 16u)
            {
                float pre = 0.0f;
                for (uint pl = 0; pl < (q.z >> 8); ++pl)
                {
                    const float w0 = st[6 + 3*pl + n];
                    const float w = fma(c[6*pl + n], w0, c[6*pl + 3 + n] * f);
                    pre = pre + (w - w0);
                    st[6 + 3*pl + n] = w;
                }
                st[3+n] = pre;
            }
            else if (q.z & (eStep ? 4u : 8u))
            {
                st[3+n] = fma(c[6+n], st[n], st[3+n]);
                st[n] = st[n] * c[n];
                st[n] = fma(c[3+n], f - st[3+n], st[n]);
            }
            else
            {
                st[n] = st[n] * c[n];
                st[n] = fma(c[3+n], f, st[n]);
            }
        }
    }
}

inline void adePost(device const float* Fd, device const uint4* ar, device const uint4* ae, device const float* ac,
                    device float* as, uint r0, uint r1, int2 sx, int2 sy, uint cs, uint tid, uint tgs)
{
    const uint cnt = 3*(r1 - r0);
    for (uint i = tid; i < cnt; i += tgs)
    {
        const uint4 R = ar[r0 + i/3];
        if (!inSpan(R.y, sx, sy)) continue;
        const uint n = i % 3;
        const float vr = Fd[R.x + n*cs];
        for (uint l = R.z; l < R.w; ++l)
        {
            const uint4 q = ae[l];
            if ((q.z & 16u) == 0) continue;
            device const float* c = ac + q.x;
            device float* st = as + q.y;
            const uint P = q.z >> 8;
            const float v = (vr - st[3+n]) * c[6*P + n];
            st[n] = vr - v;
            for (uint pl = 0; pl < P; ++pl)
                st[6 + 3*pl + n] = fma(c[6*pl + 3 + n], v, st[6 + 3*pl + n]);
        }
    }
}

// Engine_Ext_Dispersive::Apply2*Lines for the runs of one extension: field -= ADE, a run's elements in their order
inline void adeApply(device float* Fd, device const uint4* ar, device const uint4* ae, device const float* as, uint r0, uint r1,
                     int2 sx, int2 sy, uint cs, uint tid, uint tgs)
{
    const uint cnt = 3*(r1 - r0);
    for (uint i = tid; i < cnt; i += tgs)
    {
        const uint4 R = ar[r0 + i/3];
        if (!inSpan(R.y, sx, sy)) continue;
        const uint n = i % 3, fi = R.x + n*cs;
        float v = Fd[fi];
        for (uint l = R.z; l < R.w; ++l)
            v = v - as[ae[l].y + n];
        Fd[fi] = v;
    }
}

// One threadgroup advances one tile over p.depth timesteps in place, z complete, with a threadgroup barrier between
// the stages; the extensions run per tile in the CPU engine's order (Pre in reverse extension order, update, Post,
// Apply in extension order). Metal has 31 buffer slots, so the extension data come in three blobs (U, Fc, St).
kernel void diamond(device float4* volt [[buffer(0)]], device float4* curr [[buffer(1)]],
                    device const float4* pv [[buffer(2)]], device const float4* pi [[buffer(3)]],
                    device const uint* opi [[buffer(4)]], device const float4* pc [[buffer(5)]],
                    device const int2* xr0 [[buffer(6)]], device const int2* yr0 [[buffer(7)]],
                    device const int2* xr1 [[buffer(14)]], device const int2* yr1 [[buffer(15)]],
                    constant Phase& p [[buffer(8)]], constant Ext& e [[buffer(9)]],
                    device const uint* U [[buffer(10)]], device const float* Fc [[buffer(11)]],
                    device float* St [[buffer(12)]], constant Ofs& o [[buffer(13)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint3 tptg [[threads_per_threadgroup]])
{
    device const uint4* xrun = (device const uint4*)(U + o.xrun);
    device const uint* xro = U + o.xro;
    device const uint2* xele = (device const uint2*)(U + o.xele);
    device const MurFace* mf = (device const MurFace*)(U + o.mf);
    device const uint* apply = U + o.apply;
    device const uint* rlcI = U + o.rlcI;
    device const uint* roff = U + o.roff;
    device const uint* rlist = U + o.rlist;
    device const Upml* up = (device const Upml*)(U + o.upml);
    device const float* vsig = Fc + o.vsig;
    device const float* csig = Fc + o.csig;
    device const float* mcoef = Fc + o.mcoef;
    device const float* rlcC = Fc + o.rlcC;
    device const float* utab = Fc + o.ucoef;
    device const uint4* arun = (device const uint4*)(U + o.arun);
    device const uint* aro = U + o.aro;
    device const uint4* aele = (device const uint4*)(U + o.aele);
    device const float* adeC = Fc + o.adeC;
    device float* mscr = St + o.mscr;
    device float* rlcS = St + o.rlcS;
    device float* adeS = St + o.adeS;
    constant Dims& d = dims;
    const uint tgs = tptg.x;
    // this threadgroup's tile: x and y mountains in xr0, yr0, valleys in xr1, yr1; a two-set dispatch has the grid of the
    // larger set in each direction
    const uint set = tg.z;
    const uint xtype = (p.types >> (2*set)) & 1u, ytype = (p.types >> (2*set + 1)) & 1u;
    const uint lbase = set ? p.lbase1 : p.lbase0, ntx = set ? p.ntx1 : p.ntx0, nty = set ? p.nty1 : p.nty0;
    if (tg.x >= ntx || tg.y >= nty)
        return;   // the same for the whole threadgroup
    const int2 rx = (xtype ? xr1 : xr0)[tg.x], ry = (ytype ? yr1 : yr0)[tg.y];
    device float* E = (device float*)volt;
    device float* H = (device float*)curr;
    const uint cs = 4*d.nx*d.ny*d.nv;   // the floats of one field component
    // this tile's entry lists and runs, the same for every half-step: read once (a device load per half-step costs a few
    // per cent on small grids, where a half-step is about one cell update per thread)
    const uint li = lbase + tg.x*nty + tg.y;
#if DIAMOND_RLC
    const bool rlcTile = roff[li+1] > roff[li];
#endif
#if DIAMOND_ADE
    const uint3 ar = uint3(aro[2*li*e.numAde], aro[(2*li+1)*e.numAde], aro[(2*li+2)*e.numAde]);   // [x, y) E, [y, z) H
#endif
#if DIAMOND_EXC
    const uint3 xq = uint3(xro[2*li], xro[2*li+1], xro[2*li+2]);   // [x, y) voltages, [y, z) currents
#endif
    for (uint s = 0; s < 2 * p.depth; ++s)
    {
        const int2 sx = span(rx, xtype, s, int(d.nx)), sy = span(ry, ytype, s, int(d.ny));
        const int wx = sx.y - sx.x, wy = sy.y - sy.x;
        if (wx <= 0 || wy <= 0)
            continue;   // the same for the whole threadgroup
        const uint ts = p.ts0 + s / 2;
        const bool eStep = (s & 1) == 0;
        bool rlc = false;   // lumped elements in this tile
#if DIAMOND_RLC
        rlc = eStep && rlcTile;
#endif
        bool ade = false;   // dispersive runs in this tile for this step type: [a0, a1); ab: their ranges per extension
        uint ab = 0, a0 = 0, a1 = 0;
#if DIAMOND_ADE
        ab = (li*2 + (eStep ? 0u : 1u)) * e.numAde;
        a0 = eStep ? ar.x : ar.y;
        a1 = eStep ? ar.y : ar.z;
        ade = a1 > a0;
#endif
        uint mur = 0;   // the active Mur faces with cells in this tile
        uint murAxes = 0;   // their axes (bit ny)
#if DIAMOND_MUR
        if (eStep)
            for (uint f = 0; f < e.numMur; ++f)
            {
                const int L = int(mf[f].line);
                if (ts < mf[f].startTS) continue;
                if (mf[f].ny == 0 && (L < sx.x || L >= sx.y)) continue;
                if (mf[f].ny == 1 && (L < sy.x || L >= sy.y)) continue;
                mur |= 1u << f;
                murAxes |= 1u << mf[f].ny;
            }
#endif
        if (mur || ade)
        {
            // the Pre stages that only read the field (Mur, Drude/Lorentz) run before the UPML's, which writes it
#if DIAMOND_MUR
            for (uint f = 0; f < e.numMur; ++f)
                if (mur & (1u << f)) murStage(0, mf[f], E, mcoef, mscr, sx, sy, d, tid, tgs);
#endif
#if DIAMOND_ADE
            if (ade) adePre(eStep, eStep ? E : H, arun, aele, adeC, adeS, a0, a1, sx, sy, cs, tid, tgs);
#endif
            threadgroup_barrier(mem_flags::mem_device);
        }
        uint upml = 0;   // the UPML boxes with cells in this tile (the boxes do not overlap: one barrier per stage)
#if DIAMOND_UPML
        for (uint u = 0; u < e.numUpml; ++u)
            if (int(up[u].x0) < sx.y && int(up[u].x0 + up[u].nx) > sx.x && int(up[u].y0) < sy.y && int(up[u].y0 + up[u].ny) > sy.x)
                upml |= 1u << u;
        if (upml)
        {
            for (uint u = e.numUpml; u-- > 0; )
                if (upml & (1u << u)) upmlStage(0, eStep, up[u], eStep ? E : H, U, utab, St, sx, sy, d, tid, tgs);
            threadgroup_barrier(mem_flags::mem_device);
        }
#endif
        const uint count = uint(wx) * uint(wy) * d.nv;
        const uint uwy = uint(wy), rcp = rcpW(uwy);
        const bool okY = okW(uint(wx) * uwy, uwy);
        for (uint i = tid; i < count; i += tgs)
        {
            const uint zv = i % d.nv, r = i / d.nv;
            const uint q = divW(r, uwy, rcp, okY);
            const uint y = uint(sy.x) + (r - q*uwy), x = uint(sx.x) + q;
            if (eStep) cellV(volt, curr, pv, pc, opi, d, x, y, zv);
            else if (x + 1 < d.nx && y + 1 < d.ny) cellI(curr, volt, pi, pc, opi, d, x, y, zv);
        }
        threadgroup_barrier(mem_flags::mem_device);
#if DIAMOND_UPML
        if (upml)
        {
            for (uint u = 0; u < e.numUpml; ++u)
                if (upml & (1u << u)) upmlStage(1, eStep, up[u], eStep ? E : H, U, utab, St, sx, sy, d, tid, tgs);
            threadgroup_barrier(mem_flags::mem_device);
        }
#endif
        bool deb = false;   // Debye positions (their Post stage reads the field only, as Mur's)
#if DIAMOND_DEBYE
        deb = eStep && ade;
#endif
        // Mur faces of one axis have no cell in common, and none of them writes another's shift line (DIAMOND_MURMERGE,
        // set when every Mur face is at least two lines from its shift line's opposite face). So when the faces in this
        // tile are all on one axis, each thread does Post and Apply of its elements in one go (murOne: one barrier
        // less per E step), provided the Mur faces come first in the Apply order (DIAMOND_MURFIRST) or nothing else is
        // applied in this tile, and no Debye Post, which reads the field, runs beside them; otherwise consecutive faces
        // of one axis in the Apply order share a barrier.
        bool murOne = false;
#if DIAMOND_MUR && DIAMOND_MURMERGE
        murOne = mur && (murAxes == 1u || murAxes == 2u || murAxes == 4u) && !deb && (DIAMOND_MURFIRST || (!rlc && !ade));
#endif
        if (mur || deb)
        {
#if DIAMOND_MUR
            for (uint f = 0; f < e.numMur; ++f)
                if (mur & (1u << f)) murStage(murOne ? 3 : 1, mf[f], E, mcoef, mscr, sx, sy, d, tid, tgs);
#endif
#if DIAMOND_DEBYE
            if (deb) adePost(E, arun, aele, adeC, adeS, a0, a1, sx, sy, cs, tid, tgs);
#endif
            threadgroup_barrier(mem_flags::mem_device);
        }
        if ((mur && !murOne) || rlc || ade)
        {
            // Apply in the engine's extension order (kind << 16 | index; 0 = Mur face, 1 = lumped RLC, 2 = Drude/Lorentz),
            // one after the other: Mur faces of different axes share their edge cells, an element may sit on a face; in
            // the H step only the Drude/Lorentz currents
            uint pend = 3;   // the axis of the Mur faces applied since the last barrier (3: none)
            for (uint a = 0; a < e.numApply; ++a)
            {
                const uint idx = apply[a] & 0xffffu;
                if ((apply[a] >> 16) == 0)
                {
#if DIAMOND_MUR
                    if (mur & (1u << idx))
                    {
#if DIAMOND_MURMERGE
                        if (pend != 3 && pend != mf[idx].ny)
                            threadgroup_barrier(mem_flags::mem_device);
                        murStage(2, mf[idx], E, mcoef, mscr, sx, sy, d, tid, tgs);
                        pend = mf[idx].ny;
#else
                        murStage(2, mf[idx], E, mcoef, mscr, sx, sy, d, tid, tgs);
                        threadgroup_barrier(mem_flags::mem_device);
#endif
                    }
#endif
                }
                else if ((apply[a] >> 16) == 2)
                {
#if DIAMOND_ADE
                    if (ade && aro[ab + idx + 1] > aro[ab + idx])
                    {
                        if (pend != 3) { threadgroup_barrier(mem_flags::mem_device); pend = 3; }
                        adeApply(eStep ? E : H, arun, aele, adeS, aro[ab + idx], aro[ab + idx + 1], sx, sy, cs, tid, tgs);
                        threadgroup_barrier(mem_flags::mem_device);
                    }
#endif
                }
                else if (rlc)
                {
#if DIAMOND_RLC
                    if (pend != 3) { threadgroup_barrier(mem_flags::mem_device); pend = 3; }
                    rlcStage(0, idx, E, rlcI, rlcC, rlcS, roff, rlist, li, sx, sy, d, tid, tgs, e.rlcSerial != 0);
                    threadgroup_barrier(mem_flags::mem_device);
                    rlcStage(1, idx, E, rlcI, rlcC, rlcS, roff, rlist, li, sx, sy, d, tid, tgs, e.rlcSerial != 0);
                    threadgroup_barrier(mem_flags::mem_device);
#endif
                }
            }
            if (pend != 3)
                threadgroup_barrier(mem_flags::mem_device);
        }
#if DIAMOND_EXC
        // the excitation of this step for the cells this tile updated, as runs: the entries of one field value in entry
        // order (Engine_Ext_Excitation), one thread per run; run: float index, x << 16 | y, first and end entry; entry:
        // amplitude (float bits), delay
        const uint x0 = eStep ? xq.x : xq.y, x1 = eStep ? xq.y : xq.z;
        if (x1 > x0)
        {
            device const float* sig = eStep ? vsig : csig;
            device float* f = eStep ? E : H;
            for (uint r = x0 + tid; r < x1; r += tgs)
            {
                const uint4 R = xrun[r];
                if (!inSpan(R.y, sx, sy)) continue;
                float v = f[R.x];
                for (uint l = R.z; l < R.w; ++l)
                    v = fma(as_type<float>(xele[l].x), sig[excPos(ts, xele[l].y, e)], v);
                f[R.x] = v;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
#endif
    }
}
)MSL";

struct Phase { uint32_t depth, ts0, types, lbase0, ntx0, nty0, lbase1, ntx1, nty1; };
struct Ext { uint32_t vcount, ccount, length, period0, numMur, numApply, rlcSerial, numUpml, numAde; };
struct MurFace { uint32_t ny, nyP, nyPP, line, shift, n1, startTS, coef, size; };
struct Upml { uint32_t x0, y0, z0, nx, ny, nz, cidx, flux, rcpZ, okZ; };
typedef std::array<uint32_t,4> U4;   // a uint4 in the kernel: 16-byte aligned in U
struct Ofs { uint32_t xrun, xro, xele, mf, apply, rlcI, roff, rlist, upml, arun, aro, aele;
			 uint32_t vsig, csig, mcoef, rlcC, ucoef, adeC;
			 uint32_t mscr, rlcS, adeS; };

unsigned int EnvUInt(const char* name, unsigned int def)
{
	const char* s = getenv(name);
	if ((s==NULL) || (*s==0))
		return def;
	return (unsigned int)strtoul(s, NULL, 10);
}

// Mountains of width M with valleys of width V between them along an axis of n cells, the first mountain starting at
// cell 2 and the last one ending at n-2 at the latest, so that the two cells at each domain edge (a Mur face and its
// shift line) always lie in the same (edge) valley. A rest of at least 2K-1 cells becomes a narrower last mountain.
void DiamondTiles(int n, int M, int V, int K, std::vector<int32_t>& mountains, std::vector<int32_t>& valleys)
{
	mountains.clear();
	valleys.clear();
	std::vector<int> a, b;
	int lo = 2;
	while (lo + M <= n-2)
	{
		a.push_back(lo);
		b.push_back(lo+M);
		lo += M + V;
	}
	if ((n-2) - lo >= 2*K-1)
	{
		a.push_back(lo);
		b.push_back(n-2);
	}
	for (size_t m=0; m<a.size(); ++m)
	{
		mountains.push_back(a[m]);
		mountains.push_back(b[m]);
	}
	// valleys: (left mountain's hi, right mountain's lo); the edge valleys run from 0 and to n
	int prev = 0;
	for (size_t m=0; m<a.size(); ++m)
	{
		valleys.push_back(prev);
		valleys.push_back(a[m]);
		prev = b[m];
	}
	valleys.push_back(prev);
	valleys.push_back(n);
}

} // namespace

struct Engine_Metal_Diamond::State
{
	bool tried = false;   // SetupGPU ran
	bool gpu = false;     // the timesteps run on the GPU
	id<MTLDevice> dev = nil;
	id<MTLCommandQueue> queue = nil;
	id<MTLComputePipelineState> pso = nil;
	id<MTLBuffer> volt = nil, curr = nil;
	id<MTLBuffer> pv = nil, pi = nil, opi = nil, pc = nil;
	bool split = false;   // the split coefficient tables
	// a tiling: K time steps per block, mountains of width M and valleys of width V; its tiles (x mountains, x valleys,
	// y mountains, y valleys), threads per tile, and the extension words (U: the shared ones, then its per-tile lists)
	struct Tiling
	{
		unsigned int K = 4, M = 8, V = 8, tg = 0;
		id<MTLBuffer> tiles[4] = {nil, nil, nil, nil};
		unsigned int numTiles[4] = {0, 0, 0, 0}, lbase[4] = {0, 0, 0, 0};
		id<MTLBuffer> U = nil;
		Ofs ofs = {};
		double best = 0;           // while tuning: the least GPU time per step of its timed command buffers
		unsigned int samples = 0;
	};
	std::vector<Tiling> tilings;
	unsigned int cur = 0;          // the tiling in use
	bool tuning = false;           // the command buffers take the tilings in turn until each has `tune` timed ones
	bool warm = false;             // the first command buffer of the run is encoded
	unsigned int tune = 4;
	unsigned long long rr = 0, tuneSteps = 0;
	id<MTLBuffer> Fc = nil, St = nil;   // the extension data of all tilings: read-only floats, state
	unsigned int numRLC = 0;
	Ext ext = {};
	unsigned int batch = 64;
	bool merge = true;  // the mountain x valley and valley x mountain tiles in one dispatch (OPENEMS_DIAMOND_MERGE=0: four dispatches per block)
	bool even = true;   // a call's rest of fewer than K steps is spread over its last blocks (OPENEMS_DIAMOND_EVEN=0: K, K, .., rest)
	// OPENEMS_DIAMOND_TIMING=1 prints these at the end: setup and IterateTS wall times, the calls, the GPU time of the
	// command buffers (busy) and from the first start to the last end of each call (span); =2 also commits every phase
	// dispatch in a command buffer of its own and adds the GPU time per phase (MM, MV, VM, VV; slower, for the split)
	unsigned int timing = 0;
	double setupSec = 0, runSec = 0, gpuBusy = 0, gpuSpan = 0, phaseSec[4] = {0, 0, 0, 0};
	unsigned long long steps = 0, calls = 0, cbs = 0;
};

Operator_Metal_Diamond* Operator_Metal_Diamond::New()
{
	cout << "Create FDTD operator (compressed SSE, multithreaded setup, for the Metal diamond engine)" << endl;
	Operator_Metal_Diamond* op = new Operator_Metal_Diamond();
	op->setNumThreads(0);   // CalcECOperator sizes the setup pool itself (every core); the engine has no CPU threads
	op->Init();
	return op;
}

Engine* Operator_Metal_Diamond::CreateEngine()
{
	if (!m_Use_Compression)
	{
		cerr << "Operator_Metal_Diamond: the operator could not be compressed, running the SSE engine" << endl;
		m_Engine = Engine_sse::New(this);
	}
	else
		m_Engine = Engine_Metal_Diamond::New(this);
	return m_Engine;
}

Engine_Metal_Diamond* Engine_Metal_Diamond::New(const Operator_SSE_Compressed* op)
{
	cout << "Create FDTD engine (Metal, diamond-blocked)" << endl;
	Engine_Metal_Diamond* e = new Engine_Metal_Diamond(op);
	e->Init();
	return e;
}

Engine_Metal_Diamond::Engine_Metal_Diamond(const Operator_SSE_Compressed* op) : Engine_SSE_Compressed(op)
{
	m_md = new State;
}

Engine_Metal_Diamond::~Engine_Metal_Diamond()
{
	if (m_md && m_md->gpu && m_md->timing)
	{
		const State& md = *m_md;
		cout << "Engine_Metal_Diamond: setup " << md.setupSec << " s, " << md.steps << " timesteps in " << md.runSec << " s = "
			 << (md.runSec>0 ? 1e-6*md.steps*numLines[0]*numLines[1]*numLines[2]/md.runSec : 0) << " MC/s on the GPU; "
			 << md.calls << " IterateTS calls (" << (md.calls ? (double)md.steps/md.calls : 0) << " steps each), " << md.cbs
			 << " command buffers, GPU busy " << md.gpuBusy << " s, span " << md.gpuSpan << " s";
		if (md.timing>1)
			cout << "; per phase MM " << md.phaseSec[0] << " MV " << md.phaseSec[1] << " VM " << md.phaseSec[2] << " VV " << md.phaseSec[3] << " s";
		const State::Tiling& c = md.tilings[md.cur];
		cout << "; K=" << c.K << " tiles " << c.M << "," << c.V << (md.tuning ? " (still tuning)" : "") << endl;
	}
	Reset();
	delete m_md;
	m_md = NULL;
}

void Engine_Metal_Diamond::Init()
{
	Engine_SSE_Compressed::Init();
	@autoreleasepool
	{
		m_md->dev = MTLCreateSystemDefaultDevice();
		if (m_md->dev==nil)
		{
			cerr << "Engine_Metal_Diamond: no Metal device, running the CPU loops" << endl;
			return;
		}
		// the field arrays move into shared Metal buffers (zero, as the engine's own arrays), in N-I-J-K order: the
		// I-J-K-N order of the CPU engines puts the three components of a cell side by side, so the loads of a SIMD
		// group are 48 B apart (7-9 % slower on Mur boxes of 121^3 and 201^3); the CPU fallback runs the generic loops
		ArrayLib::ArrayENG<f4vector>* arr[2] = {f4_volt_ptr, f4_curr_ptr};
		id<MTLBuffer> buf[2];
		for (int n=0; n<2; ++n)
		{
			const size_t bytes = (size_t)arr[n]->size()*sizeof(f4vector);
			buf[n] = [m_md->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
			if (buf[n]==nil)
			{
				cerr << "Engine_Metal_Diamond: cannot allocate " << bytes << " bytes of shared Metal memory, running the CPU loops" << endl;
				m_md->dev = nil;
				return;
			}
			memset([buf[n] contents], 0, bytes);
		}
		m_md->volt = buf[0];
		m_md->curr = buf[1];
		f4_volt_ptr->InitView("f4_volt", {numLines[0], numLines[1], numVectors}, (f4vector*)[m_md->volt contents], true);
		f4_curr_ptr->InitView("f4_curr", {numLines[0], numLines[1], numVectors}, (f4vector*)[m_md->curr contents], true);
	}
}

void Engine_Metal_Diamond::Reset()
{
	// the arrays point into the Metal buffers: drop them first
	Engine_SSE_Compressed::Reset();
	if (m_md)
	{
		State* s = m_md;
		m_md = new State;
		delete s;
	}
}

bool Engine_Metal_Diamond::SetupGPU()
{
	State& md = *m_md;
	md.tried = true;
	if (md.dev==nil)
		return false;
	ArrayLib::ArrayENG<f4vector>& V = *f4_volt_ptr;
	if ((V.linearIndex({1,0,0,0})!=numLines[0]*numLines[1]*numVectors) || (V.linearIndex({0,0,0,1})!=1) || ((uint64_t)V.size()*4 >= (1ull<<32)))
	{
		cerr << "Engine_Metal_Diamond: the kernels need the N-I-J-K field layout and fewer than 2^32 floats per field, running the CPU loops" << endl;
		return false;
	}

	// the extensions: UPML, excitation, Mur, lumped RLC and the dispersive materials (Drude/Lorentz, conducting sheets,
	// Debye) on the GPU, in the engine's extension order (by priority: UPML first, Mur, RLC and the dispersive ones with the
	// same priority, the excitation last); Apply order = that order. The dispersive ones only as Engine_Ext_LorentzMaterial
	// or Engine_Ext_DebyeMaterial themselves (not a subclass).
	std::vector<Engine_Ext_UPML*> upmls;
	std::vector<Engine_Ext_Mur_ABC*> murs;
	std::vector<Engine_Ext_LumpedRLC*> rlcs;
	std::vector<Engine_Extension*> lors;   // Engine_Ext_LorentzMaterial or Engine_Ext_DebyeMaterial
	bool hasDebye = false;
	std::vector<uint32_t> applyOrder;
	bool excSeen = false;
	for (size_t n=0; n<m_Eng_exts.size(); ++n)
	{
		Engine_Extension* ext = m_Eng_exts[n];
		if (dynamic_cast<Engine_Ext_Excitation*>(ext))
		{
			excSeen = true;
			continue;
		}
		Engine_Ext_UPML* upml = dynamic_cast<Engine_Ext_UPML*>(ext);
		if (upml && !excSeen && murs.empty() && rlcs.empty() && lors.empty())
		{
			const Operator_Ext_UPML* o = upml->m_Op_UPML;
			if (o->m_numLines[0] && o->m_numLines[1] && o->m_numLines[2])
				upmls.push_back(upml);
			continue;
		}
		Engine_Ext_Mur_ABC* mur = dynamic_cast<Engine_Ext_Mur_ABC*>(ext);
		if (mur && !excSeen)
		{
			applyOrder.push_back(murs.size());
			murs.push_back(mur);
			continue;
		}
		if (((typeid(*ext)==typeid(Engine_Ext_LorentzMaterial)) || (typeid(*ext)==typeid(Engine_Ext_DebyeMaterial))) && !excSeen)
		{
			applyOrder.push_back((2u<<16) | lors.size());
			lors.push_back(ext);
			hasDebye |= typeid(*ext)==typeid(Engine_Ext_DebyeMaterial);
			continue;
		}
		Engine_Ext_LumpedRLC* rlc = dynamic_cast<Engine_Ext_LumpedRLC*>(ext);
		if (rlc && !excSeen)
		{
			if (rlc->m_Op_Ext_RLC->RLC_count)
			{
				applyOrder.push_back((1u<<16) | rlcs.size());
				rlcs.push_back(rlc);
			}
			continue;
		}
		cerr << "Engine_Metal_Diamond: the extension \"" << ext->GetExtensionName() << "\" has no GPU version here, running the CPU loops" << endl;
		return false;
	}
	if ((murs.size()>32) || (rlcs.size()>0xffff) || (upmls.size()>32) || (lors.size()>0xffff))
		return false;
	// the kernel runs each UPML stage of all boxes of a tile without a barrier between them: the boxes must not overlap
	// (Create_UPML leaves out the edges and corners another box has)
	for (size_t a=0; a<upmls.size(); ++a)
		for (size_t b=a+1; b<upmls.size(); ++b)
		{
			const Operator_Ext_UPML* p = upmls[a]->m_Op_UPML;
			const Operator_Ext_UPML* q = upmls[b]->m_Op_UPML;
			bool overlap = true;
			for (int n=0; n<3; ++n)
				overlap &= (p->m_StartPos[n] < q->m_StartPos[n]+q->m_numLines[n]) && (q->m_StartPos[n] < p->m_StartPos[n]+p->m_numLines[n]);
			if (overlap)
			{
				cerr << "Engine_Metal_Diamond: two UPML boxes overlap, running the CPU loops" << endl;
				return false;
			}
		}
	// the runs of the excitation and the dispersive materials keep x and y in 16 bits each
	if ((numLines[0]>0xffff) || (numLines[1]>0xffff))
	{
		cerr << "Engine_Metal_Diamond: more than 65535 lines in x or y, running the CPU loops" << endl;
		return false;
	}

	// operator: packed coefficients (6 f4vectors per operator, in access order: vv0 vi0 vv1 vi1 vv2 vi2, ii0 iv0 ..) and the
	// index per f4vector; or split (OPENEMS_DIAMOND_COEF=packed or split): the distinct f4vectors in one table pc, vv and ii
	// first (at most 1024: 10-bit indices), and per operator the indices of vi0 vi1 vi2 and vv0 | vv1 << 10 | vv2 << 20 (ii,
	// iv the same). The operators share most of their f4vectors (the SMA launch: 12.4 MB packed; split 1.4 MB and 2.1 MB
	// of indices), so the caches hold the coefficients of more cells, for one more dependent load per cell. Split by default
	// when the packed coefficients of a 16x16-column block exceed 32 KB on average (SMA launch 275 KB, GCPW line 46 KB:
	// split 4-6 % and 3-4 % faster; patch x5 21 KB, notch x3 6 KB: no clear difference either way)
	const size_t numOps = Op->f4_vv_Compressed[0].size();
	double blockKB = 0;
	{
		std::vector<uint32_t> seen(numOps, 0);
		size_t blocks = 0, reads = 0;
		for (unsigned int bx=0; bx<numLines[0]; bx+=16)
			for (unsigned int by=0; by<numLines[1]; by+=16)
			{
				++blocks;
				for (unsigned int x=bx; x<std::min(bx+16, numLines[0]); ++x)
					for (unsigned int y=by; y<std::min(by+16, numLines[1]); ++y)
						for (unsigned int zv=0; zv<numVectors; ++zv)
						{
							const unsigned int op = Op->m_Op_index(x,y,zv);
							if (seen[op]!=blocks)
							{
								seen[op] = blocks;
								++reads;
							}
						}
			}
		blockKB = blocks ? 192.0*reads/blocks/1024 : 0;
	}
	bool split = blockKB > 32;
	if (const char* c = getenv("OPENEMS_DIAMOND_COEF"))
		if (*c)
			split = (strcmp(c, "split")==0);
	std::vector<f4vector> pv, pi, pc;
	std::vector<U4> qv, qi;
	if (split)
	{
		std::map<std::array<uint32_t,4>, uint32_t> lut;
		auto idx = [&](const f4vector& v) -> uint32_t
		{
			std::array<uint32_t,4> key;
			memcpy(key.data(), &v, sizeof(key));
			auto it = lut.find(key);
			if (it==lut.end())
			{
				it = lut.insert(std::make_pair(key, (uint32_t)pc.size())).first;
				pc.push_back(v);
			}
			return it->second;
		};
		qv.assign(numOps, U4{0, 0, 0, 0});
		qi.assign(numOps, U4{0, 0, 0, 0});
		for (size_t k=0; k<numOps; ++k)
			for (int n=0; n<3; ++n)
			{
				qv[k][3] |= idx(Op->f4_vv_Compressed[n][k]) << (10*n);
				qi[k][3] |= idx(Op->f4_ii_Compressed[n][k]) << (10*n);
			}
		if (pc.size()>1024)
			split = false;
		else
			for (size_t k=0; k<numOps; ++k)
				for (int n=0; n<3; ++n)
				{
					qv[k][n] = idx(Op->f4_vi_Compressed[n][k]);
					qi[k][n] = idx(Op->f4_iv_Compressed[n][k]);
				}
	}
	if (!split)
	{
		pc.clear();
		pv.resize(6*numOps);
		pi.resize(6*numOps);
		for (size_t k=0; k<numOps; ++k)
			for (int n=0; n<3; ++n)
			{
				pv[6*k+2*n] = Op->f4_vv_Compressed[n][k];
				pv[6*k+2*n+1] = Op->f4_vi_Compressed[n][k];
				pi[6*k+2*n] = Op->f4_ii_Compressed[n][k];
				pi[6*k+2*n+1] = Op->f4_iv_Compressed[n][k];
			}
	}

	@autoreleasepool
	{
		NSError* error = nil;
		MTLCompileOptions* options = [MTLCompileOptions new];
		// IEEE float semantics: the fields must agree with the CPU engines
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
		if (@available(macOS 15.0, *))
			options.mathMode = MTLMathModeSafe;
		else
#endif
		{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			options.fastMathEnabled = NO;
#pragma clang diagnostic pop
		}
		// Mur faces of one axis share barriers (see the kernel) when their lines keep them apart: a face line is never the
		// opposite face's shift line with three lines or more. OPENEMS_DIAMOND_MURMERGE=0: one barrier per face and stage,
		// as before.
		bool murMerge = EnvUInt("OPENEMS_DIAMOND_MURMERGE", 1)!=0;
		for (const Engine_Ext_Mur_ABC* m : murs)
			murMerge = murMerge && (numLines[m->m_ny]>=3);
		// the Mur faces come before every lumped RLC and dispersive extension in the Apply order
		bool murFirst = true;
		for (size_t a=1; a<applyOrder.size(); ++a)
			murFirst = murFirst && !(((applyOrder[a]>>16)==0) && ((applyOrder[a-1]>>16)!=0));
		// the kernel without the parts this model does not use (fewer registers, more tiles in flight)
		options.preprocessorMacros = @{@"DIAMOND_MUR": @(murs.empty() ? 0 : 1),
									   @"DIAMOND_MURMERGE": @(murMerge ? 1 : 0),
									   @"DIAMOND_MURFIRST": @(murFirst ? 1 : 0),
									   @"DIAMOND_FASTIDX": @(EnvUInt("OPENEMS_DIAMOND_FASTIDX", 1) ? 1 : 0),
									   @"DIAMOND_EXC": @((excSeen && Op->GetExcitationExtension()) ? 1 : 0),
									   @"DIAMOND_RLC": @(rlcs.empty() ? 0 : 1),
									   @"DIAMOND_UPML": @(upmls.empty() ? 0 : 1),
									   @"DIAMOND_ADE": @(lors.empty() ? 0 : 1),
									   @"DIAMOND_DEBYE": @(hasDebye ? 1 : 0),
									   @"DIAMOND_SPLIT": @(split ? 1 : 0),
									   @"DIAMOND_NX": @(numLines[0]), @"DIAMOND_NY": @(numLines[1]),
									   @"DIAMOND_NZ": @(numLines[2]), @"DIAMOND_NV": @(numVectors)};
		id<MTLLibrary> lib = [md.dev newLibraryWithSource:[NSString stringWithUTF8String:DIAMOND_SOURCE] options:options error:&error];
		if (lib==nil)
		{
			cerr << "Engine_Metal_Diamond: kernel compilation failed: " << [[error localizedDescription] UTF8String] << endl;
			return false;
		}
		id<MTLFunction> fn = [lib newFunctionWithName:@"diamond"];
		// The threads per tile are 256, 512 or 1024 unless OPENEMS_DIAMOND_TG says otherwise: whole SIMD groups, which the
		// pipeline may be told (OPENEMS_DIAMOND_TGFLAG=0: not told). Checked against the pipeline's width below.
		const unsigned int envTG = EnvUInt("OPENEMS_DIAMOND_TG", 0);
		auto pipeline = [&](bool whole) -> id<MTLComputePipelineState>
		{
			MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
			desc.computeFunction = fn;
			desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = whole ? YES : NO;
			return [md.dev newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
		};
		bool whole = (EnvUInt("OPENEMS_DIAMOND_TGFLAG", 1)!=0) && (envTG%32==0);
		md.pso = pipeline(whole);
		if ((md.pso!=nil) && whole)
		{
			const NSUInteger w = md.pso.threadExecutionWidth, mx = md.pso.maxTotalThreadsPerThreadgroup;
			bool ok = (w>0);
			for (NSUInteger tg : {(NSUInteger)256, (NSUInteger)512, (NSUInteger)1024, (NSUInteger)envTG})
				ok = ok && (tg==0 || (std::min(tg, mx) % w)==0);
			if (!ok)
				md.pso = pipeline(false);
		}
		if (md.pso==nil)
		{
			cerr << "Engine_Metal_Diamond: pipeline creation failed: " << [[error localizedDescription] UTF8String] << endl;
			return false;
		}
		md.queue = [md.dev newCommandQueue];

		auto buffer = [&](const void* data, size_t bytes) -> id<MTLBuffer>
		{
			if (bytes==0) // Metal buffers cannot be empty; the kernels do not read these
				return [md.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
			return [md.dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
		};

		if (split)
		{
			md.pv = buffer(qv.data(), qv.size()*sizeof(U4));
			md.pi = buffer(qi.data(), qi.size()*sizeof(U4));
		}
		else
		{
			md.pv = buffer(pv.data(), pv.size()*sizeof(f4vector));
			md.pi = buffer(pi.data(), pi.size()*sizeof(f4vector));
		}
		md.pc = buffer(pc.data(), pc.size()*sizeof(f4vector));
		md.split = split;
		std::vector<uint32_t> opi((size_t)numLines[0]*numLines[1]*numVectors);
		for (unsigned int x=0; x<numLines[0]; ++x)
			for (unsigned int y=0; y<numLines[1]; ++y)
				for (unsigned int zv=0; zv<numVectors; ++zv)
					opi[((size_t)x*numLines[1]+y)*numVectors+zv] = Op->m_Op_index(x,y,zv);
		md.opi = buffer(opi.data(), opi.size()*sizeof(uint32_t));

		md.batch = std::max(1u, EnvUInt("OPENEMS_DIAMOND_BATCH", 64));
		md.even = EnvUInt("OPENEMS_DIAMOND_EVEN", 1)!=0;
		md.merge = EnvUInt("OPENEMS_DIAMOND_MERGE", 1)!=0;
		md.timing = EnvUInt("OPENEMS_DIAMOND_TIMING", 0);

		// the extension data in three blobs: read-only words (Ub), read-only floats (Fb) and the state (Sn floats, zero); the
		// per-tile lists depend on the tiling and follow the shared words in each tiling's copy of them (below)
		std::vector<uint32_t> Ub;
		std::vector<float> Fb;
		size_t Sn = 0;
		auto addU = [&](const uint32_t* p, size_t n) -> uint32_t { const size_t o = Ub.size(); Ub.insert(Ub.end(), p, p+n); return (uint32_t)o; };
		auto addU4 = [&](const std::vector<U4>& v) -> uint32_t { Ub.resize((Ub.size()+3) & ~(size_t)3, 0); return addU((const uint32_t*)v.data(), 4*v.size()); };
		auto addF = [&](const float* p, size_t n) -> uint32_t { const size_t o = Fb.size(); Fb.insert(Fb.end(), p, p+n); return (uint32_t)o; };
		auto addS = [&](size_t n) -> uint32_t { const size_t o = Sn; Sn += n; return (uint32_t)o; };

		// excitation entries and the signals, as runs per step type (voltages, currents): the entries of one field value in
		// entry order (the CPU's: its line hooks sort the entries stably by x); run: float index, x << 16 | y, first and end
		// entry; entry: amplitude (float bits), delay
		Operator_Ext_Excitation* exc = Op->GetExcitationExtension();
		std::vector<std::vector<U4> > xruns(2);
		std::vector<uint32_t> xele;
		md.ext = {0, 0, 0, 0, (uint32_t)murs.size(), (uint32_t)applyOrder.size(), 0, (uint32_t)upmls.size()};
		Ofs ofs = {};
		if (excSeen && exc)
		{
			for (int st=0; st<2; ++st)
			{
				const unsigned int count = st ? exc->Curr_Count : exc->Volt_Count;
				unsigned int* const* index = st ? exc->Curr_index : exc->Volt_index;
				const unsigned short* dir = st ? exc->Curr_dir : exc->Volt_dir;
				const unsigned int* delay = st ? exc->Curr_delay : exc->Volt_delay;
				const FDTD_FLOAT* amp = st ? exc->Curr_amp : exc->Volt_amp;
				std::map<std::pair<uint32_t,uint32_t>, std::vector<unsigned int> > at;   // (x << 16 | y, float index) -> its entries
				for (unsigned int k=0; k<count; ++k)
				{
					const uint32_t px = index[0][k], py = index[1][k], pz = index[2][k];
					at[{(px<<16) | py, 4*(((dir[k]*numLines[0] + px)*numLines[1] + py)*numVectors + pz%numVectors) + pz/numVectors}].push_back(k);
				}
				for (const auto& r : at)
				{
					xruns[st].push_back({r.first.second, r.first.first, (uint32_t)(xele.size()/2), (uint32_t)(xele.size()/2 + r.second.size())});
					for (unsigned int k : r.second)
					{
						uint32_t a;
						memcpy(&a, &amp[k], 4);
						xele.push_back(a);
						xele.push_back(delay[k]);
					}
				}
			}
			Excitation* sig = exc->m_Exc;
			const unsigned int len = sig->GetLength();
			md.ext.vcount = exc->Volt_Count;
			md.ext.ccount = exc->Curr_Count;
			md.ext.length = len;
			md.ext.period0 = sig->GetSignalPeriod()>0 ? (uint32_t)int(sig->GetSignalPeriod()/sig->GetTimestep()) : 0;
			ofs.vsig = addF(sig->GetVoltageSignal(), (exc->Volt_Count && len) ? len : 0);
			ofs.csig = addF(sig->GetCurrentSignal(), (exc->Curr_Count && len) ? len : 0);
		}
		Ub.resize((Ub.size()+3) & ~(size_t)3, 0);   // the entries are uint2
		ofs.xele = addU(xele.data(), xele.size());

		// Mur faces, in Apply order: coefficients nyP then nyPP per face, a scratch array of the same size
		std::vector<MurFace> faces;
		std::vector<float> coef;
		for (Engine_Ext_Mur_ABC* m : murs)
		{
			const uint32_t size = m->m_numLines[0]*m->m_numLines[1];
			faces.push_back({(uint32_t)m->m_ny, (uint32_t)m->m_nyP, (uint32_t)m->m_nyPP, m->m_LineNr, (uint32_t)m->m_LineNr_Shift,
							 m->m_numLines[1], m->m_start_TS, (uint32_t)coef.size(), size});
			coef.insert(coef.end(), m->m_Mur_Coeff_nyP.data(), m->m_Mur_Coeff_nyP.data()+size);
			coef.insert(coef.end(), m->m_Mur_Coeff_nyPP.data(), m->m_Mur_Coeff_nyPP.data()+size);
		}
		ofs.mf = addU((const uint32_t*)faces.data(), faces.size()*sizeof(MurFace)/4);
		ofs.mcoef = addF(coef.data(), coef.size());
		ofs.mscr = addS(coef.size());

		// lumped RLC elements of all extensions in Apply order (their per-tile lists below) and a zero state (as the CPU's)
		std::vector<uint32_t> ri;
		std::vector<float> rc;
		std::set<std::array<uint32_t,5> > cells;
		for (size_t r=0; r<rlcs.size(); ++r)
		{
			const Operator_Ext_LumpedRLC* o = rlcs[r]->m_Op_Ext_RLC;
			for (unsigned int k=0; k<o->RLC_count; ++k)
			{
				const uint32_t q[5] = {o->v_RLC_pos[0][k], o->v_RLC_pos[1][k], o->v_RLC_pos[2][k], (uint32_t)o->v_RLC_dir[k], (uint32_t)r};
				ri.insert(ri.end(), q, q+5);
				const FDTD_FLOAT c[8] = {o->v_RLC_i2v[k]*o->v_RLC_ilv[k], o->v_RLC_vvd[k], o->v_RLC_vv2[k], o->v_RLC_vj1[k], o->v_RLC_vj2[k],
										 o->v_RLC_ib0[k], o->v_RLC_b1[k]*o->v_RLC_ib0[k], o->v_RLC_b2[k]*o->v_RLC_ib0[k]};
				rc.insert(rc.end(), c, c+8);
				if (!cells.insert({q[0], q[1], q[2], q[3], q[4]}).second)
					md.ext.rlcSerial = 1;   // two elements on one cell: the last one's voltage stays, as on the CPU
			}
		}
		md.numRLC = ri.size()/5;
		ofs.rlcI = addU(ri.data(), ri.size());
		ofs.rlcC = addF(rc.data(), rc.size());
		ofs.rlcS = addS((size_t)md.numRLC*5);
		ofs.apply = addU(applyOrder.data(), applyOrder.size());

		// UPML boxes: one table of the distinct coefficient sets (vv vvfo vvfn ii iifo iifn) of all boxes, an index per
		// component and box cell [n][x][y][z], the flux (zero, as the CPU's): voltages [n][x][y][z], then the currents
		std::vector<Upml> boxes;
		std::vector<float> utab;
		std::map<std::array<float,6>, uint32_t> lut;
		size_t upmlCells = 0;
		for (Engine_Ext_UPML* u : upmls)
		{
			const Operator_Ext_UPML* o = u->m_Op_UPML;
			const unsigned int NX = o->m_numLines[0], NY = o->m_numLines[1], NZ = o->m_numLines[2];
			const size_t N = (size_t)NX*NY*NZ;
			std::vector<uint32_t> idx(3*N);
			for (unsigned int n=0; n<3; ++n)
				for (unsigned int lx=0; lx<NX; ++lx)
					for (unsigned int ly=0; ly<NY; ++ly)
						for (unsigned int lz=0; lz<NZ; ++lz)
						{
							const std::array<float,6> key = {o->vv(n,lx,ly,lz), o->vvfo(n,lx,ly,lz), o->vvfn(n,lx,ly,lz),
															 o->ii(n,lx,ly,lz), o->iifo(n,lx,ly,lz), o->iifn(n,lx,ly,lz)};
							auto it = lut.find(key);
							if (it==lut.end())
							{
								it = lut.insert(std::make_pair(key, (uint32_t)(utab.size()/6))).first;
								utab.insert(utab.end(), key.begin(), key.end());
							}
							idx[((n*(size_t)NX+lx)*NY+ly)*NZ+lz] = it->second;
						}
			// i / NZ in the kernel as a multiplication (see divW): exact for every float of a tile, at most 3 * NZ per column
			// and fewer than 2^16 columns, when 2^16 * 3 * NZ * (NZ - 1) < 2^32
			const uint32_t rcpZ = (NZ>1) ? (uint32_t)(0xffffffffu/NZ + 1u) : 0;
			const uint32_t okZ = ((NZ>1) && (3ull*NZ*(NZ-1) < (1ull<<16))) ? 1 : 0;
			boxes.push_back({o->m_StartPos[0], o->m_StartPos[1], o->m_StartPos[2], NX, NY, NZ, addU(idx.data(), idx.size()), addS(6*N), rcpZ, okZ});
			upmlCells += N;
		}
		ofs.upml = addU((const uint32_t*)boxes.data(), boxes.size()*sizeof(Upml)/4);
		ofs.ucoef = addF(utab.data(), utab.size());

		// dispersive materials: flags per (extension, order): 1 voltage ADE, 2 current ADE, 4 voltage Lorentz term, 8 current
		// Lorentz term, 16 Debye (poles << 8); Lorentz: 18 coefficients and 12 state floats per element, Debye with P poles:
		// 6P+3 coefficients, 6+3P state floats; the state is zero (as the CPU's). The kernel goes over runs: the elements of
		// one extension at one cell for one step type (E: voltage ADE, H: current ADE) in the CPU's order (the orders one after
		// the other, the list order within an order), so one thread subtracts them all from its field value in that order
		// (Engine_Ext_Dispersive::Apply2*Lines) and the orders share one pass. Run: the float index of component 0, x << 16 | y,
		// first and end element; element: coefficient and state offset (the H step's parts included), flags. Per tile of each
		// phase, the runs inside the union of its ranges, by step type and extension (below).
		std::vector<float> acf;
		std::vector<U4> aele;
		std::vector<std::vector<U4> > aruns(2*lors.size());   // [step type * extensions + extension], by cell
		size_t adeState = 0, numADE = 0;
		for (size_t xi=0; xi<lors.size(); ++xi)
		{
			Engine_Extension* l = lors[xi];
			std::map<std::array<uint32_t,3>, std::vector<U4> > at[2];   // per step type: cell (x, y, z) -> its elements in order
			if (typeid(*l)==typeid(Engine_Ext_DebyeMaterial))
			{
				const Operator_Ext_DebyeMaterial* o = static_cast<Engine_Ext_DebyeMaterial*>(l)->m_Op_Ext_Deb;
				const uint32_t P = o->m_PoleCount;
				for (unsigned int i=0; i<o->m_LM_Count.at(0); ++i)
				{
					at[0][{o->m_LM_pos[0][0][i], o->m_LM_pos[0][1][i], o->m_LM_pos[0][2][i]}].push_back({(uint32_t)acf.size(), (uint32_t)adeState, 1 | 16 | (P<<8), 0});
					for (uint32_t pl=0; pl<P; ++pl)
					{
						for (int n=0; n<3; ++n)
							acf.push_back(o->v_relax_ADE[pl][n][i]);
						for (int n=0; n<3; ++n)
							acf.push_back(o->v_drive_ADE[pl][n][i]);
					}
					for (int n=0; n<3; ++n)
						acf.push_back(o->v_solve_ADE[n][i]);
					adeState += 6 + 3*P;
					++numADE;
				}
			}
			else
			{
				const Operator_Ext_LorentzMaterial* o = static_cast<Engine_Ext_LorentzMaterial*>(l)->m_Op_Ext_Lor;
				for (int ord=0; ord<o->m_Order; ++ord)
				{
					const bool vOn = o->m_volt_ADE_On[ord], cOn = o->m_curr_ADE_On[ord];
					const uint32_t g = (vOn ? 1 : 0) | (cOn ? 2 : 0) | ((vOn && o->m_volt_Lor_ADE_On[ord]) ? 4 : 0) | ((cOn && o->m_curr_Lor_ADE_On[ord]) ? 8 : 0);
					if (!vOn && !cOn)
						continue;
					for (unsigned int i=0; i<o->m_LM_Count.at(ord); ++i)
					{
						const std::array<uint32_t,3> cell = {o->m_LM_pos[ord][0][i], o->m_LM_pos[ord][1][i], o->m_LM_pos[ord][2][i]};
						if (vOn)
							at[0][cell].push_back({(uint32_t)acf.size(), (uint32_t)adeState, g, 0});
						if (cOn)
							at[1][cell].push_back({(uint32_t)acf.size()+9, (uint32_t)adeState+6, g, 0});
						float c[18] = {};
						for (int n=0; n<3; ++n)
						{
							if (vOn)
							{
								c[n] = o->v_int_ADE[ord][n][i];
								c[3+n] = o->v_ext_ADE[ord][n][i];
								if (o->m_volt_Lor_ADE_On[ord])
									c[6+n] = o->v_Lor_ADE[ord][n][i];
							}
							if (cOn)
							{
								c[9+n] = o->i_int_ADE[ord][n][i];
								c[12+n] = o->i_ext_ADE[ord][n][i];
								if (o->m_curr_Lor_ADE_On[ord])
									c[15+n] = o->i_Lor_ADE[ord][n][i];
							}
						}
						acf.insert(acf.end(), c, c+18);
						adeState += 12;
						++numADE;
					}
				}
			}
			for (int st=0; st<2; ++st)
				for (const auto& cl : at[st])
				{
					const uint32_t px = cl.first[0], py = cl.first[1], pz = cl.first[2];
					const uint32_t f0 = 4*((px*numLines[1] + py)*numVectors + pz%numVectors) + pz/numVectors;
					aruns[st*lors.size()+xi].push_back({f0, (px<<16) | py, (uint32_t)aele.size(), (uint32_t)(aele.size()+cl.second.size())});
					aele.insert(aele.end(), cl.second.begin(), cl.second.end());
				}
		}
		md.ext.numAde = lors.size();
		ofs.aele = addU4(aele);
		ofs.adeC = addF(acf.data(), acf.size());
		ofs.adeS = addS(adeState);

		if ((Fb.size()>=(1ull<<32)) || (Sn>=(1ull<<32)))
		{
			cerr << "Engine_Metal_Diamond: the extension data exceed 2^32 words, running the CPU loops" << endl;
			return false;
		}
		md.Fc = buffer(Fb.data(), Fb.size()*4);
		md.St = [md.dev newBufferWithLength:std::max<size_t>(16, Sn*4) options:MTLResourceStorageModeShared];
		if ((md.Fc==nil) || (md.St==nil))
		{
			cerr << "Engine_Metal_Diamond: cannot allocate the extension buffers, running the CPU loops" << endl;
			return false;
		}
		memset([md.St contents], 0, std::max<size_t>(16, Sn*4));

		// tilings: K time steps per block, mountains of width M and valleys of width V in x and y. OPENEMS_DIAMOND_K and
		// OPENEMS_DIAMOND_TILE (M,V or M:V) set one; otherwise IterateTS gives its first command buffers the tilings below
		// in turn and keeps the one with the least GPU time per step (OPENEMS_DIAMOND_TUNE: timed command buffers per
		// tiling, default 4; 0: K=4 8,8 without tuning). The best of the six benchmark models' tile and K sweeps is one of
		// these four, up to 12 % ahead of K=4 8,8 (paired runs on cg23 and cg27: K=4 8,8 for patch x5 and the notch
		// tutorial, K=4 12,8 for the GCPW line and notch x3, K=3 8,6 for the SMA launch, K=3 6,6 for the patch tutorial).
		// Every tiling gives every value the same operations in the same order (only independent cells change their
		// order), so the fields do not depend on the choice.
		std::vector<std::array<unsigned int,3> > cand;   // K, M, V
		{
			const char* envK = getenv("OPENEMS_DIAMOND_K");
			const char* envT = getenv("OPENEMS_DIAMOND_TILE");
			md.tune = EnvUInt("OPENEMS_DIAMOND_TUNE", 4);
			if ((envK && *envK) || (envT && *envT) || (md.tune==0) || (md.timing>1))
			{
				const unsigned int K = std::max(1u, EnvUInt("OPENEMS_DIAMOND_K", 4));
				unsigned int M = 2*K, Vw = 2*K;
				if (envT && *envT)
				{
					unsigned int m=0, v=0;
					if ((sscanf(envT, "%u%*[,:]%u", &m, &v)==2) && (m>=2*K-1))
					{
						M = m;
						Vw = v;
					}
					else
						cerr << "Engine_Metal_Diamond: OPENEMS_DIAMOND_TILE=" << envT << " ignored (M,V or M:V with M >= 2K-1 = " << 2*K-1 << ")" << endl;
				}
				cand.push_back({K, M, Vw});
			}
			else
				cand = {{4, 8, 8}, {4, 12, 8}, {3, 8, 6}, {3, 6, 6}};
		}
		for (const auto& kmv : cand)
		{
			State::Tiling c;
			c.K = kmv[0];
			c.M = kmv[1];
			c.V = kmv[2];
			std::vector<int32_t> t[4];
			DiamondTiles(numLines[0], c.M, c.V, c.K, t[0], t[1]);
			DiamondTiles(numLines[1], c.M, c.V, c.K, t[2], t[3]);
			for (int n=0; n<4; ++n)
			{
				c.numTiles[n] = t[n].size()/2;
				c.tiles[n] = buffer(t[n].data(), t[n].size()*sizeof(int32_t));
			}
			unsigned int numAll = 0;
			for (int ph=0; ph<4; ++ph)
			{
				c.lbase[ph] = numAll;
				numAll += c.numTiles[ph/2]*c.numTiles[2+ph%2];
			}
			// threads per tile: by default the power of two at or below 3/5 of a mountain's f4vectors per half-step (M*M*nv),
			// 256 to 1024 (OPENEMS_DIAMOND_TG; a K, threads sweep on the five models that run on the GPU: idle threads in a
			// small tile cost barrier time; at K=4 the patch tutorial (768 f4vectors) ran best with 256, the SMA launch (960)
			// with 512, patch x5 (2880) with 512-1024)
			unsigned int tgDef = 256;
			while ((tgDef < 1024) && (5*2*tgDef <= 3*(size_t)c.M*c.M*numVectors))
				tgDef *= 2;
			c.tg = std::min<unsigned int>(EnvUInt("OPENEMS_DIAMOND_TG", 0) ? EnvUInt("OPENEMS_DIAMOND_TG", 0) : tgDef,
										  (unsigned int)md.pso.maxTotalThreadsPerThreadgroup);

			// per tile of each phase (the kernel's li = lbase[phase] + tile x * tiles in y + tile y), the entries inside the
			// union of its ranges (a mountain stays in [lo, hi), a valley in [lo-K, hi+K)), in entry order
			auto lists = [&](const std::vector<uint32_t>& ent, size_t stride, std::vector<uint32_t>& off, std::vector<uint32_t>& list)
			{
				const size_t count = ent.size()/stride;
				off.assign(1, 0);
				list.clear();
				for (int ph=0; ph<4; ++ph)
				{
					const int xt = ph/2, yt = ph%2;
					for (unsigned int tx=0; tx<c.numTiles[xt]; ++tx)
					{
						const int x0 = t[xt][2*tx] - (xt ? (int)c.K : 0), x1 = t[xt][2*tx+1] + (xt ? (int)c.K : 0);
						for (unsigned int ty=0; ty<c.numTiles[2+yt]; ++ty)
						{
							const int y0 = t[2+yt][2*ty] - (yt ? (int)c.K : 0), y1 = t[2+yt][2*ty+1] + (yt ? (int)c.K : 0);
							for (size_t k=0; k<count; ++k)
							{
								const int x = ent[stride*k], y = ent[stride*k+1];
								if ((x>=x0) && (x<x1) && (y>=y0) && (y<y1))
									list.push_back(k);
							}
							off.push_back(list.size());
						}
					}
				}
			};
			// the same for runs (float index, x << 16 | y, ...; each segment sorted by x): per tile, the runs of every segment
			// inside the union of its ranges, the segments one after the other; off: the start of each (tile, segment), then
			// the end
			auto tileRuns = [&](const std::vector<std::vector<U4> >& segs, std::vector<U4>& out, std::vector<uint32_t>& off)
			{
				out.clear();
				off.clear();
				for (int ph=0; ph<4; ++ph)
				{
					const int xt = ph/2, yt = ph%2;
					for (unsigned int tx=0; tx<c.numTiles[xt]; ++tx)
					{
						const int x0 = std::max(0, t[xt][2*tx] - (xt ? (int)c.K : 0)), x1 = t[xt][2*tx+1] + (xt ? (int)c.K : 0);
						for (unsigned int ty=0; ty<c.numTiles[2+yt]; ++ty)
						{
							const int y0 = t[2+yt][2*ty] - (yt ? (int)c.K : 0), y1 = t[2+yt][2*ty+1] + (yt ? (int)c.K : 0);
							for (const std::vector<U4>& seg : segs)
							{
								off.push_back(out.size());
								auto r = std::lower_bound(seg.begin(), seg.end(), (uint32_t)x0 << 16, [](const U4& a, uint32_t v) { return a[1] < v; });
								for (; (r!=seg.end()) && ((int)((*r)[1] >> 16) < x1); ++r)
								{
									const int y = (*r)[1] & 0xffff;
									if ((y>=y0) && (y<y1))
										out.push_back(*r);
								}
							}
						}
					}
				}
				off.push_back(out.size());
			};
			std::vector<U4> xrun, arun;
			std::vector<uint32_t> xro, ro, rl, aro;
			tileRuns(xruns, xrun, xro);
			lists(ri, 5, ro, rl);
			tileRuns(aruns, arun, aro);

			// this tiling's words: the shared ones, then its lists
			std::vector<uint32_t> W(Ub);
			auto addW = [&](const uint32_t* p, size_t n) -> uint32_t { const size_t o = W.size(); W.insert(W.end(), p, p+n); return (uint32_t)o; };
			auto addW4 = [&](const std::vector<U4>& v) -> uint32_t { W.resize((W.size()+3) & ~(size_t)3, 0); return addW((const uint32_t*)v.data(), 4*v.size()); };
			c.ofs = ofs;
			c.ofs.xro = addW(xro.data(), xro.size());
			c.ofs.xrun = addW4(xrun);
			c.ofs.roff = addW(ro.data(), ro.size());
			c.ofs.rlist = addW(rl.data(), rl.size());
			c.ofs.aro = addW(aro.data(), aro.size());
			c.ofs.arun = addW4(arun);
			if (W.size()>=(1ull<<32))
			{
				cerr << "Engine_Metal_Diamond: the extension data exceed 2^32 words, running the CPU loops" << endl;
				return false;
			}
			c.U = buffer(W.data(), W.size()*4);
			if (c.U==nil)
			{
				cerr << "Engine_Metal_Diamond: cannot allocate the extension buffers, running the CPU loops" << endl;
				return false;
			}
			md.tilings.push_back(c);
		}
		md.cur = 0;
		md.tuning = md.tilings.size()>1;

		if (g_settings.GetVerboseLevel()>0)
		{
			cout << "Engine_Metal_Diamond: " << numLines[0] << "x" << numLines[1] << "x" << numLines[2] << " cells (" << numVectors
				 << " z-vectors, " << numOps << " operators, " << (int)(blockKB+0.5) << " KB per 16x16 columns, "
				 << (md.split ? "split coefficients (" + std::to_string(pc.size()) + " f4vectors)" : std::string("packed coefficients")) << "), ";
			if (md.tuning)
				cout << "the fastest of";
			for (const State::Tiling& c : md.tilings)
				cout << (&c==&md.tilings[0] ? " " : ", ") << "K=" << c.K << " tiles " << c.M << "," << c.V << " (x " << c.numTiles[0] << "+"
					 << c.numTiles[1] << ", y " << c.numTiles[2] << "+" << c.numTiles[3] << ", " << c.tg << " threads per tile)";
			cout << ", " << murs.size() << " Mur faces, " << md.numRLC << " lumped RLC elements, " << upmls.size() << " UPML boxes ("
				 << upmlCells << " cells, " << utab.size()/6 << " coefficient sets), " << numADE << " dispersive positions, "
				 << md.ext.vcount << "+" << md.ext.ccount << " excitation entries" << endl;
		}
	}
	md.gpu = true;
	return true;
}

bool Engine_Metal_Diamond::IterateTS(unsigned int iterTS)
{
	if (!m_md->tried)
	{
		const auto t0 = std::chrono::steady_clock::now();
		SetupGPU();
		m_md->setupSec = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
	}
	if (!m_md->gpu)
		return Engine_SSE_Compressed::IterateTS(iterTS);   // the CPU loops on the same memory

	State& md = *m_md;
	const auto t0 = std::chrono::steady_clock::now();
	NSMutableArray* timed = md.timing ? [NSMutableArray array] : nil;   // OPENEMS_DIAMOND_TIMING: the command buffers
	std::vector<int> timedPhase;                                         // and their phase (-1 = several)
	// while tuning: the command buffers in flight (at most two, so the choice follows the measurements closely), their
	// tiling, steps, and whether they count (a full batch or a whole call: every tiling is timed on the same work)
	NSMutableArray* tuned = md.tuning ? [NSMutableArray array] : nil;
	std::vector<std::array<unsigned int,3> > tunedInfo;
	auto harvest = [&](NSUInteger keep)
	{
		while (tuned && (tuned.count>keep))
		{
			id<MTLCommandBuffer> b = tuned[0];
			[b waitUntilCompleted];
			const std::array<unsigned int,3> info = tunedInfo.front();
			[tuned removeObjectAtIndex:0];
			tunedInfo.erase(tunedInfo.begin());
			const double sec = (b.GPUEndTime - b.GPUStartTime)/info[1];
			if (!md.tuning || !info[2] || (b.status!=MTLCommandBufferStatusCompleted) || !(sec>0))
				continue;
			State::Tiling& c = md.tilings[info[0]];
			c.best = c.samples ? std::min(c.best, sec) : sec;
			++c.samples;
			bool ready = true;
			unsigned int best = 0;
			for (unsigned int n=0; n<md.tilings.size(); ++n)
			{
				ready = ready && (md.tilings[n].samples>=md.tune);
				if (md.tilings[n].best<md.tilings[best].best)
					best = n;
			}
			if (!ready)
				continue;
			md.tuning = false;
			md.cur = best;
			if ((g_settings.GetVerboseLevel()>0) || md.timing)
			{
				const double cells = 1e-6*numLines[0]*numLines[1]*numLines[2];
				cout << "Engine_Metal_Diamond: K=" << md.tilings[best].K << " tiles " << md.tilings[best].M << "," << md.tilings[best].V
					 << " after " << md.tuneSteps << " timesteps of tuning (MC/s of the fastest timed command buffer:";
				for (const State::Tiling& t : md.tilings)
					cout << (&t==&md.tilings[0] ? " " : ", ") << "K=" << t.K << " " << t.M << "," << t.V << " " << (int)(cells/t.best+0.5);
				cout << ")" << endl;
			}
			for (unsigned int n=0; n<md.tilings.size(); ++n)   // the command buffers in flight keep what they use
				if (n!=best)
				{
					md.tilings[n].U = nil;
					for (int k=0; k<4; ++k)
						md.tilings[n].tiles[k] = nil;
				}
		}
	};
	unsigned int done = 0;
	while (done<iterTS)
	{
		@autoreleasepool
		{
			// while tuning, the command buffers that are timed (a full batch, or a whole call: the same work per step for every
			// tiling) take the tilings in turn; the others (the end of a call, and the first buffer of the run, which also maps
			// the buffers and warms the GPU up) run the first tiling, so no tiling always gets the short ones
			const unsigned int batch = md.tuning ? std::max(1u, md.batch/2) : md.batch;   // tuning: more samples in fewer steps
			const bool timedCB = md.tuning && md.warm && ((done==0) || (iterTS-done>=batch));
			const unsigned int ti = timedCB ? (unsigned int)(md.rr++ % md.tilings.size()) : md.cur;
			md.warm = true;
			const State::Tiling& c = md.tilings[ti];
			const unsigned int first = done;
			id<MTLCommandBuffer> cb = nil;
			id<MTLComputeCommandEncoder> enc = nil;
			auto begin = [&]()
			{
				cb = [md.queue commandBuffer];
				enc = [cb computeCommandEncoder];
				[enc setComputePipelineState:md.pso];
				id<MTLBuffer> bufs[] = {md.volt, md.curr, md.pv, md.pi, md.opi, md.pc};
				for (int n=0; n<6; ++n)
					[enc setBuffer:bufs[n] offset:0 atIndex:n];
				[enc setBytes:&md.ext length:sizeof(Ext) atIndex:9];
				[enc setBuffer:c.U offset:0 atIndex:10];
				[enc setBuffer:md.Fc offset:0 atIndex:11];
				[enc setBuffer:md.St offset:0 atIndex:12];
				[enc setBytes:&c.ofs length:sizeof(Ofs) atIndex:13];
				[enc setBuffer:c.tiles[0] offset:0 atIndex:6];    // x mountains, y mountains, x valleys, y valleys
				[enc setBuffer:c.tiles[2] offset:0 atIndex:7];
				[enc setBuffer:c.tiles[1] offset:0 atIndex:14];
				[enc setBuffer:c.tiles[3] offset:0 atIndex:15];
			};
			auto end = [&](int ph)
			{
				[enc endEncoding];
				[cb commit];
				if (timed)
				{
					[timed addObject:cb];
					timedPhase.push_back(ph);
				}
			};
			begin();
			unsigned int inCB = 0;
			while ((done<iterTS) && (inCB<batch))
			{
				// the block depth: K, and towards the end of a call the rest in blocks of nearly equal depth (10 steps at K=4:
				// 4, 3, 3 and not 4, 4, 2; a shallow block reuses its tile's cached fields less). The fields do not depend
				// on the depths.
				unsigned int depth = std::min(c.K, iterTS-done);
				if (md.even)
				{
					const unsigned int rest = iterTS-done, blocks = (rest+c.K-1)/c.K;
					depth = (rest+blocks-1)/blocks;
				}
				// MM, MV, VM, VV. A mountain x valley tile reads mountain x mountain cells and its own, a valley x mountain
				// tile the same, and no cell is in both kinds within a block (a cell leaves the x mountain or the y mountain
				// first): the two phases do not depend on each other and go in one dispatch (three per block, not four; the
				// per-phase timing keeps them apart).
				const bool two = md.merge && (md.timing<=1);
				for (int ph=0; ph<4; ++ph)
				{
					const int xt = ph/2, yt = ph%2;
					const uint32_t ntx = c.numTiles[xt], nty = c.numTiles[2+yt];
					if (two && (ph==2))
						continue;   // went with MV
					Phase p = {depth, numTS+done, (uint32_t)(xt | (yt<<1)), c.lbase[ph], ntx, nty, 0, 0, 0};
					MTLSize grid = MTLSizeMake(ntx, nty, 1);
					if (two && (ph==1))
					{
						p.types |= (1u | (0u<<1)) << 2;   // VM
						p.lbase1 = c.lbase[2];
						p.ntx1 = c.numTiles[1];
						p.nty1 = c.numTiles[2];
						if ((ntx==0) || (nty==0))
							p.ntx0 = p.nty0 = 0;
						if ((p.ntx1==0) || (p.nty1==0))
							p.ntx1 = p.nty1 = 0;
						grid = MTLSizeMake(std::max(p.ntx0, p.ntx1), std::max(p.nty0, p.nty1), 2);
					}
					if ((grid.width==0) || (grid.height==0))
						continue;
					[enc setBytes:&p length:sizeof(Phase) atIndex:8];
					[enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(c.tg, 1, 1)];
					if (md.timing>1)
					{
						end(ph);
						begin();
					}
				}
				done += depth;
				inCB += depth;
			}
			end(-1);
			if (md.tuning)
			{
				[tuned addObject:cb];
				tunedInfo.push_back({ti, done-first, timedCB ? 1u : 0u});
				md.tuneSteps += done-first;
			}
			if (done>=iterTS)
			{
				[cb waitUntilCompleted];
				if (cb.status==MTLCommandBufferStatusError)
					throw std::runtime_error(std::string("Engine_Metal_Diamond: command buffer failed: ") + [[cb.error localizedDescription] UTF8String]);
			}
			harvest(1);
		}
	}
	harvest(0);
	numTS += iterTS;
	md.runSec += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
	md.steps += iterTS;
	if (timed && (timed.count>0))
	{
		double first = 0, last = 0;
		for (NSUInteger n=0; n<timed.count; ++n)
		{
			id<MTLCommandBuffer> b = timed[n];
			[b waitUntilCompleted];   // done already (one queue, in order)
			const double g = b.GPUEndTime - b.GPUStartTime;
			md.gpuBusy += g;
			if (timedPhase[n]>=0)
				md.phaseSec[timedPhase[n]] += g;
			first = (n==0) ? b.GPUStartTime : std::min(first, b.GPUStartTime);
			last = std::max(last, b.GPUEndTime);
		}
		md.gpuSpan += last - first;
		md.cbs += timed.count;
	}
	++md.calls;
	return true;
}
