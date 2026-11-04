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
struct Phase { uint depth, ts0, xtype, ytype, lbase, nty; }; // type 0 = mountain, 1 = valley; lbase, nty: entry lists
struct Ext { uint vcount, ccount, length, period0, numMur, numApply, rlcSerial, numUpml, adeSerial; }; // period0: 0 = no signal period
struct MurFace { uint ny, nyP, nyPP, line, shift, n1, startTS, coef, size; };
struct Upml { uint x0, y0, z0, nx, ny, nz, cidx, flux; };   // box start and lines, coefficient index (U), flux (St)
// where the extension data sit in the read-only words (U), the read-only floats (Fc) and the state (St)
struct Ofs { uint vexc, voff, vlist, cexc, coff, clist, mf, apply, rlcI, roff, rlist, upml, adeE, adeG, adeX, aoff, alist;  // U
             uint vamp, camp, vsig, csig, mcoef, rlcC, ucoef, adeC;                                           // Fc
             uint mscr, rlcS, adeS; };                                                                        // St

// the f4vector of component n at (x, y, zv) in the N-I-J-K layout, and the float of cell z (z = lane*nv + zv)
inline uint I(uint n, uint x, uint y, uint zv, constant Dims& d) { return ((n*d.nx + x)*d.ny + y)*d.nv + zv; }
inline uint F(uint n, uint x, uint y, uint z, constant Dims& d) { return I(n, x, y, z % d.nv, d)*4 + z / d.nv; }

// Engine_SSE_Compressed::UpdateVoltages_IJKN / UpdateCurrents_IJKN for one f4vector, same operands in the same
// order; packed coefficients per operator: vv0 vi0 vv1 vi1 vv2 vi2 (ii0 iv0 ii1 iv1 ii2 iv2 for the currents)
inline void cellV(device float4* volt, device const float4* curr, device const float4* pv, device const uint* opi,
                  constant Dims& d, uint x, uint y, uint zv)
{
    device const float4* c = pv + 6*opi[(x*d.ny + y)*d.nv + zv];
    const uint xm = x > 0 ? x-1 : x, ym = y > 0 ? y-1 : y;
    const float4 c0 = curr[I(0,x,y,zv,d)], c1 = curr[I(1,x,y,zv,d)], c2 = curr[I(2,x,y,zv,d)];
    float4 c1zm, c0zm;
    if (zv > 0) { c1zm = curr[I(1,x,y,zv-1,d)]; c0zm = curr[I(0,x,y,zv-1,d)]; }
    else {
        const float4 a = curr[I(1,x,y,d.nv-1,d)]; c1zm = float4(0.0f, a.x, a.y, a.z);
        const float4 b = curr[I(0,x,y,d.nv-1,d)]; c0zm = float4(0.0f, b.x, b.y, b.z);
    }
    const float4 c2ym = curr[I(2,x,ym,zv,d)], c2xm = curr[I(2,xm,y,zv,d)];
    const float4 c1xm = curr[I(1,xm,y,zv,d)], c0ym = curr[I(0,x,ym,zv,d)];
    float4 v;
    v = volt[I(0,x,y,zv,d)]; v = v * c[0]; v = fma(c[1], ((c2 - c2ym) - c1) + c1zm, v); volt[I(0,x,y,zv,d)] = v;
    v = volt[I(1,x,y,zv,d)]; v = v * c[2]; v = fma(c[3], ((c0 - c0zm) - c2) + c2xm, v); volt[I(1,x,y,zv,d)] = v;
    v = volt[I(2,x,y,zv,d)]; v = v * c[4]; v = fma(c[5], ((c1 - c1xm) - c0) + c0ym, v); volt[I(2,x,y,zv,d)] = v;
}

inline void cellI(device float4* curr, device const float4* volt, device const float4* pi, device const uint* opi,
                  constant Dims& d, uint x, uint y, uint zv)
{
    device const float4* c = pi + 6*opi[(x*d.ny + y)*d.nv + zv];
    const float4 v0 = volt[I(0,x,y,zv,d)], v1 = volt[I(1,x,y,zv,d)], v2 = volt[I(2,x,y,zv,d)];
    float4 v1zp, v0zp;
    if (zv+1 < d.nv) { v1zp = volt[I(1,x,y,zv+1,d)]; v0zp = volt[I(0,x,y,zv+1,d)]; }
    else {
        const float4 a = volt[I(1,x,y,0,d)]; v1zp = float4(a.y, a.z, a.w, 0.0f);
        const float4 b = volt[I(0,x,y,0,d)]; v0zp = float4(b.y, b.z, b.w, 0.0f);
    }
    const float4 v2yp = volt[I(2,x,y+1,zv,d)], v0yp = volt[I(0,x,y+1,zv,d)];
    const float4 v2xp = volt[I(2,x+1,y,zv,d)], v1xp = volt[I(1,x+1,y,zv,d)];
    float4 i;
    i = curr[I(0,x,y,zv,d)]; i = i * c[0]; i = fma(c[1], ((v2 - v2yp) - v1) + v1zp, i); curr[I(0,x,y,zv,d)] = i;
    i = curr[I(1,x,y,zv,d)]; i = i * c[2]; i = fma(c[3], ((v0 - v0zp) - v2) + v2xp, i); curr[I(1,x,y,zv,d)] = i;
    i = curr[I(2,x,y,zv,d)]; i = i * c[4]; i = fma(c[5], ((v1 - v1xp) - v0) + v0yp, i); curr[I(2,x,y,zv,d)] = i;
}

// The range of a tile at half-step s: a mountain [lo, hi) is [lo + s/2, hi - (s+1)/2), a valley between two
// mountains (lo = the left mountain's hi, hi = the right mountain's lo) is [lo - (s+1)/2, hi + s/2); clipped to [0, n).
inline int2 span(int2 r, uint type, uint s, int n)
{
    const int fl = int(s / 2), ce = int((s + 1) / 2);
    int a = type == 0 ? r.x + fl : r.x - ce, b = type == 0 ? r.y - ce : r.y + fl;
    return int2(max(a, 0), min(b, n));
}

// Engine_Ext_Excitation: index into the signal for timestep ts
inline int excPos(uint ts, uint delay, constant Ext& e)
{
    int pos = int(ts) - int(delay);
    pos *= (pos > 0);
    pos %= int(e.period0 ? e.period0 : ts + 1);
    pos *= (pos < int(e.length));
    return pos;
}

// The Mur face cells inside the tile: the face line lies in the tile's x (y) range for an x (y) face, the shift line
// too (the domain edges sit inside the edge valleys), a z face is complete in every tile
inline uint murCount(device const MurFace& m, int2 sx, int2 sy, constant Dims& d)
{
    const uint wx = uint(sx.y - sx.x), wy = uint(sy.y - sy.x);
    return m.ny == 0 ? wy*d.nz : (m.ny == 1 ? wx*d.nz : wx*wy);
}
inline void murPos(device const MurFace& m, uint i, int2 sx, int2 sy, constant Dims& d, thread uint* pos)
{
    const uint wy = uint(sy.y - sy.x);
    if (m.ny == 0)      { pos[0] = m.line; pos[1] = uint(sy.x) + i / d.nz; pos[2] = i % d.nz; }
    else if (m.ny == 1) { pos[0] = uint(sx.x) + i / d.nz; pos[1] = m.line; pos[2] = i % d.nz; }
    else                { pos[0] = uint(sx.x) + i / wy; pos[1] = uint(sy.x) + i % wy; pos[2] = m.line; }
}

// stage 0: Pre (before the E update), 1: Post (after it), 2: Apply; Engine_Ext_Mur_ABC's MurFast arithmetic
// (vP = E_shift - c*E_face is one fmsub on arm64, vP += c*E_shift one fmadd)
inline void murStage(uint stage, device const MurFace& m, device float* E, device const float* mc, device float* ms,
                     int2 sx, int2 sy, constant Dims& d, uint tid, uint tgs)
{
    const uint count = murCount(m, sx, sy, d);
    for (uint i = tid; i < count; i += tgs)
    {
        uint pos[3];
        murPos(m, i, sx, sy, d, pos);
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
        else
        {
            ms[m.coef + k]          = fma(cP,  E[sP],  ms[m.coef + k]);
            ms[m.coef + m.size + k] = fma(cPP, E[sPP], ms[m.coef + m.size + k]);
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
inline void upmlStage(uint stage, bool eStep, device const Upml& u, device float* Fd, device const uint* U,
                      device const float* tab, device float* St, int2 sx, int2 sy, constant Dims& d, uint tid, uint tgs)
{
    const int x0 = max(sx.x, int(u.x0)), x1 = min(sx.y, int(u.x0 + u.nx));
    const int y0 = max(sy.x, int(u.y0)), y1 = min(sy.y, int(u.y0 + u.ny));
    if (x1 <= x0 || y1 <= y0) return;
    const uint wy = uint(y1 - y0), N = u.nx*u.ny*u.nz;
    const uint count = uint(x1 - x0) * wy * 3 * u.nz;
    device float* flux = St + u.flux + (eStep ? 0 : 3*N);
    const uint co = eStep ? 0 : 3;
    for (uint i = tid; i < count; i += tgs)
    {
        const uint lz = i % u.nz;
        uint r = i / u.nz;
        const uint n = r % 3;
        r /= 3;
        const uint y = uint(y0) + r % wy, x = uint(x0) + r / wy;
        const uint c = ((n*u.nx + (x - u.x0))*u.ny + (y - u.y0))*u.nz + lz;
        device const float* t = tab + 6*U[u.cidx + c] + co;
        const uint fi = F(n, x, y, u.z0 + lz, d);
        if (stage == 0) { const float h = fma(t[0], Fd[fi], -(t[1]*flux[c])); Fd[fi] = flux[c]; flux[c] = h; }
        else            { const float h = flux[c]; flux[c] = Fd[fi]; Fd[fi] = fma(t[2], flux[c], h); }
    }
}

// The dispersive materials for the positions in this tile: Engine_Ext_LorentzMaterial (Drude/Lorentz, conducting sheets)
// and Engine_Ext_DebyeMaterial. Groups = (extension, order), flags: 1 voltage ADE, 2 current ADE, 4 voltage Lorentz term,
// 8 current Lorentz term, 16 Debye (poles << 8). Element: x, y, z, group, coefficient offset, state offset.
// Lorentz coefficients v_int, v_ext, v_Lor, i_int, i_ext, i_Lor [3] each; state: volt ADE, volt Lorentz, curr ADE,
// curr Lorentz [3]. Pre (DoPre*UpdatesLines, before the update; reads the field only): Lor += v_Lor*ADE one fmadd,
// ADE *= v_int, then ADE += v_ext*(f - Lor) one fmadd (two statements: not contracted across them).
// Debye coefficients (v_relax, v_drive)[3] per pole, then v_solve[3]; state: the correction (volt ADE)[3], pre[3], W[pole][3].
// Pre: W = v_relax*W_old + v_drive*V^n (clang: fma(v_relax, W_old, v_drive*V)), pre += W - W_old; Post (after the update):
// V = (V_raw - pre)*v_solve, correction = V_raw - V, W += v_drive*V one fmadd.
inline void adePre(bool eStep, device const float* Fd, device const uint* el, device const uint* grp, device const float* ac,
                   device float* as, device const uint* off, device const uint* list, uint li, int2 sx, int2 sy,
                   constant Dims& d, uint tid, uint tgs)
{
    const uint l0 = off[li], cnt = 3*(off[li+1] - l0);
    for (uint i = tid; i < cnt; i += tgs)
    {
        const uint k = list[l0 + i/3], n = i % 3;
        device const uint* q = el + 6*k;
        const int x = int(q[0]), y = int(q[1]);
        if (x < sx.x || x >= sx.y || y < sy.x || y >= sy.y) continue;
        const uint g = grp[q[3]];
        if ((g & (eStep ? 1u : 2u)) == 0) continue;
        const float f = Fd[F(n, uint(x), uint(y), q[2], d)];
        if (g & 16u)
        {
            device const float* c = ac + q[4];
            device float* st = as + q[5];
            float pre = 0.0f;
            for (uint pl = 0; pl < (g >> 8); ++pl)
            {
                const float w0 = st[6 + 3*pl + n];
                const float w = fma(c[6*pl + n], w0, c[6*pl + 3 + n] * f);
                pre = pre + (w - w0);
                st[6 + 3*pl + n] = w;
            }
            st[3+n] = pre;
            continue;
        }
        device const float* c = ac + q[4] + (eStep ? 0 : 9);
        device float* st = as + q[5] + (eStep ? 0 : 6);
        if (g & (eStep ? 4u : 8u))
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

inline void adePost(device const float* Fd, device const uint* el, device const uint* grp, device const float* ac,
                    device float* as, device const uint* off, device const uint* list, uint li, int2 sx, int2 sy,
                    constant Dims& d, uint tid, uint tgs)
{
    const uint l0 = off[li], cnt = 3*(off[li+1] - l0);
    for (uint i = tid; i < cnt; i += tgs)
    {
        const uint k = list[l0 + i/3], n = i % 3;
        device const uint* q = el + 6*k;
        const uint g = grp[q[3]];
        if ((g & 16u) == 0) continue;
        const int x = int(q[0]), y = int(q[1]);
        if (x < sx.x || x >= sx.y || y < sy.x || y >= sy.y) continue;
        device const float* c = ac + q[4];
        device float* st = as + q[5];
        const uint P = g >> 8;
        const float vr = Fd[F(n, uint(x), uint(y), q[2], d)];
        const float v = (vr - st[3+n]) * c[6*P + n];
        st[n] = vr - v;
        for (uint pl = 0; pl < P; ++pl)
            st[6 + 3*pl + n] = fma(c[6*pl + 3 + n], v, st[6 + 3*pl + n]);
    }
}

// Engine_Ext_Dispersive::Apply2*Lines: field -= ADE, the orders of extension x one after the other (a position may have
// several); one thread in list order when a position repeats within an order
inline void adeApply(bool eStep, uint x, device float* Fd, device const uint* el, device const uint* grp,
                     device const uint* gx, device const float* as, device const uint* off, device const uint* list,
                     uint li, int2 sx, int2 sy, constant Dims& d, uint tid, uint tgs, bool serial)
{
    const uint l0 = off[li], cnt = 3*(off[li+1] - l0);
    for (uint g = gx[2*x]; g < gx[2*x+1]; ++g)
    {
        if (grp[g] & (eStep ? 1u : 2u))
            for (uint i = serial ? (tid == 0 ? 0 : cnt) : tid; i < cnt; i += serial ? 1 : tgs)
            {
                const uint k = list[l0 + i/3], n = i % 3;
                device const uint* q = el + 6*k;
                if (q[3] != g) continue;
                const int px = int(q[0]), py = int(q[1]);
                if (px < sx.x || px >= sx.y || py < sy.x || py >= sy.y) continue;
                const uint fi = F(n, uint(px), uint(py), q[2], d);
                Fd[fi] = Fd[fi] - as[q[5] + (eStep ? 0 : 6) + n];
            }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// One threadgroup advances one tile over p.depth timesteps in place, z complete, with a threadgroup barrier between
// the stages; the extensions run per tile in the CPU engine's order (Pre in reverse extension order, update, Post,
// Apply in extension order). Metal has 31 buffer slots, so the extension data come in three blobs (U, Fc, St).
kernel void diamond(device float4* volt [[buffer(0)]], device float4* curr [[buffer(1)]],
                    device const float4* pv [[buffer(2)]], device const float4* pi [[buffer(3)]],
                    device const uint* opi [[buffer(4)]], constant Dims& d [[buffer(5)]],
                    device const int2* xr [[buffer(6)]], device const int2* yr [[buffer(7)]],
                    constant Phase& p [[buffer(8)]], constant Ext& e [[buffer(9)]],
                    device const uint* U [[buffer(10)]], device const float* Fc [[buffer(11)]],
                    device float* St [[buffer(12)]], constant Ofs& o [[buffer(13)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint3 tptg [[threads_per_threadgroup]])
{
    device const uint* vexc = U + o.vexc;
    device const uint* voff = U + o.voff;
    device const uint* vlist = U + o.vlist;
    device const uint* cexc = U + o.cexc;
    device const uint* coff = U + o.coff;
    device const uint* clist = U + o.clist;
    device const MurFace* mf = (device const MurFace*)(U + o.mf);
    device const uint* apply = U + o.apply;
    device const uint* rlcI = U + o.rlcI;
    device const uint* roff = U + o.roff;
    device const uint* rlist = U + o.rlist;
    device const Upml* up = (device const Upml*)(U + o.upml);
    device const float* vamp = Fc + o.vamp;
    device const float* camp = Fc + o.camp;
    device const float* vsig = Fc + o.vsig;
    device const float* csig = Fc + o.csig;
    device const float* mcoef = Fc + o.mcoef;
    device const float* rlcC = Fc + o.rlcC;
    device const float* utab = Fc + o.ucoef;
    device const uint* adeE = U + o.adeE;
    device const uint* adeG = U + o.adeG;
    device const uint* adeX = U + o.adeX;
    device const uint* aoff = U + o.aoff;
    device const uint* alist = U + o.alist;
    device const float* adeC = Fc + o.adeC;
    device float* mscr = St + o.mscr;
    device float* rlcS = St + o.rlcS;
    device float* adeS = St + o.adeS;
    const uint tgs = tptg.x;
    const int2 rx = xr[tg.x], ry = yr[tg.y];
    device float* E = (device float*)volt;
    device float* H = (device float*)curr;
    for (uint s = 0; s < 2 * p.depth; ++s)
    {
        const int2 sx = span(rx, p.xtype, s, int(d.nx)), sy = span(ry, p.ytype, s, int(d.ny));
        const int wx = sx.y - sx.x, wy = sy.y - sy.x;
        if (wx <= 0 || wy <= 0)
            continue;   // the same for the whole threadgroup
        const uint ts = p.ts0 + s / 2;
        const bool eStep = (s & 1) == 0;
        const uint li = p.lbase + tg.x*p.nty + tg.y;   // this tile's entry lists
        bool rlc = false;   // lumped elements in this tile
#if DIAMOND_RLC
        rlc = eStep && roff[li+1] > roff[li];
#endif
        bool ade = false;   // Drude/Lorentz positions (conducting sheets) in this tile
#if DIAMOND_ADE
        ade = aoff[li+1] > aoff[li];
#endif
        uint mur = 0;   // the active Mur faces with cells in this tile
#if DIAMOND_MUR
        if (eStep)
            for (uint f = 0; f < e.numMur; ++f)
            {
                const int L = int(mf[f].line);
                if (ts < mf[f].startTS) continue;
                if (mf[f].ny == 0 && (L < sx.x || L >= sx.y)) continue;
                if (mf[f].ny == 1 && (L < sy.x || L >= sy.y)) continue;
                mur |= 1u << f;
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
            if (ade) adePre(eStep, eStep ? E : H, adeE, adeG, adeC, adeS, aoff, alist, li, sx, sy, d, tid, tgs);
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
        for (uint i = tid; i < count; i += tgs)
        {
            const uint zv = i % d.nv, r = i / d.nv;
            const uint y = uint(sy.x) + r % uint(wy), x = uint(sx.x) + r / uint(wy);
            if (eStep) cellV(volt, curr, pv, opi, d, x, y, zv);
            else if (x + 1 < d.nx && y + 1 < d.ny) cellI(curr, volt, pi, opi, d, x, y, zv);
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
        if (mur || deb)
        {
#if DIAMOND_MUR
            for (uint f = 0; f < e.numMur; ++f)
                if (mur & (1u << f)) murStage(1, mf[f], E, mcoef, mscr, sx, sy, d, tid, tgs);
#endif
#if DIAMOND_DEBYE
            if (deb) adePost(E, adeE, adeG, adeC, adeS, aoff, alist, li, sx, sy, d, tid, tgs);
#endif
            threadgroup_barrier(mem_flags::mem_device);
        }
        if (mur || rlc || ade)
            // Apply in the engine's extension order (kind << 16 | index; 0 = Mur face, 1 = lumped RLC, 2 = Drude/Lorentz),
            // one after the other: Mur faces share their edge cells, an element may sit on a face; in the H step only the
            // Drude/Lorentz currents
            for (uint a = 0; a < e.numApply; ++a)
            {
                const uint idx = apply[a] & 0xffffu;
                if ((apply[a] >> 16) == 0)
                {
#if DIAMOND_MUR
                    if (mur & (1u << idx))
                    {
                        murStage(2, mf[idx], E, mcoef, mscr, sx, sy, d, tid, tgs);
                        threadgroup_barrier(mem_flags::mem_device);
                    }
#endif
                }
                else if ((apply[a] >> 16) == 2)
                {
#if DIAMOND_ADE
                    if (ade) adeApply(eStep, idx, eStep ? E : H, adeE, adeG, adeX, adeS, aoff, alist, li, sx, sy, d, tid, tgs, e.adeSerial != 0);
#endif
                }
                else if (rlc)
                {
#if DIAMOND_RLC
                    rlcStage(0, idx, E, rlcI, rlcC, rlcS, roff, rlist, li, sx, sy, d, tid, tgs, e.rlcSerial != 0);
                    threadgroup_barrier(mem_flags::mem_device);
                    rlcStage(1, idx, E, rlcI, rlcC, rlcS, roff, rlist, li, sx, sy, d, tid, tgs, e.rlcSerial != 0);
                    threadgroup_barrier(mem_flags::mem_device);
#endif
                }
            }
#if DIAMOND_EXC
        const uint ecount = eStep ? e.vcount : e.ccount;
        if (ecount > 0)
        {
            // the excitation of this step for the cells this tile updated, in entry order (one thread: entries may
            // share a cell); entries: x, y, z, direction, delay; the tile's list holds the entries its ranges reach
            if (tid == 0)
            {
                device const uint* ex = eStep ? vexc : cexc;
                device const float* amp = eStep ? vamp : camp;
                device const float* sig = eStep ? vsig : csig;
                device const uint* off = eStep ? voff : coff;
                device const uint* list = eStep ? vlist : clist;
                device float* f = eStep ? E : H;
                for (uint l = off[li]; l < off[li+1]; ++l)
                {
                    const uint k = list[l];
                    const int x = int(ex[5*k]), y = int(ex[5*k+1]);
                    if (x < sx.x || x >= sx.y || y < sy.x || y >= sy.y) continue;
                    const uint fi = F(ex[5*k+3], uint(x), uint(y), ex[5*k+2], d);
                    f[fi] = fma(amp[k], sig[excPos(ts, ex[5*k+4], e)], f[fi]);
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
#endif
    }
}
)MSL";

struct Dims { uint32_t nx, ny, nz, nv; };
struct Phase { uint32_t depth, ts0, xtype, ytype, lbase, nty; };
struct Ext { uint32_t vcount, ccount, length, period0, numMur, numApply, rlcSerial, numUpml, adeSerial; };
struct MurFace { uint32_t ny, nyP, nyPP, line, shift, n1, startTS, coef, size; };
struct Upml { uint32_t x0, y0, z0, nx, ny, nz, cidx, flux; };
struct Ofs { uint32_t vexc, voff, vlist, cexc, coff, clist, mf, apply, rlcI, roff, rlist, upml, adeE, adeG, adeX, aoff, alist;
			 uint32_t vamp, camp, vsig, csig, mcoef, rlcC, ucoef, adeC;
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
	id<MTLBuffer> pv = nil, pi = nil, opi = nil;
	id<MTLBuffer> tiles[4] = {nil, nil, nil, nil};   // x mountains, x valleys, y mountains, y valleys
	unsigned int numTiles[4] = {0, 0, 0, 0};
	id<MTLBuffer> U = nil, Fc = nil, St = nil;   // the extension data: read-only words, read-only floats, state
	Ofs ofs = {};
	unsigned int numRLC = 0;
	unsigned int lbase[4] = {0, 0, 0, 0};
	Dims dims = {0, 0, 0, 0};
	Ext ext = {};
	unsigned int K = 4, tg = 0, batch = 64;
	double setupSec = 0, runSec = 0;   // OPENEMS_DIAMOND_TIMING=1 prints them at the end
	unsigned long long steps = 0;
};

Operator_Metal_Diamond* Operator_Metal_Diamond::New()
{
	cout << "Create FDTD operator (compressed SSE, for the Metal diamond engine)" << endl;
	Operator_Metal_Diamond* op = new Operator_Metal_Diamond();
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
	if (m_md && m_md->gpu && EnvFlag("OPENEMS_DIAMOND_TIMING"))
		cout << "Engine_Metal_Diamond: setup " << m_md->setupSec << " s, " << m_md->steps << " timesteps in " << m_md->runSec << " s = "
			 << (m_md->runSec>0 ? 1e-6*m_md->steps*numLines[0]*numLines[1]*numLines[2]/m_md->runSec : 0) << " MC/s on the GPU" << endl;
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
		// the kernel without the parts this model does not use (fewer registers, more tiles in flight)
		options.preprocessorMacros = @{@"DIAMOND_MUR": @(murs.empty() ? 0 : 1),
									   @"DIAMOND_EXC": @((excSeen && Op->GetExcitationExtension()) ? 1 : 0),
									   @"DIAMOND_RLC": @(rlcs.empty() ? 0 : 1),
									   @"DIAMOND_UPML": @(upmls.empty() ? 0 : 1),
									   @"DIAMOND_ADE": @(lors.empty() ? 0 : 1),
									   @"DIAMOND_DEBYE": @(hasDebye ? 1 : 0)};
		id<MTLLibrary> lib = [md.dev newLibraryWithSource:[NSString stringWithUTF8String:DIAMOND_SOURCE] options:options error:&error];
		if (lib==nil)
		{
			cerr << "Engine_Metal_Diamond: kernel compilation failed: " << [[error localizedDescription] UTF8String] << endl;
			return false;
		}
		id<MTLFunction> fn = [lib newFunctionWithName:@"diamond"];
		md.pso = [md.dev newComputePipelineStateWithFunction:fn error:&error];
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

		// operator: packed coefficients (6 f4vectors per operator, in access order) and the index per f4vector
		const size_t numOps = Op->f4_vv_Compressed[0].size();
		std::vector<f4vector> pv(6*numOps), pi(6*numOps);
		for (size_t k=0; k<numOps; ++k)
			for (int n=0; n<3; ++n)
			{
				pv[6*k+2*n] = Op->f4_vv_Compressed[n][k];
				pv[6*k+2*n+1] = Op->f4_vi_Compressed[n][k];
				pi[6*k+2*n] = Op->f4_ii_Compressed[n][k];
				pi[6*k+2*n+1] = Op->f4_iv_Compressed[n][k];
			}
		md.pv = buffer(pv.data(), pv.size()*sizeof(f4vector));
		md.pi = buffer(pi.data(), pi.size()*sizeof(f4vector));
		std::vector<uint32_t> opi((size_t)numLines[0]*numLines[1]*numVectors);
		for (unsigned int x=0; x<numLines[0]; ++x)
			for (unsigned int y=0; y<numLines[1]; ++y)
				for (unsigned int zv=0; zv<numVectors; ++zv)
					opi[((size_t)x*numLines[1]+y)*numVectors+zv] = Op->m_Op_index(x,y,zv);
		md.opi = buffer(opi.data(), opi.size()*sizeof(uint32_t));
		md.dims = {numLines[0], numLines[1], numLines[2], numVectors};

		// tiles
		md.K = std::max(1u, EnvUInt("OPENEMS_DIAMOND_K", 4));
		unsigned int M = 2*md.K, Vw = 2*md.K;
		if (const char* t = getenv("OPENEMS_DIAMOND_TILE"))
			if (*t)
			{
				unsigned int m=0, v=0;
				if ((sscanf(t, "%u%*[,:]%u", &m, &v)==2) && (m>=2*md.K-1))
				{
					M = m;
					Vw = v;
				}
				else
					cerr << "Engine_Metal_Diamond: OPENEMS_DIAMOND_TILE=" << t << " ignored (M,V or M:V with M >= 2K-1 = " << 2*md.K-1 << ")" << endl;
			}
		std::vector<int32_t> t[4];
		DiamondTiles(numLines[0], M, Vw, md.K, t[0], t[1]);
		DiamondTiles(numLines[1], M, Vw, md.K, t[2], t[3]);
		for (int n=0; n<4; ++n)
		{
			md.numTiles[n] = t[n].size()/2;
			md.tiles[n] = buffer(t[n].data(), t[n].size()*sizeof(int32_t));
		}
		// threads per tile: by default the power of two at or below 3/5 of a mountain's f4vectors per half-step (M*M*nv),
		// 256 to 1024 (OPENEMS_DIAMOND_TG; a K, threads sweep on the five models that run on the GPU: idle threads in a
		// small tile cost barrier time; at K=4 the patch tutorial (768 f4vectors) ran best with 256, the SMA launch (960)
		// with 512, patch x5 (2880) with 512-1024)
		unsigned int tgDef = 256;
		while ((tgDef < 1024) && (5*2*tgDef <= 3*(size_t)M*M*numVectors))
			tgDef *= 2;
		md.tg = std::min<unsigned int>(EnvUInt("OPENEMS_DIAMOND_TG", 0) ? EnvUInt("OPENEMS_DIAMOND_TG", 0) : tgDef,
									   (unsigned int)md.pso.maxTotalThreadsPerThreadgroup);
		md.batch = std::max(1u, EnvUInt("OPENEMS_DIAMOND_BATCH", 64));

		// the extension data in three blobs: read-only words (Ub), read-only floats (Fb) and the state (Sn floats, zero)
		std::vector<uint32_t> Ub;
		std::vector<float> Fb;
		size_t Sn = 0;
		auto addU = [&](const uint32_t* p, size_t n) -> uint32_t { const size_t o = Ub.size(); Ub.insert(Ub.end(), p, p+n); return (uint32_t)o; };
		auto addF = [&](const float* p, size_t n) -> uint32_t { const size_t o = Fb.size(); Fb.insert(Fb.end(), p, p+n); return (uint32_t)o; };
		auto addS = [&](size_t n) -> uint32_t { const size_t o = Sn; Sn += n; return (uint32_t)o; };

		// excitation entries (x, y, z, direction, delay) and the signals
		Operator_Ext_Excitation* exc = Op->GetExcitationExtension();
		std::vector<uint32_t> ve, ce;
		std::vector<float> va, ca;
		md.ext = {0, 0, 0, 0, (uint32_t)murs.size(), (uint32_t)applyOrder.size(), 0, (uint32_t)upmls.size()};
		md.ofs = {};
		if (excSeen && exc)
		{
			for (unsigned int k=0; k<exc->Volt_Count; ++k)
			{
				const uint32_t e[5] = {exc->Volt_index[0][k], exc->Volt_index[1][k], exc->Volt_index[2][k], exc->Volt_dir[k], exc->Volt_delay[k]};
				ve.insert(ve.end(), e, e+5);
				va.push_back(exc->Volt_amp[k]);
			}
			for (unsigned int k=0; k<exc->Curr_Count; ++k)
			{
				const uint32_t e[5] = {exc->Curr_index[0][k], exc->Curr_index[1][k], exc->Curr_index[2][k], exc->Curr_dir[k], exc->Curr_delay[k]};
				ce.insert(ce.end(), e, e+5);
				ca.push_back(exc->Curr_amp[k]);
			}
			Excitation* sig = exc->m_Exc;
			const unsigned int len = sig->GetLength();
			md.ext.vcount = exc->Volt_Count;
			md.ext.ccount = exc->Curr_Count;
			md.ext.length = len;
			md.ext.period0 = sig->GetSignalPeriod()>0 ? (uint32_t)int(sig->GetSignalPeriod()/sig->GetTimestep()) : 0;
			md.ofs.vsig = addF(sig->GetVoltageSignal(), (exc->Volt_Count && len) ? len : 0);
			md.ofs.csig = addF(sig->GetCurrentSignal(), (exc->Curr_Count && len) ? len : 0);
		}
		// per tile of each phase, the entries inside the union of its ranges (a mountain stays in [lo, hi), a valley in
		// [lo-K, hi+K)), in entry order
		auto lists = [&](const std::vector<uint32_t>& ent, size_t stride, std::vector<uint32_t>& off, std::vector<uint32_t>& list)
		{
			const size_t count = ent.size()/stride;
			off.assign(1, 0);
			list.clear();
			for (int ph=0; ph<4; ++ph)
			{
				const int xt = ph/2, yt = ph%2;
				md.lbase[ph] = off.size()-1;
				for (unsigned int tx=0; tx<md.numTiles[xt]; ++tx)
				{
					const int x0 = t[xt][2*tx] - (xt ? (int)md.K : 0), x1 = t[xt][2*tx+1] + (xt ? (int)md.K : 0);
					for (unsigned int ty=0; ty<md.numTiles[2+yt]; ++ty)
					{
						const int y0 = t[2+yt][2*ty] - (yt ? (int)md.K : 0), y1 = t[2+yt][2*ty+1] + (yt ? (int)md.K : 0);
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
		std::vector<uint32_t> vo, vl, co, cl;
		lists(ve, 5, vo, vl);
		lists(ce, 5, co, cl);
		md.ofs.voff = addU(vo.data(), vo.size());
		md.ofs.vlist = addU(vl.data(), vl.size());
		md.ofs.coff = addU(co.data(), co.size());
		md.ofs.clist = addU(cl.data(), cl.size());
		md.ofs.vexc = addU(ve.data(), ve.size());
		md.ofs.cexc = addU(ce.data(), ce.size());
		md.ofs.vamp = addF(va.data(), va.size());
		md.ofs.camp = addF(ca.data(), ca.size());

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
		md.ofs.mf = addU((const uint32_t*)faces.data(), faces.size()*sizeof(MurFace)/4);
		md.ofs.mcoef = addF(coef.data(), coef.size());
		md.ofs.mscr = addS(coef.size());

		// lumped RLC elements of all extensions in Apply order, their per-tile lists and a zero state (as the CPU's)
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
		std::vector<uint32_t> ro, rl;
		lists(ri, 5, ro, rl);
		md.ofs.rlcI = addU(ri.data(), ri.size());
		md.ofs.rlcC = addF(rc.data(), rc.size());
		md.ofs.rlcS = addS((size_t)md.numRLC*5);
		md.ofs.roff = addU(ro.data(), ro.size());
		md.ofs.rlist = addU(rl.data(), rl.size());
		md.ofs.apply = addU(applyOrder.data(), applyOrder.size());

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
			boxes.push_back({o->m_StartPos[0], o->m_StartPos[1], o->m_StartPos[2], NX, NY, NZ, addU(idx.data(), idx.size()), addS(6*N)});
			upmlCells += N;
		}
		md.ofs.upml = addU((const uint32_t*)boxes.data(), boxes.size()*sizeof(Upml)/4);
		md.ofs.ucoef = addF(utab.data(), utab.size());

		// dispersive positions: groups = (extension, order) with their flags, the group range per extension, elements
		// (x, y, z, group, coefficient offset, state offset); Lorentz: 18 coefficients, 12 state floats; Debye with P poles:
		// 6P+3 coefficients, 6+3P state floats; the state is zero (as the CPU's); per-tile lists
		std::vector<uint32_t> ae, ag, ax;
		std::vector<float> acf;
		size_t adeState = 0;
		for (Engine_Extension* l : lors)
		{
			ax.push_back(ag.size());
			if (typeid(*l)==typeid(Engine_Ext_DebyeMaterial))
			{
				const Operator_Ext_DebyeMaterial* o = static_cast<Engine_Ext_DebyeMaterial*>(l)->m_Op_Ext_Deb;
				const uint32_t g = ag.size(), P = o->m_PoleCount;
				ag.push_back(1 | 16 | (P<<8));
				std::set<std::array<uint32_t,3> > cells;
				for (unsigned int i=0; i<o->m_LM_Count.at(0); ++i)
				{
					const uint32_t q[6] = {o->m_LM_pos[0][0][i], o->m_LM_pos[0][1][i], o->m_LM_pos[0][2][i], g, (uint32_t)acf.size(), (uint32_t)adeState};
					ae.insert(ae.end(), q, q+6);
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
					if (!cells.insert({q[0], q[1], q[2]}).second)
						md.ext.adeSerial = 1;
				}
				ax.push_back(ag.size());
				continue;
			}
			const Operator_Ext_LorentzMaterial* o = static_cast<Engine_Ext_LorentzMaterial*>(l)->m_Op_Ext_Lor;
			for (int ord=0; ord<o->m_Order; ++ord)
			{
				const uint32_t g = ag.size();
				const bool vOn = o->m_volt_ADE_On[ord], cOn = o->m_curr_ADE_On[ord];
				ag.push_back((vOn ? 1 : 0) | (cOn ? 2 : 0) | ((vOn && o->m_volt_Lor_ADE_On[ord]) ? 4 : 0) | ((cOn && o->m_curr_Lor_ADE_On[ord]) ? 8 : 0));
				if (!vOn && !cOn)
					continue;
				std::set<std::array<uint32_t,3> > cells;
				for (unsigned int i=0; i<o->m_LM_Count.at(ord); ++i)
				{
					const uint32_t q[6] = {o->m_LM_pos[ord][0][i], o->m_LM_pos[ord][1][i], o->m_LM_pos[ord][2][i], g, (uint32_t)acf.size(), (uint32_t)adeState};
					ae.insert(ae.end(), q, q+6);
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
					if (!cells.insert({q[0], q[1], q[2]}).second)
						md.ext.adeSerial = 1;   // a position twice in one order: the CPU's sequence, one thread
				}
			}
			ax.push_back(ag.size());
		}
		const size_t numADE = ae.size()/6;
		std::vector<uint32_t> ao, al;
		lists(ae, 6, ao, al);
		md.ofs.adeE = addU(ae.data(), ae.size());
		md.ofs.adeG = addU(ag.data(), ag.size());
		md.ofs.adeX = addU(ax.data(), ax.size());
		md.ofs.aoff = addU(ao.data(), ao.size());
		md.ofs.alist = addU(al.data(), al.size());
		md.ofs.adeC = addF(acf.data(), acf.size());
		md.ofs.adeS = addS(adeState);

		if ((Ub.size()>=(1ull<<32)) || (Fb.size()>=(1ull<<32)) || (Sn>=(1ull<<32)))
		{
			cerr << "Engine_Metal_Diamond: the extension data exceed 2^32 words, running the CPU loops" << endl;
			return false;
		}
		md.U = buffer(Ub.data(), Ub.size()*4);
		md.Fc = buffer(Fb.data(), Fb.size()*4);
		md.St = [md.dev newBufferWithLength:std::max<size_t>(16, Sn*4) options:MTLResourceStorageModeShared];
		if ((md.U==nil) || (md.Fc==nil) || (md.St==nil))
		{
			cerr << "Engine_Metal_Diamond: cannot allocate the extension buffers, running the CPU loops" << endl;
			return false;
		}
		memset([md.St contents], 0, std::max<size_t>(16, Sn*4));

		if (g_settings.GetVerboseLevel()>0)
			cout << "Engine_Metal_Diamond: " << numLines[0] << "x" << numLines[1] << "x" << numLines[2] << " cells (" << numVectors
				 << " z-vectors), K=" << md.K << ", tiles " << M << "," << Vw << " (x " << md.numTiles[0] << "+" << md.numTiles[1]
				 << ", y " << md.numTiles[2] << "+" << md.numTiles[3] << "), " << md.tg << " threads per tile, " << murs.size()
				 << " Mur faces, " << md.numRLC << " lumped RLC elements, " << upmls.size() << " UPML boxes (" << upmlCells << " cells, "
				 << utab.size()/6 << " coefficient sets), " << numADE << " dispersive positions, " << md.ext.vcount << "+" << md.ext.ccount << " excitation entries" << endl;
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
	unsigned int done = 0;
	while (done<iterTS)
	{
		@autoreleasepool
		{
			id<MTLCommandBuffer> cb = [md.queue commandBuffer];
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:md.pso];
			id<MTLBuffer> bufs[] = {md.volt, md.curr, md.pv, md.pi, md.opi};
			for (int n=0; n<5; ++n)
				[enc setBuffer:bufs[n] offset:0 atIndex:n];
			[enc setBytes:&md.dims length:sizeof(Dims) atIndex:5];
			[enc setBytes:&md.ext length:sizeof(Ext) atIndex:9];
			[enc setBuffer:md.U offset:0 atIndex:10];
			[enc setBuffer:md.Fc offset:0 atIndex:11];
			[enc setBuffer:md.St offset:0 atIndex:12];
			[enc setBytes:&md.ofs length:sizeof(Ofs) atIndex:13];
			unsigned int inCB = 0;
			while ((done<iterTS) && (inCB<md.batch))
			{
				const unsigned int depth = std::min(md.K, iterTS-done);
				for (int ph=0; ph<4; ++ph)   // MM, MV, VM, VV
				{
					const int xt = ph/2, yt = ph%2;
					if ((md.numTiles[xt]==0) || (md.numTiles[2+yt]==0))
						continue;
					const Phase p = {depth, numTS+done, (uint32_t)xt, (uint32_t)yt, md.lbase[ph], md.numTiles[2+yt]};
					[enc setBuffer:md.tiles[xt] offset:0 atIndex:6];
					[enc setBuffer:md.tiles[2+yt] offset:0 atIndex:7];
					[enc setBytes:&p length:sizeof(Phase) atIndex:8];
					[enc dispatchThreadgroups:MTLSizeMake(md.numTiles[xt], md.numTiles[2+yt], 1)
						threadsPerThreadgroup:MTLSizeMake(md.tg, 1, 1)];
				}
				done += depth;
				inCB += depth;
			}
			[enc endEncoding];
			[cb commit];
			if (done>=iterTS)
			{
				[cb waitUntilCompleted];
				if (cb.status==MTLCommandBufferStatusError)
					throw std::runtime_error(std::string("Engine_Metal_Diamond: command buffer failed: ") + [[cb.error localizedDescription] UTF8String]);
			}
		}
	}
	numTS += iterTS;
	md.runSec += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
	md.steps += iterTS;
	return true;
}
