# type: ignore

import cutlass
from cutlass import Float32, Int64, const_expr, cute

from ..ptx import (
    ds_scale,
    ld_global_nc_f32,
    ld_shared_v2_f32,
    ldsm_x4,
    ldsm_x4_trans,
    mma_fp8,
    pack_e4m3x4,
    pack_fixed2,
    pack_half2,
    pow2_inv,
    prmt,
    red_add_u64,
    st_global_v2_b32,
    st_shared_e4m3_pair,
)
from ..utils.fragments import ldsm_a, ldsm_b
from .bwd_common import g_index
from .bwd_sm80 import AttnBwdSm80


def _split_b8(r):
    # b16 transposed `ldmatrix` rows hold two dims per key; split them into the even and odd dim B operands
    return [
        [prmt(r[0], r[1], 0x6420), prmt(r[2], r[3], 0x6420)],
        [prmt(r[0], r[1], 0x7531), prmt(r[2], r[3], 0x7531)],
    ]


def _pack_k32(src, kk2):
    # 32 queries or keys as an e4m3 A operand, in the order `_split_b8` gives the B operand
    o = 16 * kk2
    return [
        pack_e4m3x4(src[o], src[o + 1], src[o + 4], src[o + 5]),
        pack_e4m3x4(src[o + 2], src[o + 3], src[o + 6], src[o + 7]),
        pack_e4m3x4(src[o + 8], src[o + 9], src[o + 12], src[o + 13]),
        pack_e4m3x4(src[o + 10], src[o + 11], src[o + 14], src[o + 15]),
    ]


class AttnBwdSm89(AttnBwdSm80):
    """Adds e4m3 inputs for SM89 with ``mma.sync`` m16n8k32, adding each tile's products into fp32.

    dO enters dP as two e4m3 terms, and dK and dQ use power-of-two scales per key and per query, as on SM120.
    """

    FP8 = True

    def _g_row(self, row0):
        return g_index(row0)

    def _dolo_tile(self, sbase, st):
        return self._tile(sbase, self.dolo_off + st * self.q_tile_bytes, self.BM)

    @cute.jit
    def _mma_sdp(self, sbase, st, warp, lane, vf, s_acc, dp_acc):
        if const_expr(self.bf16):
            AttnBwdSm80._mma_sdp(self, sbase, st, warp, lane, vf, s_acc, dp_acc)
        else:
            ktile = self._tile(sbase, self.k_off, self.BC)
            vtile = self._tile(sbase, self.v_off, self.BC)
            qtile = self._q_tile(sbase, st)
            otile = self._do_tile(sbase, st)
            ltile = self._dolo_tile(sbase, st)
            s_acc.fill(0.0)
            dp_acc.fill(0.0)
            zero = [Float32(0.0)] * 4
            # `S^T = K_w Q^T`, `dP^T = V_w (dO_hi + dO_lo/16)^T`, rows are this warp's 16 keys
            for kk in cutlass.range_constexpr(self.row_bytes // 32):
                a_k = ldsm_a(ktile, warp * 16, kk, lane)
                if const_expr(self.v_regs):
                    a_v = [vf[4 * kk + x] for x in range(4)]
                else:
                    a_v = ldsm_a(vtile, warp * 16, kk, lane)
                for npair in cutlass.range_constexpr(self.nt // 2):
                    b_q = ldsm_b(qtile, 16 * npair, kk, lane)
                    b_o = ldsm_b(otile, 16 * npair, kk, lane)
                    b_l = ldsm_b(ltile, 16 * npair, kk, lane)
                    for h in cutlass.range_constexpr(2):
                        o = 4 * (2 * npair + h)
                        c = mma_fp8(
                            a_k,
                            [b_q[2 * h], b_q[2 * h + 1]],
                            [s_acc[o + x] for x in range(4)],
                        )
                        for x in cutlass.range_constexpr(4):
                            s_acc[o + x] = c[x]
                        # dP cancels against D, so each k step is added in fp32
                        hi = mma_fp8(a_v, [b_o[2 * h], b_o[2 * h + 1]], zero)
                        lo = mma_fp8(a_v, [b_l[2 * h], b_l[2 * h + 1]], zero)
                        for x in cutlass.range_constexpr(4):
                            dp_acc[o + x] = (
                                dp_acc[o + x] + hi[x] + lo[x] * Float32(1.0 / 16.0)
                            )

    @cute.jit
    def _key_scales(self, dp_acc):
        m = [Float32(0.0), Float32(0.0)]
        for nt in cutlass.range_constexpr(self.nt):
            for r in cutlass.range_constexpr(2):
                for e in cutlass.range_constexpr(2):
                    x = dp_acc[4 * nt + 2 * r + e]
                    m[r] = cute.arch.fmax(m[r], x if x > Float32(0.0) else -x)
        # dK scales of this thread's two keys, from the largest |dS| over the tile's queries
        return [
            ds_scale(cute.arch.warp_reduction_max(m[r], threads_in_group=4))
            for r in range(2)
        ]

    @cute.jit
    def _store_ds(self, sbase, st, dst, warp, lane, dp_acc):
        if const_expr(self.bf16):
            AttnBwdSm80._store_ds(self, sbase, st, dst, warp, lane, dp_acc)
        else:
            g = lane // 4
            t = lane % 4
            gq_base = sbase + self.ld_off + st * self.ld_stage_bytes + 2 * self.BM * 4
            # dS as `[query][key]`, scaled per query, keys in the order `_split_b8` gives K
            pos = 32 * (warp // 2) + 16 * (warp % 2) + 4 * (g // 2) + g % 2
            for nt in cutlass.range_constexpr(self.nt):
                gq = ld_shared_v2_f32(gq_base + (8 * nt + 2 * t) * 4)
                for e in cutlass.range_constexpr(2):
                    a = dst.addr(8 * nt + 2 * t + e, pos)
                    st_shared_e4m3_pair(
                        a,
                        a + 2,
                        dp_acc[4 * nt + e] * gq[e],
                        dp_acc[4 * nt + 2 + e] * gq[e],
                    )

    @cute.jit
    def _mma_dvdk(self, sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc, reds=None):
        if const_expr(self.bf16):
            AttnBwdSm80._mma_dvdk(
                self, sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc, reds
            )
        else:
            qtile = self._q_tile(sbase, st)
            otile = self._do_tile(sbase, st)
            gk = self._key_scales(dp_acc)
            ik = [pow2_inv(gk[0]), pow2_inv(gk[1])]
            # entry i belongs to key row `(i // 2) % 2` of this thread
            ds = [dp_acc[i] * gk[(i // 2) % 2] for i in range(4 * self.nt)]
            nk = self.BM // 32
            pa = [_pack_k32(s_acc, kk2) for kk2 in range(nk)]
            da = [_pack_k32(ds, kk2) for kk2 in range(nk)]
            zero = [Float32(0.0)] * 4
            steps = self.d // 16
            for npair in cutlass.range_constexpr(steps):
                pv = [zero, zero]
                pk = [zero, zero]
                # `dV += P^T dO`, `dK += dS^T Q`, this tile's products added in fp32
                for kk2 in cutlass.range_constexpr(nk):
                    bo = _split_b8(ldsm_x4_trans(otile.addr(32 * kk2 + lane, npair)))
                    bq = _split_b8(ldsm_x4_trans(qtile.addr(32 * kk2 + lane, npair)))
                    for h in cutlass.range_constexpr(2):
                        pv[h] = mma_fp8(pa[kk2], bo[h], pv[h])
                        pk[h] = mma_fp8(da[kk2], bq[h], pk[h])
                for h in cutlass.range_constexpr(2):
                    for x in cutlass.range_constexpr(4):
                        i = 4 * (2 * npair + h) + x
                        dv_acc[i] = dv_acc[i] + pv[h][x]
                        dk_acc[i] = dk_acc[i] + pk[h][x] * ik[x // 2]
                if const_expr(reds is not None):
                    per = (len(reds) + steps - 1) // steps
                    self._red_dq(reds[npair * per : (npair + 1) * per])

    @cute.jit
    def _dq(self, sbase, st, ds, row0, gdq, warp, lane):
        if const_expr(self.bf16):
            return AttnBwdSm80._dq(self, sbase, st, ds, row0, gdq, warp, lane)
        out = []
        d = self.d
        g = lane // 4
        t = lane % 4
        ktile = self._tile(sbase, self.k_off, self.BC)
        dchunks = d // 16
        zero = [Float32(0.0)] * 4
        for i in cutlass.range_constexpr(self.nb):
            blk = warp + self.WARPS * i
            q0 = (blk // dchunks) * 16  # this block's 16 queries and 16 columns
            c16 = blk % dchunks
            part = [zero, zero]
            for ks in cutlass.range_constexpr(self.BC // 32):
                a = ldsm_x4(
                    ds.addr(
                        q0 + lane % 8 + 8 * ((lane // 8) % 2),
                        16 * (2 * ks + lane // 16),
                    )
                )
                bk = _split_b8(ldsm_x4_trans(ktile.addr(32 * ks + lane, c16)))
                for h in cutlass.range_constexpr(2):
                    part[h] = mma_fp8(a, bk[h], part[h])
            for r in cutlass.range_constexpr(2):
                q_loc = q0 + g + 8 * r
                # `cq = fq / gq` turns this tile's products into dQ in fixed point
                cq = ld_global_nc_f32(self._gg + Int64(g_index(row0 + q_loc) + 128) * 4)
                for e in cutlass.range_constexpr(2):
                    # even dims from the first B operand, odd from the second
                    col = c16 * 16 + 4 * t + 2 * e
                    addr = gdq + (Int64(row0 + q_loc) * d + Int64(col)) * 4
                    out.append(
                        (
                            addr,
                            pack_fixed2(
                                part[0][2 * r + e] * cq, part[1][2 * r + e] * cq
                            ),
                        )
                    )
        return out

    def _red_dq(self, parts):
        if const_expr(self.bf16):
            AttnBwdSm80._red_dq(self, parts)
        else:
            for addr, v in parts:
                red_add_u64(addr, v)

    @cute.jit
    def _store_dkdv(
        self, warp, lane, kblock, seqlen_k, s_dk, s_dv, k_row0, dk_acc, dv_acc, mdK, mdV
    ):
        if const_expr(self.bf16):
            AttnBwdSm80._store_dkdv(
                self,
                warp,
                lane,
                kblock,
                seqlen_k,
                s_dk,
                s_dv,
                k_row0,
                dk_acc,
                dv_acc,
                mdK,
                mdV,
            )
        else:
            d = self.d
            g = lane // 4
            t = lane % 4
            gdk = Int64(mdK.iterator.toint())
            gdv = Int64(mdV.iterator.toint())
            for r in cutlass.range_constexpr(2):
                if kblock * self.BC + warp * 16 + g + 8 * r < seqlen_k:
                    krow = Int64(k_row0 + warp * 16 + g + 8 * r)
                    # tiles 2p and 2p + 1 hold the even and odd dims of 16p + 4t .. 16p + 4t + 3
                    for p in cutlass.range_constexpr(d // 16):
                        e0 = 4 * (2 * p) + 2 * r
                        e1 = 4 * (2 * p + 1) + 2 * r
                        off = (krow * d + Int64(16 * p + 4 * t)) * 2
                        for gbase, acc, s in ((gdk, dk_acc, s_dk), (gdv, dv_acc, s_dv)):
                            st_global_v2_b32(
                                gbase + off,
                                pack_half2(acc[e0] * s, acc[e1] * s, "bf16"),
                                pack_half2(acc[e0 + 1] * s, acc[e1 + 1] * s, "bf16"),
                            )
