# type: ignore

import math

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, Int64, const_expr, cute

from ..arch import smem_limit
from ..ptx import (
    cp_async16,
    ldsm_x4_trans,
    mma_16816,
    pack_half2,
    st_global_b32,
    st_global_v4_b32,
)
from ..utils.fragments import ldsm_a, ldsm_b
from ..utils.layout import SmemTile
from ..utils.warp import quad_transpose
from .constants import BF16_ONE_X2, LOG2E, P_OFFSET


class AttnFwdBase:
    """Attention forward for one block of 64 or 128 queries per CTA, 16 or 32 queries per warp, K and V by ``cp.async``.

    Subclasses provide ``_mma`` and may replace the K/V pipeline and the PV step.
    """

    BM = 128
    NUM_THREADS = 256
    FP8 = False  # whether the class has an e4m3 path
    # bf16 row sums from an MMA of the rounded P, or from fp32 adds of P
    BF16_ROWSUM_MMA = False

    def __init__(
        self,
        head_dim: int,
        causal: bool,
        num_stages: int = 4,
        pipelined: bool = True,
        block_n: int = 64,
        tile_local: bool = True,
        lpt: bool = False,
        return_lse: bool = True,
        has_seqused: bool = False,
        lse_natural: bool = False,
        bf16: bool = False,
        arch: int = 80,
        block_m: int = 128,
        o_vec: bool = False,
        warp_rows: int = 16,
        rev_m: bool = False,
        ctas_per_sm: int = 0,
        split_kv: bool = False,
    ):
        assert bf16 or self.FP8, f"{type(self).__name__} needs bf16 inputs"
        assert (
            head_dim in (64, 128) and block_n in (32, 64, 128) and block_m in (64, 128)
        )
        self.BM = block_m
        # O rows 16-byte aligned, so each lane can store 16 contiguous bytes
        self.o_vec = o_vec
        # longest causal query blocks of each head first, keeping one head's CTAs together for L2
        self.rev_m = rev_m
        # a register bound so this many CTAs fit on one SM, or 0 for none
        self.ctas_per_sm = ctas_per_sm
        assert warp_rows == 16 or (warp_rows == 32 and bf16 and not pipelined), (
            "32 queries per warp need bf16 inputs and the non-pipelined loop"
        )
        self.MW = (
            warp_rows // 16
        )  # m16 row blocks per warp, which share each K and V fragment
        self.NUM_THREADS = 32 * block_m // warp_rows
        row_bytes = head_dim * (2 if bf16 else 1)
        assert block_n * row_bytes // 16 >= self.NUM_THREADS, (
            "a K/V tile needs at least one 16-byte copy per thread"
        )
        assert num_stages >= (3 if pipelined else 2)
        # K and V of a tile in separate copy groups, so QK starts before V lands
        assert not (split_kv and pipelined), "split_kv needs the non-pipelined loop"
        self.split_kv = split_kv
        self.arch = arch
        self.d = head_dim
        self.causal = causal
        self.stages = num_stages
        self.pipelined = pipelined
        self.BN = block_n
        self.nt = block_n // 8
        self.tile_local = tile_local
        self.lpt = lpt
        self.threads = self.NUM_THREADS
        self.return_lse = return_lse
        self.has_seqused = has_seqused
        self.lse_natural = lse_natural
        self.sw_bits = 3 if head_dim == 128 else 2
        self.bf16 = bf16
        # bf16 P needs no offset against underflow
        self.p_off = 0.0 if bf16 else P_OFFSET
        self.row_bytes = head_dim * (2 if bf16 else 1)
        self.q_bytes = self.BM * self.row_bytes
        self.kv_tile_bytes = self.BN * self.row_bytes
        self.k_off = self.q_bytes
        self.v_off = self.q_bytes + num_stages * self.kv_tile_bytes
        self.smem_bytes = self.q_bytes + 2 * num_stages * self.kv_tile_bytes
        assert self.smem_bytes + self._smem_overhead() <= smem_limit(arch), (
            f"shared memory does not fit: {self.smem_bytes} B > {smem_limit(arch)} B "
            f"(d={head_dim}, block_n={block_n}, num_stages={num_stages})"
        )
        self.shared_storage = self._make_storage()

    def _smem_overhead(self):
        return 0

    def _make_storage(self):
        nbytes = self.smem_bytes

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]

        return SharedStorage

    def _mma(self, a, b, c):
        raise NotImplementedError

    def _k_tile(self, sbase, jt):
        return SmemTile(
            sbase + self.k_off + (jt % self.stages) * self.kv_tile_bytes,
            self.row_bytes,
            self.BN,
        )

    def _v_tile(self, sbase, jt):
        return SmemTile(
            sbase + self.v_off + (jt % self.stages) * self.kv_tile_bytes,
            self.row_bytes,
            self.BN,
        )

    def _load_rows(self, dst, g_addr, nrows, tidx, row_stride):
        for i in range(nrows * dst.chunks // self.NUM_THREADS):
            idx = tidx + i * self.NUM_THREADS
            row = idx // dst.chunks
            ch = idx % dst.chunks
            cp_async16(
                dst.addr(row, ch), g_addr + Int64(row) * row_stride + Int64(ch * 16)
            )

    @cute.jit
    def _load_tile(
        self,
        jn: Int32,
        n_blocks: Int32,
        sbase: Int32,
        gk: Int64,
        gv: Int64,
        tidx: Int32,
    ):
        self._load_k(jn, n_blocks, sbase, gk, tidx)
        self._load_v(jn, n_blocks, sbase, gv, tidx)

    @cute.jit
    def _load_k(self, jn: Int32, n_blocks: Int32, sbase: Int32, gk: Int64, tidx: Int32):
        if jn < n_blocks:
            self._load_rows(
                self._k_tile(sbase, jn),
                gk + Int64(jn) * Int64(self.BN) * self._kss,
                self.BN,
                tidx,
                self._kss,
            )

    @cute.jit
    def _load_v(self, jn: Int32, n_blocks: Int32, sbase: Int32, gv: Int64, tidx: Int32):
        if jn < n_blocks:
            self._load_rows(
                self._v_tile(sbase, jn),
                gv + Int64(jn) * Int64(self.BN) * self._vss,
                self.BN,
                tidx,
                self._vss,
            )

    @cute.jit
    def _qk(
        self, jt: Int32, sbase: Int32, lane: Int32, qf: cute.Tensor, acc: cute.Tensor
    ):
        ktile = self._k_tile(sbase, jt)
        acc.fill(0.0)
        for kk in cutlass.range_constexpr(self.row_bytes // 32):
            for npair in cutlass.range_constexpr(self.nt // 2):
                bfr = ldsm_b(ktile, 16 * npair, kk, lane)
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = self._mma(
                        [qf[4 * kk], qf[4 * kk + 1], qf[4 * kk + 2], qf[4 * kk + 3]],
                        [bfr[2 * h], bfr[2 * h + 1]],
                        [
                            acc[4 * nt],
                            acc[4 * nt + 1],
                            acc[4 * nt + 2],
                            acc[4 * nt + 3],
                        ],
                    )
                    for x in cutlass.range_constexpr(4):
                        acc[4 * nt + x] = c[x]

    @cute.jit
    def _qk_mw(self, jt, sbase, lane, qfs, accs):
        ktile = self._k_tile(sbase, jt)
        for acc in accs:
            acc.fill(0.0)
        for kk in cutlass.range_constexpr(self.row_bytes // 32):
            for npair in cutlass.range_constexpr(self.nt // 2):
                bfr = ldsm_b(ktile, 16 * npair, kk, lane)
                for m in cutlass.range_constexpr(self.MW):
                    qf = qfs[m]
                    acc = accs[m]
                    for h in cutlass.range_constexpr(2):
                        o = 4 * (2 * npair + h)
                        c = self._mma(
                            [qf[4 * kk + x] for x in range(4)],
                            [bfr[2 * h], bfr[2 * h + 1]],
                            [acc[o + x] for x in range(4)],
                        )
                        for x in cutlass.range_constexpr(4):
                            acc[o + x] = c[x]

    @cute.jit
    def _pv_bf16_mw(self, vtile, accs, o_accs, l_accs):
        lane = cute.arch.thread_idx()[0] % 32
        for kk in cutlass.range_constexpr(self.BN // 16):
            nt = 2 * kk
            pas = []
            for m in cutlass.range_constexpr(self.MW):
                acc = accs[m]
                l_acc = l_accs[m]
                pas.append(
                    [
                        pack_half2(acc[4 * nt + 2 * x], acc[4 * nt + 2 * x + 1], "bf16")
                        for x in range(4)
                    ]
                )
                for r in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(2):
                        o = 4 * (nt + j) + 2 * r
                        l_acc[2 * r] = l_acc[2 * r] + (acc[o] + acc[o + 1])
            key = 16 * kk + (lane % 8) + ((lane // 8) % 2) * 8
            for npair in cutlass.range_constexpr(self.d // 16):
                vb = ldsm_x4_trans(vtile.addr(key, 2 * npair + lane // 16))
                for m in cutlass.range_constexpr(self.MW):
                    o_acc = o_accs[m]
                    for h in cutlass.range_constexpr(2):
                        on = 4 * (2 * npair + h)
                        c = mma_16816(
                            pas[m],
                            [vb[2 * h], vb[2 * h + 1]],
                            [o_acc[on + x] for x in range(4)],
                            "bf16",
                        )
                        for x in cutlass.range_constexpr(4):
                            o_acc[on + x] = c[x]

    @cute.jit
    def _tile_mw(
        self,
        j,
        n_blocks,
        sbase,
        gk,
        gv,
        tidx,
        lane,
        t,
        beta,
        lims,
        mask_start,
        qfs,
        s_accs,
        o_accs,
        l_accs,
        row_maxs,
        row_runs,
    ):
        self._tile_wait(j, n_blocks, sbase, gk, gv, tidx)
        self._qk_mw(j, sbase, lane, qfs, s_accs)
        for m in cutlass.range_constexpr(self.MW):
            self._mask(j, t, lims[m][0], lims[m][1], mask_start, s_accs[m])
            self._softmax(
                beta, s_accs[m], o_accs[m], l_accs[m], row_maxs[m], row_runs[m]
            )
        self._v_wait(j, n_blocks, sbase, gv, tidx)
        self._pv_bf16_mw(self._v_tile(sbase, j), s_accs, o_accs, l_accs)
        self._tile_release(j, lane)

    @cute.jit
    def _mask(
        self,
        jt: Int32,
        t: Int32,
        lim0: Int32,
        lim1: Int32,
        mask_start: Int32,
        acc: cute.Tensor,
    ):
        neg_inf = Float32(-math.inf)
        if (
            jt * self.BN + (self.BN - 1) >= mask_start
        ):  # tiles ending before the first masked key need no mask
            kb = jt * self.BN + t * 2
            for r in cutlass.range_constexpr(2):
                lim = lim0 if r == 0 else lim1
                for nt in cutlass.range_constexpr(self.nt):
                    for e in cutlass.range_constexpr(2):
                        idx = 4 * nt + 2 * r + e
                        acc[idx] = acc[idx] if kb + (8 * nt + e) < lim else neg_inf

    @cute.jit
    def _softmax(
        self,
        beta: Float32,
        acc: cute.Tensor,
        o_acc: cute.Tensor,
        l_acc: cute.Tensor,
        row_max: cute.Tensor,
        row_run: cute.Tensor,
    ):
        neg_inf = Float32(-math.inf)
        for r in cutlass.range_constexpr(2):
            mx = neg_inf
            for nt in cutlass.range_constexpr(self.nt):
                for e in cutlass.range_constexpr(2):
                    mx = cute.arch.fmax(mx, acc[4 * nt + 2 * r + e])
            mx = cute.arch.fmax(mx, cute.arch.shuffle_sync_bfly(mx, offset=1))
            mx = cute.arch.fmax(mx, cute.arch.shuffle_sync_bfly(mx, offset=2))
            m_tile = mx * beta
            m_ref = row_max[r]  # reference for `O` and `l`
            m_run = cute.arch.fmax(row_run[r], m_tile)  # true running max
            row_run[r] = m_run
            # the tile max keeps small P values representable
            if const_expr(self.tile_local):
                # at least m_run - 60
                m_new = cute.arch.fmax(m_tile, m_run - Float32(60.0))
                m_new = m_ref if m_tile == neg_inf else m_new
            else:
                m_new = m_run
            m_safe = Float32(0.0) if m_new == neg_inf else m_new
            corr = cute.math.exp2(m_ref - m_safe, fastmath=True)
            row_max[r] = m_new
            for nt in cutlass.range_constexpr(self.d // 8):
                o_acc[4 * nt + 2 * r] = o_acc[4 * nt + 2 * r] * corr
                o_acc[4 * nt + 2 * r + 1] = o_acc[4 * nt + 2 * r + 1] * corr
            l_acc[2 * r] = l_acc[2 * r] * corr
            l_acc[2 * r + 1] = l_acc[2 * r + 1] * corr
            bias = Float32(self.p_off) - m_safe  # base-2 exponent plus the e4m3 offset
            for nt in cutlass.range_constexpr(self.nt):
                for e in cutlass.range_constexpr(2):
                    idx = 4 * nt + 2 * r + e
                    acc[idx] = cute.math.exp2(acc[idx] * beta + bias, fastmath=True)
    @cute.jit
    def _pv(
        self,
        jt: Int32,
        sbase: Int32,
        v_key: Int32,
        acc: cute.Tensor,
        o_acc: cute.Tensor,
        l_acc: cute.Tensor,
    ):
        self._pv_bf16(self._v_tile(sbase, jt), acc, o_acc, l_acc)

    @cute.jit
    def _pv_bf16(self, vtile, acc, o_acc, l_acc):
        lane = cute.arch.thread_idx()[0] % 32
        ones = Int32(BF16_ONE_X2)
        for kk in cutlass.range_constexpr(self.BN // 16):
            nt = 2 * kk
            # P of 16 keys as the A operand: rows g and g + 8, then keys + 8
            pa = [
                pack_half2(acc[4 * nt], acc[4 * nt + 1], "bf16"),
                pack_half2(acc[4 * nt + 2], acc[4 * nt + 3], "bf16"),
                pack_half2(acc[4 * nt + 4], acc[4 * nt + 5], "bf16"),
                pack_half2(acc[4 * nt + 6], acc[4 * nt + 7], "bf16"),
            ]
            if const_expr(self.BF16_ROWSUM_MMA):
                # row sums of the same rounded P
                c = mma_16816(
                    pa, [ones, ones], [l_acc[0], l_acc[1], l_acc[2], l_acc[3]], "bf16"
                )
                for x in cutlass.range_constexpr(4):
                    l_acc[x] = c[x]
            else:
                # this thread's part of each row sum, added across the quad at the end
                for r in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(2):
                        o = 4 * (nt + j) + 2 * r
                        l_acc[2 * r] = l_acc[2 * r] + (acc[o] + acc[o + 1])
            key = 16 * kk + (lane % 8) + ((lane // 8) % 2) * 8
            for npair in cutlass.range_constexpr(self.d // 16):
                vb = ldsm_x4_trans(vtile.addr(key, 2 * npair + lane // 16))
                for h in cutlass.range_constexpr(2):
                    on = 2 * npair + h
                    c = mma_16816(
                        pa,
                        [vb[2 * h], vb[2 * h + 1]],
                        [
                            o_acc[4 * on],
                            o_acc[4 * on + 1],
                            o_acc[4 * on + 2],
                            o_acc[4 * on + 3],
                        ],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        o_acc[4 * on + x] = c[x]

    @cute.jit
    def _store_o(self, go, row, t, oss, o_acc, r, inv, valid):
        if const_expr(self.o_vec):
            words = [
                pack_half2(
                    o_acc[4 * nt + 2 * r] * inv, o_acc[4 * nt + 2 * r + 1] * inv, "bf16"
                )
                for nt in range(self.d // 8)
            ]
            upw = self.d // 32
            # transpose within each quad so every lane holds 16 contiguous bytes
            units = quad_transpose(
                [words[q * upw : (q + 1) * upw] for q in range(4)], t
            )
            for w in cutlass.range_constexpr(upw):
                if valid:
                    st_global_v4_b32(
                        go + (Int64(row) * oss + Int64(8 * (t * upw + w))) * 2,
                        units[0][w],
                        units[1][w],
                        units[2][w],
                        units[3][w],
                    )
        else:
            if valid:
                for nt in cutlass.range_constexpr(self.d // 8):
                    addr = go + (Int64(row) * oss + Int64(t * 2 + 8 * nt)) * 2
                    st_global_b32(
                        addr,
                        pack_half2(
                            o_acc[4 * nt + 2 * r] * inv,
                            o_acc[4 * nt + 2 * r + 1] * inv,
                            "bf16",
                        ),
                    )

    @cute.jit
    def _prologue(
        self, n_blocks: Int32, sbase: Int32, gk: Int64, gv: Int64, tidx: Int32
    ):
        if const_expr(self.split_kv):
            for s in cutlass.range_constexpr(self.stages - 1):
                self._load_k(Int32(s), n_blocks, sbase, gk, tidx)
                cute.arch.cp_async_commit_group()
                self._load_v(Int32(s), n_blocks, sbase, gv, tidx)
                cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(2 * self.stages - 3)  # Q and K of tile 0
        else:
            for s in cutlass.range_constexpr(self.stages - 1):
                self._load_tile(Int32(s), n_blocks, sbase, gk, gv, tidx)
                cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.stages - 2)  # Q and tile 0 resident

    def _first_wait(self, n_blocks):
        pass

    @cute.jit
    def _tile_wait(
        self, j: Int32, n_blocks: Int32, sbase: Int32, gk: Int64, gv: Int64, tidx: Int32
    ):
        if const_expr(self.split_kv):
            cute.arch.cp_async_wait_group(2 * self.stages - 3)  # K of tile j
            cute.arch.sync_threads()
            self._load_k(j + (self.stages - 1), n_blocks, sbase, gk, tidx)
            cute.arch.cp_async_commit_group()
        elif const_expr(self.pipelined):
            cute.arch.cp_async_wait_group(
                self.stages - 3
            )  # tiles j and j+1 must be loaded
        else:
            cute.arch.cp_async_wait_group(self.stages - 2)  # tile j must be loaded
        if const_expr(not self.split_kv):
            cute.arch.sync_threads()
            self._load_tile(j + (self.stages - 1), n_blocks, sbase, gk, gv, tidx)
            cute.arch.cp_async_commit_group()

    @cute.jit
    def _v_wait(self, j: Int32, n_blocks: Int32, sbase: Int32, gv: Int64, tidx: Int32):
        if const_expr(self.split_kv):
            cute.arch.cp_async_wait_group(2 * self.stages - 3)  # V of tile j
            cute.arch.sync_threads()
            self._load_v(j + (self.stages - 1), n_blocks, sbase, gv, tidx)
            cute.arch.cp_async_commit_group()

    def _tile_release(self, j, lane):
        pass

    @cute.jit
    def _tile(
        self,
        j: Int32,
        n_blocks: Int32,
        sbase: Int32,
        gk: Int64,
        gv: Int64,
        tidx: Int32,
        lane: Int32,
        t: Int32,
        beta: Float32,
        lim0: Int32,
        lim1: Int32,
        v_key: Int32,
        mask_start: Int32,
        qf: cute.Tensor,
        s_acc: cute.Tensor,
        o_acc: cute.Tensor,
        l_acc: cute.Tensor,
        row_max: cute.Tensor,
        row_run: cute.Tensor,
    ):
        self._tile_wait(j, n_blocks, sbase, gk, gv, tidx)
        self._qk(j, sbase, lane, qf, s_acc)
        self._mask(j, t, lim0, lim1, mask_start, s_acc)
        self._softmax(beta, s_acc, o_acc, l_acc, row_max, row_run)
        self._v_wait(j, n_blocks, sbase, gv, tidx)
        self._pv(j, sbase, v_key, s_acc, o_acc, l_acc)
        self._tile_release(j, lane)

    @cute.jit
    def _tile_pipelined(
        self,
        j: Int32,
        n_blocks: Int32,
        sbase: Int32,
        gk: Int64,
        gv: Int64,
        tidx: Int32,
        lane: Int32,
        t: Int32,
        beta: Float32,
        lim0: Int32,
        lim1: Int32,
        v_key: Int32,
        mask_start: Int32,
        qf: cute.Tensor,
        s_cur: cute.Tensor,
        s_next: cute.Tensor,
        o_acc: cute.Tensor,
        l_acc: cute.Tensor,
        row_max: cute.Tensor,
        row_run: cute.Tensor,
    ):
        self._tile_wait(j, n_blocks, sbase, gk, gv, tidx)
        # overlaps tile j's softmax; the last read is unused
        self._qk(j + 1, sbase, lane, qf, s_next)
        self._softmax(beta, s_cur, o_acc, l_acc, row_max, row_run)
        self._pv(j, sbase, v_key, s_cur, o_acc, l_acc)
        self._tile_release(j, lane)
        self._mask(j + 1, t, lim0, lim1, mask_start, s_next)
        for x in cutlass.range_constexpr(4 * self.nt):
            s_cur[x] = s_next[x]

    def _tma_args(self, mK, mV):
        return None

    def _setup(self, storage, tidx, hkv, b, tma_args):
        return Int32(
            storage.buf.get_tensor(cute.make_layout(self.smem_bytes)).iterator.toint()
        )

    def _grid(self, nq_pad, h_q, batch):
        if self.lpt:
            return [
                h_q * batch,
                nq_pad // self.BM,
                1,
            ]  # longest causal CTAs launch first
        return [nq_pad // self.BM, h_q, batch]

    def _min_blocks(self):
        return self.ctas_per_sm

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mPmax: cute.Tensor,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDv: cute.Tensor,
        mSeqK: cute.Tensor,
        softmax_scale: Float32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        nq_pad: Int32,
        nk_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
        batch: Int32,
        qsb: Int64,
        qsh: Int64,
        qss: Int64,
        ksb: Int64,
        ksh: Int64,
        kss: Int64,
        vsb: Int64,
        vsh: Int64,
        vss: Int64,
        osb: Int64,
        osh: Int64,
        oss: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mPmax,
            mDq,
            mDk,
            mDv,
            mSeqK,
            softmax_scale,
            seqlen_q,
            seqlen_k,
            nq_pad,
            nk_pad,
            h_q,
            h_kv,
            qsb,
            qsh,
            qss,
            ksb,
            ksh,
            kss,
            vsb,
            vsh,
            vss,
            osb,
            osh,
            oss,
            self._tma_args(mK, mV),
        ).launch(
            grid=self._grid(nq_pad, h_q, batch),
            block=[self.threads, 1, 1],
            min_blocks_per_mp=self._min_blocks(),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    def _producer(self, warp, n_blocks):
        pass

    def _consumer_start(self):
        pass

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mPmax: cute.Tensor,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDv: cute.Tensor,
        mSeqK: cute.Tensor,
        softmax_scale: Float32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        nq_pad: Int32,
        nk_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
        qsb: Int64,
        qsh: Int64,
        qss: Int64,
        ksb: Int64,
        ksh: Int64,
        kss: Int64,
        vsb: Int64,
        vsh: Int64,
        vss: Int64,
        osb: Int64,
        osh: Int64,
        oss: Int64,
        tma_args,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        lane = tidx % 32
        self._kss = kss
        self._vss = vss
        env = [
            mQ,
            mK,
            mV,
            mSeqK,
            mDq,
            mDk,
            mDv,
            softmax_scale,
            seqlen_q,
            seqlen_k,
            nq_pad,
            nk_pad,
            h_q,
            h_kv,
            qsb,
            qsh,
            qss,
            ksb,
            ksh,
            vsb,
            vsh,
        ]
        out = [mO, mLSE, mPmax, osb, osh, oss]
        if const_expr(self.lpt):
            lin, mb, _ = cute.arch.block_idx()
            hq = lin % h_q
            b = lin // h_q
            m_block = nq_pad // self.BM - 1 - mb
        else:
            m_block, hq, b = cute.arch.block_idx()
            if const_expr(self.rev_m):
                m_block = nq_pad // self.BM - 1 - m_block
        hkv = hq // (h_q // h_kv)
        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sbase = self._setup(storage, tidx, hkv, b, tma_args)
        _, _, _, _, gq, gk, gv, _, n_blocks = self._item_args(env, m_block, hq, b)
        if warp >= self.NUM_THREADS // 32:  # producer warp group
            self._producer(warp, n_blocks)
        else:
            self._consumer_start()
            # Q first, then the first K/V stages
            self._load_rows(
                SmemTile(sbase, self.row_bytes, self.BM), gq, self.BM, tidx, qss
            )
            cute.arch.cp_async_commit_group()
            self._prologue(n_blocks, sbase, gk, gv, tidx)
            # consumers only
            cute.arch.barrier(barrier_id=1, number_of_threads=self.NUM_THREADS)
            self._run_item(env, out, sbase, tidx, warp, lane, m_block, hq, b)

    @cute.jit
    def _item_args(self, env, m_block, hq, b):
        (
            mQ,
            mK,
            mV,
            mSeqK,
            mDq,
            mDk,
            mDv,
            softmax_scale,
            seqlen_q,
            seqlen_k,
            nq_pad,
            nk_pad,
            h_q,
            h_kv,
        ) = env[:14]
        qsb, qsh, qss, ksb, ksh, vsb, vsh = env[14:]
        hkv = hq // (h_q // h_kv)
        beta = softmax_scale * mDq[b, hkv] * mDk[b, hkv] * LOG2E
        sv = mDv[b, hkv]
        q_row0 = (b * h_q + hq) * nq_pad + m_block * self.BM
        gq = (
            Int64(mQ.iterator.toint())
            + Int64(b) * qsb
            + Int64(hq) * qsh
            + Int64(m_block * self.BM) * qss
        )
        gk = Int64(mK.iterator.toint()) + Int64(b) * ksb + Int64(hkv) * ksh
        gv = Int64(mV.iterator.toint()) + Int64(b) * vsb + Int64(hkv) * vsh
        sk = seqlen_k
        if const_expr(self.has_seqused):
            used = mSeqK[b]
            sk = min(seqlen_k, used)
        n_blocks = nk_pad // self.BN
        if const_expr(self.has_seqused):
            n_used = (sk + self.BN - 1) // self.BN
            n_blocks = min(n_blocks, n_used)
        if const_expr(self.causal):
            kmax = m_block * self.BM + sk - seqlen_q + (self.BM - 1)
            nb = (kmax + self.BN) // self.BN
            nb = nb if nb > 0 else Int32(0)
            n_blocks = min(n_blocks, nb)
        return hkv, beta, sv, q_row0, gq, gk, gv, sk, n_blocks

    @cute.jit
    def _run_item(self, env, out, sbase, tidx, warp, lane, m_block, hq, b):
        mO, mLSE, mPmax, osb, osh, oss = out
        seqlen_q = env[8]
        d = self.d
        kq_steps = self.row_bytes // 32
        o_tiles = d // 8
        g = lane // 4
        t = lane % 4
        _, beta, sv, q_row0, _, gk, gv, sk, n_blocks = self._item_args(
            env, m_block, hq, b
        )
        qtile = SmemTile(sbase, self.row_bytes, self.BM)
        mw = self.MW
        wrow = warp * 16 * mw  # this warp's first query row in the block
        qfs, o_accs, l_accs, row_maxs, row_runs, s_curs, lims = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        mask_start = sk
        if const_expr(self.causal):
            ms = m_block * self.BM + (sk - seqlen_q + 1)
            mask_start = min(sk, ms)
        for m in cutlass.range_constexpr(mw):
            qf = cute.make_rmem_tensor(4 * kq_steps, Int32)
            for kq in cutlass.range_constexpr(kq_steps):
                qfrag = ldsm_a(qtile, wrow + 16 * m, kq, lane)
                for x in cutlass.range_constexpr(4):
                    qf[4 * kq + x] = qfrag[x]
            o_acc = cute.make_rmem_tensor(o_tiles * 4, Float32)
            o_acc.fill(0.0)
            l_acc = cute.make_rmem_tensor(4, Float32)
            l_acc.fill(0.0)
            row_max = cute.make_rmem_tensor(2, Float32)
            row_max.fill(-math.inf)
            row_run = cute.make_rmem_tensor(2, Float32)
            row_run.fill(-math.inf)
            q_idx = m_block * self.BM + wrow + 16 * m + g
            lim0 = sk
            lim1 = sk
            if const_expr(self.causal):
                lr0 = q_idx + (sk - seqlen_q + 1)
                lim0 = min(sk, lr0)
                lim1 = min(sk, lr0 + 8)
            qfs.append(qf)
            o_accs.append(o_acc)
            l_accs.append(l_acc)
            row_maxs.append(row_max)
            row_runs.append(row_run)
            s_curs.append(cute.make_rmem_tensor(4 * self.nt, Float32))
            lims.append((lim0, lim1))

        li = lane % 16
        v_key = ((li % 4) // 2) * 8 + (li // 4) * 2 + (li % 2) + (lane // 16) * 16

        if const_expr(mw > 1):
            for j in cutlass.range(n_blocks, unroll=1):
                self._tile_mw(
                    j,
                    n_blocks,
                    sbase,
                    gk,
                    gv,
                    tidx,
                    lane,
                    t,
                    beta,
                    lims,
                    mask_start,
                    qfs,
                    s_curs,
                    o_accs,
                    l_accs,
                    row_maxs,
                    row_runs,
                )
        elif const_expr(self.pipelined):
            s_next = cute.make_rmem_tensor(4 * self.nt, Float32)
            self._first_wait(n_blocks)
            self._qk(Int32(0), sbase, lane, qfs[0], s_curs[0])
            self._mask(Int32(0), t, lims[0][0], lims[0][1], mask_start, s_curs[0])
            for j in cutlass.range(n_blocks, unroll=1):
                self._tile_pipelined(
                    j,
                    n_blocks,
                    sbase,
                    gk,
                    gv,
                    tidx,
                    lane,
                    t,
                    beta,
                    lims[0][0],
                    lims[0][1],
                    v_key,
                    mask_start,
                    qfs[0],
                    s_curs[0],
                    s_next,
                    o_accs[0],
                    l_accs[0],
                    row_maxs[0],
                    row_runs[0],
                )
        else:
            for j in cutlass.range(n_blocks, unroll=1):
                self._tile(
                    j,
                    n_blocks,
                    sbase,
                    gk,
                    gv,
                    tidx,
                    lane,
                    t,
                    beta,
                    lims[0][0],
                    lims[0][1],
                    v_key,
                    mask_start,
                    qfs[0],
                    s_curs[0],
                    o_accs[0],
                    l_accs[0],
                    row_maxs[0],
                    row_runs[0],
                )

        # normalize and write O, the base-2 LSE and the max probability
        go = Int64(mO.iterator.toint()) + (Int64(b) * osb + Int64(hq) * osh) * 2
        for m in cutlass.range_constexpr(mw):
            o_acc = o_accs[m]
            l_acc = l_accs[m]
            row_max = row_maxs[m]
            row_run = row_runs[m]
            q_idx0 = m_block * self.BM + wrow + 16 * m + g
            o_row0 = q_row0 + wrow + 16 * m + g
            if const_expr(self.bf16 and not self.BF16_ROWSUM_MMA):
                for r in cutlass.range_constexpr(2):
                    s = l_acc[2 * r]
                    s = s + cute.arch.shuffle_sync_bfly(s, offset=1)
                    l_acc[2 * r] = s + cute.arch.shuffle_sync_bfly(s, offset=2)
            for r in cutlass.range_constexpr(2):
                lsum = l_acc[2 * r]
                valid = lsum > Float32(0.0)
                inv = sv / lsum if valid else Float32(0.0)
                lse = (
                    row_max[r]
                    + cute.math.log2(lsum, fastmath=True)
                    - Float32(self.p_off)
                    if valid
                    else Float32(math.inf)
                )
                pmax = (
                    cute.math.exp2(row_run[r] - lse, fastmath=True)
                    if valid
                    else Float32(0.0)
                )
                orow = o_row0 + 8 * r
                if q_idx0 + 8 * r < seqlen_q:  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                    if const_expr(self.return_lse):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                        if t == 0:
                            if const_expr(self.lse_natural):
                                mLSE[orow] = lse * Float32(math.log(2.0))
                            else:
                                mLSE[orow] = lse
                            mPmax[orow] = pmax
                # every lane runs the store's shuffles; only rows inside `seqlen_q` write
                self._store_o(
                    go,
                    q_idx0 + 8 * r,
                    t,
                    oss,
                    o_acc,
                    r,
                    inv,
                    q_idx0 + 8 * r < seqlen_q,
                )
