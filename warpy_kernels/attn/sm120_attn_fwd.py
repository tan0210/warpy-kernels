# type: ignore

import math

import cuda.bindings.driver as cuda
import cutlass
import torch
import torch.nn.functional as F
from cutlass import Float32, Int32, Int64, const_expr, cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from ..sm120.ptx import (
    SMEM_LIMIT,
    cp_async16,
    ldsm_x2_trans_b8,
    ldsm_x4,
    mma_fp8,
    pack_e4m3x4,
    pack_half2,
    st_global_b32,
    swizzle_offset,
    warp_sync,
)
from ..utils import (
    autotune,
    check_descale,
    compile_cached,
    flat,
    round_up,
    tune_key,
    tune_lookup,
)
from .constants import E4M3_ONE_X4, LOG2E, P_OFFSET


class AttnFwdSm120:
    BM = 128
    NUM_THREADS = 256
    PRODUCER_REGS = 24  # 8 * 240 + 4 * 24 fills 12 * 168
    CONSUMER_REGS = 240

    def __init__(
        self,
        head_dim: int,
        causal: bool,
        num_stages: int = 4,
        pipelined: bool = True,
        block_n: int = 64,
        tile_local: bool = True,
        lpt: bool = False,
        tma: bool = False,
        return_lse: bool = True,
        has_seqused: bool = False,
        lse_natural: bool = False,
        warp_spec: bool = False,
    ):
        assert head_dim in (64, 128) and block_n in (64, 128)
        assert num_stages >= (3 if pipelined else 2)
        self.d = head_dim
        self.causal = causal
        self.stages = num_stages
        self.pipelined = pipelined
        self.BN = block_n
        self.nt = block_n // 8
        self.tile_local = tile_local
        self.lpt = lpt
        self.tma = tma
        assert tma or not warp_spec
        self.warp_spec = warp_spec
        self.threads = self.NUM_THREADS + (128 if warp_spec else 0)
        self.return_lse = return_lse
        self.has_seqused = has_seqused
        self.lse_natural = lse_natural
        self.sw_bits = 3 if head_dim == 128 else 2
        self.q_bytes = self.BM * head_dim
        self.kv_tile_bytes = self.BN * head_dim
        self.k_off = self.q_bytes
        self.v_off = self.q_bytes + num_stages * self.kv_tile_bytes
        self.smem_bytes = self.q_bytes + 2 * num_stages * self.kv_tile_bytes
        assert self.smem_bytes + (1024 if tma else 0) <= SMEM_LIMIT, (
            f"shared memory does not fit: {self.smem_bytes} B > {SMEM_LIMIT} B "
            f"(d={head_dim}, block_n={block_n}, num_stages={num_stages})"
        )
        nbytes = self.smem_bytes
        if tma:
            n_stage, tile_b, q_b = num_stages, self.kv_tile_bytes, self.q_bytes

            @cute.struct
            class SharedStorage:
                # full barriers, then empty barriers
                mbar: cute.struct.MemRange[cutlass.Int64, 2 * n_stage]
                # Q, K, V stages, 1024-byte aligned for the swizzle
                q: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, q_b], 1024]
                k: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, n_stage * tile_b], 1024
                ]
                v: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, n_stage * tile_b], 1024
                ]
        else:

            @cute.struct
            class SharedStorage:
                buf: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, nbytes], 1024
                ]

        self.shared_storage = SharedStorage

    def _swz_off(self, row, chunk):
        return swizzle_offset(self.d, row, chunk)

    def _load_rows(self, s_addr, g_addr, nrows, tidx, row_stride):
        chunks = self.d // 16
        for i in range(nrows * chunks // self.NUM_THREADS):
            idx = tidx + i * self.NUM_THREADS
            row = idx // chunks
            ch = idx % chunks
            cp_async16(
                s_addr + self._swz_off(row, ch),
                g_addr + Int64(row) * row_stride + Int64(ch * 16),
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
        if jn < n_blocks:
            sn = jn % self.stages
            self._load_rows(
                sbase + self.k_off + sn * self.kv_tile_bytes,
                gk + Int64(jn) * Int64(self.BN) * self._kss,
                self.BN,
                tidx,
                self._kss,
            )
            self._load_rows(
                sbase + self.v_off + sn * self.kv_tile_bytes,
                gv + Int64(jn) * Int64(self.BN) * self._vss,
                self.BN,
                tidx,
                self._vss,
            )

    @cute.jit
    def _stage_layout(self):
        sw = cute.make_swizzle(self.sw_bits, 4, 3)  # 128B for d=128, 64B for d=64
        return cute.make_composed_layout(
            sw,
            0,
            cute.make_layout(
                (self.BN, self.d, self.stages), stride=(self.d, 1, self.kv_tile_bytes)
            ),
        )

    @cute.jit
    def _tma_load(self, jn: Int32, n_blocks: Int32):
        if jn < n_blocks:
            s = jn % self.stages
            lane0 = (cute.arch.thread_idx()[0] % 32) == 0
            if lane0:  # lane 0 waits for the stage and arms the barrier
                cute.arch.mbarrier_wait(self._empty + s, ((jn // self.stages) & 1) ^ 1)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    self._full + s, 2 * self.kv_tile_bytes
                )
            warp_sync()  # TMA copies need the full warp
            cute.copy(
                self._atom_k,
                self._tKgK[(None, self._tile0 + jn)],
                self._tKsK[(None, s)],
                tma_bar_ptr=self._full + s,
            )
            cute.copy(
                self._atom_v,
                self._tVgV[(None, self._tile0 + jn)],
                self._tVsV[(None, s)],
                tma_bar_ptr=self._full + s,
            )

    @cute.jit
    def _tma_release(self, jt: Int32, lane: Int32):
        warp_sync()
        if lane == 0:
            cute.arch.mbarrier_arrive(self._empty + jt % self.stages)  # stage is free

    @cute.jit
    def _qk(
        self, jt: Int32, sbase: Int32, lane: Int32, qf: cute.Tensor, acc: cute.Tensor
    ):
        kbase = sbase + self.k_off + (jt % self.stages) * self.kv_tile_bytes
        acc.fill(0.0)
        for kk in cutlass.range_constexpr(self.d // 32):
            for npair in cutlass.range_constexpr(self.nt // 2):
                key = (lane % 8) + (lane // 16) * 8 + 16 * npair
                ch = (lane // 8) % 2 + 2 * kk
                bfr = ldsm_x4(kbase + self._swz_off(key, ch))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_fp8(
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
    def _mask(
        self,
        jt: Int32,
        t: Int32,
        lim0: Int32,
        lim1: Int32,
        mask_start: Int32,
        acc: cute.Tensor,
    ):
        # tiles ending before the first masked key need no mask
        neg_inf = Float32(-math.inf)
        if jt * self.BN + (self.BN - 1) >= mask_start:
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
            bias = Float32(P_OFFSET) - m_safe  # base-2 exponent plus the `2^8` offset
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
        vbase = sbase + self.v_off + (jt % self.stages) * self.kv_tile_bytes
        ones = Int32(E4M3_ONE_X4)
        for kk2 in cutlass.range_constexpr(self.BN // 32):
            t0 = 4 * kk2
            # P packed from the S accumulators
            pa = [
                pack_e4m3x4(
                    acc[4 * t0], acc[4 * t0 + 1], acc[4 * t0 + 4], acc[4 * t0 + 5]
                ),
                pack_e4m3x4(
                    acc[4 * t0 + 2], acc[4 * t0 + 3], acc[4 * t0 + 6], acc[4 * t0 + 7]
                ),
                pack_e4m3x4(
                    acc[4 * t0 + 8], acc[4 * t0 + 9], acc[4 * t0 + 12], acc[4 * t0 + 13]
                ),
                pack_e4m3x4(
                    acc[4 * t0 + 10],
                    acc[4 * t0 + 11],
                    acc[4 * t0 + 14],
                    acc[4 * t0 + 15],
                ),
            ]
            c = mma_fp8(pa, [ones, ones], [l_acc[0], l_acc[1], l_acc[2], l_acc[3]])
            for x in cutlass.range_constexpr(4):
                l_acc[x] = c[x]
            key = v_key + 32 * kk2
            for npair in cutlass.range_constexpr(self.d // 16):
                # V via transposing 8-bit `ldmatrix`
                vb = ldsm_x2_trans_b8(vbase + self._swz_off(key, npair))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_fp8(
                        pa,
                        [vb[h], vb[2 + h]],
                        [
                            o_acc[4 * nt],
                            o_acc[4 * nt + 1],
                            o_acc[4 * nt + 2],
                            o_acc[4 * nt + 3],
                        ],
                    )
                    for x in cutlass.range_constexpr(4):
                        o_acc[4 * nt + x] = c[x]

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
        if const_expr(self.tma):
            if const_expr(not self.warp_spec):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                if tidx < 32:
                    self._tma_load(j + (self.stages - 1), n_blocks)
            cute.arch.mbarrier_wait(
                self._full + j % self.stages, (j // self.stages) & 1
            )
        else:
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            self._load_tile(j + (self.stages - 1), n_blocks, sbase, gk, gv, tidx)
            cute.arch.cp_async_commit_group()
        self._qk(j, sbase, lane, qf, s_acc)
        self._mask(j, t, lim0, lim1, mask_start, s_acc)
        self._softmax(beta, s_acc, o_acc, l_acc, row_max, row_run)
        self._pv(j, sbase, v_key, s_acc, o_acc, l_acc)
        if const_expr(self.tma):
            self._tma_release(j, lane)

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
        if const_expr(self.tma):
            if const_expr(not self.warp_spec):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                if tidx < 32:
                    self._tma_load(j + (self.stages - 1), n_blocks)
            if j + 1 < n_blocks:  # tile j+1 must be loaded
                cute.arch.mbarrier_wait(
                    self._full + (j + 1) % self.stages, ((j + 1) // self.stages) & 1
                )
        else:
            cute.arch.cp_async_wait_group(self.stages - 3)  # tile j+1 must be loaded
            cute.arch.sync_threads()
            self._load_tile(j + (self.stages - 1), n_blocks, sbase, gk, gv, tidx)
            cute.arch.cp_async_commit_group()
        # overlaps tile j's softmax; the last read is unused
        self._qk(j + 1, sbase, lane, qf, s_next)
        self._softmax(beta, s_cur, o_acc, l_acc, row_max, row_run)
        self._pv(j, sbase, v_key, s_cur, o_acc, l_acc)
        if const_expr(self.tma):
            self._tma_release(j, lane)
        self._mask(j + 1, t, lim0, lim1, mask_start, s_next)
        for x in cutlass.range_constexpr(4 * self.nt):
            s_cur[x] = s_next[x]

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
        if const_expr(self.tma):
            one_stage = cute.slice_(self._stage_layout(), (None, None, 0))
            # view [B, H, N, d] as (N, d, H, B)
            mK_t = cute.make_tensor(
                mK.iterator, cute.select(mK.layout, mode=[2, 3, 1, 0])
            )
            mV_t = cute.make_tensor(
                mV.iterator, cute.select(mV.layout, mode=[2, 3, 1, 0])
            )
            atom_k, tk = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mK_t, one_stage, (self.BN, self.d)
            )
            atom_v, tv = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mV_t, one_stage, (self.BN, self.d)
            )
        else:
            atom_k = None
            tk = None
            atom_v = None
            tv = None
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
            atom_k,
            tk,
            atom_v,
            tv,
        ).launch(
            # longest causal CTAs launch first
            grid=[h_q * batch, nq_pad // self.BM, 1]
            if self.lpt
            else [nq_pad // self.BM, h_q, batch],
            block=[self.threads, 1, 1],
            min_blocks_per_mp=1 if self.warp_spec else 0,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

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
        atom_k,
        tk,
        atom_v,
        tv,
    ):
        d = self.d
        kq_steps = d // 32
        o_tiles = d // 8

        tidx, _, _ = cute.arch.thread_idx()
        if const_expr(self.lpt):
            lin, mb, _ = cute.arch.block_idx()
            hq = lin % h_q
            b = lin // h_q
            m_block = nq_pad // self.BM - 1 - mb
        else:
            m_block, hq, b = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        g = lane // 4
        t = lane % 4

        hkv = hq // (h_q // h_kv)
        beta = softmax_scale * mDq[b, hkv] * mDk[b, hkv] * LOG2E
        sv = mDv[b, hkv]

        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        if const_expr(self.tma):
            # Q, K, V stages are consecutive, matching the `cp.async` offsets
            sbase = Int32(
                storage.q.get_tensor(cute.make_layout(self.q_bytes)).iterator.toint()
            )
            stage_layout = self._stage_layout()
            sK = storage.k.get_tensor(stage_layout.outer, swizzle=stage_layout.inner)
            sV = storage.v.get_tensor(stage_layout.outer, swizzle=stage_layout.inner)
            self._full = storage.mbar.data_ptr()
            self._empty = self._full + self.stages
            self._atom_k = atom_k
            self._atom_v = atom_v
            if tidx == 0:
                for s in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(self._full + s, 1)
                    cute.arch.mbarrier_init(self._empty + s, self.NUM_THREADS // 32)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            self._tile0 = Int32(0)
            gKt = cute.local_tile(tk[None, None, hkv, b], (self.BN, d), (None, 0))
            gVt = cute.local_tile(tv[None, None, hkv, b], (self.BN, d), (None, 0))
            self._tKsK, self._tKgK = cpasync.tma_partition(
                atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 2),
                cute.group_modes(gKt, 0, 2),
            )
            self._tVsV, self._tVgV = cpasync.tma_partition(
                atom_v,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gVt, 0, 2),
            )
        else:
            sbase = Int32(
                storage.buf.get_tensor(
                    cute.make_layout(self.smem_bytes)
                ).iterator.toint()
            )

        q_row0 = (b * h_q + hq) * nq_pad + m_block * self.BM
        gq = (
            Int64(mQ.iterator.toint())
            + Int64(b) * qsb
            + Int64(hq) * qsh
            + Int64(m_block * self.BM) * qss
        )
        gk = Int64(mK.iterator.toint()) + Int64(b) * ksb + Int64(hkv) * ksh
        gv = Int64(mV.iterator.toint()) + Int64(b) * vsb + Int64(hkv) * vsh
        self._kss = kss
        self._vss = vss

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

        if warp >= self.NUM_THREADS // 32:  # producer warp group
            if const_expr(self.warp_spec):
                cute.arch.setmaxregister_decrease(self.PRODUCER_REGS)
                if warp == self.NUM_THREADS // 32:
                    for j in cutlass.range(n_blocks, unroll=1):
                        self._tma_load(j, n_blocks)
        else:
            if const_expr(self.warp_spec):
                cute.arch.setmaxregister_increase(self.CONSUMER_REGS)
            # Q first, then the first K/V stages
            self._load_rows(sbase, gq, self.BM, tidx, qss)
            cute.arch.cp_async_commit_group()
            if const_expr(self.tma):
                if const_expr(not self.warp_spec):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                    if tidx < 32:
                        for s in cutlass.range_constexpr(self.stages - 1):
                            self._tma_load(Int32(s), n_blocks)
                cute.arch.cp_async_wait_group(0)
            else:
                for s in cutlass.range_constexpr(self.stages - 1):
                    self._load_tile(Int32(s), n_blocks, sbase, gk, gv, tidx)
                    cute.arch.cp_async_commit_group()
                # Q and tile 0 resident
                cute.arch.cp_async_wait_group(self.stages - 2)
            # consumers only
            cute.arch.barrier(barrier_id=1, number_of_threads=self.NUM_THREADS)
            qf = cute.make_rmem_tensor(4 * kq_steps, Int32)
            for kq in cutlass.range_constexpr(kq_steps):
                qrow = (lane % 8) + ((lane // 8) % 2) * 8 + warp * 16
                qch = lane // 16 + 2 * kq
                qfrag = ldsm_x4(sbase + self._swz_off(qrow, qch))
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
            s_cur = cute.make_rmem_tensor(4 * self.nt, Float32)

            q_idx0 = m_block * self.BM + warp * 16 + g
            lim0 = sk
            lim1 = sk
            mask_start = sk
            if const_expr(self.causal):
                lr0 = q_idx0 + (sk - seqlen_q + 1)
                lr1 = lr0 + 8
                lim0 = min(sk, lr0)
                lim1 = min(sk, lr1)
                ms = m_block * self.BM + (sk - seqlen_q + 1)
                mask_start = min(sk, ms)

            li = lane % 16
            v_key = ((li % 4) // 2) * 8 + (li // 4) * 2 + (li % 2) + (lane // 16) * 16

            if const_expr(self.pipelined):
                s_next = cute.make_rmem_tensor(4 * self.nt, Float32)
                if const_expr(self.tma):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                    if n_blocks > 0:
                        cute.arch.mbarrier_wait(self._full, 0)
                self._qk(Int32(0), sbase, lane, qf, s_cur)
                self._mask(Int32(0), t, lim0, lim1, mask_start, s_cur)
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
                        lim0,
                        lim1,
                        v_key,
                        mask_start,
                        qf,
                        s_cur,
                        s_next,
                        o_acc,
                        l_acc,
                        row_max,
                        row_run,
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
                        lim0,
                        lim1,
                        v_key,
                        mask_start,
                        qf,
                        s_cur,
                        o_acc,
                        l_acc,
                        row_max,
                        row_run,
                    )

            # normalize and write O, the base-2 LSE and the max probability
            go = Int64(mO.iterator.toint()) + (Int64(b) * osb + Int64(hq) * osh) * 2
            o_row0 = q_row0 + warp * 16 + g
            for r in cutlass.range_constexpr(2):
                lsum = l_acc[2 * r]
                valid = lsum > Float32(0.0)
                inv = sv / lsum if valid else Float32(0.0)
                lse = (
                    row_max[r] + cute.math.log2(lsum, fastmath=True) - Float32(P_OFFSET)
                    if valid
                    else Float32(math.inf)
                )
                pmax = (
                    cute.math.exp2(row_run[r] - lse, fastmath=True)
                    if valid
                    else Float32(0.0)
                )
                orow = o_row0 + 8 * r
                if q_idx0 + 8 * r < seqlen_q:
                    if const_expr(self.return_lse):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                        if t == 0:
                            if const_expr(self.lse_natural):
                                mLSE[orow] = lse * Float32(math.log(2.0))
                            else:
                                mLSE[orow] = lse
                            mPmax[orow] = pmax
                    for nt in cutlass.range_constexpr(o_tiles):
                        addr = (
                            go
                            + (Int64(q_idx0 + 8 * r) * oss + Int64(t * 2 + 8 * nt)) * 2
                        )
                        st_global_b32(
                            addr,
                            pack_half2(
                                o_acc[4 * nt + 2 * r] * inv,
                                o_acc[4 * nt + 2 * r + 1] * inv,
                                "bf16",
                            ),
                        )


FWD_CANDIDATES = {
    128: [
        {"pipelined": False, "num_stages": 2, "block_n": 128},
        {"pipelined": True, "num_stages": 4, "block_n": 64},
        {"pipelined": False, "num_stages": 3, "block_n": 64},
    ],
    64: [
        {"pipelined": True, "num_stages": 4, "block_n": 64},
        {"pipelined": False, "num_stages": 3, "block_n": 128},
        {"pipelined": True, "num_stages": 3, "block_n": 128},
        {"pipelined": False, "num_stages": 2, "block_n": 64},
    ],
}
FWD_DEFAULTS = {
    128: {"pipelined": False, "num_stages": 2, "block_n": 128, "tma": True},
    64: {"pipelined": True, "num_stages": 3, "block_n": 128, "tma": True},
}


def _kernel_view(x, n_pad):
    xb = x.view(torch.uint8)
    n = x.shape[2]
    direct = (
        n == n_pad
        and x.stride(3) == 1
        and x.data_ptr() % 16 == 0
        and all(st % 16 == 0 for st in x.stride()[:3])
    )
    return xb if direct else F.pad(xb, (0, 0, 0, n_pad - n)).contiguous()


def make_attn_fwd(
    q,
    k,
    v,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    num_stages=None,
    pipelined=None,
    block_n=None,
    tile_local=True,
    lpt=None,
    tma=None,
    out=None,
    return_lse=True,
    seqused_k=None,
    lse_natural=False,
    warp_spec=None,
):
    """Prepare an attention forward launch on SM120.

    ``launch()`` may not see later in-place changes to the inputs. Call ``make_attn_fwd`` again for new values.

    Args:
        q: ``[B, Hq, Nq, d]`` ``torch.float8_e4m3fn``. ``d`` is 64 or 128. Any strides work if the last one is 1,
            such as a transposed ``[B, N, H, d]`` tensor. Inputs are copied if a stride or the base address is
            not a multiple of 16 bytes, or if ``Nq`` is not a multiple of 128.
        k: ``[B, Hkv, Nk, d]`` ``torch.float8_e4m3fn``, with the same stride rules. ``Hq`` must be a multiple of
            ``Hkv``. ``Nk`` must be a multiple of ``block_n`` unless ``tma`` is on, or it is copied.
        v: Same shape and dtype as ``k``.
        q_descale: ``[B, Hkv]`` float32 dequantization scale, or ``None`` for 1.
        k_descale: Same as ``q_descale``, for ``k``.
        v_descale: Same as ``q_descale``, for ``v``.
        causal: Whether to apply a bottom-right aligned causal mask.
        softmax_scale: Scale on the logits. Defaults to ``1 / sqrt(d)``.
        num_stages: Depth of the K/V shared-memory pipeline, at least 3 when ``pipelined``. ``None`` takes the
            autotuned or built-in choice. See ``autotune_attn_fwd``.
        pipelined: Whether to issue the next tile's QK^T before the current tile's softmax. ``None`` takes the
            autotuned or built-in choice.
        block_n: Keys per tile, 64 or 128. ``None`` takes the autotuned or built-in choice.
        tile_local: Whether to quantize P against each tile's own row max, which keeps long-context tails
            accurate.
        lpt: Whether to launch the longest causal CTAs first. ``None`` enables it for causal attention only.
        tma: Whether to load K and V with TMA instead of ``cp.async``. Q always uses ``cp.async``. ``None``
            takes the autotuned or built-in choice.
        out: Output buffer, ``[B, Hq, Nq, d]`` bf16 with last stride 1 and even other strides. ``None`` allocates
            one with the memory layout of ``q``.
        return_lse: Whether to compute ``lse`` and ``pmax``. The backward pass needs them.
        seqused_k: ``[B]`` int32 number of keys each batch item attends to, at most ``Nk``. ``None`` attends to all
            ``Nk``. Causal masking is aligned to this length. The backward pass does not support it.
        lse_natural: Whether ``lse`` is the natural-log log-sum-exp instead of base 2. The backward pass needs
            base 2.
        warp_spec: Whether a dedicated producer warp group issues the K/V TMA loads. Requires ``tma``. ``None``
            takes the autotuned choice, which is off by default.

    Returns:
        ``(launch, out, lse, pmax)``. Calling ``launch()`` runs the kernel and fills the outputs. ``out`` is
        ``[B, Hq, Nq, d]`` bf16, ``lse`` is the ``[B, Hq, Nq]`` base-2 log-sum-exp of the scaled logits
        (``+inf`` for rows with no visible key), and ``pmax`` is the ``[B, Hq, Nq]`` largest probability per row.
        ``lse`` and ``pmax`` are ``None`` if ``return_lse`` is false.

    Raises:
        ValueError: If ``q``, ``k`` or ``v`` has the wrong dtype or shape, a descale is not ``[B, Hkv]``, ``out``
            has the wrong shape, dtype or strides, or ``seqused_k`` is not an int32 CUDA tensor of shape ``[B]``.
    """
    B, Hq, Nq, d = q.shape
    Hkv, Nk = k.shape[1], k.shape[2]
    if not (q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn):
        raise ValueError("q, k and v must be torch.float8_e4m3fn")
    if (
        k.dim() != 4
        or v.shape != k.shape
        or k.shape[0] != B
        or k.shape[3] != d
        or d not in (64, 128)
        or Hq % Hkv
    ):
        raise ValueError(
            "need q [B, Hq, Nq, d] and k, v [B, Hkv, Nk, d] with d 64 or 128 and Hq a multiple of Hkv, "
            f"got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    for x, name in (
        (q_descale, "q_descale"),
        (k_descale, "k_descale"),
        (v_descale, "v_descale"),
    ):
        check_descale(x, B, Hkv, name)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    if (
        num_stages is None
        or pipelined is None
        or block_n is None
        or tma is None
        or warp_spec is None
    ):
        tuned = (
            tune_lookup(tune_key("fwd", d, causal, Nq, Nk, Hq // Hkv))
            or FWD_DEFAULTS[d]
        )
        num_stages = tuned["num_stages"] if num_stages is None else num_stages
        pipelined = tuned["pipelined"] if pipelined is None else pipelined
        block_n = tuned["block_n"] if block_n is None else block_n
        tma = bool(tuned.get("tma", False)) if tma is None else tma
        warp_spec = (
            bool(tuned.get("warp_spec", False)) if warp_spec is None else warp_spec
        )
    warp_spec = bool(warp_spec) and bool(tma)
    if lpt is None:
        lpt = causal
    dev = q.device
    ones = torch.ones(B, Hkv, dtype=torch.float32, device=dev)
    dq_, dk_, dv_ = [
        (x if x is not None else ones).float().contiguous()
        for x in (q_descale, k_descale, v_descale)
    ]

    nq_pad = round_up(Nq, AttnFwdSm120.BM)
    nk_pad = round_up(Nk, block_n)
    qv = _kernel_view(q, nq_pad)
    kv = _kernel_view(k, Nk if tma else nk_pad)  # TMA zero-fills the rows past `Nk`
    vv = _kernel_view(v, Nk if tma else nk_pad)
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    elif (
        out.shape != q.shape
        or out.dtype != torch.bfloat16
        or out.device != dev
        or out.stride(3) != 1
        or any(st % 2 for st in out.stride()[:3])
        or out.data_ptr() % 4
    ):
        raise ValueError(
            "out must be a bf16 tensor shaped like q, with last stride 1 and even other strides"
        )
    if seqused_k is not None and (
        seqused_k.shape != (B,)
        or seqused_k.dtype != torch.int32
        or seqused_k.device != dev
    ):
        raise ValueError("seqused_k must be an int32 CUDA tensor of shape [B]")
    seq = (
        seqused_k.contiguous()
        if seqused_k is not None
        else torch.zeros(1, dtype=torch.int32, device=dev)
    )
    if return_lse:
        lse = torch.empty(B, Hq, nq_pad, dtype=torch.float32, device=dev)
        pmax = torch.empty(B, Hq, nq_pad, dtype=torch.float32, device=dev)
    else:
        lse = pmax = torch.empty(1, dtype=torch.float32, device=dev)

    def dyn(x):
        return from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=3)

    targs = [dyn(qv), dyn(kv), dyn(vv), dyn(out), flat(lse), flat(pmax)]
    targs += [from_dlpack(x) for x in (dq_, dk_, dv_, seq)]
    strides = [Int64(st) for x in (qv, kv, vv, out) for st in x.stride()[:3]]
    sargs = [
        Float32(softmax_scale),
        Int32(Nq),
        Int32(Nk),
        Int32(nq_pad),
        Int32(nk_pad),
        Int32(Hq),
        Int32(Hkv),
        Int32(B),
    ] + strides

    ctor = (
        d,
        causal,
        num_stages,
        pipelined,
        block_n,
        tile_local,
        lpt,
        tma,
        return_lse,
        seqused_k is not None,
        lse_natural,
        warp_spec,
    )
    compiled = compile_cached(
        AttnFwdSm120, ctor, (B, Hq, Hkv, nq_pad, nk_pad), *targs, *sargs
    )

    def launch():
        compiled(*targs, *sargs)

    if not return_lse:
        return launch, out, None, None
    return launch, out, lse[:, :, :Nq], pmax[:, :, :Nq]


def attn_fwd(
    q,
    k,
    v,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    num_stages=None,
    pipelined=None,
    block_n=None,
    tile_local=True,
    lpt=None,
    tma=None,
    out=None,
    return_lse=True,
    seqused_k=None,
    lse_natural=False,
    warp_spec=None,
):
    """Run the attention forward once.

    Args:
        q: As in ``make_attn_fwd``.
        k: As in ``make_attn_fwd``.
        v: As in ``make_attn_fwd``.
        q_descale: As in ``make_attn_fwd``.
        k_descale: As in ``make_attn_fwd``.
        v_descale: As in ``make_attn_fwd``.
        causal: As in ``make_attn_fwd``.
        softmax_scale: As in ``make_attn_fwd``.
        num_stages: As in ``make_attn_fwd``.
        pipelined: As in ``make_attn_fwd``.
        block_n: As in ``make_attn_fwd``.
        tile_local: As in ``make_attn_fwd``.
        lpt: As in ``make_attn_fwd``.
        tma: As in ``make_attn_fwd``.
        out: As in ``make_attn_fwd``.
        return_lse: As in ``make_attn_fwd``.
        seqused_k: As in ``make_attn_fwd``.
        lse_natural: As in ``make_attn_fwd``.
        warp_spec: As in ``make_attn_fwd``.

    Returns:
        ``(out, lse, pmax)``, as returned by ``make_attn_fwd`` after the launch.

    Raises:
        ValueError: If ``out`` or ``seqused_k`` is invalid. See ``make_attn_fwd``.
    """
    launch, out, lse, pmax = make_attn_fwd(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        causal,
        softmax_scale,
        num_stages,
        pipelined,
        block_n,
        tile_local,
        lpt,
        tma,
        out,
        return_lse,
        seqused_k,
        lse_natural,
        warp_spec,
    )
    launch()
    return out, lse, pmax


def autotune_attn_fwd(
    q,
    k,
    v,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    candidates=None,
    iters=10,
    verbose=True,
):
    """Time candidate forward configs for this shape and cache the fastest for later calls.

    Args:
        q: As in ``make_attn_fwd``.
        k: As in ``make_attn_fwd``.
        v: As in ``make_attn_fwd``.
        q_descale: As in ``make_attn_fwd``.
        k_descale: As in ``make_attn_fwd``.
        v_descale: As in ``make_attn_fwd``.
        causal: As in ``make_attn_fwd``.
        softmax_scale: As in ``make_attn_fwd``.
        candidates: Config dicts with keys ``pipelined``, ``num_stages``, ``block_n``, ``tma`` and ``warp_spec``.
            Defaults to every config in ``FWD_CANDIDATES`` for the head dim, with ``cp.async``, TMA, and TMA
            with a producer warp group.
        iters: Timed launches per candidate.
        verbose: Whether to print one line per candidate.

    Returns:
        ``(best_config, results)``, where ``results`` is a list of ``(config, milliseconds or error string)``.

    Raises:
        RuntimeError: If no candidate runs.
    """
    d = q.shape[-1]
    if candidates is None:
        candidates = (
            FWD_CANDIDATES[d]
            + [dict(c, tma=True) for c in FWD_CANDIDATES[d]]
            + [dict(c, tma=True, warp_spec=True) for c in FWD_CANDIDATES[d]]
        )

    def build(cfg):
        return make_attn_fwd(
            q, k, v, q_descale, k_descale, v_descale, causal, softmax_scale, **cfg
        )[0]

    return autotune(
        "fwd",
        d,
        causal,
        q.shape[2],
        k.shape[2],
        q.shape[1] // k.shape[1],
        candidates,
        build,
        iters,
        verbose,
    )
