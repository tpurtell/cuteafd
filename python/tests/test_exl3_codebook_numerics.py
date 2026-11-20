"""Opt-in exact-AOT codebook GPU qualification; no export, subprocess, or service startup.

Point CUTEAFD_EXL3_MCG_AOT and CUTEAFD_EXL3_MUL1_AOT at matching path-separated
already-built EXL3 variant directories (e.g. m1:m80 of the same layout).
Run: python3 -m unittest discover -s python/tests -p test_exl3_codebook_numerics.py -v
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import unittest


_REF_GEMM_CACHE: dict[tuple[int, int, int, int], object] = {}
_REF_HAD_CACHE: dict[int, object] = {}
_REF_ACT_CACHE: dict[tuple[bool, int, int], object] = {}
_REF_FC2_ROT_CACHE: dict[tuple[int, int], object] = {}


def _decode_gptqmodel_trellis_f16(raw_trellis, bits: int, codebook: str):
    """Vectorized GPU reconstruction of GPTQModel [k16, n16, 16*bits] int16 tiles to [K, N] FP16."""
    import torch

    device = raw_trellis.device
    k16, n16, packed_words = raw_trellis.shape
    assert packed_words == 16 * bits
    num_tiles = k16 * n16
    words = raw_trellis.contiguous().view(torch.uint16).reshape(num_tiles, 16 * bits).to(torch.int32) & 0xFFFF
    swapped = words.reshape(num_tiles, 8 * bits, 2).flip(-1).reshape(num_tiles, 16 * bits)
    bit_shifts = torch.arange(15, -1, -1, device=device, dtype=torch.int32)
    bit_plane = ((swapped.unsqueeze(-1) >> bit_shifts) & 1).reshape(num_tiles, 256 * bits)

    ends = torch.arange(1, 257, device=device, dtype=torch.long) * bits - 1
    shifts = torch.arange(16, device=device, dtype=torch.long)
    idx = (ends[:, None] - shifts[None, :]) % (256 * bits)
    states = ((bit_plane[:, idx].to(torch.int64) << shifts[None, None, :]).sum(dim=-1) & 0xFFFF).to(torch.int64)

    if codebook == "mcg":
        x = ((states * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60
        low = (x & 0xFFFF).to(torch.uint16).view(torch.float16)
        high = ((x >> 16) & 0xFFFF).to(torch.uint16).view(torch.float16)
        vals = low + high
    elif codebook == "mul1":
        prod = (states * 0x83DCD12D) & 0xFFFFFFFF
        bsum = (prod & 0xFF) + ((prod >> 8) & 0xFF) + ((prod >> 16) & 0xFF) + ((prod >> 24) & 0xFF)
        h = (0x6400 + bsum).to(torch.uint16).view(torch.float16)
        kinv = torch.tensor([0x1EEE], dtype=torch.uint16, device=device).view(torch.float16)
        kbias = torch.tensor([0xC931], dtype=torch.uint16, device=device).view(torch.float16)
        vals = torch.addcmul(kbias.to(torch.float64), h.to(torch.float64), kinv.to(torch.float64)).to(torch.float16)
    else:
        raise ValueError(f"unsupported codebook {codebook!r}")

    row_idx = []
    col_idx = []
    for t in range(32):
        r, c = 2 * (t % 4), t // 4
        for rr, cc in ((r, c), (r + 1, c), (r + 8, c), (r + 9, c),
                       (r, c + 8), (r + 1, c + 8), (r + 8, c + 8), (r + 9, c + 8)):
            row_idx.append(rr)
            col_idx.append(cc)
    tiles = torch.empty((num_tiles, 16, 16), dtype=torch.float16, device=device)
    tiles[:, torch.tensor(row_idx, device=device), torch.tensor(col_idx, device=device)] = vals
    return tiles.reshape(k16, n16, 16, 16).permute(0, 2, 1, 3).reshape(k16 * 16, n16 * 16).contiguous()


def _bitwise_reference_moe(
    x,
    topk_ids,
    topk_weights,
    gate_w,
    up_w,
    down_w,
    gate_suh,
    up_suh,
    inter_rot,
    down_svh,
    *,
    swiglu_limit,
    output_dtype,
    tile_k: int,
):
    """Independent CuTe m16n8k16 + H128 bitwise reference (uses zero b12x kernel modules)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import Float16, Float32, Int32, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T
    import torch

    @cute.jit
    def _pack_half2(src: cute.Tensor, i0: Int32, i1: Int32) -> Uint32:
        a = src[i0].to(Float32)
        b = src[i1].to(Float32)
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Float32(a).ir_value(), Float32(b).ir_value()],
                "cvt.rn.f16x2.f32 $0, $2, $1;",
                "=r,f,f",
                has_side_effects=False,
                is_align_stack=False,
            )
        )

    @cute.jit
    def _mma_m16n8k16(d0: Float32, d1: Float32, d2: Float32, d3: Float32,
                      a0: Uint32, a1: Uint32, a2: Uint32, a3: Uint32,
                      b0: Uint32, b1: Uint32):
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
            [
                Uint32(a0).ir_value(), Uint32(a1).ir_value(), Uint32(a2).ir_value(), Uint32(a3).ir_value(),
                Uint32(b0).ir_value(), Uint32(b1).ir_value(),
                Float32(d0).ir_value(), Float32(d1).ir_value(), Float32(d2).ir_value(), Float32(d3).ir_value(),
            ],
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), res, [0])),
            Float32(llvm.extractvalue(T.f32(), res, [1])),
            Float32(llvm.extractvalue(T.f32(), res, [2])),
            Float32(llvm.extractvalue(T.f32(), res, [3])),
        )

    @cute.jit
    def _had128_quad(v0: Float32, v1: Float32, v2: Float32, v3: Float32, lane: Int32):
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        rs = Float32(0.088388347648)
        return h0 * rs, h1 * rs, h2 * rs, h3 * rs

    def run_had128(x_f32):
        blocks = x_f32.contiguous().reshape(-1, 128)
        nblk = int(blocks.shape[0])
        out = torch.empty_like(blocks)
        if nblk not in _REF_HAD_CACHE:
            @cute.kernel
            def _had_k(src: cute.Tensor, dst: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                blk, _, _ = cute.arch.block_idx()
                base = blk * Int32(128) + lane * Int32(4)
                h0, h1, h2, h3 = _had128_quad(
                    src[base + Int32(0)].to(Float32),
                    src[base + Int32(1)].to(Float32),
                    src[base + Int32(2)].to(Float32),
                    src[base + Int32(3)].to(Float32),
                    lane,
                )
                dst[base + Int32(0)] = h0
                dst[base + Int32(1)] = h1
                dst[base + Int32(2)] = h2
                dst[base + Int32(3)] = h3

            @cute.jit
            def _launch(src: cute.Tensor, dst: cute.Tensor, stream):
                _had_k(src, dst).launch(grid=(nblk, 1, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_HAD_CACHE[nblk] = cute.compile(_launch, from_dlpack(blocks.reshape(-1)), from_dlpack(out.reshape(-1)), st)
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_HAD_CACHE[nblk](from_dlpack(blocks.reshape(-1)), from_dlpack(out.reshape(-1)), st)
        return out.reshape(x_f32.shape)

    def run_gemm(a_f16, b_f16):
        m_rows, k_dim = a_f16.shape
        _, n_dim = b_f16.shape
        chains = tile_k // 32
        pad_m = ((max(m_rows, 1) + 7) // 8) * 8
        a_work = torch.zeros((pad_m, k_dim), dtype=torch.float16, device=a_f16.device)
        a_work[:m_rows].copy_(a_f16)
        b_work = b_f16.contiguous()
        c_work = torch.empty((pad_m, n_dim), dtype=torch.float16, device=a_f16.device)
        key = (k_dim, n_dim, chains, pad_m)
        if key not in _REF_GEMM_CACHE:
            k_iters = k_dim // tile_k

            @cute.kernel
            def _gk(a_t: cute.Tensor, b_t: cute.Tensor, c_t: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                m_blk, n_tile, _ = cute.arch.block_idx()
                r = Int32(2) * (lane % Int32(4))
                c = lane // Int32(4)
                m_row = m_blk * Int32(8) + c
                n_col0 = n_tile * Int32(16) + c
                n_col1 = n_col0 + Int32(8)
                acc = cute.make_rmem_tensor((chains, 4), Float32)
                for ch in cutlass.range_constexpr(chains):
                    for j in cutlass.range_constexpr(4):
                        acc[ch, j] = Float32(0.0)
                for kt in range(k_iters):
                    for ch in cutlass.range_constexpr(chains):
                        for sub in cutlass.range_constexpr(2):
                            kb = Int32((kt * tile_k) + (ch * 32) + (sub * 16))
                            k0 = kb + r
                            k1 = k0 + Int32(1)
                            k2 = k0 + Int32(8)
                            k3 = k0 + Int32(9)
                            a0 = _pack_half2(a_t, m_row * Int32(k_dim) + k0, m_row * Int32(k_dim) + k1)
                            a1 = _pack_half2(a_t, m_row * Int32(k_dim) + k2, m_row * Int32(k_dim) + k3)
                            o0 = _pack_half2(b_t, k0 * Int32(n_dim) + n_col0, k1 * Int32(n_dim) + n_col0)
                            o1 = _pack_half2(b_t, k2 * Int32(n_dim) + n_col0, k3 * Int32(n_dim) + n_col0)
                            o2 = _pack_half2(b_t, k0 * Int32(n_dim) + n_col1, k1 * Int32(n_dim) + n_col1)
                            o3 = _pack_half2(b_t, k2 * Int32(n_dim) + n_col1, k3 * Int32(n_dim) + n_col1)
                            d0, d1, d2, d3 = _mma_m16n8k16(
                                acc[ch, 0], acc[ch, 1], acc[ch, 2], acc[ch, 3],
                                o0, o2, o1, o3, a0, a1,
                            )
                            acc[ch, 0] = d0
                            acc[ch, 1] = d1
                            acc[ch, 2] = d2
                            acc[ch, 3] = d3
                if cutlass.const_expr(chains == 4):
                    s0 = acc[0, 0] + ((acc[1, 0] + acc[3, 0]) + acc[2, 0])
                    s1 = acc[0, 1] + ((acc[1, 1] + acc[3, 1]) + acc[2, 1])
                    s2 = acc[0, 2] + ((acc[1, 2] + acc[3, 2]) + acc[2, 2])
                    s3 = acc[0, 3] + ((acc[1, 3] + acc[3, 3]) + acc[2, 3])
                else:
                    s0 = acc[0, 0] + acc[1, 0]
                    s1 = acc[0, 1] + acc[1, 1]
                    s2 = acc[0, 2] + acc[1, 2]
                    s3 = acc[0, 3] + acc[1, 3]
                out_r0 = m_blk * Int32(8) + r
                out_r1 = out_r0 + Int32(1)
                c0 = n_tile * Int32(16) + c
                c1 = c0 + Int32(8)
                c_t[out_r0 * Int32(n_dim) + c0] = s0.to(Float16)
                c_t[out_r1 * Int32(n_dim) + c0] = s1.to(Float16)
                c_t[out_r0 * Int32(n_dim) + c1] = s2.to(Float16)
                c_t[out_r1 * Int32(n_dim) + c1] = s3.to(Float16)

            @cute.jit
            def _launch_g(a_t: cute.Tensor, b_t: cute.Tensor, c_t: cute.Tensor, stream):
                _gk(a_t, b_t, c_t).launch(grid=(pad_m // 8, n_dim // 16, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_GEMM_CACHE[key] = cute.compile(
                _launch_g, from_dlpack(a_work.reshape(-1)), from_dlpack(b_work.reshape(-1)), from_dlpack(c_work.reshape(-1)), st
            )
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_GEMM_CACHE[key](
            from_dlpack(a_work.reshape(-1)), from_dlpack(b_work.reshape(-1)), from_dlpack(c_work.reshape(-1)), st
        )
        return c_work[:m_rows]

    def run_inter_act(g_f16, u_f16, rot_row):
        rows, inter = g_f16.shape
        nblk = inter // 128
        out_f16 = torch.empty((rows, inter), dtype=torch.float16, device=g_f16.device)
        has_lim = swiglu_limit is not None
        lim_v = float(swiglu_limit or 0.0)
        key = (has_lim, rows, inter)
        if key not in _REF_ACT_CACHE:
            @cute.kernel
            def _ak(g_t: cute.Tensor, u_t: cute.Tensor, r_t: cute.Tensor, o_t: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                unit, _, _ = cute.arch.block_idx()
                row = unit // Int32(nblk)
                blk = unit % Int32(nblk)
                col0 = blk * Int32(128) + lane * Int32(4)
                base = row * Int32(inter) + col0
                gh0, gh1, gh2, gh3 = _had128_quad(
                    g_t[base + Int32(0)].to(Float32), g_t[base + Int32(1)].to(Float32),
                    g_t[base + Int32(2)].to(Float32), g_t[base + Int32(3)].to(Float32), lane
                )
                uh0, uh1, uh2, uh3 = _had128_quad(
                    u_t[base + Int32(0)].to(Float32), u_t[base + Int32(1)].to(Float32),
                    u_t[base + Int32(2)].to(Float32), u_t[base + Int32(3)].to(Float32), lane
                )
                gh = cute.make_rmem_tensor((4,), Float32)
                uh = cute.make_rmem_tensor((4,), Float32)
                gh[0], gh[1], gh[2], gh[3] = gh0, gh1, gh2, gh3
                uh[0], uh[1], uh[2], uh[3] = uh0, uh1, uh2, uh3
                out_v = cute.make_rmem_tensor((4,), Float32)
                for j in cutlass.range_constexpr(4):
                    c_idx = col0 + Int32(j)
                    ig = gh[j] * r_t[c_idx].to(Float32)
                    iu = uh[j] * r_t[Int32(inter) + c_idx].to(Float32)
                    sd = r_t[Int32(2 * inter) + c_idx].to(Float32)
                    if cutlass.const_expr(has_lim):
                        lim = Float32(lim_v)
                        nlim = Float32(-lim_v)
                        if ig > lim:
                            ig = lim
                        if iu < nlim:
                            iu = nlim
                        if iu > lim:
                            iu = lim
                    sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-ig, fastmath=True))
                    out_v[j] = ig * sig * iu * sd
                rh0, rh1, rh2, rh3 = _had128_quad(out_v[0], out_v[1], out_v[2], out_v[3], lane)
                o_t[base + Int32(0)] = rh0.to(Float16)
                o_t[base + Int32(1)] = rh1.to(Float16)
                o_t[base + Int32(2)] = rh2.to(Float16)
                o_t[base + Int32(3)] = rh3.to(Float16)

            @cute.jit
            def _launch_a(g_t: cute.Tensor, u_t: cute.Tensor, r_t: cute.Tensor, o_t: cute.Tensor, stream):
                _ak(g_t, u_t, r_t, o_t).launch(grid=(rows * nblk, 1, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_ACT_CACHE[key] = cute.compile(
                _launch_a, from_dlpack(g_f16.reshape(-1)), from_dlpack(u_f16.reshape(-1)),
                from_dlpack(rot_row.reshape(-1)), from_dlpack(out_f16.reshape(-1)), st
            )
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_ACT_CACHE[key](
            from_dlpack(g_f16.reshape(-1)), from_dlpack(u_f16.reshape(-1)),
            from_dlpack(rot_row.reshape(-1)), from_dlpack(out_f16.reshape(-1)), st
        )
        return out_f16

    def run_fc2_rot(fc2_f16, svh_row):
        rows, hid = fc2_f16.shape
        nblk = hid // 128
        out_f32 = torch.empty((rows, hid), dtype=torch.float32, device=fc2_f16.device)
        key = (rows, hid)
        if key not in _REF_FC2_ROT_CACHE:
            @cute.kernel
            def _fk(src: cute.Tensor, svh: cute.Tensor, dst: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                unit, _, _ = cute.arch.block_idx()
                row = unit // Int32(nblk)
                blk = unit % Int32(nblk)
                col0 = blk * Int32(128) + lane * Int32(4)
                base = row * Int32(hid) + col0
                h0, h1, h2, h3 = _had128_quad(
                    src[base + Int32(0)].to(Float32), src[base + Int32(1)].to(Float32),
                    src[base + Int32(2)].to(Float32), src[base + Int32(3)].to(Float32), lane
                )
                dst[base + Int32(0)] = h0 * svh[col0 + Int32(0)].to(Float32)
                dst[base + Int32(1)] = h1 * svh[col0 + Int32(1)].to(Float32)
                dst[base + Int32(2)] = h2 * svh[col0 + Int32(2)].to(Float32)
                dst[base + Int32(3)] = h3 * svh[col0 + Int32(3)].to(Float32)

            @cute.jit
            def _launch_f(src: cute.Tensor, svh: cute.Tensor, dst: cute.Tensor, stream):
                _fk(src, svh, dst).launch(grid=(rows * nblk, 1, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_FC2_ROT_CACHE[key] = cute.compile(
                _launch_f, from_dlpack(fc2_f16.reshape(-1)), from_dlpack(svh_row.reshape(-1)),
                from_dlpack(out_f32.reshape(-1)), st
            )
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_FC2_ROT_CACHE[key](
            from_dlpack(fc2_f16.reshape(-1)), from_dlpack(svh_row.reshape(-1)),
            from_dlpack(out_f32.reshape(-1)), st
        )
        return out_f32

    m_rows, hidden = x.shape
    _, top_k = topk_ids.shape
    route_outputs = torch.zeros((m_rows, top_k, hidden), dtype=torch.float32, device=x.device)
    ids_cpu = topk_ids.detach().cpu()
    for expert in sorted({int(e) for e in ids_cpu.reshape(-1).tolist() if int(e) >= 0}):
        pos = (ids_cpu == expert).nonzero(as_tuple=False)
        token_idx = pos[:, 0].to(device=x.device, dtype=torch.long)
        slot_idx = pos[:, 1].to(device=x.device, dtype=torch.long)
        x_e = x.index_select(0, token_idx).to(torch.float16).to(torch.float32)
        rot_g = run_had128(x_e * gate_suh[expert].to(torch.float32)).to(torch.float16)
        rot_u = run_had128(x_e * up_suh[expert].to(torch.float32)).to(torch.float16)
        fc1_g = run_gemm(rot_g, gate_w[expert])
        fc1_u = run_gemm(rot_u, up_w[expert])
        act = run_inter_act(fc1_g, fc1_u, inter_rot[expert])
        fc2 = run_gemm(act, down_w[expert])
        route_outputs[token_idx, slot_idx] = run_fc2_rot(fc2, down_svh[expert])

    acc = torch.zeros((m_rows, hidden), dtype=torch.float32, device=x.device)
    w_f32 = topk_weights.to(torch.float32)
    for k_idx in range(top_k):
        acc = torch.addcmul(acc, route_outputs[:, k_idx], w_f32[:, k_idx : k_idx + 1])
    return acc.to(torch.bfloat16) if output_dtype == torch.bfloat16 else acc


@unittest.skipUnless(
    os.environ.get("CUTEAFD_EXL3_MCG_AOT") and os.environ.get("CUTEAFD_EXL3_MUL1_AOT"),
    "opt-in exact-AOT codebook GPU qualification",
)
class Exl3CodebookNumericsTests(unittest.TestCase):
    def test_mcg_and_mul1_packages_match_bitwise_oracle_and_differ(self):
        import torch

        mcg_dirs = [Path(p) for p in os.environ["CUTEAFD_EXL3_MCG_AOT"].split(os.pathsep) if p]
        mul1_dirs = [Path(p) for p in os.environ["CUTEAFD_EXL3_MUL1_AOT"].split(os.pathsep) if p]
        self.assertTrue(mcg_dirs)
        self.assertEqual(len(mcg_dirs), len(mul1_dirs))
        torch.cuda.set_device(0)

        for mcg_dir, mul1_dir in zip(mcg_dirs, mul1_dirs):
            with self.subTest(mcg=str(mcg_dir), mul1=str(mul1_dir)):
                self._qualify_pair(mcg_dir, mul1_dir, torch)

    def _qualify_variant(self, root: Path, expected_codebook: str, torch):
        from b12x.moe._shared.kernels.w4a16.mixed_trellis import _descriptor_bits
        from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

        meta = json.loads((root / "v41_exl3.json").read_text())
        if expected_codebook == "mcg":
            self.assertNotIn("codebook", meta)
        else:
            self.assertEqual(meta.get("codebook"), expected_codebook)

        hidden = int(meta["hidden"])
        width = int(meta["intermediate"])
        experts = int(meta["experts"])
        capacity = int(meta["capacity"])
        topk = int(meta["top_k"])
        bits = tuple(int(b) for b in meta["bits"])
        tile = tuple(int(t) for t in meta["tile"])
        swiglu_limit = meta["swiglu_limit"]
        out_dtype = torch.bfloat16 if meta["output_dtype"] == "bf16" else torch.float32
        device = torch.device("cuda", 0)
        props = torch.cuda.get_device_properties(0)
        self.assertEqual([props.major, props.minor], meta["compute"])

        active_experts = max(8, topk)
        if active_experts % 2:
            active_experts += 1
        slots_per_tier = active_experts // 2
        gate_tiers = tuple(e % 2 for e in range(active_experts))
        up_tiers = tuple((e + 1) % 2 for e in range(active_experts))
        down_tiers = tuple((e // 2) % 2 for e in range(active_experts))

        gen = torch.Generator(device="cpu").manual_seed(9100 + capacity)
        signs = (-1.0, 1.0)
        gate_suh_small = torch.where(
            torch.randint(0, 2, (active_experts, hidden), generator=gen) == 0,
            torch.tensor(signs[0], dtype=torch.float16),
            torch.tensor(signs[1], dtype=torch.float16),
        ).to(device)
        up_suh_small = torch.where(
            torch.randint(0, 2, (active_experts, hidden), generator=gen) == 0,
            torch.tensor(signs[0], dtype=torch.float16),
            torch.tensor(signs[1], dtype=torch.float16),
        ).to(device)
        inter_rot_small = (torch.randn((active_experts, 3 * width), generator=gen) * 0.25).to(
            device=device, dtype=torch.float16
        )
        down_svh_small = (torch.randn((active_experts, hidden), generator=gen) * 0.125).to(
            device=device, dtype=torch.float16
        )

        gate_raw = []
        up_raw = []
        down_raw = []
        gate_w = []
        up_w = []
        down_w = []
        for e in range(active_experts):
            gb = bits[gate_tiers[e]]
            ub = bits[up_tiers[e]]
            db = bits[down_tiers[e]]
            gr = torch.randint(-32768, 32768, (hidden // 16, width // 16, 16 * gb),
                               generator=gen, dtype=torch.int32).to(device=device, dtype=torch.int16)
            ur = torch.randint(-32768, 32768, (hidden // 16, width // 16, 16 * ub),
                               generator=gen, dtype=torch.int32).to(device=device, dtype=torch.int16)
            dr = torch.randint(-32768, 32768, (width // 16, hidden // 16, 16 * db),
                               generator=gen, dtype=torch.int32).to(device=device, dtype=torch.int16)
            gate_raw.append(gr)
            up_raw.append(ur)
            down_raw.append(dr)
            gate_w.append(_decode_gptqmodel_trellis_f16(gr, gb, expected_codebook))
            up_w.append(_decode_gptqmodel_trellis_f16(ur, ub, expected_codebook))
            down_w.append(_decode_gptqmodel_trellis_f16(dr, db, expected_codebook))

        prepared_tiers = []
        for t_idx, t_bits in enumerate(bits):
            g_ids = [e for e in range(active_experts) if gate_tiers[e] == t_idx]
            u_ids = [e for e in range(active_experts) if up_tiers[e] == t_idx]
            d_ids = [e for e in range(active_experts) if down_tiers[e] == t_idx]
            w13_t = torch.stack(
                (torch.stack([gate_raw[e] for e in g_ids], dim=0),
                 torch.stack([up_raw[e] for e in u_ids], dim=0)),
                dim=0,
            ).contiguous()
            w2_t = torch.stack([down_raw[e] for e in d_ids], dim=0).contiguous()
            prepared_tiers.append(
                prepare_trellis256_moe_weights(
                    w13_t,
                    w2_t,
                    hidden_size=hidden,
                    intermediate_size=width,
                    num_experts=slots_per_tier,
                    activation="silu",
                    fc1_tile_n=tile[1],
                    fc2_tile_n=tile[3],
                    params_dtype=torch.float16,
                    w13_layout="trellis_t256_proj",
                    trellis_bits=t_bits,
                    codebook=expected_codebook,
                    gate_suh=gate_suh_small[:slots_per_tier].contiguous(),
                    up_suh=up_suh_small[:slots_per_tier].contiguous(),
                    intermediate_rotations=inter_rot_small[:slots_per_tier].contiguous(),
                    down_svh=down_svh_small[:slots_per_tier].contiguous(),
                    tile_config=tile,
                )
            )

        desc_bits = _descriptor_bits(experts)
        descriptor = torch.full((3, 2 * experts), -1, device=device, dtype=torch.int32)
        for row_idx, proj_tiers in enumerate((gate_tiers, up_tiers, down_tiers)):
            counters = [0, 0]
            for e_idx, t_id in enumerate(proj_tiers):
                local = counters[t_id]
                counters[t_id] += 1
                descriptor[row_idx, e_idx] = (t_id << desc_bits) | local

        expert_map = torch.full((experts,), -1, device=device, dtype=torch.int32)
        expert_map[:active_experts] = torch.arange(active_experts, device=device, dtype=torch.int32)

        rotations = torch.zeros((2 * experts, 3 * width), device=device, dtype=torch.float16)
        rotations[:active_experts].copy_(inter_rot_small)
        gate_suh = torch.zeros((2 * experts, hidden), device=device, dtype=torch.float16)
        gate_suh[:active_experts].copy_(gate_suh_small)
        up_suh = torch.zeros((2 * experts, hidden), device=device, dtype=torch.float16)
        up_suh[:active_experts].copy_(up_suh_small)
        down_svh = torch.zeros((2 * experts, hidden), device=device, dtype=torch.float16)
        down_svh[:active_experts].copy_(down_svh_small)

        buffers = {}
        for name, spec in meta["buffers"].items():
            if name == spec["allocation"]:
                dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
                tensor = torch.empty(spec["shape"], dtype=dtype, device=device)
                self.assertEqual(tensor.numel() * tensor.element_size(), spec["bytes"])
                if spec["zero_on_create"]:
                    tensor.zero_()
                buffers[name] = tensor
        for name, spec in meta["buffers"].items():
            buffers[name] = buffers[spec["allocation"]]
        original_addresses = {name: value.data_ptr() for name, value in buffers.items()}

        x_full = (torch.randn((capacity, hidden), generator=gen, dtype=torch.float32) * 0.25).to(
            device=device, dtype=torch.bfloat16
        )
        ids_full = torch.stack(
            [torch.randperm(active_experts, generator=gen)[:topk] for _ in range(capacity)], dim=0
        ).to(device=device, dtype=torch.int32).contiguous()
        weights_full = torch.softmax(
            torch.randn((capacity, topk), generator=gen, dtype=torch.float32), dim=-1
        ).to(device=device, dtype=torch.float32).contiguous()

        lut_raw = (root / "trellis_lut.bin").read_bytes()
        self.assertEqual(hashlib.sha256(lut_raw).hexdigest(), meta["trellis_lut"]["sha256"])
        lut = torch.frombuffer(bytearray(lut_raw), dtype=torch.float16).to(device)

        pointers = dict(buffers)
        pointers.update(
            rotation_input_ptr=x_full,
            raw_topk_ids=ids_full,
            topk_weights_ptr=weights_full,
            descriptor_map_ptr=descriptor,
            global_to_combined_ptr=expert_map,
            intermediate_rotations_ptr=rotations,
            gate_suh_ptr=gate_suh,
            up_suh_ptr=up_suh,
            svh_ptr=down_svh,
            trellis_lut_ptr=lut,
            fc2_ptr=buffers["fc2"],
            output_ptr=buffers["output"],
            route_expert_ids_ptr=ids_full,
            expert_map_ptr=expert_map,
        )
        scalars = dict(route_num_experts=experts, weight_num_experts=2 * experts)
        for tier_idx, t_bits in enumerate(bits):
            prep = prepared_tiers[tier_idx]
            w13_small = prep.w13.contiguous().view(torch.int32).reshape(-1)
            w2_small = prep.w2.contiguous().view(torch.int32).reshape(-1)
            plane_words = hidden * width * t_bits // 32
            self.assertEqual(w13_small.numel(), 2 * slots_per_tier * plane_words)
            self.assertEqual(w2_small.numel(), slots_per_tier * plane_words)
            pointers[f"t{tier_idx}_w13_ptr"] = w13_small.clone()
            pointers[f"t{tier_idx}_w2_ptr"] = w2_small.clone()
            for key in ("w13_scales", "w2_scales"):
                pointers[f"t{tier_idx}_{key}_ptr"] = torch.zeros(16, device=device, dtype=torch.uint8)
            for key in ("w13_global", "w2_global"):
                pointers[f"t{tier_idx}_{key}_ptr"] = torch.ones(experts, device=device, dtype=torch.float32)
            scalars.update(
                {
                    f"tier{tier_idx}_num_experts": experts,
                    f"tier{tier_idx}_fc2_experts": slots_per_tier,
                    f"tier{tier_idx}_gate_experts": slots_per_tier,
                    f"tier{tier_idx}_up_experts": slots_per_tier,
                }
            )

        from cutlass.cute.export.aot_config import get_libdir
        ct.CDLL(str(Path(get_libdir()) / "libcute_dsl_runtime.so"), mode=ct.RTLD_GLOBAL)
        lib = ct.CDLL(str(root / "libcuteafd_exl3.so"))
        lib.cuteafd_exl3_create.argtypes = [ct.POINTER(ct.c_void_p)]
        lib.cuteafd_exl3_create.restype = ct.c_int
        lib.cuteafd_exl3_destroy.argtypes = [ct.c_void_p]
        for fn_role in ("core", "sum"):
            fn = getattr(lib, "cuteafd_exl3_" + fn_role)
            fn.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_int32), ct.c_void_p]
            fn.restype = ct.c_int

        context = ct.c_void_p()
        self.assertEqual(lib.cuteafd_exl3_create(ct.byref(context)), 0)
        route_lib, route_context, graph = None, ct.c_void_p(), None
        active_rows = sorted({1, min(7, capacity)})
        outputs = {}
        try:
            if meta["requires_route_preparation"]:
                route_path = root / meta["route_preparation"]["manifest"]
                self.assertEqual(
                    hashlib.sha256(route_path.read_bytes()).hexdigest(),
                    meta["route_preparation"]["sha256"],
                )
                route_lib = ct.CDLL(str(route_path.parent / "libv41_exl3_routes.so"))
                route_lib.cuteafd_exl3_routes_create.argtypes = [ct.POINTER(ct.c_void_p)]
                route_lib.cuteafd_exl3_routes_create.restype = ct.c_int
                route_lib.cuteafd_exl3_routes_destroy.argtypes = [ct.c_void_p]
                route_lib.cuteafd_exl3_routes_destroy.restype = ct.c_int
                route_lib.cuteafd_exl3_routes_launch.argtypes = [
                    ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint64), ct.c_int32, ct.c_void_p
                ]
                route_lib.cuteafd_exl3_routes_launch.restype = ct.c_int
                route_tensors = [ids_full, expert_map] + [
                    buffers[name] for name in (
                        "packed_route_indices", "block_expert_ids",
                        "packed_route_count", "expert_offsets", "expert_counts",
                    )
                ]
                route_p = (ct.c_void_p * 7)(*[t.data_ptr() for t in route_tensors])
                route_bytes = (ct.c_uint64 * 7)(*[t.numel() * t.element_size() for t in route_tensors])
                self.assertEqual(route_lib.cuteafd_exl3_routes_create(ct.byref(route_context)), 0)

            def poison():
                for name in ("rotation_gate", "rotation_up", "fc1", "activated", "fc2", "output", "fc1_scratch", "fc2_scratch"):
                    if name in buffers:
                        buffers[name].fill_(float("nan"))
                for name in ("packed_route_indices", "block_expert_ids", "packed_route_count", "expert_offsets", "expert_counts"):
                    if name in buffers:
                        buffers[name].fill_(-777)

            def launch_native(rows: int):
                scalars.update(active_m=rows, grid_x=props.multi_processor_count * int(meta["blocks_per_sm"]))
                stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
                if route_lib is not None:
                    route_bytes[0] = rows * topk * ids_full.element_size()
                    self.assertEqual(
                        route_lib.cuteafd_exl3_routes_launch(route_context, route_p, route_bytes, rows, stream), 0
                    )
                for entry in meta["objects"]:
                    p = (ct.c_void_p * len(entry["pointer_slots"]))(
                        *[pointers[n].data_ptr() for n in entry["pointer_slots"]]
                    )
                    s = (ct.c_int32 * len(entry["scalar_slots"]))(*[scalars[n] for n in entry["scalar_slots"]])
                    fn_role = entry["label"].rsplit("_", 1)[1]
                    self.assertEqual(getattr(lib, "cuteafd_exl3_" + fn_role)(context, p, s, stream), 0)

            raw_view = torch.uint16 if out_dtype == torch.bfloat16 else torch.int32
            for rows in active_rows:
                poison()
                launch_native(rows)
                torch.cuda.synchronize()
                actual = buffers["output"][:rows].clone()
                ref = _bitwise_reference_moe(
                    x_full[:rows], ids_full[:rows], weights_full[:rows],
                    gate_w, up_w, down_w,
                    gate_suh_small, up_suh_small, inter_rot_small, down_svh_small,
                    swiglu_limit=swiglu_limit, output_dtype=out_dtype, tile_k=tile[0],
                )
                torch.cuda.synchronize()
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertGreater(float(actual.float().abs().sum().item()), 0.0)
                mismatches = int((actual.view(raw_view) != ref.view(raw_view)).sum().item())
                max_abs = float((actual.float() - ref.float()).abs().max().item())
                self.assertEqual(
                    mismatches, 0,
                    f"{expected_codebook} cap={capacity} m={rows} bitwise mismatch: {mismatches}/{actual.numel()} max_abs={max_abs:.6e}",
                )
                outputs[rows] = actual

            rows = active_rows[-1]
            poison()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch_native(rows)
            poison()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(buffers["output"][:rows].view(raw_view), outputs[rows].view(raw_view)))
            self.assertEqual(original_addresses, {name: value.data_ptr() for name, value in buffers.items()})
        finally:
            torch.cuda.synchronize()
            if graph is not None:
                del graph
            if route_context.value:
                self.assertEqual(route_lib.cuteafd_exl3_routes_destroy(route_context), 0)
            lib.cuteafd_exl3_destroy(context)

        so_sha = hashlib.sha256((root / "libcuteafd_exl3.so").read_bytes()).hexdigest()
        print(json.dumps({
            "aot": str(root),
            "codebook": expected_codebook,
            "compute": meta["compute"],
            "capacity": capacity,
            "rows": active_rows,
            "output_dtype": meta["output_dtype"],
            "library_sha256": so_sha,
            "bitwise_equal": True,
            "cuda_graph_replay": True,
        }), flush=True)
        return outputs, so_sha, out_dtype

    def _qualify_pair(self, mcg_dir: Path, mul1_dir: Path, torch):
        mcg_out, mcg_sha, out_dtype = self._qualify_variant(mcg_dir, "mcg", torch)
        mul1_out, mul1_sha, _ = self._qualify_variant(mul1_dir, "mul1", torch)
        self.assertNotEqual(mcg_sha, mul1_sha)
        raw_view = torch.uint16 if out_dtype == torch.bfloat16 else torch.int32
        for rows in mcg_out:
            self.assertFalse(
                torch.equal(mcg_out[rows].view(raw_view), mul1_out[rows].view(raw_view)),
                f"MCG and MUL1 outputs unexpectedly identical at m={rows}",
            )


if __name__ == "__main__":
    unittest.main()
