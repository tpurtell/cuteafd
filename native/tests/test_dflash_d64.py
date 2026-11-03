"""Native d64 DFlash2 gate; run with CUTEAFD_DFLASH_LIBRARY inside CUDA pytest.

The Qwen3 RMSNorm/rotate-half reference is the existing DFlash2 reference,
python/reference/families/glm5/dflash2/reference.py. No checkpoint weights are
needed to exercise its per-head arithmetic and block attention contract.
"""
import ctypes as C
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from draft_kv_selftest import pack, plane_bytes, unpack

P, I, S = C.c_void_p, C.c_int32, C.c_float
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native():
    path = os.environ.get("CUTEAFD_DFLASH_LIBRARY")
    if not path:
        pytest.skip("set CUTEAFD_DFLASH_LIBRARY to the native CUDA library")
    lib = C.CDLL(path)
    for suffix in ("", "_fp8"):
        fn = getattr(lib, "cuteafd_glm_dflash_qk_rope_d64" + suffix)
        fn.argtypes, fn.restype = [P] * 8 + [I] * 3 + [S] * 2 + [P], I
        fn = getattr(lib, "cuteafd_glm_dflash_attention_d64" + suffix)
        fn.argtypes, fn.restype = [P] * 10 + [I] * 7 + [S, P], I
    fn = lib.cuteafd_glm_dflash_attention_workspace_d64
    fn.argtypes, fn.restype = [I] * 3, C.c_uint64
    return lib


@pytest.fixture(scope="module")
def reference():
    path = ROOT / "python/reference/families/glm5/dflash2/reference.py"
    spec = importlib.util.spec_from_file_location("dflash_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stream():
    return torch.cuda.current_stream().cuda_stream


@pytest.mark.parametrize("heads", [0, 128])
@pytest.mark.parametrize("positions", [[0, 1, 7, 31, 63, 64, 2047, 2048], [0] * 8])
def test_qk_reference_and_records(native, reference, heads, positions):
    torch.manual_seed(718)
    rows, kvh, dim = len(positions), 16, 64
    source = torch.randn(rows, heads + 2 * kvh, dim, device="cuda").bfloat16()
    source[0].zero_()
    qn = (torch.randn(dim, device="cuda") * .1 + 1).bfloat16()
    kn = (torch.randn_like(qn.float()) * .1 + 1).bfloat16()
    pos = torch.tensor(positions, device="cuda", dtype=torch.int64)
    slots = torch.tensor([3, -1, 0, 7, 2, 6, 5, 4], device="cuda", dtype=torch.int32)
    q = torch.empty(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(rows, kvh, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    cq = torch.empty_like(q)
    ck = torch.full((rows, plane_bytes(kvh, dim)), 0xA5, device="cuda", dtype=torch.uint8)
    cv = torch.full_like(ck, 0xA5)

    def launch(src, pp, ss, outputs, count, suffix=""):
        assert getattr(native, "cuteafd_glm_dflash_qk_rope_d64" + suffix)(
            src.data_ptr(), qn.data_ptr(), kn.data_ptr(), pp.data_ptr(),
            ss.data_ptr() if ss is not None else None,
            *[x.data_ptr() for x in outputs], count, heads, kvh,
            10000., reference.EPS, stream()) == 0

    launch(source, pos, slots, (q, k, v), rows)
    launch(source, pos, slots, (cq, ck, cv), rows, "_fp8")
    torch.cuda.synchronize()
    assert torch.equal(q, cq)
    written = slots[slots >= 0].long()
    assert torch.equal(pack(k)[written], ck[written])
    assert torch.equal(pack(v)[written], cv[written])
    assert torch.all(ck[1] == 0xA5) and torch.all(cv[1] == 0xA5)
    for i in range(rows):
        # Independent rows and reordered slots must keep every stored bit.
        qi = torch.empty(1, heads, dim, device="cuda", dtype=torch.bfloat16)
        ki = torch.empty(1, kvh, dim, device="cuda", dtype=torch.bfloat16)
        vi = torch.empty_like(ki)
        launch(source[i:i+1], pos[i:i+1], None, (qi, ki, vi), 1)
        assert torch.equal(q[i], qi[0])
        if slots[i] < 0:
            continue
        at = slots[i].item()
        assert torch.equal(k[at], ki[0]) and torch.equal(v[at], vi[0])
        assert torch.equal(v[at], source[i, heads+kvh:])
        for x, weight, actual in ((source[i:i+1, :heads], qn, qi),
                                  (source[i:i+1, heads:heads+kvh], kn, ki)):
            if not x.numel():
                continue
            expected = reference.rope(reference.rms(x, weight), pos[i:i+1], 10000.)
            # F32 RMS reductions and libdevice trigonometry can cross a BF16
            # midpoint; bound that arithmetic, while discrete steps are exact.
            torch.testing.assert_close(actual, expected, atol=.03125, rtol=.016)


@pytest.mark.parametrize("context", [0, 33, 2047, 2048, 2055])
@pytest.mark.parametrize("window", [0, 2048])
@pytest.mark.parametrize("amplitude", [.2, 4.])
def test_attention_reference_and_row_invariance(native, context, window, amplitude):
    torch.manual_seed(731)
    sequences, block, heads, kvh, dim = 3, 8, 128, 16, 64
    ring = max(96, context + 17)
    shape = (sequences, block, heads, dim)
    q = (torch.randn(shape, device="cuda") * amplitude).bfloat16()
    kb = torch.randn(sequences, block, kvh, dim, device="cuda").bfloat16()
    vb = torch.randn_like(kb)
    kr = torch.randn(sequences * ring, kvh, dim, device="cuda").bfloat16()
    vr = torch.randn_like(kr)
    ck, cv = pack(kr), pack(vr)
    kr, vr = unpack(ck, kvh, dim), unpack(cv, kvh, dim)
    slots = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([max(0, context - 7*i) for i in range(sequences)], device="cuda", dtype=torch.int32)
    ends = torch.tensor([2*ring + context + 13 + i for i in range(sequences)], device="cuda", dtype=torch.int32)
    max_keys = context + block
    size = native.cuteafd_glm_dflash_attention_workspace_d64(sequences, kvh, max_keys)
    assert size > 0
    scratch = torch.full((size + 256,), 173, device="cuda", dtype=torch.uint8)

    def launch(query, key, value, ss, ll, ee, out, count, fp8):
        assert getattr(native, "cuteafd_glm_dflash_attention_d64" + ("_fp8" if fp8 else ""))(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            (ck if fp8 else kr).data_ptr(), (cv if fp8 else vr).data_ptr(),
            ss.data_ptr(), ll.data_ptr(), ee.data_ptr(), out.data_ptr(), scratch.data_ptr(),
            count, block, heads, kvh, ring, max_keys, window, dim**-.5, stream()) == 0

    out, compact = torch.empty_like(q), torch.empty_like(q)
    launch(q, kb, vb, slots, lengths, ends, out, sequences, False)
    launch(q, kb, vb, slots, lengths, ends, compact, sequences, True)
    torch.cuda.synchronize()
    assert torch.equal(out, compact)
    assert torch.all(scratch[size:] == 173)
    order = torch.tensor([1, 2, 0], device="cuda")
    permuted = torch.empty_like(q)
    launch(q[order], kb[order], vb[order], slots[order], lengths[order],
           ends[order], permuted, sequences, True)
    assert torch.equal(permuted, out[order]), "fixed-width wave permutation changes row bytes"
    for s in range(sequences):
        one = torch.empty_like(q[s:s+1])
        launch(q[s:s+1], kb[s:s+1], vb[s:s+1], slots[s:s+1], lengths[s:s+1],
               ends[s:s+1], one, 1, True)
        # The existing shared WMMA path chooses partitions from concurrency.
        # Check exact row stability within a fixed launch shape; cross-shape
        # invariance is a separate shared defect, not changed by this port.
        again = torch.empty_like(one)
        launch(q[s:s+1], kb[s:s+1], vb[s:s+1], slots[s:s+1], lengths[s:s+1],
               ends[s:s+1], again, 1, True)
        assert torch.equal(one, again)
        repeated = torch.empty_like(q)
        launch(q, kb, vb, slots, lengths, ends, repeated, sequences, True)
        assert torch.equal(out, repeated)
        ctx, end, slot = lengths[s].item(), ends[s].item(), slots[s].item()
        positions = torch.arange(end-ctx, end, device="cuda")
        indices = slot * ring + positions % ring
        keys = torch.cat((kr[indices], kb[s])).transpose(0, 1)
        values = torch.cat((vr[indices], vb[s])).transpose(0, 1)
        mask = torch.ones(block, ctx + block, device="cuda", dtype=torch.bool)
        if window:
            mask[:, :ctx] = positions[None] >= end + torch.arange(block, device="cuda")[:, None] - window + 1
        # All block rows see all other block rows: deliberately noncausal.
        ref = F.scaled_dot_product_attention(q[s].transpose(0, 1)[None].double(),
            keys[None].double(), values[None].double(), attn_mask=mask,
            is_causal=False, enable_gqa=True, scale=dim**-.5)[0].transpose(0, 1)
        delta = out[s].double() - ref
        relative = (delta.square().mean() / ref.square().mean().clamp_min(1e-20)).sqrt()
        assert delta.abs().max() < .025 and relative < .01
