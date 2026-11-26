"""Shared block dSpark rotary/attention numerical gate (CUDA PyTorch).

Usage: python native/tests/block_dspark_selftest.py LIBRARY
The library can contain only glmf_dspark.cu and shared draft_attention.cu.
"""
import argparse
import ctypes as C

import torch

P = C.c_void_p
I = C.c_int32
F = C.c_float


def bind(path):
    lib = C.CDLL(path)
    signatures = {
        "cuteafd_glmf_dspark_markov_workspace": ([I, I], C.c_uint64),
        "cuteafd_glmf_dspark_markov_norms": ([P, P, I, I, P], I),
        "cuteafd_glmf_dspark_markov": ([P] * 7 + [I] * 4 + [P], I),
        "cuteafd_glmf_dspark_qk_rope": ([P] * 8 + [I] * 3 + [F] * 2 + [P], I),
        "cuteafd_block_dspark_qk_rope": ([P] * 9 + [I] * 4 + [F] * 3 + [P], I),
        "cuteafd_block_dspark_qk_rope_fp8": ([P] * 9 + [I] * 4 + [F] * 3 + [P], I),
        "cuteafd_block_dspark_attention_workspace": ([I] * 6, C.c_uint64),
        "cuteafd_block_dspark_attention": ([P] * 10 + [I] * 9 + [F, P], I),
        "cuteafd_block_dspark_attention_paged": ([P] * 11 + [I] * 10 + [F, P], I),
        "cuteafd_block_dspark_attention_fp8": ([P] * 10 + [I] * 9 + [F, P], I),
        "cuteafd_block_dspark_attention_paged_fp8": ([P] * 11 + [I] * 10 + [F, P], I),
        "cuteafd_glmf_dspark_attention_workspace": ([I] * 3, C.c_uint64),
        "cuteafd_glmf_dspark_attention": ([P] * 10 + [I] * 8 + [F, P], I),
        "cuteafd_glmf_dspark_attention_fp8": ([P] * 10 + [I] * 8 + [F, P], I),
    }
    for name, (arguments, result) in signatures.items():
        getattr(lib, name).argtypes = arguments
        getattr(lib, name).restype = result
    return lib


def rotary_check(lib, baseline=None):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    stream = torch.cuda.current_stream().cuda_stream
    positions = torch.tensor([0, 8191, 8192, 262143], device="cuda", dtype=torch.int64)
    slots = torch.tensor([2, -1, 0, 3], device="cuda", dtype=torch.int32)
    for dim, heads, kvh, yarn in ((64,64,64,False),(128,40,8,True),(128,0,8,True),
                                  (64,64,16,True),(64,0,16,True)):
        rows = positions.numel()
        source = torch.randn(rows, heads + 2 * kvh, dim, device="cuda").bfloat16()
        qnorm = (torch.randn(dim, device="cuda") * 0.1 + 1).bfloat16()
        knorm = (torch.randn(dim, device="cuda") * 0.1 + 1).bfloat16()
        q = torch.empty(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.full((rows, kvh, dim), -123, device="cuda", dtype=torch.bfloat16)
        v = torch.full_like(k, -123)
        if yarn:
            config = Qwen3Config(hidden_size=5120, num_attention_heads=40, num_key_value_heads=8,
                head_dim=dim, max_position_embeddings=1048576,
                rope_parameters=dict(rope_type="yarn", rope_theta=10000000. if dim==128 else 10000.,
                    factor=32. if dim==128 else 16.,
                    original_max_position_embeddings=8192 if dim==128 else 65536, beta_fast=32., beta_slow=1.))
            rotary = Qwen3RotaryEmbedding(config, device="cuda")
            inv_freq = rotary.inv_freq.contiguous()
            factor = rotary.attention_scaling
        else:
            inv_freq, factor = None, 1.
        pointers = [source.data_ptr(), qnorm.data_ptr(), knorm.data_ptr(), positions.data_ptr(), slots.data_ptr()]
        outputs = [q.data_ptr(), k.data_ptr(), v.data_ptr()]
        assert lib.cuteafd_block_dspark_qk_rope(*pointers,
            inv_freq.data_ptr() if inv_freq is not None else None, *outputs,
            rows, heads, kvh, dim, 10000., factor, 1e-6, stream) == 0
        if baseline is not None and (not yarn or dim == 128):
            previous = [torch.empty_like(q), torch.full_like(k, -123), torch.full_like(v, -123)]
            assert baseline.cuteafd_block_dspark_qk_rope(*pointers,
                inv_freq.data_ptr() if inv_freq is not None else None, *[x.data_ptr() for x in previous],
                rows, heads, kvh, dim, 10000., factor, 1e-6, stream) == 0
            torch.cuda.synchronize()
            assert all(torch.equal(a,b) for a,b in zip((q,k,v),previous)), "baseline rotary bytes changed"
        if not yarn:
            legacy = [torch.empty_like(q), torch.full_like(k, -123), torch.full_like(v, -123)]
            assert lib.cuteafd_glmf_dspark_qk_rope(*pointers, *[x.data_ptr() for x in legacy],
                rows, heads, kvh, 10000., 1e-6, stream) == 0
            torch.cuda.synchronize()
            assert all(torch.equal(a, b) for a, b in zip((q, k, v), legacy)), "legacy rotary changed"
        else:
            cos, sin = rotary(source[:, :1], positions[None])

            def expected(x, weight):
                unit = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
                normed = unit * weight
                half = torch.cat((-normed[..., dim // 2:], normed[..., :dim // 2]), dim=-1)
                return normed * cos[0, :, None] + half * sin[0, :, None]

            expected_q = expected(source[:, :heads], qnorm)
            expected_k = expected(source[:, heads:heads + kvh], knorm)
            torch.cuda.synchronize()
            # FP32 norm reductions and libdevice sin/cos can straddle a BF16
            # midpoint; no blanket FP32 equality is asserted for those ties.
            if heads:
                torch.testing.assert_close(q, expected_q, rtol=0.016, atol=0.016)
            for row, slot in enumerate(slots.tolist()):
                if slot >= 0:
                    torch.testing.assert_close(k[slot], expected_k[row], rtol=0.016, atol=0.016)
            print(f"rotary dim={dim} heads={heads}/{kvh} max_q_error="
                  f"{(q.float() - expected_q.float()).abs().max().item() if heads else 0:.6g}")
        from draft_kv_selftest import pack, plane_bytes
        compact_q = torch.empty_like(q)
        compact_k = torch.full((rows,plane_bytes(kvh,dim)),0xA5,device="cuda",dtype=torch.uint8)
        compact_v = torch.full_like(compact_k,0xA5)
        assert lib.cuteafd_block_dspark_qk_rope_fp8(*pointers,
            inv_freq.data_ptr() if inv_freq is not None else None,
            compact_q.data_ptr(),compact_k.data_ptr(),compact_v.data_ptr(),
            rows,heads,kvh,dim,10000.,factor,1e-6,stream)==0
        torch.cuda.synchronize()
        assert torch.equal(q,compact_q), "FP8 KV changed query bytes"
        if baseline is not None and (not yarn or dim == 128):
            old_q=torch.empty_like(q)
            old_k=torch.full_like(compact_k,0xA5)
            old_v=torch.full_like(compact_v,0xA5)
            assert baseline.cuteafd_block_dspark_qk_rope_fp8(*pointers,
                inv_freq.data_ptr() if inv_freq is not None else None,
                old_q.data_ptr(),old_k.data_ptr(),old_v.data_ptr(),
                rows,heads,kvh,dim,10000.,factor,1e-6,stream)==0
            torch.cuda.synchronize()
            assert all(torch.equal(a,b) for a,b in zip((compact_q,compact_k,compact_v),(old_q,old_k,old_v))), "baseline FP8 rotary bytes changed"
        expected_k,expected_v=pack(k),pack(v)
        for row,slot in enumerate(slots.tolist()):
            if slot>=0:
                assert torch.equal(compact_k[slot],expected_k[slot]), "FP8 YaRN K writer bytes differ"
                assert torch.equal(compact_v[slot],expected_v[slot]), "FP8 V writer bytes differ"
        assert torch.all(compact_k[1]==0xA5) and torch.all(compact_v[1]==0xA5)
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                assert torch.equal(v[slot], source[row, heads + kvh:]), "V changed"
        assert torch.all(k[1] == -123) and torch.all(v[1] == -123), "negative slot wrote cache"
    print("rotary legacy64 dispatch is byte-identical; YaRN64/128 boundary positions passed")


def attention_check(lib):
    stream = torch.cuda.current_stream().cuda_stream
    for dim, heads, kvh, block in ((64, 64, 64, 8), (128, 40, 8, 7), (64,64,16,5)):
        for sequences in (1, 8):
            for context in (0, 33, 1024):
                for causal in (False, True):
                    capacity = max(context + 17, 64)
                    q = torch.randn(sequences, block, heads, dim, device="cuda").bfloat16()
                    k = torch.randn(sequences, block, kvh, dim, device="cuda").bfloat16()
                    v = torch.randn_like(k)
                    kc = torch.randn(sequences, capacity, kvh, dim, device="cuda").bfloat16()
                    vc = torch.randn_like(kc)
                    slots = torch.arange(sequences, device="cuda", dtype=torch.int32).flip(0).contiguous()
                    lengths = torch.tensor([max(0, context - i % 3) for i in range(sequences)],
                                           device="cuda", dtype=torch.int32)
                    ends = lengths.clone()
                    max_keys = context + block
                    size = lib.cuteafd_block_dspark_attention_workspace(16, heads, kvh, block, max_keys, dim)
                    assert size > 0
                    scratch = torch.full((size + 256,), 173, device="cuda", dtype=torch.uint8)
                    out = torch.full_like(q, float("nan"))
                    pointers = [x.data_ptr() for x in (q, k, v, kc, vc, slots, lengths, ends)]
                    geometry = [sequences, block, heads, kvh, capacity, max_keys, 0, int(causal)]
                    assert lib.cuteafd_block_dspark_attention(*pointers, out.data_ptr(), scratch.data_ptr(),
                        *geometry, dim, dim ** -0.5, stream) == 0
                    # Fragmented pages must preserve the exact arithmetic of the
                    # slot-local route, including page boundaries and peaked scores.
                    width = (capacity + 63) // 64
                    page_ids = torch.randperm(sequences * width, device="cuda", dtype=torch.int32)
                    page_tables = page_ids.reshape(sequences, width)[slots.long()].contiguous()
                    kp = torch.zeros(sequences * width, 64, kvh, dim, device="cuda", dtype=torch.bfloat16)
                    vp = torch.zeros_like(kp)
                    for slot in range(sequences):
                        padded_k = torch.nn.functional.pad(kc[slot], (0, 0, 0, 0, 0, width * 64 - capacity))
                        padded_v = torch.nn.functional.pad(vc[slot], (0, 0, 0, 0, 0, width * 64 - capacity))
                        kp[page_ids.reshape(sequences, width)[slot].long()] = padded_k.reshape(width, 64, kvh, dim)
                        vp[page_ids.reshape(sequences, width)[slot].long()] = padded_v.reshape(width, 64, kvh, dim)
                    from draft_kv_selftest import pack
                    compact_k = pack(kc.reshape(-1, kvh, dim))
                    compact_v = pack(vc.reshape(-1, kvh, dim))
                    compact_kp = torch.zeros(sequences * width, 64, compact_k.shape[-1],
                                             device="cuda", dtype=torch.uint8)
                    compact_vp = torch.zeros_like(compact_kp)
                    for slot in range(sequences):
                        for logical in range(width):
                            first, end = logical * 64, min((logical + 1) * 64, capacity)
                            physical = page_ids.reshape(sequences, width)[slot, logical].long()
                            compact_kp[physical, :end - first] = compact_k[slot * capacity + first:slot * capacity + end]
                            compact_vp[physical, :end - first] = compact_v[slot * capacity + first:slot * capacity + end]
                    for amplitude in (1., 8.):
                        query = (q.float() * amplitude).bfloat16()
                        contiguous = torch.empty_like(q)
                        paged = torch.empty_like(q)
                        assert lib.cuteafd_block_dspark_attention(query.data_ptr(), *pointers[1:],
                            contiguous.data_ptr(), scratch.data_ptr(), *geometry, dim, dim ** -0.5, stream) == 0
                        paged_pointers = [query.data_ptr(), k.data_ptr(), v.data_ptr(), kp.data_ptr(), vp.data_ptr(),
                                          slots.data_ptr(), lengths.data_ptr(), ends.data_ptr(), page_tables.data_ptr()]
                        assert lib.cuteafd_block_dspark_attention_paged(*paged_pointers, paged.data_ptr(),
                            scratch.data_ptr(), *geometry, dim, width, dim ** -0.5, stream) == 0
                        torch.cuda.synchronize()
                        assert torch.equal(contiguous, paged), "fragmented pages changed attention bytes"
                        assert lib.cuteafd_block_dspark_attention_fp8(query.data_ptr(), k.data_ptr(), v.data_ptr(),
                            compact_k.data_ptr(), compact_v.data_ptr(), slots.data_ptr(), lengths.data_ptr(),
                            ends.data_ptr(), contiguous.data_ptr(), scratch.data_ptr(), *geometry, dim,
                            dim ** -0.5, stream) == 0
                        assert lib.cuteafd_block_dspark_attention_paged_fp8(query.data_ptr(), k.data_ptr(), v.data_ptr(),
                            compact_kp.data_ptr(), compact_vp.data_ptr(), slots.data_ptr(), lengths.data_ptr(),
                            ends.data_ptr(), page_tables.data_ptr(), paged.data_ptr(), scratch.data_ptr(),
                            *geometry, dim, width, dim ** -0.5, stream) == 0
                        torch.cuda.synchronize()
                        assert torch.equal(contiguous, paged), "fragmented FP8 pages changed attention bytes"
                    if dim == 64:
                        legacy_size = lib.cuteafd_glmf_dspark_attention_workspace(16, kvh, max_keys)
                        assert size == legacy_size
                        legacy = torch.empty_like(q)
                        assert lib.cuteafd_glmf_dspark_attention(*pointers, legacy.data_ptr(), scratch.data_ptr(),
                            *geometry, dim ** -0.5, stream) == 0
                        torch.cuda.synchronize()
                        assert torch.equal(out, legacy), "legacy attention changed"
                    reference = torch.empty_like(q, dtype=torch.float64)
                    for s in range(sequences):
                        count, slot = lengths[s].item(), slots[s].item()
                        keys = torch.cat((kc[slot, :count], k[s])).double()
                        values = torch.cat((vc[slot, :count], v[s])).double()
                        for h in range(heads):
                            kvhead = h // (heads // kvh)
                            scores = q[s, :, h].double() @ keys[:, kvhead].T * dim ** -0.5
                            if causal:
                                mask = torch.arange(block, device="cuda")[None, :] > torch.arange(block, device="cuda")[:, None]
                                scores[:, count:].masked_fill_(mask, -torch.inf)
                            reference[s, :, h] = scores.softmax(-1) @ values[:, kvhead]
                    delta = out.double() - reference
                    maximum = delta.abs().max().item()
                    rms = (delta.square().mean() / reference.square().mean().clamp_min(1e-20)).sqrt().item()
                    assert torch.all(scratch[size:] == 173), "workspace overrun"
                    assert torch.isfinite(out).all() and maximum < 0.025 and rms < 0.01, (maximum, rms)
                    print(f"attention dim={dim} heads={heads}/{kvh} block={block} sequences={sequences} "
                          f"context={context} causal={causal} max_abs={maximum:.4g} rel_rms={rms:.4g}")
    print("attention legacy64 bytes and 35-query/two-slice FP64 gate passed")


def sliding_attention_check(lib):
    """Wrapped, long-position rows must use the checkpoint's query-local mask."""
    from draft_kv_selftest import pack, unpack
    stream = torch.cuda.current_stream().cuda_stream
    # 17 exercises scalar attention; 2048/3072 force WMMA. Non-default
    # windows prevent accidentally substituting the K3.1 checkpoint's 2048.
    for window, position in ((17, 2500), (2048, 2500), (3072, 8192)):
        for dim, heads, kvh, block in ((64, 4, 4, 8), (128, 10, 2, 7)):
            q = torch.zeros(1, block, heads, dim, device="cuda", dtype=torch.bfloat16)
            k = torch.zeros(1, block, kvh, dim, device="cuda", dtype=torch.bfloat16)
            v = torch.zeros_like(k)
            kc = torch.zeros(window, kvh, dim, device="cuda", dtype=torch.bfloat16)
            vc = torch.zeros_like(kc)
            # Uniform scores make each old boundary key independently observable.
            # The block marker also proves the denominator is exactly window.
            for row in range(block):
                vc[(position - window + row) % window, :, row] = 256
            v[..., block] = 256
            tables = torch.tensor([0, window, position], device="cuda", dtype=torch.int32)
            size = lib.cuteafd_block_dspark_attention_workspace(1, heads, kvh, block, window + block, dim)
            scratch = torch.full((size + 256,), 173, device="cuda", dtype=torch.uint8)
            physical = torch.arange(position - window, position, device="cuda") % window
            key_positions = torch.arange(position - window, position + block, device="cuda")
            query_positions = torch.arange(position, position + block, device="cuda")
            for fp8 in (False, True):
                retained_k, retained_v = (pack(kc), pack(vc)) if fp8 else (kc, vc)
                decoded_k, decoded_v = (unpack(retained_k, kvh, dim), unpack(retained_v, kvh, dim)) if fp8 else (kc, vc)
                keys = torch.cat((decoded_k[physical], k[0])).double().repeat_interleave(heads // kvh, dim=1)
                values = torch.cat((decoded_v[physical], v[0])).double().repeat_interleave(heads // kvh, dim=1)
                for causal in (False, True):
                    mask = key_positions[None] >= query_positions[:, None] - window + 1
                    if causal:
                        mask &= key_positions[None] <= query_positions[:, None]
                        assert torch.all(mask.sum(-1) == window), "causal rows must see exactly window keys"
                    else:
                        # Noncausal dSpark keeps the whole block visible.
                        mask[:, window:] = True
                        assert torch.equal(mask.sum(-1), window - torch.arange(block, device="cuda") - 1 + block)
                    scores = torch.einsum("qhd,khd->hqk", q[0].double(), keys) * dim ** -.5
                    expected = torch.einsum("hqk,khd->qhd", scores.masked_fill(~mask[None], -torch.inf).softmax(-1), values)
                    out = torch.empty_like(q)
                    suffix = "_fp8" if fp8 else ""
                    pointers = [q.data_ptr(), k.data_ptr(), v.data_ptr(), retained_k.data_ptr(), retained_v.data_ptr(),
                                tables.data_ptr(), tables.data_ptr() + 4, tables.data_ptr() + 8,
                                out.data_ptr(), scratch.data_ptr()]
                    geometry = [1, block, heads, kvh, window, window + block, window, int(causal)]
                    assert getattr(lib, "cuteafd_block_dspark_attention" + suffix)(
                        *pointers, *geometry, dim, dim ** -.5, stream) == 0
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out[0].double(), expected, atol=1e-5, rtol=.01)
                    if dim == 64:
                        legacy = torch.empty_like(out)
                        assert getattr(lib, "cuteafd_glmf_dspark_attention" + suffix)(
                            *pointers[:8], legacy.data_ptr(), scratch.data_ptr(), *geometry, dim ** -.5, stream) == 0
                        torch.cuda.synchronize()
                        assert torch.equal(out, legacy), "legacy sliding attention differs"
                    assert torch.all(scratch[size:] == 173), "sliding workspace overrun"
            print(f"sliding attention window={window} position={position} dim={dim}: BF16/FP8 reference masks passed")


def markov_check(lib, baseline=None, rank=256):
    stream = torch.cuda.current_stream().cuda_stream
    vocab, block = 8192, 3
    w1 = torch.randn(vocab, rank, device="cuda").bfloat16()
    w2 = (torch.randn(vocab, rank, device="cuda") * 0.05).bfloat16()
    norms = torch.empty(vocab, device="cuda")
    assert lib.cuteafd_glmf_dspark_markov_norms(w2.data_ptr(), norms.data_ptr(), vocab, rank, stream) == 0
    if baseline is not None:
        old_norms = torch.empty_like(norms)
        assert baseline.cuteafd_glmf_dspark_markov_norms(w2.data_ptr(), old_norms.data_ptr(), vocab, rank, stream) == 0
        assert torch.equal(norms, old_norms), "rank256 Markov norm bytes changed"
    logits = torch.randn(64, block, vocab, device="cuda")
    anchors = torch.randint(vocab, (64,), device="cuda", dtype=torch.int32)

    def run(library, scores, previous):
        sequences = previous.numel()
        size = library.cuteafd_glmf_dspark_markov_workspace(sequences, block)
        scratch = torch.full((size + 256,), 173, device="cuda", dtype=torch.uint8)
        drafts = torch.empty(sequences, block, device="cuda", dtype=torch.int32)
        def launch():
            active_stream = torch.cuda.current_stream().cuda_stream
            assert library.cuteafd_glmf_dspark_markov(scores.data_ptr(), w1.data_ptr(), w2.data_ptr(),
                norms.data_ptr(), previous.data_ptr(), drafts.data_ptr(), scratch.data_ptr(),
                sequences, block, vocab, rank, active_stream) == 0
        launch()
        torch.cuda.synchronize()
        assert torch.all(scratch[size:] == 173), "Markov workspace overrun"
        return drafts, launch

    for sequences in (1, 8, 16, 32, 33, 64):
        actual, launch = run(lib, logits[:sequences], anchors[:sequences])
        previous = anchors[:sequences].long()
        for step in range(block):
            scores = logits[:sequences, step] + w1[previous].float() @ w2.float().T
            expected = scores.argmax(-1)
            assert torch.equal(actual[:, step].long(), expected), f"rank{rank} C{sequences} step{step} reference mismatch"
            previous = expected
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph.replay()
        torch.cuda.synchronize()
        tiles = [run(baseline or lib, logits[first:min(first + 32, sequences)],
                     anchors[first:min(first + 32, sequences)])[0]
                 for first in range(0, sequences, 32)]
        assert torch.equal(actual, torch.cat(tiles)), f"C{sequences} differs from independent C32 tiles"
        if baseline is not None and sequences <= 32:
            expected, old_launch = run(baseline, logits[:sequences], anchors[:sequences])
            assert torch.equal(actual, expected), f"C{sequences} baseline bytes changed"
            if sequences in (1, 16, 32):
                def timing(call):
                    call()  # warm up before timing
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        call()
                    end.record()
                    end.synchronize()
                    return start.elapsed_time(end) / 10
                old_ms, new_ms = timing(old_launch), timing(launch)
                print(f"Markov C{sequences}: baseline={old_ms:.4f}ms candidate={new_ms:.4f}ms ratio={new_ms / old_ms:.4f}")
        print(f"Markov C{sequences}: byte-exact independent sequence tiles; guard intact")
    invalid = lib.cuteafd_glmf_dspark_markov(None, None, None, None, None, None, None,
                                            65, block, vocab, rank, stream)
    assert invalid != 0, "C65 must be rejected before launching"
    for invalid_rank in (0, 128, 257, 1024):
        assert lib.cuteafd_glmf_dspark_markov(None, None, None, None, None, None, None,
            1, block, vocab, invalid_rank, stream) != 0
        assert lib.cuteafd_glmf_dspark_markov_norms(None, None, vocab, invalid_rank, stream) != 0


def batch_composition_invariance_check(lib):
    """A sequence's WMMA attention output must be bitwise identical across batch
    sizes, companion context lengths, and batch positions (both ring and paged, BF16 and FP8)."""
    from draft_kv_selftest import pack
    stream = torch.cuda.current_stream().cuda_stream
    for dim, heads, kvh, block in ((128, 24, 8, 8), (128, 48, 8, 8)):
        max_seqs = 32
        lengths_list = [0, 17, 33, 64, 127, 192, 256, 583, 1024, 1500, 2048,
                        31, 95, 311, 777, 1024, 0, 5, 193, 600, 1024, 1200,
                        48, 128, 512, 999, 1024, 1800, 63, 255, 1023, 2048]
        assert len(lengths_list) == max_seqs
        capacity = max(lengths_list) + 64
        width = (capacity + 63) // 64
        q_all = torch.randn(max_seqs, block, heads, dim, device="cuda").bfloat16()
        k_all = torch.randn(max_seqs, block, kvh, dim, device="cuda").bfloat16()
        v_all = torch.randn_like(k_all)
        kc = torch.randn(max_seqs, capacity, kvh, dim, device="cuda").bfloat16()
        vc = torch.randn_like(kc)
        compact_k = pack(kc.reshape(-1, kvh, dim))
        compact_v = pack(vc.reshape(-1, kvh, dim))
        page_ids = torch.randperm(max_seqs * width, device="cuda", dtype=torch.int32)
        kp = torch.zeros(max_seqs * width, 64, kvh, dim, device="cuda", dtype=torch.bfloat16)
        vp = torch.zeros_like(kp)
        compact_kp = torch.zeros(max_seqs * width, 64, compact_k.shape[-1], device="cuda", dtype=torch.uint8)
        compact_vp = torch.zeros_like(compact_kp)
        for slot in range(max_seqs):
            padded_k = torch.nn.functional.pad(kc[slot], (0, 0, 0, 0, 0, width * 64 - capacity))
            padded_v = torch.nn.functional.pad(vc[slot], (0, 0, 0, 0, 0, width * 64 - capacity))
            kp[page_ids.reshape(max_seqs, width)[slot].long()] = padded_k.reshape(width, 64, kvh, dim)
            vp[page_ids.reshape(max_seqs, width)[slot].long()] = padded_v.reshape(width, 64, kvh, dim)
            for logical in range(width):
                first, end = logical * 64, min((logical + 1) * 64, capacity)
                physical = page_ids.reshape(max_seqs, width)[slot, logical].long()
                compact_kp[physical, :end - first] = compact_k[slot * capacity + first:slot * capacity + end]
                compact_vp[physical, :end - first] = compact_v[slot * capacity + first:slot * capacity + end]
        size = lib.cuteafd_block_dspark_attention_workspace(max_seqs, heads, kvh, block, capacity + block, dim)
        scratch = torch.full((size + 256,), 173, device="cuda", dtype=torch.uint8)

        def run(subset, fp8, paged):
            idx = torch.tensor(subset, device="cuda", dtype=torch.long)
            n = len(subset)
            q = q_all[idx].contiguous()
            k = k_all[idx].contiguous()
            v = v_all[idx].contiguous()
            slots = idx.to(torch.int32).contiguous()
            lens = torch.tensor([lengths_list[s] for s in subset], device="cuda", dtype=torch.int32)
            ends = lens.clone()
            max_keys = max(lengths_list[s] for s in subset) + block
            out = torch.full_like(q, float("nan"))
            if paged:
                pt = page_ids.reshape(max_seqs, width)[idx].contiguous()
                fn = lib.cuteafd_block_dspark_attention_paged_fp8 if fp8 else lib.cuteafd_block_dspark_attention_paged
                k_src, v_src = (compact_kp, compact_vp) if fp8 else (kp, vp)
                assert fn(q.data_ptr(), k.data_ptr(), v.data_ptr(), k_src.data_ptr(), v_src.data_ptr(),
                          slots.data_ptr(), lens.data_ptr(), ends.data_ptr(), pt.data_ptr(),
                          out.data_ptr(), scratch.data_ptr(), n, block, heads, kvh, max_seqs * width * 64,
                          max_keys, 0, 0, dim, width, dim ** -0.5, stream) == 0
            else:
                fn = lib.cuteafd_block_dspark_attention_fp8 if fp8 else lib.cuteafd_block_dspark_attention
                k_src, v_src = (compact_k, compact_v) if fp8 else (kc, vc)
                assert fn(q.data_ptr(), k.data_ptr(), v.data_ptr(), k_src.data_ptr(), v_src.data_ptr(),
                          slots.data_ptr(), lens.data_ptr(), ends.data_ptr(),
                          out.data_ptr(), scratch.data_ptr(), n, block, heads, kvh, capacity,
                          max_keys, 0, 0, dim, dim ** -0.5, stream) == 0
            torch.cuda.synchronize()
            assert torch.all(scratch[size:] == 173), "batch invariance workspace overrun"
            return out

        probes = [0, 2, 5, 7, 8, 10]  # lengths: 0, 33, 192, 583, 1024, 2048
        compositions = [
            [7, 10],
            [10, 7],
            [0, 2, 5, 7, 8, 10, 1, 3],
            [10, 8, 7, 5, 2, 0, 15, 20, 11, 14, 25],
            list(range(16)),
            list(range(32)),
            list(reversed(range(32))),
        ]
        for fp8 in (False, True):
            for paged in (False, True):
                solo = {s: run([s], fp8, paged)[0] for s in probes}
                for comp in compositions:
                    batched = run(comp, fp8, paged)
                    for pos, s in enumerate(comp):
                        if s in solo:
                            assert torch.equal(batched[pos], solo[s]), (
                                f"WMMA attention batch-invariance mismatch: heads={heads}/{kvh} "
                                f"fp8={fp8} paged={paged} slot={s} len={lengths_list[s]} "
                                f"batch_size={len(comp)} pos={pos}"
                            )
        print(f"batch-composition invariance heads={heads}/{kvh} block={block} batches=1,2,8,11,16,32: exact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--baseline", help="Old native library for Markov byte/timing comparisons")
    parser.add_argument("--markov-only", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.1)
    torch.manual_seed(276)
    lib = bind(args.library)
    markov_check(lib, bind(args.baseline) if args.baseline else None)
    markov_check(lib, rank=512)
    if not args.markov_only:
        rotary_check(lib, bind(args.baseline) if args.baseline else None)
        attention_check(lib)
        sliding_attention_check(lib)
        batch_composition_invariance_check(lib)
    print("block dSpark kernels: all checks passed")


if __name__ == "__main__":
    main()
