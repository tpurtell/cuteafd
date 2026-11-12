#!/usr/bin/env python3
"""Replay saved MiniMax FFN inputs through the shipping ABI6 NVFP4 package.

GPU-only numerical test; use golden-run.py --command with a sealed SM120 slot.
Loads four experts selected by the saved reference router. No kernel compilation,
reference bridge, serving, or alternate launch driver. Reference math is the
independent checkpoint QDQ/GEMM oracle, never b12x's quantizer.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct


def aligned(value):
    return (value + 1023) // 1024 * 1024


def scratch_regions(experts, hidden, intermediate, rows, capacity):
    """Shipping fused stream-A4 scratch contract; check its static ABI extent."""
    def tiles(n):
        return (n * 4 + 127) // 128 + min(experts, n * 4)
    pairs, padded = rows * 4, tiles(rows) * 128
    sizes = [4 * (4 + 3 * experts + tiles(capacity)), pairs * 4, pairs * 4,
             padded * hidden // 2, padded * hidden // 16, 0, 0,
             padded * intermediate // 2, padded * intermediate // 16,
             pairs * hidden * 2]
    offsets, end = [], 0
    for size in sizes:
        offsets.append(end)
        end += aligned(size)
    return offsets, sizes, end


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_payload(directory, name, value, shape, dtype):
    """Persist a drained byte view, including enough metadata for CPU replay."""
    path = directory / (name + '.bin')
    with path.open('xb') as stream:
        stream.write(value)
    return {'path': path.name, 'shape': list(shape), 'dtype': dtype,
            'bytes': len(value), 'sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--taps', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--rows-per-expert', type=int, default=16)
    parser.add_argument('--max-nrmse', type=float, default=.01)
    args = parser.parse_args()
    import numpy as np
    import torch
    from families.minimax_m3 import model

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = json.loads((args.snapshot / 'config.json').read_bytes())
    cfg = model.Config.from_dict(root)
    text = root.get('text_config', root)
    assert text['hidden_act'] == 'swigluoai'
    assert text['swiglu_alpha'] == 1.702 and text['swiglu_limit'] == 7.0
    assert 0 < args.rows_per_expert <= 16
    h, n, e, k = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts, cfg.num_experts_per_tok
    assert (h, n, e, k) == (6144, 3072, 128, 4)
    base = f'layer{args.layer:02d}'
    source_path, ids_path = args.taps / (base + '.ffn_norm.bin'), args.taps / (base + '.router_ids.bin')
    source = torch.from_numpy(np.fromfile(source_path, dtype='<f4').reshape(-1, h).copy())
    saved_ids = torch.from_numpy(np.fromfile(ids_path, dtype='<i4').reshape(-1, k).copy())
    assert len(source) == len(saved_ids) and torch.isfinite(source).all()
    assert torch.equal(source, source.bfloat16().float()), 'saved input is not exact BF16'
    counts = torch.bincount(saved_ids.long().flatten(), minlength=e)
    selected = sorted(range(e), key=lambda expert: (-int(counts[expert]), expert))[:4]
    row_groups = [torch.where((saved_ids == expert).any(-1))[0][:args.rows_per_expert] for expert in selected]
    assert all(len(rows) == args.rows_per_expert for rows in row_groups)
    row_ids = torch.cat(row_groups)
    expert_ids = torch.cat([torch.full((len(rows),), expert, dtype=torch.int32)
                            for expert, rows in zip(selected, row_groups)])
    x = source[row_ids].bfloat16().cuda()
    rows = len(x)
    ids = torch.full((rows, k), -1, dtype=torch.int32, device='cuda')
    ids[:, 0] = expert_ids.cuda()
    routing = torch.zeros((rows, k), dtype=torch.float32, device='cuda')
    routing[:, 0] = 1
    actual = torch.full_like(x, float('nan'))

    lib = C.CDLL(str(args.library))
    def function(name, types):
        fn = getattr(lib, 'cuteafd_fp8moe_' + name)
        fn.argtypes, fn.restype = types, C.c_int32
        return fn
    query = function('info', [C.POINTER(C.c_uint32), C.c_uint32])
    words = (C.c_uint32 * 18)()
    assert query(words, 18) == 0
    assert words[0] == int.from_bytes(bytes([6, 4, 3, 1]), 'little')
    assert list(words)[1:8] == [h, n, e, k, n, 1, 1]
    fp = lambda word: struct.unpack('<f', struct.pack('<I', word))[0]
    assert fp(words[8]) == 7 and fp(words[16]) == fp(struct.unpack('<I', struct.pack('<f', 1.702))[0])
    assert fp(words[17]) == 1
    capacity = next(c for c in list(words)[10:10 + words[9]] if c >= rows)
    scratch_query = function('scratch_bytes', [C.c_uint32, C.POINTER(C.c_uint64)])
    scratch_bytes = C.c_uint64()
    assert scratch_query(capacity, C.byref(scratch_bytes)) == 0
    _, _, expected_scratch = scratch_regions(e, h, n, capacity, capacity)
    assert scratch_bytes.value == expected_scratch, 'package is not the expected fused A4 scratch layout'

    # One named packed backing, zeroed unused experts, separate gate/up matrices.
    region_sizes = []
    for height, width in [(n, h), (n, h), (h, n)]:
        region_sizes += [e * height * width // 2, e * height * width // 16 + 8 * e]
    weight_bytes = sum(aligned(size) for size in region_sizes)
    assert weight_bytes + scratch_bytes.value + 2 * 1024**3 < 10 * 1024**3
    arena = torch.zeros(weight_bytes, dtype=torch.uint8, device='cuda')
    regions, offset = [], 0
    for size in region_sizes:
        regions.append(arena[offset:offset + size])
        offset += aligned(size)
    weights = model.Weights(args.snapshot, 'cuda')
    independent, plain, stage_rows = [], [], []
    for group, (expert, original_rows) in enumerate(zip(selected, row_groups)):
        stem = f'{model.TEXT}layers.{args.layer}.block_sparse_moe.experts.{expert}'
        owned = {}
        for projection, (name, height, width) in enumerate([('w1', n, h), ('w3', n, h), ('w2', h, n)]):
            tensors = {suffix: weights.tensor(stem + '.' + name + '.' + suffix)
                       for suffix in ('weight_packed', 'weight_scale', 'weight_global_scale', 'input_global_scale')}
            owned[name] = tensors
            packed, scales = regions[projection * 2:projection * 2 + 2]
            matrix, grid = height * width // 2, height * width // 16
            packed[expert * matrix:(expert + 1) * matrix].copy_(tensors['weight_packed'].flatten())
            scales[expert * grid:(expert + 1) * grid].copy_(tensors['weight_scale'].view(torch.uint8).flatten())
            for tail, scalar in [(0, tensors['weight_global_scale'].reciprocal()),
                                 (1, tensors['input_global_scale'].reciprocal())]:
                start = e * grid + tail * e * 4 + expert * 4
                scales[start:start + 4].copy_(scalar.float().reshape(1).view(torch.uint8))
        assert torch.equal(owned['w1']['weight_global_scale'], owned['w3']['weight_global_scale'])
        assert torch.equal(owned['w1']['input_global_scale'], owned['w3']['input_global_scale'])
        start, stop = group * args.rows_per_expert, (group + 1) * args.rows_per_expert
        def project(value, name):
            tensor = owned[name]
            q = model.quantize_nvfp4(value, tensor['input_global_scale'])
            w = model.dequant_nvfp4(tensor['weight_packed'], tensor['weight_scale'], tensor['weight_global_scale'])
            return model.linear(q, w).bfloat16()
        gate, up = project(x[start:stop], 'w1'), project(x[start:stop], 'w3')
        oai = model.swiglu_oai(gate, up, cfg.swiglu_alpha, cfg.swiglu_limit)
        # Both controls receive the SAME BF16 reference GEMM outputs, not native
        # preactivations (those are fused/shared-memory and not exported).
        plain_act = (torch.nn.functional.silu(gate.float().clamp(max=7))
                     * up.float().clamp(-7, 7)).bfloat16()
        independent.append(project(oai, 'w2'))
        plain.append(project(plain_act, 'w2'))
        stage_rows.append((owned['w2']['input_global_scale'].clone(),
                           model.quantize_nvfp4(oai, owned['w2']['input_global_scale']),
                           model.quantize_nvfp4(plain_act, owned['w2']['input_global_scale'])))
        weights.files.clear()
        del owned
    expected, plain_expected = torch.cat(independent), torch.cat(plain)
    scratch = torch.empty(scratch_bytes.value, dtype=torch.uint8, device='cuda')
    context = C.c_void_p()
    create = function('create', [C.POINTER(C.c_void_p)])
    launch = function('launch', [C.c_void_p, C.c_uint32, C.POINTER(C.c_void_p), C.c_int32, C.c_void_p])
    destroy = function('destroy', [C.c_void_p])
    destroy.restype = None
    assert create(C.byref(context)) == 0 and context.value
    pointers = (C.c_void_p * 11)(*[t.data_ptr() for t in (x, ids, routing, *regions, actual, scratch)])
    try:
        assert launch(context, capacity, pointers, rows, torch.cuda.current_stream().cuda_stream) == 0
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all(), 'native output incomplete/nonfinite'
        offsets, sizes, _ = scratch_regions(e, h, n, rows, capacity)
        def view(index, dtype):
            return scratch[offsets[index]:offsets[index] + sizes[index]].view(dtype)
        meta = view(0, torch.int32).cpu()
        pair_pos = view(2, torch.int32).reshape(rows, k)[:, 0].cpu()
        tile_codes = meta[4 + 3 * e:4 + 3 * e + int(meta[1])]
        xq = view(3, torch.uint8).reshape(-1, h // 2)
        x_s = view(4, torch.uint8).reshape(-1, h // 16)
        aq = view(7, torch.uint8).reshape(-1, n // 2)
        a_s = view(8, torch.uint8).reshape(-1, n // 16)
        args.out.mkdir(parents=True, exist_ok=False)
        payloads = {}
        def persist(name, value, dtype):
            raw = value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            payloads[name] = write_payload(args.out, name, raw, value.shape, dtype)
        persist('input', x, 'bf16-le')
        persist('input_row_ids', row_ids, 'i64-le')
        persist('expert_ids', expert_ids, 'i32-le')
        persist('pair_positions', pair_pos, 'i32-le')
        # Exclude unused capacity and padding: only initialized metadata/rows
        # belong in a replay, not arbitrary allocator bytes.
        persist('metadata', meta[:4 + 3 * e + int(meta[1])], 'i32-le')
        def metrics(value, reference):
            delta = value.float() - reference.float()
            assert torch.isfinite(delta).all()
            return {'nrmse': (delta.square().mean().sqrt() / reference.float().square().mean().sqrt()).item(),
                    'max_abs': delta.abs().max().item()}
        result = {'layer': args.layer, 'experts': selected, 'rows_per_expert': args.rows_per_expert,
                  'saved_input_sha256': digest(source_path), 'saved_ids_sha256': digest(ids_path),
                  'checkpoint_config_sha256': digest(args.snapshot / 'config.json'),
                  'library_sha256': digest(args.library), 'abi': list(words), 'capacity': capacity,
                  'weight_arena_bytes': weight_bytes, 'scratch_bytes': scratch_bytes.value,
                  'activation': {'hidden_act': text['hidden_act'], 'alpha': text['swiglu_alpha'],
                                 'beta': 1, 'limit': text['swiglu_limit'], 'gate_clamp': 'upper only',
                                 'up_clamp': 'symmetric', 'weights': 'separate w1 gate and w3 up'},
                  'oai': metrics(actual, expected), 'plain': metrics(actual, plain_expected), 'per_expert': []}
        for group, expert in enumerate(selected):
            start, stop = group * args.rows_per_expert, (group + 1) * args.rows_per_expert
            tile = torch.where((tile_codes & 65535) == expert)[0]
            assert len(tile) == 1, 'fixture expects one chunk per selected expert'
            group_start = int(meta[4 + e + expert])
            compact = pair_pos[start:stop].long() - group_start + int(tile[0]) * 128
            global_scale, oai_q, plain_q = stage_rows[group]
            packed = aq[compact.cuda()]
            scales = a_s[compact.cuda()].view(torch.float8_e4m3fn)
            native_q = model.dequant_nvfp4(packed, scales, global_scale)
            prefix = f'expert{expert:03d}_'
            for name, value, dtype in [
                    ('scratch_rows', compact, 'i64-le'),
                    ('input_packed', xq[compact.cuda()], 'e2m1-pairs-u8'),
                    ('input_scales', x_s[compact.cuda()], 'e4m3fn-u8'),
                    ('postactivation_packed', packed, 'e2m1-pairs-u8'),
                    ('postactivation_scales', scales.view(torch.uint8), 'e4m3fn-u8'),
                    ('postactivation_qdq', native_q, 'f32-le'),
                    ('oai_postactivation_qdq', oai_q, 'f32-le'),
                    ('plain_postactivation_qdq', plain_q, 'f32-le'),
                    ('down_global_scale', global_scale.reshape(1).float(), 'f32-le')]:
                persist(prefix + name, value, dtype)
            row = {'expert': expert, 'input_rows': row_groups[group].tolist(),
                   'oai': metrics(actual[start:stop], expected[start:stop]),
                   'plain': metrics(actual[start:stop], plain_expected[start:stop]),
                   'postactivation_qdq_oai': metrics(native_q, oai_q),
                   'postactivation_qdq_plain': metrics(native_q, plain_q)}
            result['per_expert'].append(row)
        for name, value in [('native', actual), ('oai', expected), ('plain', plain_expected)]:
            persist(name, value.float(), 'f32-le')
        result['replay'] = {'schema': 1, 'snapshot': str(args.snapshot),
                            'scratch_offsets': offsets, 'scratch_sizes': sizes,
                            'row_order': 'selected expert order, then input row order',
                            'padding_saved': False, 'payloads': payloads}
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        assert all(row['oai']['nrmse'] < args.max_nrmse for row in result['per_expert']), 'native expert exceeds stop bar'
        assert result['oai']['nrmse'] < result['plain']['nrmse'], 'native matches plain activation better than OAI'
        print('PASS MiniMax native NVFP4 selected-expert OAI oracle', flush=True)
    finally:
        torch.cuda.synchronize()
        destroy(context)


if __name__ == '__main__':
    main()
