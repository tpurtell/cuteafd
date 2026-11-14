#!/usr/bin/env python3
"""Bitwise live-row invariance of installed MiniMax MXFP8 graph buckets.

Run with golden-run.py --command in a sealed SM120 slot. Uses the shipping
program table, not a JIT implementation or a private model launcher.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    import torch

    torch.cuda.set_device(0)
    library = C.CDLL(str(args.library))

    class Info(C.Structure):
        _fields_ = [('name', C.c_char * 64), ('pointers', C.c_uint32),
                    ('scalars', C.c_uint32), ('scalar_kinds', C.c_char * 16)]

    library.cuteafd_dsv4_program_count.restype = C.c_uint32
    library.cuteafd_dsv4_program_info.argtypes = [C.c_uint32, C.POINTER(Info)]
    library.cuteafd_dsv4_program_load.argtypes = [C.c_uint32]
    library.cuteafd_dsv4_program_launch.argtypes = [C.c_uint32, C.POINTER(C.c_void_p),
                                                 C.POINTER(C.c_uint64), C.c_void_p]
    indices = {}
    for index in range(library.cuteafd_dsv4_program_count()):
        info = Info()
        assert library.cuteafd_dsv4_program_info(index, C.byref(info)) == 0
        indices[info.name.decode()] = (index, info.pointers, info.scalars, info.scalar_kinds)
    manifest = json.loads(args.manifest.read_bytes())
    programs = {program['name']: program for program in manifest['programs']}
    names = sorted(name for name in programs if re.fullmatch(r'exlr8_dense_mxfp8_k\d+_n\d+_s[12]_a8', name))
    assert names, 'installed MiniMax native MXFP8 programs missing'
    stream = torch.cuda.current_stream()
    cta_limit = torch.cuda.get_device_properties(0).multi_processor_count * 4
    generator = torch.Generator(device='cuda').manual_seed(1856)
    cases = []

    for name in names:
        match = re.fullmatch(r'exlr8_dense_mxfp8_k(\d+)_n(\d+)_s([12])_a8', name)
        k, n, side = map(int, match.groups())
        quant_name = f'exlr8_dense_quant_k{k}_bf16_a8'
        for program, pointers in [(name, 5), (quant_name, 2)]:
            index, actual, scalars, kinds = indices[program]
            assert (actual, scalars, kinds) == (pointers, 3, b'iii'), program
            assert library.cuteafd_dsv4_program_load(index) == 0, program
        geometry = programs[name]['geometry']
        assert geometry['encoding'] == 'mxfp8' and geometry['input_dtype'] == 'bf16'
        x = torch.randn((64, k), generator=generator, device='cuda').bfloat16()
        x[0].zero_()
        x[1].mul_(1e-4)
        table = torch.tensor([x.data_ptr()], dtype=torch.int64, device='cuda')
        payload = torch.randn((n, k), generator=generator, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.randint(120, 126, (n, k // 32), generator=generator, dtype=torch.uint8, device='cuda')
        if side == 2:
            payload = payload.reshape(n, k // 128, 128).permute(1, 0, 2).contiguous()
            scales = scales.reshape(n, k // 128, 4).permute(1, 0, 2).contiguous()
        records = torch.empty(4 * (k // 32) * 528, dtype=torch.uint8, device='cuda')
        output = torch.empty((64, n), dtype=torch.float32, device='cuda')
        quant_pointers = (C.c_void_p * 2)(table.data_ptr(), records.data_ptr())
        linear_pointers = (C.c_void_p * 5)(table.data_ptr(), payload.data_ptr(), scales.data_ptr(),
                                        records.data_ptr(), output.data_ptr())

        def launch(rows):
            scalars = (C.c_uint64 * 3)(rows, 0, cta_limit)
            for program, pointers in [(quant_name, quant_pointers), (name, linear_pointers)]:
                status = library.cuteafd_dsv4_program_launch(indices[program][0], pointers, scalars,
                                                            torch.cuda.current_stream().cuda_stream)
                assert status == 0, (program, rows, status)

        # Include boot-C clamped tails as well as the shared bucket inventory.
        for bucket in (1, 3, 4, 16, 17, 64):
            launch(bucket)
            torch.cuda.synchronize()
            executable = torch.cuda.CUDAGraph()
            with torch.cuda.graph(executable):
                launch(bucket)
            for live in sorted({1, max(1, bucket - 1), bucket}):
                launch(live)
                torch.cuda.synchronize()
                expected = output[:live].clone()
                expected_bf16 = expected.bfloat16().view(torch.int16).clone()
                output.fill_(float('nan'))
                executable.replay()
                torch.cuda.synchronize()
                assert torch.isfinite(output[:live]).all(), (name, bucket, live)
                assert torch.equal(output[:live].view(torch.int32), expected.view(torch.int32)), (name, bucket, live, 'F32')
                assert torch.equal(output[:live].bfloat16().view(torch.int16), expected_bf16), (name, bucket, live, 'BF16')
                cases.append({'program': name, 'bucket': bucket, 'live': live, 'f32_bits_equal': True,
                              'bf16_bits_equal': True})
            del executable
        del x, table, payload, scales, records, output
    result = {'schema': 'cuteafd.minimax.padding/1', 'device': torch.cuda.get_device_name(0),
              'programs': names, 'cases': cases, 'passed': True}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'programs': len(names), 'cases': len(cases), 'passed': True}))


if __name__ == '__main__':
    main()
