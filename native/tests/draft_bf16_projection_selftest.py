"""Qualify generic BF16-weight projection row invariance and BF16/F32 output.

Run inside the coordinator dev container with a task-owned native library:
  python native/tests/draft_bf16_projection_selftest.py LIB [--shape K,N ...]
Each family contributes its context FC, QKV, context KV, O, gate/up and down.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import warnings

import torch


def shapes():
    # hidden, query heads, KV heads, head width, MLP width, tap count
    families = {
        "dspark-qwen": (5120, 40, 8, 128, 10240, 5),
        "dspark-kimi": (7168, 64, 16, 64, 14336, 5),
        "dflash2-glm": (6144, 64, 8, 128, 12288, 6),
        "dflash2-glmf": (4096, 32, 8, 128, 12288, 5),
        "dflash-qwen27": (5120, 32, 8, 128, 17408, 5),
        "dflash-qwen35": (2048, 32, 8, 128, 6144, 8),
        "dflash-kimi-modal": (7168, 32, 8, 128, 14336, 6),
        "dflash-mimo-pro": (6144, 128, 8, 128, 16384, 5),
        "dflash-mimo-flash": (4096, 64, 8, 128, 16384, 5),
        "dflash2-muse-radix": (8192, 64, 8, 128, 28672, 5),
        "dflash2-muse-nvidia": (8192, 128, 16, 64, 24576, 5),
    }
    result = {}
    for family, (h, heads, kv, d, inter, taps) in families.items():
        for k, n in ((h * taps, h), (h, (heads + 2 * kv) * d),
                     (h, 2 * kv * d), (heads * d, h), (h, 2 * inter), (inter, h)):
            result.setdefault((k, n), []).append(family)
    for h in (2048, 4096, 5120, 6144, 7168, 8192):
        result.setdefault((h, h // 4), []).append("dflash-convolution")
        result.setdefault((h, 256), []).append("dflash-selector")
    for h, inter in ((7168, 14336), (6144, 24576)):
        result.setdefault((h, 2 * inter), []).append("mla-dflash/ml4-ffn")
        result.setdefault((inter, h), []).append("mla-dflash/ml4-ffn")
    for h in (2048, 4096, 5120, 6144):
        result.setdefault((2 * h, h), []).append("mimo/qwen-mtp")
    result.setdefault((30720, 6144), []).append("ml4-context-fc")
    for k, n in ((2048, 16384), (2048, 8192), (16384, 6144), (8192, 6144),
                 (1536, 18432), (1536, 9216), (12288, 7168), (6144, 7168),
                 (12288, 4096), (6144, 4096), (8192, 8192), (4096, 8192),
                 (8192, 4096)):
        result.setdefault((k, n), []).append("hqa-native")
    return result


F32_SHAPES = {(6144, 7168), (6144, 4096), (4096, 8192), (8192, 6144)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--out-dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--shape", action="append", help="restrict to K,N; repeatable")
    parser.add_argument("--rows", action="append", type=int, default=[],
                        help="additional live row count (1..2048); repeatable")
    parser.add_argument("--support-library", help="load native dependencies for isolated translation-unit iteration")
    parser.add_argument("--screen", action="store_true", help="compare fixed-Lt, padding and existing row-local TMA")
    parser.add_argument("--skinny-screen", action="store_true",
                        help="compare canonical TMA with row-retiled M16 tensor-core arms")
    args = parser.parse_args()
    if any(rows < 1 or rows > 2048 for rows in args.rows):
        parser.error("--rows must be between 1 and 2048")
    row_counts = tuple(sorted({1, 4, 8, 16, 17, 32, 64, 128, 140, 2048, *args.rows}))
    assert not (args.screen and args.skinny_screen), "select one screen"
    assert args.out_dtype == "bf16" or not (args.screen or args.skinny_screen), "screens require BF16 output"
    output_dtype = torch.bfloat16 if args.out_dtype == "bf16" else torch.float32
    bits_dtype = torch.int16 if args.out_dtype == "bf16" else torch.int32
    poison = 0x7FFF if args.out_dtype == "bf16" else 0x7FFFFFFF
    # Empty captures invalidate both the replay proof and event timings.
    warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
    support = ct.CDLL(args.support_library, mode=ct.RTLD_GLOBAL) if args.support_library else None
    library = ct.CDLL(args.library)
    # Resolve through the native library's linked cuBLAS (same toolkit and ABI).
    blas = library
    handle = ct.c_void_p()
    blas.cublasCreate_v2.argtypes = [ct.POINTER(ct.c_void_p)]
    blas.cublasSetStream_v2.argtypes = [ct.c_void_p, ct.c_void_p]
    blas.cublasDestroy_v2.argtypes = [ct.c_void_p]
    blas.cublasGemmEx.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
        ct.c_int, ct.c_int]
    assert blas.cublasCreate_v2(ct.byref(handle)) == 0
    assert blas.cublasSetStream_v2(handle, torch.cuda.current_stream().cuda_stream) == 0
    alpha, beta = ct.c_float(1), ct.c_float(0)
    launch = library.cuteafd_cuda_linear_bf16_cublas_async
    launch.argtypes = [ct.c_void_p] * 4 + [ct.c_size_t] * 3 + [ct.c_void_p]
    launch.restype = ct.c_int
    library.cuteafd_cuda_prepare_cublas_handle.argtypes = []
    library.cuteafd_cuda_prepare_cublas_handle.restype = ct.c_int
    assert library.cuteafd_cuda_prepare_cublas_handle() == 0
    indices = {}
    if not (args.screen or args.skinny_screen):
        class Info(ct.Structure):
            _fields_ = [("name", ct.c_char * 64), ("pointers", ct.c_uint32),
                        ("scalars", ct.c_uint32), ("scalar_kinds", ct.c_char * 16)]
        library.cuteafd_dsv4_program_info.argtypes = [ct.c_uint32, ct.POINTER(Info)]
        library.cuteafd_dsv4_program_load.argtypes = [ct.c_uint32]
        library.cuteafd_dsv4_program_launch.argtypes = [ct.c_uint32, ct.POINTER(ct.c_void_p),
                                                       ct.POINTER(ct.c_uint64), ct.c_void_p]
        for index in range(library.cuteafd_dsv4_program_count()):
            info = Info()
            assert library.cuteafd_dsv4_program_info(index, ct.byref(info)) == 0
            if info.name.startswith(b"draft_bf16_"):
                assert (info.pointers, info.scalars, info.scalar_kinds) == (3, 1, b"i")
                indices[info.name.decode()] = index
    inventory = shapes()
    if args.out_dtype == "f32":
        inventory = {shape: inventory[shape] for shape in sorted(F32_SHAPES)}
    if args.shape:
        requested = {tuple(map(int, value.split(","))) for value in args.shape}
        inventory = {shape: inventory.get(shape, ["explicit"]) for shape in sorted(requested)}
    torch.manual_seed(1045)
    qualified = []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for (k, n), families in sorted(inventory.items()):
            x = (torch.randn(2048, k, device="cuda") * .1).bfloat16()
            offsets = (0, 1, 17, 63, 64, 65, 127, 139, 511, 1023, 2047)
            row = x[0].clone()
            for offset in offsets:
                x[offset].copy_(row)
            w = (torch.randn(n, k, device="cuda") * .02).bfloat16()
            oracle = x.float() @ w.float().T

            if not (args.screen or args.skinny_screen):
                suffix = "_f32" if args.out_dtype == "f32" else ""
                program_index = indices[f"draft_bf16_k{k}_n{n}{suffix}"]
                assert library.cuteafd_dsv4_program_load(program_index) == 0

            def run(input_, output):
                if args.screen or args.skinny_screen:
                    status = launch(input_.data_ptr(), w.data_ptr(), None, output.data_ptr(),
                                    input_.shape[0], k, n, torch.cuda.current_stream().cuda_stream)
                else:
                    pointers = (ct.c_void_p * 3)(input_.data_ptr(), w.data_ptr(), output.data_ptr())
                    scalars = (ct.c_uint64 * 1)(input_.shape[0])
                    status = library.cuteafd_dsv4_program_launch(program_index, pointers, scalars,
                                                                torch.cuda.current_stream().cuda_stream)
                assert status == 0, (k, n, input_.shape[0], status)

            def default(input_, output):
                # Capture uses a different stream than eager warm-up.
                assert blas.cublasSetStream_v2(handle, torch.cuda.current_stream().cuda_stream) == 0
                status = blas.cublasGemmEx(handle, 1, 0, n, input_.shape[0], k,
                    ct.byref(alpha), w.data_ptr(), 14, k, input_.data_ptr(), 14, k,
                    ct.byref(beta), output.data_ptr(), 14 if args.out_dtype == "bf16" else 0, n, 68, 99)
                assert status == 0, (k, n, status)

            arms = {"default": default, "candidate": run}
            if args.screen or args.skinny_screen:
                sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/sparkinfer"))
                from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
                from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
                local = TmaBf16Projection(n, k, tiles_by_rows=())
                program = compile_program(local, name="draft_bf16_row_local_screen", operands=(
                    Operand("x", torch.bfloat16, f"[rows,{k}]"),
                    Operand("w", torch.bfloat16, f"[{n},{k}]"),
                    Operand("out", torch.bfloat16, f"[rows,{n}]", "out")),
                    scalars=(Scalar("rows"),), key=local.key())

                def tma(input_, output):
                    program.launch(input_, w, output, scalars=(input_.shape[0],))

                arms["tma"] = tma
                if args.skinny_screen:
                    arms.pop("candidate")
                    # Retile M/N only: K64 stages and ordered K16 MMA stay canonical.
                    for tile_n, stages in ((32, 8), (64, 6)):
                        skinny = TmaBf16Projection(n, k,
                            tiles_by_rows=((16, 1, 4, 16, tile_n, stages),))
                        skinny_program = compile_program(skinny,
                            name=f"draft_bf16_skinny_n{tile_n}_screen", operands=(
                                Operand("x", torch.bfloat16, f"[rows,{k}]"),
                                Operand("w", torch.bfloat16, f"[{n},{k}]"),
                                Operand("out", torch.bfloat16, f"[rows,{n}]", "out")),
                            scalars=(Scalar("rows"),), key=skinny.key())

                        def skinny_tma(input_, output, compiled=skinny_program):
                            compiled.launch(input_, w, output, scalars=(input_.shape[0],))

                        arms[f"tma_m16_n{tile_n}"] = skinny_tma
                # Include staging copies in padding costs, not just the larger GEMM.
                for name, multiple in ((("pad64", 64), ("pad2048", 2048)) if args.screen else ()):
                    padded_x = torch.zeros_like(x)
                    padded_out = torch.empty((2048, n), dtype=torch.bfloat16, device="cuda")

                    def padded(input_, output, px=padded_x, po=padded_out, tile=multiple):
                        rows = input_.shape[0]
                        extent = ((rows + tile - 1) // tile) * tile
                        px[:rows].copy_(input_)
                        run(px[:extent], po[:extent])
                        output.copy_(po[:rows])

                    arms[name] = padded

            results = {}
            canonical_outputs = {}
            for arm, operation in arms.items():
                outputs, timing = {}, {}
                graph_exact = True
                for rows in row_counts:
                    storage = torch.empty((2049, n), dtype=output_dtype, device="cuda")
                    storage.view(bits_dtype).fill_(poison)
                    out = storage[:rows]
                    operation(x[:rows], out)
                    torch.cuda.synchronize()
                    assert torch.all(storage[rows:].view(bits_dtype) == poison), (k, n, arm, rows, "eager tail")
                    assert torch.isfinite(out).all(), (k, n, arm, rows, "nonfinite output")
                    torch.testing.assert_close(out.float(), oracle[:rows],
                        rtol=1e-2 if args.out_dtype == "bf16" else 5e-5,
                        atol=2e-3 if args.out_dtype == "bf16" else 5e-5)
                    expected = out.clone()
                    outputs[rows] = expected
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        operation(x[:rows], out)
                    # A replay must overwrite poison; checking stale eager output proves nothing.
                    storage.view(bits_dtype).fill_(poison)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.all(storage[rows:].view(bits_dtype) == poison), (k, n, arm, rows, "graph tail")
                    exact = torch.equal(out.view(bits_dtype), expected.view(bits_dtype))
                    graph_exact = graph_exact and exact
                    assert exact, (k, n, arm, rows, "graph replay did not reproduce eager bytes")
                    for _ in range(5):
                        graph.replay()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    timing[f"n{rows}_us"] = start.elapsed_time(end) * 1000 / 30
                mismatches = []
                for rows, out in outputs.items():
                    if not torch.equal(out.view(bits_dtype), outputs[2048][:rows].view(bits_dtype)):
                        mismatches.append(f"n{rows}/n2048")
                    for offset in offsets:
                        if offset < rows and not torch.equal(outputs[1][0].view(bits_dtype), out[offset].view(bits_dtype)):
                            mismatches.append(f"n{rows}/offset{offset}")
                canonical_exact = None
                if arm == "tma":
                    canonical_outputs = outputs
                elif arm.startswith("tma_m16_"):
                    canonical_exact = all(torch.equal(out.view(bits_dtype),
                        canonical_outputs[rows].view(bits_dtype)) for rows, out in outputs.items())
                    assert canonical_exact, (k, n, arm, "retile changed canonical TMA bytes")
                results[arm] = dict(row_invariant=not mismatches, mismatches=mismatches,
                                    graph_exact=graph_exact, canonical_exact=canonical_exact, timing=timing)
            if args.skinny_screen:
                qualified.append(all(result["row_invariant"] for arm, result in results.items()
                                     if arm.startswith("tma")))
            else:
                qualified.append(results["tma" if args.screen else "candidate"]["row_invariant"])
            print(json.dumps(dict(k=k, n=n, families=families, output_dtype=args.out_dtype, offsets=offsets, arms=results)), flush=True)
        assert all(qualified), "selected row-local/candidate arm failed row invariance"
        print(f"PASS: {len(inventory)} projection shapes, rows={row_counts}, offsets and nonempty graphs", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        assert blas.cublasDestroy_v2(handle) == 0


if __name__ == "__main__":
    main()
