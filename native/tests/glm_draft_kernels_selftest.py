"""GLM 5.3 Flash drafter kernels: the vocabulary head's tensor-op launch and the FP8 GEMM modes.

Usage: python native/tests/glm_draft_kernels_selftest.py /path/to/libcuteafd_native.so
           [--head SHARD.safetensors] [--quick]

Run with Python containing CUDA PyTorch, on one GPU.

1. Vocabulary head (`cuteafd_vocabulary_head_launch_tensor_op`, `--draft-head tensor`): BF16
   tensor cores with FP32 accumulation against an FP64 reference and against the pedantic FP32
   path (`cuteafd_vocabulary_head_launch_width`) and the few-row kernel (`cuteafd_vocab_head_rows`).
   Reports the drafter's top-16 agreement (BF16-rounded logits, larger first, lower index on
   ties, as `glm_dflash_topk` ranks them) and times each path by row count. `--head` reads
   `lm_head.weight` from a checkpoint shard; otherwise the weight is synthetic.
2. FP8 GEMMs (`cuteafd_fp8_linear`, `--draft-linear`): mode 1 (passes of 128 rows) must equal
   mode 0 (passes of 64) bit for bit at every row count; mode 2 (W8A8 past 8 rows) must equal
   mode 0 bit for bit up to 8 rows; past them it must match mode 0 to accumulation rounding on
   activations that E4M3 holds exactly (every row and 128-wide K block scaled to an amax of 448
   times a power of two), and bit for bit on ternary operands, whose every partial sum is exact
   in any accumulator (a fragment-layout slip misses by O(1) either way). The drafters take BF16
   outputs: in every mode and at every row count the BF16 output must be the FP32 output rounded
   to nearest even, bit for bit (the plain W8A16 entry point too). Times each mode on the
   drafter's GEMM shapes.
"""
import argparse
import ctypes as C
import json
import struct
import sys

import torch

P = C.c_void_p
VOCAB, WIDTH = 154880, 4096
# The DFlash2 drafter's GEMMs (k, n): attention and MLP convolutions, q|k|v, o, gate|up, down,
# the selector projection, and the context update's fc and k|v rows.
SHAPES = [(4096, 1024), (4096, 6144), (4096, 4096), (4096, 24576), (12288, 4096), (4096, 256), (20480, 4096),
          (4096, 2048)]


def bind(path):
    lib = C.CDLL(path)
    sig = {
        "cuteafd_vocab_head_rows": ([P, P, P, C.c_int32, C.c_int32, C.c_int32, P], C.c_int32),
        "cuteafd_vocabulary_head_create_vocab": ([P, C.c_uint64, C.c_int32, C.c_int32, C.c_int32, C.POINTER(P)],
                                                 C.c_int32),
        "cuteafd_vocabulary_head_launch_width": ([P, P, P, P, C.c_int32, P], C.c_int32),
        "cuteafd_vocabulary_head_launch_tensor_op": ([P, P, P, P, C.c_int32, P], C.c_int32),
        "cuteafd_v41_markov_destroy": ([P], C.c_int32),
        "cuteafd_fp8_w8a16_pack": ([P, P, P, C.c_int32, C.c_int32, C.c_int32, P], C.c_int32),
        "cuteafd_fp8_w8a16_workspace": ([C.c_int32, C.c_int32, C.c_int32], C.c_size_t),
        "cuteafd_fp8_w8a16_linear": ([P, P, P, P, C.c_int32, C.c_int32, C.c_int32, C.c_int32, P, C.c_size_t, P],
                                     C.c_int32),
        "cuteafd_fp8_linear_workspace": ([C.c_int32, C.c_int32, C.c_int32, C.c_int32], C.c_size_t),
        "cuteafd_fp8_linear": ([P, P, P, P, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32, P, C.c_size_t, P],
                               C.c_int32),
    }
    for name, (args, ret) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, ret
    return lib


def read_head(path):
    """lm_head.weight [VOCAB, WIDTH] BF16 from a safetensors shard, onto the GPU."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        meta = header["lm_head.weight"]
        assert meta["dtype"] == "BF16" and meta["shape"] == [VOCAB, WIDTH], meta
        start, end = meta["data_offsets"]
        f.seek(8 + size + start)
        data = bytearray(f.read(end - start))
    return torch.frombuffer(data, dtype=torch.bfloat16).view(VOCAB, WIDTH).cuda()


def timed(fn, runs):
    stream = torch.cuda.current_stream()
    fn()
    times = []
    for _ in range(runs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(stream)
        fn()
        b.record(stream)
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]


def top16(logits):
    """The drafter's top 16 of each row: BF16-rounded logits, larger first, lower index on ties."""
    rounded = logits.to(torch.bfloat16).float()
    index = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float64)
    # Exact for |logit| < 2^20 at this vocabulary: the BF16 value, then the lower index.
    key = rounded.double() * 2.0 ** 20 - index / 2.0 ** 18
    return torch.topk(key, 16, dim=1).indices


def head_check(lib, args):
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(53)
    weight = read_head(args.head) if args.head else (torch.randn(VOCAB, WIDTH, device="cuda") * 0.02).bfloat16()
    workspace = torch.empty(4 << 20, device="cuda", dtype=torch.uint8)
    handle = P()
    assert lib.cuteafd_vocabulary_head_create_vocab(workspace.data_ptr(), workspace.numel(), WIDTH, 128, VOCAB,
                                                    C.byref(handle)) == 0
    rows_all = 128
    # Final-norm-like rows: unit RMS with a per-row scale.
    x = torch.randn(rows_all, WIDTH, device="cuda") * torch.linspace(0.5, 2.0, rows_all, device="cuda")[:, None]
    x = x.bfloat16()
    out = {name: torch.empty(rows_all, VOCAB, device="cuda") for name in ("pedantic", "tensor", "few")}

    def launch(name, rows):
        if name == "pedantic":
            status = lib.cuteafd_vocabulary_head_launch_width(handle, x.data_ptr(), weight.data_ptr(),
                                                              out[name].data_ptr(), rows, stream)
        elif name == "tensor":
            status = lib.cuteafd_vocabulary_head_launch_tensor_op(handle, x.data_ptr(), weight.data_ptr(),
                                                                  out[name].data_ptr(), rows, stream)
        else:
            status = lib.cuteafd_vocab_head_rows(x.data_ptr(), weight.data_ptr(), out[name].data_ptr(), rows, WIDTH,
                                                 VOCAB, stream)
        assert status == 0, (name, rows, status)

    for name in ("pedantic", "tensor"):
        launch(name, rows_all)
    torch.cuda.synchronize()
    # FP64 reference for 8 rows, in vocabulary chunks.
    ref = torch.cat([x[:8].double() @ weight[v:v + 16384].double().T for v in range(0, VOCAB, 16384)], dim=1)
    scale = ref.abs().amax().item()
    err = {name: (out[name][:8].double() - ref).abs().amax().item() / scale for name in ("pedantic", "tensor")}
    print(f"head: max |error| / max |logit| over 8 rows: pedantic {err['pedantic']:.2e}, tensor {err['tensor']:.2e}")
    assert err["pedantic"] < 1e-5, err
    assert err["tensor"] < 1e-3, err  # wrong operands or layouts give O(1)
    def agreement(name, rows):
        a, b = top16(out["pedantic"][:rows]), top16(out[name][:rows])
        same_top = (a == b).all(dim=1).float().mean().item()
        same_first = (a[:, 0] == b[:, 0]).float().mean().item()
        same_set = sum(set(a[r].tolist()) == set(b[r].tolist()) for r in range(rows)) / rows
        print(f"head: {name} against pedantic over {rows} rows: top-1 {100 * same_first:.2f}%, top-16 set "
              f"{100 * same_set:.2f}%, top-16 in order {100 * same_top:.2f}%")
        return same_first

    assert agreement("tensor", rows_all) >= 0.95
    # The few-row kernel is the exact route up to 24 rows: FP32 products and sums like pedantic, in
    # another order (the baseline for how often two FP32 orders reorder the top 16).
    launch("few", 24)
    torch.cuda.synchronize()
    few_err = (out["few"][:8].double() - ref).abs().amax().item() / scale
    assert few_err < 1e-5, few_err
    agreement("few", 24)
    runs = 5 if args.quick else 20
    print("head ms (median of %d): rows, few-row kernel, pedantic cuBLAS, tensor-op cuBLAS" % runs)
    for rows in (8, 16, 24, 32, 64, 96, 128):
        few = f"{timed(lambda: launch('few', rows), runs):8.3f}" if rows <= 24 else "       -"
        print(f"  {rows:4d} {few} {timed(lambda: launch('pedantic', rows), runs):8.3f} "
              f"{timed(lambda: launch('tensor', rows), runs):8.3f}")
    assert lib.cuteafd_v41_markov_destroy(handle) == 0


def e4m3_exact_rows(rows, k, generator):
    """BF16 activations that E4M3 holds exactly at the W8A8 scale: E4M3 values times 2^j per row and
    128-wide block, each block with one entry of magnitude 448 * 2^j (so amax / 448 = 2^j)."""
    codes = torch.randint(0, 256, (rows, k), generator=generator, dtype=torch.int32)
    codes = codes[(codes & 0x7F) != 0x7F]  # no NaN codes
    codes = torch.cat([codes, torch.zeros(rows * k, dtype=torch.int32)])[:rows * k].view(rows, k)
    values = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
    values[:, ::128] = 448.0 * torch.where(torch.rand(rows, k // 128, generator=generator) < 0.5, -1.0, 1.0)
    exponents = torch.randint(-12, -2, (rows, k // 128), generator=generator).float()
    values = values * torch.repeat_interleave(torch.exp2(exponents), 128, dim=1)
    assert torch.equal(values.bfloat16().float(), values)
    return values.bfloat16().cuda()


def ternary(rows, k, scale, generator):
    """{-1, 0, 1} * scale per row and 128-wide block (2^j for activations), the first entry of
    each block 1, so every block's amax is its scale."""
    values = torch.randint(-1, 2, (rows, k), generator=generator).float()
    values[:, ::128] = 1.0
    return (values * scale).bfloat16().cuda()


def fp8_check(lib, args):
    stream = torch.cuda.current_stream().cuda_stream
    generator = torch.Generator().manual_seed(8)
    shapes = SHAPES[:3] if args.quick else SHAPES
    # 136 and 200: a 128-row pass and a shorter one (8 rows: the W8A16 path; 72: another wide pass).
    row_counts = (1, 8, 9, 16, 33, 64, 65, 96, 127, 128, 136, 200)
    for k, n in shapes:
        weight = (torch.randn(n, k, generator=generator) * 0.02).bfloat16().cuda()
        packed = torch.empty(n * k, device="cuda", dtype=torch.uint8)
        scale = torch.empty(n * k // 128, device="cuda", dtype=torch.float32)
        assert lib.cuteafd_fp8_w8a16_pack(weight.data_ptr(), packed.data_ptr(), scale.data_ptr(), n, k, 1, stream) == 0
        rows_max = max(row_counts)
        sizes = [lib.cuteafd_fp8_linear_workspace(rows_max, k, n, mode) for mode in (0, 1, 2)]
        assert sizes[0] == lib.cuteafd_fp8_w8a16_workspace(rows_max, k, n), sizes
        assert sizes[0] <= sizes[1] <= sizes[2], sizes
        workspace = torch.empty(sizes[2], device="cuda", dtype=torch.uint8)

        def run(mode, x, rows, out):
            # FP32 output for an FP32 `out`, BF16 for a BF16 one.
            out_f32 = int(out.dtype == torch.float32)
            status = lib.cuteafd_fp8_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(), out.data_ptr(), out_f32,
                                            rows, k, n, mode, workspace.data_ptr(), workspace.numel(), stream)
            assert status == 0, (k, n, mode, rows, out_f32, status)

        x = (torch.randn(rows_max, k, generator=generator) * torch.linspace(0.1, 3.0, rows_max)[:, None]).bfloat16()
        x = x.cuda()
        exact = e4m3_exact_rows(rows_max, k, generator)
        # Ternary operands: weights {-1, 0, 1} / 64 (E4M3 256 at the pow2 scale 2^-14), activations
        # {-1, 0, 1} * 448 * 2^j per block; every product and partial sum is exact.
        weight3 = ternary(n, k, 2.0 ** -6, generator)
        packed3 = torch.empty(n * k, device="cuda", dtype=torch.uint8)
        scale3 = torch.empty(n * k // 128, device="cuda", dtype=torch.float32)
        assert lib.cuteafd_fp8_w8a16_pack(weight3.data_ptr(), packed3.data_ptr(), scale3.data_ptr(), n, k, 1,
                                          stream) == 0
        x3 = ternary(rows_max, k, torch.repeat_interleave(
            448.0 * torch.exp2(torch.randint(-12, -2, (rows_max, k // 128), generator=generator).float()), 128, dim=1),
            generator)
        for rows in row_counts:
            outs = {}
            for mode in (0, 1, 2):
                outs[mode] = torch.full((rows, n), float("nan"), device="cuda")
                run(mode, x, rows, outs[mode])
            # The plain entry point is mode 0.
            legacy = torch.empty(rows, n, device="cuda")
            assert lib.cuteafd_fp8_w8a16_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(), legacy.data_ptr(), 1,
                                                rows, k, n, workspace.data_ptr(), workspace.numel(), stream) == 0
            # BF16 outputs (the drafters' dtype): the FP32 ones rounded, in every mode.
            outs16 = {}
            for mode in (0, 1, 2):
                outs16[mode] = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
                run(mode, x, rows, outs16[mode])
            legacy16 = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
            assert lib.cuteafd_fp8_w8a16_linear(x.data_ptr(), packed.data_ptr(), scale.data_ptr(), legacy16.data_ptr(),
                                                0, rows, k, n, workspace.data_ptr(), workspace.numel(), stream) == 0
            torch.cuda.synchronize()
            assert torch.equal(legacy, outs[0]), (k, n, rows)
            assert torch.equal(legacy16, outs[0].bfloat16()), ("BF16 output of the W8A16 entry point", k, n, rows)
            for mode in (0, 1, 2):
                assert torch.equal(outs16[mode], outs[mode].bfloat16()), ("BF16 output is not the FP32 one rounded",
                                                                          k, n, mode, rows)
            assert torch.equal(outs[1], outs[0]), ("wide is not bit-identical", k, n, rows)
            if rows <= 8:
                assert torch.equal(outs[2], outs[0]), ("w8a8 changed one draft block", k, n, rows)
            else:
                rms = outs[0].pow(2).mean().sqrt().item()
                drift = (outs[2] - outs[0]).pow(2).mean().sqrt().item() / rms
                exact0, exact2 = torch.empty(rows, n, device="cuda"), torch.empty(rows, n, device="cuda")
                run(0, exact, rows, exact0)
                run(2, exact, rows, exact2)
                torch.cuda.synchronize()
                # Both sum exact products in FP32; FP8 tensor cores may hold fewer bits while
                # they sum within an MMA. A wrong fragment layout misses by O(1).
                bound = exact0.abs().amax().item() * 2e-3
                gap = (exact2 - exact0).abs().amax().item()
                assert gap <= bound, ("w8a8 differs on exact activations", k, n, rows, gap, bound)
                tern = [torch.empty(rows, n, device="cuda") for _ in range(2)]
                for mode, out in zip((0, 2), tern):
                    assert lib.cuteafd_fp8_linear(x3.data_ptr(), packed3.data_ptr(), scale3.data_ptr(), out.data_ptr(),
                                                  1, rows, k, n, mode, workspace.data_ptr(), workspace.numel(),
                                                  stream) == 0
                torch.cuda.synchronize()
                assert torch.equal(tern[0], tern[1]), ("w8a8 differs on ternary operands", k, n, rows)
                if rows in (16, 128):
                    print(f"fp8 [{k}, {n}] {rows} rows: w8a8 relative RMS drift {drift:.2e} (E4M3 activations); "
                          f"on exact activations max |gap| {gap:.2e} (bound {bound:.2e})")
        runs = 5 if args.quick else 20
        line = f"fp8 [{k}, {n}] ms (median of {runs}), rows: w8a16 / wide / w8a8:"
        for rows in (8, 16, 32, 64, 128):
            out = torch.empty(rows, n, device="cuda")
            line += " %d: %s" % (rows, " / ".join(f"{timed(lambda: run(mode, x, rows, out), runs):.4f}"
                                                  for mode in (0, 1, 2)))
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--head", help="checkpoint shard holding lm_head.weight (default: a synthetic head)")
    parser.add_argument("--quick", action="store_true", help="three GEMM shapes, fewer timing runs")
    parser.add_argument("--skip-head", action="store_true")
    args = parser.parse_args()
    lib = bind(args.library)
    print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).multi_processor_count, "SMs")
    if not args.skip_head:
        head_check(lib, args)
    fp8_check(lib, args)
    print("glm draft kernels: all checks passed")


if __name__ == "__main__":
    sys.exit(main())
