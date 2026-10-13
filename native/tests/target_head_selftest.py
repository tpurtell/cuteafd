"""Target vocabulary heads on a checkpoint's real BF16 head: error and time of each route.

Usage: python native/tests/target_head_selftest.py /path/to/libcuteafd_native.so SNAPSHOT [SNAPSHOT ...]
           [--rows 1,4,8,16,64] [--runs 50]

Run with Python containing CUDA PyTorch, on one GPU. For each snapshot, reads the BF16 head
(`lm_head.weight` or DeepSeek's `head.weight`) and the final norm weight, builds final-norm-like
rows (unit-RMS Gaussian rows times the norm weight, BF16) and compares three routes against an FP64
product of the same BF16 operands:

- pedantic: `cuteafd_vocabulary_head_launch_width`, cuBLAS with CUBLAS_COMPUTE_32F_PEDANTIC (FP32
  products and sums on CUDA cores) when the width is 4096 or more;
- few: `cuteafd_vocab_head_rows`, FP32 products and sums, one read of the head per 8 rows (up to
  24 rows, widths it admits);
- tensor: `cuteafd_vocabulary_head_launch_tensor_op`, BF16 tensor cores with FP32 accumulation.

Per route: max |error| / max |logit|, top-1 agreement with FP64, and the mean and max KL(FP64 ||
route) of the softmax. Times: the median of single launches and the mean over a back-to-back burst
(as decode runs them, so a power-capped GPU shows its throttled clock).
"""
import argparse
import ctypes as C
import json
import os
import struct
import sys

import torch

P = C.c_void_p


def bind(path):
    lib = C.CDLL(path)
    sig = {
        "cuteafd_vocab_head_rows": ([P, P, P, C.c_int32, C.c_int32, C.c_int32, P], C.c_int32),
        "cuteafd_vocabulary_head_create_vocab": ([P, C.c_uint64, C.c_int32, C.c_int32, C.c_int32, C.POINTER(P)],
                                                 C.c_int32),
        "cuteafd_vocabulary_head_launch_width": ([P, P, P, P, C.c_int32, P], C.c_int32),
        "cuteafd_vocabulary_head_launch_tensor_op": ([P, P, P, P, C.c_int32, P], C.c_int32),
        "cuteafd_v41_markov_destroy": ([P], C.c_int32),
    }
    for name, (args, ret) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, ret
    return lib


def tensor(snapshot, names):
    """The first of `names` in the snapshot as a GPU BF16 tensor, and its name."""
    weight_map = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    name = next(n for n in names if n in weight_map)
    with open(os.path.join(snapshot, weight_map[name]), "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(size))[name]
        assert meta["dtype"] == "BF16", (name, meta)
        start, end = meta["data_offsets"]
        f.seek(8 + size + start)
        data = bytearray(f.read(end - start))
    return torch.frombuffer(data, dtype=torch.bfloat16).view(*meta["shape"]).cuda(), name


def timed(fn, runs):
    fn()
    torch.cuda.synchronize()
    single = []
    for _ in range(runs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        single.append(a.elapsed_time(b))
    single.sort()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(runs * 4):
        fn()
    b.record()
    b.synchronize()
    return single[len(single) // 2], a.elapsed_time(b) / (runs * 4)


def check(lib, snapshot, rows_list, runs):
    weight, head_name = tensor(snapshot, ["lm_head.weight", "head.weight", "language_model.lm_head.weight"])
    norm, norm_name = tensor(snapshot, ["model.norm.weight", "norm.weight", "model.language_model.norm.weight"])
    norm = norm.float()
    if norm.abs().mean() < 0.5:  # zero-centred RMSNorm (Qwen 3.x: 1 + w)
        norm = 1 + norm
    vocab, width = weight.shape
    max_rows = max(rows_list)
    torch.manual_seed(53)
    x = torch.randn(max_rows, width, device="cuda")
    x = (x / x.pow(2).mean(1, keepdim=True).sqrt() * norm).bfloat16()
    workspace = torch.empty(4 << 20, device="cuda", dtype=torch.uint8)
    handle = P()
    assert lib.cuteafd_vocabulary_head_create_vocab(workspace.data_ptr(), workspace.numel(), width, max_rows, vocab,
                                                    C.byref(handle)) == 0
    out = torch.empty(max_rows, vocab, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    few_ok = width % 256 == 0 and 8 * width * 2 <= 99 * 1024

    def launch(route, rows):
        if route == "pedantic":
            status = lib.cuteafd_vocabulary_head_launch_width(handle, x.data_ptr(), weight.data_ptr(), out.data_ptr(),
                                                              rows, stream)
        elif route == "tensor":
            status = lib.cuteafd_vocabulary_head_launch_tensor_op(handle, x.data_ptr(), weight.data_ptr(),
                                                                  out.data_ptr(), rows, stream)
        else:
            status = lib.cuteafd_vocab_head_rows(x.data_ptr(), weight.data_ptr(), out.data_ptr(), rows, width, vocab,
                                                 stream)
        assert status == 0, (route, rows, status)

    routes = ["pedantic", "tensor"] + (["few"] if few_ok else [])
    name = os.path.basename(os.path.dirname(os.path.dirname(snapshot.rstrip("/")))) or snapshot
    print(f"{name}: {head_name} [{vocab}, {width}]")
    # Errors over 16 rows (or fewer) against FP64 of the same BF16 operands.
    n = min(16, max_rows)
    ref = torch.cat([x[:n].double() @ weight[v:v + 16384].double().T for v in range(0, vocab, 16384)], dim=1)
    ref_log = torch.log_softmax(ref, dim=1)
    scale = ref.abs().amax().item()
    for route in routes:
        launch(route, n)
        torch.cuda.synchronize()
        got = out[:n].double()
        err = (got - ref).abs().amax().item() / scale
        top1 = (got.argmax(1) == ref.argmax(1)).float().mean().item()
        kl = (ref_log.exp() * (ref_log - torch.log_softmax(got, dim=1))).sum(1)
        print(f"  {route:8s} max|err|/max|logit| {err:.2e}  top-1 {100 * top1:.1f}%  KL mean {kl.mean().item():.2e} "
              f"max {kl.max().item():.2e}  max|dlogit| {(got - ref).abs().amax().item():.2e}")
    print(f"  ms (median single / burst mean of {runs * 4}): rows " + "  ".join(f"{r:>17s}" for r in routes))
    for rows in rows_list:
        cells = []
        for route in routes:
            if route == "few" and rows > 24:
                cells.append(f"{'-':>17s}")
                continue
            single, burst = timed(lambda: launch(route, rows), runs)
            cells.append(f"{single:8.3f} / {burst:6.3f}")
        print(f"  {rows:4d} " + "  ".join(cells))
    assert lib.cuteafd_v41_markov_destroy(handle) == 0
    del weight, out, ref, ref_log
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--rows", default="1,4,8,16,64")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    lib = bind(args.library)
    print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).multi_processor_count, "SMs")
    rows = [int(r) for r in args.rows.split(",")]
    for snapshot in args.snapshots:
        check(lib, snapshot, rows, args.runs)
    print("target heads: done")


if __name__ == "__main__":
    sys.exit(main())
