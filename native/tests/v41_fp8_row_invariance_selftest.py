"""Pinned bitwise row-invariance regression test for native V4.1 target FP8 AOT ops.

Verifies issue #2793: every target verify FP8 projection (including q_a and kv)
must produce bitwise-identical BF16 outputs for every row regardless of batch
width or capacity bucket across the full admitted decode/verify range 1..256,
both eagerly and under CUDA graph capture/replay, while prefill buckets
(64, 256) remain untouched at split_k=1. Pass libcuteafd_native.so as argv[1]
(or set CUTEAFD_NATIVE_LIB).
"""
import ctypes as C
import os
import sys
import torch


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("abi", "capacity", "k", "n")] + [
        (n, C.c_uint64) for n in ("scratch", "values", "row_scales", "mma_scales", "weight_scales")
    ]


class ReductionInfo(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("abi", "capacity", "expected_rows", "split_slices", "all_rows")]


# All V4.1 target-path FP8 projections (drafter-only `main` 15360x5120 is excluded).
# (name, k, n, decode_required_slices, decode_max_cap)
TARGET_PROJECTIONS = (
    ("q_a", 5120, 1280, 4, 256),
    ("kv", 5120, 512, 4, 256),
    ("o_b", 8192, 5120, 1, 80),
    ("o_b_tp2", 8192, 2560, 1, 80),
    ("engram", 6144, 25600, None, 256),
    ("ffn_up", 5120, 2304, None, 256),
    ("ffn_down", 2304, 5120, None, 256),
    ("q_b", 1280, 32768, None, 256),
    ("o_a", 32768, 8192, None, 256),
    ("index_q", 1280, 4096, None, 256),
    ("q_b_tp2", 1280, 16384, None, 256),
    ("ffn_tp2_up", 5120, 1152, None, 256),
    ("ffn_tp2_down", 1152, 5120, None, 256),
)

DECODE_CAPACITIES = (1, 16, 64, 80, 256)
TEST_ROWS = (1, 2, 5, 8, 16, 17, 23, 32, 37, 48, 64, 65, 80, 96, 128, 129, 192, 256)


def select_capacity(rows: int) -> int:
    for cap in DECODE_CAPACITIES:
        if rows <= cap:
            return cap
    raise ValueError(f"rows {rows} exceeds decode/verify capacities")


def main() -> None:
    lib_path = sys.argv[1] if len(sys.argv) > 1 else os.environ["CUTEAFD_NATIVE_LIB"]
    lib = C.CDLL(lib_path)
    P, I, U = C.c_void_p, C.c_int32, C.c_uint64

    def bind(name, argtypes):
        fn = getattr(lib, "cuteafd_v41_fp8_" + name)
        fn.argtypes, fn.restype = argtypes, I

        def checked(*args):
            status = fn(*args)
            if status != 0:
                raise RuntimeError(f"cuteafd_v41_fp8_{name} failed with status {status}")
        return checked

    prefill_red_fn = bind("matrix_reduction_info", [I, I, I, C.POINTER(ReductionInfo)])
    decode_info_fn = bind("decode_matrix_info", [I, I, I, C.POINTER(Info)])
    decode_red_fn = bind("decode_matrix_reduction_info", [I, I, I, C.POINTER(ReductionInfo)])
    decode_init_fn = bind("decode_matrix_initialize", [I, I, I, C.POINTER(P)])
    pack_fn = bind("matrix_pack_scales", [P, P, I, I, P])
    init_scratch_fn = bind("initialize_scratch", [P, P, U, P, P])
    launch_fn = bind("launch", [P, P, P, P, P, U, P, P, I, P])

    for name, k, n, required_slices, required_max_cap in TARGET_PROJECTIONS:
        # 1. Verify load-time reduction metadata contract for decode and prefill.
        recipes = {}
        for cap in DECODE_CAPACITIES:
            r = ReductionInfo()
            decode_red_fn(cap, k, n, C.byref(r))
            recipes[cap] = (r.expected_rows, r.split_slices, r.all_rows)
        if required_slices is not None:
            expected_tuple = (16, required_slices, 1)
            for cap, got in recipes.items():
                if cap <= required_max_cap:
                    assert got == expected_tuple, (
                        f"{name} ({k}x{n}) decode cap={cap}: expected {expected_tuple}, got {got}"
                    )
        if name in ("q_a", "kv"):
            for prefill_cap in (64, 256, 1024, 4096):
                pr = ReductionInfo()
                prefill_red_fn(prefill_cap, k, n, C.byref(pr))
                assert (pr.expected_rows, pr.split_slices, pr.all_rows) == (prefill_cap, 1, 0), (
                    f"{name} prefill cap={prefill_cap} changed: "
                    f"got ({pr.expected_rows}, {pr.split_slices}, {pr.all_rows})"
                )

        # 2. Prepare inputs, weights, packed scales, and per-capacity decode handles/scratch.
        torch.manual_seed(279300 + k + n)
        x_full = (torch.randn((256, k), device="cuda") * 0.25).bfloat16().contiguous()
        w = (torch.randn((n, k), device="cuda") * 0.125).to(torch.float8_e4m3fn).contiguous()
        raw_scales = torch.randint(123, 129, (n // 32, k // 32), device="cuda", dtype=torch.uint8).contiguous()

        handles, infos, scratches = {}, {}, {}
        stream = torch.cuda.current_stream().cuda_stream
        alpha = torch.empty(1, device="cuda", dtype=torch.float32)
        for cap in DECODE_CAPACITIES:
            info, handle = Info(), P()
            decode_info_fn(cap, k, n, C.byref(info))
            decode_init_fn(cap, k, n, C.byref(handle))
            scratch = torch.empty(info.scratch, device="cuda", dtype=torch.uint8)
            init_scratch_fn(handle, scratch.data_ptr(), info.scratch, alpha.data_ptr(), stream)
            handles[cap], infos[cap], scratches[cap] = handle, info, scratch

        packed_scales = torch.empty(infos[1].weight_scales, device="cuda", dtype=torch.uint8)
        pack_fn(raw_scales.data_ptr(), packed_scales.data_ptr(), k, n, stream)

        def run_eager(cap: int, x_in: torch.Tensor, rows: int, out_buf: torch.Tensor) -> None:
            s = torch.cuda.current_stream().cuda_stream
            launch_fn(
                handles[cap], x_in.data_ptr(), w.data_ptr(), packed_scales.data_ptr(),
                scratches[cap].data_ptr(), infos[cap].scratch, alpha.data_ptr(),
                out_buf.data_ptr(), rows, s,
            )

        # 3. Serial reference: compute every row 0..255 individually at capacity=1, rows=1.
        serial_out = torch.empty((256, n), device="cuda", dtype=torch.bfloat16)
        row_buf = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        for r in range(256):
            x_row = x_full[r:r + 1].contiguous()
            run_eager(1, x_row, 1, row_buf)
            serial_out[r].copy_(row_buf[0])
        torch.cuda.synchronize()
        assert torch.isfinite(serial_out).all()

        # 4. Batched verification across every admitted decode/verify width 1..256.
        batched_out = torch.empty((256, n), device="cuda", dtype=torch.bfloat16)
        for rows in TEST_ROWS:
            cap = select_capacity(rows)
            batched_out.fill_(float("nan"))
            run_eager(cap, x_full, rows, batched_out)
            torch.cuda.synchronize()
            assert torch.isfinite(batched_out[:rows]).all(), f"{name} non-finite at rows={rows}"
            if not torch.equal(batched_out[:rows].view(torch.uint16), serial_out[:rows].view(torch.uint16)):
                diffs = (batched_out[:rows].view(torch.uint16) != serial_out[:rows].view(torch.uint16)).sum().item()
                max_err = (batched_out[:rows].float() - serial_out[:rows].float()).abs().max().item()
                raise AssertionError(
                    f"ROW-INVARIANCE FAIL: {name} ({k}x{n}) rows={rows} cap={cap} "
                    f"recipes={recipes} diffs={diffs}/{rows * n} max_abs={max_err:.6g}"
                )

        # 5. CUDA graph capture/replay check on narrow fixed-split ops across m=16->17->64->128->256.
        if name in ("q_a", "kv"):
            for rows, cap in ((17, 64), (64, 64), (80, 80), (128, 256), (256, 256)):
                graph_out = torch.empty((cap, n), device="cuda", dtype=torch.bfloat16)
                x_mut = x_full[:cap].clone()
                run_eager(cap, x_mut, rows, graph_out)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream()
                with torch.cuda.graph(graph, stream=capture_stream):
                    run_eager(cap, x_mut, rows, graph_out)
                x_mut.mul_(-0.5)
                graph_out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for check_row in (0, 15, 16, rows - 1):
                    run_eager(1, x_mut[check_row:check_row + 1].contiguous(), 1, row_buf)
                    torch.cuda.synchronize()
                    assert torch.equal(
                        graph_out[check_row:check_row + 1].view(torch.uint16),
                        row_buf.view(torch.uint16),
                    ), f"GRAPH ROW-INVARIANCE FAIL: {name} rows={rows} cap={cap} row={check_row}"

        print(f"PASS {name:12s} K={k:5d} N={n:5d} recipes={recipes} rows=1..256", flush=True)


if __name__ == "__main__":
    main()
