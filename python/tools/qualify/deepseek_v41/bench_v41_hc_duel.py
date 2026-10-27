#!/usr/bin/env python3
"""Frozen native V4.1 mHC audit; run in coordinator-dev on queued GPU1.

Build with --build-only outside hardware locks. Default execution checks all
outputs and poisoned guards, then measures rotated CUDA-graph timing rounds.
--profile-kernel runs one warmed oracle launch inside a CUDA profiler region.
No AOT projection export is needed: small-row checks cover the CUDA fallback
project/finish path, not the separate serving AOT workspace entry point.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
FROZEN = Path(__file__).resolve().parent / "fixtures/v41_hc_ea60b6db.cu"
ORACLE_BLOB = "ea60b6db4f1aacb3fcf7e62bd886857494f7c962"
ARMS = ("current", "current_control", "candidate")
DEFAULT_SNAPSHOT = "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_oracle():
    data = FROZEN.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != ORACLE_BLOB:
        raise RuntimeError(f"frozen oracle changed: {blob} != {ORACLE_BLOB}")


def build(args):
    check_oracle()
    subprocess.run([sys.executable, str(ROOT / "scripts/build/assert-build-filesystem.py"),
                    str(args.build_dir)], check=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    candidate = ROOT / "native/families/deepseek_v41/cuda/v41_hc.cu"
    include = ROOT / "native/families/deepseek_v41/include"
    provenance = dict(oracle_blob=ORACLE_BLOB, oracle_sha256=sha(FROZEN),
                      candidate_sha256=sha(candidate),
                      header_sha256=sha(include / "cuteafd_v41_hc.h"),
                      nvcc=subprocess.check_output(["nvcc", "--version"], text=True), arms={})
    for arm in ARMS:
        source = candidate if arm == "candidate" else FROZEN
        library = args.build_dir / f"{arm}.so"
        command = ["nvcc", "-O3", "-DNDEBUG", "-std=c++17", "-arch=sm_120f",
                   "-lineinfo", "-shared", "-Xcompiler=-fPIC", "-I", str(include),
                   str(source), "-o", str(library)]
        subprocess.run(command, check=True, timeout=180)
        provenance["arms"][arm] = dict(command=command, source_sha256=sha(source),
                                       library_sha256=sha(library))
    (args.build_dir / "build.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=Path(DEFAULT_SNAPSHOT))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--site", choices=("attn", "ffn"), default="attn")
    parser.add_argument("--rows", default="1,2,6,16,17,80,1024,2048,2752,4096")
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--warmup", type=int, default=20, help="untimed graph replays")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--calls-per-graph", type=int, default=10)
    parser.add_argument("--profile-kernel", choices=("mixes", "post"))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.build_only:
        build(args)
        return
    check_oracle()
    if min(args.warmup, args.rounds, args.replays, args.calls_per_graph) < 1:
        parser.error("timing counts must be positive")
    rows_list = [int(v) for v in args.rows.split(",")]
    if any(not 1 <= r <= 4096 for r in rows_list):
        parser.error("rows must be in 1..4096")
    import torch
    from safetensors import safe_open

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this frozen duel targets RTX SM120")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    provenance = json.loads((args.build_dir / "build.json").read_text())
    if provenance["oracle_blob"] != ORACLE_BLOB:
        raise RuntimeError("build oracle identity mismatch")
    include = ROOT / "native/families/deepseek_v41/include/cuteafd_v41_hc.h"
    if sha(include) != provenance["header_sha256"]:
        raise RuntimeError("header changed since build")
    libs = {}
    for arm in ARMS:
        path = args.build_dir / f"{arm}.so"
        source = ROOT / "native/families/deepseek_v41/cuda/v41_hc.cu" if arm == "candidate" else FROZEN
        recorded = provenance["arms"][arm]
        if sha(path) != recorded["library_sha256"] or sha(source) != recorded["source_sha256"]:
            raise RuntimeError(f"stale or modified {arm}; rebuild outside hardware locks")
        lib = C.CDLL(str(path.resolve()), mode=C.RTLD_LOCAL)
        lib.cuteafd_v41_hc_mixes.argtypes = [C.c_void_p] * 7 + [C.c_int32, C.c_void_p]
        lib.cuteafd_v41_hc_post.argtypes = [C.c_void_p] * 5 + [C.c_int32, C.c_void_p]
        lib.cuteafd_v41_hc_pre.argtypes = [C.c_void_p] * 3 + [C.c_int32, C.c_void_p]
        libs[arm] = lib
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    weights, weight_info = [], {}
    for suffix in ("fn", "scale", "base"):
        name = f"layers.{args.layer}.hc_{args.site}_{suffix}"
        with safe_open(args.snapshot / index[name], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name).float().contiguous()
        weight_info[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
        weights.append(value.cuda())
    fn, scale, base = weights
    if [v.numel() for v in weights] != [24 * 20480, 3, 24]:
        raise RuntimeError("unexpected HC geometry")
    # First-layer residuals originate in embed.weight. Match the RMS of a
    # bounded, deterministic real embedding sample rather than guessing scale.
    with safe_open(args.snapshot / index["embed.weight"], framework="pt", device="cpu") as handle:
        embedding = handle.get_slice("embed.weight")[256:512].float()
    residual_std = embedding.square().mean().sqrt().item()
    report = dict(argv=sys.argv, oracle_blob=ORACLE_BLOB, build=provenance,
                  snapshot=str(args.snapshot), weights=weight_info, seed=args.seed,
                  residual=dict(distribution="seeded normal, BF16", std=residual_std,
                                scale_source="RMS embed.weight[256:512], first-layer reference"),
                  gpu=str(torch.cuda.get_device_properties(0)), torch=torch.__version__,
                  cuda=torch.version.cuda, timing=dict(warmup=args.warmup, rounds=args.rounds,
                  replays=args.replays, calls_per_graph=args.calls_per_graph), cases=[])
    print(json.dumps({k: v for k, v in report.items() if k not in ("build", "cases")}), flush=True)

    def checked_call(function, tensors, rows):
        status = function(*[v.data_ptr() for v in tensors], rows, stream.cuda_stream)
        if status:
            raise RuntimeError(f"CUDA status {status}, rows {rows}")

    def guarded(rows, width, dtype):
        parent = torch.full((rows + 4, width), float("nan"), device="cuda", dtype=dtype)
        return parent, parent[2:rows + 2]

    def bits(value):
        return value.view(torch.int16 if value.dtype == torch.bfloat16 else torch.int32)

    def guards_ok(parent, rows):
        expected = torch.full_like(parent[:2], float("nan"))
        return torch.equal(bits(parent[:2]), bits(expected)) and torch.equal(bits(parent[rows + 2:]), bits(expected))

    for rows in ([2752] if args.profile_kernel else rows_list):
        residual = (torch.randn((rows, 20480), device="cuda") * residual_std).bfloat16()
        sublayer = (torch.randn((rows, 5120), device="cuda") * residual_std).bfloat16()
        coeff = [torch.empty((rows, width), device="cuda") for width in (4, 4, 16)]
        torch.cuda.synchronize()
        checked_call(libs["current"].cuteafd_v41_hc_mixes, [residual, *weights, *coeff], rows)
        stream.synchronize()
        for kernel in ([args.profile_kernel] if args.profile_kernel else ["mixes", "post", "pre_check"]):
            outputs, parents, calls = {}, {}, {}
            for arm in ARMS:
                widths = (4, 4, 16) if kernel == "mixes" else (20480,) if kernel == "post" else (5120,)
                dtype = torch.float32 if kernel == "mixes" else torch.bfloat16
                pairs = [guarded(rows, width, dtype) for width in widths]
                parents[arm], outputs[arm] = [p[0] for p in pairs], [p[1] for p in pairs]
                tensors = ([residual, *weights, *outputs[arm]] if kernel == "mixes" else
                           [sublayer, residual, coeff[1], coeff[2], *outputs[arm]] if kernel == "post" else
                           [residual, coeff[0], *outputs[arm]])
                function = getattr(libs[arm], "cuteafd_v41_hc_" + ("pre" if kernel == "pre_check" else kernel))
                calls[arm] = lambda f=function, ts=tensors: checked_call(f, ts, rows)
            torch.cuda.synchronize()
            if args.profile_kernel:
                for _ in range(3):
                    calls["current"]()
                stream.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                calls["current"]()
                stream.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                print(f"profiled oracle {kernel} rows={rows}", flush=True)
                continue
            for call in calls.values():
                call()
            stream.synchronize()
            for arm in ARMS:
                for a, b, parent in zip(outputs["current"], outputs[arm], parents[arm]):
                    if not torch.isfinite(b).all().item() or not torch.equal(bits(a), bits(b)):
                        raise RuntimeError(f"nonexact/nonfinite {arm} {kernel} rows={rows}")
                    if not guards_ok(parent, rows):
                        raise RuntimeError(f"poisoned guard overwritten {arm} {kernel} rows={rows}")
            item = dict(rows=rows, kernel=kernel, byte_exact=True, poisoned_guards=True)
            if kernel == "pre_check":
                report["cases"].append(item)
                continue
            graphs = {}
            for arm in ARMS:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.calls_per_graph):
                        calls[arm]()
                graphs[arm] = graph
            for _ in range(args.warmup):
                for graph in graphs.values():
                    with torch.cuda.stream(stream):
                        graph.replay()
            stream.synchronize()
            samples = {arm: [] for arm in ARMS}
            for round_index in range(args.rounds):
                order = list(ARMS[round_index % 3:] + ARMS[:round_index % 3])
                if round_index % 2:
                    order.reverse()
                for arm in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record()
                        for _ in range(args.replays):
                            graphs[arm].replay()
                        end.record()
                    end.synchronize()
                    samples[arm].append(start.elapsed_time(end) * 1000 / args.replays / args.calls_per_graph)
            medians = {arm: statistics.median(values) for arm, values in samples.items()}
            item.update(us=medians, samples_us=samples,
                        control_spread_pct=100 * (medians["current_control"] / medians["current"] - 1),
                        candidate_speedup=medians["current"] / medians["candidate"])
            byte_floor = rows * (40960 + 96) + 24 * 20480 * 4 + 108 if kernel == "mixes" else rows * (40960 + 10240 + 80 + 40960)
            item.update(ideal_dram_bytes=byte_floor, ideal_dram_us=byte_floor / 1.79e6)
            report["cases"].append(item)
            print(json.dumps(item), flush=True)
        # Release graph references before the next shape's buffers are allocated.
        if not args.profile_kernel:
            del graphs, graph
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
