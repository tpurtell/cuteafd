#!/usr/bin/env python3
"""Compare the shared TP2 Rust fixture against independently decoded FP64 EXL3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def decode(packed: np.ndarray) -> np.ndarray:
    # Vectorize the independent lane oracle in sparkinfer's
    # tests/moe/test_fused_moe_trellis.py. MCG's FP16 codebook sum is intentional;
    # every operation after this weight reconstruction uses FP64.
    bits = packed.shape[-1] // 16
    u16 = packed.view(np.uint16).reshape(*packed.shape[:2], 8 * bits, 2)
    words = u16[..., 0].astype(np.uint32) | (u16[..., 1].astype(np.uint32) << np.uint32(16))
    block = np.empty((*packed.shape[:2], 16, 16), dtype=np.float16)
    for lane in range(32):
        row = (lane % 4) * 2
        rows = (row, row + 1, row + 8, row + 9)
        for weight in range(8):
            end = (lane * 8 + weight + 257) * bits
            first, last = (end - 16) // 32, (end - 1) // 32
            merged = (
                words[..., first % (8 * bits)].astype(np.uint64) << np.uint64(32)
            ) | words[..., last % (8 * bits)]
            window = (
                (merged >> np.uint64((last + 1) * 32 - end)) & np.uint64(65535)
            ).astype(np.uint32)
            value = (window.astype(np.uint64) * np.uint64(0xCBAC1FED)).astype(np.uint32)
            value = (value & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
            lo = (value & np.uint32(65535)).astype(np.uint16).view(np.float16)
            hi = (value >> np.uint32(16)).astype(np.uint16).view(np.float16)
            col = 2 * (lane // 8 + (4 if weight >= 4 else 0)) + ((lane >> 2) & 1)
            block[..., rows[weight % 4], col] = (lo + hi).astype(np.float16)
    return (
        block.transpose(0, 2, 1, 3)
        .reshape(packed.shape[0] * 16, packed.shape[1] * 16)
        .astype(np.float64)
    )


def had(value: np.ndarray) -> np.ndarray:
    work = value.reshape(-1, value.shape[-1] // 128, 128).copy()
    step = 1
    while step < 128:
        view = work.reshape(*work.shape[:-1], -1, 2, step)
        a, b = view[..., 0, :].copy(), view[..., 1, :].copy()
        view[..., 0, :], view[..., 1, :] = a + b, a - b
        step *= 2
    return work.reshape(value.shape) / np.sqrt(128.0)


def run(snapshot: Path, fixture_output: Path) -> dict:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((snapshot / "config.json").read_text())
    h, k = config["hidden_size"], config["num_experts_per_tok"]

    def tensor(name: str) -> np.ndarray:
        with safe_open(snapshot / index[name], framework="numpy") as shard:
            return shard.get_tensor(name)

    # Exact inputs/routes of local/shared_tp2_tests.rs, not fresh sampled data.
    x = np.asarray(
        [[(0.5, -0.5, 1.0, -1.0)[(r + c) % 4] for c in range(h)] for r in range(4)],
        dtype=np.float64,
    )
    reference = np.zeros((80, h), dtype=np.float64)
    used = sorted({i * 7 % config["n_routed_experts"] for i in range(80 * k)})
    for sequence, expert in enumerate(used):
        base = f"model.layers.0.mlp.experts.{expert}"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert int(tensor(base + f".{projection}.mcg")) & 0xFFFFFFFF == 0xCBAC1FED
        gate = had(
            had(x * tensor(base + ".gate_proj.suh").astype(np.float64))
            @ decode(tensor(base + ".gate_proj.trellis"))
        ) * tensor(base + ".gate_proj.svh").astype(np.float64)
        up = had(
            had(x * tensor(base + ".up_proj.suh").astype(np.float64))
            @ decode(tensor(base + ".up_proj.trellis"))
        ) * tensor(base + ".up_proj.svh").astype(np.float64)
        gate = np.minimum(gate, config["swiglu_limit"])
        up = np.clip(up, -config["swiglu_limit"], config["swiglu_limit"])
        activation = gate / (1.0 + np.exp(-gate)) * up
        down = had(
            had(activation * tensor(base + ".down_proj.suh").astype(np.float64))
            @ decode(tensor(base + ".down_proj.trellis"))
        ) * tensor(base + ".down_proj.svh").astype(np.float64)
        for row in range(80):
            for slot in range(k):
                if (row * k + slot) * 7 % config["n_routed_experts"] == expert:
                    weight = np.float32((slot + 1) / (k * (k + 1) / 2))
                    reference[row] += down[row % 4] * float(weight)
        if sequence % 16 == 0:
            print(f"FP64_PROGRESS {sequence + 1}/{len(used)}", flush=True)
    reference.astype("<f8").tofile(fixture_output / "pro-reference.f64")
    report = {}
    for rows in (1, 16, 80):
        expected = reference[:rows]
        values = {}
        for backend in ("tp1", "tp2"):
            actual = (
                np.fromfile(fixture_output / f"pro-{rows}.{backend}.f32", dtype="<f4")
                .reshape(rows, h)
                .astype(np.float64)
            )
            delta = actual - expected
            values[backend] = {
                "max_abs": float(np.abs(delta).max()),
                "rel_l2": float(np.linalg.norm(delta) / np.linalg.norm(expected)),
            }
        values["error_ratio"] = values["tp2"]["rel_l2"] / values["tp1"]["rel_l2"]
        report[rows] = values
        print("PRO_FP64", rows, json.dumps(values), flush=True)
    (fixture_output / "pro-fp64.json").write_text(json.dumps(report, indent=2) + "\n")
    assert all(values["error_ratio"] <= 2.0 for values in report.values()), report
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.fixture_output)
