"""After-norm error of quantized TP expert partials (Muse Spark), numpy only.

TP slices each expert's intermediate 1/N over the Sparks; each rank's partial
down-proj output for a (row, expert) is quantized for the wire, the
coordinator dequantizes, sums the N partials, then applies the expert's
input-scaled post_expert_norm (``x * bf16(1 + w)`` rounded to BF16, then RMS)
and the routing weight. This measures, per routed (row, expert), the relative
error of the normed output against the unquantized partial sum, and the
partial-to-sum magnitude ratio (cancellation): max and RMS over the sample.

Wire formats: ``fp8`` E4M3 with a power-of-two (UE8M0) scale per 32 elements;
``nvfp4`` E2M1 with an E4M3 scale per 16 elements (scale = amax / 6, the
hardware cvt's round-to-nearest-even grid).

Inputs: the official checkpoint's layer (experts, router, pre-expert
projection and norm) and real token embeddings through the layer's
pre_expert_proj + pre_expert_norm (a proxy until the golden's layer
activations exist; ``--activations`` takes [rows, 4096] .npy expert inputs).

    python3 ep_numerics.py SNAPSHOT --layer 0 --rows 128 --ranks 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PREFIX = "model.language_model."
E, INTER, EXPERTS, TOPK = 4096, 4096, 256, 8
UNITS = INTER // 128


def bf16(x: np.ndarray) -> np.ndarray:
    """Round F32 to BF16 (RNE), returned as F32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32)


class Snapshot:
    def __init__(self, path: Path):
        self.path = path
        self.index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str, rows=None) -> np.ndarray:
        """Tensor as F32 (BF16 widened exactly); `rows` selects leading indices."""
        import struct
        shard = self.path / self.index[name]
        with shard.open("rb") as stream:
            count = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(count))
            meta = header[name]
            base = 8 + count + meta["data_offsets"][0]
            shape, dtype = meta["shape"], meta["dtype"]
            item = {"BF16": 2, "F32": 4}[dtype]
            inner = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            def read(first, n):
                stream.seek(base + first * inner * item)
                raw = np.frombuffer(stream.read(n * inner * item), dtype=np.uint16 if item == 2 else np.float32)
                return (raw.astype(np.uint32) << 16).view(np.float32) if item == 2 else raw.copy()
            if rows is None:
                return read(0, shape[0]).reshape(shape)
            return np.stack([read(int(r), 1).reshape(shape[1:]) for r in rows])


def silu(x):
    return x / (1.0 + np.exp(-x))


def rms(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def post_norm(x, w, eps=1e-8):
    """Input-scaled: s = bf16(x * bf16(1 + w)); bf16(s * rsqrt(mean(s^2) + eps))."""
    s = bf16(bf16(x) * bf16(1.0 + w))
    return bf16(rms(s, eps))


def quant_fp8(x: np.ndarray) -> np.ndarray:
    """E4M3 per 32 with a power-of-two scale (UE8M0); returns dequantized F32."""
    blocks = x.reshape(*x.shape[:-1], -1, 32)
    amax = np.abs(blocks).max(-1, keepdims=True)
    scale = np.where(amax > 0, 2.0 ** np.ceil(np.log2(np.maximum(amax, 1e-30) / 448.0)), 1.0)
    return (e4m3(blocks / scale) * scale).reshape(x.shape)


def e4m3(v):
    """Round to the E4M3 grid (RNE; max 448; subnormals step 2^-9)."""
    v = np.clip(v, -448.0, 448.0)
    a = np.abs(v)
    exp = np.floor(np.log2(np.maximum(a, 2.0 ** -9)))
    exp = np.maximum(exp, -6.0)
    step = 2.0 ** (exp - 3)
    return np.sign(v) * np.round(a / step) * step


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1(v):
    """Round to the E2M1 grid (RNE among {0, .5, 1, 1.5, 2, 3, 4, 6})."""
    a = np.clip(np.abs(v), 0, 6.0)
    idx = np.abs(a[..., None] - E2M1).argmin(-1)
    return np.sign(v) * E2M1[idx]


def quant_nvfp4(x: np.ndarray) -> np.ndarray:
    """E2M1 per 16 with an E4M3 block scale amax/6; returns dequantized F32."""
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    amax = np.abs(blocks).max(-1, keepdims=True)
    scale = e4m3(np.where(amax > 0, amax / 6.0, 1.0))
    scale = np.where(scale > 0, scale, 2.0 ** -9)
    return (e2m1(blocks / scale) * scale).reshape(x.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--ranks", type=int, default=6)
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    snap = Snapshot(args.snapshot)
    root = f"{PREFIX}layers.{args.layer}.mlp."
    rng = np.random.default_rng(args.seed)
    if args.activations:
        x = np.load(args.activations).astype(np.float32)[:args.rows]
        hidden = None
    else:
        tokens = rng.integers(1000, 200000, size=args.rows)
        emb = snap.get(PREFIX + "embed_tokens.weight", rows=tokens)
        hidden = bf16(rms(emb, 1e-5))                          # weightless embedding norm
        w_pre = snap.get(root + "pre_expert_proj.weight")      # [4096, 8192]
        n_pre = snap.get(root + "pre_expert_norm.weight")
        x = bf16(hidden @ w_pre.T)
        x = bf16(bf16(rms(x, 1e-5)) * bf16(1.0 + n_pre))       # centered pre-expert norm
    # Routing on the pre-feedforward-normalized hidden (proxy: the embedding row).
    router = snap.get(root + "gate.weight")                    # F32 [256, 8192]
    bias = snap.get(root + "gate.e_score_correction_bias")
    if hidden is None:
        ids = rng.choice(EXPERTS, size=(len(x), TOPK))
    else:
        scores = 1.0 / (1.0 + np.exp(-(hidden @ router.T) * 0.17677669529663687))
        ids = np.argsort(-(scores + bias), axis=1, kind="stable")[:, :TOPK]
    norm_w = snap.get(root + "experts.post_expert_norm.weight")
    used = np.unique(ids)
    gate_up = snap.get(root + "experts.gate_up_proj", rows=used)   # [n, 8192, 4096]
    down = snap.get(root + "experts.down_proj", rows=used)         # [n, 4096, 4096]
    where = {int(e): i for i, e in enumerate(used)}
    stats = {fmt: {"err": [], "ratio": []} for fmt in ("fp8", "nvfp4")}
    for row in range(len(x)):
        for e in ids[row]:
            i = where[int(e)]
            gu = bf16(gate_up[i] @ x[row])
            act = bf16(bf16(silu(gu[:INTER])) * gu[INTER:])
            # TP partials: units of 128 intermediate columns dealt round-robin from offset e.
            owner = (np.arange(UNITS) + int(e)) % args.ranks
            partials = []
            for r in range(args.ranks):
                cols = np.concatenate([np.arange(u * 128, (u + 1) * 128) for u in np.flatnonzero(owner == r)])
                partials.append(down[i][:, cols] @ act[cols])           # F32 partial [4096]
            exact = post_norm(np.sum(partials, 0), norm_w)
            total = np.abs(np.sum(partials, 0))
            mag = np.max(np.abs(np.stack(partials)), axis=0)
            for fmt, quant in (("fp8", quant_fp8), ("nvfp4", quant_nvfp4)):
                summed = np.sum([quant(p) for p in partials], 0)
                got = post_norm(summed, norm_w)
                stats[fmt]["err"].append(np.linalg.norm(got - exact) / max(np.linalg.norm(exact), 1e-30))
                stats[fmt]["ratio"].append(np.median(mag / np.maximum(total, 1e-30)))
    print(f"layer {args.layer}, {len(x)} rows x top-{TOPK}, {args.ranks} ranks, "
          f"{'golden activations' if args.activations else 'embedding-proxy activations'}")
    print("format  rel L2 err after norm (per routed row): RMS / max   partial/sum magnitude (median per elem): median / p99")
    for fmt, s in stats.items():
        err, ratio = np.array(s["err"]), np.array(s["ratio"])
        print(f"{fmt:6}  {np.sqrt(np.mean(err ** 2)):.4e} / {err.max():.4e}   "
              f"{np.median(ratio):.3f} / {np.percentile(ratio, 99):.3f}")


if __name__ == "__main__":
    main()
