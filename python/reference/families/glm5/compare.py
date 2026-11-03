#!/usr/bin/env python3
"""Compare two golden directories (golden.py outputs): per-layer cosine for
layers both saved, logits top-1 agreement, mean KL(a || b), and each side's
next-token accuracy and mean NLL on the prompt (the robust quality measure
when routes flip on rounding-level changes).

  compare.py A_DIR B_DIR
"""
import sys
from pathlib import Path

import numpy as np


def bf16(path: Path) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32)


def cosine_error(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("activation shapes differ or contain nonfinite values")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    cos = float(x @ y / (nx * ny)) if nx and ny else float(np.array_equal(x, y))
    rel = float(np.linalg.norm(x - y) / max(nx, 1e-30))
    return cos, rel


def compare_trace(a, b):
    """Reference F32 when available, native BF16, exact int32 router identity."""
    files_a = {p.name for p in a.glob("layer*.bin") if not p.name.endswith(".f32.bin")}
    files_b = {p.name for p in b.glob("layer*.bin") if not p.name.endswith(".f32.bin")}
    if not files_a or files_a != files_b:
        raise ValueError("activation trace component sets differ or are empty")
    for name in sorted(files_b, key=lambda name: (int(name.split("-", 1)[0][5:]), name)):
        if name.endswith("-route_ids.bin"):
            x, y = (np.fromfile(folder / name, dtype="<i4") for folder in (a, b))
            print(f"{name}: ids_equal {np.array_equal(x, y)} ids_set_equal {np.array_equal(np.sort(x), np.sort(y))}")
        else:
            reference = (a / name).with_suffix(".f32.bin")
            x = np.fromfile(reference, dtype="<f4") if reference.exists() else bf16(a / name)
            y = bf16(b / name)
            cos, rel = cosine_error(x, y)
            print(f"{name}: cosine {cos:.9f} rel_l2 {rel:.6e}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--activation-trace":
        a, b = map(Path, args[1:])
        compare_trace(a, b)
        return
    a, b = map(Path, args)
    for layer in sorted(p.name for p in b.glob("layer*.bin")):
        x, y = bf16(a / layer), bf16(b / layer)
        cos, rel = cosine_error(x, y)
        print(f"{layer}: cosine {cos:.6f} rel_l2 {rel:.3e}")
    tokens = np.frombuffer((a / "tokens.bin").read_bytes(), dtype=np.int32)
    la = np.frombuffer((a / "logits.bin").read_bytes(), dtype=np.float32).reshape(len(tokens), -1)
    lb = np.frombuffer((b / "logits.bin").read_bytes(), dtype=np.float32).reshape(len(tokens), -1)
    agree = float((la.argmax(1) == lb.argmax(1)).mean())
    def logsoftmax(l):
        l = l - l.max(1, keepdims=True)
        return l - np.log(np.exp(l).sum(1, keepdims=True))
    pa, pb = logsoftmax(la.astype(np.float64)), logsoftmax(lb.astype(np.float64))
    kl = float((np.exp(pa) * (pa - pb)).sum(1).mean())
    next_a = float((la.argmax(1)[:-1] == tokens[1:]).mean())
    next_b = float((lb.argmax(1)[:-1] == tokens[1:]).mean())
    rows = np.arange(len(tokens) - 1)
    nll_a, nll_b = float(-pa[rows, tokens[1:]].mean()), float(-pb[rows, tokens[1:]].mean())
    print(f"logits: top-1 agreement {agree:.4f} | mean KL {kl:.5f} nats | next-token accuracy {next_a:.4f} vs {next_b:.4f}"
          f" | mean NLL {nll_a:.4f} vs {nll_b:.4f}")


if __name__ == "__main__":
    main()
