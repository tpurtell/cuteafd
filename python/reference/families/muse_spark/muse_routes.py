"""Real Muse Spark routes from the fidelity golden (schema cuteafd.muse.routes/1).

One directory per layer (``layerNNN``) holding ``index.json`` (the layer, top-k,
expert count, dtypes and a sha256 per file) and, per window ``wid``,
``{wid}.ids.bin`` (``int16 [T, top_k]``, the bias-selected experts) and
``{wid}.weights.bin`` (``float32 [T, top_k]``, renormalized weights). Every file
is checked against its index digest before use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "cuteafd.muse.routes/1"


def layer_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("layer"))


def load_layer(path: Path, weights: bool = False) -> tuple[int, dict[str, np.ndarray]]:
    """(layer, {window: ids [T, top_k]}) (or the weights), digests verified."""
    index = json.loads((path / "index.json").read_text())
    if index.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {index.get('schema')!r}, expected {SCHEMA}")
    top_k, experts = int(index["top_k"]), int(index["experts"])
    kind, dtype = ("weights", index["weights_dtype"]) if weights else ("ids", index["ids_dtype"])
    out = {}
    for name, digest in sorted(index["files"].items()):
        window, suffix = name.rsplit(".", 1)
        if suffix != kind:
            continue
        data = (path / f"{name}.bin").read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"{path / name}.bin: sha256 differs from index.json")
        array = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(-1, top_k)
        if not weights and (array.min() < 0 or array.max() >= experts):
            raise SystemExit(f"{path / name}.bin: expert id outside 0..{experts}")
        out[window] = array
    if not out:
        raise SystemExit(f"{path}: no {kind} files")
    return int(index["layer"]), out
