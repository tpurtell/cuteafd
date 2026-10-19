#!/usr/bin/env python3
"""Backfill golden window NLL from stored F32 logits and token IDs, without inference."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from fidelity_windows import canonical, scored_logits_nll


def backfill(golden: Path) -> dict:
    golden = golden.resolve()
    for root in (golden, golden.parent):
        if any((root / name).exists() for name in
               ("golden-seal.json", "full.sha256.json", "reference-validation.json")):
            raise ValueError("refusing to modify a sealed or validated golden")
    meta_path = golden / "meta.json"
    meta = json.loads(meta_path.read_text())
    entries = meta["windows"]
    if not entries or len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("golden must contain distinct windows")
    writes, folders = [], set()
    loss, count = 0.0, 0
    for entry in entries:
        folder = (golden / entry["path"]).resolve()
        if folder == golden or not folder.is_relative_to(golden) or folder in folders:
            raise ValueError("invalid golden window path")
        folders.add(folder)
        for name in ("tokens.bin", "logits.bin", "meta.json"):
            if not (folder / name).resolve().is_relative_to(golden):
                raise ValueError("golden file escapes its root")
        tokens_path, logits_path = folder / "tokens.bin", folder / "logits.bin"
        if tokens_path.stat().st_size % 4:
            raise ValueError("golden tokens extent mismatch")
        tokens = np.fromfile(tokens_path, dtype="<i4")
        positions, vocab = entry["positions"], entry["vocab"]
        if (not positions or any(a >= b for a, b in zip(positions, positions[1:]))
                or not isinstance(vocab, int) or vocab <= 0
                or logits_path.stat().st_size != len(positions) * vocab * 4):
            raise ValueError("golden logits geometry/extent mismatch")
        logits = np.memmap(logits_path, dtype="<f4", mode="r", shape=(len(positions), vocab))
        score = scored_logits_nll(logits, tokens, positions)
        del logits
        entry.update(score)
        writes.append((folder / "meta.json", canonical(entry) + b"\n"))
        loss += score["nll"] * score["scored_tokens"]
        count += score["scored_tokens"]
    nll = loss / count
    if (not math.isclose(meta["nll"], nll, rel_tol=1e-9, abs_tol=1e-8)
            or ("scored_tokens" in meta and meta["scored_tokens"] != count)):
        raise ValueError("aggregate NLL/scored_tokens mismatch; no metadata written")
    meta["scored_tokens"] = count
    writes.append((meta_path, canonical(meta) + b"\n"))
    # Validate every window and the aggregate before replacing any metadata.
    for path, data in writes:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".nll-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return {"windows": len(entries), "nll": nll, "scored_tokens": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True, type=Path, help="golden directory containing meta.json")
    args = parser.parse_args()
    print(json.dumps(backfill(args.golden), sort_keys=True))


if __name__ == "__main__":
    main()
