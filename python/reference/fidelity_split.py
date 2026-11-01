"""Standard-v2 split from fidelity_split.json, the one table Python and Rust both read.

A set's windows must be exactly its own per-block counts under the sealed ids;
each id's half is the sealed table, then decode-first alternation past each
block's sealed range. Mirrors rust/crates/cuteafd-bench/src/fidelity_dataset.rs.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

SPLIT_SHA256 = "2af36f78359d9ce7f37d0d2d4740c57ca11f0a77734bc13e774d37e416eee65b"
ROWS = 512


@lru_cache(maxsize=1)
def split() -> dict:
    data = json.loads((Path(__file__).with_name("fidelity_split.json")).read_text())
    joined = "\n".join(data["sealed"]["decode"] + data["sealed"]["prefill"])
    if hashlib.sha256(joined.encode()).hexdigest() != SPLIT_SHA256:
        raise ValueError("sealed standard split changed")
    return data


def standard_half(window_id: str) -> str:
    data = split()
    for half in ("decode", "prefill"):
        if window_id in data["sealed"][half]:
            return half
    match = re.fullmatch(r"([a-z])(\d{2,3})", window_id)
    if not match or match.group(1) not in data["blocks"].values():
        raise ValueError(f"window id {window_id} outside the standard split")
    prefix, index = match.group(1), int(match.group(2))
    anchor = data["anchor"]["id"]
    sealed = sum(1 for w in data["sealed"]["decode"] + data["sealed"]["prefill"]
                 if w.startswith(prefix) and w != anchor)
    if index < sealed:
        raise ValueError(f"window id {window_id} outside the standard split")
    return "decode" if (index - sealed) % 2 == 0 else "prefill"


def standard_layout(windows: list[dict]) -> dict:
    """Per-block counts derived from the windows; ids must be exactly those counts."""
    data = split()
    counts = Counter(w["block"] for w in windows)
    expected = set()
    for block, count in counts.items():
        if block not in data["blocks"]:
            raise ValueError(f"block {block} outside the standard split")
        anchored = int(block == data["anchor"]["block"])
        expected |= {f"{data['blocks'][block]}{i:02d}" for i in range(count - anchored)}
        if anchored:
            expected.add(data["anchor"]["id"])
    ids = [w["id"] for w in windows]
    if len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError("dataset window ids differ from its block counts (gap, duplicate or misnamed window)")
    for w in windows:
        if w["id"] != data["anchor"]["id"] and not w["id"].startswith(data["blocks"][w["block"]]):
            raise ValueError(f"window {w['id']} block {w['block']} differs")
        standard_half(w["id"])
    return dict(sorted(counts.items()))


def standard_ids(windows: list[dict], half: str) -> list[str]:
    standard_layout(windows)
    sealed = split()["sealed"][half]
    ids = [w["id"] for w in windows if standard_half(w["id"]) == half]
    return sorted(ids, key=lambda wid: sealed.index(wid) if wid in sealed else len(sealed))
