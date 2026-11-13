#!/usr/bin/env python3
"""Unpinned diagnostic wrapper around the pinned Kimi K3.1 reference runner.

The published reference runner (``python/reference/families/kimi_k3/golden.py``)
is imported and executed unmodified; this script only rebinds two module
globals before ``extend_reference`` builds the DSA adapter, so the runner's own
source pins, window seal and provenance checks still cover exactly the files
they always covered. Nothing here is a qualification reference.

With ``CUTEAFD_KIMI_REF_DSA_DUMP`` set, each requested DSA layer records the
reference's per-position selection over its causal frontier: the 2048 selected
logical token ids (ascending, the runner's order), their F32 scores, the cutoff
(2048th-ranked) score, and the next 256 just-below-cutoff slots. That is what
the engine's ``CUTEAFD_KIMI_DSA_SCORES`` dump is compared against.

``CUTEAFD_KIMI_REF_DSA_QDQ=1`` re-runs the same window with the served indexer
contract applied to the reference's own F32 indexer projections: per-(row,head)
E4M3 query QDQ over 128 dims with a UE8M0 power-of-two scale (amax floor
1e-10), and per-token E4M3 key QDQ with the key rounded to BF16 first (amax
floor 1e-4). The head weights already carry the equivalent weight_scale, so
dequantized q/k reproduce the served scores up to reduction order. The
projections themselves remain the reference's F32 upcast arithmetic; the
serving producer accumulates them from BF16 weights, which this arm does not
emulate.

Selection is the runner's own rule (stable descending score sort, exact ties to
lower ids, selected ids ascending). Candidate columns are 0..position, one
position per row; positions are indexes into the window token sequence, the
same convention as the engine's ``DSA positions``.

Stop behavior: after the deepest requested layer's last tile is recorded the
wrapper raises an internal stop, so the runner never reaches output scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))

import shape_invariant  # noqa: E402
import torch  # noqa: E402
from families.mistral_large4 import ops  # noqa: E402
from families.kimi_k3 import golden, k31  # noqa: E402

ENV_DIR = "CUTEAFD_KIMI_REF_DSA_DUMP"
ENV_LAYERS = "CUTEAFD_KIMI_REF_DSA_LAYERS"
ENV_FROM = "CUTEAFD_KIMI_REF_DSA_FROM"
ENV_QDQ = "CUTEAFD_KIMI_REF_DSA_QDQ"

FLOAT8_E4M3_MAX = 448.0
QUERY_AMAX_FLOOR = 1e-10
KEY_AMAX_FLOOR = 1e-4
NEAR_BAND = 256

DIR = os.environ.get(ENV_DIR)
LAYERS = frozenset(int(part) for part in os.environ.get(ENV_LAYERS, "").split(",") if part.strip())
FROM = int(os.environ.get(ENV_FROM, "0"))
QDQ = os.environ.get(ENV_QDQ, "0") not in ("", "0", "false", "False")
STOP_LAYER = max(LAYERS) if LAYERS else None

RECORDS: dict[int, list] = {}
QDQ_CALLS = 0


class _StopDiagnostic(Exception):
    """Internal: the deepest requested layer has been recorded."""


def _dsa_site(frame):
    """Return (layer, start, length) for a call made by the K3.1 DSA forward."""
    attn = frame.f_locals.get("self") if frame is not None else None
    if attn is None or type(attn).__name__ != "DSAAttention":
        return None
    layer = getattr(attn, "layer_idx", None)
    start = frame.f_locals.get("start")
    length = frame.f_locals.get("length")
    if not (isinstance(layer, int) and isinstance(start, int) and isinstance(length, int)):
        return None
    return layer, start, length


def _ue8m0_ceil(scale: torch.Tensor) -> torch.Tensor:
    """Bit-exact replica of the device's ``pow2_ceil_ue8m0``.

    Mirrors ``b12x/_lib/intrinsics.py:pow2_ceil_ue8m0_torch``: round positive
    F32 scales up to a power of two by bumping the exponent whenever the
    mantissa is nonzero (zero maps to zero).
    """
    bits = scale.to(torch.float32).contiguous().view(torch.int32)
    mantissa = bits & 0x007FFFFF
    bumped = torch.where(mantissa != 0, (bits + 0x00800000) & 0x7F800000, bits)
    return bumped.view(torch.float32)


def _quantize_dequantize(value: torch.Tensor, floor: float) -> torch.Tensor:
    """E4M3 payload with a UE8M0 power-of-two scale, back to F32."""
    amax = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(floor)
    scale = _ue8m0_ceil(amax / FLOAT8_E4M3_MAX)
    payload = (value.float() / scale).to(torch.float8_e4m3fn).to(torch.float32)
    return payload * scale


_original_topk = shape_invariant.fixed_index_topk
_original_reduction = ops.indexer_score_reduction


def _reduction(query, key, head_weights):
    global QDQ_CALLS
    if QDQ and _dsa_site(sys._getframe(1)) is not None:
        # The producer rounds the normalized key to BF16 before its FP8 cast;
        # the query stays F32 until the QDQ scale.
        query = _quantize_dequantize(query, QUERY_AMAX_FLOOR)
        key = _quantize_dequantize(key.to(torch.bfloat16).to(torch.float32), KEY_AMAX_FLOOR)
        QDQ_CALLS += 1
    return _original_reduction(query, key, head_weights)


def _topk(scores, slots):
    result = _original_topk(scores, slots)
    site = _dsa_site(sys._getframe(1))
    if site is None or DIR is None:
        return result
    layer, start, length = site
    if LAYERS and layer not in LAYERS:
        return result
    rows = int(scores.shape[0])
    ids = result.indices.detach().cpu()
    values = result.values.detach().cpu()
    for offset in range(rows):
        position = start + offset
        if position < FROM or position >= length:
            continue
        chosen = [int(value) for value in ids[offset]]
        chosen_scores = [float(value) for value in values[offset]]
        if not all(value == value and value != float("-inf") for value in chosen_scores):
            continue
        order = torch.argsort(scores[offset], descending=True, stable=True)
        near = [(int(index), float(scores[offset][index])) for index in order[slots:slots + NEAR_BAND]]
        RECORDS.setdefault(layer, []).append({
            "layer": layer,
            "position": position,
            "candidates": position + 1,
            "selected": chosen,
            "scores": chosen_scores,
            "cutoff": min(chosen_scores),
            "near": near,
        })
    if STOP_LAYER is not None and layer == STOP_LAYER and start + rows >= length:
        raise _StopDiagnostic(f"recorded layer {layer} through position {length - 1}")
    return result


def _install():
    shape_invariant.fixed_index_topk = _topk
    ops.indexer_score_reduction = _reduction
    original_extend = k31.extend_reference

    def extend(reference, config):
        # Rebind inside the runner's own import scope so the DSA adapter binds
        # the instrumented symbols; the pinned sources stay untouched.
        shape_invariant.fixed_index_topk = _topk
        ops.indexer_score_reduction = _reduction
        return original_extend(reference, config)

    k31.extend_reference = extend


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_outputs():
    if DIR is None:
        return
    base = Path(DIR) / ("qdq" if QDQ else "")
    base.mkdir(parents=True, exist_ok=True)
    written = {}
    for layer, rows in sorted(RECORDS.items()):
        rows.sort(key=lambda row: row["position"])
        path = base / f"layer-{layer:02}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        written[str(path.relative_to(base))] = {
            "rows": len(rows),
            "first_position": rows[0]["position"],
            "last_position": rows[-1]["position"],
            "sha256": _digest(path),
        }
    reference = ROOT / "python/reference"
    manifest = {
        "schema": "cuteafd.kimi.ref-dsa-scores/1",
        "diagnostic_only": True,
        "qualification_reference": False,
        "qdq": QDQ,
        "layers": sorted(RECORDS),
        "from": FROM,
        "near_band": NEAR_BAND,
        "positions": "index into the window token sequence; candidates are 0..position",
        "selection": "stable descending F32 score sort, exact ties to lower ids, selected ids ascending",
        "scores": "indexer_score_reduction output (F32 weighted ReLU dot products) before masking",
        "arithmetic": ("served E4M3/UE8M0 QDQ applied to the reference's F32 indexer projections: "
                       "per-(row,head) query scale (amax floor 1e-10), per-token key scale on the "
                       "BF16-rounded normalized key (amax floor 1e-4); the per-head weights and "
                       "the residual stream remain the reference's F32 arithmetic (serving folds "
                       "weight_scale into BF16 head weights)"
                       if QDQ else
                       "ideal F32 indexer arithmetic; the F32 oracle does not emulate serving "
                       "E4M3/UE8M0 QDQ or BF16 head weights"),
        "source_sha256": {
            name: _digest(reference / name)
            for name in ("families/kimi_k3/k31.py", "families/kimi_k3/golden.py",
                         "families/mistral_large4/ops.py", "shape_invariant.py")
        },
        "files": written,
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"reference DSA selection dump: {len(written)} layer files under {base}", flush=True)


def _problems():
    """Refuse a silently empty or silently F32-only diagnostic run."""
    if DIR is None:
        return []
    problems = []
    if not RECORDS:
        problems.append("no DSA layer recorded; the instrumentation site never matched")
    missing = sorted(LAYERS - set(RECORDS))
    if missing:
        problems.append(f"requested layers {missing} recorded nothing")
    if QDQ and QDQ_CALLS == 0:
        problems.append("QDQ arm applied to no DSA reduction; refusing to report an F32 dump as QDQ")
    return problems


def main():
    _install()
    status = 0
    try:
        golden.main()
    except _StopDiagnostic as stop:
        print(f"reference DSA selection dump stopped: {stop}", flush=True)
    except BaseException:
        status = 1
        raise
    finally:
        _write_outputs()
    problems = _problems()
    for problem in problems:
        print(f"reference DSA selection dump invalid: {problem}", file=sys.stderr, flush=True)
    return 1 if problems else status


if __name__ == "__main__":
    sys.exit(main())
