#!/usr/bin/env python3
"""Shared three-component calibration corpus renderer, preparer, and loader.

Single source of truth for ``wrldsuksgo2mars/agentic-quant-panel-v2`` across
MC quantizations, QMLA calibrations, and other future quant pipelines.

Canonical calibration stream order:
1. ``agentic``: single-turn prompts rendered once with the checkpoint's own
   chat template (``add_generation_prompt=True``), one unpadded prompt per batch.
2. ``exl3_standard``: raw text packed to 250 x 2,048 rows via
   ``reproduction/pack_exl3_standard.py`` (contract ``exl3-standard-pack-v1``).
3. ``long_context``: 16K and 32K multi-turn agentic traces (tool calls, tool
   results, optional reasoning) plus long-document QA, rendered once with the
   checkpoint's own chat template (``add_generation_prompt=False``), unpadded,
   one record per batch, honoring ``calibration`` vs ``heldout`` splits.

Flipping to the three-component panel once published is one edit: set
``THREE_COMPONENT_REVISION`` and ``THREE_COMPONENT_SHA256SUMS_SHA256`` below.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from kimi_k3 import prepare_corpus as v1_panel
from kimi_k3.uniform_k2.storage import atomic_json, checkpoint_io, fsync_directory

PANEL_REPO = "wrldsuksgo2mars/agentic-quant-panel-v2"
TWO_COMPONENT_REVISION = "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"
TWO_COMPONENT_SHA256SUMS_SHA256 = "23f07f7dbe7b8248a09891b6ae2475371f4653e499a9b8f930e267ff3d5c6d57"

# Single edit point once the three-component revision is published:
THREE_COMPONENT_REVISION: str | None = None
THREE_COMPONENT_SHA256SUMS_SHA256: str | None = None

COMPONENT_ORDER: tuple[str, ...] = ("agentic", "exl3_standard", "long_context")
BASE_COMPONENTS: tuple[str, ...] = ("agentic", "exl3_standard")
SPLITS: tuple[str, ...] = ("calibration", "screening", "heldout")
LONG_CONTEXT_SUBSETS: tuple[str, ...] = ("16k", "32k")
LONG_CONTEXT_SPLITS: tuple[str, ...] = ("calibration", "heldout")
LONG_CONTEXT_SCHEMA = "agentic-quant-panel-v2-long-context-v1"
EXL3_STANDARD_PACKER = "reproduction/pack_exl3_standard.py"
EXL3_STANDARD_CONTRACT = "exl3-standard-pack-v1"
PREPARED_SCHEMA = "cuteafd-panel-corpus-v3"

# Disjoint rendered-token bounds per length class ([0.70x, 1.35x] of 16,384 / 32,768).
LONG_CONTEXT_LENGTH_BOUNDS: dict[str, tuple[int, int]] = {
    "16k": (11_468, 22_118),
    "32k": (22_937, 44_236),
}

_FORBIDDEN_KEYS = frozenset({"input_ids", "token_ids", "tokens"})
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

canonical = v1_panel.canonical
sha256_file = v1_panel.sha256_file
ids_sha256 = v1_panel.ids_sha256
software_identity = v1_panel.software_identity


def default_panel_revision() -> str:
    return THREE_COMPONENT_REVISION or TWO_COMPONENT_REVISION


def default_panel_sha256sums() -> str | None:
    if THREE_COMPONENT_REVISION is None:
        return TWO_COMPONENT_SHA256SUMS_SHA256
    return THREE_COMPONENT_SHA256SUMS_SHA256


def default_components() -> tuple[str, ...]:
    return COMPONENT_ORDER if THREE_COMPONENT_REVISION is not None else BASE_COMPONENTS


def default_dataset() -> dict[str, Any]:
    entry: dict[str, Any] = {"repo_id": PANEL_REPO, "revision": default_panel_revision()}
    sums = default_panel_sha256sums()
    if sums is not None:
        entry["sha256sums_sha256"] = sums
    return entry


def resolve_panel_policy(
    *,
    panel_revision: str | None = None,
    components: Sequence[str] | None = None,
    full: bool = True,
) -> tuple[str, str, tuple[str, ...]]:
    """Resolve ``(repo_id, revision, canonical_components)`` from the module pin."""
    if THREE_COMPONENT_REVISION is not None and not _HEX40.match(THREE_COMPONENT_REVISION):
        raise ValueError(f"THREE_COMPONENT_REVISION must be a 40-char SHA, got {THREE_COMPONENT_REVISION!r}")
    rev = default_panel_revision() if panel_revision is None else panel_revision
    if not isinstance(rev, str) or not _HEX40.match(rev):
        raise ValueError(f"panel_revision must be a 40-char lowercase SHA, got {rev!r}")

    if components is None:
        ordered = BASE_COMPONENTS if rev == TWO_COMPONENT_REVISION else COMPONENT_ORDER
    else:
        seen = []
        for name in components:
            if not isinstance(name, str) or name not in COMPONENT_ORDER:
                raise ValueError(f"unknown panel component {name!r}; allowed: {COMPONENT_ORDER}")
            if name not in seen:
                seen.append(name)
        if not seen:
            raise ValueError("components must be nonempty")
        ordered = tuple(c for c in COMPONENT_ORDER if c in seen)

    if "long_context" in ordered and rev == TWO_COMPONENT_REVISION:
        raise ValueError(f"long_context is not present in two-component revision {TWO_COMPONENT_REVISION}")

    if full:
        if THREE_COMPONENT_REVISION is not None:
            if rev == TWO_COMPONENT_REVISION or ordered != COMPONENT_ORDER:
                raise ValueError(
                    f"full calibration requires all three components {COMPONENT_ORDER} once "
                    f"{THREE_COMPONENT_REVISION} is published; got revision={rev} components={ordered}"
                )
        elif not set(BASE_COMPONENTS) <= set(ordered):
            raise ValueError(f"full calibration requires at least {BASE_COMPONENTS}; got {ordered}")
    return PANEL_REPO, rev, ordered


def component_input_ids_sha256(sequences) -> str:
    """SHA256 of little-endian int64 token IDs concatenated in record order."""
    digest = hashlib.sha256()
    for seq in sequences:
        buf = array("q", seq)
        if sys.byteorder != "little":
            buf.byteswap()
        digest.update(buf.tobytes())
    return digest.hexdigest()


def max_context_from_config(source_snapshot: Path | str) -> int | None:
    cfg_path = Path(source_snapshot) / "config.json"
    if not cfg_path.is_file():
        return None
    doc = json.loads(cfg_path.read_text(encoding="utf-8"))
    for scope in (doc.get("text_config") or {}, doc):
        if isinstance(scope, Mapping):
            for key in ("max_position_embeddings", "model_max_length", "max_seq_len"):
                val = scope.get(key)
                if type(val) is int and val > 0:
                    return val
    return None


def _content_hash(messages, tools) -> str:
    return hashlib.sha256(canonical({"messages": messages, "tools": tools})).hexdigest()


def validate_long_context_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a model-neutral ``long_context`` record before template rendering."""
    if not isinstance(record, Mapping):
        raise ValueError("long_context record must be a mapping")
    if _FORBIDDEN_KEYS.intersection(record):
        raise ValueError(f"long_context record must not store token IDs: {sorted(_FORBIDDEN_KEYS & set(record))}")
    rid, split, length_class = record.get("id"), record.get("split"), record.get("length_class")
    if not isinstance(rid, str) or not rid or split not in LONG_CONTEXT_SPLITS or length_class not in LONG_CONTEXT_LENGTH_BOUNDS:
        raise ValueError(f"invalid long_context id/split/length_class: {rid!r}/{split!r}/{length_class!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[0].get("role") not in ("system", "user"):
        raise ValueError(f"{rid}: long_context requires nonempty messages starting with system or user")

    tools = record.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or any(
            not isinstance(t, Mapping) or not isinstance((t.get("function") or {}).get("name"), str) or not t["function"]["name"]
            for t in tools
        ):
            raise ValueError(f"{rid}: invalid tools schema list")

    actual_hash = _content_hash(messages, tools)
    if record.get("content_sha256") is not None and record["content_sha256"] != actual_hash:
        raise ValueError(f"{rid}: long_context content_sha256 mismatch")

    has_tool_calls = has_tool_results = has_reasoning = False
    pending: set[str] = set()
    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping) or msg.get("role") not in _VALID_ROLES:
            raise ValueError(f"{rid}: invalid message[{idx}]")
        role = msg["role"]
        if role == "assistant":
            if pending:
                raise ValueError(f"{rid}: unanswered tool_calls before turn {idx}: {sorted(pending)}")
            rc = msg.get("reasoning_content")
            if rc is not None:
                if not isinstance(rc, str):
                    raise ValueError(f"{rid}: message[{idx}].reasoning_content must be str")
                has_reasoning = has_reasoning or bool(rc.strip())
            for tc in msg.get("tool_calls") or ():
                fn = tc.get("function") if isinstance(tc, Mapping) else None
                if not isinstance(fn, Mapping) or not isinstance(fn.get("name"), str) or not fn["name"] or not isinstance(fn.get("arguments"), (dict, str)):
                    raise ValueError(f"{rid}: invalid tool_call in message[{idx}]")
                has_tool_calls = True
                if isinstance(tc.get("id"), str) and tc["id"]:
                    pending.add(tc["id"])
        elif role == "tool":
            if not isinstance(msg.get("content"), str):
                raise ValueError(f"{rid}: tool message[{idx}] requires string content")
            has_tool_results = True
            pending.discard(str(msg.get("tool_call_id") or ""))
        elif not isinstance(msg.get("content"), str):
            raise ValueError(f"{rid}: {role} message[{idx}] requires string content")
    if pending or (has_tool_calls and not has_tool_results):
        raise ValueError(f"{rid}: unmatched tool_calls and tool results")

    return dict(id=rid, split=split, length_class=length_class, has_tools=bool(tools),
                has_tool_calls=has_tool_calls, has_tool_results=has_tool_results,
                has_reasoning=has_reasoning, content_sha256=actual_hash)


def _snippet(text: str, limit: int = 48) -> str:
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= 8:
            return s[:limit]
    return text.strip()[:limit]


def _check_template_and_decoded(tokenizer, record: Mapping[str, Any], meta: Mapping[str, Any], decoded: str | None = None) -> dict[str, bool | None]:
    rid = meta["id"]
    if decoded is None:
        template = getattr(tokenizer, "chat_template", None)
        if isinstance(template, Mapping):
            template = template.get("default") or next(iter(template.values()), None)
        if isinstance(template, str):
            if meta["has_tools"] and "tools" not in template:
                raise ValueError("checkpoint chat template does not support tools declarations")
            if meta["has_tool_calls"] and "tool_calls" not in template:
                raise ValueError("checkpoint chat template cannot express assistant tool_calls")
            if meta["has_tool_results"] and ("'tool'" not in template and '"tool"' not in template):
                raise ValueError("checkpoint chat template cannot express role='tool' results")
        return {}

    if meta["has_tools"]:
        for tool in record["tools"]:
            if tool["function"]["name"] not in decoded:
                raise ValueError(f"{rid}: rendered tokens dropped declared tool {tool['function']['name']!r}")
    if meta["has_tool_calls"]:
        for idx, msg in enumerate(record["messages"]):
            for tc in msg.get("tool_calls") or ():
                fn = tc["function"]
                if fn["name"] not in decoded:
                    raise ValueError(f"{rid}: rendered tokens dropped tool_call {fn['name']!r} at turn {idx}")
                args = fn.get("arguments")
                if isinstance(args, Mapping) and args and next(iter(args)) not in decoded:
                    raise ValueError(f"{rid}: rendered tokens dropped tool_call argument at turn {idx}")
    if meta["has_tool_results"]:
        for idx, msg in enumerate(record["messages"]):
            if msg["role"] == "tool":
                s = _snippet(msg["content"])
                if s and s not in decoded:
                    raise ValueError(f"{rid}: rendered tokens dropped tool result at turn {idx}")

    rc_snips = [_snippet(m["reasoning_content"]) for m in record["messages"]
                if m.get("role") == "assistant" and isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip()]
    rc_snips = [s for s in rc_snips if s]
    if not rc_snips:
        return {"history_reasoning_preserved": None, "final_reasoning_preserved": None}
    return {
        "history_reasoning_preserved": all(s in decoded for s in rc_snips[:-1]) if len(rc_snips) >= 2 else None,
        "final_reasoning_preserved": rc_snips[-1] in decoded,
    }


def _call_kwargs_for(fn, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    accepted = set(sig.parameters)
    out = {}
    for k, v in kwargs.items():
        if k in accepted:
            out[k] = v
        elif k != "return_dict":
            raise ValueError(f"tokenizer.apply_chat_template does not accept required argument {k!r}")
    return out


def render_long_context_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    chat_kwargs: Mapping[str, Any] | None = None,
    max_context: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Render one multi-turn ``long_context`` record in one pass; fail closed."""
    meta = validate_long_context_record(record)
    rid = meta["id"]
    _check_template_and_decoded(tokenizer, record, meta)

    call_kwargs = dict(chat_kwargs or {})
    call_kwargs.update(tokenize=True, add_generation_prompt=False, padding=False, truncation=False, return_dict=False)
    if meta["has_tools"]:
        call_kwargs["tools"] = record["tools"]

    try:
        raw = tokenizer.apply_chat_template(record["messages"], **_call_kwargs_for(tokenizer.apply_chat_template, call_kwargs))
    except Exception as exc:
        raise ValueError(f"{rid}: checkpoint chat template failed on multi-turn record: {exc}") from exc

    if isinstance(raw, Mapping):
        raw = raw.get("input_ids")
    elif hasattr(raw, "input_ids"):
        raw = raw.input_ids
    if isinstance(raw, list) and raw and isinstance(raw[0], list) and len(raw) == 1:
        raw = raw[0]
    vocab = len(tokenizer)
    if not isinstance(raw, list) or not raw or any(type(t) is not int or not 0 <= t < vocab for t in raw):
        raise ValueError(f"{rid}: invalid token IDs from checkpoint template")

    if hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode(raw, skip_special_tokens=False)
        if not isinstance(decoded, str):
            raise ValueError(f"{rid}: tokenizer.decode did not return a string")
        reasoning_stats = _check_template_and_decoded(tokenizer, record, meta, decoded)
    elif meta["has_tools"] or meta["has_tool_calls"] or meta["has_tool_results"]:
        raise ValueError(f"{rid}: tokenizer lacks decode(); cannot verify tool rendering")
    else:
        reasoning_stats = {"history_reasoning_preserved": None, "final_reasoning_preserved": None}

    low, high = LONG_CONTEXT_LENGTH_BOUNDS[meta["length_class"]]
    if not low <= len(raw) <= high:
        raise ValueError(f"{rid}: rendered length {len(raw)} outside declared {meta['length_class']!r} bounds [{low}, {high}]")
    if max_context is not None and (type(max_context) is not int or max_context <= 0 or len(raw) > max_context):
        raise ValueError(f"{rid}: rendered length {len(raw)} exceeds model max_context {max_context}")

    entry = dict(
        id=rid, component="long_context", split=meta["split"], length_class=meta["length_class"],
        tokens=len(raw), content_sha256=meta["content_sha256"], input_ids_sha256=ids_sha256(raw),
        has_tools=meta["has_tools"], has_tool_calls=meta["has_tool_calls"],
        has_reasoning=meta["has_reasoning"], **reasoning_stats,
    )
    return raw, entry


def tokenize_long_context_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    chat_kwargs: Mapping[str, Any] | None = None,
    max_context: int | None = None,
) -> tuple[array, list[int], dict[str, Any]]:
    flat, offsets, records, seqs, seen = array("q"), [0], [], [], set()
    stream, by_class = hashlib.sha256(), {}
    for row in rows:
        ids, entry = render_long_context_record(row, tokenizer, chat_kwargs=chat_kwargs, max_context=max_context)
        if entry["id"] in seen:
            raise ValueError(f"duplicate long_context record id: {entry['id']}")
        seen.add(entry["id"])
        flat.extend(ids)
        offsets.append(len(flat))
        seqs.append(ids)
        stream.update(canonical({"id": entry["id"], "input_ids": ids}) + b"\n")
        records.append(entry)
        b = by_class.setdefault(entry["length_class"], {"count": 0, "tokens": 0})
        b["count"] += 1
        b["tokens"] += entry["tokens"]
    if not records:
        raise ValueError("empty tokenized long_context split")
    return flat, offsets, dict(
        records=records, count=len(records), tokens=len(flat),
        shortest=min(r["tokens"] for r in records), longest=max(r["tokens"] for r in records),
        input_ids_sha256=component_input_ids_sha256(seqs),
        token_stream_sha256=stream.hexdigest(), length_classes=by_class,
    )


def _read_sha256sums(root: Path) -> dict[str, str]:
    path = root / "SHA256SUMS"
    if not path.is_file():
        raise ValueError(f"missing panel SHA256SUMS: {path}")
    return {name: digest for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            for digest, name in (line.split(maxsplit=1),)}


def read_long_context_component(root: Path, *, hashes: Mapping[str, str], agentic_provenance=None):
    for rel in ("long_context/manifest.json", *(f"long_context/{s}.jsonl" for s in LONG_CONTEXT_SUBSETS)):
        target = root / rel
        if rel not in hashes or not target.is_file() or sha256_file(target) != hashes[rel]:
            raise ValueError(f"long_context file missing or hash differs: {rel}")

    lc_manifest = json.loads((root / "long_context/manifest.json").read_text(encoding="utf-8"))
    if lc_manifest.get("schema") != LONG_CONTEXT_SCHEMA or lc_manifest.get("component") != "long_context":
        raise ValueError("long_context manifest schema or component differs")

    by_split: dict[str, list[dict[str, Any]]] = {"calibration": [], "heldout": []}
    all_rows, seen = [], set()
    for subset in LONG_CONTEXT_SUBSETS:
        rel = f"long_context/{subset}.jsonl"
        decl = lc_manifest.get("subsets", {}).get(subset)
        if not isinstance(decl, Mapping) or decl.get("file") != rel or decl.get("sha256") != hashes[rel]:
            raise ValueError(f"long_context subset manifest mismatch for {subset}")
        lines = (root / rel).read_text(encoding="utf-8").splitlines()
        if len(lines) != decl.get("records"):
            raise ValueError(f"long_context subset {subset} record count differs")
        for line in lines:
            row = json.loads(line)
            meta = validate_long_context_record(row)
            if meta["length_class"] != subset or meta["id"] in seen:
                raise ValueError(f"{meta['id']}: bad subset class or duplicate id")
            seen.add(meta["id"])
            by_split[meta["split"]].append(row)
            all_rows.append(row)

    manifest_rows = {r["id"]: r for r in lc_manifest.get("records", [])}
    if set(manifest_rows) != seen or any(
        manifest_rows[r["id"]].get(f) != r.get(f)
        for r in all_rows for f in ("split", "length_class", "group", "content_sha256", "source_identity")
    ):
        raise ValueError("long_context manifest record inventory differs from payloads")

    cal, hold = by_split["calibration"], by_split["heldout"]
    if not cal or not hold:
        raise ValueError("long_context requires nonempty calibration and heldout splits")
    for key in ("group", "content_sha256", "source_identity"):
        if {r[key] for r in cal} & {r[key] for r in hold}:
            raise ValueError("long_context calibration and heldout splits overlap")
    if agentic_provenance is not None:
        ag_cal = {r["group"] for r in agentic_provenance["splits"]["calibration"]["records"]}
        ag_hold = {r["group"] for r in agentic_provenance["splits"]["heldout"]["records"]}
        if ({r["group"] for r in cal} & ag_hold) or ({r["group"] for r in hold} & ag_cal):
            raise ValueError("long_context splits overlap agentic splits")
    return by_split, lc_manifest


def read_panel_v2(
    root: Path | str,
    *,
    panel_revision: str | None = None,
    panel_sha256sums: str | None = None,
    components: Sequence[str] | None = None,
    full: bool = True,
) -> dict[str, Any]:
    """Attest and load the pinned ``agentic-quant-panel-v2`` snapshot."""
    root = Path(root)
    repo_id, rev, resolved = resolve_panel_policy(panel_revision=panel_revision, components=components, full=full)
    if root.name != rev:
        raise ValueError(f"require the pinned v2 snapshot {rev}, got {root.name!r}")

    sums_sha = sha256_file(root / "SHA256SUMS")
    expected_sums = panel_sha256sums
    if expected_sums is None:
        if rev == TWO_COMPONENT_REVISION:
            expected_sums = TWO_COMPONENT_SHA256SUMS_SHA256
        elif THREE_COMPONENT_REVISION is not None and rev == THREE_COMPONENT_REVISION:
            expected_sums = THREE_COMPONENT_SHA256SUMS_SHA256
    if expected_sums is not None and sums_sha != expected_sums:
        raise ValueError(f"panel SHA256SUMS digest mismatch for {rev}")

    hashes = _read_sha256sums(root)
    base_names = (
        "panel.json", "manifest.json", *(s + ".jsonl" for s in SPLITS),
        *("data/" + s + ".parquet" for s in (*SPLITS, "exl3_standard")),
        "exl3_standard/manifest.json", "exl3_standard/segments.jsonl",
        "exl3_standard/upstream/tokenizer.py", EXL3_STANDARD_PACKER,
    )
    if any(name not in hashes for name in base_names):
        raise ValueError("panel SHA256SUMS missing expected base files")
    if any(hashes[name] != v1_panel.PAYLOAD_SHA256[name] for name in ("manifest.json", *(s + ".jsonl" for s in SPLITS))):
        raise ValueError("v2 agentic provenance differs from the released v1 panel")

    agentic_rows, agentic_prov = v1_panel.read_panel(root, payload_sha256={n: hashes[n] for n in base_names})
    lc_rows = lc_manifest = None
    if "long_context" in resolved:
        lc_rows, lc_manifest = read_long_context_component(root, hashes=hashes, agentic_provenance=agentic_prov)

    return dict(
        dataset={"repo_id": repo_id, "revision": rev, "sha256sums_sha256": sums_sha},
        components=resolved, agentic_rows=agentic_rows, agentic_provenance=agentic_prov,
        long_context_rows=lc_rows, long_context_manifest=lc_manifest,
    )


def pack_exl3_standard(corpus_root: Path, source_snapshot: Path, stage: Path, tokenizer_sha256: str):
    import torch
    from safetensors.torch import load_file

    packed = stage / "exl3_standard.safetensors"
    subprocess.run([sys.executable, str(corpus_root / EXL3_STANDARD_PACKER), str(corpus_root), str(source_snapshot), str(packed)], check=True)
    receipt = json.loads(packed.with_suffix(".json").read_text(encoding="utf-8"))
    values = load_file(str(packed))["input_ids"]
    digest = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    if (values.shape != (250, 2048) or values.dtype != torch.int64
            or receipt.get("contract") != EXL3_STANDARD_CONTRACT
            or receipt.get("tokenizer_sha256") != tokenizer_sha256
            or receipt.get("input_ids_sha256") != digest):
        raise ValueError("ExLlamaV3 calibration pack contract or checksum differs")

    rows_out, stream = [], hashlib.sha256()
    for idx, (row, prov) in enumerate(zip(values, receipt["rows"], strict=True)):
        tokens = row.tolist()
        row_id = f"exl3_standard/{idx:03d}"
        if ids_sha256(tokens) != prov["token_ids_sha256"]:
            raise ValueError(f"standard panel row digest differs: {row_id}")
        stream.update(canonical({"id": row_id, "input_ids": tokens}) + b"\n")
        rows_out.append(dict(id=row_id, component="exl3_standard", input_ids=tokens, tokens=len(tokens),
                             input_ids_sha256=prov["token_ids_sha256"], source=prov["source"], kind=prov["kind"]))
    packed.unlink()
    return rows_out, dict(contract=receipt["contract"], count=values.shape[0], tokens=values.numel(),
                          input_ids_sha256=digest, token_stream_sha256=stream.hexdigest(), receipt="exl3_standard.json")


def _append_to_split(ids: array, offsets: list[int], report: dict[str, Any], extra_meta, extra_seqs) -> None:
    stream = hashlib.sha256()
    for i, row in enumerate(report["records"]):
        stream.update(canonical({"id": row["id"], "input_ids": ids[offsets[i]:offsets[i + 1]].tolist()}) + b"\n")
    for meta, seq in zip(extra_meta, extra_seqs, strict=True):
        toks = list(seq)
        ids.extend(toks)
        offsets.append(len(ids))
        entry = {k: v for k, v in meta.items() if k != "input_ids"}
        report["records"].append(entry)
        stream.update(canonical({"id": entry["id"], "input_ids": toks}) + b"\n")
    lengths = [r["tokens"] for r in report["records"]]
    report.update(tokens=len(ids), count=len(offsets) - 1, shortest=min(lengths),
                  longest=max(lengths), token_stream_sha256=stream.hexdigest())


def prepare_panel_corpus(
    source: Path | str,
    corpus: Path | str,
    output: Path | str,
    *,
    source_repo: str,
    source_revision: str,
    chat_kwargs: Mapping[str, Any],
    schema: str = PREPARED_SCHEMA,
    contract: Mapping[str, Any] | None = None,
    assets: Sequence[str] = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"),
    components: Sequence[str] | None = None,
    full: bool = True,
    panel_revision: str | None = None,
    panel_sha256sums: str | None = None,
    tokenizer: Any = None,
    long_witness_fn=None,
) -> dict[str, Any]:
    """Prepare and attest a 2- or 3-component calibration corpus on disk."""
    import torch
    from transformers import PreTrainedTokenizerFast

    source, corpus, output = Path(source).absolute(), Path(corpus).absolute(), Path(output).absolute()
    if source.name != source_revision or output.exists():
        raise ValueError("require the pinned source and a fresh corpus directory")
    identity = {name: sha256_file(source / name) for name in assets}
    if tokenizer is None:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
        if (not tokenizer.is_fast or ("chat_template.jinja" in identity and (
                not isinstance(tokenizer.chat_template, str)
                or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != identity["chat_template.jinja"]))):
            raise ValueError("native tokenizer did not load the pinned chat template")

    max_context = max_context_from_config(source)
    panel = read_panel_v2(corpus, panel_revision=panel_revision, panel_sha256sums=panel_sha256sums,
                          components=components, full=full)
    active = panel["components"]
    contract_dict = dict(contract) if contract is not None else dict(
        renderer="native-chat-template", raw_calibration=EXL3_STANDARD_CONTRACT,
        batch_size=1, packing=False, order="dataset", **dict(chat_kwargs),
        **({"long_context": LONG_CONTEXT_SCHEMA} if "long_context" in active else {}),
    )
    manifest: dict[str, Any] = dict(
        schema=schema, status="passed", contract=contract_dict, dataset=panel["dataset"],
        component_order=list(active), calibration_mode="full" if full else "subset", components={},
        historical_selection=panel["agentic_provenance"]["selection"],
        source=dict(repo_id=source_repo, revision=source_revision, files=identity),
        software=software_identity(), tokenizer=dict(vocab_size=len(tokenizer), max_context=max_context), splits={},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in SPLITS:
            ids, offsets, report = array("q"), [0], dict(records=[], count=0, tokens=0, shortest=0, longest=0, token_stream_sha256="")
            if "agentic" in active:
                ids, offsets, report = v1_panel.tokenize_rows(panel["agentic_rows"][split], tokenizer, chat_kwargs=dict(chat_kwargs))
                for r in report["records"]:
                    r["component"] = "agentic"
                if split == "calibration":
                    ag_seqs = [ids[offsets[i]:offsets[i + 1]].tolist() for i in range(len(report["records"]))]
                    manifest["components"]["agentic"] = dict(
                        count=report["count"], tokens=report["tokens"],
                        input_ids_sha256=component_input_ids_sha256(ag_seqs),
                        token_stream_sha256=report["token_stream_sha256"],
                    )
            if split == "calibration" and "exl3_standard" in active:
                std_rows, std_meta = pack_exl3_standard(corpus, source, stage, identity["tokenizer.json"])
                _append_to_split(ids, offsets, report, std_rows, [r["input_ids"] for r in std_rows])
                manifest["components"]["exl3_standard"] = std_meta
            if "long_context" in active and split in LONG_CONTEXT_SPLITS:
                lc_ids, lc_off, lc_rep = tokenize_long_context_rows(
                    panel["long_context_rows"][split], tokenizer, chat_kwargs=chat_kwargs, max_context=max_context)
                lc_seqs = [lc_ids[lc_off[i]:lc_off[i + 1]].tolist() for i in range(len(lc_rep["records"]))]
                _append_to_split(ids, offsets, report, lc_rep["records"], lc_seqs)
                summary = dict(count=lc_rep["count"], tokens=lc_rep["tokens"],
                               input_ids_sha256=lc_rep["input_ids_sha256"],
                               token_stream_sha256=lc_rep["token_stream_sha256"],
                               length_classes=lc_rep["length_classes"])
                if split == "calibration":
                    manifest["components"]["long_context"] = dict(schema=LONG_CONTEXT_SCHEMA, **summary)
                else:
                    manifest["components"]["long_context"]["heldout"] = summary
            if not report["records"]:
                continue
            path = stage / (split + ".safetensors")
            receipt = checkpoint_io().write_safetensors(
                path, dict(input_ids=torch.frombuffer(ids, dtype=torch.int64).clone(),
                           offsets=torch.tensor(offsets, dtype=torch.int64)),
                metadata=dict(schema=schema, split=split), hash_algorithm="sha256")
            manifest["splits"][split] = dict(report, file=path.name, sha256=receipt["sha256"])

        if long_witness_fn is not None:
            witness = long_witness_fn(tokenizer, chat_kwargs)
            atomic_json(stage / "long-context-smoke.json", witness)
            manifest["long_context_smoke"] = dict(file="long-context-smoke.json", tokens=witness["tokens"],
                                                  sha256=sha256_file(stage / "long-context-smoke.json"), hessian_eligible=False)
        if identity != {name: sha256_file(source / name) for name in assets}:
            raise ValueError("source tokenizer changed during preparation")
        atomic_json(stage / "manifest.json", manifest)
        for split in manifest["splits"]:
            load_token_records(stage, split, schema=schema, contract=contract_dict, dataset=panel["dataset"], full=full)
        stage.rename(output)
        fsync_directory(output.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def _record_component(row: Mapping[str, Any], index: int, manifest: Mapping[str, Any], split: str) -> str:
    if isinstance(row.get("component"), str):
        return row["component"]
    if str(row.get("id", "")).startswith("exl3_standard/"):
        return "exl3_standard"
    if split == "calibration":
        cursor = 0
        for name in COMPONENT_ORDER:
            if name in manifest.get("components", {}):
                count = int(manifest["components"][name]["count"])
                if cursor <= index < cursor + count:
                    return name
                cursor += count
    return "agentic"


def _verify_components(records: list[dict[str, Any]], manifest: Mapping[str, Any], split: str, *, require_components, full) -> None:
    comps = manifest.get("components")
    if not isinstance(comps, Mapping) or not comps:
        if require_components is not None:
            raise ValueError("prepared manifest is missing components provenance")
        return
    if set(comps) - set(COMPONENT_ORDER):
        raise ValueError(f"unknown components in prepared manifest: {sorted(set(comps) - set(COMPONENT_ORDER))}")
    present = [c for c in COMPONENT_ORDER if c in comps]
    if manifest.get("component_order") is not None and list(manifest["component_order"]) != present:
        raise ValueError(f"component_order {manifest['component_order']} disagrees with {present}")

    enforce_full = (manifest.get("calibration_mode") == "full") if full is None else full
    if split == "calibration" and enforce_full and THREE_COMPONENT_REVISION is not None and tuple(present) != COMPONENT_ORDER:
        raise ValueError(
            f"full calibration requires all three components {COMPONENT_ORDER} once "
            f"{THREE_COMPONENT_REVISION} is published; prepared corpus has {tuple(present)}"
        )
    if require_components is not None:
        if set(require_components) - set(COMPONENT_ORDER):
            raise ValueError(f"unknown required component in {require_components!r}")
        missing = [c for c in COMPONENT_ORDER if c in require_components and c not in comps]
        if missing:
            raise ValueError(f"prepared corpus missing required components: {missing}")

    if split == "calibration":
        cursor = 0
        for name in present:
            info = comps[name]
            chunk = records[cursor:cursor + int(info["count"])]
            if len(chunk) != int(info["count"]) or any(r["component"] != name for r in chunk):
                raise ValueError(f"calibration stream component order or count mismatch in {name!r}")
            if sum(len(r["input_ids"]) for r in chunk) != int(info["tokens"]):
                raise ValueError(f"component {name!r} token count differs from manifest")
            if info.get("input_ids_sha256") is not None and component_input_ids_sha256(r["input_ids"] for r in chunk) != info["input_ids_sha256"]:
                raise ValueError(f"component {name!r} input_ids_sha256 differs from manifest")
            cursor += int(info["count"])
        if cursor != len(records):
            raise ValueError("calibration stream contains trailing records outside declared components")
    elif split == "heldout":
        seen_lc, lc_chunk = False, []
        for r in records:
            if r["component"] == "exl3_standard":
                raise ValueError("exl3_standard must never appear in heldout split")
            if r["component"] == "long_context":
                seen_lc = True
                lc_chunk.append(r)
            elif seen_lc:
                raise ValueError("heldout stream out of order: agentic record after long_context")
        lc_hold = comps.get("long_context", {}).get("heldout") if "long_context" in comps else None
        if lc_hold is not None:
            if (len(lc_chunk) != int(lc_hold["count"])
                    or sum(len(r["input_ids"]) for r in lc_chunk) != int(lc_hold["tokens"])
                    or component_input_ids_sha256(r["input_ids"] for r in lc_chunk) != lc_hold["input_ids_sha256"]):
                raise ValueError("long_context heldout provenance differs from manifest")


def load_token_records(
    root: Path | str,
    split: str = "calibration",
    *,
    schema: str | None = PREPARED_SCHEMA,
    contract: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    require_components: Sequence[str] | None = None,
    full: bool | None = None,
) -> list[dict[str, Any]]:
    """Verify a prepared split and return ordered records with component tags."""
    import torch
    from safetensors.torch import load_file

    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "passed"
            or (schema is not None and manifest.get("schema") != schema)
            or (contract is not None and manifest.get("contract") != dict(contract))
            or (dataset is not None and manifest.get("dataset") != dict(dataset))):
        raise ValueError("prepared corpus contract differs")

    report = manifest["splits"][split]
    path = root / report["file"]
    if report.get("file") != split + ".safetensors" or sha256_file(path) != report["sha256"]:
        raise ValueError("prepared split file or hash differs")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"input_ids", "offsets"} or any(t.dtype != torch.int64 or t.ndim != 1 for t in tensors.values()):
        raise ValueError("invalid token tensor layout")

    flat, offsets, records = tensors["input_ids"], tensors["offsets"].tolist(), report["records"]
    if (not records or len(offsets) != len(records) + 1 or offsets[0] != 0 or offsets[-1] != flat.numel()
            or any(a >= b for a, b in zip(offsets, offsets[1:])) or len({r["id"] for r in records}) != len(records)):
        raise ValueError("invalid token offsets or record identities")

    stream, result, vocab = hashlib.sha256(), [], manifest["tokenizer"]["vocab_size"]
    for idx, (row, start, stop) in enumerate(zip(records, offsets, offsets[1:], strict=False)):
        ids = flat[start:stop].tolist()
        if len(ids) != row["tokens"] or ids_sha256(ids) != row["input_ids_sha256"] or any(not 0 <= t < vocab for t in ids):
            raise ValueError("prepared record token identity differs")
        stream.update(canonical({"id": row["id"], "input_ids": ids}) + b"\n")
        entry = dict(id=row["id"], component=_record_component(row, idx, manifest, split), input_ids=ids)
        if "length_class" in row:
            entry["length_class"] = row["length_class"]
        result.append(entry)

    if (report["count"] != len(records) or report["tokens"] != flat.numel()
            or report["shortest"] != min(r["tokens"] for r in records)
            or report["longest"] != max(r["tokens"] for r in records)
            or report["token_stream_sha256"] != stream.hexdigest()):
        raise ValueError("prepared split token summary differs")

    _verify_components(result, manifest, split, require_components=require_components, full=full)
    return result


def load_token_batches(
    root: Path | str,
    split: str = "calibration",
    *,
    schema: str | None = PREPARED_SCHEMA,
    contract: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    require_components: Sequence[str] | None = None,
    full: bool | None = None,
):
    """Yield unpadded CPU batches (one record per batch) in canonical order."""
    import torch

    for row in load_token_records(
        root, split, schema=schema, contract=contract, dataset=dataset,
        require_components=require_components, full=full,
    ):
        ids = torch.tensor([row["input_ids"]], dtype=torch.int64)
        yield dict(id=row["id"], component=row["component"], input_ids=ids, attention_mask=torch.ones_like(ids))


def load_component_records(
    root: Path | str,
    split: str = "calibration",
    *,
    components: Sequence[str] | None = None,
    schema: str | None = None,
    contract: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    full: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load selected components and frozen provenance (importable by MC and QMLA)."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    all_records = load_token_records(
        root, split, schema=schema, contract=contract, dataset=dataset,
        require_components=components, full=full,
    )
    wanted = set(components) if components is not None else {r["component"] for r in all_records}
    filtered = [r for r in all_records if r["component"] in wanted]
    comp_prov: dict[str, dict[str, Any]] = {}
    for comp in COMPONENT_ORDER:
        rows = [r for r in filtered if r["component"] == comp]
        if rows:
            comp_prov[comp] = dict(
                count=len(rows), sequences=len(rows),
                tokens=sum(len(r["input_ids"]) for r in rows),
                input_ids_sha256=component_input_ids_sha256(r["input_ids"] for r in rows),
            )
    combined_sha = hashlib.sha256(
        "\n".join(f"{k}:{comp_prov[k]['input_ids_sha256']}" for k in COMPONENT_ORDER if k in comp_prov).encode()
    ).hexdigest()
    return filtered, dict(
        dataset_repo=manifest["dataset"]["repo_id"],
        dataset_revision=manifest["dataset"]["revision"],
        component_order=[k for k in COMPONENT_ORDER if k in comp_prov],
        components=comp_prov,
        input_ids_sha256=combined_sha,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-revision", default=None)
    parser.add_argument("--components", default=None)
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--date-string", default=None)
    args = parser.parse_args()

    chat = dict(tokenize=True, add_generation_prompt=True, return_dict=False, padding=False, truncation=False)
    if args.reasoning_effort is not None:
        chat["reasoning_effort"] = args.reasoning_effort
    if args.date_string is not None:
        chat["date_string"] = args.date_string
    comps = [c.strip() for c in args.components.split(",") if c.strip()] if args.components else None
    manifest = prepare_panel_corpus(
        args.source, args.corpus, args.output, source_repo=args.source_repo,
        source_revision=args.source.resolve().name, chat_kwargs=chat,
        components=comps, full=not args.subset, panel_revision=args.panel_revision,
    )
    print(json.dumps(dict(
        dataset=manifest["dataset"], component_order=manifest["component_order"],
        splits={k: {f: v[f] for f in ("count", "tokens", "longest")} for k, v in manifest["splits"].items()},
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
