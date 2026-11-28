#!/usr/bin/env python3
"""Shared three-component calibration corpus renderer, preparer, and loader.

Single source of truth for ``wrldsuksgo2mars/agentic-quant-panel-v2`` across
full MC quantizations, QMLA calibrations, and other future quant pipelines.

Components in canonical calibration order:
1. ``agentic``: single-turn prompts rendered once with the checkpoint's own
   chat template (``add_generation_prompt=True``), unpadded, one prompt per batch.
2. ``exl3_standard``: raw text packed to fixed-width rows (250 x 2,048 tokens)
   via ``reproduction/pack_exl3_standard.py`` (contract ``exl3-standard-pack-v1``).
3. ``long_context``: 16K and 32K multi-turn agentic traces (with tool calls,
   tool results, and optional reasoning) plus long-document QA, rendered once
   with the checkpoint's own chat template (``add_generation_prompt=False``),
   unpadded, one record per batch, honoring ``calibration`` vs ``heldout`` splits.

Bumping the panel revision once ``long_context`` is published is a single edit:
set ``THREE_COMPONENT_REVISION`` and ``THREE_COMPONENT_SHA256SUMS_SHA256`` below.
Before that revision is published, the default stays on the pinned two-component
revision ``54683731eb01c7c2fa5b801a0888e029a0e1b1c7``. Once published, the default
switches to all three components and any ``full=True`` calibration requesting
fewer than all three components fails closed.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from kimi_k3 import prepare_corpus as v1_panel
from kimi_k3.uniform_k2.storage import atomic_json, checkpoint_io, fsync_directory


# ---------------------------------------------------------------------------
# Single source of truth for panel pinning and component ordering
# ---------------------------------------------------------------------------

PANEL_REPO = "wrldsuksgo2mars/agentic-quant-panel-v2"

# Pinned two-component revision (agentic + exl3_standard).
TWO_COMPONENT_REVISION = "54683731eb01c7c2fa5b801a0888e029a0e1b1c7"
TWO_COMPONENT_SHA256SUMS_SHA256 = "23f07f7dbe7b8248a09891b6ae2475371f4653e499a9b8f930e267ff3d5c6d57"

# Set these two constants in one edit once the three-component revision lands
# on wrldsuksgo2mars/agentic-quant-panel-v2.
THREE_COMPONENT_REVISION: str | None = None
THREE_COMPONENT_SHA256SUMS_SHA256: str | None = None

COMPONENT_ORDER: tuple[str, ...] = ("agentic", "exl3_standard", "long_context")
BASE_COMPONENTS: tuple[str, ...] = ("agentic", "exl3_standard")
FULL_COMPONENTS: tuple[str, ...] = COMPONENT_ORDER
KNOWN_COMPONENTS: tuple[str, ...] = COMPONENT_ORDER

SPLITS: tuple[str, ...] = ("calibration", "screening", "heldout")
LONG_CONTEXT_SUBSETS: tuple[str, ...] = ("16k", "32k")
LONG_CONTEXT_SPLITS: tuple[str, ...] = ("calibration", "heldout")
LONG_CONTEXT_SCHEMA = "agentic-quant-panel-v2-long-context-v1"

EXL3_STANDARD_PACKER = "reproduction/pack_exl3_standard.py"
EXL3_STANDARD_CONTRACT = "exl3-standard-pack-v1"
PREPARED_SCHEMA = "cuteafd-panel-corpus-v3"

# Rendered token bounds per length class ([0.70x, 1.35x] of 16,384 / 32,768).
# Strictly disjoint between 16k and 32k; wide enough for cross-tokenizer
# vocabulary differences and native chat-template history-reasoning pruning
# while rejecting truncated or misclassified records.
LONG_CONTEXT_LENGTH_BOUNDS: dict[str, tuple[int, int]] = {
    "16k": (11_468, 22_118),
    "32k": (22_937, 44_236),
}

_FORBIDDEN_RECORD_KEYS = frozenset({"input_ids", "token_ids", "tokens"})
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

canonical = v1_panel.canonical
sha256_file = v1_panel.sha256_file
ids_sha256 = v1_panel.ids_sha256
software_identity = v1_panel.software_identity


def _is_hex(value: Any, length: int) -> bool:
    if not isinstance(value, str):
        return False
    pattern = _HEX40 if length == 40 else _HEX64
    return bool(pattern.match(value))


def default_panel_revision(*, three_component_revision: str | None = None) -> str:
    """Return the active default panel revision from the single source of truth."""
    rev = THREE_COMPONENT_REVISION if three_component_revision is None else three_component_revision
    return rev if rev is not None else TWO_COMPONENT_REVISION


def default_panel_sha256sums(
    *,
    three_component_revision: str | None = None,
    three_component_sha256sums: str | None = None,
) -> str | None:
    """Return the pinned SHA256SUMS digest for the active default panel revision."""
    rev = THREE_COMPONENT_REVISION if three_component_revision is None else three_component_revision
    if rev is None:
        return TWO_COMPONENT_SHA256SUMS_SHA256
    return THREE_COMPONENT_SHA256SUMS_SHA256 if three_component_sha256sums is None else three_component_sha256sums


def default_components(*, three_component_revision: str | None = None) -> tuple[str, ...]:
    """Return the default calibration component tuple (2 before 3-component publish, 3 after)."""
    rev = THREE_COMPONENT_REVISION if three_component_revision is None else three_component_revision
    return FULL_COMPONENTS if rev is not None else BASE_COMPONENTS


def default_dataset(
    *,
    three_component_revision: str | None = None,
    three_component_sha256sums: str | None = None,
) -> dict[str, Any]:
    """Return the canonical dataset identity dictionary for the active default pin."""
    rev = default_panel_revision(three_component_revision=three_component_revision)
    sums = default_panel_sha256sums(
        three_component_revision=three_component_revision,
        three_component_sha256sums=three_component_sha256sums,
    )
    entry: dict[str, Any] = {"repo_id": PANEL_REPO, "revision": rev}
    if sums is not None:
        entry["sha256sums_sha256"] = sums
    return entry


DEFAULT_PANEL_REVISION: str = default_panel_revision()
DEFAULT_PANEL_SHA256SUMS_SHA256: str | None = default_panel_sha256sums()
DEFAULT_COMPONENTS: tuple[str, ...] = default_components()
DEFAULT_DATASET: dict[str, Any] = default_dataset()


def resolve_panel_policy(
    *,
    panel_revision: str | None = None,
    components: Sequence[str] | None = None,
    full: bool = True,
    three_component_revision: str | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    """Resolve and validate ``(repo_id, revision, canonical_components)``.

    Rules:
    - Component names are restricted to ``COMPONENT_ORDER`` and always returned
      in canonical order (``agentic``, ``exl3_standard``, ``long_context``).
    - Before ``three_component_revision`` is published (``None``): default is
      ``(TWO_COMPONENT_REVISION, ("agentic", "exl3_standard"))``. Requesting
      ``long_context`` against ``TWO_COMPONENT_REVISION`` fails closed.
    - Once ``three_component_revision`` is published: default is
      ``(three_component_revision, COMPONENT_ORDER)``, and any ``full=True``
      request that omits ``long_context`` or pins the legacy two-component
      revision fails closed.
    - Non-full runs (``full=False``, e.g. pilot subsets) may select a nonempty
      subset of ``COMPONENT_ORDER``.
    """
    active_three = THREE_COMPONENT_REVISION if three_component_revision is None else three_component_revision
    if active_three is not None and not _is_hex(active_three, 40):
        raise ValueError(f"three_component_revision must be a 40-char lowercase SHA, got {active_three!r}")

    resolved_rev = (
        (active_three if active_three is not None else TWO_COMPONENT_REVISION)
        if panel_revision is None
        else panel_revision
    )
    if not _is_hex(resolved_rev, 40):
        raise ValueError(f"panel_revision must be a 40-char lowercase SHA, got {resolved_rev!r}")

    if components is None:
        if resolved_rev == TWO_COMPONENT_REVISION:
            selected = list(BASE_COMPONENTS)
        elif active_three is not None or resolved_rev != TWO_COMPONENT_REVISION:
            selected = list(FULL_COMPONENTS)
        else:
            selected = list(BASE_COMPONENTS)
    else:
        selected = []
        for name in components:
            if not isinstance(name, str) or name not in COMPONENT_ORDER:
                raise ValueError(
                    f"unknown panel component {name!r}; allowed components are {COMPONENT_ORDER}"
                )
            if name not in selected:
                selected.append(name)
        if not selected:
            raise ValueError("components must be nonempty")

    ordered = tuple(name for name in COMPONENT_ORDER if name in selected)

    if "long_context" in ordered and resolved_rev == TWO_COMPONENT_REVISION:
        raise ValueError(
            "long_context is not present in the two-component panel revision "
            f"{TWO_COMPONENT_REVISION}"
        )

    if full:
        if active_three is not None:
            if resolved_rev == TWO_COMPONENT_REVISION or ordered != FULL_COMPONENTS:
                raise ValueError(
                    "full calibration requires all three panel components "
                    f"{FULL_COMPONENTS} once the three-component revision ({active_three}) is published; "
                    f"got revision={resolved_rev} components={ordered}"
                )
        else:
            if not set(BASE_COMPONENTS) <= set(ordered):
                raise ValueError(
                    f"full calibration requires at least {BASE_COMPONENTS}; got {ordered}"
                )

    return PANEL_REPO, resolved_rev, ordered


def component_input_ids_sha256(sequences: Iterable[Sequence[int]]) -> str:
    """SHA256 of little-endian int64 token IDs concatenated in record order.

    Matches both ``exl3_standard``'s packed ``int64`` array digest and QMLA's
    per-component ``input_ids_sha256``.
    """
    hasher = hashlib.sha256()
    for seq in sequences:
        buf = array("q", seq)
        if sys.byteorder != "little":
            buf.byteswap()
        hasher.update(buf.tobytes())
    return hasher.hexdigest()


def max_context_from_config(source_snapshot: Path | str) -> int | None:
    """Resolve the checkpoint's maximum context length from ``config.json``."""
    cfg_path = Path(source_snapshot) / "config.json"
    if not cfg_path.is_file():
        return None
    doc = json.loads(cfg_path.read_text(encoding="utf-8"))
    for scope in (doc.get("text_config") or {}, doc):
        if not isinstance(scope, Mapping):
            continue
        for key in ("max_position_embeddings", "model_max_length", "max_seq_len"):
            val = scope.get(key)
            if type(val) is int and val > 0:
                return val
    return None


# ---------------------------------------------------------------------------
# Multi-turn long_context renderer (single pass, fail-closed)
# ---------------------------------------------------------------------------

def _canonical_content_hash(messages: list[dict[str, Any]], tools: Any) -> str:
    return hashlib.sha256(canonical({"messages": messages, "tools": tools})).hexdigest()


def validate_long_context_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a model-neutral ``long_context`` record before template rendering."""
    if not isinstance(record, Mapping):
        raise ValueError("long_context record must be a mapping")
    forbidden = _FORBIDDEN_RECORD_KEYS.intersection(record.keys())
    if forbidden:
        raise ValueError(f"long_context record must not store token IDs: {sorted(forbidden)}")

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("long_context record requires a nonempty string id")

    split = record.get("split")
    if split not in LONG_CONTEXT_SPLITS:
        raise ValueError(f"{record_id}: invalid long_context split {split!r}")

    length_class = record.get("length_class")
    if length_class not in LONG_CONTEXT_LENGTH_BOUNDS:
        raise ValueError(f"{record_id}: unknown long_context length_class {length_class!r}")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record_id}: long_context record requires nonempty messages list")

    first_role = messages[0].get("role") if isinstance(messages[0], Mapping) else None
    if first_role not in ("system", "user"):
        raise ValueError(f"{record_id}: first message role must be system or user, got {first_role!r}")

    tools = record.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or not all(isinstance(t, Mapping) for t in tools):
            raise ValueError(f"{record_id}: tools must be a list of tool schema objects")
        for tool in tools:
            fn = tool.get("function")
            if not isinstance(fn, Mapping) or not isinstance(fn.get("name"), str) or not fn["name"]:
                raise ValueError(f"{record_id}: each tool requires function.name")

    expected_hash = record.get("content_sha256")
    if expected_hash is not None:
        actual_hash = _canonical_content_hash(messages, tools)
        if actual_hash != expected_hash:
            raise ValueError(f"{record_id}: long_context content_sha256 mismatch")

    has_tool_calls = False
    has_tool_results = False
    has_reasoning = False
    pending_ids: set[str] = set()
    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise ValueError(f"{record_id}: message[{idx}] must be an object")
        role = msg.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"{record_id}: message[{idx}] has invalid role {role!r}")
        if role == "assistant":
            if pending_ids:
                raise ValueError(f"{record_id}: unanswered tool_calls before assistant turn {idx}: {sorted(pending_ids)}")
            rc = msg.get("reasoning_content")
            if rc is not None:
                if not isinstance(rc, str):
                    raise ValueError(f"{record_id}: message[{idx}].reasoning_content must be str")
                if rc.strip():
                    has_reasoning = True
            tcs = msg.get("tool_calls")
            if tcs:
                if not isinstance(tcs, list):
                    raise ValueError(f"{record_id}: message[{idx}].tool_calls must be a list")
                has_tool_calls = True
                for tc in tcs:
                    if not isinstance(tc, Mapping):
                        raise ValueError(f"{record_id}: invalid tool_call entry in message[{idx}]")
                    fn = tc.get("function")
                    if not isinstance(fn, Mapping) or not isinstance(fn.get("name"), str) or not fn["name"]:
                        raise ValueError(f"{record_id}: tool_call missing function.name in message[{idx}]")
                    args = fn.get("arguments")
                    if not isinstance(args, (dict, str)):
                        raise ValueError(f"{record_id}: tool_call arguments must be dict or str in message[{idx}]")
                    tc_id = tc.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        pending_ids.add(tc_id)
        elif role == "tool":
            has_tool_results = True
            content = msg.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{record_id}: tool message[{idx}] requires string content")
            tc_id = msg.get("tool_call_id")
            if isinstance(tc_id, str) and tc_id:
                pending_ids.discard(tc_id)
        else:
            content = msg.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{record_id}: {role} message[{idx}] requires string content")

    if pending_ids:
        raise ValueError(f"{record_id}: ends with unanswered tool_calls: {sorted(pending_ids)}")
    if has_tool_calls and not has_tool_results:
        raise ValueError(f"{record_id}: tool_calls present without matching tool results")

    return dict(
        id=record_id,
        split=split,
        length_class=length_class,
        has_tools=bool(tools),
        has_tool_calls=has_tool_calls,
        has_tool_results=has_tool_results,
        has_reasoning=has_reasoning,
        content_sha256=expected_hash or _canonical_content_hash(messages, tools),
    )


def _check_template_tool_support(tokenizer: Any, *, need_tools: bool, need_tool_calls: bool, need_tool_role: bool) -> None:
    """Fail closed if a Jinja chat template cannot express tools/tool_calls/tool results."""
    if not (need_tools or need_tool_calls or need_tool_role):
        return
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, Mapping):
        template = template.get("default") or next(iter(template.values()), None)
    if isinstance(template, str):
        if need_tools and "tools" not in template:
            raise ValueError("checkpoint chat template does not support tools declarations")
        if need_tool_calls and "tool_calls" not in template:
            raise ValueError("checkpoint chat template cannot express assistant tool_calls")
        if need_tool_role and ("'tool'" not in template and '"tool"' not in template):
            raise ValueError("checkpoint chat template cannot express role='tool' results")


def _accepted_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Filter optional HF kwargs (like return_dict) only when the callable rejects **kwargs."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    accepted = set(sig.parameters)
    filtered = {}
    for key, value in kwargs.items():
        if key in accepted:
            filtered[key] = value
        elif key == "return_dict":
            continue
        else:
            # Explicit template arguments (e.g. tools, reasoning_effort) must not be silently dropped.
            raise ValueError(f"tokenizer.apply_chat_template does not accept required argument {key!r}")
    return filtered


def _unwrap_token_ids(raw: Any, record_id: str) -> list[int]:
    if isinstance(raw, Mapping):
        raw = raw.get("input_ids")
    elif hasattr(raw, "input_ids"):
        raw = raw.input_ids
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{record_id}: tokenizer.apply_chat_template returned empty or non-list token IDs")
    if isinstance(raw[0], list):
        if len(raw) != 1:
            raise ValueError(f"{record_id}: expected single unbatched token sequence")
        raw = raw[0]
    return raw


def _representative_snippet(text: str, limit: int = 48) -> str:
    """Extract a clean non-whitespace substring from a message/tool output for presence check."""
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 8:
            return stripped[:limit]
    stripped = text.strip()
    return stripped[:limit]


def _verify_decoded_tools_and_results(
    decoded: str,
    record: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> None:
    """Verify the rendered token stream preserved tools, tool calls, and tool results."""
    record_id = meta["id"]
    if meta["has_tools"]:
        for tool in record["tools"]:
            name = tool["function"]["name"]
            if name not in decoded:
                raise ValueError(
                    f"{record_id}: rendered tokens dropped declared tool {name!r}; "
                    "checkpoint template does not properly render tools"
                )
    if meta["has_tool_calls"]:
        for idx, msg in enumerate(record["messages"]):
            for tc in msg.get("tool_calls") or ():
                fn = tc["function"]
                fn_name = fn["name"]
                if fn_name not in decoded:
                    raise ValueError(
                        f"{record_id}: rendered tokens dropped tool_call function {fn_name!r} at turn {idx}"
                    )
                args = fn.get("arguments")
                if isinstance(args, Mapping) and args:
                    first_key = next(iter(args.keys()))
                    if first_key not in decoded:
                        raise ValueError(
                            f"{record_id}: rendered tokens dropped tool_call argument key {first_key!r} at turn {idx}"
                        )
    if meta["has_tool_results"]:
        for idx, msg in enumerate(record["messages"]):
            if msg["role"] == "tool":
                snippet = _representative_snippet(msg["content"])
                if snippet and snippet not in decoded:
                    raise ValueError(
                        f"{record_id}: rendered tokens dropped tool result content at turn {idx}"
                    )


def _reasoning_preservation_stats(decoded: str, messages: Sequence[Mapping[str, Any]]) -> dict[str, bool | None]:
    """Diagnostic-only check of whether the template kept history vs final reasoning."""
    rc_snippets: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                snippet = _representative_snippet(rc)
                if snippet:
                    rc_snippets.append(snippet)
    if not rc_snippets:
        return {"history_reasoning_preserved": None, "final_reasoning_preserved": None}
    final_kept = rc_snippets[-1] in decoded
    history_kept = all(s in decoded for s in rc_snippets[:-1]) if len(rc_snippets) >= 2 else None
    return {
        "history_reasoning_preserved": history_kept,
        "final_reasoning_preserved": final_kept,
    }


def render_long_context_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    chat_kwargs: Mapping[str, Any] | None = None,
    max_context: int | None = None,
    length_bounds: Mapping[str, tuple[int, int]] = LONG_CONTEXT_LENGTH_BOUNDS,
) -> tuple[list[int], dict[str, Any]]:
    """Render one multi-turn ``long_context`` record in a single pass.

    - Calls ``tokenizer.apply_chat_template(messages, tools=..., tokenize=True, ...)``
      exactly once; never re-encodes rendered text.
    - Fails closed if the template cannot express tool calls or tool results,
      if the rendered length falls outside the record's declared length class,
      or if it exceeds ``max_context``.
    - Leaves ``reasoning_content`` untouched so the checkpoint's own template
      applies its exact native history-reasoning rules.
    """
    meta = validate_long_context_record(record)
    record_id = meta["id"]

    _check_template_tool_support(
        tokenizer,
        need_tools=meta["has_tools"],
        need_tool_calls=meta["has_tool_calls"],
        need_tool_role=meta["has_tool_results"],
    )

    # Start from the checkpoint's chat kwargs (preserving thinking/effort/date mode)
    # and enforce single-pass unpadded/untruncated rendering with add_generation_prompt=False
    # for completed multi-turn traces.
    call_kwargs: dict[str, Any] = dict(chat_kwargs or {})
    call_kwargs.update(
        tokenize=True,
        add_generation_prompt=False,
        padding=False,
        truncation=False,
        return_dict=False,
    )
    if meta["has_tools"]:
        call_kwargs["tools"] = record["tools"]

    filtered_kwargs = _accepted_kwargs(tokenizer.apply_chat_template, call_kwargs)
    try:
        raw_ids = tokenizer.apply_chat_template(record["messages"], **filtered_kwargs)
    except Exception as exc:
        raise ValueError(
            f"{record_id}: checkpoint chat template failed to render multi-turn record: {exc}"
        ) from exc

    ids = _unwrap_token_ids(raw_ids, record_id)
    vocab_size = len(tokenizer)
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError(f"{record_id}: invalid token IDs from checkpoint template")

    if hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if not isinstance(decoded, str):
            raise ValueError(f"{record_id}: tokenizer.decode did not return a string")
        _verify_decoded_tools_and_results(decoded, record, meta)
        reasoning_stats = _reasoning_preservation_stats(decoded, record["messages"])
    else:
        if meta["has_tools"] or meta["has_tool_calls"] or meta["has_tool_results"]:
            raise ValueError(
                f"{record_id}: tokenizer lacks decode(); cannot verify tool call/result rendering"
            )
        reasoning_stats = {"history_reasoning_preserved": None, "final_reasoning_preserved": None}

    token_count = len(ids)
    low, high = length_bounds[meta["length_class"]]
    if not low <= token_count <= high:
        raise ValueError(
            f"{record_id}: rendered length {token_count} is outside declared "
            f"{meta['length_class']!r} bounds [{low}, {high}]"
        )
    if max_context is not None:
        if type(max_context) is not int or max_context <= 0:
            raise ValueError(f"max_context must be a positive integer, got {max_context!r}")
        if token_count > max_context:
            raise ValueError(
                f"{record_id}: rendered length {token_count} exceeds model max_context {max_context}"
            )

    entry = dict(
        id=record_id,
        component="long_context",
        split=meta["split"],
        length_class=meta["length_class"],
        tokens=token_count,
        content_sha256=meta["content_sha256"],
        input_ids_sha256=ids_sha256(ids),
        has_tools=meta["has_tools"],
        has_tool_calls=meta["has_tool_calls"],
        has_reasoning=meta["has_reasoning"],
        **reasoning_stats,
    )
    return ids, entry


def tokenize_long_context_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    chat_kwargs: Mapping[str, Any] | None = None,
    max_context: int | None = None,
    length_bounds: Mapping[str, tuple[int, int]] = LONG_CONTEXT_LENGTH_BOUNDS,
) -> tuple[array, list[int], dict[str, Any]]:
    """Render an ordered sequence of ``long_context`` records into flat int64 IDs + offsets."""
    flat, offsets, records = array("q"), [0], []
    stream = hashlib.sha256()
    seen_ids: set[str] = set()
    by_class: dict[str, dict[str, int]] = {}
    id_sequences: list[list[int]] = []

    for row in rows:
        ids, entry = render_long_context_record(
            row,
            tokenizer,
            chat_kwargs=chat_kwargs,
            max_context=max_context,
            length_bounds=length_bounds,
        )
        if entry["id"] in seen_ids:
            raise ValueError(f"duplicate long_context record id: {entry['id']}")
        seen_ids.add(entry["id"])
        flat.extend(ids)
        offsets.append(len(flat))
        id_sequences.append(ids)
        stream.update(canonical({"id": entry["id"], "input_ids": ids}) + b"\n")
        records.append(entry)
        cls_bucket = by_class.setdefault(entry["length_class"], {"count": 0, "tokens": 0})
        cls_bucket["count"] += 1
        cls_bucket["tokens"] += entry["tokens"]

    if not records:
        raise ValueError("empty tokenized long_context split")

    report = dict(
        records=records,
        count=len(records),
        tokens=len(flat),
        shortest=min(r["tokens"] for r in records),
        longest=max(r["tokens"] for r in records),
        input_ids_sha256=component_input_ids_sha256(id_sequences),
        token_stream_sha256=stream.hexdigest(),
        length_classes=by_class,
    )
    return flat, offsets, report


# ---------------------------------------------------------------------------
# Panel snapshot reading & attestation (2-component and 3-component v2)
# ---------------------------------------------------------------------------

def _read_sha256sums(root: Path) -> dict[str, str]:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ValueError(f"missing panel SHA256SUMS: {sums_path}")
    result: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def read_long_context_component(
    root: Path,
    *,
    hashes: Mapping[str, str],
    agentic_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Validate and load ``long_context/{16k,32k}.jsonl`` split by calibration/heldout."""
    required_files = [
        "long_context/manifest.json",
        *(f"long_context/{subset}.jsonl" for subset in LONG_CONTEXT_SUBSETS),
    ]
    for rel in required_files:
        if rel not in hashes:
            raise ValueError(f"panel SHA256SUMS missing long_context file: {rel}")
        target = root / rel
        if not target.is_file() or sha256_file(target) != hashes[rel]:
            raise ValueError(f"long_context file missing or hash differs: {rel}")

    lc_manifest = json.loads((root / "long_context/manifest.json").read_text(encoding="utf-8"))
    if lc_manifest.get("schema") != LONG_CONTEXT_SCHEMA or lc_manifest.get("component") != "long_context":
        raise ValueError("long_context manifest schema or component differs")

    subsets_decl = lc_manifest.get("subsets", {})
    by_split: dict[str, list[dict[str, Any]]] = {"calibration": [], "heldout": []}
    all_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for subset in LONG_CONTEXT_SUBSETS:
        rel = f"long_context/{subset}.jsonl"
        subset_meta = subsets_decl.get(subset)
        if not isinstance(subset_meta, Mapping) or subset_meta.get("file") != rel or subset_meta.get("sha256") != hashes[rel]:
            raise ValueError(f"long_context subset manifest mismatch for {subset}")
        lines = (root / rel).read_text(encoding="utf-8").splitlines()
        if len(lines) != subset_meta.get("records"):
            raise ValueError(f"long_context subset {subset} record count differs")
        for line in lines:
            row = json.loads(line)
            meta = validate_long_context_record(row)
            if meta["length_class"] != subset:
                raise ValueError(f"{meta['id']}: length_class {meta['length_class']!r} != {subset!r}")
            if meta["id"] in seen_ids:
                raise ValueError(f"duplicate long_context record id: {meta['id']}")
            seen_ids.add(meta["id"])
            by_split[meta["split"]].append(row)
            all_records.append(row)

    # Verify manifest record summary matches loaded JSONL records.
    manifest_records = {r["id"]: r for r in lc_manifest.get("records", [])}
    if set(manifest_records) != seen_ids:
        raise ValueError("long_context manifest record inventory differs from JSONL payloads")
    for row in all_records:
        decl = manifest_records[row["id"]]
        for field in ("split", "length_class", "group", "content_sha256", "source_identity"):
            if decl.get(field) != row.get(field):
                raise ValueError(f"{row['id']}: long_context manifest {field} differs from payload")

    # Enforce strict calibration / heldout disjointness (and cross-component disjointness).
    cal_rows, hold_rows = by_split["calibration"], by_split["heldout"]
    if not cal_rows or not hold_rows:
        raise ValueError("long_context requires nonempty calibration and heldout splits")

    cal_groups = {r["group"] for r in cal_rows}
    hold_groups = {r["group"] for r in hold_rows}
    cal_hashes = {r["content_sha256"] for r in cal_rows}
    hold_hashes = {r["content_sha256"] for r in hold_rows}
    cal_sources = {r["source_identity"] for r in cal_rows}
    hold_sources = {r["source_identity"] for r in hold_rows}
    if (cal_groups & hold_groups) or (cal_hashes & hold_hashes) or (cal_sources & hold_sources):
        raise ValueError("long_context calibration and heldout splits overlap")

    if agentic_provenance is not None:
        ag_cal_groups = {r["group"] for r in agentic_provenance["splits"]["calibration"]["records"]}
        ag_hold_groups = {r["group"] for r in agentic_provenance["splits"]["heldout"]["records"]}
        if (cal_groups & ag_hold_groups) or (hold_groups & ag_cal_groups):
            raise ValueError("long_context splits overlap agentic splits across calibration/heldout")

    return by_split, lc_manifest


def read_panel_v2(
    root: Path | str,
    *,
    panel_revision: str | None = None,
    panel_sha256sums: str | None = None,
    components: Sequence[str] | None = None,
    full: bool = True,
    three_component_revision: str | None = None,
    three_component_sha256sums: str | None = None,
) -> dict[str, Any]:
    """Attest and load the pinned ``agentic-quant-panel-v2`` snapshot.

    Supports both the two-component revision (``agentic`` + ``exl3_standard``)
    and the three-component revision (adding ``long_context``).
    """
    root = Path(root)
    repo_id, resolved_rev, resolved_components = resolve_panel_policy(
        panel_revision=panel_revision,
        components=components,
        full=full,
        three_component_revision=three_component_revision,
    )
    if root.name != resolved_rev:
        raise ValueError(
            f"require the pinned v2 snapshot {resolved_rev}, got directory {root.name!r}"
        )

    sums_sha = sha256_file(root / "SHA256SUMS")
    expected_sums = panel_sha256sums
    if expected_sums is None:
        if resolved_rev == TWO_COMPONENT_REVISION:
            expected_sums = TWO_COMPONENT_SHA256SUMS_SHA256
        else:
            active_three = THREE_COMPONENT_REVISION if three_component_revision is None else three_component_revision
            if active_three is not None and resolved_rev == active_three:
                expected_sums = (
                    THREE_COMPONENT_SHA256SUMS_SHA256
                    if three_component_sha256sums is None
                    else three_component_sha256sums
                )
    if expected_sums is not None and sums_sha != expected_sums:
        raise ValueError(
            f"panel SHA256SUMS digest mismatch for {resolved_rev}: expected {expected_sums}, got {sums_sha}"
        )

    hashes = _read_sha256sums(root)
    base_names = [
        "panel.json",
        "manifest.json",
        *(s + ".jsonl" for s in SPLITS),
        *("data/" + s + ".parquet" for s in (*SPLITS, "exl3_standard")),
        "exl3_standard/manifest.json",
        "exl3_standard/segments.jsonl",
        "exl3_standard/upstream/tokenizer.py",
        EXL3_STANDARD_PACKER,
    ]
    missing = [name for name in base_names if name not in hashes]
    if missing:
        raise ValueError(f"panel SHA256SUMS missing expected files: {missing}")

    if any(
        hashes[name] != v1_panel.PAYLOAD_SHA256[name]
        for name in ("manifest.json", *(s + ".jsonl" for s in SPLITS))
    ):
        raise ValueError("v2 agentic provenance differs from the released v1 panel")

    agentic_rows, agentic_provenance = v1_panel.read_panel(
        root, payload_sha256={name: hashes[name] for name in base_names}
    )

    long_context_rows: dict[str, list[dict[str, Any]]] | None = None
    long_context_manifest: dict[str, Any] | None = None
    if "long_context" in resolved_components:
        long_context_rows, long_context_manifest = read_long_context_component(
            root, hashes=hashes, agentic_provenance=agentic_provenance
        )

    dataset = {
        "repo_id": repo_id,
        "revision": resolved_rev,
        "sha256sums_sha256": sums_sha,
    }
    return {
        "dataset": dataset,
        "components": resolved_components,
        "agentic_rows": agentic_rows,
        "agentic_provenance": agentic_provenance,
        "long_context_rows": long_context_rows,
        "long_context_manifest": long_context_manifest,
        "hashes": hashes,
    }


# ---------------------------------------------------------------------------
# exl3_standard packing + combined corpus preparation & loading
# ---------------------------------------------------------------------------

def pack_exl3_standard(
    corpus_root: Path,
    source_snapshot: Path,
    stage: Path,
    tokenizer_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the panel's ``reproduction/pack_exl3_standard.py`` and attest its rows."""
    import torch
    from safetensors.torch import load_file

    packed = stage / "exl3_standard.safetensors"
    subprocess.run(
        [sys.executable, str(corpus_root / EXL3_STANDARD_PACKER), str(corpus_root), str(source_snapshot), str(packed)],
        check=True,
    )
    receipt_path = packed.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    values = load_file(str(packed))["input_ids"]
    digest = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    if (
        values.shape != (250, 2048)
        or values.dtype != torch.int64
        or receipt.get("contract") != EXL3_STANDARD_CONTRACT
        or receipt.get("tokenizer_sha256") != tokenizer_sha256
        or receipt.get("input_ids_sha256") != digest
    ):
        raise ValueError("ExLlamaV3 calibration pack contract or checksum differs")

    rows_out: list[dict[str, Any]] = []
    stream = hashlib.sha256()
    for index, (row, provenance) in enumerate(zip(values, receipt["rows"], strict=True)):
        tokens = row.tolist()
        row_id = f"exl3_standard/{index:03d}"
        if ids_sha256(tokens) != provenance["token_ids_sha256"]:
            raise ValueError(f"standard panel row digest differs: {row_id}")
        stream.update(canonical({"id": row_id, "input_ids": tokens}) + b"\n")
        rows_out.append(
            dict(
                id=row_id,
                component="exl3_standard",
                input_ids=tokens,
                tokens=len(tokens),
                input_ids_sha256=provenance["token_ids_sha256"],
                source=provenance["source"],
                kind=provenance["kind"],
            )
        )
    packed.unlink()  # The combined split owns all token IDs.
    meta = dict(
        contract=receipt["contract"],
        count=values.shape[0],
        tokens=values.numel(),
        input_ids_sha256=digest,
        token_stream_sha256=stream.hexdigest(),
        receipt=receipt_path.name,
    )
    return rows_out, meta


def _append_records_to_split(
    ids: array,
    offsets: list[int],
    report: dict[str, Any],
    extra_records: Sequence[Mapping[str, Any]],
    extra_id_sequences: Sequence[Sequence[int]],
) -> None:
    """Append ordered token records to an existing split and recompute stream hash."""
    stream = hashlib.sha256()
    for i, row in enumerate(report["records"]):
        stream.update(
            canonical({"id": row["id"], "input_ids": ids[offsets[i] : offsets[i + 1]].tolist()}) + b"\n"
        )
    for meta, tokens in zip(extra_records, extra_id_sequences, strict=True):
        tok_list = list(tokens)
        ids.extend(tok_list)
        offsets.append(len(ids))
        entry = {k: v for k, v in meta.items() if k != "input_ids"}
        report["records"].append(entry)
        stream.update(canonical({"id": entry["id"], "input_ids": tok_list}) + b"\n")

    lengths = [r["tokens"] for r in report["records"]]
    report.update(
        tokens=len(ids),
        count=len(offsets) - 1,
        shortest=min(lengths),
        longest=max(lengths),
        token_stream_sha256=stream.hexdigest(),
    )


def prepare_panel_corpus(
    source: Path | str,
    corpus: Path | str,
    output: Path | str,
    *,
    tokenizer: Any = None,
    source_repo: str,
    source_revision: str,
    chat_kwargs: Mapping[str, Any],
    long_context_chat_kwargs: Mapping[str, Any] | None = None,
    schema: str = PREPARED_SCHEMA,
    contract: Mapping[str, Any] | None = None,
    assets: Sequence[str] = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"),
    components: Sequence[str] | None = None,
    full: bool = True,
    panel_revision: str | None = None,
    panel_sha256sums: str | None = None,
    three_component_revision: str | None = None,
    three_component_sha256sums: str | None = None,
    max_context: int | None = None,
    length_bounds: Mapping[str, tuple[int, int]] = LONG_CONTEXT_LENGTH_BOUNDS,
    long_witness_fn: Callable[[Any, Mapping[str, Any]], dict[str, Any]] | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Prepare and attest a 2- or 3-component calibration corpus on disk.

    Stream order in ``calibration.safetensors``:
    ``agentic`` -> ``exl3_standard`` -> ``long_context`` (calibration split only).
    Stream order in ``heldout.safetensors``:
    ``agentic`` -> ``long_context`` (heldout split only, when ``long_context`` is active).
    ``screening.safetensors`` holds ``agentic`` screening prompts.
    """
    import torch
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    source = Path(source).expanduser().absolute()
    corpus = Path(corpus).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    if source.name != source_revision or output.exists():
        raise ValueError("require the pinned source and a fresh corpus directory")

    present_assets = [name for name in assets if (source / name).is_file()]
    missing_assets = [name for name in assets if name not in present_assets]
    if missing_assets:
        raise ValueError(f"missing source tokenizer assets: {missing_assets}")
    identity = {name: sha256_file(source / name) for name in assets}

    if tokenizer is None:
        if trust_remote_code:
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
        else:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
            if (
                not tokenizer.is_fast
                or ("chat_template.jinja" in identity and (
                    not isinstance(tokenizer.chat_template, str)
                    or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != identity["chat_template.jinja"]
                ))
            ):
                raise ValueError("native tokenizer did not load the pinned chat template")

    if max_context is None:
        max_context = max_context_from_config(source)

    panel = read_panel_v2(
        corpus,
        panel_revision=panel_revision,
        panel_sha256sums=panel_sha256sums,
        components=components,
        full=full,
        three_component_revision=three_component_revision,
        three_component_sha256sums=three_component_sha256sums,
    )
    active_components: tuple[str, ...] = panel["components"]
    lc_chat = dict(chat_kwargs if long_context_chat_kwargs is None else long_context_chat_kwargs)

    if contract is None:
        contract_dict: dict[str, Any] = dict(
            renderer="native-chat-template",
            raw_calibration=EXL3_STANDARD_CONTRACT,
            batch_size=1,
            packing=False,
            order="dataset",
            **dict(chat_kwargs),
        )
        if "long_context" in active_components:
            contract_dict["long_context"] = LONG_CONTEXT_SCHEMA
    else:
        contract_dict = dict(contract)

    manifest: dict[str, Any] = dict(
        schema=schema,
        status="passed",
        contract=contract_dict,
        dataset=panel["dataset"],
        component_order=list(active_components),
        calibration_mode="full" if full else "subset",
        components={},
        historical_selection=panel["agentic_provenance"]["selection"],
        source=dict(repo_id=source_repo, revision=source_revision, files=identity),
        software=software_identity(),
        tokenizer=dict(vocab_size=len(tokenizer), max_context=max_context),
        splits={},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in SPLITS:
            ids = array("q")
            offsets = [0]
            report: dict[str, Any] = dict(records=[], count=0, tokens=0, shortest=0, longest=0, token_stream_sha256="")

            # 1. agentic (all three splits when enabled)
            if "agentic" in active_components:
                ag_ids, ag_offsets, ag_report = v1_panel.tokenize_rows(
                    panel["agentic_rows"][split], tokenizer, chat_kwargs=dict(chat_kwargs)
                )
                for row in ag_report["records"]:
                    row["component"] = "agentic"
                ag_seqs = [
                    ag_ids[ag_offsets[i] : ag_offsets[i + 1]].tolist()
                    for i in range(len(ag_report["records"]))
                ]
                ag_input_sha = component_input_ids_sha256(ag_seqs)
                ids, offsets, report = ag_ids, ag_offsets, ag_report
                if split == "calibration":
                    manifest["components"]["agentic"] = dict(
                        count=ag_report["count"],
                        tokens=ag_report["tokens"],
                        input_ids_sha256=ag_input_sha,
                        token_stream_sha256=ag_report["token_stream_sha256"],
                    )

            # 2. exl3_standard (calibration split only)
            if split == "calibration" and "exl3_standard" in active_components:
                tok_sha = identity.get("tokenizer.json")
                if tok_sha is None:
                    raise ValueError("exl3_standard packing requires tokenizer.json in source assets")
                std_rows, std_meta = pack_exl3_standard(corpus, source, stage, tok_sha)
                _append_records_to_split(
                    ids, offsets, report, std_rows, [row["input_ids"] for row in std_rows]
                )
                manifest["components"]["exl3_standard"] = std_meta

            # 3. long_context (calibration split -> calibration, heldout split -> heldout)
            if "long_context" in active_components and split in LONG_CONTEXT_SPLITS:
                lc_rows = panel["long_context_rows"][split]
                lc_ids, lc_offsets, lc_report = tokenize_long_context_rows(
                    lc_rows,
                    tokenizer,
                    chat_kwargs=lc_chat,
                    max_context=max_context,
                    length_bounds=length_bounds,
                )
                lc_seqs = [
                    lc_ids[lc_offsets[i] : lc_offsets[i + 1]].tolist()
                    for i in range(len(lc_report["records"]))
                ]
                _append_records_to_split(ids, offsets, report, lc_report["records"], lc_seqs)
                if split == "calibration":
                    manifest["components"]["long_context"] = dict(
                        schema=LONG_CONTEXT_SCHEMA,
                        count=lc_report["count"],
                        tokens=lc_report["tokens"],
                        input_ids_sha256=lc_report["input_ids_sha256"],
                        token_stream_sha256=lc_report["token_stream_sha256"],
                        length_classes=lc_report["length_classes"],
                    )
                elif split == "heldout":
                    manifest["components"]["long_context"]["heldout"] = dict(
                        count=lc_report["count"],
                        tokens=lc_report["tokens"],
                        input_ids_sha256=lc_report["input_ids_sha256"],
                        token_stream_sha256=lc_report["token_stream_sha256"],
                        length_classes=lc_report["length_classes"],
                    )

            if not report["records"]:
                continue
            path = stage / (split + ".safetensors")
            receipt = checkpoint_io().write_safetensors(
                path,
                dict(
                    input_ids=torch.frombuffer(ids, dtype=torch.int64).clone(),
                    offsets=torch.tensor(offsets, dtype=torch.int64),
                ),
                metadata=dict(schema=schema, split=split),
                hash_algorithm="sha256",
            )
            manifest["splits"][split] = dict(report, file=path.name, sha256=receipt["sha256"])

        if long_witness_fn is not None:
            witness = long_witness_fn(tokenizer, chat_kwargs)
            atomic_json(stage / "long-context-smoke.json", witness)
            manifest["long_context_smoke"] = dict(
                file="long-context-smoke.json",
                tokens=witness["tokens"],
                sha256=sha256_file(stage / "long-context-smoke.json"),
                hessian_eligible=False,
            )

        if identity != {name: sha256_file(source / name) for name in assets}:
            raise ValueError("source tokenizer changed during preparation")

        atomic_json(stage / "manifest.json", manifest)
        for split in manifest["splits"]:
            load_token_records(
                stage,
                split,
                schema=schema,
                contract=contract_dict,
                dataset=panel["dataset"],
                full=full,
                three_component_revision=three_component_revision,
            )
        stage.rename(output)
        fsync_directory(output.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def _infer_record_component(row: Mapping[str, Any], index: int, manifest: Mapping[str, Any], split: str) -> str:
    """Identify the component of a record, supporting both tagged v3 and legacy v2 manifests."""
    comp = row.get("component")
    if isinstance(comp, str):
        return comp
    row_id = str(row.get("id", ""))
    if row_id.startswith("exl3_standard/"):
        return "exl3_standard"
    if split == "calibration":
        comps = manifest.get("components", {})
        cursor = 0
        for name in COMPONENT_ORDER:
            if name not in comps:
                continue
            count = int(comps[name]["count"])
            if cursor <= index < cursor + count:
                return name
            cursor += count
    return "agentic"


def _verify_component_order_and_hashes(
    records_with_ids: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    split: str,
    *,
    require_components: Sequence[str] | None,
    full: bool | None,
    three_component_revision: str | None,
) -> None:
    """Validate component ordering, per-component hashes, and full-run policy."""
    comps_meta = manifest.get("components")
    if not isinstance(comps_meta, Mapping) or not comps_meta:
        if require_components is not None:
            raise ValueError("prepared manifest is missing components provenance")
        return

    present_order = [name for name in COMPONENT_ORDER if name in comps_meta]
    unknown = set(comps_meta.keys()) - set(COMPONENT_ORDER)
    if unknown:
        raise ValueError(f"unknown components in prepared manifest: {sorted(unknown)}")

    declared_order = manifest.get("component_order")
    if declared_order is not None and list(declared_order) != present_order:
        raise ValueError(
            f"prepared corpus component_order {declared_order} disagrees with canonical order {present_order}"
        )

    active_three = THREE_COMPONENT_REVISION if three_component_revision is None else three_component_revision
    enforce_full = (manifest.get("calibration_mode") == "full") if full is None else full
    if split == "calibration" and enforce_full and active_three is not None:
        if tuple(present_order) != FULL_COMPONENTS:
            raise ValueError(
                f"full calibration requires all three components {FULL_COMPONENTS} "
                f"once revision {active_three} is published; prepared corpus has {tuple(present_order)}"
            )

    if require_components is not None:
        req = tuple(name for name in COMPONENT_ORDER if name in require_components)
        if set(req) != set(require_components):
            raise ValueError(f"unknown required component in {require_components!r}")
        missing = [name for name in req if name not in comps_meta]
        if missing:
            raise ValueError(f"prepared corpus missing required components: {missing}")

    if split == "calibration":
        # Verify strict canonical component order in the calibration stream and per-component hashes.
        cursor = 0
        for comp_name in present_order:
            comp_info = comps_meta[comp_name]
            count = int(comp_info["count"])
            chunk = records_with_ids[cursor : cursor + count]
            if len(chunk) != count:
                raise ValueError(f"calibration stream truncated in component {comp_name!r}")
            if any(row["component"] != comp_name for row in chunk):
                raise ValueError(f"calibration stream component order mismatch inside {comp_name!r}")
            tokens = sum(len(row["input_ids"]) for row in chunk)
            if tokens != int(comp_info["tokens"]):
                raise ValueError(f"component {comp_name!r} token count differs from manifest")
            expected_input_sha = comp_info.get("input_ids_sha256")
            if expected_input_sha is not None:
                actual_sha = component_input_ids_sha256(row["input_ids"] for row in chunk)
                if actual_sha != expected_input_sha:
                    raise ValueError(f"component {comp_name!r} input_ids_sha256 differs from manifest")
            cursor += count
        if cursor != len(records_with_ids):
            raise ValueError("calibration stream contains trailing records outside declared components")
    elif split == "heldout":
        # Heldout must contain only agentic then optional long_context, never exl3_standard.
        seen_lc = False
        lc_chunk = []
        for row in records_with_ids:
            if row["component"] == "exl3_standard":
                raise ValueError("exl3_standard must never appear in heldout split")
            if row["component"] == "long_context":
                seen_lc = True
                lc_chunk.append(row)
            elif seen_lc:
                raise ValueError("heldout stream out of order: agentic record after long_context")
        lc_heldout = comps_meta.get("long_context", {}).get("heldout") if "long_context" in comps_meta else None
        if lc_heldout is not None:
            if len(lc_chunk) != int(lc_heldout["count"]):
                raise ValueError("long_context heldout record count differs from manifest")
            if sum(len(r["input_ids"]) for r in lc_chunk) != int(lc_heldout["tokens"]):
                raise ValueError("long_context heldout token count differs from manifest")
            if component_input_ids_sha256(r["input_ids"] for r in lc_chunk) != lc_heldout["input_ids_sha256"]:
                raise ValueError("long_context heldout input_ids_sha256 differs from manifest")


def load_token_records(
    root: Path | str,
    split: str = "calibration",
    *,
    schema: str | None = PREPARED_SCHEMA,
    contract: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    require_components: Sequence[str] | None = None,
    full: bool | None = None,
    three_component_revision: str | None = None,
) -> list[dict[str, Any]]:
    """Verify a prepared split and return ordered token records with component tags."""
    import torch
    from safetensors.torch import load_file

    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("prepared corpus contract differs")
    if schema is not None and manifest.get("schema") != schema:
        raise ValueError("prepared corpus contract differs")
    if contract is not None and manifest.get("contract") != dict(contract):
        raise ValueError("prepared corpus contract differs")
    if dataset is not None and manifest.get("dataset") != dict(dataset):
        raise ValueError("prepared corpus contract differs")

    report = manifest["splits"][split]
    if report.get("file") != split + ".safetensors":
        raise ValueError("prepared split file differs")
    path = root / report["file"]
    if sha256_file(path) != report["sha256"]:
        raise ValueError("prepared token file hash differs")
    tensors = load_file(str(path), device="cpu")
    if (
        set(tensors) != {"input_ids", "offsets"}
        or any(t.dtype != torch.int64 or t.ndim != 1 for t in tensors.values())
    ):
        raise ValueError("invalid token tensor layout")

    flat, offsets = tensors["input_ids"], tensors["offsets"].tolist()
    records = report["records"]
    if (
        not records
        or len(offsets) != len(records) + 1
        or offsets[0] != 0
        or offsets[-1] != flat.numel()
        or any(a >= b for a, b in zip(offsets, offsets[1:]))
        or len({row["id"] for row in records}) != len(records)
    ):
        raise ValueError("invalid token offsets or record identities")

    stream = hashlib.sha256()
    result: list[dict[str, Any]] = []
    vocabulary_size = manifest["tokenizer"]["vocab_size"]
    for idx, (row, start, stop) in enumerate(zip(records, offsets, offsets[1:], strict=False)):
        ids = flat[start:stop].tolist()
        if (
            len(ids) != row["tokens"]
            or ids_sha256(ids) != row["input_ids_sha256"]
            or any(not 0 <= token < vocabulary_size for token in ids)
        ):
            raise ValueError("prepared record token identity differs")
        stream.update(canonical({"id": row["id"], "input_ids": ids}) + b"\n")
        comp = _infer_record_component(row, idx, manifest, split)
        entry = dict(id=row["id"], component=comp, input_ids=ids)
        if "length_class" in row:
            entry["length_class"] = row["length_class"]
        result.append(entry)

    if (
        report["count"] != len(records)
        or report["tokens"] != flat.numel()
        or report["shortest"] != min(row["tokens"] for row in records)
        or report["longest"] != max(row["tokens"] for row in records)
        or report["token_stream_sha256"] != stream.hexdigest()
    ):
        raise ValueError("prepared split token summary differs")

    _verify_component_order_and_hashes(
        result,
        manifest,
        split,
        require_components=require_components,
        full=full,
        three_component_revision=three_component_revision,
    )
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
    three_component_revision: str | None = None,
):
    """Yield unpadded CPU calibration batches (one record per batch) in canonical order."""
    import torch

    for row in load_token_records(
        root,
        split,
        schema=schema,
        contract=contract,
        dataset=dataset,
        require_components=require_components,
        full=full,
        three_component_revision=three_component_revision,
    ):
        ids = torch.tensor([row["input_ids"]], dtype=torch.int64)
        yield dict(
            id=row["id"],
            component=row["component"],
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
        )


def load_component_records(
    root: Path | str,
    split: str = "calibration",
    *,
    components: Sequence[str] | None = None,
    schema: str | None = None,
    contract: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    full: bool | None = None,
    three_component_revision: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load selected components from a prepared directory along with frozen provenance.

    Suitable for both MC quantization runners and QMLA calibration pipelines.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    all_records = load_token_records(
        root,
        split,
        schema=schema,
        contract=contract,
        dataset=dataset,
        require_components=components,
        full=full,
        three_component_revision=three_component_revision,
    )
    wanted = set(components) if components is not None else {r["component"] for r in all_records}
    filtered = [r for r in all_records if r["component"] in wanted]
    comp_provenance: dict[str, dict[str, Any]] = {}
    for comp in COMPONENT_ORDER:
        if comp not in wanted:
            continue
        rows = [r for r in filtered if r["component"] == comp]
        if not rows:
            continue
        comp_provenance[comp] = {
            "count": len(rows),
            "sequences": len(rows),
            "tokens": sum(len(r["input_ids"]) for r in rows),
            "input_ids_sha256": component_input_ids_sha256(r["input_ids"] for r in rows),
        }
    combined_sha = hashlib.sha256(
        "\n".join(
            f"{k}:{comp_provenance[k]['input_ids_sha256']}" for k in COMPONENT_ORDER if k in comp_provenance
        ).encode("utf-8")
    ).hexdigest()
    provenance = {
        "dataset_repo": manifest["dataset"]["repo_id"],
        "dataset_revision": manifest["dataset"]["revision"],
        "component_order": [k for k in COMPONENT_ORDER if k in comp_provenance],
        "components": comp_provenance,
        "input_ids_sha256": combined_sha,
    }
    return filtered, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-revision", default=None)
    parser.add_argument("--components", default=None, help="Comma-separated components from agentic,exl3_standard,long_context")
    parser.add_argument("--subset", action="store_true", help="Allow a non-full pilot subset of components")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--date-string", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    chat_kwargs: dict[str, Any] = dict(
        tokenize=True, add_generation_prompt=True, return_dict=False, padding=False, truncation=False
    )
    if args.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.date_string is not None:
        chat_kwargs["date_string"] = args.date_string
    comp_list = [c.strip() for c in args.components.split(",") if c.strip()] if args.components else None

    manifest = prepare_panel_corpus(
        args.source,
        args.corpus,
        args.output,
        source_repo=args.source_repo,
        source_revision=args.source.resolve().name,
        chat_kwargs=chat_kwargs,
        components=comp_list,
        full=not args.subset,
        panel_revision=args.panel_revision,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "component_order": manifest["component_order"],
                "components": manifest["components"],
                "splits": {
                    name: {k: report[k] for k in ("count", "tokens", "longest")}
                    for name, report in manifest["splits"].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
