"""Tests for the shared three-component calibration corpus renderer and loader.

All multi-turn samples and tokenizers are read directly from disk (never inline
fixture transcripts):
- Raw traces: ``~/.cache/cuteafd/builds/long-context-bank/raw_traces/*.json``
- Real checkpoints: local HF hub cache (Hy4, Hy4-flash, GLM 5.3, Kimi K3.1)
- Base v2 panel: local HF hub cache (``wrldsuksgo2mars/agentic-quant-panel-v2``)
"""
from __future__ import annotations

from array import array
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import panel_corpus as pc
from hy4 import export as hy4_export
from hy4_flash import export as hy4_flash_export, prepare_corpus as hy4_flash_corpus
from kimi_k3 import prepare_corpus as v1_panel


RAW_TRACES_DIR = Path.home() / ".cache/cuteafd/builds/long-context-bank/raw_traces"


def _hf_hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache/huggingface/hub"


def _require_snapshot(repo_dir: str, revision: str) -> Path:
    snap = _hf_hub_dir() / repo_dir / "snapshots" / revision
    if not snap.is_dir():
        pytest.skip(f"cached snapshot not available locally: {snap}")
    return snap


def _load_raw_trace(filename: str) -> dict:
    path = RAW_TRACES_DIR / filename
    if not path.is_file():
        pytest.skip(f"long-context sample trace not available on disk: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Policy & default two-versus-three switch tests
# ---------------------------------------------------------------------------

def test_default_policy_before_and_after_three_component_publish():
    # Before publication (default on this branch): 2 components at 54683731eb.
    assert pc.THREE_COMPONENT_REVISION is None
    repo, rev, comps = pc.resolve_panel_policy()
    assert repo == pc.PANEL_REPO
    assert rev == pc.TWO_COMPONENT_REVISION
    assert comps == ("agentic", "exl3_standard")
    assert pc.DEFAULT_PANEL_REVISION == pc.TWO_COMPONENT_REVISION
    assert pc.DEFAULT_COMPONENTS == ("agentic", "exl3_standard")

    # Requesting long_context on the legacy 2-component revision fails closed.
    with pytest.raises(ValueError, match="not present in the two-component"):
        pc.resolve_panel_policy(
            panel_revision=pc.TWO_COMPONENT_REVISION,
            components=("agentic", "exl3_standard", "long_context"),
        )

    # Once a three-component revision is published: default switches to all 3.
    rev3 = "a" * 40
    repo3, got_rev3, comps3 = pc.resolve_panel_policy(three_component_revision=rev3)
    assert repo3 == pc.PANEL_REPO
    assert got_rev3 == rev3
    assert comps3 == ("agentic", "exl3_standard", "long_context")
    assert pc.default_panel_revision(three_component_revision=rev3) == rev3
    assert pc.default_components(three_component_revision=rev3) == ("agentic", "exl3_standard", "long_context")

    # Asking for a full run with only 2 components fails closed once rev3 exists.
    with pytest.raises(ValueError, match="full calibration requires all three"):
        pc.resolve_panel_policy(
            panel_revision=rev3,
            components=("agentic", "exl3_standard"),
            full=True,
            three_component_revision=rev3,
        )
    with pytest.raises(ValueError, match="full calibration requires all three"):
        pc.resolve_panel_policy(
            panel_revision=pc.TWO_COMPONENT_REVISION,
            full=True,
            three_component_revision=rev3,
        )

    # Pilot / non-full subset is still explicitly allowed.
    _, sub_rev, sub_comps = pc.resolve_panel_policy(
        panel_revision=rev3,
        components=("long_context", "agentic"),
        full=False,
        three_component_revision=rev3,
    )
    assert sub_rev == rev3
    # Canonical order is always preserved regardless of input order.
    assert sub_comps == ("agentic", "long_context")

    # Unknown components fail closed.
    with pytest.raises(ValueError, match="unknown panel component"):
        pc.resolve_panel_policy(components=("agentic", "arbitrary"), full=False)


# ---------------------------------------------------------------------------
# Real-template multi-turn renderer tests (Hy4, GLM 5.3, Kimi K3.1)
# ---------------------------------------------------------------------------

FAMILY_TOKENIZERS = [
    pytest.param(
        "hy4",
        "models--tencent--Hy4",
        "c10c0de5b6f850703fb0a90e65d1c32b558b9167",
        dict(reasoning_effort="high", date_string="2026-11-22"),
        False,
        id="hy4",
    ),
    pytest.param(
        "glm5",
        "models--zai-org--GLM-5.3",
        "aca966e4e02791568aa6a4ced368624b3d897f42",
        dict(reasoning_effort="max"),
        False,
        id="glm5",
    ),
    pytest.param(
        "kimi_k31",
        "models--moonshotai--Kimi-K3.1-Pro",
        "6d0a0202956796543364d666ab19c6ad1a494ee1",
        dict(thinking=True, thinking_effort="max"),
        True,
        id="kimi_k31",
    ),
]


def _load_real_tokenizer(repo_dir: str, revision: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    snap = _require_snapshot(repo_dir, revision)
    return snap, AutoTokenizer.from_pretrained(snap, local_files_only=True, trust_remote_code=trust_remote_code)


@pytest.mark.parametrize("family,repo_dir,revision,chat_kwargs,trust", FAMILY_TOKENIZERS)
def test_renderer_on_real_templates_tools_and_reasoning(family, repo_dir, revision, chat_kwargs, trust):
    snap, tok = _load_real_tokenizer(repo_dir, revision, trust)
    max_ctx = pc.max_context_from_config(snap)
    assert max_ctx is not None and max_ctx >= 131_072

    # 1. 16K agentic coding trace with tools, tool_calls, tool results, and reasoning.
    trace_tools = _load_raw_trace("lc-16k-cal-01.json")
    ids1, entry1 = pc.render_long_context_record(
        trace_tools, tok, chat_kwargs=chat_kwargs, max_context=max_ctx
    )
    assert entry1["id"] == trace_tools["id"]
    assert entry1["component"] == "long_context"
    assert entry1["length_class"] == "16k"
    assert entry1["tokens"] == len(ids1)
    assert entry1["has_tools"] is True
    assert entry1["has_tool_calls"] is True
    assert entry1["has_reasoning"] is True
    assert entry1["input_ids_sha256"] == pc.ids_sha256(ids1)
    low16, high16 = pc.LONG_CONTEXT_LENGTH_BOUNDS["16k"]
    assert low16 <= len(ids1) <= high16

    # 2. 16K multi-turn QA trace with reasoning in earlier turns + final turn.
    trace_qa = _load_raw_trace("lc-16k-cal-06.json")
    ids2, entry2 = pc.render_long_context_record(
        trace_qa, tok, chat_kwargs=chat_kwargs, max_context=max_ctx
    )
    assert entry2["has_tools"] is False
    assert entry2["has_tool_calls"] is False
    assert entry2["has_reasoning"] is True
    assert entry2["final_reasoning_preserved"] is True
    if family == "hy4":
        # Hy4's native template drops history reasoning before later user turns.
        assert entry2["history_reasoning_preserved"] is False
    assert low16 <= len(ids2) <= high16

    # 3. 32K agentic coding trace without reasoning (Claude CLI trace).
    trace_32k = _load_raw_trace("lc-32k-cal-03.json")
    ids3, entry3 = pc.render_long_context_record(
        trace_32k, tok, chat_kwargs=chat_kwargs, max_context=max_ctx
    )
    assert entry3["length_class"] == "32k"
    assert entry3["has_tools"] is True
    assert entry3["has_tool_calls"] is True
    assert entry3["has_reasoning"] is False
    assert entry3["history_reasoning_preserved"] is None
    assert entry3["final_reasoning_preserved"] is None
    low32, high32 = pc.LONG_CONTEXT_LENGTH_BOUNDS["32k"]
    assert low32 <= len(ids3) <= high32


def test_renderer_one_pass_and_never_reencodes_text():
    snap, tok = _load_real_tokenizer(
        "models--tencent--Hy4", "c10c0de5b6f850703fb0a90e65d1c32b558b9167", False
    )
    trace = _load_raw_trace("lc-16k-cal-01.json")

    class OnePassSpy:
        def __init__(self, inner):
            self._inner = inner
            self.chat_template = inner.chat_template
            self.template_calls = 0

        def __len__(self):
            return len(self._inner)

        def encode(self, *args, **kwargs):
            raise AssertionError("renderer must never call tokenizer.encode() on rendered text")

        def apply_chat_template(self, messages, **kwargs):
            self.template_calls += 1
            assert kwargs["tokenize"] is True
            assert kwargs["add_generation_prompt"] is False
            assert kwargs["padding"] is False
            assert kwargs["truncation"] is False
            return self._inner.apply_chat_template(messages, **kwargs)

        def decode(self, ids, **kwargs):
            return self._inner.decode(ids, **kwargs)

    spy = OnePassSpy(tok)
    ids, entry = pc.render_long_context_record(
        trace,
        spy,
        chat_kwargs=dict(reasoning_effort="high", date_string="2026-11-22"),
        max_context=pc.max_context_from_config(snap),
    )
    assert spy.template_calls == 1
    assert len(ids) == entry["tokens"]


def test_renderer_fails_closed_on_missing_tool_support_and_length_bounds():
    snap, tok = _load_real_tokenizer(
        "models--tencent--Hy4", "c10c0de5b6f850703fb0a90e65d1c32b558b9167", False
    )
    trace16 = _load_raw_trace("lc-16k-cal-01.json")
    trace32 = _load_raw_trace("lc-32k-cal-01.json")
    chat = dict(reasoning_effort="high", date_string="2026-11-22")

    # 1. Fail closed if template cannot express tools/tool_calls/tool results (never flatten).
    tok_no_tools = deepcopy(tok)
    tok_no_tools.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    )
    with pytest.raises(ValueError, match="does not support tools|cannot express"):
        pc.render_long_context_record(trace16, tok_no_tools, chat_kwargs=chat)

    # 2. Fail closed if template mentions 'tools'/'tool_calls'/'tool' but silently drops tool results.
    tok_silent_drop = deepcopy(tok)
    tok_silent_drop.chat_template = (
        "{# tools tool_calls 'tool' #}"
        "{% for m in messages %}{% if m['role'] != 'tool' %}{{ m['role'] }}: {{ m['content'] or '' }}\n{% endif %}{% endfor %}"
    )
    with pytest.raises(ValueError, match="dropped"):
        pc.render_long_context_record(trace16, tok_silent_drop, chat_kwargs=chat)

    # 3. Fail closed if rendered length doesn't match declared length class.
    bad_class = dict(trace16, length_class="32k")
    with pytest.raises(ValueError, match="outside declared '32k' bounds"):
        pc.render_long_context_record(bad_class, tok, chat_kwargs=chat)
    bad_class_32 = dict(trace32, length_class="16k")
    with pytest.raises(ValueError, match="outside declared '16k' bounds"):
        pc.render_long_context_record(bad_class_32, tok, chat_kwargs=chat)

    # 4. Fail closed if rendered length exceeds model max_context.
    with pytest.raises(ValueError, match="exceeds model max_context"):
        pc.render_long_context_record(trace16, tok, chat_kwargs=chat, max_context=8192)

    # 5. Fail closed if record stores forbidden token IDs or tampered content_sha256.
    with pytest.raises(ValueError, match="must not store token IDs"):
        pc.render_long_context_record(dict(trace16, input_ids=[1, 2, 3]), tok, chat_kwargs=chat)
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        pc.render_long_context_record(dict(trace16, content_sha256="0" * 64), tok, chat_kwargs=chat)


# ---------------------------------------------------------------------------
# End-to-end 3-component panel reader, prepared stream loader & tamper checks
# ---------------------------------------------------------------------------

def _stage_three_component_panel(tmp_path: Path, revision: str) -> Path:
    """Build a valid 3-component panel snapshot in tmp_path from real disk files."""
    base_v2 = _require_snapshot(
        "datasets--wrldsuksgo2mars--agentic-quant-panel-v2",
        pc.TWO_COMPONENT_REVISION,
    )
    t16_cal = _load_raw_trace("lc-16k-cal-01.json")
    t16_hold = dict(_load_raw_trace("lc-16k-cal-06.json"), id="lc-16k-hold-test", split="heldout", group="wiki:heldout-test")
    t32_cal = _load_raw_trace("lc-32k-cal-01.json")
    t32_hold = dict(_load_raw_trace("lc-32k-cal-03.json"), id="lc-32k-hold-test", split="heldout", group="agentic:heldout-test")

    stage = tmp_path / revision
    stage.mkdir(parents=True)
    # Reuse every v2 file via symlinks (no payload duplication).
    for item in base_v2.iterdir():
        if item.name in ("SHA256SUMS", "panel.json"):
            continue
        (stage / item.name).symlink_to(item)

    lc_dir = stage / "long_context"
    lc_dir.mkdir()
    subsets = {"16k": [t16_cal, t16_hold], "32k": [t32_cal, t32_hold]}
    subsets_manifest = {}
    all_records = []
    for cls, rows in subsets.items():
        jl = lc_dir / f"{cls}.jsonl"
        jl.write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )
        subsets_manifest[cls] = {
            "file": f"long_context/{cls}.jsonl",
            "sha256": pc.sha256_file(jl),
            "records": len(rows),
        }
        all_records.extend(rows)

    lc_manifest = {
        "schema": pc.LONG_CONTEXT_SCHEMA,
        "component": "long_context",
        "splits": ["calibration", "heldout"],
        "subsets": subsets_manifest,
        "records": [
            {
                "id": r["id"],
                "split": r["split"],
                "length_class": r["length_class"],
                "group": r["group"],
                "content_sha256": r["content_sha256"],
                "source_identity": r["source_identity"],
            }
            for r in all_records
        ],
    }
    (lc_dir / "manifest.json").write_text(json.dumps(lc_manifest, sort_keys=True), encoding="utf-8")

    panel_doc = json.loads((base_v2 / "panel.json").read_text(encoding="utf-8"))
    panel_doc["components"].append(
        {"name": "long_context", "manifest": "long_context/manifest.json", "splits": ["calibration", "heldout"]}
    )
    (stage / "panel.json").write_text(json.dumps(panel_doc, sort_keys=True), encoding="utf-8")

    # Build SHA256SUMS preserving all base v2 hashes + new long_context & panel.json hashes.
    sums = pc._read_sha256sums(base_v2)
    sums["panel.json"] = pc.sha256_file(stage / "panel.json")
    for rel in ("long_context/manifest.json", "long_context/16k.jsonl", "long_context/32k.jsonl"):
        sums[rel] = pc.sha256_file(stage / rel)
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )
    return stage


def test_read_panel_v2_three_components_and_split_disjointness(tmp_path):
    rev3 = "b" * 40
    stage = _stage_three_component_panel(tmp_path, rev3)
    sums_sha = pc.sha256_file(stage / "SHA256SUMS")

    loaded = pc.read_panel_v2(
        stage,
        panel_revision=rev3,
        panel_sha256sums=sums_sha,
        components=pc.FULL_COMPONENTS,
        full=True,
        three_component_revision=rev3,
        three_component_sha256sums=sums_sha,
    )
    assert loaded["components"] == ("agentic", "exl3_standard", "long_context")
    assert loaded["dataset"] == {"repo_id": pc.PANEL_REPO, "revision": rev3, "sha256sums_sha256": sums_sha}
    cal_ids = [r["id"] for r in loaded["long_context_rows"]["calibration"]]
    hold_ids = [r["id"] for r in loaded["long_context_rows"]["heldout"]]
    assert cal_ids == ["lc-16k-cal-01", "lc-32k-cal-01"]
    assert hold_ids == ["lc-16k-hold-test", "lc-32k-hold-test"]
    assert set(cal_ids).isdisjoint(hold_ids)

    # Injecting a group overlap between calibration and heldout must fail closed.
    jl16 = stage / "long_context/16k.jsonl"
    rows16 = [json.loads(line) for line in jl16.read_text(encoding="utf-8").splitlines()]
    rows16[1]["group"] = rows16[0]["group"]
    jl16.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows16), encoding="utf-8")
    lcm = json.loads((stage / "long_context/manifest.json").read_text(encoding="utf-8"))
    lcm["subsets"]["16k"]["sha256"] = pc.sha256_file(jl16)
    for rec in lcm["records"]:
        if rec["id"] == rows16[1]["id"]:
            rec["group"] = rows16[0]["group"]
    (stage / "long_context/manifest.json").write_text(json.dumps(lcm, sort_keys=True), encoding="utf-8")
    sums = pc._read_sha256sums(stage)
    sums["long_context/16k.jsonl"] = pc.sha256_file(jl16)
    sums["long_context/manifest.json"] = pc.sha256_file(stage / "long_context/manifest.json")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="splits overlap"):
        pc.read_panel_v2(
            stage,
            panel_revision=rev3,
            panel_sha256sums=pc.sha256_file(stage / "SHA256SUMS"),
            components=pc.FULL_COMPONENTS,
            full=True,
            three_component_revision=rev3,
        )


def _write_prepared_corpus(
    out_dir: Path,
    *,
    dataset: dict,
    contract: dict,
    vocab_size: int,
    agentic_cal: list[tuple[str, list[int]]],
    std_cal: list[tuple[str, list[int]]],
    lc_cal: list[tuple[str, list[int], str]] | None,
    agentic_hold: list[tuple[str, list[int]]],
    lc_hold: list[tuple[str, list[int], str]] | None,
    calibration_mode: str = "full",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_order = ["agentic", "exl3_standard"] + (["long_context"] if lc_cal is not None else [])
    components = {}

    def build_split(split_name: str, items: list[tuple[str, str, list[int], str | None]]) -> dict:
        flat, offsets, recs = array("q"), [0], []
        stream = hashlib.sha256()
        for rid, comp, ids, lcls in items:
            flat.extend(ids)
            offsets.append(len(flat))
            entry = dict(id=rid, component=comp, tokens=len(ids), input_ids_sha256=pc.ids_sha256(ids))
            if lcls is not None:
                entry["length_class"] = lcls
            recs.append(entry)
            stream.update(pc.canonical({"id": rid, "input_ids": ids}) + b"\n")
        path = out_dir / f"{split_name}.safetensors"
        save_file(
            dict(
                input_ids=torch.tensor(list(flat), dtype=torch.int64),
                offsets=torch.tensor(offsets, dtype=torch.int64),
            ),
            str(path),
        )
        return dict(
            file=path.name,
            sha256=pc.sha256_file(path),
            records=recs,
            count=len(recs),
            tokens=len(flat),
            shortest=min(r["tokens"] for r in recs),
            longest=max(r["tokens"] for r in recs),
            token_stream_sha256=stream.hexdigest(),
        )

    cal_items = (
        [(rid, "agentic", ids, None) for rid, ids in agentic_cal]
        + [(rid, "exl3_standard", ids, None) for rid, ids in std_cal]
        + ([(rid, "long_context", ids, lcls) for rid, ids, lcls in lc_cal] if lc_cal else [])
    )
    hold_items = (
        [(rid, "agentic", ids, None) for rid, ids in agentic_hold]
        + ([(rid, "long_context", ids, lcls) for rid, ids, lcls in lc_hold] if lc_hold else [])
    )

    components["agentic"] = dict(
        count=len(agentic_cal),
        tokens=sum(len(ids) for _, ids in agentic_cal),
        input_ids_sha256=pc.component_input_ids_sha256(ids for _, ids in agentic_cal),
    )
    components["exl3_standard"] = dict(
        contract=pc.EXL3_STANDARD_CONTRACT,
        count=len(std_cal),
        tokens=sum(len(ids) for _, ids in std_cal),
        input_ids_sha256=pc.component_input_ids_sha256(ids for _, ids in std_cal),
    )
    if lc_cal is not None:
        components["long_context"] = dict(
            schema=pc.LONG_CONTEXT_SCHEMA,
            count=len(lc_cal),
            tokens=sum(len(ids) for _, ids, _ in lc_cal),
            input_ids_sha256=pc.component_input_ids_sha256(ids for _, ids, _ in lc_cal),
        )
        if lc_hold is not None:
            components["long_context"]["heldout"] = dict(
                count=len(lc_hold),
                tokens=sum(len(ids) for _, ids, _ in lc_hold),
                input_ids_sha256=pc.component_input_ids_sha256(ids for _, ids, _ in lc_hold),
            )

    manifest = dict(
        schema=pc.PREPARED_SCHEMA,
        status="passed",
        contract=contract,
        dataset=dataset,
        component_order=comp_order,
        calibration_mode=calibration_mode,
        components=components,
        tokenizer=dict(vocab_size=vocab_size),
        splits=dict(
            calibration=build_split("calibration", cal_items),
            heldout=build_split("heldout", hold_items),
        ),
    )
    (out_dir / "manifest.json").write_bytes(pc.canonical(manifest))
    return manifest


def test_prepared_loader_order_splits_hashes_and_fail_closed_full_check(tmp_path):
    snap, tok = _load_real_tokenizer(
        "models--tencent--Hy4", "c10c0de5b6f850703fb0a90e65d1c32b558b9167", False
    )
    chat = dict(reasoning_effort="high", date_string="2026-11-22")
    max_ctx = pc.max_context_from_config(snap)

    # Render real long_context traces from disk with Hy4's tokenizer.
    t16_cal = _load_raw_trace("lc-16k-cal-01.json")
    t16_hold = dict(_load_raw_trace("lc-16k-cal-06.json"), id="lc-16k-hold-01", split="heldout")
    lc_cal_ids, _ = pc.render_long_context_record(t16_cal, tok, chat_kwargs=chat, max_context=max_ctx)
    lc_hold_ids, _ = pc.render_long_context_record(t16_hold, tok, chat_kwargs=chat, max_context=max_ctx)

    # Read 2 real agentic prompts from the cached v2 panel on disk.
    v2 = _require_snapshot("datasets--wrldsuksgo2mars--agentic-quant-panel-v2", pc.TWO_COMPONENT_REVISION)
    ag_cal_row = json.loads((v2 / "calibration.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ag_hold_row = json.loads((v2 / "heldout.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ag_cal_ids = tok.apply_chat_template(
        [dict(role="user", content=ag_cal_row["prompt"])],
        tokenize=True, add_generation_prompt=True, return_dict=False, **chat,
    )
    ag_hold_ids = tok.apply_chat_template(
        [dict(role="user", content=ag_hold_row["prompt"])],
        tokenize=True, add_generation_prompt=True, return_dict=False, **chat,
    )
    # Real raw text from exl3_standard/segments.jsonl on disk.
    std_seg = json.loads((v2 / "exl3_standard/segments.jsonl").read_text(encoding="utf-8").splitlines()[0])
    std_ids = tok.encode(std_seg["text"], add_special_tokens=False)[:2048]

    rev3 = "c" * 40
    ds3 = {"repo_id": pc.PANEL_REPO, "revision": rev3}
    contract3 = {"renderer": "native-chat-template", "packing": False}
    dir3 = tmp_path / "prepared-3comp"
    _write_prepared_corpus(
        dir3,
        dataset=ds3,
        contract=contract3,
        vocab_size=len(tok),
        agentic_cal=[(ag_cal_row["id"], ag_cal_ids)],
        std_cal=[("exl3_standard/000", std_ids)],
        lc_cal=[(t16_cal["id"], lc_cal_ids, "16k")],
        agentic_hold=[(ag_hold_row["id"], ag_hold_ids)],
        lc_hold=[(t16_hold["id"], lc_hold_ids, "16k")],
    )

    # Order: agentic -> exl3_standard -> long_context; heldout never enters calibration.
    cal_recs = pc.load_token_records(
        dir3, "calibration", contract=contract3, dataset=ds3, full=True, three_component_revision=rev3
    )
    assert [r["component"] for r in cal_recs] == ["agentic", "exl3_standard", "long_context"]
    assert [r["id"] for r in cal_recs] == [ag_cal_row["id"], "exl3_standard/000", t16_cal["id"]]
    assert t16_hold["id"] not in {r["id"] for r in cal_recs}

    hold_recs = pc.load_token_records(
        dir3, "heldout", contract=contract3, dataset=ds3, full=True, three_component_revision=rev3
    )
    assert [r["component"] for r in hold_recs] == ["agentic", "long_context"]
    assert [r["id"] for r in hold_recs] == [ag_hold_row["id"], t16_hold["id"]]

    # Unpadded CPU batches (batch size 1, all-ones mask).
    batches = list(pc.load_token_batches(
        dir3, "calibration", contract=contract3, dataset=ds3, full=True, three_component_revision=rev3
    ))
    assert [b["component"] for b in batches] == ["agentic", "exl3_standard", "long_context"]
    for rec, batch in zip(cal_recs, batches, strict=True):
        assert tuple(batch["input_ids"].shape) == (1, len(rec["input_ids"]))
        assert batch["input_ids"].dtype == torch.int64
        assert torch.equal(batch["attention_mask"], torch.ones_like(batch["input_ids"]))

    # load_component_records returns QMLA/MC-compatible per-component provenance.
    comp_recs, prov = pc.load_component_records(
        dir3, "calibration", contract=contract3, dataset=ds3, full=True, three_component_revision=rev3
    )
    assert len(comp_recs) == 3
    assert prov["component_order"] == ["agentic", "exl3_standard", "long_context"]
    assert prov["components"]["long_context"]["tokens"] == len(lc_cal_ids)
    assert prov["components"]["long_context"]["input_ids_sha256"] == pc.component_input_ids_sha256([lc_cal_ids])

    # Tampering a component's input_ids_sha256 fails closed.
    bad_manifest = json.loads((dir3 / "manifest.json").read_text(encoding="utf-8"))
    bad_manifest["components"]["long_context"]["input_ids_sha256"] = "0" * 64
    (dir3 / "manifest.json").write_bytes(pc.canonical(bad_manifest))
    with pytest.raises(ValueError, match="input_ids_sha256 differs"):
        pc.load_token_records(dir3, "calibration", contract=contract3, dataset=ds3)

    # A 2-component prepared corpus loads cleanly BEFORE rev3 is published,
    # and FAILS CLOSED on full=True AFTER rev3 is published!
    ds2 = {"repo_id": pc.PANEL_REPO, "revision": pc.TWO_COMPONENT_REVISION}
    dir2 = tmp_path / "prepared-2comp"
    _write_prepared_corpus(
        dir2,
        dataset=ds2,
        contract=contract3,
        vocab_size=len(tok),
        agentic_cal=[(ag_cal_row["id"], ag_cal_ids)],
        std_cal=[("exl3_standard/000", std_ids)],
        lc_cal=None,
        agentic_hold=[(ag_hold_row["id"], ag_hold_ids)],
        lc_hold=None,
    )
    recs2 = pc.load_token_records(
        dir2, "calibration", contract=contract3, dataset=ds2, full=True, three_component_revision=None
    )
    assert [r["component"] for r in recs2] == ["agentic", "exl3_standard"]

    with pytest.raises(ValueError, match="full calibration requires all three"):
        pc.load_token_records(
            dir2, "calibration", contract=contract3, dataset=ds2, full=True, three_component_revision=rev3
        )
    # Explicit non-full reload still works for reproducing historical 2-component artifacts.
    recs2_sub = pc.load_token_records(
        dir2, "calibration", contract=contract3, dataset=ds2, full=False, three_component_revision=rev3
    )
    assert len(recs2_sub) == 2


def test_hy4_flash_prepare_corpus_uses_single_source_of_truth():
    assert hy4_flash_corpus.PANEL_REVISION == pc.DEFAULT_PANEL_REVISION
    assert hy4_flash_corpus.PANEL_SUMS == pc.DEFAULT_PANEL_SHA256SUMS_SHA256
    assert hy4_flash_corpus.DATASET == pc.DEFAULT_DATASET
    base_v2 = _require_snapshot(
        "datasets--wrldsuksgo2mars--agentic-quant-panel-v2",
        pc.TWO_COMPONENT_REVISION,
    )
    rows, provenance = hy4_flash_corpus.read_v2(base_v2)
    assert set(rows) == {"calibration", "screening", "heldout"}
    assert len(rows["calibration"]) == provenance["splits"]["calibration"]["summary"]["records"]


def test_tokenize_long_context_rows_preserves_order_and_class_totals():
    snap, tok = _load_real_tokenizer(
        "models--tencent--Hy4", "c10c0de5b6f850703fb0a90e65d1c32b558b9167", False
    )
    t16 = _load_raw_trace("lc-16k-cal-01.json")
    t32 = _load_raw_trace("lc-32k-cal-01.json")
    flat, offsets, report = pc.tokenize_long_context_rows(
        [t16, t32],
        tok,
        chat_kwargs=dict(reasoning_effort="high", date_string="2026-11-22"),
        max_context=pc.max_context_from_config(snap),
    )
    assert report["count"] == 2
    assert offsets[0] == 0 and offsets[-1] == len(flat) == report["tokens"]
    assert [r["id"] for r in report["records"]] == [t16["id"], t32["id"]]
    assert report["length_classes"]["16k"]["count"] == 1
    assert report["length_classes"]["32k"]["count"] == 1
    assert len(report["input_ids_sha256"]) == 64
    assert len(report["token_stream_sha256"]) == 64


def test_hy4_model_cards_record_three_components(tmp_path):
    identity3 = dict(
        codebook="mul1",
        codebook_multiplier=0x83DCD12D,
        seed=787,
        sigma=0.025,
        code={"fork": "a" * 40},
        dataset={"repo_id": pc.PANEL_REPO, "revision": "d" * 40},
        components=dict(
            agentic=dict(count=1426, tokens=1_080_000, input_ids_sha256="1" * 64),
            exl3_standard=dict(count=250, tokens=512_000, input_ids_sha256="2" * 64),
            long_context=dict(count=12, tokens=256_000, input_ids_sha256="3" * 64),
        ),
    )
    view_bytes = {"original": 10 << 30, "k8-compact": 9 << 30, "k6-compact": 8 << 30}
    engram = {"proof": {"layer1": {"mismatch_count": 0, "rows": 100}}}

    flash_card = hy4_flash_export.model_card(
        tmp_path / "flash.md",
        engram,
        k6_drafter_bits=6,
        selection={"lm_head": {}},
        identity=identity3,
        view_bytes=view_bytes,
    ).read_text(encoding="utf-8")
    assert "1,848,000 tokens" in flash_card
    assert "12 long-context traces (256,000 tokens)" in flash_card
    assert "3" * 64 in flash_card
    assert "exercising DSA selection beyond the top-2,048 cutoff" in flash_card

    full_card = hy4_export.model_card(
        tmp_path / "full.md",
        None,
        k6_drafter_bits=6,
        selection={"lm_head": {}},
        identity=identity3,
        view_bytes=view_bytes,
    ).read_text(encoding="utf-8")
    assert "256,000 tokens (12 traces)" in full_card
    assert "exercised DSA selection beyond 2,048 tokens during calibration" in full_card
