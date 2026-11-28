"""Tests for the shared three-component calibration corpus renderer and loader.

All multi-turn samples and tokenizers are read directly from disk (never inline
fixture transcripts):
- Raw traces: ``~/.cache/cuteafd/builds/long-context-bank/raw_traces/*.json``
- Real checkpoints: local HF hub cache (Hy4, GLM 5.3, Kimi K3.1)
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
from hy4_flash import prepare_corpus as hy4_flash_corpus

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


def _load_real_tokenizer(repo_dir: str, revision: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    snap = _require_snapshot(repo_dir, revision)
    return snap, AutoTokenizer.from_pretrained(snap, local_files_only=True, trust_remote_code=trust_remote_code)


def test_default_policy_two_versus_three_switch(monkeypatch):
    assert pc.THREE_COMPONENT_REVISION is None
    assert pc.resolve_panel_policy() == (pc.PANEL_REPO, pc.TWO_COMPONENT_REVISION, ("agentic", "exl3_standard"))
    assert pc.default_panel_revision() == pc.TWO_COMPONENT_REVISION
    assert pc.default_components() == ("agentic", "exl3_standard")

    with pytest.raises(ValueError, match="not present in two-component"):
        pc.resolve_panel_policy(panel_revision=pc.TWO_COMPONENT_REVISION, components=pc.COMPONENT_ORDER)

    rev3 = "a" * 40
    monkeypatch.setattr(pc, "THREE_COMPONENT_REVISION", rev3)
    assert pc.default_panel_revision() == rev3
    assert pc.default_components() == pc.COMPONENT_ORDER
    assert pc.resolve_panel_policy() == (pc.PANEL_REPO, rev3, pc.COMPONENT_ORDER)

    # Full run with only 2 components fails closed once rev3 is published.
    with pytest.raises(ValueError, match="full calibration requires all three"):
        pc.resolve_panel_policy(panel_revision=rev3, components=("agentic", "exl3_standard"), full=True)
    with pytest.raises(ValueError, match="full calibration requires all three"):
        pc.resolve_panel_policy(panel_revision=pc.TWO_COMPONENT_REVISION, full=True)

    # Pilot subset still allowed and canonicalizes component order.
    assert pc.resolve_panel_policy(panel_revision=rev3, components=("long_context", "agentic"), full=False) == (
        pc.PANEL_REPO, rev3, ("agentic", "long_context")
    )
    with pytest.raises(ValueError, match="unknown panel component"):
        pc.resolve_panel_policy(components=("agentic", "unknown"), full=False)


FAMILY_TOKENIZERS = [
    pytest.param("hy4", "models--tencent--Hy4", "c10c0de5b6f850703fb0a90e65d1c32b558b9167",
                 dict(reasoning_effort="high", date_string="2026-11-22"), False, id="hy4"),
    pytest.param("glm5", "models--zai-org--GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42",
                 dict(reasoning_effort="max"), False, id="glm5"),
    pytest.param("kimi_k31", "models--moonshotai--Kimi-K3.1-Pro", "6d0a0202956796543364d666ab19c6ad1a494ee1",
                 dict(thinking=True, thinking_effort="max"), True, id="kimi_k31"),
]


@pytest.mark.parametrize("family,repo_dir,revision,chat_kwargs,trust", FAMILY_TOKENIZERS)
def test_renderer_on_real_templates_tools_and_reasoning(family, repo_dir, revision, chat_kwargs, trust):
    snap, tok = _load_real_tokenizer(repo_dir, revision, trust)
    max_ctx = pc.max_context_from_config(snap)
    assert max_ctx is not None and max_ctx >= 131_072

    t_tools = _load_raw_trace("lc-16k-cal-01.json")
    ids1, e1 = pc.render_long_context_record(t_tools, tok, chat_kwargs=chat_kwargs, max_context=max_ctx)
    assert e1["tokens"] == len(ids1) and e1["has_tools"] and e1["has_tool_calls"] and e1["has_reasoning"]
    assert pc.LONG_CONTEXT_LENGTH_BOUNDS["16k"][0] <= len(ids1) <= pc.LONG_CONTEXT_LENGTH_BOUNDS["16k"][1]

    t_qa = _load_raw_trace("lc-16k-cal-06.json")
    ids2, e2 = pc.render_long_context_record(t_qa, tok, chat_kwargs=chat_kwargs, max_context=max_ctx)
    assert not e2["has_tools"] and e2["has_reasoning"] and e2["final_reasoning_preserved"] is True
    if family == "hy4":
        assert e2["history_reasoning_preserved"] is False

    t_32k = _load_raw_trace("lc-32k-cal-03.json")
    ids3, e3 = pc.render_long_context_record(t_32k, tok, chat_kwargs=chat_kwargs, max_context=max_ctx)
    assert e3["length_class"] == "32k" and e3["has_tool_calls"] and not e3["has_reasoning"]
    assert pc.LONG_CONTEXT_LENGTH_BOUNDS["32k"][0] <= len(ids3) <= pc.LONG_CONTEXT_LENGTH_BOUNDS["32k"][1]


def test_renderer_one_pass_and_fail_closed():
    snap, tok = _load_real_tokenizer("models--tencent--Hy4", "c10c0de5b6f850703fb0a90e65d1c32b558b9167", False)
    chat = dict(reasoning_effort="high", date_string="2026-11-22")
    t16, t32 = _load_raw_trace("lc-16k-cal-01.json"), _load_raw_trace("lc-32k-cal-01.json")

    class OnePassSpy:
        def __init__(self, inner):
            self._inner, self.chat_template, self.calls = inner, inner.chat_template, 0
        def __len__(self): return len(self._inner)
        def encode(self, *a, **kw): raise AssertionError("must never re-encode rendered text")
        def apply_chat_template(self, messages, **kw):
            self.calls += 1
            assert kw["tokenize"] is True and kw["add_generation_prompt"] is False
            return self._inner.apply_chat_template(messages, **kw)
        def decode(self, ids, **kw): return self._inner.decode(ids, **kw)

    spy = OnePassSpy(tok)
    ids, entry = pc.render_long_context_record(t16, spy, chat_kwargs=chat, max_context=pc.max_context_from_config(snap))
    assert spy.calls == 1 and len(ids) == entry["tokens"]

    # Missing tool support fails closed (never flattens).
    no_tools = deepcopy(tok)
    no_tools.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    with pytest.raises(ValueError, match="does not support tools|cannot express"):
        pc.render_long_context_record(t16, no_tools, chat_kwargs=chat)

    # Silent tool-result drop fails closed.
    silent_drop = deepcopy(tok)
    silent_drop.chat_template = "{# tools tool_calls 'tool' #}{% for m in messages %}{% if m['role'] != 'tool' %}{{ m['content'] or '' }}{% endif %}{% endfor %}"
    with pytest.raises(ValueError, match="dropped"):
        pc.render_long_context_record(t16, silent_drop, chat_kwargs=chat)

    # Length-class mismatch, max_context, forbidden token IDs, and content hash mismatch fail closed.
    with pytest.raises(ValueError, match="outside declared '32k'"):
        pc.render_long_context_record(dict(t16, length_class="32k"), tok, chat_kwargs=chat)
    with pytest.raises(ValueError, match="outside declared '16k'"):
        pc.render_long_context_record(dict(t32, length_class="16k"), tok, chat_kwargs=chat)
    with pytest.raises(ValueError, match="exceeds model max_context"):
        pc.render_long_context_record(t16, tok, chat_kwargs=chat, max_context=8192)
    with pytest.raises(ValueError, match="must not store token IDs"):
        pc.render_long_context_record(dict(t16, input_ids=[1, 2]), tok, chat_kwargs=chat)
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        pc.render_long_context_record(dict(t16, content_sha256="0" * 64), tok, chat_kwargs=chat)


def _stage_three_component_panel(tmp_path: Path, revision: str) -> Path:
    base_v2 = _require_snapshot("datasets--wrldsuksgo2mars--agentic-quant-panel-v2", pc.TWO_COMPONENT_REVISION)
    t16_cal = _load_raw_trace("lc-16k-cal-01.json")
    t16_hold = dict(_load_raw_trace("lc-16k-cal-06.json"), id="lc-16k-hold-test", split="heldout", group="wiki:heldout-test")
    t32_cal = _load_raw_trace("lc-32k-cal-01.json")
    t32_hold = dict(_load_raw_trace("lc-32k-cal-03.json"), id="lc-32k-hold-test", split="heldout", group="agentic:heldout-test")

    stage = tmp_path / revision
    stage.mkdir(parents=True)
    for item in base_v2.iterdir():
        if item.name not in ("SHA256SUMS", "panel.json"):
            (stage / item.name).symlink_to(item)

    lc_dir = stage / "long_context"
    lc_dir.mkdir()
    subsets_meta, all_rows = {}, []
    for cls, rows in (("16k", [t16_cal, t16_hold]), ("32k", [t32_cal, t32_hold])):
        jl = lc_dir / f"{cls}.jsonl"
        jl.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        subsets_meta[cls] = {"file": f"long_context/{cls}.jsonl", "sha256": pc.sha256_file(jl), "records": len(rows)}
        all_rows.extend(rows)

    lc_manifest = dict(
        schema=pc.LONG_CONTEXT_SCHEMA, component="long_context", splits=["calibration", "heldout"],
        subsets=subsets_meta,
        records=[{k: r[k] for k in ("id", "split", "length_class", "group", "content_sha256", "source_identity")} for r in all_rows],
    )
    (lc_dir / "manifest.json").write_text(json.dumps(lc_manifest, sort_keys=True), encoding="utf-8")
    panel_doc = json.loads((base_v2 / "panel.json").read_text(encoding="utf-8"))
    panel_doc["components"].append({"name": "long_context", "manifest": "long_context/manifest.json", "splits": ["calibration", "heldout"]})
    (stage / "panel.json").write_text(json.dumps(panel_doc, sort_keys=True), encoding="utf-8")

    sums = pc._read_sha256sums(base_v2)
    for rel in ("panel.json", "long_context/manifest.json", "long_context/16k.jsonl", "long_context/32k.jsonl"):
        sums[rel] = pc.sha256_file(stage / rel)
    (stage / "SHA256SUMS").write_text("".join(f"{d}  {n}\n" for n, d in sorted(sums.items())), encoding="utf-8")
    return stage


def test_read_panel_v2_three_components_and_split_disjointness(tmp_path, monkeypatch):
    rev3 = "b" * 40
    stage = _stage_three_component_panel(tmp_path, rev3)
    sums_sha = pc.sha256_file(stage / "SHA256SUMS")
    monkeypatch.setattr(pc, "THREE_COMPONENT_REVISION", rev3)
    monkeypatch.setattr(pc, "THREE_COMPONENT_SHA256SUMS_SHA256", sums_sha)

    loaded = pc.read_panel_v2(stage)
    assert loaded["components"] == pc.COMPONENT_ORDER
    assert [r["id"] for r in loaded["long_context_rows"]["calibration"]] == ["lc-16k-cal-01", "lc-32k-cal-01"]
    assert [r["id"] for r in loaded["long_context_rows"]["heldout"]] == ["lc-16k-hold-test", "lc-32k-hold-test"]

    # Overlapping group between calibration and heldout fails closed.
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
    (stage / "SHA256SUMS").write_text("".join(f"{d}  {n}\n" for n, d in sorted(sums.items())), encoding="utf-8")
    monkeypatch.setattr(pc, "THREE_COMPONENT_SHA256SUMS_SHA256", pc.sha256_file(stage / "SHA256SUMS"))
    with pytest.raises(ValueError, match="splits overlap"):
        pc.read_panel_v2(stage)


def _write_prepared(out_dir: Path, *, dataset, contract, vocab_size, cal_parts, hold_parts):
    out_dir.mkdir(parents=True, exist_ok=True)
    components = {}

    def build_split(name, items):
        flat, offsets, recs, stream = array("q"), [0], [], hashlib.sha256()
        for rid, comp, ids in items:
            flat.extend(ids)
            offsets.append(len(flat))
            recs.append(dict(id=rid, component=comp, tokens=len(ids), input_ids_sha256=pc.ids_sha256(ids)))
            stream.update(pc.canonical({"id": rid, "input_ids": ids}) + b"\n")
        path = out_dir / f"{name}.safetensors"
        save_file(dict(input_ids=torch.tensor(list(flat), dtype=torch.int64),
                       offsets=torch.tensor(offsets, dtype=torch.int64)), str(path))
        return dict(file=path.name, sha256=pc.sha256_file(path), records=recs, count=len(recs),
                    tokens=len(flat), shortest=min(r["tokens"] for r in recs),
                    longest=max(r["tokens"] for r in recs), token_stream_sha256=stream.hexdigest())

    cal_items = []
    for comp, rows in cal_parts:
        cal_items.extend((rid, comp, ids) for rid, ids in rows)
        components[comp] = dict(count=len(rows), tokens=sum(len(i) for _, i in rows),
                                input_ids_sha256=pc.component_input_ids_sha256(i for _, i in rows))
    hold_items = []
    for comp, rows in hold_parts:
        hold_items.extend((rid, comp, ids) for rid, ids in rows)
        if comp == "long_context":
            components["long_context"]["heldout"] = dict(
                count=len(rows), tokens=sum(len(i) for _, i in rows),
                input_ids_sha256=pc.component_input_ids_sha256(i for _, i in rows),
            )
    manifest = dict(
        schema=pc.PREPARED_SCHEMA, status="passed", contract=contract, dataset=dataset,
        component_order=[c for c, _ in cal_parts], calibration_mode="full", components=components,
        tokenizer=dict(vocab_size=vocab_size),
        splits=dict(calibration=build_split("calibration", cal_items), heldout=build_split("heldout", hold_items)),
    )
    (out_dir / "manifest.json").write_bytes(pc.canonical(manifest))


def test_prepared_loader_order_splits_hashes_and_full_check(tmp_path, monkeypatch):
    snap, tok = _load_real_tokenizer("models--tencent--Hy4", "c10c0de5b6f850703fb0a90e65d1c32b558b9167", False)
    chat = dict(reasoning_effort="high", date_string="2026-11-22")
    max_ctx = pc.max_context_from_config(snap)

    t16_cal = _load_raw_trace("lc-16k-cal-01.json")
    t16_hold = dict(_load_raw_trace("lc-16k-cal-06.json"), id="lc-16k-hold-01", split="heldout")
    lc_cal_ids, _ = pc.render_long_context_record(t16_cal, tok, chat_kwargs=chat, max_context=max_ctx)
    lc_hold_ids, _ = pc.render_long_context_record(t16_hold, tok, chat_kwargs=chat, max_context=max_ctx)

    v2 = _require_snapshot("datasets--wrldsuksgo2mars--agentic-quant-panel-v2", pc.TWO_COMPONENT_REVISION)
    ag_cal = json.loads((v2 / "calibration.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ag_hold = json.loads((v2 / "heldout.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ag_cal_ids = tok.apply_chat_template([dict(role="user", content=ag_cal["prompt"])], tokenize=True, add_generation_prompt=True, return_dict=False, **chat)
    ag_hold_ids = tok.apply_chat_template([dict(role="user", content=ag_hold["prompt"])], tokenize=True, add_generation_prompt=True, return_dict=False, **chat)
    std_ids = tok.encode(json.loads((v2 / "exl3_standard/segments.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"], add_special_tokens=False)[:2048]

    rev3 = "c" * 40
    ds3, contract3 = {"repo_id": pc.PANEL_REPO, "revision": rev3}, {"renderer": "native-chat-template", "packing": False}
    dir3 = tmp_path / "prepared-3"
    _write_prepared(dir3, dataset=ds3, contract=contract3, vocab_size=len(tok),
                    cal_parts=[("agentic", [(ag_cal["id"], ag_cal_ids)]),
                               ("exl3_standard", [("exl3_standard/000", std_ids)]),
                               ("long_context", [(t16_cal["id"], lc_cal_ids)])],
                    hold_parts=[("agentic", [(ag_hold["id"], ag_hold_ids)]),
                                ("long_context", [(t16_hold["id"], lc_hold_ids)])])

    monkeypatch.setattr(pc, "THREE_COMPONENT_REVISION", rev3)
    cal_recs = pc.load_token_records(dir3, "calibration", contract=contract3, dataset=ds3, full=True)
    assert [r["component"] for r in cal_recs] == ["agentic", "exl3_standard", "long_context"]
    assert t16_hold["id"] not in {r["id"] for r in cal_recs}
    assert [r["id"] for r in pc.load_token_records(dir3, "heldout", contract=contract3, dataset=ds3)] == [ag_hold["id"], t16_hold["id"]]

    batches = list(pc.load_token_batches(dir3, "calibration", contract=contract3, dataset=ds3, full=True))
    assert [tuple(b["input_ids"].shape) for b in batches] == [(1, len(r["input_ids"])) for r in cal_recs]
    _, prov = pc.load_component_records(dir3, "calibration", contract=contract3, dataset=ds3, full=True)
    assert prov["component_order"] == ["agentic", "exl3_standard", "long_context"]
    assert prov["components"]["long_context"]["input_ids_sha256"] == pc.component_input_ids_sha256([lc_cal_ids])

    # Tampered component hash fails closed.
    bad = json.loads((dir3 / "manifest.json").read_text(encoding="utf-8"))
    bad["components"]["long_context"]["input_ids_sha256"] = "0" * 64
    (dir3 / "manifest.json").write_bytes(pc.canonical(bad))
    with pytest.raises(ValueError, match="input_ids_sha256 differs"):
        pc.load_token_records(dir3, "calibration", contract=contract3, dataset=ds3)

    # 2-component prepared corpus loads pre-rev3, fails closed on full=True post-rev3.
    ds2 = {"repo_id": pc.PANEL_REPO, "revision": pc.TWO_COMPONENT_REVISION}
    dir2 = tmp_path / "prepared-2"
    _write_prepared(dir2, dataset=ds2, contract=contract3, vocab_size=len(tok),
                    cal_parts=[("agentic", [(ag_cal["id"], ag_cal_ids)]), ("exl3_standard", [("exl3_standard/000", std_ids)])],
                    hold_parts=[("agentic", [(ag_hold["id"], ag_hold_ids)])])
    with pytest.raises(ValueError, match="full calibration requires all three"):
        pc.load_token_records(dir2, "calibration", contract=contract3, dataset=ds2, full=True)
    assert len(pc.load_token_records(dir2, "calibration", contract=contract3, dataset=ds2, full=False)) == 2
    monkeypatch.setattr(pc, "THREE_COMPONENT_REVISION", None)
    assert len(pc.load_token_records(dir2, "calibration", contract=contract3, dataset=ds2, full=True)) == 2


def test_panel_corpus_two_component_read_matches_frozen_hy4_flash_loader():
    base_v2 = _require_snapshot("datasets--wrldsuksgo2mars--agentic-quant-panel-v2", pc.TWO_COMPONENT_REVISION)
    loaded = pc.read_panel_v2(base_v2)
    hy4_rows, hy4_prov = hy4_flash_corpus.read_v2(base_v2)
    assert loaded["dataset"] == hy4_flash_corpus.DATASET
    assert loaded["agentic_rows"] == hy4_rows
    assert loaded["agentic_provenance"] == hy4_prov
    assert loaded["components"] == ("agentic", "exl3_standard")
