"""Admission and recovery checks for the shortened Step-5 format pilot."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quantization"))
from step5.core_k6 import pilot
from step5 import prepare_corpus as corpus


def record(name, length):
    return dict(id=name, input_ids=list(range(length)))


def complete_report():
    rows = []
    for layer in (*pilot.LAYERS, pilot.LATE_LAYER):
        for name in pilot.modules(layer):
            for arm in pilot.ARMS:
                rows.append(dict(tensor=name, arm=arm, nmse=0 if arm == "bf16" else 1e-4,
                    late_expert_format_witness=layer == 88, causal_input_layer=7 if layer == 88 else layer,
                    source_dtype="torch.float32" if name.endswith("sparse_indexer_w") else "torch.bfloat16",
                    native_effective_dtype="torch.bfloat16", reference_basis="native-effective-projection-weight"))
    return dict(status="passed", native_checks=dict.fromkeys(pilot.CHECKS, True), results=rows,
        head_measured=False, full_model_quality_measured=False,
        sparse_scope=dict(panel_modified=False, native_csa_indexer_measured=True,
            global_topk_pruning_measured=False, maximum_pilot_prompt_tokens=1050,
            pruning_requires_prompt_tokens=4105))


def test_whole_prompts_and_window_witness_are_not_truncated():
    cal = [record("a", 400), record("b", 300), record("window", 603)]
    held = [record("held", 305)]
    actual, validation, witness = pilot.select_records(cal, held, 401, 256)
    assert actual == cal and validation == held and witness == cal[2]
    assert len(actual[-1]["input_ids"]) == 603


def test_prefers_real_pruning_witness_only_if_the_panel_contains_one():
    cal = [record("window", 603), record("pruning", 4105)]
    actual, _, witness = pilot.select_records(cal, [record("held", 300)], 300, 256)
    assert actual == cal and witness == cal[1]


def test_missing_window_or_overlapping_heldout_fails():
    with pytest.raises(ValueError, match="window-crossing"):
        pilot.select_records([record("a", 500)], [record("held", 300)], 300, 256)
    with pytest.raises(ValueError, match="overlap"):
        pilot.select_records([record("a", 603)], [record("a", 603)], 300, 256)


def test_metric_inventory_and_absolute_bound_admit_fp8_advantage():
    report = complete_report()
    for row in report["results"]:
        if row["arm"] == "fp8-block128":
            row["nmse"] = 1e-9
    assert pilot.validate_qualification(report) is report
    assert len(report["results"]) == 105
    report["results"][0]["nmse"] = .00101
    with pytest.raises(ValueError, match="absolute format bound"):
        pilot.validate_qualification(report)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "negative", "nonfinite", "native", "control", "late", "quality"])
def test_qualification_cannot_promote_incomplete_or_mislabelled_evidence(failure):
    report = complete_report()
    if failure == "missing":
        report["results"].pop()
    elif failure == "duplicate":
        report["results"][1] = copy.copy(report["results"][0])
    elif failure == "negative":
        report["results"][0]["nmse"] = -1e-5
    elif failure == "nonfinite":
        report["results"][0]["nmse"] = float("nan")
    elif failure == "native":
        report["native_checks"][pilot.CHECKS[-1]] = False
    elif failure == "control":
        report["results"][4]["nmse"] = 1e-7
    elif failure == "late":
        report["results"][-1]["causal_input_layer"] = 88
    elif failure == "quality":
        report["full_model_quality_measured"] = True
    with pytest.raises(ValueError):
        pilot.validate_qualification(report)


def test_tp2_padding_and_replicated_indexer_accounting():
    # The 64-head output gate pads to128 before K6 storage; each TP rank's32
    # outputs also pads to128. Splitting scale-inclusive bytes inhalf iswrong.
    rank = pilot.tp2_projection_bytes("model.layers.1.self_attn.g_proj", [64, 4096], "mcg-k6")
    assert rank == [4096 * 128 * 6 // 8 + 2 * (4096 + 128) + 4] * 2
    indexer = pilot.tp2_projection_bytes("model.layers.7.self_attn.sparse_indexer_w", [16, 4096], "bf16")
    assert indexer == [16 * 4096 * 2] * 2
    down = pilot.tp2_projection_bytes("model.layers.1.mlp.down_proj", [4096, 13824], "mcg-k4")
    assert down == [4096 * 6912 // 2 + 2 * (4096 + 6912) + 4] * 2


def test_native_thinking_renderer_contract():
    class Native:
        bos_token = "<BOS>"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["reasoning_effort"] == "max"
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["padding"] is False and kwargs["truncation"] is False
            text = self.bos_token + "<|im_start|>system\nReasoning: max\n\n<|im_end|>\n" + messages[0]["content"] + "<|im_start|>assistant\n<think>"
            return self.encode(text, add_special_tokens=False) if kwargs["tokenize"] else text
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode())
    assert len(corpus.tokenizer_probes(Native())) == 5
    class Broken(Native):
        def encode(self, text, *, add_special_tokens):
            return [1]
        def apply_chat_template(self, messages, **kwargs):
            return [2] if kwargs["tokenize"] else super().apply_chat_template(messages, **kwargs)
    with pytest.raises(ValueError, match="template tokenization"):
        corpus.tokenizer_probes(Broken())


def test_sparse_probe_observes_pruning_and_restores_native_function():
    import torch
    from types import SimpleNamespace
    packed = torch.arange(4 * 513, dtype=torch.int32).reshape(4, 513)
    counts = torch.tensor([1, 513, 513, 513], dtype=torch.int32)
    def original(*args, **kwargs):
        return packed, counts
    source = SimpleNamespace(kernel=SimpleNamespace(prefill_sparse_meta=original))
    positions = torch.tensor([0, 4096, 4104, 4200], dtype=torch.int32)
    with pilot.SparseProbe(source, 4105) as probe:
        actual = source.kernel.prefill_sparse_meta(None, positions, None, topk=512, region_size=8)
        assert actual[0] is packed and actual[1] is counts
    assert source.kernel.prefill_sparse_meta is original
    evidence = probe.summary()
    assert evidence["query_rows"] == 3 and evidence["pruned_query_rows"] == 1
    assert evidence["maximum_available_regions"] == 514
    assert evidence["maximum_selected_regions"] == 513
    assert evidence["sparse_pruning_exercised"] is True
    assert len(evidence["metadata_sha256"]) == 64


def test_native_tokenizer_regex_is_preserved_and_heuristic_changes_fail(tmp_path):
    import json
    from types import SimpleNamespace
    original = dict(model={"vocab": {"hello":0}}, normalizer=None,
        pre_tokenizer={"pattern":"native-step-regex"}, decoder={"type":"ByteLevel"})
    (tmp_path / "tokenizer.json").write_text(json.dumps(original))
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:json.dumps(original)))
    assert corpus.verify_native_tokenizer(tokenizer, tmp_path)["heuristic_regex_patch"] is False
    altered = copy.deepcopy(original)
    altered["pre_tokenizer"]["pattern"] = "different-model-regex"
    tokenizer.backend_tokenizer.to_str = lambda: json.dumps(altered)
    with pytest.raises(ValueError, match="native serialized"):
        corpus.verify_native_tokenizer(tokenizer, tmp_path)


def test_three_stage_alignment_uses_successive_shift_without_repeating_target_rows():
    import torch
    spec = importlib.util.spec_from_file_location("step5_test_mtp_oracle", ROOT / "python/reference/families/step5/mtp.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    tokens, positions = torch.arange(10), torch.arange(500, 510)
    hidden = torch.arange(10).float()[:, None].repeat(1, 4)
    for stage in range(3):
        tokens, previous, positions = oracle.teacher_inputs(tokens, hidden, positions)
        assert tokens.tolist() == list(range(stage + 1, 10))
        assert positions.tolist() == list(range(501 + stage, 510))
        assert previous[:, 0].tolist() == list(range(stage, 9))
        hidden = previous + 1  # Explicit oracle chain predecessor, not target reused.


def test_pilot_retirement_refuses_symlink_before_removing_external_data(tmp_path, monkeypatch):
    from step5.core_k6 import finalize
    import json
    report = complete_report()
    report["identity"] = {"captured": "same"}
    (tmp_path / "qualification.json").write_text(json.dumps(report))
    (tmp_path / "finalization-audit.json").write_text(json.dumps(dict(status="passed", identity=report["identity"])))
    external = tmp_path / "keep"
    external.mkdir()
    (external / "sentinel").write_text("keep")
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration/frontier").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(finalize, "validate_report_code", lambda r: r)
    with pytest.raises(ValueError, match="symlink"):
        finalize.retire_intermediates(tmp_path)
    assert (external / "sentinel").read_text() == "keep"


def test_uncommitted_capture_retry_removes_only_its_partial_evidence(tmp_path):
    from types import SimpleNamespace
    frontier = tmp_path / "frontier"
    current, partial = frontier / "layer-006", frontier / "layer-007"
    for p in (current, partial, tmp_path / "proof-inputs", tmp_path / "hessians"):
        p.mkdir(parents=True)
    (current / "committed").write_text("keep input frontier")
    (partial / "batch-00000.safetensors").write_text("partial")
    keep = tmp_path / "proof-inputs/layer-004-00000.safetensors"
    keep.write_text("keep committed earlier proof")
    partial_proof = tmp_path / "proof-inputs/layer-007-00000.safetensors"
    partial_proof.write_text("partial proof from failed attempt")
    late_hessian = tmp_path / "hessians" / (pilot.modules(88)[0] + ".safetensors")
    late_hessian.write_text("partial late witness")
    partial_hessian = tmp_path / "hessians" / (pilot.modules(7)[0] + ".safetensors.partial")
    partial_hessian.write_text("partial Hessian write")
    events = tmp_path / "events.jsonl"
    events.write_text("append-only history\n")
    boundary = SimpleNamespace(current=lambda **kwargs: {"layer":6}, directory=lambda layer:frontier / f"layer-{layer:03d}")
    pilot.reset_partial_capture(tmp_path, boundary, 7)
    pilot.reset_partial_capture(tmp_path, boundary, 7)  # Retrying cleanup itself is safe.
    assert keep.read_text() == "keep committed earlier proof"
    assert (current / "committed").exists() and events.read_text() == "append-only history\n"
    assert not any(p.exists() for p in (partial, partial_proof, late_hessian, partial_hessian))
    with pytest.raises(ValueError, match="next uncommitted"):
        pilot.reset_partial_capture(tmp_path, boundary, 6)


def test_partial_capture_cleanup_refuses_symlink(tmp_path):
    from types import SimpleNamespace
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep").write_text("keep")
    (tmp_path / "proof-inputs").symlink_to(external, target_is_directory=True)
    boundary = SimpleNamespace(current=lambda **kwargs: {"layer":6}, directory=lambda layer:tmp_path / "frontier" / f"layer-{layer:03d}")
    with pytest.raises(ValueError, match="symlink"):
        pilot.reset_partial_capture(tmp_path, boundary, 7)
    assert (external / "keep").read_text() == "keep"


def test_capture_inventory_error_names_the_missing_functional_projection(tmp_path):
    raw = {name:{} for name in pilot.modules(7) if not name.endswith("sparse_indexer_w")}
    with pytest.raises(ValueError, match='"missing":\\["model.layers.7.self_attn.sparse_indexer_w"\\]'):
        pilot._save_hessians(tmp_path, {}, 7, raw, 3)


def test_sparse_indexer_w_scoring_uses_native_bf16_without_mutating_source(monkeypatch):
    import torch
    original = torch.tensor([[1.0001, -1.0002]], dtype=torch.float32)
    saved = original.clone()
    monkeypatch.setattr(pilot, "_weight", lambda source,name: original)
    value, dtype = pilot.score_weight(None, "model.layers.7.self_attn.sparse_indexer_w")
    assert dtype == "torch.float32" and value.dtype == torch.bfloat16
    assert torch.equal(value, original.bfloat16()) and torch.equal(original, saved)
    report = complete_report()
    row = next(r for r in report["results"] if r["tensor"].endswith("sparse_indexer_w"))
    row["native_effective_dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="native BF16 operand"):
        pilot.validate_qualification(report)


def test_mtp_candidate_reuses_committed_payload_after_interrupted_stage(tmp_path):
    import torch
    from types import SimpleNamespace
    calls = []
    def quantize(weight, raw, *, bits):
        calls.append(bits)
        return {"trellis":torch.arange(8,dtype=torch.int16)}, {"measured":1}, {"bits":4}
    numerical = SimpleNamespace(quantize_projection=quantize)
    first = pilot.mtp_candidate(numerical, None, {"count":3}, tmp_path, {"identity":"fixed"}, 1)
    second = pilot.mtp_candidate(numerical, None, {"count":3}, tmp_path, {"identity":"fixed"}, 1)
    assert calls == [4] and first[1:] == second[1:]
    assert torch.equal(first[0]["trellis"],second[0]["trellis"])
    with pytest.raises(ValueError, match="identity differs"):
        pilot.mtp_candidate(numerical, None, {"count":2}, tmp_path, {"identity":"fixed"}, 1)


def test_retry_event_lookup_keeps_append_only_history_and_uses_latest_success():
    from step5.core_k6.finalize import latest_event
    events = [dict(event="native_boundary_exact",layer=7,attempt=1),
              dict(event="native_boundary_exact",layer=7,attempt=2)]
    before = copy.deepcopy(events)
    assert latest_event(events,"native_boundary_exact",layer=7)["attempt"] == 2
    assert events == before
    with pytest.raises(ValueError, match="missing measured"):
        latest_event(events,"layer_committed",layer=7)
