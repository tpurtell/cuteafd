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
                    late_expert_format_witness=layer == 88, causal_input_layer=7 if layer == 88 else layer))
    return dict(status="passed", native_checks=dict.fromkeys(pilot.CHECKS, True), results=rows,
        head_measured=False, full_model_quality_measured=False)


def test_whole_prompts_and_window_witness_are_not_truncated():
    cal = [record("a", 400), record("b", 300), record("window", 4105)]
    held = [record("held", 305)]
    actual, validation, witness = pilot.select_records(cal, held, 401, 256)
    assert actual == cal and validation == held and witness == cal[2]
    assert len(actual[-1]["input_ids"]) == 4105


def test_missing_window_or_overlapping_heldout_fails():
    with pytest.raises(ValueError, match="window-crossing"):
        pilot.select_records([record("a", 500)], [record("held", 300)], 300, 256)
    with pytest.raises(ValueError, match="overlap"):
        pilot.select_records([record("a", 4105)], [record("a", 4105)], 300, 256)


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
