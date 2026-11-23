"""Crash boundaries and evidence required by the GLM Flash production coordinator."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc import run


def test_frontier_advances_only_after_weights_metrics_and_successor_then_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "PHASES", ("initial", "layer-00", "head", "layer-45"))
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    live, sealed, called = set(), set(), []
    def make(label, stage):
        called.append(label)
        sealed.add(label)
        live.add(stage)
        if label != "source-native":
            run.atomic_json(metrics / (label + ".json"), [])
        return stage
    def target(index, incoming):
        assert incoming == "initial"
        return make(f"layer-{index:02}", "next")
    engine = SimpleNamespace(initial=lambda:make("source-native", "initial"), target=target,
                             head=lambda _:make("head", "next"), mtp=lambda _:make("layer-45","mtp"))
    def retire(name):
        assert run.read(tmp_path/"frontier.json")["history"][-1]["stage"] != name
        live.discard(name)
    stages = SimpleNamespace(load=lambda name:{} if name in live else None, retire=retire)
    writer = SimpleNamespace(group_complete=lambda label:label in sealed and label!="layer-00")
    kwargs = dict(attempt=1, interrupted_phase=None, status=lambda *_:None)
    with pytest.raises(ValueError, match="unsealed"):
        run.traverse(tmp_path, engine, stages, writer, {}, **kwargs)
    assert [row["phase"] for row in run.read(tmp_path/"frontier.json")["history"]] == ["initial"]
    assert "initial" in live
    writer.group_complete = sealed.__contains__
    kwargs.update(attempt=2, interrupted_phase="layer-00")
    history = run.traverse(tmp_path, engine, stages, writer, {}, **kwargs)
    assert called == ["source-native","layer-00","layer-00","head","layer-45"]
    assert live == {"mtp"}
    assert [r["interrupted"] for r in history] == [False,True,False,False]
    run.traverse(tmp_path, engine, stages, writer, {}, **kwargs)
    assert called.count("layer-45") == 1


def test_interrupted_attempt_needs_diagnosis_and_changed_boot_when_requested(tmp_path, monkeypatch):
    root = tmp_path/"run"
    root.mkdir()
    output = tmp_path/"out"
    monkeypatch.setattr(run, "numerical_identity", lambda *args: {"fake":"identity"})
    def fail(*args, status, **kwargs):
        status("running","layer-00")
        raise OSError("EIO")
    monkeypatch.setattr(run, "produce", fail)
    with pytest.raises(OSError, match="EIO"):
        run.run(root, output, tmp_path/"source")
    prior = run.read(root/"status.json")
    assert (prior["attempt"], prior["phase"]) == (1,"layer-00")
    with pytest.raises(ValueError, match="diagnosis"):
        run.run(root, output, tmp_path/"source")
    with pytest.raises(ValueError, match="changed boot"):
        run.run(root, output, tmp_path/"source", diagnosis="source restored", after_boot_id=prior["boot_id"])
    with pytest.raises(OSError, match="EIO"):
        run.run(root, output, tmp_path/"source", diagnosis="source restored")
    assert run.read(root/"attempts/0002.json")["previous_status"] == prior


def test_source_changed_before_resume_refuses_any_work(tmp_path):
    run.bind(tmp_path/"identity.json", {"source":"rev1"})
    with pytest.raises(ValueError, match="identity differs"):
        run.bind(tmp_path/"identity.json", {"source":"rev2"})


def test_csv_requires_every_width_and_audited_recovery(tmp_path):
    rows = [dict(module="foo", bits=4, side=1, codebook="mul1", hessian_weighted_relative_error=.001,
                 natural_rows=100, augmented_rows=24, identity_rows=900, effective_rows=1024,
                 augmented_rank_counts=[24]+[0]*7, selection_xxh3_128="a"*32)]
    run.atomic_json(tmp_path/"metrics/layer-03.json", rows)
    result = run.write_error_csv(tmp_path, {("foo",4)})
    assert "augmented_rank_counts" in result.read_text()
    with pytest.raises(ValueError, match="exactly once"):
        run.write_error_csv(tmp_path, {("foo",4),("head",8)})
    rows[0]["natural_rows"] = 101
    run.atomic_json(tmp_path/"metrics/layer-03.json", rows)
    with pytest.raises(ValueError, match="recovery evidence"):
        run.write_error_csv(tmp_path, {("foo",4)})


def test_completed_stays_completed_even_after_payload_retirement(tmp_path, monkeypatch):
    root = tmp_path/"run"
    root.mkdir()
    receipt = dict(status="passed", identity={}, repository=run.source.TARGET_REPO, output=str(tmp_path/"out"))
    run.atomic_json(root/"numerical-complete.json", receipt)
    run.atomic_json(root/"identity.json", {})
    monkeypatch.setattr(run, "produce", lambda *_: pytest.fail("restart"))
    assert run.run(root, tmp_path/"out", tmp_path/"src") == receipt


def test_gb10_evidence_bound_to_kernel_files_and_native_bank():
    names = ("experts.py", "numerics.py", "qualification.py")
    code = dict(fork="a"*40, files={"quantization/glm5_flash/mul1_mc/"+n:"b"*64 for n in names})
    receipt = dict(status="passed", device="NVIDIA GB10", fork_revision=code["fork"],
                   code_sha256={n:"b"*64 for n in names}, compared_k4_encodes=3,
                   native_bf16_compared_k4_encodes=3, raw_hessians_equal=True, packed_bytes_equal=True,
                   heldout_sum_equal=True, sealed_replay_equal=True, selection_equal=True,
                   gate_evidence_equal=True, workspace_reused=True, native_bf16_bank_equal=True)
    run.validate_gb10(receipt, code)
    with pytest.raises(ValueError, match="exactness witness"):
        run.validate_gb10(receipt | dict(native_bf16_bank_equal=False), code)
    with pytest.raises(ValueError, match="exactness witness"):
        run.validate_gb10(receipt | dict(fork_revision="c"*40), code)
