"""CPU gates for ERNIE's candidate, failed attempt recovery and immutable producer identity."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ernie55_flash import identity as ids


def test_committed_files_excludes_uncommitted_changes(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", tmp_path, *args], check=True, capture_output=True)
    git("init")
    module = tmp_path / "forward.py"
    module.write_text("a")
    git("add", ".")
    git("-c", "user.name=test", "-c", "user.email=test@example.org", "commit", "-m", "init")
    assert ids.committed_files(tmp_path, module.name) == {module.name: ids.sha256_file(module)}
    module.write_text("b")
    with pytest.raises(ValueError, match="committed"):
        ids.committed_files(tmp_path, module.name)


def test_fork_pin_refuses_override(monkeypatch):
    monkeypatch.setenv("INKLING_GPTQMODEL_ROOT", "/another/checkout")
    with pytest.raises(ValueError, match="GPTQModel"):
        ids.fork_identity(ids.REPO)


def parity_inputs(monkeypatch):
    current = {"m.py": "sha"}
    vendor = dict(schema="ernie55-vllm-ernie-parity-v1", status="passed",
                  source_revision=ids.BASE_REVISION, reference_files_sha256=ids.digest(current),
                  measurements={p: dict(error=.001, bound=.002) for p in
                                ("hca", "mla_dsa", "mhc", "dense", "latent_moe", "situ", "lm_head", "dspark")})
    inputs = dict(reference_files=current, code={})
    local = dict(vendor, schema="ernie55-quant-forward-parity-v1", host="maxq",
                 input_identity_sha256=ids.digest(inputs), vendor_receipt_sha256=ids.digest(vendor))
    monkeypatch.setitem(sys.modules, "ernie55_flash.forward", SimpleNamespace(
        reference=SimpleNamespace(__file__=str(ids.REPO / ids.REFERENCE_FILES[0]))))
    monkeypatch.setattr(ids, "committed_files", lambda *_: current)
    monkeypatch.setattr(ids, "git", lambda *_: "merged-blob")
    return inputs, vendor, local


def test_measured_gates_bind_every_phase(monkeypatch):
    inputs, vendor, local = parity_inputs(monkeypatch)
    assert ids.verified_identity(inputs, vendor, local)["inputs"] == inputs
    with pytest.raises(ValueError, match="maxq forward parity"):
        ids.verified_identity(inputs, vendor, {**local, "input_identity_sha256": "different"})
    for value in (None, float("nan"), .01):
        row = dict(vendor, measurements={**vendor["measurements"], "hca": dict(error=value, bound=.002)})
        with pytest.raises(ValueError, match="parity bound failed"):
            ids.verified_identity(inputs, row, local)
    with pytest.raises(ValueError, match="missing measured"):
        ids.verified_identity(inputs, dict(vendor, measurements={}), local)
    monkeypatch.setattr(ids, "git", lambda *_args: _args[-1])
    with pytest.raises(ValueError, match="merged dev"):
        ids.verified_identity(inputs, vendor, local)


def session_factory(tmp_path, monkeypatch):
    inputs, vendor, local = parity_inputs(monkeypatch)
    monkeypatch.setattr(ids, "candidate", lambda *_: inputs)
    @contextmanager
    def session():
        with ids.producer_session(tmp_path, None, None, vendor, local) as identity:
            yield identity
    return session


def test_failure_interruption_and_completed_guards(tmp_path, monkeypatch):
    session = session_factory(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="source EIO"):
        with session() as frozen:
            with pytest.raises(BlockingIOError), session():
                pytest.fail("second producer entered")
            raise RuntimeError("source EIO")
    failure = ids.read(tmp_path / "attempt.json")
    assert failure["detail"] == "source EIO"
    diagnosis = tmp_path / "diagnoses" / (ids.digest(failure) + ".json")
    with pytest.raises(ValueError, match="missing diagnosis"), session():
        pytest.fail("undiagnosed retry")
    evidence = dict(identity_sha256=ids.digest(frozen), failure_sha256="stale", cause="local I/O", resolution="mount repaired")
    ids.atomic_json(diagnosis, evidence)
    with pytest.raises(ValueError, match="exact producer failure"), session():
        pytest.fail("unrelated diagnosis")
    ids.atomic_json(diagnosis, dict(evidence, failure_sha256=ids.digest(failure)))
    with session() as resumed:
        assert resumed == frozen
    assert ids.read(tmp_path / "failures" / (ids.digest(failure) + ".json")) == failure
    # SIGKILL leaves a different current attempt, so the previous diagnosis is stale.
    attempt = tmp_path / "attempt.json"
    ids.atomic_json(attempt, dict(ids.read(attempt), status="running"))
    with pytest.raises(ValueError, match="missing diagnosis"), session():
        pytest.fail("interrupted job resumed")
    ids.atomic_json(tmp_path / "numerical-complete.json", dict(status="passed"))
    with pytest.raises(ValueError, match="already completed"), session():
        pytest.fail("completed job restarted")


def test_recovery_rejects_missing_or_changed_frozen_identity(tmp_path, monkeypatch):
    session = session_factory(tmp_path, monkeypatch)
    with session():
        pass
    path = tmp_path / "identity.json"
    ids.atomic_json(path, {"schema": "stale"})
    with pytest.raises(ValueError, match="frozen numerical identity differs"), session():
        pytest.fail("changed identity")
    path.unlink()
    with pytest.raises(ValueError, match="without a frozen identity"), session():
        pytest.fail("orphaned production state")
