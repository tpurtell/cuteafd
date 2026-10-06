"""Paired encoder gates decide against golden rows, never direct arm KL."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("media_paired_probe", ROOT / "scripts/bench/media-paired-probe.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def feature_metadata(key, family):
    metadata = {"key": key, "sha256": "c" * 64}
    if family == "glm5_flash":
        metadata["snapshot_identity"] = {
            "modeling_sha256": module.GLM_MODELING_SHA256,
            "image_processing_sha256": module.GLM_IMAGE_PROCESSING_SHA256}
    return metadata


def reference_provenance(metadata, family):
    provenance = {"mode": "reference_features", "probe_only": True,
                  "encoder_bypassed": True, "features": [metadata]}
    if family == "glm5_flash":
        provenance["modeling_source_revision"] = module.GLM_TRANSFORMERS_REVISION
    return provenance


@pytest.mark.parametrize("family", module.FAMILIES)
def test_capture_retains_http_error_body(tmp_path, monkeypatch, family):
    import io
    import urllib.error
    from types import SimpleNamespace
    window = {"id": "vision00", "tokens": [9, 2], "score_from": 1,
              "media": [{"fixture": {"path": "image.png", "sha256": "a" * 64}}]}
    panel = {"family": family, "checkpoint": "model", "quick_windows": ["vision00"], "windows": [window]}
    def load_set(path, expected):
        assert expected == family
        return panel
    monkeypatch.setattr(module, "load_set", load_set)
    monkeypatch.setattr(module, "read_fixture", lambda *args: b"png")
    def http(url, token, body=None, timeout=240):
        if body is None:
            return {"data": [{"id": "model", "capabilities": {"vision": True}}]}
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"media guard"}'))
    monkeypatch.setattr(module, "http", http)
    args = SimpleNamespace(windows=tmp_path / "windows", out=tmp_path / "out", url="http://localhost",
        bench_token=None, mode="native", features=None, quick=False, media_root=tmp_path,
        host_dump=tmp_path / "host", server_dump=tmp_path / "server", timeout=10, family=family)
    with pytest.raises(urllib.error.HTTPError):
        module.capture(args)
    saved = json.loads((args.out / "vision00.http-error.json").read_text())
    assert saved == {"status": 400, "body": '{"error":"media guard"}'}
    assert not (args.out / "capture.json").exists()


def test_window_bootstrap_is_paired_and_deterministic():
    stats, counts = [[.02, .01], [.04, .02]], [10, 20]
    result = module.paired_bounds(stats, counts, 500, 7)
    assert result == module.paired_bounds(stats, counts, 500, 7)
    assert result["kl_increase"] == pytest.approx(.002)
    assert result["top1_loss"] == pytest.approx(.001)
    assert result["kl_increase_upper95"] == pytest.approx(.002)
    assert result["top1_loss_upper95"] == pytest.approx(.001)
    with pytest.raises(ValueError):
        module.paired_bounds([[0, 0]], [1])


def test_dump_positions_and_safe_files(tmp_path):
    window = {"score_from": 2, "tokens": [1, 2, 3, 4]}
    rows = [{"position": pos, "vocab_size": 3, "file": f"row-{pos}.safetensors",
             "tensor": "log_probs", "dtype": "F32", "byte_order": "little"} for pos in [2, 3]]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert list(module.dump_rows(tmp_path, window, 3)) == [2, 3]
    rows[1]["position"] = 2
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="duplicate"):
        module.dump_rows(tmp_path, window, 3)
    rows[1]["position"], rows[0]["file"] = 3, "../escape"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        module.dump_rows(tmp_path, window, 3)


def test_record_requires_native_identity_and_override_metadata(tmp_path):
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2]}
    window = {"tokens": [9, 2], "score_from": 1, "media": [span]}
    record = {"engine": "mimo", "cold": True, "no_speculation": True, "cached_tokens": 0,
              "score_path": "decode", "prompt_ids": [9, 2], "media": [span],
              "rows": [{"position": 1, "finite": True}], "scored": 1}
    response = {"probe": record, "server": {"model": "model"}}
    module.check_record(window, response, "model", "native")
    response["server"]["family"] = "mimo_v2"
    module.check_record(window, response, "model", "native", family="mimo_v2")
    with pytest.raises(ValueError, match="identity"):
        module.check_record(window, response, "model", "native", family="qwen4")
    record["media"] = []
    with pytest.raises(ValueError, match="echo"):
        module.check_record(window, response, "model", "native")
    record["media"] = [span]
    metadata = {"key": span["key"], "sha256": "b" * 64}
    (tmp_path / (span["key"] + ".json")).write_text(json.dumps(metadata))
    record["provenance"] = {"mode": "reference_features", "probe_only": True,
                            "encoder_bypassed": True, "features": [metadata]}
    module.check_record(window, response, "model", "reference", tmp_path)
    with pytest.raises(ValueError, match="native arm"):
        module.check_record(window, response, "model", "native")
    record["provenance"]["features"][0] = {"sha256": "c" * 64}
    with pytest.raises(ValueError, match="provenance"):
        module.check_record(window, response, "model", "reference", tmp_path)


@pytest.mark.parametrize("family", module.FAMILIES)
def test_reference_provenance_keeps_family_contract_exact(tmp_path, family):
    import copy
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2]}
    window = {"tokens": [9, 2], "score_from": 1, "media": [span]}
    metadata = feature_metadata(span["key"], family)
    (tmp_path / (span["key"] + ".json")).write_text(json.dumps(metadata))
    provenance = reference_provenance(metadata, family)
    response = {"server": {"model": "model", "family": family}, "probe": {
        "engine": family, "cold": True, "no_speculation": True, "cached_tokens": 0,
        "score_path": "decode", "prompt_ids": window["tokens"], "media": [span],
        "rows": [{"position": 1, "finite": True}], "scored": 1, "provenance": provenance}}
    module.check_record(window, response, "model", "reference", tmp_path, family)
    changes = [{**provenance, "unverified_source": "extra"}]
    for field in ("mode", "probe_only", "encoder_bypassed", "features"):
        missing = copy.deepcopy(provenance)
        missing.pop(field)
        changes.append(missing)
    mismatched = copy.deepcopy(provenance)
    mismatched["features"][0]["sha256"] = "d" * 64
    changes.append(mismatched)
    if family == "glm5_flash":
        missing = copy.deepcopy(provenance)
        missing.pop("modeling_source_revision")
        changes.extend([missing, {**provenance, "modeling_source_revision": "0" * 40}])
    else:
        changes.append({**provenance, "modeling_source_revision": module.GLM_TRANSFORMERS_REVISION})
    for changed in changes:
        response["probe"]["provenance"] = changed
        with pytest.raises(ValueError, match="provenance"):
            module.check_record(window, response, "model", "reference", tmp_path, family)
    response["probe"]["provenance"] = provenance
    with pytest.raises(ValueError, match="native arm"):
        module.check_record(window, response, "model", "native", tmp_path, family)


@pytest.mark.parametrize("field", ["modeling_sha256", "image_processing_sha256"])
@pytest.mark.parametrize("bad_value", [None, "0" * 64])
def test_glm_provenance_requires_sealed_pinned_sources(tmp_path, field, bad_value):
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2]}
    window = {"tokens": [9, 2], "score_from": 1, "media": [span]}
    metadata = feature_metadata(span["key"], "glm5_flash")
    if bad_value is None:
        metadata["snapshot_identity"].pop(field)
    else:
        metadata["snapshot_identity"][field] = bad_value
    (tmp_path / (span["key"] + ".json")).write_text(json.dumps(metadata))
    response = {"server": {"model": "model", "family": "glm5_flash"}, "probe": {
        "engine": "glm5_flash", "cold": True, "no_speculation": True, "cached_tokens": 0,
        "score_path": "decode", "prompt_ids": window["tokens"], "media": [span],
        "rows": [{"position": 1, "finite": True}], "scored": 1,
        "provenance": reference_provenance(metadata, "glm5_flash")}}
    with pytest.raises(ValueError, match="modeling source"):
        module.check_record(window, response, "model", "reference", tmp_path, "glm5_flash")


def test_log_probs_validate_shape_and_normalize(tmp_path):
    from safetensors.numpy import save_file
    path = tmp_path / "row.safetensors"
    save_file({"log_probs": np.array([-1., -2., -3.], dtype=np.float32)}, str(path))
    values = module.log_probs(path, 3)
    assert np.exp(values).sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="shape"):
        module.log_probs(path, 4)
    with pytest.raises(ValueError, match="nonfinite"):
        module.normalize([0, float("nan")])


@pytest.mark.parametrize("family", module.FAMILIES)
def test_compare_uses_golden_difference_not_direct_kl(tmp_path, monkeypatch, family):
    import hashlib
    from types import SimpleNamespace
    from safetensors.numpy import save_file
    from fidelity_windows import canonical, set_hash
    panel = {"schema": "cuteafd.fidelity.set/1", "family": family, "checkpoint": "model",
             "quick_windows": ["w0", "w1"], "windows": []}
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2],
            "fixture": {"path": "image.png", "sha256": "b" * 64}}
    for name in ["w0", "w1"]:
        panel["windows"].append({"id": name, "block": "vision", "bucket": "0-2K",
                                "tokens": [9] * 577, "roles": ["ctx"] * 65 + ["gen"] * 512,
                                "score_from": 65, "media": [span]})
    panel["set_sha256"] = set_hash(panel)
    windows = tmp_path / "windows.json"
    windows.write_bytes(canonical(panel))
    identity = {"snapshot_revision": "pinned"}
    proof = {"schema": "cuteafd.fidelity.prefix/1", "passed": True, "finite": True,
             "rows": 512, "different_rows": [], "argmax_disagreements": 0,
             "lengths": [576, 640], "score_from": 64, "fixed_rows": 128,
             "family": family, "set_sha256": panel["set_sha256"], "snapshot_identity": identity,
             "source_window": "w0", "media": [span], "vocab": 2}
    golden = tmp_path / "golden"
    golden.mkdir()
    meta = {"set_sha256": panel["set_sha256"], "checkpoint": "model", "snapshot_identity": identity,
            "prefix_qualification": proof, "windows": []}
    # Golden is uniform. Symmetric arms are equally distant from it, but very far
    # from each other: the deciding paired KL increase is zero, direct KL is large.
    for window in panel["windows"]:
        folder = golden / window["id"]
        folder.mkdir()
        np.zeros((512, 2), dtype="<f4").tofile(folder / "logits.bin")
        np.asarray(window["tokens"], dtype="<i4").tofile(folder / "tokens.bin")
        meta["windows"].append({"id": window["id"], "path": window["id"], "positions": list(range(65, 577)),
                                "vocab": 2, "media": [span]})
    (golden / "meta.json").write_bytes(canonical(meta))
    (golden / "windows.json").write_bytes(canonical(panel))
    seal = tmp_path / "golden-seal.json"
    monkeypatch.setattr("sys.argv", ["media-paired-probe.py", "seal", "--family", family,
        "--windows", str(windows), "--golden", str(golden), "--out", str(seal)])
    module.main()
    assert json.loads(seal.read_text()) == module.golden_seal(golden, panel)
    features = tmp_path / "features"
    features.mkdir()
    metadata = feature_metadata(span["key"], family)
    (features / (span["key"] + ".json")).write_text(json.dumps(metadata))
    for mode, row in [("native", [0., -5.]), ("reference", [-5., 0.])]:
        root = tmp_path / mode
        root.mkdir()
        capture = {"schema": "cuteafd.media.paired.capture/1", "mode": mode, "set_sha256": panel["set_sha256"],
                   "checkpoint": "model", "quick": False,
                   "server": {"model": "model", "family": family,
                              "build": {"commit": "same", "image": mode}}, "windows": []}
        for window in panel["windows"]:
            dump = root / window["id"]
            dump.mkdir()
            manifest, files = [], []
            for pos in range(65, 577):
                path = dump / f"row-{pos}.safetensors"
                save_file({"log_probs": np.array(row, dtype=np.float32)}, str(path))
                manifest.append({"position": pos, "vocab_size": 2, "file": path.name,
                                 "tensor": "log_probs", "dtype": "F32", "byte_order": "little"})
                files.append({"position": pos, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (dump / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in manifest))
            response = {"server": capture["server"], "probe": {"engine": "mimo", "cold": True,
                "no_speculation": True, "cached_tokens": 0, "score_path": "decode",
                "prompt_ids": window["tokens"], "media": [span], "scored": 512,
                "rows": [{"position": pos, "finite": True} for pos in range(65, 577)]}}
            if mode == "reference":
                response["probe"]["provenance"] = reference_provenance(metadata, family)
            response_path = root / (window["id"] + ".json")
            response_path.write_bytes(canonical(response))
            capture["windows"].append({"id": window["id"], "path": str(dump), "files": files,
                "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256((dump / "manifest.jsonl").read_bytes()).hexdigest()})
        (root / "capture.json").write_bytes(canonical(capture))
    out = tmp_path / "g4.json"
    args = SimpleNamespace(windows=windows, golden=golden, golden_seal=seal, native=tmp_path / "native",
                           reference=tmp_path / "reference", features=features, out=out, bootstrap=100, seed=7)
    if family != "mimo_v2":
        with pytest.raises(ValueError, match="set family"):
            module.compare(args)
        args.family = family
    assert module.compare(args)
    result = json.loads(out.read_text())
    assert result["kl_increase_upper95"] == pytest.approx(0)
    assert result["top1_loss_upper95"] == pytest.approx(-1)
    assert result["windows"][0]["direct_reference_native_kl"] > 4
    assert result["criterion"] == module.CRITERION
    capture_path = tmp_path / "reference/capture.json"
    sealed = json.loads(capture_path.read_text())
    sealed["windows"].pop()
    capture_path.write_bytes(canonical(sealed))
    args.out = tmp_path / "incomplete.json"
    with pytest.raises(ValueError, match="incomplete windows"):
        module.compare(args)
    with (golden / "w0/logits.bin").open("r+b") as changed:
        changed.write(np.asarray([1.], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="immutable seal"):
        module.compare(args)
