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


def test_server_key_translation_preserves_sealed_window():
    import copy
    span = {"start": 2, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2],
            "fixture": {"path": "image.png", "sha256": "b" * 64}}
    window = {"tokens": [1, 2, 9, 3], "roles": ["ctx"] * 4, "score_from": 3, "media": [span]}
    original = copy.deepcopy(window)
    mapping = {span["key"]: {"old_key": span["key"], "new_key": "c" * 64,
        "grid": span["grid"], "len": span["len"], "fixture": span["fixture"]}}
    translated = module.mapped_window(window, mapping)
    assert window == original
    assert translated["media"][0]["key"] == "c" * 64
    translated["media"][0]["key"] = span["key"]
    assert translated == original
    mapping[span["key"]]["grid"] = [1, 4, 4]
    with pytest.raises(ValueError, match="geometry"):
        module.mapped_window(window, mapping)


def test_normal_server_echo_supplies_key_and_preserves_pixels(tmp_path, monkeypatch):
    import hashlib
    import io
    from types import SimpleNamespace
    Image = pytest.importorskip("PIL.Image")
    data = io.BytesIO()
    pixels = Image.new("RGB", (4, 4), (17, 29, 53))
    pixels.save(data, format="PNG")
    data = data.getvalue()
    span = {"key": "a" * 64, "len": 1, "grid": [1, 2, 2],
        "fixture": {"path": "image.png", "sha256": hashlib.sha256(data).hexdigest()}}
    panel = {"checkpoint": "model", "family": "qwen4", "windows": [{"media": [span]}, {"media": [span]}]}
    monkeypatch.setattr(module, "read_fixture", lambda *args: data)
    calls = []
    def http(url, token, body, timeout):
        calls.append(body)
        assert "prompt_ids" not in body["spec"] and "media" not in body["spec"]
        assert body["body"]["messages"][0]["content"][0]["type"] == "image_url"
        return {"server": {"model": "model", "family": "qwen4"}, "probe": {"media": [{"kind": "image",
            "key": "c" * 64, "grid": span["grid"], "len": 1}]}}
    monkeypatch.setattr(module, "http", http)
    args = SimpleNamespace(media_root=tmp_path, out=tmp_path, url="http://localhost", bench_token=None,
                           timeout=10, mode="native")
    mapping = module.discover_keys(args, panel)
    assert len(calls) == 1
    assert mapping[span["key"]]["new_key"] == "c" * 64
    assert mapping[span["key"]]["pixel_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    capture = {"key_mapping_sha256": module.file_hash(tmp_path / "key-mapping.json"), "server": {"model": "model", "family": "qwen4"}}
    assert module.capture_mapping(tmp_path, capture) == mapping
    (tmp_path / (span["key"] + ".prepare.json")).write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        module.capture_mapping(tmp_path, capture)


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
@pytest.mark.parametrize("server_keys", (False, True))
def test_compare_uses_golden_difference_not_direct_kl(tmp_path, monkeypatch, family, server_keys):
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
    metadata = {"key": span["key"], "sha256": "c" * 64}
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
                response["probe"]["provenance"] = {"mode": "reference_features", "probe_only": True,
                    "encoder_bypassed": True, "features": [metadata]}
            response_path = root / (window["id"] + ".json")
            response_path.write_bytes(canonical(response))
            capture["windows"].append({"id": window["id"], "path": str(dump), "files": files,
                "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256((dump / "manifest.jsonl").read_bytes()).hexdigest()})
        if server_keys:
            mapping = {span["key"]: {"old_key": span["key"], "new_key": "d" * 64,
                "grid": span["grid"], "len": span["len"], "fixture": span["fixture"]}}
            monkeypatch.setattr(module, "capture_mapping", lambda *args: mapping)
            for window in panel["windows"]:
                path = root / (window["id"] + ".json")
                response = json.loads(path.read_bytes())
                response["probe"]["media"][0]["key"] = "d" * 64
                if mode == "reference":
                    response["probe"]["provenance"]["features"][0]["key"] = "d" * 64
                path.write_bytes(canonical(response))
                entry = next(e for e in capture["windows"] if e["id"] == window["id"])
                entry["response_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
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
