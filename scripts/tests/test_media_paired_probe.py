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
    response["server"]["family"] = "glm5"
    with pytest.raises(ValueError, match="identity"):
        module.check_record(window, response, "model", "native", family="mistral_large4")
    response["server"]["family"] = "mistral_large4"
    module.check_record(window, response, "model", "native", family="mistral_large4")
    with pytest.raises(ValueError, match="identity"):
        module.check_record(window, response, "other", "native", family="mistral_large4")
    response["server"]["family"] = "qwen4"
    with pytest.raises(ValueError, match="identity"):
        module.check_record(window, response, "model", "native", family="mistral_large4")
    response["server"]["family"] = "mimo_v2"
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


def compare_case(tmp_path, monkeypatch, family, control=False, native_row=(0., -5.), geometry="tiled", quick=False):
    import hashlib
    from types import SimpleNamespace
    from safetensors.numpy import save_file
    from fidelity_windows import canonical, set_hash
    panel = {"schema": "cuteafd.fidelity.set/1", "family": family, "checkpoint": "model",
             "quick_windows": ["w1", "w2"] if quick else ["w0", "w1"], "windows": []}
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2],
            "fixture": {"path": "image.png", "sha256": "b" * 64}}
    for name in (["w0", "w1", "w2"] if quick else ["w0", "w1"]):
        panel["windows"].append({"id": name, "block": "vision", "bucket": "0-2K",
                                "tokens": [9] * 577, "roles": ["ctx"] * 65 + ["gen"] * 512,
                                "score_from": 65, "media": [span]})
    if control:
        panel["windows"][1].pop("media")
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
                                "vocab": 2, "media": window.get("media", [])})
    if geometry == "official-full":
        import copy
        from fidelity_windows import validate_reference
        meta.pop("prefix_qualification")
        meta.update(geometry=geometry, attention="eager", use_cache=False, mtp=False,
                    source_pin={"official": "pinned"})
        meta["official_full"] = {"schema": "cuteafd.reference.official-full/1", "family": panel["family"],
            "set_sha256": panel["set_sha256"], "snapshot_identity": identity,
            "layers": 72, "completed_layers": 72, "tiling": False, "head": "full-length",
            "windows": {w["id"]: {"tokens": len(w["tokens"]),
                "tokens_sha256": hashlib.sha256(canonical(w["tokens"])).hexdigest()} for w in panel["windows"]}}
        assert validate_reference(meta, panel) == 2
        for change in ({"tiling": True}, {"completed_layers": 71}, {"head": "fixed-M128"}, {"windows": {}}):
            bad = copy.deepcopy(meta)
            bad["official_full"].update(change)
            with pytest.raises(ValueError, match="incomplete or mismatched"):
                validate_reference(bad, panel)
        bad = copy.deepcopy(meta)
        bad["prefix_qualification"] = {}
        with pytest.raises(ValueError, match="must not claim"):
            validate_reference(bad, panel)
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
    for mode, row in [("native", native_row), ("reference", [-5., 0.])]:
        root = tmp_path / mode
        root.mkdir()
        capture = {"schema": "cuteafd.media.paired.capture/1", "mode": mode, "set_sha256": panel["set_sha256"],
                   "checkpoint": "model", "quick": quick,
                   "server": {"model": "model", "family": family,
                              "build": {"commit": "same", "image": mode}}, "windows": []}
        for window in panel["windows"]:
            if quick and window["id"] not in panel["quick_windows"]:
                continue
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
                "prompt_ids": window["tokens"], "media": window.get("media", []), "scored": 512,
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
    return args, panel


@pytest.mark.parametrize("geometry", ["tiled", "official-full"])
@pytest.mark.parametrize("family", module.FAMILIES)
def test_compare_uses_golden_difference_not_direct_kl(tmp_path, monkeypatch, family, geometry):
    from fidelity_windows import canonical
    args, panel = compare_case(tmp_path, monkeypatch, family, geometry=geometry)
    if family != "mimo_v2":
        with pytest.raises(ValueError, match="set family"):
            module.compare(args)
        args.family = family
    assert module.compare(args)
    result = json.loads(args.out.read_text())
    assert result["kl_increase_upper95"] == pytest.approx(0)
    assert result["top1_loss_upper95"] == pytest.approx(-1)
    assert result["windows"][0]["direct_reference_native_kl"] > 4
    assert result["criterion"] == module.CRITERION
    # Freeze the old two-arm serialized result, including its floating-point math.
    g, n, r = (module.normalize(v) for v in ([0., 0.], [0., -5.], [-5., 0.]))
    values = np.asarray([[float(np.dot(np.exp(g), g - n)), float(np.dot(np.exp(g), g - r)),
                          1., 0., float(np.dot(np.exp(r), r - n)),
                          float(np.dot(np.exp(n), n - r))]] * 512)
    means = values.mean(axis=0)
    stats = [float((values[:, 0] - values[:, 1]).sum()), float((values[:, 3] - values[:, 2]).sum())]
    expected = {"schema": "cuteafd.media.g4/1", "criterion": module.CRITERION,
        "set_sha256": panel["set_sha256"], "checkpoint": "model",
        "golden_seal_sha256": module.file_hash(args.golden_seal), "path": "decode-shaped",
        "windows": [{"id": name, "rows": 512, "native_golden_kl": float(means[0]),
            "reference_golden_kl": float(means[1]), "native_top1": float(means[2]),
            "reference_top1": float(means[3]), "direct_reference_native_kl": float(means[4]),
            "direct_native_reference_kl": float(means[5])} for name in ["w0", "w1"]],
        **module.paired_bounds([stats, stats], [512, 512], 100, 7), "pass": True,
        "scope": "encoder-swap gate only; no prefill qualification or precision-default promotion"}
    assert args.out.read_bytes() == canonical(expected) + b"\n"
    capture_path = tmp_path / "reference/capture.json"
    sealed = json.loads(capture_path.read_text())
    sealed["windows"].pop()
    capture_path.write_bytes(canonical(sealed))
    args.out = tmp_path / "incomplete.json"
    with pytest.raises(ValueError, match="incomplete windows"):
        module.compare(args)
    with (args.golden / "w0/logits.bin").open("r+b") as changed:
        changed.write(np.asarray([1.], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="immutable seal"):
        module.compare(args)


def test_official_full_golden_seals_and_compares_without_prefix(tmp_path, monkeypatch):
    import copy
    import hashlib
    from fidelity_windows import canonical, validate_reference
    args, panel = compare_case(tmp_path, monkeypatch, "mimo_v2", control=True, native_row=(0., 0.))
    meta = json.loads((args.golden / "meta.json").read_text())
    meta.pop("prefix_qualification")
    meta.update(geometry="official-full", attention="eager", use_cache=False, mtp=False, source_pin={"official": "pinned"})
    meta["official_full"] = {"schema": "cuteafd.reference.official-full/1", "family": panel["family"],
        "set_sha256": panel["set_sha256"], "snapshot_identity": meta["snapshot_identity"],
        "layers": 72, "completed_layers": 72, "tiling": False, "head": "full-length",
        "windows": {w["id"]: {"tokens": len(w["tokens"]),
            "tokens_sha256": hashlib.sha256(canonical(w["tokens"])).hexdigest()} for w in panel["windows"]}}
    assert validate_reference(meta, panel) == 2
    for change in ({"tiling": True}, {"completed_layers": 71}, {"head": "fixed-M128"}, {"windows": {}}):
        bad = copy.deepcopy(meta)
        bad["official_full"].update(change)
        with pytest.raises(ValueError, match="incomplete or mismatched"):
            validate_reference(bad, panel)
    bad = copy.deepcopy(meta)
    bad["prefix_qualification"] = {}
    with pytest.raises(ValueError, match="must not claim"):
        validate_reference(bad, panel)
    (args.golden / "meta.json").write_bytes(canonical(meta))
    args.golden_seal = tmp_path / "official-full-seal.json"
    module.write_new(args.golden_seal, module.golden_seal(args.golden, panel))
    args.reference = args.features = None
    args.expert_bpw = 4.
    assert module.compare(args)


@pytest.mark.parametrize("bpw,kl,top1", [(2.99, .07, .90), (3., .05, .93), (4., .05, .93), (6., .03, .95)])
def test_native_expert_bit_tiers(bpw, kl, top1):
    bar = module.fidelity_bar(bpw)
    assert (bar["max_kl"], bar["min_top1"]) == (kl, top1)


@pytest.mark.parametrize("bpw", [None, True, "4", 0, -1, float("nan"), float("inf")])
def test_native_expert_bits_fail_closed(bpw):
    with pytest.raises(ValueError, match="expert-bpw"):
        module.fidelity_bar(bpw)


def test_native_metrics_weight_rows_not_windows():
    metrics = module.native_metrics([
        {"rows": 1, "native_golden_kl": .08, "native_top1": 0},
        {"rows": 9, "native_golden_kl": 0., "native_top1": 1}], module.fidelity_bar(4))
    assert metrics["native_golden_kl"] == pytest.approx(.008)
    assert metrics["native_top1"] == pytest.approx(.9)
    assert not metrics["pass"]
    assert module.native_metrics([], module.fidelity_bar(4)) is None


@pytest.mark.parametrize("row,passed", [((0., 0.), True), ((0., -5.), False)])
def test_native_only_official_golden_and_text_control(tmp_path, monkeypatch, row, passed):
    from fidelity_windows import canonical
    args, panel = compare_case(tmp_path, monkeypatch, "mimo_v2", control=True, native_row=row)
    args.reference = args.features = None
    args.expert_bpw = 4.
    assert module.compare(args) is passed
    result = json.loads(args.out.read_text())
    assert result["schema"] == "cuteafd.media.g4.native/1"
    assert result["pass"] is passed
    assert result["pooled"]["rows"] == 1024
    assert result["image"]["rows"] == result["control"]["rows"] == 512
    assert result["image"]["pass"] is result["control"]["pass"] is passed
    assert result["bar"]["expert_bpw"] == 4.
    assert [w["kind"] for w in result["windows"]] == ["media", "control"]
    assert result["pooled"]["native_top1"] == 1.
    assert result["pooled"]["native_golden_kl"] == pytest.approx(0 if passed else 1.8135681679291724)
    args.out = tmp_path / "rejected.json"
    args.features = tmp_path / "features"
    with pytest.raises(ValueError, match="supplied together"):
        module.compare(args)
    args.features = None
    capture_path = args.native / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["quick"] = True
    capture_path.write_bytes(canonical(capture))
    with pytest.raises(ValueError, match="complete panel"):
        module.compare(args)
    capture["quick"] = False
    capture["windows"].pop()
    capture_path.write_bytes(canonical(capture))
    with pytest.raises(ValueError, match="incomplete windows"):
        module.compare(args)
    with (args.golden / "w0/logits.bin").open("r+b") as changed:
        changed.write(np.asarray([1.], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="immutable seal"):
        module.compare(args)


def test_capture_text_control_without_media(tmp_path, monkeypatch):
    from types import SimpleNamespace
    window = {"id": "control", "tokens": [9, 2], "score_from": 1}
    panel = {"family": "mimo_v2", "checkpoint": "model", "set_sha256": "a" * 64,
             "quick_windows": ["control"], "windows": [window]}
    monkeypatch.setattr(module, "load_panel", lambda a: panel)
    monkeypatch.setattr(module, "read_fixture", lambda *a: pytest.fail("text control read image"))
    monkeypatch.setattr(module, "dump_rows", lambda *a: {})
    dump = tmp_path / "host/control"
    dump.parent.mkdir(parents=True)
    monkeypatch.setenv("CUTEAFD_API_KEY", "synthetic-env-token")
    def http(url, token, body=None, timeout=240):
        assert token == "synthetic-env-token"
        if body is None:
            return {"data": [{"id": "model", "capabilities": {"vision": True}}]}
        assert body["spec"]["media"] == []
        dump.mkdir()
        (dump / "manifest.jsonl").write_text("")
        return {"server": {"model": "model", "family": "mimo_v2"}, "probe": {
            "engine": "mimo", "cold": True, "no_speculation": True, "cached_tokens": 0,
            "score_path": "decode", "prompt_ids": [9, 2], "scored": 1,
            "rows": [{"position": 1, "finite": True}]}}
    monkeypatch.setattr(module, "http", http)
    args = SimpleNamespace(out=tmp_path / "out", url="http://localhost", bench_token=None,
        mode="native", features=None, quick=False, media_root=tmp_path,
        host_dump=tmp_path / "host", server_dump=tmp_path / "server", timeout=10)
    module.capture(args)
    assert json.loads((args.out / "capture.json").read_text())["windows"][0]["id"] == "control"
    assert "synthetic-env-token" not in (args.out / "capture.json").read_text()


def test_bench_token_sources_are_private_and_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTEAFD_API_KEY", "synthetic-env-token")
    assert module.probe_token() == "synthetic-env-token"
    assert module.probe_token("synthetic-explicit") == "synthetic-explicit"
    key = tmp_path / "synthetic-token"
    with pytest.raises(ValueError, match="cannot read bench token file"):
        module.probe_token(token_file=key)
    key.write_text(" synthetic-file-token\n")
    assert module.probe_token(token_file=key) == "synthetic-file-token"
    with pytest.raises(ValueError, match="choose one"):
        module.probe_token("synthetic-explicit", key)
    for value in ["", "synthetic secret", "synthetic\nsecret", "synthetic\x7fsecret", "synthetic-é"]:
        key.write_text(value)
        with pytest.raises(ValueError) as error:
            module.probe_token(token_file=key)
        assert "synthetic" not in str(error.value)
    key.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="cannot read bench token file"):
        module.probe_token(token_file=key)


def test_capture_cli_accepts_token_file_without_argv_secret(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["media-paired-probe.py", "capture", "--windows", "w",
        "--media-root", "m", "--out", "o", "--host-dump", "h", "--server-dump", "s",
        "--mode", "native", "--url", "http://localhost", "--bench-token-file", str(tmp_path / "key")])
    seen = []
    monkeypatch.setattr(module, "capture", seen.append)
    module.main()
    assert seen[0].bench_token is None
    assert seen[0].bench_token_file == tmp_path / "key"


def test_input_binding_keeps_set_identity_and_rejects_wrong_sidecars(tmp_path):
    span = {"key": "a" * 64}
    panel = {"set_sha256": "b" * 64, "windows": [{"media": [span]}]}
    sidecar = {"schema": "cuteafd.media.input-binding/1", "set_sha256": panel["set_sha256"],
               "images": {span["key"]: {"encoder": "c" * 64, "processor": "d" * 64}}}
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(sidecar))
    bound, provenance = module.bound_panel(panel, path)
    assert bound["set_sha256"] == panel["set_sha256"] and "identity" not in span
    assert bound["windows"][0]["media"][0]["identity"] == sidecar["images"][span["key"]]
    assert provenance["document"] == sidecar and len(provenance["sha256"]) == 64
    sidecar["set_sha256"] = "e" * 64
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="set identity"):
        module.bound_panel(panel, path)


@pytest.mark.parametrize("quick", [False, True])
def test_compare_rebound_capture_uses_bound_windows_and_sealed_admission(tmp_path, monkeypatch, quick):
    import hashlib
    from fidelity_windows import canonical
    args, panel = compare_case(tmp_path, monkeypatch, "mimo_v2", quick=quick)
    sidecar = {"schema": "cuteafd.media.input-binding/1", "set_sha256": panel["set_sha256"],
               "images": {"a" * 64: {"encoder": "b" * 64, "processor": "c" * 64}}}
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(canonical(sidecar))
    _, binding = module.bound_panel(panel, binding_path)
    manifests = []
    for root in (args.native, args.reference):
        path = root / "capture.json"
        capture = json.loads(path.read_text())
        capture["input_binding"] = binding
        for entry in capture["windows"]:
            response_path = root / (entry["id"] + ".json")
            response = json.loads(response_path.read_text())
            for span in response["probe"]["media"]:
                span.update(key="d" * 64, input_key="a" * 64,
                            identity={"encoder": "e" * 64, "processor": "c" * 64})
            response_path.write_bytes(canonical(response))
            entry["response_sha256"] = hashlib.sha256(response_path.read_bytes()).hexdigest()
            entry["admitted_media"] = response["probe"]["media"]
        path.write_bytes(canonical(capture))
        manifests.append(capture)
    assert module.compare(args)
    report = json.loads(args.out.read_text())
    assert report["input_bindings"] == [binding, binding]
    assert report["admitted_media"][0][0]["media"][0]["identity"]["encoder"] == "e" * 64
    assert [w["id"] for w in report["windows"]] == (["w1", "w2"] if quick else ["w0", "w1"])
    if quick:
        assert not (args.native / "w0").exists()
        assert not (args.reference / "w0").exists()
    for mutation in ("missing", "encoder", "processor", "key", "response"):
        capture = json.loads(json.dumps(manifests[0]))
        admitted = capture["windows"][0]["admitted_media"]
        if mutation == "missing":
            admitted.clear()
        elif mutation == "encoder":
            admitted[0]["identity"].pop("encoder")
        elif mutation == "processor":
            admitted[0]["identity"]["processor"] = "f" * 64
        elif mutation == "key":
            admitted[0]["key"] = "invalid"
        else:
            admitted[0]["identity"]["encoder"] = "f" * 64
        (args.native / "capture.json").write_bytes(canonical(capture))
        args.out = tmp_path / (mutation + ".json")
        with pytest.raises(ValueError, match="media|identity"):
            module.compare(args)


def test_rebound_capture_requires_verified_input_and_admitted_encoder():
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2],
            "identity": {"encoder": "b" * 64, "processor": "c" * 64}}
    window = {"tokens": [9, 2], "score_from": 1, "media": [span]}
    actual = {**span, "key": "d" * 64, "identity": {"encoder": "e" * 64, "processor": "c" * 64},
              "input_key": span["key"]}
    record = {"engine": "glm5", "cold": True, "no_speculation": True, "cached_tokens": 0,
              "score_path": "decode", "prompt_ids": [9, 2], "media": [actual],
              "rows": [{"position": 1, "finite": True}], "scored": 1}
    response = {"probe": record, "server": {"model": "model"}}
    module.check_record(window, response, "model", "native")
    for field, value in [("input_key", "f" * 64), ("identity", {"processor": "c" * 64})]:
        broken = json.loads(json.dumps(response)); broken["probe"]["media"][0][field] = value
        with pytest.raises(ValueError, match="identity"):
            module.check_record(window, broken, "model", "native")
