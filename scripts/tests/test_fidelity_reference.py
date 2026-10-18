"""Schema 2 row selection, compact serialization, and legacy equality gates."""
import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import (canonical, finish_golden, load_set, set_hash, validate_set,
                              write_scored_logits, verify_snapshot, prefix_comparison,
                              qualify_prefix, validate_qualification, CheckpointStorage,
                              release_checkpoint)

spec = importlib.util.spec_from_file_location("make_fidelity_reference", ROOT / "scripts/bench/make-fidelity-reference.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


@pytest.mark.parametrize("tail_logit", [-25.0, -100.0])
def test_compact_tail_retains_mass_below_top_probability_resolution(tail_logit):
    values = np.array([0.0, tail_logit, tail_logit], dtype=np.float64)
    logp = values - np.log(np.exp(values).sum())
    expected = max(tail_logit + np.log(2) - np.log(np.exp(values).sum()), np.log(1e-30))
    assert converter.tail_log_mass(logp, np.array([0])) == pytest.approx(expected, abs=1e-12)
    assert converter.tail_log_mass(logp, np.arange(3)) == pytest.approx(np.log(1e-30))


def test_dense_golden_requires_pinned_imported_source():
    spec = importlib.util.spec_from_file_location("dense_golden", ROOT / "python/reference/families/qwen3_5/golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    source = (ROOT / "third_party/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py").read_bytes()
    pin = golden.verify_reference_source(source)
    assert pin["revision"] == json.loads((ROOT / "third_party/transformers.lock.json").read_text())["revision"]
    assert pin["modeling_sha256"] == hashlib.sha256(source).hexdigest()
    with pytest.raises(ValueError, match="differs from pinned source"):
        golden.verify_reference_source(source + b"\n# drift\n")


def test_reference_only_validator_seals_and_rejects_mutation(tmp_path):
    spec = importlib.util.spec_from_file_location("validate_dataset", ROOT / "scripts/bench/validate-fidelity-dataset.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    window = dict(id="w", block="E", bucket="0-2K", tokens=[0] * 576,
                  roles=["ctx"] * 576, score_from=64)
    identity = dict(snapshot_revision="abcdef", checkpoint="official", precision="BF16")
    manifest = dict(schema="cuteafd.fidelity.set/1", family="qwen3_5", checkpoint="official",
                    model="official", generation_arm={"checkpoint": "official"}, reference_snapshot=identity,
                    windows=[window], quick_windows=["w"])
    manifest["set_sha256"] = set_hash(manifest)
    proof = dict(schema="cuteafd.fidelity.prefix/1", family="qwen3_5", set_sha256=manifest["set_sha256"],
                 snapshot_identity=identity, passed=True, finite=True, rows=512, different_rows=[],
                 argmax_disagreements=0, lengths=[576, 640], score_from=64, fixed_rows=128, vocab=2)
    logits = np.zeros((512, 2), dtype=np.float32)
    golden = tmp_path / "golden"
    entry = write_scored_logits(golden, window, logits)
    entry.update(nll=float(np.log(2)), scored_tokens=512)
    folder = golden / entry["path"]
    (folder / "meta.json").write_bytes(canonical(entry))
    # Keep raw common-prefix evidence, not just a self-attested pass flag.
    source = {**window, "tokens": [0] * 640, "roles": ["ctx"] * 640}
    manifest["windows"] = [source]
    manifest["set_sha256"] = set_hash(manifest)
    proof.update(source_window="w", set_sha256=manifest["set_sha256"])
    # The main fixture also uses 576 scored rows after this extension.
    logits = np.zeros((576, 2), dtype=np.float32)
    entry = write_scored_logits(golden, source, logits)
    entry.update(nll=float(np.log(2)), scored_tokens=576)
    (folder / "meta.json").write_bytes(canonical(entry))
    probe_root = golden / "prefix-gate-test"
    probe_windows, probe_entries, hashes = [], [], []
    for length, name in zip((576, 640), ("prefix_short", "prefix_extended")):
        probe_window = {**source, "id": name, "tokens": [0] * length, "roles": ["ctx"] * length}
        probe_entry = write_scored_logits(probe_root, probe_window, np.zeros((length - 64, 2), dtype=np.float32))
        probe_windows.append(probe_window)
        probe_entries.append(probe_entry)
        hashes.append(hashlib.sha256((probe_root / probe_entry["path"] / "logits.bin").read_bytes()).hexdigest())
    panel = {**manifest, "windows": probe_windows, "quick_windows": [w["id"] for w in probe_windows]}
    panel["set_sha256"] = set_hash(panel)
    finish_golden(probe_root, panel, probe_entries, snapshot_identity=identity)
    proof["logits_sha256"] = hashes
    finish_golden(golden, manifest, [entry], snapshot_identity=identity, prefix_qualification=proof,
                  mtp=False, use_cache=False, attention="eager", nll=float(np.log(2)), scored_tokens=576)
    args = SimpleNamespace(windows=golden / "windows.json", golden=golden, rows_dir=tmp_path / "full",
                           out=tmp_path / "reference.json", top_k=2, model=["official"], name=None,
                           kl_max=None, top1_min=None, quick_out=None)
    converter.convert_windows(args)
    result = validator.validate_reference_only(tmp_path, seal=True)
    assert result["passed"] and not result["engine_parity"]
    assert result["scored_tokens"] == 576
    assert validator.validate_reference_only(tmp_path) == result
    compact_path = tmp_path / "reference.json"
    original = compact_path.read_bytes()
    compact = json.loads(original)
    compact["windows"][0]["positions"][0]["top"][0]["lp"] += 0.01
    compact_path.write_bytes(canonical(compact))
    with pytest.raises(ValueError, match="top probabilities"):
        validator.validate_reference_only(tmp_path)
    compact_path.write_bytes(original)
    compact = json.loads(original)
    compact["windows"][0]["positions"][0]["tail_lp"] += 0.01
    compact_path.write_bytes(canonical(compact))
    with pytest.raises(ValueError, match="tail probability"):
        validator.validate_reference_only(tmp_path)
    compact_path.write_bytes(original)
    with pytest.raises(ValueError, match="overwrite"):
        validator.validate_reference_only(tmp_path, seal=True)
    (folder / "tokens.bin").write_bytes(np.ones(576, dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="tokens differ"):
        validator.validate_reference_only(tmp_path)


def tiny_set():
    windows = []
    for name in ("legacy", "a00"):
        windows.append({"id": name, "block": "E" if name == "legacy" else "A", "bucket": "0-2K",
                        "tokens": [1, 2, 3, 4, 5], "roles": ["ctx", "ctx", "gen", "gen", "ctx"], "score_from": 2})
    manifest = {"schema": "cuteafd.fidelity.set/1", "family": "deepseek_v41", "model": "test-model",
                "checkpoint": "test-model", "generation_arm": {"head": "bf16"},
                "quick_windows": ["legacy"], "windows": windows}
    manifest["set_sha256"] = set_hash(manifest)
    return manifest


def fixture_proof(manifest, identity=None, vocab=16):
    # Synthetic qualification metadata for converter tests, not hardware evidence.
    return {"schema": "cuteafd.fidelity.prefix/1", "passed": True, "finite": True,
            "rows": 512, "vocab": vocab, "different_rows": [], "argmax_disagreements": 0,
            "lengths": [576, 640], "score_from": 64, "fixed_rows": 128,
            "family": manifest["family"], "set_sha256": manifest["set_sha256"],
            "snapshot_identity": identity or {"snapshot_revision": "fixture"}}


def fixture_golden(tmp_path):
    manifest = tiny_set()
    golden = tmp_path / "golden"
    golden.mkdir()
    logits = np.arange(48, dtype=np.float32).reshape(3, 16) / 16
    logits[0, 0] = logits[0, 15]  # Deterministic top-k boundary tie handling.
    rows = [write_scored_logits(golden, w, logits) for w in manifest["windows"]]
    finish_golden(golden, manifest, rows, snapshot="fixture", reference="fake",
                  snapshot_identity={"snapshot_revision": "fixture"}, prefix_qualification=fixture_proof(manifest))
    return manifest, golden, logits


def options(tmp_path, golden):
    return SimpleNamespace(windows=golden / "windows.json", golden=golden,
        rows_dir=tmp_path / "rows", out=tmp_path / "reference.json", quick_out=tmp_path / "quick.json",
        top_k=12, name=None, model=["test-model"], kl_max=None, top1_min=None)


@pytest.mark.parametrize("text", ["hf_" + "x" * 24, "github_pat_" + "x" * 24, "sk-" + "x" * 24,
    "-----BEGIN OPENSSH PRIVATE KEY-----", "api_key=" + "x" * 24, "tpurtell"])
def test_public_text_keeps_credentials_keys_and_personal_data_blocked(text):
    from fidelity_windows import validate_public_text
    with pytest.raises(ValueError, match="publication blocker"):
        validate_public_text(text, scored_text=True)


def test_dataset_finalizer_requires_reference_not_engine_evidence(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/bench/finalize-fidelity-dataset.py")]
    for name in ("source", "out", "tokenizer", "license"):
        command += ["--" + name, str(tmp_path / name)]
    command += ["--config", "fixture"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "windows.json" in result.stderr
    assert not (tmp_path / "out").exists()
    for removed in ("--arms", "--comparison-policy-commit", "--validator-source-commit"):
        assert removed not in subprocess.check_output(command[:2] + ["--help"], text=True)


def test_dataset_finalizer_preserves_only_sealed_official_licence_contacts(tmp_path):
    from fidelity_windows import validate_public_metadata, validate_public_text
    tree = ast.parse((ROOT / "scripts/bench/finalize-fidelity-dataset.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "validate_output_file")
    namespace = {"digest": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
                 "json": json, "validate_public_metadata": validate_public_metadata,
                 "validate_public_text": validate_public_text}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "output-privacy", "exec"), namespace)
    check = namespace["validate_output_file"]
    licence = tmp_path / "CHECKPOINT_LICENSE"
    official = "Official licence contact: model-business@notice.qwencloud.com\n"
    licence.write_text(official)
    entry = {"path": licence.name, "sha256": hashlib.sha256(licence.read_bytes()).hexdigest()}
    manifest = {"licence_files": [entry]}
    check(licence, tmp_path, manifest)
    assert licence.read_text() == official
    licence.write_text(official + "tampered\n")
    with pytest.raises(AssertionError, match="Licence checksum"):
        check(licence, tmp_path, manifest)
    licence.write_text(official)
    check(licence, tmp_path, {"licence_files": []})
    ordinary = tmp_path / "LICENSE"
    ordinary.write_text(official)
    check(ordinary, tmp_path, {"licence_files": [{"path": ordinary.name,
          "sha256": hashlib.sha256(ordinary.read_bytes()).hexdigest()}]})


def finalizer_media_checks():
    from fidelity_windows import validate_public_metadata, validate_public_text
    tree = ast.parse((ROOT / "scripts/bench/finalize-fidelity-dataset.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("validate_media_input", "validate_output_file")]
    namespace = {"Path": pathlib.Path, "json": json,
                 "digest": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
                 "validate_public_metadata": validate_public_metadata, "validate_public_text": validate_public_text}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "media-output-privacy", "exec"), namespace)
    return namespace


def png_fixture(metadata=None):
    import struct
    import zlib
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    if metadata is not None:
        data += chunk(metadata, b"Comment\0/home/private-user/private-file")
    return data + chunk(b"IDAT", zlib.compress(b"\0\0\0\0")) + chunk(b"IEND", b"")


@pytest.mark.parametrize("suffix, payload", [(".bf16", b"\xff\xff"), (".png", png_fixture())])
def test_dataset_finalizer_accepts_only_matching_declared_binary_media(tmp_path, suffix, payload):
    checks = finalizer_media_checks()
    path = tmp_path / ("input" + suffix)
    path.write_bytes(payload)
    entry = dict(path=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    manifest = dict(media_inputs=[entry])
    checks["validate_media_input"](tmp_path, entry)
    checks["validate_output_file"](path, tmp_path, manifest)
    with pytest.raises(UnicodeDecodeError):
        checks["validate_output_file"](path, tmp_path, {})
    entry["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="size/checksum"):
        checks["validate_output_file"](path, tmp_path, manifest)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    entry["bytes"] += 1
    with pytest.raises(AssertionError, match="size/checksum"):
        checks["validate_media_input"](tmp_path, entry)


@pytest.mark.parametrize("metadata", [b"tEXt", b"iTXt", b"zTXt", b"eXIf"])
def test_dataset_finalizer_rejects_png_privacy_metadata(tmp_path, metadata):
    payload = png_fixture(metadata)
    path = tmp_path / "private.png"
    path.write_bytes(payload)
    entry = dict(path=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(AssertionError, match="privacy metadata"):
        finalizer_media_checks()["validate_output_file"](path, tmp_path, dict(media_inputs=[entry]))


@pytest.mark.parametrize("relative", ["../escape.bf16", "/absolute.bf16"])
def test_dataset_finalizer_rejects_unsafe_media_paths(tmp_path, relative):
    with pytest.raises(AssertionError, match="Unsafe media input"):
        finalizer_media_checks()["validate_media_input"](tmp_path, dict(path=relative, bytes=2, sha256="0" * 64))


def test_public_text_allows_public_fabric_without_mutating_tokens():
    from fidelity_windows import validate_public_text
    text = "ostrich dodo emu kiwi rhea moa raptor sparknest 10.55.1.22"
    assert validate_public_text(text, scored_text=True) is None


def test_checkpoint_retirement_drains_before_releasing_handles(monkeypatch):
    import fidelity_windows as storage
    events = []

    class Files(dict):
        def clear(self):
            events.append("clear")
            super().clear()

    reader = SimpleNamespace(files=Files(shard=SimpleNamespace(
        __exit__=lambda *_args: events.append("close"))))
    cuda = SimpleNamespace(synchronize=lambda: events.append("drain"),
                           empty_cache=lambda: events.append("empty"))
    monkeypatch.setattr(storage.gc, "collect", lambda: events.append("gc"))
    release_checkpoint(cuda, reader)
    assert events == ["drain", "close", "clear", "gc", "empty"]
    assert not reader.files


def test_checkpoint_rss_guard_fails_closed_and_does_not_ratchet(monkeypatch):
    import fidelity_windows as storage
    rss = iter([2 * 2**30, 2 * 2**30, 3 * 2**30, 4 * 2**30 + 1])
    monkeypatch.setattr(storage, "rss_bytes", lambda: next(rss))
    memory = CheckpointStorage(None, max_rss_gib=80, max_growth_gib=2)
    assert memory.check("layer 0") == 3 * 2**30
    with pytest.raises(RuntimeError, match="RSS bound exceeded after layer 1"):
        memory.check("layer 1")


def test_checkpoint_read_log_labels_logical_bytes_and_time(monkeypatch, capsys):
    import fidelity_windows as storage
    monkeypatch.setattr(storage.time, "monotonic", lambda: 12.0)
    readers = (SimpleNamespace(read_bytes=4_000_000, read_seconds=0.5),
               SimpleNamespace(read_bytes=2_000_000, read_seconds=0.25))
    storage.log_checkpoint_reads("layer 1 load", readers, 2_000_000, 10.0)
    output = capsys.readouterr().out
    assert "checkpoint reads layer 1 load: utc=" in output
    assert "bytes=4000000 elapsed=2.000s MB/s=2.000" in output
    assert "cumulative_read_seconds=0.750" in output


@pytest.mark.parametrize("family", ["deepseek_v4", "deepseek_v41", "glm5", "glm5_flash",
                                   "mimo_v2/mimo_v2", "mimo_v2/mimo_v26", "qwen4"])
def test_every_golden_retires_layer_storage_and_owns_cpu_reads(family):
    tree = ast.parse((ROOT / "python/reference/families" / family / "golden.py").read_text())
    reader = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Weights")
    assert ".clone()" in ast.unparse(reader)
    for function in (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("main", "run_windows")):
        source = ast.unparse(function)
        assert "CheckpointStorage(" in source
        assert "memory.release()" in source and "memory.check(" in source
    assert "release_checkpoint(" in ast.unparse(tree)


def test_schema2_scored_rows_and_full_manifest(tmp_path):
    manifest, golden, logits = fixture_golden(tmp_path)
    args = options(tmp_path, golden)
    converter.convert_windows(args)
    reference = json.loads(args.out.read_text())
    assert reference["schema"] == "cuteafd.fidelity.reference/2"
    assert reference["set_sha256"] == manifest["set_sha256"]
    assert reference["checkpoint"] == "test-model"
    assert reference["expect"] == {"kl_max": 0.06, "top1_min": 0.90}
    assert [w["id"] for w in json.loads(args.quick_out.read_text())["windows"]] == ["legacy"]
    rows = json.loads((args.rows_dir / "rows.json").read_text())
    assert rows["dtype"] == "<f2" and rows["kind"] == "log_softmax"
    for entry, window in zip(rows["windows"], reference["windows"]):
        path = args.rows_dir / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["shape"] == [3, 16] and entry["positions"] == [2, 3, 4]
        full = np.fromfile(path, dtype="<f2").reshape(3, 16)
        exact = logits.astype(np.float64)
        exact -= np.log(np.exp(exact).sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(full, exact.astype("<f2"))
        for pos, compact in enumerate(window["positions"]):
            assert compact["pos"] == pos + 2 and compact["next"] == manifest["windows"][0]["tokens"][pos + 2]
            assert compact["next_lp"] == round(float(exact[pos, compact["next"]]), 5)
            assert compact["top"][0]["id"] == (0 if pos == 0 else 15)


@pytest.mark.parametrize("change", ["hash", "checkpoint", "tokens", "positions", "nonfinite", "duplicate", "missing", "extra"])
def test_converter_refuses_mismatched_or_corrupt_goldens(tmp_path, change):
    manifest, golden, logits = fixture_golden(tmp_path)
    meta = json.loads((golden / "meta.json").read_text())
    if change in ("hash", "checkpoint"):
        meta["set_sha256" if change == "hash" else "checkpoint"] = "wrong"
    elif change == "duplicate":
        meta["windows"].append(dict(meta["windows"][0]))
    elif change == "missing":
        meta["windows"].pop()
    elif change == "extra":
        meta["windows"].append({**meta["windows"][0], "id": "extra"})
    elif change == "positions":
        meta["windows"][0]["positions"] = [1, 2, 3]
    elif change == "tokens":
        np.array([1, 2, 7, 4, 5], dtype="<i4").tofile(golden / "windows/legacy/tokens.bin")
    else:
        logits[0, 0] = np.nan
        logits.tofile(golden / "windows/legacy/logits.bin")
    (golden / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        converter.convert_windows(options(tmp_path, golden))


def test_legacy_output_is_byte_identical_to_base_script(tmp_path, monkeypatch):
    golden = tmp_path / "legacy-golden"
    golden.mkdir()
    rng = np.random.default_rng(72)
    tokens = rng.integers(0, 64, 514, dtype=np.int32)
    tokens.tofile(golden / "tokens.bin")
    logits = rng.normal(size=(514, 64)).astype(np.float32)
    logits.tofile(golden / "logits.bin")
    (golden / "meta.json").write_text(json.dumps({"snapshot": str(tmp_path / "snapshot"), "reference": "synthetic"}))
    original = subprocess.check_output(["git", "-C", str(ROOT), "show",
        "7e5f769:scripts/bench/make-fidelity-reference.py"], text=True)
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    monkeypatch.setattr(sys, "argv", ["make", "--golden", str(golden), "--model", "test", "--out", str(before)])
    exec(compile(original, "original_make_reference.py", "exec"), {"__name__": "__main__"})
    monkeypatch.setattr(sys, "argv", ["make", "--golden", str(golden), "--model", "test", "--out", str(after)])
    converter.main()
    assert before.read_bytes() == after.read_bytes()
    # Schema2 --top-k 12 reproduces all legacy compact fields to five decimals.
    legacy = json.loads(after.read_text())
    manifest = tiny_set()
    manifest["windows"] = [{"id": "legacy", "block": "E", "bucket": "0-2K", "tokens": legacy["tokens"],
        "roles": ["ctx"] * len(legacy["tokens"]), "score_from": 1}]
    manifest["set_sha256"] = set_hash(manifest)
    rows = [write_scored_logits(golden, manifest["windows"][0], logits[:512])]
    finish_golden(golden, manifest, rows, snapshot=str(tmp_path / "snapshot"),
                  snapshot_identity={"snapshot_revision": "fixture"}, prefix_qualification=fixture_proof(manifest, vocab=64))
    args = options(tmp_path, golden)
    converter.convert_windows(args)
    window = json.loads(args.out.read_text())["windows"][0]
    assert [[t["id"] for t in p["top"]] for p in window["positions"]] == legacy["ids"]
    assert [[t["lp"] for t in p["top"]] for p in window["positions"]] == legacy["lps"]
    assert [p["tail_lp"] for p in window["positions"]] == legacy["tail_lp"]
    assert [p["next_lp"] for p in window["positions"]] == legacy["next_lp"]


def test_qualified_scored_window_regenerates_schema1_without_fake_rows(tmp_path, monkeypatch):
    manifest, golden, _ = fixture_golden(tmp_path)
    args = options(tmp_path, golden)
    converter.convert_windows(args)
    window = json.loads(args.out.read_text())["windows"][0]
    out = tmp_path / "qualified-legacy.json"
    monkeypatch.setattr(sys, "argv", ["make", "--golden", str(golden), "--legacy-window", "legacy",
        "--model", "test-model", "--out", str(out)])
    converter.main()
    legacy = json.loads(out.read_text())
    assert legacy["schema"] == "cuteafd.bench.reference/1"
    assert legacy["tokens"] == manifest["windows"][0]["tokens"]
    assert legacy["score_from"] == 2
    assert legacy["ids"] == [[t["id"] for t in p["top"]] for p in window["positions"]]
    assert legacy["lps"] == [[t["lp"] for t in p["top"]] for p in window["positions"]]
    assert legacy["tail_lp"] == [p["tail_lp"] for p in window["positions"]]
    assert legacy["next_lp"] == [p["next_lp"] for p in window["positions"]]
    assert legacy["source"]["prefix_qualification"]["set_sha256"] == manifest["set_sha256"]


@pytest.mark.parametrize("change", ["proof", "positions", "tokens", "extent", "nonfinite", "duplicate", "path"])
def test_qualified_legacy_conversion_fails_closed(tmp_path, monkeypatch, change):
    _, golden, logits = fixture_golden(tmp_path)
    meta = json.loads((golden / "meta.json").read_text())
    if change == "proof":
        meta["prefix_qualification"]["passed"] = False
    elif change == "positions":
        meta["windows"][0]["positions"] = [1, 2, 3]
    elif change == "tokens":
        np.array([1, 2, 7, 4, 5], dtype="<i4").tofile(golden / "windows/legacy/tokens.bin")
    elif change == "extent":
        logits[:2].tofile(golden / "windows/legacy/logits.bin")
    elif change == "nonfinite":
        logits[0, 0] = np.nan
        logits.tofile(golden / "windows/legacy/logits.bin")
    elif change == "duplicate":
        meta["windows"].append(dict(meta["windows"][0]))
    else:
        meta["windows"][0]["path"] = "../escape"
    (golden / "meta.json").write_text(json.dumps(meta))
    out = tmp_path / "legacy.json"
    monkeypatch.setattr(sys, "argv", ["make", "--golden", str(golden), "--legacy-window", "legacy",
        "--model", "test-model", "--out", str(out)])
    with pytest.raises(ValueError):
        converter.main()
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["length", "mask", "id", "bucket", "duplicate", "quick"])
def test_window_contract_failures(mutation):
    m = tiny_set()
    w = m["windows"][0]
    if mutation == "length":
        w["tokens"] = [0] * 16385
        w["roles"] = ["ctx"] * 16385
    elif mutation == "mask":
        w["roles"][0] = "assistant"
    elif mutation == "id":
        w["id"] = "../escape"
    elif mutation == "bucket":
        w["bucket"] = "8-16K"
    elif mutation == "duplicate":
        m["windows"].append(w)
    else:
        m["quick_windows"] = ["absent"]
    m["set_sha256"] = set_hash(m)
    with pytest.raises(ValueError):
        validate_set(m)


@pytest.mark.parametrize("change", ["missing", "failed", "nonfinite", "family", "hash", "snapshot", "signed_zero"])
def test_prefix_qualification_fails_closed(tmp_path, change):
    manifest, golden, _ = fixture_golden(tmp_path)
    meta = json.loads((golden / "meta.json").read_text())
    proof = meta["prefix_qualification"]
    if change == "missing":
        del meta["prefix_qualification"]
    elif change == "failed":
        proof["passed"] = False
    elif change == "nonfinite":
        proof["finite"] = False
    elif change == "family":
        proof["family"] = "other"
    elif change == "hash":
        proof["set_sha256"] = "other"
    elif change == "snapshot":
        proof["snapshot_identity"] = {"snapshot_revision": "other"}
    else:
        left = np.zeros((512, 16), dtype=np.float32)
        right = left.copy()
        right[10, 3] = -0.0
        proof.update(prefix_comparison(left, right))
    (golden / "meta.json").write_bytes(canonical(meta))
    args = options(tmp_path, golden)
    with pytest.raises(ValueError, match="prefix"):
        converter.convert_windows(args)
    assert not args.out.exists() and not args.rows_dir.exists()


@pytest.mark.parametrize("bad", [False, True])
def test_actual_prefix_runner_compares_full_rows_and_binds_set(tmp_path, bad):
    manifest = tiny_set()
    window = manifest["windows"][0]
    window["tokens"], window["roles"] = [i % 16 for i in range(640)], ["ctx"] * 640
    manifest["set_sha256"] = set_hash(manifest)
    identity = {"snapshot_revision": "fixture"}
    a = SimpleNamespace(out=tmp_path, layers=[0])

    def execute(probe):
        assert probe._prefix_probe and probe.layers is None
        panel = load_set(probe.windows)
        entries = []
        for w in panel["windows"]:
            count = len(w["tokens"]) - w["score_from"]
            logits = np.arange(count * 16, dtype=np.float32).reshape(count, 16)
            if bad and w["id"] == "prefix_extended":
                logits[3, 2] += .125  # Same argmax, different non-top probability.
            entries.append(write_scored_logits(probe.out, w, logits))
        finish_golden(probe.out, panel, entries, snapshot_identity=identity, seconds=1.0)

    if bad:
        with pytest.raises(ValueError, match="prefix invariance"):
            qualify_prefix(a, manifest, execute)
    else:
        proof = qualify_prefix(a, manifest, execute)
        validate_qualification(proof, manifest, identity)
        assert proof["vocab"] == 16 and proof["source_window"] == window["id"]
    assert a.layers == [0] and not hasattr(a, "_prefix_probe")
    saved = json.loads(next(tmp_path.glob("prefix-gate-*/qualification.json")).read_text())
    assert saved["passed"] is not bad
    assert saved["different_rows"] == ([3] if bad else [])


def test_prefix_comparison_rejects_nonfinite_logits():
    rows = np.ones((512, 4), dtype=np.float32)
    for value in (np.nan, np.inf, -np.inf):
        rows[3, 2] = value
        proof = prefix_comparison(rows, rows.copy())
        assert not proof["passed"] and not proof["finite"]


def test_snapshot_provenance_must_match_pinned_set(tmp_path):
    snapshot = tmp_path / "revision"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "config.json").write_text('{"layers":70}')
    manifest = tiny_set()
    identity = verify_snapshot(manifest, snapshot)
    manifest["tokenizer_sha256"] = identity["tokenizer_sha256"]
    manifest["generation_arm"]["snapshot_revision"] = "revision"
    assert verify_snapshot(manifest, snapshot) == identity
    manifest["tokenizer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="tokenizer"):
        verify_snapshot(manifest, snapshot)
    manifest["tokenizer_sha256"] = identity["tokenizer_sha256"]
    manifest["generation_arm"]["snapshot_revision"] = "other"
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(manifest, snapshot)


def test_glm_kda_grouping_preserves_all_heads_and_recurrent_state():
    tree = ast.parse((ROOT / "python/reference/families/glm5_flash/golden.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bounded_kda")
    scope = {"torch": SimpleNamespace(cat=lambda values, dim: np.concatenate(values, axis=dim))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "bounded_kda", "exec"), scope)
    calls = []
    def official(query, key, value, g, beta, **kwargs):
        calls.append((query.shape[2], kwargs["chunk_size"], kwargs["use_qk_l2norm_in_kernel"]))
        assert key.shape == value.shape == g.shape == query.shape
        assert beta.shape == query.shape[:-1]
        return query + key + value, kwargs["initial_state"] if kwargs["output_final_state"] else None
    data = np.arange(1 * 67 * 7 * 8).reshape(1, 67, 7, 8)
    state = np.arange(1 * 7 * 8 * 8).reshape(1, 7, 8, 8)
    grouped = scope["bounded_kda"](official, heads=3)
    output, final = grouped(data, data, data, data, data[..., 0], initial_state=state,
                            output_final_state=True, use_qk_l2norm_in_kernel=True)
    np.testing.assert_array_equal(output, data * 3)
    np.testing.assert_array_equal(final, state)
    assert calls == [(3, 64, True), (3, 64, True), (1, 64, True)]
    assert grouped(data, data, data, data, data[..., 0])[1] is None


def test_eager_query_blocks_keep_keys_masks_and_selected_rows():
    tree = ast.parse((ROOT / "python/reference/shape_invariant.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("bounded_eager", "bounded_sparse")]
    scope = {"torch": SimpleNamespace(cat=lambda values, dim: np.concatenate(values, axis=dim))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "bounded_attention", "exec"), scope)
    q = np.arange(1 * 3 * 7 * 2).reshape(1, 3, 7, 2)
    kv = np.zeros((1, 3, 11, 2))
    mask = np.arange(7 * 11).reshape(1, 1, 7, 11)
    selected = np.arange(1 * 7 * 4 * 2).reshape(1, 7, 4, 2)
    calls = []
    def official(module, query, key, value, attention_mask, **kwargs):
        assert key is kv and value is kv
        calls.append((query.shape[-2], attention_mask.copy(), kwargs["selected_kv"].copy()))
        return query.transpose(0, 2, 1, 3), np.ones((1, 3, query.shape[-2], 11))
    module = SimpleNamespace(training=False, attention_dropout=.3,
                             config=SimpleNamespace(attention_dropout=.3))
    wrapped = scope["bounded_eager"](official, rows=3)
    output, weights = wrapped(module, q, kv, kv, mask, .25, selected_kv=selected,
                              selected_valid=np.ones((1, 7, 4)))
    assert weights is None
    np.testing.assert_array_equal(output, q.transpose(0, 2, 1, 3))
    assert [c[0] for c in calls] == [3, 3, 1]
    np.testing.assert_array_equal(np.concatenate([c[1] for c in calls], axis=2), mask)
    np.testing.assert_array_equal(np.concatenate([c[2] for c in calls], axis=1), selected)
    eval_output, _ = wrapped(module, q, kv, kv, mask, .25, dropout=.1,
                             selected_kv=selected, selected_valid=np.ones((1, 7, 4)))
    np.testing.assert_array_equal(eval_output, output)
    module.training = True
    with pytest.raises(ValueError, match="inference"):
        wrapped(module, q, kv, kv, mask, .25)
    sparse_calls = []
    def sparse(query, key, sink, indices, scale):
        sparse_calls.append(query.shape[1])
        assert key is kv and scale == .25
        return query + indices[..., :1, None]
    sparse_q = q.transpose(0, 2, 1, 3)
    indices = np.arange(7).reshape(1, 7, 1)
    actual = scope["bounded_sparse"](sparse, rows=3)(sparse_q, kv, None, indices, .25)
    np.testing.assert_array_equal(actual, sparse_q + indices[..., :1, None])
    assert sparse_calls == [3, 3, 1]


def test_fixed_index_topk_masks_padding_orders_ids_and_resolves_cutoff_ties():
    tree = ast.parse((ROOT / "python/reference/shape_invariant.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "fixed_index_topk")

    class Tensor(np.ndarray):
        def argsort(self, dim=-1, descending=False, stable=False):
            assert descending and stable
            return np.argsort(-np.asarray(self), axis=dim, kind="stable").view(Tensor)

        def sort(self, dim=-1):
            return SimpleNamespace(values=np.sort(np.asarray(self), axis=dim).view(Tensor))

        def gather(self, dim, indices):
            return np.take_along_axis(self, indices, axis=dim)

    def pad(scores, extent, value):
        assert extent[0] == 0
        return np.pad(scores, ((0, 0), (0, extent[1])),
                      constant_values=value).view(Tensor)

    scope = {"F": SimpleNamespace(pad=pad)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fixed_topk", "exec"), scope)
    select = scope["fixed_index_topk"]
    short = np.array([[2., 1., -np.inf], [0., 0., -np.inf]]).view(Tensor)
    long = np.pad(short, ((0, 0), (0, 2)), constant_values=-np.inf).view(Tensor)
    a, b = select(short, 8), select(long, 8)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.indices.shape == (2, 8)
    assert np.isneginf(a.values[:, 2:]).all()
    tied = np.array([[0., 4., 4., 4., 9.]]).view(Tensor)
    np.testing.assert_array_equal(select(tied, 3).indices, [[1, 2, 4]])


def test_index_topk_adapter_changes_only_selection_and_fails_closed():
    tree = ast.parse((ROOT / "python/reference/shape_invariant.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "install_index_topk")
    calls = []
    def selection(scores, slots):
        calls.append((scores, slots))
        return (None, [0, 1, 2, 3])
    scope = {"fixed_index_topk": selection}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "install_topk", "exec"), scope)

    class Indexer:
        index_topk = 4
        def forward(self, score):
            index_score = score + 10
            return index_score.topk(min(self.index_topk, 2), dim=-1)[1]

    module = SimpleNamespace(Indexer=Indexer)
    scope["install_index_topk"](module)
    scope["install_index_topk"](module)
    assert Indexer().forward(7) == [0, 1, 2, 3]
    assert calls == [(17, 4)]

    class Unsupported:
        def forward(self, score):
            return score

    with pytest.raises(ValueError, match="unsupported official"):
        scope["install_index_topk"](SimpleNamespace(Indexer=Unsupported))


def test_v4_compressed_slots_keep_order_mask_padding_and_original_cache():
    tree = ast.parse((ROOT / "python/reference/families/deepseek_v4/golden.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install_compressed_slots")
    scope = {"torch": SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(
        pad=lambda x, extent, value: np.pad(x, [(0, 0), (0, 0), (0, extent[1])], constant_values=value))))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "compressed_slots", "exec"), scope)
    calls = []
    def original(ratio, bsz, seqlen, start_pos, offset):
        calls.append((ratio, bsz, seqlen, start_pos, offset))
        return np.arange(seqlen // ratio).reshape(1, 1, -1) + offset
    original.cache_clear = lambda: calls.append("clear")
    module = SimpleNamespace(get_compress_topk_idxs=original)
    scope["install_compressed_slots"](module, 1024)
    ids = module.get_compress_topk_idxs(128, 1, 576, 0, 576)
    assert ids.tolist() == [[[576, 577, 578, 579, -1, -1, -1, -1]]]
    scope["install_compressed_slots"](module, 640)
    assert module.get_compress_topk_idxs(128, 1, 576, 0, 576).shape[-1] == 5
    module.get_compress_topk_idxs.cache_clear()
    assert calls[-1] == "clear"
    with pytest.raises(ValueError, match="panel slot extent"):
        module.get_compress_topk_idxs(128, 1, 768, 0, 0)


def test_official_reference_identity_can_differ_from_generation_checkpoint(tmp_path):
    snapshot = tmp_path / "official-fp8"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "config.json").write_text('{"layers":48}')
    manifest = tiny_set()
    identity = verify_snapshot(manifest, snapshot)
    manifest["generation_arm"].update(snapshot_revision="exl3", config_sha256="quant-config")
    manifest["reference_snapshot"] = identity.copy()
    manifest["tokenizer_sha256"] = identity["tokenizer_sha256"]
    assert verify_snapshot(manifest, snapshot) == identity
    manifest["reference_snapshot"]["config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="config"):
        verify_snapshot(manifest, snapshot)
    manifest["reference_snapshot"] = {"snapshot_revision": snapshot.name}
    with pytest.raises(ValueError, match="all identity hashes"):
        verify_snapshot(manifest, snapshot)


def test_v41_two_lengths_isolate_module_and_cross_layer_state():
    tree = ast.parse((ROOT / "python/reference/families/deepseek_v41/golden.py").read_text())
    names = {"initial_runtime_buffers", "reset_runtime_buffers", "stage_shared_attention", "restore_shared_attention"}
    helpers = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = {}
    exec(compile(helpers, "golden_runtime_helpers", "exec"), scope)

    class Tensor:
        def __init__(self, array): self.array = np.asarray(array)
        def detach(self): return self
        def cpu(self): return self
        def clone(self): return Tensor(self.array.copy())
        def to(self, _device): return self
        def copy_(self, other): np.copyto(self.array, other.array)

    class Indexer:
        def __init__(self):
            self._buffers = {"k_cache": Tensor(np.zeros(5))}
            self._non_persistent_buffers_set = {"k_cache"}
            self.freqs_cis = "previous window"

    class Shared:
        def __init__(self):
            self.compress_kv = self.index_k = self.topk_idxs = self.candidates = None

    indexer = Indexer()
    compressor = SimpleNamespace(_buffers={"kv_state": Tensor(np.zeros(5)),
        "score_state": Tensor(np.full(5, -np.inf))}, _non_persistent_buffers_set={"kv_state", "score_state"})
    layer = SimpleNamespace(_buffers={"window_kv_cache": Tensor(np.zeros(5)),
        "freqs_cis": Tensor(np.arange(5))}, _non_persistent_buffers_set={"window_kv_cache", "freqs_cis"})
    layer.modules = lambda: [layer, indexer, compressor]
    ref = SimpleNamespace(Indexer=Indexer, SharedAttentionRuntime=Shared, shared_attn=Shared())
    initial = scope["initial_runtime_buffers"](layer)
    snapshots = []
    for length in [2, 5]:
        scope["reset_runtime_buffers"](initial, layer, ref)
        assert indexer.freqs_cis is None
        assert not indexer._buffers["k_cache"].array.any()
        assert not layer._buffers["window_kv_cache"].array.any()
        assert np.isneginf(compressor._buffers["score_state"].array).all()
        assert np.array_equal(layer._buffers["freqs_cis"].array, np.arange(5))
        indexer._buffers["k_cache"].array[:length] = length
        for name in vars(ref.shared_attn):
            setattr(ref.shared_attn, name, Tensor(indexer._buffers["k_cache"].array[:length]))
        snapshots.append(scope["stage_shared_attention"](ref.shared_attn))
        indexer._buffers["k_cache"].array[:] = -99
        compressor._buffers["score_state"].array[:] = 99
    for length, state in zip([2, 5], snapshots):
        scope["restore_shared_attention"](ref, state, "cpu")
        for value in vars(ref.shared_attn).values():
            assert value.array.shape == (length,)
            assert np.array_equal(value.array, np.full(length, length))


@pytest.mark.parametrize("family,media", [("glm5_flash", False), ("glm5_flash", True), ("glm5", False)])
def test_glm_window_dsa_handoff_is_per_window(tmp_path, monkeypatch, family, media):
    from contextlib import nullcontext
    manifest = tiny_set()
    manifest["family"] = family
    w = manifest["windows"][1]
    w["tokens"], w["roles"] = [1, 2, 3, 4, 5, 6, 7], ["ctx"] * 7
    if media:
        from test_fidelity_media import span, glm_config
        for window in manifest["windows"]:
            window.update(tokens=[1] + [9] * 4 + [2] + [3] * 634, roles=["ctx"] * 640,
                          score_from=64, media=[span()])
        (tmp_path / "config.json").write_text(json.dumps(glm_config()))
    manifest["set_sha256"] = set_hash(manifest)
    path = tmp_path / "input.json"
    path.write_bytes(canonical(manifest))
    tree = ast.parse((ROOT / "python/reference/families" / family / "golden.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_windows")
    calls = []

    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cpu(self): return Tensor(self.data.copy())
        def cuda(self): return Tensor(self.data.copy())
        def float(self): return self
        def numpy(self): return self.data.astype(np.float32)
        @property
        def shape(self): return self.data.shape
        def __getitem__(self, key): return Tensor(self.data[key])
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.data, axis))
        def repeat(self, *shape): return Tensor(np.tile(self.data, shape))
        def expand(self, *shape):
            shape = tuple(old if new == -1 else new for old, new in zip(self.data.shape, shape))
            return Tensor(np.broadcast_to(self.data, shape))
        def contiguous(self): return Tensor(self.data.copy())
        def mean(self, dim): return Tensor(self.data.mean(axis=dim))
        @property
        def dtype(self): return "bf16"
        def triu(self, diagonal): return Tensor(np.triu(self.data, diagonal))
        def to(self, _dtype): return self
        def copy_(self, other): np.copyto(self.data, other.data)

    class Layer:
        def __init__(self, _config, layer_id): self.layer_id = layer_id
        def to_empty(self, **_kwargs): return self
        def eval(self): return self
        def named_parameters(self): return []
        def named_buffers(self): return []
        def __call__(self, h, *, prev_topk_indices, **_kwargs):
            length = h.data.shape[1]
            incoming = None if prev_topk_indices is None else int(prev_topk_indices.data[0, 0, 0])
            calls.append((self.layer_id, length, incoming))
            np.testing.assert_array_equal(_kwargs["position_ids"].data, np.arange(length)[None])
            if media:
                np.testing.assert_array_equal(h.data[0, 1:5], np.full((4, 4, 1), 42))
                assert h.data[0, 0, 0, 0] == 1 and h.data[0, 5, 0, 0] == 2
            expected = length if self.layer_id == 1 else None
            assert incoming == expected
            topk = Tensor(np.full((1, length, 1), length)) if self.layer_id == 0 else None
            return h, topk

    class Norm:
        weight = Tensor([1.0])
        def cuda(self): return self
        def to(self, _dtype): return self
        def __call__(self, h): return h

    weights = SimpleNamespace(files={}, read_bytes=0, stage_layer=lambda *_args: None, staged={}, get=lambda name: Tensor(np.ones((16, 1)) if "lm_head" in name else [1.0]))
    torch = SimpleNamespace(inference_mode=nullcontext, device=lambda _name: nullcontext(),
        bfloat16="bf16", float32="f32", bool="bool", set_default_dtype=lambda _dtype: None,
        tensor=lambda data, **_kwargs: Tensor(data), arange=lambda n, **_kwargs: Tensor(np.arange(n)),
        ones=lambda *shape, **_kwargs: Tensor(np.ones(shape)),
        full=lambda shape, value, **_kwargs: Tensor(np.full(shape, value)),
        cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
        nn=SimpleNamespace(functional=SimpleNamespace(
            embedding=lambda ids, _weights: Tensor(ids.data[..., None]),
            linear=lambda x, w: Tensor(x.data @ w.data.T))))
    import time
    coordinator = {"snapshot_revision": "dense", "modeling_sha256": "a" * 64}
    expert_identity = {"snapshot_revision": "official-experts", "index_sha256": "b" * 64}
    events = []
    if media:
        import glm_flash_media
        monkeypatch.setattr(glm_flash_media, "snapshot_identity", lambda *_args: coordinator)
        def expert_source(path):
            assert path == tmp_path / "experts"
            events.append("expert-identity")
            return expert_identity
        monkeypatch.setattr(glm_flash_media, "expert_snapshot_identity", expert_source)
        def features(*_args):
            assert events == ["expert-identity", "qualify"]
            events.append("tower")
            return {"a" * 64: Tensor(np.full((4, 1), 42))}, coordinator
        monkeypatch.setattr(glm_flash_media, "window_features", features)
        identity = {**coordinator, "coordinator_tower": coordinator,
                    "experts": expert_identity, "tower_dtype": "bf16"}
        proof = fixture_proof(manifest, identity)
        proof.update(source_window=manifest["windows"][0]["id"], media=manifest["windows"][0]["media"])
    def qualification(*_args):
        if media:
            assert events == ["expert-identity"]
            events.append("qualify")
            return proof
    scope = dict(torch=torch, time=time, json=json, load_set=load_set, verify_snapshot=lambda *_args: {},
        qualify=qualification, write_scored_logits=write_scored_logits, finish_golden=finish_golden,
        PREFIX="model.language_model.", FP32_KEYS=(), load_layer=lambda *_args: None,
        CheckpointStorage=CheckpointStorage, install_dsa=lambda *_args: None,
        log_checkpoint_reads=lambda *_args: None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "glm_window_runner", "exec"), scope)
    config = SimpleNamespace(hc_mult=4, num_hidden_layers=3, hidden_size=1, rms_norm_eps=1e-6,
                             layer_types=["dsa", "dsa", "kda"])
    ref = SimpleNamespace(Glm5NextTextDecoderLayer=Layer, Glm5NextTextRMSNorm=lambda *_args: Norm(),
        GlmMoeDsaDecoderLayer=Layer, GlmMoeDsaRMSNorm=lambda *_args: Norm(),
        GlmMoeDsaRotaryEmbedding=lambda **_kwargs: SimpleNamespace(
            cuda=lambda: lambda h, **_kw: (h, h)))
    args = SimpleNamespace(windows=path, snapshot=tmp_path, out=tmp_path, layers=None,
                           experts_snapshot=tmp_path / "experts" if media else None,
                           media=media, tower_dtype="bf16")
    if family == "glm5_flash":
        scope["run_windows"](args, config, ref, weights, weights)
    else:
        scope["run_windows"](args, config, ref, weights)
    if media:
        assert calls == [(0, 640, None)] * 2 + [(1, 640, 640)] * 2 + [(2, 640, None)] * 2
    else:
        assert calls == [(0, 5, None), (0, 7, None), (1, 5, 5), (1, 7, 7), (2, 5, None), (2, 7, None)]
    meta = json.loads((tmp_path / "meta.json").read_text())
    if media:
        assert events == ["expert-identity", "qualify", "tower"]
        assert meta["snapshot_identity"] == meta["prefix_qualification"]["snapshot_identity"] == identity
        validate_qualification(meta["prefix_qualification"], manifest, identity)
        changed = {**identity, "experts": {**expert_identity, "snapshot_revision": "other"}}
        with pytest.raises(ValueError, match="provenance"):
            validate_qualification(meta["prefix_qualification"], manifest, changed)
        assert [w["positions"] for w in meta["windows"]] == [list(range(64, 640))] * 2
        assert "not a full official-model byte-equivalence claim" in meta["reference"]
    else:
        assert [w["positions"] for w in meta["windows"]] == [list(range(2, 5)), list(range(2, 7))]


@pytest.mark.parametrize("family", ["deepseek_v41", "deepseek_v4", "mimo_v2/mimo_v26", "qwen4", "glm5_flash", "glm5"])
def test_goldens_have_layer_major_window_loops_and_scored_head_selection(family):
    path = ROOT / "python/reference/families" / family / "golden.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_windows")
    layer_loop = next(n for n in ast.walk(function) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "layer_id")
    assert any(isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) for n in ast.walk(layer_loop))
    source = ast.unparse(function)
    assert "score_from" in source and "write_scored_logits" in source and "finish_golden" in source
    assert "a.layers is not None" in source
    assert "load_set" in source


def test_dsv4_reset_preserves_compressor_buffer_aliases():
    tree = ast.parse((ROOT / "python/reference/families/deepseek_v4/golden.py").read_text())
    names = {"initial_runtime_buffers", "reset_runtime_buffers"}
    scope = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name in names], type_ignores=[]), "dsv4_buffers", "exec"), scope)

    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def detach(self): return self
        def cpu(self): return self
        def clone(self): return Tensor(self.data.copy())
        def copy_(self, other): np.copyto(self.data, other.data)

    cache = Tensor(np.zeros((7, 2)))
    scores = Tensor(np.full((4, 2), -np.inf))
    module = SimpleNamespace(_buffers={"kv_cache": cache, "score_state": scores,
        "freqs_cis": Tensor(np.arange(7))},
        _non_persistent_buffers_set={"kv_cache", "score_state", "freqs_cis"})
    module.modules = lambda: [module]
    alias = cache.data[2:]
    initial = scope["initial_runtime_buffers"](module)
    for length in (2, 7, 3):
        scope["reset_runtime_buffers"](initial)
        assert module._buffers["kv_cache"] is cache
        assert not alias.any() and np.isneginf(scores.data).all()
        np.testing.assert_array_equal(module._buffers["freqs_cis"].data, np.arange(7))
        cache.data[:length] = length
        scores.data[:] = length


def test_qwen_window_and_legacy_layers_enter_eval_after_loading():
    tree = ast.parse((ROOT / "python/reference/families/qwen4/golden.py").read_text())
    for function_name in ("run_windows", "main"):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
        source = ast.unparse(function)
        loaded = source.index('load_experts(layer.mlp.experts')
        evaluated = source.index('layer.eval()', loaded)
        executed = source.index('h = layer(', evaluated)
        assert loaded < evaluated < executed


def qwen_storage_scope():
    tree = ast.parse((ROOT / "python/reference/families/qwen4/golden.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    nodes += [n for n in tree.body if getattr(n, "name", None) in ("verify_ple_storage", "Weights")]
    import os
    scope = {"Path": pathlib.Path, "hashlib": hashlib, "json": json, "os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "qwen-storage", "exec"), scope)
    return scope


def ple_storage_fixture(tmp_path):
    snapshot = tmp_path / "pinned-revision"
    local = tmp_path / "local"
    snapshot.mkdir()
    local.mkdir()
    table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    index = {table: "table.safetensors", "lm_head.weight": "head.safetensors"}
    index_bytes = json.dumps({"weight_map": index}).encode()
    (snapshot / "model.safetensors.index.json").write_bytes(index_bytes)
    data = b"synthetic table bytes"
    (local / "table.safetensors").write_bytes(data)
    seal = {"complete": True, "snapshot_revision": snapshot.name,
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(), "total_bytes": len(data),
            "files": [{"path": "table.safetensors", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}
    (local / "seal.json").write_text(json.dumps(seal))
    return snapshot, local, index, table, seal


def test_qwen_local_ple_mapping_keeps_dense_weights_streamed(tmp_path):
    scope = qwen_storage_scope()
    snapshot, local, index, table, seal = ple_storage_fixture(tmp_path)
    weights = scope["Weights"](snapshot, local)
    assert weights.ple_path(table) == local / index[table]
    assert weights.ple_storage_seal_sha256 == hashlib.sha256((local / "seal.json").read_bytes()).hexdigest()
    assert scope["Weights"](snapshot).ple_path(table) == snapshot / index[table]
    with pytest.raises(ValueError, match="non-table"):
        weights.ple_path("lm_head.weight")
    assert "self.snapshot / shard" in ast.unparse(ast.parse((ROOT / "python/reference/families/qwen4/golden.py").read_text()))


@pytest.mark.parametrize("change", ["incomplete", "revision", "index", "bytes", "hash", "missing", "extra", "total", "symlink"])
def test_qwen_local_ple_storage_fails_closed(tmp_path, change):
    scope = qwen_storage_scope()
    snapshot, local, index, table, seal = ple_storage_fixture(tmp_path)
    if change == "incomplete": seal["complete"] = False
    elif change == "revision": seal["snapshot_revision"] = "other"
    elif change == "index": seal["index_sha256"] = "0" * 64
    elif change == "bytes": seal["files"][0]["bytes"] += 1
    elif change == "hash": (local / "table.safetensors").write_bytes(b"corrupted table bytes")
    elif change == "missing": seal["files"] = []
    elif change == "extra": seal["files"].append(dict(seal["files"][0], path="head.safetensors"))
    elif change == "total": seal["total_bytes"] += 1
    elif change == "symlink":
        (local / "table.safetensors").rename(tmp_path / "outside")
        (local / "table.safetensors").symlink_to(tmp_path / "outside")
    (local / "seal.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError):
        scope["Weights"](snapshot, local)


def test_qwen_layer_diagnostic_cannot_publish_golden_evidence():
    tree = ast.parse((ROOT / "python/reference/families/qwen4/golden.py").read_text())
    run = ast.unparse(next(n for n in tree.body if getattr(n, "name", None) == "run_windows"))
    assert "None if diagnostic_stop is not None else qualify" in run
    assert "'qualification': False" in run and "'kind': 'layer-timing-only'" in run
    stop = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "diagnostic_stop == layer_id")
    assert isinstance(stop.body[-1], ast.Return)
    assert "finish_golden" not in ast.unparse(stop)


def test_rolling_layer_checkpoint_exact_resume_and_retirement(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1024 * 2**30))
    from fidelity_windows import LayerCheckpoints
    shapes = {"short": [1, 3, 2, 4], "long": [1, 7, 2, 4]}
    binding = {"source_seal_sha256": "a" * 64, "set_sha256": "b" * 64}
    states = [np.arange(np.prod(s), dtype=np.uint16).reshape(s) for s in shapes.values()]
    store = LayerCheckpoints(tmp_path / "first", binding, shapes)
    store.commit(0, states, [1.0])
    store.commit(1, states, [1.0, 2.0])
    assert not (store.root / "layer00").exists()
    assert (store.root / "layer01" / "seal.json").exists()
    resumed = LayerCheckpoints(tmp_path / "resumed", binding, shapes, resume=store.root)
    layer, restored, times = resumed.resumed
    assert layer == 1 and times == [1.0, 2.0]
    for expected, actual in zip(states, restored):
        np.testing.assert_array_equal(expected, actual)
    # Interrupted successor files cannot replace the last committed complete layer.
    (store.root / "layer02").mkdir()
    assert store.read(store.root)[0] == 1
    assert (store.root / "layer01").exists()


@pytest.mark.parametrize("change", ["binding", "seal", "file", "shape"])
def test_layer_checkpoint_resume_fails_closed(tmp_path, change, monkeypatch):
    import shutil
    from types import SimpleNamespace
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1024 * 2**30))
    from fidelity_windows import LayerCheckpoints
    shapes = {"w": [1, 2, 1, 2]}
    binding = {"source_seal_sha256": "a" * 64}
    store = LayerCheckpoints(tmp_path / "first", binding, shapes)
    store.commit(0, [np.zeros(shapes["w"], dtype=np.uint16)], [1.0])
    if change == "binding": binding = {"source_seal_sha256": "c" * 64}
    elif change == "shape": shapes = {"w": [1, 1, 1, 4]}
    elif change == "seal": (store.root / "layer00/seal.json").write_text("{}")
    elif change == "file": (store.root / "layer00/w.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        LayerCheckpoints(tmp_path / "next", binding, shapes, resume=store.root)


def test_streamed_checkpoints_load_one_window_and_retain_predecessor(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from fidelity_windows import LayerCheckpoints, WindowCheckpoint
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1024 * 2**30))
    shapes = {"a": [1, 3, 1, 2], "b": [1, 7, 1, 2]}
    store = LayerCheckpoints(tmp_path / "streamed", {"source": "new"}, shapes,
                             streaming=True, retain_previous=True)
    states = [np.arange(np.prod(s), dtype=np.uint16).reshape(s) for s in shapes.values()]
    for layer in range(3):
        folder = store.begin(layer)
        descriptors = [store.write_window(folder, name, value) for name, value in zip(shapes, states)]
        store.commit(layer, descriptors, [1.] * (layer + 1))
    assert not (store.root / "layer00").exists()
    assert (store.root / "layer01").exists() and (store.root / "layer02").exists()
    resumed = LayerCheckpoints(tmp_path / "fresh", {"source": "qualified-new"}, shapes,
                              streaming=True, resume_binding={"source": "new"}, resume=store.root)
    layer, values, times = resumed.resumed
    assert layer == 2 and len(times) == 3
    for expected, descriptor in zip(states, values):
        assert isinstance(descriptor, WindowCheckpoint)
        np.testing.assert_array_equal(expected, descriptor.load())
    (values[0].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        values[0].load()


def test_streamed_partial_layer_cannot_replace_complete_pointer(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from fidelity_windows import LayerCheckpoints
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1024 * 2**30))
    shapes = {"a": [1, 2], "b": [1, 2]}
    store = LayerCheckpoints(tmp_path / "store", {}, shapes, streaming=True)
    states = [np.zeros(s, dtype=np.uint16) for s in shapes.values()]
    store.commit(0, states, [1.])
    folder = store.begin(1)
    partial = store.write_window(folder, "a", states[0])
    with pytest.raises(ValueError, match="incomplete"):
        store.commit(1, [partial], [1., 2.])
    assert store.read(store.root)[0] == 0


@pytest.mark.parametrize("capability", [(12, 1), (12, 0)])
@pytest.mark.parametrize("cuda_free,host_available", [(8, 32), (32, 8), (32, 32)])
def test_unified_admission_uses_reclaimable_host_availability(monkeypatch, cuda_free, host_available, capability):
    from pathlib import Path
    from types import SimpleNamespace
    import fidelity_windows as fw
    cuda = SimpleNamespace(synchronize=lambda: None, current_device=lambda: 0, mem_get_info=lambda device=0: (cuda_free * 2**30, 64 * 2**30),
                           get_device_properties=lambda device: SimpleNamespace(
                               is_integrated=capability == (12, 1), total_memory=64 * 2**30))
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw:
                        f"MemAvailable: {host_available * 2**20} kB\n" if str(p) == "/proc/meminfo"
                        else "" if str(p) == "/proc/self/cgroup" else original(p, *a, **kw))
    if (host_available if capability == (12, 1) else cuda_free) < 20:
        with pytest.raises(RuntimeError, match="before allocation"):
            fw.admit_reference_memory(cuda, 16 * 2**30)
    else:
        fw.admit_reference_memory(cuda, 16 * 2**30)


def test_reference_cuda_telemetry_distinguishes_live_and_cached(capsys):
    from fidelity_windows import log_reference_cuda_memory
    events = []
    cuda = SimpleNamespace(synchronize=lambda: events.append("drained"),
                           mem_get_info=lambda: (70 * 2**30, 120 * 2**30),
                           memory_allocated=lambda: 2**30, memory_reserved=lambda: 3 * 2**30)
    stats = log_reference_cuda_memory(cuda, "retired")
    assert events == ["drained"] and stats["allocated"] == 2**30 and stats["reserved"] == 3 * 2**30
    assert "allocated=1.000GiB reserved=3.000GiB free=70.000GiB" in capsys.readouterr().out


@pytest.mark.parametrize("complete", [True, False])
def test_reference_cache_release_requires_exact_host_ack(tmp_path, monkeypatch, complete):
    import fidelity_windows as fw
    request, response = tmp_path / "request", tmp_path / "response"
    original = pathlib.Path.write_bytes

    def acknowledge(path, data):
        result = original(path, data)
        if path.parent == request:
            record = json.loads(data)
            original(response / path.name, fw.canonical({"token": record["token"], "complete": complete}))
        return result

    monkeypatch.setattr(pathlib.Path, "write_bytes", acknowledge)
    if complete:
        fw.release_reference_page_cache(request, response, "layer2")
    else:
        with pytest.raises(RuntimeError, match="release failed"):
            fw.release_reference_page_cache(request, response, "layer2")
    assert len(list(request.iterdir())) == 1


def test_reference_cache_release_times_out_without_host(tmp_path):
    from fidelity_windows import release_reference_page_cache
    with pytest.raises(RuntimeError, match="timed out"):
        release_reference_page_cache(tmp_path / "request", tmp_path / "response", "layer2", timeout=0)


def test_layer_checkpoint_admission_retains_nvme_headroom(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from fidelity_windows import LayerCheckpoints
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30))
    with pytest.raises(RuntimeError, match="headroom"):
        LayerCheckpoints(tmp_path / "first", {}, {"w": [1, 2, 1, 2]})


def test_glm5_checkpoints_preserve_hidden_and_shared_dsa_indices():
    tree = ast.parse((ROOT / "python/reference/families/glm5/golden.py").read_text())
    run = ast.unparse(next(n for n in tree.body if getattr(n, "name", None) == "run_windows"))
    assert "-hidden" in run and "-index" in run
    assert "config.index_topk, 2" in run
    assert "checkpoint_states(states)" in run and "restore_states(arrays)" in run
    assert run.index("del layer") < run.index("checkpoints.commit")
    assert "not getattr(a, '_prefix_probe', False)" in run
    assert "attention_mask=mask" in run
    assert "log_checkpoint_reads" in run


def test_glm5_fixed_dsa_hooks_are_fail_closed_and_panel_bound():
    tree = ast.parse((ROOT / "python/reference/families/glm5/golden.py").read_text())
    hook = ast.unparse(next(n for n in tree.body if getattr(n, "name", None) == "install_dsa"))
    assert "unsupported official DSA score site" in hook
    assert "unsupported official DSA selection site" in hook
    assert "fixed_index_topk(scores, slots)" in hook
    assert "values.to(torch.int32)" in hook
    assert "value=-float('inf')" in hook
    assert "range(0, length, ROWS)" in hook
    run = ast.unparse(next(n for n in tree.body if getattr(n, "name", None) == "run_windows"))
    assert run.index("install_dsa") < run.index("qualify")
    assert "_dsa_extent" in run


def test_glm5_timing_pilot_cannot_publish_golden_evidence():
    tree = ast.parse((ROOT / "python/reference/families/glm5/golden.py").read_text())
    function = next(n for n in tree.body if getattr(n, "name", None) == "run_windows")
    source = ast.unparse(function)
    assert "None if diagnostic_stop is not None else qualify" in source
    stop = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "diagnostic_stop == layer_id")
    assert isinstance(stop.body[-1], ast.Return)
    assert "'qualification': False" in ast.unparse(stop)
    assert "finish_golden" not in ast.unparse(stop)


def test_glm5_layer_reads_are_physical_order_and_stop_on_slow_archive():
    tree = ast.parse((ROOT / "python/reference/families/glm5/golden.py").read_text())
    weights = next(n for n in tree.body if getattr(n, "name", None) == "Weights")
    stage = ast.unparse(next(n for n in weights.body if getattr(n, "name", None) == "stage_layer"))
    assert "data_offsets" in stage and "handle.get_tensor(name).clone()" in stage
    guard = ast.unparse(next(n for n in weights.body if getattr(n, "name", None) == "check_layer_read_rate"))
    assert "check_layer_read_rate" in stage
    assert "minimum_mbps=200" in guard and "average < minimum_mbps" in guard
    assert "slow_mbps=100" in guard and "consecutive_slow_layers=2" in guard
    assert "self.staged.clear()" in guard and "stop, do not crawl" in guard
    run = ast.unparse(next(n for n in tree.body if getattr(n, "name", None) == "run_windows"))
    assert run.index("weights.stage_layer") < run.index("GlmMoeDsaDecoderLayer")


@pytest.mark.parametrize("slow", [False, True])
def test_glm5_sequential_layer_staging_consumes_whole_tensors(tmp_path, slow):
    import struct
    tree = ast.parse((ROOT / "python/reference/families/glm5/golden.py").read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == "Weights")
    names = ["model.layers.0.z", "model.layers.0.a"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(names, "shard")}))
    header = json.dumps({names[0]: {"data_offsets": [0, 10]}, names[1]: {"data_offsets": [10, 20]}}).encode()
    (tmp_path / "shard").write_bytes(struct.pack("<Q", len(header)) + header)
    order = []
    class Tensor:
        def clone(self): return self
        def numel(self): return 300_000_000
        def element_size(self): return 2
    class Handle:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def get_tensor(self, name): order.append(name); return Tensor()
    now = iter(range(100)) if not slow else iter(range(0, 100000, 1000))
    scope = dict(Path=pathlib.Path, json=json, struct=struct,
        time=SimpleNamespace(monotonic=lambda: next(now)),
        torch=SimpleNamespace(Tensor=Tensor, cuda=None), safe_open=lambda *_args, **_kw: Handle(),
        release_checkpoint=lambda *_args: None)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "glm5-staging", "exec"), scope)
    weights = scope["Weights"](tmp_path)
    if slow:
        with pytest.raises(RuntimeError, match="stop, do not crawl"):
            weights.stage_layer("model.layers.0.")
        assert weights.staged == {}
    else:
        weights.stage_layer("model.layers.0.")
        assert order == names and weights.read_bytes == 1_200_000_000
        before = weights.read_bytes
        assert isinstance(weights.raw(names[0]), Tensor)
        assert weights.read_bytes == before and names[0] not in weights.staged


@pytest.mark.parametrize("rates,failed", [
    ([600, 123, 400], None),
    ([1000, 90, 90], 2),
    ([600, 70, 400, 70], None),
    ([300, 123, 123], 2),
    ([199], 0),
    ([200], None),
    ([1000, 100, 100], None),
])
def test_glm5_archive_guard_uses_cumulative_and_consecutive_rates(tmp_path, rates, failed):
    tree = ast.parse((ROOT / "python/reference/families/glm5/golden.py").read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == "Weights")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    released = []
    scope = dict(Path=pathlib.Path, json=json, torch=SimpleNamespace(Tensor=object, cuda=None),
                 release_checkpoint=lambda *_args: released.append(True))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "glm5-rate-guard", "exec"), scope)
    weights = scope["Weights"](tmp_path)
    for layer, rate in enumerate(rates):
        weights.staged = {"tensor": object()}
        if layer == failed:
            with pytest.raises(RuntimeError, match="stop, do not crawl"):
                weights.check_layer_read_rate(f"model.layers.{layer}.", rate * 1_000_000, 1)
            assert weights.staged == {} and released == [True]
            break
        weights.check_layer_read_rate(f"model.layers.{layer}.", rate * 1_000_000, 1)
        assert weights.staged and not released
    # A new prefix probe cannot inherit a prior pass's throughput or slow streak.
    weights.check_layer_read_rate("model.layers.0.", 600_000_000, 1)
    assert weights.layer_read_bytes == 600_000_000
    assert weights.layer_read_seconds == 1 and weights.slow_layers == 0
