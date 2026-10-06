"""Media fidelity contracts: CPU-only, no checkpoint or GPU needed."""
import copy
import hashlib
import multiprocessing
import os
import pathlib
import stat
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_media import publish_immutable, read_fixture, require_media_flag, validate_media, write_features


def load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts/bench" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g6_additive_ui_question_contract(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    import json
    generator = load_script("generate-media-fixtures")
    source = tmp_path / "source"
    generator.generate(source)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    g6 = load_script("generate-media-g6")
    monkeypatch.setattr(g6, "render", lambda html, png, browser: png.write_bytes(b"owned-ui"))
    monkeypatch.setattr(g6.subprocess, "check_output", lambda *args, **kwargs: "test-browser")
    one = g6.generate(source, tmp_path / "one", "browser")
    two = g6.generate(source, tmp_path / "two", "browser")
    assert one == two
    assert len(one["fixtures"]) == 9 and len(one["questions"]) == 24
    assert {q["category"] for q in one["questions"]} == {
        "ocr", "chart_values", "shapes_colors", "ui_labels"}
    assert len({q["id"] for q in one["questions"]}) == 24
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert "data:font/ttf;base64," in g6.ui_html("settings")
    assert "Apply" in g6.ui_html("settings") and "Ready" in g6.ui_html("ide")
    assert json.loads((tmp_path / "one/g6.json").read_text()) == one
    with pytest.raises(ValueError, match="must be new"):
        g6.generate(source, tmp_path / "one", "browser")
    (source / "code0.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="seal differs"):
        g6.generate(source, tmp_path / "bad", "browser")


def test_fixture_bytes_reproduce_and_vision_recipe(tmp_path):
    pytest.importorskip("PIL")
    import json
    generator = load_script("generate-media-fixtures")
    one = generator.generate(tmp_path / "one")
    two = generator.generate(tmp_path / "two")
    assert one == two
    assert len(one["fixtures"]) == 8
    for f in one["fixtures"]:
        assert (tmp_path / "one" / f["path"]).read_bytes() == (tmp_path / "two" / f["path"]).read_bytes()
    builder = load_script("fidelity-set")
    requests = []
    def probe(body):
        requests.append(body)
        assert body["max_tokens"] == 2048
        task = body["messages"][1]["content"][1]["text"]
        assert "at least 800 words" in task and "at least eight" in task
        assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        return {"probe": {"engine": "fake", "prompt_ids": [1] + [9] * 4 + [2],
                "generated": [3] * 600, "media": [{k: v for k, v in span().items() if k != "fixture"}]},
                "server": {"model": "model", "family": "mimo_v2"}}
    arm = {"checkpoint": "model", "head": "bf16", "activations": "bf16", "kv": "bf16",
           "state": "bf16", "speculation": False, "prefix_cache": False}
    result = builder.build_vision_set(family="mimo_v2", model="model", checkpoint="model", version="media1",
        arm=arm, probe=probe, fixtures=tmp_path / "one", tokenizer_sha256="c" * 64)
    assert len(result["windows"]) == 8 and len(result["quick_windows"]) == 2
    assert all(w["block"] == "vision" and set(w["roles"][w["score_from"]:]) == {"gen"} for w in result["windows"])
    tasks = [body["messages"][1]["content"][1]["text"] for body in requests]
    assert "visible line of code" in tasks[0] and "Python parser" in tasks[2]
    assert "matplotlib program" in tasks[4] and "validator" in tasks[6]
    def short(body):
        response = probe(body)
        response["probe"]["generated"] = [3] * 525
        return response
    with pytest.raises(ValueError, match="576 real generated"):
        builder.build_vision_set(family="mimo_v2", model="model", checkpoint="model", version="media2",
            arm=arm, probe=short, fixtures=tmp_path / "one", tokenizer_sha256="c" * 64)
    def long(body):
        response = probe(body)
        response["probe"]["generated"] = [3] * 1536
        return response
    expanded = builder.build_vision_set(family="mimo_v2", model="model", checkpoint="model", version="media2",
        arm=arm, probe=long, fixtures=tmp_path / "one", tokenizer_sha256="c" * 64)
    assert all(w["provenance"]["generated_tokens"] == 1536 for w in expanded["windows"])
    def missing(body):
        response = probe(body)
        del response["probe"]["media"]
        return response
    with pytest.raises(ValueError, match="prepared image"):
        builder.build_vision_set(family="mimo_v2", model="model", checkpoint="model", version="media1",
            arm=arm, probe=missing, fixtures=tmp_path / "one", tokenizer_sha256="c" * 64)


def span():
    return {"start": 1, "len": 4, "kind": "image", "key": "a" * 64,
            "grid": [1, 4, 4], "fixture": {"path": "code.png", "sha256": "b" * 64}}


def test_media_contract():
    s = span()
    validate_media([s], [1, 9, 9, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)
    for field, value in [("len", 0), ("start", -1), ("grid", [1, 3, 3]),
                         ("key", "A" * 64), ("fixture", {"path": "../x", "sha256": "b" * 64})]:
        bad = copy.deepcopy(s)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_media([bad], [1, 9, 9, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)
    with pytest.raises(ValueError):
        validate_media([s, s], [1, 9, 9, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)
    with pytest.raises(ValueError):
        validate_media([s], [1, 9, 8, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)


def test_media_prefix_keeps_images_and_cannot_use_text_evidence(tmp_path):
    from types import SimpleNamespace
    from fidelity_windows import (SET_SCHEMA, bucket, finish_golden, qualify_prefix,
        set_hash, validate_qualification, write_scored_logits)
    import json
    s = span()
    tokens = [1] + [9] * 4 + [3] * 700
    window = {"id": "vision00", "block": "vision", "bucket": bucket(193),
              "tokens": tokens, "roles": ["ctx"] * 193 + ["gen"] * 512,
              "score_from": 193, "media": [s]}
    manifest = {"schema": SET_SCHEMA, "family": "mimo_v2", "checkpoint": "model",
                "quick_windows": ["vision00"], "windows": [window]}
    manifest["set_sha256"] = set_hash(manifest)
    identity = {"snapshot_revision": "pinned"}
    seen = []
    def execute(args):
        panel = json.loads(args.windows.read_text())
        rows = []
        for w in panel["windows"]:
            seen.append(w["media"])
            rows.append(write_scored_logits(args.out, w, np.zeros((len(w["tokens"]) - w["score_from"], 4), dtype=np.float32)))
        finish_golden(args.out, panel, rows, snapshot_identity=identity, seconds=0)
    a = SimpleNamespace(out=tmp_path, windows=tmp_path / "windows.json", layers=None)
    proof = qualify_prefix(a, manifest, execute)
    assert proof["passed"] and seen == [[s], [s]] and proof["media"] == [s]
    validate_qualification(proof, manifest, identity)
    del proof["media"]
    with pytest.raises(ValueError, match="image evidence"):
        validate_qualification(proof, manifest, identity)
    def dropped_media(args):
        execute(args)
        meta_path = args.out / "meta.json"
        meta = json.loads(meta_path.read_text())
        for entry in meta["windows"]:
            entry.pop("media", None)
        meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="media identity"):
        qualify_prefix(a, manifest, dropped_media)


def test_feature_writer_binds_immutable_bytes(tmp_path):
    s = span()
    values = np.full((4, 8), 0x3f80, dtype=np.uint16)
    metadata = write_features(tmp_path, s, values, tower_dtype="bf16", identity={"snapshot_revision": "pinned"})
    assert metadata["shape"] == [4, 8]
    assert metadata["sha256"] == hashlib.sha256(values.astype("<u2").tobytes()).hexdigest()
    assert write_features(tmp_path, s, values, tower_dtype="bf16", identity={"snapshot_revision": "pinned"}) == metadata
    with pytest.raises(ValueError, match="identity reused"):
        write_features(tmp_path, s, values + 1, tower_dtype="bf16", identity={"snapshot_revision": "pinned"})
    with pytest.raises(ValueError, match="non-finite"):
        write_features(tmp_path, s, np.full((4, 8), 0x7f80, dtype=np.uint16), tower_dtype="bf16", identity={})


def test_immutable_publish_preserves_existing_inode_and_complete_bytes(tmp_path):
    path = tmp_path / "features.json"
    content = b'{"features":["first"]}\n'
    publish_immutable(path, content)
    before = path.stat()
    publish_immutable(path, content)
    assert path.stat().st_ino == before.st_ino and path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="immutable"):
        publish_immutable(path, b"different")
    assert path.read_bytes() == content and not list(tmp_path.glob(".feature-*"))


@pytest.mark.parametrize("mask,mode", [(0o022, 0o644), (0o027, 0o640), (0o077, 0o600)])
def test_atomic_feature_creation_preserves_ordinary_umask_permissions(tmp_path, mask, mode):
    previous = os.umask(mask)
    try:
        write_features(tmp_path, span(), np.full((4, 8), 0x3f80, dtype=np.uint16),
                       tower_dtype="bf16", identity={})
        publish_immutable(tmp_path / "features.json", b"index")
    finally:
        os.umask(previous)
    for name in (span()["key"] + ".bf16", span()["key"] + ".json", "features.json", ".features.lock"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == mode


def test_feature_lock_can_be_reused_without_write_permission(tmp_path, monkeypatch):
    import fidelity_media
    lock = tmp_path / ".features.lock"
    lock.touch(mode=0o444)
    actual_open = os.open
    flags = []

    def checked_open(path, flag, *args):
        if pathlib.Path(path) == lock:
            flags.append(flag)
        return actual_open(path, flag, *args)

    monkeypatch.setattr(fidelity_media.os, "open", checked_open)
    publish_immutable(tmp_path / "features.json", b"index")
    assert len(flags) == 1 and flags[0] & os.O_ACCMODE == os.O_RDONLY
    assert stat.S_IMODE(lock.stat().st_mode) == 0o444


def _feature_racing_writer(root, barrier, results, kind, variant):
    try:
        barrier.wait(timeout=10)
        if kind == "index":
            publish_immutable(root / "features.json", b'{"set":' + str(variant).encode() + b'}\n')
        else:
            write_features(root, span(), np.full((4, 8), 0x3f80 + variant, dtype=np.uint16),
                           tower_dtype="bf16", identity={"snapshot_revision": "pinned"})
        results.put((variant, True))
    except ValueError:
        results.put((variant, False))


@pytest.mark.parametrize("kind,equal", [("index", False), ("index", True), ("rows", False), ("rows", True)])
def test_feature_writers_race_without_replacing_identity(tmp_path, kind, equal):
    import json
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(2), context.Queue()
    variants = [0, 0 if equal else 1]
    children = [context.Process(target=_feature_racing_writer, args=(tmp_path, barrier, results, kind, v))
                for v in variants]
    try:
        for child in children:
            child.start()
        outcomes = [results.get(timeout=15) for _ in children]
        for child in children:
            child.join(timeout=15)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()
        results.close()
        results.join_thread()
    assert sum(success for _, success in outcomes) == (2 if equal else 1)
    winner = next(v for v, success in outcomes if success)
    if kind == "index":
        assert json.loads((tmp_path / "features.json").read_text()) == {"set": winner}
    else:
        data = (tmp_path / (span()["key"] + ".bf16")).read_bytes()
        meta = json.loads((tmp_path / (span()["key"] + ".json")).read_text())
        assert np.frombuffer(data, dtype="<u2").tolist() == [0x3f80 + winner] * 32
        assert meta["sha256"] == hashlib.sha256(data).hexdigest()
    assert not list(tmp_path.glob(".feature-*"))


def test_feature_pair_conflict_does_not_publish_other_member(tmp_path):
    key = span()["key"]
    (tmp_path / (key + ".json")).write_bytes(b"conflicting metadata")
    with pytest.raises(ValueError, match="identity reused"):
        write_features(tmp_path, span(), np.full((4, 8), 0x3f80, dtype=np.uint16),
                       tower_dtype="bf16", identity={})
    assert not (tmp_path / (key + ".bf16")).exists()


def test_fixture_hash_and_family_fail_closed(tmp_path):
    s = span()
    (tmp_path / "code.png").write_bytes(b"fixture")
    s["fixture"]["sha256"] = hashlib.sha256(b"fixture").hexdigest()
    assert read_fixture(tmp_path, s) == b"fixture"
    (tmp_path / "code.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned"):
        read_fixture(tmp_path, s)
    with (tmp_path / "code.png").open("wb") as f:
        f.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="byte cap"):
        read_fixture(tmp_path, s)
    manifest = {"windows": [{"media": [s]}]}
    assert require_media_flag(manifest, True, "mimo_v2")
    assert require_media_flag(manifest, True, "qwen4")
    for family, flag in [("mimo_v2", False), ("qwen4", False), ("glm5_flash", True)]:
        with pytest.raises(ValueError):
            require_media_flag(manifest, flag, family)
