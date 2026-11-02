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


@pytest.mark.parametrize("family", ["mimo_v2", "glm5_flash", "qwen4"])
def test_scoring_launch_admission_precedes_media_work(family):
    adapter = (ROOT / "rust/crates/cuteafd-daemon/src/families" / family / "serve.rs").read_text()
    if family == "qwen4":
        source = (ROOT / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
        assert "super::media::prepare(" in adapter
        assert "super::media::probe_features(" in adapter
        operations = ("engine.prepare(", "engine.probe_features(")
    else:
        source = adapter
        operations = ("super::media::prepare(", "super::media::probe_features(")
    receipt = source.index("let job = if !busy && media.is_empty()")
    guard = source.index("probe::validate_scoring(&job.probe,", receipt)
    assert source.count("probe::validate_scoring(") == 1
    for operation in ("!job.media.is_empty()", "probe::prompt_ids(", *operations, "media.enqueue("):
        assert receipt < guard < source.index(operation, receipt)
    if family == "mimo_v2":
        assert source.index("failures.watch(&job.events)", receipt) < guard


def test_g6_additive_ui_question_contract(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")
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
    pytest.importorskip("matplotlib")
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
    # Pin the original builder's complete legacy body, not two calls to the new path.
    import ast
    function = next(n for n in ast.parse(pathlib.Path(builder.__file__).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "build_vision_set")
    function.args.kwonlyargs = function.args.kwonlyargs[:-2]
    function.args.kw_defaults = function.args.kw_defaults[:-2]
    del function.body[1]  # The sole new statement dispatches --vision-manifest.
    assert hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest() == (
        "bfc3c3bebc7d46288ef4c83a9bdb862c559f744314e53671f8f0962622a1a1f6")
    assert builder.canonical(result) == builder.canonical(builder.build_vision_set(
        family="mimo_v2", model="model", checkpoint="model", version="media1", arm=arm,
        probe=probe, fixtures=tmp_path / "one", tokenizer_sha256="c" * 64, vision_manifest=None))
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


@pytest.fixture
def real_vision_input(tmp_path):
    import json
    from PIL import Image
    root = tmp_path / "sources"
    snapshot = tmp_path / "snapshot"
    root.mkdir()
    snapshot.mkdir()
    Image.new("RGB", (56, 56), (80, 120, 160)).save(root / "photo.png")
    digest = hashlib.sha256((root / "photo.png").read_bytes()).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"images": [{"file": "photo.png", "sha256": digest}],
        "resize": ["photo.png@4x4"]}))
    (snapshot / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 32},
        "vision_config": {"patch_size": 14}, "spatial_merge_size": 2, "image_token_index": 9}))
    (snapshot / "processor_config.json").write_text(json.dumps({"spatial_merge_size": 2,
        "image_processor": {"image_processor_type": "PixtralImageProcessor", "patch_size": 14}}))
    (snapshot / "params.json").write_text(json.dumps({"vision_encoder": {
        "image_token_id": 9, "image_break_token_id": 8, "image_end_token_id": 7}}))
    return manifest, snapshot, tmp_path / "fixtures"


def test_real_vision_manifest_pins_resizes_and_scores_following_text(real_vision_input):
    from fidelity_media import processor_contract
    manifest, snapshot, destination = real_vision_input
    builder = load_script("fidelity-set")
    records, contract = builder.real_vision_fixtures(manifest, destination, snapshot)
    assert contract == processor_contract(snapshot)
    assert len(records) == 2 and records[1]["provenance"]["grid"] == [1, 4, 4]
    assert records[1]["provenance"]["derived_sha256"] == records[1]["fixture"]["sha256"]
    calls = []
    def probe(body):
        calls.append(body)
        image = body["messages"][0]["content"][0]["type"] == "image_url"
        return {"probe": {"engine": "fake", "prompt_ids": [1, 9, 9, 8, 9, 9, 7, 2] if image else [1, 2],
            "generated": [3] * 600, "media": [{"start": 1, "len": 6, "grid": [1, 4, 4],
                "key": "a" * 64, "kind": "image"}] if image else []},
            "server": {"model": "model", "family": "mistral_large4"}}
    arm = {"checkpoint": "model", "head": "bf16", "activations": "bf16", "kv": "bf16",
        "state": "bf16", "speculation": False, "prefix_cache": False}
    result = builder.build_vision_set(family="mistral_large4", model="model", checkpoint="model",
        version="real1", arm=arm, probe=probe, fixtures=destination, tokenizer_sha256="c" * 64,
        vision_manifest=manifest, snapshot=snapshot)
    assert [w["id"] for w in result["windows"]] == ["real_00", "resize_00", "text_control"]
    for window in result["windows"]:
        assert set(window["roles"][window["score_from"]:]) == {"gen"}
        for image in window.get("media", []):
            assert image["start"] + image["len"] <= window["score_from"]
    assert not result["windows"][-1].get("media") and len(calls) == 3
    assert result["windows"][0]["media"][0]["fixture"] == records[0]["fixture"]


@pytest.mark.parametrize("mutation,match", [
    ("hash", "pinned media identity"), ("path", "relative path"),
    ("resize", "pinned FILE"), ("pixels", "geometry"), ("merge", "image merge"),
])
def test_real_vision_manifest_rejects_unbound_inputs(real_vision_input, mutation, match):
    import json
    manifest, snapshot, destination = real_vision_input
    document = json.loads(manifest.read_text())
    if mutation == "hash":
        document["images"][0]["sha256"] = "0" * 64
    elif mutation == "path":
        document["images"][0]["file"] = "../photo.png"
    elif mutation == "resize":
        document["resize"] = ["unbound.png@4x4"]
    elif mutation == "pixels":
        document["resize"] = ["photo.png@2048x2048"]
    else:
        processor = json.loads((snapshot / "processor_config.json").read_text())
        processor["spatial_merge_size"] = True
        (snapshot / "processor_config.json").write_text(json.dumps(processor))
        config = json.loads((snapshot / "config.json").read_text())
        config["spatial_merge_size"] = True
        (snapshot / "config.json").write_text(json.dumps(config))
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        load_script("fidelity-set").real_vision_fixtures(manifest, destination, snapshot)


def test_shared_processor_contract_supports_legacy_processor_not_family_name(tmp_path):
    import json
    from fidelity_media import processor_contract
    (tmp_path / "config.json").write_text("{}")
    for kind in ("Qwen2VLImageProcessor", "Glm5NextImageProcessor"):
        (tmp_path / "preprocessor_config.json").write_text(json.dumps({
            "image_processor_type": kind, "patch_size": 16, "merge_size": 2}))
        assert processor_contract(tmp_path) == {"merge": 2, "row_markers": 0}


def span():
    return {"start": 1, "len": 4, "kind": "image", "key": "a" * 64,
            "grid": [1, 4, 4], "fixture": {"path": "code.png", "sha256": "b" * 64}}


def test_processor_geometry_preserves_merged_counts_and_checks_row_markers():
    from fidelity_media import image_extent
    for h in range(2, 112, 2):
        for w in range(2, 112, 2):
            assert image_extent([1, h, w]) == (h // 2, w // 2, h * w // 4)
            assert image_extent([1, h, w], {"merge": 2, "row_markers": 1}) == (
                h // 2, w // 2, h // 2 * (w // 2 + 1))
    contract = {"merge": 2, "row_markers": 1, "image_token_id": 9, "row_break_token_id": 8, "end_token_id": 7}
    s = {**span(), "len": 6}
    tokens = [1, 9, 9, 8, 9, 9, 7, 2]
    assert validate_media([s], tokens, ["ctx"] * 7 + ["gen"], 7, processor=contract) == [s]
    for i in range(1, 7):
        wrong = tokens.copy()
        wrong[i] = 3
        with pytest.raises(ValueError, match="row markers"):
            validate_media([s], wrong, ["ctx"] * 7 + ["gen"], 7, processor=contract)
    with pytest.raises(ValueError, match="merged token extent"):
        validate_media([s], tokens, ["ctx"] * 7 + ["gen"], 7)


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


@pytest.mark.parametrize("family", ["glm5_flash", "qwen4"])
def test_explicit_implemented_family_opt_in(family):
    manifest = {"windows": [{"media": [span()]}]}
    assert require_media_flag(manifest, True, family, implemented_families=(family,))
    with pytest.raises(ValueError, match="not implemented"):
        require_media_flag(manifest, True, "deepseek_v4", implemented_families=(family,))
    with pytest.raises(ValueError, match="explicit --media"):
        require_media_flag(manifest, False, family, implemented_families=(family,))
    with pytest.raises(ValueError, match="pinned media"):
        require_media_flag({"windows": [{}]}, True, family, implemented_families=(family,))


def glm_config():
    return {"model_type": "glm5_next", "image_start_token_id": 1,
            "image_end_token_id": 2, "image_token_id": 9,
            "text_config": {"vocab_size": 32, "hidden_size": 8,
                            "moe_intermediate_size": 4, "n_routed_experts": 1,
                            "first_k_dense_replace": 0, "num_hidden_layers": 1}}


@pytest.mark.parametrize("bad", [None, "placeholder", "begin", "end", "ids", "capacity"])
def test_glm_spans_use_actual_checkpoint_markers(bad):
    from glm_flash_media import validate_spans
    config = glm_config()
    image = span()
    tokens = [1] + [9] * 4 + [2, 3]
    if bad == "placeholder": tokens[2] = 8
    if bad == "begin": tokens[0] = 8
    if bad == "end": tokens[5] = 8
    if bad == "ids": config["image_token_id"] = True
    if bad == "capacity":
        image["len"] = 4097
        tokens = [1] + [9] * 4097 + [2, 3]
    manifest = {"windows": [{"tokens": tokens, "media": [image]}]}
    if bad:
        with pytest.raises(ValueError): validate_spans(manifest, config)
    else:
        validate_spans(manifest, config)


@pytest.mark.parametrize("token", [1, 2, 9])
@pytest.mark.parametrize("where", ["context", "generated", "text_window"])
def test_glm_rejects_unbound_image_tokens_in_full_sequence(token, where):
    from glm_flash_media import validate_spans
    window = {"tokens": [1] + [9] * 4 + [2, 3], "media": [span()]}
    if where == "context":
        window["tokens"].insert(0, token)
        window["media"][0]["start"] += 1
    elif where == "generated":
        window["tokens"].append(token)
    else:
        window = {"tokens": [3, token, 3], "media": []}
    with pytest.raises(ValueError, match="pinned span list"):
        validate_spans({"windows": [window]}, glm_config())


def mock_glm_sources(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import glm_flash_media as media
    root = tmp_path / "checkout/src/transformers"
    modules, originals = {}, {}
    for name in media.SOURCE_MODULES:
        path = root / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# " + name + "\n")
        modules["transformers." + name] = SimpleNamespace(__file__=str(path))
        originals[name] = path.read_bytes()
    package = ModuleType("transformers.models.glm5_next")
    package.modeling_glm5_next = modules["transformers.models.glm5_next.modeling_glm5_next"]
    package.image_processing_pil_glm5_next = modules["transformers.models.glm5_next.image_processing_pil_glm5_next"]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(media.importlib, "import_module", modules.__getitem__)
    state = {"revision": media.TRANSFORMERS_REVISION, "changes": ""}
    def git(command, **kwargs):
        assert command[:3] == ["git", "-C", str(root.parent.parent)] and kwargs["timeout"] == 10
        if command[3] == "rev-parse": return state["revision"] + "\n"
        if command[3] == "status": return state["changes"]
        assert command[3] == "show"
        relative = command[4].split(":src/transformers/")[1]
        return originals[relative.removesuffix(".py").replace("/", ".")]
    monkeypatch.setattr(media.subprocess, "check_output", git)
    return media, package, modules, state


def test_glm_source_identity_hashes_loaded_code_and_nested_processor(tmp_path, monkeypatch):
    import json
    media, package, _modules, _state = mock_glm_sources(tmp_path, monkeypatch)
    for name in ("config.json", "tokenizer.json", "processor_config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text(json.dumps({"source": name}))
    identity = media.snapshot_identity(tmp_path)
    for key, path in (("modeling", pathlib.Path(package.modeling_glm5_next.__file__)),
                      ("image_processing", pathlib.Path(package.image_processing_pil_glm5_next.__file__)),
                      ("preprocessor", tmp_path / "processor_config.json"),
                      ("index", tmp_path / "model.safetensors.index.json")):
        assert identity[key + "_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(identity) == {"snapshot_revision", "config_sha256", "tokenizer_sha256", "preprocessor_sha256",
                             "modeling_sha256", "image_processing_sha256", "index_sha256"}


@pytest.mark.parametrize("change", ["revision", "dirty", "shadow", "bytes", "git"])
def test_glm_source_admission_rejects_changed_pin_or_loaded_helper(tmp_path, monkeypatch, change):
    media, package, modules, state = mock_glm_sources(tmp_path, monkeypatch)
    helper = modules["transformers.vision_utils"]
    if change == "revision": state["revision"] = "0" * 40
    if change == "dirty": state["changes"] = " M src/transformers/image_transforms.py\n"
    if change == "shadow": helper.__file__ = str(tmp_path / "site-packages/vision_utils.py")
    if change == "bytes": pathlib.Path(helper.__file__).write_text("# modified while Git ignores changes\n")
    if change == "git":
        def fail(*args, **kwargs): raise OSError("unavailable checkout")
        monkeypatch.setattr(media.subprocess, "check_output", fail)
    with pytest.raises(ValueError, match="pinned"):
        media.verify_sources(package.modeling_glm5_next, package.image_processing_pil_glm5_next)


def test_glm_processor_uses_nested_config_and_cap(tmp_path, monkeypatch):
    import json
    from types import ModuleType
    from glm_flash_media import processor
    name = "transformers.models.glm5_next.image_processing_pil_glm5_next"
    module = ModuleType(name)
    module.Glm5NextImageProcessorPil = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, name, module)
    config = {"image_processor": {"max_image_tokens": 8000, "patch_size": 14,
                                  "image_processor_type": "Glm5NextImageProcessor"},
              "video_processor": {"max_image_tokens": 240000}}
    (tmp_path / "processor_config.json").write_text(json.dumps(config))
    assert processor(tmp_path) == {"max_image_tokens": 4096, "patch_size": 14}


@pytest.mark.parametrize("fail", [False, True])
def test_glm_tower_restores_global_lm_arithmetic_on_exception(monkeypatch, fail):
    from types import ModuleType, SimpleNamespace
    from glm_flash_media import official_tower_arithmetic
    torch = ModuleType("torch")
    F = ModuleType("torch.nn.functional")
    nn = ModuleType("torch.nn")
    nn.functional = F
    torch.nn = nn
    padded_linear, padded_einsum, official_linear, official_einsum = [object() for _ in range(4)]
    F.linear, torch.einsum = padded_linear, padded_einsum
    for name, module in (("torch", torch), ("torch.nn", nn), ("torch.nn.functional", F)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "shape_invariant",
                        SimpleNamespace(_linear=official_linear, _einsum=official_einsum))
    def encode():
        with official_tower_arithmetic():
            assert F.linear is official_linear and torch.einsum is official_einsum
            if fail: raise RuntimeError("tower failed")
    if fail:
        with pytest.raises(RuntimeError, match="tower failed"): encode()
    else:
        encode()
    assert F.linear is padded_linear and torch.einsum is padded_einsum


def make_glm_experts(tmp_path, dtype="BF16", quant=None):
    import json
    import struct
    config = glm_config()
    if quant: config["quantization_config"] = {"quant_method": quant}
    (tmp_path / "config.json").write_text(json.dumps(config))
    headers = {}
    for projection in ("gate", "up", "down"):
        name = f"model.language_model.layers.0.mlp.experts.0.{projection}_proj.weight"
        shape = [8, 4] if projection == "down" else [4, 8]
        headers[name] = {"dtype": dtype, "shape": shape, "data_offsets": [0, 64]}
        if dtype == "F8_E4M3":
            headers[name.removesuffix("weight") + "weight_scale_inv"] = {
                "dtype": "F32", "shape": [1, 1], "data_offsets": [64, 68]}
    encoded = json.dumps(headers).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        name: "weights.safetensors" for name in headers}}))
    return headers


@pytest.mark.parametrize("dtype,quant", [("BF16", None), ("F8_E4M3", "fp8"), ("U8", None), ("BF16", "exl3")])
def test_glm_media_experts_fail_closed_without_torch(tmp_path, dtype, quant):
    from glm_flash_media import expert_snapshot_identity
    make_glm_experts(tmp_path, dtype, quant)
    if dtype == "U8" or quant == "exl3":
        with pytest.raises(ValueError, match="unsupported|EXL3"):
            expert_snapshot_identity(tmp_path)
    else:
        identity = expert_snapshot_identity(tmp_path)
        assert identity["storage_dtypes"] == [dtype]
        assert "tensor bytes not hashed" in identity["scope"]
        assert identity["snapshot_revision"] == tmp_path.name


def test_glm_expert_fp8_scale_and_missing_tensor_rejected(tmp_path):
    import json
    import struct
    from glm_flash_media import expert_snapshot_identity
    headers = make_glm_experts(tmp_path, "F8_E4M3", "fp8")
    scale = next(name for name in headers if name.endswith("weight_scale_inv"))
    headers[scale]["dtype"] = "BF16"
    encoded = json.dumps(headers).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    with pytest.raises(ValueError, match="block scale"): expert_snapshot_identity(tmp_path)
    headers = make_glm_experts(tmp_path)
    name = next(iter(headers))
    del headers[name]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        key: "weights.safetensors" for key in headers}}))
    with pytest.raises(ValueError, match="missing official expert tensor"): expert_snapshot_identity(tmp_path)


def test_glm_injects_single_stream_before_four_hc_copies():
    from glm_flash_media import inject_embeddings
    class Tensor(np.ndarray):
        def copy_(self, value): np.copyto(self, value)
    embeddings = np.zeros((1, 7, 8)).view(Tensor)
    features = {"a" * 64: np.arange(32).reshape(4, 8)}
    inject_embeddings(embeddings, [span()], features)
    hc = np.repeat(embeddings[:, :, None], 4, axis=2)
    for lane in range(4): np.testing.assert_array_equal(hc[0, 1:5, lane], features["a" * 64])
    assert not embeddings[:, [0, 5, 6]].any()
    with pytest.raises(ValueError, match="widths"):
        inject_embeddings(embeddings, [span()], {"a" * 64: np.zeros((4, 4))})


def fake_torch(monkeypatch):
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace
    torch = ModuleType("torch")
    nn, F = ModuleType("torch.nn"), ModuleType("torch.nn.functional")
    nn.functional = F
    F.linear, torch.einsum = object(), object()
    nn.attention = SimpleNamespace(sdpa_kernel=lambda _backend: nullcontext(),
                                  SDPBackend=SimpleNamespace(MATH="math"))
    torch.nn = nn
    torch.inference_mode = torch.no_grad = nullcontext
    torch.bfloat16, torch.float32, torch.uint16 = "bf16", "fp32", "uint16"
    torch.cuda = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
    for name, module in (("torch", torch), ("torch.nn", nn), ("torch.nn.functional", F)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "shape_invariant", raising=False)
    from PIL import ImageOps
    image_utils = ModuleType("transformers.image_utils")
    image_utils.load_image = lambda image: ImageOps.exif_transpose(image).convert("RGB")
    monkeypatch.setitem(sys.modules, image_utils.__name__, image_utils)
    return torch


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("orientation", [None, 3, 6])
def test_glm_encode_uses_pooler_not_last_hidden_state(tmp_path, monkeypatch, dtype, orientation):
    import io
    from types import SimpleNamespace
    from PIL import Image
    from glm_flash_media import encode_span
    torch = fake_torch(monkeypatch)
    data = io.BytesIO()
    source = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
    source.putpixel((0, 0), (100, 110, 120, 128))
    exif = Image.Exif()
    if orientation is not None: exif[274] = orientation
    source.save(data, format="PNG", exif=exif)
    (tmp_path / "code.png").write_bytes(data.getvalue())
    image_span = span()
    image_span["fixture"]["sha256"] = hashlib.sha256(data.getvalue()).hexdigest()
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        @property
        def shape(self): return self.data.shape
        def cuda(self): return self
        def to(self, actual): assert actual == dtype; return self
        def tolist(self): return self.data.tolist()
        def cpu(self): return self
    value = Tensor(np.arange(32).reshape(4, 8))
    torch.isfinite = lambda features: np.isfinite(features.data)
    def process(*, images, return_tensors):
        assert images.mode == "RGB" and images.size == (4, 4)
        corner = {None: (0, 0), 3: (3, 3), 6: (3, 0)}[orientation]
        assert images.getpixel(corner) == (100, 110, 120) and images.getpixel((1, 1)) == (10, 20, 30)
        assert return_tensors == "pt"
        return {"image_grid_thw": Tensor([[1, 4, 4]]), "pixel_values": Tensor([[0.]])}
    class Tower:
        config = SimpleNamespace(out_hidden_size=8)
        def __call__(self, _pixels, _grid):
            return SimpleNamespace(pooler_output=value, last_hidden_state=Tensor(np.zeros((4, 4096))))
    assert encode_span(Tower(), process, tmp_path, image_span, dtype=dtype) is value
    image_span["grid"] = [1, 2, 8]
    with pytest.raises(ValueError, match="grid differs"):
        encode_span(Tower(), process, tmp_path, image_span, dtype=dtype)


@pytest.mark.parametrize("change", [None, "missing", "dtype", "shape"])
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_glm_tower_loader_consumes_only_bf16_visual_parameters(tmp_path, monkeypatch, change, dtype):
    import json
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace
    from glm_flash_media import load_tower
    torch = fake_torch(monkeypatch)
    events, defaults = [], ["bf16"]
    torch.get_default_dtype = lambda: defaults[-1]
    torch.set_default_dtype = lambda dtype: defaults.append(dtype)
    torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()),
                                     cudnn=SimpleNamespace())
    class Tensor:
        dtype = "bf16" if change != "dtype" else "fp32"
        shape = (2, 2) if change != "shape" else (3, 2)
        def copy_(self, value): events.append(("copy", value))
    class ParameterData:
        def to(self, actual): events.append(("parameter_to", actual)); return self
    frequencies = np.array([1.0, 0.316227766, 0.1], dtype=np.float32)
    class Tower:
        def __init__(self, config):
            assert torch.get_default_dtype() == "fp32" and config._attn_implementation == "sdpa"
            self.parameter = SimpleNamespace(shape=(2, 2), data=ParameterData(),
                                             copy_=lambda value: events.append(("copy", value)))
            self.inv_freq, self.original_inv_freq = frequencies.copy(), frequencies.copy()
        def named_parameters(self): return [("weight", self.parameter)]
        def to(self, dtype): pytest.fail("whole-model dtype conversion rounds canonical RoPE buffers")
        def cuda(self): events.append("cuda"); return self
        def eval(self): events.append("eval"); return self
    config_module = ModuleType("transformers.models.glm5_next.configuration_glm5_next")
    config_module.Glm5NextVisionConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    model_module = ModuleType("transformers.models.glm5_next.modeling_glm5_next")
    model_module.Glm5NextVisionModel = Tower
    safetensors = ModuleType("safetensors")
    def opened(path, **kwargs):
        assert path == str(tmp_path / "visual.safetensors")
        return nullcontext(SimpleNamespace(get_tensor=lambda name: Tensor()))
    safetensors.safe_open = opened
    for module in (config_module, model_module, safetensors):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    config = glm_config()
    config["vision_config"] = {"out_hidden_size": 8}
    (tmp_path / "config.json").write_text(json.dumps(config))
    index = {"lm_head.weight": "never-opened.safetensors"}
    if change != "missing": index["model.visual.weight"] = "visual.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    if change:
        with pytest.raises(ValueError, match="missing|dtype/shape"):
            load_tower(tmp_path, dtype)
    else:
        model, _config = load_tower(tmp_path, dtype)
        assert model.inv_freq.dtype == model.original_inv_freq.dtype == np.float32
        assert model.inv_freq.tobytes() == model.original_inv_freq.tobytes() == frequencies.tobytes()
        assert events[-2:] == ["cuda", "eval"]
        assert [entry for entry in events if isinstance(entry, tuple) and entry[0] == "parameter_to"] == [("parameter_to", dtype)]
    assert defaults[-1] == "bf16"


def test_glm_feature_export_and_prefix_probe_immutability(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import glm_flash_media as media
    from fidelity_windows import canonical
    torch = fake_torch(monkeypatch)
    events = []
    torch.cuda.synchronize = lambda: events.append("drain")
    torch.cuda.empty_cache = lambda: events.append("empty")
    class Tensor:
        def bfloat16(self): return self
        def contiguous(self): return self
        def view(self, dtype): assert dtype == "uint16"; return self
        def numpy(self): return np.full((4, 8), 0x3f80, dtype=np.uint16)
    (tmp_path / "config.json").write_text(json.dumps(glm_config()))
    identity = {"snapshot_revision": "test", "tokenizer_sha256": "a" * 64}
    monkeypatch.setattr(media, "snapshot_identity", lambda *_args: identity)
    monkeypatch.setattr(media, "load_tower", lambda *_args: (object(), {}))
    monkeypatch.setattr(media, "processor", lambda *_args: object())
    calls = []
    monkeypatch.setattr(media, "encode_span", lambda *_args, **_kwargs: calls.append("encode") or Tensor())
    image = span()
    window = {"tokens": [1] + [9] * 4 + [2], "media": [image]}
    manifest = {"windows": [window, window], "family": "glm5_flash", "checkpoint": "test", "set_sha256": "b" * 64}
    args = SimpleNamespace(snapshot=tmp_path, media_root=tmp_path, windows=tmp_path / "input.json",
                           media_features_out=tmp_path / "features", tower_dtype="fp32")
    values, actual_identity = media.window_features(args, manifest)
    assert len(values) == 1 and calls == ["encode"] and actual_identity == identity
    assert events == ["drain", "empty"]
    metadata = json.loads((args.media_features_out / (image["key"] + ".json")).read_text())
    assert metadata["tower_dtype"] == "fp32" and metadata["dtype"] == "bf16-le"
    index = args.media_features_out / "features.json"
    published = index.stat()
    media.window_features(args, manifest)
    assert (index.stat().st_ino, index.stat().st_mtime_ns) == (published.st_ino, published.st_mtime_ns)
    files = {p.name: p.read_bytes() for p in args.media_features_out.iterdir()}
    args._prefix_probe = True
    media.window_features(args, manifest)
    assert {p.name: p.read_bytes() for p in args.media_features_out.iterdir()} == files
    args._prefix_probe = False
    index = args.media_features_out / "features.json"
    index.write_bytes(canonical({"tampered": True}))
    with pytest.raises(ValueError, match="index is immutable"): media.window_features(args, manifest)
    conflict = copy.deepcopy(window)
    conflict["media"][0]["fixture"]["sha256"] = "c" * 64
    manifest["windows"] = [window, conflict]
    with pytest.raises(ValueError, match="different fixtures/grids"): media.window_features(args, manifest)


@pytest.mark.parametrize("family", [None, "glm5_flash"])
def test_official_tower_export_family_dispatch_preserves_mimo_default(tmp_path, monkeypatch, family):
    import json
    from types import ModuleType, SimpleNamespace
    import golden_media as exporter
    expected = family or "mimo_v2"
    manifest = {"family": expected, "windows": [{"media": [span()]}], "checkpoint": "test",
                "set_sha256": "b" * 64}
    monkeypatch.setattr(exporter, "load_set", lambda _path, actual: manifest if actual == expected else pytest.fail("wrong family"))
    monkeypatch.setattr(exporter, "verify_snapshot", lambda *_args: {"snapshot_revision": "test"})
    seen = []
    module = ModuleType("glm_flash_media" if family else "mimo_media")
    def features(args, panel):
        assert panel is manifest and args.family == expected and args.tower_dtype == "bf16"
        assert args.media_features_out == tmp_path / "features"
        args.out.mkdir()
        seen.append(expected)
        return {span()["key"]: SimpleNamespace(shape=(4, 8))}, {"modeling_sha256": "a" * 64}
    module.window_features = features
    monkeypatch.setitem(sys.modules, module.__name__, module)
    torch = fake_torch(monkeypatch)
    torch.cuda.set_device = lambda value: seen.append(value)
    argv = ["golden_media", "--snapshot", str(tmp_path), "--windows", str(tmp_path / "input.json"),
            "--out", str(tmp_path / "features")]
    if family: argv += ["--family", family]
    monkeypatch.setattr(sys, "argv", argv)
    exporter.main()
    output = json.loads((tmp_path / "features/export.json").read_text())
    assert seen == [0, expected] and output["feature_shapes"] == {span()["key"]: [4, 8]}
    assert output["scope"] == "tower only; not an LM prefix qualification"
    assert output["snapshot_identity"] == {"snapshot_revision": "test", "modeling_sha256": "a" * 64}
    if family: assert output["family"] == family
    else: assert "family" not in output
    with pytest.raises(ValueError, match="not implemented"): exporter.family_features("deepseek_v4")
