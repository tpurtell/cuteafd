"""MC publication/loader cross-language golden vectors; no safetensors needed."""
import copy
import importlib.util
import json
import re
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mc_manifest", ROOT / "scripts/lib/mc_manifest.py")
mc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mc)
FIXTURE = Path(__file__).with_name("fixtures") / "mc-v1"
SHARED = "1fb4551df22a63f50f22340099e2b495813979999164ec641931f466fe674655"
FINGERPRINTS = {
    "original": "4d585770a7aee778ae778d5d7e11dd013e69efd2740654c4a852f6f232d9ed16",
    "k6-compact": "2c998320fea2f7f60a2c8e605ab68e85bd63cf96188b3724e4fccd3a96c2f10e",
    "k8-compact": "86470216c28b0b319efd25d2eba3dc7ac12c2d0a955ec0747f3d2d01558da76b",
}


def test_cross_language_golden_vectors():
    manifest = json.loads((FIXTURE / mc.MANIFEST).read_text())
    assert mc.validate(manifest, FIXTURE) == manifest
    assert manifest["shared_inventory_sha256"] == SHARED
    for v in manifest["variants"]:
        assert v["fingerprint"] == FINGERPRINTS[v["name"]]
    assert mc.build(FIXTURE, source_repo=manifest["source"]["repo"],
                    source_revision=manifest["source"]["revision"],
                    core_tensors={v["name"]: v["core_tensors"] for v in manifest["variants"]}) == manifest


@pytest.mark.parametrize("damage", ["fp8", "duplicate", "hash", "unsafe", "unknown", "core", "missing"])
def test_manifest_fails_closed(damage):
    manifest = json.loads((FIXTURE / mc.MANIFEST).read_text())
    if damage == "fp8":
        manifest["original_core_dtype"] = "F8_E4M3"
    elif damage == "duplicate":
        manifest["variants"][1] = copy.deepcopy(manifest["variants"][0])
    elif damage == "hash":
        manifest["variants"][0]["fingerprint"] = "0" * 64
    elif damage == "unsafe":
        manifest["variants"][0]["config"] = "../config.json"
    elif damage == "unknown":
        manifest["extra"] = True
    elif damage == "core":
        manifest["variants"][0]["core_tensors"] *= 2
    else:
        manifest["variants"] = [v for v in manifest["variants"] if v["name"] != "original"]
    with pytest.raises(ValueError):
        mc.validate(manifest, FIXTURE)


def test_selected_view_does_not_read_other_views(tmp_path):
    manifest = json.loads((FIXTURE / mc.MANIFEST).read_text())
    for filename in mc.FILES["k8-compact"][:3]:
        shutil.copyfile(FIXTURE / filename, tmp_path / filename)
    mc.validate(manifest, tmp_path, "k8-compact")
    with pytest.raises(FileNotFoundError):
        mc.validate(manifest, tmp_path, "original")


def test_selected_metadata_tamper_and_shared_drift(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    manifest = json.loads((tmp_path / mc.MANIFEST).read_text())
    (tmp_path / "config.k8.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata digest"):
        mc.validate(manifest, tmp_path, "k8-compact")
    with pytest.raises(ValueError, match="missing"):
        mc.shared_inventory_digest({}, ["missing.weight"])


@pytest.mark.parametrize("family", ["glm5/mod.rs", "glm5_flash/mod.rs", "kimi_k3/serve.rs",
                                    "muse_spark/serve.rs", "step5/serve.rs",
                                    "minimax_m3/serve.rs", "minimax_m3/golden.rs"])
def test_resolved_core_identity_is_precision_only(family):
    """Every family publishes its core identity through the one shared publisher, which reports it
    only for checkpoints that declare views (#1755); those names stay precision-only for pairing."""
    source = (ROOT / "rust/crates/cuteafd-daemon/src/families" / family).read_text()
    cli = (ROOT / "rust/crates/cuteafd-daemon/src/cli.rs").read_text()
    bench = (ROOT / "rust/crates/cuteafd-bench/src/fidelity.rs").read_text()
    precision = re.search(r"const PRECISION_SETTINGS:.*?= &\[(.*?)\];", bench, re.S)
    assert precision is not None
    allowed = set(re.findall(r'"([^"]+)"', precision.group(1)))
    identity = {"core-view", "core-view-fingerprint", "core-fp8-recipe"}
    assert "crate::cli::publish_core_view(" in source, family
    assert not re.findall(r'set_resolved\("(core-[^"]+)"', source), family
    settings = cli.split("fn core_view_settings(")[1].split("\n}\n")[0]
    published = set(re.findall(r'\("(core-[^"]+)",', settings))
    assert published == identity, published
    assert published <= allowed, published - allowed
    assert "declared_view()" in settings
