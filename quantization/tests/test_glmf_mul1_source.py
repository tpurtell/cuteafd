"""GLM Flash MUL1 source pin, payload accounting, and drafter-width witnesses."""
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.mul1_mc import source


@pytest.mark.parametrize("bits", [6, 8])
def test_core_and_mtp_ride_selected_width_and_share_native_eh(bits):
    modules = source.core_modules(bits)
    prefix = "model.language_model.layers.45."
    assert len(modules) == 323
    assert modules[prefix + "self_attn.q_b_proj"]["bits"] == bits
    assert prefix + "eh_proj" not in modules
    assert all(g["bits"] == bits for g in modules.values())
    assert prefix + "self_attn.kv_b_proj" not in modules
    assert prefix + "self_attn.indexer.wq_b" not in modules


def test_payloads_store_experts_only_once():
    assert source.payloads() == dict(shared_expert_bytes=156_283_454_592,
        retained_native_bytes=3_457_792_888, expert_modules=37_152, core_modules_per_view=323,
        view_payload_bytes={"k6-compact": 165_703_606_020, "k8-compact": 167_688_822_532,
                            "original": 175_622_979_576}, package_payload_bytes=189_532_913_168)


def test_cache_inventory_binds_asset_content_and_expected_oid(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / source.BASE_REVISION
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    asset = b"123"
    oid = hashlib.sha1(b"blob 3\0" + asset).hexdigest()
    (blobs / oid).write_bytes(asset)
    (snapshot / "config.json").symlink_to("../../blobs/" + oid)
    manifest = {"config.json": {"bytes": 3, "oid": oid}}
    with pytest.raises(ValueError, match="OIDs differ"):
        source.cache_manifest(snapshot)
    digest = hashlib.sha256(source.canonical(manifest).encode()).hexdigest()
    monkeypatch.setattr(source, "SOURCE_MANIFEST_SHA256", digest)
    assert source.cache_manifest(snapshot) == manifest
    (blobs / oid).write_bytes(b"456")
    with pytest.raises(ValueError, match="asset content differs"):
        source.cache_manifest(snapshot)
