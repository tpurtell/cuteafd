#!/usr/bin/env python3
"""CPU-only regressions for verifying an EXL3 root with nested families.

Release images keep one well-known `exl3/` root. Multi-family builds nest
`exl3-kXX` packages beneath it, while a direct `manifest.json` in the root is
the legacy single-family layout. `build.sh` verifies and checksums the exported
root, so both layouts must be handled, strictly:

* every `exl3-*` family must be verified with the unchanged strict
  single-package validation -- one valid package must never mask a broken
  sibling, and a family without a manifest is an error rather than skipped;
* a root mixing a flat manifest with family packages is rejected as ambiguous;
* a root with no package at all is an error.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "python" / "tools" / "aot" / "package_exl3_aot.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("package_exl3_aot", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("geometry", ["dsv4f", "dsv4p", "glm", "glmf", "qwen4"])
def test_generic_spark_profiles_cover_every_nonempty_whole_block_rank(geometry):
    tool = load_tool()
    intermediate = tool.GEOMETRIES[geometry][1]
    profiles = tool.profiles_for_role("spark", geometry)
    for tp in range(1, 9):
        ranks = {}
        for _name, width, _experts, _topk, dtype, destinations in profiles:
            assert dtype == "bf16"
            for destination in destinations:
                if destination.startswith(f"tp{tp}-rank"):
                    rank = int(destination.split("rank")[1])
                    assert rank not in ranks
                    ranks[rank] = width
        if tp > 6 or tp > intermediate // 128:
            assert not ranks
        else:
            assert sorted(ranks) == list(range(tp))
            assert sum(ranks.values()) == intermediate
            assert all(width > 0 and width % 128 == 0 for width in ranks.values())
            assert max(ranks.values()) - min(ranks.values()) <= 128


@pytest.mark.parametrize("geometry", ["dsv4f", "dsv4p", "glm", "glmf", "qwen4"])
def test_all_counts_are_opt_in_and_legacy_build_profiles_remain(geometry):
    tool = load_tool()
    legacy = tool.build_profiles("spark", geometry)
    worlds = {int(profile[0].split("-")[0][2:]) for profile in legacy}
    assert worlds == ({1, 2, 3, 4} if geometry == "qwen4" else {2, 3, 4, 6})
    assert tool.build_profiles("spark", geometry, all_spark_counts=True) == tool.profiles_for_role("spark", geometry)


def test_kimi_coordinator_exports_whole_and_tp2_fp32_latent_experts():
    tool = load_tool()
    assert tool.GEOMETRIES["kimik3"] == (3584, 3072, 896, 16)
    assert tool.activation("kimik3") == "situ"
    assert tool.build_profiles("coordinator", "kimik3") == [
        ("rtx-tp1", 3072, 896, 16, "fp32", ["rtx-tp1"]),
        ("rtx-tp2", 1536, 896, 16, "fp32", ["rtx-tp2"]),
    ]
    assert tool.parse_requested_layouts(["rtx-tp2"], "coordinator", "kimik3") == ["rtx-tp2"]
    assert tool.package_capacities("kimik3", "1,16,80,256,1024,4096") == [1, 80, 256, 1024, 4096]
    assert tool.build_profiles("spark", "kimik3") == [
        ("tp6-width512", 512, 896, 16, "bf16", [f"tp6-rank{rank}" for rank in range(6)]),
    ]


def _manifest(path: Path, role: str = "coordinator") -> None:
    path.write_text(json.dumps({"schema": "cuteafd.exl3-package.v1", "role": role}))


@pytest.fixture()
def recorded(monkeypatch):
    """Replace the strict single-package verifier with a recorder."""
    tool = load_tool()
    calls: list[Path] = []

    def fake_verify(package, revision=None, runtime=None, role=None):
        calls.append(Path(package))
        return {"role": "coordinator", "variants": [1]}

    monkeypatch.setattr(tool, "verify", fake_verify)
    return tool, calls


def test_legacy_flat_root_verifies_the_root_itself(tmp_path, recorded):
    tool, calls = recorded
    _manifest(tmp_path / "manifest.json")
    manifests = tool.verify_root(tmp_path)
    assert calls == [tmp_path]
    assert len(manifests) == 1


def test_nested_family_root_verifies_every_family(tmp_path, recorded):
    tool, calls = recorded
    for family in ("exl3-k23", "exl3-k34"):
        child = tmp_path / family
        child.mkdir()
        _manifest(child / "manifest.json")
    manifests = tool.verify_root(tmp_path)
    assert calls == [tmp_path / "exl3-k23", tmp_path / "exl3-k34"]
    assert len(manifests) == 2


def test_missing_family_manifest_fails_closed(tmp_path, recorded):
    """A broken sibling must fail the whole root, not be skipped."""
    tool, calls = recorded
    good = tmp_path / "exl3-k23"
    good.mkdir()
    _manifest(good / "manifest.json")
    (tmp_path / "exl3-k34").mkdir()  # no manifest.json
    with pytest.raises(ValueError, match="missing its manifest.json"):
        tool.verify_root(tmp_path)


def test_ambiguous_flat_and_nested_root_is_rejected(tmp_path, recorded):
    tool, _calls = recorded
    _manifest(tmp_path / "manifest.json")
    child = tmp_path / "exl3-k34"
    child.mkdir()
    _manifest(child / "manifest.json")
    with pytest.raises(ValueError, match="ambiguous EXL3 root"):
        tool.verify_root(tmp_path)


def test_unexpected_package_directory_is_rejected(tmp_path, recorded):
    tool, _calls = recorded
    child = tmp_path / "k34-not-a-family"
    child.mkdir()
    _manifest(child / "manifest.json")
    with pytest.raises(ValueError, match="unexpected EXL3 package directory"):
        tool.verify_root(tmp_path)


def test_root_without_any_package_is_an_error(tmp_path, recorded):
    tool, _calls = recorded
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no EXL3 package or family manifest"):
        tool.verify_root(tmp_path)


def test_missing_root_is_an_error(tmp_path, recorded):
    tool, _calls = recorded
    with pytest.raises(ValueError, match="no EXL3 package or family manifest"):
        tool.verify_root(tmp_path / "absent")


def test_build_sh_verifies_the_exported_root():
    text = (REPO / "build.sh").read_text()
    assert 'verify --package "$repo_root/dist/$role/exl3"' in text


def test_build_sh_checksums_real_exl3_manifest_paths():
    """The dist checksum list must not name a flat manifest that never exists."""
    text = (REPO / "build.sh").read_text()
    checksum_block = text.split("sha256sum \\", 1)[1].split(">SHA256SUMS", 1)[0]
    assert "coordinator/exl3/manifest.json" not in checksum_block
    assert "spark-expert/exl3/manifest.json" not in checksum_block
    assert '"${dist_exl3_manifests[@]}"' in checksum_block


def test_sm121_object_requires_actual_cuda_target(tmp_path):
    import struct
    tool = load_tool()
    blob = bytearray(64)
    blob[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', blob, 18, 190)
    struct.pack_into('<I', blob, 48, 0x06007902)
    path = tmp_path / 'kernel.o'
    path.write_bytes(b'host object prefix' + blob)
    tool.verify_object_target(path, (12, 1))
    with pytest.raises(ValueError, match='target differs'):
        tool.verify_object_target(path, (12, 0))
    struct.pack_into('<I', blob, 48, 0x06007802)
    path.write_bytes(blob)
    with pytest.raises(ValueError, match='target differs'):
        tool.verify_object_target(path, (12, 1))


def test_worker_target_is_explicit_and_failures_are_observed_unordered():
    source = TOOL.read_text()
    assert 'args, target_compute, raw' in source
    assert 'describe_gb10()' in source
    assert "meta['compute'] != list(target_compute)" in source
    assert 'as_completed(futures)' in source and 'pool.map(' not in source
    assert 'process.kill()' in source and 'process.join()' in source
    assert 'for group in groups:' in source
    assert source.index('for group in groups:') > source.index('process.join(timeout=5)')


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_verification_checks_every_packaged_dso_including_routes(tmp_path):
    import struct
    tool = load_tool()
    files = {}
    for name, compute in [('core.so', 121), ('routes/metadata.so', 120)]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytearray(64)
        data[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<H', data, 18, 190)
        struct.pack_into('<I', data, 48, compute << 8)
        path.write_bytes(data)
        files[name] = {'bytes': len(data), 'sha256': tool.digest(path)}
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'schema': 'cuteafd.exl3-package.v1', 'role': 'coordinator',
        'variants': [], 'files': files, 'compute': [12, 1]}))
    with pytest.raises(ValueError, match='metadata.so'):
        tool.verify(tmp_path)


@pytest.mark.parametrize("geometry,profile,bits", [
    ("glm", "tp6-compact-width384", [4, 5]),
    ("glmf", "tp4-width512", [3, 4]),
    ("qwen4", "tp4-compact-width256", [4, 5]),
    ("kimik3", "tp6-width512", [2, 3]),
])
def test_only_measured_spark_decode_profiles_default_to_pure_hint(geometry, profile, bits):
    tool = load_tool()
    measured = [1, 16, 80] if geometry in ("glm", "qwen4") else [1, 80]
    for capacity in measured:
        assert tool.default_decode_schedule(geometry, "spark", profile, capacity, bits) == tool.PURE_EVICT_FIRST
    for capacity in set([1, 16, 80, 256, 1024, 4096]) - set(measured):
        assert tool.default_decode_schedule(geometry, "spark", profile, capacity, bits) is None
    assert tool.default_decode_schedule(geometry, "coordinator", profile, 1, bits) is None
    assert tool.default_decode_schedule(geometry, "spark", profile + "-other", 1, bits) is None
    assert tool.default_decode_schedule(geometry, "spark", profile, 1, [2, 3, 4]) is None


@pytest.mark.parametrize("geometry,bits", [("glm", [4, 5]), ("glmf", [3, 4])])
def test_only_measured_coordinator_capacities_default_to_pure_hint(geometry, bits):
    tool = load_tool()
    for capacity in [1, 16, 80, 256, 1024, 4096]:
        expected = tool.PURE_EVICT_FIRST if capacity in [1, 80] else None
        assert tool.default_decode_schedule(geometry, "coordinator", "rtx-tp2", capacity, bits) == expected
        assert tool.default_decode_schedule(geometry, "coordinator", "rtx-tp1", capacity, bits) is None
        for other_bits in ([2, 3], [3, 4], [4, 5], [3, 4, 5]):
            if other_bits != bits:
                assert tool.default_decode_schedule(geometry, "coordinator", "rtx-tp2", capacity, other_bits) is None


@pytest.mark.parametrize("schedule,allowed", [(None, True), ("l2=2,pf1=0,pf2=0,pdl=1", True),
                                               ("l2=2,pf1=1,pf2=0,pdl=1", False)])
@pytest.mark.parametrize("explicit", [False, True])
def test_compact_package_accepts_only_pure_l2_schedule(tmp_path, monkeypatch, schedule, allowed, explicit):
    tool = load_tool()
    monkeypatch.setattr(tool, "verify_object_target", lambda *args: None)
    suffix = "-hint" if schedule and explicit else ""
    directory = f"tp6-compact-rank0/m80{suffix}"
    variant = dict(directory=directory, capacity=80, intermediate=384, experts=256,
                   top_k=8, output_dtype="bf16", bits=[4, 5], descriptor_rows=7, native_info_version=3)
    meta = {**variant, "hidden": 6144, "compute": [12, 1], "tile": [64, 128, 64, 128],
            "requires_route_preparation": False, "sparkinfer_revision": "test"}
    if schedule:
        variant["decode_schedule"] = schedule
        if explicit:
            variant["schedule"] = "hint"
        meta["decode_schedule"] = schedule
    dest = tmp_path / directory
    dest.mkdir(parents=True)
    (dest / "v41_exl3.json").write_text(json.dumps(meta))
    (dest / "trellis_lut.bin").write_bytes(b"test")
    (dest / "libcuteafd_exl3.so").write_bytes(b"test")
    files = {str(p.relative_to(tmp_path)): dict(bytes=p.stat().st_size, sha256=tool.digest(p))
             for p in dest.iterdir()}
    manifest = dict(schema="cuteafd.exl3-package.v1", role="spark", geometry="glm",
                    compute=[12, 1], sparkinfer_revision="test", variants=[variant], files=files)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if allowed:
        assert tool.verify(tmp_path) == manifest
        variant["decode_schedule"] = "mismatched-package-identity"
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="variant decode schedule mismatch"):
            tool.verify(tmp_path)
    else:
        with pytest.raises(ValueError, match="invalid compact geometry or execution contract"):
            tool.verify(tmp_path)
