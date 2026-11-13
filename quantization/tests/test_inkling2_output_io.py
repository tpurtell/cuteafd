import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2.output_io import (BACKEND_DIRECTORY, OPERATIONAL_MARKER,
    OutputNumpyProxy, _digest)
from inkling2 import output_io
from kimi_k3.uniform_k2.storage import atomic_json, file_identity, sha256_file


class Fixture:
    def __init__(self, tmp_path, *, legacy=(1, 3, 5), flat=True, layer=0):
        self.root = tmp_path / "task"
        self.layer = self.root / "routes" / f"layer-{layer:03d}"
        self.layer.mkdir(parents=True)
        self.path = self.layer / "outputs.bf16"
        self.identity = dict(frozen="numerics", recipe="K4-K8-K6")
        self.shape = (13, 8, 5)
        t, rank, column = np.indices(self.shape)
        # Opaque finite BF16 bit patterns, including sign and low mantissa bits.
        self.expected = (0x3f00 + t * 40 + rank * 5 + column).astype(np.uint16)
        self.expected[::2, :, 0] |= np.uint16(0x8000)
        self.routes = ((np.arange(13)[:, None] * 3 + np.arange(8)[None, :]) % 16).astype(np.int16)
        route_path = self.layer / "routes.i16"
        route_path.write_bytes(self.routes.tobytes())
        self.route_manifest = dict(identity=self.identity, tokens=13, width=5, experts=24,
            files={"routes.i16": dict(sha256=sha256_file(route_path), file_identity=file_identity(route_path))})
        atomic_json(self.layer / "manifest.json", self.route_manifest)
        if flat:
            old = np.full(self.shape, np.uint16(0x7fc0), dtype=np.uint16)  # poison uncommitted slots
            for expert in legacy:
                mask = self.routes == expert
                old[mask] = self.expected[mask]
            self.path.write_bytes(old.tobytes())
            self.binding()
            for expert in legacy:
                self.frozen_receipt(expert)
        self.flat_before = self.path.read_bytes() if flat else None
        self.flat_identity = file_identity(self.path) if flat else None
        self.proxy = OutputNumpyProxy(self.root, self.identity)

    def binding(self):
        atomic_json(self.path.with_suffix(".identity.json"), dict(identity=self.identity,
            route_manifest=_digest(self.route_manifest), file_identity=file_identity(self.path)))

    def frozen_receipt(self, expert):
        atomic_json(self.layer / f"expert-{expert:03d}.json", dict(identity=self.identity, expert=expert,
            natural_rows=int(np.any(self.routes == expert, axis=1).sum()),
            scatter_inode=self.path.stat().st_ino, route_manifest=_digest(self.route_manifest)))

    def writer(self, mode="r+"):
        return self.proxy.memmap(self.path, dtype=np.uint16, mode=mode, shape=self.shape)

    def rows(self, expert):
        rows, ranks = np.nonzero(self.routes == expert)
        return rows, ranks

    def append(self, writer, expert, *, chunk=3, commit=True):
        rows, ranks = self.rows(expert)
        for start in range(0, len(rows), chunk):
            selected, selected_ranks = rows[start:start + chunk], ranks[start:start + chunk]
            writer[selected, selected_ranks] = self.expected[selected, selected_ranks]
        writer.flush()
        if commit:
            self.frozen_receipt(expert)

    def fill(self, writer):
        for expert in range(24):
            if not (self.layer / f"expert-{expert:03d}.json").exists():
                self.append(writer, expert)

    def reader(self):
        return self.proxy.memmap(self.path, dtype=np.uint16, mode="r", shape=self.shape)


def bf16_rank_sum(values):
    # Independent BF16 decoding and the frozen runner's eight FP32 additions.
    decoded = (values.astype(np.uint32) << 16).view(np.float32)
    result = np.zeros((len(values), values.shape[-1]), dtype=np.float32)
    for rank in range(8):
        result = result + decoded[:, rank]
    return result


def test_mixed_legacy_and_new_exact_slots_bytes_and_rank_sum(tmp_path):
    f = Fixture(tmp_path)
    assert not (f.root / OPERATIONAL_MARKER).exists()  # activation is lazy
    writer = f.writer()
    f.fill(writer)
    writer._mmap.close()
    writer._mmap.close()
    assert f.path.read_bytes() == f.flat_before
    assert file_identity(f.path) == f.flat_identity
    manifest = json.loads((f.layer / BACKEND_DIRECTORY / "manifest.json").read_text())
    assert set(manifest["legacy_receipts"]) == {"1", "3", "5"}
    assert manifest["code_sha256"] == json.loads((f.root / OPERATIONAL_MARKER).read_text())["code_sha256"]
    provenance = json.loads((f.root / "output-io-provenance/layer-000.json").read_text())
    assert provenance["backend_manifest_sha256"] == _digest(manifest)
    assert provenance["legacy_experts"] == [1, 3, 5]
    assert provenance["legacy_receipts_sha256"] == manifest["legacy_receipts_sha256"]
    reader = f.reader()
    for selection in (slice(0, 1), slice(1, 6), slice(6, 13), slice(0, 13), slice(4, 4)):
        actual, expected = reader[selection], f.expected[selection]
        assert actual.tobytes() == expected.tobytes()
        assert bf16_rank_sum(actual).tobytes() == bf16_rank_sum(expected).tobytes()
    value = reader[0:3]
    value.fill(0)
    assert reader[0:3].tobytes() == f.expected[0:3].tobytes()  # owned result
    reader._mmap.close()
    reader._mmap.close()
    assert file_identity(f.path) == f.flat_identity


def test_new_layer_sparse_placeholder_and_zero_route_experts(tmp_path):
    f = Fixture(tmp_path, legacy=(), flat=False, layer=1)
    writer = f.writer("w+")
    assert f.path.stat().st_size == np.prod(f.shape) * 2
    # NumPy may allocate its final page when extending the sparse placeholder.
    # Neither expert appends nor prompt reads may dirty any additional pages.
    allocated_before = f.path.stat().st_blocks
    f.binding()  # exactly what the frozen runner does after np.memmap returns
    before = file_identity(f.path)
    f.fill(writer)
    assert not (f.layer / BACKEND_DIRECTORY / "expert-023.bf16").exists()
    writer._mmap.close()
    reader = f.reader()
    assert reader[:].tobytes() == f.expected.tobytes()
    reader.close()
    assert file_identity(f.path) == before
    assert f.path.stat().st_blocks == allocated_before
    assert not json.loads((f.layer / BACKEND_DIRECTORY / "manifest.json").read_text())["legacy_receipts"]


def test_frozen_cumsum_numpy_integer_prompt_bounds(tmp_path):
    f = Fixture(tmp_path)
    writer = f.writer()
    f.fill(writer)
    writer.close()
    reader = f.reader()
    bounds = f.proxy.cumsum([0, 1, 5, 7])
    assert isinstance(bounds[0], np.integer)
    for index in range(len(bounds) - 1):
        actual = reader[bounds[index]:bounds[index + 1]:np.int64(1)]
        expected = f.expected[bounds[index]:bounds[index + 1]]
        assert actual.tobytes() == expected.tobytes()
        assert bf16_rank_sum(actual).tobytes() == bf16_rank_sum(expected).tobytes()
    for invalid in (slice(True, 3), slice(0, np.bool_(True)), slice(0, 3, True),
                    slice(np.bool_(False), 3), slice(0, 3, np.bool_(True)),
                    slice(0.0, 3), slice(0, 3.0), slice(0, 3, 1.0),
                    slice(np.int64(-1), 3), slice(0, np.int64(14)), slice(0, 3, np.int64(2))):
        with pytest.raises(ValueError):
            reader[invalid]
    reader.close()


def test_chunk_values_are_copied_when_appended(tmp_path):
    f = Fixture(tmp_path, legacy=())
    writer = f.writer()
    rows, ranks = f.rows(0)
    values = f.expected[rows, ranks].copy()
    writer[rows, ranks] = values
    values.fill(0)
    writer.flush()
    payload = f.layer / BACKEND_DIRECTORY / "expert-000.bf16"
    assert payload.read_bytes() == f.expected[rows, ranks].tobytes()
    writer.close()


def test_partial_flush_never_commits_and_partial_scratch_can_restart(tmp_path):
    f = Fixture(tmp_path, legacy=())
    writer = f.writer()
    rows, ranks = f.rows(0)
    writer[rows[:1], ranks[:1]] = f.expected[rows[:1], ranks[:1]]
    with pytest.raises(ValueError, match="partial expert"):
        writer.flush()
    writer.close()
    assert not (f.layer / BACKEND_DIRECTORY / "expert-000.json").exists()
    writer = f.writer()
    f.fill(writer)
    writer.close()
    reader = f.reader()
    assert reader[:].tobytes() == f.expected.tobytes()
    reader.close()


def test_orphan_commit_replays_exact_bytes_without_replacing_payload(tmp_path):
    f = Fixture(tmp_path, legacy=())
    writer = f.writer()
    f.append(writer, 0, commit=False)
    writer.close()
    payload = f.layer / BACKEND_DIRECTORY / "expert-000.bf16"
    receipt = payload.with_suffix(".json")
    before = (file_identity(payload), receipt.read_bytes())
    writer = f.writer()
    f.append(writer, 0, chunk=1)
    assert (file_identity(payload), receipt.read_bytes()) == before
    f.fill(writer)
    writer.close()
    reader = f.reader()
    assert reader[:].tobytes() == f.expected.tobytes()
    reader.close()


def test_orphan_commit_rejects_regenerated_numerical_difference(tmp_path):
    f = Fixture(tmp_path, legacy=())
    writer = f.writer()
    f.append(writer, 0, commit=False)
    writer.close()
    writer = f.writer()
    rows, ranks = f.rows(0)
    wrong = f.expected[rows, ranks].copy()
    wrong[0, 0] ^= 1
    with pytest.raises(ValueError, match="orphan output bytes differ"):
        writer[rows, ranks] = wrong
    writer.close()
    assert not (f.layer / "expert-000.json").exists()


def test_payload_fsync_precedes_commit_receipt_and_checksums_written_bytes(tmp_path, monkeypatch):
    f = Fixture(tmp_path, legacy=())
    writer = f.writer()
    events = []
    real_fsync, real_atomic = output_io.os.fsync, output_io.atomic_json
    def fsync(fd):
        events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        return real_fsync(fd)
    def atomic(path, value):
        real_atomic(path, value)
        events.append(("receipt", str(path)))
    monkeypatch.setattr(output_io.os, "fsync", fsync)
    monkeypatch.setattr(output_io, "atomic_json", atomic)
    f.append(writer, 0, commit=False)
    payload = f.layer / BACKEND_DIRECTORY / "expert-000.bf16"
    receipt = json.loads(payload.with_suffix(".json").read_text())
    import xxhash
    assert receipt["xxh3_128"] == xxhash.xxh3_128(payload.read_bytes()).hexdigest()
    assert 'sha256' not in receipt
    data_sync = next(n for n, (kind, path) in enumerate(events) if kind == "fsync" and path.endswith("expert-000.partial"))
    proof = next(n for n, (kind, path) in enumerate(events) if kind == "receipt" and path.endswith("expert-000.json"))
    assert data_sync < proof
    writer.close()


def test_crash_after_payload_rename_before_receipt_rewrites_only_unreferenced_file(tmp_path, monkeypatch):
    f = Fixture(tmp_path, legacy=())
    writer = f.writer()
    real_atomic = output_io.atomic_json
    def interrupted(path, value):
        if Path(path).name == "expert-000.json":
            raise RuntimeError("simulated interrupted receipt commit")
        return real_atomic(path, value)
    monkeypatch.setattr(output_io, "atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="interrupted receipt"):
        f.append(writer, 0, commit=False)
    writer.close()
    payload = f.layer / BACKEND_DIRECTORY / "expert-000.bf16"
    assert payload.exists() and not payload.with_suffix(".json").exists()
    monkeypatch.setattr(output_io, "atomic_json", real_atomic)
    writer = f.writer()
    f.fill(writer)
    writer.close()
    reader = f.reader()
    assert reader[:].tobytes() == f.expected.tobytes()
    reader.close()


def test_missing_or_partial_slots_cannot_be_read(tmp_path):
    f = Fixture(tmp_path)
    with pytest.raises(ValueError, match="not been activated"):
        f.reader()
    assert not (f.root / OPERATIONAL_MARKER).exists()
    assert not (f.layer / BACKEND_DIRECTORY).exists()
    writer = f.writer()
    f.append(writer, 0)
    writer.close()
    with pytest.raises(ValueError, match="uncommitted expert slots"):
        f.reader()


@pytest.mark.parametrize("case", ["payload", "frozen_receipt", "route", "flat", "manifest", "marker"])
def test_corruption_or_identity_change_fails_closed(tmp_path, case):
    f = Fixture(tmp_path)
    writer = f.writer()
    f.fill(writer)
    writer.close()
    if case == "payload":
        path = f.layer / BACKEND_DIRECTORY / "expert-000.bf16"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    elif case == "frozen_receipt":
        path = f.layer / "expert-001.json"
        saved = json.loads(path.read_text())
        saved["natural_rows"] += 1
        atomic_json(path, saved)
    elif case == "route":
        path = f.layer / "routes.i16"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    elif case == "flat":
        with f.path.open("r+b") as stream:
            stream.write(b"\0\0")
    else:
        path = f.layer / BACKEND_DIRECTORY / "manifest.json" if case == "manifest" else f.root / OPERATIONAL_MARKER
        saved = json.loads(path.read_text())
        saved["code_sha256"] = "0" * 64
        atomic_json(path, saved)
    with pytest.raises(ValueError):
        f.reader()


def test_legacy_snapshot_is_not_reinterpreted_after_new_receipts(tmp_path):
    f = Fixture(tmp_path)
    writer = f.writer()
    f.append(writer, 0)
    writer.close()
    manifest = (f.layer / BACKEND_DIRECTORY / "manifest.json").read_bytes()
    writer = f.writer()
    assert (f.layer / BACKEND_DIRECTORY / "manifest.json").read_bytes() == manifest
    assert writer.legacy_ids == {1, 3, 5}  # new frozen receipt0 is not legacy
    with pytest.raises(ValueError, match="frozen committed"):
        rows, ranks = f.rows(0)
        writer[rows, ranks] = f.expected[rows, ranks]
    writer.close()


@pytest.mark.parametrize("case", ["order", "duplicate", "mixed_experts", "wrong_rank", "dtype", "shape", "bounds"])
def test_invalid_assignment_does_not_publish(tmp_path, case):
    f = Fixture(tmp_path, legacy=())
    writer = f.writer()
    rows, ranks = f.rows(0)
    values = f.expected[rows, ranks].copy()
    if case == "order":
        rows, ranks, values = rows[::-1], ranks[::-1], values[::-1]
    elif case == "duplicate":
        rows[1], ranks[1] = rows[0], ranks[0]
    elif case == "mixed_experts":
        ranks[0] = (ranks[0] + 1) % 8
    elif case == "wrong_rank":
        ranks[:] = 8
    elif case == "dtype":
        values = values.astype(np.int16)
    elif case == "shape":
        values = values[:, :-1]
    else:
        rows[0] = 13
    with pytest.raises(ValueError):
        writer[rows, ranks] = values
    writer.close()
    assert not list((f.layer / BACKEND_DIRECTORY).glob("expert-*.json"))


def test_unknown_paths_geometry_and_symlinks_are_rejected(tmp_path):
    f = Fixture(tmp_path)
    paths = [tmp_path / "other" / "outputs.bf16", f.root / "routes" / "layer-048" / "outputs.bf16",
             f.root / "routes" / "wrong" / "outputs.bf16"]
    for path in paths:
        with pytest.raises(ValueError, match="scope"):
            f.proxy.memmap(path, dtype=np.uint16, mode="w+", shape=f.shape)
    for kwargs in (dict(dtype=np.float16), dict(shape=(13, 7, 5)), dict(offset=2), dict(order="F"), dict(mode="c")):
        with pytest.raises(ValueError, match="unsupported"):
            f.proxy.memmap(f.path, **{**dict(dtype=np.uint16, mode="r+", shape=f.shape), **kwargs})
    link = f.root / "routes" / "layer-002"
    link.symlink_to(f.layer)
    with pytest.raises(ValueError, match="scope"):
        f.proxy.memmap(link / "outputs.bf16", dtype=np.uint16, shape=f.shape)


def test_only_run_proxy_is_changed_and_other_numpy_calls_delegate(tmp_path):
    f = Fixture(tmp_path)
    assert f.proxy.memmap is not np.memmap
    assert f.proxy.asarray is np.asarray
    assert f.proxy.uint16 is np.uint16
    raw = tmp_path / "not-output.i16"
    mapping = f.proxy.memmap(raw, dtype=np.int16, mode="w+", shape=(2,))
    assert isinstance(mapping, np.memmap)
    mapping._mmap.close()
    assert not (f.root / OPERATIONAL_MARKER).exists()


def test_guard_runs_before_any_activation_state_and_concurrent_writer_is_rejected(tmp_path):
    f = Fixture(tmp_path)
    def guard():
        raise RuntimeError("producer lock unavailable")
    proxy = OutputNumpyProxy(f.root, f.identity, activation_guard=guard)
    with pytest.raises(RuntimeError, match="producer lock"):
        proxy.memmap(f.path, dtype=np.uint16, shape=f.shape)
    assert not (f.root / OPERATIONAL_MARKER).exists()
    writer = f.writer()
    with pytest.raises(ValueError, match="already in use"):
        f.writer()
    writer.close()
