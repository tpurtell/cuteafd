"""Sequential expert-output storage behind the frozen runner's NumPy API.

Install ``OutputNumpyProxy`` on ``inkling2.run.np`` only, then call that imported
module's ``main`` directly. The caller must hold the frozen producer lock and
quiesce the legacy producer before the first intercepted memmap call. Merely
constructing the proxy does not snapshot or modify an active layer.

The legacy flat output remains an immutable inode/extent placeholder. New
results are appended in natural-row order to one contiguous file per expert;
prompt reads combine only legacy-committed slots with new committed payloads.
No numerical file, projection receipt or frozen expert receipt is rewritten.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import operator
from pathlib import Path
import re

import numpy as np
import xxhash

from kimi_k3.uniform_k2.storage import (atomic_json, canonical, file_identity,
    fsync_directory, sha256_file, verify_receipt)

SCHEMA = "inkling2-sequential-expert-outputs-v2"
BACKEND_DIRECTORY = "output-sequential"
OPERATIONAL_MARKER = "output-io-code.json"


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_sha256():
    return sha256_file(Path(__file__))


def _plain(path):
    if path.is_symlink():
        raise ValueError("refuse symlink output storage")


def _fingerprint(path):
    stat = path.stat()
    return dict(file_identity(path), device=stat.st_dev)


def _rows_hash(rows):
    return xxhash.xxh3_128(rows.astype("<i8", copy=False).tobytes()).hexdigest()


def _json(path):
    _plain(path)
    return json.loads(path.read_text())


class OutputNumpyProxy:
    """Delegate NumPy except the exact task/layer output memmap calls.

    ``activation_guard`` may assert the wrapper's outer producer-lock lease.
    It runs before every intercepted open, before any backend state is touched.
    Other memmaps delegate unchanged, including the separate RouteSpool module.
    """

    def __init__(self, task_root, identity, *, activation_guard=None):
        root = Path(task_root).expanduser().absolute()
        if root.resolve() != root:
            raise ValueError("output task root must be a real, absolute path")
        self.root, self.identity = root, json.loads(canonical(identity))
        self.activation_guard = activation_guard

    def __getattr__(self, name):
        return getattr(np, name)

    def memmap(self, filename, dtype=np.uint8, mode="r+", offset=0, shape=None, order="C"):
        if not isinstance(filename, (str, bytes, os.PathLike)):
            return np.memmap(filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order=order)
        path = Path(os.fsdecode(filename)).expanduser().absolute()
        if path.name != "outputs.bf16":
            return np.memmap(filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order=order)
        relative = path.relative_to(self.root) if self.root in path.parents else None
        if (relative is None or len(relative.parts) != 3 or relative.parts[0] != "routes"
                or not re.fullmatch(r"layer-(?:00[0-9]|0[1-3][0-9]|04[0-7])", relative.parts[1])
                or path.resolve() != path):
            raise ValueError("output memmap path outside the exact Inkling task/layer scope")
        if (np.dtype(dtype) != np.dtype(np.uint16) or mode not in ("r", "r+", "w+")
                or offset != 0 or order != "C" or not isinstance(shape, (tuple, list))
                or len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
                or shape[1] != 8):
            raise ValueError("unsupported frozen output dtype/mode/shape/offset/order")
        if self.activation_guard is not None:
            self.activation_guard()
        return _Reader(self, path, tuple(shape)) if mode == "r" else _Writer(self, path, tuple(shape), mode)

    def _record_code(self):
        path = self.root / OPERATIONAL_MARKER
        expected = dict(schema=SCHEMA, code_sha256=code_sha256(), identity=self.identity,
            payload_checksum='xxh3-128', checksum_seed=0,
            activation_requirement="frozen-producer-lock-held-and-legacy-writer-quiescent")
        if path.exists():
            if _json(path) != expected:
                raise ValueError("output I/O operational code/identity marker differs")
        else:
            atomic_json(path, expected)
        return expected


class _Backend:
    def __init__(self, proxy, path, shape, *, writable, mode="r"):
        self.proxy, self.path, self.shape = proxy, path, shape
        self.dtype, self.closed, self._mmap = np.dtype(np.uint16), False, self
        self.directory = path.parent / BACKEND_DIRECTORY
        self.routes = self.legacy = None
        self.payload_maps = {}
        self.lock = None
        self.manifest = None
        try:
            _plain(path)
            _plain(self.directory)
            if not writable and not (self.directory / "manifest.json").exists():
                raise ValueError("output backend has not been activated")
            self.directory.mkdir(exist_ok=True)
            lock_path = self.directory / "lock"
            _plain(lock_path)
            self.lock = lock_path.open("a+b")
            try:
                fcntl.flock(self.lock.fileno(), (fcntl.LOCK_EX if writable else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("output backend already in use") from error
            marker = proxy._record_code()
            route_manifest = _json(path.parent / "manifest.json")
            if (route_manifest.get("identity") != proxy.identity
                    or route_manifest.get("tokens") != shape[0] or route_manifest.get("width") != shape[2]
                    or type(route_manifest.get("experts")) is not int
                    or not 8 <= route_manifest["experts"] <= 32767):
                raise ValueError("output geometry/identity differs from immutable route spool")
            self.experts = route_manifest["experts"]
            self.route_digest = _digest(route_manifest)
            route_path = path.parent / "routes.i16"
            _plain(route_path)
            route_receipt = route_manifest["files"]["routes.i16"]
            verify_receipt(route_path, route_receipt, checksum=True)
            if route_path.stat().st_size != shape[0] * 8 * 2:
                raise ValueError("output route extent differs")
            self.route_fingerprint = _fingerprint(route_path)
            self.routes = np.memmap(route_path, mode="r", dtype=np.int16, shape=shape[:2])
            self.natural = self._natural_rows()
            if not path.exists():
                if not writable or mode != "w+" or any(path.parent.glob("expert-*.json")):
                    raise ValueError("missing immutable legacy output placeholder")
                # ftruncate-backed sparse placeholder; no scatter pages are dirtied.
                placeholder = np.memmap(path, mode="w+", dtype=np.uint16, shape=shape)
                placeholder._mmap.close()
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                fsync_directory(path.parent)
            if path.stat().st_size != int(np.prod(shape)) * 2:
                raise ValueError("legacy output placeholder extent differs")
            frozen_receipts = self._frozen_receipts()
            binding_path = path.with_suffix(".identity.json")
            if binding_path.exists():
                binding = _json(binding_path)
                self._binding_valid(binding)
            else:
                if not writable or mode != "w+" or frozen_receipts:
                    raise ValueError("legacy scatter binding missing")
                binding = dict(identity=proxy.identity, route_manifest=self.route_digest,
                               file_identity=file_identity(path))
            manifest_path = self.directory / "manifest.json"
            if manifest_path.exists():
                self.manifest = _json(manifest_path)
                expected = dict(schema=SCHEMA, code_sha256=marker["code_sha256"], identity=proxy.identity,
                    shape=list(shape), dtype="uint16", route_manifest=self.route_digest,
                    route_file=self.route_fingerprint, legacy_file=_fingerprint(path), binding_sha256=_digest(binding))
                if any(self.manifest.get(k) != value for k, value in expected.items()):
                    raise ValueError("immutable output backend manifest differs")
                legacy = self.manifest["legacy_receipts"]
                if self.manifest.get("legacy_receipts_sha256") != _digest(legacy):
                    raise ValueError("legacy expert receipt inventory digest differs")
                for expert, receipt in legacy.items():
                    if frozen_receipts.get(expert) != receipt:
                        raise ValueError("legacy committed expert metadata changed")
            else:
                if not writable:
                    raise ValueError("output backend has not been activated")
                if set(p.name for p in self.directory.iterdir()) != {"lock"}:
                    raise ValueError("unowned files in a new output backend")
                self.manifest = dict(schema=SCHEMA, code_sha256=marker["code_sha256"], identity=proxy.identity,
                    shape=list(shape), dtype="uint16", route_manifest=self.route_digest,
                    route_file=self.route_fingerprint, legacy_file=_fingerprint(path), binding_sha256=_digest(binding),
                    legacy_receipts=frozen_receipts, legacy_receipts_sha256=_digest(frozen_receipts))
                atomic_json(manifest_path, self.manifest)
            self.backend_digest = _digest(self.manifest)
            self._record_layer_provenance()
            self.legacy_ids = {int(n) for n in self.manifest["legacy_receipts"]}
            for expert in frozen_receipts:
                if int(expert) not in self.legacy_ids and len(self.natural[int(expert)]):
                    if self._payload_receipt(int(expert)) is None:
                        raise ValueError("frozen expert receipt references missing sequential output")
            self.frozen_receipts = frozen_receipts
        except BaseException:
            self.close()
            raise

    def _record_layer_provenance(self):
        directory = self.proxy.root / "output-io-provenance"
        _plain(directory)
        path = directory / (self.path.parent.name + ".json")
        expected = dict(schema=SCHEMA, code_sha256=self.manifest["code_sha256"],
            identity_sha256=_digest(self.proxy.identity), layer=self.path.parent.name,
            backend_manifest_sha256=self.backend_digest, shape=list(self.shape),
            route_manifest=self.route_digest, legacy_file=self.manifest["legacy_file"],
            legacy_experts=sorted(int(n) for n in self.manifest["legacy_receipts"]),
            legacy_receipts_sha256=self.manifest["legacy_receipts_sha256"])
        if path.exists():
            if _json(path) != expected:
                raise ValueError("persistent output I/O layer provenance differs")
        else:
            atomic_json(path, expected)

    def _natural_rows(self):
        pieces = [[] for _ in range(self.experts)]
        for start in range(0, self.shape[0], 8192):
            values = self.routes[start:start + 8192]
            if (np.any(values < 0) or np.any(values >= self.experts)
                    or np.any(np.diff(np.sort(values, axis=1), axis=1) == 0)):
                raise ValueError("invalid immutable natural expert routes")
            for expert in np.unique(values):
                pieces[expert].append(np.flatnonzero(np.any(values == expert, axis=1)) + start)
        return [np.concatenate(rows) if rows else np.empty(0, dtype=np.int64) for rows in pieces]

    def _binding_valid(self, binding):
        if (binding.get("identity") != self.proxy.identity or binding.get("route_manifest") != self.route_digest
                or binding.get("file_identity", {}).get("inode") != self.path.stat().st_ino):
            raise ValueError("legacy scatter binding differs")

    def _frozen_receipts(self):
        result = {}
        for path in self.path.parent.glob("expert-*.json"):
            match = re.fullmatch(r"expert-([0-9]{3})\.json", path.name)
            if not match or int(match[1]) >= self.experts:
                raise ValueError("unreviewed frozen expert receipt name")
            expert, saved = int(match[1]), _json(path)
            if (saved.get("identity") != self.proxy.identity or saved.get("expert") != expert
                    or saved.get("natural_rows") != len(self.natural[expert])
                    or saved.get("scatter_inode") != self.path.stat().st_ino
                    or saved.get("route_manifest") != self.route_digest):
                raise ValueError("frozen expert scatter receipt differs")
            result[str(expert)] = saved
        return result

    def _paths(self, expert):
        return (self.directory / f"expert-{expert:03d}.bf16",
                self.directory / f"expert-{expert:03d}.json",
                self.directory / f"expert-{expert:03d}.partial")

    def _payload_receipt(self, expert):
        payload, receipt_path, _ = self._paths(expert)
        _plain(payload)
        _plain(receipt_path)
        if not receipt_path.exists():
            return None
        saved = _json(receipt_path)
        expected = dict(schema=SCHEMA, backend=self.backend_digest, expert=expert,
            natural_rows=len(self.natural[expert]), row_order_xxh3_128=_rows_hash(self.natural[expert]),
            shape=[len(self.natural[expert]), self.shape[2]], dtype="uint16")
        if (any(saved.get(k) != v for k, v in expected.items())
                or not re.fullmatch(r"[0-9a-f]{32}", saved.get("xxh3_128", ""))):
            raise ValueError("sequential output receipt geometry/identity differs")
        verify_receipt(payload, saved)
        if payload.stat().st_size != len(self.natural[expert]) * self.shape[2] * 2:
            raise ValueError("sequential output payload extent differs")
        return saved

    def _unchanged(self):
        if self.closed:
            raise ValueError("output backend is closed")
        if (_fingerprint(self.path) != self.manifest["legacy_file"]
                or _fingerprint(self.path.parent / "routes.i16") != self.route_fingerprint):
            raise ValueError("immutable legacy output or routes changed")

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        for mapping in getattr(self, "payload_maps", {}).values():
            mapping._mmap.close()
        for name in ("routes", "legacy"):
            mapping = getattr(self, name, None)
            if mapping is not None:
                mapping._mmap.close()
                setattr(self, name, None)
        if self.lock is not None:
            self.lock.close()
            self.lock = None


class _Writer(_Backend):
    def __init__(self, proxy, path, shape, mode):
        self.active, self.stream, self.replay = None, None, None
        self.count = 0
        super().__init__(proxy, path, shape, writable=True, mode=mode)

    def __setitem__(self, key, values):
        self._unchanged()
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("sequential output writes require paired natural rows and ranks")
        rows, ranks = (np.asarray(v) for v in key)
        if (rows.ndim != 1 or ranks.shape != rows.shape or not len(rows)
                or rows.dtype.kind not in "iu" or ranks.dtype.kind not in "iu"
                or np.any(rows < 0) or np.any(rows >= self.shape[0])
                or np.any(ranks < 0) or np.any(ranks >= 8)
                or not isinstance(values, np.ndarray) or values.dtype != np.uint16
                or values.shape != (len(rows), self.shape[2])):
            raise ValueError("invalid sequential output rows/ranks/dtype/shape")
        experts = np.asarray(self.routes[rows, ranks])
        expert = int(experts[0])
        if np.any(experts != expert) or expert in self.legacy_ids:
            raise ValueError("write must belong to one new expert, never a committed legacy slot")
        if self.active is None:
            self._start(expert)
        if self.active != expert or not np.array_equal(rows, self.natural[expert][self.count:self.count + len(rows)]):
            raise ValueError("expert outputs must append the exact ascending natural-row inventory")
        raw = np.ascontiguousarray(values).tobytes()
        if self.replay is not None:
            if self.replay.read(len(raw)) != raw:
                raise ValueError("regenerated orphan output bytes differ from committed payload")
        else:
            if self.stream.write(raw) != len(raw):
                raise OSError("short sequential expert-output write")
        self.hasher.update(raw)
        self.count += len(rows)

    def _start(self, expert):
        frozen = self.path.parent / f"expert-{expert:03d}.json"
        if frozen.exists():
            raise ValueError("refuse to rewrite a frozen committed expert output")
        payload, _, partial = self._paths(expert)
        _plain(partial)
        saved = self._payload_receipt(expert)
        if saved is not None:
            # A crash after new receipt but before frozen receipt is a replay,
            # not permission to overwrite durable output under the same identity.
            self.replay = payload.open("rb")
            self.replay_saved = saved
        else:
            # Only names owned by this backend and unreferenced by any frozen
            # receipt may be replaced. Interrupted .partial files are not state.
            if partial.exists():
                partial.unlink()
            if payload.exists():
                payload.unlink()
            self.stream = partial.open("xb")
        self.active, self.count, self.hasher = expert, 0, xxhash.xxh3_128()

    def flush(self):
        self._unchanged()
        if self.active is None:
            return
        expert = self.active
        if self.count != len(self.natural[expert]):
            raise ValueError("partial expert output is uncommitted; complete natural rows required")
        if self.stream is None and self.replay is None:
            raise ValueError("interrupted output commit must be diagnosed and resumed")
        payload, receipt_path, partial = self._paths(expert)
        checksum = self.hasher.hexdigest()
        if self.replay is not None:
            if self.replay.read(1) or checksum != self.replay_saved["xxh3_128"]:
                raise ValueError("regenerated orphan output checksum differs")
            self.replay.close()
            self.replay = None
        else:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None
            os.replace(partial, payload)
            fsync_directory(self.directory)
            atomic_json(receipt_path, dict(schema=SCHEMA, backend=self.backend_digest, expert=expert,
                natural_rows=self.count, row_order_xxh3_128=_rows_hash(self.natural[expert]),
                shape=[self.count, self.shape[2]], dtype="uint16", xxh3_128=checksum,
                file_identity=file_identity(payload)))
        self.active, self.count = None, 0

    def close(self):
        # A finally/close during a partial expert must never publish its bytes.
        for name in ("stream", "replay"):
            stream = getattr(self, name, None)
            if stream is not None:
                stream.close()
                setattr(self, name, None)
        super().close()


class _Reader(_Backend):
    def __init__(self, proxy, path, shape):
        super().__init__(proxy, path, shape, writable=False)
        try:
            self.receipts = {}
            for expert, rows in enumerate(self.natural):
                if not len(rows) or expert in self.legacy_ids:
                    continue
                receipt = self._payload_receipt(expert)
                if str(expert) not in self.frozen_receipts or receipt is None:
                    raise ValueError("prompt output has uncommitted expert slots")
                self.receipts[expert] = receipt
            if self.legacy_ids:
                self.legacy = np.memmap(path, dtype=np.uint16, mode="r", shape=shape)
        except BaseException:
            self.close()
            raise

    def __getitem__(self, selection):
        self._unchanged()
        if not isinstance(selection, slice):
            raise ValueError("prompt output reads require a bounded unit-step token slice")
        def integer(value, default):
            if value is None:
                return default
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("prompt output slice bounds must be integers, not booleans")
            try:
                return operator.index(value)
            except TypeError as error:
                raise ValueError("prompt output slice bounds must be integers") from error
        start = integer(selection.start, 0)
        stop = integer(selection.stop, self.shape[0])
        step = integer(selection.step, 1)
        if step != 1 or not 0 <= start <= self.shape[0] or not 0 <= stop <= self.shape[0]:
            raise ValueError("prompt output reads require a bounded unit-step token slice")
        if stop < start:
            raise ValueError("reversed prompt output slice")
        values = np.empty((stop - start, 8, self.shape[2]), dtype=np.uint16)
        filled = np.zeros((stop - start, 8), dtype=bool)
        routes = np.asarray(self.routes[start:stop])
        for expert in np.unique(routes):
            expert = int(expert)
            row, rank = np.nonzero(routes == expert)
            if expert in self.legacy_ids:
                values[row, rank] = self.legacy[start + row, rank]
            else:
                receipt = self.receipts[expert]
                payload = self._paths(expert)[0]
                verify_receipt(payload, receipt)
                if expert not in self.payload_maps:
                    self.payload_maps[expert] = np.memmap(payload, dtype=np.uint16, mode="r",
                                                        shape=(len(self.natural[expert]), self.shape[2]))
                left, right = np.searchsorted(self.natural[expert], [start, stop])
                selected = self.natural[expert][left:right] - start
                if not np.array_equal(selected, row):
                    raise ValueError("prompt route inventory differs from sequential output rows")
                values[row, rank] = self.payload_maps[expert][left:right]
            if np.any(filled[row, rank]):
                raise ValueError("duplicate prompt expert-output slot")
            filled[row, rank] = True
        if not filled.all():
            raise ValueError("prompt expert-output slots are incomplete")
        self._unchanged()
        return values

    def flush(self):
        self._unchanged()
