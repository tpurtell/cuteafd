"""Host-only contracts shared by the set builder, goldens and converter."""
from __future__ import annotations

import copy
import ctypes
import gc
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

SET_SCHEMA = "cuteafd.fidelity.set/1"
MAX_TOKENS = 16384


def validate_public_text(text: str, *, scored_text: bool = False) -> None:
    """Reject publication blockers without changing text or blocking public fabric names."""
    credentials = re.compile(
        r"\bhf_[A-Za-z0-9]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b"
        r"|\bgithub_pat_[A-Za-z0-9_]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b"
        r"|-----BEGIN (?:[A-Z0-9 ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----"
        r"|\b(?:api[_-]?key|access[_-]?token|authorization|password|secret)"
        r"\s*[=:]\s*[\"']?[A-Za-z0-9_+/.-]{20,}", re.I)
    if credentials.search(text):
        raise ValueError("credential or private-key publication blocker")
    if re.search(r"\b(?:tj|tpurtell)\b", text, re.I):
        raise ValueError("personal-data publication blocker")
    for address in re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        domain = address.rsplit("@", 1)[1].lower()
        reserved = domain in ("example.com", "example.org", "example.net") or domain.endswith(".example")
        if not (scored_text and reserved):
            raise ValueError("email publication blocker")


def validate_public_metadata(value, *, synthetic_exemption: bool = False) -> None:
    """Permit reserved-email audit records without exempting arbitrary metadata."""
    if isinstance(value, str):
        validate_public_text(value, scored_text=synthetic_exemption)
    elif isinstance(value, list):
        for item in value:
            validate_public_metadata(item, synthetic_exemption=synthetic_exemption)
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_public_text(key)
            if key == "synthetic_email_exemptions":
                if not isinstance(item, list):
                    raise ValueError("invalid synthetic-email audit records")
                for record in item:
                    if not isinstance(record, dict) or record.get("role") != "gen":
                        raise ValueError("synthetic-email audit must name generated text")
                    address = record.get("address")
                    if not isinstance(address, str) or "@" not in address:
                        raise ValueError("synthetic-email audit lacks address")
                    domain = address.rsplit("@", 1)[1].lower()
                    if domain not in ("example.com", "example.org", "example.net") and not domain.endswith(".example"):
                        raise ValueError("non-reserved synthetic-email audit address")
                    for field, content in record.items():
                        validate_public_text(field)
                        validate_public_metadata(content, synthetic_exemption=field == "address")
            else:
                validate_public_metadata(item)


def rss_bytes() -> int:
    """Current Linux RSS, not ru_maxrss's irreversible peak."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def release_checkpoint(cuda, *readers):
    # Copy completion precedes releasing mapped checkpoint backing storage.
    if cuda is not None:
        cuda.synchronize()
    for reader in readers:
        for handle in reader.files.values():
            handle.__exit__(None, None, None)
            del handle
        reader.files.clear()
    gc.collect()
    # glibc may retain freed CPU staging arenas, especially on unified-memory ARM.
    # Return them to the OS so the next layer's admission sees actual live storage.
    libc = ctypes.CDLL(None)
    trim = getattr(libc, "malloc_trim", None)
    if trim is not None:
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
        trim(0)
    if cuda is not None:
        cuda.empty_cache()


def log_checkpoint_reads(label, readers, before, started):
    """Logical cloned/sliced tensor bytes, not physical FUSE or archive traffic."""
    elapsed = time.monotonic() - started
    size = sum(reader.read_bytes for reader in readers) - before
    read_seconds = sum(reader.read_seconds for reader in readers)
    print(f"checkpoint reads {label}: utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
          f"bytes={size} elapsed={elapsed:.3f}s MB/s={size / 1e6 / max(elapsed, 1e-9):.3f} "
          f"cumulative_read_seconds={read_seconds:.3f}", flush=True)


class CheckpointStorage:
    """Bound CPU shard lifetimes; CUDA is injected so CPU contracts stay host-only."""

    def __init__(self, cuda, *readers, max_rss_gib=80, max_growth_gib=8):
        self.cuda, self.readers = cuda, readers
        self.max_rss = int(max_rss_gib * 2**30)
        self.max_growth = int(max_growth_gib * 2**30)
        if self.max_rss <= 0 or self.max_growth < 0:
            raise ValueError("invalid checkpoint RSS bounds")
        self.release()
        self.baseline = rss_bytes()
        self.check("embedding")

    def release(self):
        release_checkpoint(self.cuda, *self.readers)

    def check(self, label):
        current = rss_bytes()
        limit = min(self.max_rss, self.baseline + self.max_growth)
        print(f"checkpoint memory {label}: RSS={current / 2**30:.3f} GiB "
              f"baseline={self.baseline / 2**30:.3f} limit={limit / 2**30:.3f}", flush=True)
        if current > limit:
            raise RuntimeError(f"checkpoint RSS bound exceeded after {label}: {current} > {limit}")
        self.baseline = min(self.baseline, current)
        return current


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def set_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "set_sha256"})).hexdigest()


def validate_set(manifest: dict) -> dict:
    if manifest.get("schema") != SET_SCHEMA:
        raise ValueError("unsupported fidelity set schema")
    if manifest.get("set_sha256") != set_hash(manifest):
        raise ValueError("fidelity set hash mismatch")
    seen = set()
    for w in manifest["windows"]:
        name = w["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("unsafe window id")
        if name in seen:
            raise ValueError("duplicate window id")
        seen.add(name)
        tokens, roles, start = w["tokens"], w["roles"], w["score_from"]
        if not 1 <= start < len(tokens) <= MAX_TOKENS:
            raise ValueError("window length/score_from exceeds the 16K contract")
        if any(type(t) is not int or not 0 <= t <= 2147483647 for t in tokens):
            raise ValueError("token ids must be nonnegative i32")
        if len(roles) != len(tokens) or any(r not in ("ctx", "gen") for r in roles):
            raise ValueError("one ctx/gen role is required per token")
        if w["bucket"] != bucket(start):
            raise ValueError("context bucket does not match first scored position")
        from fidelity_media import validate_media
        validate_media(w.get("media", []), tokens, roles, start)
    if not seen or len(set(manifest["quick_windows"])) != len(manifest["quick_windows"]) or not set(manifest["quick_windows"]) <= seen:
        raise ValueError("invalid quick subset")
    return manifest


def load_set(path: Path, family: str | None = None) -> dict:
    manifest = validate_set(json.loads(path.read_text()))
    if family is not None and manifest["family"] != family:
        raise ValueError(f"set family {manifest['family']} does not match {family}")
    if (family is not None and family not in ("mimo_v2", "qwen4", "glm5_flash")
            and any(w.get("media") for w in manifest["windows"])):
        raise ValueError(f"{family} golden does not support media; refusing text-only scoring")
    return manifest


def bucket(position: int) -> str:
    return "0-2K" if position <= 2048 else "2-8K" if position <= 8192 else "8-16K" if position < MAX_TOKENS else "16K+"


def verify_snapshot(manifest: dict, snapshot: Path) -> dict:
    """Check cheap, stable identities without hashing hundreds of GB of weights."""
    tokenizer_sha = hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
    config_sha = hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
    if manifest.get("tokenizer_sha256") not in (None, tokenizer_sha):
        raise ValueError("snapshot tokenizer differs from the pinned fidelity set")
    arm = manifest.get("reference_snapshot", manifest.get("generation_arm", {}))
    revision = arm.get("snapshot_revision")
    if revision is not None and snapshot.name != revision:
        raise ValueError("snapshot revision differs from the pinned reference identity")
    if arm.get("config_sha256") not in (None, config_sha):
        raise ValueError("snapshot config differs from the pinned reference identity")
    if "reference_snapshot" in manifest and (
            not revision or arm.get("config_sha256") != config_sha
            or arm.get("tokenizer_sha256") != tokenizer_sha):
        raise ValueError("explicit reference snapshot requires all identity hashes")
    identity = {"snapshot_revision": snapshot.name, "tokenizer_sha256": tokenizer_sha,
                "config_sha256": config_sha}
    for field in ("checkpoint", "precision"):
        if field in arm:
            identity[field] = arm[field]
    return identity


def prefix_comparison(short: np.ndarray, extended: np.ndarray) -> dict:
    """Require finite, bit-identical f32 common-prefix rows, not just argmax."""
    if short.ndim != 2 or extended.ndim != 2 or short.shape[1] != extended.shape[1] or extended.shape[0] < short.shape[0]:
        raise ValueError("prefix qualification logits have incompatible shapes")
    left = np.asarray(short, dtype="<f4")
    right = np.asarray(extended[:len(short)], dtype="<f4")
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    changed = np.any(left.view("<u4") != right.view("<u4"), axis=1)
    return {"passed": finite and not bool(changed.any()), "finite": finite,
            "rows": len(left), "vocab": left.shape[1],
            "different_rows": np.flatnonzero(changed).tolist(),
            "argmax_disagreements": int(np.count_nonzero(left.argmax(1) != right.argmax(1)))}


def qualify_prefix(a, manifest: dict, execute) -> dict:
    """Fail closed before the full panel; execute the family's actual golden loop."""
    def gate_start(w):
        return max([64] + [s["start"] + s["len"] for s in w.get("media", [])])
    media_panel = any(w.get("media") for w in manifest["windows"])
    source = next((w for w in manifest["windows"]
                   if (not media_panel or w.get("media"))
                   and len(w["tokens"]) >= gate_start(w) + 576), None)
    if source is None:
        raise ValueError("prefix qualification requires 576 text tokens after media (640 for text)")
    score_from = gate_start(source)
    lengths = [score_from + 512, score_from + 576]
    root = Path(tempfile.mkdtemp(prefix="prefix-gate-", dir=a.out))
    panel = copy.deepcopy(manifest)
    panel["windows"] = []
    for size, name in zip(lengths, ("prefix_short", "prefix_extended")):
        window = copy.deepcopy(source)
        window.update(id=name, tokens=source["tokens"][:size], roles=["ctx"] * size,
                      score_from=score_from, bucket=bucket(score_from))
        panel["windows"].append(window)
    panel["quick_windows"] = ["prefix_short", "prefix_extended"]
    panel["set_sha256"] = set_hash(panel)
    validate_set(panel)
    windows = root / "input.json"
    windows.write_bytes(canonical(panel) + b"\n")
    probe = copy.copy(a)
    probe.windows, probe.out, probe.layers, probe._prefix_probe = windows, root, None, True
    if media_panel and not getattr(probe, "media_root", None):
        probe.media_root = a.windows.parent
    execute(probe)
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("set_sha256") != panel["set_sha256"] or meta.get("family") != manifest["family"]:
        raise ValueError("prefix golden provenance mismatch")
    entries = {w["id"]: w for w in meta["windows"]}
    arrays, hashes = [], []
    for window in panel["windows"]:
        entry = entries[window["id"]]
        if entry.get("media", []) != window.get("media", []):
            raise ValueError("prefix golden media identity mismatch")
        positions = list(range(window["score_from"], len(window["tokens"])))
        if entry["positions"] != positions:
            raise ValueError("prefix golden positions mismatch")
        folder = (root / entry["path"]).resolve()
        if not folder.is_relative_to(root.resolve()):
            raise ValueError("prefix golden path escapes output")
        if not np.array_equal(np.fromfile(folder / "tokens.bin", dtype="<i4"), window["tokens"]):
            raise ValueError("prefix golden tokens mismatch")
        path = folder / "logits.bin"
        if path.stat().st_size != len(positions) * entry["vocab"] * 4:
            raise ValueError("prefix golden logits extent mismatch")
        arrays.append(np.memmap(path, dtype="<f4", mode="r", shape=(len(positions), entry["vocab"])))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    result = prefix_comparison(*arrays)
    proof = {"schema": "cuteafd.fidelity.prefix/1", "family": manifest["family"],
             "set_sha256": manifest["set_sha256"], "source_window": source["id"],
             "lengths": lengths, "score_from": score_from, "fixed_rows": 128,
             **({"media": source["media"]} if source.get("media") else {}),
             **({"reference_geometry": meta["reference_geometry"]} if "reference_geometry" in meta else {}),
             "snapshot_identity": meta["snapshot_identity"], "logits_sha256": hashes,
             "seconds": meta["seconds"], **result}
    (root / "qualification.json").write_bytes(canonical(proof) + b"\n")
    print(f"prefix qualification: {len(result['different_rows'])}/{result['rows']} different rows; {root}", flush=True)
    if not result["passed"]:
        raise ValueError(f"reference prefix invariance failed; see {root / 'qualification.json'}")
    return proof


def validate_qualification(proof: dict | None, manifest: dict, identity: dict) -> None:
    if not isinstance(identity, dict) or not identity.get("snapshot_revision"):
        raise ValueError("reference prefix qualification lacks snapshot identity")
    if not isinstance(proof, dict) or proof.get("schema") != "cuteafd.fidelity.prefix/1":
        raise ValueError("reference lacks prefix-invariance qualification")
    required = {"passed": True, "finite": True, "rows": 512, "different_rows": [],
                "argmax_disagreements": 0, "lengths": [576, 640], "score_from": 64,
                "fixed_rows": 128, "family": manifest["family"],
                "set_sha256": manifest["set_sha256"], "snapshot_identity": identity}
    if any(w.get("media") for w in manifest["windows"]):
        source = next((w for w in manifest["windows"] if w["id"] == proof.get("source_window")), None)
        if source is None or not source.get("media") or proof.get("media") != source["media"]:
            raise ValueError("media prefix qualification lacks pinned image evidence")
        start = max([64] + [s["start"] + s["len"] for s in source["media"]])
        required.update(score_from=start, lengths=[start + 512, start + 576])
    if any(proof.get(k) != v for k, v in required.items()):
        raise ValueError("reference prefix-invariance qualification failed or mismatched provenance")


def write_scored_logits(out: Path, window: dict, logits: np.ndarray) -> dict:
    """Rows are already selected: row i predicts token score_from + i."""
    positions = list(range(window["score_from"], len(window["tokens"])))
    if logits.ndim != 2 or logits.shape[0] != len(positions):
        raise ValueError("scored logits shape does not match window positions")
    folder = out / "windows" / window["id"]
    folder.mkdir(parents=True, exist_ok=True)
    np.asarray(window["tokens"], dtype="<i4").tofile(folder / "tokens.bin")
    np.asarray(logits, dtype="<f4").tofile(folder / "logits.bin")
    return {"id": window["id"], "path": str(folder.relative_to(out)),
            "positions": positions, "vocab": logits.shape[1],
            **({"media": window["media"]} if window.get("media") else {})}


def finish_golden(out: Path, manifest: dict, rows: list[dict], **meta) -> None:
    (out / "windows.json").write_bytes(canonical(manifest) + b"\n")
    (out / "meta.json").write_bytes(canonical({"schema": "cuteafd.fidelity.golden/2",
        "set_sha256": manifest["set_sha256"], "family": manifest["family"],
        "checkpoint": manifest["checkpoint"], "windows": rows, **meta}) + b"\n")
