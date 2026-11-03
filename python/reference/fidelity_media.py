"""Host-only media identity, fixture and probe-feature contracts."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import secrets

HEX = re.compile(r"[0-9a-f]{64}\Z")

# Existing adapters predate the <family>_media module convention.
MEDIA_MODULES = {"mimo_v2": "mimo_media", "qwen4": "qwen_media", "glm5_flash": "glm_flash_media"}


def media_adapter_name(family):
    if not isinstance(family, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", family):
        raise ValueError("invalid official media family")
    return MEDIA_MODULES.get(family, family + "_media")


def has_media_adapter(family):
    """Discover official adapters without importing Torch or constructing a tower."""
    name = media_adapter_name(family)
    return name not in ("fidelity_media", "golden_media") and Path(__file__).with_name(name + ".py").is_file()


def media_families():
    aliases = {module: family for family, module in MEDIA_MODULES.items()}
    return tuple(sorted({aliases.get(path.stem, path.stem[:-6])
        for path in Path(__file__).parent.glob("*_media.py")
        if path.stem not in ("fidelity_media", "golden_media")}))


def official_media_adapter(family):
    import importlib
    if not has_media_adapter(family):
        raise ValueError(f"{family} official media exporter is not implemented")
    adapter = importlib.import_module(media_adapter_name(family))
    if not callable(getattr(adapter, "window_features", None)):
        raise ValueError(f"{family} official media adapter lacks window_features")
    return adapter


def processor_contract(snapshot):
    """The shared image-span layout, derived from the checkpoint processor."""
    snapshot = Path(snapshot)
    config = json.loads((snapshot / "config.json").read_bytes())
    path = snapshot / "processor_config.json"
    document = json.loads(path.read_bytes()) if path.exists() else {}
    processor = document.get("image_processor")
    if processor is None:
        processor = json.loads((snapshot / "preprocessor_config.json").read_bytes())
    kind = processor.get("image_processor_type", "")
    if kind in ("PixtralImageProcessor", "PixtralImageProcessorPil"):
        merge = document.get("spatial_merge_size")
        vision = json.loads((snapshot / "params.json").read_bytes())["vision_encoder"]
        ids = [vision.get(k) for k in ("image_token_id", "image_break_token_id", "image_end_token_id")]
        vocab = config.get("text_config", config)["vocab_size"]
        if (processor.get("patch_size") != config["vision_config"]["patch_size"]
                or merge != config.get("spatial_merge_size")
                or config.get("image_token_index") != ids[0]
                or any(type(n) is not int or not 0 <= n < vocab for n in ids) or len(set(ids)) != 3):
            raise ValueError("checkpoint processor geometry or image markers disagree")
        contract = {"merge": merge, "row_markers": 1, "image_token_id": ids[0],
            "row_break_token_id": ids[1], "end_token_id": ids[2]}
    elif kind in ("Qwen2VLImageProcessor", "Qwen2VLImageProcessorPil", "Qwen2VLImageProcessorFast",
                  "Glm5NextImageProcessor", "Glm5NextImageProcessorPil", "Glm5NextImageProcessorFast"):
        contract = {"merge": processor.get("merge_size", 2), "row_markers": 0}
    elif all(type(processor.get(field)) is int for field in ("merge_size", "patch_size", "temporal_patch_size")):
        vision = config.get("vision_config", {})
        for field in ("merge_size", "patch_size", "temporal_patch_size"):
            target = "patch_temporal" if field == "temporal_patch_size" and "patch_temporal" in vision else field
            if processor[field] != vision.get(target):
                raise ValueError(f"processor_config.{field}={processor[field]} disagrees with vision_config.{target}={vision.get(target)}")
        contract = {"merge": processor["merge_size"], "row_markers": 0}
    else:
        raise ValueError(f"unsupported shared image processor: {kind}")
    merge = contract["merge"]
    if type(merge) is not int or not 1 <= merge <= 8:
        raise ValueError("invalid checkpoint image merge")
    return contract


def image_extent(grid, contract=None):
    """Same patch-grid/merge/row-marker contract as the serving processor."""
    contract = contract or {"merge": 2, "row_markers": 0}
    merge, markers = contract.get("merge"), contract.get("row_markers")
    if (type(merge) is not int or merge <= 0 or type(markers) is not int or markers not in (0, 1)
            or not isinstance(grid, list) or len(grid) != 3
            or any(type(n) is not int or n <= 0 for n in grid)
            or grid[0] != 1 or grid[1] % merge or grid[2] % merge):
        raise ValueError("image grid does not match merged token extent")
    h, w = grid[1] // merge, grid[2] // merge
    return h, w, h * (w + markers)


def validate_media(media, tokens, roles, score_from, *, processor=None, audio_processor=None):
    if not isinstance(media, list):
        raise ValueError("media must be an ordered list")
    previous = 0
    for span in media:
        if not isinstance(span, dict) or span.get("kind") not in ("image", "audio"):
            raise ValueError("unsupported fidelity media kind")
        start, length = span.get("start"), span.get("len")
        if (type(start) is not int or type(length) is not int or length < 1
                or start < previous or start + length > score_from):
            raise ValueError("media spans must be sorted, disjoint and before scoring")
        if span["kind"] == "image":
            h, w, expected = image_extent(span.get("grid"), processor)
            if expected != length:
                raise ValueError("image grid does not match merged token extent")
        else:
            policy = audio_processor or {}
            samples, stride = span.get("samples"), policy.get("samples_per_token")
            if (policy.get("sampling_rate") != 24000 or type(stride) is not int or stride <= 0
                    or type(samples) is not int or not 1 <= samples <= 7_200_000
                    or length != (samples + stride - 1) // stride):
                raise ValueError("audio span differs from declared PCM processor geometry")
            if not isinstance(span.get("pcm_sha256"), str) or not HEX.fullmatch(span["pcm_sha256"]):
                raise ValueError("audio span requires canonical PCM SHA256")
        if not isinstance(span.get("key"), str) or not HEX.fullmatch(span["key"]):
            raise ValueError("media key must be lowercase SHA256 hex")
        fixture = span.get("fixture")
        if not isinstance(fixture, dict):
            raise ValueError("fixture requires a relative path and source hash")
        path = fixture.get("path")
        if (not isinstance(path, str) or not path or "\\" in path
                or Path(path).is_absolute() or any(p in (".", "..") for p in path.split("/"))
                or any(not p for p in path.split("/"))):
            raise ValueError("fixture path escapes fixture root")
        if not isinstance(fixture.get("sha256"), str) or not HEX.fullmatch(fixture["sha256"]):
            raise ValueError("fixture lacks a SHA256")
        if any(r != "ctx" for r in roles[start:start + length]):
            raise ValueError("image placeholders cannot be generated positions")
        if span["kind"] == "image" and processor and processor.get("row_markers") == 1:
            ids = [processor.get(key) for key in ("image_token_id", "row_break_token_id", "end_token_id")]
            if any(type(n) is not int or n < 0 for n in ids) or len(set(ids)) != 3:
                raise ValueError("processor image marker ids are absent or invalid")
            image, row_break, end = ids
            for row in range(h):
                offset = start + row * (w + 1)
                if (tokens[offset:offset + w] != [image] * w
                        or tokens[offset + w] != (end if row + 1 == h else row_break)):
                    raise ValueError("image span differs from processor row markers")
        elif len(set(tokens[start:start + length])) != 1:
            raise ValueError("image span must contain one repeated placeholder id")
        previous = start + length
    return media


def read_fixture(root: Path, span: dict) -> bytes:
    path = (root / span["fixture"]["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("fixture path escapes fixture root")
    cap = 32 * 1024 * 1024
    with path.open("rb") as source:
        if path.stat().st_size > cap:
            raise ValueError("fixture exceeds image byte cap")
        data = source.read(cap + 1)
    if len(data) > cap:
        raise ValueError("fixture exceeds image byte cap")
    if hashlib.sha256(data).hexdigest() != span["fixture"]["sha256"]:
        raise ValueError("fixture bytes differ from pinned media identity")
    return data


def read_pcm_fixture(root: Path, span: dict):
    """Pinned mono 24 kHz IEEE float WAV; never resample or silently quantize."""
    import struct
    import numpy as np
    data = read_fixture(root, span)
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE" or struct.unpack_from("<I", data, 4)[0] != len(data) - 8:
        raise ValueError("audio fixture requires canonical mono 24000 Hz float WAV")
    offset, chunks = 12, {}
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated PCM WAV chunk")
        name, size = struct.unpack_from("<4sI", data, offset)
        offset += 8
        if offset + size > len(data) or name in chunks:
            raise ValueError("invalid or duplicate PCM WAV chunk")
        chunks[name] = data[offset:offset + size]
        offset += size + size % 2
    fmt, payload = chunks.get(b"fmt ", b""), chunks.get(b"data", b"")
    if (len(fmt) < 16 or struct.unpack_from("<HHIIHH", fmt) != (3, 1, 24000, 96000, 4, 32)
            or len(payload) != span["samples"] * 4 or not 1 <= span["samples"] <= 7_200_000):
        raise ValueError("audio fixture requires canonical mono 24000 Hz float WAV sample count")
    values = np.frombuffer(payload, dtype="<f4").copy()
    if not np.isfinite(values).all() or hashlib.sha256(values.astype("<f4").tobytes()).hexdigest() != span["pcm_sha256"]:
        raise ValueError("audio fixture differs from canonical PCM identity")
    return values


def require_media_flag(manifest, enabled, family, *, implemented_families=()):
    """Family runners opt in only after implementing their official tower hook."""
    present = any(w.get("media") for w in manifest["windows"])
    if present and not enabled:
        raise ValueError("media windows require explicit --media; refusing text-only scoring")
    if enabled and not present:
        raise ValueError("--media requires pinned media windows")
    if present and family not in {"mimo_v2", "qwen4", *implemented_families}:
        raise ValueError(f"{family} official media golden is not implemented")
    return present


@contextmanager
def _feature_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    # Directory-authorized writers need only read access to the persistent lock.
    fd = os.open(root / ".features.lock", os.O_RDONLY | os.O_CREAT, 0o666)
    with os.fdopen(fd, "rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _publish_locked(contents):
    """Validate every existing member before publishing any new member."""
    missing = []
    for path, data in contents:
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("feature identity reused with different bytes or metadata; feature index is immutable")
        else:
            missing.append((path, data))
    for path, data in missing:
        temporary = None
        try:
            for _ in range(100):
                candidate = path.parent / (".feature-" + secrets.token_hex(16))
                try:
                    fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                except FileExistsError:
                    continue
                temporary = candidate
                break
            else:
                raise FileExistsError("cannot allocate an exclusive feature staging file")
            # Ordinary creation permissions preserve the caller's umask, unlike mkstemp.
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            # Readers see complete bytes; the held lock prevents writer replacement.
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if missing:
        directory = os.open(missing[0][0].parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def publish_immutable(path: Path, content: bytes):
    """Publish complete immutable bytes under the shared feature-set writer lock."""
    with _feature_lock(path.parent):
        _publish_locked([(path, content)])


def write_features(root: Path, span: dict, values, *, tower_dtype: str, identity: dict):
    """Write immutable row-major BF16 probe rows; values are CPU uint16 bits."""
    import numpy as np
    bits = np.asarray(values)
    if bits.dtype != np.uint16 or bits.ndim != 2 or bits.shape[0] != span["len"]:
        raise ValueError("feature rows must be uint16 BF16 [media.len, hidden]")
    if not np.isfinite((bits.astype(np.uint32) << 16).view(np.float32)).all():
        raise ValueError("non-finite reference features")
    if tower_dtype not in ("bf16", "fp32"):
        raise ValueError("unsupported tower precision")
    if not HEX.fullmatch(span["key"]):
        raise ValueError("unsafe feature key")
    data = bits.astype("<u2").tobytes()
    meta = {"schema": "cuteafd.media.features/1", "key": span["key"],
            **({"grid": span["grid"]} if span["kind"] == "image" else
               {"kind": "audio", "samples": span["samples"], "pcm_sha256": span["pcm_sha256"]}),
            "shape": list(bits.shape), "dtype": "bf16-le",
            "sha256": hashlib.sha256(data).hexdigest(), "tower_dtype": tower_dtype,
            "fixture_sha256": span["fixture"]["sha256"], "snapshot_identity": identity}
    encoded = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with _feature_lock(root):
        _publish_locked([(root / (span["key"] + ".bf16"), data),
                         (root / (span["key"] + ".json"), encoded)])
    return meta


def prepared_span(span):
    """Preserve the encoder/preprocessor identity sealed by the generation server."""
    result = {k: span[k] for k in ("start", "len", "kind", "key", "grid")}
    if "identity" in span:
        identity = span["identity"]
        if (not isinstance(identity, dict) or set(identity) != {"encoder", "processor"}
                or any(not isinstance(v, str) or not HEX.fullmatch(v) for v in identity.values())):
            raise ValueError("invalid prepared input identity")
        result["identity"] = dict(identity)
    return result
