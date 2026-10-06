#!/usr/bin/env python3
"""Capture media teacher-forced rows and gate paired encoder effects against goldens."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from fidelity_media import read_fixture
from fidelity_windows import canonical, load_set, validate_qualification

CRITERION = ("the paired difference against the golden decides (\u22640.005 nat KL increase, "
             "\u22640.5 point top-1 loss, one-sided 95% window bootstrap). "
             "Direct KL between the arms is a diagnostic only.")


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(canonical(value) + b"\n")


def http(url, token, body=None, timeout=240):
    headers = {"Content-Type": "application/json"}
    if token:
        headers.update({"x-cuteafd-bench": token, "Authorization": "Bearer " + token})
    request = urllib.request.Request(url, None if body is None else canonical(body), headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        data = error.read()
        print(f"HTTP {error.code} {url}: {data.decode('utf-8', errors='replace')}", file=sys.stderr)
        import io
        error.fp = io.BytesIO(data)
        raise


def mapped_window(window, mapping):
    """Translate only encoder-dependent keys, never sealed tokens/geometry/fixtures."""
    result = json.loads(json.dumps(window))
    for span in result["media"]:
        entry = mapping[span["key"]]
        if entry["old_key"] != span["key"] or entry["grid"] != span["grid"] or entry["len"] != span["len"] or entry["fixture"] != span["fixture"]:
            raise ValueError("server key mapping changes sealed image geometry/fixture")
        if len(entry["new_key"]) != 64 or any(c not in "0123456789abcdef" for c in entry["new_key"]):
            raise ValueError("invalid echoed image key")
        span["key"] = entry["new_key"]
    return result


def discover_keys(a, panel):
    """Normal server preparation is authoritative; never derive an ImageKey here."""
    from PIL import Image
    import io
    mapping = {}
    for window in panel["windows"]:
        for span in window["media"]:
            if span["key"] in mapping:
                continue
            data = read_fixture(a.media_root, span)
            pixels = Image.open(io.BytesIO(data)).convert("RGB")
            body = {"body": {"model": panel["checkpoint"], "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
                {"type": "text", "text": "Reply OK."}]}], "max_tokens": 1, "temperature": 0},
                "spec": {"cold": True, "no_speculation": True, "record_first": True, "top_k": 1}}
            response = http(a.url.rstrip("/") + "/v1/bench/probe", a.bench_token, body, a.timeout)
            record = response.get("probe", {})
            echoed = record.get("media", [])
            if record.get("error") or len(echoed) != 1 or response["server"]["model"] != panel["checkpoint"] or response["server"].get("family") != panel["family"]:
                raise ValueError("normal server image preparation failed")
            image = echoed[0]
            if image["grid"] != span["grid"] or image["len"] != span["len"] or image["kind"] != "image":
                raise ValueError("normal server preparation changed sealed grid")
            evidence = a.out / (span["key"] + ".prepare.json")
            write_new(evidence, response)
            fixture_copy = a.out / (span["key"] + ".source.png")
            with fixture_copy.open("xb") as f:
                f.write(data)
            mapping[span["key"]] = {"old_key": span["key"], "new_key": image["key"],
                "grid": span["grid"], "len": span["len"], "fixture": span["fixture"],
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "pixel_size": list(pixels.size),
                "source_file": fixture_copy.name,
                "prepare_file": evidence.name, "prepare_sha256": file_hash(evidence)}
            if a.mode == "reference":
                if a.mapped_features is None:
                    raise ValueError("reference rekey requires --mapped-features server bind")
                meta = json.loads((a.features / (span["key"] + ".json")).read_bytes())
                payload = (a.features / (span["key"] + ".bf16")).read_bytes()
                if meta["key"] != span["key"] or meta["grid"] != span["grid"] or meta["fixture_sha256"] != span["fixture"]["sha256"] or hashlib.sha256(payload).hexdigest() != meta["sha256"]:
                    raise ValueError("sealed reference payload identity differs")
                meta["key"] = image["key"]
                a.mapped_features.mkdir(parents=True, exist_ok=True)
                with (a.mapped_features / (image["key"] + ".bf16")).open("xb") as f:
                    f.write(payload)
                write_new(a.mapped_features / (image["key"] + ".json"), meta)
    write_new(a.out / "key-mapping.json", mapping)
    return mapping


def capture_mapping(root, capture):
    seal = capture.get("key_mapping_sha256")
    if seal is None:
        return None
    path = root / "key-mapping.json"
    if file_hash(path) != seal:
        raise ValueError("server key mapping changed")
    mapping = json.loads(path.read_bytes())
    for entry in mapping.values():
        from PIL import Image
        source = root / entry["source_file"]
        if source.name != entry["source_file"] or file_hash(source) != entry["fixture"]["sha256"]:
            raise ValueError("mapping source pixels differ from sealed fixture")
        pixels = Image.open(source).convert("RGB")
        if hashlib.sha256(pixels.tobytes()).hexdigest() != entry["pixel_sha256"] or list(pixels.size) != entry["pixel_size"]:
            raise ValueError("mapping decoded pixels changed")
        path = root / entry["prepare_file"]
        if path.name != entry["prepare_file"] or file_hash(path) != entry["prepare_sha256"]:
            raise ValueError("normal prepare evidence changed")
        response = json.loads(path.read_bytes())
        if response["server"] != capture["server"]:
            raise ValueError("key preparation server differs from scored server")
        media = response["probe"]["media"]
        if len(media) != 1 or media[0]["key"] != entry["new_key"] or media[0]["grid"] != entry["grid"] or media[0]["len"] != entry["len"]:
            raise ValueError("mapping differs from authoritative server echo")
    return mapping


FAMILIES = ("mimo_v2", "qwen4", "glm5_flash")


def load_panel(a):
    family = getattr(a, "family", "mimo_v2")
    if family not in FAMILIES:
        raise ValueError("unsupported paired media family")
    return load_set(a.windows, family)


def check_record(window, response, checkpoint, mode, features_root=None, family=None, feature_mapping=None):
    record = response.get("probe", {})
    if (record.get("error") or not record.get("engine") or not record.get("cold")
            or not record.get("no_speculation") or record.get("cached_tokens") != 0
            or record.get("score_path") != "decode" or record.get("prompt_ids") != window["tokens"]
            or response.get("server", {}).get("model") != checkpoint
            or family is not None and response.get("server", {}).get("family") != family):
        raise ValueError("media scoring identity/cold/path contract not honored")
    fields = ("start", "len", "kind", "key", "grid")
    expected = [{k: span[k] for k in fields} for span in window["media"]]
    actual = [{k: span.get(k) for k in fields} for span in record.get("media", [])]
    if actual != expected:
        raise ValueError("prepared media echo differs")
    positions = list(range(window["score_from"], len(window["tokens"])))
    if ([row["position"] for row in record.get("rows", [])] != positions
            or record.get("scored") != len(positions)
            or not all(row.get("finite") for row in record["rows"])):
        raise ValueError("incomplete/nonfinite scored rows")
    provenance = record.get("provenance")
    if mode == "native":
        if provenance is not None:
            raise ValueError("native arm unexpectedly reports feature override provenance")
    else:
        metadata = []
        reverse = {entry["new_key"]: old for old, entry in (feature_mapping or {}).items()}
        for span in window["media"]:
            source_key = reverse.get(span["key"], span["key"])
            meta = json.loads((features_root / (source_key + ".json")).read_text())
            if source_key != span["key"]:
                if meta["key"] != source_key:
                    raise ValueError("original feature identity differs")
                meta["key"] = span["key"]
            metadata.append(meta)
        if provenance != {"mode": "reference_features", "probe_only": True,
                          "encoder_bypassed": True, "features": metadata}:
            raise ValueError("reference feature provenance differs from sealed export")


def capture(a):
    panel = load_panel(a)
    if a.out.exists():
        raise ValueError("capture output must be new")
    models = http(a.url.rstrip("/") + "/v1/models", a.bench_token)
    records = [m for m in models.get("data", []) if m.get("id") == panel["checkpoint"]]
    if len(records) != 1 or records[0].get("capabilities", {}).get("vision") is not True:
        raise ValueError("checkpoint must advertise vision")
    if a.mode == "reference" and a.features is None:
        raise ValueError("reference capture needs --features")
    selected = [w for w in panel["windows"] if not a.quick or w["id"] in panel["quick_windows"]]
    a.out.mkdir(parents=True)
    mapping = discover_keys(a, panel) if getattr(a, "server_keys", False) else None
    feature_root = a.mapped_features if mapping is not None and a.mode == "reference" else a.features
    entries, server = [], None
    started = time.monotonic()
    for window in selected:
        window = mapped_window(window, mapping) if mapping is not None else window
        media = []
        for span in window["media"]:
            data = read_fixture(a.media_root, span)
            media.append({**span, "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}})
        leaf = window["id"]
        if (a.host_dump / leaf).exists():
            raise ValueError("dump leaf already exists")
        body = {"body": {"model": panel["checkpoint"], "messages": [{"role": "user", "content": "media fidelity probe"}],
                         "max_tokens": 1, "temperature": 0, "stream": False},
                "spec": {"prompt_ids": window["tokens"], "media": media, "score_from": window["score_from"],
                         "cold": True, "no_speculation": True, "score_path": "decode", "verify_rows": 1,
                         "top_k": 1, "dump_rows": str(a.server_dump / leaf)}}
        try:
            response = http(a.url.rstrip("/") + "/v1/bench/probe", a.bench_token, body, a.timeout)
        except urllib.error.HTTPError as error:
            write_new(a.out / (leaf + ".http-error.json"), {"status": error.code,
                "body": error.read().decode("utf-8", errors="replace")})
            raise
        # Retain an invalid response too; it is evidence of a failed live gate.
        write_new(a.out / (leaf + ".json"), response)
        check_record(window, response, panel["checkpoint"], a.mode, feature_root, panel["family"])
        current = response["server"]
        if server is not None and current != server:
            raise ValueError("server build/settings changed during capture")
        server = current
        rows = dump_rows(a.host_dump / leaf, window, None)
        entries.append({"id": leaf, "response_sha256": hashlib.sha256((a.out / (leaf + ".json")).read_bytes()).hexdigest(),
                        "path": str((a.host_dump / leaf).resolve()),
                        "manifest_sha256": hashlib.sha256((a.host_dump / leaf / "manifest.jsonl").read_bytes()).hexdigest(),
                        "files": [{"position": pos, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for pos, path in rows.items()]})
    write_new(a.out / "capture.json", {"schema": "cuteafd.media.paired.capture/1", "mode": a.mode,
        "set_sha256": panel["set_sha256"], "checkpoint": panel["checkpoint"], "server": server,
        "quick": a.quick, "seconds": time.monotonic() - started, "windows": entries,
        **({"key_mapping_sha256": file_hash(a.out / "key-mapping.json")} if mapping is not None else {})})


def dump_rows(root, window, vocab):
    rows = {}
    for line in (root / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        name = row["file"]
        if (Path(name).name != name or name in (".", "..") or row["tensor"] != "log_probs"
                or row["dtype"] != "F32" or row["byte_order"] != "little"
                or vocab is not None and row["vocab_size"] != vocab or row["position"] in rows):
            raise ValueError("invalid/duplicate dump row identity")
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("dump row escapes root")
        rows[row["position"]] = path
    if sorted(rows) != list(range(window["score_from"], len(window["tokens"]))):
        raise ValueError("dump position coverage differs from panel")
    return rows


def log_probs(path, vocab):
    from safetensors.numpy import load_file
    tensors = load_file(str(path))
    value = tensors.get("log_probs")
    if set(tensors) != {"log_probs"} or value.dtype != np.float32 or value.shape != (vocab,):
        raise ValueError("dump tensor shape/dtype differs")
    return normalize(value)


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite vocabulary row")
    return values - (values.max() + np.log(np.exp(values - values.max()).sum()))


def paired_bounds(stats, counts, replicates=5000, seed=20260829):
    stats, counts = np.asarray(stats, dtype=np.float64), np.asarray(counts)
    if replicates < 100 or len(counts) < 2 or np.any(counts <= 0):
        raise ValueError("paired bootstrap requires at least two nonempty windows")
    picks = np.random.default_rng(seed).integers(len(counts), size=(replicates, len(counts)))
    samples = stats[picks].sum(axis=1) / counts[picks].sum(axis=1)[:, None]
    mean = stats.sum(axis=0) / counts.sum()
    return {"kl_increase": float(mean[0]), "top1_loss": float(mean[1]),
            "kl_increase_upper95": float(np.quantile(samples[:, 0], .95)),
            "top1_loss_upper95": float(mean[1] + 1.6448536269514722 * samples[:, 1].std(ddof=1)),
            "bootstrap": replicates, "seed": seed, "unit": "whole windows, ratio of sums"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def golden_seal(root, panel):
    meta = json.loads((root / "meta.json").read_text())
    validate_qualification(meta.get("prefix_qualification"), panel, meta.get("snapshot_identity"))
    if meta["set_sha256"] != panel["set_sha256"] or meta["checkpoint"] != panel["checkpoint"]:
        raise ValueError("golden seal provenance differs")
    files = {"meta.json": file_hash(root / "meta.json"), "windows.json": file_hash(root / "windows.json")}
    for entry in meta["windows"]:
        folder = (root / entry["path"]).resolve()
        if not folder.is_relative_to(root.resolve()):
            raise ValueError("golden path escapes root")
        for name in ("tokens.bin", "logits.bin"):
            path = folder / name
            files[str(path.relative_to(root.resolve()))] = file_hash(path)
    return {"schema": "cuteafd.media.golden.seal/1", "set_sha256": panel["set_sha256"],
            "checkpoint": panel["checkpoint"], "files": files}


def compare(a):
    panel = load_panel(a)
    seal = json.loads(a.golden_seal.read_text())
    if seal != golden_seal(a.golden, panel):
        raise ValueError("golden files differ from immutable seal")
    meta = json.loads((a.golden / "meta.json").read_text())
    validate_qualification(meta.get("prefix_qualification"), panel, meta.get("snapshot_identity"))
    if meta["set_sha256"] != panel["set_sha256"] or meta["checkpoint"] != panel["checkpoint"]:
        raise ValueError("golden provenance differs")
    captures = [json.loads((root / "capture.json").read_text()) for root in (a.native, a.reference)]
    for capture_record, mode in zip(captures, ("native", "reference")):
        if (capture_record["schema"] != "cuteafd.media.paired.capture/1" or capture_record["mode"] != mode
                or capture_record["set_sha256"] != panel["set_sha256"]
                or capture_record["checkpoint"] != panel["checkpoint"]):
            raise ValueError("arm capture identity differs")
    # Only the image tag may differ; commit and all effective model knobs must match.
    servers = json.loads(json.dumps([c["server"] for c in captures]))
    for server in servers:
        server.get("build", {}).pop("image", None)
    if servers[0] != servers[1]:
        raise ValueError("arms have different builds or non-encoder settings")
    key_maps = [capture_mapping(root, capture) for root, capture in zip((a.native, a.reference), captures)]
    if (key_maps[0] is None) != (key_maps[1] is None):
        raise ValueError("arms use different media key policies")
    if key_maps[0] is not None:
        comparable = lambda mapping: {key: {k: v for k, v in entry.items() if k not in ("prepare_file", "prepare_sha256")} for key, entry in mapping.items()}
        if comparable(key_maps[0]) != comparable(key_maps[1]):
            raise ValueError("arms have different prepared media identities")
    arm_maps = [{w["id"]: w for w in c["windows"]} for c in captures]
    expected = {w["id"] for w in panel["windows"]}
    if captures[0]["quick"] or captures[1]["quick"]:
        expected = set(panel["quick_windows"])
    if (captures[0]["quick"] != captures[1]["quick"] or set(arm_maps[0]) != expected
            or set(arm_maps[1]) != expected
            or any(len(arm) != len(c["windows"]) for arm, c in zip(arm_maps, captures))):
        raise ValueError("arms score different/incomplete windows")
    gold = {w["id"]: w for w in meta["windows"]}
    if set(gold) != {w["id"] for w in panel["windows"]} or len(gold) != len(meta["windows"]):
        raise ValueError("golden panel coverage differs")
    statistics, counts, summaries = [], [], []
    for window in panel["windows"]:
        if window["id"] not in arm_maps[0]:
            continue
        entry = gold[window["id"]]
        positions = list(range(window["score_from"], len(window["tokens"])))
        if entry["positions"] != positions or entry.get("media") != window["media"]:
            raise ValueError("golden media/position identity differs")
        vocab = entry["vocab"]
        if vocab != meta["prefix_qualification"]["vocab"]:
            raise ValueError("golden vocabulary differs from prefix gate")
        folder = (a.golden / entry["path"]).resolve()
        if not folder.is_relative_to(a.golden.resolve()):
            raise ValueError("golden path escapes root")
        path = folder / "logits.bin"
        if path.stat().st_size != len(positions) * vocab * 4:
            raise ValueError("golden logits extent differs")
        if np.fromfile(folder / "tokens.bin", dtype="<i4").tolist() != window["tokens"]:
            raise ValueError("golden tokens differ")
        logits = np.memmap(path, dtype="<f4", mode="r", shape=(len(positions), vocab))
        arms = []
        for arm, capture_root, mode in zip(arm_maps, (a.native, a.reference), ("native", "reference")):
            sealed = arm[window["id"]]
            response_path = capture_root / (window["id"] + ".json")
            if hashlib.sha256(response_path.read_bytes()).hexdigest() != sealed["response_sha256"]:
                raise ValueError("capture response changed")
            response = json.loads(response_path.read_text())
            key_map = key_maps[0 if mode == "native" else 1]
            prepared_window = mapped_window(window, key_map) if key_map is not None else window
            check_record(prepared_window, response, panel["checkpoint"], mode, a.features, panel["family"], key_map)
            if response["server"] != captures[0 if mode == "native" else 1]["server"]:
                raise ValueError("capture server differs from manifest")
            root = Path(sealed["path"])
            if hashlib.sha256((root / "manifest.jsonl").read_bytes()).hexdigest() != sealed["manifest_sha256"]:
                raise ValueError("dump manifest changed")
            rows = dump_rows(root, window, vocab)
            hashes = {f["position"]: f["sha256"] for f in sealed["files"]}
            if set(hashes) != set(rows):
                raise ValueError("sealed file coverage differs")
            for pos, rowpath in rows.items():
                if hashlib.sha256(rowpath.read_bytes()).hexdigest() != hashes[pos]:
                    raise ValueError("dump row changed")
            arms.append(rows)
        values = []
        for i, pos in enumerate(positions):
            if window["roles"][pos] != "gen":
                raise ValueError("paired gate must score assistant-generated rows")
            g, n, r = normalize(logits[i]), log_probs(arms[0][pos], vocab), log_probs(arms[1][pos], vocab)
            argmax = int(np.argmax(g))
            values.append([float(np.dot(np.exp(g), g - n)), float(np.dot(np.exp(g), g - r)),
                           float(np.argmax(n) == argmax), float(np.argmax(r) == argmax),
                           float(np.dot(np.exp(r), r - n)), float(np.dot(np.exp(n), n - r))])
        values = np.asarray(values)
        means = values.mean(axis=0)
        statistics.append([float((values[:, 0] - values[:, 1]).sum()),
                           float((values[:, 3] - values[:, 2]).sum())])
        counts.append(len(values))
        summaries.append({"id": window["id"], "rows": len(values), "native_golden_kl": float(means[0]),
            "reference_golden_kl": float(means[1]), "native_top1": float(means[2]), "reference_top1": float(means[3]),
            "direct_reference_native_kl": float(means[4]), "direct_native_reference_kl": float(means[5])})
    bounds = paired_bounds(statistics, counts, a.bootstrap, a.seed)
    result = {"schema": "cuteafd.media.g4/1", "criterion": CRITERION, "set_sha256": panel["set_sha256"],
              "checkpoint": panel["checkpoint"], "golden_seal_sha256": file_hash(a.golden_seal),
              "path": "decode-shaped", "windows": summaries, **bounds,
              "pass": bounds["kl_increase_upper95"] <= .005 and bounds["top1_loss_upper95"] <= .005,
              "scope": "encoder-swap gate only; no prefill qualification or precision-default promotion"}
    write_new(a.out, result)
    print(json.dumps(result, indent=2))
    return result["pass"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    capture_parser = actions.add_parser("capture")
    for flag in ("windows", "media-root", "out", "host-dump", "server-dump"):
        capture_parser.add_argument("--" + flag, type=Path, required=True)
    capture_parser.add_argument("--mode", choices=("native", "reference"), required=True)
    capture_parser.add_argument("--features", type=Path)
    capture_parser.add_argument("--server-keys", action="store_true", help="map sealed keys through normal server image preparation")
    capture_parser.add_argument("--mapped-features", type=Path, help="private writable reference-feature directory mounted on the server")
    capture_parser.add_argument("--url", required=True)
    capture_parser.add_argument("--bench-token")
    capture_parser.add_argument("--timeout", type=float, default=240)
    capture_parser.add_argument("--quick", action="store_true")
    seal_parser = actions.add_parser("seal")
    for flag in ("windows", "golden", "out"):
        seal_parser.add_argument("--" + flag, type=Path, required=True)
    compare_parser = actions.add_parser("compare")
    for flag in ("windows", "golden", "golden-seal", "native", "reference", "features", "out"):
        compare_parser.add_argument("--" + flag, type=Path, required=True)
    compare_parser.add_argument("--bootstrap", type=int, default=5000)
    compare_parser.add_argument("--seed", type=int, default=20260829)
    for command in (capture_parser, seal_parser, compare_parser):
        command.add_argument("--family", choices=FAMILIES, default="mimo_v2",
                             help="require this family in the pinned set and probe responses")
    args = parser.parse_args()
    if args.action == "capture":
        capture(args)
    elif args.action == "seal":
        write_new(args.out, golden_seal(args.golden, load_panel(args)))
    elif not compare(args):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
