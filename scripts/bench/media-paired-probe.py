#!/usr/bin/env python3
"""Capture media teacher-forced rows and gate paired encoder effects against goldens."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from fidelity_media import has_media_adapter, media_families, read_fixture
from fidelity_windows import canonical, load_set, validate_reference

CRITERION = ("the paired difference against the golden decides (\u22640.005 nat KL increase, "
             "\u22640.5 point top-1 loss, one-sided 95% window bootstrap). "
             "Direct KL between the arms is a diagnostic only.")


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(canonical(value) + b"\n")


def probe_token(token=None, token_file=None):
    if token_file is not None:
        if token is not None:
            raise ValueError("choose one bench token source")
        try:
            token = Path(token_file).read_text().strip()
        except (OSError, UnicodeError):
            raise ValueError("cannot read bench token file") from None
        if not token:
            raise ValueError("bench token file is empty")
    elif token is None:
        token = os.environ.get("CUTEAFD_API_KEY")
    if token and (not token.isascii() or any(ord(c) < 33 or ord(c) == 127 for c in token)):
        raise ValueError("invalid bench token")
    return token


def http(url, token, body=None, timeout=240):
    headers = {"Content-Type": "application/json"}
    if token:
        headers.update({"x-cuteafd-bench": token, "Authorization": "Bearer " + token})
    request = urllib.request.Request(url, None if body is None else canonical(body), headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


FAMILIES = media_families()
GLM_TRANSFORMERS_REVISION = "62d7ebd7de4938e072b7aaeb881593b79dc56835"
GLM_MODELING_SHA256 = "f6dfea36d663a5257cd10ad61924c4f90c29c38f14f40b2af1e446714168b05d"
GLM_IMAGE_PROCESSING_SHA256 = "c3065c32ece0cd46adf74a1aef5707da48a0596578503dabf47f874da0716663"


def load_panel(a):
    family = getattr(a, "family", "mimo_v2")
    if not has_media_adapter(family):
        raise ValueError("unsupported paired media family")
    return load_set(a.windows, family)


def digest_hex(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def check_media_echo(window, actual):
    expected = window.get("media", [])
    if not isinstance(actual, list) or len(expected) != len(actual):
        raise ValueError("prepared media echo differs")
    for span, got in zip(expected, actual):
        if not isinstance(got, dict) or any(got.get(k) != span[k] for k in ("start", "len", "kind", "grid")):
            raise ValueError("prepared media echo differs")
        if span.get("identity"):
            identity = got.get("identity")
            if (not isinstance(identity, dict) or not digest_hex(got.get("key"))
                    or got.get("input_key") != span["key"]
                    or identity.get("processor") != span["identity"]["processor"]
                    or any(not digest_hex(identity.get(k)) for k in ("encoder", "processor"))):
                raise ValueError("prepared input/admitted encoder identity differs")
        elif got.get("key") != span["key"]:
            raise ValueError("prepared media echo differs")


def check_record(window, response, checkpoint, mode, features_root=None, family=None):
    record = response.get("probe", {})
    if (record.get("error") or not record.get("engine") or not record.get("cold")
            or not record.get("no_speculation") or record.get("cached_tokens") != 0
            or record.get("score_path") != "decode" or record.get("prompt_ids") != window["tokens"]
            or response.get("server", {}).get("model") != checkpoint
            or family is not None and response.get("server", {}).get("family") != family):
        raise ValueError("media scoring identity/cold/path contract not honored")
    images = [span for span in window.get("media", []) if span.get("kind", "image") == "image"]
    check_media_echo({"media": images}, record.get("media", []))
    audio_fields = ("start", "len", "key", "samples", "pcm_sha256")
    expected_audio = [{k: span[k] for k in audio_fields} for span in window.get("media", []) if span.get("kind") == "audio"]
    actual_audio = [{k: span.get(k) for k in audio_fields} for span in record.get("audio", [])]
    if actual_audio != expected_audio:
        raise ValueError("prepared audio echo differs")
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
        metadata = [json.loads((features_root / (s["key"] + ".json")).read_text()) for s in window.get("media", [])]
        expected_provenance = {"mode": "reference_features", "probe_only": True,
                               "encoder_bypassed": True, "features": metadata}
        if family == "glm5_flash":
            # GLM echoes a source revision outside its hash-only feature identity.
            # Bind that revision to both locked arithmetic sources, not just its label.
            for feature in metadata:
                identity = feature.get("snapshot_identity")
                if (not isinstance(identity, dict)
                        or identity.get("modeling_sha256") != GLM_MODELING_SHA256
                        or identity.get("image_processing_sha256") != GLM_IMAGE_PROCESSING_SHA256):
                    raise ValueError("reference feature modeling source differs from pinned GLM export")
            expected_provenance["modeling_source_revision"] = GLM_TRANSFORMERS_REVISION
        if provenance != expected_provenance:
            raise ValueError("reference feature provenance differs from sealed export")


def bound_panel(panel, path):
    """A sidecar adds input identity without changing the immutable set or golden."""
    if path is None:
        return panel, None
    payload = Path(path).read_bytes()
    sidecar = json.loads(payload)
    result, _ = bound_panel_document(panel, sidecar)
    return result, {"sha256": hashlib.sha256(payload).hexdigest(),
                    "canonical_sha256": hashlib.sha256(canonical(sidecar)).hexdigest(), "document": sidecar}


def bound_panel_document(panel, sidecar):
    if (sidecar.get("schema") != "cuteafd.media.input-binding/1"
            or sidecar.get("set_sha256") != panel["set_sha256"]):
        raise ValueError("input binding set identity differs")
    identities = sidecar.get("images", {})
    keys = {s["key"] for w in panel["windows"] for s in w.get("media", [])}
    if set(identities) != keys:
        raise ValueError("input binding image keys differ")
    result = json.loads(json.dumps(panel))
    for window in result["windows"]:
        for span in window.get("media", []):
            identity = identities[span["key"]]
            if (not isinstance(identity, dict) or set(identity) != {"encoder", "processor"}
                    or any(not isinstance(v, str) or len(v) != 64
                           or any(c not in "0123456789abcdef" for c in v) for v in identity.values())):
                raise ValueError("invalid input binding identity")
            if span.get("identity", identity) != identity:
                raise ValueError("input binding conflicts with sealed identity")
            span["identity"] = identity
    return result, None


def capture(a):
    a.bench_token = probe_token(a.bench_token, getattr(a, "bench_token_file", None))
    panel, binding = bound_panel(load_panel(a), getattr(a, "input_binding", None))
    if a.out.exists():
        raise ValueError("capture output must be new")
    models = http(a.url.rstrip("/") + "/v1/models", a.bench_token)
    records = [m for m in models.get("data", []) if m.get("id") == panel["checkpoint"]]
    kinds = {span.get("kind", "image") for window in panel["windows"] for span in window.get("media", [])}
    required = {"audio" if kind == "audio" else "vision" for kind in kinds} or {"vision"}
    if len(records) != 1 or any(records[0].get("capabilities", {}).get(kind) is not True for kind in required):
        raise ValueError("checkpoint must advertise " + "/".join(sorted(required)))
    if a.mode == "reference" and a.features is None:
        raise ValueError("reference capture needs --features")
    selected = [w for w in panel["windows"] if not a.quick or w["id"] in panel["quick_windows"]]
    a.out.mkdir(parents=True)
    entries, server = [], None
    admitted_encoder = None
    started = time.monotonic()
    for window in selected:
        media, audio, audio_content = [], [], []
        for span in window.get("media", []):
            data = read_fixture(a.media_root, span)
            if span.get("kind") == "audio":
                from fidelity_media import read_pcm_fixture
                read_pcm_fixture(a.media_root, span)
                audio.append({k: span[k] for k in ("start", "len", "key", "samples", "pcm_sha256")})
                audio_content.append({"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(data).decode(), "format": "wav"}})
            else:
                media.append({**span, "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}})
        leaf = window["id"]
        if (a.host_dump / leaf).exists():
            raise ValueError("dump leaf already exists")
        body = {"body": {"model": panel["checkpoint"], "messages": [{"role": "user", "content": "media fidelity probe"}],
                         "max_tokens": 1, "temperature": 0, "stream": False},
                "spec": {"prompt_ids": window["tokens"], "media": media, "score_from": window["score_from"],
                         "cold": True, "no_speculation": True, "score_path": "decode", "verify_rows": 1,
                         "top_k": 1, "dump_rows": str(a.server_dump / leaf)}}
        if audio:
            if a.mode != "native":
                raise ValueError("audio capture requires native mode and an official-feature LM golden")
            body["spec"]["audio"] = audio
            body["body"]["messages"][0]["content"] = [*audio_content, {"type": "text", "text": "media fidelity probe"}]
        try:
            response = http(a.url.rstrip("/") + "/v1/bench/probe", a.bench_token, body, a.timeout)
        except urllib.error.HTTPError as error:
            write_new(a.out / (leaf + ".http-error.json"), {"status": error.code,
                "body": error.read().decode("utf-8", errors="replace")})
            raise
        # Retain an invalid response too; it is evidence of a failed live gate.
        write_new(a.out / (leaf + ".json"), response)
        check_record(window, response, panel["checkpoint"], a.mode, a.features, panel["family"])
        for span in response["probe"].get("media", []):
            identity = span.get("identity")
            if identity is not None:
                if admitted_encoder is not None and identity != admitted_encoder:
                    raise ValueError("admitted encoder changed during capture")
                admitted_encoder = identity
        current = response["server"]
        if server is not None and current != server:
            raise ValueError("server build/settings changed during capture")
        server = current
        rows = dump_rows(a.host_dump / leaf, window, None)
        entries.append({"id": leaf, "admitted_media": response["probe"].get("media", []), "response_sha256": hashlib.sha256((a.out / (leaf + ".json")).read_bytes()).hexdigest(),
                        "path": str((a.host_dump / leaf).resolve()),
                        "manifest_sha256": hashlib.sha256((a.host_dump / leaf / "manifest.jsonl").read_bytes()).hexdigest(),
                        "files": [{"position": pos, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for pos, path in rows.items()]})
    write_new(a.out / "capture.json", {"schema": "cuteafd.media.paired.capture/1", "mode": a.mode,
        "set_sha256": panel["set_sha256"], "checkpoint": panel["checkpoint"], "server": server,
        "input_binding": binding,
        "quick": a.quick, "seconds": time.monotonic() - started, "windows": entries})


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
    validate_reference(meta, panel)
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


def fidelity_bar(expert_bpw):
    if (isinstance(expert_bpw, bool) or not isinstance(expert_bpw, (float, int))
            or not np.isfinite(expert_bpw) or expert_bpw <= 0):
        raise ValueError("native-only compare requires finite positive --expert-bpw")
    kl, top1 = (.03, .95) if expert_bpw >= 6 else ((.05, .93) if expert_bpw >= 3 else (.07, .90))
    return {"expert_bpw": expert_bpw, "max_kl": kl, "min_top1": top1,
            "source": "checkpoint routed-expert bits per weight; excludes dense/core/head precision"}


def native_metrics(summaries, bar):
    if not summaries:
        return None
    rows = sum(s["rows"] for s in summaries)
    kl = sum(s["native_golden_kl"] * s["rows"] for s in summaries) / rows
    top1 = sum(s["native_top1"] * s["rows"] for s in summaries) / rows
    return {"rows": rows, "native_golden_kl": kl, "native_top1": top1,
            "pass": kl <= bar["max_kl"] and top1 >= bar["min_top1"]}


def report_provenance(captures, paired):
    admitted = [[{"id": w["id"], "media": w["admitted_media"]} for w in capture["windows"]
                 if any(s.get("identity") for s in w.get("admitted_media", []))] for capture in captures]
    bindings = [capture.get("input_binding") for capture in captures]
    fields = {"admitted_media": admitted if paired else admitted[0],
              "input_bindings" if paired else "input_binding": bindings if paired else bindings[0]}
    return {key: value for key, value in fields.items()
            if (any(value) if paired else bool(value))}


def compare(a):
    paired = a.reference is not None
    if paired != (a.features is not None):
        raise ValueError("--reference and --features must be supplied together")
    bar = None if paired else fidelity_bar(getattr(a, "expert_bpw", None))
    roots = (a.native, a.reference) if paired else (a.native,)
    panel = load_panel(a)
    seal = json.loads(a.golden_seal.read_text())
    if seal != golden_seal(a.golden, panel):
        raise ValueError("golden files differ from immutable seal")
    meta = json.loads((a.golden / "meta.json").read_text())
    validate_reference(meta, panel)
    if meta["set_sha256"] != panel["set_sha256"] or meta["checkpoint"] != panel["checkpoint"]:
        raise ValueError("golden provenance differs")
    captures = [json.loads((root / "capture.json").read_text()) for root in roots]
    identity_panels = []
    for capture_record, mode in zip(captures, ("native", "reference")):
        if (capture_record["schema"] != "cuteafd.media.paired.capture/1" or capture_record["mode"] != mode
                or capture_record["set_sha256"] != panel["set_sha256"]
                or capture_record["checkpoint"] != panel["checkpoint"]):
            raise ValueError("arm capture identity differs")
        binding = capture_record.get("input_binding")
        identity_panel = panel
        if binding is not None:
            if (not digest_hex(binding.get("sha256"))
                    or hashlib.sha256(canonical(binding["document"])).hexdigest() != binding["canonical_sha256"]
                    or binding["document"].get("set_sha256") != panel["set_sha256"]):
                raise ValueError("capture input binding provenance differs")
            identity_panel, _ = bound_panel_document(panel, binding["document"])
        identity_windows = {w["id"]: w for w in identity_panel["windows"]}
        for sealed in capture_record["windows"]:
            window = identity_windows.get(sealed["id"])
            if window is None:
                raise ValueError("capture binding windows differ")
            if binding is not None or "admitted_media" in sealed or any(s.get("identity") for s in window.get("media", [])):
                check_media_echo(window, sealed.get("admitted_media", []))
        identity_panels.append(identity_windows)
    # Only the image tag may differ; commit and all effective model knobs must match.
    servers = json.loads(json.dumps([c["server"] for c in captures]))
    for server in servers:
        server.get("build", {}).pop("image", None)
    if paired and servers[0] != servers[1]:
        raise ValueError("arms have different builds or non-encoder settings")
    arm_maps = [{w["id"]: w for w in c["windows"]} for c in captures]
    expected = {w["id"] for w in panel["windows"]}
    if any(c["quick"] for c in captures):
        if not paired:
            raise ValueError("native-only gate requires the complete panel, not quick windows")
        expected = set(panel["quick_windows"])
    if (any(c["quick"] != captures[0]["quick"] for c in captures)
            or any(set(arm) != expected or len(arm) != len(c["windows"])
                   for arm, c in zip(arm_maps, captures))):
        raise ValueError("arms score different/incomplete windows")
    qualified_vocab = validate_reference(meta, panel)
    gold = {w["id"]: w for w in meta["windows"]}
    if set(gold) != {w["id"] for w in panel["windows"]} or len(gold) != len(meta["windows"]):
        raise ValueError("golden panel coverage differs")
    statistics, counts, summaries = [], [], []
    for window in panel["windows"]:
        if window["id"] not in arm_maps[0]:
            continue
        entry = gold[window["id"]]
        positions = list(range(window["score_from"], len(window["tokens"])))
        if entry["positions"] != positions or entry.get("media", []) != window.get("media", []):
            raise ValueError("golden media/position identity differs")
        vocab = entry["vocab"]
        if vocab != qualified_vocab:
            raise ValueError("golden vocabulary differs from qualified reference")
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
        for arm, capture_root, mode, identity_windows in zip(arm_maps, roots, ("native", "reference"), identity_panels):
            sealed = arm[window["id"]]
            response_path = capture_root / (window["id"] + ".json")
            if hashlib.sha256(response_path.read_bytes()).hexdigest() != sealed["response_sha256"]:
                raise ValueError("capture response changed")
            response = json.loads(response_path.read_text())
            check_record(identity_windows[window["id"]], response, panel["checkpoint"], mode, a.features, panel["family"])
            if "admitted_media" in sealed and sealed["admitted_media"] != response["probe"].get("media", []):
                raise ValueError("capture admitted media differs from sealed response")
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
            g, n = normalize(logits[i]), log_probs(arms[0][pos], vocab)
            argmax = int(np.argmax(g))
            if not paired:
                values.append([float(np.dot(np.exp(g), g - n)), float(np.argmax(n) == argmax)])
                continue
            r = log_probs(arms[1][pos], vocab)
            values.append([float(np.dot(np.exp(g), g - n)), float(np.dot(np.exp(g), g - r)),
                           float(np.argmax(n) == argmax), float(np.argmax(r) == argmax),
                           float(np.dot(np.exp(r), r - n)), float(np.dot(np.exp(n), n - r))])
        values = np.asarray(values)
        means = values.mean(axis=0)
        if not paired:
            summary = {"id": window["id"], "rows": len(values),
                       "kind": "media" if window.get("media", []) else "control",
                       "native_golden_kl": float(means[0]), "native_top1": float(means[1])}
            summary["pass"] = native_metrics([summary], bar)["pass"]
            summaries.append(summary)
            continue
        statistics.append([float((values[:, 0] - values[:, 1]).sum()),
                           float((values[:, 3] - values[:, 2]).sum())])
        counts.append(len(values))
        summaries.append({"id": window["id"], "rows": len(values), "native_golden_kl": float(means[0]),
            "reference_golden_kl": float(means[1]), "native_top1": float(means[2]), "reference_top1": float(means[3]),
            "direct_reference_native_kl": float(means[4]), "direct_native_reference_kl": float(means[5])})
    if not paired:
        image_ids = {w["id"] for w in panel["windows"]
                     if any(span["kind"] == "image" for span in w.get("media", []))}
        groups = {"pooled": summaries,
                  "image": [s for s in summaries if s["id"] in image_ids],
                  "media": [s for s in summaries if s["kind"] == "media"],
                  "control": [s for s in summaries if s["kind"] == "control"]}
        metrics = {name: native_metrics(group, bar) for name, group in groups.items()}
        result = {"schema": "cuteafd.media.g4.native/1", "set_sha256": panel["set_sha256"],
                  "checkpoint": panel["checkpoint"], "golden_seal_sha256": file_hash(a.golden_seal),
                  "path": "decode-shaped", "bar": bar, "windows": summaries, **metrics,
                  **report_provenance(captures, paired=False),
                  "criterion": "native vs official golden; pooled and each nonempty media/image/control group meet expert bit-tier bar",
                  "pass": all(m["pass"] for m in metrics.values() if m is not None),
                  "scope": "following-text decode fidelity only; no prefill qualification or precision-default promotion"}
        write_new(a.out, result)
        print(json.dumps(result, indent=2))
        return result["pass"]
    bounds = paired_bounds(statistics, counts, a.bootstrap, a.seed)
    result = {"schema": "cuteafd.media.g4/1", "criterion": CRITERION, "set_sha256": panel["set_sha256"],
              "checkpoint": panel["checkpoint"], "golden_seal_sha256": file_hash(a.golden_seal),
              "path": "decode-shaped", "windows": summaries, **bounds,
              **report_provenance(captures, paired=True),
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
    capture_parser.add_argument("--input-binding", type=Path)
    capture_parser.add_argument("--url", required=True)
    auth = capture_parser.add_mutually_exclusive_group()
    auth.add_argument("--bench-token")
    auth.add_argument("--bench-token-file", type=Path)
    capture_parser.add_argument("--timeout", type=float, default=240)
    capture_parser.add_argument("--quick", action="store_true")
    seal_parser = actions.add_parser("seal")
    for flag in ("windows", "golden", "out"):
        seal_parser.add_argument("--" + flag, type=Path, required=True)
    compare_parser = actions.add_parser("compare")
    for flag in ("windows", "golden", "golden-seal", "native", "out"):
        compare_parser.add_argument("--" + flag, type=Path, required=True)
    compare_parser.add_argument("--reference", type=Path, help="optional reference-feature diagnostic arm")
    compare_parser.add_argument("--features", type=Path, help="sealed features for the reference arm")
    compare_parser.add_argument("--expert-bpw", type=float,
                                help="checkpoint routed-expert bpw; required for native-only fidelity gate")
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
