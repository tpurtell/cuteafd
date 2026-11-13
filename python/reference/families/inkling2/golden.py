#!/usr/bin/env python3
"""Layer-major Inkling 2 MXFP8 text golden; interrupted layers are replayed.

Checkpoint tensors stay native until use. Residuals and completed six-layer
blocks are fsynced F32 files, bound to the source, runtime and input panel.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import (admit_reference_memory, canonical, finish_golden, load_set,
                              scored_logits_nll, verify_snapshot, write_scored_logits)
from families.inkling2.model import (CHECKPOINT, REVISION, TEXT, Config, Weights,
    head_forward, layer_forward, output_norm, rms_norm, tensor_contract)
from families.inkling2.sources import ARITHMETIC, SGLANG, VLLM


SNAPSHOT_PINS = {
    "config.json": "7da68011c7c85af5e040114d3ebd5c8fdc8ce9030851a5706ebe88c379632405",
    "hf_quant_config.json": "778af00cedf00beb3f5b038abf75c992933d4639e07bba986ffa87efbb1eaf85",
    "model.safetensors.index.json": "51f87de7e81264fb2d8ea0b8f8b0b542da662204f26e2c83d06ae1c04f0addc9",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def source_identity(snapshot):
    if snapshot.name != REVISION or any(digest(snapshot / name) != sha for name, sha in SNAPSHOT_PINS.items()):
        raise ValueError("Inkling golden requires the pinned official MXFP8 snapshot")
    return expected_source_identity({})


def expected_source_identity(runtime_digests):
    if runtime_digests:
        raise ValueError("Inkling reference has no dynamically imported model source")
    return {"vllm": VLLM, "sglang": SGLANG, "snapshot": SNAPSHOT_PINS,
            "reference_adapter_sha256": digest(Path(__file__).with_name("model.py")),
            "reference_runner_sha256": digest(Path(__file__)),
            "arithmetic_sources_sha256": digest(Path(__file__).with_name("sources.py")),
            "shape_invariant_sha256": digest(ROOT / "python/reference/shape_invariant.py")}


def upstream_evidence(path, manifest, pins):
    """Bind an independently run pure-Torch helper comparison to this panel."""
    import math
    report = json.loads(Path(path).read_bytes())
    if (report.get("passed") is not True or report.get("panel_diagnostic_only")
            or report.get("panel_set_sha256") != manifest["set_sha256"]
            or report.get("reference_adapter_sha256") != pins["reference_adapter_sha256"]
            or report.get("revision") != REVISION
            or report.get("upstream", {}).get("revision") != VLLM["revision"]):
        raise ValueError("upstream helper evidence differs from this reference panel")
    window = next(w for w in manifest["windows"] if len(w["tokens"]) >= 640)
    if (report.get("window") != window["id"] or report.get("tokens") != len(window["tokens"])
            or report.get("window_tokens_sha256") != hashlib.sha256(canonical(window["tokens"])).hexdigest()):
        raise ValueError("upstream helper evidence window differs")
    for name in ("test_attention.py", "test_core_ops.py", "test_moe.py"):
        key = "tests/models/inkling2/" + name
        if report["upstream"].get("files", {}).get(key) != VLLM["files"][key]:
            raise ValueError("upstream helper evidence source hashes differ")
    for index in (0, 5):
        for name in ("static_conv", "l2_clip", "relative_attention_sink_gate", "route_gamma",
                     "dequant_4608", "dequant_2304"):
            check = report.get("measurements", {}).get(f"layer{index}/{name}", {})
            error = check.get("max_abs", float("nan"))
            if check.get("calls", 0) < 1 or not math.isfinite(error) or error > 2e-5:
                raise ValueError("upstream helper evidence lacks a passing intermediate")
    return {**report, "evidence_sha256": digest(path),
            "comparison": "Pinned unmodified upstream pure-Torch helpers on first real text window, "
                          "layers 0 and 5; intermediates only, not an independent full decoder"}


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


class LayerState:
    """Publish a complete manifest before retiring the previous partial stream."""
    def __init__(self, root, binding):
        self.root, self.binding = Path(root), binding
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "latest.json"
        self.latest = json.loads(path.read_bytes()) if path.exists() else None
        if self.latest:
            if (self.latest.get("schema") != "cuteafd.inkling2.layer-state/1"
                    or self.latest["binding"] != binding
                    or set(self.latest["windows"]) != set(binding["windows"])):
                raise ValueError("reference resume source, panel or runtime differs")
            for entry in self.latest["windows"].values():
                for blob in [entry["hidden"], *entry["blocks"]]:
                    self.verify(blob)

    def contained(self, name):
        path = self.root / name
        if not path.resolve().is_relative_to(self.root.resolve()) or path.is_symlink():
            raise ValueError("unsafe reference state path")
        return path

    def verify(self, blob):
        path = self.contained(blob["path"])
        if path.stat().st_size != blob["bytes"] or digest(path) != blob["sha256"]:
            raise ValueError("reference state blob differs")
        return path

    def load(self, wid, device):
        import numpy as np
        import torch
        def read(blob):
            raw = np.fromfile(self.verify(blob), dtype="<f4")
            return torch.from_numpy(raw.reshape(blob["shape"])).to(device)
        entry = self.latest["windows"][wid]
        return read(entry["hidden"]), [read(blob) for blob in entry["blocks"]]

    def begin(self, index):
        if index != (-1 if self.latest is None else self.latest["layer"] + 1):
            raise ValueError("reference state layers must be consecutive")
        if shutil.disk_usage(self.root).free < self.binding["state_bytes_per_layer"] + 100 * 2**30:
            raise ValueError("reference state write would violate 100 GiB NVMe headroom")
        folder = self.root / ("layer%03d" % index)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        return {"schema": "cuteafd.inkling2.layer-state/1", "binding": self.binding,
                "layer": index, "windows": {}, "seconds_per_layer":
                list(self.latest["seconds_per_layer"]) if self.latest else []}

    def save(self, pending, wid, hidden, *, append_block=False):
        import torch
        if (wid not in self.binding["windows"] or wid in pending["windows"]
                or hidden.dtype != torch.float32 or hidden.ndim != 2
                or not torch.isfinite(hidden).all()):
            raise ValueError("reference stream must be unique, finite F32")
        path = self.root / ("layer%03d" % pending["layer"]) / (wid + ".bin")
        with path.open("wb") as stream:
            hidden.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        blob = {"path": str(path.relative_to(self.root)), "bytes": path.stat().st_size,
                "sha256": digest(path), "shape": list(hidden.shape)}
        previous = self.latest["windows"][wid] if self.latest else None
        blocks = list(previous["blocks"]) if previous else []
        if append_block:
            if previous is None:
                raise ValueError("cannot append a missing residual block")
            folder = self.root / "blocks" / ("block%03d" % pending["layer"])
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (wid + ".bin")
            if target.exists():
                target.unlink()
            os.link(self.verify(previous["hidden"]), target)
            sync_directory(folder)
            sync_directory(folder.parent)
            sync_directory(self.root)
            blocks.append({**previous["hidden"], "path": str(target.relative_to(self.root))})
        pending["windows"][wid] = {"hidden": blob, "blocks": blocks}

    def commit(self, pending, seconds):
        if set(pending["windows"]) != set(self.binding["windows"]):
            raise ValueError("reference state window set is incomplete")
        for entry in pending["windows"].values():
            for blob in [entry["hidden"], *entry["blocks"]]:
                self.verify(blob)
        pending["seconds_per_layer"].append(seconds)
        sync_directory(self.root / ("layer%03d" % pending["layer"]))
        atomic_json(self.root / "latest.json", pending)
        previous, self.latest = self.latest, pending
        if previous:
            shutil.rmtree(self.root / ("layer%03d" % previous["layer"]))


def configure_f32():
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return {**ARITHMETIC, "matmul_precision": torch.get_float32_matmul_precision(),
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "deterministic_algorithms": True}


def execution_memory(cfg, length, index):
    """Live arithmetic bytes, excluding the separately counted native layer."""
    heads, _, dim = cfg.geometry(index)
    history = 1 + index // cfg.n_layers_per_attn_res_block
    residual = 4 * length * cfg.hidden_size * (3 * history + 12)
    attention = (4 * length * (heads * (3 * dim + cfg.rel_extent + cfg.d_rel) + 4 * cfg.hidden_size)
                 + 4 * 128 * (length + 128) * 8)
    moe = 4 * (length * (cfg.num_experts_per_tok * cfg.hidden_size + cfg.n_routed_experts
                        + 6 * cfg.intermediate_size + 3 * cfg.hidden_size)
               + 2 * cfg.hidden_size * cfg.intermediate_size)
    return residual + max(attention, moe)


def release(device):
    import torch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def execute(a):
    import importlib.metadata
    import numpy as np
    import torch
    import torch.nn.functional as F
    from shape_invariant import qualify
    started = time.monotonic()
    pins = source_identity(a.snapshot)
    cfg = Config.from_dict(json.loads((a.snapshot / "config.json").read_bytes()))
    weights = Weights(a.snapshot, a.device)
    weights.validate(cfg)
    print("native Inkling MXFP8/BF16/F32 shape preflight passed", flush=True)
    if a.check_only:
        return
    arithmetic = configure_f32()
    torch.manual_seed(0)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    manifest = load_set(a.windows, "inkling2")
    if manifest["checkpoint"] != CHECKPOINT or any(w.get("media") for w in manifest["windows"]):
        raise ValueError("Inkling golden accepts only its pinned text panel")
    for window in manifest["windows"]:
        if any(token >= cfg.unpadded_vocab_size for token in window["tokens"]):
            raise ValueError("Inkling panel token exceeds the unpadded vocabulary")
    snapshot = verify_snapshot(manifest, a.snapshot)
    a.out.mkdir(parents=True, exist_ok=True)
    stop = a.stop_after_layer
    if stop is not None and not 0 <= stop < cfg.num_hidden_layers:
        raise ValueError("diagnostic stop layer is outside checkpoint")
    if stop is not None and any((a.out / name).exists() for name in ("meta.json", "prefix-qualification.json")):
        raise ValueError("diagnostic stop cannot reuse a qualified output")
    evidence = None
    if stop is None and not getattr(a, "_prefix_probe", False):
        if not getattr(a, "upstream_evidence", None):
            raise ValueError("full reference requires real-panel upstream helper evidence")
        evidence = upstream_evidence(a.upstream_evidence, manifest, pins)
    harness = os.environ.get("CUTEAFD_REFERENCE_COMMIT", "")
    if harness and (len(harness) != 40 or any(c not in "0123456789abcdef" for c in harness)):
        raise ValueError("reference harness commit must be a full Git SHA")
    proof = qualify(a, manifest, execute) if stop is None else None
    windows = manifest["windows"]
    binding = {"set_sha256": manifest["set_sha256"], "source_pin": pins,
               "snapshot_identity": snapshot, "torch": torch.__version__,
               "cuda": torch.version.cuda, "device": str(device), "arithmetic": arithmetic,
               "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
               "image_sha256": os.environ.get("CUTEAFD_REFERENCE_IMAGE_SHA256", ""),
               "state_dtype": "float32", "harness_commit": harness,
               "state_bytes_per_layer": sum(len(w["tokens"]) * cfg.hidden_size * 4 for w in windows),
               "windows": {w["id"]: hashlib.sha256(canonical(w["tokens"])).hexdigest() for w in windows}}
    state = LayerState(a.out.parent / (a.out.name + "-layer-state"), binding)
    with torch.inference_mode():
        if state.latest is None:
            tick = time.monotonic()
            embedding = weights.tensor(TEXT + "embed.weight").float()
            norm = weights.tensor(TEXT + "embed_norm.weight").float()
            weights.close()
            pending = state.begin(-1)
            for window in windows:
                tokens = torch.tensor(window["tokens"], device=device, dtype=torch.long)
                hidden = rms_norm(F.embedding(tokens, embedding), norm, cfg.rms_norm_eps)
                state.save(pending, window["id"], hidden)
                del hidden, tokens
            state.commit(pending, time.monotonic() - tick)
            del embedding, norm
            release(device)
        for index in range(state.latest["layer"] + 1, cfg.num_hidden_layers if stop is None else stop + 1):
            tick = time.monotonic()
            if device.type == "cuda":
                native = sum(weights.descriptor(name)["data_offsets"][1] - weights.descriptor(name)["data_offsets"][0]
                             for name in tensor_contract(cfg, index))
                # Dense tensors are expanded to F32 as the layer is loaded.
                dense = sum(2 * torch.tensor(shape).prod().item() for shape, dtype in tensor_contract(cfg, index).values()
                            if dtype == "BF16")
                peak = native + dense + execution_memory(cfg, max(len(w["tokens"]) for w in windows), index)
                admit_reference_memory(torch.cuda, peak)
            layer = weights.load_layer(index, cfg)
            pending = state.begin(index)
            for window in windows:
                hidden, blocks = state.load(window["id"], device)
                result = layer_forward(hidden, blocks, layer, cfg, index)
                state.save(pending, window["id"], result,
                           append_block=index % cfg.n_layers_per_attn_res_block == 0)
                del hidden, blocks, result
            state.commit(pending, time.monotonic() - tick)
            del layer
            release(device)
            print(f"layer {index + 1}/{cfg.num_hidden_layers} committed; {time.monotonic() - tick:.1f}s; "
                  f"logical reads {weights.read_bytes / 1e9:.3f}GB", flush=True)
        if stop is not None:
            print(f"diagnostic stopped at layer {stop}; no scores or publication", flush=True)
            return
        head = weights.tensor(TEXT + "unembed.weight").float()
        norm = weights.tensor(TEXT + "norm.weight").float()
        query = weights.tensor(TEXT + "attn_res.output_query").float()
        weights.close()
        entries, role_stats = [], {role: {"rows": 0, "loss": 0., "top1": 0} for role in ("gen", "ctx")}
        for window in windows:
            hidden, blocks = state.load(window["id"], device)
            hidden = output_norm(hidden, blocks, query, norm, cfg)
            start, end = window["score_from"], len(window["tokens"])
            # Keep full-vocabulary output on disk, never all windows on the GPU.
            logits = np.empty((end - start, cfg.unpadded_vocab_size), dtype="<f4")
            for first in range(start, end, 128):
                last = min(first + 128, end)
                logits[first - start:last - start] = head_forward(hidden[first - 1:last - 1], head, cfg).cpu().numpy()
            entry = write_scored_logits(a.out, window, logits)
            atomic_json(a.out / entry["path"] / "meta.json", entry)
            entries.append(entry)
            roles = np.asarray(window["roles"][start:])
            positions = np.arange(start, end)
            tokens = np.asarray(window["tokens"])
            for role, stats in role_stats.items():
                mask = roles == role
                if mask.any():
                    selected = logits[mask]
                    rows = int(mask.sum())
                    stats["loss"] += scored_logits_nll(selected, tokens, positions[mask])["nll"] * rows
                    stats["rows"] += rows
                    stats["top1"] += int((selected.argmax(-1) == tokens[positions[mask]]).sum())
            del hidden, blocks, logits
        rows = sum(entry["scored_tokens"] for entry in entries)
        platform = ({"capability": list(torch.cuda.get_device_capability(device)),
                     "gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda}
                    if device.type == "cuda" else {"device": "cpu"})
        finish_golden(a.out, manifest, entries, snapshot_identity=snapshot, source_pin=pins,
            source_identity_provider="families.inkling2.golden",
            harness_commit=harness, **({"official_forward": evidence} if evidence else {}),
            nll=sum(e["nll"] * e["scored_tokens"] for e in entries) / rows, scored_tokens=rows,
            roles={role: {"scored_rows": s["rows"], "nll": s["loss"] / s["rows"] if s["rows"] else None,
                          "api_text_top1": s["top1"] / s["rows"] if s["rows"] else None}
                   for role, s in role_stats.items()},
            seconds=time.monotonic() - started, seconds_per_layer=state.latest["seconds_per_layer"][1:],
            torch=torch.__version__, tokenizers=importlib.metadata.version("tokenizers"),
            transformers="not used; independent text forward", attention="eager", mtp=False, use_cache=False,
            prefix_qualification=proof, fixed_rows=128, arithmetic=arithmetic, platform=platform,
            root_arithmetic={"source": "Independent F32 port of two pinned vendor Inkling 2 integrations",
                             "arithmetic": "; ".join(ARITHMETIC.values())})
    print(f"reference complete: {rows} scored rows; {a.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--upstream-evidence", type=Path,
                        help="passing real-panel pinned pure-Torch helper evidence JSON")
    args = parser.parse_args()
    if not args.check_only and (args.windows is None or args.out is None):
        parser.error("--windows and --out are required for scoring")
    execute(args)


if __name__ == "__main__":
    main()
