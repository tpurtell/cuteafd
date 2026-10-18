#!/usr/bin/env python3
"""Official Kimi K3 MXFP4 text oracle, one native layer resident at a time.

Block-residual history and current BF16 streams are durable NVMe files. Only an
atomic, fsynced layer manifest advances resume, so an interruption loses at most
one layer. Prefix qualification includes an independent unmodified-decoder arm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import (canonical, finish_golden, load_set, verify_snapshot,
                              write_scored_logits)
from families.kimi_k3.model import (LayerWeights, geometry, load_official,
                                    package_source_identity, source_identity)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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
    def __init__(self, root, binding):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        latest = root / "latest.json"
        self.latest = json.loads(latest.read_bytes()) if latest.exists() else None
        if self.latest and self.latest["binding"] != binding:
            raise ValueError("resume source, set, arithmetic or runtime differs")
        self.binding = binding
        if self.latest:
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

    def load(self, wid, hidden_size):
        import numpy as np
        import torch
        entry = self.latest["windows"][wid]
        def read(blob):
            raw = np.fromfile(self.contained(blob["path"]), dtype="<u2")
            return torch.from_numpy(raw).view(torch.bfloat16).reshape(-1, hidden_size).cuda()
        hidden = read(entry["hidden"])[None]
        blocks = [read(blob) for blob in entry["blocks"]]
        history = (torch.stack(blocks, dim=1) if blocks else
                   hidden.new_empty((hidden.shape[1], 0, hidden_size)))
        return hidden, history

    def begin(self, layer):
        required = self.binding.get("state_bytes_per_layer", 0) + 100 * 1024 ** 3
        if shutil.disk_usage(self.root).free < required:
            raise ValueError("reference state write would violate 100 GiB NVMe headroom")
        pending = self.root / ("layer%03d" % layer)
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        return {"schema": "cuteafd.kimi.layer-state/1", "binding": self.binding,
                "layer": layer, "windows": {},
                "checks": list(self.latest.get("checks", [])) if self.latest else [],
                "native_fp32_parameters": list(self.latest.get("native_fp32_parameters", [])) if self.latest else []}

    def save(self, pending, wid, hidden, append_block=False):
        import torch
        if hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all():
            raise ValueError("reference stream must remain finite native BF16")
        path = self.root / ("layer%03d" % pending["layer"]) / (wid + ".bin")
        raw = hidden.detach().cpu().contiguous().view(torch.uint16).numpy()
        with path.open("wb") as stream:
            raw.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        blob = {"path": str(path.relative_to(self.root)), "bytes": path.stat().st_size,
                "sha256": digest(path)}
        previous = self.latest["windows"][wid] if self.latest else None
        blocks = list(previous["blocks"]) if previous else []
        if append_block:
            folder = self.root / "blocks" / ("block%03d" % pending["layer"])
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (wid + ".bin")
            if target.exists():
                target.unlink()
            os.link(self.contained(previous["hidden"]["path"]), target)
            sync_directory(folder)
            blocks.append({**previous["hidden"], "path": str(target.relative_to(self.root))})
        pending["windows"][wid] = {"hidden": blob, "blocks": blocks}

    def commit(self, pending):
        sync_directory(self.root / ("layer%03d" % pending["layer"]))
        atomic_json(self.root / "latest.json", pending)
        previous = self.latest
        self.latest = pending
        if previous:
            shutil.rmtree(self.root / ("layer%03d" % previous["layer"]))


def native_layer(reference, config, weights, index):
    import torch
    with torch.device("meta"):
        layer = reference.KimiDecoderLayer(config, index)
    prefix = f"language_model.model.layers.{index}"
    required = 0
    for name, parameter in layer.named_parameters():
        full = prefix + "." + name
        if full in weights.index:
            _, descriptor = weights.descriptor(full)
            sizes = {"BF16": 2, "F32": 4}
            if descriptor["dtype"] not in sizes:
                raise ValueError("unsupported native dense dtype: " + full)
            required += parameter.numel() * sizes[descriptor["dtype"]]
        else:
            required += parameter.numel() * 2
    free, _ = torch.cuda.mem_get_info()
    if required + 16 * 1024 ** 3 > free:
        raise ValueError("native layer plus 16 GiB execution reserve does not fit")
    return weights.load_module(layer, prefix)


def official_mask(hidden):
    import torch
    size = hidden.shape[1]
    mask = hidden.new_zeros((size, size))
    mask.masked_fill_(torch.ones((size, size), dtype=torch.bool, device=hidden.device).triu(1), float("-inf"))
    return mask[None, None]


def official_check(out, arrays, per_layer):
    import numpy as np
    native, adapted = arrays
    logp = native.astype(np.float64)
    candidate = adapted.astype(np.float64)
    logp -= np.logaddexp.reduce(logp, axis=-1, keepdims=True)
    candidate -= np.logaddexp.reduce(candidate, axis=-1, keepdims=True)
    kl = float((np.exp(logp) * (logp - candidate)).sum(-1).mean())
    top1 = float(np.mean(native.argmax(-1) == adapted.argmax(-1)))
    relative = float(np.linalg.norm(native - adapted) / max(np.linalg.norm(native), 1e-30))
    result = {"schema": "cuteafd.kimi.official-forward/1", "tokens": len(native),
              "top1": top1, "kl": kl, "relative_logit_l2": relative,
              "max_abs_logit_difference": float(np.abs(native - adapted).max()),
              "comparison": "independent native decoder streams; original eager versus fixed-M128 causal tiles; same exact native weights and F32 output head",
              "passed": top1 >= .99 and kl <= 1e-4 and relative <= .01,
              "limits": {"top1_min": .99, "kl_max": 1e-4, "relative_logit_l2_max": .01},
              "layers": per_layer}
    atomic_json(out / "official-forward.json", result)
    if not result["passed"]:
        raise ValueError("official-forward numerical check failed")
    return result


def execute(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    import tokenizers
    import fla
    import compressed_tensors
    from shape_invariant import linear, qualify

    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(0)
    manifest = load_set(a.windows)
    if manifest["family"] != "kimi_k3" or any(w.get("media") for w in manifest["windows"]):
        raise ValueError("Kimi oracle supports only its own text panel")
    snapshot = verify_snapshot(manifest, a.snapshot)
    reference, config, quantization = load_official(a.snapshot)
    weights = LayerWeights(a.snapshot, quantization)
    # Reject an inconsistent release before embeddings, GPU admission or shard payload reads.
    with torch.device("meta"):
        first = reference.KimiDecoderLayer(config, 0)
    weights.validate_module(first, "language_model.model.layers.0")
    del first
    pins = {**source_identity(a.snapshot), "fla_source_sha256": package_source_identity(fla),
            "transformers_source_sha256": package_source_identity(transformers),
            "compressed_tensors_source_sha256": package_source_identity(compressed_tensors),
            "reference_adapter_sha256": digest(Path(__file__).with_name("model.py")),
            "reference_runner_sha256": digest(Path(__file__))}
    checking = getattr(a, "_prefix_probe", False)
    if not checking:
        a._kimi_probe_state = a.out.parent / (a.out.name + "-prefix-state")
        # Keep interrupted evidence, but leave exactly one raw probe in the seal.
        for previous in a.out.glob("prefix-gate-*"):
            archive = a.out.parent / "prefix-history"
            archive.mkdir(exist_ok=True)
            previous.rename(archive / previous.name)
    proof = qualify(a, manifest, execute)
    windows = copy.deepcopy(manifest["windows"])
    if checking:
        for name in ("official_short", "adapted_short"):
            source = copy.deepcopy(windows[0])
            source.update(id=name, tokens=source["tokens"][:128], score_from=1, roles=["ctx"] * 128)
            windows.append(source)
    binding = {"set_sha256": manifest["set_sha256"], "source_pin": pins,
               "snapshot_identity": snapshot, "torch": torch.__version__,
               "transformers": transformers.__version__, "fla": fla.__version__,
               "compressed_tensors": compressed_tensors.__version__,
               "compute_capability": list(torch.cuda.get_device_capability()),
               "image_sha256": os.environ.get("CUTEAFD_REFERENCE_IMAGE_SHA256", ""),
               "state_bytes_per_layer": sum(len(w["tokens"]) * config.hidden_size * 2 for w in windows),
               "windows": {w["id"]: hashlib.sha256(canonical(w["tokens"])).hexdigest() for w in windows}}
    state = LayerState(a._kimi_probe_state if checking else
                       a.out.parent / (a.out.name + "-layer-state"), binding)
    with torch.inference_mode():
        if state.latest is None:
            embedding = weights.weight("language_model.model.embed_tokens.weight")
            weights.close()
            pending = state.begin(-1)
            for window in windows:
                ids = torch.tensor(window["tokens"], device="cuda", dtype=torch.long)
                state.save(pending, window["id"], F.embedding(ids, embedding)[None])
            state.commit(pending)
            del embedding
        print(f"resume after layer {state.latest['layer']}; {len(windows)} streams", flush=True)
        for index in range(state.latest["layer"] + 1, config.num_hidden_layers):
            tick = time.monotonic()
            layer = native_layer(reference, config, weights, index)
            pending = state.begin(index)
            short_outputs = {}
            for window in windows:
                wid = window["id"]
                hidden, blocks = state.load(wid, config.hidden_size)
                original_blocks = blocks.shape[1]
                if wid == "official_short":
                    mask = None if layer.is_linear_attn else official_mask(hidden)
                    result, history = layer(hidden, attention_mask=mask, block_residual=blocks,
                                            use_cache=False, output_attentions=False)
                else:
                    with geometry(reference):
                        result, history = layer(hidden, attention_mask=None, block_residual=blocks,
                                                use_cache=False, output_attentions=False)
                append = index % config.attn_res_block_size == 0
                if history.shape[1] != original_blocks + int(append):
                    raise ValueError("official block-residual count differs")
                if append and not torch.equal(history[:, -1], hidden[0]):
                    raise ValueError("official block-residual state differs from exact prefix sum")
                state.save(pending, wid, result, append_block=append)
                if checking and wid in ("official_short", "adapted_short"):
                    short_outputs[wid] = result.float().cpu()
                del hidden, blocks, result, history
            if checking:
                difference = short_outputs["official_short"] - short_outputs["adapted_short"]
                pending["checks"].append({"layer": index, "max_abs_hidden_difference": difference.abs().max().item(),
                                          "relative_hidden_l2": (difference.norm() /
                                              short_outputs["official_short"].norm().clamp_min(1e-30)).item()})
            pending["native_fp32_parameters"] = sorted(set(pending["native_fp32_parameters"]) | weights.native_fp32)
            state.commit(pending)
            del layer
            torch.cuda.empty_cache()
            print(f"layer {index + 1}/{config.num_hidden_layers} committed; {time.monotonic() - tick:.1f}s; read {weights.bytes_read / 1e9:.3f} GB", flush=True)
        with torch.device("meta"):
            norm = reference.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            output_norm = reference.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            output_proj = torch.nn.Linear(config.hidden_size, 1, bias=False)
        weights.load_module(norm, "language_model.model.norm")
        weights.load_module(output_norm, "language_model.model.output_attn_res_norm")
        weights.load_module(output_proj, "language_model.model.output_attn_res_proj")
        head = weights.weight("language_model.lm_head.weight").float()
        weights.close()
        entries, short_logits = [], {}
        total_nll, total_rows = 0.0, 0
        for window in windows:
            wid = window["id"]
            hidden, blocks = state.load(wid, config.hidden_size)
            hidden = reference._apply_attn_res(hidden[0], blocks, output_proj, output_norm)[None]
            hidden = norm(hidden)
            start = window["score_from"]
            selected = hidden[0, start - 1:len(window["tokens"]) - 1]
            logits = linear(selected.float(), head).float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError("non-finite reference logits")
            if wid in ("official_short", "adapted_short"):
                short_logits[wid] = logits
            else:
                entry = write_scored_logits(a.out, window, logits)
                atomic_json(a.out / entry["path"] / "meta.json", entry)
                entries.append(entry)
                targets = torch.tensor(window["tokens"][start:], dtype=torch.long)
                total_nll += F.cross_entropy(torch.from_numpy(logits).double(), targets, reduction="sum").item()
                total_rows += len(logits)
            del hidden, blocks, selected, logits
        comparison = None
        if checking:
            comparison = official_check(a.out, [short_logits["official_short"], short_logits["adapted_short"]], state.latest["checks"])
        else:
            evidence = next(a.out.glob("prefix-gate-*/official-forward.json"), None)
            if evidence is None:
                raise ValueError("missing independent official forward evidence")
            comparison = json.loads(evidence.read_bytes())
            shutil.copyfile(evidence, a.out / "official-forward.json")
        finish_golden(a.out, manifest, entries, snapshot_identity=snapshot, source_pin=pins,
                      nll=total_nll / total_rows, scored_tokens=total_rows, seconds=time.monotonic() - started,
                      torch=torch.__version__, transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                      attention="eager", mtp=False, use_cache=False, prefix_qualification=proof,
                      fixed_rows=128, native_fp32_parameters=sorted(set(state.latest["native_fp32_parameters"]) | weights.native_fp32),
                      gates={"situ_beta": config.activation_situ_beta,
                             "situ_linear_beta": config.activation_situ_linear_beta,
                             "kda_gate_lower_bound": config.linear_attn_config["gate_lower_bound"]},
                      official_forward=comparison,
                      root_arithmetic={"source": "official checkpoint Python decoder and compressed-tensors MXFP4 decompressor",
                          "arithmetic": "Exact native group-32 MXFP4 dequantization; shipped dense dtypes; BF16 stream and full block-residual history; official Situ, latent MoE, bounded-gate FLA KDA and eager MLA; F32 accumulations and widened F32 head; fixed-M128 linears; MTP/cache off."},
                      platform={"capability": list(torch.cuda.get_device_capability()),
                                "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
    print(f"reference NLL {total_nll / total_rows:.12f}; {total_rows} scored rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "windows", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers", type=int, default=None)
    args = parser.parse_args()
    if args.layers is not None:
        raise ValueError("partial layers cannot qualify a Kimi reference")
    execute(args)


if __name__ == "__main__":
    main()
