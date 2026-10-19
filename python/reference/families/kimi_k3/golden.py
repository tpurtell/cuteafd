#!/usr/bin/env python3
"""Official Kimi K3 MXFP4 text oracle, one native layer resident at a time.

Block-residual history and current F32 streams are durable NVMe files. Only an
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
from families.kimi_k3.model import (LayerWeights, configure_f32, geometry, load_official,
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
    def __init__(self, root, binding, *, diagnostic_prefix=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        latest = root / "latest.json"
        self.latest = json.loads(latest.read_bytes()) if latest.exists() else None
        self.diagnostic_prefix = diagnostic_prefix
        if self.latest and self.latest.get("diagnostic_prefix") != diagnostic_prefix:
            raise ValueError("diagnostic prefix state cannot qualify or change its stop layer")
        if self.latest and self.latest["binding"] != binding:
            raise ValueError("resume source, set, arithmetic or runtime differs")
        self.binding = binding
        if self.latest:
            for entry in self.latest["windows"].values():
                for blob in [entry["hidden"], *entry["blocks"]]:
                    self.verify(blob)

    def require_scoring(self):
        if self.latest and "diagnostic_prefix" in self.latest:
            raise ValueError("diagnostic prefix state cannot be scored or published")

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
            f32 = self.binding.get("state_dtype") == "float32"
            raw = np.fromfile(self.contained(blob["path"]), dtype="<f4" if f32 else "<u2")
            value = torch.from_numpy(raw)
            if not f32:
                value = value.view(torch.bfloat16)
            return value.reshape(-1, hidden_size).cuda()
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
        flags = {} if self.diagnostic_prefix is None else {"diagnostic_prefix": self.diagnostic_prefix}
        return {"schema": "cuteafd.kimi.layer-state/1", "binding": self.binding, **flags,
                "layer": layer, "windows": {},
                "checks": list(self.latest.get("checks", [])) if self.latest else [],
                "native_fp32_parameters": list(self.latest.get("native_fp32_parameters", [])) if self.latest else []}

    def save(self, pending, wid, hidden, append_block=False):
        import torch
        f32 = self.binding.get("state_dtype") == "float32"
        expected = torch.float32 if f32 else torch.bfloat16
        if hidden.dtype != expected or not torch.isfinite(hidden).all():
            raise ValueError("reference stream must remain finite " + str(expected))
        path = self.root / ("layer%03d" % pending["layer"]) / (wid + ".bin")
        value = hidden.detach().cpu().contiguous()
        raw = (value if f32 else value.view(torch.uint16)).numpy()
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
    if getattr(weights, "dtype", None) == torch.float32:
        weights.release_experts()
        entries = weights.validate_module(layer, prefix)
        required, largest = 0, 0
        for _, _, full, _, parameter in entries:
            if ".experts." in full:
                stem = full.removesuffix(".weight")
                for suffix in (".weight_packed", ".weight_scale"):
                    _, descriptor = weights.descriptor(stem + suffix)
                    required += descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
                largest = max(largest, parameter.numel() * 4)
            else:
                required += parameter.numel() * 4
        free, _ = torch.cuda.mem_get_info()
        # Admission covers all 16K-window state/combine/KDA buffers plus slack.
        if required + largest * 5 + 64 * 1024 ** 3 > free:
            raise ValueError("packed layer plus strict F32 execution reserve does not fit")
        return weights.load_module(layer, prefix, stream_experts=True)
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
    stop = getattr(a, "stop_after_layer", None)
    state_root = (a._kimi_probe_state if getattr(a, "_prefix_probe", False) else
                  a.out.parent / (a.out.name + "-layer-state"))
    latest = state_root / "latest.json"
    if latest.exists() and "diagnostic_prefix" in json.loads(latest.read_bytes()) and stop is None:
        raise ValueError("diagnostic prefix state cannot be scored or published")
    if stop is not None and any((a.out / name).exists() for name in
                               ("meta.json", "prefix-qualification.json", "golden-seal.json")):
        raise ValueError("diagnostic prefix cannot reuse a qualified output directory")
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
    arithmetic = configure_f32()
    torch.manual_seed(0)
    manifest = load_set(a.windows)
    if manifest["family"] != "kimi_k3" or any(w.get("media") for w in manifest["windows"]):
        raise ValueError("Kimi oracle supports only its own text panel")
    snapshot = verify_snapshot(manifest, a.snapshot)
    reference, config, quantization = load_official(a.snapshot)
    if stop is not None and not 0 <= stop < config.num_hidden_layers:
        raise ValueError("diagnostic stop layer is outside checkpoint")
    weights = LayerWeights(a.snapshot, quantization, dtype=torch.float32,
                           kda_num_heads=config.linear_attn_config["num_heads"])
    # Check every layer shape and only the small A_log tails before GPU admission.
    for index in range(config.num_hidden_layers):
        with torch.device("meta"):
            layer = reference.KimiDecoderLayer(config, index)
        weights.validate_module(layer, f"language_model.model.layers.{index}")
        weights.close()
        del layer
    with torch.device("meta"):
        outer = [(torch.nn.Embedding(config.vocab_size, config.hidden_size), "language_model.model.embed_tokens"),
                 (torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False), "language_model.lm_head"),
                 (reference.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps), "language_model.model.norm"),
                 (reference.KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps), "language_model.model.output_attn_res_norm"),
                 (torch.nn.Linear(config.hidden_size, 1, bias=False), "language_model.model.output_attn_res_proj")]
    for module, prefix in outer:
        weights.validate_module(module, prefix)
    del outer, module
    print(f"native shape preflight passed; {len(weights.narrowed_a_log)} zero-padded KDA A_log tensors", flush=True)
    pins = {**source_identity(a.snapshot), "fla_source_sha256": package_source_identity(fla),
            "transformers_source_sha256": package_source_identity(transformers),
            "compressed_tensors_source_sha256": package_source_identity(compressed_tensors),
            "reference_adapter_sha256": digest(Path(__file__).with_name("model.py")),
            "reference_runner_sha256": digest(Path(__file__))}
    checking = getattr(a, "_prefix_probe", False)
    if not checking and stop is None:
        a._kimi_probe_state = a.out.parent / (a.out.name + "-prefix-state")
        # Keep interrupted evidence, but leave exactly one raw probe in the seal.
        for previous in a.out.glob("prefix-gate-*"):
            archive = a.out.parent / "prefix-history"
            archive.mkdir(exist_ok=True)
            previous.rename(archive / previous.name)
    # A diagnostic prefix is never passed to qualification or output scoring.
    proof = qualify(a, manifest, execute) if stop is None else None
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
               "state_dtype": "float32", "arithmetic": arithmetic,
               "expert_storage": "GPU packed cache; one F32 matrix resident per native Linear call",
               "state_bytes_per_layer": sum(len(w["tokens"]) * config.hidden_size * 4 for w in windows),
               "windows": {w["id"]: hashlib.sha256(canonical(w["tokens"])).hexdigest() for w in windows}}
    state = LayerState(a._kimi_probe_state if checking else
                       a.out.parent / (a.out.name + "-layer-state"), binding,
                       diagnostic_prefix=stop)
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
        for index in range(state.latest["layer"] + 1, config.num_hidden_layers if stop is None else stop + 1):
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
            weights.release_experts()
            torch.cuda.empty_cache()
            print(f"layer {index + 1}/{config.num_hidden_layers} committed; {time.monotonic() - tick:.1f}s; read {weights.bytes_read / 1e9:.3f} GB", flush=True)
        if stop is not None:
            weights.close()
            print(f"diagnostic only: retained layer {state.latest['layer']}; no scores or publication", flush=True)
            return
        state.require_scoring()
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
                      gates={"f32_policy": arithmetic, "expert_storage": binding["expert_storage"],
                             "situ_beta": config.activation_situ_beta,
                             "situ_linear_beta": config.activation_situ_linear_beta,
                             "kda_gate_lower_bound": config.linear_attn_config["gate_lower_bound"],
                             "kda_a_log_zero_padding": weights.padding_provenance()},
                      official_forward=comparison, arithmetic=arithmetic,
                      expert_storage=binding["expert_storage"],
                      root_arithmetic={"source": "official checkpoint Python decoder; exact E2M1/E8M0 group-32 MXFP4 decoded directly to F32",
                          "arithmetic": "Reference computed entirely in F32 from the native checkpoint: exact group-32 MXFP4 to F32; shipped BF16/F32 dense tensors upcast to F32; F32 activations, block history, router, norms, KDA recurrence and eager attention; TF32 disabled in Torch and Triton, highest matmul precision; one expert matrix dequantized at a time from a per-layer packed GPU cache; fixed-M128 linears; MTP/cache off.\n\nCheckpoint-format note: the release zero-pads native KDA A_log from 96 active heads to 128. Each affected tensor is narrowed to the first 96 only after all 32 unused F32 values are verified bit-exact positive zero. Per-layer shapes and tail SHA256 checks are recorded in reference_provenance.gates.kda_a_log_zero_padding; all other tensor shapes remain strict."},
                      platform={"capability": list(torch.cuda.get_device_capability()),
                                "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
    print(f"reference NLL {total_nll / total_rows:.12f}; {total_rows} scored rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "windows", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--stop-after-layer", type=int, default=None,
                        help="diagnostic only: retain residual at this zero-based layer; never score or publish")
    parser.add_argument("--diagnose-layer", type=int, default=None,
                        help="compare native/fixed sub-ops at layer 1 without running a golden")
    parser.add_argument("--diagnose-official-noise", action="store_true",
                        help="compare unmodified native batch 2 with retained batch-1 probe states")
    parser.add_argument("--diagnose-blas", choices=("cublas", "cublaslt"),
                        help="select a native Torch BLAS backend for both diagnostic arms")
    parser.add_argument("--diagnose-linear-variants", action="store_true",
                        help="compare fixed-row native/F32 expert arithmetic on controlled inputs")
    parser.add_argument("--probe-state", type=Path,
                        help="saved probe state containing exact block000 short-stream embeddings")
    args = parser.parse_args()
    if args.layers is not None:
        raise ValueError("partial layers cannot qualify a Kimi reference")
    if args.stop_after_layer is not None and (args.stop_after_layer < 0
            or args.diagnose_layer is not None or args.diagnose_official_noise
            or args.diagnose_linear_variants or args.diagnose_blas or args.probe_state):
        raise ValueError("--stop-after-layer requires a nonnegative layer and no other diagnostic mode")
    if args.diagnose_official_noise:
        if args.diagnose_layer is not None or args.diagnose_linear_variants or args.diagnose_blas:
            raise ValueError("official noise diagnosis uses only native cuBLAS batch geometry")
        from families.kimi_k3.diagnostic import diagnose_official_noise
        diagnose_official_noise(args, native_layer=native_layer, atomic_json=atomic_json,
                                digest=digest, state_type=LayerState, official_mask=official_mask)
        return
    if args.diagnose_layer is not None:
        from families.kimi_k3.diagnostic import diagnose
        diagnose(args, native_layer=native_layer, atomic_json=atomic_json, digest=digest,
                 state_type=LayerState, official_mask=official_mask)
    else:
        if args.probe_state is not None or args.diagnose_linear_variants or args.diagnose_blas:
            raise ValueError("diagnostic options are only supported with --diagnose-layer")
        execute(args)


if __name__ == "__main__":
    main()
