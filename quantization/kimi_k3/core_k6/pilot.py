#!/usr/bin/env python3
"""Bounded format decisions using CoreEngine's published causal predecessors.

Calibration/held-out prompts are disjoint. Candidates never affect replay.
This mode cannot export or publish a checkpoint or change production recipes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

LEGACY_ARMS = ("mcg", "mcg-post-e4m3", "e4m3")
FORMAT_ARMS = ("mcg-k6", "e4m3-k6", "mcg-k5", "e4m3-k5", "mcg-k4", "e4m3-k4", "fp8-block128", "fp8-mx32")
EXPERT_ARMS = ("mcg-k2", "e4m3-k2", "e4m3-k3")
ARMS = LEGACY_ARMS
SCHEMA = "cuteafd-core-codebook-pilot-v1"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot", "replay-snapshot", "tokens", "root"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--layers", default="1,3")
    result.add_argument("--arms", default=",".join(ARMS))
    result.add_argument("--calibration-tokens", type=int, default=32768)
    result.add_argument("--heldout-tokens", type=int, default=4096)
    result.add_argument("--expert-count", type=int, choices=(0, 32), default=0)
    result.add_argument("--expert-source-snapshot", type=Path)
    result.add_argument("--preflight", action="store_true")
    return result


def configuration(argv=None):
    args = parser().parse_args(argv)
    args.layers = tuple(int(value) for value in args.layers.split(","))
    args.arms = tuple(args.arms.split(","))
    if (not args.layers or tuple(sorted(set(args.layers))) != args.layers
            or args.layers[0] < 1 or args.layers[-1] > 3):
        raise ValueError("bounded pilot requires sorted unique layers in 1..3")
    if args.arms not in (LEGACY_ARMS, FORMAT_ARMS):
        raise ValueError("pilot requires one complete ordered arm inventory")
    if not 32768 <= args.calibration_tokens <= 65536 or not 1 <= args.heldout_tokens <= 16384:
        raise ValueError("pilot token budget outside admission")
    if args.expert_count and (args.expert_source_snapshot is None or 1 not in args.layers):
        raise ValueError("expert pilot requires original expert source and layer 1")
    return args


def selection_identity(args):
    return dict(schema=SCHEMA, layers=list(args.layers), arms=list(args.arms),
                expert_count=args.expert_count, expert_layer=1 if args.expert_count else None)


def subset(records, budget):
    """Deterministic original-order prefix of complete, unmodified prompts."""
    selected, count = [], 0
    for row in records:
        selected.append(row)
        count += len(row["input_ids"])
        if count >= budget:
            return selected
    raise ValueError("corpus split is smaller than the requested pilot")


def stratified_experts(counts):
    """Eight evenly spaced experts per count quartile, ties broken by index."""
    if len(counts) != 896 or any(count < 0 for count in counts):
        raise ValueError("expected 896 nonnegative route counts")
    order = sorted(range(len(counts)), key=lambda index: (counts[index], index))
    return [(order[quartile * 224 + slot * 223 // 7], quartile + 1)
            for quartile in range(4) for slot in range(8)]


def payload_bytes(arm, shape):
    outputs, inputs = shape
    if min(shape) <= 0 or inputs % 128:
        raise ValueError("invalid pilot geometry")
    if arm == "fp8-block128":
        return outputs * inputs + math.ceil(outputs / 128) * math.ceil(inputs / 128) * 4
    if arm == "fp8-mx32":
        return outputs * inputs + outputs * math.ceil(inputs / 32)
    bits = 6 if arm in LEGACY_ARMS else int(arm.rsplit("-k", 1)[1])
    stored_outputs = math.ceil(outputs / 128) * 128
    return inputs * stored_outputs * bits // 8 + 2 * (inputs + stored_outputs) + 4


def fp8_quantize(weight, arm):
    """Native-coordinate E4M3 RTN, without activation quantization or rotation."""
    import torch
    if weight.ndim != 2 or not bool(torch.isfinite(weight).all()):
        raise ValueError("FP8 pilot requires finite matrix weights")
    outputs, inputs = weight.shape
    if arm == "fp8-block128":
        padded = torch.nn.functional.pad(weight.float(), (0, (-inputs) % 128, 0, (-outputs) % 128))
        groups = padded.reshape(-1, 128, padded.shape[1] // 128, 128).permute(0, 2, 1, 3)
        scales = groups.abs().amax(dim=(-2, -1)) / 448.
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        codes = (groups / scales[..., None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        codes = codes.permute(0, 2, 1, 3).reshape(padded.shape)[:outputs, :inputs]
        return dict(codes=codes.view(torch.uint8).cpu(), scales=scales.cpu())
    if arm != "fp8-mx32" or inputs % 32:
        raise ValueError("unknown or unaligned FP8 format")
    groups = weight.float().reshape(outputs, inputs // 32, 32)
    amax = groups.abs().amax(dim=-1)
    # OCP MX shared exponent: floor(log2(max_abs)) - emax(E4M3), with
    # saturation for the top of the binade. E8M0 byte 255 is reserved for NaN.
    exponent = (torch.floor(torch.log2(amax.clamp_min(2. ** -126))) - 8).clamp(-127, 127)
    exponent = torch.where(amax == 0, torch.zeros_like(exponent), exponent)
    scales = torch.exp2(exponent)
    codes = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return dict(codes=codes.reshape(weight.shape).view(torch.uint8).cpu(),
                scales=(exponent + 127).to(torch.uint8).cpu())


def fp8_reconstruct(packed, arm, device):
    import torch
    codes = packed["codes"].to(device).view(torch.float8_e4m3fn).float()
    scales = packed["scales"].to(device)
    if arm == "fp8-mx32":
        return (codes.reshape(codes.shape[0], -1, 32) *
                torch.exp2(scales.float() - 127)[..., None]).reshape(codes.shape)
    return codes * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:codes.shape[0], :codes.shape[1]]


def quantize_candidate(weight, raw, arm, codec, quantizer):
    import torch
    if arm.startswith("fp8-"):
        return fp8_quantize(weight, arm)
    if arm in ("mcg", "e4m3", "mcg-k6"):
        return codec.quantize_core_projection(weight, raw, codebook="e4m3" if arm == "e4m3" else "mcg")[0]
    codebook, bit_string = arm.split("-k")
    outputs, inputs = weight.shape
    working = weight.new_zeros((inputs, math.ceil(outputs / 128) * 128), dtype=torch.float32)
    working[:, :outputs].copy_(weight.T)
    options = dict(K=int(bit_string), seed=787, sigma_reg=.025,
                   devices=[weight.device], apply_out_scales=None, **{codebook: True})
    _, _, packed = quantizer.quantize_exl3(working,
        dict(H=raw["H"].clone(), count=raw["count"], finalized=False), options, return_weight_q=False)
    metrics = options["error_metrics"]
    if (metrics["quantizer_path"] != "hessian_ldlq" or metrics["hessian_metric_status"] != "ok"
            or metrics["hessian_numerical_contract"] != "signed-block-hadamard-congruence-fp64-v1"):
        raise ValueError("pilot requires calibrated FP64-congruence LDLQ")
    packed = {key: value.detach().cpu() for key, value in packed.items()}
    if codec.validate_core_tensors(packed, list(weight.shape))["bits"] != int(bit_string):
        raise ValueError("candidate bitrate differs")
    return packed


def reconstruct(packed, arm, shape, codec, device):
    import torch
    if arm.startswith("fp8-"):
        return fp8_reconstruct(packed, arm, device)
    if arm == "mcg-post-e4m3":
        packed = {("e4m3" if key == "mcg" else key): value for key, value in packed.items()}
    return codec.reconstruct_core_projection(packed, shape, device=device, dtype=torch.float32)


def validate_result(row):
    base = {"schema", "tensor", "arm", "nmse", "bf16_output_proxy_nmse",
            "calibration_tokens", "heldout_tokens", "weight_nmse"}
    extra = {"payload_bytes", "bpw", "tp2_bytes"}
    if set(row) not in (base, base | extra):
        raise ValueError("invalid pilot result fields")
    if row["schema"] != SCHEMA or row["arm"] not in LEGACY_ARMS + FORMAT_ARMS or not row["tensor"]:
        raise ValueError("invalid pilot result identity")
    for key in ("nmse", "bf16_output_proxy_nmse", "weight_nmse"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("nonfinite or negative pilot metric")
    for key in ("calibration_tokens", "heldout_tokens"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] <= 0:
            raise ValueError("invalid pilot observation count")
    if extra <= set(row) and any(not math.isfinite(row[key]) or row[key] <= 0 for key in extra):
        raise ValueError("invalid storage metric")
    return row


def markdown(rows):
    by_tensor = {}
    for row in rows:
        validate_result(row)
        if row["arm"] in by_tensor.setdefault(row["tensor"], {}):
            raise ValueError("duplicate pilot result")
        by_tensor[row["tensor"]][row["arm"]] = row
    arms = FORMAT_ARMS if any(row["arm"] == "mcg-k6" for row in rows) else LEGACY_ARMS
    lines = ["| Tensor | " + " | ".join(arm + " NMSE (ratio)" for arm in arms) + " |",
             "| --- | " + " | ".join("---:" for _ in arms) + " |"]
    for tensor, values in sorted(by_tensor.items()):
        if set(values) != set(arms):
            raise ValueError("incomplete pilot arm inventory")
        baseline = values[arms[0]]["nmse"]
        cells = [f"{values[arm]['nmse']:.8g} ({values[arm]['nmse']/baseline:.4f}x)" for arm in arms]
        lines.append(f"| {tensor} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = configuration(argv)
    from . import run
    from .recipe import reviewed_modules
    from .storage import StageStore
    from kimi_k3.uniform_k2 import run as expert_run
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from gptqmodel.utils import kimi_k3_core_format as codec
    from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as quantizer

    args.root.mkdir(parents=True, exist_ok=True)
    calibration = subset(run.load_token_records(args.tokens, "calibration"), args.calibration_tokens)
    heldout = subset(run.load_token_records(args.tokens, "heldout"), args.heldout_tokens)
    if {row["id"] for row in calibration} & {row["id"] for row in heldout}:
        raise ValueError("pilot calibration and held-out prompt IDs overlap")
    token_hash = lambda records: hashlib.sha256(run.canonical(records).encode()).hexdigest()
    identity = dict(**selection_identity(args),
        calibration_tokens=sum(len(row["input_ids"]) for row in calibration),
        heldout_tokens=sum(len(row["input_ids"]) for row in heldout),
        calibration_sha256=token_hash(calibration), heldout_sha256=token_hash(heldout),
        calibration="deterministic-original-prompt-subset-rebuild",
        replay="published-Core-K6-predecessors", head=False,
        source_index_sha256=run.sha256_file(args.source_snapshot / "model.safetensors.index.json"),
        replay_index_sha256=run.sha256_file(args.replay_snapshot / "model.safetensors.index.json"))
    if args.expert_count:
        identity["expert_source_index_sha256"] = run.sha256_file(args.expert_source_snapshot / "model.safetensors.index.json")
    run.atomic_json(args.root / "pilot-plan.json", identity)
    if args.preflight:
        for snapshot in (args.source_snapshot, args.replay_snapshot):
            run.header_source(snapshot)
        if args.expert_count:
            run.header_source(args.expert_source_snapshot)
        print(run.canonical(identity), flush=True)
        return
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86 * run.GIB / torch.cuda.get_device_properties(0).total_memory)
    source = codec.KimiK3EXL3Source(args.source_snapshot)
    replay = codec.KimiK3EXL3Source(args.replay_snapshot)
    expert_source = codec.KimiK3Source(args.expert_source_snapshot) if args.expert_count else None
    selection = reviewed_modules()
    candidates = args.root / "candidates"
    candidates.mkdir(exist_ok=True)
    rows, expert_rows = [], []

    class PublishedCheckpoints:
        def load(self, module):
            return None

    class PilotEngine(run.CoreEngine):
        def __init__(self, split, records):
            root = args.root / split
            root.mkdir(exist_ok=True)
            super().__init__(replay, records, root, root / "unused-export", identity,
                             selection, run.Progress(root, host_limit=50))
            self.split = split

        def input_sample(self, block, store, phase, name):
            _, _, state = next(self._read_batches(store, "a" if phase == "b" else phase))
            if phase == "a":
                value = state["hidden"]
            elif phase == "b":
                value = block.self_attn.q_a_layernorm(block.self_attn.q_a_proj(state["hidden"]))
            elif phase == "c":
                value = state["o_input"]
            elif phase == "d":
                value = state["hidden"]
            elif name == "routed-up":
                value = state["routed_input"]
            else:
                _, _, prior = next(self._read_batches(store, "d"))
                value = self.math.shared_down_input(block.block_sparse_moe.shared_experts, prior["hidden"])
            return value.reshape(-1, value.shape[-1])[:128].contiguous()

        def _quant_group(self, block, checkpoints, modules, store, phase, hessian):
            if block.layer_idx not in args.layers:
                return
            raw = self._raw(store, phase, hessian)
            sample = self.input_sample(block, store, phase, hessian) if self.split == "heldout" else None
            for module in modules:
                weight = source.tensor(module + ".weight", self.device)
                shape = tuple(weight.shape)
                if self.split == "calibration":
                    for arm in args.arms:
                        if arm == "mcg-post-e4m3":
                            continue
                        packed = quantize_candidate(weight, raw, arm, codec, quantizer)
                        actual = sum(value.numel() * value.element_size() for value in packed.values())
                        if actual != payload_bytes(arm, shape):
                            raise ValueError("serialized payload accounting differs")
                        run.commit_safetensors(candidates / (module + "." + arm + ".safetensors"), packed)
                        self.progress.emit("pilot_candidate", module=module, arm=arm)
                        del packed
                else:
                    denominator = float(((weight.float() @ raw["H"]) * weight.float()).sum(dtype=torch.float64))
                    reference = sample @ weight.T
                    for arm in args.arms:
                        candidate_arm = "mcg" if arm == "mcg-post-e4m3" else arm
                        packed = load_file(str(candidates / (module + "." + candidate_arm + ".safetensors")))
                        dense = reconstruct(packed, arm, shape, codec, self.device)
                        error = dense - weight.float()
                        numerator = float(((error @ raw["H"]) * error).sum(dtype=torch.float64))
                        proxy = sample @ dense.to(torch.bfloat16).T
                        size = payload_bytes(arm, shape)
                        result = validate_result(dict(schema=SCHEMA, tensor=module, arm=arm,
                            nmse=max(0., numerator) / denominator,
                            bf16_output_proxy_nmse=float((proxy.float() - reference.float()).square().sum(dtype=torch.float64) / reference.float().square().sum(dtype=torch.float64)),
                            weight_nmse=float(error.square().sum(dtype=torch.float64) / weight.float().square().sum(dtype=torch.float64)),
                            calibration_tokens=identity["calibration_tokens"], heldout_tokens=identity["heldout_tokens"],
                            payload_bytes=size, bpw=8 * size / weight.numel(), tp2_bytes=size / 2))
                        rows.append(result)
                        run.atomic_json(args.root / "results.json", dict(identity=identity, results=rows))
                        self.progress.emit("pilot_result", **result)
                        del dense, error, proxy, packed
                del weight
                run.compact_memory()
            del raw, sample
            run.compact_memory()

        def _phase_d(self, block, current, store):
            result = super()._phase_d(block, current, store)
            if args.expert_count and block.layer_idx == 1:
                # Reuse the uniform-K2 capture/routing/near-miss recipe unchanged.
                spool = expert_run.capture_layer(block, self.boundary, current, self.records,
                    self.root / "expert-capture", self.progress, identity)
                try:
                    self.experts(spool)
                finally:
                    spool.close()
            return result

        def experts(self, spool):
            sample_path = args.root / "expert-selection.json"
            if self.split == "calibration":
                chosen = stratified_experts(spool.counts)
                run.atomic_json(sample_path, dict(selection=chosen, route_counts=spool.counts,
                    policy="count-quartiles-tie-expert-id-eight-equally-spaced-including-extrema"))
            else:
                chosen = json.loads(sample_path.read_text())["selection"]
            for expert, quartile in chosen:
                prefix = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}"
                # All arms' down projections use the same SHIPPED K2 gate/up
                # activations, matching published predecessor replay.
                projections = {key: replay.decoded(prefix + "." + key + ".weight", self.device)
                               for key in ("w1", "w3")}
                input_h = None
                for projection in ("w1", "w3", "w2"):
                    name = prefix + "." + projection
                    weight = expert_source.decoded(name + ".weight", self.device)
                    if self.split == "calibration":
                        if projection == "w2":
                            raw, evidence = expert_run.capture_hessian(spool, expert, projections=projections)
                        else:
                            if input_h is None:
                                input_h, input_evidence = expert_run.capture_hessian(spool, expert)
                            raw, evidence = input_h, input_evidence
                        run.atomic_json(candidates / (name + ".hessian.json"), evidence)
                        for arm in EXPERT_ARMS:
                            packed = quantize_candidate(weight, raw, arm, codec, quantizer)
                            run.commit_safetensors(candidates / (name + "." + arm + ".safetensors"), packed)
                            self.progress.emit("expert_candidate", module=name, arm=arm, quartile=quartile, **evidence)
                            del packed
                        del raw
                    else:
                        # No recovery rows or identity on HELD-OUT scoring.
                        natural = spool.route_rows[expert]
                        H = torch.zeros((weight.shape[1], weight.shape[1]), device=self.device)
                        for start in range(0, len(natural), 2048):
                            values = spool.gather(natural[start:start + 2048])
                            if projection == "w2":
                                values = expert_run.situ(values @ projections["w1"].T,
                                                         values @ projections["w3"].T).to(torch.bfloat16)
                            values = values.float()
                            H.addmm_(values.T, values)
                        denominator = float(((weight.float() @ H) * weight.float()).sum(dtype=torch.float64))
                        for arm in EXPERT_ARMS:
                            packed = load_file(str(candidates / (name + "." + arm + ".safetensors")))
                            dense = reconstruct(packed, arm, tuple(weight.shape), codec, self.device)
                            error = dense - weight.float()
                            numerator = max(0., float(((error @ H) * error).sum(dtype=torch.float64)))
                            size = sum(value.numel() * value.element_size() for value in packed.values())
                            if size != payload_bytes(arm, tuple(weight.shape)):
                                raise ValueError("expert payload accounting differs")
                            row = dict(tensor=name, expert=expert, quartile=quartile, projection=projection,
                                arm=arm, numerator=numerator, denominator=denominator,
                                nmse=numerator / denominator if denominator > 0 else None,
                                heldout_rows=len(natural), payload_bytes=size, bpw=8 * size / weight.numel())
                            expert_rows.append(row)
                            run.atomic_json(args.root / "expert-results.json", dict(identity=identity, results=expert_rows))
                            self.progress.emit("expert_result", **row)
                            del packed, dense, error
                        del H
                    del weight
                    run.compact_memory()
                del input_h, projections
                run.compact_memory()

        @torch.inference_mode()
        def pilot_layer(self, layer, current):
            store = StageStore(self.root / f"layers/{layer:03d}/stages", identity)
            block = replay.load_block(layer, self.device, include_experts=False)
            self._execute_layer(block, current, store, PublishedCheckpoints(),
                                {name: None for name in ("a", "b", "c", "d", "e", "f")})
            del block
            run.compact_memory()
            manifest = store.load("f")
            batches = run.hardlink_frontier(self.boundary, layer, store.directory("f"), manifest)
            current = self.boundary.commit(layer, batches)
            store.clear()
            self.boundary.retire_old()
            return current

    with torch.inference_mode(), run.exclusive_run(args.root):
        for split, records in (("calibration", calibration), ("heldout", heldout)):
            engine = PilotEngine(split, records)
            current = engine.initial()
            for layer in range(max(args.layers) + 1):
                current = engine.pilot_layer(layer, current)
            import shutil
            shutil.rmtree(engine.root / "frontier")
        table = markdown(rows)
        (args.root / "pilot-table.md").write_text(table)
        # Storage projection covers all non-head core matrices, not just the
        # sampled layers. Head stays in shipped K6 and is explicitly separate.
        sizes = {arm: sum(payload_bytes(arm, (geometry["out_features"], geometry["in_features"]))
            for name, geometry in selection.items() if name != "language_model.lm_head") / 2 for arm in args.arms}
        run.atomic_json(args.root / "summary.json", dict(identity=identity,
            stop="pilot-only-no-full-quant", results=rows, expert_results=expert_rows,
            non_head_core_tp2_bytes=sizes,
            unchanged_head_k6_total_bytes=payload_bytes("mcg-k6", (163840, 7168))))
        print(table, flush=True)
        print("Stopping after pilot; no full quantization.", flush=True)
    source.close()
    replay.close()


if __name__ == "__main__":
    main()
