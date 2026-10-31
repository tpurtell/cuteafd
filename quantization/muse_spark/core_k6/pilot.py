#!/usr/bin/env python3
"""Bounded native-BF16 Muse format pilot with immutable, resumable receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "muse_spark.core_k6"

import torch
from safetensors.torch import load_file
from kimi_k3.uniform_k2.storage import atomic_json, canonical, commit_safetensors, sha256_file, verify_receipt
from kimi_k3.core_k6.pilot import subset
from muse_spark.prepare_corpus import load_token_records
from . import codec, numerical
from .source import MuseSource
from .recipe import BASE_REVISION, BASE_REPO, inventory, packed_specs, reviewed_core_modules

SCHEMA = "cuteafd-muse-spark-core-pilot-v1"
LAYERS = (0, 1, 61)
ARMS = (6, 5, 4)
CORE = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "self_attn.gate_proj", "mlp.pre_expert_proj", "mlp.post_expert_proj")
PREFIX = "model.language_model."
GIB = 1024 ** 3
LIMIT = 1e-3


def now():
    return datetime.now(timezone.utc).isoformat()


def compact():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite pilot numerical evidence")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)


@torch.inference_mode()
def weighted_nmse(weight, candidate, hessian, *, device="cuda:0", rows=1024):
    """trace(E H E^T)/trace(W H W^T), with FP64 final accumulation.

    This is the existing GLM pilot metric, tiled by output rows so the
    202,048-row head never needs three complete FP32 copies on the GPU.
    """
    if weight.shape != candidate.shape or hessian.shape != (weight.shape[1],) * 2:
        raise ValueError("incompatible projection metric geometry")
    H = hessian.to(device=device, dtype=torch.float32)
    if not torch.isfinite(H).all():
        raise ValueError("nonfinite heldout Hessian")
    numerator = denominator = 0.0
    for start in range(0, len(weight), rows):
        w = weight[start:start + rows].to(device=device, dtype=torch.float32)
        q = candidate[start:start + rows].to(device=device, dtype=torch.float32)
        if not torch.isfinite(w).all() or not torch.isfinite(q).all():
            raise ValueError("nonfinite projection metric operand")
        error = q - w
        denominator += float(((w @ H) * w).sum(dtype=torch.float64))
        numerator += float(((error @ H) * error).sum(dtype=torch.float64))
    if (not math.isfinite(numerator) or not math.isfinite(denominator)
            or denominator <= 0 or numerator < -1e-6 * denominator):
        raise ValueError("nonfinite, negative or zero-energy heldout metric")
    return max(0.0, numerator) / denominator


def payload_bytes(shape, bits):
    outputs, inputs = shape
    if min(shape) <= 0 or inputs % 128 or bits not in (4, 5, 6, 16):
        raise ValueError("invalid Muse pilot storage geometry")
    if bits == 16:
        return outputs * inputs * 2
    stored = (outputs + 127) // 128 * 128
    return inputs * stored * bits // 8 + 2 * (inputs + stored) + 4


def tp2_bytes(name, shape, bits):
    """Whole operands use side 1; o/post bridges split their input dimension."""
    outputs, inputs = shape
    if name == "lm_head":
        return [payload_bytes(shape, bits), 0]
    local = (outputs, inputs // 2) if name.endswith(("o_proj", "post_expert_proj")) else (outputs // 2, inputs)
    return [payload_bytes(local, bits)] * 2


def full_name(layer, relative):
    return f"{PREFIX}layers.{layer}.{relative}"


def stored_shapes():
    return {name: dict(in_features=g["in_features"], out_features=g["out_features"],
        stored_out_features=g["stored_out_features"], sharded_side=1,
        stored_tensors={key: [dtype, shape] for key, (dtype, shape) in packed_specs(name, g).items()})
        for name, g in reviewed_core_modules().items()}


def make_plan(source, tokens, native_reference_root=None):
    from transformers import AutoTokenizer
    from muse_spark.prepare_corpus import CHAT_KWARGS
    summary = inventory(source)
    if source.snapshot.name != BASE_REVISION:
        raise ValueError("pilot requires the pinned Muse source")
    records = {"calibration": subset(load_token_records(tokens, "calibration"), 4096),
               "heldout": subset(load_token_records(tokens, "heldout"), 1024)}
    tokenizer = AutoTokenizer.from_pretrained(source.snapshot, local_files_only=True)
    witness = tokenizer.apply_chat_template([dict(role="user", content="hello " * 2200)], **CHAT_KWARGS)
    if not 2048 < len(witness) <= 4096:
        raise ValueError("window witness must be one complete native prompt above 2048 tokens")
    paths = [Path(__file__), Path(codec.__file__), Path(numerical.__file__), Path(__file__).with_name("source.py"),
        Path(__file__).with_name("recipe.py"), REPO / "quantization/muse_spark/prepare_corpus.py",
        REPO / "quantization/kimi_k3/prepare_corpus.py", REPO / "quantization/kimi_k3/core_k6/pilot.py",
        REPO / "quantization/kimi_k3/uniform_k2/storage.py"]
    paths += [Path(__file__).with_name("requalify.py")]
    paths += [REPO / "third_party/gptqmodel/gptqmodel/utils" / name for name in
              ("exl3_hessian.py",)]
    paths += list((REPO / "third_party/gptqmodel/gptqmodel/exllamav3").rglob("*.py"))
    paths += [p for p in (REPO / "third_party/gptqmodel/gptqmodel_ext/exllamav3").rglob("*") if p.is_file()]
    code = {str(p.relative_to(REPO)): sha256_file(p) for p in sorted(paths) if "__pycache__" not in p.parts}
    identity = dict(schema=SCHEMA, source_repo=BASE_REPO, source_revision=BASE_REVISION,
        source_config_sha256=sha256_file(source.snapshot / "config.json"),
        source_index_sha256=sha256_file(source.snapshot / "model.safetensors.index.json"),
        token_manifest_sha256=sha256_file(tokens / "manifest.json"), code=code,
        native=numerical.source_identity(),
        gptqmodel_commit=subprocess.check_output(["git", "-C", str(REPO / "third_party/gptqmodel"),
            "rev-parse", "HEAD"], text=True).strip(), layers=list(LAYERS), arms=list(ARMS),
        records={split: [dict(id=r["id"], tokens=len(r["input_ids"]),
            ids_sha256=hashlib.sha256(canonical(r["input_ids"]).encode()).hexdigest()) for r in rows]
            for split, rows in records.items()}, witness=dict(input_ids=witness,
                tokens=len(witness), use="native-sliding-window-proof-only-excluded-from-all-Hessians"),
        recipe=dict(seed=787, sigma_reg=.025, codebook="mcg", core_bits=6,
            native_reference=True, attention_gate="keep-BF16", head_bits=6,
            expert_bits=4, absolute_k6_nmse_bound=LIMIT, hessian_accumulation=codec.ACCUMULATION_CONTRACT))
    if native_reference_root is not None:
        from .requalify import native_origin
        identity["native_reference_origin"] = native_origin(native_reference_root, identity, repo=REPO)
    return identity, records, summary


def save_tensor(path, tensors, identity):
    receipt = dict(identity=identity, filename=Path(path).name,
        **commit_safetensors(path, tensors))
    atomic_json(Path(path).with_suffix(".receipt.json"), receipt)
    return receipt


def read_tensor(path, identity):
    receipt = json.loads(Path(path).with_suffix(".receipt.json").read_text())
    if receipt.get("identity") != identity or receipt.get("filename") != Path(path).name:
        raise ValueError("pilot tensor producer differs")
    verify_receipt(path, receipt, checksum=True)
    return load_file(str(path), device="cpu")


def preserve_partial(path):
    """Incomplete attempts are never mistaken for commitments or destroyed."""
    path = Path(path)
    if path.exists():
        if path.is_symlink():
            raise ValueError("refuse pilot symlink recovery path")
        target = path.with_name(path.name + ".incomplete-" + str(time.time_ns()))
        os.rename(path, target)
    path.mkdir(parents=True)


class Capture:
    def __init__(self, block):
        self.raw, self.handles, self.expert_rows = {}, [], []
        for relative in CORE:
            linear = block.get_submodule(relative)
            self.raw[relative] = codec.DenseRawHessian(linear.in_features)
            self.handles.append(linear.register_forward_pre_hook(self.hook(relative)))
        self.handles.append(block.mlp.experts.register_forward_pre_hook(self.experts))

    def hook(self, name):
        def accumulate(_module, args):
            self.raw[name].add(args[0].detach())
        return accumulate

    def experts(self, _module, args):
        inputs, selected = args[0], args[1]
        if inputs.ndim != 2 or inputs.shape[1] != 4096 or selected.shape != (len(inputs), 8):
            raise ValueError("native expert capture geometry differs")
        if not torch.isfinite(inputs).all():
            raise ValueError("nonfinite native expert inputs")
        self.expert_rows.append((inputs.detach().cpu(), selected.detach().cpu()))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, directory, identity):
        directory.mkdir(parents=True)
        counts = {}
        for name, accumulator in self.raw.items():
            raw = accumulator.raw_dict()
            save_tensor(directory / (name + ".safetensors"), dict(H=raw["H"]), identity)
            counts[name] = raw["count"]
        save_tensor(directory / "experts.safetensors", dict(
            inputs=torch.cat([r[0] for r in self.expert_rows]),
            selected=torch.cat([r[1] for r in self.expert_rows])), identity)
        atomic_json(directory / "manifest.json", dict(identity=identity, counts=counts))


@torch.inference_mode()
def byte_proof(block, hidden, rotary):
    native = numerical.native_forward(block, dict(hidden=hidden), rotary)["hidden"]
    staged = numerical.staged_forward(block, dict(hidden=hidden), rotary)["hidden"]
    if (not torch.isfinite(native).all() or not torch.isfinite(staged).all()
            or native.dtype != staged.dtype or native.shape != staged.shape
            or not torch.equal(native.contiguous().view(torch.uint8), staged.contiguous().view(torch.uint8))):
        raise ValueError("Muse native/staged layer state differs bytewise")
    return dict(status="passed", byte_exact=True, tokens=hidden.shape[1], dtype=str(native.dtype),
        output_sha256=hashlib.sha256(native.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())


@torch.inference_mode()
def window_proof(block, hidden, rotary):
    context = numerical.prepare_context(block, hidden, rotary)
    mask = context["attention_mask"]
    if mask is None or not isinstance(mask, torch.Tensor):
        raise ValueError("long sliding-window proof needs an explicit native mask")
    visible = mask[0, 0, -1] if mask.dtype == torch.bool else mask[0, 0, -1].eq(0)
    expected = torch.arange(hidden.shape[1], device=hidden.device) >= hidden.shape[1] - 2048
    if not torch.equal(visible, expected):
        raise ValueError("native sliding-window visibility differs at the 2048-token cutoff")
    return dict(byte_proof(block, hidden, rotary), visible_tail_tokens=int(visible.sum()),
        first_visible=hidden.shape[1] - 2048, hessian_rows=0)


@torch.inference_mode()
def capture(source, records, root, identity):
    current_path = root / "capture-current.json"
    batches = [(split, row) for split in ("calibration", "heldout") for row in records[split]]
    if not current_path.exists():
        directory = root / "frontier-embedding"
        preserve_partial(directory)
        embed = numerical.embedding(source)
        for index, (_, row) in enumerate(batches):
            hidden = embed(torch.tensor([row["input_ids"]], device="cuda:0"))
            save_tensor(directory / f"{index:03d}.safetensors", dict(hidden=hidden), identity)
        witness = embed(torch.tensor([identity["witness"]["input_ids"]], device="cuda:0"))
        save_tensor(directory / "witness.safetensors", dict(hidden=witness), identity)
        del embed, hidden, witness
        atomic_json(current_path, dict(identity=identity, layer=-1, directory=directory.name, proofs={}))
        compact()
    current = json.loads(current_path.read_text())
    if current["identity"] != identity:
        raise ValueError("pilot capture identity differs")
    rotary = numerical.rotary(source)
    for layer in range(current["layer"] + 1, 62):
        started = time.monotonic()
        directory = root / f"frontier-{layer:03d}"
        completed = directory / "manifest.json"
        if completed.exists():
            candidate = json.loads(completed.read_text())
            if candidate.get("identity") != identity or candidate.get("layer") != layer:
                raise ValueError("completed pilot frontier identity differs")
            for index in range(len(batches)):
                read_tensor(directory / f"{index:03d}.safetensors", identity)
            current = candidate
            atomic_json(current_path, current)
            continue
        preserve_partial(directory)
        block = numerical.load_block(source, layer, include_experts=True)
        proofs = dict(current["proofs"])
        for split in records:
            accumulator = Capture(block) if layer in LAYERS else None
            try:
                for index, (batch_split, row) in enumerate(batches):
                    if batch_split != split:
                        continue
                    hidden = read_tensor(root / current["directory"] / f"{index:03d}.safetensors", identity)["hidden"].cuda()
                    result = numerical.native_forward(block, dict(hidden=hidden), rotary)["hidden"]
                    if not torch.isfinite(result).all():
                        raise ValueError("nonfinite native BF16 pilot layer")
                    save_tensor(directory / f"{index:03d}.safetensors", dict(hidden=result), identity)
                    # Close hooks while proving the same block so duplicate
                    # proof rows never enter either split's Hessian.
                if accumulator is not None:
                    accumulator.close()
                    accumulator.save(directory / split, identity)
            finally:
                if accumulator is not None:
                    accumulator.close()
            del accumulator
        if layer in LAYERS:
            hidden = read_tensor(root / current["directory"] / "000.safetensors", identity)["hidden"].cuda()
            proofs[str(layer)] = byte_proof(block, hidden, rotary)
        if layer == 0:
            hidden = read_tensor(root / "frontier-embedding" / "witness.safetensors", identity)["hidden"].cuda()
            proofs["sliding_window"] = window_proof(block, hidden, rotary)
        current = dict(identity=identity, layer=layer, directory=directory.name, proofs=proofs)
        atomic_json(completed, current)
        atomic_json(current_path, current)
        print(canonical(dict(event="pilot_native_layer", layer=layer, seconds=time.monotonic() - started)), flush=True)
        del block, hidden, result
        compact()
    return current


def hessian(root, layer, split, relative, identity):
    directory = root / f"frontier-{layer:03d}" / split
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("pilot Hessian producer differs")
    return dict(H=read_tensor(directory / (relative + ".safetensors"), identity)["H"],
        count=manifest["counts"][relative], finalized=False, identity_rows=0,
        accumulation_contract=codec.ACCUMULATION_CONTRACT)


def raw_identity(raw):
    H = raw["H"].detach().cpu().contiguous()
    return dict(count=raw["count"], finalized=raw["finalized"], dtype=str(H.dtype),
        shape=list(H.shape), sha256=hashlib.sha256(H.view(torch.uint8).numpy().tobytes()).hexdigest())


@torch.inference_mode()
def candidate(root, name, weight, calibration, heldout, bits, identity, *, basis):
    key = hashlib.sha256((name + ":" + str(bits) + ":" + basis).encode()).hexdigest()
    directory = root / "candidates" / key
    path = directory / "result.json"
    covariance = dict(calibration=raw_identity(calibration), heldout=raw_identity(heldout))
    if path.exists():
        result = json.loads(path.read_text())
        if (result.get("identity") != identity or result.get("name") != name or result.get("bits") != bits
                or result.get("covariance") != covariance):
            raise ValueError("committed pilot candidate producer differs")
        packed = read_tensor(directory / "packed.safetensors", identity)
        dense = read_tensor(directory / "reconstruction.safetensors", identity)["weight"]
        codec.validate_projection(packed, list(weight.shape))
        finite_tree(result)
        return result["observation"], dense
    preserve_partial(directory)
    packed, metrics, evidence = codec.quantize_projection(weight, calibration, bits=bits)
    save_tensor(directory / "packed.safetensors", packed, identity)
    # Reload the committed bytes: scores never use a transient quantizer result.
    restored = read_tensor(directory / "packed.safetensors", identity)
    dense = codec.reconstruct_projection(restored, list(weight.shape), device="cuda:0").cpu()
    save_tensor(directory / "reconstruction.safetensors", dict(weight=dense), identity)
    error = weighted_nmse(weight, dense, heldout["H"])
    size = sum(v.numel() * v.element_size() for v in restored.values())
    if size != payload_bytes(list(weight.shape), bits):
        raise ValueError("candidate serialized payload differs from storage model")
    observation = dict(name=name, bits=bits, shape=list(weight.shape), heldout_nmse=error,
        scale_inclusive_bytes=size, scale_inclusive_bpw=size * 8 / weight.numel(),
        tp2_bytes=tp2_bytes(name, list(weight.shape), bits), calibration_rows=calibration["count"],
        heldout_rows=heldout["count"], calibration_basis=basis, metrics=metrics,
        codec=evidence, candidate=str(directory.relative_to(root)))
    finite_tree(observation)
    atomic_json(path, dict(identity=identity, name=name, bits=bits, covariance=covariance,
                          observation=observation))
    print(canonical(dict(event="pilot_candidate", name=name, bits=bits, heldout_nmse=error)), flush=True)
    return observation, dense


def raw_from_rows(rows):
    h = codec.DenseRawHessian(rows.shape[-1])
    h.add(rows.cuda())
    result = h.raw_dict()
    result["H"] = result["H"].cpu()
    return result


@torch.inference_mode()
def score_expert(source, layer, root, identity):
    observations = {split: read_tensor(root / f"frontier-{layer:03d}" / split / "experts.safetensors", identity)
                    for split in ("calibration", "heldout")}
    counts = {split: torch.bincount(value["selected"].flatten(), minlength=256)
              for split, value in observations.items()}
    options = [e for e in range(256) if counts["calibration"][e] > 0 and counts["heldout"][e] > 0]
    if not options:
        raise ValueError("no naturally routed expert shared by calibration and heldout")
    expert = min(options, key=lambda e: (-int(counts["calibration"][e]), e))
    rows = {split: value["inputs"][(value["selected"] == expert).any(dim=1)]
            for split, value in observations.items()}
    raw = {split: raw_from_rows(x) for split, x in rows.items()}
    block = numerical.load_block(source, layer)
    results = []
    for bits in ((4, 5) if layer == 1 else (4,)):
        reconstructed = {}
        for part in ("gate_proj", "up_proj"):
            name = full_name(layer, f"mlp.experts.{expert}.{part}")
            weight = source.expert(layer, expert, part)
            result, dense = candidate(root, name, weight, raw["calibration"], raw["heldout"], bits,
                identity, basis="BF16-reference-natural-routed-expert-inputs")
            results.append(result)
            reconstructed[part] = dense.cuda()
        down_raw = {}
        for split, x in rows.items():
            inputs = numerical.expert_down_input(block, reconstructed["gate_proj"],
                reconstructed["up_proj"], x.cuda())
            down_raw[split] = raw_from_rows(inputs)
        name = full_name(layer, f"mlp.experts.{expert}.down_proj")
        weight = source.expert(layer, expert, "down_proj")
        result, dense = candidate(root, name, weight, down_raw["calibration"], down_raw["heldout"], bits,
            identity, basis=f"serialized-K{bits}-gate-up-reconstructed-natural-routed-inputs")
        results.append(result)
        del reconstructed, inputs, dense, down_raw, weight
        compact()
    del block
    return dict(layer=layer, expert=expert, calibration_rows=len(rows["calibration"]),
        heldout_rows=len(rows["heldout"]), observations=results,
        fractional_k35="unsupported: one standard per-module trellis has integer K; no mixed-width recipe introduced")


@torch.inference_mode()
def score(source, records, root, identity, current):
    results = []
    for layer in LAYERS:
        for relative in CORE:
            name = full_name(layer, relative)
            weight = source.tensor(name + ".weight")
            raw = {split: hessian(root, layer, split, relative, identity) for split in records}
            for bits in ARMS:
                observation, dense = candidate(root, name, weight, raw["calibration"], raw["heldout"], bits,
                    identity, basis="BF16-source-predecessors-natural-core-inputs")
                results.append(observation)
                del dense
                compact()
            results.append(dict(name=name, bits=16, shape=list(weight.shape), heldout_nmse=0.,
                scale_inclusive_bytes=weight.numel() * 2, scale_inclusive_bpw=16.,
                tp2_bytes=tp2_bytes(name, list(weight.shape), 16), reference=True))
            del raw, weight
            compact()
    experts = [score_expert(source, layer, root, identity) for layer in LAYERS]
    head = {}
    batches = [(split, row) for split in ("calibration", "heldout") for row in records[split]]
    for split in records:
        path = root / ("head-hessian-" + split + ".safetensors")
        count = sum(len(row["input_ids"]) for row in records[split])
        if path.with_suffix(".receipt.json").exists():
            H = read_tensor(path, identity)["H"]
        else:
            accumulator = codec.DenseRawHessian(8192)
            for index, (batch_split, _) in enumerate(batches):
                if split == batch_split:
                    hidden = read_tensor(root / current["directory"] / f"{index:03d}.safetensors", identity)["hidden"].cuda()
                    accumulator.add(numerical.final_norm(source, hidden))
            raw = accumulator.raw_dict()
            if raw["count"] != count:
                raise ValueError("native head Hessian token count differs")
            H = raw["H"].cpu()
            if path.exists():
                os.rename(path, path.with_name(path.name + ".incomplete-" + str(time.time_ns())))
            save_tensor(path, dict(H=H), identity)
            del accumulator, raw
        head[split] = dict(H=H, count=count, finalized=False, identity_rows=0,
            accumulation_contract=codec.ACCUMULATION_CONTRACT)
        compact()
    weight = source.tensor("lm_head.weight")
    observation, dense = candidate(root, "lm_head", weight, head["calibration"], head["heldout"], 6,
        identity, basis="BF16-source-full62-layers-final-native-norm")
    results.append(observation)
    del dense, weight, head
    compact()
    return dict(schema=SCHEMA, status="passed", identity=identity,
        native_proofs=current["proofs"], observations=results, experts=experts,
        stored_shapes=stored_shapes(), attention_gate_decision="keep-BF16",
        comparison="BF16 source predecessors; no candidate changes the reference capture",
        expert_down_contract="both splits use reconstructed serialized gate/up before down capture",
        completed=now())


def validate_report(value):
    report = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    finite_tree(report)
    if report.get("schema") != SCHEMA or report.get("status") != "passed":
        raise ValueError("pilot report failed or schema differs")
    identity = report.get("identity", {})
    if (identity.get("schema") != SCHEMA or identity.get("source_repo") != BASE_REPO
            or identity.get("source_revision") != BASE_REVISION):
        raise ValueError("pilot source/numerical identity absent")
    if report.get("identity", {}).get("recipe", {}).get("hessian_accumulation") != codec.ACCUMULATION_CONTRACT:
        raise ValueError("pilot must qualify the bounded K128 covariance producer")
    proofs = report.get("native_proofs", {})
    for key in (*map(str, LAYERS), "sliding_window"):
        if proofs.get(key, {}).get("status") != "passed" or proofs[key].get("byte_exact") is not True:
            raise ValueError("native/staged byte proof absent")
    witness = proofs["sliding_window"]
    if witness.get("tokens", 0) <= 2048 or witness.get("visible_tail_tokens") != 2048 or witness.get("hessian_rows") != 0:
        raise ValueError("separate long sliding-window proof absent")
    required = {full_name(layer, relative) for layer in LAYERS for relative in CORE}
    expected = {(name, bits) for name in required for bits in (*ARMS, 16)} | {("lm_head", 6)}
    observations = report.get("observations", [])
    if len(observations) != len(expected) or {(r["name"], r["bits"]) for r in observations} != expected:
        raise ValueError("pilot comparison inventory differs")
    for row in observations:
        error = row.get("heldout_nmse")
        if not isinstance(error, (float, int)) or error < 0:
            raise ValueError("invalid heldout error")
        if row["bits"] == 6 and not row["name"].endswith("self_attn.gate_proj") and error > LIMIT:
            raise ValueError("absolute K6 activation-weighted NMSE exceeds 1e-3")
        shape = row.get("shape")
        expected_shape = ([202048, 8192] if row["name"] == "lm_head" else
            [8192, 4096] if row["name"].endswith("post_expert_proj") else
            [4096, 8192] if row["name"].endswith("pre_expert_proj") else
            [1024, 8192] if row["name"].endswith(("k_proj", "v_proj")) else [8192, 8192])
        if (shape != expected_shape or row.get("scale_inclusive_bytes") != payload_bytes(shape, row["bits"])
                or row.get("tp2_bytes") != tp2_bytes(row["name"], shape, row["bits"])):
            raise ValueError("pilot observed storage geometry/payload differs")
        if row["bits"] != 16:
            evidence = row.get("codec", {})
            if (evidence.get("bits") != row["bits"] or evidence.get("native_shape") != shape
                    or evidence.get("stored_shape") != [(shape[0] + 127) // 128 * 128, shape[1]]
                    or row.get("calibration_rows", 0) <= 0 or row.get("heldout_rows", 0) <= 0
                    or row.get("metrics", {}).get("quantizer_path") != "hessian_ldlq"):
                raise ValueError("pilot serialized/calibrated candidate evidence absent")
    if report.get("attention_gate_decision") != "keep-BF16" or report.get("stored_shapes") != stored_shapes():
        raise ValueError("pilot recipe/stored geometry differs")
    experts = report.get("experts", [])
    if len(experts) != len(LAYERS) or {row["layer"] for row in experts} != set(LAYERS):
        raise ValueError("natural expert pilot coverage differs")
    for item in experts:
        expected_bits = {4, 5} if item["layer"] == 1 else {4}
        if (item.get("calibration_rows", 0) <= 0 or item.get("heldout_rows", 0) <= 0
                or {(r["name"].rsplit(".", 1)[-1], r["bits"]) for r in item["observations"]}
                   != {(part, bits) for part in ("gate_proj", "up_proj", "down_proj") for bits in expected_bits}):
            raise ValueError("expert arm/observation inventory differs")
        for row in item["observations"]:
            if not 0 <= row.get("heldout_nmse", -1) <= 1e-2 or (row["name"].endswith("down_proj")
                    and not row.get("calibration_basis", "").startswith("serialized-")):
                raise ValueError("expert error is catastrophic or down calibration omitted serialized predecessors")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot", "tokens", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-reference-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    source = MuseSource(args.source_snapshot)
    identity, records, summary = make_plan(source, args.tokens, args.native_reference_root)
    path = args.root / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("pilot immutable numerical producer/input identity differs; assess cached work before a new run")
    if args.preflight:
        print(canonical(dict(identity=identity, inventory=summary, required_free_bytes=32 * GIB)), flush=True)
        return
    if path.exists() and not args.resume:
        raise ValueError("existing pilot requires explicit --resume")
    if args.resume and not path.exists():
        raise ValueError("pilot resume requires committed identity")
    args.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.root).free < 32 * GIB:
        raise OSError("insufficient room for bounded pilot receipts/frontiers/candidates")
    from kimi_k3.core_k6.run import exclusive_run
    with exclusive_run(args.root), torch.inference_mode():
        if not path.exists():
            atomic_json(path, identity)
        if args.output.exists():
            report = validate_report(args.output)
            if report.get("identity") != identity:
                raise ValueError("completed pilot producer differs")
            print(canonical(dict(event="pilot_complete", output=str(args.output))))
            return
        torch.set_num_threads(8)
        numerical.configure_math()
        torch.cuda.set_per_process_memory_fraction(86 * GIB / torch.cuda.get_device_properties(0).total_memory)
        try:
            if args.native_reference_root is None:
                current = capture(source, records, args.root, identity)
            else:
                from .requalify import capture_from_native
                current = capture_from_native(source, records, args.root, identity, args.native_reference_root)
            report = score(source, records, args.root, identity, current)
            # Preserve all numerical evidence even if the decision fails.
            atomic_json(args.root / "observations.json", report)
            validate_report(report)
            atomic_json(args.output, report)
            print(canonical(dict(event="pilot_complete", output=str(args.output))), flush=True)
        except BaseException as error:
            atomic_json(args.root / ("failure-" + str(time.time_ns()) + ".json"),
                dict(identity=identity, time=now(), error_type=type(error).__name__, message=str(error),
                    recovery="preserve all commitments; numerical failures require cause/cache assessment; OOM may resume unchanged"))
            raise


if __name__ == "__main__":
    main()
