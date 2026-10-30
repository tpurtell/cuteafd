#!/usr/bin/env python3
"""Fixed short Step-5 format pilot; never exports or publishes a model.

Replay eight BF16 target layers. Late-expert format scoring deliberately uses
layer-7 FFN inputs, not an asserted layer-88 causal stream. Production must
capture actual layer-88 observations. No format decision changes replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("STEP5_GPTQMODEL_SOURCE", str(REPO / "third_party/gptqmodel")))
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "step5.core_k6"

from glm5.core_k6.pilot import weighted_nmse
from kimi_k3.core_k6.pilot import fp8_quantize, fp8_reconstruct, payload_bytes, subset
from kimi_k3.uniform_k2.storage import (
    RollingBoundary, atomic_json, canonical, commit_safetensors, sha256_file, verify_receipt,
)

ARMS = ("mcg-k6", "mcg-k5", "mcg-k4", "fp8-block128", "bf16")
LAYERS = (1, 4, 7)
PREFIX = "model."
SCHEMA = "cuteafd-step5-short-pilot-v1"
CHECKS = ("sliding_boundary_byte_exact", "sparse_boundary_byte_exact",
          "serialized_projection_replay", "mtp_three_stage_alignment_byte_exact",
          "mtp_serialized_k4_byte_exact")
LATE_LAYER = 88
LATE_EXPERT = 0
WITNESS_NAME = "model.layers.7.share_expert.gate_proj"
GIB = 1024**3


def modules(layer):
    prefix = f"model.layers.{layer}."
    if layer == 1:
        suffixes = ["mlp." + part for part in ("gate_proj", "up_proj", "down_proj")]
        suffixes += ["self_attn.g_proj"]
    elif layer == 4:
        suffixes = ["self_attn." + part for part in ("q_proj", "k_proj", "v_proj", "o_proj")]
        suffixes += ["share_expert.gate_proj", "share_expert.down_proj"]
    elif layer == 7:
        suffixes = ["self_attn." + part for part in ("q_proj", "k_proj", "v_proj", "o_proj",
            "sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_w", "sparse_indexer_z")]
    elif layer == LATE_LAYER:
        suffixes = [f"moe.experts.{LATE_EXPERT}." + part for part in ("gate_proj", "up_proj", "down_proj")]
    else:
        raise ValueError("layer is outside the fixed shortened pilot")
    return [prefix + name for name in suffixes]


def select_records(calibration, heldout, calibration_tokens=2048, heldout_tokens=768):
    """Whole prompt prefix plus one actual window-crossing calibration prompt."""
    cal, held = subset(calibration, calibration_tokens), subset(heldout, heldout_tokens)
    # Prefer a real sparse-pruning witness if one exists. The public panel is
    # deliberately kept unmodified; its Step-5 prompts currently stop at1050
    # tokens, so it exercises native CSA/indexing but not global top-k pruning.
    witness = next((r for r in calibration if len(r["input_ids"]) > 4104), None)
    if witness is None:
        witness = next((r for r in calibration if len(r["input_ids"]) > 512), None)
    if witness is None:
        raise ValueError("calibration contains no complete window-crossing prompt (>512 tokens)")
    if witness["id"] not in {r["id"] for r in cal}:
        cal.append(witness)
    if {r["id"] for r in cal} & {r["id"] for r in held}:
        raise ValueError("pilot calibration and heldout overlap")
    return cal, held, witness


def tp2_projection_bytes(name, shape, arm):
    """Scale-inclusive projection payload, not a serving allocation estimate."""
    n, k = shape
    def size(dims):
        return math.prod(dims) * 2 if arm == "bf16" else payload_bytes(arm, dims)
    if "sparse_indexer_" in name:
        return [size(shape), size(shape)]
    local = (n, k // 2) if name.endswith(("o_proj", "down_proj")) else (n // 2, k)
    if n % 2 or k % 2:
        raise ValueError("pilot TP2 projection is not divisible")
    return [size(local), size(local)]


def decision_policy():
    return dict(schema="cuteafd-step5-format-admission-v1",
        metric="heldout-activation-weighted-projection-NMSE", maximum_k6_nmse=1e-3,
        fp8_control="informational", native_checks="all-required", full_model_quality_claim=False,
        late_expert="source-layer88-weight-layer7-FFN-input-format-witness-not-causal-calibration")


def exact_state(actual, expected):
    import torch
    if set(actual) != set(expected) or any(not torch.equal(actual[k], expected[k]) for k in actual):
        raise ValueError("native/staged Step-5 boundary differs")


class SparseProbe:
    """Observe actual returned sparse metadata without modifying selection."""
    def __init__(self, source, tokens):
        self.kernel, self.tokens = source.kernel, tokens
        self.original = None
        self.counts, self.positions, self.topk, self.region_sizes = [], [], set(), set()
        self.packed_hash = hashlib.sha256()

    def __enter__(self):
        self.original = self.kernel.prefill_sparse_meta
        def observe(logits, positions, block_table, request_indices=None, **kwargs):
            packed, counts = self.original(logits, positions, block_table, request_indices, **kwargs)
            valid = positions < self.tokens
            self.positions.extend(positions[valid].cpu().tolist())
            self.counts.extend(counts[valid].cpu().tolist())
            self.packed_hash.update(packed[valid].contiguous().cpu().numpy().tobytes())
            self.topk.add(kwargs["topk"])
            self.region_sizes.add(kwargs["region_size"])
            return packed, counts
        self.kernel.prefill_sparse_meta = observe
        return self

    def __exit__(self, *args):
        self.kernel.prefill_sparse_meta = self.original

    def summary(self):
        if not self.counts or self.topk != {512} or self.region_sizes != {8}:
            raise ValueError("Step-5 sparse proof did not observe reviewed native metadata")
        available = [p // 8 + 1 for p in self.positions]
        pruned = sum(count < total for count,total in zip(self.counts, available))
        return dict(tokens=self.tokens, query_rows=len(self.counts), topk=512, region_size=8,
            maximum_selected_regions=max(self.counts), maximum_available_regions=max(available),
            pruned_query_rows=pruned, sparse_pruning_exercised=pruned > 0,
            metadata_sha256=self.packed_hash.hexdigest(),
            region_counts_sha256=hashlib.sha256(canonical(self.counts).encode()).hexdigest(),
            query_positions_sha256=hashlib.sha256(canonical(self.positions).encode()).hexdigest())


def native_staged_boundary(source, numerical, block, state, capture=None):
    from contextlib import nullcontext
    sparse = block.self_attn.is_sparse
    with SparseProbe(source, len(state["hidden"])) if sparse else nullcontext() as native_probe:
        with capture if capture else nullcontext():
            native = numerical.native_forward(block, state)
    with SparseProbe(source, len(state["hidden"])) if sparse else nullcontext() as staged_probe:
        staged = numerical.staged_forward(block, state)
    exact_state(native, staged)
    native_meta = native_probe.summary() if sparse else None
    if sparse and native_meta != staged_probe.summary():
        raise ValueError("Step-5 native/staged sparse metadata differs")
    return native, native_meta


def compact():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def progress(root, event, **fields):
    import psutil
    import torch
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    row = dict(event=event, time=datetime.now(timezone.utc).isoformat(),
        rss_gib=psutil.Process().memory_info().rss/GIB,
        cuda_gib=torch.cuda.memory_allocated()/GIB, gpu_peak_gib=torch.cuda.max_memory_allocated()/GIB,
        free_gib=shutil.disk_usage(root).free/GIB, **fields)
    with (root / "events.jsonl").open("a") as f:
        f.write(canonical(row) + "\n")
    atomic_json(root / "status.json", row)
    print(canonical(row), flush=True)
    if row["rss_gib"] > 78 or row["cuda_gib"] > 86 or row["free_gib"] < 64:
        raise MemoryError("Step-5 pilot exceeded host/GPU/disk admission")


class Capture:
    """Raw actual X.T@X, with a separately labelled late-expert input witness."""
    def __init__(self, numerical, block, layer):
        self.math, self.block, self.layer = numerical, block, layer
        self.raw, self.witness, self.context = {}, [], None

    def observe(self, name, value):
        import torch
        if name == WITNESS_NAME:
            self.witness.append(value.detach().to("cpu", copy=True))
        if name not in modules(self.layer):
            return
        x = value.detach().reshape(-1, value.shape[-1]).float()
        if name not in self.raw:
            self.raw[name] = dict(H=torch.zeros((x.shape[1],)*2, device=x.device), count=0)
        self.raw[name]["H"].addmm_(x.T, x)
        self.raw[name]["count"] += len(x)

    def __enter__(self):
        names = modules(self.layer) + ([WITNESS_NAME] if self.layer == 7 else [])
        self.context = self.math.capture_inputs(self.block, names, self.observe)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)


def _device(state):
    return {key: value.to("cuda:0") for key, value in state.items()}


def _save_batch(directory, index, state):
    path = Path(directory) / f"batch-{index:05d}.safetensors"
    return dict(filename=path.name, **commit_safetensors(path, state))


def _save_hessians(root, identity, layer, raw, expected_count):
    import torch
    if set(raw) != set(modules(layer)):
        raise ValueError("pilot captured projection inventory differs")
    receipts = {}
    for name, entry in raw.items():
        if entry["count"] != expected_count:
            raise ValueError("pilot Hessian must contain only actual observed rows")
        receipts[name] = commit_safetensors(root / "hessians" / (name + ".safetensors"),
            dict(H=entry["H"], count=torch.tensor(entry["count"])))
    atomic_json(root / "hessians" / f"layer-{layer}.json", dict(identity=identity, files=receipts))


def capture_split(source, numerical, records, root, identity):
    import torch
    from contextlib import nullcontext
    from safetensors.torch import load_file
    root.mkdir(parents=True, exist_ok=True)
    boundary = RollingBoundary(root / "frontier", identity)
    current = boundary.current(checksum=True)
    if current is None:
        partial = boundary.directory(-1)
        if partial.is_symlink():
            raise ValueError("refuse pilot frontier symlink")
        if partial.exists():
            shutil.rmtree(partial)
        embedding = source.tensor("model.embed_tokens.weight", "cuda:0")
        entries = []
        for index, record in enumerate(records):
            ids = torch.tensor(record["input_ids"], device="cuda:0")
            entries.append(_save_batch(partial, index, dict(hidden=embedding[ids].float(),
                position_ids=torch.arange(len(ids), device="cuda:0"))))
        current = boundary.commit(-1, entries)
        del embedding, ids
        source.close()
        compact()
    for layer in range(current["layer"] + 1, max(LAYERS) + 1):
        partial = boundary.directory(layer)
        if partial.is_symlink():
            raise ValueError("refuse pilot successor symlink")
        if partial.exists():
            shutil.rmtree(partial)
        block = source.load_block(layer, "cuda:0", include_experts=True)
        capture = Capture(numerical, block, layer) if layer in LAYERS else None
        entries, proofs = [], []
        try:
            for index, entry in enumerate(current["manifest"]["batches"]):
                state = _device(load_file(str(boundary.root / current["directory"] / entry["filename"])))
                if index == 0 or records[index]["id"] == identity["long_context_witness"]:
                    successor, metadata = native_staged_boundary(source, numerical, block, state, capture)
                    if layer in LAYERS:
                        path = root / "proof-inputs" / f"layer-{layer:03d}-{index:05d}.safetensors"
                        proofs.append(dict(layer=layer, row_id=records[index]["id"],
                            tokens=len(records[index]["input_ids"]), file=str(path.relative_to(root)),
                            receipt=commit_safetensors(path, state)))
                    progress(root, "native_boundary_exact", layer=layer, row_id=records[index]["id"],
                        tokens=len(records[index]["input_ids"]), attention="sparse" if layer in (3, 7) else "sliding",
                        sparse_metadata=metadata)
                    if (metadata is not None and len(records[index]["input_ids"]) > 4104
                            and records[index]["id"] == identity["long_context_witness"]
                            and not metadata["sparse_pruning_exercised"]):
                        raise ValueError("natural Step-5 long witness did not exercise actual sparse pruning")
                else:
                    with capture if capture else nullcontext():
                        successor = numerical.native_forward(block, state)
                entries.append(_save_batch(partial, index, successor))
                del state, successor
            if capture:
                _save_hessians(root, identity, layer, capture.raw, sum(len(r["input_ids"]) for r in records))
                atomic_json(root / "proof-inputs" / f"layer-{layer}.json", dict(identity=identity, entries=proofs))
                if layer == 7:
                    late_witness_hessians(source, numerical, capture.witness, root, identity)
            current = boundary.commit(layer, entries)
        finally:
            del capture, block
            source.close()
            compact()
        progress(root, "layer_committed", layer=layer)
    return current


def _weight(source, name):
    if ".moe.experts." in name:
        return source.expert_tensor(LATE_LAYER, LATE_EXPERT, name.rsplit(".", 1)[-1], "cuda:0")
    return source.tensor(name + ".weight", "cuda:0")


def late_witness_hessians(source, numerical, witness, root, identity):
    """Same real layer-7 FFN input for late weights, explicitly not a layer-88 replay."""
    import torch
    names = modules(LATE_LAYER)
    gate, up = (_weight(source, name) for name in names[:2])
    raw = {}
    for value in witness:
        x = value.to("cuda:0").reshape(-1, value.shape[-1])
        # Invoke the checkpoint's own clamped activation, including its BF16
        # boundary; do not substitute an unclamped SwiGLU in this format witness.
        limits = source.config.get("swiglu_limits")
        limit = limits[LATE_LAYER] if limits else 7.0
        activated = source.release.clamped_swiglu(torch.nn.functional.linear(x, gate),
            torch.nn.functional.linear(x, up), limit)
        for name, inputs in ((names[0], x), (names[1], x), (names[2], activated)):
            inputs = inputs.float()
            if name not in raw:
                raw[name] = dict(H=torch.zeros((inputs.shape[-1],)*2, device=inputs.device), count=0)
            raw[name]["H"].addmm_(inputs.T, inputs)
            raw[name]["count"] += len(inputs)
    count = sum(len(value.reshape(-1, value.shape[-1])) for value in witness)
    _save_hessians(root, identity, LATE_LAYER, raw, count)
    progress(root, "late_expert_witness", source_layer=LATE_LAYER, input_layer=7, expert=LATE_EXPERT,
        tokens=count, actual_layer88_calibration=False)


def load_hessian(root, split, layer, name, identity):
    from safetensors.torch import load_file
    directory = root / split / "hessians"
    manifest = json.loads((directory / f"layer-{layer}.json").read_text())
    if manifest["identity"] != identity or set(manifest["files"]) != set(modules(layer)):
        raise ValueError("pilot Hessian identity/inventory differs")
    path = directory / (name + ".safetensors")
    verify_receipt(path, manifest["files"][name], checksum=True)
    raw = load_file(str(path), device="cuda:0")
    if int(raw["count"]) != identity[split + "_tokens"]:
        raise ValueError("pilot observation count differs")
    return dict(H=raw["H"], count=int(raw["count"]), finalized=False)


def score(source, numerical, root, identity):
    from safetensors.torch import load_file
    rows = []
    for layer in (*LAYERS, LATE_LAYER):
        for name in modules(layer):
            raw = load_hessian(root, "calibration", layer, name, identity)
            weight = _weight(source, name)
            shape = list(weight.shape)
            source.close()
            for arm in ARMS:
                if arm == "bf16":
                    dense, size = weight.float(), weight.numel() * weight.element_size()
                else:
                    path = root / "candidates" / (name + "." + arm + ".safetensors")
                    receipt = path.with_suffix(".json")
                    if receipt.exists():
                        prior = json.loads(receipt.read_text())
                        if prior["identity"] != identity:
                            raise ValueError("pilot candidate identity differs")
                        verify_receipt(path, prior["receipt"], checksum=True)
                        packed = load_file(str(path))
                    else:
                        path.unlink(missing_ok=True)
                        packed = (numerical.quantize_projection(weight, raw, bits=int(arm[-1]))[0]
                            if arm.startswith("mcg-k") else fp8_quantize(weight, arm))
                        rec = commit_safetensors(path, packed)
                        atomic_json(receipt, dict(identity=identity, receipt=rec))
                    size = sum(t.numel() * t.element_size() for t in packed.values())
                    if size != payload_bytes(arm, shape):
                        raise ValueError("pilot candidate byte accounting differs")
                    dense = (numerical.reconstruct_projection(packed, shape, device="cuda:0", dtype=weight.dtype).float()
                        if arm.startswith("mcg-k") else fp8_reconstruct(packed, arm, "cuda:0").to(weight.dtype).float())
                    del packed
                heldout = load_hessian(root, "heldout", layer, name, identity)
                tp2 = tp2_projection_bytes(name, shape, arm)
                if arm == "bf16" and weight.element_size() == 4:
                    tp2 = [2 * value for value in tp2]
                row = dict(tensor=name, arm=arm, source_dtype=str(weight.dtype),
                    nmse=weighted_nmse(weight, dense, heldout["H"]), payload_bytes=size,
                    bpw=size * 8 / weight.numel(), tp2_bytes=tp2,
                    calibration_tokens=identity["calibration_tokens"], heldout_tokens=identity["heldout_tokens"],
                    causal_input_layer=7 if layer == LATE_LAYER else layer,
                    late_expert_format_witness=layer == LATE_LAYER)
                rows.append(row)
                atomic_json(root / "results.json", dict(identity=identity, results=rows))
                progress(root, "pilot_result", **row)
                del dense, heldout
            del raw, weight
            compact()
    return rows


def qualify_serialized_replay(source, numerical, root, identity):
    import torch
    from safetensors.torch import load_file
    for layer in LAYERS:
        block = source.load_block(layer, "cuda:0", include_experts=True)
        for name in modules(layer):
            path = root / "candidates" / (name + ".mcg-k6.safetensors")
            receipt = json.loads(path.with_suffix(".json").read_text())
            if receipt["identity"] != identity:
                raise ValueError("serialized replay identity differs")
            verify_receipt(path, receipt["receipt"], checksum=True)
            packed = load_file(str(path))
            module = numerical.module(block, name.removeprefix(f"model.layers.{layer}."))
            module.weight.copy_(numerical.reconstruct_projection(packed, list(module.weight.shape),
                device="cuda:0", dtype=module.weight.dtype))
            del packed
        numerical.invalidate_attention(block)
        manifest = json.loads((root / "calibration/proof-inputs" / f"layer-{layer}.json").read_text())
        if manifest["identity"] != identity or not manifest["entries"]:
            raise ValueError("serialized replay proof inventory differs")
        for entry in manifest["entries"]:
            path = root / "calibration" / entry["file"]
            verify_receipt(path, entry["receipt"], checksum=True)
            state = _device(load_file(str(path)))
            native, metadata = native_staged_boundary(source, numerical, block, state)
            if not bool(torch.isfinite(native["hidden"]).all()):
                raise ValueError("serialized replay is nonfinite")
            progress(root, "serialized_boundary_exact", layer=layer, row_id=entry["row_id"], tokens=entry["tokens"], sparse_metadata=metadata)
            if (metadata is not None and entry["tokens"] > 4104 and entry["row_id"] == identity["long_context_witness"]
                    and not metadata["sparse_pruning_exercised"]):
                raise ValueError("serialized Step-5 long witness did not exercise actual sparse pruning")
            del native, state
        del block
        source.close()
        compact()
    return {name: True for name in CHECKS[:3]}


def qualify_mtp(source, numerical, root, identity, row):
    """Three-stage teacher wiring on bounded prefix hidden; not final-target quality.

    The independent reference supplies token/position slicing and norm-concat.
    Decoder checks retain native GEMM/reduction geometry. The reference's F32
    accuracy-oracle GEMMs are deliberately not called a byte-exact native path.
    """
    import importlib.util
    import torch
    from safetensors.torch import load_file
    spec = importlib.util.spec_from_file_location("step5_pilot_mtp_oracle",
        REPO / "python/reference/families/step5/mtp.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    boundary = RollingBoundary(root / "calibration/frontier", identity)
    current = boundary.current(checksum=True)
    if current is None or current["layer"] != max(LAYERS):
        raise ValueError("Step-5 bounded target-prefix MTP witness is absent")
    entry = current["manifest"]["batches"][0]
    state = _device(load_file(str(boundary.root / current["directory"] / entry["filename"])))
    tokens = torch.tensor(row["input_ids"], device="cuda:0", dtype=torch.int64)
    hidden, positions = state["hidden"], state["position_ids"]
    if len(hidden) != len(tokens):
        raise ValueError("MTP witness row does not match the prefix frontier")
    embedding = source.tensor("model.embed_tokens.weight", "cuda:0")
    source.close()
    stages = []
    for stage in range(3):
        block = numerical.load_mtp(source, stage, "cuda:0")
        aligned = numerical.aligned_inputs(hidden, tokens, embedding, position_ids=positions)
        next_tokens, previous, next_positions = oracle.teacher_inputs(tokens, hidden, positions)
        if (not torch.equal(aligned["input_ids"], next_tokens)
                or not torch.equal(aligned["target_hidden"], previous)
                or not torch.equal(aligned["position_ids"], next_positions)
                or not torch.equal(aligned["inputs_embeds"], embedding[next_tokens])):
            raise ValueError("Step-5 MTP shifted token/hidden/position alignment differs")
        envelope = numerical.mtp_envelope(block, aligned["inputs_embeds"], previous)
        expected = oracle.norm_concat(embedding[next_tokens], previous,
            {"enorm.weight": block.enorm.weight, "hnorm.weight": block.hnorm.weight},
            source.config["rms_norm_eps"])
        if not torch.equal(envelope.float(), expected):
            raise ValueError("Step-5 MTP native envelope differs from independent norm-concat oracle")
        def compare():
            actual = numerical.forward_aligned(block, aligned)
            projected = block.eh_proj(envelope).float()
            native = numerical.native_forward(block, dict(hidden=projected, position_ids=next_positions))
            staged = numerical.staged_forward(block, dict(hidden=projected, position_ids=next_positions))
            exact_state(native, staged)
            if (not torch.equal(actual["hidden"], native["hidden"])
                    or not torch.equal(actual["position_ids"], native["position_ids"])
                    or not torch.equal(actual["normalized_hidden"], numerical.mtp_head_input(block, native["hidden"]))):
                raise ValueError("Step-5 MTP native/aligned decoder wiring differs")
            return actual
        original = compare()
        x = envelope.float()
        raw = dict(H=x.T @ x, count=len(x), finalized=False)
        packed, metrics, evidence = numerical.quantize_projection(block.eh_proj.weight, raw, bits=4)
        path = root / "candidates" / f"mtp-stage{stage}-eh-k4.safetensors"
        receipt = commit_safetensors(path, packed)
        reloaded = load_file(str(path))
        block.eh_proj.weight.copy_(numerical.reconstruct_projection(reloaded, list(block.eh_proj.weight.shape),
            device="cuda:0", dtype=block.eh_proj.weight.dtype))
        restored = compare()
        if not torch.isfinite(restored["hidden"]).all():
            raise ValueError("Step-5 serialized MTP result is nonfinite")
        proof = dict(stage=stage, layer=92 + stage, origin=stage + 1, shifted_pairs=len(next_tokens),
            byte_exact=True, serialized_k4_byte_exact=True, final_target_frontier=False,
            projection_receipt=receipt, projection_metrics=metrics, projection_evidence=evidence,
            contract="raw-residual[t]-shifted-token-and-absolute-position[t+1]-per-stage")
        stages.append(proof)
        progress(root, "mtp_boundary_exact", **proof)
        # The pilot teacher chain uses source BF16 predecessors for every stage;
        # candidate EH parameters never alter the next stage's input witness.
        tokens, hidden, positions = next_tokens, original["hidden"], next_positions
        del block, aligned, envelope, expected, x, raw, packed, reloaded, original, restored
        source.close()
        compact()
    proof = dict(byte_exact=True, serialized_k4_byte_exact=True, stages=stages,
        prefix_layers=8, final_target_frontier=False, tokens=len(row["input_ids"]), source_row=row["id"])
    atomic_json(root / "mtp-proof.json", proof)
    return proof


def validate_qualification(report):
    if (report.get("status") != "passed" or set(report.get("native_checks", {})) != set(CHECKS)
            or any(value is not True for value in report["native_checks"].values())):
        raise ValueError("Step-5 pilot native qualification is incomplete")
    expected = {(name, arm) for layer in (*LAYERS, LATE_LAYER) for name in modules(layer) for arm in ARMS}
    rows = report.get("results", [])
    if len(rows) != len(expected) or {(r["tensor"], r["arm"]) for r in rows} != expected:
        raise ValueError("Step-5 pilot result inventory differs")
    for row in rows:
        if not isinstance(row.get("nmse"), (int, float)) or not math.isfinite(row["nmse"]) or row["nmse"] < 0:
            raise ValueError("Step-5 pilot metric is invalid")
        if row["arm"] == "bf16" and row["nmse"] != 0:
            raise ValueError("Step-5 source precision control differs")
        if row["arm"] == "mcg-k6" and row["nmse"] > decision_policy()["maximum_k6_nmse"]:
            raise ValueError("Step-5 K6 error exceeds the absolute format bound: " + row["tensor"])
        late = row["tensor"].startswith("model.layers.88.")
        if row.get("late_expert_format_witness") is not late or row.get("causal_input_layer") != (7 if late else int(row["tensor"].split(".")[2])):
            raise ValueError("Step-5 pilot causal witness label differs")
    if report.get("full_model_quality_measured") is not False or report.get("head_measured") is not False:
        raise ValueError("short Step-5 pilot cannot claim full-model/head quality")
    scope = report.get("sparse_scope")
    if (not isinstance(scope, dict) or scope.get("panel_modified") is not False
            or scope.get("native_csa_indexer_measured") is not True
            or scope.get("global_topk_pruning_measured") is not (scope.get("maximum_pilot_prompt_tokens", 0) > 4104)
            or scope.get("pruning_requires_prompt_tokens") != 4105):
        raise ValueError("Step-5 pilot sparse-context scope differs")
    return report


def validate_report_code(report, repo=REPO):
    repo = Path(repo)
    identity = report["identity"]
    for relative, digest in identity["code"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo.resolve()) or sha256_file(path) != digest:
            raise ValueError("Step-5 pilot numerical producer changed: " + relative)
    fork = os.environ.get("STEP5_GPTQMODEL_SOURCE", str(repo / "third_party/gptqmodel"))
    actual = subprocess.check_output(["git", "-C", fork, "rev-parse", "HEAD"], text=True).strip()
    if actual != identity["gptqmodel_commit"]:
        raise ValueError("Step-5 qualification fork changed")
    return report


def make_identity(source, tokens, calibration, heldout, witness, reviewed):
    relatives = ["quantization/step5/core_k6/pilot.py", "quantization/step5/core_k6/recipe.py",
        "quantization/step5/core_k6/run.py", "quantization/step5/core_k6/finalize.py",
        "quantization/step5/prepare_corpus.py", "quantization/kimi_k3/prepare_corpus.py",
        "quantization/glm5/core_k6/pilot.py", "quantization/kimi_k3/core_k6/pilot.py",
        "third_party/gptqmodel/gptqmodel/utils/step5_core.py",
        "python/reference/families/step5/golden.py", "python/reference/families/step5/mtp.py",
        "python/reference/checkpoint_tokenizer.py"]
    return dict(schema=SCHEMA, source=str(source.snapshot), arms=list(ARMS), layers=list(LAYERS),
        reference_runtime=source.reference.verify_runtime_source(),
        source_inventory=reviewed, source_config_sha256=sha256_file(source.snapshot / "config.json"),
        source_index_sha256=sha256_file(source.snapshot / "model.safetensors.index.json"),
        token_manifest_sha256=sha256_file(tokens / "manifest.json"),
        tokens=str(tokens.resolve()), calibration_ids=[r["id"] for r in calibration],
        heldout_ids=[r["id"] for r in heldout],
        calibration_sha256=hashlib.sha256(canonical(calibration).encode()).hexdigest(),
        heldout_sha256=hashlib.sha256(canonical(heldout).encode()).hexdigest(),
        calibration_tokens=sum(len(r["input_ids"]) for r in calibration),
        heldout_tokens=sum(len(r["input_ids"]) for r in heldout),
        replay="fixed-native-bf16-eight-layer-prefix-no-candidate-affects-replay",
        selection="whole-prompts-original-order-prefix-plus-natural-window-witness-prefer-pruning-if-present",
        long_context_witness=witness["id"], late_expert=decision_policy()["late_expert"],
        maximum_pilot_prompt_tokens=max(len(r["input_ids"]) for r in (*calibration, *heldout)),
        mtp="bounded-prefix-arithmetic-and-alignment-witness-not-final-target-calibration",
        head="unmeasured-requires-finished-92-layer-stream", code={p: sha256_file(REPO / p) for p in relatives},
        gptqmodel_commit=subprocess.check_output(["git", "-C", str(REPO / "third_party/gptqmodel"), "rev-parse", "HEAD"], text=True).strip(),
        packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-snapshot", "tokens", "root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-tokens", type=int, default=2048)
    parser.add_argument("--heldout-tokens", type=int, default=768)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.calibration_tokens <= 4096 or not 256 <= args.heldout_tokens <= 1536:
        parser.error("short pilot token budget outside admission")
    from step5.prepare_corpus import load_token_records
    from .recipe import BASE_REVISION, inventory
    from gptqmodel.utils import step5_core as numerical
    source = numerical.Step5BF16Source(args.source_snapshot, reference_root=REPO / "python/reference")
    if source.snapshot.name != BASE_REVISION:
        raise ValueError("pilot source is not the pinned Step-5 BF16 revision")
    calibration, heldout, witness = select_records(load_token_records(args.tokens, "calibration"),
        load_token_records(args.tokens, "heldout"), args.calibration_tokens, args.heldout_tokens)
    identity = make_identity(source, args.tokens, calibration, heldout, witness, inventory(source))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = args.output or root / "qualification.json"
    if (root / "plan.json").exists() and json.loads((root / "plan.json").read_text()) != identity:
        raise ValueError("pilot identity differs; use a fresh root")
    if args.preflight:
        print(canonical(dict(admission="passed", identity=identity)), flush=True)
        return
    if output.exists():
        report = validate_report_code(validate_qualification(json.loads(output.read_text())))
        if report["identity"] != identity:
            raise ValueError("completed pilot identity differs")
        from .finalize import audit_evidence, retire_intermediates
        if not (root / "recovery-retired.json").exists():
            audit_evidence(root, report)
            retire_intermediates(root)
        print("Pilot already complete; no GPU work.")
        return
    import torch
    from kimi_k3.core_k6.run import exclusive_run
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86 * GIB / torch.cuda.get_device_properties(0).total_memory)
    started = time.monotonic()
    with torch.inference_mode(), exclusive_run(root):
        atomic_json(root / "plan.json", identity)
        for split, records in (("calibration", calibration), ("heldout", heldout)):
            capture_split(source, numerical, records, root / split, identity)
        rows = score(source, numerical, root, identity)
        checks = qualify_serialized_replay(source, numerical, root, identity)
        mtp = qualify_mtp(source, numerical, root, identity, calibration[0])
        checks.update(mtp_three_stage_alignment_byte_exact=True, mtp_serialized_k4_byte_exact=True)
        report = dict(status="passed", identity=identity, native_checks=checks, results=rows, mtp=mtp,
            decision_policy=decision_policy(), head_measured=False, full_model_quality_measured=False,
            sparse_scope=dict(panel_modified=False, native_csa_indexer_measured=True,
                global_topk_pruning_measured=identity["maximum_pilot_prompt_tokens"] > 4104,
                maximum_pilot_prompt_tokens=identity["maximum_pilot_prompt_tokens"],
                pruning_requires_prompt_tokens=4105,
                limitation="The unmodified public panel exercises native CSA/indexing; global top-k pruning requires more than4104 tokens."),
            elapsed_seconds=time.monotonic() - started)
        validate_qualification(report)
        atomic_json(output, report)
        atomic_json(root / "summary.json", report)
        # Audit all candidate and input receipts before retiring recovery payloads.
        from .finalize import audit_evidence, retire_intermediates
        audit_evidence(root, report)
        retire_intermediates(root)
        progress(root, "complete", results=len(rows), elapsed_seconds=report["elapsed_seconds"])
    source.close()


if __name__ == "__main__":
    main()
