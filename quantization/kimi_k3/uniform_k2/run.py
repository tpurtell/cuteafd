#!/usr/bin/env python3
"""Single-device, resumable routed EXL3 K2 with bounded host allocations.

Capture each layer's latent inputs once, then gather each expert's rows from a
read-only disk mapping. Gate/up are quantized first; down Hessians see their
serialized EXL3 reconstructions. Replay the completed layer before its successor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "third_party/gptqmodel"))
sys.path.insert(0, str(REPO / "quantization"))
os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from kimi_k3.prepare_corpus import load_token_records
from kimi_k3.uniform_k2.storage import (
    RollingBoundary, assemble_shard, atomic_json, canonical, commit_safetensors,
    file_identity, fsync_directory, read_header, scratch_receipt, sha256_file, verify_receipt,
)
from gptqmodel.utils.kimi_k3 import KimiK3Source, prepare_moe, situ
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, reconstruct_exl3_tensors

GIB = 1024 ** 3
RECIPE = dict(bits=2, scope="routed-experts-only", seed=787, sigma_reg=.025,
              codebook="mcg", out_scales="auto", capture="raw-xtx-sum-fp32-v1",
              hessian="signed-block-hadamard-congruence-fp64-v1",
              recovery=dict(target=1024, ranks=[17, 32], order="rank-major-then-corpus-row",
                            residual="missing-rows-times-identity"),
              replay="serialized-exl3-reconstruction-fp16-to-bf16",
              frontier="prefix-sum-and-attention-residual-bank")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Progress:
    def __init__(self, root, *, host_limit=48, gpu_limit=86):
        self.root = Path(root)
        self.host_limit = host_limit * GIB
        self.gpu_limit = gpu_limit * GIB
        self.state = dict(status="running", started=utc_now(), pid=os.getpid(), hostname=os.uname().nodename)
        self.process = psutil.Process()

    def emit(self, event, **fields):
        memory = self.process.memory_full_info()
        memory_fields = dict(rss_gib=memory.rss / GIB, anonymous_gib=getattr(memory, "uss", memory.rss) / GIB,
                             cuda_allocated_gib=torch.cuda.memory_allocated() / GIB,
                             cuda_reserved_gib=torch.cuda.memory_reserved() / GIB,
                             disk_free_gib=shutil.disk_usage(self.root).free / GIB)
        record = dict(time=utc_now(), event=event, **memory_fields, **fields)
        with (self.root / "events.jsonl").open("a") as file:
            file.write(canonical(record) + "\n")
            file.flush()
        self.state.update(record)
        atomic_json(self.root / "status.json", self.state)
        print(canonical(record), flush=True)
        # RSS includes bounded reclaimable mappings. Anonymous/USS is a second
        # diagnostic; reject a genuinely oversized process before swapping.
        if memory.rss > self.host_limit:
            raise MemoryError(f"host RSS exceeds {self.host_limit / GIB:g} GiB admission")
        if torch.cuda.memory_allocated() > self.gpu_limit:
            raise MemoryError(f"CUDA allocated exceeds {self.gpu_limit / GIB:g} GiB admission")
        if shutil.disk_usage(self.root).free < 64 * GIB:
            raise OSError("root disk has less than 64 GiB reserve")


@contextmanager
def exclusive_run(root):
    path = Path(root) / "run.lock"
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another coordinator holds this run") from None
        yield


def compact_memory():
    gc.collect()
    torch.cuda.empty_cache()


def source_plan(source, token_root, export):
    text = source.text_config
    if (text["num_hidden_layers"], text["num_experts"], text["num_experts_per_token"],
        text["routed_expert_hidden_size"], text["moe_intermediate_size"], text["first_k_dense_replace"]) != (93, 896, 16, 3584, 3072, 1):
        raise ValueError("unreviewed Kimi K3 geometry")
    expert_parameters = 92 * 896 * 3 * 3584 * 3072
    native_bytes, source_fingerprints = 0, {}
    for shard in sorted(set(source.weight_map.values())):
        path = source.snapshot / shard
        header, _ = read_header(path)
        source_fingerprints[shard] = file_identity(path)
        for key, value in header.items():
            if key != "__metadata__" and ".block_sparse_moe.experts." not in key:
                native_bytes += value["data_offsets"][1] - value["data_offsets"][0]
    manifest = json.loads((Path(token_root) / "manifest.json").read_text())
    tokens = manifest["splits"]["calibration"]["tokens"]
    code_files = [Path(__file__), Path(__file__).with_name("storage.py"),
                  REPO / "quantization/kimi_k3/prepare_corpus.py",
                  REPO / "third_party/gptqmodel/gptqmodel/utils/kimi_k3.py"]
    identity = dict(schema="cuteafd-kimi-k3-uniform-k2-v1", source_revision=source.snapshot.name,
                    source_config_sha256=sha256_file(source.snapshot / "config.json"),
                    source_index_sha256=sha256_file(source.snapshot / "model.safetensors.index.json"),
                    source_code={name: sha256_file(source.snapshot / name) for name in
                                 ("configuration_kimi_k3.py", "modeling_kimi_linear.py", "modeling_kimi_k3.py")},
                    corpus_sha256=sha256_file(Path(token_root) / "manifest.json"), recipe=RECIPE,
                    gptqmodel_commit=subprocess.check_output(["git", "-C", str(REPO / "third_party/gptqmodel"),
                                                             "rev-parse", "HEAD"], text=True).strip(),
                    code={str(path.relative_to(REPO)): sha256_file(path) for path in code_files},
                    packages={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "fla-core", "safetensors", "numpy")})
    output_bytes = native_bytes + expert_parameters // 4 + 92 * 896 * 3 * (2 * (3584 + 3072) + 4)
    # At most two full residual banks, current latent input/route spool and one
    # layer's packed candidates + shard assembly. No full export duplication.
    frontier_bytes = tokens * 7168 * 2 * (1 + 8)
    reserve_bytes = 3 * frontier_bytes + tokens * (7168 * 2 + 3584 * 2 + 16 * 8 + 16 * 2) + 32 * GIB
    return dict(identity=identity, source_fingerprints=source_fingerprints,
                source=str(source.snapshot), token_root=str(Path(token_root).resolve()), export=str(Path(export).resolve()),
                source_native_bytes=native_bytes, expected_output_bytes=output_bytes,
                frontier_max_bytes=frontier_bytes, temporary_reserve_bytes=reserve_bytes,
                required_free_bytes=output_bytes + reserve_bytes + 64 * GIB,
                layers=93, routed_layers=92, experts=896, projections=92 * 896 * 3,
                publication_repo="wrldsuksgo2mars/Kimi-K3-EXL3-K2")


def load_boundary_batch(boundary, current, index, device="cuda:0"):
    entry = current["manifest"]["batches"][index]
    path = boundary.root / current["directory"] / entry["filename"]
    values = load_file(str(path), device="cpu")
    return {key: value.to(device) for key, value in values.items()}


@torch.inference_mode()
def initial_boundary(source, boundary, records, progress):
    current = boundary.current()
    if current is not None:
        boundary.retire_old()
        return current
    progress.emit("embedding_started", layer=-1)
    embedding = source.tensor("language_model.model.embed_tokens.weight", "cuda:0")
    directory = boundary.directory(-1)
    entries = []
    for index, row in enumerate(records):
        filename = f"batch-{index:05d}.safetensors"
        path = directory / filename
        if path.exists():
            # An uncommitted initial bank is deterministic to regenerate; never
            # infer a completed batch from a partial/unattested file.
            path.unlink()
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda:0")
        hidden = F.embedding(ids, embedding).unsqueeze(0)
        residual = hidden.new_empty((ids.numel(), 0, hidden.shape[-1]))
        receipt = commit_safetensors(path, dict(prefix_sum=hidden, block_residual=residual))
        entries.append(dict(filename=filename, id=row["id"], tokens=ids.numel(), **receipt))
        if index % 128 == 0:
            progress.emit("embedding_progress", layer=-1, batches=index + 1, total_batches=len(records))
        del hidden, residual, ids
    del embedding
    compact_memory()
    current = boundary.commit(-1, entries)
    progress.emit("embedding_committed", layer=-1, batches=len(records))
    return current


class RouteSpool:
    """Compact global latent rows and slot indices, backed by root NVMe."""
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        self.tokens = manifest["tokens"]
        self.inputs = np.memmap(self.root / "latent.bf16", dtype=np.uint16, mode="r",
                                shape=(self.tokens, 3584))
        self.routes = np.memmap(self.root / "routes.i16", dtype=np.int16, mode="r", shape=(self.tokens, 16))
        self.weights = np.memmap(self.root / "weights.f32", dtype=np.float32, mode="r", shape=(self.tokens, 16))
        self.candidates = np.memmap(self.root / "candidates.i16", dtype=np.int16, mode="r", shape=(16, self.tokens))
        self.route_rows = []
        self.recovery_rows = []
        self.counts = []
        # Each scan has bounded intermediates (~40 MB), and total inverted
        # natural routes is 154 MB. Preserve token order for Hessian summation.
        for expert in range(896):
            rows = np.nonzero(np.any(self.routes == expert, axis=1))[0]
            self.route_rows.append(rows)
            self.counts.append(len(rows))
            missing = max(0, 1024 - len(rows))
            selections = []
            for rank in range(16):
                if not missing:
                    break
                near = np.flatnonzero(self.candidates[rank] == expert)[:missing]
                selections.extend(near.tolist())
                missing -= len(near)
            self.recovery_rows.append(np.asarray(selections, dtype=np.int64))
        if sum(self.counts) != self.tokens * 16:
            raise ValueError("natural routing inventory loses or duplicates slots")

    def gather(self, indices, device="cuda:0"):
        # Advanced indexing returns owned bytes; Torch never retains mappings.
        values = np.array(self.inputs[indices], copy=True)
        return torch.from_numpy(values).view(torch.bfloat16).to(device)

    def close(self):
        for key in ("inputs", "routes", "weights", "candidates"):
            value = getattr(self, key)
            value._mmap.close()
            setattr(self, key, None)


@torch.inference_mode()
def capture_layer(block, boundary, current, records, directory, progress, identity):
    metadata_path = directory / "capture.json"
    if metadata_path.exists():
        manifest = json.loads(metadata_path.read_text())
        if manifest["identity"] != identity or manifest["layer"] != block.layer_idx:
            raise ValueError("capture identity differs")
        for name, receipt in manifest["files"].items():
            verify_receipt(directory / name, receipt, checksum=True)
        for entry in manifest["batches"]:
            verify_receipt(directory / entry["filename"], entry, checksum=True)
        return RouteSpool(directory, manifest)
    directory.mkdir(parents=True, exist_ok=True)
    tokens = sum(len(row["input_ids"]) for row in records)
    shapes = dict(latent=(tokens, 3584), routes=(tokens, 16), weights=(tokens, 16), candidates=(16, tokens))
    dtypes = dict(latent=np.uint16, routes=np.int16, weights=np.float32, candidates=np.int16)
    names = dict(latent="latent.bf16", routes="routes.i16", weights="weights.f32", candidates="candidates.i16")
    maps = {key: np.memmap(directory / names[key], mode="w+", dtype=dtypes[key], shape=shape)
            for key, shape in shapes.items()}
    batches, offset = [], 0
    try:
        for index, row in enumerate(records):
            state = load_boundary_batch(boundary, current, index)
            prefix, residual, hidden = prepare_moe(block, state["prefix_sum"], state["block_residual"])
            moe = block.block_sparse_moe
            selected, weights = moe.gate(hidden)
            flat = hidden.reshape(-1, hidden.shape[-1])
            latent = moe.routed_expert_down_proj(flat)
            logits = F.linear(flat.float(), moe.gate.weight.float())
            scores = logits.sigmoid() + moe.gate.e_score_correction_bias.unsqueeze(0)
            # Recovery candidates must exclude the EXACT live selected set,
            # including a possible tied top-k boundary. Never alter live routes.
            scores.scatter_(1, selected, float("-inf"))
            near = scores.topk(16, dim=-1, sorted=True).indices
            length = flat.shape[0]
            maps["latent"][offset:offset + length] = latent.cpu().view(torch.uint16).numpy()
            maps["routes"][offset:offset + length] = selected.cpu().short().numpy()
            maps["weights"][offset:offset + length] = weights.cpu().float().numpy()
            maps["candidates"][:, offset:offset + length] = near.cpu().short().numpy().T
            # Only the small post-attention state is saved; MoE input hidden
            # can be recreated from latent + shared output after quantization.
            filename = f"attention-{index:05d}.safetensors"
            target = directory / filename
            if target.exists():
                target.unlink()
            shared = moe.shared_experts(hidden)
            receipt = commit_safetensors(target, dict(prefix_sum=prefix, block_residual=residual, shared=shared))
            batches.append(dict(filename=filename, id=row["id"], offset=offset, tokens=length, **receipt))
            offset += length
            if index % 64 == 0:
                progress.emit("capture_progress", layer=block.layer_idx, batches=index + 1, total_batches=len(records))
            del state, prefix, residual, hidden, latent, logits, scores, near, selected, weights, shared, flat
        if offset != tokens:
            raise ValueError("capture token count differs")
        files = {}
        for key, mapping in maps.items():
            mapping.flush()
            mapping._mmap.close()
            path = directory / names[key]
            with path.open("rb") as file:
                os.fsync(file.fileno())
            files[names[key]] = scratch_receipt(path)
        maps.clear()
        fsync_directory(directory)
        manifest = dict(identity=identity, layer=block.layer_idx, tokens=tokens, batches=batches, files=files)
        atomic_json(metadata_path, manifest)
    finally:
        for mapping in maps.values():
            mapping._mmap.close()
    progress.emit("capture_committed", layer=block.layer_idx, tokens=tokens)
    return RouteSpool(directory, manifest)


@torch.inference_mode()
def capture_hessian(spool, expert, *, projections=None, chunk_rows=2048):
    down = projections is not None
    width = 3072 if down else 3584
    hessian = torch.zeros((width, width), dtype=torch.float32, device="cuda:0")
    natural = spool.route_rows[expert]
    recovery = spool.recovery_rows[expert]
    for selection in (natural, recovery):
        for start in range(0, len(selection), chunk_rows):
            rows = spool.gather(selection[start:start + chunk_rows])
            if down:
                gate = F.linear(rows, projections["w1"])
                up = F.linear(rows, projections["w3"])
                rows = situ(gate, up).to(torch.bfloat16)
            value = rows.float()
            hessian.addmm_(value.T, value)
    residual_rows = max(0, 1024 - len(natural) - len(recovery))
    hessian.diagonal().add_(float(residual_rows))
    count = len(natural) + len(recovery) + residual_rows
    if not bool(torch.isfinite(hessian).all()) or float(hessian.diagonal().sum()) <= 0:
        raise ValueError("invalid captured Hessian")
    evidence = dict(natural_rows=len(natural), augmented_rows=len(recovery), identity_rows=residual_rows,
                    effective_rows=count, candidate_selection_sha256=hashlib.sha256(recovery.tobytes()).hexdigest(),
                    ranks=[17, 32], selection_policy="rank-major-then-corpus-row-v1",
                    down_inputs="serialized-quantized-gate-up-situ-bf16" if down else None)
    return dict(H=hessian, count=count, finalized=False), evidence


@torch.inference_mode()
def quantize_projection(weight, hessian):
    if hessian["finalized"]:
        raise ValueError("projection must start from the immutable raw Hessian")
    args = dict(K=2, devices=[torch.device("cuda:0")], apply_out_scales=None,
                sigma_reg=.025, seed=787, mcg=True)
    # Independent finalization is required for deterministic partial-expert
    # resume: a cached finalized H would consume a different RNG subsequence
    # for output signs than a freshly rebuilt H after restarting after w1.
    working = dict(H=hessian["H"].clone(), count=hessian["count"], finalized=False)
    _, _, packed = quantize_exl3(weight.T.contiguous().float(), working, args, return_weight_q=False)
    metrics = args["error_metrics"]
    if (metrics["quantizer_path"] != "hessian_ldlq" or metrics["hessian_metric_status"] != "ok"
            or metrics["hessian_numerical_contract"] != RECIPE["hessian"]):
        raise ValueError("EXL3 projection did not use calibrated Hessian path")
    packed = {key: value.detach().cpu() for key, value in packed.items()}
    return packed, metrics


@torch.inference_mode()
def quantize_experts(block, spool, directory, progress, identity):
    prefix = f"language_model.model.layers.{block.layer_idx}.block_sparse_moe.experts"
    receipts = []
    started = time.monotonic()
    for expert, module in enumerate(block.block_sparse_moe.experts):
        inputs_h, input_evidence = None, None
        entry = dict(expert=expert, projections={})
        for projection in ("w1", "w3", "w2"):
            name = f"{prefix}.{expert}.{projection}"
            path = directory / f"expert-{expert:03d}-{projection}.safetensors"
            receipt_path = path.with_suffix(".json")
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                if saved["identity"] != identity or saved["module"] != name:
                    raise ValueError("projection checkpoint identity differs")
                verify_receipt(path, saved, checksum=True)
                tensors = load_file(str(path))
                packed = {key.removeprefix(name + "."): value for key, value in tensors.items()}
            else:
                if path.exists():
                    path.unlink()
                if projection == "w2":
                    hessian, evidence = capture_hessian(spool, expert, projections={
                        key: getattr(module, key).weight for key in ("w1", "w3")})
                else:
                    if inputs_h is None:
                        inputs_h, input_evidence = capture_hessian(spool, expert)
                    hessian, evidence = inputs_h, input_evidence
                weight = getattr(module, projection).weight
                packed, metrics = quantize_projection(weight, hessian)
                receipt = commit_safetensors(path, {name + "." + key: value for key, value in packed.items()})
                saved = dict(identity=identity, module=name, evidence=evidence, metrics=metrics, **receipt)
                atomic_json(receipt_path, saved)
                progress.emit("projection_committed", layer=block.layer_idx, expert=expert, projection=projection,
                              natural_rows=evidence["natural_rows"], recovery_rows=evidence["augmented_rows"],
                              identity_rows=evidence["identity_rows"],
                              hessian_error=metrics["hessian_weighted_relative_error"])
                if projection == "w2":
                    del hessian
            # Actual exported half scales and trellis are authoritative. Never
            # propagate quantizer's pre-serialization dense candidate weights.
            dense = reconstruct_exl3_tensors(packed, device="cuda:0", dtype=torch.bfloat16).T.contiguous()
            getattr(module, projection).weight = nn.Parameter(dense, requires_grad=False)
            entry["projections"][projection] = dict(filename=path.name, receipt=saved)
            del packed, dense
        receipts.append(entry)
        del inputs_h, input_evidence
        if expert % 16 == 0:
            compact_memory()
            elapsed = time.monotonic() - started
            progress.emit("expert_progress", layer=block.layer_idx, expert=expert, experts=896,
                          seconds_per_expert=elapsed / (expert + 1),
                          quantization_eta_seconds=(elapsed / (expert + 1)) *
                          ((92 - block.layer_idx) * 896 + 895 - expert))
    atomic_json(directory / "experts-complete.json", dict(identity=identity, layer=block.layer_idx, experts=receipts))
    return receipts


@torch.inference_mode()
def replay_layer(block, spool, boundary, records, progress):
    layer = block.layer_idx
    batches = []
    directory = boundary.directory(layer)
    for index, (row, captured) in enumerate(zip(records, spool.manifest["batches"], strict=True)):
        state = load_file(str(spool.root / captured["filename"]), device="cpu")
        state = {key: value.to("cuda:0") for key, value in state.items()}
        start, length = captured["offset"], captured["tokens"]
        latent = spool.gather(np.arange(start, start + length))
        routes = torch.from_numpy(np.array(spool.routes[start:start + length], copy=True)).long().cuda()
        weights = torch.from_numpy(np.array(spool.weights[start:start + length], copy=True)).cuda()
        moe = block.block_sparse_moe
        # Use the checkpoint's exact token/slot ordering and weighted reduction.
        output = moe.moe_infer(latent, routes, weights)
        output = moe.routed_expert_norm(output)
        output = moe.routed_expert_up_proj(output).view(1, length, -1) + state["shared"]
        prefix = state["prefix_sum"] + output
        if not bool(torch.isfinite(prefix).all()):
            raise ValueError("nonfinite replay frontier")
        filename = f"batch-{index:05d}.safetensors"
        path = directory / filename
        if path.exists():
            path.unlink()  # Current layer is still uncommitted; never old bank.
        receipt = commit_safetensors(path, dict(prefix_sum=prefix, block_residual=state["block_residual"]))
        batches.append(dict(filename=filename, id=row["id"], tokens=length, **receipt))
        if index % 64 == 0:
            progress.emit("replay_progress", layer=layer, batches=index + 1, total_batches=len(records))
        del state, latent, routes, weights, output, prefix
    return batches


def assemble_layer(source, layer, candidate_dir, export, identity):
    prefix = f"language_model.model.layers.{layer}."
    entries = {name: (source.snapshot / shard, name) for name, shard in source.weight_map.items()
               if name.startswith(prefix) and ".block_sparse_moe.experts." not in name}
    metrics = []
    if layer:
        completed = json.loads((candidate_dir / "experts-complete.json").read_text())
        if completed["identity"] != identity:
            raise ValueError("assembled layer identity differs")
        for expert in completed["experts"]:
            for projection, entry in expert["projections"].items():
                path = candidate_dir / entry["filename"]
                verify_receipt(path, entry["receipt"])
                header, _ = read_header(path)
                entries.update({name: (path, name) for name in header if name != "__metadata__"})
                metrics.append(dict(module=entry["receipt"]["module"], evidence=entry["receipt"]["evidence"],
                                    metrics=entry["receipt"]["metrics"]))
    shard = export / f"model-{layer + 1:05d}-of-00094.safetensors"
    receipt_path = export / f"layer-{layer:03d}-receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity:
            raise ValueError("export shard identity differs")
        verify_receipt(shard, receipt)
        return receipt
    if shard.exists():
        shard.unlink()  # Orphaned assembly has no durable receipt.
    receipt = assemble_shard(entries, shard)
    # Keep error evidence beside the run, then publish it as aggregate metadata.
    atomic_json(candidate_dir / "layer-error-summary.json", dict(layer=layer, projections=metrics))
    receipt.update(identity=identity, filename=shard.name, layer=layer, tensors=len(entries))
    atomic_json(receipt_path, receipt)
    return receipt


@torch.inference_mode()
def dense_layer(source, block, boundary, current, records, progress):
    batches = []
    for index, row in enumerate(records):
        state = load_boundary_batch(boundary, current, index)
        prefix, residual = block(state["prefix_sum"], block_residual=state["block_residual"], use_cache=False)
        if not bool(torch.isfinite(prefix).all()):
            raise ValueError("nonfinite dense frontier")
        filename = f"batch-{index:05d}.safetensors"
        path = boundary.directory(0) / filename
        if path.exists():
            path.unlink()
        receipt = commit_safetensors(path, dict(prefix_sum=prefix, block_residual=residual))
        batches.append(dict(filename=filename, id=row["id"], tokens=len(row["input_ids"]), **receipt))
        if index % 64 == 0:
            progress.emit("dense_progress", layer=0, batches=index + 1, total_batches=len(records))
        del state, prefix, residual
    return batches


def finalize_export(source, export, identity):
    extras = {name: (source.snapshot / shard, name) for name, shard in source.weight_map.items()
              if not name.startswith("language_model.model.layers.")}
    shard = export / "model-00094-of-00094.safetensors"
    receipt_path = export / "extras-receipt.json"
    if receipt_path.exists():
        verify_receipt(shard, json.loads(receipt_path.read_text()))
    else:
        if shard.exists():
            shard.unlink()
        receipt = assemble_shard(extras, shard)
        atomic_json(receipt_path, dict(identity=identity, filename=shard.name, **receipt))
    weight_map, total = {}, 0
    actual_specs = {}
    for path in sorted(export.glob("model-*.safetensors")):
        header, _ = read_header(path)
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map:
                raise ValueError("duplicate exported tensor")
            weight_map[name] = path.name
            actual_specs[name] = (value["dtype"], value["shape"])
            total += value["data_offsets"][1] - value["data_offsets"][0]
    expected = {name for name in source.weight_map if ".block_sparse_moe.experts." not in name}
    tensor_storage = {}
    for layer in range(1, 93):
        for expert in range(896):
            for projection in ("w1", "w3", "w2"):
                prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}"
                expected.update(prefix + "." + suffix for suffix in ("trellis", "suh", "svh", "mcg"))
                inputs, outputs = (3072, 3584) if projection == "w2" else (3584, 3072)
                tensor_storage[prefix] = dict(quant_format="exl3", bits_per_weight=2,
                    mcg_multiplier=0xCBAC1FED, stored_tensors={
                        prefix + ".trellis": dict(shape=[inputs // 16, outputs // 16, 32], torch_dtype="int16"),
                        prefix + ".suh": dict(shape=[inputs], torch_dtype="float16"),
                        prefix + ".svh": dict(shape=[outputs], torch_dtype="float16"),
                        prefix + ".mcg": dict(shape=[], torch_dtype="int32")})
                for suffix, dtype, shape in (("trellis", "I16", [inputs // 16, outputs // 16, 32]),
                                             ("suh", "F16", [inputs]), ("svh", "F16", [outputs]),
                                             ("mcg", "I32", [])):
                    if actual_specs.get(prefix + "." + suffix) != (dtype, shape):
                        raise ValueError(f"exported EXL3 dtype/geometry differs: {prefix}.{suffix}")
    if set(weight_map) != expected:
        raise ValueError("final export tensor inventory differs")
    atomic_json(export / "model.safetensors.index.json", dict(metadata=dict(total_size=total), weight_map=weight_map))
    config = json.loads(canonical(source.config))
    declaration = dict(quant_method="exl3", format="exl3", checkpoint_format="exl3", bits=2.)
    config["quantization_config"] = declaration
    config["text_config"]["quantization_config"] = declaration
    atomic_json(export / "config.json", config)
    atomic_json(export / "quantize_config.json", dict(method="exl3", quant_method="exl3", format="exl3",
                checkpoint_format="exl3", bits=2., codebook="mcg", out_scales="auto", group_size=-1,
                desc_act=False, module_include=[r"language_model\.model\.layers\.[0-9]+\.block_sparse_moe\.experts\.[0-9]+\.w[123]$"],
                tensor_storage=tensor_storage, meta=dict(quantization_provenance=identity)))
    for path in source.snapshot.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name not in {
                "config.json", "model.safetensors.index.json", "README.md", ".gitattributes"}:
            shutil.copyfile(path, export / path.name)
    card = f'''---
license: other
license_name: kimi-k3
license_link: https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
base_model: moonshotai/Kimi-K3
datasets:
- wrldsuksgo2mars/agentic-quant-panel-v1
tags:
- exl3
- quantized
- kimi_k3
---
# Kimi K3 EXL3 K2

Uniform 2-bit EXL3 MCG trellis quantization of all 247,296 routed expert
projections in the 92 MoE layers. The first dense layer, attention, latent MoE
projections, routers, shared experts, embeddings, head and native MoonViT-V2
vision tensors are preserved byte-for-byte from the source checkpoint.

Calibration uses all 1,426 prompts (1,201,086 Kimi tokens) from the pinned
agentic panel, in original order with the checkpoint's segmented chat encoder,
thinking enabled and thinking effort max. Source revision: `{identity['source_revision']}`.
Seed 787; sigma regularization 0.025; automatic output scales; FP64 signed
Hadamard Hessian congruence. Experts below 1,024 natural rows use adjacent
router ranks 17–32, then an explicitly recorded identity residual. Down
projections are calibrated after gate/up EXL3 serialization and reload.

The entire model does not fit on a single 96 GB GPU. Quantization ran one
layer at a time on a 96 GB GPU with a disk-backed rolling attention-residual bank.
The source license and custom tokenizer/processor files are included.
Structural verification and per-projection error evidence accompany this
artifact. Full-model generation/benchmark evaluation has not been performed.
'''
    (export / "README.md").write_text(card)
    atomic_json(export / "quantization-provenance.json", identity)
    return dict(tensors=len(weight_map), payload_bytes=total, shards=94, routed_projections=247296)


def _receipt_digests(export):
    """Writer SHA256 per shard from its layer receipt; HTTP LFS never rehashes weights."""
    digests = {}
    for path in Path(export).glob("*.safetensors"):
        name = ("extras-receipt.json" if path.name.startswith("model-00094")
                else f"layer-{int(path.name.split('-')[1]) - 1:03d}-receipt.json")
        receipt = json.loads((Path(export) / name).read_text())
        verify_receipt(path, receipt)
        digests[path.name] = receipt["sha256"]
    return digests


def publish(export, root, repo_id, *, options=None):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    import hub_lfs
    from hub_requirements import require_qualified_hub
    require_qualified_hub()  # HTTP LFS before any remote mutation.
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    # Verified, resumable HTTP LFS batches; hashes of small files persist under root.
    hub_lfs.upload_folder(api, repo_id, export, ignore_patterns=["*-receipt.json", "layer-*-receipt.json", ".cache/**"], state=root,
                          digests=_receipt_digests(export), options=options,
                          commit_message="Publish calibrated EXL3 checkpoint")
    info = api.model_info(repo_id, files_metadata=True)
    remote = {file.rfilename: file for file in info.siblings}
    published_files = {}
    for path in export.rglob("*"):
        relative = str(path.relative_to(export))
        if not path.is_file() or path.name.endswith("-receipt.json") or relative.startswith(".cache/"):
            continue
        if relative not in remote or remote[relative].size != path.stat().st_size:
            raise ValueError(f"published file absent or wrong size: {relative}")
        remote_file = remote[relative]
        if path.name.endswith(".safetensors"):
            receipt_name = "extras-receipt.json" if path.name.startswith("model-00094") else (
                f"layer-{int(path.name.split('-')[1]) - 1:03d}-receipt.json")
            receipt = json.loads((export / receipt_name).read_text())
            lfs = remote_file.lfs
            if lfs is None or lfs.sha256 != receipt["sha256"]:
                raise ValueError(f"published LFS checksum differs: {path.name}")
        elif remote_file.lfs is not None:
            if remote_file.lfs.sha256 != sha256_file(path):
                raise ValueError(f"published metadata checksum differs: {relative}")
        else:
            digest = hashlib.sha1()
            digest.update(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as file:
                while block := file.read(4 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != remote_file.blob_id:
                raise ValueError(f"published Git blob differs: {relative}")
        published_files[relative] = path
    # Hub may append its own LFS rules. Fetch only that small generated asset,
    # then materialize a complete snapshot with same-filesystem hardlinks.
    if ".gitattributes" in remote and ".gitattributes" not in published_files:
        path = Path(hf_hub_download(repo_id, ".gitattributes", revision=info.sha, local_dir=export))
        published_files[".gitattributes"] = path
    snapshot = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / info.sha
    for name, source_path in published_files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(source_path, target):
                raise ValueError(f"existing snapshot file differs: {name}")
        else:
            os.link(source_path, target)
            fsync_directory(target.parent)
    refs = snapshot.parents[1] / "refs"
    refs.mkdir(exist_ok=True)
    temporary = refs / ".main.partial"
    with temporary.open("w") as file:
        file.write(info.sha)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, refs / "main")
    fsync_directory(refs)
    cached = snapshot_download(repo_id, revision=info.sha, local_files_only=True)
    if Path(cached) != snapshot:
        raise ValueError("offline standard HF cache does not resolve the published revision")
    atomic_json(root / "publication-complete.json", dict(repo_id=repo_id, revision=info.sha,
                local_snapshot=str(snapshot), files=len(published_files), time=utc_now()))
    return info.sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    import hub_lfs
    hub_lfs.add_arguments(parser)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    source = KimiK3Source(args.source_snapshot)
    plan = source_plan(source, args.tokens, args.export)
    print(canonical(plan), flush=True)
    if args.plan_only:
        return
    if args.publish_only:
        with exclusive_run(args.root):
            if json.loads((args.root / "plan.json").read_text()) != plan:
                raise ValueError("publication recovery plan identity differs")
            complete = json.loads((args.root / "quantization-complete.json").read_text())
            if complete["identity"] != plan["identity"]:
                raise ValueError("publication requires a verified complete quantization")
            progress = Progress(args.root)
            progress.emit("publication_started", layer=92)
            revision = publish(args.export, args.root, plan["publication_repo"], options=hub_lfs.options_from(args))
            progress.state["status"] = "complete"
            progress.emit("complete", layer=92, revision=revision, **complete["summary"])
        return
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = load_token_records(args.tokens)
    args.export.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.root)
    with exclusive_run(args.root):
        plan_path = args.root / "plan.json"
        if plan_path.exists():
            if not args.resume:
                raise FileExistsError("existing run requires --resume")
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("run/source/code/recipe identity changed; review recovery compatibility first")
        else:
            if args.resume:
                raise FileNotFoundError("resume requires existing plan")
            if shutil.disk_usage(args.root).free < plan["required_free_bytes"]:
                raise OSError("insufficient root NVMe space for output and rolling banks")
            atomic_json(plan_path, plan)
        complete_path = args.root / "quantization-complete.json"
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if complete["identity"] != plan["identity"]:
                raise ValueError("completed quantization identity differs")
            progress.state["status"] = "quantized"
            progress.emit("quantization_complete", layer=92, **complete["summary"])
            return
        boundary = RollingBoundary(args.root / "frontier", plan["identity"])
        if args.resume:
            boundary.current(checksum=True)
        current = initial_boundary(source, boundary, records, progress)
        # An interruption after current.json publication may precede transient
        # retirement. Clean those committed layers once their shard is verified.
        for completed_layer in range(current["layer"] + 1):
            receipt = json.loads((args.export / f"layer-{completed_layer:03d}-receipt.json").read_text())
            if receipt["identity"] != plan["identity"]:
                raise ValueError("completed shard run identity differs")
            verify_receipt(args.export / receipt["filename"], receipt)
            captured = args.root / f"layers/{completed_layer:03d}/capture"
            if captured.exists():
                shutil.rmtree(captured)
            for path in (args.root / f"layers/{completed_layer:03d}/projections").glob("*.safetensors"):
                path.unlink()
        try:
            for layer in range(current["layer"] + 1, 93):
                started = time.monotonic()
                candidate_dir = args.root / f"layers/{layer:03d}/projections"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                progress.emit("layer_load_started", layer=layer)
                block = source.load_block(layer)
                progress.emit("layer_loaded", layer=layer)
                if layer == 0:
                    batches = dense_layer(source, block, boundary, current, records, progress)
                else:
                    spool = capture_layer(block, boundary, current, records, args.root / f"layers/{layer:03d}/capture",
                                          progress, plan["identity"])
                    quantize_experts(block, spool, candidate_dir, progress, plan["identity"])
                    batches = replay_layer(block, spool, boundary, records, progress)
                    spool.close()
                    del spool
                del block
                compact_memory()
                receipt = assemble_layer(source, layer, candidate_dir, args.export, plan["identity"])
                current = boundary.commit(layer, batches)
                # All successor inputs and final layer weights are durable now.
                # Preserve receipts/error evidence; retire only bulky transients.
                layer_dir = args.root / f"layers/{layer:03d}"
                capture_dir = layer_dir / "capture"
                if capture_dir.exists():
                    shutil.rmtree(capture_dir)
                for path in candidate_dir.glob("*.safetensors"):
                    path.unlink()
                elapsed = time.monotonic() - started
                atomic_json(layer_dir / "complete.json", dict(identity=plan["identity"], layer=layer,
                            receipt=receipt, seconds=elapsed, time=utc_now()))
                progress.emit("layer_committed", layer=layer, seconds=elapsed,
                              remaining_layers=92 - layer, layer_eta_seconds=elapsed * (92 - layer))
            summary = finalize_export(source, args.export, plan["identity"])
            evidence_dir = args.export / "quantization-errors"
            evidence_dir.mkdir(exist_ok=True)
            for path in args.root.glob("layers/*/projections/layer-error-summary.json"):
                shutil.copyfile(path, evidence_dir / f"layer-{path.parents[1].name}.json")
            progress.emit("export_verified", layer=92, **summary)
            atomic_json(args.root / "quantization-complete.json", dict(identity=plan["identity"],
                        summary=summary, time=utc_now()))
            shutil.rmtree(boundary.root)
            if args.publish:
                revision = publish(args.export, args.root, plan["publication_repo"], options=hub_lfs.options_from(args))
                progress.emit("published", revision=revision, repo=plan["publication_repo"])
            progress.state["status"] = "complete" if args.publish else "quantized"
            progress.emit("complete" if args.publish else "quantization_complete", layer=92, **summary)
        except BaseException as error:
            progress.state.update(status="failed", error=repr(error))
            # Do not let an exhausted-memory guard hide the original exception.
            atomic_json(args.root / "status.json", dict(progress.state, time=utc_now()))
            raise


if __name__ == "__main__":
    main()
