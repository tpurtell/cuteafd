#!/usr/bin/env python3
"""Incrementally quantize Kimi K3's dense core over immutable K2 experts.

The numerical stages live in the GPTQModel fork. This coordinator checkpoints
whole-corpus raw Hessians, serialized projections, and rolling causal replay
states; it never keeps a whole model or corpus of GPU activations in memory.
Publication is a separate CPU invocation after the numerical process exits.
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
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("KIMI_GPTQMODEL_SOURCE", str(REPO / "third_party/gptqmodel")))
sys.path.insert(0, str(REPO / "quantization"))
os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import psutil
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from kimi_k3.prepare_corpus import load_token_records
from kimi_k3.uniform_k2.storage import (
    RollingBoundary, atomic_json, canonical, commit_safetensors, file_identity,
    fsync_directory, sha256_file, verify_receipt,
)
from kimi_k3.core_k6.recipe import (
    BASE_REPO, BASE_REVISION, MODULE_COUNT, TARGET_PAYLOAD_BYTES, TARGET_REPO,
    selected_modules,
)
from kimi_k3.core_k6.storage import StageStore

GIB = 1024 ** 3
RECIPE = dict(core_bits=6, experts="existing-serialized-k2-unchanged", seed=787,
              sigma_reg=.025, codebook="mcg", out_scales="auto",
              capture="all-calibration-tokens-raw-xtx-fp32-v1",
              hessian="signed-block-hadamard-congruence-fp64-v1",
              identity_rows=0, augmentation_rows=0,
              replay="serialized-exl3-fp16-scales-reconstructed-to-bf16",
              causal_stages="attention-entry-qb-o-moe-entry-shared-down-latent-up-head-v1",
              frontier="prefix-sum-and-complete-attention-residual-bank")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def compact_memory():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class Progress:
    def __init__(self, root, *, host_limit=58, gpu_limit=86):
        self.root = Path(root)
        self.host_limit, self.gpu_limit = host_limit * GIB, gpu_limit * GIB
        self.process = psutil.Process()
        self.state = dict(status="running", started=utc_now(), pid=os.getpid(), hostname=os.uname().nodename)

    def emit(self, event, **fields):
        memory = self.process.memory_full_info()
        record = dict(time=utc_now(), event=event, rss_gib=memory.rss / GIB,
                      open_files=self.process.num_fds(),
                      uss_gib=getattr(memory, "uss", memory.rss) / GIB,
                      cuda_allocated_gib=torch.cuda.memory_allocated() / GIB,
                      cuda_reserved_gib=torch.cuda.memory_reserved() / GIB,
                      disk_free_gib=shutil.disk_usage(self.root).free / GIB, **fields)
        self.state.update(record)
        with (self.root / "events.jsonl").open("a") as file:
            file.write(canonical(record) + "\n")
            file.flush()
        atomic_json(self.root / "status.json", self.state)
        print(canonical(record), flush=True)
        if memory.rss > self.host_limit:
            raise MemoryError(f"host RSS exceeds {self.host_limit / GIB:g} GiB admission")
        if torch.cuda.memory_allocated() > self.gpu_limit:
            raise MemoryError(f"CUDA allocated exceeds {self.gpu_limit / GIB:g} GiB admission")
        if shutil.disk_usage(self.root).free < 64 * GIB:
            raise OSError("root disk has less than 64 GiB reserve")


@contextmanager
def exclusive_run(root):
    with (Path(root) / "run.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another coordinator holds this run") from None
        yield


def make_plan(source, tokens, export, selection):
    """Bind every producer input before numerical work, without CUDA weights."""
    tokens, export = Path(tokens).resolve(), Path(export).resolve()
    manifest = json.loads((tokens / "manifest.json").read_text())
    split = manifest["splits"]["calibration"]
    if split["tokens"] != 1_201_086:
        raise ValueError("core calibration requires the reviewed full native Kimi stream")
    if len(selection) != MODULE_COUNT:
        raise ValueError("core projection selection differs from the reviewed recipe")
    relative_files = ["quantization/kimi_k3/core_k6/" + name for name in
                      ("run.py", "storage.py", "recipe.py", "export.py", "container.py")]
    relative_files += ["quantization/kimi_k3/uniform_k2/storage.py",
                       "quantization/kimi_k3/prepare_corpus.py"]
    relative_files += ["third_party/gptqmodel/gptqmodel/utils/" + name for name in
                       ("kimi_k3.py", "kimi_k3_core_format.py", "kimi_k3_core_stream.py")]
    fingerprint = {name: file_identity(source.snapshot / name) for name in sorted(set(source.weight_map.values()))}
    identity = dict(schema="cuteafd-kimi-k3-k2-core-k6-v1", base_repo=BASE_REPO,
                    base_revision=BASE_REVISION, source_config_sha256=sha256_file(source.snapshot / "config.json"),
                    base_quantize_config_sha256=sha256_file(source.snapshot / "quantize_config.json"),
                    source_index_sha256=sha256_file(source.snapshot / "model.safetensors.index.json"),
                    source_code={name: sha256_file(source.snapshot / name) for name in
                                 ("configuration_kimi_k3.py", "modeling_kimi_linear.py", "modeling_kimi_k3.py")},
                    corpus_sha256=sha256_file(tokens / "manifest.json"), recipe=RECIPE,
                    code={name: sha256_file(REPO / name) for name in relative_files},
                    gptqmodel_commit=subprocess.check_output(
                        ["git", "-C", str(REPO / "third_party/gptqmodel"), "rev-parse", "HEAD"], text=True).strip(),
                    packages={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "fla-core", "safetensors", "numpy")})
    # Worst phase transition: old frontier + D's full bank + successor F.
    # Independent narrow phase inputs and raw Hessians add at most ~64 GiB.
    frontier = split["tokens"] * 7168 * 2 * 9
    reserve = 3 * frontier + 64 * GIB
    return dict(identity=identity, source=str(source.snapshot), source_fingerprints=fingerprint,
                token_root=str(tokens), export=str(export), selection=selection,
                expected_output_bytes=TARGET_PAYLOAD_BYTES, frontier_max_bytes=frontier,
                temporary_reserve_bytes=reserve, required_free_bytes=TARGET_PAYLOAD_BYTES + reserve + 64 * GIB,
                layers=93, core_projections=MODULE_COUNT, publication_repo=TARGET_REPO)


def header_source(snapshot):
    """A source view for preflight/publication, importing no decoder or GPU code."""
    snapshot = Path(snapshot).resolve()
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "kimi_k3":
        raise ValueError("expected Kimi K3 base checkpoint")
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(weight_map.values()):
        if Path(shard).name != shard or not (snapshot / shard).is_file():
            raise ValueError("invalid or missing base checkpoint shard")
    return SimpleNamespace(snapshot=snapshot, config=config, text_config=config["text_config"], weight_map=weight_map)


def load_tensors(path, device):
    tensors = load_file(str(path), device="cpu")
    owned = {name: value.clone() for name, value in tensors.items()}
    del tensors
    return {name: value.to(device) for name, value in owned.items()}


def boundary_batch(boundary, current, index, device):
    entry = current["manifest"]["batches"][index]
    return load_tensors(boundary.root / current["directory"] / entry["filename"], device)


def batch_entries(manifest):
    return [entry for entry in manifest["entries"] if entry.get("kind") == "batch"]


def commit_batch(directory, index, row, tensors):
    filename = f"batch-{index:05d}.safetensors"
    receipt = commit_safetensors(Path(directory) / filename, tensors)
    return dict(filename=filename, kind="batch", id=row["id"], tokens=len(row["input_ids"]), **receipt)


def hardlink_frontier(boundary, layer, phase_directory, manifest):
    """Link a committed F bank into its standard boundary, retaining receipts."""
    current = boundary.current()
    if current is not None and layer != current["layer"] + 1:
        raise ValueError("hardlinked frontier must succeed the current boundary")
    entries = batch_entries(manifest)
    if not entries:
        raise ValueError("completed phase has no frontier batches")
    directory = boundary.directory(layer)
    directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        source, target = Path(phase_directory) / entry["filename"], directory / entry["filename"]
        verify_receipt(source, entry)
        if target.is_symlink():
            raise ValueError("refuse symlink successor frontier")
        if target.exists():
            if not os.path.samefile(source, target):
                target.unlink()  # This layer is not the published current boundary.
                os.link(source, target)
        else:
            os.link(source, target)
        verify_receipt(target, entry)
    fsync_directory(directory)
    return [{key: value for key, value in entry.items() if key != "kind"} for entry in entries]


class ProjectionCheckpoints:
    def __init__(self, directory, evidence_directory, identity, selection, progress, *, device="cuda:0", numerical=None):
        self.directory, self.evidence_directory = Path(directory), Path(evidence_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.evidence_directory.mkdir(parents=True, exist_ok=True)
        self.identity, self.selection, self.progress, self.device = identity, selection, progress, device
        if numerical is None:
            from gptqmodel.utils import kimi_k3_core_format as numerical
        self.numerical = numerical

    def paths(self, module):
        if module not in self.selection:
            raise ValueError("projection is outside reviewed selection")
        return self.directory / (module + ".safetensors"), self.directory / (module + ".json")

    def load(self, module):
        path, receipt_path = self.paths(module)
        if not receipt_path.exists():
            return None
        saved = json.loads(receipt_path.read_text())
        geometry = self.selection[module]
        shape = [geometry["out_features"], geometry["in_features"]]
        if saved.get("identity") != self.identity or saved.get("module") != module or saved.get("logical_shape") != shape:
            raise ValueError("projection checkpoint identity/shape differs")
        verify_receipt(path, saved, checksum=True)
        self._evidence(module, saved)
        return saved

    def _evidence(self, module, saved):
        public = {key: saved[key] for key in ("module", "logical_shape", "metrics", "evidence")}
        atomic_json(self.evidence_directory / (module + ".json"), public)

    @torch.inference_mode()
    def ensure(self, module, weight, raw_hessian=None):
        path, receipt_path = self.paths(module)
        saved = self.load(module)
        if saved is None:
            if raw_hessian is None:
                raise ValueError(f"missing raw Hessian for uncommitted projection: {module}")
            if path.exists():
                path.unlink()
            packed, metrics, evidence = self.numerical.quantize_core_projection(
                weight, raw_hessian, bits=6, seed=787, sigma=.025, device=self.device)
            receipt = commit_safetensors(path, {module + "." + suffix: value for suffix, value in packed.items()})
            saved = dict(identity=self.identity, module=module, logical_shape=list(weight.shape),
                         metrics=metrics, evidence=evidence, **receipt)
            atomic_json(receipt_path, saved)
            self._evidence(module, saved)
            self.progress.emit("projection_committed", module=module, hessian_rows=evidence["hessian_rows"],
                               hessian_error=metrics["hessian_weighted_relative_error"])
        return path

    @torch.inference_mode()
    def install(self, module, linear):
        saved = self.load(module)
        if saved is None:
            return False
        path, _ = self.paths(module)
        tensors = load_file(str(path), device="cpu")
        packed = {key.removeprefix(module + "."): value for key, value in tensors.items()}
        dense = self.numerical.reconstruct_core_projection(packed, saved["logical_shape"], device=self.device, dtype=torch.bfloat16)
        linear._parameters["weight"] = nn.Parameter(dense, requires_grad=False)
        return True

    def replacements(self, modules):
        result = {}
        for module in modules:
            if self.load(module) is None:
                raise ValueError(f"export is missing projection checkpoint: {module}")
            result[module] = self.paths(module)[0]
        return result

    def retire_payloads(self):
        for path in self.directory.glob("*.safetensors"):
            path.unlink()
        fsync_directory(self.directory)


class CoreEngine:
    """Whole-corpus causal phases for one decoder block at a time."""
    def __init__(self, source, records, root, export, identity, selection, progress, *, device="cuda:0", numerical=None):
        self.source, self.records, self.root, self.export = source, records, Path(root), Path(export)
        self.identity, self.selection, self.progress, self.device = identity, selection, progress, device
        self.tokens = sum(len(row["input_ids"]) for row in records)
        if numerical is None:
            from gptqmodel.utils import kimi_k3_core_stream as numerical
        self.math = numerical
        self.boundary = RollingBoundary(self.root / "frontier", identity)
        self.layer_seconds = []

    def _emit_batch(self, phase, layer, index):
        if index % 64 == 0 or index + 1 == len(self.records):
            self.progress.emit("phase_progress", layer=layer, phase=phase, batches=index + 1, total_batches=len(self.records))

    @torch.inference_mode()
    def initial(self):
        current = self.boundary.current()
        if current is not None:
            self.boundary.retire_old()
            return current
        self.progress.emit("embedding_started", layer=-1)
        embedding = self.source.tensor("language_model.model.embed_tokens.weight", self.device)
        directory = self.boundary.directory(-1)
        if directory.exists():
            shutil.rmtree(directory)
        entries = []
        for index, row in enumerate(self.records):
            ids = torch.tensor(row["input_ids"], dtype=torch.long, device=self.device)
            hidden = F.embedding(ids, embedding).unsqueeze(0)
            bank = hidden.new_empty((ids.numel(), 0, hidden.shape[-1]))
            entry = commit_batch(directory, index, row, dict(prefix_sum=hidden, block_residual=bank))
            entries.append({key: value for key, value in entry.items() if key != "kind"})
            self._emit_batch("embedding", -1, index)
            del ids, hidden, bank
        del embedding
        compact_memory()
        current = self.boundary.commit(-1, entries)
        self.progress.emit("embedding_committed", layer=-1)
        return current

    def _hessian(self, width):
        return self.math.DenseRawHessian(width, device=self.device)

    def _commit_phase(self, store, phase, entries, hessians=()):
        for name, accumulator in hessians:
            raw = accumulator.raw_dict(minimum_rows=self.tokens)
            if raw["count"] != self.tokens:
                raise ValueError("dense Hessian observation count differs from the full calibration corpus")
            filename = f"hessian-{name}.safetensors"
            receipt = commit_safetensors(store.directory(phase) / filename, dict(H=raw["H"]))
            entries.append(dict(filename=filename, kind="hessian", hessian=name, count=raw["count"], **receipt))
        manifest = store.commit(phase, entries, metadata=dict(tokens=self.tokens, batches=len(self.records)))
        self.progress.emit("phase_committed", phase=phase)
        return manifest

    def _raw(self, store, phase, name):
        manifest = store.load(phase)
        if manifest is None:
            raise ValueError(f"required raw Hessian phase is missing: {phase}")
        entries = [entry for entry in manifest["entries"] if entry.get("kind") == "hessian" and entry.get("hessian") == name]
        if len(entries) != 1 or entries[0]["count"] != self.tokens:
            raise ValueError("raw Hessian inventory/count differs")
        H = load_tensors(store.directory(phase) / entries[0]["filename"], self.device)["H"]
        return dict(H=H, count=self.tokens, finalized=False)

    def _read_batches(self, store, phase):
        manifest = store.load(phase)
        if manifest is None:
            raise ValueError(f"required replay phase is missing: {phase}")
        entries = batch_entries(manifest)
        if len(entries) != len(self.records):
            raise ValueError("replay phase batch inventory differs")
        for index, (row, entry) in enumerate(zip(self.records, entries, strict=True)):
            if entry["id"] != row["id"] or entry["tokens"] != len(row["input_ids"]):
                raise ValueError("replay phase corpus order differs")
            yield index, row, load_tensors(store.directory(phase) / entry["filename"], self.device)

    def _new_phase(self, store, phase):
        store.retire(phase)
        store.directory(phase).mkdir(parents=True)

    @staticmethod
    def _local_name(layer, module):
        prefix = f"language_model.model.layers.{layer}."
        if not module.startswith(prefix):
            raise ValueError("projection is outside current layer")
        return module.removeprefix(prefix)

    def _quant_group(self, block, checkpoints, modules, store, phase, hessian):
        missing = [module for module in modules if checkpoints.load(module) is None]
        raw = self._raw(store, phase, hessian) if missing else None
        for module in modules:
            linear = block.get_submodule(self._local_name(block.layer_idx, module))
            checkpoints.ensure(module, linear.weight, raw)
            checkpoints.install(module, linear)
        del raw
        compact_memory()

    @torch.inference_mode()
    def _phase_a(self, block, current, store):
        self._new_phase(store, "a")
        hessian, entries = self._hessian(7168), []
        for index, row in enumerate(self.records):
            state = boundary_batch(self.boundary, current, index, self.device)
            hidden, prefix, bank = self.math.attention_inputs(block, state["prefix_sum"], state["block_residual"])
            hessian.add(hidden)
            entries.append(commit_batch(store.directory("a"), index, row, dict(hidden=hidden)))
            self._emit_batch("a", block.layer_idx, index)
            del state, hidden, prefix, bank
        return self._commit_phase(store, "a", entries, [("input", hessian)])

    @torch.inference_mode()
    def _phase_b(self, block, store):
        self._new_phase(store, "b")
        hessian = self._hessian(1536)
        for index, _, state in self._read_batches(store, "a"):
            hidden = block.self_attn.q_a_layernorm(block.self_attn.q_a_proj(state["hidden"]))
            hessian.add(hidden)
            self._emit_batch("b", block.layer_idx, index)
            del state, hidden
        return self._commit_phase(store, "b", [], [("q-b", hessian)])

    @torch.inference_mode()
    def _phase_c(self, block, store):
        self._new_phase(store, "c")
        hessian, entries = self._hessian(12288), []
        for index, row, state in self._read_batches(store, "a"):
            inputs = self.math.capture_linear_input(block.self_attn.o_proj, block.self_attn,
                                                   hidden_states=state["hidden"], attention_mask=None, use_cache=False)
            hessian.add(inputs)
            entries.append(commit_batch(store.directory("c"), index, row, dict(o_input=inputs)))
            self._emit_batch("c", block.layer_idx, index)
            del state, inputs
        result = self._commit_phase(store, "c", entries, [("o", hessian)])
        store.retire("a", "b")
        return result

    @torch.inference_mode()
    def _phase_d(self, block, current, store):
        self._new_phase(store, "d")
        hessian, entries = self._hessian(7168) if block.layer_idx else None, []
        for index, row, state in self._read_batches(store, "c"):
            original = boundary_batch(self.boundary, current, index, self.device)
            hidden, prefix, bank = self.math.attention_inputs(block, original["prefix_sum"], original["block_residual"])
            output = block.self_attn.o_proj(state["o_input"])
            prefix, bank, hidden = self.math.attention_finish(block, prefix, bank, output)
            if hessian is not None:
                hessian.add(hidden)
            entries.append(commit_batch(store.directory("d"), index, row,
                                        dict(prefix_sum=prefix, block_residual=bank, hidden=hidden)))
            self._emit_batch("d", block.layer_idx, index)
            del original, state, hidden, prefix, bank, output
        result = self._commit_phase(store, "d", entries, [] if hessian is None else [("input", hessian)])
        store.retire("a", "b", "c")
        return result

    @staticmethod
    def release_experts(block):
        for expert in block.block_sparse_moe.experts:
            for module in expert.modules():
                for name, weight in list(module._parameters.items()):
                    if weight is not None and not weight.is_meta:
                        module._parameters[name] = nn.Parameter(torch.empty_like(weight, device="meta"), requires_grad=False)
        compact_memory()

    @torch.inference_mode()
    def _phase_e(self, block, store):
        self._new_phase(store, "e")
        shared_h, routed_h, entries = self._hessian(6144), self._hessian(3584), []
        self.progress.emit("routed_k2_load_started", layer=block.layer_idx)
        self.source.materialize_routed_experts(block, self.device)
        self.progress.emit("routed_k2_loaded", layer=block.layer_idx)
        try:
            for index, row, state in self._read_batches(store, "d"):
                shared = self.math.shared_down_input(block.block_sparse_moe.shared_experts, state["hidden"])
                shared_h.add(shared)
                routed = self.math.routed_up_input(block.block_sparse_moe, state["hidden"])
                routed_h.add(routed)
                entries.append(commit_batch(store.directory("e"), index, row, dict(routed_input=routed)))
                self._emit_batch("e", block.layer_idx, index)
                del state, shared, routed
        finally:
            self.release_experts(block)
        return self._commit_phase(store, "e", entries, [("shared-down", shared_h), ("routed-up", routed_h)])

    @torch.inference_mode()
    def _phase_f(self, block, store):
        self._new_phase(store, "f")
        entries = []
        routed_batches = self._read_batches(store, "e") if block.layer_idx else None
        for index, row, state in self._read_batches(store, "d"):
            if block.layer_idx:
                routed_index, routed_row, routed = next(routed_batches)
                if routed_index != index or routed_row["id"] != row["id"]:
                    raise ValueError("routed/shared replay corpus order differs")
                shared = self.math.shared_down_input(block.block_sparse_moe.shared_experts, state["hidden"])
                prefix, bank = self.math.complete_moe(state["prefix_sum"], state["block_residual"], shared,
                                                     routed["routed_input"], block.block_sparse_moe)
                del routed, shared
            else:
                prefix, bank = state["prefix_sum"] + block.mlp(state["hidden"]), state["block_residual"]
            if not bool(torch.isfinite(prefix).all()):
                raise ValueError("nonfinite successor frontier")
            entries.append(commit_batch(store.directory("f"), index, row, dict(prefix_sum=prefix, block_residual=bank)))
            self._emit_batch("f", block.layer_idx, index)
            del state, prefix, bank
        result = self._commit_phase(store, "f", entries)
        store.retire("a", "b", "c", "d", "e")
        return result

    def _checkpoints(self, layer):
        name = "head" if layer == "head" else f"layer-{layer:03d}"
        directory = self.root / "head/projections" if layer == "head" else self.root / f"layers/{layer:03d}/projections"
        return ProjectionCheckpoints(directory, self.root / "evidence" / name, self.identity,
                                     self.selection, self.progress, device=self.device)

    @torch.inference_mode()
    def layer(self, layer, current):
        from kimi_k3.core_k6.export import assemble_layer
        started = time.monotonic()
        store = StageStore(self.root / f"layers/{layer:03d}/stages", self.identity)
        # Verify every committed phase once on process recovery, before trusting
        # a successor whose predecessor may already have been retired.
        phases = {name: store.load(name, checksum=True) for name in ("a", "b", "c", "d", "e", "f")}
        checkpoints = self._checkpoints(layer)
        prefix = f"language_model.model.layers.{layer}."
        modules = [module for module in self.selection if module.startswith(prefix)]
        if phases["f"] is None:
            self.progress.emit("layer_load_started", layer=layer)
            block = self.source.load_block(layer, self.device, include_experts=False)
            for module in modules:
                checkpoints.install(module, block.get_submodule(self._local_name(layer, module)))
            self.progress.emit("layer_loaded", layer=layer)
            self._execute_layer(block, current, store, checkpoints, phases)
            del block
            compact_memory()
        replacements = checkpoints.replacements(modules)
        # The complete exported shard is the weight-retirement barrier. F and
        # the old current boundary survive a crash during shard construction.
        self.progress.emit("layer_assembly_started", layer=layer)
        receipt = assemble_layer(self.source, layer, replacements, self.export, self.identity)
        manifest = store.load("f")
        batches = hardlink_frontier(self.boundary, layer, store.directory("f"), manifest)
        current = self.boundary.commit(layer, batches)
        store.clear()
        checkpoints.retire_payloads()
        elapsed = time.monotonic() - started
        self.layer_seconds.append(elapsed)
        atomic_json(self.root / f"layers/{layer:03d}/complete.json",
                    dict(identity=self.identity, layer=layer, receipt=receipt, seconds=elapsed, time=utc_now()))
        durations = self.layer_seconds[-8:]
        self.progress.emit("layer_committed", layer=layer, seconds=elapsed, remaining_layers=92 - layer,
                           layer_eta_seconds=sum(durations) / len(durations) * (92 - layer))
        return current

    def _execute_layer(self, block, current, store, checkpoints, phases):
        """Shared causal replay stages for production and bounded pilots."""
        layer = block.layer_idx
        prefix = f"language_model.model.layers.{layer}."
        attention = prefix + "self_attn."
        if phases["d"] is None and phases["e"] is None:
            if phases["c"] is None:
                if phases["a"] is None:
                    self._phase_a(block, current, store)
                entry = [attention + name for name in
                         (("q_proj", "k_proj", "v_proj", "g_proj") if block.is_linear_attn else
                          ("q_a_proj", "kv_a_proj_with_mqa", "g_proj"))]
                self._quant_group(block, checkpoints, entry, store, "a", "input")
                if not block.is_linear_attn:
                    if phases["b"] is None and checkpoints.load(attention + "q_b_proj") is None:
                        self._phase_b(block, store)
                    self._quant_group(block, checkpoints, [attention + "q_b_proj"], store, "b", "q-b")
                self._phase_c(block, store)
            self._quant_group(block, checkpoints, [attention + "o_proj"], store, "c", "o")
            self._phase_d(block, current, store)
        if layer:
            moe = prefix + "block_sparse_moe."
            if phases["e"] is None:
                group = [moe + name for name in ("shared_experts.gate_proj", "shared_experts.up_proj", "routed_expert_down_proj")]
                self._quant_group(block, checkpoints, group, store, "d", "input")
                self._phase_e(block, store)
            self._quant_group(block, checkpoints, [moe + "shared_experts.down_proj"], store, "e", "shared-down")
            self._quant_group(block, checkpoints, [moe + "routed_expert_up_proj"], store, "e", "routed-up")
        self._phase_f(block, store)

    def retire_committed(self, current):
        """Finish retirement if a previous process died after current.json."""
        for layer in range(current["layer"] + 1):
            path = self.export / f"layer-{layer:03d}-receipt.json"
            receipt = json.loads(path.read_text())
            if receipt["identity"] != self.identity:
                raise ValueError("committed layer export identity differs")
            verify_receipt(self.export / receipt["filename"], receipt, checksum=True)
            StageStore(self.root / f"layers/{layer:03d}/stages", self.identity).clear()
            directory = self.root / f"layers/{layer:03d}/projections"
            for candidate in directory.glob("*.safetensors"):
                candidate.unlink()

    @torch.inference_mode()
    def head(self, current):
        from kimi_k3.core_k6.export import finalize_export
        store = StageStore(self.root / "head/stages", self.identity)
        checkpoints = self._checkpoints("head")
        module = "language_model.lm_head"
        if checkpoints.load(module) is None:
            if store.load("head", checksum=True) is None:
                self._new_phase(store, "head")
                reference = self.source.reference
                proj = nn.Linear(7168, 1, bias=False, device="meta")
                proj.weight = nn.Parameter(self.source.tensor("language_model.model.output_attn_res_proj.weight", self.device), requires_grad=False)
                mixer_norm, norm = reference.KimiRMSNorm(7168), reference.KimiRMSNorm(7168)
                mixer_norm.weight = nn.Parameter(self.source.tensor("language_model.model.output_attn_res_norm.weight", self.device), requires_grad=False)
                norm.weight = nn.Parameter(self.source.tensor("language_model.model.norm.weight", self.device), requires_grad=False)
                mixer_norm.variance_epsilon = norm.variance_epsilon = self.source.text_config["rms_norm_eps"]
                hessian = self._hessian(7168)
                for index, _ in enumerate(self.records):
                    state = boundary_batch(self.boundary, current, index, self.device)
                    hidden = self.math.final_head_input(state["prefix_sum"], state["block_residual"],
                        output_attn_res_proj=proj, output_attn_res_norm=mixer_norm, model_norm=norm,
                        apply_attn_res=reference._apply_attn_res)
                    hessian.add(hidden)
                    self._emit_batch("head", 93, index)
                    del state, hidden
                self._commit_phase(store, "head", [], [("input", hessian)])
                del hessian, proj, mixer_norm, norm
                compact_memory()
            raw = self._raw(store, "head", "input")
            # Head is quantized after the last block has been released. Its
            # source remains on CPU while the quantizer owns device admission.
            weight = self.source.tensor(module + ".weight", "cpu")
            checkpoints.ensure(module, weight, raw)
            del weight, raw
            compact_memory()
        self.progress.emit("final_export_started", layer=93)
        summary = finalize_export(self.source, self.selection, checkpoints.replacements([module]),
                                  self.export, self.identity, self.root / "evidence")
        atomic_json(self.root / "quantization-complete.json", dict(identity=self.identity, summary=summary, time=utc_now()))
        # All weights, metadata and evidence are durable before the final bank
        # and last candidate are retired. A complete-marker resume needs none.
        store.clear()
        checkpoints.retire_payloads()
        shutil.rmtree(self.boundary.root)
        self.progress.state["status"] = "quantized"
        self.progress.emit("quantization_complete", layer=93, **summary)
        return summary


def publish(export, root, repo_id):
    """CPU-only upload, remote checksum audit, and hardlinked HF snapshot."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from kimi_k3.uniform_k2.publish import install_upload_compat
    install_upload_compat()
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=str(export),
                            ignore_patterns=["*-receipt.json", ".cache/**"], num_workers=4,
                            print_report=True, print_report_every=1800)
    info = api.model_info(repo_id, files_metadata=True)
    remote, published = {file.rfilename: file for file in info.siblings}, {}
    for path in Path(export).rglob("*"):
        relative = str(path.relative_to(export))
        if not path.is_file() or path.name.endswith("-receipt.json") or relative.startswith(".cache/"):
            continue
        if relative not in remote or remote[relative].size != path.stat().st_size:
            raise ValueError(f"published file absent or wrong size: {relative}")
        item = remote[relative]
        if path.name.endswith(".safetensors"):
            receipt_name = "extras-receipt.json" if path.name.startswith("model-00094") else f"layer-{int(path.name.split('-')[1]) - 1:03d}-receipt.json"
            receipt = json.loads((Path(export) / receipt_name).read_text())
            if item.lfs is None or item.lfs.sha256 != receipt["sha256"]:
                raise ValueError(f"published LFS checksum differs: {relative}")
        elif item.lfs is not None:
            if item.lfs.sha256 != sha256_file(path):
                raise ValueError(f"published metadata checksum differs: {relative}")
        else:
            digest = hashlib.sha1()
            digest.update(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as file:
                while block := file.read(4 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != item.blob_id:
                raise ValueError(f"published Git blob differs: {relative}")
        published[relative] = path
    if ".gitattributes" in remote and ".gitattributes" not in published:
        published[".gitattributes"] = Path(hf_hub_download(repo_id, ".gitattributes", revision=info.sha, local_dir=export))
    snapshot = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / info.sha
    for name, source in published.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(source, target):
                raise ValueError(f"existing snapshot file differs: {name}")
        else:
            os.link(source, target)
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
    if Path(snapshot_download(repo_id, revision=info.sha, local_files_only=True)) != snapshot:
        raise ValueError("offline standard HF cache does not resolve the published revision")
    atomic_json(Path(root) / "publication-complete.json", dict(repo_id=repo_id, revision=info.sha,
                local_snapshot=str(snapshot), files=len(published), time=utc_now()))
    return info.sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", "--plan-only", dest="preflight", action="store_true")
    parser.add_argument("--publish", action="store_true", help="record publication intent; supervisor runs a separate CPU publisher")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if args.publish_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source = header_source(args.source_snapshot)
    selection = selected_modules(source)
    plan = make_plan(source, args.tokens, args.export, selection)
    print(canonical(plan), flush=True)
    if args.preflight:
        ancestor = args.root
        while not ancestor.exists():
            ancestor = ancestor.parent
        plan_path = args.root / "plan.json"
        if plan_path.exists():
            if not args.resume:
                raise FileExistsError("existing run requires --resume")
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("run/source/code/recipe identity changed")
        elif args.resume:
            raise FileNotFoundError("resume requires an existing plan")
        available = shutil.disk_usage(ancestor).free
        if not plan_path.exists() and available < plan["required_free_bytes"]:
            raise OSError("insufficient root NVMe space for output and rolling stages")
        if available < 64 * GIB:
            raise OSError("root disk has less than 64 GiB reserve")
        print(canonical(dict(available_bytes=available,
                             required_free_bytes=plan["required_free_bytes"])), flush=True)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    args.export.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.root)
    with exclusive_run(args.root):
        plan_path = args.root / "plan.json"
        if plan_path.exists():
            if not args.resume and not args.publish_only:
                raise FileExistsError("existing run requires --resume")
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("run/source/code/recipe identity changed; review compatibility before recovery")
        else:
            if args.resume or args.publish_only:
                raise FileNotFoundError("resume/publication requires an existing plan")
            if shutil.disk_usage(args.root).free < plan["required_free_bytes"]:
                raise OSError("insufficient root NVMe space for output and rolling stages")
            atomic_json(plan_path, plan)
        complete_path = args.root / "quantization-complete.json"
        try:
            if args.publish_only:
                complete = json.loads(complete_path.read_text())
                if complete["identity"] != plan["identity"]:
                    raise ValueError("publication requires verified complete quantization identity")
                progress.emit("publication_started", layer=93)
                revision = publish(args.export, args.root, TARGET_REPO)
                progress.state["status"] = "complete"
                progress.emit("complete", layer=93, revision=revision, **complete["summary"])
                return
            if complete_path.exists():
                complete = json.loads(complete_path.read_text())
                if complete["identity"] != plan["identity"]:
                    raise ValueError("completed quantization identity differs")
                # The marker precedes retirement by design. Finish an interrupted
                # retirement without regenerating activations or decoder objects.
                for directory in [args.root / "frontier", args.root / "head/stages"]:
                    if directory.exists():
                        shutil.rmtree(directory)
                for path in (args.root / "head/projections").glob("*.safetensors"):
                    path.unlink()
                progress.state["status"] = "quantized"
                progress.emit("quantization_complete", layer=93, **complete["summary"])
                return
            torch.set_num_threads(8)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(86 * GIB / torch.cuda.get_device_properties(0).total_memory)
            records = load_token_records(args.tokens)
            if len(records) != 1426 or sum(len(row["input_ids"]) for row in records) != 1_201_086:
                raise ValueError("calibration corpus differs from reviewed full native panel")
            from gptqmodel.utils.kimi_k3_core_format import KimiK3EXL3Source
            source = KimiK3EXL3Source(args.source_snapshot)
            engine = CoreEngine(source, records, args.root, args.export, plan["identity"], selection, progress)
            if args.resume:
                engine.boundary.current(checksum=True)
            current = engine.initial()
            engine.retire_committed(current)
            for layer in range(current["layer"] + 1, 93):
                current = engine.layer(layer, current)
            engine.head(current)
        except BaseException as error:
            progress.state.update(status="failed", error=repr(error), time=utc_now())
            atomic_json(args.root / "status.json", progress.state)
            raise
        finally:
            if hasattr(source, "close"):
                source.close()


if __name__ == "__main__":
    main()
