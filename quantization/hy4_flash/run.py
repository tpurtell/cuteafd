"""One K8-causal Hy4 quantization pass, resumed at committed stages."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from itertools import islice
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import time

import torch
import torch.nn.functional as F

from inkling2 import numerical
from kimi_k3.uniform_k2.storage import atomic_json, fsync_directory, verify_receipt
from . import gates, recipe
from .export import DirectMCWriter
from .seal import seal_package
from .forward import AttentionTap, Drafter, Layer, Model, Reader, native_projection, recalled_context, reference
from .prepare_corpus import load_token_batches
from .source import HeaderSource
from .storage import CANDIDATE_GAMMA_CONTRACT, IDENTITY_CONTRACT, MINIMUM_EXPERT_ROWS, Routes, Stages, candidate_routes


REPO = Path(__file__).resolve().parents[2]
CODE = (
    "quantization/hy4_flash/run.py", "quantization/hy4_flash/storage.py",
    "quantization/hy4_flash/gates.py", "quantization/hy4_flash/recipe.py", "quantization/hy4_flash/forward.py",
    "quantization/hy4_flash/source.py", "quantization/hy4_flash/prepare_corpus.py",
    "quantization/hy4_flash/parity.py", "quantization/hy4_flash/engram.py",
    "quantization/hy4_flash/export.py", "quantization/hy4_flash/seal.py", "quantization/core_export.py",
    "quantization/inkling2/numerical.py",
    "quantization/inkling2/export.py", "quantization/kimi_k3/core_k6/storage.py",
    "quantization/kimi_k3/uniform_k2/storage.py", "quantization/kimi_k3/prepare_corpus.py",
    "quantization/glm5/core_k6/recipe.py", "quantization/mc_package.py",
    "scripts/lib/mc_manifest.py",
    "python/reference/families/hy4_flash", "python/reference/shape_invariant.py",
)
SEGMENT_BYTES = 256 * (1 << 20)
EXPERT_GROUP = 32
EXPERT_ROWS = 2048


def read_json(path):
    return json.loads(path.read_text())


def git(*args, cwd=REPO):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def verify_preflight(report, corpus_digest):
    reference_dir = REPO / "python/reference/families/hy4_flash"
    files = [Path(__file__).with_name("parity.py"), Path(__file__).with_name("forward.py"),
             reference_dir / "model.py", reference_dir / "golden.py"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if (report.get("schema") != "cuteafd-hy4-forward-parity-v2" or report.get("status") != "passed"
            or report.get("code_identity") != hashes or report.get("corpus_manifest_sha256") != corpus_digest
            or report.get("source_revision") != recipe.BASE_REVISION
            or report.get("reference_revision") != recipe.REFERENCE_REVISION
            or [r["layer"] for r in report.get("layers", [])] != list(range(recipe.LAYER_COUNT))
            or len(report.get("head", [])) != 3 or len(report.get("dfly2", [])) != 3
            or any(len(r["layers"]) != recipe.MTP_LAYER_COUNT for r in report["dfly2"])
            or {r["tokens"] for r in report.get("context", []) if r["hessian_eligible"] is False} != {2709, 513}):
        raise ValueError("Hy4 requires completed parity on these source files, calibration corpus and references")


def identity(source, corpus, preflight):
    if os.environ.get("INKLING_GPTQMODEL_ROOT"):
        raise ValueError("Hy4 production requires the committed GPTQModel pin")
    if git("status", "--porcelain", "--", *CODE):
        raise ValueError("Hy4 producer files must be committed before production")
    submodule = REPO / "third_party/gptqmodel"
    fork = git("rev-parse", "HEAD", cwd=submodule)
    if (git("status", "--porcelain", cwd=submodule)
            or git("ls-tree", "HEAD", "third_party/gptqmodel").split()[2] != fork):
        raise ValueError("Hy4 GPTQModel differs from the committed fork pin")
    lock = read_json(REPO / "third_party/gptqmodel.lock.json")
    digest = runpy.run_path(str(REPO / "scripts/build/verify-sparkinfer-source.py"))["source_tree_sha256"](submodule)
    if lock["revision"] != fork or lock["source_tree_sha256"] != digest:
        raise ValueError("GPTQModel differs from the committed source lock")
    manifest = read_json(corpus / "manifest.json")
    corpus_digest = hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest()
    verify_preflight(preflight, corpus_digest)
    paths = git("ls-files", "--", *CODE).splitlines()
    return dict(schema="cuteafd-hy4-causal-mc-v2", source_repo=recipe.BASE_REPO,
                source_revision=recipe.BASE_REVISION, reference_revision=recipe.REFERENCE_REVISION,
                source_fingerprints=source.source_fingerprints,
                corpus_manifest_sha256=corpus_digest,
                code=dict(fork=fork, files={path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
                                           for path in paths}),
                preflight=preflight, dataset=manifest["dataset"], components=manifest["components"],
                codebook="mul1", codebook_multiplier=recipe.MUL1_MULTIPLIER,
                seed=787, sigma=.025, expert_bits=4, core_bits=[8, 6],
                upstream_core_bits=8, calibration="ieee-fp32-reference",
                covariance=numerical.ACCUMULATION_CONTRACT,
                expert_rows=EXPERT_ROWS, expert_group=EXPERT_GROUP, minimum_expert_rows=MINIMUM_EXPERT_ROWS,
                candidate_gamma=CANDIDATE_GAMMA_CONTRACT, expert_identity=IDENTITY_CONTRACT,
                replay_weights="serialized-exl3-reconstructed-bf16-promoted-fp32",
                drafter_anchor="last-committed-target-state-k8-head-greedy-bonus")


def checkpoint_identity(root, candidate):
    path = root / "identity.json"
    if path.exists():
        if read_json(path) != candidate:
            raise ValueError("Hy4 numerical identity differs from the frozen v2 run")
    else:
        if (root / "frontier.json").exists() or (root / "stages").exists():
            raise ValueError("Hy4 stages without a frozen v2 identity")
        atomic_json(path, candidate)
    return candidate


class Progress:
    def __init__(self, root, total):
        self.root, self.total = root, total
        self.base = dict(target_layers=recipe.LAYER_COUNT, drafter_layers=recipe.MTP_LAYER_COUNT,
                         total_encode_operations=total, completed_target_layers=0,
                         completed_drafter_layers=0, committed_encodes=0)

    def event(self, phase, *, snapshot=True, **fields):
        row = self.base | fields | dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat())
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if snapshot:
            atomic_json(self.root / "status.json", dict(status="running", **row))
        print(json.dumps(row), flush=True)


@torch.inference_mode()
def raw(acc, identity_rows=0):
    if acc.count + identity_rows == 0:
        raise ValueError("Hy4 projection has no calibration rows")
    matrix = acc.snapshot()
    if identity_rows:
        matrix.diagonal().add_(identity_rows)
    return dict(H=matrix, count=acc.count + identity_rows, finalized=False, identity_rows=identity_rows)


def reconstruction(values, name, bits, shape, device):
    packed = {suffix: values[f"k{bits}.{name}.{suffix}"] for suffix in recipe.SUFFIXES}
    return numerical.reconstruct_projection(packed, list(shape), device=str(device),
                                            dtype=torch.bfloat16, codebook="mul1").float()


def encode(name, weight, covariance, bits, device, metrics):
    values = {}
    for width in bits:
        packed, measured, evidence = numerical.quantize_projection(
            weight, covariance, bits=width, device=str(device), codebook="mul1")
        values.update({f"k{width}.{name}.{key}": value for key, value in packed.items()})
        if width == 6:
            error = measured.get("hessian_weighted_relative_error")
            if (not isinstance(error, (float, int)) or not math.isfinite(error)
                    or not 0 <= error <= gates.BARS[6]):
                raise ValueError(f"{name} K6 training error {error} outside [0, {gates.BARS[6]}]")
        metrics.append(evidence | measured | dict(module=name, bits=width,
            effective_bits_per_weight=gates.packed_bpw(values, name, width, weight.shape)))
    return values


def weights_project(overrides):
    def project(name, inputs, weight):
        return native_projection(name, inputs, overrides.get(name, weight))
    return project


class Runner:
    def __init__(self, args):
        self.args, self.root = args, args.root.resolve()
        self.output = args.output.resolve()
        self.device = torch.device(args.device)
        self.root.mkdir(parents=True, exist_ok=True)
        self.header = HeaderSource(args.source)
        preflight = read_json(args.preflight)
        self.policy = dict(include_head=True, include_kv_b=True, k6_drafter_bits=6)
        self.identity = checkpoint_identity(self.root, identity(self.header, args.corpus, preflight))
        self.core = recipe.reviewed_core_modules(
            include_head=self.policy["include_head"], include_kv_b=self.policy["include_kv_b"])
        self.total = len(self.core) * 2 + recipe.EXPERT_MODULE_COUNT
        self.progress = Progress(self.root, self.total)
        self.stages = Stages(self.root / "stages", self.identity)
        self.writer = DirectMCWriter(self.output, self.header.snapshot, self.identity,
                                     self.header.as_inventory())
        self.records = [r["input_ids"].reshape(-1).cpu()
                        for r in load_token_batches(args.corpus, split="calibration")]
        if not self.records or any(not len(tokens) for tokens in self.records):
            raise ValueError("Hy4 needs nonempty causal calibration prompts")
        self.calibration_count = len(self.records)
        heldout = [r["input_ids"].reshape(-1).cpu()
                   for r in load_token_batches(args.corpus, split="heldout")]
        if not heldout or any(not len(tokens) for tokens in heldout):
            raise ValueError("Hy4 production gates need the disjoint agentic held-out split")
        self.records.extend(heldout)
        self.native_modules = set()
        self.expert_witness = None
        for path in (self.root / "gates").glob("*.json"):
            self.native_modules.update(row["module"] for row in read_json(path)["projections"]
                                       if row.get("kept_native"))
        self.reader = Reader(self.header.snapshot, self.device)
        self.model = Model(self.reader)
        self.embedding = self.reader.raw(reference.OUTER_NAMES[0])
        self.frontier_path = self.root / "frontier.json"
        self.frontier = self.load_frontier()
        self.progress.base.update(completed_target_layers=self.frontier["target"],
                                  completed_drafter_layers=self.frontier["drafter"],
                                  committed_encodes=self.frontier["encodes"])

    def load_frontier(self):
        if not self.frontier_path.exists():
            return dict(stage="initial", target=0, drafter=0, head=False, encodes=0)
        frontier = read_json(self.frontier_path)
        if frontier.pop("identity") != self.identity:
            raise ValueError("Hy4 frontier numerical identity differs")
        target, draft, head = (frontier[key] for key in ("target", "drafter", "head"))
        if (type(target) is not int or not 0 <= target <= recipe.LAYER_COUNT
                or type(draft) is not int or not 0 <= draft <= recipe.MTP_LAYER_COUNT
                or type(head) is not bool or (draft and not head)
                or (head and target != recipe.LAYER_COUNT)):
            raise ValueError("invalid Hy4 rolling frontier")
        expected = ["initial"] if target == 0 else [f"target-{target - 1:03d}-next"]
        if head:
            expected = [f"draft-{draft - 1:03d}-next"] if draft else ["draft-initial"]
        if frontier["stage"] not in expected or not self.stages.load(frontier["stage"]):
            raise ValueError("committed Hy4 successor stage is missing")
        return frontier

    def advance(self, *, completed, old_stages, **fields):
        self.frontier.update(fields)
        self.frontier["encodes"] += completed
        atomic_json(self.frontier_path, dict(self.frontier, identity=self.identity))
        self.progress.base.update(completed_target_layers=self.frontier["target"],
                                  completed_drafter_layers=self.frontier["drafter"],
                                  committed_encodes=self.frontier["encodes"])
        self.stages.retire(*old_stages)

    def rows(self, stage, *keys, start=0):
        return self.stages.records(stage, device=str(self.device), keys=keys or None, start=start)

    def save(self, name, batches):
        def checked(start):
            for index, values in batches(start):
                checks = [torch.isfinite(value).all() for value in values.values() if value.is_floating_point()]
                if checks and not bool(torch.stack(checks).all()):
                    raise ValueError("nonfinite Hy4 calibration stage")
                yield index, values
        self.stages.write_activations(name, checked, len(self.records), segment_bytes=SEGMENT_BYTES)

    def core_fit(self, stage, layer, groups, samples, overrides):
        """Fit one shared covariance per input boundary and replay serialized K8."""
        groups = [tuple(name for name in group if name in self.core) for group in groups]
        names = [name for group in groups for name in group]
        if not names:
            return None
        metrics = []
        gate = (layer.index in gates.TARGET_GATES or layer.prefix.startswith("model.dfly_layers.")
                or not layer.prefix)
        def fit():
            accumulators = [numerical.covariance_accumulator(
                self.core[group[0]]["in_features"], device=self.device) if group else None
                for group in groups]
            # One sample per forward record. Held-out tails observe the same
            # upstream K8 activations but never contribute to a covariance.
            witnesses = [[] for _ in groups]
            count = 0
            for count, sample in enumerate(samples(), 1):
                for acc, inputs, tails in zip(accumulators, sample, witnesses, strict=True):
                    if acc is not None:
                        if count <= self.calibration_count:
                            acc.add(inputs)
                        elif gate:
                            tails.append(inputs.reshape(-1, inputs.shape[-1])[-gates.TAIL:].clone())
            if count != len(self.records):
                raise ValueError(f"{stage} fit samples differ from the frozen prompt partition")
            ordinal = 0
            for group, acc, tails in zip(groups, accumulators, witnesses, strict=True):
                if acc is None:
                    continue
                covariance = raw(acc)
                for name in group:
                    self.progress.event("core_projection", stage=stage, projection=name)
                    weight = layer.weights[name.removeprefix(layer.prefix) + ".weight"]
                    values = encode(name, weight, covariance, (8, 6), self.device, metrics)
                    if gate:
                        inputs = torch.cat(tails)
                        pair = metrics[-2:]
                        for row in pair:
                            matrix = reconstruction(values, name, row["bits"], weight.shape, self.device)
                            gates.evaluated(row, inputs, weight, matrix, "last-16-rows-each-heldout-prompt")
                        draft = layer.prefix.startswith("model.dfly_layers.")
                        # A K6-only drafter miss triggers the whole-drafter K8 alias.
                        fallback = any(not row["heldout_passed"] for row in pair
                                       if not draft or row["bits"] == 8)
                        for row in pair:
                            row["kept_native"] = fallback
                    yield ordinal, values
                    ordinal += 1
        manifest = self.stages.write_batches(stage, fit(), {"names": names, "metrics": metrics})
        if manifest["metadata"]["names"] != names:
            raise ValueError("Hy4 compact scratch inventory differs")
        self.native_modules.update(row["module"] for row in manifest["metadata"]["metrics"]
                                   if row.get("kept_native"))
        for index, name in enumerate(names):
            if name in self.native_modules:
                continue
            geometry = self.core[name]
            overrides[name] = reconstruction(self.stages.read_batch(stage, index), name, 8,
                (geometry["out_features"], geometry["in_features"]), self.device)
        return stage

    @staticmethod
    def first_names(prefix):
        return (prefix + leaf for leaf in ("q_a_proj", "kv_a_proj_with_mqa", "linear_gate"))

    def compact_entries(self, stages, *, draft=False):
        entries = {6: {}, 8: {}}
        metrics = []
        for stage in filter(None, stages):
            manifest = self.stages.load(stage)
            metrics.extend(manifest["metadata"]["metrics"])
            for index, name in enumerate(manifest["metadata"]["names"]):
                if name in self.native_modules:
                    continue
                for bits in (8, 6):
                    if draft and bits == 6 and self.policy["k6_drafter_bits"] == 8:
                        continue
                    for suffix in recipe.SUFFIXES:
                        key = f"{name}.{suffix}"
                        entries[bits][key] = self.stages.reference(stage, index, f"k{bits}.{key}")
        return entries, metrics

    def save_metrics(self, label, metrics):
        directory = self.root / "metrics"
        directory.mkdir(exist_ok=True)
        path = directory / f"{label}.json"
        if path.exists():
            if read_json(path) != metrics:
                raise ValueError(f"committed Hy4 metrics changed for {label}")
        else:
            atomic_json(path, metrics)

    def selected_core(self):
        return {name: g for name, g in self.core.items() if name not in self.native_modules}

    def save_gate(self, label, metrics, tick, **fields):
        sizes = recipe.size_report(self.header, self.selected_core(),
                                   k6_drafter_bits=self.policy["k6_drafter_bits"])
        gates.gate_report(self.root, label, metrics, sizes, time.monotonic() - tick, **fields)
        self.progress.event("production_gate", snapshot=False, group=label,
                            receipt=str(self.root / "gates" / f"{label}.json"))

    def source_group(self, prefix, *, exclude=()):
        native = {name for name in self.header.specs if name.startswith(prefix)}
        selected = {recipe.source_tensor(name)[0] for name in self.core if name not in self.native_modules}
        original = native & selected
        native -= original | set(exclude) | recipe.ENGRAM_TABLES.keys()
        return self.header.source_entries(native), self.header.source_entries(original)

    def target(self, index):
        label = f"target-{index:03d}"
        tick = time.monotonic()
        incoming = self.frontier["stage"]
        successor = label + "-next"
        if self.writer.group_complete(label):
            self.recover_complete_group(label, successor, target=index + 1)
            return
        self.progress.base.update(current_layer=index, expert_projections=0,
                                  total_expert_projections=recipe.EXPERTS_PER_LAYER * 3 if index else 0)
        self.progress.event("target_layer")
        layer = self.model.load_layer(index)
        p, a = layer.prefix, layer.prefix + "self_attn."
        overrides = {}
        project = weights_project(overrides)
        capture = index in recipe.DFLY_TARGET_LAYERS
        moe = index > 0
        mlp = p + ("mlp.shared_experts." if moe else "mlp.")
        a_start = label + "-attention-input"
        mlp_stage = label + "-mlp"
        route_stage = label + "-routes"
        fit_stages = []

        def attention_input(start):
            for number, values in self.rows(incoming, start=start):
                tokens = self.records[number].to(self.device)
                residual, post, inputs = self.model.attention_start(values["state"], tokens, layer)
                yield number, dict(residual=residual, post=post, inputs=inputs)

        self.save(a_start, attention_input)
        fit_stages.append(self.core_fit(label + "-attn-first", layer, [self.first_names(a)],
            lambda: ((values["inputs"],) for _, values in self.rows(a_start, "inputs")), overrides))

        def attention_first(*keys, start=0):
            for number, values in self.rows(a_start, *keys, start=start):
                first = self.model.attention_first(values["inputs"], layer, project)
                yield number, values, first

        fit_stages.append(self.core_fit(label + "-attn-expand", layer,
            [(a + "q_b_proj",), (a + "kv_b_proj",)],
            lambda: ((first.normalized_q, first.kv) for _, _, first in attention_first("inputs")), overrides))

        def attention_output(*keys, start=0):
            for number, values, first in attention_first(*keys, start=start):
                output, tap = self.model.attention_output_inputs(
                    values["inputs"], first, layer, project, capture_tap=capture)
                yield number, values, output, tap

        fit_stages.append(self.core_fit(label + "-attn-out", layer, [(a + "o_proj",)],
            lambda: ((o,) for _, _, o, _ in attention_output("inputs")), overrides))

        def attention_finish(start):
            for number, values, output, tap in attention_output(start=start):
                attended = self.model.attention_finish(output, layer, project)
                state, post, inputs = self.model.mlp_start(values["residual"], values["post"],
                                                           attended, layer)
                result = dict(state=state, post=post, inputs=inputs)
                if tap is not None:
                    result.update(kv=tap.kv, rope_key=tap.rope_key, groups=tap.groups)
                yield number, result

        self.save(mlp_stage, attention_finish)
        if capture:
            tap_stage = f"tap-{index:03d}"
            self.save(tap_stage, lambda start: self.stages.records(mlp_stage, keys=("kv", "rope_key", "groups"), start=start))

        entry = [mlp + "gate_proj", mlp + "up_proj"]
        if moe:
            entry.append(p + "mlp.routed_expert_down_proj")
        fit_stages.append(self.core_fit(label + "-mlp-first", layer, [entry],
            lambda: ((values["inputs"],) for _, values in self.rows(mlp_stage, "inputs")), overrides))

        def activated(inputs):
            return F.silu(layer.project((mlp + "gate_proj").removeprefix(p), inputs, project)) * (
                layer.project((mlp + "up_proj").removeprefix(p), inputs, project))

        fit_stages.append(self.core_fit(label + "-mlp-down", layer, [(mlp + "down_proj",)],
            lambda: ((activated(values["inputs"]),) for _, values in self.rows(mlp_stage, "inputs")), overrides))
        packed, extra_metrics, expert_stages = {}, [], []
        combined = None
        if moe:
            def routes(start):
                for number, values in self.rows(mlp_stage, "inputs", start=start):
                    inputs = values["inputs"]
                    config = self.model.config
                    scores, choice = reference.routing_scores(inputs, layer.weights, config)
                    ids, gamma = reference.native_routes(scores, choice, config)
                    latent = layer.project("mlp.routed_expert_down_proj", inputs, project)
                    record = dict(latent=latent, ids=ids, gamma=gamma)
                    if number < self.calibration_count:
                        near, near_gamma = candidate_routes(scores, choice, ids, config)
                        record.update(candidate_ids=near, candidate_gamma=near_gamma)
                    yield number, record
            self.save(route_stage, routes)
            def route_batches(start, stop):
                for number in range(start, stop):
                    yield number - start, self.stages.read_batch(route_stage, number)
            held = self.calibration_count
            lengths = list(map(len, self.records))
            spool = Routes(route_batches(0, held), recipe.EXPERTS_PER_LAYER,
                           sum(lengths[:held]), device=self.device, recover=True)
            witness = Routes(route_batches(held, len(lengths)), recipe.EXPERTS_PER_LAYER,
                             sum(lengths[held:]), device=self.device)
            self.expert_witness = gates.ExpertWitness(witness, index, self.model.config.swiglu_limit)
            packed, extra_metrics, expert_stages, combined = self.experts(label, layer, spool)
            heldout_combined = self.expert_witness.combined
            del spool, witness
            self.expert_witness = None
            def routed_batches():
                yield from combined.split(lengths[:held])
                yield from heldout_combined.split(lengths[held:])
            fit_stages.append(self.core_fit(label + "-latent-up", layer,
                [(p + "mlp.routed_expert_up_proj",)],
                lambda: ((rows,) for rows in routed_batches()), overrides))

        def finish(start):
            routed = islice(routed_batches(), start, None) if moe else None
            for number, values in self.rows(mlp_stage, "state", "post", "inputs", start=start):
                result = self.model.mlp_finish(values["state"], values["post"], values["inputs"],
                    layer, next(routed) if moe else None, project)
                yield number, dict(state=result)

        self.save(successor, finish)
        views, metrics = self.compact_entries(fit_stages)
        metrics += extra_metrics
        self.save_metrics(label, metrics)
        if index in gates.TARGET_GATES or index in recipe.ENGRAM_ROWS:
            engram = gates.engram_coverage(self.model.config, layer, self.records[:self.calibration_count])
            if index in gates.TARGET_GATES:
                self.save_gate(label, metrics, tick, recovery=gates.recovery_distribution(metrics), engram=engram)
            elif engram:
                atomic_json(self.root / "metrics" / f"engram-coverage-{index:03d}.receipt", engram)
        fused = {p + leaf for leaf in reference.EXPERTS} if moe else ()
        native, original = self.source_group(p, exclude=fused)
        self.writer.write_group(label, shared=native | packed, original=original,
                                k8=views[8], k6=views[6])
        self.advance(completed=len(metrics), target=index + 1, stage=successor,
            old_stages=[incoming, a_start, mlp_stage, route_stage, *filter(None, fit_stages), *expert_stages])
        self.progress.event("target_layer_complete", current_layer=index,
                            layer_seconds=time.monotonic() - tick)
        self.reader.close()

    def experts(self, label, layer, spool):
        packed, errors, stages = {}, [], []
        combined = torch.zeros_like(spool.latent)
        for start in range(0, recipe.EXPERTS_PER_LAYER, EXPERT_GROUP):
            stage = f"{label}-expert-{start:03d}"
            metrics = []
            tick = time.monotonic()
            manifest = self.stages.load(stage)
            if manifest is None:
                def encode_group():
                    for expert in range(start, min(start + EXPERT_GROUP, recipe.EXPERTS_PER_LAYER)):
                        yield expert - start, self.expert(layer, spool, expert, metrics, combined)

                manifest = self.stages.write_batches(stage, encode_group(), {"metrics": metrics},
                                                      segment_bytes=SEGMENT_BYTES, segment_batches=EXPERT_GROUP)
            else:
                for entry in manifest["entries"]:
                    self.replay_expert(layer, spool, start + entry["index"], combined, stage, entry["index"])
            for entry in manifest["entries"]:
                ordinal = entry["index"]
                for name in self.expert_names(layer, start + ordinal):
                    for suffix in recipe.SUFFIXES:
                        tensor = f"{name}.{suffix}"
                        packed[tensor] = self.stages.reference(stage, ordinal, f"k4.{tensor}")
            errors.extend(manifest["metadata"]["metrics"])
            stages.append(stage)
            self.progress.base["expert_projections"] = len(errors)
            phase = "expert_group_complete" if metrics else "expert_group_replayed"
            self.progress.event(phase, current_layer=layer.index,
                                expert_group_seconds=time.monotonic() - tick,
                                new_expert_projections=len(metrics))
        return packed, errors, stages, combined

    @staticmethod
    def expert_names(layer, expert):
        prefix = f"{layer.prefix}mlp.experts.{expert}."
        return tuple(prefix + leaf for leaf in ("gate_proj", "up_proj", "down_proj"))

    def expert_activations(self, spool, rows, names, gate, up):
        limit = self.model.config.swiglu_limit
        for selection in rows.split(EXPERT_ROWS):
            chunk = spool.latent.index_select(0, selection)
            g = native_projection(names[0], chunk, gate).clamp_max(limit)
            u = native_projection(names[1], chunk, up).clamp(-limit, limit)
            yield F.silu(g) * u

    def accumulate_expert(self, combined, rows, gamma, name, down, activations):
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        for selection, coefficients, chunk in zip(
                rows.split(EXPERT_ROWS), gamma.split(EXPERT_ROWS), activations, strict=True):
            weighted = native_projection(name, chunk, down) * coefficients[:, None]
            finite.logical_and_(torch.isfinite(weighted).all())
            # Every expert visits a token at most once. Add in ascending expert
            # order, as the stock reference and the legacy CPU reduction do.
            combined.index_copy_(0, selection, combined.index_select(0, selection) + weighted)
        if not bool(finite):
            raise ValueError(f"nonfinite Hy4 expert output: {name}")

    def replay_expert(self, layer, spool, expert, combined, stage, ordinal):
        rows, gamma = spool.expert(expert)
        if not len(rows) and self.expert_witness is None:
            return
        rows, gamma = rows.to(self.device), gamma.to(self.device)
        names = self.expert_names(layer, expert)
        keys = tuple(f"k4.{name}.{suffix}" for name in names for suffix in recipe.SUFFIXES)
        # Select only packed weights, including when a frozen group still
        # contains the legacy unsummed output. Replaying needs no Hessian.
        values = self.stages.fields(stage, ordinal, keys)
        latent = spool.latent.shape[-1]
        intermediate = self.model.config.moe_intermediate_size
        gate, up, down = (reconstruction(values, name, 4, shape, self.device)
                          for name, shape in zip(names, ((intermediate, latent), (intermediate, latent),
                                                        (latent, intermediate)), strict=True))
        if self.expert_witness is not None:
            self.expert_witness.add(expert, (gate, up, down))
        if not len(rows):
            return
        activated = self.expert_activations(spool, rows, names, gate, up)
        self.accumulate_expert(combined, rows, gamma, names[2], down, activated)

    def expert(self, layer, spool, expert, metrics, combined):
        selected, coefficients, evidence = spool.calibration(expert)
        rows, gamma = selected.to(self.device), coefficients.to(self.device)
        natural = evidence["natural_rows"]
        identity = evidence["identity_rows"]
        metric_start = len(metrics)
        names = self.expert_names(layer, expert)
        weights = tuple(w.to(self.device) for w in self.reader.expert_weights(layer.index, expert))
        acc = numerical.covariance_accumulator(weights[0].shape[-1], device=self.device)
        if len(rows):
            for part in rows.split(EXPERT_ROWS):
                acc.add(spool.latent.index_select(0, part))
        covariance = raw(acc, identity)
        values = {}
        for name, weight in zip(names[:2], weights[:2], strict=True):
            values.update(encode(name, weight, covariance, (4,), self.device, metrics))
        gate, up = (reconstruction(values, name, 4, weight.shape, self.device)
                    for name, weight in zip(names[:2], weights[:2], strict=True))
        acc = numerical.covariance_accumulator(weights[2].shape[-1], device=self.device)
        if len(rows):
            activations = self.expert_activations(spool, rows, names, gate, up)
            # Gamma scales the linear down output in Hy4's native forward. In
            # its Hessian, scale the input by the same coefficient to measure
            # the error on that weighted output; gate/up observe latent inputs.
            for chunk, scale in zip(activations, gamma.split(EXPERT_ROWS), strict=True):
                acc.add(chunk * scale[:, None])
        values.update(encode(names[2], weights[2], raw(acc, identity),
                             (4,), self.device, metrics))
        for metric in metrics[metric_start:]:
            metric.update(evidence)
        down = reconstruction(values, names[2], 4, weights[2].shape, self.device)
        if natural:
            activated = self.expert_activations(spool, rows[:natural], names, gate, up)
            self.accumulate_expert(combined, rows[:natural], gamma[:natural], names[2], down, activated)
        if self.expert_witness is not None:
            self.expert_witness.add(expert, (gate, up, down), names=names, native=weights, metrics=metrics)
        return values

    def recover_complete_group(self, label, successor, **frontier):
        """A sealed output and successor authorize adopting a missing frontier."""
        if self.stages.load(successor) is None:
            raise ValueError("Hy4 sealed output has no committed successor; diagnose before reuse")
        metrics = read_json(self.root / "metrics" / f"{label}.json")
        # If a crash preceded this barrier, all former stages are still
        # available. Retire only after committing the adoption.
        discard = [path.name for path in self.stages.root.iterdir()
                   if path.name.startswith(label + "-") and path.name != successor]
        if "drafter" in frontier:
            discard = [name for name in discard if name not in
                       {f"{label}-{stage}" for stage in gates.DRAFT_STAGES}]
        discard.append(self.frontier["stage"])
        self.advance(completed=len(metrics), old_stages=discard, stage=successor, **frontier)
        self.progress.event("group_frontier_recovered", group=label)

    def head_and_draft_anchors(self):
        label = "head"
        incoming = self.frontier["stage"]
        starter = "draft-initial"
        if self.writer.group_complete(label):
            self.recover_complete_group(label, starter, head=True)
            return
        tick = time.monotonic()
        weights = {name: self.reader.load(name) for name in reference.OUTER_NAMES[1:5]}
        weights["lm_head.weight"] = self.reader.load("lm_head.weight")
        layer = Layer(recipe.LAYER_COUNT, "", weights)
        overrides = {}
        self.progress.event("head_calibration")
        fit = self.core_fit(label + "-fit", layer, [("lm_head",)],
            lambda: ((self.model.head_features(row["state"], weights),)
                     for _, row in self.rows(incoming)), overrides)
        aux = self.reader.load_draft_aux()
        drafter = Drafter(self.model,
            layers=[self.reader.load_draft_layer(index) for index in range(recipe.MTP_LAYER_COUNT)],
            aux=aux)

        def anchors(start):
            for number, row in self.rows(incoming, start=start):
                features = self.model.head_features(row["state"][-1:], weights)
                bonus = self.model.head(features, project=weights_project(overrides),
                                        weight=weights["lm_head.weight"]).argmax()
                state = drafter.start(bonus, features[0], self.embedding)
                result = dict(state=state)
                if number >= self.calibration_count:
                    result.update(native=state, k6=state, bonus=bonus.reshape(1))
                yield number, result

        self.save(starter, anchors)
        views, metrics = self.compact_entries([fit])
        self.save_metrics(label, metrics)
        if metrics:
            self.save_gate(label, metrics, tick)
        native, original = self.source_group("lm_head.")
        self.writer.write_group(label, shared=native, original=original, k8=views[8], k6=views[6])
        self.advance(completed=len(metrics), head=True, stage=starter,
                     old_stages=[incoming, *filter(None, [fit])])
        self.progress.event("head_complete")
        self.reader.close()

    def draft_taps(self, drafter, index, prompt):
        taps = {}
        for source in {drafter.kv_layers[index], drafter.index_layers[index]}:
            values = self.stages.read_batch(f"tap-{source:03d}", prompt, device=str(self.device))
            taps[source] = AttentionTap(values["kv"], values["rope_key"], values["groups"])
        return taps

    def draft(self, drafter, index):
        label = f"draft-{index:03d}"
        incoming = self.frontier["stage"]
        successor = label + "-next"
        if self.writer.group_complete(label):
            self.recover_complete_group(label, successor, drafter=index + 1)
            return
        self.progress.event("drafter_layer", current_drafter_layer=index)
        tick = time.monotonic()
        layer = drafter.layers[index]
        p = layer.prefix
        a, mlp = p + "self_attn.", p + "mlp."
        overrides = {}
        project = weights_project(overrides)

        def inputs(start=0):
            for number, values in self.rows(incoming, start=start):
                state = values["state"]
                yield number, state, drafter.attention_inputs(state, index)

        fit_stages = [self.core_fit(label + "-attn-first", layer, [self.first_names(a)],
            lambda: ((boundary,) for _, _, boundary in inputs()), overrides)]

        def first(start=0):
            for number, state, boundary in inputs(start):
                yield number, state, self.model.attention_first(boundary, layer, project)

        def expansions():
            ratio = drafter.config.compressed_indexer_compress_ratio
            for number, _, projected in first():
                kv = projected.kv
                if self.policy["include_kv_b"]:
                    taps = self.draft_taps(drafter, index, number)
                    end = len(self.records[number])
                    ids = recalled_context(taps[drafter.index_layers[index]], end, ratio)
                    kv = torch.cat((taps[drafter.kv_layers[index]].kv[ids], kv))
                yield projected.normalized_q, kv

        fit_stages.append(self.core_fit(label + "-attn-expand", layer,
            [(a + "q_b_proj",), (a + "kv_b_proj",)], expansions, overrides))

        def attention(start=0):
            for number, state, projected in first(start):
                taps = self.draft_taps(drafter, index, number)
                end = len(self.records[number])
                boundary = drafter.attention_output_inputs(projected, index, taps, end, project)
                yield number, state, boundary

        fit_stages.append(self.core_fit(label + "-attn-out", layer, [(a + "o_proj",)],
            lambda: ((out,) for _, _, out in attention()), overrides))

        mlp_stage = label + "-mlp"
        def after_attention(start):
            for number, state, out in attention(start):
                result = self.model.attention_finish(out, layer, project)
                residual, normalized = drafter.ffn_inputs(state, result, index)
                yield number, dict(residual=residual, inputs=normalized)

        self.save(mlp_stage, after_attention)
        fit_stages.append(self.core_fit(label + "-mlp-first", layer,
            [(mlp + "gate_proj", mlp + "up_proj")],
            lambda: ((values["inputs"],) for _, values in self.rows(mlp_stage, "inputs")), overrides))

        def down_inputs():
            for _, v in self.rows(mlp_stage, "inputs"):
                gate = layer.project("mlp.gate_proj", v["inputs"], project)
                up = layer.project("mlp.up_proj", v["inputs"], project)
                yield (F.silu(gate) * up,)

        fit_stages.append(self.core_fit(label + "-mlp-down", layer, [(mlp + "down_proj",)],
                                       down_inputs, overrides))

        def finish(start):
            alternatives = {}
            for stage in filter(None, fit_stages):
                for order, name in enumerate(self.stages.load(stage)["metadata"]["names"]):
                    if name in self.native_modules:
                        continue
                    weight = layer.weights[name.removeprefix(layer.prefix) + ".weight"]
                    alternatives[name] = reconstruction(self.stages.read_batch(stage, order), name, 6,
                                                        weight.shape, self.device)
            project_k6 = weights_project(alternatives)
            for number, v in self.rows(mlp_stage, start=start):
                result = self.model.ffn_parts(v["inputs"], layer, "mlp.", project)
                record = dict(state=v["residual"] + result)
                if number >= self.calibration_count:
                    prior = self.stages.fields(incoming, number, ("native", "k6", "bonus"),
                                               device=self.device)
                    taps = self.draft_taps(drafter, index, number)
                    end = len(self.records[number])
                    record.update(native=drafter.layer(prior["native"], index, taps, end),
                                  k6=drafter.layer(prior["k6"], index, taps, end, project_k6),
                                  bonus=prior["bonus"])
                yield number, record

        self.save(successor, finish)
        views, metrics = self.compact_entries(fit_stages, draft=True)
        self.save_metrics(label, metrics)
        extra = {"hidden_correction": "retained native", "hidden_correction_parameters":
                 sum(t.numel() for key, t in drafter.aux.items() if key.startswith("hidden_correction."))}
        if index == recipe.MTP_LAYER_COUNT - 1:
            head = self.reader.load("lm_head.weight")
            matches = torch.zeros(2, dtype=torch.int64, device=self.device)
            count = 0
            for number in range(self.calibration_count, len(self.records)):
                saved = self.stages.read_batch(successor, number, device=self.device)
                bonus = saved["bonus"].reshape(())
                def proposals(state):
                    return drafter.proposals(drafter.finish(state), bonus, self.embedding,
                                             head_weight=head).argmax(-1)
                expected = proposals(saved["native"])
                matches += torch.stack([(proposals(saved[key]) == expected).sum() for key in ("state", "k6")])
                count += expected.numel()
            agreement = {str(bits): hits/count for bits, hits in zip((8, 6), matches.tolist(), strict=True)}
            extra.update(dfly_top1_agreement=agreement, compared_slots=count,
                         comparison="native drafter on identical K8 target anchors, taps and common native head")
            if agreement["8"] < .98:
                self.save_gate(label, metrics, tick, extra=extra | {"status": "failed-dfly-k8-top1"})
                raise ValueError(f"DFly2 K8 held-out top-1 agreement {agreement['8']} below 0.98")
            failed_k6 = any(row["bits"] == 6 and not row["heldout_passed"]
                            for n in range(recipe.MTP_LAYER_COUNT)
                            for row in read_json(self.root / "metrics" / f"draft-{n:03d}.json"))
            extra["published_k6_drafter_bits"] = 8 if failed_k6 or agreement["6"] < .98 else 6
            self.policy["k6_drafter_bits"] = extra["published_k6_drafter_bits"]
        self.save_gate(label, metrics, tick, extra=extra)
        native, original = self.source_group(p)
        # K6 drafter publication depends on the final full-drafter top-1 gate.
        # Keep only its small packed fit stages until that decision is durable.
        self.writer.write_group(label, shared=native, original=original, k8=views[8])
        self.advance(completed=len(metrics), drafter=index + 1, stage=successor,
                     old_stages=[incoming, mlp_stage])
        self.progress.event("drafter_layer_complete", current_drafter_layer=index,
                            layer_seconds=time.monotonic() - tick)

    def seal_draft_k6(self):
        final = read_json(self.root / "gates" / f"draft-{recipe.MTP_LAYER_COUNT-1:03d}.json")
        self.policy["k6_drafter_bits"] = final["published_k6_drafter_bits"]
        for index in range(recipe.MTP_LAYER_COUNT):
            label = f"draft-{index:03d}"
            fit_stages = [f"{label}-{suffix}" for suffix in gates.DRAFT_STAGES]
            group = label + "-k6"
            selected = any(name.startswith(f"model.dfly_layers.{index}.") for name in self.selected_core())
            if selected and self.policy["k6_drafter_bits"] == 6 and not self.writer.group_complete(group):
                views, _ = self.compact_entries(fit_stages, draft=True)
                if not views[6]:
                    raise ValueError(f"{group} missing selected K6 drafter modules")
                self.writer.write_group(group, k6=views[6])
            self.stages.retire(*fit_stages)

    def globals(self):
        if self.writer.group_complete("globals"):
            return
        native = {name for name in self.header.specs
                  if not name.startswith(("model.layers.", "lm_head."))
                  and not any(name.startswith(f"model.dfly_layers.{index}.")
                              for index in range(recipe.MTP_LAYER_COUNT))}
        self.writer.write_group("globals", shared=self.header.source_entries(native))

    def write_error_csv(self):
        rows = [row for path in sorted((self.root / "metrics").glob("*.json"))
                for row in read_json(path)]
        expected = {(module, bits) for module in self.core for bits in (8, 6)}
        expected.update((module, 4) for module in recipe.reviewed_expert_modules())
        if len(rows) != len(expected) or {(row["module"], row["bits"]) for row in rows} != expected:
            raise ValueError("Hy4 calibration metrics do not cover every expected encode")
        names = ["module", "bits"] + sorted(set().union(*(row.keys() for row in rows)) - {"module", "bits"})
        path = self.root / "per-module-errors.csv"
        temporary = path.with_suffix(".csv.partial")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, names)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(self.root)
        return path

    def export_engram(self):
        self.progress.event("engram_export")
        receipt = read_json(self.args.engram_receipt)
        source = self.args.engram_dir.resolve()
        if source.stat().st_dev != self.output.stat().st_dev:
            raise ValueError("Hy4 Engram and MC output need the same filesystem for no-copy adoption")
        tables = receipt["tables"]
        expected = set(tables.values()) | {part["file"] for layer in receipt["manifest"]["layers"]
                                           for part in layer["parts"]}
        if tables.keys() != recipe.ENGRAM_TABLES.keys() or receipt["files"].keys() != expected:
            raise ValueError("Hy4 Engram receipt has an unexpected file inventory")
        for name, proof in receipt["files"].items():
            if Path(name).name != name or not name.startswith("engram-") or not name.endswith(".safetensors"):
                raise ValueError("invalid Hy4 Engram file receipt")
            before = source / name
            after = self.output / name
            if before.is_symlink() or after.is_symlink():
                raise ValueError("Hy4 Engram adoption requires regular files")
            verify_receipt(before, proof)
            if not after.exists():
                os.link(before, after)
            verify_receipt(after, proof)
        fsync_directory(self.output)
        return receipt

    def run(self):
        reference.configure_f32()
        self.progress.event("global_tensors")
        self.globals()
        if not self.frontier["target"]:
            self.progress.event("initial_activations")
            self.save("initial", lambda start: (
                (i, {"state": self.model.initial(self.records[i].to(self.device), self.embedding)})
                for i in range(start, len(self.records))))
        for layer in range(self.frontier["target"], recipe.LAYER_COUNT):
            self.target(layer)
        if not self.frontier["head"]:
            self.head_and_draft_anchors()
        if self.frontier["drafter"] < recipe.MTP_LAYER_COUNT:
            drafter = Drafter(self.model)
            for index in range(self.frontier["drafter"], recipe.MTP_LAYER_COUNT):
                self.draft(drafter, index)
            del drafter
        self.reader.close()
        if self.frontier["encodes"] != self.total:
            raise ValueError("Hy4 completed encode count differs from the production recipe")
        self.seal_draft_k6()
        error_csv = self.write_error_csv()
        tables = self.export_engram()
        self.progress.event("package_audit")
        return seal_package(self.root, self.writer, self.identity, self.header.document,
                            self.selected_core(), self.native_modules, tables,
                            self.policy["k6_drafter_bits"], error_csv, self.progress.base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "preflight", "root", "output", "engram-dir", "engram-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    complete = args.root / "numerical-complete.json"
    if complete.exists():
        receipt = read_json(complete)
        package = args.output.resolve() / "package-validation.json"
        if (receipt["status"] != "passed" or receipt["output"] != str(args.output.resolve())
                or receipt["identity"] != read_json(args.root / "identity.json")
                or receipt["package_validation"] != str(package)
                or read_json(package)["status"] != "validated-headers-and-byte-exactness"):
            raise ValueError("Hy4 completion receipt disagrees with requested output")
        print(json.dumps(receipt), flush=True)
        return
    runner = None
    try:
        with torch.inference_mode():
            runner = Runner(args)
            result = runner.run()
        print(json.dumps(result), flush=True)
    except Exception as error:
        path = args.root / "status.json"
        if path.exists():
            status = read_json(path)
        else:
            status = {}
        if args.root.is_dir():
            atomic_json(path, status | dict(status="blocked",
                error=f"{type(error).__name__}: {error}",
                updated_at=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        if runner is not None:
            runner.reader.close()


if __name__ == "__main__":
    main()
