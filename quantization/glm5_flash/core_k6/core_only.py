"""Calibrated target-core producer; no experts, MTP or native tensors exported."""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import torch

from kimi_k3.uniform_k2.storage import atomic_json, canonical, sha256_file
from . import recipe
from .run import (Engine, Progress, RECIPE, REPO, GIB, cleanup_complete,
                  compact, exclusive_run)
from .storage import CoreProjectionStore


def core_selection(bits):
    if type(bits) is not int or bits not in (6, 8):
        raise ValueError("core-only GLM Flash requires K6 or K8")
    # The eight MTP matrices are shared K4, not alternate target-core views.
    return recipe.reviewed_core_modules(include_mtp=False, core_bits=bits)


class CoreOnlyEngine(Engine):
    def checkpoints(self, layer):
        return CoreProjectionStore(
            self.root / ("head/projections" if layer == "head" else f"layers/{layer:03d}/projections"),
            self.evidence_root / ("head" if layer == "head" else f"layer-{layer:03d}"),
            self.identity, self.selection, self.progress, self.math, self.device)

    def _experts(self, block, layer, store, checkpoints):
        # Native routed weights must participate in every successor frontier.
        # An empty quantized expert selection must never install empty tensors.
        self.progress.emit("source_experts_load_started", layer=layer, phase="native-experts")
        self.source.materialize_routed_experts(block, layer=layer, device=self.device)
        self.progress.emit("source_experts_loaded", layer=layer, quantized=False)

    def assemble_layer(self, layer, replacements):
        from core_export import assemble_core_shard
        selection = {name: self.selection[name] for name in replacements}
        return assemble_core_shard(replacements, selection, self.export,
            f"core-layer-{layer:03d}.safetensors", self.identity)


def make_core_plan(source, tokens, root, export, *, core_bits):
    summary = recipe.inventory(source)
    selected = core_selection(core_bits)
    manifest = json.loads((tokens / "manifest.json").read_text())
    if (source.snapshot.name != recipe.BASE_REVISION
            or manifest["splits"]["calibration"]["count"] != 1426):
        raise ValueError("production requires the pinned full BF16 source and full GLM calibration stream")
    if root == export or root in export.parents or export in root.parents:
        raise ValueError("core-only run and export directories must be disjoint")
    numerical_recipe = deepcopy(RECIPE)
    numerical_recipe.update(core_bits=core_bits, expert_bits=None, mtp_bits=None,
        experts="native-BF16-propagation-only", mtp="excluded-shared-K4-drafter",
        ordering="interleaved-serialized-core-and-native-experts-before-successor-layer")
    code = [*(Path(__file__).with_name(name) for name in
              ("core_only.py", "run.py", "storage.py", "recipe.py")),
        REPO / "quantization/core_export.py",
        REPO / "quantization/glm5_flash/prepare_corpus.py",
        REPO / "quantization/glm5/prepare_corpus.py",
        REPO / "quantization/kimi_k3/prepare_corpus.py",
        REPO / "quantization/glm5/core_k6/storage.py",
        REPO / "quantization/kimi_k3/core_k6/run.py",
        REPO / "quantization/kimi_k3/core_k6/storage.py",
        REPO / "quantization/kimi_k3/uniform_k2/storage.py"]
    fork = REPO / "third_party/gptqmodel"
    code += [fork / "gptqmodel/utils" / name for name in
             ("glm5_core.py", "glm5_flash_core.py", "kimi_k3_core_format.py")]
    identity = dict(schema="cuteafd-glm53-flash-core-only-v1",
        source_repo=recipe.BASE_REPO, source_revision=recipe.BASE_REVISION,
        source_config_sha256=sha256_file(source.snapshot / "config.json"),
        source_index_sha256=sha256_file(source.snapshot / "model.safetensors.index.json"),
        token_manifest_sha256=sha256_file(tokens / "manifest.json"), recipe=numerical_recipe,
        calibration=dict(samples=manifest["splits"]["calibration"]["count"],
                         tokens=manifest["splits"]["calibration"]["tokens"]),
        code={str(path.relative_to(REPO)): sha256_file(path) for path in code},
        gptqmodel_commit=subprocess.check_output(
            ["git", "-C", str(fork), "rev-parse", "HEAD"], text=True).strip(),
        sparkinfer_commit=subprocess.check_output(
            ["git", "-C", str(REPO / "third_party/sparkinfer"), "rev-parse", "HEAD"], text=True).strip(),
        packages={name: importlib.metadata.version(name) for name in
                  ("torch", "transformers", "numpy", "safetensors")})
    payload = sum(recipe.packed_bytes(g) for g in selected.values())
    selected_source = sum(g["in_features"] * g["out_features"] * 2 for g in selected.values())
    summary.update(selected_modules=len(selected), selected_source_bytes=selected_source,
        selected_packed_bytes=payload, target_payload_bytes=payload,
        target_tensors=len(selected) * 4, preserved_bytes=0, preserved_tensors=0,
        parts=dict(base_core=dict(modules=len(selected), source_bytes=selected_source,
                                  packed_bytes=payload)))
    return dict(identity=identity, source=str(source.snapshot), tokens=str(tokens),
        root=str(root), export=str(export), inventory=summary,
        temporary_reserve_bytes=192 * GIB,
        required_free_bytes=payload + (192 + 64) * GIB,
        publication_repo=None)


def verify_core_evidence(selection, root, identity):
    for name, g in selection.items():
        layer = "head" if name == "lm_head" else f"layer-{int(name.split('.layers.')[1].split('.')[0]):03d}"
        path = root / "evidence" / layer / (name + ".json")
        saved = json.loads(path.read_text())
        evidence = saved.get("evidence", {})
        oracle = evidence.get("packed_byte_oracle", {})
        reconstruction = evidence.get("reconstruction", {})
        if (saved.get("identity") != identity or saved.get("module") != name
                or saved.get("bits") != g["bits"]
                or saved.get("logical_shape") != [g["out_features"], g["in_features"]]
                or oracle.get("passed") is not True or oracle.get("bits") != g["bits"]
                or reconstruction.get("finite") is not True
                or reconstruction.get("logical_shape") != saved["logical_shape"]
                or evidence.get("identity_rows") != 0
                or evidence.get("hessian_rows") != identity["calibration"]["tokens"]):
            raise ValueError(f"incomplete calibrated core reconstruction/oracle evidence: {name}")


def run_core_only(args):
    from gptqmodel.utils.glm5_flash_core import GlmFlashBF16Source
    from core_export import assemble_core_shard, finalize_core_export

    source = GlmFlashBF16Source(args.source_snapshot)
    try:
        args.tokens, args.root, args.export = (
            path.expanduser().resolve() for path in (args.tokens, args.root, args.export))
        plan = make_core_plan(source, args.tokens, args.root, args.export, core_bits=args.core_bits)
        args.root.mkdir(parents=True, exist_ok=True)
        prior = args.root / "plan.json"
        if prior.exists() and json.loads(prior.read_text()) != plan:
            raise ValueError("immutable production plan differs")
        if args.preflight:
            if not prior.exists() and shutil.disk_usage(args.root).free < plan["required_free_bytes"]:
                raise OSError("not enough root NVMe space for core output and bounded recovery reserve")
            print(canonical(plan), flush=True)
            return
        if prior.exists() and not args.resume:
            raise ValueError("existing production run requires explicit --resume")
        if not prior.exists() and args.resume:
            raise ValueError("recovery requires a committed plan")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # The 96 GiB unified-memory container reserves 32 GiB for the host.
        # Legacy full export retains its independent 86 GiB allocator limit.
        torch.cuda.set_per_process_memory_fraction(
            min(1., 64 * GIB / torch.cuda.get_device_properties(0).total_memory))
        with torch.inference_mode(), exclusive_run(args.root):
            if not prior.exists():
                if args.export.exists() and any(args.export.iterdir()):
                    raise ValueError("core-only export must start empty")
                atomic_json(prior, plan)
            progress = Progress(args.root,host_limit_gib=32,gpu_limit_gib=64)
            progress.state["total_projections"] = len(core_selection(args.core_bits))
            if (args.root / "quantization-complete.json").exists():
                completed = json.loads((args.root / "quantization-complete.json").read_text())
                if completed["identity"] != plan["identity"]:
                    raise ValueError("completed core producer identity differs")
                verify_core_evidence(core_selection(args.core_bits), args.root, plan["identity"])
                finalize_core_export(core_selection(args.core_bits), args.export, plan["identity"])
                cleanup_complete(args.root)
                progress.state["status"] = "quantized"
                progress.emit("quantization_complete")
                return
            from glm5_flash.prepare_corpus import load_token_records
            records = load_token_records(args.tokens, "calibration")
            calibration = plan["identity"]["calibration"]
            if (len(records) != calibration["samples"]
                    or sum(len(row["input_ids"]) for row in records) != calibration["tokens"]):
                raise ValueError("calibration records differ from full production manifest")
            engine = CoreOnlyEngine(source, records, args.root, args.export, plan["identity"],
                core_selection(args.core_bits), progress)
            current = engine.initial()
            for layer in range(current["layer"] + 1, recipe.TARGET_LAYER_COUNT):
                current = engine.layer(layer, current)
            head = engine.head(current)
            # The head is not replayed by a successor, but it still needs a
            # serialized reconstruction/finite scalar witness like every layer.
            dense = head.dense("lm_head")
            del dense
            compact()
            assemble_core_shard(head.replacements(["lm_head"]),
                {"lm_head": engine.selection["lm_head"]}, args.export,
                "core-head.safetensors", engine.identity)
            verify_core_evidence(engine.selection, args.root, engine.identity)
            summary = finalize_core_export(engine.selection, args.export, engine.identity)
            atomic_json(args.root / "quantization-complete.json",
                dict(identity=engine.identity, summary=summary))
            cleanup_complete(args.root)
            progress.state["status"] = "quantized"
            progress.emit("quantization_complete", **summary)
    finally:
        source.close()
