"""End-to-end Hy4-flash QMLA calibration driver (reference MLA pipeline to copy).

Supports both short pilot runs (``--pilot`` or explicit ``--layers``, ``--modes``,
``--train-sequences``, ``--val-sequences``, ``--test-sequences``, ``--steps``) and full
12-DSA-layer calibration. Both fit every width (content c2/c4/c8, RoPE r4/r8) and
measure all nine combinations by default.

Splits: ``train`` fits, ``val`` selects fit steps and the shipped transform, ``test``
produces every reported attention error. The sidecar ships ONE content and ONE RoPE
transform per layer, chosen on ``val`` across every combination; the report also keeps
the per-width fits' test errors and the full candidate cross-table for analysis.

Corpus contract: ``quantization/panel_corpus.py`` owns the panel pin. The default is
``wrldsuksgo2mars/agentic-quant-panel-v3`` with all three components (``agentic``,
``exl3_standard``, ``long_context``); a full run requires all three, a pilot may pass a
subset with ``--components``. ``--corpus-dir`` is a ``panel_corpus.prepare_panel_corpus``
output (windowed ``long_context``, 2,048-token windows), loaded and verified through
``panel_corpus.load_component_records``. ``--long-eval`` renders full-length 16K/32K
``long_context`` records from the pinned panel snapshot in the local HF cache.

Protection default (Correction 2): ``--protection sink4_local128`` is enabled by
default on full-history MLA layers only (Hy4-flash's 30 SWA-512 layers are untouched).

Outputs (Correction 3 — no separate ``mla_calibration.json``):
- ``<output>/sidecar/mla_calibration.safetensors`` (self-describing with embedded
  ``__metadata__["cuteafd_qmla"]``, ``artifact_status="calibrated_unqualified"``)
- ``<output>/qmla-validation-report.json``
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmla.capture import CapturedLayerBank, SPLIT_NAMES, load_layer_capture, save_layer_capture, split_train_val_test
from qmla.fit import FitConfig
from qmla.long_eval import LongContextEvaluator, LongContextRecord, load_long_context_records, panel_snapshot_dir
from qmla.quantizer import parse_qmla_mode
from qmla.sidecar import load_sidecar_transforms

from . import forward as hy4_forward
from . import prepare_corpus as hy4_corpus
from . import recipe as hy4_recipe
from .qmla_capture import Hy4FlashCaptureAdapter
from .qmla_mc import MCModel, MCReader

DEFAULT_PILOT_LAYERS = (0, 21, 41)
DEFAULT_FULL_MODES = ("8:8", "4:8", "4:4", "2:4", "8:0", "4:0", "2:0", "0:8", "0:4")
# The pilot is for understanding: it fits every width and measures every combination.
DEFAULT_PILOT_MODES = DEFAULT_FULL_MODES


from qmla.pipeline import (
    _git_head_sha, attach_long_context_eval, build_run_parser, calibrate_from_banks, FULL_SPLITS, PILOT_SPLITS,
    check_reused_captures, corpus_split_description, load_pinned_qmla_corpus,
    parse_components_arg, parse_layers_arg, parse_modes_arg, required_profile_bits,
    resolve_settings, split_identity,
)

calibrate_hy4_flash_from_banks = calibrate_from_banks


def build_parser() -> argparse.ArgumentParser:
    return build_run_parser(__doc__, target_repo=hy4_recipe.TARGET_REPO,
                            base_repo=hy4_recipe.BASE_REPO, base_revision=hy4_recipe.BASE_REVISION)


def resolve_run_settings(args, eligible_layers):
    return resolve_settings(args, eligible_layers, pilot_layers=DEFAULT_PILOT_LAYERS,
                            pilot_modes=DEFAULT_PILOT_MODES, full_modes=DEFAULT_FULL_MODES)


def _routed_forward(model, layer, inputs):
    """Read each expert once for all sequences; keep the reference's expert-ID sum order."""
    latent, ids, gamma = model.moe_route(torch.cat(inputs), layer)
    routed = torch.zeros_like(latent)
    for expert in ids.unique(sorted=True).tolist():
        rows, weights = model.expert_rows(ids, gamma, expert)
        routed[rows] += model.expert(expert, latent[rows], layer) * weights[:, None]
    return routed.split([len(x) for x in inputs])


def _stream_hy4_flash_captures(
    reader: hy4_forward.Reader,
    adapter: Hy4FlashCaptureAdapter,
    target_layers: Sequence[int],
    token_records: Sequence[tuple[str, torch.Tensor]],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    capture_dir: Path,
    provenance: dict[str, Any],
) -> dict[int, CapturedLayerBank]:
    """Run one frozen-weight layer-major forward up to max(target_layers) and capture target MLA layers."""
    model = adapter.model
    max_layer = max(target_layers)
    target_set = set(target_layers)
    embedding = reader.raw(hy4_forward.reference.OUTER_NAMES[0])

    provenance = {**provenance, "splits": corpus_split_description(
        token_records, train_count=train_count, val_count=val_count, test_count=test_count, seed=seed
    )}
    train_items, val_items, test_items = split_train_val_test(
        token_records, train_count=train_count, val_count=val_count, test_count=test_count,
        key_fn=lambda r: r[0], seed=seed,
    )
    active = [*train_items, *val_items, *test_items]
    active = [(sid, tokens.to(reader.device)) for sid, tokens in active]
    states = deque(model.initial(tokens, embedding=embedding) for _, tokens in active)
    banks = {}
    for layer_idx in range(max_layer + 1):
        started = time.monotonic()
        layer = model.load_layer(layer_idx)
        captures, mlp_args = [], deque()
        for seq_id, tokens in active:
            residual, attention_post, attn_in = model.attention_start(states.popleft(), tokens, layer)
            if layer_idx in target_set:
                cap, attn_out, _ = adapter.capture_sequence(attn_in, layer, seq_id=seq_id)
                captures.append(cap)
            else:
                attn_out, _ = model.attention(attn_in, layer)
            if layer_idx < max_layer:
                mlp_args.append(model.mlp_start(residual, attention_post, attn_out, layer))
            del residual, attention_post, attn_in, attn_out

        if layer_idx in target_set:
            bank = CapturedLayerBank(adapter.geometry_for(layer_idx), adapter.layer_params(layer),
                                     tuple(captures[:train_count]),
                                     tuple(captures[train_count:train_count + val_count]),
                                     tuple(captures[train_count + val_count:]), provenance)
            save_layer_capture(capture_dir, bank)
            del bank, captures, cap
            banks[layer_idx] = load_layer_capture(capture_dir, layer_idx)
        if mlp_args:
            routed = [None] * len(mlp_args)
            if model.config.mlp_layer_types[layer_idx] == "sparse":
                routed = _routed_forward(model, layer, [args[2] for args in mlp_args])
            for result in routed:
                states.append(model.mlp_finish(*mlp_args.popleft(), layer, result))
            del routed, result
        print(json.dumps({"phase": "capture", "completed_layer": layer_idx,
                          "seconds": time.monotonic() - started}), flush=True)
        del layer
    return banks


@torch.no_grad()
def run_hy4_flash_long_eval(
    adapter: Hy4FlashCaptureAdapter,
    *,
    sidecar_path: Path | str,
    records: Sequence[LongContextRecord],
    corpus_meta: Mapping[str, Any],
    device: torch.device | str = "cpu",
    layers: Sequence[int] | None = None,
    modes: Sequence[tuple[int, int]] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    embedding: torch.Tensor | None = None,
    phase: str = "long_eval",
) -> dict[str, Any]:
    """Forward-only streaming long-context (16K/32K) evaluation against a shipped QMLA sidecar.

    Streams one layer at a time matching ``_stream_hy4_flash_captures``, capturing compact DSA
    group IDs (never ``[T, T]`` bool masks), evaluating shipped / RTN / Hadamard per sequence
    at each target layer, and freeing every capture immediately after scoring.
    """
    if not records:
        raise ValueError("run_hy4_flash_long_eval requires at least one LongContextRecord")
    started = time.monotonic()
    dev = torch.device(device)
    model = adapter.model
    manifest, content_by_layer, rope_by_layer = load_sidecar_transforms(
        sidecar_path, expected_source=expected_source, strict=True,
    )
    sidecar_layers = sorted(set(content_by_layer) | set(rope_by_layer))
    target_layers = tuple(sorted(sidecar_layers if layers is None else layers))
    missing = [lid for lid in target_layers if lid not in set(sidecar_layers)]
    if missing:
        raise ValueError(f"sidecar is missing requested long-eval layers: {missing}")

    if modes is None:
        eval_modes = tuple(parse_qmla_mode(m) for m in manifest["supported_modes"])
    else:
        supported = set(manifest["supported_modes"])
        eval_modes = tuple(modes)
        unsupported = [f"{c}:{r}" for c, r in eval_modes if f"{c}:{r}" not in supported]
        if unsupported:
            raise ValueError(f"sidecar does not support requested long-eval modes: {unsupported}")

    evaluator = LongContextEvaluator(
        modes=eval_modes,
        content_transforms=content_by_layer,
        rope_transforms=rope_by_layer,
        protection_policy=manifest["protection_policy"],
        dsa_indexer_note=adapter.dsa_indexer_note,
    )

    max_layer = max(target_layers)
    target_set = set(target_layers)
    if embedding is None:
        embedding = model.reader.raw(hy4_forward.reference.OUTER_NAMES[0])
    active = [(rec, rec.input_ids.to(dev)) for rec in records]
    states = deque(model.initial(tokens, embedding=embedding) for _, tokens in active)

    for layer_idx in range(max_layer + 1):
        layer_start = time.monotonic()
        layer = model.load_layer(layer_idx)
        is_target = layer_idx in target_set
        params = adapter.layer_params(layer) if is_target else None
        geom = adapter.geometry_for(layer_idx) if is_target else None
        # Replace each HC state as it advances. The full long-context panel carries
        # about 15 GiB of HC state per generation on Hy4-flash.
        mlp_args = deque()
        for rec, tokens in active:
            residual, attention_post, attn_in = model.attention_start(states.popleft(), tokens, layer)
            if is_target:
                assert params is not None and geom is not None
                cap, attn_out, _ = adapter.capture_sequence(
                    attn_in, layer, seq_id=rec.seq_id, compact_mask=True, capture_device=dev,
                )
                evaluator.evaluate_layer_sequence(cap, params, geom, split=rec.split, device=dev)
                del cap
            else:
                attn_out, _ = model.attention(attn_in, layer)
            if layer_idx < max_layer:
                mlp_args.append(model.mlp_start(residual, attention_post, attn_out, layer))
            del residual, attention_post, attn_in, attn_out
        if mlp_args:
            routed = [None] * len(mlp_args)
            if model.config.mlp_layer_types[layer_idx] == "sparse":
                routed = _routed_forward(model, layer, [args[2] for args in mlp_args])
            for result in routed:
                states.append(model.mlp_finish(*mlp_args.popleft(), layer, result))
            del routed, result
        evaluator.release_layer_cache()
        print(
            json.dumps({"phase": phase, "completed_layer": layer_idx, "evaluated": is_target,
                        "seconds": time.monotonic() - layer_start}),
            flush=True,
        )
        del layer

    result = evaluator.finalize(
        corpus_meta=corpus_meta,
        transform_fingerprint=manifest["transform_fingerprint"],
        wall_seconds=time.monotonic() - started,
        device=dev,
    )
    view = getattr(model.reader, "variant", {"name": "original"})
    result["reference"] = f"{view['name']} 0:0"
    return result


LONG_EVAL_CHAT = {k: hy4_corpus.CHAT[k] for k in ("reasoning_effort", "date_string")}


def load_long_eval_records(checkpoint: Path, panel_revision: str | None):
    """Full-length v3 ``long_context`` records (heldout, then calibration) for ``--long-eval``."""
    return load_long_context_records(panel_snapshot_dir(panel_revision), checkpoint_dir=checkpoint,
                                     chat_kwargs=LONG_EVAL_CHAT, panel_revision=panel_revision)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    code_commit = _git_head_sha(Path(__file__).resolve().parents[2])
    if args.sidecar is not None and not args.long_eval:
        parser.error("--sidecar requires --long-eval")

    checkpoint = args.checkpoint.resolve()
    config_bytes = (checkpoint / "config.json").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    forward_identity = {
        "target_revision": args.source_revision,
        "forward_revision": args.source_revision,
        "forward_config_sha256": config_sha256,
        "measurement_reference": f"{args.view} 0:0",
    }
    if args.checkpoint_kind == "mc":
        reader = MCReader(checkpoint, args.device, view=args.view)
        if (checkpoint.name != args.source_revision
                or reader.source != {"repo": args.base_repo, "revision": args.base_revision}):
            raise ValueError("MC snapshot or upstream source identity differs from the calibration request")
        model = MCModel(reader)
        forward_identity["forward_view_fingerprint"] = reader.variant["fingerprint"]
    else:
        reader = hy4_forward.Reader(checkpoint, device=args.device)
        model = hy4_forward.Model(reader)
    adapter = Hy4FlashCaptureAdapter(model, include_swa=False)
    eligible = tuple(g.layer_id for g in adapter.mla_geometries(include_swa=False))

    # Standalone --long-eval against an existing --sidecar (no refitting).
    if args.sidecar is not None:
        sidecar_manifest, _, _ = load_sidecar_transforms(args.sidecar, strict=True)
        if (args.checkpoint_kind == "mc" and sidecar_manifest["calibration_provenance"].get("forward_view_fingerprint")
                != forward_identity["forward_view_fingerprint"]):
            raise ValueError("sidecar was not fitted on this MC original view")
        expected_source = {
            "repo": args.source_repo,
            "revision": args.source_revision,
            "family": "hy4_flash",
            "config_sha256": config_sha256,
        }
        long_layers = parse_layers_arg(args.layers, eligible) if args.layers else None
        long_modes = parse_modes_arg(args.modes) if args.modes else None
        long_records, long_corpus_meta = load_long_eval_records(checkpoint, args.panel_revision)
        long_eval = run_hy4_flash_long_eval(
            adapter,
            sidecar_path=args.sidecar,
            records=long_records,
            corpus_meta=long_corpus_meta,
            device=args.device,
            layers=long_layers,
            modes=long_modes,
            expected_source=expected_source,
        )
        attach_long_context_eval(args.output, long_eval, manifest=sidecar_manifest)
        print(
            json.dumps({
                "status": "passed",
                "phase": "long_eval",
                "layers_evaluated": long_eval["layers_evaluated"],
                "combinations": sorted(long_eval["combinations"]),
                "transform_fingerprint": long_eval["transform_fingerprint"],
                "report": str(args.output.resolve() / "qmla-validation-report.json"),
            }),
            flush=True,
        )
        return 0

    resolved = resolve_run_settings(args, eligible)

    if args.corpus_dir is None and args.reuse_captures is None:
        parser.error("provide --corpus-dir PATH or --reuse-captures DIR")

    capture_dir = (args.reuse_captures or (args.output / "captures")).resolve()
    all_records = None
    if args.corpus_dir is not None:
        all_records, corpus_provenance = load_pinned_qmla_corpus(
            args.corpus_dir,
            panel_revision=resolved["panel_revision"],
            components=resolved["components"],
            full=resolved["full"],
        )
    split_counts = {f"{name}_count": resolved[f"{name}_sequences"] for name in SPLIT_NAMES}
    if args.reuse_captures is not None:
        banks = {lid: load_layer_capture(capture_dir, lid) for lid in resolved["layers"]}
        corpus_provenance = check_reused_captures(
            banks, resolved, forward_identity, seed=args.seed,
            expected_splits=None if all_records is None else corpus_split_description(
                all_records, seed=args.seed, **split_counts
            ),
        )
    else:
        assert all_records is not None
        corpus_provenance.update(forward_identity)
        banks = _stream_hy4_flash_captures(
            reader,
            adapter,
            resolved["layers"],
            all_records,
            seed=args.seed,
            capture_dir=capture_dir,
            provenance=corpus_provenance,
            **split_counts,
        )

    source = {
        "repo": args.source_repo,
        "revision": args.source_revision,
        "base_repo": args.base_repo,
        "base_revision": args.base_revision,
        "family": "hy4_flash",
        "model_type": "hy_v4_flash",
        "config_sha256": config_sha256,
        "checkpoint_kind": args.checkpoint_kind,
        "calibrated_view": args.view,
    }
    fit_cfg = FitConfig(steps=resolved["steps"], seed=args.seed, protection=resolved["protection"])
    template_path = checkpoint / "chat_template.jinja"
    template_sha = hashlib.sha256(template_path.read_bytes()).hexdigest() if template_path.is_file() else ""

    result = calibrate_hy4_flash_from_banks(
        banks,
        output_dir=args.output,
        source=source,
        modes=resolved["modes"],
        corpus_provenance=corpus_provenance,
        fit_config=fit_cfg,
        device=args.device,
        renderer_template_sha256=template_sha,
        code_commit=code_commit,
    )
    if args.long_eval:
        del banks
        long_records, long_corpus_meta = load_long_eval_records(checkpoint, resolved["panel_revision"])
        long_eval = run_hy4_flash_long_eval(
            adapter,
            sidecar_path=result["sidecar_dir"],
            records=long_records,
            corpus_meta=long_corpus_meta,
            device=args.device,
            layers=resolved["layers"],
            modes=resolved["modes"],
            expected_source={"repo": source["repo"], "revision": source["revision"], "family": "hy4_flash"},
        )
        attach_long_context_eval(args.output, long_eval, manifest=result["manifest"])
    print(
        json.dumps({
            "status": result["validation"]["status"],
            "artifact_status": result["manifest"]["artifact_status"],
            "protection_policy": result["manifest"]["protection_policy"],
            "layers": list(resolved["layers"]),
            "supported_modes": result["manifest"]["supported_modes"],
            "transform_fingerprint": result["manifest"]["transform_fingerprint"],
            "sidecar_dir": result["sidecar_dir"],
            "long_eval": bool(args.long_eval),
        }),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
