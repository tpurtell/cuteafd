"""Fit the GLM 5.3 Flash MUL1 target's content-only MLA cache calibration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

from qmla.capture import (CapturedLayerBank, SPLIT_NAMES, load_layer_capture, save_layer_capture,
                          split_train_val_test)
from qmla.fit import FitConfig
from qmla.long_eval import LongContextEvaluator, load_long_context_records, panel_snapshot_dir
from qmla.pipeline import (_git_head_sha, attach_long_context_eval, build_run_parser, calibrate_from_banks,
                           check_reused_captures, corpus_split_description, load_pinned_qmla_corpus,
                           parse_modes_arg, resolve_settings)
from qmla.sidecar import load_sidecar_transforms

TARGET_REPO = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC-MUL1"
TARGET_REVISION = "1bca67feffab2747e0e489e571185bc06336ee26"
BASE_REPO = "zai-org/GLM-5.3-Flash-BF16"
BASE_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
PILOT_LAYERS = (3, 23, 43)
MODES = ("8:0", "4:0", "2:0")


def build_parser():
    return build_run_parser(__doc__, target_repo=TARGET_REPO, base_repo=BASE_REPO, base_revision=BASE_REVISION)


def resolve_run_settings(args, layers):
    settings = resolve_settings(args, layers, pilot_layers=PILOT_LAYERS, pilot_modes=MODES, full_modes=MODES)
    if any(rope for _, rope in settings["modes"]):
        raise ValueError("GLM 5.3 Flash has no RoPE key cache (qk_rope_head_dim=0)")
    return settings


def read_identities(checkpoint, args):
    if getattr(args, "teacher_checkpoint", None) is not None:
        raise ValueError("GLM Flash requires the MC original view, including its shared K4 experts")
    if (checkpoint.name, args.source_repo, args.source_revision, args.base_repo, args.base_revision,
            args.view, args.checkpoint_kind) != (
            TARGET_REVISION, TARGET_REPO, TARGET_REVISION, BASE_REPO, BASE_REVISION, "original", "mc"):
        raise ValueError("GLM Flash QMLA requires the pinned MUL1 snapshot and original view")
    mc = json.loads((checkpoint / "cuteafd-mc.json").read_text())
    if mc["source"] != {"repo": BASE_REPO, "revision": BASE_REVISION}:
        raise ValueError("MUL1 original source provenance differs from the pinned parent")
    variant, = (v for v in mc["variants"] if v["name"] == "original")
    if (mc["schema"], mc["original_core_dtype"], mc["exl3_codebook"]["name"],
            variant["core_bits"], variant["index"]) != (
            "cuteafd-mc-v1", "BF16", "mul1", 16, "model.safetensors.index.json"):
        raise ValueError("unexpected MC schema, codebook or original core inventory")
    forward = dict(repo=TARGET_REPO, revision=TARGET_REVISION, view="original", routed_experts="exl3-k4-mul1",
                   index_sha256=hashlib.sha256((checkpoint / variant["index"]).read_bytes()).hexdigest(),
                   view_fingerprint=variant["fingerprint"], shared_inventory_sha256=mc["shared_inventory_sha256"])
    source = dict(repo=TARGET_REPO, revision=TARGET_REVISION, base_repo=BASE_REPO, base_revision=BASE_REVISION,
                  family="glm5_flash", model_type="glm5_next_text", checkpoint_kind="mc", calibrated_view="original",
                  config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
                  forward=forward)
    return source, {"capture_forward": forward}


@torch.no_grad()
def stream_layers(source, adapter, layers, records, device, *, long_eval=None, phase="capture"):
    """Walk the native decoder layer-major. Only one sequence's recurrence is on the device.

    Yields one capture bank's inputs at a time. During long eval score each row immediately
    and keep only the compact DSA selection for that sequence.
    """
    from .qmla_mc import native

    last = max(layers)
    embedding = source.tensor(native.PREFIX + "embed_tokens.weight", device)
    states = [torch.nn.functional.embedding(tokens.to(device), embedding)[None, :, None, :].expand(
        -1, -1, source.config["hc_mult"], -1).contiguous().cpu() for _, tokens in records]
    del embedding
    source.close()
    for index in range(last + 1):
        started = time.monotonic()
        block = source.load_block(index, device, include_experts=index < last)
        captured = []
        geometry = adapter.geometry_for(index) if index in layers else None
        params = adapter.layer_params(block.self_attn) if geometry is not None else None
        if getattr(block.self_attn, "next_skip_topk", False):
            raise ValueError("GLM Flash QMLA requires each DSA block to own its indexer")
        for row, (seq_id, _) in enumerate(records):
            state = states[row].to(device)
            if geometry is None:
                result = native.native_forward(block, {"hidden": state})
                if "prev_topk_indices" in result:
                    raise ValueError("unexpected shared-indexer state in GLM Flash")
                states[row] = result["hidden"].cpu()
                del result
            else:
                normalized = native.attention_inputs(block, state)
                seq, output = adapter.capture_sequence(normalized, block.self_attn, seq_id=seq_id,
                                                       compact_mask=long_eval is not None)
                if long_eval is None:
                    captured.append(seq.to("cpu"))
                else:
                    evaluator, splits = long_eval
                    evaluator.evaluate_layer_sequence(seq, params, geometry, split=splits[seq_id], device=device)
                del seq
                if index < last:
                    streams, mlp_inputs = native.attention_finish(block, state, output.reshape_as(normalized))
                    states[row] = native.finish_layer(block, streams, mlp_inputs).cpu()
                    del streams, mlp_inputs
                del normalized, output
            del state
        del block
        if geometry is not None:
            yield index, params, captured
            if long_eval is not None:
                evaluator.release_layer_cache()
        del captured, params
        print(json.dumps(dict(phase=phase, completed_layer=index,
                              seconds=time.monotonic() - started)), flush=True)


def capture_banks(source, adapter, records, settings, capture_dir, provenance, seed, device):
    counts = {f"{name}_count": settings[f"{name}_sequences"] for name in SPLIT_NAMES}
    parts = split_train_val_test(records, **counts, key_fn=lambda row: row[0], seed=seed)
    provenance = {**provenance, "splits": corpus_split_description(records, seed=seed, **counts)}
    train_n, val_n = len(parts[0]), len(parts[1])
    banks = {}
    for idx, params, caps in stream_layers(source, adapter, settings["layers"], sum(parts, []), device):
        bank = CapturedLayerBank(adapter.geometry_for(idx), params.to("cpu"), tuple(caps[:train_n]),
                                 tuple(caps[train_n:train_n + val_n]), tuple(caps[train_n + val_n:]), provenance)
        save_layer_capture(capture_dir, bank)
        caps.clear()
        del bank
        banks[idx] = load_layer_capture(capture_dir, idx)
    return banks


def run_evaluation(source, adapter, sidecar_path, records, corpus_meta, *, expected_source, device,
                   layers=None, modes=None, phase="long_eval"):
    """Score native long-context records or the compact view's frozen test records layer-major."""
    manifest, content, rope = load_sidecar_transforms(sidecar_path, expected_source=expected_source, strict=True)
    layers = sorted(content) if layers is None else tuple(layers)
    if set(layers) - set(content):
        raise ValueError("evaluation requested a layer missing from the sidecar")
    if modes is None:
        modes = parse_modes_arg(",".join(manifest["supported_modes"]))
    if any(f"{c}:{r}" not in manifest["supported_modes"] for c, r in modes):
        raise ValueError("evaluation requested a mode missing from the sidecar")
    evaluator = LongContextEvaluator(modes=modes, content_transforms=content, rope_transforms=rope,
                                     protection_policy=manifest["protection_policy"],
                                     dsa_indexer_note=adapter.dsa_indexer_note)
    start = time.monotonic()
    seqs = [(r.seq_id, r.input_ids) for r in records]
    for _ in stream_layers(source, adapter, layers, seqs, device, phase=phase,
                           long_eval=(evaluator, {r.seq_id: r.split for r in records})):
        pass
    return evaluator.finalize(corpus_meta=corpus_meta, transform_fingerprint=manifest["transform_fingerprint"],
                              device=device, wall_seconds=time.monotonic()-start)


def long_context_eval(source, adapter, checkpoint, source_id, path, args, settings):
    records, meta = load_long_context_records(panel_snapshot_dir(settings["panel_revision"]), checkpoint_dir=checkpoint,
                                             chat_kwargs={}, panel_revision=settings["panel_revision"])
    section = run_evaluation(source, adapter, path, records, meta, expected_source=source_id, device=args.device,
                             layers=settings["layers"] if args.layers else None,
                             modes=settings["modes"] if args.modes else None)
    manifest, _, _ = load_sidecar_transforms(path, expected_source=source_id, strict=True)
    attach_long_context_eval(args.output, section, manifest=manifest)


def main(argv=None):
    from .qmla_mc import GlmFlashMCSource
    from .qmla_capture import Glm5FlashCaptureAdapter

    parser = build_parser()
    args = parser.parse_args(argv)
    code_sha = _git_head_sha(Path(__file__).resolve().parents[2])
    checkpoint = args.checkpoint.resolve()
    source_id, capture_identity = read_identities(checkpoint, args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = GlmFlashMCSource(checkpoint, view="original")
    adapter = Glm5FlashCaptureAdapter(source.native_config())
    settings = resolve_run_settings(args, [g.layer_id for g in adapter.mla_geometries()])
    if args.sidecar:
        if not args.long_eval:
            parser.error("--sidecar requires --long-eval")
        long_context_eval(source, adapter, checkpoint, source_id, args.sidecar, args, settings)
        return 0
    if args.corpus_dir is None:
        parser.error("--corpus-dir is required to bind captures to the native rendered panel")
    corpus_manifest = json.loads((args.corpus_dir / "manifest.json").read_text())
    template_hash = hashlib.sha256((checkpoint / "chat_template.jinja").read_bytes()).hexdigest()
    renderer = corpus_manifest["source"]
    if (renderer["files"]["chat_template.jinja"] != template_hash
            or renderer["repo_id"] != TARGET_REPO or renderer["revision"] != TARGET_REVISION
            or any(key in corpus_manifest["contract"] for key in ("reasoning_effort", "enable_thinking", "clear_thinking"))):
        raise ValueError("prepared corpus differs from the pinned GLM MC tokenizer and served default reasoning policy")
    records, prov = load_pinned_qmla_corpus(args.corpus_dir, panel_revision=settings["panel_revision"],
                                            components=settings["components"], full=settings["full"])
    if args.reuse_captures:
        banks = {lid: load_layer_capture(args.reuse_captures, lid) for lid in settings["layers"]}
        counts = {f"{k}_count": settings[f"{k}_sequences"] for k in SPLIT_NAMES}
        prov = check_reused_captures(banks, settings, capture_identity, seed=args.seed,
                                     expected_splits=corpus_split_description(records, seed=args.seed, **counts))
    else:
        prov.update(capture_identity)
        banks = capture_banks(source, adapter, records, settings, args.output / "captures", prov, args.seed, args.device)
    result = calibrate_from_banks(banks, output_dir=args.output, source=source_id, modes=settings["modes"],
                                  corpus_provenance=prov, fit_config=FitConfig(steps=settings["steps"], seed=args.seed,
                                  protection=settings["protection"]), device=args.device, renderer_template_sha256=template_hash,
                                  thinking_mode="enabled (served template default, effective reasoning effort max)",
                                  code_commit=code_sha)
    del banks
    if args.long_eval:
        long_context_eval(source, adapter, checkpoint, source_id, result["sidecar_dir"], args, settings)
    print(json.dumps({"status": result["validation"]["status"], "sidecar_dir": result["sidecar_dir"],
                      "layers": settings["layers"], "supported_modes": result["manifest"]["supported_modes"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
