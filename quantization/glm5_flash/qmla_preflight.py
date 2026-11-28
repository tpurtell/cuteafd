"""Compare the GLM QMLA replay with the pinned, native BF16 attention on real panel prompts."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import panel_corpus
from qmla.capture import _atomic_json, eval_attention_output


TEACHER_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
MAX_RELATIVE_ERROR = 2e-5


def select_witnesses(records, index_topk, pool):
    """Two distinct short prefixes and one with enough complete pools for real top-k pruning."""
    selected = []
    for length in (64, 193, index_topk + 8 * pool):
        row = next((r for r in records if len(r["input_ids"]) >= length
                    and all(r["id"] != name for name, _ in selected)), None)
        if row is None:
            raise ValueError(f"missing distinct panel witness of at least {length} tokens")
        selected.append((row["id"], row["input_ids"][:length]))
    return selected


def check_selection(seq, native_mask, sparse_required):
    """Compare all rows against the native attention mask, including causal tail tokens."""
    length = seq.seq_len
    expected = native_mask.reshape(length, length)
    if expected.dtype != torch.bool:
        expected = expected.eq(0)  # The pinned eager path returns an additive mask.
    positions = torch.arange(length, device=expected.device)
    allowed = positions[None, :] <= positions[:, None]
    if seq.allowed_mask is not None:
        actual = seq.allowed_mask
    elif seq.sparse_selection is not None:
        actual = seq.sparse_selection.tile_mask(0, length, 0, length, positions) & allowed
    else:
        actual = allowed
    counts = expected.sum(-1)
    pruned = int((counts < positions + 1).sum())
    return dict(mask_matches=bool(torch.equal(actual, expected)), sparse_required=sparse_required,
                pruned_rows=pruned, last_visible_keys=int(counts[-1]),
                mask_is_causal=bool((~expected | allowed).all()),
                sparse_witness_passed=(pruned > 0 if sparse_required else pruned == 0))


@torch.inference_mode()
def run(teacher, corpus_dir, report):
    from mistral_large4.core_k6.numerical import fork_module
    from .qmla_capture import Glm5FlashCaptureAdapter

    if teacher.name != TEACHER_REVISION:
        raise ValueError("expected the pinned BF16 teacher snapshot")
    rows, provenance = panel_corpus.load_component_records(corpus_dir, "calibration",
                                                         components=panel_corpus.default_components())
    if provenance["dataset_revision"] != panel_corpus.V3_PANEL.revision:
        raise ValueError("expected the pinned v3 calibration panel")
    native = fork_module("utils.glm5_flash_core")
    source = native.GlmFlashBF16Source(teacher)
    cfg = source.native_config()
    adapter = Glm5FlashCaptureAdapter(cfg)
    layer_id = min(g.layer_id for g in adapter.mla_geometries())
    geometry = adapter.geometry_for(layer_id)
    witnesses = select_witnesses(rows, cfg.index_topk, cfg.index_kpool)
    report.update(teacher_revision=TEACHER_REVISION, layer=layer_id, panel=provenance["dataset_revision"],
                  attention_backend=cfg._attn_implementation, relative_l2_limit=MAX_RELATIVE_ERROR, cases=[])
    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    embedding = source.tensor(native.PREFIX + "embed_tokens.weight", device)
    states = [F.embedding(torch.tensor([tokens], device=device), embedding).unsqueeze(2)
              .expand(-1, -1, cfg.hc_mult, -1).contiguous() for _, tokens in witnesses]
    del embedding
    source.close()
    for layer in range(layer_id):
        block = source.load_block(layer, device, include_experts=True)
        if block.block_type != "linear_attention":
            raise ValueError("unexpected predecessor before the first MLA layer")
        for i, hidden in enumerate(states):
            state = native.native_forward(block, {"hidden": hidden})
            states[i] = state["hidden"]
        del block, hidden, state
    block = source.load_block(layer_id, device, include_experts=False)
    attn = block.self_attn
    params = adapter.layer_params(attn).to(device)
    for i, ((name, tokens), hidden) in enumerate(zip(witnesses, states, strict=True)):
        inputs = native.attention_inputs(block, hidden)
        context = native.prepare_context(hidden)
        indices = []
        with attn.indexer.register_forward_hook(lambda module, args, out: indices.append(out)):
            output, _, _ = attn(hidden_states=inputs, past_key_values=None, **context)
        if output.dtype != torch.bfloat16 or len(indices) != 1:
            raise ValueError("native BF16 attention or DSA indexer was bypassed")
        native_mask = attn.build_attention_mask_from_topk(indices[0], output, len(tokens))
        expected = output[0].float()
        with adapter.capture_hook(attn, seq_id=f"preflight-{i}", compact_mask=(i == 2)) as captured:
            attn(hidden_states=inputs, past_key_values=None, **context)
        seq, = captured
        seq = seq.to(device)
        seq.validate(geometry)
        selection = check_selection(seq, native_mask, sparse_required=(i == 2))
        actual = eval_attention_output(seq, params, geometry)
        if actual.shape != expected.shape or seq.ref_output.shape != expected.shape:
            raise ValueError("reference and replay output shapes differ")
        energy = torch.linalg.vector_norm(expected)
        rel = float(torch.linalg.vector_norm(actual - expected) / energy)
        captured_rel = float(torch.linalg.vector_norm(seq.ref_output.float() - expected) / energy)
        passed = (math.isfinite(rel) and rel < MAX_RELATIVE_ERROR
                  and math.isfinite(captured_rel) and captured_rel < MAX_RELATIVE_ERROR
                  and selection["mask_matches"] and selection["mask_is_causal"]
                  and selection["sparse_witness_passed"])
        report["cases"].append(dict(panel_record=name, tokens=len(tokens), token_sha256=panel_corpus.ids_sha256(tokens),
                                    relative_l2=rel if math.isfinite(rel) else None,
                                    captured_ref_relative_l2=captured_rel if math.isfinite(captured_rel) else None,
                                    reference_dtype="bfloat16", **selection, passed=passed))
        print(report["cases"][-1], flush=True)
    if not all(case["passed"] for case in report["cases"]):
        raise ValueError("GLM BF16 native reference preflight failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="preflight receipt JSON")
    args = parser.parse_args()
    report = {"schema": "glmf-qmla-preflight-v1", "status": "failed"}
    start = time.monotonic()
    try:
        run(args.teacher, args.corpus_dir, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        _atomic_json(args.output, report)


if __name__ == "__main__":
    main()
