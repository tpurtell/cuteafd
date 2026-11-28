"""Compare the GLM QMLA replay with the pinned, native BF16 attention on real panel prompts."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import panel_corpus
from qmla.capture import _atomic_json, eval_attention_output
from qmla.fit import FitConfig, check_content_identity
from qmla.measurement import _Scorer


MC_REVISION = "1bca67feffab2747e0e489e571185bc06336ee26"
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


def fp64_reference(query, expanded_kv, native_mask, output_weight, geometry):
    """Evaluate the same native Q/K/V operands and selected keys entirely in FP64."""
    length = query.shape[0]
    q = query.reshape(length, geometry.heads, geometry.qk_nope_dim).double().transpose(0, 1)
    kv = expanded_kv.reshape(length, geometry.heads, geometry.qk_nope_dim + geometry.v_head_dim)
    k, v = kv.double().transpose(0, 1).split([geometry.qk_nope_dim, geometry.v_head_dim], dim=-1)
    allowed = native_mask.reshape(length, length)
    if allowed.dtype != torch.bool:
        allowed = allowed.eq(0)
    attended = []
    for start in range(0, length, 128):
        stop = min(length, start + 128)
        logits = (q[:, start:stop] @ k.transpose(-1, -2)) * geometry.effective_scale
        probs = logits.masked_fill(~allowed[start:stop], -torch.inf).softmax(-1)
        attended.append((probs @ v).transpose(0, 1))
    return F.linear(torch.cat(attended).reshape(length, -1), output_weight.double())


def relative_l2(observed, reference):
    ref = reference.double()
    norm = torch.linalg.vector_norm(ref)
    if not bool(torch.isfinite(norm) & (norm > 0)):
        raise ValueError("invalid reference norm")
    return float(torch.linalg.vector_norm(observed.double() - ref) / norm)


@torch.inference_mode()
def run(checkpoint, corpus_dir, report):
    from mistral_large4.core_k6.numerical import fork_module
    from .qmla_capture import Glm5FlashCaptureAdapter

    from .qmla_mc import GlmFlashMCSource

    if checkpoint.name != MC_REVISION:
        raise ValueError("expected the pinned MUL1 MC snapshot")
    rows, provenance = panel_corpus.load_component_records(corpus_dir, "calibration",
                                                         components=panel_corpus.default_components())
    if provenance["dataset_revision"] != panel_corpus.V3_PANEL.revision:
        raise ValueError("expected the pinned v3 calibration panel")
    native = fork_module("utils.glm5_flash_core")
    source = GlmFlashMCSource(checkpoint, view="original")
    cfg = source.native_config()
    adapter = Glm5FlashCaptureAdapter(cfg)
    layer_id = min(g.layer_id for g in adapter.mla_geometries())
    geometry = adapter.geometry_for(layer_id)
    witnesses = select_witnesses(rows, cfg.index_topk, cfg.index_kpool)
    report.update(capture_revision=MC_REVISION, capture_view="original", shared_experts="K4 MUL1", layer=layer_id, panel=provenance["dataset_revision"],
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
        indices, queries, expanded = [], [], []
        with (attn.indexer.register_forward_hook(lambda module, args, out: indices.append(out)),
              attn.q_b_proj.register_forward_hook(lambda module, args, out: queries.append(out)),
              attn.kv_b_proj.register_forward_hook(lambda module, args, out: expanded.append(out))):
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
        check_content_identity((seq,), params, geometry)
        actual = eval_attention_output(seq, params, geometry)
        if actual.shape != expected.shape or seq.ref_output.shape != expected.shape:
            raise ValueError("reference and replay output shapes differ")
        rel = relative_l2(actual, expected)
        captured_rel = relative_l2(seq.ref_output, expected)
        details = {}
        if i == 2:
            oracle = fp64_reference(queries[0][0], expanded[0], native_mask, attn.o_proj.weight, geometry)
            native_err = relative_l2(expected, oracle)
            replay_err = relative_l2(actual, oracle)
            config = FitConfig()
            scorer = _Scorer(None, None, config, device, geom=geometry, params=params, seqs=(seq,))
            int8 = eval_attention_output(seq, params, geometry, **scorer.content_hooks(8, "rtn"), **scorer.protection)
            int8_err = relative_l2(int8, expected)
            details = dict(native_vs_fp64=native_err, replay_vs_fp64=replay_err, int8_rtn_vs_native=int8_err,
                           replay_budget=0.01 * int8_err, oracle="native projected Q/K/V, exact DSA mask, FP64 softmax, value product and o_proj")
            passed = replay_err <= native_err and rel < 0.01 * int8_err
        else:
            passed = rel < MAX_RELATIVE_ERROR
        passed = (passed and captured_rel == 0 and selection["mask_matches"]
                  and selection["mask_is_causal"] and selection["sparse_witness_passed"])
        report["cases"].append(dict(panel_record=name, tokens=len(tokens), token_sha256=panel_corpus.ids_sha256(tokens),
                                    relative_l2=rel, captured_ref_relative_l2=captured_rel,
                                    reference_dtype="bfloat16", identity_byte_identical=True, **details, **selection, passed=passed))
        print(report["cases"][-1], flush=True)
    if not all(case["passed"] for case in report["cases"]):
        raise ValueError("GLM BF16 native reference preflight failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="preflight receipt JSON")
    args = parser.parse_args()
    report = {"schema": "glmf-qmla-preflight-v2", "status": "failed"}
    start = time.monotonic()
    try:
        run(args.checkpoint, args.corpus_dir, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        _atomic_json(args.output, report)


if __name__ == "__main__":
    main()
