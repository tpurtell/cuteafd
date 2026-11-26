"""Compare the staged, unquantized ERNIE calibration forward against the pinned reference on maxq."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import islice
import json
import math
from pathlib import Path
import socket
import tempfile
import time

import torch
import torch.nn.functional as F

from . import identity
from .forward import ErnieForward, MlpInput, reference as ref
from .prepare_corpus import load_token_batches, load_witnesses, panel
from .routing import ErnieRoutes
from .storage import RoutedWorkspace

PILOT_LAYERS = (0, 1, 7, 20, 21, 42)
RELATIVE_BOUND = 1e-4


class Measurements:
    """Worst relative L2 over compared samples, rejecting non-finite or incompatible tensors."""

    def __init__(self):
        self.rows = {}

    def compare(self, path, actual, expected):
        if actual.shape != expected.shape or not actual.numel() or actual.dtype != expected.dtype:
            raise ValueError(f"{path} parity geometry differs: {actual.shape} vs {expected.shape}")
        if actual.is_floating_point():
            diff = torch.linalg.vector_norm(actual - expected)
            norm = torch.linalg.vector_norm(expected)
            error = (diff / norm.clamp_min(1e-20)).item()
            bound = RELATIVE_BOUND
        else:
            error = float(not torch.equal(actual, expected))
            bound = 0
        if not math.isfinite(error) or error > bound:
            raise ValueError(f"{path} parity failed: error {error}, bound {bound}")
        row = self.rows.setdefault(path, dict(error=0., bound=bound, comparisons=0))
        row["error"] = max(error, row["error"])
        row["comparisons"] += 1


@dataclass
class Prompt:
    ids: torch.Tensor
    provenance: dict
    saved: dict = field(default_factory=dict)
    candidate_taps: dict = field(default_factory=dict)


def read_prompts(corpus, per_split=4, tail=256):
    manifest = identity.read(corpus / "manifest.json")
    prompts = []
    seen = set()
    for split in ("calibration", "heldout"):
        records = manifest["splits"][split]["records"]
        for index, batch in enumerate(islice(load_token_batches(corpus, split), per_split)):
            row = records[index]
            ids = batch["input_ids"][0]
            suffix = ids[-tail:].clone()
            tail_hash = panel.ids_sha256(suffix.tolist())
            if (len(suffix) < min(tail, 129) or row["prompt_sha256"] in seen or tail_hash in seen
                    or row["input_ids_sha256"] != panel.ids_sha256(ids.tolist())):
                raise ValueError("short, duplicate or unverified pilot prompt tail")
            seen.update((row["prompt_sha256"], tail_hash))
            provenance = dict(split=split, id=row["id"], file=f"{split}/{index:03}.pt",
                              input_ids_sha256=row["input_ids_sha256"], prompt_sha256=row["prompt_sha256"],
                              rendered_sha256=row["rendered_sha256"], tail_ids_sha256=tail_hash,
                              start=len(ids)-len(suffix), tokens=len(suffix), source_tokens=len(ids))
            prompts.append(Prompt(suffix, provenance, {"ids": suffix}))
    if len(prompts) != per_split * 2:
        raise ValueError("missing real prompt tails in preflight panel")
    return prompts


def embed(ids, embedding, config):
    x = F.embedding(ids.to(embedding.device), embedding)
    return x[:, None].expand(-1, config.hc_mult, -1).contiguous()


def reference_mlp_input(hidden, weights, config, index, positions):
    """Native per-prompt attention. Only the positionwise FFN can batch across prompts."""
    post, comb, x = ref.mhc_pre(hidden, weights, "self_attention_hyper_connection",
                               weights["input_layernorm.weight"], config)
    if config.is_dsa_layer(index):
        attended = ref.mla_dsa_attention(x, weights, config, positions, enable_flashindexer=False)
    else:
        attended = ref.hca_attention(x, weights, config, positions, rope_mode="yarn")
    residual = ref.mhc_post(attended, hidden, post, comb)
    post, comb, x = ref.mhc_pre(residual, weights, "mlp_hyper_connection",
                               weights["post_attention_layernorm.weight"], config)
    return MlpInput(residual, post, comb, x)


def stage_sums(model, inputs, weights, checks, *, reference_inputs=None):
    cfg = model.config
    batch = model.route_batch(inputs, weights)
    ids, gamma = ref.moe_routing(inputs if reference_inputs is None else reference_inputs, weights, cfg)
    checks.compare("route_ids", batch["ids"], ids)
    checks.compare("route_gamma", batch["gamma"], gamma)
    routes = ErnieRoutes([(0, {k: v.cpu() for k, v in batch.items()})], cfg.n_routed_experts, len(inputs))
    workspace = RoutedWorkspace(routes, inputs.device)
    store = weights["experts"]

    def projection(expert):
        w1, w3, w2 = store.get(expert, inputs.device)
        return lambda x: ref.linear(model.activated(x, w1, w3), w2)

    return workspace.replay(projection)


def target_step(model, weights, index, samples, expected, actual, checks, verify_native):
    cfg = model.config
    native_states, staged_states = [], []
    for hidden, staged in zip(expected, actual, strict=True):
        positions = torch.arange(len(hidden), device=hidden.device)
        native = reference_mlp_input(hidden, weights, cfg, index, positions)
        state = model.mlp_input(staged, weights, index, positions)
        checks.compare("mla_dsa" if cfg.is_dsa_layer(index) else "hca", state.residual, native.residual)
        checks.compare("mhc", state.x, native.x)
        native_states.append(native)
        staged_states.append(state)

    inputs = torch.cat([s.x for s in native_states])
    lengths = [len(s.x) for s in native_states]
    if cfg.is_moe_layer(index):
        natural = ref.latent_moe_ffn(inputs, weights, cfg)
        routed = stage_sums(model, torch.cat([s.x for s in staged_states]), weights, checks, reference_inputs=inputs)
        outputs = [model.finish_moe(s, y, weights) for s, y in zip(staged_states, routed.split(lengths), strict=True)]
        metric = "latent_moe"
        first, second = ("block_sparse_moe.shared_experts.w1.weight", "block_sparse_moe.shared_experts.w3.weight")
    else:
        natural = ref.dense_ffn(inputs, weights, cfg)
        outputs = [model.finish_dense(s, weights) for s in staged_states]
        metric = "dense"
        first, second = "mlp.gate_proj.weight", "mlp.up_proj.weight"
    probe = inputs[:ref.ROWS]
    checks.compare("situ", model.activated(probe, weights[first], weights[second]),
                   ref.situ(ref.linear(probe, weights[first]), ref.linear(probe, weights[second]),
                            cfg.activation_situ_beta, cfg.activation_situ_linear_beta))
    native_outputs = [s.finish(y) for s, y in zip(native_states, natural.split(lengths), strict=True)]
    if verify_native:
        # Also compare to the unchanged monolithic reference entry point at every pilot layer.
        checks.compare("native_layer_entry", native_outputs[0], model.native_layer(expected[0], weights, index))
    for sample, hidden, out, native in zip(samples, expected, outputs, native_outputs, strict=True):
        checks.compare(metric, out, native)
        if verify_native:
            sample.saved[f"layer_{index}"] = hidden.cpu()
        if index in cfg.dspark_target_layer_ids:
            sample.saved[f"tap_{index}"] = native.mean(dim=1).cpu()
            sample.candidate_taps[index] = out.mean(dim=1)
    return native_outputs, outputs


def check_witness(model, weights, index, tail_ids, embedding, checks):
    """Exercise the true compressed HCA and pruned DSA paths with bounded attention matrices."""
    c = model.config
    is_dsa = c.is_dsa_layer(index)
    if is_dsa:
        count = c.sliding_window + c.index_topk + 1
        start = len(tail_ids) - count
    else:
        count = 2 * ref.COMPRESS_RATIO_HCA + 1
        start = 0  # Target HCA's compressor assigns absolute positions from its prompt origin.
    ids = tail_ids[start:start+count]
    if len(ids) != count or start < 0:
        raise ValueError("witness shorter than the attention selection boundary")
    hidden = embed(ids, embedding, c)
    positions = torch.arange(start, start + count, device=hidden.device)
    native = reference_mlp_input(hidden, weights, c, index, positions)
    stage = model.mlp_input(hidden, weights, index, positions)
    metric = "long_mla_dsa" if is_dsa else "long_hca"
    checks.compare(metric, stage.residual, native.residual)
    # Exactly the final query exceeds the sparse cutoff. Keep its error visible
    # even if all preceding dense rows happen to agree.
    checks.compare(metric + "_last_query", stage.residual[-1], native.residual[-1])
    return dict(layer=index, start=start, stop=start+count, full_tokens=len(tail_ids),
                scope="embedding-input attention operator window",
                pruned_query_rows=count-c.sliding_window-c.index_topk if is_dsa else 0,
                compressed_chunks=0 if is_dsa else count//ref.COMPRESS_RATIO_HCA)


def draft_block(model, sample, anchor, auxiliary, stages, outer, checks):
    cfg = model.config
    device = next(iter(outer.values())).device
    taps = [sample.saved[f"tap_{i}"].to(device) for i in cfg.dspark_target_layer_ids]
    candidate_taps = [sample.candidate_taps[i] for i in cfg.dspark_target_layer_ids]
    native = model.native_proposal(taps, anchor, auxiliary, stages, outer)

    def stage(index, hidden, weights, positions, context, kv):
        sample.saved.update({f"draft{index}_hidden": hidden.cpu(), f"draft{index}_context_kv": kv.cpu(),
                             f"draft{index}_positions": positions.cpu()})
        if index == 0:
            sample.saved["draft_context_features"] = context.cpu()
        state = model.mlp_input(hidden, weights, index, positions, draft_context_kv=kv)
        out = model.finish_moe(state, stage_sums(model, state.x, weights, checks), weights)
        checks.compare("draft_stage", out, model.native_draft_stage(hidden, weights, positions, kv))
        return out

    staged = model.staged_proposal(candidate_taps, anchor, auxiliary, stages, outer, stage)
    for metric, result, expected in zip(("draft_tokens", "dspark", "draft_confidence"), staged, native, strict=True):
        checks.compare(metric, result, expected)
    sample.saved["anchor"] = torch.tensor(anchor)
    sample.saved["reference_proposal"] = native[0].cpu()


@torch.inference_mode()
def run_forward(model, samples, witness_ids, *, pilot_layers=PILOT_LAYERS):
    cfg, checkpoint = model.config, model.checkpoint
    checks = Measurements()
    embedding = checkpoint.load(ref.OUTER_NAMES[0])
    actual = [embed(sample.ids, embedding, cfg) for sample in samples]
    expected = list(actual)
    dsa_layer = cfg.dsa_layers()[0]
    witness_results = [[] for _ in witness_ids]
    for index in range(cfg.num_hidden_layers):
        start = time.monotonic()
        print(json.dumps(dict(phase="reference-forward", layer=index, event="loading")), flush=True)
        weights = checkpoint.layer(cfg, index, load_experts=True, experts_on_device=True, load_flashindexer=False)
        expected, actual = target_step(model, weights, index, samples, expected, actual, checks, index in pilot_layers)
        if index in (0, dsa_layer):
            for ids, result in zip(witness_ids, witness_results, strict=True):
                result.append(check_witness(model, weights, index, ids, embedding, checks))
        del weights
        checkpoint.close()
        print(json.dumps(dict(phase="reference-forward", layer=index, seconds=round(time.monotonic()-start, 2),
                              event="passed")), flush=True)

    outer = {name: checkpoint.load(name) for name in ref.OUTER_NAMES[1:]}
    outer[ref.OUTER_NAMES[0]] = embedding
    anchors = []
    for a, b in zip(actual, expected, strict=True):
        logits = model.logits(a[-1:], outer)
        native = ref.lm_head_logits(b[-1:], outer, cfg, eps_outside_sqrt=False)
        checks.compare("lm_head", logits, native)
        checks.compare("target_token", logits.argmax(-1), native.argmax(-1))
        anchors.append(int(native.argmax(-1).item()))
    del actual, expected
    # Three compact BF16 stores fit together on maxq. Each expert is promoted to
    # FP32 only while its projections execute. The independent reference needs all three.
    stages = [checkpoint.mtp_stage(cfg, i, load_experts=True, experts_on_device=True)
              for i in range(cfg.num_mtp_stages)]
    aux = checkpoint.mtp_aux(cfg)
    for sample, anchor in zip(samples, anchors, strict=True):
        draft_block(model, sample, anchor, aux, stages, outer, checks)
    checkpoint.close()
    return checks.rows, witness_results


def write_fixtures(destination, samples, corpus, local):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = identity.read(corpus / "manifest.json")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".ernie55-fixtures-") as tmp:
        stage = Path(tmp) / "fixtures"
        records = []
        for sample in samples:
            path = stage / sample.provenance["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample.saved, path)
            records.append(dict(sample.provenance, sha256=identity.sha256_file(path)))
        result = dict(schema="ernie55-pilot-inputs-v1", status="passed", records=records,
                      input_identity_sha256=local["input_identity_sha256"], source_revision=local["source_revision"],
                      corpus_manifest_sha256=identity.sha256_file(corpus / "manifest.json"),
                      source_files={k: v["sha256"] for k, v in manifest["splits"].items()},
                      tensor_format="torch weights_only", head_anchor="argmax of final reference target logits")
        identity.atomic_json(stage / "manifest.json", result)
        stage.rename(destination)
    return identity.sha256_file(destination / "manifest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("snapshot", "corpus", "vendor-receipt", "output", "fixtures"):
        ap.add_argument("--" + key, type=Path, required=True)
    args = ap.parse_args()
    host = socket.gethostname().lower().split(".")[0]
    if host != "maxq" or not torch.cuda.is_available():
        raise RuntimeError("measured forward preflight requires the maxq CUDA worker (container hostname maxq)")
    if args.output.exists() or args.fixtures.exists():
        raise FileExistsError("preflight output or fixture directory exists")
    vendor = identity.read(args.vendor_receipt)
    inputs = identity.candidate(args.snapshot, args.corpus, identity.REPO)
    ref_hash = identity.digest(inputs["reference_files"])
    if (vendor.get("schema") != "ernie55-vllm-ernie-parity-v1" or vendor.get("status") != "passed"
            or vendor.get("source_revision") != inputs["source_revision"]
            or vendor.get("reference_files_sha256") != ref_hash):
        raise ValueError("upstream vLLM parity receipt differs from the current reference")
    model = ErnieForward.from_snapshot(args.snapshot, "cuda")
    prompts = read_prompts(args.corpus)
    witnesses = [item[0] for item in load_witnesses(args.corpus)]
    measured, checks = run_forward(model, prompts, witnesses)
    full_manifest = identity.read(args.corpus / "manifest.json")
    witness_records = [dict(id=row["id"], input_ids_sha256=row["input_ids_sha256"],
                            hessian_eligible=False, windows=windows) for row, windows in
                       zip(full_manifest["long_context_witnesses"]["records"], checks, strict=True)]
    local = dict(schema="ernie55-quant-forward-parity-v1", status="passed", host=host,
                 source_revision=inputs["source_revision"], reference_files_sha256=ref_hash,
                 input_identity_sha256=identity.parity_input_digest(inputs), vendor_receipt_sha256=identity.digest(vendor),
                 measurements=measured, error_metric="worst per-comparison relative L2; exact IDs",
                 target_layers=model.config.num_hidden_layers, draft_stages=model.config.num_mtp_stages,
                 device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                 torch_matmul_precision=torch.get_float32_matmul_precision(),
                 index_heads=model.config.index_n_heads, flashindexer=False, long_witnesses=witness_records)
    identity.verified_identity(inputs, vendor, local)
    local["fixtures_manifest_sha256"] = write_fixtures(args.fixtures, prompts, args.corpus, local)
    identity.atomic_json(args.output, local)
    print(json.dumps(dict(status="passed", receipt=str(args.output), measurements=measured)), flush=True)


if __name__ == "__main__":
    main()
