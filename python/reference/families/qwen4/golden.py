#!/usr/bin/env python3
"""Golden activations for Qwen 3.8 Flash Next (qwen4_exp) from transformers' reference.

Runs one Qwen4ExpTextDecoderLayer at a time on one GPU (the routed experts do
not fit): the four hyper-connection streams (``[T, 4 * hidden]``) go in and
out of every layer as a full-sequence prefill without a cache. Gated DeltaNet
layers take the chunked path; full-attention layers run the real QSA indexer
(every token is selected up to 2051 visible tokens, so shorter prompts are
dense causal GQA) and eager attention. Positions, rotary embeddings and the
causal mask follow Qwen4ExpTextModel.forward; media windows use the pinned
Qwen4ExpModel.get_rope_index T/H/W axes and native ids for PLE. The official
BF16 tower replaces embeddings before their four HC copies. The final collapse is ``hyper_connection_mixer``
(Qwen4ExpTextGatedResidual without the injection); there is no final norm,
``lm_head`` runs in FP32.

PLE (layer 1): the n-gram ids come from the module's own hashing
(Qwen4ExpTextNGramEmbedding.forward is unchanged); only the gathered rows
are read from the 128 ``ngram_embedding.shard_N`` tensors (2,500,012 rows of
160 each, row r in shard r // 2500012). FP8 tables are
``bf16(bf16(e4m3) * weight_scale)`` (BF16 scale, round to nearest even),
BF16 tables are used as stored. The rest of Qwen4ExpTextPLELayer is the real
module code.

Weights: coordinator tensors come from ``--snapshot`` (BF16 as stored; FP8
tensors times their 128x128 block ``weight_scale_inv``). Routed experts come
from ``--experts-snapshot`` (default the same snapshot): FP8 per-expert
tensors (times their BF16 128x128 block scales) or BF16 fused
``mlp.experts.gate_up_proj`` / ``down_proj``. EXL3 checkpoints cannot be
expert sources (their dense tensors are usable as ``--snapshot``). Routed
experts are summed in FP32 and rounded once (transformers' eager experts sum
in BF16, which moves results on rounding-level changes: compare engines by
NLL as well as agreement); the softmax top-10 router is the module as is.
Run with ``PYTHONPATH=<cuteafd>/third_party/transformers/src`` (the pinned
transformers carries qwen4_exp) and ``USE_HUB_KERNELS=0``.

Writes the raw files ``qwen4-golden`` reads:

  tokens.bin      i32  [T]
  layerNN.bin     bf16 [T, 4, hidden]   hyper-connection streams after layer NN
  logits.bin      f32  [T, vocab]
  meta.json

  golden.py --snapshot SNAP [--experts-snapshot SNAP] --text-file prompt.txt --out DIR
            [--layers 0 1 ...] [--stop-after N] [--max-tokens T]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Purge freed CPU staging pages immediately on the ARM Torch allocator.
import os
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")

import torch
from safetensors import safe_open

PREFIX = "model.language_model."
SHARD_ROWS = 2_500_012

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fidelity_windows import CheckpointStorage, release_checkpoint, load_set, write_scored_logits, finish_golden, verify_snapshot, log_checkpoint_reads


def run_windows(a, config, ref, dense, experts_src, create_causal_mask):
    manifest = load_set(a.windows, "qwen4")
    from qwen_media import configure_qsa_eager
    rows = configure_qsa_eager(ref, manifest, getattr(a, "qsa_key_rows", None))
    geometry = {"qsa_key_rows": rows,
                "qsa_padding": "zero-key/value-tail; additive-mask-negative-infinity"}
    fixed_gdn_rows = getattr(a, "fixed_gdn_rows", None)
    if fixed_gdn_rows is not None:
        from qwen_media import fixed_gdn_forward
        fixed_gdn_forward(None, fixed_gdn_rows)  # Validate before qualification or weight reads.
        if any(len(w["tokens"]) > fixed_gdn_rows for w in manifest["windows"]):
            raise ValueError("panel exceeds fixed GDN geometry")
        geometry.update(gdn_sequence_rows=fixed_gdn_rows,
                        gdn_padding="zero-hidden-tail; no-cache; cropped-output")
    from fidelity_media import require_media_flag
    media = require_media_flag(manifest, getattr(a, "media", False), "qwen4")
    identity = verify_snapshot(manifest, a.snapshot)
    if media:
        from qwen_media import snapshot_identity, validate_window
        identity.update(snapshot_identity(a.snapshot))
        for window in manifest["windows"]:
            validate_window(window)
    from shape_invariant import qualify
    proof = qualify(a, manifest, lambda probe: run_windows(probe, config, ref, dense, experts_src, create_causal_mask))
    if getattr(a, "prefix_only", False) and not getattr(a, "_prefix_probe", False):
        return
    if PREFIX + "norm.weight" in dense:
        raise ValueError("unexpected final norm: model feeds stream mixer into lm_head")
    started, times, rows, states = time.time(), [], [], []
    features, window_positions = {}, []
    if media:
        from qwen_media import window_features, rope_positions, inject_embeddings
        features, _ = window_features(a, manifest)
    with torch.inference_mode():
        embed_weight = dense.get(PREFIX + "embed_tokens.weight")
        for w in manifest["windows"]:
            ids = torch.tensor([w["tokens"]], device="cuda")
            embed = torch.nn.functional.embedding(ids, embed_weight)
            if media:
                states.append(inject_embeddings(embed, w, features, config.hc_count).cpu())
                window_positions.append(rope_positions(w, a.snapshot, device="cpu"))
            else:
                states.append(embed.repeat(1, 1, config.hc_count).cpu())
                window_positions.append(torch.arange(len(w["tokens"])).view(1, 1, -1).expand(4, 1, -1))
        del embed_weight, ids, embed, features
        rotary = ref.Qwen4ExpTextRotaryEmbedding(config=config).cuda()
        memory = CheckpointStorage(torch.cuda, dense, experts_src)
        for layer_id in range(config.num_hidden_layers):
            start = time.time()
            read_start = time.monotonic()
            read_before = dense.read_bytes + experts_src.read_bytes if dense is not experts_src else dense.read_bytes
            read_sources = (dense, experts_src) if dense is not experts_src else (dense,)
            kind = config.layer_types[layer_id]
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.Qwen4ExpTextDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            skip = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
            if layer.ple is not None:
                emb = layer.ple.ple_embedding
                table_rows, dim = emb.ngram_embedding.num_embeddings, emb.ngram_embedding.embedding_dim
                if table_rows != 128 * SHARD_ROWS:
                    raise ValueError("unexpected n-gram table shape")
                emb.ngram_embedding = torch.nn.Identity()
                del emb
            layer = layer.to_empty(device="cuda")
            if layer.ple is not None:
                prefix = f"{PREFIX}layers.{layer_id}.ple.ple_embedding.ngram_embedding."
                layer.ple.ple_embedding.ngram_embedding = LazyNgramTable(dense, prefix, table_rows, dim)
            load_module(layer, dense, f"{PREFIX}layers.{layer_id}.", skip)
            load_experts(layer.mlp.experts, experts_src, f"{PREFIX}layers.{layer_id}.")
            log_checkpoint_reads(f"layer {layer_id} load", read_sources, read_before, read_start)
            layer.eval()
            if fixed_gdn_rows is not None and kind == "linear_attention":
                layer.linear_attn.forward = fixed_gdn_forward(layer.linear_attn.forward, fixed_gdn_rows)
            if layer.ple is not None:
                emb = layer.ple.ple_embedding
                expected = ref._build_layer_multipliers(emb.unigram_vocab_size, emb.ngram_size, emb.ple_layer_index, emb.seed)
                if not (torch.equal(emb.layer_multipliers.cpu(), expected)
                        and emb.ngram_heads_vocab_sizes.tolist() == emb.head_vocab_sizes
                        and emb.ngram_heads_offsets.tolist() == emb.head_offsets):
                    raise ValueError("checkpoint n-gram hash buffers differ from module")
                del emb, expected
            for i, w in enumerate(manifest["windows"]):
                h = states[i].cuda()
                ids = torch.tensor([w["tokens"]], device="cuda")
                positions = window_positions[i].cuda()
                # Masks/rotary depend on shape, not on the embedding values; one
                # hidden-width view matches the original embedding's geometry.
                embed_shape = h[..., :config.hidden_size]
                causal = create_causal_mask(config=config, inputs_embeds=embed_shape, attention_mask=None,
                    past_key_values=None, position_ids=positions[0], allow_is_causal_skip=False)
                position_embeddings = rotary(embed_shape, positions[1:])
                h = layer(h, position_embeddings=position_embeddings, attention_mask=causal, conv_mask=None,
                          past_key_values=None, ple_input_ids=ids)
                if a.layers is not None and layer_id in a.layers:
                    folder = a.out / "windows" / w["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"layer{layer_id:02d}.bin").write_bytes(h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
                states[i] = h.cpu()
                del h, ids, positions, embed_shape, causal, position_embeddings
            log_checkpoint_reads(f"layer {layer_id} total (includes lazy PLE)", read_sources, read_before, read_start)
            del layer
            memory.release()
            memory.check(f"layer {layer_id}")
            times.append(time.time() - start)
            print(f"layer {layer_id} ({kind}) {times[-1]:.1f}s ({len(states)} windows)", flush=True)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            mixer = ref.Qwen4ExpTextGatedResidual(config, use_combine=False)
        torch.set_default_dtype(torch.float32)
        mixer = mixer.to_empty(device="cuda")
        load_module(mixer, dense, PREFIX + "hyper_connection_mixer.", set())
        head = dense.get("lm_head.weight").float()
        for i, w in enumerate(manifest["windows"]):
            h = states[i][:, w["score_from"] - 1:len(w["tokens"]) - 1].cuda()
            logits = torch.nn.functional.linear(mixer(h)[0].float(), head)
            rows.append(write_scored_logits(a.out, w, logits.cpu().numpy()))
            states[i] = None
            del h, logits
    finish_golden(a.out, manifest, rows, snapshot=str(a.snapshot),
        experts_snapshot=str(a.experts_snapshot or a.snapshot),
        reference="transformers qwen4_exp (pinned 62d7ebd7; eager, FP32 routed sum, lazy PLE rows)",
        seconds=time.time() - started, seconds_per_layer=times, snapshot_identity=identity,
        **({"reference_geometry": geometry} if geometry is not None else {}),
        prefix_qualification=proof)


class Weights:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}
        self.read_bytes = 0
        self.read_seconds = 0.0

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def handle(self, name: str):
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.files[shard]

    def raw(self, name: str) -> torch.Tensor:
        start = time.monotonic()
        value = self.handle(name).get_tensor(name).clone()
        self.read_bytes += value.numel() * value.element_size()
        self.read_seconds += time.monotonic() - start
        return value

    def get(self, name: str, device: str = "cuda") -> torch.Tensor:
        """BF16/FP32/int tensors as stored; FP8 weights times their 128x128 block scales."""
        value = self.raw(name).to(device)
        if value.dtype != torch.float8_e4m3fn:
            return value
        scale = self.raw(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
        rows, cols = value.shape
        grown = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
        return (value.float() * grown).bfloat16()


class LazyNgramTable(torch.nn.Module):
    """Stands in for ``ngram_embedding`` (nn.Embedding): gathers only the requested rows."""

    def __init__(self, weights: Weights, prefix: str, rows: int, dim: int):
        super().__init__()
        self.w, self.prefix, self.rows, self.dim = weights, prefix, rows, dim
        scale_name = prefix + "weight_scale"
        self.scale = weights.raw(scale_name) if scale_name in weights else None
        self.weight = torch.empty(0, device="cuda")  # plain attribute: the module reads .weight.device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu()
        if int(flat.max()) >= self.rows or int(flat.min()) < 0:
            raise ValueError("n-gram row id out of range")
        unique, inverse = torch.unique(flat, return_inverse=True)
        out = torch.empty(len(unique), self.dim, dtype=torch.bfloat16)
        shard_of = unique // SHARD_ROWS
        for shard in torch.unique(shard_of).tolist():
            pick = (shard_of == shard).nonzero().flatten()
            name = f"{self.prefix}shard_{shard}.weight"
            # PLE may touch all 128 shards in one layer; do not cache these handles.
            with safe_open(str(self.w.snapshot / self.w.index[name]), framework="pt", device="cpu") as handle:
                view = handle.get_slice(name)
                for i in pick.tolist():
                    r = int(unique[i]) - shard * SHARD_ROWS
                    read_start = time.monotonic()
                    row = view[r:r + 1]
                    self.w.read_bytes += row.numel() * row.element_size()
                    if row.dtype == torch.float8_e4m3fn:
                        if self.scale is None:
                            raise KeyError(f"{self.prefix}weight_scale: FP8 table without its scale")
                        row = row.to(torch.bfloat16) * self.scale.to(torch.bfloat16)
                    elif row.dtype != torch.bfloat16:
                        raise TypeError(f"{name}: unsupported table dtype {row.dtype}")
                    out[i] = row[0]
                    self.w.read_seconds += time.monotonic() - read_start
                del row, view
        return out[inverse].reshape(*ids.shape, self.dim).to(ids.device)


def load_experts(experts, src: Weights, prefix: str) -> None:
    fused = prefix + "mlp.experts.gate_up_proj"
    with torch.no_grad():
        if fused in src:
            experts.gate_up_proj.copy_(src.get(fused).to(torch.bfloat16))
            experts.down_proj.copy_(src.get(prefix + "mlp.experts.down_proj").to(torch.bfloat16))
            release_checkpoint(torch.cuda, src)
            return
        first = prefix + "mlp.experts.0."
        if first + "gate_proj.trellis" in src:
            raise ValueError("EXL3 expert tensors cannot be a golden expert source (use FP8 or BF16)")
        for e in range(experts.gate_up_proj.shape[0]):
            base = f"{prefix}mlp.experts.{e}."
            experts.gate_up_proj[e].copy_(torch.cat([src.get(base + "gate_proj.weight"),
                                                     src.get(base + "up_proj.weight")], 0))
            experts.down_proj[e].copy_(src.get(base + "down_proj.weight"))
    release_checkpoint(torch.cuda, src)


def load_module(module: torch.nn.Module, dense: Weights, prefix: str, skip: set[str]) -> None:
    for key, param in list(module.named_parameters()) + list(module.named_buffers()):
        if key in skip:
            continue
        name = prefix + key
        if name not in dense:
            raise KeyError(f"{name}: no checkpoint tensor for {key}")
        with torch.no_grad():
            param.copy_(dense.get(name).reshape(param.shape).to(param.dtype))
    release_checkpoint(torch.cuda, dense)


def experts_fp32(self, hidden_states, top_k_index, top_k_weights):
    """Qwen4ExpTextExperts.forward with the routed sum in FP32 and one BF16 rounding."""
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        expert = expert[0]
        slot, token = torch.where(mask[expert])
        gate, up = torch.nn.functional.linear(hidden_states[token], self.gate_up_proj[expert]).chunk(2, dim=-1)
        out = torch.nn.functional.linear(self.act_fn(gate) * up, self.down_proj[expert])
        final.index_add_(0, token, out.float() * top_k_weights[token, slot, None].float())
    return final.to(hidden_states.dtype)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True, help="coordinator weights, config and tokenizer")
    p.add_argument("--experts-snapshot", type=Path,
                   help="routed experts (FP8 per-expert or BF16 fused; not EXL3); default --snapshot")
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--windows", type=Path, help="pinned fidelity set; scored-row logits; streams saved only with --layers")
    p.add_argument("--text-file", type=Path, help="prompt text from a file")
    p.add_argument("--max-tokens", type=int, help="keep the first T tokens")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose streams to save (default all)")
    p.add_argument("--stop-after", type=int, help="run only layers 0..N (no logits)")
    p.add_argument("--prefix-only", action="store_true", help="qualify reference prefix arithmetic without running the full panel")
    p.add_argument("--media", action="store_true", help="inject the pinned official BF16 Qwen tower for media windows")
    p.add_argument("--media-root", type=Path, help="fixture root (default: windows manifest directory)")
    p.add_argument("--media-features-out", type=Path, help="write immutable BF16 paired-probe feature files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--qsa-key-rows", type=int, help="fixed QSA key extent; must match the set when specified")
    p.add_argument("--fixed-gdn-rows", type=int,
                   help="diagnostic fixed no-cache GDN geometry (multiple of 64); requires --windows")
    a = p.parse_args()
    if a.qsa_key_rows is not None and not a.windows:
        p.error("--qsa-key-rows requires --windows")
    if a.fixed_gdn_rows is not None and not a.windows:
        p.error("--fixed-gdn-rows requires --windows")
    if a.prefix_only and not a.windows:
        p.error("--prefix-only requires --windows")
    if (a.media or a.media_root or a.media_features_out) and not (a.media and a.windows):
        p.error("media options require --media --windows")

    from tokenizers import Tokenizer
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref

    ref.Qwen4ExpTextExperts.forward = experts_fp32
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from shape_invariant import install
    install()
    from shape_invariant import install_eager
    install_eager(ref)
    config = AutoConfig.from_pretrained(a.snapshot).text_config
    config._attn_implementation = "eager"
    if a.windows:
        if a.text or a.text_file or a.max_tokens or a.stop_after is not None:
            p.error("--windows cannot be combined with legacy text/truncation/stop options")
        a.out.mkdir(parents=True, exist_ok=True)
        dense = Weights(a.snapshot)
        experts_src = Weights(a.experts_snapshot) if a.experts_snapshot else dense
        run_windows(a, config, ref, dense, experts_src, create_causal_mask)
        return
    text = a.text_file.read_text() if a.text_file else a.text
    tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    if a.max_tokens:
        tokens = tokens[:a.max_tokens]
    dense = Weights(a.snapshot)
    experts_src = Weights(a.experts_snapshot) if a.experts_snapshot else dense
    if PREFIX + "norm.weight" in dense:
        raise ValueError("unexpected final norm: this model feeds the stream mixer straight into lm_head")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tokens.bin").write_bytes(torch.tensor(tokens, dtype=torch.int32).numpy().tobytes())
    n_layers = config.num_hidden_layers
    layers = n_layers if a.stop_after is None else min(a.stop_after + 1, n_layers)
    save = set(range(layers)) if not a.layers else set(a.layers)
    t = len(tokens)
    ids = torch.tensor([tokens], device="cuda")
    timings = []
    with torch.inference_mode():
        embed = torch.nn.functional.embedding(ids, dense.get(PREFIX + "embed_tokens.weight"))
        # Qwen4ExpTextModel.forward: 4 position rows (text, then the 3 mRoPE axes).
        position_ids = torch.arange(t, device="cuda").view(1, 1, -1).expand(4, 1, -1)
        text_positions, mrope_positions = position_ids[0], position_ids[1:]
        causal = create_causal_mask(config=config, inputs_embeds=embed, attention_mask=None,
                                    past_key_values=None, position_ids=text_positions,
                                    allow_is_causal_skip=False)
        conv_mask = None  # create_recurrent_attention_mask: None for an unpadded prompt
        rotary = ref.Qwen4ExpTextRotaryEmbedding(config=config).cuda()
        position_embeddings = rotary(embed, mrope_positions)
        h = embed.repeat(1, 1, config.hc_count)
        memory = CheckpointStorage(torch.cuda, dense, experts_src)
        for layer_id in range(layers):
            start = time.time()
            kind = config.layer_types[layer_id]
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.Qwen4ExpTextDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            skip = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
            if layer.ple is not None:
                emb = layer.ple.ple_embedding
                rows, dim = emb.ngram_embedding.num_embeddings, emb.ngram_embedding.embedding_dim
                if rows != 128 * SHARD_ROWS:
                    raise ValueError(f"n-gram table has {rows} rows, expected 128 shards of {SHARD_ROWS}")
                emb.ngram_embedding = torch.nn.Identity()  # dropped before allocation
                del emb
            layer = layer.to_empty(device="cuda")
            if layer.ple is not None:
                table_prefix = f"{PREFIX}layers.{layer_id}.ple.ple_embedding.ngram_embedding."
                layer.ple.ple_embedding.ngram_embedding = LazyNgramTable(dense, table_prefix, rows, dim)
            load_module(layer, dense, f"{PREFIX}layers.{layer_id}.", skip)
            load_experts(layer.mlp.experts, experts_src, f"{PREFIX}layers.{layer_id}.")
            layer.eval()
            if layer.ple is not None:
                emb = layer.ple.ple_embedding
                expected = ref._build_layer_multipliers(emb.unigram_vocab_size, emb.ngram_size,
                                                        emb.ple_layer_index, emb.seed)
                if not (torch.equal(emb.layer_multipliers.cpu(), expected)
                        and emb.ngram_heads_vocab_sizes.tolist() == emb.head_vocab_sizes
                        and emb.ngram_heads_offsets.tolist() == emb.head_offsets):
                    raise ValueError("checkpoint n-gram hash buffers differ from the module's construction")
                del emb, expected
            h = layer(h, position_embeddings=position_embeddings, attention_mask=causal, conv_mask=conv_mask,
                      past_key_values=None, ple_input_ids=ids)
            if layer_id in save:
                (a.out / f"layer{layer_id:02d}.bin").write_bytes(
                    h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del layer
            memory.release()
            memory.check(f"layer {layer_id}")
            timings.append(time.time() - start)
            print(f"layer {layer_id} ({kind}) {timings[-1]:.1f}s", flush=True)
        if layers < n_layers:
            return
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            mixer = ref.Qwen4ExpTextGatedResidual(config, use_combine=False)
        torch.set_default_dtype(torch.float32)
        mixer = mixer.to_empty(device="cuda")
        load_module(mixer, dense, PREFIX + "hyper_connection_mixer.", set())
        final = mixer(h)
        head = dense.get("lm_head.weight").float()
        logits = torch.nn.functional.linear(final[0].float(), head)
        del head
        (a.out / "logits.bin").write_bytes(logits.contiguous().cpu().numpy().tobytes())
        argmax = logits.argmax(-1)
        next_ok = (argmax[:-1] == ids[0, 1:]).float().mean().item()
        nll_sum = 0.0
        for first in range(0, t - 1, 512):
            last = min(first + 512, t - 1)
            lp = torch.log_softmax(logits[first:last].double(), -1)
            nll_sum -= lp.gather(1, ids[0, first + 1:last + 1, None]).sum().item()
        nll = nll_sum / max(t - 1, 1)
    (a.out / "meta.json").write_text(json.dumps({
        "tokens": t, "snapshot": str(a.snapshot), "experts_snapshot": str(a.experts_snapshot or a.snapshot),
        "reference": "transformers qwen4_exp (pinned 62d7ebd7; eager, FP32 routed sum, lazy PLE rows)",
        "argmax_last": int(argmax[-1]), "next_token_accuracy": next_ok, "mean_nll": nll,
        "seconds_per_layer": [round(s, 2) for s in timings],
    }, indent=1))
    print(f"argmax of last position: {int(argmax[-1])}; next-token accuracy {next_ok:.3f}; mean NLL {nll:.4f}")


if __name__ == "__main__":
    main()
