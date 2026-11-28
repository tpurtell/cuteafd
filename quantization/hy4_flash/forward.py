"""Layer-major Hy4 calibration with the independent IEEE-F32 text reference.

The explicit stage boundaries let the caller collect one Hessian and substitute
the reconstructed projection before collecting downstream activations. A layer
owns only its non-expert device weights. Engram tables and fused routed expert
weights remain host mappings, so neither needs a full device allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

REFERENCE = Path(__file__).resolve().parents[2] / "python" / "reference"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
from families.hy4_flash import model as reference
from shape_invariant import linear
from .engram import vendor_round_trip

DRAFT_BLOCK_SIZE = 6
DRAFT_SOURCE_REVISION = "ccb8d9ccaa309f71dda42e37517a7b0ee6afae8d"
Project = Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor]


def source_engram_round_trip(rows):
    """Stock Hy4 rounds fetched BF16 rows before the calibration forward resumes in F32."""
    return vendor_round_trip()(rows.to(torch.bfloat16)).float()


def native_projection(name, inputs, weight):
    return linear(inputs, weight)


@dataclass(frozen=True)
class Layer:
    index: int
    prefix: str
    weights: dict[str, torch.Tensor]

    def project(self, name, inputs, project):
        return project(self.prefix + name, inputs, self.weights[name + ".weight"])


@dataclass(frozen=True)
class AttentionFirst:
    normalized_q: torch.Tensor
    kv: torch.Tensor
    rope_key: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True)
class AttentionTap:
    """Target compressed KV and per-query group indices for DFly2.

    Empty group columns represent a densely attended prefix. Group IDs are
    stored before token expansion. Negative entries mask slots outside the
    causal prefix. Context excludes draft slots and their own KV projections.
    """

    kv: torch.Tensor
    rope_key: torch.Tensor
    groups: torch.Tensor


class Reader(reference.Checkpoint):
    """Read projected operands and token rows lazily from the pinned source."""

    def __init__(self, snapshot, device="cpu"):
        super().__init__(snapshot, device)
        self.document = json.loads((self.snapshot / "config.json").read_bytes())
        config = reference.Config.parse(self.document)
        # The released multimodal config declares the bundled drafter at the
        # root. Text dimensions and MLA geometry still come from text_config.
        drafter = {key: value for key, value in self.document.items() if key.startswith("dfly_")}
        self.config = reference.Config(config.text | drafter, config.masked_tokens)

    def load_layer(self, index):
        prefix = f"model.layers.{index}."
        weights = {}
        for name in reference.tensor_contract(self.config, index):
            leaf = name.removeprefix(prefix)
            if leaf in reference.EXPERTS or leaf == reference.TABLE:
                weights[leaf] = self.raw(name)
            else:
                weights[leaf] = self.load(name, native=leaf == "engram_layer.lookup_table")
        return Layer(index, prefix, weights)

    @staticmethod
    def draft_targets(config):
        return draft_targets(config)

    def load_draft_layer(self, index):
        prefix = f"model.dfly_layers.{index}."
        weights = {name.removeprefix(prefix): self.load(name)
                   for name in self.index if name.startswith(prefix)}
        if not weights:
            raise ValueError(f"missing Hy4 DFly2 layer {index}")
        targets, _ = self.draft_targets(self.config)
        return Layer(targets[index], prefix, weights)

    def load_draft_aux(self):
        prefix = "model.dfly_layers."
        prefixes = ("mtp_fusion.", "hidden_correction.", "final_norm.")
        return {name.removeprefix(prefix): self.load(name) for name in self.index
                if name.startswith(prefix) and name.removeprefix(prefix).startswith(prefixes)}

    def expert_weights(self, layer, expert):
        gate_up = self.raw(f"model.layers.{layer}.mlp.experts.gate_up_proj")[expert]
        down = self.raw(f"model.layers.{layer}.mlp.experts.down_proj")[expert]
        return (*gate_up.chunk(2, dim=0), down)


def _with_sink(query, key, value, allowed, sink):
    # Input tensors are [token, head, channel]; reductions use FP32.
    scores = query.transpose(0, 1) @ key.permute(1, 2, 0)
    scores = (scores * (query.shape[-1] ** -.5)).masked_fill(
        ~allowed, torch.finfo(scores.dtype).min)
    scores = torch.cat((scores, sink[:, None, None].expand(-1, len(query), -1)), dim=-1)
    probs = (scores - scores.max(dim=-1, keepdim=True).values).softmax(-1)[..., :-1]
    return (probs @ value.transpose(0, 1)).transpose(0, 1)


class Model:
    attention_arithmetic = "f32_tiled"

    @staticmethod
    def cache_weighted_content(weighted):
        """Return the content operand read by kv_b on this forward path."""
        return weighted

    def __init__(self, reader):
        self.reader = reader
        self.config = reader.config

    def load_layer(self, index):
        return self.reader.load_layer(index)

    def initial(self, tokens, embedding=None):
        if embedding is None:
            embedding = self.reader.raw(reference.OUTER_NAMES[0])
        # Gather on its owning device to avoid allocating the full vocabulary on GPU.
        values = embedding.index_select(0, tokens.to(embedding.device))
        values = values.to(self.reader.device, dtype=torch.float32)
        return values.unsqueeze(-2).expand(-1, self.config.hc_mult, -1).contiguous()

    def attention_start(self, state, tokens, layer):
        c, w, index = self.config, layer.weights, layer.index
        if index + 1 in c.engram_layer_ids:
            state = state + reference.engram(state, tokens, w, c, index, round_trip=source_engram_round_trip)
        collapsed, post = reference.hyper(state, w, "hc_attn_layer", c)
        return state, post, reference.norm(collapsed, w["input_layernorm.weight"], c.rms_norm_eps)

    def attention_first(self, inputs, layer, project=native_projection):
        c, w = self.config, layer.weights
        p = "self_attn."
        rope = c.attention(layer.index)[2]
        q = layer.project(p + "q_a_proj", inputs, project)
        q = reference.norm(q, w[p + "q_a_layernorm.weight"], c.rms_norm_eps)
        kv, kr = layer.project(p + "kv_a_proj_with_mqa", inputs, project).split(
            [c.kv_lora_rank, rope], dim=-1)
        kv = self.cache_weighted_content(reference.norm(kv, w[p + "kv_a_layernorm.weight"], c.rms_norm_eps))
        gate = layer.project(p + "linear_gate", inputs, project).sigmoid()
        return AttentionFirst(q, kv, kr, gate)

    rotate = staticmethod(reference.rotate)
    rotate_tail = staticmethod(reference.rotate_tail)

    def rope_base(self, index):
        return self.config.rope_parameters[self.config.layer_types[index]]["rope_theta"]

    def attention_expand(self, projected, layer, positions, project=native_projection,
                         *, context=None):
        """Expand q_b and kv_b. DFly2 uses the same draft kv_b on shared latents."""
        c, w = self.config, layer.weights
        h, nope, rope, vdim = c.attention(layer.index)
        p = "self_attn."
        base = self.rope_base(layer.index)
        cos, sin = reference.phase(positions, rope, base)
        q = layer.project(p + "q_b_proj", projected.normalized_q, project)
        q = self.rotate_tail(q.reshape(len(q), h, nope + rope), cos, sin)
        kr = self.rotate(projected.rope_key.unsqueeze(1), cos, sin).squeeze(1)
        kv = projected.kv
        if context is not None:
            ck, cr = context
            kv = torch.cat((ck, kv), dim=0)
            kr = torch.cat((cr, kr), dim=0)
        expanded = layer.project(p + "kv_b_proj", kv, project)
        k, v = expanded.reshape(len(expanded), h, nope + vdim).split([nope, vdim], dim=-1)
        keys = torch.cat((k, kr.unsqueeze(1).expand(-1, h, -1)), dim=-1)
        return q, keys, v, kr, w[p + "learnable_sink_param"]

    def attention_output_inputs(self, inputs, first, layer, project=native_projection,
                                *, capture_tap=False):
        """Return attended gated values consumed by o_proj and the optional DFly tap."""
        c, w, index = self.config, layer.weights, layer.index
        kind = c.layer_types[index]
        pos = torch.arange(len(inputs), device=inputs.device)
        q, k, v, kr, sink = self.attention_expand(first, layer, pos, project)
        indexed = kind == reference.SPARSE and len(inputs) > c.compressed_indexer_candidate_topk
        if indexed:
            base = c.rope_parameters[kind]["rope_theta"]
            iq, ik, iw = reference.index_projections(inputs, first.normalized_q, w, c, pos, base)
        attended, group_parts = [], []
        ratio = c.compressed_indexer_compress_ratio
        rows = reference.ROWS
        for start in range(0, len(inputs), rows):
            stop = min(start + rows, len(inputs))
            first_key = max(0, start + 1 - c.sliding_window) if kind == reference.SLIDING else 0
            width = start + rows - first_key
            query = F.pad(q[start:stop], (0, 0, 0, 0, 0, rows - (stop - start)))
            keys = F.pad(k[first_key:stop], (0, 0, 0, 0, 0, width - (stop - first_key)))
            values = F.pad(v[first_key:stop], (0, 0, 0, 0, 0, width - (stop - first_key)))
            key_id = torch.arange(first_key, start + rows, device=inputs.device)
            query_id = torch.arange(start, start + rows, device=inputs.device)
            distance = query_id[:, None] - key_id
            allowed = (distance >= 0) & (key_id < stop)
            if kind == reference.SLIDING:
                allowed &= distance < c.sliding_window
            elif indexed:
                current_group = pos[start:stop] // ratio
                groups = reference.recall_groups(iq[start:stop], ik, iw[start:stop],
                                                 current_group, c.compressed_indexer_group_topk)
                slots = (groups[..., None] * ratio + torch.arange(ratio, device=inputs.device)).flatten(1)
                slots = slots.clamp_max(len(inputs) - 1).clamp_max(width)
                selected = torch.zeros((rows, width + 1), dtype=torch.bool, device=inputs.device)
                selected[:stop - start].scatter_(1, slots, True)
                allowed &= selected[:, :width]
                if capture_tap:
                    valid = groups <= current_group[:, None]
                    group_parts.append(groups.masked_fill(~valid, -1).to(torch.int32))
            attended.append(_with_sink(query, keys, values, allowed, sink)[:stop - start])
        gated = torch.cat(attended).flatten(1) * first.gate
        tap = None
        if capture_tap:
            if kind != reference.SPARSE:
                raise ValueError("Hy4 DFly2 supports only sparse attention target taps")
            groups = torch.cat(group_parts) if group_parts else pos.new_empty((len(inputs), 0), dtype=torch.int32)
            tap = AttentionTap(first.kv, kr, groups)
        return gated, tap

    def attention_finish(self, inputs, layer, project=native_projection):
        output = layer.project("self_attn.o_proj", inputs, project)
        weight = layer.weights["self_attn.post_proj_norm.weight"]
        return reference.norm(output, weight, self.config.rms_norm_eps)

    def attention(self, inputs, layer, project=native_projection, *, capture_tap=False, **index_args):
        first = self.attention_first(inputs, layer, project)
        output_inputs, tap = self.attention_output_inputs(inputs, first, layer, project,
                                                         capture_tap=capture_tap, **index_args)
        return self.attention_finish(output_inputs, layer, project), tap

    def mlp_start(self, residual, attention_post, output, layer):
        state = attention_post * output.unsqueeze(1) + residual
        collapsed, post = reference.hyper(state, layer.weights, "hc_mlp_layer", self.config)
        inputs = reference.norm(collapsed, layer.weights["post_attention_layernorm.weight"],
                                self.config.rms_norm_eps)
        return state, post, inputs

    @staticmethod
    def ffn_parts(inputs, layer, prefix, project=native_projection):
        gate = layer.project(prefix + "gate_proj", inputs, project)
        up = layer.project(prefix + "up_proj", inputs, project)
        active = F.silu(gate) * up
        return layer.project(prefix + "down_proj", active, project)

    def routing_scores(self, inputs, layer):
        return reference.routing_scores(inputs, layer.weights, self.config)

    def moe_route(self, inputs, layer, project=native_projection):
        # Router and shared expert see full-width states. Only routed experts
        # see the hidden -> latent projection.
        ids, gamma = reference.routing(inputs, layer.weights, self.config)
        latent = layer.project("mlp.routed_expert_down_proj", inputs, project)
        return latent, ids, gamma

    def routed_expert_weights(self, index, latent, layer):
        fused = layer.weights[reference.EXPERTS[0]][index].to(latent.device, dtype=latent.dtype)
        down = layer.weights[reference.EXPERTS[1]][index].to(latent.device, dtype=latent.dtype)
        return (*fused.chunk(2, dim=0), down)

    def expert(self, index, latent, layer, project=native_projection):
        gate, up, down = self.routed_expert_weights(index, latent, layer)
        prefix = layer.prefix + f"mlp.experts.{index}."
        gate = project(prefix + "gate_proj", latent, gate)
        up = project(prefix + "up_proj", latent, up)
        limit = self.config.swiglu_limit
        activated = F.silu(gate.clamp_max(limit)) * up.clamp(-limit, limit)
        return project(prefix + "down_proj", activated, down)

    @staticmethod
    def expert_rows(ids, gamma, expert):
        """Expert token IDs and post-down routing weights in corpus order."""
        rows, slots = torch.where(ids == expert)
        return rows, gamma[rows, slots]

    def mlp_finish(self, state, post, inputs, layer, routed=None, project=native_projection):
        c = self.config
        if c.mlp_layer_types[layer.index] == "dense":
            if routed is not None:
                raise ValueError("dense Hy4 layer cannot accept routed expert outputs")
            result = self.ffn_parts(inputs, layer, "mlp.", project)
            weight = layer.weights["mlp.post_fc2_norm.weight"]
        else:
            if routed is None:
                raise ValueError("Hy4 MoE requires the gamma-weighted latent sum")
            result = layer.project("mlp.routed_expert_up_proj", routed, project)
            result = result + self.ffn_parts(inputs, layer, "mlp.shared_experts.", project)
            weight = layer.weights["mlp.combine_output_norm.weight"]
        return post * reference.norm(result, weight, c.rms_norm_eps).unsqueeze(1) + state

    def forward_layer(self, state, tokens, layer, project=native_projection, *, capture_tap=False, **index_args):
        residual, attention_post, inputs = self.attention_start(state, tokens, layer)
        output, tap = self.attention(inputs, layer, project, capture_tap=capture_tap, **index_args)
        state, post, inputs = self.mlp_start(residual, attention_post, output, layer)
        routed = None
        if self.config.mlp_layer_types[layer.index] == "sparse":
            latent, ids, gamma = self.moe_route(inputs, layer, project)
            routed = torch.zeros_like(latent)
            for expert in ids.unique().tolist():
                rows, weights = self.expert_rows(ids, gamma, expert)
                routed[rows] += self.expert(expert, latent[rows], layer, project) * weights[:, None]
        return self.mlp_finish(state, post, inputs, layer, routed, project), tap

    def head_features(self, states, weights=None):
        if weights is None:
            weights = {name: self.reader.load(name) for name in reference.OUTER_NAMES[1:5]}
        return reference.head_features(states, weights, self.config)

    def head(self, features, project=native_projection, weight=None):
        if weight is None:
            weight = self.reader.load("lm_head.weight")
        return project("lm_head", features.float(), weight.float())


def draft_targets(config):
    """Check the released DFly2 topology before accepting target taps."""
    kv = tuple(config.dfly_target_kv_layers)
    selections = tuple(config.dfly_target_index_layer)
    masks = config.dfly_mask_token_id
    n = config.dfly_num_hidden_layers
    if (config.dfly_mtp_fusion != "slot0" or config.dfly_context_local_window not in (None, 0)
            or config.dfly_enable_hidden_correction is not True
            or config.dfly_target_index_topk != config.compressed_indexer_candidate_topk
            or n != len(kv) or n != len(selections) or len(set(kv)) != n or len(set(selections)) != n
            or not 0 <= masks < config.vocab_size
            or any(not 0 <= layer < config.num_hidden_layers or config.layer_types[layer] != reference.SPARSE
                   for layer in kv + selections)):
        raise ValueError("unsupported Hy4 DFly2 configuration")
    return kv, selections


def recalled_context(tap, end, ratio):
    """Reuse the last committed target token's selection for every draft slot."""
    if not 1 <= end <= len(tap.kv):
        raise ValueError("DFly2 context boundary lies outside the tapped target states")
    groups = tap.groups[end - 1].long()
    if not groups.numel():
        return torch.arange(end, device=tap.kv.device)
    groups = groups[groups >= 0]
    ids = (groups[:, None] * ratio + torch.arange(ratio, device=groups.device)).flatten()
    # The last, incomplete group has only the actual context slots. Gathering
    # the official clamped duplicates would assign the last token extra mass.
    return ids[ids < end]


class Drafter:
    """Torch implementation of the pinned SGLang DFly2 proposal forward.

    The tap represents a text prefix of length end. Slot zero is the uncached
    target bonus token, not a replacement for the final committed token.
    """

    linear = staticmethod(linear)
    block_size = DRAFT_BLOCK_SIZE
    target_layers = staticmethod(draft_targets)

    def selection_ratio(self):
        return self.config.compressed_indexer_compress_ratio

    def __init__(self, model, *, layers=None, aux=None):
        self.model = model
        self.config = model.config
        self.kv_layers, self.index_layers = self.target_layers(self.config)
        self.layers = tuple(layers) if layers is not None else tuple(
            model.reader.load_draft_layer(i) for i in range(len(self.kv_layers)))
        if tuple(layer.index for layer in self.layers) != self.kv_layers:
            raise ValueError("DFly2 layers disagree with their target KV geometry")
        self.aux = model.reader.load_draft_aux() if aux is None else aux

    def embed(self, ids, embedding):
        values = embedding.index_select(0, ids.to(embedding.device))
        return values.to(ids.device, dtype=torch.float32)

    def start(self, bonus, target_hidden, embedding, *, width=None):
        width = self.block_size if width is None else width
        if bonus.ndim != 0 or target_hidden.ndim != 1 or not 2 <= width <= 17:
            raise ValueError("invalid Hy4 DFly2 anchor or block width")
        w, c = self.aux, self.config
        block = torch.full((width,), c.dfly_mask_token_id, dtype=torch.long, device=bonus.device)
        block[0] = bonus
        states = self.embed(block, embedding)
        anchor = states[0]
        p = "mtp_fusion."
        joined = torch.cat((
            reference.norm(anchor, w[p + "enorm.weight"], c.rms_norm_eps),
            reference.norm(target_hidden, w[p + "hnorm.weight"], c.rms_norm_eps),
        ))
        states[0] = anchor + w[p + "gate"] * self.linear(joined, w[p + "eh_proj.weight"])
        return states

    def attention_inputs(self, hidden, stage):
        weight = self.layers[stage].weights["input_layernorm.weight"]
        return reference.norm(hidden, weight, self.config.rms_norm_eps)

    def attention_output_inputs(self, first, stage, taps, end, project=native_projection):
        kv = taps[self.kv_layers[stage]]
        selection = taps[self.index_layers[stage]]
        indices = recalled_context(selection, end, self.selection_ratio())
        if len(kv.kv) != len(selection.kv):
            raise ValueError("DFly2 KV and selection tap lengths differ")
        context = (kv.kv[indices], kv.rope_key[indices])
        pos = torch.arange(end, end + len(first.kv), device=first.kv.device)
        q, k, v, _, sink = self.model.attention_expand(first, self.layers[stage], pos, project,
                                                        context=context)
        slots = torch.arange(len(k), device=q.device) - len(indices)
        allowed = slots[None, :] <= torch.arange(len(q), device=q.device)[:, None]
        return _with_sink(q, k, v, allowed, sink).flatten(1) * first.gate

    def ffn_inputs(self, state, attended, stage):
        residual = state + attended
        weight = self.layers[stage].weights["post_attention_layernorm.weight"]
        normalized = reference.norm(residual, weight, self.config.rms_norm_eps)
        return residual, normalized

    def layer(self, state, stage, taps, end, project=native_projection):
        layer = self.layers[stage]
        inputs = self.attention_inputs(state, stage)
        first = self.model.attention_first(inputs, layer, project)
        o_inputs = self.attention_output_inputs(first, stage, taps, end, project)
        attended = self.model.attention_finish(o_inputs, layer, project)
        residual, inputs = self.ffn_inputs(state, attended, stage)
        return residual + self.model.ffn_parts(inputs, layer, "mlp.", project)

    def finish(self, hidden):
        return reference.norm(hidden, self.aux["final_norm.weight"], self.config.rms_norm_eps)

    def corrected(self, hidden, previous_embed):
        w, c = self.aux, self.config
        p = "hidden_correction."
        joined = torch.cat((
            reference.norm(hidden, w[p + "hidden_norm.weight"], c.rms_norm_eps),
            reference.norm(previous_embed, w[p + "embed_norm.weight"], c.rms_norm_eps),
        ))
        gate = self.linear(joined, w[p + "gate_proj.weight"])
        up = self.linear(joined, w[p + "up_proj.weight"])
        return hidden + self.linear(F.silu(gate) * up, w[p + "down_proj.weight"])

    def proposals(self, hidden, bonus, embedding, *, head=native_projection, head_weight=None):
        """Return the sequentially corrected FP32 logits of draft slots 1..m."""
        logits = []
        previous = bonus
        for slot in hidden[1:]:
            feature = self.corrected(slot, self.embed(previous.view(1), embedding)[0])
            row = self.model.head(feature, head, head_weight)
            logits.append(row)
            previous = row.argmax(-1)
        return torch.stack(logits)

    def forward(self, bonus, target_hidden, taps, end, embedding, project=native_projection,
                *, head=native_projection, head_weight=None, width=None):
        hidden = self.start(bonus, target_hidden, embedding, width=width)
        for stage in range(len(self.layers)):
            hidden = self.layer(hidden, stage, taps, end, project)
        return self.proposals(self.finish(hidden), bonus, embedding, head=head, head_weight=head_weight)
