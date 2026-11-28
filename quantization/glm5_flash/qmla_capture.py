"""Content-only QMLA capture from the pinned GLM-5.3-Flash attention modules."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager

import torch
import torch.nn.functional as F

from qmla.capture import (
    CapturedLayerParams,
    CapturedSequence,
    MlaLayerGeometry,
    QmlaCaptureAdapter,
    SparseSelection,
    unit_rms_norm,
    native_masked_attention,
)
from shape_invariant import ROWS


def glm5_flash_mla_geometries(config) -> tuple[MlaLayerGeometry, ...]:
    if config.model_type != "glm5_next_text" or config.qk_rope_head_dim != 0:
        raise ValueError("GLM Flash QMLA requires a pure-NoPE glm5_next_text checkpoint")
    if (config.num_attention_heads != config.num_key_value_heads or config.attention_bias
            or not config.index_kpool_always_select_tail or config.index_topk % config.index_kpool):
        raise ValueError("unsupported GLM Flash attention or index pool geometry")
    if len(config.layer_types) != config.num_hidden_layers or len(config.indexer_types) != config.num_hidden_layers:
        raise ValueError("incomplete GLM Flash layer inventory")
    result = []
    for index, kind in enumerate(config.layer_types):
        if kind == "linear_attention":
            continue
        if kind != "deepseek_sparse_attention" or config.indexer_types[index] != "full":
            raise ValueError(f"unsupported attention/indexer at GLM Flash layer {index}")
        geom = MlaLayerGeometry(
            layer_id=index, kv_source_layer=index, kind="dsa", heads=config.num_attention_heads,
            kv_lora_rank=config.kv_lora_rank, qk_nope_dim=config.qk_nope_head_dim,
            qk_rope_dim=0, v_head_dim=config.v_head_dim, rope_kind="none",
            rms_norm_eps=config.rms_norm_eps, attention_scale=config.qk_nope_head_dim ** -0.5,
        )
        geom.validate()
        result.append(geom)
    if not result:
        raise ValueError("GLM Flash checkpoint has no eligible MLA layers")
    return tuple(result)


def _pool_selection(indices, indexer):
    """Undo the official indexer's complete-pool expansion and preserve its partial tail."""
    seq_len = indices.shape[0]
    pool = indexer.index_kpool
    complete = min(seq_len // pool, indexer.index_topk // pool)
    group_starts = indices[:, :complete * pool:pool]
    if pool > 1:
        group_starts = torch.cat((group_starts, indices[:, complete * pool:complete * pool + 1]), dim=1)
    return SparseSelection(group_starts // pool, pool, indexer.index_topk)


def _tiled_selection(x, q_resid, indexer):
    """Score the same complete pools and append the visible tail with bounded query tiles."""
    device, count = x.device, x.shape[1]
    pool = indexer.index_kpool
    key = indexer.k_norm(indexer.wk(x))
    gate = F.linear(x, indexer.index_kpool_compress_gate)
    packed = torch.cat((key, gate, key.new_ones((1, count, 1))), dim=-1)
    keys, pool_indices, _ = indexer.get_pooled_states(packed)
    ends = pool_indices[0, :, -1]
    query = indexer.wq_b(q_resid).view(1, count, indexer.n_heads, indexer.head_dim)
    weights = indexer.weights_proj(x).float() * indexer.n_heads ** -0.5
    k = min(indexer.index_topk // pool, ends.numel())
    groups = []
    for start in range(0, count, ROWS):
        stop = min(start + ROWS, count)
        slots = torch.arange(start, stop, device=device)
        scores = torch.matmul(query[:, start:stop].float(), keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * indexer.softmax_scale)
        scores = torch.matmul(weights[:, start:stop].unsqueeze(-2), scores)[0, :, 0]
        visible = ends[None, :] <= slots[:, None]
        selected = scores.masked_fill(~visible, torch.finfo(scores.dtype).min).topk(k, dim=-1).indices
        selected = selected.masked_fill(~visible.gather(-1, selected), -1)
        if pool > 1:
            tail = (slots // pool).masked_fill((slots + 1) % pool == 0, -1)
            selected = torch.cat((selected, tail[:, None]), dim=1)
        groups.append(selected.to(torch.int32))
    return SparseSelection(torch.cat(groups), pool, indexer.index_topk)


def _arithmetic(layer):
    dtype = layer.kv_b_proj.weight.dtype
    backend = layer.config._attn_implementation
    if dtype == torch.float32:
        return "f32_tiled"
    if dtype != torch.bfloat16 or backend not in ("sdpa", "eager"):
        raise ValueError(f"unsupported GLM Flash attention arithmetic {dtype} {backend}")
    return f"bf16_{backend}"


def _tiled_forward(x, attention):
    """Pinned eager equations with query tiling for long records; retain native tensor dtypes."""
    count = x.shape[1]
    q_resid = attention.q_a_layernorm(attention.q_a_proj(x))
    query = attention.q_b_proj(q_resid).reshape(count, attention.num_heads, attention.qk_nope_head_dim)
    raw = attention.kv_a_proj_with_mqa(x)
    kv = attention.kv_a_layernorm(raw).unsqueeze(1)
    keys, values = attention.expand_kv(kv, kv[..., :0])
    selection = _tiled_selection(x, q_resid, attention.indexer)
    output = []
    backend = attention.config._attn_implementation
    for start in range(0, count, ROWS):
        stop = min(start + ROWS, count)
        slots = torch.arange(start, stop, device=x.device)
        allowed = selection.tile_mask(start, stop, 0, count, slots)
        allowed &= torch.arange(count, device=x.device)[None, :] <= slots[:, None]
        attended = native_masked_attention(query[start:stop], keys[0].transpose(0, 1),
                                           values[0].transpose(0, 1), allowed,
                                           attention.scaling, f"bf16_{backend}")
        output.append(attended.reshape(stop - start, -1))
    out = attention.o_proj(torch.cat(output))
    return raw[0], query, selection, out


class Glm5FlashCaptureAdapter(QmlaCaptureAdapter):
    family_id = "glm5_flash"
    has_rope = False
    rope_kind = "none"
    dsa_indexer_depends_on_mla_latents = False
    dsa_indexer_note = (
        "Glm5NextTextIndexer reads the layer input and q_a_layernorm(q_a_proj(input)); "
        "its wk, gate, wq_b and head weights are independent of the stored MLA content. "
        "With native indexer keys, intra-layer selection drift is structurally zero."
    )

    def __init__(self, config):
        self.config = config
        self._geometries = {g.layer_id: g for g in glm5_flash_mla_geometries(config)}

    def mla_geometries(self, *, include_swa=False):
        return tuple(self._geometries.values())

    def geometry_for(self, layer_id):
        if layer_id not in self._geometries:
            raise ValueError(f"GLM Flash layer {layer_id} is outside the full-history MLA set")
        return self._geometries[layer_id]

    def layer_params(self, layer):
        geom = self.geometry_for(layer.layer_idx)
        params = CapturedLayerParams(
            attention_arithmetic=_arithmetic(layer),
            kv_norm_weight=layer.kv_a_layernorm.weight.detach().cpu().float(),
            kv_b_weight=layer.kv_b_proj.weight.detach().cpu().float(),
            o_proj_weight=layer.o_proj.weight.detach().cpu().float(),
        )
        params.validate(geom)
        return params

    def _sequence(self, layer, seq_id, positions, raw, query, out, *, mask=None, selection=None):
        geom = self.geometry_for(layer.layer_idx)
        count = raw.shape[0]
        u = unit_rms_norm(raw, geom.rms_norm_eps).to(raw.dtype)
        seq = CapturedSequence(
            seq_id=seq_id,
            positions=torch.arange(count) if positions is None else positions.detach().cpu(),
            u=u.detach().cpu(), q_nope=query.detach().reshape(count, geom.heads, geom.qk_nope_dim).cpu(),
            q_rope=None, k_rope=None, allowed_mask=None if mask is None else mask.cpu(),
            sparse_selection=None if selection is None else selection.to("cpu"),
            gate=None, ref_output=out.detach().float().cpu(),
        )
        seq.validate(geom)
        return seq

    @torch.no_grad()
    def replay_input(self, layer, params, row):
        from dataclasses import replace
        from qmla.capture import eval_attention_output
        dev = layer.q_a_proj.weight.device
        x = row.hidden.to(device=dev, dtype=layer.q_a_proj.weight.dtype).unsqueeze(0)
        raw = layer.kv_a_proj_with_mqa(x)[0]
        query = layer.q_b_proj(layer.q_a_layernorm(layer.q_a_proj(x)))
        geom = self.geometry_for(layer.layer_idx)
        u = unit_rms_norm(raw, geom.rms_norm_eps).to(raw.dtype).float()
        seq = CapturedSequence(row.seq_id, row.positions.to(dev), u,
                               query.reshape(row.seq_len, geom.heads, geom.qk_nope_dim).float(),
                               None, None, row.allowed_mask.to(dev), None,
                               torch.empty((row.seq_len, x.shape[-1]), device=dev))
        return replace(seq, ref_output=eval_attention_output(seq, params.to(dev), geom))

    @contextmanager
    def capture_hook(self, layer, *, seq_id, positions=None, compact_mask=False):
        """Observe one real decoder invocation; projection hooks leave its output untouched."""
        self.geometry_for(layer.layer_idx)
        tensors = {}
        result = []

        def save(name):
            def record(module, args, output):
                tensors[name] = output.detach()
            return record

        def after(module, args, kwargs, output):
            mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            if mask is None or mask.ndim != 2 or mask.shape[0] != 1 or not bool(mask.all()):
                raise ValueError("GLM QMLA requires one unpadded prefill record")
            if kwargs.get("past_key_values") is not None:
                raise ValueError("GLM QMLA requires uncached prefill")
            indices = tensors["indices"][0]
            count = mask.shape[1]
            if indices.shape[0] != count or tensors["raw"].shape[:2] != (1, count):
                raise ValueError("GLM capture must cover the whole sequence")
            dense, selection = None, None
            if compact_mask:
                selection = _pool_selection(indices, module.indexer)
            else:
                dense = torch.zeros((count, count + 1), device=indices.device, dtype=torch.bool)
                dense.scatter_(1, indices.masked_fill(indices < 0, count).long(), True)
                dense = dense[:, :count]
            result.append(self._sequence(module, seq_id, positions, tensors["raw"][0], tensors["query"][0],
                                         output[0][0], mask=dense, selection=selection))

        with ExitStack() as stack:
            for mod, name in ((layer.kv_a_proj_with_mqa, "raw"), (layer.q_b_proj, "query"),
                              (layer.indexer, "indices")):
                stack.callback(mod.register_forward_hook(save(name)).remove)
            stack.callback(layer.register_forward_hook(after, with_kwargs=True).remove)
            yield result
        if len(result) != 1:
            raise ValueError(f"GLM capture expected one attention forward, saw {len(result)}")

    @torch.no_grad()
    def capture_sequence(self, inputs, layer, *, seq_id, positions=None, compact_mask=False):
        self.geometry_for(layer.layer_idx)
        x = inputs.unsqueeze(0) if inputs.ndim == 2 else inputs
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("GLM QMLA requires one prefill sequence")
        if compact_mask:
            raw, query, selection, out = _tiled_forward(x, layer)
            return self._sequence(layer, seq_id, positions, raw, query, out, selection=selection), out
        with self.capture_hook(layer, seq_id=seq_id, positions=positions) as result:
            out = layer(x, attention_mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))[0][0]
        return result[0], out
