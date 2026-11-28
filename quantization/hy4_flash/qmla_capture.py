"""Hy4-flash mode-agnostic QMLA capture adapter (reference MLA family adapter).

Wraps ``quantization/hy4_flash/forward.py`` so both the full MC causal pass and
the standalone incremental calibration pass capture identical FP32 MLA states:
weightless RMSNorm_1 latent ``u``, ``q_nope``, post-RoPE ``q_rope`` / ``k_rope``
(1D or 3D interleaved M-RoPE ``[12, 10, 10]``), exact sparse compressed-indexer
or SWA-512 attention masks, sigmoid ``linear_gate``, learnable sink,
``post_proj_norm``, and post-projection reference outputs.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from qmla.capture import (
    CapturedLayerBank,
    CapturedLayerParams,
    CapturedSequence,
    MlaLayerGeometry,
    QmlaCaptureAdapter,
    SparseSelection,
    causal_mask,
    describe_splits,
    sequence_sha256,
    split_train_val_test,
    unit_rms_norm,
)

from . import forward as hy4_forward

reference = hy4_forward.reference
Project = hy4_forward.Project


def _mrope_section(config: Any, kind: str) -> tuple[int, int, int] | None:
    params = getattr(config, "rope_parameters", {}).get(kind, {})
    raw = params.get("mrope_section")
    if raw is None:
        return None
    if len(raw) != 3:
        raise ValueError(f"Hy4-flash mrope_section must have 3 axes, got {raw!r}")
    return (int(raw[0]), int(raw[1]), int(raw[2]))


def hy4_flash_mla_geometries(config: Any, *, include_swa: bool = False) -> tuple[MlaLayerGeometry, ...]:
    """Build per-layer MLA geometries from a Hy4-flash config (DSA by default)."""
    geometries: list[MlaLayerGeometry] = []
    for index, kind in enumerate(config.layer_types):
        is_sparse = kind == reference.SPARSE
        if not is_sparse and not include_swa:
            continue
        heads, nope, rope, vdim = config.attention(index)
        geom = MlaLayerGeometry(
            layer_id=index,
            kv_source_layer=index,
            kind="dsa" if is_sparse else "swa",
            heads=int(heads),
            kv_lora_rank=int(config.kv_lora_rank),
            qk_nope_dim=int(nope),
            qk_rope_dim=int(rope),
            v_head_dim=int(vdim),
            rope_kind="interleaved",
            mrope_section=_mrope_section(config, kind),
            rms_norm_eps=float(config.rms_norm_eps),
            attention_scale=float((nope + rope) ** -0.5),
            sliding_window=None if is_sparse else int(config.sliding_window),
            shared_consumers=(index,),
        )
        geom.validate()
        geometries.append(geom)
    return tuple(geometries)


def _rope_phase(
    positions: torch.Tensor,
    rope_dim: int,
    base: float,
    section: tuple[int, int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim == 2:
        if section is None:
            raise ValueError("3D M-RoPE positions require mrope_section in config")
        return reference.mrope_phase(positions, rope_dim, base, section)
    if positions.ndim == 1:
        return reference.phase(positions, rope_dim, base)
    raise ValueError(f"unsupported Hy4-flash positions ndim={positions.ndim}")


class Hy4FlashCaptureAdapter(QmlaCaptureAdapter):
    """Reference MLA capture adapter for Hy4-flash (copy/adapt for GLM 5.3 Flash)."""

    family_id = "hy4_flash"
    dsa_indexer_depends_on_mla_latents = False
    dsa_indexer_note = (
        "Hy4-flash compressed_indexer computes scores solely from layer input x "
        "(token_key_projection, head_weight_projection) and q_a_proj(x) (query_projection); "
        "QMLA-quantized MLA content latent u and k_rope do not feed it and indexer keys stay native, "
        "so intra-layer DSA top-k selection drift is structurally zero."
    )

    def __init__(self, model: hy4_forward.Model, *, include_swa: bool = False):
        self.model = model
        self.config = model.config
        self.include_swa = bool(include_swa)
        self._geometries = {
            g.layer_id: g for g in hy4_flash_mla_geometries(self.config, include_swa=True)
        }

    def mla_geometries(self, *, include_swa: bool | None = None) -> tuple[MlaLayerGeometry, ...]:
        use_swa = self.include_swa if include_swa is None else bool(include_swa)
        return hy4_flash_mla_geometries(self.config, include_swa=use_swa)

    def geometry_for(self, layer_id: int) -> MlaLayerGeometry:
        if layer_id not in self._geometries:
            raise ValueError(f"Hy4-flash layer {layer_id} is outside configured layers")
        return self._geometries[layer_id]

    def layer_params(self, layer: hy4_forward.Layer) -> CapturedLayerParams:
        w = layer.weights
        p = "self_attn."
        params = CapturedLayerParams(
            kv_norm_weight=w[p + "kv_a_layernorm.weight"].detach().float().cpu(),
            kv_b_weight=w[p + "kv_b_proj.weight"].detach().float().cpu(),
            o_proj_weight=w[p + "o_proj.weight"].detach().float().cpu(),
            sink=w[p + "learnable_sink_param"].detach().float().cpu(),
            post_proj_norm_weight=w[p + "post_proj_norm.weight"].detach().float().cpu(),
        )
        params.validate(self.geometry_for(layer.index))
        return params

    def capture_sequence(
        self,
        inputs: torch.Tensor,
        layer: hy4_forward.Layer,
        *,
        seq_id: str,
        positions: torch.Tensor | None = None,
        project: Project = hy4_forward.native_projection,
        capture_tap: bool = False,
        compact_mask: bool = False,
    ) -> tuple[CapturedSequence, torch.Tensor, hy4_forward.AttentionTap | None]:
        """Run exact Hy4-flash attention and capture mode-agnostic MLA tensors.

        When ``compact_mask=True`` (used for 16K/32K long-context evaluation), skips
        allocating the dense ``[T, T]`` boolean mask and records compact DSA group IDs in
        ``CapturedSequence.sparse_selection`` instead.
        """
        c = self.config
        w = layer.weights
        index = layer.index
        geom = self.geometry_for(index)
        kind = c.layer_types[index]
        h, nope, rope, vdim = c.attention(index)
        p = "self_attn."
        seq_len = int(inputs.shape[0])
        device = inputs.device

        if positions is None:
            positions = torch.arange(seq_len, device=device, dtype=torch.int64)
        else:
            positions = positions.to(device=device, dtype=torch.int64)
        slot_pos = torch.arange(seq_len, device=device, dtype=torch.int64)

        # Stage 1: low-rank projections; preserve weightless RMSNorm_1 latent u.
        q_a = layer.project(p + "q_a_proj", inputs, project)
        q_norm = reference.norm(q_a, w[p + "q_a_layernorm.weight"], c.rms_norm_eps)
        kv_raw, kr_raw = layer.project(p + "kv_a_proj_with_mqa", inputs, project).split(
            [c.kv_lora_rank, rope], dim=-1
        )
        u = unit_rms_norm(kv_raw, c.rms_norm_eps)
        kv = u * w[p + "kv_a_layernorm.weight"].to(device=device, dtype=torch.float32)
        gate = layer.project(p + "linear_gate", inputs, project).sigmoid()

        # Stage 2: expand queries and apply exact 1D or 3D interleaved M-RoPE.
        base = self.model.rope_base(index)
        section = _mrope_section(c, kind)
        cos, sin = _rope_phase(positions, rope, base, section)
        q_full = layer.project(p + "q_b_proj", q_norm, project).reshape(seq_len, h, nope + rope)
        q_nope, q_rope_raw = q_full.split([nope, rope], dim=-1)
        q_rope = self.model.rotate(q_rope_raw, cos, sin)
        q_rot = torch.cat((q_nope, q_rope), dim=-1)
        kr_rot = self.model.rotate(kr_raw.unsqueeze(1), cos, sin).squeeze(1)

        expanded = layer.project(p + "kv_b_proj", kv, project)
        k_nope, v = expanded.reshape(seq_len, h, nope + vdim).split([nope, vdim], dim=-1)
        keys = torch.cat((k_nope, kr_rot.unsqueeze(1).expand(-1, h, -1)), dim=-1)
        sink = w[p + "learnable_sink_param"]

        # Stage 3: exact causal / SWA / compressed-indexer DSA mask and attention (128-row tiles).
        indexed = kind == reference.SPARSE and seq_len > c.compressed_indexer_candidate_topk
        if indexed:
            rope_arg = positions if positions.ndim == 2 else None
            iq, ik, iw = reference.index_projections(inputs, q_norm, w, c, slot_pos, base, rope_arg)

        ratio = c.compressed_indexer_compress_ratio
        rows = reference.ROWS
        attended_parts: list[torch.Tensor] = []
        group_parts: list[torch.Tensor] = []
        full_mask = None if compact_mask else causal_mask(
            seq_len,
            device=device,
            sliding_window=c.sliding_window if kind == reference.SLIDING else None,
        )

        for start in range(0, seq_len, rows):
            stop = min(start + rows, seq_len)
            first_key = max(0, start + 1 - c.sliding_window) if kind == reference.SLIDING else 0
            width = start + rows - first_key
            query = F.pad(q_rot[start:stop], (0, 0, 0, 0, 0, rows - (stop - start)))
            key_tile = F.pad(keys[first_key:stop], (0, 0, 0, 0, 0, width - (stop - first_key)))
            val_tile = F.pad(v[first_key:stop], (0, 0, 0, 0, 0, width - (stop - first_key)))
            key_id = torch.arange(first_key, start + rows, device=device)
            query_id = torch.arange(start, start + rows, device=device)
            distance = query_id[:, None] - key_id
            allowed = (distance >= 0) & (key_id < stop)
            if kind == reference.SLIDING:
                allowed &= distance < c.sliding_window
            elif indexed:
                current_group = slot_pos[start:stop] // ratio
                groups = reference.recall_groups(
                    iq[start:stop], ik, iw[start:stop], current_group, c.compressed_indexer_group_topk
                )
                slots = (groups[..., None] * ratio + torch.arange(ratio, device=device)).flatten(1)
                slots = slots.clamp_max(seq_len - 1).clamp_max(width)
                selected = torch.zeros((rows, width + 1), dtype=torch.bool, device=device)
                selected[: stop - start].scatter_(1, slots, True)
                allowed &= selected[:, :width] | (query_id[:, None] < c.compressed_indexer_candidate_topk)
                if full_mask is not None:
                    full_mask[start:stop, :stop] = allowed[: stop - start, :stop]
                if capture_tap or compact_mask:
                    valid = groups <= current_group[:, None]
                    group_parts.append(groups.masked_fill(~valid, -1).to(torch.int32))
            attended_parts.append(hy4_forward._with_sink(query, key_tile, val_tile, allowed, sink)[: stop - start])

        gated = torch.cat(attended_parts, dim=0).flatten(1) * gate
        ref_output = self.model.attention_finish(gated, layer, project)

        sparse_selection = None
        if compact_mask and indexed:
            sparse_selection = SparseSelection(
                groups=torch.cat(group_parts, dim=0),
                compress_ratio=int(ratio),
                candidate_topk=int(c.compressed_indexer_candidate_topk),
            )

        tap = None
        if capture_tap:
            if kind != reference.SPARSE:
                raise ValueError("Hy4 DFly2 supports only sparse attention target taps")
            groups_cat = (
                torch.cat(group_parts, dim=0)
                if group_parts
                else slot_pos.new_empty((seq_len, 0), dtype=torch.int32)
            )
            tap = hy4_forward.AttentionTap(kv, kr_rot, groups_cat)

        captured = CapturedSequence(
            seq_id=seq_id,
            positions=positions.detach().cpu(),
            u=u.detach().float().cpu(),
            q_nope=q_nope.detach().float().cpu(),
            q_rope=q_rope.detach().float().cpu(),
            k_rope=kr_rot.detach().float().cpu(),
            allowed_mask=None if full_mask is None else full_mask.detach().cpu(),
            gate=gate.detach().float().cpu(),
            ref_output=ref_output.detach().float().cpu(),
            sparse_selection=None if sparse_selection is None else sparse_selection.to("cpu"),
        )
        captured.validate(geom)
        return captured, ref_output, tap

    def capture_layer(
        self,
        layer: hy4_forward.Layer,
        sequences: Sequence[tuple[str, torch.Tensor, torch.Tensor | None]],
        *,
        train_count: int,
        val_count: int,
        test_count: int,
        seed: int = 0,
        project: Project = hy4_forward.native_projection,
        provenance: dict[str, Any] | None = None,
    ) -> CapturedLayerBank:
        """Capture a disjoint train/val/test CapturedLayerBank for one Hy4-flash layer."""
        geom = self.geometry_for(layer.index)
        params = self.layer_params(layer)
        splits = split_train_val_test(
            sequences,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            key_fn=lambda row: row[0],
            seed=seed,
        )
        captured = tuple(
            tuple(
                self.capture_sequence(inp, layer, seq_id=sid, positions=pos, project=project)[0]
                for sid, inp, pos in items
            )
            for items in splits
        )
        description = describe_splits(
            {
                name: [(sid, sequence_sha256(inp)) for sid, inp, _ in items]
                for name, items in zip(("train", "val", "test"), splits, strict=True)
            },
            seed=seed,
        )
        bank = CapturedLayerBank(
            geometry=geom,
            params=params,
            train=captured[0],
            val=captured[1],
            test=captured[2],
            provenance={**dict(provenance or {}), "splits": description},
        )
        bank.validate()
        return bank
