"""Independent IEEE-F32 reference for ERNIE 5.5 Flash (target + bundled DSpark drafter).

Zero engine imports. Exposes Config, Checkpoint, layer_forward, lm_head_logits,
and DSpark drafter entry points so both layer_major golden scoring and
quantization/ernie55_flash/forward.py can wrap it directly.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct

import torch
import torch.nn.functional as F

from shape_invariant import ROWS, linear
from families.hy4_common import (
    Checkpoint as _BaseCheckpoint,
    configure_f32,
    inverse_rms,
    norm,
    phase,
    phase_split_half,
    rotate,
    rotate_split_half,
    rotate_tail,
)

OUTER_NAMES = (
    "model.embed_tokens.weight",
    "model.mhc_contract.hc_head_fn",
    "model.mhc_contract.hc_head_base",
    "model.mhc_contract.hc_head_scale",
    "model.norm.weight",
    "lm_head.weight",
    "model.lm_head.multimax_ranges",
    "model.lm_head.multimax_ts",
)

MTP_AUX_NAMES = (
    "model.mtp.feature_projection.weight",
    "model.mtp.feature_norm.weight",
    "model.mtp.mhc_contract.hc_head_fn",
    "model.mtp.mhc_contract.hc_head_base",
    "model.mtp.mhc_contract.hc_head_scale",
    "model.mtp.norm.weight",
    "model.mtp.markov_head.markov_w1.weight",
    "model.mtp.markov_head.markov_w2.weight",
    "model.mtp.confidence_head.projection.weight",
)

FLASHINDEXER_FILE = "flashindexer/flashindexer.safetensors"
FLASHINDEXER_TOPK_HEADS = 8
COMPRESS_RATIO_HCA = 128
COMPRESS_RATIO_MLA = -2
VHA_RANK = 16
HC_POST_ALPHA = 2.0

# Defaults updated after 2x2 empirical DSpark acceptance measurement on real weights.
DEFAULT_DSPARK_ROPE_MODE = "plain"
DEFAULT_DSPARK_MULTIMAX_AFTER_MARKOV = True


@dataclass(frozen=True)
class Config:
    raw: dict

    @classmethod
    def parse(cls, raw: dict) -> "Config":
        if (
            raw.get("model_type") != "ernie5_5_moe"
            or raw.get("quantization_config")
            or raw.get("tie_word_embeddings")
            or raw.get("hidden_act") != "situ"
            or raw.get("multimax") != "lm_head"
            or not raw.get("moe_split_feature_routing")
            or not raw.get("latent_moe_use_norm")
            or not raw.get("routed_scaling_factor_learnable")
            or raw.get("n_group", 1) != 1
            or raw.get("topk_group", 1) != 1
        ):
            raise ValueError("unsupported ERNIE 5.5 Flash config")
        n_layers = int(raw["num_hidden_layers"])
        ratios = list(raw["compress_ratios"])
        if len(ratios) != n_layers or set(ratios) - {COMPRESS_RATIO_HCA, COMPRESS_RATIO_MLA}:
            raise ValueError(f"invalid ERNIE 5.5 compress_ratios: {ratios}")
        if int(raw["hc_mult"]) != 4 or int(raw["o_groups"]) != 8:
            raise ValueError("unsupported ERNIE 5.5 hc_mult or o_groups")
        return cls(dict(raw))

    def __getattr__(self, name: str):
        try:
            return self.raw[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def hc_post_alpha(self) -> float:
        return HC_POST_ALPHA

    @property
    def num_mtp_stages(self) -> int:
        ids = self.raw.get("dspark_target_layer_ids") or []
        return len(ids)

    def is_dsa_layer(self, index: int) -> bool:
        return self.compress_ratios[index] == COMPRESS_RATIO_MLA

    def is_hca_layer(self, index: int) -> bool:
        return self.compress_ratios[index] == COMPRESS_RATIO_HCA

    def is_moe_layer(self, index: int) -> bool:
        return index >= self.first_k_dense_replace

    def dsa_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.is_dsa_layer(i))


def _add_mhc_contract(add, prefix: str, h: int, hc: int) -> None:
    width = 2 * hc + hc * hc
    add(f"{prefix}.mapping_proj.weight", hc * h, width, dtype="F32")
    add(f"{prefix}.bias", width, dtype="F32")
    for name in ("alpha_pre", "alpha_post", "alpha_res"):
        add(f"{prefix}.{name}", 1, dtype="F32")


def _add_moe_contract(add, c: Config, *, include_experts: bool) -> None:
    h = c.hidden_size
    ne = c.n_routed_experts
    latent = c.moe_latent_size
    inter = c.moe_intermediate_size
    shared_inter = inter * c.n_shared_experts
    p = "block_sparse_moe."
    add(p + "gate.weight", ne, h, dtype="F32")
    add(p + "gate.weight_1", ne, h, dtype="F32")
    add(p + "e_score_correction_bias", ne, dtype="F32")
    add(p + "gate.routed_scaling_factor_param", ne, dtype="F32")
    add(p + "fc1_latent_proj.weight", latent, h)
    add(p + "fc2_latent_proj.weight", h, latent)
    add(p + "latent_norm.weight", latent)
    add(p + "shared_experts.w1.weight", shared_inter, h)
    add(p + "shared_experts.w2.weight", h, shared_inter)
    add(p + "shared_experts.w3.weight", shared_inter, h)
    if include_experts:
        for e in range(ne):
            add(f"{p}experts.{e}.w1.weight", inter, latent)
            add(f"{p}experts.{e}.w2.weight", latent, inter)
            add(f"{p}experts.{e}.w3.weight", inter, latent)


def _add_hca_contract(add, c: Config, *, with_compressor: bool) -> None:
    h = c.hidden_size
    nh = c.num_attention_heads
    hd = c.head_dim
    ql = c.q_lora_rank
    og = c.o_groups
    ol = c.o_lora_rank
    p = "self_attn."
    add(p + "linear_q_down_proj.weight", ql, h)
    add(p + "q_layernorm.weight", ql)
    add(p + "linear_q_up_proj.weight", nh * hd, ql)
    add(p + "linear_kv_proj.weight", hd, h)
    add(p + "kv_layernorm.weight", hd)
    add(p + "gate_proj.weight", og * ol, ql)
    add(p + "linear_o_group_proj", og * ol, (nh // og) * hd)
    add(p + "o_proj.weight", h, og * ol)
    add(p + "vha_postmix_U", nh, VHA_RANK)
    add(p + "vha_postmix_V", nh, VHA_RANK)
    add(p + "core_attention.attn_sink", nh, dtype="F32")
    if with_compressor:
        cp = p + "core_attention.compressor."
        add(cp + "ape", COMPRESS_RATIO_HCA, hd, dtype="F32")
        add(cp + "linear_wkv.weight", hd, h)
        add(cp + "linear_wgate.weight", hd, h)
        add(cp + "norm.weight", hd)


def _add_mla_dsa_contract(add, c: Config) -> None:
    h = c.hidden_size
    nh = c.hybrid_mla_num_attention_heads
    ql = c.hybrid_mla_q_lora_rank
    kvl = c.hybrid_mla_kv_lora_rank
    nope = c.hybrid_mla_qk_nope_head_dim
    rope = c.hybrid_mla_qk_rope_head_dim
    vd = c.hybrid_mla_v_head_dim
    inh = c.index_n_heads
    ihd = c.index_head_dim
    p = "self_attn."
    add(p + "q_a_proj.weight", ql, h)
    add(p + "q_a_layernorm.weight", ql)
    add(p + "q_b_proj.weight", nh * (nope + rope), ql)
    add(p + "kv_a_proj_with_mqa.weight", kvl + rope, h)
    add(p + "kv_a_layernorm.weight", kvl)
    add(p + "kv_b_proj.weight", nh * (nope + vd), kvl)
    add(p + "gate_proj.weight", nh * vd, ql)
    add(p + "o_proj.weight", h, nh * vd)
    add(p + "vha_postmix_U", nh, VHA_RANK)
    add(p + "vha_postmix_V", nh, VHA_RANK)
    add(p + "core_attention.softmax_offset", nh)
    ip = p + "core_attention.indexer."
    add(ip + "wq_b.weight", inh * ihd, ql)
    add(ip + "wk.weight", ihd, h)
    add(ip + "k_norm.weight", ihd)
    add(ip + "k_norm.bias", ihd)
    add(ip + "weights_proj.weight", inh, h)


def tensor_contract(c: Config, index: int | None = None, *, include_experts: bool = True) -> dict[str, tuple[list[int], str]]:
    """Return exact target tensor names, shapes and storage dtypes."""
    h, hc = c.hidden_size, c.hc_mult
    items: dict[str, tuple[list[int], str]] = {}

    def add(name: str, *shape: int, dtype: str = "BF16") -> None:
        items[name] = (list(shape), dtype)

    if index is None:
        add(OUTER_NAMES[0], c.vocab_size, h)
        add(OUTER_NAMES[1], hc * h, hc, dtype="F32")
        add(OUTER_NAMES[2], hc, dtype="F32")
        add(OUTER_NAMES[3], 1, dtype="F32")
        add(OUTER_NAMES[4], h)
        add(OUTER_NAMES[5], c.vocab_size, h)
        add(OUTER_NAMES[6], 4)
        add(OUTER_NAMES[7], 4)
        return items

    add("input_layernorm.weight", h)
    add("post_attention_layernorm.weight", h)
    _add_mhc_contract(add, "self_attention_hyper_connection", h, hc)
    _add_mhc_contract(add, "mlp_hyper_connection", h, hc)
    if c.is_dsa_layer(index):
        _add_mla_dsa_contract(add, c)
    else:
        _add_hca_contract(add, c, with_compressor=True)
    if c.is_moe_layer(index):
        _add_moe_contract(add, c, include_experts=include_experts)
    else:
        add("mlp.gate_proj.weight", c.intermediate_size, h)
        add("mlp.up_proj.weight", c.intermediate_size, h)
        add("mlp.down_proj.weight", h, c.intermediate_size)
    return {f"model.layers.{index}.{name}": value for name, value in items.items()}


def flashindexer_contract(c: Config) -> dict[str, tuple[list[int], str]]:
    """Return exact flashindexer/flashindexer.safetensors tensor contract."""
    items: dict[str, tuple[list[int], str]] = {}
    for layer_id in c.dsa_layers():
        p = f"model.layers.{layer_id}.self_attn.core_attention.indexer."
        items[p + "linear_gate.weight"] = ([c.index_n_heads, c.hidden_size], "BF16")
        items[p + "flashindexer_bias"] = ([c.index_n_heads], "BF16")
    return items


def mtp_contract(c: Config, stage: int | None = None, *, include_experts: bool = True) -> dict[str, tuple[list[int], str]]:
    """Return exact bundled DSpark drafter (model.mtp.*) tensor contract."""
    h, hc = c.hidden_size, c.hc_mult
    items: dict[str, tuple[list[int], str]] = {}

    def add(name: str, *shape: int, dtype: str = "BF16") -> None:
        items[name] = (list(shape), dtype)

    if stage is None:
        rank = int(c.dspark_markov_rank)
        n_taps = c.num_mtp_stages
        add(MTP_AUX_NAMES[0], h, h * n_taps)
        add(MTP_AUX_NAMES[1], h)
        add(MTP_AUX_NAMES[2], hc * h, hc, dtype="F32")
        add(MTP_AUX_NAMES[3], hc, dtype="F32")
        add(MTP_AUX_NAMES[4], 1, dtype="F32")
        add(MTP_AUX_NAMES[5], h)
        add(MTP_AUX_NAMES[6], rank, c.vocab_size)
        add(MTP_AUX_NAMES[7], c.vocab_size, rank)
        add(MTP_AUX_NAMES[8], 1, h + rank)
        return items

    add("input_layernorm.weight", h)
    add("post_attention_layernorm.weight", h)
    _add_mhc_contract(add, "self_attention_hyper_connection", h, hc)
    _add_mhc_contract(add, "mlp_hyper_connection", h, hc)
    _add_hca_contract(add, c, with_compressor=False)
    _add_moe_contract(add, c, include_experts=include_experts)
    return {f"model.mtp.{stage}.{name}": value for name, value in items.items()}


class LazyExpertStore:
    """On-demand per-expert BF16->F32 streamer (zero resident expert memory)."""

    def __init__(self, checkpoint: "Checkpoint", prefix: str):
        self.checkpoint = checkpoint
        self.prefix = prefix

    def get(self, expert: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep = f"{self.prefix}block_sparse_moe.experts.{expert}."
        return (
            self.checkpoint.read_tensor(ep + "w1.weight").to(device, dtype=torch.float32),
            self.checkpoint.read_tensor(ep + "w3.weight").to(device, dtype=torch.float32),
            self.checkpoint.read_tensor(ep + "w2.weight").to(device, dtype=torch.float32),
        )


class PackedBF16ExpertStore:
    """One layer's 512 experts packed in BF16 (12 GiB), upcast one expert at a time to F32."""

    def __init__(self, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor):
        self.w1 = w1
        self.w3 = w3
        self.w2 = w2

    def get(self, expert: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.w1[expert].to(device, dtype=torch.float32),
            self.w3[expert].to(device, dtype=torch.float32),
            self.w2[expert].to(device, dtype=torch.float32),
        )


class DictExpertStore:
    """In-memory expert lookup for unit tests and extracted single-layer modules."""

    def __init__(self, weights: dict[str, torch.Tensor], prefix: str = ""):
        self.weights = weights
        self.prefix = prefix

    def get(self, expert: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep = f"{self.prefix}block_sparse_moe.experts.{expert}."
        return (
            self.weights[ep + "w1.weight"].to(device, dtype=torch.float32),
            self.weights[ep + "w3.weight"].to(device, dtype=torch.float32),
            self.weights[ep + "w2.weight"].to(device, dtype=torch.float32),
        )


class Checkpoint(_BaseCheckpoint):
    """Pread-backed checkpoint reader for ERNIE 5.5 main index + flashindexer subfolder."""

    def __init__(self, snapshot, device="cpu"):
        super().__init__(snapshot, device)
        self._fi_header: dict | None = None
        self._fi_offset: int = 0

    def _load_flashindexer_header(self) -> tuple[dict, int]:
        if self._fi_header is None:
            path = self.snapshot / FLASHINDEXER_FILE
            with path.open("rb") as stream:
                size, = struct.unpack("<Q", stream.read(8))
                if not 2 <= size <= 64 * 2**20:
                    raise ValueError(f"invalid flashindexer header size: {size}")
                header = json.loads(stream.read(size))
            end = max(entry["data_offsets"][1] for key, entry in header.items() if key != "__metadata__")
            if end + 8 + size != path.stat().st_size:
                raise ValueError("incomplete flashindexer safetensors file")
            self._fi_header = header
            self._fi_offset = 8 + size
        return self._fi_header, self._fi_offset

    def check_flashindexer(self, contract: dict[str, tuple[list[int], str]]) -> None:
        header, _ = self._load_flashindexer_header()
        present = {k for k in header if k != "__metadata__"}
        if present != set(contract):
            raise ValueError(f"flashindexer inventory mismatch: {sorted(present ^ set(contract))}")
        for name, (shape, dtype) in contract.items():
            d = header[name]
            first, last = d["data_offsets"]
            itemsize = 2 if dtype == "BF16" else 4
            if d["shape"] != shape or d["dtype"] != dtype or first < 0 or last - first != math.prod(shape) * itemsize:
                raise ValueError(f"wrong flashindexer tensor shape, dtype or extent: {name}: {d}")

    def load_flashindexer(self, name: str) -> torch.Tensor:
        header, base_offset = self._load_flashindexer_header()
        d = header[name]
        dtype = torch.bfloat16 if d["dtype"] == "BF16" else torch.float32
        first, last = d["data_offsets"]
        nbytes = last - first
        tensor = torch.empty(d["shape"], dtype=dtype)
        mv = memoryview(tensor.view(torch.uint8).numpy().ravel())
        fd = os.open(self.snapshot / FLASHINDEXER_FILE, os.O_RDONLY)
        try:
            pos = 0
            while pos < nbytes:
                chunk = os.pread(fd, min(16 * 2**20, nbytes - pos), base_offset + first + pos)
                if not chunk:
                    raise ValueError(f"short read on flashindexer tensor {name}")
                mv[pos:pos + len(chunk)] = chunk
                pos += len(chunk)
        finally:
            os.close(fd)
        return tensor.to(self.device, dtype=torch.float32, copy=True)

    def _load_packed_experts(self, prefix: str, c: Config, target_device: torch.device) -> PackedBF16ExpertStore:
        """Read 512 BF16 experts into 3 compact BF16 tensors (12 GiB total, never F32)."""
        ne = c.n_routed_experts
        inter = c.moe_intermediate_size
        latent = c.moe_latent_size
        # Allocate on host first to read via pread, then move each tensor to target_device.
        # On GB10 unified memory (device=cuda), allocate directly or copy cleanly without pinned memory.
        w1 = torch.empty((ne, inter, latent), dtype=torch.bfloat16)
        w3 = torch.empty((ne, inter, latent), dtype=torch.bfloat16)
        w2 = torch.empty((ne, latent, inter), dtype=torch.bfloat16)
        # Group reads by shard so each shard opens once.
        by_shard: dict[str, list[tuple[int, str, int, int]]] = {}
        for e in range(ne):
            for tag in ("w1", "w3", "w2"):
                name = f"{prefix}block_sparse_moe.experts.{e}.{tag}.weight"
                d = self.descriptor(name)
                shard = self.index[name]
                first, last = d["data_offsets"]
                by_shard.setdefault(shard, []).append((e, tag, self.header_offsets[shard] + first, last - first))

        bufs = {"w1": w1, "w3": w3, "w2": w2}

        def read_shard(item):
            shard, entries = item
            fd = os.open(self.snapshot / shard, os.O_RDONLY)
            total = 0
            try:
                for e, tag, offset, nbytes in entries:
                    mv = memoryview(bufs[tag][e].view(torch.uint8).numpy().ravel())
                    pos = 0
                    while pos < nbytes:
                        chunk = os.pread(fd, nbytes - pos, offset + pos)
                        if not chunk:
                            raise ValueError(f"short read on expert {e}.{tag}")
                        mv[pos:pos + len(chunk)] = chunk
                        pos += len(chunk)
                    total += nbytes
            finally:
                os.close(fd)
            return total

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, max(1, len(by_shard)))) as pool:
            self.read_bytes += sum(pool.map(read_shard, by_shard.items()))

        if target_device.type != "cpu":
            w1 = w1.to(target_device)
            w3 = w3.to(target_device)
            w2 = w2.to(target_device)
            torch.cuda.synchronize(target_device)
        return PackedBF16ExpertStore(w1, w3, w2)

    def layer(
        self,
        config: Config,
        index: int,
        *,
        load_experts: bool = True,
        experts_on_device: bool = True,
        load_flashindexer: bool = False,
    ) -> dict[str, object]:
        prefix = f"model.layers.{index}."
        contract = tensor_contract(config, index, include_experts=False)
        result: dict[str, object] = {}
        for name in contract:
            result[name.removeprefix(prefix)] = self.load(name)
        if load_flashindexer and config.is_dsa_layer(index):
            for name in flashindexer_contract(config):
                if name.startswith(prefix):
                    result[name.removeprefix(prefix)] = self.load_flashindexer(name)
        if config.is_moe_layer(index):
            if load_experts:
                dev = self.device if experts_on_device else torch.device("cpu")
                result["experts"] = self._load_packed_experts(prefix, config, dev)
            else:
                result["experts"] = LazyExpertStore(self, prefix)
        return result

    def mtp_aux(self, config: Config) -> dict[str, torch.Tensor]:
        result = {}
        for name in mtp_contract(config, None):
            result[name.removeprefix("model.mtp.")] = self.load(name)
        return result

    def mtp_stage(
        self,
        config: Config,
        stage: int,
        *,
        load_experts: bool = True,
        experts_on_device: bool = True,
    ) -> dict[str, object]:
        prefix = f"model.mtp.{stage}."
        contract = mtp_contract(config, stage, include_experts=False)
        result: dict[str, object] = {}
        for name in contract:
            result[name.removeprefix(prefix)] = self.load(name)
        if load_experts:
            dev = self.device if experts_on_device else torch.device("cpu")
            result["experts"] = self._load_packed_experts(prefix, config, dev)
        else:
            result["experts"] = LazyExpertStore(self, prefix)
        return result


def yarn_phase(
    positions: torch.Tensor,
    width: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved GPT-J YaRN RoPE phase (mscale=0), matching vLLM 0.32.0 & SGLang."""
    def find_correction_dim(num_rot: float) -> float:
        return width * math.log(original_seq_len / (num_rot * 2.0 * math.pi)) / (2.0 * math.log(base))

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), width - 1)
    if low == high:
        high_f = high + 0.001
    else:
        high_f = float(high)
    half = width // 2
    ramp = ((torch.arange(half, dtype=torch.float32, device=positions.device) - low) / (high_f - low)).clamp(0.0, 1.0)
    smooth = 1.0 - ramp
    inv_base = 1.0 / (base ** (torch.arange(0, width, 2, dtype=torch.float32, device=positions.device) / width))
    inv_freq = inv_base / factor * (1.0 - smooth) + inv_base * smooth
    freq = positions.float().unsqueeze(-1) * inv_freq
    return (
        torch.cos(freq).repeat_interleave(2, dim=-1).unsqueeze(1),
        torch.sin(freq).repeat_interleave(2, dim=-1).unsqueeze(1),
    )


def hca_phase(positions: torch.Tensor, c: Config, *, rope_mode: str = "yarn") -> tuple[torch.Tensor, torch.Tensor]:
    width = c.qk_rope_head_dim
    if rope_mode == "plain":
        return phase(positions, width, float(c.rope_theta))
    if rope_mode != "yarn":
        raise ValueError(f"unsupported rope_mode: {rope_mode!r}")
    rs = c.rope_scaling
    return yarn_phase(
        positions,
        width=width,
        base=float(c.compress_rope_theta),
        original_seq_len=int(rs["original_max_position_embeddings"]),
        factor=float(rs["factor"]),
        beta_fast=float(rs["beta_fast"]),
        beta_slow=float(rs["beta_slow"]),
    )


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float = 4.0, linear_beta: float = 25.0) -> torch.Tensor:
    """SiTU gated activation: (beta * tanh(g/beta) * sigmoid(g)) * (linear_beta * tanh(u/linear_beta))."""
    g = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    u = linear_beta * torch.tanh(up / linear_beta)
    return g * u


def mhc_pre(
    residual: torch.Tensor,
    weights: dict,
    prefix: str,
    norm_weight: torch.Tensor,
    c: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre-mix with eps OUTSIDE sqrt and 20 Sinkhorn iterations."""
    hc = c.hc_mult
    x_flat = residual.flatten(-2)
    rms_inv = 1.0 / (x_flat.square().mean(dim=-1, keepdim=True).sqrt() + c.rms_norm_eps)
    fn = weights[prefix + ".mapping_proj.weight"].t()
    mixes = linear(x_flat, fn) * rms_inv
    bias = weights[prefix + ".bias"]
    a_pre = weights[prefix + ".alpha_pre"]
    a_post = weights[prefix + ".alpha_post"]
    a_res = weights[prefix + ".alpha_res"]
    pre = torch.sigmoid(mixes[:, :hc] * a_pre + bias[:hc]) + c.hc_eps
    post = torch.sigmoid(mixes[:, hc : 2 * hc] * a_post + bias[hc : 2 * hc]) * c.hc_post_alpha
    comb = (mixes[:, 2 * hc :].view(-1, hc, hc) * a_res) + bias[2 * hc :].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + c.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + c.hc_eps)
    for _ in range(c.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + c.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + c.hc_eps)
    mixed = (pre.unsqueeze(-1) * residual).sum(dim=1)
    layer_input = norm(mixed, norm_weight, c.rms_norm_eps)
    return post.unsqueeze(-1), comb, layer_input


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """mHC residual stream update: einsum('tij,tih->tjh', comb, residual) + post * x."""
    return torch.einsum("tij,tih->tjh", comb, residual) + post * x.unsqueeze(1)


def hc_head_pre_norm(
    hidden: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    c: Config,
    *,
    eps_outside_sqrt: bool = False,
) -> torch.Tensor:
    """Collapse mHC streams before final RMSNorm (default: vLLM-ERNIE eps inside sqrt)."""
    x_flat = hidden.flatten(-2)
    if eps_outside_sqrt:
        rms_inv = 1.0 / (x_flat.square().mean(dim=-1, keepdim=True).sqrt() + c.rms_norm_eps)
    else:
        rms_inv = inverse_rms(x_flat, c.rms_norm_eps)
    mix = linear(x_flat, fn.t()) * rms_inv
    pre = torch.sigmoid(mix * scale + base) + c.hc_eps
    return (pre.unsqueeze(-1) * hidden).sum(dim=1)


def vha_postmix(o: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Virtual Head Attention cross-head low-rank postmix: o <- (I + V @ U^T) @ o."""
    eye = torch.eye(u.shape[0], dtype=o.dtype, device=o.device)
    mix = torch.addmm(eye, v, u.t())
    return torch.matmul(mix, o)


def hca_attention(
    x: torch.Tensor,
    weights: dict,
    c: Config,
    positions: torch.Tensor,
    *,
    context_kv: torch.Tensor | None = None,
    non_causal: bool = False,
    rope_mode: str = "yarn",
) -> torch.Tensor:
    """HCA (compressed + SWA-128) attention, or SWA-only non-causal DSpark draft attention."""
    seq_len = x.shape[0]
    device = x.device
    nh = c.num_attention_heads
    hd = c.head_dim
    og = c.o_groups
    ol = c.o_lora_rank
    window = c.sliding_window
    p = "self_attn."

    cos, sin = hca_phase(positions, c, rope_mode=rope_mode)
    q_c = norm(linear(x, weights[p + "linear_q_down_proj.weight"]), weights[p + "q_layernorm.weight"], c.rms_norm_eps)
    q = linear(q_c, weights[p + "linear_q_up_proj.weight"]).view(seq_len, nh, hd)
    q = q * inverse_rms(q, c.rms_norm_eps)
    q = rotate_tail(q, cos, sin)

    kv = norm(linear(x, weights[p + "linear_kv_proj.weight"]), weights[p + "kv_layernorm.weight"], c.rms_norm_eps)
    kv = rotate_tail(kv.unsqueeze(1), cos, sin).squeeze(1)

    sink = weights[p + "core_attention.attn_sink"][:, None, None]
    scale = hd ** -0.5

    if non_causal:
        # DSpark draft block: every draft token attends to all context_kv + all block kv.
        all_kv = kv if context_kv is None or context_kv.shape[0] == 0 else torch.cat((context_kv, kv), dim=0)
        # q: [nh, T, hd], all_kv: [K, hd]
        scores = torch.matmul(q.transpose(0, 1), all_kv.t()) * scale
        scores = torch.cat((scores, sink.expand(-1, seq_len, -1)), dim=-1)
        probs = (scores - scores.max(dim=-1, keepdim=True).values).softmax(dim=-1)[..., :-1]
        o = torch.matmul(probs, all_kv).transpose(0, 1)
    else:
        # Target HCA layer: 128-token compressor + causal sliding window of 128 tokens.
        comp_kv = None
        n_chunks = seq_len // COMPRESS_RATIO_HCA
        if n_chunks > 0:
            cp = p + "core_attention.compressor."
            x_c = x[: n_chunks * COMPRESS_RATIO_HCA]
            raw_kv = linear(x_c, weights[cp + "linear_wkv.weight"]).view(n_chunks, COMPRESS_RATIO_HCA, hd)
            raw_score = (
                linear(x_c, weights[cp + "linear_wgate.weight"]).view(n_chunks, COMPRESS_RATIO_HCA, hd)
                + weights[cp + "ape"].unsqueeze(0)
            )
            comp_w = torch.softmax(raw_score, dim=1)
            comp_kv = norm((raw_kv * comp_w).sum(dim=1), weights[cp + "norm.weight"], c.rms_norm_eps)
            chunk_pos = torch.arange(n_chunks, device=device, dtype=torch.long) * COMPRESS_RATIO_HCA
            c_cos, c_sin = hca_phase(chunk_pos, c, rope_mode=rope_mode)
            comp_kv = rotate_tail(comp_kv.unsqueeze(1), c_cos, c_sin).squeeze(1)

        out_tiles = []
        for start in range(0, seq_len, ROWS):
            stop = min(start + ROWS, seq_len)
            tile_q = F.pad(q[start:stop].transpose(0, 1), (0, 0, 0, ROWS - (stop - start)))
            # SWA slice: [first .. stop) padded to constant tile geometry for prefix invariance.
            first = max(0, start + 1 - window)
            swa_width = start + ROWS - first
            swa_valid = stop - first
            swa_k = F.pad(kv[first:stop], (0, 0, 0, swa_width - swa_valid))
            q_id = torch.arange(start, start + ROWS, device=device)
            k_id = torch.arange(first, start + ROWS, device=device)
            dist = q_id[:, None] - k_id[None, :]
            swa_allowed = (dist >= 0) & (dist < window) & (k_id[None, :] < stop)

            n_comp_tile = (stop) // COMPRESS_RATIO_HCA
            if n_comp_tile > 0 and comp_kv is not None:
                c_k = comp_kv[:n_comp_tile]
                c_id = torch.arange(n_comp_tile, device=device)
                # Chunk c (covering [128c .. 128c+127]) is visible for pos >= 128c + 127, i.e. c < (pos+1)//128.
                comp_allowed = c_id[None, :] < ((q_id[:, None] + 1) // COMPRESS_RATIO_HCA)
                all_k = torch.cat((c_k, swa_k), dim=0)
                allowed = torch.cat((comp_allowed, swa_allowed), dim=1)
            else:
                all_k = swa_k
                allowed = swa_allowed

            scores = torch.matmul(tile_q, all_k.t()) * scale
            scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
            scores = torch.cat((scores, sink.expand(-1, ROWS, -1)), dim=-1)
            probs = (scores - scores.max(dim=-1, keepdim=True).values).softmax(dim=-1)[..., :-1]
            out_tiles.append(torch.matmul(probs, all_k).transpose(0, 1)[: stop - start])
        o = torch.cat(out_tiles, dim=0)

    o = vha_postmix(o, weights[p + "vha_postmix_U"], weights[p + "vha_postmix_V"])
    o = rotate_tail(o, cos, -sin)
    # Grouped wo_a: [T, 8, 8*512] -> [T, 8, 1024] -> [T, 8192]
    wo_a = weights[p + "linear_o_group_proj"].view(og, ol, (nh // og) * hd)
    o_groups = o.view(seq_len, og, (nh // og) * hd)
    z = torch.stack([linear(o_groups[:, g], wo_a[g]) for g in range(og)], dim=1).reshape(seq_len, og * ol)
    gate = torch.sigmoid(linear(q_c, weights[p + "gate_proj.weight"]))
    return linear(z * gate, weights[p + "o_proj.weight"])


def _rotate_leading_split_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rope_dim = cos.shape[-1]
    lead, tail = x.split([rope_dim, x.shape[-1] - rope_dim], dim=-1)
    return torch.cat((rotate_split_half(lead, cos, sin), tail), dim=-1)


def dsa_select_topk(
    x: torch.Tensor,
    q_c: torch.Tensor,
    weights: dict,
    c: Config,
    positions: torch.Tensor,
    *,
    enable_flashindexer: bool = False,
) -> torch.Tensor:
    """Compute DSA top-k indices strictly before the SWA window (0 .. pos - sliding_window).

    Returns int64 [T, index_topk] with -1 for invalid/unneeded slots (queries with
    prefix_end <= index_topk do not prune any prefix token).
    """
    seq_len = x.shape[0]
    device = x.device
    inh = c.index_n_heads
    ihd = c.index_head_dim
    topk = c.index_topk
    window = c.sliding_window
    ip = "self_attn.core_attention.indexer."

    # Short sequences never need top-k pruning (all tokens <= window + topk).
    if seq_len <= window + topk:
        return torch.full((seq_len, topk), -1, dtype=torch.long, device=device)

    cos_idx, sin_idx = phase_split_half(positions, c.hybrid_mla_qk_rope_head_dim, float(c.rope_theta))
    iq = linear(q_c, weights[ip + "wq_b.weight"]).view(seq_len, inh, ihd)
    iq = _rotate_leading_split_half(iq, cos_idx, sin_idx)
    ik = linear(x, weights[ip + "wk.weight"])
    ik = F.layer_norm(ik, (ihd,), weights[ip + "k_norm.weight"], weights[ip + "k_norm.bias"], eps=1e-6)
    ik = _rotate_leading_split_half(ik.unsqueeze(1), cos_idx, sin_idx).squeeze(1)
    iw = linear(x, weights[ip + "weights_proj.weight"]) * (ihd ** -0.5) * (inh ** -0.5)

    if enable_flashindexer:
        gates = linear(x, weights[ip + "linear_gate.weight"])
        head_scores = gates + weights[ip + "flashindexer_bias"]
        sel_heads = head_scores.topk(FLASHINDEXER_TOPK_HEADS, dim=-1).indices
        iq = torch.take_along_dim(iq, sel_heads.unsqueeze(-1), dim=1)
        iw = torch.gather(iw, 1, sel_heads) * torch.gather(gates, 1, sel_heads)

    indices = torch.full((seq_len, topk), -1, dtype=torch.long, device=device)
    first_pruned = window + topk
    for start in range(first_pruned, seq_len, ROWS):
        stop = min(start + ROWS, seq_len)
        max_k = stop - window
        k_tile = ik[:max_k]  # [max_k, ihd]
        q_tile = iq[start:stop]  # [B, H, ihd]
        w_tile = iw[start:stop]  # [B, H]
        # scores[b, j] = sum_h w[b, h] * relu(dot(q[b, h], k[j]))
        raw = torch.einsum("bhd,kd->bhk", q_tile, k_tile).relu()
        agg = torch.einsum("bh,bhk->bk", w_tile, raw)
        q_pos = torch.arange(start, stop, device=device)
        k_pos = torch.arange(max_k, device=device)
        valid = k_pos[None, :] < (q_pos[:, None] + 1 - window)
        agg = agg.masked_fill(~valid, -float("inf"))
        indices[start:stop] = agg.topk(topk, dim=-1, sorted=False).indices
    return indices


def mla_dsa_attention(
    x: torch.Tensor,
    weights: dict,
    c: Config,
    positions: torch.Tensor,
    *,
    enable_flashindexer: bool = False,
) -> torch.Tensor:
    """MLA-DSA attention (layers 7/14/21/28/35/42): top-2048(prefix before SWA) union SWA-128."""
    seq_len = x.shape[0]
    device = x.device
    nh = c.hybrid_mla_num_attention_heads
    kvl = c.hybrid_mla_kv_lora_rank
    nope = c.hybrid_mla_qk_nope_head_dim
    rope = c.hybrid_mla_qk_rope_head_dim
    vd = c.hybrid_mla_v_head_dim
    window = c.sliding_window
    topk = c.index_topk
    p = "self_attn."

    cos, sin = phase(positions, rope, float(c.rope_theta))
    q_c = norm(linear(x, weights[p + "q_a_proj.weight"]), weights[p + "q_a_layernorm.weight"], c.rms_norm_eps)
    q = linear(q_c, weights[p + "q_b_proj.weight"]).view(seq_len, nh, nope + rope)
    q = rotate_tail(q, cos, sin)

    kv_a = linear(x, weights[p + "kv_a_proj_with_mqa.weight"])
    kv_c, k_pe = kv_a.split([kvl, rope], dim=-1)
    kv_c = norm(kv_c, weights[p + "kv_a_layernorm.weight"], c.rms_norm_eps)
    k_pe = rotate(k_pe.unsqueeze(1), cos, sin)  # [T, 1, rope]
    k_nope, v = linear(kv_c, weights[p + "kv_b_proj.weight"]).view(seq_len, nh, nope + vd).split([nope, vd], dim=-1)
    k = torch.cat((k_nope, k_pe.expand(-1, nh, -1)), dim=-1)  # [T, nh, nope+rope]

    topk_indices = dsa_select_topk(x, q_c, weights, c, positions, enable_flashindexer=enable_flashindexer)
    sink = weights[p + "core_attention.softmax_offset"][:, None, None]
    scale = (nope + rope) ** -0.5

    out_tiles = []
    for start in range(0, seq_len, ROWS):
        stop = min(start + ROWS, seq_len)
        width = start + ROWS
        tile_q = F.pad(q[start:stop].transpose(0, 1), (0, 0, 0, ROWS - (stop - start)))
        tile_k = F.pad(k[:stop].transpose(0, 1), (0, 0, 0, width - stop))
        tile_v = F.pad(v[:stop].transpose(0, 1), (0, 0, 0, width - stop))

        q_id = torch.arange(start, start + ROWS, device=device)
        k_id = torch.arange(width, device=device)
        dist = q_id[:, None] - k_id[None, :]
        causal = (dist >= 0) & (k_id[None, :] < stop)

        if stop > window + topk:
            allowed = torch.zeros((ROWS, width), dtype=torch.bool, device=device)
            # Rows with prefix_end <= topk attend to full causal prefix.
            dense_rows = (q_id + 1 - window) <= topk
            allowed = torch.where(dense_rows[:, None], causal, allowed)
            # Pruned rows: union of top-k prefix indices + trailing SWA window.
            swa_mask = causal & (dist < window)
            allowed |= swa_mask
            pruned_start = max(start, window + topk)
            if pruned_start < stop:
                row_slice = slice(pruned_start - start, stop - start)
                idx = topk_indices[pruned_start:stop]
                valid_idx = idx >= 0
                safe_idx = idx.clamp_min(0)
                pruned_mask = torch.zeros((stop - pruned_start, width), dtype=torch.bool, device=device)
                pruned_mask.scatter_(1, safe_idx, valid_idx)
                allowed[row_slice] |= pruned_mask
            allowed &= causal
        else:
            allowed = causal

        scores = torch.matmul(tile_q, tile_k.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
        scores = torch.cat((scores, sink.expand(-1, ROWS, -1)), dim=-1)
        probs = (scores - scores.max(dim=-1, keepdim=True).values).softmax(dim=-1)[..., :-1]
        out_tiles.append(torch.matmul(probs, tile_v).transpose(0, 1)[: stop - start])

    o = torch.cat(out_tiles, dim=0)
    o = vha_postmix(o, weights[p + "vha_postmix_U"], weights[p + "vha_postmix_V"])
    gate = torch.sigmoid(linear(q_c, weights[p + "gate_proj.weight"]))
    return linear(o.reshape(seq_len, nh * vd) * gate, weights[p + "o_proj.weight"])


def dense_ffn(x: torch.Tensor, weights: dict, c: Config) -> torch.Tensor:
    gate = linear(x, weights["mlp.gate_proj.weight"])
    up = linear(x, weights["mlp.up_proj.weight"])
    act = situ(gate, up, float(c.activation_situ_beta), float(c.activation_situ_linear_beta))
    return linear(act, weights["mlp.down_proj.weight"])


def moe_routing(x: torch.Tensor, weights: dict, c: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-router sigmoid routing with e_score_correction_bias and per-expert learnable scale."""
    p = "block_sparse_moe."
    l0 = linear(x, weights[p + "gate.weight"])
    l1 = linear(x, weights[p + "gate.weight_1"])
    scores = torch.sigmoid(l0) + torch.sigmoid(l1)
    choice = scores + weights[p + "e_score_correction_bias"]
    ids = choice.topk(c.num_experts_per_tok, dim=-1, sorted=False).indices
    probs = scores.gather(1, ids)
    if c.norm_topk_prob:
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    probs = probs * weights[p + "gate.routed_scaling_factor_param"][ids]
    return ids, probs


def latent_moe_ffn(x: torch.Tensor, weights: dict, c: Config) -> torch.Tensor:
    """Multi-router latent MoE: fc2_latent_proj(latent_norm(routed)) + shared_experts(x)."""
    p = "block_sparse_moe."
    beta = float(c.activation_situ_beta)
    linear_beta = float(c.activation_situ_linear_beta)

    ids, probs = moe_routing(x, weights, c)
    latent = linear(x, weights[p + "fc1_latent_proj.weight"])
    routed = torch.zeros_like(latent)

    store = weights.get("experts")
    if store is None:
        store = DictExpertStore(weights)

    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=c.n_routed_experts).tolist()
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        slots = order[start : start + count]
        tokens = slots // c.num_experts_per_tok
        w1, w3, w2 = store.get(expert, x.device)
        inp = latent[tokens]
        act = situ(linear(inp, w1), linear(inp, w3), beta, linear_beta)
        out = linear(act, w2)
        routed[tokens] = routed[tokens] + out * probs.flatten()[slots, None]
        start += count

    routed = norm(routed, weights[p + "latent_norm.weight"], c.rms_norm_eps)
    routed = linear(routed, weights[p + "fc2_latent_proj.weight"])

    s_gate = linear(x, weights[p + "shared_experts.w1.weight"])
    s_up = linear(x, weights[p + "shared_experts.w3.weight"])
    shared = linear(situ(s_gate, s_up, beta, linear_beta), weights[p + "shared_experts.w2.weight"])
    return routed + shared


def layer_forward(
    hidden: torch.Tensor,
    weights: dict,
    c: Config,
    index: int,
    positions: torch.Tensor | None = None,
    *,
    enable_flashindexer: bool = False,
) -> torch.Tensor:
    """Run one ERNIE 5.5 target decoder layer: hidden [T, hc_mult, H] -> [T, hc_mult, H]."""
    seq_len = hidden.shape[0]
    if positions is None:
        positions = torch.arange(seq_len, device=hidden.device, dtype=torch.long)

    post_a, comb_a, attn_in = mhc_pre(
        hidden, weights, "self_attention_hyper_connection", weights["input_layernorm.weight"], c
    )
    if c.is_dsa_layer(index):
        attn_out = mla_dsa_attention(attn_in, weights, c, positions, enable_flashindexer=enable_flashindexer)
    else:
        attn_out = hca_attention(attn_in, weights, c, positions, rope_mode="yarn")
    hidden = mhc_post(attn_out, hidden, post_a, comb_a)

    post_m, comb_m, mlp_in = mhc_pre(
        hidden, weights, "mlp_hyper_connection", weights["post_attention_layernorm.weight"], c
    )
    if c.is_moe_layer(index):
        mlp_out = latent_moe_ffn(mlp_in, weights, c)
    else:
        mlp_out = dense_ffn(mlp_in, weights, c)
    return mhc_post(mlp_out, hidden, post_m, comb_m)


def multimax(logits: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """Piece-wise multimax logit transform."""
    r = ranges.to(device=logits.device, dtype=logits.dtype)
    t = ts.to(device=logits.device, dtype=logits.dtype)
    out = logits + t[0] * (r[0] - logits).relu()
    out = out + t[1] * (logits - r[1]).relu()
    out = out + t[2] * (r[2] - logits).relu().square()
    out = out + t[3] * (logits - r[3]).relu().square()
    return out


def head_features(
    hidden: torch.Tensor,
    outer: dict[str, torch.Tensor],
    c: Config,
    *,
    eps_outside_sqrt: bool = False,
) -> torch.Tensor:
    collapsed = hc_head_pre_norm(
        hidden,
        outer[OUTER_NAMES[1]],
        outer[OUTER_NAMES[2]],
        outer[OUTER_NAMES[3]],
        c,
        eps_outside_sqrt=eps_outside_sqrt,
    )
    return norm(collapsed, outer[OUTER_NAMES[4]], c.rms_norm_eps)


def lm_head_logits(
    hidden: torch.Tensor,
    outer: dict[str, torch.Tensor],
    c: Config,
    *,
    apply_multimax: bool = True,
    eps_outside_sqrt: bool = False,
    to_cpu: bool = False,
) -> torch.Tensor:
    """Collapse mHC streams -> final RMSNorm -> untied lm_head -> optional multimax (tiled)."""
    features = head_features(hidden, outer, c, eps_outside_sqrt=eps_outside_sqrt)
    lm_w = outer[OUTER_NAMES[5]]
    ranges = outer[OUTER_NAMES[6]]
    ts = outer[OUTER_NAMES[7]]
    chunks = []
    for tile in features.split(ROWS):
        row_logits = linear(tile, lm_w)
        if apply_multimax:
            row_logits = multimax(row_logits, ranges, ts)
        chunks.append(row_logits.cpu() if to_cpu else row_logits)
    return torch.cat(chunks, dim=0)


def dspark_project_target_taps(
    taps: list[torch.Tensor] | torch.Tensor,
    mtp_aux: dict[str, torch.Tensor],
    c: Config,
) -> torch.Tensor:
    """Combine stream-averaged outputs of target layers [40, 41, 42] into main_x [T, H]."""
    if isinstance(taps, (list, tuple)):
        cat = torch.cat(taps, dim=-1)
    else:
        cat = taps
    return norm(
        linear(cat, mtp_aux["feature_projection.weight"]),
        mtp_aux["feature_norm.weight"],
        c.rms_norm_eps,
    )


def dspark_stage_forward(
    hidden: torch.Tensor,
    stage_weights: dict,
    c: Config,
    block_positions: torch.Tensor,
    context_kv: torch.Tensor,
    *,
    draft_rope_mode: str = DEFAULT_DSPARK_ROPE_MODE,
) -> torch.Tensor:
    """Run one DSpark draft stage (non-causal SWA-only HCA + multi-router latent MoE)."""
    post_a, comb_a, attn_in = mhc_pre(
        hidden, stage_weights, "self_attention_hyper_connection", stage_weights["input_layernorm.weight"], c
    )
    attn_out = hca_attention(
        attn_in,
        stage_weights,
        c,
        block_positions,
        context_kv=context_kv,
        non_causal=True,
        rope_mode=draft_rope_mode,
    )
    hidden = mhc_post(attn_out, hidden, post_a, comb_a)

    post_m, comb_m, mlp_in = mhc_pre(
        hidden, stage_weights, "mlp_hyper_connection", stage_weights["post_attention_layernorm.weight"], c
    )
    mlp_out = latent_moe_ffn(mlp_in, stage_weights, c)
    return mhc_post(mlp_out, hidden, post_m, comb_m)


def dspark_propose(
    context_taps: list[torch.Tensor] | torch.Tensor,
    anchor_token: int,
    mtp_aux: dict[str, torch.Tensor],
    stage_weights: list[dict],
    outer: dict[str, torch.Tensor],
    c: Config,
    *,
    draft_rope_mode: str = DEFAULT_DSPARK_ROPE_MODE,
    multimax_after_markov: bool = DEFAULT_DSPARK_MULTIMAX_AFTER_MARKOV,
    eps_outside_sqrt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propose one DSpark block of dspark_block_size tokens (greedy Markov walk).

    Args:
        context_taps: list of 3 tensors [L_ctx, H] (mean over mHC streams of layers 40, 41, 42)
                      or pre-concatenated [L_ctx, 3*H].
        anchor_token: bonus token id at position L_ctx (predicts position L_ctx + 1).
        mtp_aux: loaded model.mtp.* aux weights (keyed without 'model.mtp.' prefix).
        stage_weights: list of 3 stage weight dicts (model.mtp.{0,1,2}.*).
        outer: loaded OUTER_NAMES dict (needs embed_tokens, lm_head, multimax_{ranges,ts}).

    Returns:
        (draft_tokens [B], draft_logits [B, V], confidence [B])
    """
    main_x = dspark_project_target_taps(context_taps, mtp_aux, c)
    device = main_x.device
    prefix_len = main_x.shape[0]
    block_size = int(c.dspark_block_size)
    noise_id = int(c.dspark_noise_token_id)

    ctx_start = max(0, prefix_len - c.sliding_window)
    ctx_x = main_x[ctx_start:prefix_len]
    ctx_positions = torch.arange(ctx_start, prefix_len, device=device, dtype=torch.long)
    block_positions = torch.arange(prefix_len, prefix_len + block_size, device=device, dtype=torch.long)

    block_ids = torch.full((block_size,), noise_id, dtype=torch.long, device=device)
    block_ids[0] = int(anchor_token)
    embed = F.embedding(block_ids, outer[OUTER_NAMES[0]])
    hidden = embed[:, None, :].expand(-1, c.hc_mult, -1).contiguous()

    c_cos, c_sin = hca_phase(ctx_positions, c, rope_mode=draft_rope_mode)
    for stage in stage_weights:
        ctx_kv = norm(
            linear(ctx_x, stage["self_attn.linear_kv_proj.weight"]),
            stage["self_attn.kv_layernorm.weight"],
            c.rms_norm_eps,
        )
        ctx_kv = rotate_tail(ctx_kv.unsqueeze(1), c_cos, c_sin).squeeze(1)
        hidden = dspark_stage_forward(
            hidden, stage, c, block_positions, ctx_kv, draft_rope_mode=draft_rope_mode
        )

    head_hidden = hc_head_pre_norm(
        hidden,
        mtp_aux["mhc_contract.hc_head_fn"],
        mtp_aux["mhc_contract.hc_head_base"],
        mtp_aux["mhc_contract.hc_head_scale"],
        c,
        eps_outside_sqrt=eps_outside_sqrt,
    )
    normed = norm(head_hidden, mtp_aux["norm.weight"], c.rms_norm_eps)
    base_logits = linear(normed, outer[OUTER_NAMES[5]])
    ranges, ts = outer[OUTER_NAMES[6]], outer[OUTER_NAMES[7]]
    if not multimax_after_markov:
        base_logits = multimax(base_logits, ranges, ts)

    markov_w1_t = mtp_aux["markov_head.markov_w1.weight"].t()  # [V, rank]
    markov_w2 = mtp_aux["markov_head.markov_w2.weight"]  # [V, rank]
    conf_w = mtp_aux["confidence_head.projection.weight"]  # [1, H + rank]

    prev = torch.tensor(int(anchor_token), dtype=torch.long, device=device)
    draft_tokens = []
    draft_logits = []
    confidences = []
    for i in range(block_size):
        me = F.embedding(prev, markov_w1_t)  # [rank]
        bias = linear(me.unsqueeze(0), markov_w2).squeeze(0)  # [V]
        logits_i = base_logits[i] + bias
        if multimax_after_markov:
            logits_i = multimax(logits_i, ranges, ts)
        tok_i = logits_i.argmax()
        conf_in = torch.cat((head_hidden[i], me), dim=-1).unsqueeze(0)
        conf_i = torch.sigmoid(linear(conf_in, conf_w)).squeeze()
        draft_tokens.append(tok_i)
        draft_logits.append(logits_i)
        confidences.append(conf_i)
        prev = tok_i

    return (
        torch.stack(draft_tokens),
        torch.stack(draft_logits),
        torch.stack(confidences),
    )


__all__ = [
    "COMPRESS_RATIO_HCA",
    "COMPRESS_RATIO_MLA",
    "Checkpoint",
    "Config",
    "DEFAULT_DSPARK_MULTIMAX_AFTER_MARKOV",
    "DEFAULT_DSPARK_ROPE_MODE",
    "DictExpertStore",
    "FLASHINDEXER_FILE",
    "FLASHINDEXER_TOPK_HEADS",
    "HC_POST_ALPHA",
    "LazyExpertStore",
    "MTP_AUX_NAMES",
    "OUTER_NAMES",
    "PackedBF16ExpertStore",
    "ROWS",
    "VHA_RANK",
    "configure_f32",
    "dense_ffn",
    "dsa_select_topk",
    "dspark_project_target_taps",
    "dspark_propose",
    "dspark_stage_forward",
    "hc_head_pre_norm",
    "hca_attention",
    "hca_phase",
    "head_features",
    "inverse_rms",
    "latent_moe_ffn",
    "layer_forward",
    "linear",
    "lm_head_logits",
    "mhc_post",
    "mhc_pre",
    "mla_dsa_attention",
    "moe_routing",
    "mtp_contract",
    "multimax",
    "norm",
    "phase",
    "phase_split_half",
    "rotate",
    "rotate_split_half",
    "rotate_tail",
    "situ",
    "tensor_contract",
    "vha_postmix",
    "yarn_phase",
    "flashindexer_contract",
]
