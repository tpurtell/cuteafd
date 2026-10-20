"""Direct Step-5 MTP wiring oracle, with F32 reductions and checkpoint boundaries.

Semantics: StepFun-linked vLLM PR #61727, pinned below, not the target-only
inference generator. Each stage owns its SWA cache and vocabulary head. The
caller supplies checkpoint RoPE tables and already shifted token embeddings.
This module is reference math, not a serving or benchmarking driver.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

VLLM_COMMIT = "dde9aca41cd3ccceb1ffa902933b6678a09c83ce"
VLLM_SOURCES = {
    "vllm/models/step5/nvidia/ops/attention_gate.py":
        "1d49da911ce9c24cf2be11bcc517c57dc00748d85585b3a6d6070e1fc421d020",
    "vllm/model_executor/layers/logits_processor.py":
        "35cf8235cb529ba2d37358fe9b9e5a29f40b5c3cbcc9e6750c8480c51bd7da83",
    "vllm/models/step5/nvidia/model.py":
        "c912d120041dfe84b93e76a003d9180207e14154cff6d2f296c6568bb40085f4",
    "vllm/models/step5/nvidia/mtp.py":
        "24776e01df7f817d05ed3de1e1d1adfe7b156323311c1c27cf71f575d1cea71c",
    "vllm/models/step5/nvidia/layernorm.py":
        "f6213b9466d2bfe62fa9021dc8404b3c8413ff4a08e964b5b43dbe110d31b510",
    "vllm/v1/spec_decode/step3p5.py":
        "a4b61f806d6efec4ae0c8ef2cc4d3f72f75aa88e5bf3c218f155a8c2b88b6c26",
}


def provenance(source_root, *, head_dtype="float32"):
    """Attest the source read, without importing an external execution engine."""
    root = Path(source_root)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in VLLM_SOURCES}
    if hashes != VLLM_SOURCES:
        raise ValueError("Step-5 MTP semantics source differs from pinned vLLM")
    if head_dtype not in ("float32", "bfloat16"):
        raise ValueError("Step-5 MTP head dtype")
    return {"vllm_commit": VLLM_COMMIT, "source_sha256": hashes,
            "compute": "float32", "checkpoint_boundaries": "bfloat16",
            "head_dtype": head_dtype, "vllm_default_head_dtype": "bfloat16"}


def _bf16(value):
    return value.bfloat16().float()


def norm(value, weight, epsilon, *, narrow=True):
    import torch
    value = value.float()
    result = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    result = result * (1.0 + weight.float())
    return _bf16(result) if narrow else result


def norm_concat(embedding, previous_hidden, weights, epsilon):
    """vLLM mtp.py:148-155: narrow both inputs, then [enorm(e), hnorm(h)]."""
    import torch
    if embedding.shape != previous_hidden.shape or embedding.ndim != 2:
        raise ValueError("Step-5 MTP embedding/hidden geometry differs")
    return torch.cat((norm(_bf16(embedding), weights["enorm.weight"], epsilon),
                      norm(_bf16(previous_hidden), weights["hnorm.weight"], epsilon)), -1)


def teacher_inputs(tokens, hidden, positions):
    """Pair hidden at t with the already selected token and position at t+1.

    Apply once for stage 0 and again to each stage's teacher-forced output.
    This is not a free-running acceptance measurement.
    """
    if len(tokens) != len(hidden) or len(tokens) != len(positions) or len(tokens) < 2:
        raise ValueError("Step-5 MTP teacher forcing needs aligned consecutive rows")
    return tokens[1:], hidden[:-1], positions[1:]


class Stage:
    """One independently owned dense SWA stage; weights use checkpoint suffixes.

    F32 GEMMs/reductions preserve the vLLM BF16 operand/output boundaries.
    The unnormalized F32 residual is passed to the next predictor, while the
    stage-local shared-head norm/output produce logits. The caller owns
    rollback: clone/restore cache tensors before speculative execution.
    """
    def __init__(self, weights, *, heads=64, groups=4, dim=192, window=512,
                 epsilon=1e-5, limit=7.0, head_dtype="float32"):
        if head_dtype not in ("float32", "bfloat16"):
            raise ValueError("Step-5 MTP head dtype")
        self.head_dtype = head_dtype
        if heads < 1 or groups < 1 or heads % groups or dim < 2 or dim % 2 or window < 1:
            raise ValueError("Step-5 MTP SWA geometry")
        self.weights = weights
        self.heads, self.groups, self.dim = heads, groups, dim
        self.window, self.epsilon, self.limit = window, epsilon, limit
        self.cache = None

    @classmethod
    def from_checkpoint(cls, reader, config, stage, *, device="cpu", head_dtype="float32"):
        """Bind native checkpoint names through the target reference's Weights.get."""
        import torch
        cfg = config.get("text_config", config)
        layers, depth = cfg["num_hidden_layers"], cfg["num_nextn_predict_layers"]
        if (cfg["model_type"] != "step5" or layers != 92 or depth != 3
                or type(stage) is not int or not 0 <= stage < depth
                or cfg["norm_dtype"] != "float32" or not cfg["fp32_residual_connection"]):
            raise ValueError("Step-5 MTP checkpoint configuration")
        layer = layers + stage
        if cfg["layer_types"][layer] != "sliding_attention":
            raise ValueError("Step-5 MTP requires a sliding stage")
        h, heads, groups, dim, inter, vocab = (cfg[name] for name in
            ("hidden_size", "num_attention_heads", "num_attention_groups", "head_dim",
             "intermediate_size", "vocab_size"))
        shapes = {name: (h,) for name in ("enorm.weight", "hnorm.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "transformer.shared_head.norm.weight")}
        shapes.update({"eh_proj.weight": (h, 2*h),
            "self_attn.q_norm.weight": (dim,), "self_attn.k_norm.weight": (dim,),
            "self_attn.q_proj.weight": (heads*dim, h),
            "self_attn.k_proj.weight": (groups*dim, h),
            "self_attn.v_proj.weight": (groups*dim, h),
            "self_attn.g_proj.weight": (heads, h),
            "self_attn.o_proj.weight": (h, heads*dim),
            "mlp.gate_proj.weight": (inter, h), "mlp.up_proj.weight": (inter, h),
            "mlp.down_proj.weight": (h, inter),
            "transformer.shared_head.output.weight": (vocab, h)})
        weights = {}
        for suffix, shape in shapes.items():
            name = f"model.layers.{layer}.{suffix}"
            value = reader.get(name)
            dtype = torch.float32 if len(shape) == 1 else torch.bfloat16
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"{name}: native MTP shape/dtype differs")
            weights[suffix] = value.to(device=device, dtype=torch.float32)
        return cls(weights, heads=heads, groups=groups, dim=dim,
                   window=cfg["sliding_window"], epsilon=cfg["rms_norm_eps"],
                   limit=cfg["swiglu_limits_shared"][layer], head_dtype=head_dtype)

    def _linear(self, value, name):
        return _bf16(value.float() @ self.weights[name].float().T)

    def _rope(self, value, positions, cos, sin):
        import torch
        pairs = cos.shape[-1]
        if sin.shape != cos.shape or pairs * 2 > self.dim:
            raise ValueError("Step-5 MTP rotary tables")
        c, s = cos[positions].float()[:, None], sin[positions].float()[:, None]
        real, imaginary = value[..., :pairs], value[..., pairs:2 * pairs]
        rotated = torch.cat((real * c - imaginary * s, real * s + imaginary * c,
                             value[..., 2 * pairs:]), -1)
        return _bf16(rotated)

    def forward(self, embedding, previous_hidden, positions, cos, sin):
        import torch
        w = self.weights
        positions = positions.long()
        if positions.ndim != 1 or len(positions) != len(embedding) or not len(positions):
            raise ValueError("Step-5 MTP positions geometry")
        if bool((positions < 0).any()) or bool((positions[1:] != positions[:-1] + 1).any()):
            raise ValueError("Step-5 MTP positions must be nonnegative and consecutive")
        residual = self._linear(norm_concat(embedding, previous_hidden, w, self.epsilon),
                                "eh_proj.weight")
        x = norm(residual, w["input_layernorm.weight"], self.epsilon)
        q, k, v = (self._linear(x, f"self_attn.{name}_proj.weight").reshape(
            len(x), count, self.dim) for name, count in
            (("q", self.heads), ("k", self.groups), ("v", self.groups)))
        q = self._rope(norm(q, w["self_attn.q_norm.weight"], self.epsilon), positions, cos, sin)
        k = self._rope(norm(k, w["self_attn.k_norm.weight"], self.epsilon), positions, cos, sin)
        if self.cache is None:
            all_positions, all_k, all_v = positions, k, v
        else:
            old_positions, old_k, old_v = self.cache
            if int(positions[0]) != int(old_positions[-1]) + 1:
                raise ValueError("Step-5 MTP cache requires an explicit rollback before replay")
            all_positions = torch.cat((old_positions, positions))
            all_k, all_v = torch.cat((old_k, k)), torch.cat((old_v, v))
        group = torch.arange(self.heads, device=q.device) // (self.heads // self.groups)
        outputs = []
        for query, position in zip(q, positions):
            keep = (all_positions <= position) & (all_positions > position - self.window)
            keys = all_k[keep][:, group]
            values = all_v[keep][:, group]
            scores = torch.einsum("hd,khd->hk", query, keys) * self.dim ** -0.5
            probabilities = torch.softmax(scores, -1)
            outputs.append(torch.einsum("hk,khd->hd", probabilities, values))
        attention = _bf16(torch.stack(outputs))
        gate = _bf16(torch.sigmoid(self._linear(x, "self_attn.g_proj.weight")))
        attention = _bf16(attention * gate[..., None])
        residual = residual + self._linear(attention.flatten(1), "self_attn.o_proj.weight")
        x = norm(residual, w["post_attention_layernorm.weight"], self.epsilon)
        gate = self._linear(x, "mlp.gate_proj.weight")
        up = self._linear(x, "mlp.up_proj.weight")
        activated = _bf16(torch.nn.functional.silu(gate).clamp(max=self.limit)
                          * up.clamp(-self.limit, self.limit))
        hidden = residual + self._linear(activated, "mlp.down_proj.weight")
        head = norm(_bf16(hidden), w["transformer.shared_head.norm.weight"], self.epsilon)
        # The F32 accuracy oracle matches an explicit vLLM head_dtype override;
        # its default model-dtype head has a BF16 projection-output boundary.
        logits = head @ w["transformer.shared_head.output.weight"].float().T
        if self.head_dtype == "bfloat16":
            logits = _bf16(logits)
        keep = all_positions > positions[-1] - self.window
        self.cache = tuple(value[keep].clone() for value in (all_positions, all_k, all_v))
        return hidden, logits
