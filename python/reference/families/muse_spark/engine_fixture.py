"""Tiny Muse Spark checkpoint and its official logits, for the engine's CPU oracle.

Builds the official `MuseSparkTextModel` (pinned transformers, see model.py) at a
small geometry with every Muse mechanism present: a sliding (RoPE) and a full
(NoPE) layer, a window shorter than the prompt, top-2 of 6 experts with the
per-expert post norm, residual gates and the softcapped head. Weights are
stored as the release stores them (BF16, F32 router and bias); the official
model runs on those exact values in its own dtypes. The engine's Rust
reference (`cuteafd-daemon/src/families/muse_spark/reference.rs`) must match.

    PYTHONPATH=<transformers 5.22.1> python3 engine_fixture.py OUT_DIR
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main(out: Path) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
    from transformers.models.muse_spark import modeling_muse_spark as reference

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from families.muse_spark import model as muse
    muse.source_identity()

    torch.manual_seed(7)
    torch.set_num_threads(1)
    layers = 2
    text = dict(vocab_size=97, hidden_size=64, intermediate_size=64, moe_hidden_size=32, moe_intermediate_size=48,
                num_hidden_layers=layers, num_attention_heads=8, num_key_value_heads=2, head_dim=8,
                num_local_experts=6, num_experts_per_tok=2, sliding_window=5,
                layer_types=["sliding_attention", "full_attention"], mlp_layer_types=["sparse"] * layers,
                layer_rope_theta=[500000.0, 0], rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
                pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config = MuseSparkTextConfig(**text)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = reference.MuseSparkTextModel(config).eval()
    with torch.no_grad():
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            if value.dtype.is_floating_point and "inv_freq" not in name:
                value.copy_(torch.randn_like(value) * (0.3 if value.ndim > 1 else 0.5))
    head = torch.randn(config.vocab_size, config.hidden_size) * 0.3
    tensors = {muse.PREFIX + k: v.detach().clone() for k, v in model.state_dict().items() if "rotary" not in k}
    tensors[muse.HEAD] = head
    for key, value in tensors.items():
        if not key.endswith(("mlp.gate.weight", "e_score_correction_bias")):
            tensors[key] = value.bfloat16()
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"))
    # The official model on the stored values, in the checkpoint's dtypes.
    model = model.to(torch.bfloat16)
    with torch.no_grad():
        for key, value in tensors.items():
            if key == muse.HEAD:
                continue
            name = key[len(muse.PREFIX):]
            parent, _, leaf = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            target = owner._parameters.get(leaf) if leaf in owner._parameters else owner._buffers.get(leaf)
            replacement = value.clone()
            if leaf in owner._parameters:
                owner._parameters[leaf] = torch.nn.Parameter(replacement, requires_grad=False)
            else:
                owner._buffers[leaf] = replacement
            del target
    tokens = [1, 5, 9, 17, 33, 65, 3, 44, 12]
    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([tokens])).last_hidden_state
        logits = muse.logits(hidden, tensors[muse.HEAD], config)
    record = {"config": {"text_config": {**text, "model_type": "muse_spark_text", "hidden_act": "silu",
                                           "max_position_embeddings": 1024, "num_shared_experts": 0,
                                           "routed_scaling_factor": 1.0, "rms_norm_eps": config.rms_norm_eps,
                                           "post_norm_eps": config.post_norm_eps,
                                           "qk_scale_factor": config.qk_scale_factor,
                                           "output_multiplier": config.output_multiplier,
                                           "final_logit_softcapping": config.final_logit_softcapping,
                                           "residual_gate_temperature": config.residual_gate_temperature,
                                           "tie_word_embeddings": False},
                         "model_type": "muse_spark", "architectures": ["MuseSparkForConditionalGeneration"]},
              "tokens": tokens, "logits": logits[0].float().tolist(), "source": muse.source_identity()}
    (out / "official.json").write_text(json.dumps(record) + "\n")
    print(f"wrote {out}/model.safetensors and official.json: {len(tokens)} rows x {config.vocab_size}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
