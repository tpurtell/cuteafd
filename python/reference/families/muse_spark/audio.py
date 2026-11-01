"""Official Muse Spark raw-PCM tower reference (transformers 5.22.1).

Equations follow modeling_muse_spark.py MuseSparkAudioModel and
feature_extraction_muse_spark.py; all intermediate stores honor BF16. This is
an independent reference, not a serving fallback. The bundled checkpoint owns
both the architecture and processor geometry.
"""
from __future__ import annotations

import json
from pathlib import Path

PREFIX = "model.audio_tower."


def config(snapshot):
    root = json.loads((Path(snapshot) / "config.json").read_text())
    processor = json.loads((Path(snapshot) / "processor_config.json").read_text())["feature_extractor"]
    audio = root["audio_config"]
    if root["model_type"] != "muse_spark" or audio["model_type"] != "muse_spark_audio":
        raise ValueError("requires the checkpoint's bundled Muse Spark audio tower")
    if (audio["hidden_sizes"] != [256, 512, 1024, 2048] or audio["depths"] != [2] * 4
            or audio["patch_sizes"] != [16, 6, 5, 4] or audio["conv_kernel_size"] != 7
            or audio["mlp_ratio"] != 4 or audio["hidden_act"] != "gelu"
            or audio["rms_norm_eps"] != 1e-6 or audio["out_hidden_size"] != root["text_config"]["hidden_size"]):
        raise ValueError("unsupported Muse audio geometry; add matching PCM exports")
    if (processor["sampling_rate"] != 24000 or processor["samples_per_token"] != 1920
            or processor["feature_size"] != 1 or processor["padding_value"] != 0
            or processor["padding_side"] != "right"):
        raise ValueError("unsupported Muse PCM processor")
    return audio, processor


def prepare(waveform, *, sampling_rate=24000):
    import torch
    if sampling_rate != 24000:
        raise ValueError("Muse audio requires resampled 24000 Hz mono PCM")
    x = torch.as_tensor(waveform, dtype=torch.float32)
    if x.ndim != 1 or not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Muse audio requires nonempty finite mono PCM")
    tokens = token_count(x.numel())
    return torch.nn.functional.pad(x, (0, tokens * 1920 - x.numel())), tokens


def token_count(samples):
    if not 1 <= samples <= 7_200_000:
        raise ValueError("PCM must contain 1..7200000 samples")
    return (samples + 1919) // 1920


def placeholder(samples):
    return "<|soft_audio_start|>" + "<|audio|>" * token_count(samples) + "<|soft_audio_end|>"


def load_weights(snapshot, *, device="cpu"):
    from safetensors import safe_open
    snapshot = Path(snapshot)
    index = snapshot / "model.safetensors.index.json"
    shards = ({v for k, v in json.loads(index.read_text())["weight_map"].items() if k.startswith(PREFIX)}
              if index.exists() else {"model.safetensors"})
    weights = {}
    for shard in sorted(shards):
        with safe_open(snapshot / shard, framework="pt", device=device) as reader:
            for key in reader.keys():
                if key.startswith(PREFIX):
                    weights[key[len(PREFIX):]] = reader.get_tensor(key)
    if len(weights) != 72:
        raise ValueError(f"expected 72 bundled audio tensors, got {len(weights)}")
    return weights


def forward(waveform, weights, *, observer=None):
    import torch
    import torch.nn.functional as F
    def norm(x, key):
        f = x.float()
        return (f * torch.pow(f.square().mean(-1, keepdim=True) + 1e-6, -0.5)
                * weights[key].float()).to(x.dtype)
    def linear(x, key):
        return F.linear(x, weights[key + ".weight"], weights[key + ".bias"])
    def observe(stage, x):
        if observer is not None:
            observer(stage, x)
    with torch.inference_mode():
        pcm, _ = prepare(waveform)
        first = next(iter(weights.values()))
        x = pcm.to(device=first.device, dtype=torch.bfloat16).view(1, -1, 1)
        for stage, patch in enumerate((16, 6, 5, 4)):
            key = f"layers.{stage}"
            x = norm(linear(x.reshape(1, x.shape[1] // patch, -1), key + ".patch_embedder.proj"),
                     key + ".patch_embedder.norm.weight")
            observe(stage * 3, x)
            for block in range(2):
                b = key + f".conv_blocks.{block}"
                residual = x
                x = F.conv1d(F.pad(x.transpose(1, 2), (6, 0)), weights[b + ".conv.weight"],
                             weights[b + ".conv.bias"], groups=x.shape[-1]).transpose(1, 2)
                x = linear(F.gelu(linear(norm(x, b + ".norm.weight"), b + ".mlp.fc1"),
                                  approximate="none"), b + ".mlp.fc2") + residual
                observe(stage * 3 + block + 1, x)
        x = norm(linear(norm(x, "norm.weight"), "proj"), "post_proj_norm.weight")
        observe(12, x)
        return x.squeeze(0)
