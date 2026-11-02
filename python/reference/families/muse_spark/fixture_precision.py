"""Official Muse Spark model on the engine fixture at BF16, FP32 and FP64.

Writes `official_f32.json` / `official_f64.json` beside `official.json`: the same
checkpoint (BF16-valued weights, upcast exactly) run with every module in FP32
or FP64. The BF16-vs-FP32 gap is the official model's own BF16 rounding/order
noise; an oracle run without BF16 rounding must match the FP64 logits to
~1e-6, which proves the oracle's semantics independent of rounding.

    PYTHONPATH=<transformers 5.22.1> python3 fixture_precision.py FIXTURE_DIR
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main(fixture: Path) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
    from transformers.models.muse_spark import modeling_muse_spark as reference

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from families.muse_spark import model as muse
    muse.source_identity()
    torch.set_num_threads(1)
    record = json.loads((fixture / "official.json").read_bytes())
    text = {k: v for k, v in record["config"]["text_config"].items() if k not in ("model_type",)}
    tensors = load_file(str(fixture / "model.safetensors"))
    tokens = record["tokens"]
    for dtype, name in ((torch.float32, "official_f32.json"), (torch.float64, "official_f64.json")):
        config = MuseSparkTextConfig(**text)
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        with torch.device("meta"):
            model = reference.MuseSparkTextModel(config)
        for key, value in tensors.items():
            if key == muse.HEAD:
                continue
            path = key[len(muse.PREFIX):]
            parent, _, leaf = path.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            if leaf in owner._parameters:
                owner._parameters[leaf] = torch.nn.Parameter(value.to(dtype), requires_grad=False)
            else:
                owner._buffers[leaf] = value.to(dtype)
        model.rotary_emb = reference.MuseSparkTextRotaryEmbedding(config)
        model = model.to(dtype).eval()
        with torch.no_grad():
            hidden = model(input_ids=torch.tensor([tokens])).last_hidden_state
            logits = muse.logits(hidden, tensors[muse.HEAD].to(dtype), config)
        (fixture / name).write_text(json.dumps({"logits": logits[0].double().tolist()}) + "\n")
        bf16 = torch.tensor(record["logits"], dtype=torch.float64)
        print(f"{name}: max |official bf16 - official {dtype}| = {(logits[0].double() - bf16).abs().max():.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
