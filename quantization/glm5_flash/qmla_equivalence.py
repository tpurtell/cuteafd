"""Diagnose the identity content transform on a committed GLM Flash MC pilot bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qmla.capture import _atomic_json, load_layer_capture
from qmla.fit import check_content_identity


@torch.inference_mode()
def diagnose(capture_dir, layer_id, device, checkpoint):
    from .qmla_mc import GlmFlashMCSource
    from .qmla_capture import Glm5FlashCaptureAdapter
    from .qmla_run import glm_replay
    source = GlmFlashMCSource(checkpoint, view="original")
    bank = load_layer_capture(capture_dir, layer_id, glm_replay(source, Glm5FlashCaptureAdapter(source.native_config()), device))
    forward = bank.provenance["capture_forward"]
    expected = dict(repo="wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC-MUL1",
                    revision="1bca67feffab2747e0e489e571185bc06336ee26",
                    view="original", routed_experts="exl3-k4-mul1")
    if any(forward.get(key) != value for key, value in expected.items()):
        raise ValueError("capture is not from the pinned MC original view")
    geom = bank.geometry
    if (geom.has_rope or geom.kind != "dsa" or geom.kv_lora_rank != 512
            or [len(bank.train), len(bank.val), len(bank.test)] != [16, 4, 4]
            or not bank.params.attention_arithmetic.startswith("bf16_")):
        raise ValueError("expected a GLM Flash NoPE native BF16 pilot layer and its 16/4/4 split")
    manifest = json.loads((capture_dir / f"layer-{layer_id:04d}.json").read_bytes())
    params = bank.params.to(device)
    for seq in bank.val:
        check_content_identity((seq.to(device),), params, geom)
    return dict(status="passed", quantization_off=True, content_basis_space="post_norm",
                transform="R=I, S=1", split="val", layer_id=layer_id, capture_forward=forward,
                capture_xxh3_128=[row["payload"]["xxh3_128"] for row in manifest["splits"]["val"]], attention_arithmetic=params.attention_arithmetic,
                byte_identical_to_native=True, sequences=len(bank.val), tokens=sum(s.seq_len for s in bank.val))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        result = diagnose(args.captures, args.layer, args.device, args.checkpoint)
    except ValueError as exc:
        _atomic_json(args.output, dict(status="failed", layer_id=args.layer, error=str(exc)))
        raise
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
