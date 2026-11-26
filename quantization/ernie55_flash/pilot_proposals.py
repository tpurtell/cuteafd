"""Compare complete reconstructed K8 dSpark core proposals with held-out BF16 blocks."""
from pathlib import Path

import torch
from safetensors import safe_open

from .forward import reference
from .source import reviewed_roles


def install_reconstruction(weights, path, prefix, expected):
    with safe_open(path, framework="pt", device="cpu") as saved:
        found = set(saved.keys())
        if found != expected:
            raise ValueError(f"{path}: missing {sorted(expected - found)}, extra {sorted(found - expected)}")
        for name in sorted(found):
            key = name.removeprefix(prefix)
            original = weights[key]
            compact = saved.get_tensor(name)
            if compact.shape != original.shape or not compact.is_floating_point():
                raise ValueError(f"{name}: invalid reconstructed projection {compact.dtype} {compact.shape}")
            weights[key] = compact.to(original)


def compare_blocks(blocks):
    if not blocks or any(not ref or len(ref) != len(got) for _, ref, got in blocks):
        raise ValueError("empty or incomplete dSpark proposal block")
    rows = [dict(prompt=name, reference=ref, k8=got,
                 matches=sum(a == b for a, b in zip(ref, got))) for name, ref, got in blocks]
    matches = sum(row["matches"] for row in rows)
    positions = sum(len(row["reference"]) for row in rows)
    return dict(passed=100 * matches >= 95 * positions, matches=matches, positions=positions,
                agreement=matches / positions, required_agreement=0.95, blocks=rows,
                scope="K8 feature projection and all three reconstructed K8 dSpark cores, "
                      "paired on BF16 target taps with native routed experts and identical anchors")


@torch.inference_mode()
def evaluate(model, fixtures, reconstruction):
    c = model.config
    device = model.checkpoint.device
    root = Path(reconstruction)
    auxiliary = model.checkpoint.mtp_aux(c)
    install_reconstruction(auxiliary, root / "feature.safetensors", "model.mtp.",
                           {"model.mtp.feature_projection.weight"})
    roles = reviewed_roles()
    stages = []
    for n in range(c.num_mtp_stages):
        prefix = f"model.mtp.{n}."
        stage = model.load_draft_layer(n)
        expected = {name for name, role in roles.items() if role == "core" and name.startswith(prefix)}
        install_reconstruction(stage, root / f"mtp-{n}.safetensors", prefix, expected)
        stages.append(stage)
    names = [reference.OUTER_NAMES[i] for i in (0, 5, 6, 7)]
    outer = {name: model.checkpoint.load(name) for name in names}
    blocks = []
    for path in sorted((Path(fixtures) / "heldout").glob("*.pt")):
        row = torch.load(path, weights_only=True, map_location="cpu")
        taps = [row[f"tap_{idx}"].to(device) for idx in c.dspark_target_layer_ids]
        tokens, logits, confidence = model.native_proposal(taps, int(row["anchor"]), auxiliary, stages, outer)
        baseline = row["reference_proposal"]
        if tokens.shape != (c.dspark_block_size,) or baseline.shape != tokens.shape:
            raise ValueError(f"{path}: invalid proposal block length")
        if not (torch.isfinite(logits).all() & torch.isfinite(confidence).all()).item():
            raise ValueError(f"{path}: nonfinite dSpark output")
        blocks.append((path.stem, baseline.tolist(), tokens.tolist()))
    return compare_blocks(blocks)
