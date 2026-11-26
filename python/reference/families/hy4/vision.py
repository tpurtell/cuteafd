"""Independent Hy4 (`hy_v4_vl`) 40-layer ViT reference over the checkpoint's own ``vit.*`` tensors.

Matches Hy4's bundled ViT geometry (`hidden=1536, heads=16, head_dim=96,
intermediate=6144, layers=40, out=6144, patch=16, merge=2, eps=1e-5`) and
SGLang's serving tower arithmetic (`models/hunyuan_v4_vl.py`).

CLI usage (for generating ``HY4_VISION_GOLDEN`` fixtures):
    python3 python/reference/families/hy4/vision.py \\
        --snapshot /path/to/Hy4-EXL3-K4-MC \\
        --out /path/to/out_dir
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PREFIX = "vit."
HIDDEN, HEADS, HEAD_DIM, INTER, LAYERS, PATCH, MERGE, OUT = 1536, 16, 96, 6144, 40, 16, 2, 6144
EDGE = 128  # max_image_size 2048 / patch 16
EPS = 1e-5
MODES = ("serving", "f32")


def contract():
    """Exact ``vit.*`` tensor names, shapes and BF16 storage (649 tensors)."""
    items = {
        "embeddings.patch_embedding.weight": [HIDDEN, 3, PATCH, PATCH],
        "embeddings.patch_embedding.bias": [HIDDEN],
        "embeddings.position_embedding.weight": [EDGE * EDGE + 1, HIDDEN],
        "perceive.image_newline": [OUT],
        "perceive.norm.weight": [HIDDEN],
        "perceive.proj.0.weight": [HIDDEN * 4, HIDDEN, MERGE, MERGE],
        "perceive.proj.0.bias": [HIDDEN * 4],
        "perceive.proj.2.weight": [OUT, HIDDEN * 4, 1, 1],
        "perceive.proj.2.bias": [OUT],
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            items[p + norm + ".weight"] = items[p + norm + ".bias"] = [HIDDEN]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            items[p + f"self_attn.{proj}.weight"] = [HIDDEN, HIDDEN]
            items[p + f"self_attn.{proj}.bias"] = [HIDDEN]
        items[p + "mlp.dense_h_to_4h.weight"] = [INTER, HIDDEN]
        items[p + "mlp.dense_h_to_4h.bias"] = [INTER]
        items[p + "mlp.dense_4h_to_h.weight"] = [HIDDEN, INTER]
        items[p + "mlp.dense_4h_to_h.bias"] = [HIDDEN]
    return {PREFIX + k: v for k, v in items.items()}


def validate_config(raw):
    v = raw["vision_config"]
    expected = dict(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        num_hidden_layers=LAYERS,
        patch_size=PATCH,
        spatial_merge_size=MERGE,
        out_hidden_size=OUT,
        max_image_size=2048,
        rms_norm_eps=EPS,
        hidden_act="gelu",
        interpolate_mode="bilinear",
        projector_type="rmsnorm_mlp_adapter",
        cat_extra_token=0,
        temporal_patch_size=1,
        num_channels=3,
    )
    bad = {k: v.get(k) for k, want in expected.items() if v.get(k) != want}
    if bad or v.get("use_2d_rope", False) or v.get("perceive_intermediate_size") not in (None, 0):
        raise ValueError(f"unsupported Hy4 vision_config: {bad}")
    if v.get("perceive_post_norm") is not False:
        raise ValueError("Hy4 vision perceive_post_norm must be false")


def load_weights(snapshot, device="cuda"):
    """BF16 tensors exactly as stored; the tower inventory must match ``contract()``."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    validate_config(json.loads((snapshot / "config.json").read_bytes()))
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
    want = contract()
    names = {n for n in index if n.startswith(PREFIX)}
    if names != set(want):
        raise ValueError(f"vit.* inventory differs: {sorted(names ^ set(want))[:8]}")
    weights = {}
    for shard in sorted({index[n] for n in names}):
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as reader:
            for name in sorted(n for n in names if index[n] == shard):
                value = reader.get_tensor(name)
                if value.dtype != torch.bfloat16 or list(value.shape) != want[name]:
                    raise ValueError(f"{name}: {value.dtype} {list(value.shape)}; need BF16 {want[name]}")
                weights[name.removeprefix(PREFIX)] = value.to(device)
    return weights


def output_rows(grid):
    t, h, w = grid
    if t != 1 or h % MERGE or w % MERGE or h <= 0 or w <= 0 or h > EDGE or w > EDGE:
        raise ValueError("Hy4 image grid must be [1, even h<=128, even w<=128]")
    return (h // MERGE) * (w // MERGE + 1)


@dataclass
class Stages:
    patch: torch.Tensor = None
    embed: torch.Tensor = None
    layers: list = field(default_factory=list)
    merger_norm: torch.Tensor = None
    output: torch.Tensor = None


def _rms(x, weight, mode):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * weight.float()
    return y if mode == "f32" else y.to(torch.bfloat16)


def _positions(table, h, w, dtype):
    grid = table[1:].float().reshape(1, EDGE, EDGE, HIDDEN).permute(0, 3, 1, 2)
    resized = F.interpolate(
        grid,
        scale_factor=((h + 0.1) / EDGE, (w + 0.1) / EDGE),
        mode="bilinear",
        align_corners=False,
    )
    if resized.shape[-2:] != (h, w):
        raise ValueError("position interpolation extent differs from grid")
    return resized.flatten(2).transpose(1, 2)[0].to(dtype)


def encode(weights, pixels, grid, *, mode="serving", observer=None):
    return encode_batch(weights, [pixels], [grid], mode=mode, observer=observer)[0]


def encode_batch(weights, pixels, grids, *, mode="serving", observer=None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    counts = [g[1] * g[2] for g in grids]
    for value, grid, n in zip(pixels, grids, counts):
        output_rows(grid)
        if tuple(value.shape) != (n, 3 * PATCH * PATCH):
            raise ValueError("pixel rows differ from grid")
    dtype = torch.float32 if mode == "f32" else torch.bfloat16
    W = {k: v.to(dtype) for k, v in weights.items()}
    device = W["embeddings.patch_embedding.weight"].device
    total = sum(counts)
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1] + n)
    stages = [Stages() for _ in grids]

    def keep(name, value):
        for j, s in enumerate(stages):
            part = value[offsets[j] : offsets[j + 1]]
            if name == "patch":
                s.patch = part
            elif name == "embed":
                s.embed = part
            elif name == "merger_norm":
                s.merger_norm = part
            else:
                s.layers.append(part)
        if observer is not None:
            observer(name, value)

    flat = torch.cat([v.to(device, dtype) for v in pixels])
    x = F.conv2d(
        flat.reshape(total, 3, PATCH, PATCH),
        W["embeddings.patch_embedding.weight"],
        W["embeddings.patch_embedding.bias"],
        stride=PATCH,
    ).reshape(total, HIDDEN)
    keep("patch", x)
    x = x + torch.cat(
        [_positions(weights["embeddings.position_embedding.weight"], g[1], g[2], dtype) for g in grids]
    )
    keep("embed", x)
    for i in range(LAYERS):
        p = f"layers.{i}."
        res = x
        y = F.layer_norm(x, (HIDDEN,), W[p + "input_layernorm.weight"], W[p + "input_layernorm.bias"], EPS)
        q, k, v = (
            F.linear(y, W[p + f"self_attn.{n_}_proj.weight"], W[p + f"self_attn.{n_}_proj.bias"])
            .reshape(total, HEADS, HEAD_DIM)
            .transpose(0, 1)
            for n_ in "qkv"
        )
        parts = []
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for a, b in zip(offsets, offsets[1:]):
                parts.append(
                    F.scaled_dot_product_attention(
                        q[None, :, a:b], k[None, :, a:b], v[None, :, a:b], scale=HEAD_DIM**-0.5
                    )[0]
                )
        o = torch.cat(parts, dim=1)
        o = F.linear(
            o.transpose(0, 1).reshape(total, HIDDEN),
            W[p + "self_attn.o_proj.weight"],
            W[p + "self_attn.o_proj.bias"],
        )
        x = res + o
        y = F.layer_norm(
            x, (HIDDEN,), W[p + "post_attention_layernorm.weight"], W[p + "post_attention_layernorm.bias"], EPS
        )
        y = F.gelu(F.linear(y, W[p + "mlp.dense_h_to_4h.weight"], W[p + "mlp.dense_h_to_4h.bias"]))
        x = x + F.linear(y, W[p + "mlp.dense_4h_to_h.weight"], W[p + "mlp.dense_4h_to_h.bias"])
        keep(f"layer{i}", x)
    norm = _rms(x, weights["perceive.norm.weight"], mode)
    keep("merger_norm", norm)
    for j, grid in enumerate(grids):
        _, h, w = grid
        y = norm[offsets[j] : offsets[j + 1]].reshape(1, h, w, HIDDEN).permute(0, 3, 1, 2)
        y = F.gelu(F.conv2d(y, W["perceive.proj.0.weight"], W["perceive.proj.0.bias"], stride=MERGE))
        y = F.conv2d(y, W["perceive.proj.2.weight"], W["perceive.proj.2.bias"])
        newline = W["perceive.image_newline"].reshape(1, OUT, 1, 1).expand(1, OUT, h // MERGE, 1)
        y = torch.cat((y, newline), dim=-1).flatten(2).transpose(1, 2)[0]
        if y.shape != (output_rows(grid), OUT):
            raise ValueError("merged output rows differ from grid")
        stages[j].output = y
        if observer is not None:
            observer(f"output{j}", y)
    return stages


def _rgb_to_raster_patches(rgb: np.ndarray, gh: int, gw: int) -> torch.Tensor:
    """Match ``vision_patches_raster_p16_t1``: (x/255 - 0.5)/0.5 in F32, raster (gh, gw, 3, 16, 16)."""
    arr = rgb.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 4, 1, 3)  # [gh, gw, 3, 16, 16]
    f = (arr.astype(np.float64) * (1.0 / 255.0)).astype(np.float32)
    f = (f - np.float32(0.5)) / np.float32(0.5)
    return torch.from_numpy(np.ascontiguousarray(f.reshape(gh * gw, 3 * PATCH * PATCH)))


def _write_f32(path: Path, t: torch.Tensor) -> None:
    arr = np.ascontiguousarray(t.detach().float().cpu().numpy(), dtype="<f4")
    path.write_bytes(arr.tobytes())


def _structured_rgb(gh: int, gw: int, variant: int) -> np.ndarray:
    """Deterministic structured synthetic render (gradients, shapes, bands) in uint8 [H, W, 3]."""
    h, w = gh * PATCH, gw * PATCH
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = yy / max(h - 1, 1)
    xn = xx / max(w - 1, 1)
    img = np.empty((h, w, 3), dtype=np.float32)
    if variant == 0:
        # Sky-to-ground gradient with geometric blocks and a centered circle.
        img[..., 0] = 35.0 + 185.0 * yn
        img[..., 1] = 80.0 + 135.0 * (1.0 - 0.5 * yn)
        img[..., 2] = 215.0 - 140.0 * yn
        img[h // 8 : h // 3, w // 10 : w // 3] = (220.0, 40.0, 40.0)
        img[h * 2 // 3 : h * 7 // 8, w // 2 : w * 9 // 10] = (240.0, 200.0, 30.0)
        mask = (xn - 0.65) ** 2 + (yn - 0.35) ** 2 <= 0.035
        img[mask] = (30.0, 175.0, 75.0)
    elif variant == 1:
        # Chart-like light canvas with gridlines and four vertical bars.
        img[:] = (248.0, 248.0, 244.0)
        img[::32, :] = (210.0, 210.0, 205.0)
        img[:, ::32] = (210.0, 210.0, 205.0)
        img[: h // 10, :] = (32.0, 44.0, 68.0)
        bars = [(0.12, 0.24, 0.35, (42, 120, 214)), (0.32, 0.44, 0.65, (214, 96, 42)),
                (0.52, 0.64, 0.48, (46, 160, 90)), (0.72, 0.84, 0.80, (140, 70, 190))]
        for x0, x1, frac, color in bars:
            y0 = int(h * (0.92 - 0.72 * frac))
            img[y0 : int(h * 0.92), int(w * x0) : int(w * x1)] = color
    elif variant == 2:
        # Landscape diagonal bands with contrasting patches and circles.
        wave = 0.5 + 0.5 * np.sin(6.283185 * (xn * 1.5 + yn * 0.75))
        img[..., 0] = 40.0 + 180.0 * wave
        img[..., 1] = 60.0 + 150.0 * xn
        img[..., 2] = 50.0 + 170.0 * (1.0 - yn)
        img[h // 5 : h * 2 // 5, w // 6 : w * 5 // 6 : 8] = (245.0, 245.0, 245.0)
        mask = (xn - 0.3) ** 2 + (yn - 0.7) ** 2 <= 0.025
        img[mask] = (20.0, 30.0, 40.0)
    else:
        # Document/terminal layout with dark header, sidebar, and crisp text-like rows.
        img[:] = (242.0, 244.0, 248.0)
        img[: h // 8, :] = (24.0, 28.0, 36.0)
        img[h // 8 :, : w // 5] = (220.0, 226.0, 236.0)
        for row in range(h // 6, h - 24, 22):
            span = int(w * (0.35 + 0.45 * ((row * 37) % 100) / 100.0))
            img[row : row + 8, w // 4 : min(w - 16, w // 4 + span)] = (28.0, 34.0, 48.0)
    return np.ascontiguousarray(np.clip(np.rint(img), 0, 255).astype(np.uint8))


def generate_fixtures(snapshot: Path, out_dir: Path, device: str = "cuda") -> None:
    """Generate deterministic single-image and batched stage fixtures for ``hy4_full_tower_matches_the_serving_oracle_per_stage``."""
    weights = load_weights(snapshot, device=device)
    vision_dir = out_dir / "vision"
    vision_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("square_256", (16, 16), 0),
        ("square_512", (32, 32), 1),
        ("landscape_320x480", (20, 30), 2),
        ("portrait_384x256", (24, 16), 3),
    ]
    batch_pixels = []
    batch_grids = []
    batch_order = []
    batch_rows = []
    with torch.no_grad():
        for name, (gh, gw), variant in specs:
            rgb = _structured_rgb(gh, gw, variant)
            patches = _rgb_to_raster_patches(rgb, gh, gw)
            grid = [1, gh, gw]
            stages = encode(weights, patches, grid, mode="serving")
            fdir = vision_dir / name
            fdir.mkdir(parents=True, exist_ok=True)
            (fdir / "rgb.u8").write_bytes(rgb.tobytes())
            (fdir / "meta.json").write_text(json.dumps({"grid": [gh, gw], "rows": output_rows(grid)}))
            _write_f32(fdir / "stage_0.f32", stages.patch)
            _write_f32(fdir / "stage_pos.f32", stages.embed)
            for idx in (0, 1, 19, 38, 39):
                _write_f32(fdir / f"stage_{idx + 1}.f32", stages.layers[idx])
            _write_f32(fdir / f"stage_{LAYERS + 1}.f32", stages.merger_norm)
            _write_f32(fdir / "stage_out.f32", stages.output)
            batch_pixels.append(patches)
            batch_grids.append(grid)
            batch_order.append(name)
            batch_rows.append(output_rows(grid))
            print(f"fixture {name}: grid=({gh},{gw}) out={tuple(stages.output.shape)}")

        batched = encode_batch(weights, batch_pixels, batch_grids, mode="serving")
        bdir = vision_dir / "batch"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "meta.json").write_text(json.dumps({"order": batch_order, "rows": batch_rows}))
        _write_f32(bdir / "stage_out.f32", torch.cat([s.output for s in batched], dim=0))
        print(f"batch fixture: order={batch_order} rows={batch_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    generate_fixtures(args.snapshot, args.out, device=args.device)


if __name__ == "__main__":
    main()
