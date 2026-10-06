#!/usr/bin/env python3
"""Generate 24 own G1 fixtures and hashes from pinned transformers PIL methods.

No torch import or GPU is needed. AST extraction executes the pinned smart_resize,
GLM resize, patchify, rescale, normalize, and RGB conversion functions unchanged;
the small base adapter supplies Pillow resize without transformers' torch imports.
Fixtures and the manifest belong outside the checkout. Run the loader example
`cargo run -p cuteafd-loader --example media-preprocess -- MANIFEST` afterwards.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Collection
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np
import PIL
from PIL import Image, ImageOps

PIN = "62d7ebd7de4938e072b7aaeb881593b79dc56835"


class ChannelDimension(Enum):
    FIRST = "channels_first"
    LAST = "channels_last"


class SizeDict(dict):
    def __getattr__(self, key):
        return self[key]


class PilBase:
    def resize(self, image, size, resample):
        pil = Image.fromarray(image.transpose(1, 2, 0))
        return np.asarray(pil.resize((size.width, size.height), resample)).transpose(2, 0, 1)


def extract(path, functions=(), methods=()):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    if methods:
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        selected = [node for cls in classes for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert len(selected) == len(methods)
        nodes.append(ast.ClassDef(name="Reference", bases=[ast.Name(id="PilBase", ctx=ast.Load())], keywords=[], body=selected, decorator_list=[]))
    assert len(nodes) == len(functions) + bool(methods)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    env = dict(np=np, math=math, PIL=PIL, Collection=Collection, ChannelDimension=ChannelDimension,
               SizeDict=SizeDict, PilBase=PilBase, requires_backends=lambda *_: None,
               get_channel_dimension_axis=lambda image, input_data_format: 0 if input_data_format == ChannelDimension.FIRST else 2)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env


def pattern(width, height, channels=3):
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack([((x * (17 + c * 3) + y * (29 + c * 7) + c * 71) % 256).astype(np.uint8) for c in range(channels)], axis=-1)


def png16(path, values):
    height, width, channels = values.shape
    color = {3: 2, 4: 6}[channels]
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\0" + row.astype(">u2").tobytes() for row in values)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, color, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def fixtures(root):
    names = []
    def save(name, image, **kwargs):
        image.save(root / name, **kwargs)
        names.append(name)
    rgb = Image.fromarray(pattern(193, 127))
    save("jpeg420.jpg", rgb, quality=91, subsampling=2)
    save("jpeg444.jpg", rgb, quality=91, subsampling=0)
    save("progressive.jpg", rgb, quality=91, progressive=True)
    save("cmyk.jpg", rgb.convert("CMYK"), quality=91)
    save("rgb.png", rgb)
    rgba = Image.fromarray(pattern(193, 127, 4))
    save("rgba.png", rgba)
    gray = rgb.convert("L")
    save("gray.png", gray)
    la = Image.merge("LA", (gray, rgba.getchannel("A")))
    save("gray-alpha.png", la)
    palette = rgb.quantize(colors=64)
    save("palette.png", palette)
    save("palette-alpha.png", palette, transparency=bytes(range(64)))
    save("gray16.png", Image.fromarray(np.arange(193 * 127, dtype=np.uint16).reshape(127, 193)))
    for channels in (3, 4):
        name = f"{'rgb' if channels == 3 else 'rgba'}16.png"
        png16(root / name, pattern(193, 127, channels).astype(np.uint16) * 257 + 17)
        names.append(name)
    save("mono.png", gray.convert("1"))
    for orientation in range(2, 9):
        exif = Image.Exif()
        exif[274] = orientation
        save(f"exif{orientation}.jpg", rgb, quality=91, subsampling=0, exif=exif)
    for width, height in ((1, 200), (8, 8), (6000, 4000)):
        save(f"size-{width}x{height}.png", Image.fromarray(pattern(width, height)))
    assert len(names) == 24
    return names


def glm_processor_config(snapshot=None):
    config = dict(patch_size=14, merge_size=2, temporal_patch_size=2,
        patch_expand_factor=1, min_image_tokens=16, max_image_tokens=8000,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711], do_rescale=True)
    if snapshot is not None:
        value = json.loads((snapshot / "processor_config.json").read_text()).get("image_processor")
        if not isinstance(value, dict):
            raise ValueError("GLM processor_config.json requires image_processor object")
        config.update(value)
    if (config["patch_size"] != 14 or config["merge_size"] != 2
            or config["temporal_patch_size"] != 2 or config["patch_expand_factor"] != 1
            or not config["do_rescale"] or config.get("resample", 3) != 3):
        raise ValueError("unsupported GLM G1 processor geometry or rescale/resample")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--glm-snapshot", type=Path, help="use the actual nested HF GLM image processor config")
    args = parser.parse_args()
    glm_config = glm_processor_config(args.glm_snapshot)
    revision = subprocess.check_output(["git", "-C", str(args.transformers), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise SystemExit(f"reference pin differs: expected {PIN}, got {revision}")
    source = args.transformers / "src/transformers"
    transforms = extract(source / "image_transforms.py", functions=("rescale", "normalize", "convert_to_rgb"))
    qwen = extract(source / "models/qwen2_vl/image_processing_pil_qwen2_vl.py", functions=("smart_resize",), methods=("patchify",))
    glm = extract(source / "models/glm5_next/image_processing_pil_glm5_next.py", functions=("smart_resize",), methods=("resize", "patchify"))
    args.output.mkdir(parents=True, exist_ok=True)
    names = fixtures(args.output)
    cases = []
    for name in names:
        image = transforms["convert_to_rgb"](ImageOps.exif_transpose(Image.open(args.output / name)))
        for family in ("mimo", "qwen", "glm_flash"):
            for low in (False, True):
                cap = 256 if low else 4096
                patch = 14 if family == "glm_flash" else 16
                chw = np.asarray(image).transpose(2, 0, 1)
                if family == "glm_flash":
                    reference = glm["Reference"]()
                    resized = reference.resize(chw, Image.Resampling.BICUBIC,
                        glm_config["patch_size"] * glm_config["merge_size"],
                        glm_config["temporal_patch_size"], glm_config["min_image_tokens"],
                        min(glm_config["max_image_tokens"], cap))
                else:
                    reference = qwen["Reference"]()
                    minimum = 3136 if family == "mimo" else 65536
                    maximum = 12845056 if family == "mimo" else 16777216
                    height, width = qwen["smart_resize"](image.height, image.width, factor=32, min_pixels=minimum, max_pixels=min(maximum, cap * 1024))
                    resized = PilBase().resize(chw, SizeDict(height=height, width=width), Image.Resampling.BICUBIC)
                mean, std = ([0.5] * 3, [0.5] * 3) if family == "qwen" else ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                if family == "glm_flash":
                    mean, std = glm_config["image_mean"], glm_config["image_std"]
                normalized = transforms["normalize"](transforms["rescale"](resized, 1 / 255), mean, std, input_data_format=ChannelDimension.FIRST)
                patches, gh, gw = reference.patchify(normalized, patch, 2, 2)
                cases.append(dict(family=family, file=name, low=low, grid=dict(t=1, h=gh, w=gw),
                                  rgb_sha256=hashlib.sha256(resized.transpose(1, 2, 0).tobytes()).hexdigest(),
                                  patch_sha256=hashlib.sha256(patches.astype("<f4").tobytes()).hexdigest()))
    manifest = dict(transformers=revision, pillow=PIL.__version__, numpy=np.__version__, decode_policy=dict(exif_transpose=True, alpha="drop"), cases=cases)
    if args.glm_snapshot:
        manifest["glm_processor"] = {"snapshot": str(args.glm_snapshot.resolve()),
            "snapshot_revision": args.glm_snapshot.name,
            "processor_sha256": hashlib.sha256((args.glm_snapshot / "processor_config.json").read_bytes()).hexdigest(),
            "parameters": glm_config}
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(names)} fixtures, {len(cases)} reference cases: {path}")


if __name__ == "__main__":
    main()
