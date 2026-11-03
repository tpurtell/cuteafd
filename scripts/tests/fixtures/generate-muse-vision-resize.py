#!/usr/bin/env python3
"""Generate immutable Muse image-grid outputs from the official smart_resize.

Run with --source pointing at official image_processing_muse_glimmer.py.
Only that function's AST is executed; no imports from the downloaded tree run.
"""
import argparse
import ast
import hashlib
import itertools
import math
from pathlib import Path
import struct
import zlib

SIDE = 1792
PIXELS = 64 << 20


def extreme_inputs():
    # Every long side through one full-image cap; large decode extremes and
    # immediate neighbours of power-of-two and token-cap transitions, both axes.
    sides = set(range(1, 28 * 4096 + 2))
    for center in [PIXELS, *[1 << n for n in range(17, 27)]]:
        sides.update(range(max(1, center - 2), min(PIXELS, center + 2) + 1))
    for side in sorted(sides):
        yield (1, side)
        yield (side, 1)
    for short in [2, 3, 7, 14, 27, 28, 29, 56, 112, 1792]:
        maximum = PIXELS // short
        for center in [maximum, max(1, 28 * 4096 // short), 1792]:
            for long in range(max(1, center - 2), min(maximum, center + 2) + 1):
                yield short, long
                yield long, short


def official_function(source):
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "smart_resize")
    namespace = dict(math=math, itertools=itertools)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["smart_resize"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resize = official_function(args.source)
    output = bytearray()
    for h in range(1, SIDE + 1):
        for w in range(1, SIDE + 1):
            rh, rw = resize(h, w, 28, 4096)
            output.extend(struct.pack("<BB", rh // 28, rw // 28))
    for h, w in extreme_inputs():
        output.extend(struct.pack("<II", *resize(h, w, 28, 4096)))
    args.output.write_bytes(zlib.compress(output, level=9))
    print(f"{len(output)} official output bytes; sha256={hashlib.sha256(output).hexdigest()}")


if __name__ == "__main__":
    main()
