"""Native MXFP4 resident geometry and hardware-qualified slice recipes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYOUTS_PATH = Path(__file__).resolve().parents[3] / "scripts/build/mxfp4-layouts.json"


def layouts() -> dict:
    data = json.loads(LAYOUTS_PATH.read_text())
    if data["schema"] != 1:
        raise ValueError("unsupported MXFP4 layout schema")
    validate(data)
    return data


def validate(data: dict) -> None:
    keys = set()
    for row in data["native_slices"]:
        key = (row["hidden"], row["width"], row["sm"])
        prefill = row.get("prefill", [])
        if key in keys or len(set(prefill)) != len(prefill) or any(c not in (256, 1024, 4096) for c in prefill):
            raise ValueError("invalid native MXFP4 prefill recipe")
        keys.add(key)
        if not row["capacities"] and not prefill:
            raise ValueError("empty native MXFP4 recipe")
        recipes = dict(zip(row["capacities"], row["slices"], strict=True))
        if len(recipes) != len(row["capacities"]):
            raise ValueError("native MXFP4 capacities must pair uniquely with slices")
        if recipes and len({width for capacity, width in recipes.items() if capacity <= 80}) != 1:
            raise ValueError("native MXFP4 decode capacities must share one slice recipe")
        if any(width not in (32, 64, 128, 192) for width in recipes.values()):
            raise ValueError("invalid native MXFP4 slice width")
        splits = row.get("fc1_k_splits", [1] * len(recipes))
        split_recipes = dict(zip(row["capacities"], splits, strict=True))
        if any(split not in (1, 2, 4, 8) for split in splits):
            raise ValueError("invalid native MXFP4 FC1 K split")
        if split_recipes and len({split for capacity, split in split_recipes.items() if capacity <= 80}) != 1:
            raise ValueError("native MXFP4 decode capacities must share one FC1 K split")
        if any(split != 1 for capacity, split in split_recipes.items() if capacity > 80):
            raise ValueError("native MXFP4 prefill FC1 K splitting is unqualified")

    for row in data["ownership"]:
        if (row["unit"] not in (32, 64, 128) or row["intermediate"] % row["unit"]
                or not 1 <= row["world"] <= 8 or row["intermediate"] // row["unit"] < row["world"]):
            raise ValueError("invalid measured MXFP4 ownership")


def ownership_unit(intermediate: int, world: int) -> int:
    return next((row["unit"] for row in layouts()["ownership"]
                 if (row["intermediate"], row["world"]) == (intermediate, world)), 32)


def generic_stored_width(hidden: int, width: int) -> int:
    return next((row["stored"] for row in layouts()["generic_storage_holds"]
                 if (row["hidden"], row["width"]) == (hidden, width)), width)


def stored_width(hidden: int, width: int) -> int:
    if hidden <= 0 or hidden % 256 or width <= 0 or width % 32 or width > 8192:
        raise ValueError("invalid native MXFP4 geometry")
    return next((row["stored"] for row in layouts()["storage_holds"]
                 if (row["hidden"], row["width"]) == (hidden, width)), width)


def native_slice_width(hidden: int, width: int, sm: int, capacity: int,
                       matching_width: int) -> int:
    for row in layouts()["native_slices"]:
        if (row["hidden"], row["width"], row["sm"]) == (hidden, width, sm):
            return dict(zip(row["capacities"], row["slices"], strict=True)).get(
                capacity, matching_width)
    return matching_width


def native_prefill(hidden: int, width: int, sm: int, capacity: int) -> bool:
    """Qualified native STREAM recipe, independent of role names and live rows."""
    return any((row["hidden"], row["width"], row["sm"]) == (hidden, width, sm)
               and capacity in row.get("prefill", []) for row in layouts()["native_slices"])


def native_prefill_scratch(capacity: int, hidden: int, width: int,
                           experts: int, topk: int) -> tuple[list[dict], int]:
    """Pure counterpart of the pinned STREAM's seven caller-owned planes."""
    if capacity not in (256, 1024, 4096) or hidden <= 0 or hidden % 512 or width <= 0 or width % 32 or experts <= 0 or topk <= 0:
        raise ValueError("invalid native V4 prefill geometry")
    pairs = capacity * topk
    tiles = (pairs + 127) // 128 + min(experts, pairs)
    row_bytes = (width + width // 32 + 15) // 16 * 16
    specs = [("meta", "i32", (4 + 3 * experts + tiles,)),
             ("pair_row", "i32", (pairs,)), ("pair_pos", "i32", (pairs,)),
             ("act", "u8", (tiles * 128 * row_bytes,)),
             ("grouped_routing", "f32", (pairs,)), ("grouped_route", "i32", (pairs,)),
             ("output", "f32", (pairs, hidden))]
    offset, planes = 0, []
    for name, dtype, shape in specs:
        size = 1 if dtype == "u8" else 4
        for extent in shape:
            size *= extent
        offset = (offset + 15) // 16 * 16
        planes.append(dict(name=name, dtype=dtype, shape=shape, offset=offset, nbytes=size))
        offset += size
    return planes, offset


def native_fc1_k_splits(hidden: int, width: int, sm: int, capacity: int) -> int:
    for row in layouts()["native_slices"]:
        if (row["hidden"], row["width"], row["sm"]) == (hidden, width, sm):
            splits = row.get("fc1_k_splits", [1] * len(row["capacities"]))
            return dict(zip(row["capacities"], splits, strict=True)).get(capacity, 1)
    return 1


def header() -> str:
    lines = ["#pragma once", "#include <stdint.h>",
             "/* Generated from scripts/build/mxfp4-layouts.json. */",
             "static inline uint32_t cuteafd_mxfp4_stored_width(uint32_t hidden, uint32_t width) {",
             "  if (!hidden || hidden % 256 || !width || width % 32 || width > 8192) return 0;"]
    for row in layouts()["storage_holds"]:
        lines.append(f"  if (hidden == {row['hidden']} && width == {row['width']}) return {row['stored']};")
    lines += ["  return width;", "}",
              "static inline uint32_t cuteafd_mxfp4_ownership_unit(uint32_t intermediate, uint32_t world) {"]
    for row in layouts()["ownership"]:
        lines.append(f"  if (intermediate == {row['intermediate']} && world == {row['world']}) return {row['unit']};")
    return "\n".join([*lines, "  return 32;", "}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    args = parser.parse_args()
    args.header.write_text(header())
