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
    return data


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


def header() -> str:
    lines = ["#pragma once", "#include <stdint.h>",
             "/* Generated from scripts/build/mxfp4-layouts.json. */",
             "static inline uint32_t cuteafd_mxfp4_stored_width(uint32_t hidden, uint32_t width) {",
             "  if (!hidden || hidden % 256 || !width || width % 32 || width > 8192) return 0;"]
    for row in layouts()["storage_holds"]:
        lines.append(f"  if (hidden == {row['hidden']} && width == {row['width']}) return {row['stored']};")
    return "\n".join([*lines, "  return width;", "}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    args = parser.parse_args()
    args.header.write_text(header())
