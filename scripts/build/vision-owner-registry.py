#!/usr/bin/env python3
"""Bind generated vision AOT headers to geometry-keyed resident owners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def program(directory, name):
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
        raise ValueError("invalid vision program name")
    symbol = "cuteafd_" + name
    text = (directory / (name + ".h")).read_text()
    wrapper = re.search(r"static inline int32_t cute_dsl_" + symbol + r"_wrapper\(([^)]*)\)", text)
    entry = re.search(r"void (_mlir_" + symbol + r"\w+)\(void \*\*args, int32_t num_args\);", text)
    if wrapper is None or entry is None:
        raise ValueError(f"missing vision program ABI: {name}")
    arguments = [arg.strip() for arg in wrapper[1].split(",")]
    if arguments[0] != symbol + "_Kernel_Module_t *module" or arguments[-1].split()[0] != "cudaStream_t":
        raise ValueError(f"invalid vision module/stream ABI: {name}")
    pointers, kinds = 0, ""
    for argument in arguments[1:-1]:
        if re.fullmatch(r"void\s*\*\w+", argument) and not kinds:
            pointers += 1
        else:
            kind = {"int32_t": "i", "int64_t": "l", "float": "f"}.get(argument.split()[0])
            if kind is None:
                raise ValueError(f"unsupported vision argument: {argument}")
            kinds += kind
    if pointers + len(kinds) + 2 > 32:
        raise ValueError("vision program exceeds launch argument capacity")
    return '{"%s", _mlir_%s_cuda_init, _mlir_%s_cuda_load_to_device, %s, %d, "%s"}' % (
        name, symbol, symbol, entry[1], pointers, kinds)


def registry(entries):
    includes, rows, keys = [], [], set()
    for entry in entries:
        dim, windowed, pointwise = entry["head_dim"], entry["windowed"], entry["pointwise_group"]
        key = (dim, windowed, pointwise)
        if dim <= 0 or type(windowed) is not bool or key in keys:
            raise ValueError("invalid or duplicate vision owner geometry")
        keys.add(key)
        directory = Path(entry["directory"])
        specs = [(Path(p.get("directory", directory)), p["name"]) if isinstance(p, dict) else (directory, p)
                 for p in entry["programs"]]
        names = [name for _, name in specs]
        if not names or len(set(names)) != len(names):
            raise ValueError("vision owner needs unique programs, attention first")
        programs = [program(path, name) for path, name in specs]
        # Attention has the legacy 8-pointer or shared 6-pointer varlen ABI.
        if not (programs[0].endswith(', 8, "f"}') or programs[0].endswith(', 6, "f"}')):
            raise ValueError("vision entry's first program must be varlen attention")
        includes.extend('#include "%s/%s.h"' % (path.as_posix(), name) for path, name in specs)
        rows.append('{%d, %s, %s, {%s}}' % (dim, str(windowed).lower(), json.dumps(pointwise), ", ".join(programs)))
    return "#pragma once\n" + "\n".join(dict.fromkeys(includes)) + "\n#define CUTEAFD_VISION_OWNER_ENTRIES " + ", ".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(registry(json.loads(args.entries.read_text())))


if __name__ == "__main__":
    main()
