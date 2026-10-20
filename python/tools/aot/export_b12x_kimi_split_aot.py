#!/usr/bin/env python3
"""Opt-in Kimi K6 projection seams; does not replace fused BF16 exports."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def programs():
    from b12x.integration.cuteafd._kda_contract import KDAGeometry
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry
    from b12x.integration.cuteafd.kimi_split import (
        compile_kimi_kda_core_aot, compile_kimi_mla_rank_norm_aot,
        compile_kimi_mla_query_aot, compile_kimi_mla_gate_output_aot)
    result = []
    for heads in (48, 96):
        kg = KDAGeometry(hidden=7168, kda_heads=heads)
        mg = KimiMlaGeometry(heads=heads)
        for rows in (64, 4096):
            params = dict(max_rows=rows, decode=rows == 64)
            result.append((f"kda_core_h{heads}_m{rows}",
                           partial(compile_kimi_kda_core_aot, kg, **params), rows))
            for stride in (576, 640):
                result.append((f"mla_rank_norm_h{heads}_kv{stride}_m{rows}",
                    partial(compile_kimi_mla_rank_norm_aot, mg, max_rows=rows, kv_stride=stride), rows))
            for name, compiler in (("mla_query", compile_kimi_mla_query_aot),
                                   ("mla_gate_output", compile_kimi_mla_gate_output_aot)):
                result.append((f"{name}_h{heads}_m{rows}", partial(compiler, mg, max_rows=rows), rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated export names")
    args = parser.parse_args()
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.only.split(",")) if args.only else None
    table = programs()
    if selected and selected - {name for name, _, _ in table}:
        parser.error("unknown split program")
    includes, entries, records = [], [], []
    for name, compile_, capacity in table:
        if selected is not None and name not in selected:
            continue
        with exportable_compilation():
            p = compile_()
        prefix = "cuteafd_" + name
        p.export_to_c(str(output), name, prefix)
        checked = validate_exported_header(p, output / f"{name}.h", prefix)
        abi = p.abi
        kinds = "".join({"int32": "i", "int64": "l", "float32": "f"}[kind] for _, kind in abi["scalars"])
        if checked["argument_count"] != len(abi["pointers"]) + len(kinds) + 1:
            raise ValueError(f"{name}: header argument count differs")
        includes.append(f'#include "{name}.h"')
        entries.append(f'{{"{name}", _mlir_{prefix}_cuda_init, _mlir_{prefix}_cuda_load_to_device, {checked["symbol"]}, {len(abi["pointers"])}, "{kinds}", 0}}')
        records.append(dict(name=name, family="kimi_split", geometry=dict(p.geometry),
            pointers=[dict(zip(("name", "dtype", "shape", "role"), op)) for op in abi["pointers"]],
            scalars=[dict(zip(("name", "type"), op)) for op in abi["scalars"]],
            capacity_rows=capacity, scratch_bytes_at_capacity=p.scratch_bytes(capacity),
            entry=checked["symbol"], object_sha256=hashlib.sha256((output / f"{name}.o").read_bytes()).hexdigest()))
        print(f"exported {name}", flush=True)
    (output / "kimi_split_programs.json").write_text(json.dumps(dict(
        abi_version=1, sparkinfer_revision=_pinned_sparkinfer.REVISION,
        source_tree_sha256=_pinned_sparkinfer.LOCK_DATA["source_tree_sha256"], programs=records), indent=2) + "\n")
    (output / "kimi_split_programs.h").write_text("\n".join([
        "#pragma once", *includes, "#define CUTEAFD_KIMI_SPLIT_PROGRAMS " + ", ".join(entries), ""]))


if __name__ == "__main__":
    main()
