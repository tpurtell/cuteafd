#!/usr/bin/env python3
"""Compare CPU-only SM120 AOT exports, including their embedded CUDA images.

Run in the coordinator compiler image without --gpus. Differences are evidence
for review, not proof of incompatibility; --require-identical is an optional
strict gate. Reports include exact object bytes, cubins, headers and metadata.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import time

ROOT = Path(os.environ.get("CUTEAFD_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
GEOMETRIES = ("flash", "pro", "flash2", "pro2", "glm", "glm2", "glmf", "glmf2",
              "mimo", "mimo2", "mimop", "mimop2", "mimof", "mimof2", "qwen4")
CAPACITIES = "1,16,80,256,1024,4096"
ROUTED_GEOMETRIES = ("mimo", "mimop", "mimof", "glm", "glmf", "qwen4",
                     "glm_nvfp4", "glmf_nvfp4", "qwen4_nvfp4", "glmfdense_nvfp4",
                     "glm_nvfp4a4", "glmf_nvfp4a4", "qwen4_nvfp4a4", "glmfdense_nvfp4a4")
EXL3_GEOMETRIES = ("v41", "dsv4f", "dsv4p", "glm", "glmf", "qwen4")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jobs(scope):
    result = []
    # Small-row top-k deliberately exercises the fused persistent route, which
    # the production m64 table does not select.
    for geometry in ("flash", "pro", "glm", "glmf"):
        for rows in (1, 16):
            result.append((f"suspect-{geometry}-m{rows}", "export_b12x_dsv4_aot.py",
                           ["--geometry", geometry, "--decode-rows", str(rows),
                            "--only", f"index_topk_decode_m{rows}"]))
        mla = [f"sparse_mla_{mode}{kind}_m{rows}"
               for mode, rows in (("decode", 64), ("prefill", 4096))
               for kind in (("_win", "_c4", "_c128") if geometry in ("flash", "pro") else ("",))]
        result.append((f"suspect-{geometry}-mla", "export_b12x_dsv4_aot.py",
                       ["--geometry", geometry, "--only", ",".join(mla)]))
    if scope == "suspects":
        return result
    result.extend((geometry, "export_b12x_dsv4_aot.py", ["--geometry", geometry])
                  for geometry in GEOMETRIES)
    for family in ("dsv4f", "dsv4p"):
        for role in ("rtx_backbone", "rtx_tp2"):
            name = f"{family}-slices" + ("-rtx_tp2" if role == "rtx_tp2" else "")
            result.append((name, "export_b12x_slices_aot.py",
                           ["--geometry", family, "--role", role, "--rows", CAPACITIES,
                            "--width", "1:64,16:128,80:128,256:128,1024:128,4096:128",
                            "--atomic-min-capacity", "256", "--standard-names"]))
    result.extend([
        ("v41-fp8", "export_b12x_v41_fp8_aot.py", ["--rows", CAPACITIES]),
        ("v41-experts", "export_b12x_v41_experts_aot.py",
         ["--role", "coordinator", "--input-format", "bf16", "--rows", "1,2,40,16,80,256,1024,4096"]),
        ("v41-index", "export_b12x_v41_index_aot.py", []),
        ("v41-router", "export_b12x_v41_router_aot.py", []),
        ("v41-attention", "export_b12x_v41_attention_aot.py", []),
        ("v41-attention-tp2", "export_b12x_v41_attention_aot.py", ["--heads", "32"]),
    ])
    for role in ("rtx_backbone", "rtx_tp2", "dspark_tp2"):
        args = ["--role", role, "--rows", CAPACITIES, "--width", "192", "--standard-names"]
        if role != "dspark_tp2":
            args += ["--atomic-min-capacity", "256"]
        result.append((f"v41-slices-{role}", "export_b12x_slices_aot.py", args))
    for role in ("rtx_backbone", "rtx_tp2", "dspark_tp2"):
        result.append((f"v41-nvfp4-{role}", "export_b12x_v41_nvfp4_aot.py",
                       ["--role", role, "--rows", CAPACITIES, "--tile-m", "16", "--output-shards", "0"]))
    result.extend((f"routed-{geometry}", "package_fp8_moe_aot.py",
                   ["--geometry", geometry, "--capacities", CAPACITIES])
                  for geometry in ROUTED_GEOMETRIES)
    result.extend((f"exl3-{geometry}", "package_exl3_aot.py",
                   ["--geometry", geometry, "--capacities", CAPACITIES])
                  for geometry in EXL3_GEOMETRIES)
    return result


def export_routed(extra, output):
    """Use the package's exact compiler/form policy without linking its DSOs."""
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.fp8_moe import (
        GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes, prefill_forms,
    )
    import package_fp8_moe_aot as package
    geometry = extra[extra.index("--geometry") + 1]
    capacities = sorted({int(item) for item in extra[extra.index("--capacities") + 1].split(",")})
    _, g = package.layout_geometry(GEOMETRIES[geometry], "tp1")
    forms = [(rows, "auto") for rows in capacities]
    forms += [(rows, form) for rows in capacities for form in prefill_forms(g, rows, False)]
    records = []
    for rows, form in forms:
        stem = f"fp8moe_{geometry}_tp1_m{rows}" + ("" if form == "auto" else "_" + form)
        with exportable_compilation():
            program = compile_fp8_moe_aot(g, route="auto", max_rows=rows, wire=False, prefill=form)
        program.export_to_c(str(output), stem, "cuteafd_" + stem)
        checked = validate_exported_header(program, output / f"{stem}.h", "cuteafd_" + stem)
        if checked["argument_count"] != package.POINTERS + 2:
            raise ValueError(f"{stem}: unexpected routed expert ABI")
        records.append({"name": stem, "capacity": rows, "form": form, "abi": checked,
                        "scratch_bytes": fp8_moe_scratch_bytes(g, "auto", rows, False, form)})
        print(f"exported {stem}", flush=True)
    (output / "routed.json").write_text(json.dumps({"programs": records}, indent=2) + "\n")


def export_exl3(extra, output):
    """Follow production coordinator profiles and tuning, but allocate no CUDA buffers."""
    import package_exl3_aot as package
    from export_b12x_exl3_aot import export
    import torch
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis
    original_lut = mixed_trellis._trellis256_execution_lut
    # The LUT is a runtime buffer created after compilation, not a compiler input.
    # Keep its exact values on CPU; the offline worker still forbids CUDA allocation.
    mixed_trellis._trellis256_execution_lut = lambda device, codebook: original_lut(torch.device("cpu"), codebook)
    geometry = extra[extra.index("--geometry") + 1]
    capacities = sorted({int(item) for item in extra[extra.index("--capacities") + 1].split(",")})
    for profile, width, experts, topk, dtype, _ in package.profiles_for_role("coordinator", geometry):
        for rows in capacities:
            options = {"hidden": package.GEOMETRIES[geometry][0], "compile_only": True}
            block = package.route_block(geometry, rows)
            if package.warp_specialized(geometry, "coordinator", width, rows):
                block = 64
                options["warp_specialized"] = True
                if package.wire_input(geometry, "coordinator", width, rows):
                    options["wire_input"] = True
                stages = package.ws_input_stages(geometry, "coordinator", width, rows)
                if stages is not None:
                    options["ws_input_stages"] = stages
                if package.ws_dynamic_tiles(geometry, "coordinator", width, rows):
                    options["ws_dynamic_tiles"] = True
                tile = package.ws_tile(geometry, "coordinator", width, rows)
                if tile is not None:
                    options["tile"] = tile
            elif package.fused_input_rotation(geometry, "coordinator", width, rows):
                options["fused_input_rotation"] = True
            elif package.token_major_rotation(geometry, rows):
                options["token_major_rotation"] = True
            options.update(route_block=block, swiglu_limit=package.swiglu_limit(geometry))
            export(output / profile / f"m{rows}", width, experts, rows, (3, 4), "auto", topk, dtype, **options)


def worker(payload):
    output = Path(payload["output"])
    os.environ["TRITON_CACHE_DIR"] = str(output.parent / f".triton-cache-{payload['sms']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
    os.environ["CUTEAFD_EXPORT_NARROW_AOT"] = "1"
    os.environ["CUTEAFD_EXPORT_HC_LAGGED"] = "1"
    sys.path[:0] = [str(ROOT / "python/tools/lib"), str(ROOT / "python/tools/aot")]
    import _pinned_sparkinfer
    from b12x._lib.compile_pool import _initialize_worker
    _initialize_worker(0, (12, 0), "sm-export-comparison", "sm-export-comparison",
                       payload["sms"], 101376, 102400, multiprocessing.Array("i", [0, 0]))
    import torch
    import importlib.metadata
    toolchain = {name: importlib.metadata.version(name) for name in ("torch", "triton", "nvidia-cutlass-dsl")}
    from b12x.integration.cuteafd import AotProgram
    plans = {}
    original_export = AotProgram.export_to_c

    def export_with_plan(program, file_path, file_name, function_prefix):
        plans[file_name] = {"geometry": dict(program.geometry), "abi": program.abi}
        return original_export(program, file_path, file_name, function_prefix)

    AotProgram.export_to_c = export_with_plan
    start = time.monotonic()
    path = ROOT / "python/tools/aot" / payload["exporter"]
    if payload["exporter"] == "package_fp8_moe_aot.py":
        export_routed(payload["args"], output)
    elif payload["exporter"] == "package_exl3_aot.py":
        export_exl3(payload["args"], output)
    else:
        sys.argv = [str(path), "--output-dir", str(output), *payload["args"]]
        runpy.run_path(str(path), run_name="__main__")
    if torch.cuda.is_initialized():
        raise RuntimeError("SM comparison initialized CUDA")
    (Path(payload["output"]) / "comparison-plans.json").write_text(json.dumps(plans, indent=2) + "\n")
    summary = {"sms": payload["sms"], "capability": [12, 0], "cuda_initialized": False,
               "sparkinfer_revision": _pinned_sparkinfer.REVISION,
               "source_tree_sha256": _pinned_sparkinfer.LOCK_DATA["source_tree_sha256"],
               "exporter": payload["exporter"], "args": payload["args"], "toolchain": toolchain,
               "environment": {key: os.environ[key] for key in ("CUTEAFD_EXPORT_NARROW_AOT", "CUTEAFD_EXPORT_HC_LAGGED")},
               "exporter_sha256": sha(path.read_bytes()), "comparison_script_sha256": sha(Path(__file__).read_bytes()),
               "export_sources_sha256": {str(source.relative_to(ROOT)): sha(source.read_bytes())
                   for source in sorted((ROOT / "python/tools/aot").glob("*.py"))},
               "max_shared_memory_per_block": 101376, "max_shared_memory_per_multiprocessor": 102400,
               "seconds": time.monotonic() - start}
    (Path(payload["output"]) / "comparison-context.json").write_text(json.dumps(summary, indent=2) + "\n")


def cuda_sections(blob):
    """Fingerprint device instructions separately from ELF names and launch info."""
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    shsize, shcount, string_index = struct.unpack_from("<HHH", blob, 58)
    if string_index >= shcount:
        raise ValueError("invalid CUDA ELF section string table")
    string_pos, string_bytes = struct.unpack_from("<QQ", blob, shoff + string_index * shsize + 24)
    strings = blob[string_pos:string_pos + string_bytes]
    result = []
    for index in range(shcount):
        entry = shoff + index * shsize
        name_offset, kind = struct.unpack_from("<II", blob, entry)
        pos, size = struct.unpack_from("<QQ", blob, entry + 24)
        if kind == 8 or not size:
            continue
        end = strings.find(b"\0", name_offset)
        name = strings[name_offset:end if end >= 0 else len(strings)].decode("utf-8", errors="replace")
        result.append({"section": name, "bytes": size, "sha256": sha(blob[pos:pos + size])})
    return result


def clean_metadata(value):
    if isinstance(value, dict):
        return {key: clean_metadata(item) for key, item in value.items()
                if key not in ("artifacts", "artifact_sha256", "files", "object_sha256", "sha256", "device", "physical_sms")}
    if isinstance(value, list):
        return [clean_metadata(item) for item in value]
    return value


def metadata(directory):
    records = {}
    for path in sorted(directory.rglob("*.json")):
        if path.name == "comparison-context.json":
            continue
        value = json.loads(path.read_text())
        prefix = path.parent.relative_to(directory).as_posix()
        prefix = "" if prefix == "." else prefix + "/"
        if path.name == "comparison-plans.json":
            for name, record in value.items():
                records[name] = {**records.get(name, {}), **clean_metadata(record)}
            continue
        for record in value.get("programs", value.get("variants", value.get("objects", []))):
            name = record.get("label", record.get("name", record.get("stage")))
            if name:
                key = prefix + name
                records[key] = {**records.get(key, {}), **clean_metadata(record)}
                if path.name == "v41_exl3.json":
                    records[key]["launch"] = clean_metadata({k: v for k, v in value.items() if k != "objects"})
        records[prefix + path.stem] = clean_metadata(value)
    return records


def differing_fields(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [field for key in sorted(left.keys() | right.keys())
                for field in differing_fields(left.get(key), right.get(key), prefix + "." + key)]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [field for i, (a, b) in enumerate(zip(left, right))
                for field in differing_fields(a, b, prefix + f"[{i}]")]
    return [] if left == right else [prefix.lstrip(".")]


def compare(left, right):
    inspector = load_file("cuteafd_cuda_images", ROOT / "scripts/qualify/deepseek_v41/inspect-cute-object.py")
    left_names = {path.relative_to(left).as_posix() for path in left.rglob("*") if path.suffix in (".o", ".cubin")}
    right_names = {path.relative_to(right).as_posix() for path in right.rglob("*") if path.suffix in (".o", ".cubin")}
    if not left_names or left_names != right_names:
        raise ValueError(f"object set mismatch: {left_names ^ right_names}, left={len(left_names)}, right={len(right_names)}")
    contexts = [json.loads((directory / "comparison-context.json").read_text()) for directory in (left, right)]
    if any(context.get("cuda_initialized") is not False for context in contexts):
        raise ValueError("comparison context did not prove CPU-only export")
    for context in contexts:
        sms = context.get("sms")
        if type(sms) is not int or sms <= 0 or context.get("capability") != [12, 0]:
            raise ValueError("comparison context must identify positive SM count and compute 12.0")
    if contexts[0]["sms"] == contexts[1]["sms"]:
        raise ValueError("comparison contexts must have two distinct SM counts")
    for key in ("capability", "sparkinfer_revision", "source_tree_sha256", "exporter", "args",
                "toolchain", "environment", "exporter_sha256", "comparison_script_sha256", "export_sources_sha256",
                "max_shared_memory_per_block", "max_shared_memory_per_multiprocessor"):
        if any(key not in context for context in contexts):
            raise ValueError(f"comparison context missing provenance: {key}")
        if contexts[0][key] != contexts[1][key]:
            raise ValueError(f"comparison target differs beyond SM count: {key}")
    metas = [metadata(directory) for directory in (left, right)]
    if contexts[0]["exporter"] == "export_b12x_dsv4_aot.py" and "--only" not in contexts[0]["args"]:
        for target in metas:
            for name, record in target.items():
                if record.get("geometry", {}).get("route") == "paged_fused":
                    raise ValueError(f"{name}: default serving table contains co-resident paged_fused "
                                     "without a live-kernel residency gate/fallback")
    records = []
    for name in sorted(left_names):
        blobs = [(directory / name).read_bytes() for directory in (left, right)]
        images = [list(inspector.cuda_images(blob)) for blob in blobs]
        if any(not item for item in images):
            raise ValueError(f"{name}: no CUDA ELF images; cannot classify device-code differences")
        cubin_hashes = [[sha(blob) for _, blob in item] for item in images]
        sections = [[cuda_sections(blob) for _, blob in item] for item in images]
        text_hashes = [sorted(section["sha256"] for image in target for section in image
                              if section["section"].startswith(".text.")) for target in sections]
        stem = str(Path(name).with_suffix(""))
        meta_keys = [key for key in metas[0].keys() | metas[1].keys()
                     if stem == key or stem.startswith(key + "_")]
        meta_keys = [max(meta_keys, key=len)] if meta_keys else []
        fields = sorted({field for key in meta_keys
                         for field in differing_fields(metas[0].get(key), metas[1].get(key), key)})
        headers = [path.read_bytes() if path.exists() else None
                   for directory in (left, right) for path in [(directory / name).with_suffix(".h")]]
        if name.endswith(".o") and any(header is None for header in headers):
            raise ValueError(f"{name}: missing exported ABI header")
        same_object = blobs[0] == blobs[1]
        same_cubins = sorted(blob for _, blob in images[0]) == sorted(blob for _, blob in images[1])
        records.append({"program": stem, "artifact_kind": Path(name).suffix[1:], "object_identical": same_object,
                        "cubins_identical": same_cubins, "header_identical": headers[0] == headers[1],
                        "object_sha256": [sha(blob) for blob in blobs], "object_bytes": list(map(len, blobs)),
                        "cubin_sha256": cubin_hashes, "cubin_sections": sections,
                        "sass_identical": text_hashes[0] == text_hashes[1] if all(text_hashes) else None,
                        "metadata_changed_fields": fields,
                        "classification": "identical" if same_object else "host-only" if same_cubins else "cubin-bytes"})
    header_names = {path.relative_to(directory).as_posix()
                    for directory in (left, right) for path in directory.rglob("*.h")}
    headers_changed = [name for name in sorted(header_names)
                       if not all((directory / name).exists() for directory in (left, right))
                       or (left / name).read_bytes() != (right / name).read_bytes()]
    return {"contexts": contexts, "programs": records,
            "manifest_changed_fields": differing_fields(*metas), "headers_changed": headers_changed}


def parse_sms(value):
    try:
        sms = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SM counts must be integers") from exc
    if len(sms) != 2 or min(sms) <= 0 or sms[0] == sms[1]:
        raise argparse.ArgumentTypeError("supply two distinct positive SM counts")
    return sms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", type=parse_sms, default=(188, 170))
    parser.add_argument("--scope", choices=("suspects", "all"), default="all")
    parser.add_argument("--groups", help="comma-separated job names; --list prints available names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / ".cache/cuteafd/builds/plat1-sm120/sm-exports")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--require-identical", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per export context")
    parser.add_argument("--workers", type=int, default=1, help="independent CPU export groups in parallel")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(json.loads(args._worker))
        return 0
    selected_jobs = jobs(args.scope)
    if args.list:
        print("\n".join(name for name, _, _ in selected_jobs))
        return 0
    if args.groups:
        selected = set(args.groups.split(","))
        unknown = selected - {name for name, _, _ in selected_jobs}
        if unknown:
            parser.error(f"unknown groups: {sorted(unknown)}")
        selected_jobs = [job for job in selected_jobs if job[0] in selected]
    output = args.output_dir.expanduser().resolve()
    load_file("cuteafd_build_filesystem", ROOT / "scripts/build/assert-build-filesystem.py").check_path(str(output))
    output.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.timeout < 1:
        parser.error("--workers and --timeout must be positive")
    reports = []

    def export_group(job):
        name, exporter, extra = job
        directories = [output / name / str(sms) for sms in args.sms]
        for sms, directory in zip(args.sms, directories):
            if args.compare_only:
                continue
            # Never silently reuse stale objects after a partial or older run.
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f"output is not empty: {directory}; use --compare-only or a fresh output root")
            directory.mkdir(parents=True, exist_ok=True)
            payload = dict(sms=sms, exporter=exporter, args=extra, output=str(directory))
            print(f"export {name} SMs={sms}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            with (directory / "export.log").open("w") as log:
                try:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", json.dumps(payload)],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    log.flush()
                    print(f"{name} SMs={sms} failed; {(directory / 'export.log').read_text()[-12000:]}",
                          file=sys.stderr, flush=True)
                    raise
        report = compare(*directories)
        report["group"] = name
        (output / name / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        for record in report["programs"]:
            print(f"{name}/{record['program']}: {record['classification']}; "
                  f"header={'same' if record['header_identical'] else 'different'}; "
                  f"sass={record['sass_identical']}; metadata_fields={len(record['metadata_changed_fields'])}", flush=True)
        return report

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export_group, job): job[0] for job in selected_jobs}
        failures = []
        for future in as_completed(futures):
            try:
                reports.append(future.result())
            except Exception as exc:
                failures.append({"group": futures[future], "error": str(exc)})
            reports.sort(key=lambda report: report["group"])
            (output / "comparison.json").write_text(json.dumps(
                {"sms": args.sms, "groups": reports, "failures": failures}, indent=2) + "\n")
    if failures:
        print(f"Failed export groups: {', '.join(row['group'] for row in failures)}", file=sys.stderr)
        return 1
    different = sum(not row["object_identical"] for report in reports for row in report["programs"])
    total = sum(len(report["programs"]) for report in reports)
    print(f"Compared {total} programs: {total - different} identical objects, {different} different. Report: {output / 'comparison.json'}")
    return int(args.require_identical and different > 0)


if __name__ == "__main__":
    raise SystemExit(main())
