#!/usr/bin/env python3
"""Publish CPU-compiled planner inputs for the live installed checkpoint inventory.

Run through scripts/build/cuteafd-dev.sh cpu, with --output and --staging on
NVMe. Publication never switches current, and refuses an output or staging
path that is or resolves through a store's current link: validate the candidate
with CUTEAFD_DEPLOYMENT_TEST_PROGRAMS=<output>/share/PROGRAMS.json, and let the
coordinator's scripts/dev/land-pr merge and promote it on every host.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
FP8_CAPACITY_RECIPE = Path("native/cmake/shared/fp8_moe.cmake")


def fp8_capacities(root=ROOT):
    """Use the native package recipe, not EXL3's independent row policy."""
    declarations = re.findall(r'^\s*set\(CUTEAFD_FP8_MOE_CAPACITIES\s+"([0-9,]+)"\)',
                              (root / FP8_CAPACITY_RECIPE).read_text(), re.M)
    if len(declarations) != 1:
        raise ValueError("expected one literal CUTEAFD_FP8_MOE_CAPACITIES recipe")
    capacities = [int(value) for value in declarations[0].split(",")]
    if (not 1 <= len(capacities) <= 6 or capacities != sorted(set(capacities))
            or any(value <= 0 or value > 0x7fffffff for value in capacities)):
        raise ValueError("native FP8 recipe requires 1..6 increasing positive int32 capacities")
    return declarations[0]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("placement_compare", ROOT / "scripts/build/compare-sm-exports.py")
kernel_records = load("placement_kernel_records", ROOT / "scripts/build/kernel-records.py")

# Installed names match build-{wip,release}-artifacts.sh. These are allocation
# inputs to the native planner, not optional release-only executable artifacts.
V41_MANIFESTS = {
    "v41-fp8": ("v41_fp8.json", "V41_FP8_AOT.json"),
    "v41-experts": ("v41_experts.json", "V41_EXPERT_AOT.json"),
    **{f"v41-{role}": ("v41_experts.json", f"V41_EXPERT_WORKSPACES/mxfp4_{role}.json")
       for role in ("rtx_backbone", "rtx_tp2", "dspark_tp2")},
}


def live_configs(root=ROOT):
    retired = {entry["config"] for entry in json.loads((root / "benchmarks/current-matrix.json").read_text())["retired"]}
    return sorted(str(path.relative_to(root)) for path in (root / "examples/configs").glob("*.config")
                  if str(path.relative_to(root)) not in retired)


def source_hashes():
    # Keep identical to plan::testing::placement_input_source_hashes. Card report
    # rewrites do not affect compilation; only the selected example inventory does.
    paths = [ROOT / "scripts/build/export-placement-inputs.py", ROOT / "scripts/build/kernel-records.py",
             ROOT / "scripts/build/release-scope.py", ROOT / "scripts/lib/wip_aot_scopes.py", ROOT / FP8_CAPACITY_RECIPE,
             ROOT / "rust/crates/cuteafd-loader/src/serving_capacity/spark_memory.json",
             *(ROOT / config for config in live_configs(ROOT))]
    return {**compare.v4_placement_sources(),
            **{str(path.relative_to(ROOT)): compare.sha(path.read_bytes()) for path in paths},
            "installed_live_configs": compare.sha("\n".join(live_configs(ROOT)).encode())}


def export_jobs():
    scope = load("placement_scope", ROOT / "scripts/build/release-scope.py")
    native_capacities = fp8_capacities(ROOT)
    experts = set()
    for config in live_configs():
        _, _, scopes = scope.entry_scopes(ROOT, {"config": config, "set": {}})
        experts.update(scope.tokens(scopes.get("CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES", "")))
    # The current release cells also include Qwen Next, whose old example path
    # has retired cards. Its live checkpoint still needs native workspace inputs.
    experts.add("qwen4:exl3-k45")
    # Inkling 2 has no live example config yet; its MXFP8 W8A8, native dynamic-amax
    # NVFP4 W4A4 and K4 MC EXL3 expert packages are registered here.
    experts.update(("inkling2:mxfp8", "inkling2:nvfp4a4", "inkling2:exl3-k45",
                    "inkling2_mini:mxfp8", "inkling2_mini:nvfp4a4", "hy4f:exl3-k45-mul1",
                    "hy4:exl3-k45-mul1"))
    geometries = [*compare.GEOMETRIES, "kimik3", "kimik31", "muse", "muse2", "ml4", "hy4f", "hy4f2",
                  "hy4", "hy42",
                  "dense_trellis_k6", "kimi_split", "expert_partial", "expert_reply",
                  "glm_stage", "glm2_stage", "glmf_stage", "glmf2_stage", "exlr8", "exlr8_dense",
                  "minimax", "inkling2", "inkling2_mini", "native_experts"]
    jobs = [("programs", "export_b12x_dsv4_aot.py", [
        "--geometry", ",".join(geometries), "--max-context", "1048576",
        "--decode-rows", "64", "--prefill-rows", "4096", "--context-split", "on",
        "--dense-trellis-families", "kimik3,glm,glmf,step5,muse,ml4,inkling2,hy4f,hy4",
        "--exlr8-family", "glm,glmf", "--exlr8-role", "coordinator"])]
    for expert in sorted(experts):
        family, kind = expert.split(":")
        if kind.startswith("exl3-k"):
            mul1 = kind.endswith("-mul1")
            bits = kind.removesuffix("-mul1").removeprefix("exl3-k")
            suffix = "-mul1" if mul1 else ""
            codebook_args = ["--codebook", "mul1"] if mul1 else []
            jobs.append((f"exl3/exl3-{family}-k{bits}{suffix}", "package_exl3_aot.py", [
                "--geometry", family, "--bits", ",".join(bits), "--capacities",
                compare.EXL3_CAPACITIES_KIMIK3 if family == "kimik3" else compare.EXL3_CAPACITIES,
                "--role", "coordinator", *codebook_args]))
        elif kind in ("fp8", "bf16"):
            geometry = family + ("_bf16" if kind == "bf16" else "")
            suffix = "-bf16-weights" if kind == "bf16" else ""
            jobs.append((f"fp8/fp8-{family}{suffix}", "package_fp8_moe_aot.py", [
                "--geometry", geometry, "--capacities", native_capacities,
                "--layouts", "tp1,tp2", "--role", "coordinator"]))
        elif family in ("inkling2", "inkling2_mini") and kind in ("mxfp8", "nvfp4a4"):
            jobs.append((f"fp8/fp8-{family}-{kind}", "package_fp8_moe_aot.py", [
                "--geometry", f"{family}_{kind.removesuffix('a4')}", "--capacities", native_capacities,
                "--layouts", "tp1" if family == "inkling2_mini" else "tp1,tp2", "--role", "coordinator"]))
    jobs.extend([
        ("v41-fp8", "export_b12x_v41_fp8_aot.py", ["--rows", compare.V41_PROJECTION_CAPACITIES]),
        ("v41-experts", "export_b12x_v41_experts_aot.py",
         ["--role", "coordinator", "--input-format", "bf16", "--rows", compare.V41_DRAFT_CAPACITIES]),
    ])
    for role in ("rtx_backbone", "rtx_tp2", "dspark_tp2"):
        jobs.append((f"v41-{role}", "export_b12x_slices_aot.py", [
            "--role", role, "--rows", compare.V41_DRAFT_CAPACITIES if role == "dspark_tp2" else compare.V41_CAPACITIES,
            "--width", "192", "--standard-names"]))
    return jobs


def worker_command(payload, staging):
    """compare-sm-exports.py --_worker, through the shared export cache when one is mounted.

    WIP builds and republishes then reuse each other's programs (export-cache.py):
    keys are what each program read and captured, not which tool asked for it.
    """
    payload = {**payload, "host_admission": True}
    worker = [sys.executable, str(ROOT / "scripts/build/compare-sm-exports.py"), "--_worker", json.dumps(payload)]
    if os.environ.get("CUTEAFD_EXPORT_CACHE", "off") in ("", "off"):
        return worker
    return [sys.executable, str(ROOT / "scripts/build/export-cache.py"), "run", "--build-dir", str(staging), "--", *worker]


def default_workers():
    sys.path.insert(0, str(ROOT / "python/tools/lib"))
    from _export_jobs import export_jobs as admitted_jobs, host_worker_cap
    return max(1, min(admitted_jobs(), host_worker_cap()))


def program_shards(catalog, workers):
    programs = catalog["programs"]
    names = [program["name"] for program in programs]
    labels = [label for program in programs for label in program["labels"]]
    if len(set(names)) != len(names) or len(set(labels)) != len(labels):
        raise ValueError("program catalog contains duplicate names or labels")
    count = min(workers, len(programs))
    return [[label for program in programs[len(programs) * index // count:len(programs) * (index + 1) // count]
             for label in program["labels"]] for index in range(count)]


def merge_program_shards(catalog, directories, output):
    manifest = catalog["manifest"]
    programs, records, plans = {}, {}, {}
    headers, includes = {}, {}
    contexts = []
    for directory in directories:
        part = json.loads((directory / "dsv4_programs.json").read_text())
        if {key: value for key, value in part.items() if key != "programs"} != {key: value for key, value in manifest.items() if key != "programs"}:
            raise ValueError("shard manifest differs from catalog")
        for program in part["programs"]:
            if program["name"] in programs:
                raise ValueError("duplicate shard program")
            programs[program["name"]] = program
        for name, target in (("program-records.json", records), ("comparison-plans.json", plans)):
            data = json.loads((directory / name).read_text())
            if target.keys() & data.keys():
                raise ValueError(f"duplicate shard metadata: {name}")
            target.update(data)
        lines = (directory / "dsv4_programs.h").read_text().splitlines()
        for line in lines:
            if line.startswith('#include "'):
                includes[line.split('"')[1].removesuffix('.h')] = line
            elif line.startswith("#define CUTEAFD_DSV4_PROGRAMS "):
                for entry in line.removeprefix("#define CUTEAFD_DSV4_PROGRAMS ").split("}, "):
                    entry = entry if entry.endswith("}") else entry + "}"
                    headers[entry.split('"')[1]] = entry
        contexts.append(json.loads((directory / "comparison-context.json").read_text()))
        for path in directory.iterdir():
            if path.suffix in (".o", ".h") and path.name != "dsv4_programs.h":
                if (output / path.name).exists():
                    raise ValueError(f"duplicate shard artifact: {path.name}")
                shutil.copy2(path, output / path.name)
    names = [program["name"] for program in catalog["programs"]]
    labels = [label for program in catalog["programs"] for label in program["labels"]]
    if set(programs) != set(names) or set(records) != {label + ".o" for label in labels} or set(plans) != set(labels):
        raise ValueError("shards do not cover catalog exactly")
    manifest = {**manifest, "programs": [programs[name] for name in names]}
    (output / "dsv4_programs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "dsv4_programs.h").write_text("\n".join([
        "#pragma once", *[includes[label] for label in labels],
        f"#define CUTEAFD_DSV4_CC_MINOR {manifest['capability'][1]}",
        "#define CUTEAFD_DSV4_PROGRAMS " + ", ".join(headers[name] for name in names), ""]))
    for name, data in (("program-records.json", records), ("comparison-plans.json", {label: plans[label] for label in labels})):
        (output / name).write_text(json.dumps(data, indent=2, sort_keys=name == "program-records.json") + "\n")
    context = dict(contexts[0])
    for part in contexts[1:]:
        if {key: value for key, value in part.items() if key not in ("seconds", "source_reads_sha256", "backend_reuse")} != {
                key: value for key, value in context.items() if key not in ("seconds", "source_reads_sha256", "backend_reuse")}:
            raise ValueError("shard worker provenance differs")
    context["source_reads_sha256"] = dict(sorted({name: value for part in contexts for name, value in part["source_reads_sha256"].items()}.items()))
    context["seconds"] = sum(part["seconds"] for part in contexts)
    context["backend_reuse"] = {key: sum(part.get("backend_reuse", {}).get(key, 0) for part in contexts) for key in ("hits", "misses")}
    (output / "comparison-context.json").write_text(json.dumps(context, indent=2) + "\n")
    return context


def compile_program_shards(job, staging, previous, lock, timeout, backend_cache, verify_skips, pool, workers):
    name, exporter, extra = job
    output = staging / name
    output.mkdir(parents=True, exist_ok=True)
    scratch = staging / ".program-shards"
    scratch.mkdir()
    catalog_dir = scratch / "catalog"
    catalog_dir.mkdir()
    payload = dict(sms=188, exporter=exporter, args=extra, program_records=True)
    if backend_cache:
        payload["backend_cache"] = str(backend_cache)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTEAFD_SOURCE_ROOT=str(ROOT))
    started = time.monotonic()
    with (catalog_dir / "export.log").open("w") as log:
        subprocess.run(worker_command({**payload, "output": str(catalog_dir), "catalog_only": True}, staging),
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    catalog = json.loads((catalog_dir / "program-catalog.json").read_text())
    setup = time.monotonic() - started
    shards = program_shards(catalog, workers)

    def compile_shard(item):
        index, labels = item
        directory = scratch / str(index)
        directory.mkdir()
        with (directory / "export.log").open("w") as log:
            subprocess.run(worker_command({**payload, "output": str(directory), "selected_objects": labels}, staging),
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        return directory

    directories = list(pool.map(compile_shard, enumerate(shards)))
    summary = merge_program_shards(catalog, directories, output)
    sm170 = check_job_sm170(job, staging, previous, timeout, env, verify_all=verify_skips)
    summary["execution_stats"] = {"stage0_skipped": False, "verified_skip": False,
                                  "catalog_seconds": setup, "shards": len(shards),
                                  "wall_seconds": time.monotonic() - started, **sm170}
    return summary


# Fields publication_records adds to a worker's raw program-records.json entries.
DERIVED_RECORD_FIELDS = ("family", "planner_metadata", "scratch_bytes_at_capacity", "group")
WORKER_RECORDS = "program-records.json"


def worker_records(program_records, name):
    """A job's raw worker program-records.json, rebuilt from the publication's program_records.

    Publications keep the records once (INPUTS.json program_records); the raw
    per-job file is the same entries for that group minus the derived fields.
    """
    prefix = name + "/"
    return {path[len(prefix):]: {key: value for key, value in record.items() if key not in DERIVED_RECORD_FIELDS}
            for path, record in program_records.items() if path.startswith(prefix)}


def worker_records_bytes(records):
    return (json.dumps(records, indent=2, sort_keys=True) + "\n").encode()


def verified_job_files(previous, name):
    """(directory, {relative: checksum}, rebuilt program records or None) for one replayable job."""
    metadata = json.loads((previous / "INPUTS.json").read_text())
    prefix = ".workers/" + name + "/"
    files = {path[len(prefix):]: checksum for path, checksum in metadata["files"].items()
             if path.startswith(prefix)}
    if not files:
        return None
    directory = previous / ".workers" / name
    for relative, checksum in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or compare.sha((directory / path).read_bytes()) != checksum:
            raise ValueError(f"invalid or corrupt replay file: {name}/{relative}")
    rebuilt = None
    if WORKER_RECORDS not in files:
        records = worker_records(metadata.get("program_records") or {}, name)
        rebuilt = worker_records_bytes(records) if records else None
    return directory, files, rebuilt


def refresh_replay_provenance(directory, lock):
    for path in directory.rglob("*.json"):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "sparkinfer_revision" in data:
            data["sparkinfer_revision"] = lock["revision"]
            if "source_tree_sha256" in data:
                data["source_tree_sha256"] = lock["source_tree_sha256"]
            path.write_text(json.dumps(data, indent=2) + "\n")
    for path in directory.rglob("v41_exl3.json"):
        data = json.loads(path.read_text())
        route = data.get("route_preparation")
        if route:
            route["sha256"] = compare.sha((path.parent / route["manifest"]).read_bytes())
            path.write_text(json.dumps(data, indent=2) + "\n")


def verified_metadata_bytes(path, lock):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "sparkinfer_revision" in data:
        data["sparkinfer_revision"] = lock.get("revision")
        if "source_tree_sha256" in data:
            data["source_tree_sha256"] = lock.get("source_tree_sha256")
    if path.name == "program-records.json":
        for record in data.values():
            record.pop("sm170_object_sha256", None)
    if path.name == "comparison-context.json":
        for key in ("seconds", "backend_reuse", "execution_stats"):
            data.pop(key, None)
    # Route-manifest checksums change when only pin provenance changes. Compare
    # the referenced manifest's complete normalized content instead of that hash.
    if path.name == "v41_exl3.json" and data.get("route_preparation"):
        route = data["route_preparation"]
        route["sha256"] = compare.sha(verified_metadata_bytes(path.parent / route["manifest"], lock))
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def verify_job_metadata(previous, current, lock, name, prior_records=None):
    old = {path.relative_to(previous): path for path in previous.rglob("*.json") if path.name != "program-timings.json"}
    new = {path.relative_to(current): path for path in current.rglob("*.json") if path.name != "program-timings.json"}
    with tempfile.TemporaryDirectory() as scratch:
        if prior_records is not None and Path(WORKER_RECORDS) not in old:
            old[Path(WORKER_RECORDS)] = Path(scratch) / WORKER_RECORDS
            old[Path(WORKER_RECORDS)].write_bytes(prior_records)
        changed = sorted(str(path) for path in old.keys() | new.keys()
                         if path not in old or path not in new
                         or verified_metadata_bytes(old[path], lock) != verified_metadata_bytes(new[path], lock))
    if changed:
        raise ValueError(f"--verify-skips changed IR or placement metadata in skipped job {name}: {changed}")


def check_job_sm170(job, staging, previous, timeout, env, verify_all=False):
    name, exporter, extra = job
    directory = staging / name
    records_path = directory / "program-records.json"
    if not records_path.exists():
        return {"sm170_seconds": 0, "sm170_programs": 0}
    records = json.loads(records_path.read_text())
    current = publication_records(staging, [job])
    prior_records = json.loads((previous / "INPUTS.json").read_text())["program_records"] if previous else {}
    old = {path: record for path, record in prior_records.items() if path.startswith(name + "/")}
    report = kernel_records.change_report(old, current)
    selected = {entry["program"].removeprefix(name + "/")
                for category in ("added", "changed") for entry in report[category]}
    if verify_all:
        selected = set(records)
    for path, record in records.items():
        prior = old.get(name + "/" + path, {})
        if path not in selected and prior.get("object_sha256") == record["object_sha256"]:
            if "sm170_object_sha256" in prior:
                record["sm170_object_sha256"] = prior["sm170_object_sha256"]
    started = time.monotonic()
    if selected:
        output = staging / ".sm170" / name
        output.mkdir(parents=True)
        payload = dict(sms=170, exporter=exporter, args=extra, output=str(output), program_records=True)
        if exporter in ("export_b12x_dsv4_aot.py", "package_fp8_moe_aot.py"):
            payload["selected_objects"] = sorted(Path(path).stem for path in selected)
        # EXL3 exports multiple objects within one fork call: recompile its whole
        # touched job, but only compare and publish the requested changed records.
        with (output / "export.log").open("w") as log:
            subprocess.run(worker_command(payload, staging), env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        secondary = json.loads((output / "program-records.json").read_text())
        for path in selected:
            records[path]["sm170_object_sha256"] = secondary[path]["object_sha256"]
    records_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return {"sm170_seconds": round(time.monotonic() - started, 3), "sm170_programs": len(selected)}


def compile_job(job, staging, previous, lock, timeout, backend_cache=None, verify_skips=False,
                program_pool=None, workers=1):
    name, exporter, extra = job
    output = staging / name
    output.mkdir(parents=True)
    payload = dict(sms=188, exporter=exporter, args=extra, output=str(output), program_records=True)
    if backend_cache:
        payload["backend_cache"] = str(backend_cache)
    prior = verified_job_files(previous, name) if previous else None
    skipped = False
    if prior:
        prior_directory, prior_files, prior_records = prior
        prior_context = json.loads((prior_directory / "comparison-context.json").read_text())
        current_inputs = kernel_records.stage0_inputs(ROOT, payload, (12, 0))
        triggers = kernel_records.job_candidates(ROOT, prior_context, current_inputs)
        print(f"Placement job {name}: candidate triggers={triggers}", flush=True)
        skipped = not triggers
        if skipped and not verify_skips:
            for relative in prior_files:
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prior_directory / relative, target)
            if prior_records is not None:
                (output / WORKER_RECORDS).write_bytes(prior_records)
            refresh_replay_provenance(output, lock)
            summary = json.loads((output / "comparison-context.json").read_text())
            summary["execution_stats"] = {"stage0_skipped": True, "verified_skip": False,
                                          "backend_hits": 0, "backend_misses": 0}
            return summary
    if program_pool is not None and workers > 1:
        summary = compile_program_shards(job, staging, previous, lock, timeout, backend_cache,
                                         verify_skips, program_pool, workers)
        if skipped:
            verify_job_metadata(prior_directory, output, lock, name, prior_records)
        return summary
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTEAFD_SOURCE_ROOT=str(ROOT),
               CUTEAFD_SPARKINFER_SOURCE_DIR=str(ROOT / "third_party/sparkinfer"),
               CUTEAFD_SPARKINFER_LOCK_FILE=str(ROOT / "third_party/sparkinfer.lock.json"))
    print(f"Placement export {name}: {extra}", flush=True)
    with (output / "export.log").open("w") as log:
        subprocess.run(worker_command(payload, staging), env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)
    if skipped:
        verify_job_metadata(prior_directory, output, lock, name, prior_records)
    summary = json.loads((output / "comparison-context.json").read_text())
    reuse = summary.get("backend_reuse", {})
    sm170 = check_job_sm170(job, staging, previous, timeout, env, verify_all=verify_skips)
    summary["execution_stats"] = {"stage0_skipped": skipped, "verified_skip": skipped and verify_skips,
                                  "backend_hits": reuse.get("hits", 0), "backend_misses": reuse.get("misses", 0),
                                  **sm170}
    return summary


def refuse_current(*paths):
    """Only scripts/dev/land-pr flips a store's current link (after the merge)."""
    for path in paths:
        path = Path(path)
        for part in (path, *path.parents):
            link = part.parent / "current"
            if part.name == "current" or (link.is_symlink() and part.parent != part
                                          and os.path.realpath(link) == os.path.realpath(part)):
                raise ValueError(f"refusing {path}: publications never write a store's current link or its "
                                 "target; publish into <store>/<sha> and promote with scripts/dev/land-pr")


def publish(destination, staging, timeout, workers, previous=None, expect=None,
            backend_cache=None, verify_skips=False):
    refuse_current(destination, staging)
    filesystem = compare.load_file("placement_filesystem", ROOT / "scripts/build/assert-build-filesystem.py")
    if backend_cache:
        filesystem.check_path(str(backend_cache))
    verify_skips = verify_skips or os.environ.get("CUTEAFD_BUILD_KIND") == "release"
    for path in (destination, staging):
        filesystem.check_path(str(path))
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"output is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = source_hashes()
    lock = json.loads((ROOT / "third_party/sparkinfer.lock.json").read_text())
    engine = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    jobs = export_jobs()

    def compile_one(job):
        return compile_job(job, staging, previous, lock, timeout, backend_cache, verify_skips)

    report_previous = previous
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if workers > 1 and jobs[0][0] == "programs" and "--geometry" in jobs[0][2]:
            pending = [pool.submit(compile_one, job) for job in jobs[1:]]
            programs = compile_job(jobs[0], staging, previous, lock, timeout,
                                   backend_cache, verify_skips, program_pool=pool, workers=workers)
            summaries = [programs, *[future.result() for future in pending]]
        else:
            summaries = list(pool.map(compile_one, jobs))
    if sources != source_hashes() or lock != json.loads((ROOT / "third_party/sparkinfer.lock.json").read_text()):
        raise ValueError("placement export sources changed during compilation")
    if any(summary.get("sparkinfer_revision") != lock["revision"]
           or summary.get("source_tree_sha256") != lock["source_tree_sha256"]
           or summary.get("cuda_initialized") is not False for summary in summaries):
        raise ValueError("placement worker provenance mismatch or initialized CUDA")
    share = destination / "share"
    share.mkdir()
    shutil.copy2(staging / "programs/dsv4_programs.json", share / "PROGRAMS.json")
    for name, exporter, _ in jobs[1:]:
        target = destination / ("lib" if name.startswith("fp8/") else "share") / name
        if name in V41_MANIFESTS:
            source, installed = V41_MANIFESTS[name]
            target = share / installed
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / name / source, target)
        elif exporter == "package_exl3_aot.py":
            for source in (staging / name).rglob("v41_exl3.json"):
                copied = target / source.relative_to(staging / name)
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copied)
        else:
            # These capacities are the production compiler's exact scratch
            # query, including the maximum across each capacity's prefill forms.
            routed = json.loads((staging / name / "routed.json").read_text())
            records = routed["programs"]
            layouts = {}
            geometry = routed["layouts"]
            for record in records:
                capacities = layouts.setdefault(record["layout"], {})
                rows = record["capacity"]
                capacities[rows] = max(capacities.get(rows, 0), record["scratch_bytes"])
            manifest = {"layouts": {layout: {**geometry[layout], "capacities": [
                {"capacity": rows, "scratch_bytes": scratch} for rows, scratch in sorted(capacities.items())]}
                for layout, capacities in layouts.items()}}
            target.mkdir(parents=True)
            for layout in layouts:
                (target / layout).mkdir()
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    copy_replay_metadata(staging, destination, jobs)
    records = publication_records(staging, jobs)
    files = {str(path.relative_to(destination)): compare.sha(path.read_bytes())
             for path in sorted(destination.rglob("*.json"))}
    provenance = dict(schema=1, engine_commit=engine, sparkinfer_revision=lock["revision"],
                      source_tree_sha256=lock["source_tree_sha256"], engine_sources_sha256=sources,
                      exports=summaries, files=files, wall_seconds=round(time.monotonic() - started, 3))
    provenance["program_records"] = records
    (destination / "INPUTS.json").write_text(json.dumps(provenance, indent=2) + "\n")
    sm_report = kernel_records.change_report({}, records)
    if sm_report["sm170_mismatches"]:
        kernel_records.write_report(destination, sm_report)
        raise ValueError(f"SM 170 / 188 OBJECT MISMATCH: {sm_report['sm170_mismatches']}")
    if report_previous:
        old = json.loads((report_previous / "INPUTS.json").read_text()).get("program_records")
        if old is None:
            raise ValueError("previous publication has no program records; seed it with a cold export")
        expected = json.loads(expect.read_text()) if expect else None
        if isinstance(expected, dict):
            expected = expected["groups"]
        if expected is not None and (not isinstance(expected, list) or not all(isinstance(x, str) for x in expected)):
            raise ValueError("--expect must contain a JSON list of export groups, or {groups: [...]}")
        report = kernel_records.change_report(old, records, expected)
        kernel_records.write_report(destination, report)
        if report["leaks"]:
            raise ValueError(f"UNEXPECTED KERNEL CHANGES outside --expect: {report['leaks']}")
    print(f"Published placement inputs: {destination}; {provenance['wall_seconds']} s; "
          f"{sum(path.stat().st_size for path in destination.rglob('*') if path.is_file())} bytes; "
          f"INPUTS.json sha256 {compare.sha((destination / 'INPUTS.json').read_bytes())}", flush=True)


def copy_replay_metadata(staging, destination, jobs):
    """Preserve the exact metadata needed to replay each job, not heavy objects.

    Program records are kept once, in INPUTS.json program_records; replay rebuilds
    each job's raw program-records.json from them (verified_job_files)."""
    for name, _, _ in jobs:
        for source in (staging / name).rglob("*.json"):
            if source.name in ("export.log", "program-timings.json") or source == staging / name / WORKER_RECORDS:
                continue
            target = destination / ".workers" / name / source.relative_to(staging / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def publication_records(staging, jobs):
    records = {}
    for name, _, _ in jobs:
        directory = staging / name
        path = directory / "program-records.json"
        if not path.exists():
            continue  # Test fixtures can represent metadata-only workers.
        job_records = json.loads(path.read_text())
        if (directory / "dsv4_programs.json").exists():
            manifest = json.loads((directory / "dsv4_programs.json").read_text())
            for program in manifest.get("programs", []):
                for plan in program.get("card_plans", [program]):
                    label = plan.get("label", program["name"])
                    record = job_records[label + ".o"]
                    record["scratch_bytes_at_capacity"] = {
                        str(program["capacity_rows"]): plan.get("scratch_bytes_at_capacity", {})}
                    record["family"] = program["family"]
                    record["planner_metadata"] = {key: value for key, value in program.items()
                                                  if key not in ("object_sha256", "card_plans")}
                    record["planner_metadata"]["plan"] = plan
        elif name in V41_MANIFESTS:
            manifest = json.loads((directory / V41_MANIFESTS[name][0]).read_text())
            metadata = {key: value for key, value in manifest.items()
                        if key not in ("artifacts", "device", "physical_sms", "sparkinfer_revision")}
            # Bind allocation-only changes to the job without repeating the
            # complete manifest for every quantizer/projection object.
            if job_records:
                job_records[min(job_records)]["planner_metadata"] = metadata
        elif name.startswith("fp8/"):
            routed = json.loads((directory / "routed.json").read_text())
            for program in routed["programs"]:
                job_records[program["name"] + ".o"]["scratch_bytes_at_capacity"] = {
                    str(program["capacity"]): program["scratch_bytes"]}
                job_records[program["name"] + ".o"]["planner_metadata"] = {
                    **program, "geometry": routed["layouts"][program["layout"]]}
        else:
            for manifest_path in directory.rglob("v41_exl3.json"):
                manifest = json.loads(manifest_path.read_text())
                route = manifest.get("route_preparation")
                if route:
                    route["content_sha256"] = compare.sha(verified_metadata_bytes(
                        manifest_path.parent / route["manifest"], {}))
                for record_path, record in job_records.items():
                    if Path(record_path).parent == manifest_path.parent.relative_to(directory):
                        record["scratch_bytes_at_capacity"] = {
                            str(manifest["capacity"]): manifest["unique_execution_buffer_bytes"]}
                        record["planner_metadata"] = {key: value for key, value in manifest.items()
                                                      if key not in ("sparkinfer_revision", "gpu", "physical_sms", "objects")}
                        record["planner_metadata"]["objects"] = manifest["objects"]
        records.update({f"{name}/{path}": {**record, "group": name} for path, record in job_records.items()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True, help="staging directory on NVMe")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="CPU export workers (default: host-admitted export jobs)")
    parser.add_argument("--previous", type=Path, help="previous publication to compare; writes CHANGES.json and CHANGES.md")
    parser.add_argument("--expect", type=Path, help="JSON list of export job groups expected to change")
    parser.add_argument("--backend-cache", type=Path, help="content-addressed lowered-MLIR cache on NVMe")
    parser.add_argument("--verify-skips", action="store_true", help="run skipped frontends and reject changed IR")
    args = parser.parse_args()
    if args.expect and not args.previous:
        parser.error("--expect requires --previous")
    if args.workers < 1 or args.timeout < 1:
        parser.error("--workers and --timeout must be positive")
    try:
        refuse_current(args.output.expanduser().absolute(), args.staging.expanduser().absolute())
    except ValueError as error:
        parser.error(str(error))
    output = args.output.expanduser().resolve()
    publish(output, args.staging.expanduser().resolve(), args.timeout, args.workers,
            args.previous.expanduser().resolve() if args.previous else None,
            args.expect.expanduser().resolve() if args.expect else None,
            args.backend_cache.expanduser().resolve() if args.backend_cache else None,
            args.verify_skips)


if __name__ == "__main__":
    main()
