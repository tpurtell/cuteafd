#!/usr/bin/env python3
"""Publish CPU-compiled planner inputs for the live installed checkpoint inventory.

Run through scripts/build/cuteafd-dev.sh cpu, with --output and --staging on
NVMe. Publication never switches current; validate the candidate with
CUTEAFD_DEPLOYMENT_TEST_PROGRAMS=<output>/PROGRAMS.json before promoting it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("placement_compare", ROOT / "scripts/build/compare-sm-exports.py")


def live_configs(root=ROOT):
    retired = {entry["config"] for entry in json.loads((root / "benchmarks/current-matrix.json").read_text())["retired"]}
    return sorted(str(path.relative_to(root)) for path in (root / "examples/configs").glob("*.config")
                  if str(path.relative_to(root)) not in retired)


def source_hashes():
    # Keep identical to plan::testing::placement_input_source_hashes. Card report
    # rewrites do not affect compilation; only the selected example inventory does.
    paths = [ROOT / "scripts/build/export-placement-inputs.py",
             ROOT / "scripts/build/release-scope.py", ROOT / "scripts/lib/wip_aot_scopes.py",
             *(ROOT / config for config in live_configs(ROOT))]
    return {**compare.v4_placement_sources(),
            **{str(path.relative_to(ROOT)): compare.sha(path.read_bytes()) for path in paths},
            "installed_live_configs": compare.sha("\n".join(live_configs(ROOT)).encode())}


def export_jobs():
    scope = load("placement_scope", ROOT / "scripts/build/release-scope.py")
    experts = set()
    for config in live_configs():
        _, _, scopes = scope.entry_scopes(ROOT, {"config": config, "set": {}})
        experts.update(scope.tokens(scopes.get("CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES", "")))
    # The current release cells also include Qwen Next, whose old example path
    # has retired cards. Its live checkpoint still needs native workspace inputs.
    experts.add("qwen4:exl3-k45")
    geometries = [*compare.GEOMETRIES, "kimik3", "muse", "muse2", "ml4",
                  "dense_trellis_k6", "kimi_split", "expert_partial", "expert_reply",
                  "glm_stage", "glm2_stage", "glmf_stage", "glmf2_stage", "exlr8", "exlr8_dense"]
    jobs = [("programs", "export_b12x_dsv4_aot.py", [
        "--geometry", ",".join(geometries), "--max-context", "1048576",
        "--decode-rows", "64", "--prefill-rows", "4096", "--context-split", "on",
        "--dense-trellis-families", "kimik3,glm,glmf,step5,muse,ml4",
        "--exlr8-family", "glm,glmf", "--exlr8-role", "coordinator"])]
    for expert in sorted(experts):
        family, kind = expert.split(":")
        if kind.startswith("exl3-k"):
            bits = kind.removeprefix("exl3-k")
            jobs.append((f"exl3/exl3-{family}-k{bits}", "package_exl3_aot.py", [
                "--geometry", family, "--bits", ",".join(bits), "--capacities",
                compare.EXL3_CAPACITIES_KIMIK3 if family == "kimik3" else compare.EXL3_CAPACITIES,
                "--role", "coordinator"]))
        elif kind in ("fp8", "bf16"):
            geometry = family + ("_bf16" if kind == "bf16" else "")
            suffix = "-bf16-weights" if kind == "bf16" else ""
            jobs.append((f"fp8/fp8-{family}{suffix}", "package_fp8_moe_aot.py", [
                "--geometry", geometry, "--capacities", compare.EXL3_CAPACITIES,
                "--layouts", "tp1,tp2", "--role", "coordinator"]))
    return jobs


def reusable_workers(previous, staging, jobs, sources, lock):
    metadata = json.loads((previous / "INPUTS.json").read_text())
    # A publisher-only refinement can repackage identical compiler outputs. All
    # actual compiler inputs, including added/removed source files, must match.
    ignored = {"scripts/build/export-placement-inputs.py", "benchmarks/current-matrix.json",
               "installed_live_configs"}
    retired = {entry["config"] for entry in json.loads((ROOT / "benchmarks/current-matrix.json").read_text())["retired"]}
    # Older publications also hashed retired examples. Drop only those legacy
    # entries; live content and every compiler-source fingerprint still match.
    compiler_sources = lambda hashes: {key: value for key, value in hashes.items()
                                      if key not in ignored and key not in retired}
    if (metadata.get("schema") != 1 or metadata.get("sparkinfer_revision") != lock["revision"]
            or metadata.get("source_tree_sha256") != lock["source_tree_sha256"]
            or compiler_sources(metadata.get("engine_sources_sha256", {})) != compiler_sources(sources)):
        raise ValueError("reused placement compiler source provenance mismatch; run a full export")
    files = metadata.get("files", {})
    if "PROGRAMS.json" not in files:
        raise ValueError("reused placement file inventory missing")
    for path, checksum in files.items():
        if Path(path).is_absolute() or any(part in (".", "..") for part in Path(path).parts):
            raise ValueError(f"invalid reused placement path {path}")
        if compare.sha((previous / path).read_bytes()) != checksum:
            raise ValueError(f"reused placement checksum mismatch: {path}")
    summaries = metadata.get("exports", [])
    if len(summaries) != len(jobs):
        raise ValueError("reused placement worker inventory mismatch")
    for (name, exporter, args), recorded in zip(jobs, summaries):
        actual = json.loads((staging / name / "comparison-context.json").read_text())
        expected = dict(exporter=exporter, args=args, sparkinfer_revision=lock["revision"],
                        source_tree_sha256=lock["source_tree_sha256"], cuda_initialized=False,
                        sms=188, capability=[12, 0],
                        exporter_sha256=compare.sha((ROOT / "python/tools/aot" / exporter).read_bytes()),
                        comparison_script_sha256=compare.sha((ROOT / "scripts/build/compare-sm-exports.py").read_bytes()),
                        export_sources_sha256={str(path.relative_to(ROOT)): compare.sha(path.read_bytes())
                            for path in sorted((ROOT / "python/tools/aot").glob("*.py"))})
        if actual != recorded or any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"reused placement worker provenance mismatch: {name}; run a full export")
    return metadata


def publish(destination, staging, timeout, workers, reuse_workers=None):
    filesystem = compare.load_file("placement_filesystem", ROOT / "scripts/build/assert-build-filesystem.py")
    for path in (destination, staging):
        filesystem.check_path(str(path))
        if path.exists() and any(path.iterdir()) and not (reuse_workers and path == staging):
            raise ValueError(f"output is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = source_hashes()
    lock = json.loads((ROOT / "third_party/sparkinfer.lock.json").read_text())
    engine = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    jobs = export_jobs()

    def compile_one(job):
        name, exporter, extra = job
        output = staging / name
        output.mkdir(parents=True)
        payload = dict(sms=188, exporter=exporter, args=extra, output=str(output))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTEAFD_SOURCE_ROOT=str(ROOT),
                   CUTEAFD_SPARKINFER_SOURCE_DIR=str(ROOT / "third_party/sparkinfer"),
                   CUTEAFD_SPARKINFER_LOCK_FILE=str(ROOT / "third_party/sparkinfer.lock.json"))
        print(f"Placement export {name}: {extra}", flush=True)
        with (output / "export.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/build/compare-sm-exports.py"),
                            "--_worker", json.dumps(payload)], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        return json.loads((output / "comparison-context.json").read_text())

    previous = reusable_workers(reuse_workers, staging, jobs, sources, lock) if reuse_workers else None
    if previous:
        summaries = previous["exports"]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(compile_one, jobs))
    if sources != source_hashes() or lock != json.loads((ROOT / "third_party/sparkinfer.lock.json").read_text()):
        raise ValueError("placement export sources changed during compilation")
    if any(summary.get("sparkinfer_revision") != lock["revision"]
           or summary.get("source_tree_sha256") != lock["source_tree_sha256"]
           or summary.get("cuda_initialized") is not False for summary in summaries):
        raise ValueError("placement worker provenance mismatch or initialized CUDA")
    shutil.copy2(staging / "programs/dsv4_programs.json", destination / "PROGRAMS.json")
    for name, exporter, _ in jobs[1:]:
        target = destination / name
        if exporter == "package_exl3_aot.py":
            for source in (staging / name).rglob("v41_exl3.json"):
                copied = target / source.relative_to(staging / name)
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copied)
        else:
            # These capacities are the production compiler's exact scratch
            # query, including the maximum across each capacity's prefill forms.
            records = json.loads((staging / name / "routed.json").read_text())["programs"]
            layouts = {}
            for record in records:
                capacities = layouts.setdefault(record["layout"], {})
                rows = record["capacity"]
                capacities[rows] = max(capacities.get(rows, 0), record["scratch_bytes"])
            manifest = {"layouts": {layout: {"capacities": [
                {"capacity": rows, "scratch_bytes": scratch} for rows, scratch in sorted(capacities.items())]}
                for layout, capacities in layouts.items()}}
            target.mkdir(parents=True)
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = {str(path.relative_to(destination)): compare.sha(path.read_bytes())
             for path in sorted(destination.rglob("*.json"))}
    if previous and files != previous["files"]:
        raise ValueError("reused placement worker output checksum mismatch; run a full export")
    provenance = dict(schema=1, engine_commit=engine, sparkinfer_revision=lock["revision"],
                      source_tree_sha256=lock["source_tree_sha256"], engine_sources_sha256=sources,
                      exports=summaries, files=files, wall_seconds=round(time.monotonic() - started, 3))
    if previous:
        provenance.update(reused_workers_from=str(reuse_workers),
                          compiler_engine_commit=previous.get("compiler_engine_commit", previous["engine_commit"]),
                          compiler_wall_seconds=previous.get("compiler_wall_seconds", previous["wall_seconds"]))
    (destination / "INPUTS.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Published placement inputs: {destination}; {provenance['wall_seconds']} s; "
          f"{sum(path.stat().st_size for path in destination.rglob('*') if path.is_file())} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-workers", type=Path,
                        help="verified prior publication; reuse its exact retained --staging worker outputs")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("--workers and --timeout must be positive")
    publish(args.output.expanduser().resolve(), args.staging.expanduser().resolve(), args.timeout, args.workers,
            args.reuse_workers.expanduser().resolve() if args.reuse_workers else None)


if __name__ == "__main__":
    main()
