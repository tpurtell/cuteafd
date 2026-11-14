#!/usr/bin/env python3
"""Publish CPU-compiled planner inputs for the live installed checkpoint inventory.

Run through scripts/build/cuteafd-dev.sh cpu, with --output and --staging on
NVMe. Publication never switches current; validate the candidate with
CUTEAFD_DEPLOYMENT_TEST_PROGRAMS=<output>/share/PROGRAMS.json before promoting it.
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
    # Inkling 2 has no live example config yet; its MXFP8 W8A8 package is registered here.
    experts.add("inkling2:mxfp8")
    geometries = [*compare.GEOMETRIES, "kimik3", "kimik31", "muse", "muse2", "ml4",
                  "dense_trellis_k6", "kimi_split", "expert_partial", "expert_reply",
                  "glm_stage", "glm2_stage", "glmf_stage", "glmf2_stage", "exlr8", "exlr8_dense", "minimax", "inkling2"]
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
                "--geometry", geometry, "--capacities", native_capacities,
                "--layouts", "tp1,tp2", "--role", "coordinator"]))
        elif kind == "mxfp8":
            jobs.append((f"fp8/fp8-{family}-mxfp8", "package_fp8_moe_aot.py", [
                "--geometry", f"{family}_mxfp8", "--capacities", native_capacities,
                "--layouts", "tp1,tp2", "--role", "coordinator"]))
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
    worker = [sys.executable, str(ROOT / "scripts/build/compare-sm-exports.py"), "--_worker", json.dumps(payload)]
    if os.environ.get("CUTEAFD_EXPORT_CACHE", "off") in ("", "off"):
        return worker
    return [sys.executable, str(ROOT / "scripts/build/export-cache.py"), "run", "--build-dir", str(staging), "--", *worker]


def verified_job_files(previous, name):
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
    return directory, files


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


def verify_job_metadata(previous, current, lock, name):
    old = {path.relative_to(previous): path for path in previous.rglob("*.json")}
    new = {path.relative_to(current): path for path in current.rglob("*.json")}
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


def compile_job(job, staging, previous, lock, timeout, backend_cache=None, verify_skips=False):
    name, exporter, extra = job
    output = staging / name
    output.mkdir(parents=True)
    payload = dict(sms=188, exporter=exporter, args=extra, output=str(output), program_records=True)
    if backend_cache:
        payload["backend_cache"] = str(backend_cache)
    prior = verified_job_files(previous, name) if previous else None
    skipped = False
    if prior:
        prior_directory, prior_files = prior
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
            refresh_replay_provenance(output, lock)
            summary = json.loads((output / "comparison-context.json").read_text())
            summary["execution_stats"] = {"stage0_skipped": True, "verified_skip": False,
                                          "backend_hits": 0, "backend_misses": 0}
            return summary
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTEAFD_SOURCE_ROOT=str(ROOT),
               CUTEAFD_SPARKINFER_SOURCE_DIR=str(ROOT / "third_party/sparkinfer"),
               CUTEAFD_SPARKINFER_LOCK_FILE=str(ROOT / "third_party/sparkinfer.lock.json"))
    print(f"Placement export {name}: {extra}", flush=True)
    with (output / "export.log").open("w") as log:
        subprocess.run(worker_command(payload, staging), env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)
    if skipped:
        verify_job_metadata(prior_directory, output, lock, name)
    summary = json.loads((output / "comparison-context.json").read_text())
    reuse = summary.get("backend_reuse", {})
    sm170 = check_job_sm170(job, staging, previous, timeout, env, verify_all=verify_skips)
    summary["execution_stats"] = {"stage0_skipped": skipped, "verified_skip": skipped and verify_skips,
                                  "backend_hits": reuse.get("hits", 0), "backend_misses": reuse.get("misses", 0),
                                  **sm170}
    return summary


def publish(destination, staging, timeout, workers, previous=None, expect=None,
            backend_cache=None, verify_skips=False, rsync_hosts=()):
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
    # Preserve only the exact metadata needed to replay a job, not heavy objects.
    for name, _, _ in jobs:
        for source in (staging / name).rglob("*.json"):
            if source.name == "export.log":
                continue
            target = destination / ".workers" / name / source.relative_to(staging / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    records = publication_records(staging, jobs)
    if records:
        programs_path = share / "PROGRAMS.json"
        programs = json.loads(programs_path.read_text())
        programs["kernel_records"] = records
        programs_path.write_text(json.dumps(programs, indent=2) + "\n")
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
          f"{sum(path.stat().st_size for path in destination.rglob('*') if path.is_file())} bytes", flush=True)
    if rsync_hosts:
        rsync_publication(destination, rsync_hosts, timeout)


def rsync_publication(destination, hosts, timeout=7200):
    resolved = Path(destination).expanduser().resolve()
    for required in ("INPUTS.json", "share/PROGRAMS.json"):
        if not (resolved / required).is_file():
            raise ValueError(f"missing published placement input {resolved / required}")
    cleaned = [host.strip() for host in hosts if host and host.strip()]
    if not cleaned:
        return
    with tempfile.TemporaryDirectory() as tmp:
        current = Path(tmp) / "current"
        current.symlink_to(resolved.name)
        for host in cleaned:
            subprocess.run(["rsync", "-a", "--delete", "--mkpath", f"{resolved}/", f"{host}:{resolved}/"],
                           check=True, timeout=timeout)
            subprocess.run(["rsync", "-a", "--mkpath", str(current), f"{host}:{resolved.parent}/"],
                           check=True, timeout=timeout)
            print(f"Synced placement inputs to {host}:{resolved} (current -> {resolved.name})", flush=True)


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
    parser.add_argument("--staging", type=Path, help="staging directory on NVMe (required unless syncing an existing publication with --rsync-host)")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--previous", type=Path, help="previous publication to compare; writes CHANGES.json and CHANGES.md")
    parser.add_argument("--expect", type=Path, help="JSON list of export job groups expected to change")
    parser.add_argument("--backend-cache", type=Path, help="content-addressed lowered-MLIR cache on NVMe")
    parser.add_argument("--verify-skips", action="store_true", help="run skipped frontends and reject changed IR")
    parser.add_argument("--rsync-host", action="append", default=[],
                        help="remote host(s) to receive the published directory and current symlink via rsync")
    args = parser.parse_args()
    hosts = [host.strip() for raw in args.rsync_host for host in raw.split(",") if host.strip()]
    if args.expect and not args.previous:
        parser.error("--expect requires --previous")
    if args.workers < 1 or args.timeout < 1:
        parser.error("--workers and --timeout must be positive")
    output = args.output.expanduser().resolve()
    if args.staging is None:
        if not hosts:
            parser.error("--staging is required unless syncing an existing publication with --rsync-host")
        rsync_publication(output, hosts, args.timeout)
        return
    publish(output, args.staging.expanduser().resolve(), args.timeout, args.workers,
            args.previous.expanduser().resolve() if args.previous else None,
            args.expect.expanduser().resolve() if args.expect else None,
            args.backend_cache.expanduser().resolve() if args.backend_cache else None,
            args.verify_skips, rsync_hosts=hosts)


if __name__ == "__main__":
    main()
