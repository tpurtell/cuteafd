"""Post-publication zero-copy adoption and one-copy, eight-place SparkNest spread.

This runs independently of the frozen numerical producer and publisher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "kimi_k31.k2_core_k6"

from kimi_k3.uniform_k2.storage import atomic_json, file_identity
from step5.core_k6.publish import _manifest_digest
from .publish import _collect_local, _complete_plan
from .recipe import TARGET_REPO

BASE = Path.home() / ".cache/cuteafd/quantization/kimi-k31-k2-core-k6"
ROOT = BASE / "production"
EXPORT = Path.home() / ".cache/huggingface/cuteafd-exports/Kimi-K3.1-Pro-EXL3-K2-Core-K6"
LOCAL_HUB = Path.home() / ".cache/huggingface/cuteafd-local-hub"
STAGE_PARENT = Path("/srv/sparknest/import-sources")
BUILD_ROOT = Path.home() / ".cache/cuteafd/builds/kimi-k31-quant"
NAMESPACE = Path("/hf-home/hub")
MANAGED = Path("/mnt/sparknest/hf-home/hub")
REPO_NAME = "models--" + TARGET_REPO.replace("/", "--")
TARGETS = ("raptor (D2)", "raptor (D3)", "ostrich", "emu", "dodo", "kiwi", "rhea", "moa")
PRODUCER_UNIT = "kimi-k31-quant-1793900179096379410.service"
HANDOFF_UNIT = "cuteafd-kimi-k31-handoff.service"


def native(*args):
    result = subprocess.run(["nest", "--json", *map(str, args)], check=True,
                            capture_output=True, text=True, timeout=21600)
    text = result.stdout
    notice = TARGET_REPO + ": mirrored\n"
    if text.startswith(notice):
        text = text[len(notice):]
    return json.loads(text)


def job_id(result):
    value = result.get("job")
    if type(value) is not int or value <= 0:
        raise ValueError("SparkNest did not return a native job id")
    return value


def wait_job(number, *, importing=False, files=0):
    result = native("jobs", number, "--wait")
    job = result.get("job", {})
    if (job.get("finished") is not True or job.get("error") or job.get("cancelled")
            or job.get("pending", 0) or any(": failed" in n for n in job.get("notes", []))
            or any(not h.get("finished") or h.get("failed") or h.get("cancelled")
                   or h.get("done_files", 0) != h.get("total_files", 0)
                   or (h.get("total_bytes", 0) > 0
                       and h.get("done_bytes", 0) != h["total_bytes"])
                   for h in job.get("hosts", {}).values())):
        raise ValueError("native SparkNest job failed or is incomplete")
    if importing:
        counts = result.get("import", {})
        if (counts.get("finished") is not True or counts.get("errors") != []
                or counts.get("cancelled") or counts.get("copied_bytes") != 0
                or sum(counts.get(k, 0) for k in ("files", "adopted", "skipped")) != files):
            raise ValueError("zero-copy import was not verified")
    return result


def check_plan(plan, *, offgrid=False):
    goal = plan.get("goal", {})
    if (plan.get("feasible") is not True or plan.get("blocked")
            or goal.get("goal") != ("offgrid" if offgrid else "place")
            or goal.get("selector") != "hf:" + TARGET_REPO):
        raise ValueError("placement is not restricted to this checkpoint")
    if offgrid:
        if goal.get("hosts") != [] or any(s.get("kind") != "evict" for s in plan.get("steps", [])):
            raise ValueError("off-grid retirement may only remove this checkpoint's extra copies")
    elif (len(goal.get("hosts", [])) != 8 or set(goal["hosts"]) != set(TARGETS)
          or goal.get("factor") != 1 / 8 or goal.get("by_speed", False)):
        raise ValueError("placement is not the authorized even eight-place spread")
    refs = str(NAMESPACE / REPO_NAME / "refs")
    for step in plan.get("steps", []):
        if any(c.get("group") not in (TARGET_REPO, refs) for c in step.get("copies", [])):
            raise ValueError("placement touches another checkpoint")
        if step.get("kind") == "replicate" and step.get("host") not in TARGETS:
            raise ValueError("placement writes to an unauthorized disk")
        if any(c.get("skip") for c in step.get("copies", [])):
            raise ValueError("placement cannot skip selected files")
    return plan


def context(root):
    if root.is_symlink() or root.resolve() != ROOT.resolve():
        raise ValueError("handoff is restricted to this production run")
    plan = json.loads((root / "plan.json").read_text())
    published = json.loads((root / "publication-complete.json").read_text())
    revision = published.get("revision", "")
    snapshot = LOCAL_HUB / REPO_NAME / "snapshots" / revision
    if (not re.fullmatch(r"[0-9a-f]{40}", revision)
            or not re.fullmatch(r"[0-9a-f]{64}", published.get("manifest_sha256", ""))
            or not isinstance(plan.get("identity"), dict) or not plan["identity"]
            or plan.get("root") != str(root) or plan.get("publication_repo") != TARGET_REPO
            or plan.get("export") != str(EXPORT)
            or published.get("local_snapshot") != str(snapshot)
            or published.get("repo_id") != TARGET_REPO
            or published.get("identity") != plan.get("identity")):
        raise ValueError("publication identity or task output path differs")
    identity = dict(producer=plan["identity"], repo_id=TARGET_REPO,
                    revision=revision, manifest_sha256=published["manifest_sha256"])
    return plan, published, snapshot, identity


def stage_snapshot(snapshot, files, revision):
    """Reflink across /home and /srv subvolumes; never fall back to a copy."""
    stage = STAGE_PARENT / REPO_NAME
    target = stage / "snapshots" / revision
    if stage.is_symlink() or target.is_symlink():
        raise ValueError("import staging cannot be a symlink")
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = stage / "staging-receipts.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for name, value in files.items():
        source, destination = snapshot / name, target / name
        if source.is_symlink() or destination.is_symlink():
            raise ValueError("no symlinks in standalone import payload")
        if file_identity(source) != value["file_identity"]:
            raise ValueError("publication snapshot identity differs: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Reuse our successful mandatory-reflink receipt, without rehashing
            # weights. An interrupted clone without a receipt is recreated.
            receipt = receipts.get(name)
            if receipt is None:
                destination.unlink()
            elif (receipt.get("source_sha256") != value["sha256"]
                  or receipt.get("file_identity") != file_identity(destination)):
                raise ValueError("existing no-copy staging identity differs: " + name)
        if not destination.exists():
            if source.stat().st_size != value["size"]:
                raise ValueError("publication snapshot size differs: " + name)
            subprocess.run(["cp", "--reflink=always", "--preserve=timestamps", "--",
                            str(source), str(destination)], check=True, timeout=180)
            receipts[name] = dict(source_sha256=value["sha256"], file_identity=file_identity(destination))
            atomic_json(receipt_path, receipts)
    refs = stage / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(revision)
    return stage


def verify_managed(files, revision, *, allow_offgrid=False):
    relative = NAMESPACE / REPO_NAME / "snapshots" / revision
    seen, distribution = set(), dict.fromkeys(TARGETS, 0)
    largest = max(v["size"] for n, v in files.items() if n.endswith(".safetensors"))
    for directory in sorted({str(Path(name).parent) for name in files}):
        listing = native("ls", relative / directory)
        for row in listing["entries"]:
            name = (Path(directory) / row["name"]).as_posix()
            if row.get("kind") == "directory":
                continue
            hosts = row.get("hosts", [])
            placed = set(hosts) & set(TARGETS)
            if (name not in files or row.get("kind") != "regular"
                    or row.get("size") != files[name]["size"]
                    or row.get("sealed") is not True or row.get("writing_on") is not None
                    or len(placed) != 1 or (not allow_offgrid and len(hosts) != 1)):
                raise ValueError("managed inventory or one-copy placement differs: " + name)
            seen.add(name)
            if name.endswith(".safetensors"):
                distribution[next(iter(placed))] += row["size"]
    if seen != set(files) or not all(distribution.values()):
        raise ValueError("managed files do not cover all eight targets")
    if max(distribution.values()) - min(distribution.values()) > largest:
        raise ValueError("managed shard bytes are not evenly spread")
    main = MANAGED / REPO_NAME / "refs/main"
    if main.read_text().strip() != revision:
        raise ValueError("managed HF main revision differs")
    refs = native("ls", NAMESPACE / REPO_NAME / "refs")["entries"]
    if (len(refs) != 1 or refs[0].get("name") != "main"
            or refs[0].get("kind") != "regular" or refs[0].get("writing_on") is not None
            or len(set(refs[0].get("hosts", [])) & set(TARGETS)) != 1
            or (not allow_offgrid and len(refs[0]["hosts"]) != 1)):
        raise ValueError("managed main ref placement differs")
    return distribution


def unused(paths):
    """Use the repository's process scanner; refuse to delete a mount source."""
    loader = importlib.machinery.SourceFileLoader("kimi_k31_cache_gc", str(REPO / "scripts/dev/cache-gc"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    gc = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = gc
    loader.exec_module(gc)
    mounts = gc.docker_mounts()
    if mounts is None:
        raise RuntimeError("cannot inspect containers before cleanup")
    busy, partial = gc.process_paths(Path("/"))
    if partial:
        # WSL makes login infrastructure and the capability-bearing SparkNest
        # daemon non-dumpable even to their uid. Their completed native jobs
        # attest that import/placement has closed staging. Do not excuse an
        # unreadable numerical worker or any other unknown user process.
        for proc in Path("/proc").iterdir():
            try:
                if not proc.name.isdigit() or proc.stat().st_uid != os.getuid():
                    continue
                incomplete = False
                for name in ("cwd", "root", "exe"):
                    try:
                        os.readlink(proc / name)
                    except PermissionError:
                        incomplete = True
                for name in ("maps", "environ"):
                    try:
                        (proc / name).read_bytes()
                    except PermissionError:
                        incomplete = True
                if not incomplete:
                    continue
                argv = (proc / "cmdline").read_bytes().split(b"\0")
                known = (argv[:2] == [b"/usr/lib/systemd/systemd", b"--user"]
                    or argv[0] == b"(sd-pam)"
                    or argv[0].strip() == b"sshd-session: tj@notty"
                    or argv[:3] == [b"/home/linuxbrew/.linuxbrew/opt/sparknest/bin/sparknestd",
                                    b"--config", b"/srv/sparknest/node.toml"])
                if not known:
                    raise RuntimeError("cannot inspect user process before cleanup: " + proc.name)
            except FileNotFoundError:
                continue  # exited while inspecting
    for path in paths:
        if path.is_symlink():
            raise ValueError("refuse symlink retirement: " + str(path))
        resolved = path.resolve()
        if any(gc.inside(p, resolved) for p in busy):
            raise RuntimeError("task output is still used by a process: " + str(path))
        if any(gc.inside(p, resolved) for p in mounts):
            raise RuntimeError("task output contains a container mount source: " + str(path))


def retire(root):
    # Exact, task-owned leaves only. Never enumerate or remove the managed hub.
    paths = [EXPORT, LOCAL_HUB / REPO_NAME, STAGE_PARENT / REPO_NAME,
             root / "frontier", root / "head/stages", root / "head/projections",
             BASE / "kernel-cache", BASE / "pilot/candidates", BASE / "pilot-final/candidates",
             BUILD_ROOT / "exllamav3", BUILD_ROOT / "tmp"]
    paths.extend(BUILD_ROOT.glob("target*"))
    for layer in range(93):
        paths.extend(root / f"layers/{layer:03d}" / leaf
                     for leaf in ("stages", "projections", "expert-projections"))
    paths = [p for p in paths if p.exists() or p.is_symlink()]
    unused(paths)
    for path in paths:
        shutil.rmtree(path)
    atomic_json(root / "handoff-cleanup-complete.json", dict(paths=list(map(str, paths)),
        logs_and_scalar_receipts_retained=True, managed_source_untouched=True,
        time=datetime.now(timezone.utc).isoformat()))


def handoff(root):
    root = Path(root).expanduser()
    plan, published, snapshot, identity = context(root)
    with (root / "handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = root / "handoff-complete.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            if record.get("identity") != identity or record.get("status") != "passed":
                raise ValueError("handoff completion identity differs")
            return record
        state_path = root / "sparknest-handoff-state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(identity=identity)
        if state.get("identity") != identity:
            raise ValueError("storage handoff belongs to a different publication")
        if not state.get("files"):
            _complete_plan(EXPORT, root, TARGET_REPO)
            files = _collect_local(EXPORT, plan["identity"])
            if _manifest_digest(files) != published["manifest_sha256"]:
                raise ValueError("local publication manifest changed")
            if len(files) != published["files"]:
                raise ValueError("publication file count differs")
            for name, value in files.items():
                if not os.path.samefile(value["path"], snapshot / name):
                    raise ValueError("publication cache is not the verified hardlink snapshot")
            state["files"] = {n: {k: v for k, v in value.items() if k != "path"}
                              for n, value in files.items()}
            atomic_json(state_path, state)
        files = state["files"]
        if (_manifest_digest(files) != identity["manifest_sha256"]
                or len(files) != published["files"]):
            raise ValueError("persisted handoff manifest differs")
        if not state.get("import_job"):
            stage = stage_snapshot(snapshot, files, published["revision"])
            atomic_json(state_path, state)
            state["import_job"] = job_id(native("hf", "import", stage, "--offline", "--no-copy"))
            atomic_json(state_path, state)
        imported = wait_job(state["import_job"], importing=True, files=len(files))
        atomic_json(root / "sparknest-import-complete.json", imported)
        if not state.get("placement_plan"):
            arguments = [v for place in TARGETS for v in ("--host", place)]
            placement = check_plan(native("plan", "--place", "hf:" + TARGET_REPO,
                                          "--factor", "1/8", *arguments))
            atomic_json(root / "sparknest-placement-plan.json", placement)
            state["placement_plan"] = placement["id"]
            atomic_json(state_path, state)
        else:
            saved = check_plan(json.loads((root / "sparknest-placement-plan.json").read_text()))
            if saved["id"] != state["placement_plan"]:
                raise ValueError("saved placement plan identity differs")
        if not state.get("placement_job"):
            state["placement_job"] = job_id(native("plan", "apply", state["placement_plan"]))
            atomic_json(state_path, state)
        placed = wait_job(state["placement_job"])
        atomic_json(root / "sparknest-placement-complete.json", placed)
        # Native fabric placement deliberately retains off-grid adoption copies.
        # Prove the distributed copy first, then let SparkNest remove only this
        # checkpoint's extra off-grid copies, never its last managed copy.
        verify_managed(files, published["revision"], allow_offgrid=True)
        if not state.get("offgrid_plan"):
            offgrid = check_plan(native("plan", "--offgrid", "hf:" + TARGET_REPO), offgrid=True)
            atomic_json(root / "sparknest-offgrid-plan.json", offgrid)
            state["offgrid_plan"] = offgrid["id"]
            atomic_json(state_path, state)
        else:
            offgrid = check_plan(json.loads((root / "sparknest-offgrid-plan.json").read_text()),
                                 offgrid=True)
            if offgrid["id"] != state["offgrid_plan"]:
                raise ValueError("saved off-grid plan identity differs")
        if offgrid.get("steps"):
            if not state.get("offgrid_job"):
                state["offgrid_job"] = job_id(native("plan", "apply", offgrid["id"]))
                atomic_json(state_path, state)
            atomic_json(root / "sparknest-offgrid-complete.json", wait_job(state["offgrid_job"]))
        distribution = verify_managed(files, published["revision"])
        atomic_json(root / "sparknest-placement-verified.json", dict(identity=identity,
            distribution_bytes=distribution, files=len(files), one_copy_per_file=True))
        retire(root)
        record = dict(status="passed", identity=identity, repo_id=TARGET_REPO,
            revision=published["revision"], files=len(files), copied_bytes=0,
            targets=list(TARGETS), distribution_bytes=distribution,
            managed_snapshot=str(MANAGED / REPO_NAME / "snapshots" / published["revision"]),
            time=datetime.now(timezone.utc).isoformat())
        atomic_json(complete, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--wait-for-publication", action="store_true")
    args = parser.parse_args()
    if args.wait_for_publication:
        while not (args.root / "publication-complete.json").exists():
            time.sleep(60)
        # The publisher's hardlinks must be closed and its supervisor finished.
        while True:
            result = subprocess.run(["systemctl", "--user", "show", PRODUCER_UNIT,
                "-p", "ActiveState", "--value"], check=True, capture_output=True, text=True, timeout=30)
            if result.stdout.strip() == "inactive":
                break
            time.sleep(60)
    print(json.dumps(handoff(args.root)), flush=True)
    # Queue the final result before the timer retires, so completion reaches TJ.
    subprocess.run([sys.executable, str(BASE / "progress-updates/notify.py")],
                   check=True, timeout=120)
    subprocess.run(["systemctl", "--user", "disable", PRODUCER_UNIT], check=True, timeout=30)
    subprocess.run(["systemctl", "--user", "disable", "--now",
                    "cuteafd-kimi-k31-progress.timer"], check=True, timeout=30)
    subprocess.run(["systemctl", "--user", "disable", HANDOFF_UNIT], check=True, timeout=30)


if __name__ == "__main__":
    main()
