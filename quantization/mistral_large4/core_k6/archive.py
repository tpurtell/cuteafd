"""Import the published checkpoint and let native SparkNest place one copy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "mistral_large4.core_k6"

from kimi_k3.uniform_k2.storage import atomic_json, canonical
from .recipe import TARGET_REPO, TARGET_PAYLOAD_BYTES, TARGET_TENSOR_COUNT, MODULE_COUNT
from .export import SHARD_COUNT

HOSTS = ("ostrich", "dodo", "emu", "kiwi", "rhea", "moa")
NAMESPACE_HUB = Path("/hf-home/hub")
MANAGED_HUB = Path("/mnt/sparknest/hf-home/hub")


def _native(*arguments):
    result = subprocess.run(["nest", "--json", *map(str, arguments)], check=True,
                            capture_output=True, text=True, timeout=21600)
    return json.loads(result.stdout)


def _job_id(started):
    number = started.get("job")
    if type(number) is not int or number <= 0:
        raise ValueError("SparkNest did not return a native job")
    return number


def _wait(number, *, importing=False):
    result = _native("jobs", number, "--wait")
    job = result.get("job", {})
    if (job.get("finished") is not True or job.get("error") is not None
            or job.get("cancelled") is True or job.get("pending", 0)
            or any(": failed" in note for note in job.get("notes", []))
            or any(row.get("failed") or row.get("cancelled")
                   or row.get("done_files", 0) < row.get("total_files", 0)
                   for row in job.get("hosts", {}).values())):
        raise ValueError("SparkNest job did not complete successfully")
    if importing:
        counters = result.get("import", {})
        if (counters.get("finished") is not True or counters.get("errors") != []
                or counters.get("cancelled") is True):
            raise ValueError("SparkNest import did not complete successfully")
    return result


def _context(root):
    production = json.loads((root / "plan.json").read_text())
    published = json.loads((root / "publication-complete.json").read_text())
    completed = json.loads((root / "quantization-complete.json").read_text())
    identity = production.get("identity")
    if (not isinstance(identity, dict) or not identity or published.get("identity") != identity
            or completed.get("identity") != identity or published.get("repo_id") != TARGET_REPO
            or production.get("publication_repo") != TARGET_REPO
            or Path(production["root"]).resolve() != root):
        raise ValueError("archive requires the matching completed published producer")
    expected = dict(shards=SHARD_COUNT, tensors=TARGET_TENSOR_COUNT,
                    payload_bytes=TARGET_PAYLOAD_BYTES, quantized_projections=MODULE_COUNT)
    if any(completed.get("summary", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("completed checkpoint inventory differs")
    if (not re.fullmatch(r"[0-9a-f]{40}", published.get("revision", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", published.get("manifest_sha256", ""))):
        raise ValueError("invalid immutable publication revision/manifest")
    return production, published



def _placement(plan):
    goal = plan.get("goal", {})
    if (plan.get("feasible") is not True or plan.get("blocked")
            or goal.get("goal") != "place" or goal.get("selector") != "hf:" + TARGET_REPO
            or set(goal.get("hosts", [])) != set(HOSTS)
            or abs(goal.get("factor", 0) - 1 / len(HOSTS)) > 1e-9):
        raise ValueError("SparkNest placement is not the authorized six-Spark spread")
    for step in plan.get("steps", []):
        if any(row.get("group") != TARGET_REPO for row in step.get("copies", [])):
            raise ValueError("SparkNest placement includes another checkpoint")
    return plan


def _retire_local(root, production, published):
    snapshot = Path(published["local_snapshot"])
    export = Path(production["export"])
    repo_name = "models--" + TARGET_REPO.replace("/", "--")
    if (snapshot.name != published["revision"] or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != repo_name):
        raise ValueError("unexpected publication snapshot path")
    for directory in (snapshot, export):
        if directory.is_symlink() or not directory.resolve().is_relative_to(root.parent):
            raise ValueError("retirement is restricted to this task's local output")
        for i in range(1, SHARD_COUNT + 1):
            path = directory / f"model-{i:05d}-of-{SHARD_COUNT:05d}.safetensors"
            if path.is_symlink():
                raise ValueError("refuse symlink local weight retirement")
            path.unlink(missing_ok=True)
    atomic_json(root / "archive-retirement.json", dict(local_weights_retired=True,
        revision=published["revision"], time=datetime.now(timezone.utc).isoformat()))


def archive(root):
    """Resume native jobs; trust SparkNest's integrity checks without rereading weights."""
    root = Path(root).expanduser()
    if root.is_symlink():
        raise ValueError("archive root must be a regular production directory")
    root = root.resolve(strict=True)
    with (root / "archive.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another archive process holds this artifact") from None
        production, published = _context(root)
        identity = dict(producer=production["identity"], repo_id=TARGET_REPO,
            revision=published["revision"], manifest_sha256=published["manifest_sha256"])
        complete_path = root / "archive-complete.json"
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if complete.get("identity") != identity:
                raise ValueError("archive completion belongs to a different publication")
            _retire_local(root, production, published)
            return complete
        state_path = root / "native-sparknest-state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(identity=identity)
        if state.get("identity") != identity:
            raise ValueError("native storage jobs belong to a different publication")
        if not state.get("import_job"):
            source = Path(published["local_snapshot"]).parent.parent
            state["import_job"] = _job_id(_native("hf", "import", source, "--offline"))
            atomic_json(state_path, state)
        imported = _wait(state["import_job"], importing=True)
        atomic_json(root / "native-sparknest-import.json", imported)
        if not state.get("placement_plan"):
            hosts = [value for host in HOSTS for value in ("--host", host)]
            plan = _placement(_native("plan", "--place", "hf:" + TARGET_REPO,
                                      "--factor", "1/6", *hosts))
            atomic_json(root / "native-sparknest-placement-plan.json", plan)
            state["placement_plan"] = plan["id"]
            atomic_json(state_path, state)
        else:
            plan = _placement(json.loads((root / "native-sparknest-placement-plan.json").read_text()))
            if plan["id"] != state["placement_plan"]:
                raise ValueError("native placement plan identity differs")
        if not state.get("placement_job"):
            state["placement_job"] = _job_id(_native("plan", "apply", state["placement_plan"]))
            atomic_json(state_path, state)
        placed = _wait(state["placement_job"])
        atomic_json(root / "native-sparknest-placement-result.json", placed)
        repo_name = "models--" + TARGET_REPO.replace("/", "--")
        relative = Path(repo_name) / "snapshots" / published["revision"]
        listing = _native("ls", NAMESPACE_HUB / relative)
        weights = {row["name"]: row for row in listing["entries"]
                   if row["name"].endswith(".safetensors")}
        expected = {f"model-{i:05d}-of-{SHARD_COUNT:05d}.safetensors"
                    for i in range(1, SHARD_COUNT + 1)}
        if set(weights) != expected:
            raise ValueError("native managed shard inventory differs")
        counts = dict.fromkeys(HOSTS, 0)
        for row in weights.values():
            hosts = row.get("hosts", [])
            if (len(hosts) != 1 or hosts[0] not in HOSTS or row.get("sealed") is not True
                    or row.get("writing_on") is not None):
                raise ValueError("native managed shard is not sealed on exactly one Spark")
            counts[hosts[0]] += 1
        if not all(counts.values()):
            raise ValueError("native managed shards do not span six Sparks")
        complete = dict(status="complete", identity=identity, managed_snapshot=str(MANAGED_HUB / relative),
            weight_shards=SHARD_COUNT, distribution=counts, native_import_job=state["import_job"],
            native_placement_job=state["placement_job"], time=datetime.now(timezone.utc).isoformat())
        atomic_json(complete_path, complete)
        _retire_local(root, production, published)
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(archive(args.root)), flush=True)


if __name__ == "__main__":
    main()
