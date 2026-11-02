"""Distribute one published copy to six Sparks, then import it without copies.

Each Spark receives disjoint weight shards and the common metadata in a
partial HF snapshot. Offline HF imports add those partial snapshots to the
shared namespace. No cluster configuration or placement rules change.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "quantization"))
if __package__ in (None, ""):
    __package__ = "mistral_large4.core_k6"

from kimi_k3.uniform_k2.storage import atomic_json, canonical, file_identity, sha256_file
from .recipe import TARGET_REPO, TARGET_PAYLOAD_BYTES, TARGET_TENSOR_COUNT, MODULE_COUNT
from .export import SHARD_COUNT

HOSTS = ("ostrich", "dodo", "emu", "kiwi", "rhea", "moa")
MANAGED_HUB = Path("/mnt/sparknest/hf-home/hub")
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=6")

# Paths and file names are data on stdin, never interpolated into shell code.
_REMOTE_HELPER = r'''
import hashlib,json,os,shutil,stat,sys
from pathlib import Path
r=json.load(sys.stdin)
base=Path.home()/'.cache/cuteafd/builds'
stage=base/r['stage_name']
if not r['stage_name'].startswith('mistral-published-import-') or '/' in r['stage_name']:
 raise ValueError('invalid task staging name')
if stage.is_symlink():raise ValueError('refuse symlink staging root')
owner=stage/'owner.json'
identity=r['identity']
if not stage.exists() and r['operation']=='retire':
 print(json.dumps({'retired':True,'already_absent':True}));sys.exit(0)
if stage.exists():
 if not owner.is_file() or owner.is_symlink() or json.loads(owner.read_text())!=identity:
  raise ValueError('staging directory is not owned by this publication')
 if any(p.is_symlink() for p in stage.rglob('*')):
  raise ValueError('refuse symlink inside task staging')
elif r['operation']=='prepare':
 stage.mkdir(parents=True)
 owner.write_text(json.dumps(identity,sort_keys=True))
else:raise ValueError('task staging is absent')
if stage.stat().st_dev!=Path('/srv/sparknest').stat().st_dev:
 raise ValueError('staging is not on the primary managed store filesystem')
repo=stage/'hub'/r['repo_name']
snapshot=repo/'snapshots'/r['revision']
if r['operation']=='prepare':
 snapshot.mkdir(parents=True,exist_ok=True)
 (repo/'refs').mkdir(exist_ok=True)
 (repo/'refs/main').write_text(r['revision'])
 print(json.dumps({'repo':str(repo),'snapshot':str(snapshot),'same_filesystem':True}))
elif r['operation']=='verify':
 expected={'refs/main'}|{'snapshots/'+r['revision']+'/'+n for n in r['files']}
 actual=set()
 for p in repo.rglob('*'):
  if p.is_symlink():raise ValueError('staging contains a symlink')
  if p.is_file():actual.add(p.relative_to(repo).as_posix())
 if actual!=expected:raise ValueError('partial snapshot inventory differs')
 if (repo/'refs/main').read_text()!=r['revision']:raise ValueError('staging ref differs')
 total=0
 for name,want in r['files'].items():
  p=snapshot/name
  if p.stat().st_size!=want['size']:raise ValueError('staged file size differs: '+name)
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  if digest!=want['sha256']:raise ValueError('staged checksum differs: '+name)
  total+=want['size']
 print(json.dumps({'verified':True,'files':len(r['files']),'bytes':total,'identity':identity}))
elif r['operation']=='retire':
 shutil.rmtree(stage)
 print(json.dumps({'retired':True}))
else:raise ValueError('unknown staging operation')
'''


def _run(command, *, timeout=600, input=None):
    return subprocess.run(list(map(str, command)), input=input, capture_output=True, text=True,
                          timeout=timeout, check=True).stdout


def _ssh(host, command, *, timeout=600, input=None):
    if host not in HOSTS:
        raise ValueError("host is outside the authorized Spark pool")
    return _run([*SSH, host, shlex.join(list(map(str, command)))], timeout=timeout, input=input)


def _remote(host, operation, plan):
    payload = dict(operation=operation, stage_name=plan["stage_name"], identity=plan["identity"],
        repo_name=plan["repo_name"], revision=plan["revision"], files=_host_files(plan, host))
    result = json.loads(_ssh(host, ["python3", "-c", _REMOTE_HELPER],
                             timeout=7200, input=canonical(payload)))
    if not isinstance(result, dict):
        raise ValueError("invalid remote staging response")
    return result


def _relative(name):
    path = Path(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\0" in name
            or "\n" in name or "\r" in name or path.as_posix() != name):
        raise ValueError("invalid published relative path")
    return name


def _manifest_digest(files):
    entries = [[name, row["size"], row["sha256"]] for name, row in sorted(files.items())]
    return hashlib.sha256(canonical(entries).encode()).hexdigest()


def _shards():
    return {f"model-{i:05d}-of-{SHARD_COUNT:05d}.safetensors" for i in range(1, SHARD_COUNT + 1)}


def balance(files):
    """Greedy byte balance; every weight has exactly one designated Spark."""
    if {name for name in files if name.endswith(".safetensors")} != _shards():
        raise ValueError("publication shard inventory differs")
    totals = dict.fromkeys(HOSTS, 0)
    assignments = {}
    for name in sorted(_shards(), key=lambda n: (-files[n]["size"], n)):
        host = min(HOSTS, key=lambda h: (totals[h], HOSTS.index(h)))
        assignments[name] = host
        totals[host] += files[name]["size"]
    return assignments


def _host_files(plan, host):
    return {name: row for name, row in plan["files"].items()
            if not name.endswith(".safetensors") or plan["assignments"][name] == host}


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


def _new_plan(root, production, published):
    export = Path(production["export"])
    snapshot = Path(published["local_snapshot"])
    repo_name = "models--" + TARGET_REPO.replace("/", "--")
    if (export.is_symlink() or snapshot.is_symlink() or not export.is_dir() or not snapshot.is_dir()
            or snapshot.name != published["revision"] or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != repo_name):
        raise ValueError("unexpected publication export/snapshot paths")
    files = {}
    for directory, names, leaves in os.walk(snapshot, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in names):
            raise ValueError("refuse symlink publication snapshot directory")
        for leaf in leaves:
            path = Path(directory) / leaf
            name = _relative(path.relative_to(snapshot).as_posix())
            original = export / name
            if path.is_symlink() or original.is_symlink() or not path.is_file() or not original.is_file():
                raise ValueError("publication snapshot files must be regular")
            if not os.path.samefile(path, original):
                raise ValueError("local published snapshot no longer hardlinks its export")
            before = file_identity(path)
            digest = sha256_file(path)
            if file_identity(path) != before:
                raise ValueError("published file changed during checksum validation")
            files[name] = dict(size=before["bytes"], sha256=digest, file_identity=before)
    if len(files) != published.get("files") or _manifest_digest(files) != published["manifest_sha256"]:
        raise ValueError("published manifest checksum differs before distribution")
    assignments = balance(files)
    identity = dict(producer=production["identity"], repo_id=TARGET_REPO,
                    revision=published["revision"], manifest_sha256=published["manifest_sha256"])
    return dict(schema="mistral-published-spark-import-v1", identity=identity, root=str(root),
        export=str(export.resolve()), local_snapshot=str(snapshot.resolve()),
        revision=published["revision"], repo_name=repo_name, files=files, assignments=assignments,
        stage_name="mistral-published-import-" + published["revision"][:12] + "-" + published["manifest_sha256"][:12])


def _check_plan(root, plan, production, published):
    expected = dict(producer=production["identity"], repo_id=TARGET_REPO,
                    revision=published["revision"], manifest_sha256=published["manifest_sha256"])
    if (plan.get("identity") != expected or plan.get("root") != str(root)
            or plan.get("export") != str(Path(production["export"]).resolve())
            or plan.get("local_snapshot") != str(Path(published["local_snapshot"]).resolve())
            or len(plan.get("files", {})) != published.get("files")
            or _manifest_digest(plan["files"]) != published["manifest_sha256"]
            or plan.get("revision") != published["revision"]
            or plan.get("repo_name") != "models--" + TARGET_REPO.replace("/", "--")
            or plan.get("stage_name") != "mistral-published-import-" + published["revision"][:12] + "-" + published["manifest_sha256"][:12]
            or plan.get("assignments") != balance(plan["files"])):
        raise ValueError("archive recovery plan differs from the published artifact")
    for name in plan["files"]:
        _relative(name)


def _verify_sources(plan):
    for name, row in plan["files"].items():
        for root in (plan["export"], plan["local_snapshot"]):
            path = Path(root) / name
            if path.is_symlink() or file_identity(path) != row["file_identity"]:
                raise ValueError("published source changed during archive recovery: " + name)


def _transfer(host, plan, directory):
    files = _host_files(plan, host)
    listing = directory / (host + ".files0")
    listing.write_bytes(b"\0".join(name.encode() for name in sorted(files)) + b"\0")
    stage_repo = Path.home() / ".cache/cuteafd/builds" / plan["stage_name"] / "hub" / plan["repo_name"]
    destination = stage_repo / "snapshots" / plan["revision"]
    partials = stage_repo.parents[1] / "rsync-partials"
    _run(["rsync", "--archive", "--no-owner", "--no-group", "--sparse", "--fsync", "--partial",
          "--partial-dir=" + str(partials), "--protect-args", "--timeout=300", "--from0",
          "--files-from=" + str(listing), "-e", shlex.join(list(SSH)),
          plan["local_snapshot"] + "/", host + ":" + str(destination) + "/"], timeout=21600)
    return stage_repo


def _import_host(host, plan, directory):
    receipt_path = directory / (host + ".json")
    state = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if state and state.get("identity") != plan["identity"]:
        raise ValueError("host import receipt belongs to a different publication")
    if state.get("import_complete") is True:
        return state
    prepared = _remote(host, "prepare", plan)
    if prepared.get("same_filesystem") is not True:
        raise ValueError("remote staging filesystem is not admitted")
    stage_repo = _transfer(host, plan, directory)
    proof = _remote(host, "verify", plan)
    expected_files = _host_files(plan, host)
    if (proof.get("verified") is not True or proof.get("identity") != plan["identity"]
            or proof.get("files") != len(expected_files)
            or proof.get("bytes") != sum(row["size"] for row in expected_files.values())):
        raise ValueError("remote staged checksum proof differs")
    for attempt in range(2):
        if not state.get("job"):
            started = json.loads(_ssh(host, ["nest", "--json", "hf", "import", stage_repo,
                                            "--offline", "--no-copy"]))
            if type(started.get("job")) is not int or started["job"] <= 0:
                raise ValueError("native HF import did not return a job")
            state.update(identity=plan["identity"], host=host, stage_repo=str(stage_repo),
                staged_checksums=proof, files={name: {key: row[key] for key in ("size", "sha256")}
                    for name, row in expected_files.items()}, job=started["job"])
            atomic_json(receipt_path, state)
        try:
            result = json.loads(_ssh(host, ["nest", "--json", "jobs", str(state["job"]), "--wait"], timeout=21600))
            break
        except subprocess.CalledProcessError as error:
            if attempt or "no job" not in (error.stderr or "").lower():
                raise
            # Daemon restarts lose their in-memory jobs, not committed files.
            state["job"] = None
            atomic_json(receipt_path, state)
    job, counters = result.get("job", {}), result.get("import", {})
    if (job.get("finished") is not True or job.get("error") is not None
            or job.get("cancelled") is True or counters.get("finished") is not True
            or any(": failed" in note for note in job.get("notes", []))
            or counters.get("errors") != [] or counters.get("copied_bytes") != 0):
        state.setdefault("failed_jobs", []).append(result)
        state["job"] = None
        atomic_json(receipt_path, state)
        raise ValueError("native HF import was incomplete, unsafe, or copied weight bytes")
    state.update(import_complete=True, native_result=result)
    atomic_json(receipt_path, state)
    return state


def _verify_managed(plan):
    snapshot = MANAGED_HUB / plan["repo_name"] / "snapshots" / plan["revision"]
    actual = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
    if actual != set(plan["files"]):
        raise ValueError("managed published snapshot is incomplete or has unexpected files")
    listing = json.loads(_run(["nest", "--json", "ls", snapshot]))
    rows = {row["name"]: row for row in listing["entries"]}
    for name, host in plan["assignments"].items():
        row = rows.get(name, {})
        if row.get("kind") == "symlink":
            resolved = (snapshot / name).resolve(strict=True)
            if not resolved.is_relative_to(MANAGED_HUB):
                raise ValueError("managed weight link resolves outside the managed hub")
            linked = json.loads(_run(["nest", "--json", "ls", resolved])).get("entries", [])
            if len(linked) != 1:
                raise ValueError("managed weight link does not resolve to one object")
            row = linked[0]
        if (row.get("hosts") != [host] or row.get("sealed") is not True
                or row.get("writing_on") is not None or row.get("size") != plan["files"][name]["size"]):
            raise ValueError("managed weight does not have exactly its assigned sealed Spark copy: " + name)
    def verify(name):
        path, row = snapshot / name, plan["files"][name]
        if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            raise ValueError("managed published checksum differs: " + name)
        return row["size"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        total = sum(pool.map(verify, sorted(plan["files"])))
    return dict(managed_snapshot=str(snapshot), files=len(plan["files"]), bytes=total,
                weight_shards=len(plan["assignments"]), distribution={host:
                    sum(plan["files"][name]["size"] for name, owner in plan["assignments"].items() if owner == host)
                    for host in HOSTS})


def _retire(root, plan):
    path = root / "archive-retirement.json"
    state = json.loads(path.read_text()) if path.exists() else dict(identity=plan["identity"], remote=[])
    if state.get("identity") != plan["identity"]:
        raise ValueError("archive retirement identity differs")
    for host in HOSTS:
        if host in state["remote"]:
            continue
        if _remote(host, "retire", plan).get("retired") is not True:
            raise ValueError("remote staging retirement was not confirmed")
        state["remote"].append(host)
        atomic_json(path, state)
    for name in sorted(plan["assignments"]):
        for base in (plan["export"], plan["local_snapshot"]):
            source = Path(base) / name
            if source.is_symlink():
                raise ValueError("refuse symlink local weight retirement")
            if source.exists():
                if file_identity(source) != plan["files"][name]["file_identity"]:
                    raise ValueError("local weight changed before retirement")
                source.unlink()
    state["local_weights_retired"] = True
    atomic_json(path, state)


def archive(root):
    """Resume distribution/import, verify every published byte, then retire staging."""
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
        plan_path = root / "archive-plan.json"
        complete_path = root / "archive-complete.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            _check_plan(root, plan, production, published)
        else:
            if complete_path.exists():
                raise ValueError("archive completion is missing its immutable distribution plan")
            plan = _new_plan(root, production, published)
            atomic_json(plan_path, plan)
        complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
        if complete is not None and complete.get("identity") != plan["identity"]:
            raise ValueError("archive completion belongs to a different publication")
        if complete is None:
            _verify_sources(plan)
            directory = root / "archive-hosts"
            directory.mkdir(exist_ok=True)
            for host in HOSTS:
                _import_host(host, plan, directory)
        proof = _verify_managed(plan)
        if complete is None:
            complete = dict(status="complete", identity=plan["identity"], **proof,
                            time=datetime.now(timezone.utc).isoformat())
            atomic_json(complete_path, complete)
        _retire(root, plan)
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(archive(args.root)), flush=True)


if __name__ == "__main__":
    main()
