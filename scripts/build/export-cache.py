#!/usr/bin/env python3
"""Run one AOT export command through the shared, content-addressed export cache.

    export-cache.py run --build-dir DIR -- PYTHON EXPORTER ARGS...

CMake launches every export custom command through this (CUTEAFD_EXPORT_LAUNCHER,
native/CMakeLists.txt). With CUTEAFD_EXPORT_CACHE unset or "off", or without a
compiler image ID, the command just runs. Otherwise the key is the command line
with DIR and the source root replaced by placeholders, the compiler image and
the recorded observations of every Python process the export started: file
contents, directory listings, import probes, stats and environment reads
(python/tools/lib/_export_cache.py). A hit rewrites exactly the files the
recorded export wrote and skips it. A miss runs the export under the recorder
and stores its writes when nothing it did escaped the record.

    export-cache.py stats FILE      summarise a CUTEAFD_EXPORT_CACHE_STATS log
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/tools/lib"))
import _export_cache as cache  # noqa: E402

BOOTSTRAP = ROOT / "python/tools/lib/export_cache_site"


def normalized(argument: str, roots: cache.Roots) -> str:
    for path, label in ((roots.build, "<build>"), (roots.sparkinfer, "<sparkinfer>"), (roots.source, "<source>")):
        if path is not None:
            argument = argument.replace(str(path), label)
    return argument


def output_roots(command, build: Path) -> list[Path]:
    """Directories the command names under the build directory (its declared outputs).

    Parallel Ninja jobs write elsewhere in the build tree at the same time, so
    only these are compared before and after the export.
    """
    roots = set()
    values = []
    for argument in command:
        values.append(argument.partition("=")[2] if argument.startswith("--") and "=" in argument else argument)
        if argument.startswith("{"):
            try:  # compare-sm-exports.py --_worker PAYLOAD (placement-input republish)
                values.append(str(json.loads(argument)["output"]))
            except (ValueError, KeyError, TypeError):
                pass
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            continue
        resolved = Path(os.path.realpath(path))
        if resolved == build or build not in resolved.parents:
            continue
        roots.add(resolved if not resolved.suffix else resolved.parent)
    return sorted(roots)


def own_streams() -> set:
    """(device, inode) of files the launcher's own stdout/stderr write: never outputs."""
    streams = set()
    for fd in (1, 2):
        try:
            info = os.fstat(fd)
        except OSError:
            continue
        streams.add((info.st_dev, info.st_ino))
    return streams


def tree_state(roots) -> dict:
    """(size, mtime_ns, inode) of every file under each root."""
    state = {}
    for root in roots:
        state.update(_tree_state(root))
    return state


def _tree_state(root: Path) -> dict:
    if root.is_file():
        info = root.lstat()
        return {str(root): (info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev)}
    state = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = os.path.join(directory, name)
            try:
                info = os.lstat(path)
            except FileNotFoundError:
                continue
            state[path] = (info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev)
    return state


def run(args) -> int:
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("export-cache.py run: no command")
    store = cache.store_from_environment()
    toolchain = cache.toolchain_identity() if store else None
    if store is None or toolchain is None:
        reason = "cache disabled" if store is None else "missing or invalid CUTEAFD_EXPORT_IMAGE_ID"
        cache.stats("bypassed", layer="invocation", reason=reason)
        if store is not None:
            print(f"WARNING: AOT EXPORT CACHE BYPASSED: {reason}; mounted store will not be used",
                  file=sys.stderr, flush=True)
        os.execvp(command[0], command)
    roots = cache.Roots(ROOT, Path(args.build_dir))
    identity = {"schema": cache.SCHEMA, "layer": "invocation", "toolchain": toolchain,
                "outputs": [normalized(path, roots) for path in args.output],
                "argv": [normalized(argument, roots) for argument in command]}
    identity_key = cache.stable(identity)
    label = Path(next((part for part in command if part.endswith(".py")), command[0])).name
    started = time.monotonic()
    found = store.lookup(identity_key, roots)
    if found is not None:
        entry, _ = found
        if restore(store, entry, roots):
            cache.stats("hit", layer="invocation", exporter=label, files=len(entry["files"]),
                        seconds=round(time.monotonic() - started, 3))
            return 0
    if os.environ.get("CUTEAFD_EXPORT_CACHE_REQUIRE_HIT") == "1":
        cache.stats("required-hit-missed", layer="invocation", exporter=label)
        return 3
    watched = ([Path(os.path.realpath(path)) for path in args.output]
               if args.output else output_roots(command, roots.build))
    if any(path != roots.build and roots.build not in path.parents for path in watched):
        raise ValueError("declared export output is outside the build directory")
    with tempfile.TemporaryDirectory(prefix="export-reads-") as reads:
        environment = dict(os.environ)
        environment["CUTEAFD_EXPORT_CACHE_OUTPUTS"] = json.dumps([str(path) for path in watched])
        environment.update({
            cache.READS_ENV: reads, cache.ROOT_ENV: str(ROOT), cache.BUILD_ENV: str(roots.build),
            "PYTHONPATH": os.pathsep.join(filter(None, [str(BOOTSTRAP), environment.get("PYTHONPATH")])),
        })
        environment.pop("PYTHONNOUSERSITE", None)
        before = tree_state(watched)
        status = subprocess.call(command, env=environment)
        if status != 0:
            return status
        after = tree_state(watched)
        record = cache.merge_records(Path(reads))
    seconds = round(time.monotonic() - started, 3)
    reason = refusal(record, command)
    if record is not None:
        # Written then removed (a staging directory renamed or cleaned up): not outputs.
        # Directories are created along the way (mkdir also fires for existing ones);
        # only files are outputs, and a restore recreates their parents.
        record["written"] = [path for path in record["written"] if os.path.lexists(path) and not os.path.isdir(path)]
    if not reason and not watched:
        reason = "the command names no output directory under the build directory"
    if not reason and any(not any(Path(path) == root or root in Path(path).parents for root in watched)
                          for path in record["written"]):
        outside = next(path for path in record["written"]
                       if not any(Path(path) == root or root in Path(path).parents for root in watched))
        reason = f"the export wrote outside the output directories it names, e.g. {normalized(outside, roots)}"
    if not reason:
        # Every file the export created or changed must be one it was seen writing:
        # an entry restores exactly those, so an unseen write would be dropped.
        streams = own_streams()
        changed = {path for path, state in after.items() if before.get(path) != state
                   and (state[3], state[2]) not in streams}
        unseen = sorted(changed - {os.path.realpath(path) for path in record["written"]} - set(record["written"]))
        if unseen:
            reason = f"{len(unseen)} unrecorded writes, e.g. {normalized(unseen[0], roots)}"
    if not reason and not cache.still_valid(roots, record["checks"]):
        reason = "a source changed during the export"
    if reason:
        cache.stats("uncacheable", layer="invocation", exporter=label, reason=reason, seconds=seconds)
        return 0
    files = {}
    for written in record["written"]:
        path = Path(written)
        if path.is_file():
            name = roots.name(path)
            files[name] = {"blob": store.put_blob(path), "mode": path.stat().st_mode & 0o777}
    entry = {"files": files, "seconds": seconds, "argv": identity["argv"]}
    # The fork lock stays in a whole command's key: exporters derive content from
    # the pin (manifests, and digests of them compiled into headers), so outputs
    # are never carried across a pin bump. Programs are (ProgramCache).
    store.record(identity_key, record["checks"], entry)
    cache.stats("miss", layer="invocation", exporter=label, checks=len(record["checks"]),
                files=len(files), seconds=seconds)
    return 0


def refusal(record, command) -> str | None:
    if record is None:
        return "a Python process exited without its record"
    if not any(check.startswith("file|src:") for check in record["checks"]):
        return "the command started no recorded Python exporter"
    if record["unknown"]:
        return "unrecorded effects: " + ",".join(record["unknown"][:4])
    if any(not item.startswith("build:") for item in
           (cache.Roots(ROOT).name(path) or "foreign" for path in record["written"])):
        pass
    return None


def restore(store, entry, roots: cache.Roots) -> bool:
    for name, item in entry["files"].items():
        if not store.get_blob(item["blob"], roots.path(name), item["mode"]):
            return False
    return True


def summarise(path: Path) -> int:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = {("invocation", kind): (0, 0.0) for kind in ("hit", "miss", "bypassed")}
    for event in events:
        if event["event"] == "programs":
            for field, kind in (("hits", "hit"), ("misses", "miss"),
                                ("uncacheable", "uncacheable"), ("unidentified", "unidentified")):
                key = ("program", kind)
                count, seconds = rows.get(key, (0, 0.0))
                rows[key] = (count + int(event.get(field, 0)), seconds)
        else:
            key = (event.get("layer"), event["event"])
            count, seconds = rows.get(key, (0, 0.0))
            rows[key] = (count + 1, seconds + float(event.get("seconds", 0)))
    for (layer, kind), (count, seconds) in sorted(rows.items()):
        print(f"{layer}\t{kind}\t{count}\t{seconds:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--build-dir", required=True)
    runner.add_argument("--output", action="append", default=[],
                        help="exact output file or owned subtree (repeatable; excludes parallel sibling jobs)")
    runner.add_argument("command", nargs=argparse.REMAINDER)
    summary = sub.add_parser("stats")
    summary.add_argument("file", type=Path)
    args = parser.parse_args()
    if args.action == "stats":
        return summarise(args.file)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
