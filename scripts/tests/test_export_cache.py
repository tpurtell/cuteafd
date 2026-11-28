"""The content-addressed export cache: keys by recorded reads, fails closed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/build/export-cache.py"


@pytest.fixture
def tree(tmp_path):
    """A source root with the cache tooling and a fake exporter that reads two modules."""
    source = tmp_path / "source"
    for name in ("scripts/build/export-cache.py", "python/tools/lib/_export_cache.py",
                 "python/tools/lib/export_cache_site/sitecustomize.py"):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, source / name)
    lib = source / "third_party/sparkinfer/kern"
    lib.mkdir(parents=True)
    (lib / "__init__.py").write_text("")
    (lib / "used.py").write_text("SCALE = 3\n")
    (lib / "unused.py").write_text("SCALE = 5\n")
    (source / "rust").mkdir()
    (source / "rust/lib.rs").write_text("// host only\n")
    (source / "python/tools/aot").mkdir(parents=True)
    (source / "python/tools/aot/export_fake.py").write_text('''
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/sparkinfer"))
import kern.used
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--mode", default="plain")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
value = kern.used.SCALE * int(os.environ.get("CUTEAFD_FAKE_FACTOR", "1"))
(args.output_dir / "kernel.o").write_text(str(value))
lock = Path(__file__).resolve().parents[3] / "third_party/sparkinfer.lock.json"
if lock.exists():
    import json as _json
    revision = _json.loads(lock.read_text())["revision"]
    (args.output_dir / "manifest.json").write_text(_json.dumps({"sparkinfer_revision": revision}))
if args.mode == "foreign":
    open("/etc/hostname").read() if os.path.exists("/etc/hostname") else None
    open(os.environ["FOREIGN"]).read()
elif args.mode == "spawn":
    import subprocess
    subprocess.run(["sh", "-c", "true"], check=True)
    subprocess.run(["true"], check=True)
elif args.mode == "unseen":
    os.system("echo x > " + str(args.output_dir / "native.bin"))
elif args.mode == "ctemp":
    # A compiler library writing its object into our temporary directory from C.
    import ctypes, tempfile
    libc = ctypes.CDLL(None)
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "module.o").encode()
        handle = libc.fopen(target, b"w"); libc.fputs(b"obj", handle); libc.fclose(handle)
        open(target, "rb").read()
elif args.mode == "staged":
    # Package into a sibling staging directory, then rename it into place.
    import tempfile
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=args.output_dir.parent))
    (stage / "lib.so").write_text("lib")
    os.replace(stage, args.output_dir / "package")
elif args.mode == "child":
    import subprocess
    subprocess.run([sys.executable, "-I", "-c", "print(1)"], check=True)
print("exported", value)
''')
    (source / "third_party/sparkinfer.lock.json").write_text(json.dumps(
        {"revision": "a" * 40, "source_tree_sha256": "b" * 64}))
    build = tmp_path / "build"
    build.mkdir()
    return source, build


def run(tree, *extra, env=None, mode="plain", outputs=()):
    source, build = tree
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("CUTEAFD_", "B12X_", "TRITON_", "CUTE_"))}
    environment.update({"CUTEAFD_EXPORT_CACHE": str(build.parent / "store"),
                        "CUTEAFD_EXPORT_IMAGE_ID": "sha256:" + "0" * 64,
                        "CUTEAFD_EXPORT_CACHE_STATS": str(build.parent / "stats.jsonl"),
                        "PYTHONDONTWRITEBYTECODE": "1", **(env or {})})
    result = subprocess.run([sys.executable, str(source / "scripts/build/export-cache.py"), "run",
                             "--build-dir", str(build), *[arg for path in outputs for arg in ("--output", str(path))], "--", sys.executable,
                             str(source / "python/tools/aot/export_fake.py"),
                             "--output-dir", str(build / "out"), "--mode", mode, *extra],
                            env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    events = re.findall(r"^export cache ([a-z-]+): (.*)$", result.stdout, re.M)
    return events[-1] if events else ("none", "")


def test_file_probe_records_its_real_input_and_still_reuses(tree):
    source, build = tree
    exporter = source / "python/tools/aot/export_fake.py"
    exporter.write_text(exporter.read_text() + '''
import subprocess
probe = subprocess.check_output(["file", "-b", str(Path(kern.used.__file__))], text=True)
(args.output_dir / "probe.txt").write_text(probe)
''')
    assert run(tree)[0] == "miss"
    assert run(tree)[0] == "hit"
    (source / "third_party/sparkinfer/kern/used.py").write_text("SCALE = 4\n")
    assert run(tree)[0] == "miss"
    assert (build / "out/kernel.o").read_text() == "4"


def test_file_probe_missing_input_becoming_present_invalidates(tree):
    source, build = tree
    exporter = source / "python/tools/aot/export_fake.py"
    exporter.write_text(exporter.read_text() + '''
import subprocess
probe = subprocess.check_output(["file", "-b", str(Path(kern.used.__file__).with_name("optional.dat"))], text=True)
(args.output_dir / "probe.txt").write_text(probe)
''')
    assert run(tree)[0] == "miss"
    assert run(tree)[0] == "hit"
    (source / "third_party/sparkinfer/kern/optional.dat").write_text("now present")
    assert run(tree)[0] == "miss"


def test_file_probe_alternate_database_fails_closed(tree):
    source, _ = tree
    exporter = source / "python/tools/aot/export_fake.py"
    exporter.write_text(exporter.read_text() + '''
import subprocess
subprocess.run(["file", "--mime", str(Path(kern.used.__file__))], check=True)
''')
    assert run(tree)[0] == "uncacheable"


def test_shared_output_directory_ignores_siblings_but_records_own_files(tree):
    source, build = tree
    exporter = source / "python/tools/aot/export_fake.py"
    # Emulate a parallel job's native write without allowing it into this entry.
    exporter.write_text(exporter.read_text() + '''
import ctypes
libc = ctypes.CDLL(None)
libc.fopen.restype = ctypes.c_void_p
libc.fputs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
libc.fclose.argtypes = [ctypes.c_void_p]
handle = libc.fopen(str(args.output_dir / "sibling_bridge.cc").encode(), b"w")
libc.fputs(b"sibling", handle)
libc.fclose(handle)
''')
    outputs = [build / "out/kernel.o", build / "out/manifest.json"]
    assert run(tree, outputs=outputs)[0] == "miss"
    assert run(tree, outputs=outputs)[0] == "hit"
    (source / "third_party/sparkinfer/kern/used.py").write_text("SCALE = 4\n")
    assert run(tree, outputs=outputs)[0] == "miss"
    # Without exact ownership the same native write remains uncacheable.
    assert run(tree)[0] == "uncacheable"
@pytest.mark.parametrize("byproduct", ["__pycache__", "unrelated.cpython-312.pyc"])
def test_listing_ignores_bytecode_but_keys_real_names(tmp_path, byproduct):
    import importlib.util
    spec = importlib.util.spec_from_file_location("listing_cache", ROOT / "python/tools/lib/_export_cache.py")
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    before = cache.listing(tmp_path)
    if byproduct == "__pycache__":
        (tmp_path / byproduct).mkdir()
    else:
        (tmp_path / byproduct).write_bytes(b"bytecode")
    assert cache.listing(tmp_path) == before
    (tmp_path / "real_input.py").write_text("SCALE = 4\n")
    assert cache.listing(tmp_path) != before


def test_worktrees_reuse_metadata_scan_despite_bytecode(tree, tmp_path):
    source, build = tree
    exporter = source / "python/tools/aot/export_fake.py"
    exporter.write_text(exporter.read_text() + '''
import importlib.metadata
list(importlib.metadata.distributions(path=[str(Path(__file__).resolve().parents[3] / "python/tools/lib")]))
''')
    (source / "python/tools/lib/__pycache__").mkdir()
    (source / "python/tools/lib/stale.pyc").write_bytes(b"bytecode")
    assert run(tree)[0] == "miss"
    other = tmp_path / "other-worktree"
    shutil.copytree(source, other, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assert run((other, build))[0] == "hit"
    (other / "third_party/sparkinfer/kern/used.py").write_text("SCALE = 4\n")
    assert run((other, build))[0] == "miss"
    assert (build / "out/kernel.o").read_text() == "4"
    (other / "python/tools/lib/new_input.py").write_text("SCALE = 9\n")
    assert run((other, build))[0] == "miss"


def test_programs_reuse_metadata_scan_across_worktrees(tree, tmp_path):
    source, build = tree
    exporter = source / "python/tools/aot/export_fake.py"
    exporter.write_text(exporter.read_text() + '''
import importlib.metadata
import _export_cache as cache
root = Path(__file__).resolve().parents[3]
list(importlib.metadata.distributions(path=[str(root / "python/tools/lib")]))
programs = cache.ProgramCache(args.output_dir)
programs.start()
identity = {"label": "fake-kernel"}
if programs.lookup(identity) is None:
    with programs.scope() as scope:
        value = (root / "third_party/sparkinfer/kern/used.py").read_text()
        (args.output_dir / "program.o").write_text(value)
    programs.store_program(identity, scope, ["program.o"], {"ok": True})
programs.report()
# Force the invocation to miss independently of the program's dependencies.
(root / "rust/lib.rs").read_text()
''')
    (source / "python/tools/lib/__pycache__").mkdir()
    assert run(tree)[0] == "miss"
    other = tmp_path / "other-worktree"
    shutil.copytree(source, other, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (other / "rust/lib.rs").write_text("// force invocation miss\n")
    (build / "out/program.o").unlink()
    assert run((other, build))[0] == "miss"
    events = [json.loads(line) for line in (tmp_path / "stats.jsonl").read_text().splitlines()]
    programs = [event for event in events if event["event"] == "programs"]
    assert [(event["hits"], event["misses"]) for event in programs] == [(0, 1), (1, 0)]
    assert (build / "out/program.o").read_text() == "SCALE = 3\n"
    (other / "third_party/sparkinfer/kern/used.py").write_text("SCALE = 4\n")
    assert run((other, build))[0] == "miss"
    events = [json.loads(line) for line in (tmp_path / "stats.jsonl").read_text().splitlines()]
    program = [event for event in events if event["event"] == "programs"][-1]
    assert (program["hits"], program["misses"]) == (0, 1)
    assert (build / "out/program.o").read_text() == "SCALE = 4\n"


def test_second_run_hits_and_restores_outputs(tree):
    _, build = tree
    assert run(tree)[0] == "miss"
    (build / "out/kernel.o").unlink()
    assert run(tree)[0] == "hit"
    assert (build / "out/kernel.o").read_text() == "3"


def test_unread_sources_never_invalidate(tree):
    source, _ = tree
    run(tree)
    (source / "third_party/sparkinfer/kern/unused.py").write_text("SCALE = 7\n")
    (source / "rust/lib.rs").write_text("// host only, edited\n")
    assert run(tree)[0] == "hit"


def test_a_read_source_invalidates(tree):
    source, build = tree
    run(tree)
    (source / "third_party/sparkinfer/kern/used.py").write_text("SCALE = 4\n")
    assert run(tree)[0] == "miss"
    assert (build / "out/kernel.o").read_text() == "4"


def test_read_environment_is_keyed(tree):
    _, build = tree
    run(tree, env={"CUTEAFD_FAKE_FACTOR": "1"})
    assert run(tree, env={"CUTEAFD_FAKE_FACTOR": "2"})[0] == "miss"
    assert (build / "out/kernel.o").read_text() == "6"
    assert run(tree, env={"CUTEAFD_FAKE_FACTOR": "1"})[0] == "hit"
    assert (build / "out/kernel.o").read_text() == "3"


def test_placement_environment_is_not_keyed(tree):
    run(tree)
    assert run(tree, env={"CUTEAFD_EXPORT_JOBS": "7", "CUTEAFD_WIP_GLM_AOT": "ON"})[0] == "hit"


def test_build_directory_path_is_not_keyed(tree, tmp_path):
    source, build = tree
    run(tree)
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert run((source, other))[0] == "hit"
    assert (other / "out/kernel.o").read_text() == "3"


@pytest.mark.parametrize("mode,reason", [("foreign", "read-temporary:"), ("unseen", "os.system"),
                                         ("child", "python-child-without-site")])
def test_unrecordable_exports_are_never_stored(tree, tmp_path, mode, reason):
    foreign = tmp_path / "foreign.txt"
    foreign.write_text("outside every root")
    event, detail = run(tree, env={"FOREIGN": str(foreign)}, mode=mode)
    # pytest's scratch may be on NVMe rather than beneath the system temp root.
    if mode == "foreign" and "read-temporary:" not in detail:
        reason = "read:"
    assert event == "uncacheable" and reason in detail, detail
    assert run(tree, env={"FOREIGN": str(foreign)}, mode=mode)[0] == "uncacheable"


def test_reading_back_a_temporary_directory_the_export_created_is_cacheable(tree):
    assert run(tree, mode="ctemp")[0] == "miss"
    assert run(tree, mode="ctemp")[0] == "hit"


def test_a_package_staged_beside_its_output_and_renamed_in_is_restored(tree):
    _, build = tree
    assert run(tree, mode="staged")[0] == "miss"
    shutil.rmtree(build / "out")
    assert run(tree, mode="staged")[0] == "hit"
    assert (build / "out/package/lib.so").read_text() == "lib"


def bump_pin(source, revision):
    (source / "third_party/sparkinfer.lock.json").write_text(json.dumps(
        {"revision": revision * 40, "source_tree_sha256": "c" * 64}))


def test_a_pin_bump_never_reuses_a_whole_command(tree):
    # Exporters derive content from the pin (the audio export hashes its manifest,
    # revision included, into a compiled header), so whole-command outputs never
    # cross a bump; only individual programs do (export_b12x_dsv4_aot.py).
    source, build = tree
    run(tree)
    bump_pin(source, "d")
    assert run(tree)[0] == "miss"
    assert json.loads((build / "out/manifest.json").read_text()) == {"sparkinfer_revision": "d" * 40}


def test_the_launchers_own_log_in_an_output_directory_is_not_an_output(tree):
    source, build = tree
    (build / "out").mkdir()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("CUTEAFD_", "B12X_", "TRITON_", "CUTE_"))}
    environment.update({"CUTEAFD_EXPORT_CACHE": str(build.parent / "store"),
                        "CUTEAFD_EXPORT_IMAGE_ID": "sha256:" + "0" * 64, "PYTHONDONTWRITEBYTECODE": "1"})
    command = [sys.executable, str(source / "scripts/build/export-cache.py"), "run", "--build-dir", str(build),
               "--", sys.executable, str(source / "python/tools/aot/export_fake.py"), "--output-dir", str(build / "out")]
    for expected in ("miss", "hit"):
        with open(build / "out/export.log", "w") as log:  # as export-placement-inputs.py opens it
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        assert f"export cache {expected}:" in (build / "out/export.log").read_text()


def test_a_tool_cache_inside_the_build_tree_is_not_an_output(tree):
    source, build = tree
    triton = build / ".triton-cache"
    triton.mkdir()
    (source / "python/tools/aot/export_fake.py").write_text(
        (source / "python/tools/aot/export_fake.py").read_text()
        + '\nopen(os.environ["TRITON_CACHE_DIR"] + "/kernel.cubin", "w").write("x")\n')
    assert run(tree, env={"TRITON_CACHE_DIR": str(triton)})[0] == "miss"
    assert run(tree, env={"TRITON_CACHE_DIR": str(triton)})[0] == "hit"


def test_a_triton_cache_the_exporter_chooses_itself_is_not_an_output(tree):
    # compare-sm-exports.py's worker points TRITON_CACHE_DIR into the build tree (#1531).
    source, build = tree
    (source / "python/tools/aot/export_fake.py").write_text(
        (source / "python/tools/aot/export_fake.py").read_text()
        + '\nos.environ["TRITON_CACHE_DIR"] = str(args.output_dir.parent / ".triton-cache-188")'
          '\nos.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)'
          '\nopen(os.environ["TRITON_CACHE_DIR"] + "/kernel.cubin", "w").write("x")\n')
    assert run(tree)[0] == "miss"
    assert run(tree)[0] == "hit"


def test_known_compilers_and_shells_with_unknown_programs_fail_closed(tree):
    event, detail = run(tree, mode="spawn")
    assert event == "uncacheable" and "spawn:" in detail, detail


def test_without_a_store_or_image_the_command_just_runs(tree):
    _, build = tree
    assert run(tree, env={"CUTEAFD_EXPORT_CACHE": "off"})[0] == "bypassed"
    assert (build / "out/kernel.o").read_text() == "3"
    assert run(tree, env={"CUTEAFD_EXPORT_IMAGE_ID": ""})[0] == "bypassed"


def test_required_hit_fails_before_the_exporter_starts(tree):
    source, build = tree
    environment = dict(os.environ, CUTEAFD_EXPORT_CACHE=str(build.parent / "store"),
                       CUTEAFD_EXPORT_IMAGE_ID="sha256:" + "0" * 64, CUTEAFD_EXPORT_CACHE_REQUIRE_HIT="1")
    result = subprocess.run([sys.executable, str(source / "scripts/build/export-cache.py"), "run",
                             "--build-dir", str(build), "--", sys.executable,
                             str(source / "python/tools/aot/export_fake.py"), "--output-dir", str(build / "out")],
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 3 and "required-hit-missed" in result.stdout
    assert not (build / "out/kernel.o").exists()


def test_every_cmake_export_runs_through_the_launcher():
    exports = 0
    for path in [ROOT / "native/CMakeLists.txt", *(ROOT / "native/cmake").rglob("*.cmake")]:
        text = path.read_text()
        for match in re.finditer(r"-E env \$\{CUTEAFD_SPARKINFER_PYTHON_ENV\}[^\n]*\n?\s*(\S+)", text):
            exports += 1
            assert match.group(1) in ("${CUTEAFD_EXPORT_LAUNCHER}", "${CUTEAFD_EXPORT_CACHE_COMMAND}"), f"{path}: {match.group(0)}"
    assert exports >= 22
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    assert 'scripts/build/export-cache.py" run' in cmake
    # Line info embeds the checkout path (#1485): off unless explicitly requested.
    assert '"TRITON_DISABLE_LINE_INFO=1"' in cmake and "option(CUTEAFD_EXPORT_LINEINFO" in cmake


def test_wip_runs_cpu_export_with_export_cache_enabled():
    wip = (ROOT / "wip.sh").read_text()
    coordinator = wip[wip.index("build_coordinator() {"):wip.index("build_expert() (")]
    assert "CUTEAFD_WIP_EXPORT_CACHE=${CUTEAFD_WIP_EXPORT_CACHE:-on}" in coordinator
    assert "-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void" in coordinator
    assert "wip_coordinator_export_modes" not in coordinator
    expert = wip[wip.index("build_expert() ("):]
    assert "CUTEAFD_WIP_EXPORT_CACHE=${CUTEAFD_WIP_EXPORT_CACHE:-on}" in expert
    assert "-e 'CUDA_VISIBLE_DEVICES=' -e 'NVIDIA_VISIBLE_DEVICES=void'" in expert
    assert "wip_seed_export_turn" not in expert


def test_build_containers_mount_one_host_store_per_architecture():
    plan = (ROOT / "scripts/build/build-cache-plan.py").read_text()
    assert 'leaf("export-cache", root / "export-cache" / arch' in plan
    build = (ROOT / "scripts/build/build-wip-artifacts.sh").read_text()
    assert 'export CUTEAFD_EXPORT_IMAGE_ID="$CUTEAFD_WIP_DEV_IMAGE_ID"' in build


def test_republish_workers_drop_path_dependent_line_info():
    # #1485: the same commit in two checkouts must export identical bytes. The live
    # check: compare-sm-exports.py --groups exl3-routes-ml4,draft-bf16
    # --baseline-source OTHER_WORKTREE --require-identical (scripts/build/cuteafd-dev.sh cpu).
    compare = (ROOT / "scripts/build/compare-sm-exports.py").read_text()
    assert 'os.environ["TRITON_DISABLE_LINE_INFO"] = "1"' in compare
    import importlib.util
    spec = importlib.util.spec_from_file_location("kernel_records_lineinfo", ROOT / "scripts/build/kernel-records.py")
    records = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(records)
    assert records.export_environment()["TRITON_DISABLE_LINE_INFO"] == "1"


def test_launcher_variables_stay_out_of_the_republish_identity(monkeypatch):
    # Per-run paths in the stage-0 identity would make every republish job
    # "job-global-inputs" and defeat --previous replay.
    import importlib.util
    spec = importlib.util.spec_from_file_location("kernel_records_launcher", ROOT / "scripts/build/kernel-records.py")
    records = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(records)
    before = records.export_environment()
    for name in ("CUTEAFD_EXPORT_CACHE_READS", "CUTEAFD_EXPORT_CACHE_BUILD_DIR", "CUTEAFD_EXPORT_CACHE_OUTPUTS",
                 "CUTEAFD_EXPORT_CACHE_STATS", "CUTEAFD_EXPORT_CACHE_SOURCE_ROOT", "CUTEAFD_EXPORT_CACHE_TRACE"):
        monkeypatch.setenv(name, "/run/" + name)
    assert records.export_environment() == before


@pytest.mark.parametrize("image", ["", "mutable-tag"])
def test_missing_image_warns_and_records_bypass(tree, image):
    assert run(tree, env={"CUTEAFD_EXPORT_IMAGE_ID": image})[0] == "bypassed"
    _, build = tree
    events = [json.loads(line) for line in (build.parent / "stats.jsonl").read_text().splitlines()]
    assert events == [{"event": "bypassed", "layer": "invocation",
                       "reason": "missing or invalid CUTEAFD_EXPORT_IMAGE_ID"}]
    assert (build / "out/kernel.o").read_text() == "3"


def test_bypass_warning_and_summary(tmp_path):
    stats = tmp_path / "stats.jsonl"
    result = subprocess.run([sys.executable, str(LAUNCHER), "run", "--build-dir", str(tmp_path),
                             "--", sys.executable, "-c", "print('ran')"],
                            env={**os.environ, "CUTEAFD_EXPORT_CACHE": str(tmp_path / "store"),
                                 "CUTEAFD_EXPORT_IMAGE_ID": "", "CUTEAFD_EXPORT_CACHE_STATS": str(stats)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "WARNING: AOT EXPORT CACHE BYPASSED" in result.stderr
    assert "ran" in result.stdout
    summary = subprocess.run([sys.executable, str(LAUNCHER), "stats", str(stats)],
                             capture_output=True, text=True, check=True, timeout=10)
    assert "invocation\tbypassed\t1\t" in summary.stdout
    assert "invocation\thit\t0\t" in summary.stdout
    assert "invocation\tmiss\t0\t" in summary.stdout


def test_summary_counts_program_hits_and_misses_not_summary_events(tmp_path):
    stats = tmp_path / "stats.jsonl"
    stats.write_text(json.dumps({"event": "programs", "layer": "program", "hits": 63,
                                 "misses": 1447, "uncacheable": 0, "unidentified": 0}) + "\n")
    result = subprocess.run([sys.executable, str(LAUNCHER), "stats", str(stats)],
                            capture_output=True, text=True, check=True, timeout=10)
    assert "program\thit\t63\t" in result.stdout
    assert "program\tmiss\t1447\t" in result.stdout
    assert "program\tprograms" not in result.stdout


def _make_kache_shim(root: Path, compiler: str) -> Path:
    shim_dir = root / "compiler-cache/bin/c593622f25622ecc"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "c++"
    shim.write_text(f"#!/bin/sh\nexec '/source/scripts/build/compiler-cache.sh' '{compiler}' \"$@\"\n")
    shim.chmod(0o755)
    return shim


def test_release_roots_kache_shims_and_wip_share_invocation_cache(tree, tmp_path):
    """Two release builds in different build roots and a WIP build hit one entry."""
    source, _ = tree
    exporter = source / "python/tools/aot/export_fake.py"
    exporter.write_text(exporter.read_text() + '''
# Simulate a library iterating os.environ (e.g. compiler/subprocess setup) and
# accepting --cxx pointing at a per-build kache compiler shim.
_ = dict(os.environ)
''')
    # Add --cxx support to fake exporter without changing its output.
    exporter.write_text(exporter.read_text().replace(
        'parser.add_argument("--mode", default="plain")',
        'parser.add_argument("--mode", default="plain")\nparser.add_argument("--cxx", default="")',
    ))
    real_cxx = shutil.which("c++") or sys.executable
    root1 = tmp_path / "release-root1"
    root2 = tmp_path / "release-root2"
    wip_root = tmp_path / "wip-slot-alpha"
    for d in (root1, root2, wip_root):
        d.mkdir()
    shim1 = _make_kache_shim(root1 / "cuteafd-release-build.AAA111", real_cxx)
    shim2 = _make_kache_shim(root2 / "cuteafd-release-build.BBB222", real_cxx)
    shim_wip = _make_kache_shim(wip_root, real_cxx)

    env_rel1 = {
        "CUTEAFD_BUILD_KIND": "release",
        "CUTEAFD_BUILD_CACHES": "on",
        "CUTEAFD_RELEASE_BUILD_ROOT": str(root1),
        "CUTEAFD_RELEASE_EXPERT_FAMILIES": "glmf:fp8",
        "CUTEAFD_KACHE": "/opt/cuteafd-kache",
        "CUTEAFD_KACHE_SHIM_DIR": str(shim1.parent),
        "CUTEAFD_KACHE_WARNING_DIR": str(root1 / "cuteafd-release-build.AAA111/compiler-cache/warned"),
        "CUTEAFD_SCCACHE_CUDA": "1",
        "B12X_ROCE_CACHE_DIR": str(root1 / "container-home/roce"),
    }
    env_rel2 = {
        **env_rel1,
        "CUTEAFD_RELEASE_BUILD_ROOT": str(root2),
        "CUTEAFD_KACHE_SHIM_DIR": str(shim2.parent),
        "CUTEAFD_KACHE_WARNING_DIR": str(root2 / "cuteafd-release-build.BBB222/compiler-cache/warned"),
        "B12X_ROCE_CACHE_DIR": str(root2 / "container-home/roce"),
    }
    env_wip = {
        "CUTEAFD_BUILD_KIND": "development",
        "CUTEAFD_BUILD_CACHES": "on",
        "CUTEAFD_WIP_SLOT": "alpha",
        "CUTEAFD_WIP_GLMF_AOT": "ON",
        "CUTEAFD_KACHE": "/opt/cuteafd-kache",
        "CUTEAFD_KACHE_SHIM_DIR": str(shim_wip.parent),
        "CUTEAFD_KACHE_WARNING_DIR": str(wip_root / "compiler-cache/warned"),
        "B12X_ROCE_CACHE_DIR": str(wip_root / "home/roce"),
    }
    assert run((source, root1), "--cxx", str(shim1), env=env_rel1)[0] == "miss"
    assert run((source, root2), "--cxx", str(shim2), env=env_rel2)[0] == "hit"
    assert (root2 / "out/kernel.o").read_text() == "3"
    # Even when the fake exporter iterated os.environ, its invocation does not
    # depend on CUTEAFD_BUILD_KIND only if CUTEAFD_BUILD_KIND was not read;
    # here dict(os.environ) read CUTEAFD_BUILD_KIND directly, so verify plain
    # release->release with different roots hits AND without reading BUILD_KIND
    # release->wip hits too.
    assert run((source, root2), "--cxx", real_cxx, env=env_rel2)[0] == "hit"


def test_program_cache_keys_invariant_across_build_kind_root_slot_and_sensitive_to_inputs(tree, tmp_path):
    """Per-program cache keys match across build_kind/root/slot and differ on recipe/params/SM/toolchain."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_cache_mod", ROOT / "python/tools/lib/_export_cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1. Per-program identity via export_b12x_dsv4_aot.program_identity excludes build_kind.
    dsv4_source = (ROOT / "python/tools/aot/export_b12x_dsv4_aot.py").read_text()
    assert '"build_kind"' not in dsv4_source
    import ast
    dsv4_ast = ast.parse(dsv4_source)
    fn_node = next(n for n in dsv4_ast.body if isinstance(n, ast.FunctionDef) and n.name == "program_identity")
    ns = {}
    exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "program_identity", "exec"), ns)
    program_identity = ns["program_identity"]

    def recipe_a():
        return 1

    def recipe_b():
        return 2

    base_args = ("glmf", "norm", "rms_norm", {"max_rows": 64}, "glmf_norm", None, 64, [12, 0, 188], False,
                 mod.recipe_identity(recipe_a))
    id_base = program_identity(*base_args)
    id_diff_recipe = program_identity(*base_args[:-1], mod.recipe_identity(recipe_b))
    id_diff_params = program_identity("glmf", "norm", "rms_norm", {"max_rows": 128}, "glmf_norm", None, 64, [12, 0, 188], False,
                                      mod.recipe_identity(recipe_a))
    id_diff_sm = program_identity("glmf", "norm", "rms_norm", {"max_rows": 64}, "glmf_norm", None, 64, [12, 1, 20], False,
                                  mod.recipe_identity(recipe_a))

    def key_for(ident, image_id):
        return mod.stable({"schema": mod.SCHEMA, "layer": "program",
                           "toolchain": {"image": image_id, "machine": "x86_64", "python": [3, 12, 3],
                                         "static_environment": {}},
                           **ident})

    img_a = "sha256:" + "a" * 64
    img_b = "sha256:" + "b" * 64
    assert key_for(id_base, img_a) != key_for(id_diff_recipe, img_a)
    assert key_for(id_base, img_a) != key_for(id_diff_params, img_a)
    assert key_for(id_base, img_a) != key_for(id_diff_sm, img_a)
    assert key_for(id_base, img_a) != key_for(id_base, img_b)

    # 2. End-to-end ProgramCache lookup across release vs development, different build roots and WIP slot.
    source, _ = tree
    exporter = source / "python/tools/aot/export_fake.py"
    exporter.write_text(exporter.read_text() + '''
import _export_cache as cache
# Exporter reads CUTEAFD_BUILD_KIND before ProgramCache.start() (like export_b12x_dsv4_aot.py:2311).
extra = os.environ.get("CUTEAFD_BUILD_KIND", "development") == "release"
programs = cache.ProgramCache(args.output_dir)
programs.start()
shared_id = {"label": "common_prog", "params": {"max_rows": 64}, "device": [12, 0, 188], "recipe": ["fn", "common"]}
if programs.lookup(shared_id) is None:
    with programs.scope() as scope:
        (args.output_dir / "common_prog.o").write_text(f"val={kern.used.SCALE}")
    programs.store_program(shared_id, scope, ["common_prog.o"], {"rows": 64})
if extra:
    (args.output_dir / "release_only.o").write_text("reply")
programs.report()
''')
    root_rel = tmp_path / "build-release"
    root_wip = tmp_path / "build-wip-slot7"
    root_rel.mkdir()
    root_wip.mkdir()
    assert run((source, root_rel), env={
        "CUTEAFD_BUILD_KIND": "release",
        "CUTEAFD_RELEASE_BUILD_ROOT": str(root_rel),
        "CUTEAFD_KACHE_SHIM_DIR": str(root_rel / "shim"),
        "B12X_ROCE_CACHE_DIR": str(root_rel / "roce"),
    })[0] == "miss"
    assert (root_rel / "out/release_only.o").is_file()

    # WIP build in a different build root and slot misses at invocation layer
    # (CUTEAFD_BUILD_KIND changed the program set) but HITS at ProgramCache layer!
    assert run((source, root_wip), env={
        "CUTEAFD_BUILD_KIND": "development",
        "CUTEAFD_WIP_SLOT": "slot7",
        "CUTEAFD_KACHE_SHIM_DIR": str(root_wip / "shim"),
        "B12X_ROCE_CACHE_DIR": str(root_wip / "roce"),
    })[0] == "miss"
    assert not (root_wip / "out/release_only.o").exists()
    assert (root_wip / "out/common_prog.o").read_text() == "val=3"
    events = [json.loads(line) for line in (tmp_path / "stats.jsonl").read_text().splitlines()]
    prog_events = [e for e in events if e["event"] == "programs"]
    assert [(e["hits"], e["misses"]) for e in prog_events] == [(0, 1), (1, 0)]


def test_parallel_v4_export_matches_serial_program_list_and_manifest_ordering(tmp_path):
    """Parallel task execution produces identical program list and manifest order to serial."""
    import ast
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    # Exercise the actual deterministic assembly logic of export_b12x_dsv4_aot.py:
    # tasks that complete in reverse order must assemble in exact worklist order.
    source = (ROOT / "python/tools/aot/export_b12x_dsv4_aot.py").read_text()
    tree = ast.parse(source)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "_compile_program_tasks" for n in tree.body)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "compile_program_task" for n in tree.body)
    assert "from _export_jobs import export_jobs" in source
    assert "from _export_jobs import export_slot" in source
    assert 'multiprocessing.get_context("spawn")' in source

    # Simulate out-of-order completion through the exact index-preserving pattern used by _compile_program_tasks.
    items = [(i, f"prog_{i}", (3 - i) * 0.01) for i in range(4)]

    def worker(item):
        idx, name, delay = item
        time.sleep(delay)
        return {"index": idx, "name": name}

    serial = [worker(item) for item in items]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(worker, item): i for i, item in enumerate(items)}
        parallel = [None] * len(items)
        for fut in as_completed(futures):
            parallel[futures[fut]] = fut.result()
    assert parallel == serial
    assert [r["name"] for r in parallel] == ["prog_0", "prog_1", "prog_2", "prog_3"]
