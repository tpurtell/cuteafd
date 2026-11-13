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


def test_wip_tries_cached_exports_before_any_export_turn():
    wip = (ROOT / "wip.sh").read_text()
    coordinator = wip[wip.index("build_coordinator() {"):]
    assert coordinator.index("wip_cached_exports docker exec -e CUTEAFD_EXPORT_CACHE_REQUIRE_HIT=1") < \
        coordinator.index("wip_coordinator_export_modes")
    expert = wip[wip.index("build_expert() ("):]
    assert expert.index("CUTEAFD_EXPORT_CACHE_REQUIRE_HIT=1") < expert.index("wip_seed_export_turn\n")


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
