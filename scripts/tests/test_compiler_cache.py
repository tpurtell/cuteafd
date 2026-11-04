"""Opt-in cache setup, Docker argument transport, and plain compiler fallback."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/build/compiler-cache.sh"


def run_setup(tmp_path, extra=None, expression='env | sort'):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CUTEAFD_KACHE", "KACHE_", "CUTEAFD_SCCACHE", "SCCACHE_")) and key not in
           ("RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CC", "CXX",
            "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_CUDA_COMPILER_LAUNCHER")}
    env.update(extra or {})
    return subprocess.run(["bash", "-c", f'set -eu; source "{HELPER}"; '
                           'cuteafd_compiler_cache_setup "$1"; ' + expression,
                           "test", str(tmp_path / "build")], env=env,
                          text=True, capture_output=True, check=True)


def fake_kache(tmp_path):
    binary = tmp_path / "kache"
    binary.write_text('#!/bin/sh\nif [ "$1" = --version ]; then printf "kache test\\n"; exit 0; fi\nexit 42\n')
    binary.chmod(0o755)
    return str(binary)


def shim_directory(tmp_path, cc="cc", cxx="c++"):
    compilers = [str(Path(shutil.which(name)).resolve()) for name in (cc, cxx)]
    identity = hashlib.sha256('\n'.join(compilers).encode()).hexdigest()[:16]
    return tmp_path / "build/compiler-cache/bin" / identity


def test_unset_is_noop(tmp_path):
    result = run_setup(tmp_path)
    assert result.stderr == ""
    assert "KACHE_" not in result.stdout
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert not (tmp_path / "build").exists()


def test_missing_executable_warns_and_stays_plain(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": "/nonexistent/kache"})
    assert result.stderr.count("warning:") == 1
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert "CUTEAFD_KACHE_MODE=disabled" in result.stdout


def test_remote_failure_stays_plain(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
                                "CUTEAFD_KACHE_REMOTE": str(blocker)})
    assert result.stderr.count("warning:") == 1
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert "CUTEAFD_KACHE_MODE=disabled" in result.stdout


def test_existing_wrapper_is_preserved(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path), "RUSTC_WRAPPER": "/prior/wrapper"})
    assert "RUSTC_WRAPPER=/prior/wrapper" in result.stdout
    assert result.stderr.count("warning:") == 1


def test_native_launchers_and_wrapper_fallback(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       '"$RUSTC_WRAPPER" /usr/bin/printf "%s\\n" native-success; '
                       '"$RUSTC_WRAPPER" /usr/bin/printf "%s\\n" rust-success; env | sort')
    assert "native-success" in result.stdout and "rust-success" in result.stdout
    assert result.stderr.count("warning:") == 1
    assert f"CC={shim_directory(tmp_path)}/cc" in result.stdout
    assert f"CXX={shim_directory(tmp_path)}/c++" in result.stdout
    for language in ("C", "CXX"):
        assert f"CMAKE_{language}_COMPILER_LAUNCHER={HELPER}" in result.stdout
    assert "CMAKE_CUDA_COMPILER_LAUNCHER=" not in result.stdout
    assert "CUTEAFD_KACHE_MODE=enabled" in result.stdout
    assert not (tmp_path / "cache/target").exists()


def test_unset_preserves_compilers(tmp_path):
    result = run_setup(tmp_path, {"CC": "/custom/cc", "CXX": "/custom/c++"})
    assert "CC=/custom/cc\n" in result.stdout
    assert "CXX=/custom/c++\n" in result.stdout
    assert result.stderr == ""
    assert not (tmp_path / "build").exists()


def test_shims_are_single_executables_with_absolute_compilers(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       '"$CC" --version; "$CXX" --version; printf "%s\\n" "$CC" "$CXX"')
    cc, cxx = result.stdout.splitlines()[-2:]
    assert " " not in cc + cxx
    for shim in (Path(cc), Path(cxx)):
        assert shim.is_file() and os.access(shim, os.X_OK)
        import shlex
        compiler = shlex.split(shim.read_text().splitlines()[1])[2]
        assert Path(compiler).is_absolute() and Path(compiler).is_file()
    assert result.stderr.count("warning:") == 1
    assert "Copyright" in result.stdout


def test_shim_respects_custom_compiler_and_argument_boundaries(tmp_path):
    compiler = tmp_path / "custom-compiler"
    compiler.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    compiler.chmod(0o755)
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
                                "CC": str(compiler)},
                       '"$CC" "argument with spaces" -shared')
    assert result.stdout.splitlines() == ["argument with spaces", "-shared"]
    assert result.stderr.count("warning:") == 1
    assert str(compiler.resolve()) in (shim_directory(tmp_path, str(compiler)) / "cc").read_text()


def test_cmake_configures_and_builds_with_single_path_shims(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.18)\n'
                                          'project(shim_test LANGUAGES C CXX)\n'
                                          'add_library(shim_test SHARED source.c source.cpp)\n')
    (source / "source.c").write_text('int c_value(void) { return 1; }\n')
    (source / "source.cpp").write_text('int cpp_value() { return 2; }\n')
    native = tmp_path / "native"
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       f'cmake -S "{source}" -B "{native}"; cmake --build "{native}"; '
                       f'cuteafd_compiler_cache_check_cmake_compilers "{native}"')
    assert "Built target shim_test" in result.stdout
    cache = (native / "CMakeCache.txt").read_text()
    for language, name in (("C", "cc"), ("CXX", "c++")):
        assert f'CMAKE_{language}_COMPILER:FILEPATH={shim_directory(tmp_path)}/{name}\n' in cache
        assert f'CMAKE_{language}_COMPILER_ARG1:STRING=' not in cache


def test_changed_toolchain_gets_new_shims_and_requires_fresh_configure(tmp_path):
    extra = {"CUTEAFD_KACHE": fake_kache(tmp_path),
             "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")}
    original = run_setup(tmp_path, extra, 'printf "%s\\n" "$CC"').stdout.strip()
    native = tmp_path / "native"
    native.mkdir()
    (native / "CMakeCache.txt").write_text(f'CMAKE_C_COMPILER:FILEPATH={original}\n')
    compiler = tmp_path / "other-compiler"
    compiler.write_text('#!/bin/sh\nexec /usr/bin/cc "$@"\n')
    compiler.chmod(0o755)
    changed = run_setup(tmp_path, {**extra, "CC": str(compiler)},
                        f'printf "%s\\n" "$CC"; '
                        f'if cuteafd_compiler_cache_check_cmake_compilers "{native}"; then exit 99; fi')
    assert changed.stdout.strip() != original
    assert "rerun with fresh configure" in changed.stderr
    assert str(compiler) not in Path(original).read_text()


def test_cmake_launcher_wraps_real_compiler_only_once(tmp_path):
    binary = tmp_path / "kache"
    log = tmp_path / "invocations"
    binary.write_text('#!/bin/sh\nif [ "$1" = --version ]; then exit 0; fi\n'
                      f'printf "%s\\n" "$1" >> "{log}"\nexec "$@"\n')
    binary.chmod(0o755)
    run_setup(tmp_path, {"CUTEAFD_KACHE": str(binary),
                        "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
              '"$CMAKE_C_COMPILER_LAUNCHER" "$CC" --version; '
              '"$CMAKE_CXX_COMPILER_LAUNCHER" "$CXX" --version')
    assert log.read_text().splitlines() == [str(Path(shutil.which(name)).resolve())
                                           for name in ("cc", "c++")]


def test_shim_setup_failure_preserves_compilers(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
                                "CC": "/missing/compiler", "CXX": "c++"})
    assert "CC=/missing/compiler\n" in result.stdout
    assert "CXX=c++\n" in result.stdout
    assert "CUTEAFD_KACHE_MODE=disabled" in result.stdout
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert "cannot create compiler shims" in result.stderr


def test_local_index_rejects_remote_filesystem(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    findmnt = binary / "findmnt"
    findmnt.write_text('#!/bin/sh\nprintf \'{"filesystems":[{"fstype":"nfs4","options":"rw"}]}\\n\'\n')
    findmnt.chmod(0o755)
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
                                "PATH": str(binary) + os.pathsep + os.environ["PATH"]})
    assert result.stderr.count("warning:") == 1
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert not (tmp_path / "cache").exists()


def test_successful_compile_is_not_followed_by_daemon_shutdown(tmp_path):
    binary = tmp_path / "kache"
    binary.write_text('#!/bin/sh\nif [ "$1" = --version ]; then exit 0; fi\n'
                      'if [ "$1" = daemon ]; then exit 124; fi\nexec "$@"\n')
    binary.chmod(0o755)
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": str(binary),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       '"$RUSTC_WRAPPER" /bin/true')
    assert result.stderr == ""
    for name in ("compiler-cache.sh", "build-wip-artifacts.sh", "build-release-artifacts.sh"):
        assert "daemon stop" not in (ROOT / "scripts/build" / name).read_text()
        assert "sync --push" not in (ROOT / "scripts/build" / name).read_text()


def test_compiler_errors_are_not_hidden(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       'if "$RUSTC_WRAPPER" /bin/false; then exit 99; fi')
    # The plain compiler fails too: its error, not a cache fault.
    assert result.stderr.count("warning:") == 0


def test_expected_compile_failure_keeps_cache_enabled(tmp_path):
    """proc-macro2's build probe fails by design on stable rustc; a failure the plain
    compiler reproduces must not disable kache for the rest of the build."""
    binary = tmp_path / "kache"
    binary.write_text('#!/bin/sh\nif [ "$1" = --version ]; then exit 0; fi\nexec "$@"\n')
    binary.chmod(0o755)
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": str(binary),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       'if "$RUSTC_WRAPPER" /bin/false; then exit 99; fi; '
                       'test ! -e "$CUTEAFD_KACHE_WARNING_DIR"; "$RUSTC_WRAPPER" /bin/true')
    assert result.stderr == ""


def test_stale_fallback_record_is_cleared_per_build(tmp_path):
    warned = tmp_path / "build/compiler-cache/warned"
    warned.mkdir(parents=True)
    (warned / "first-failure").write_text("exit 1: old\n")
    run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                         "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")}, 'true')
    assert not warned.exists()


def test_cmake_opt_out_clears_persisted_cache_launcher(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "CMakeCache.txt").write_text(f'CMAKE_C_COMPILER_LAUNCHER:STRING={HELPER}\n')
    result = subprocess.run(["bash", "-c", f'source "{HELPER}"; unset CUTEAFD_KACHE_MODE; '
                             'cuteafd_compiler_cache_cmake_args "$1"', "test", str(native)],
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["-DCMAKE_C_COMPILER_LAUNCHER="]


@pytest.mark.parametrize("cached,current", [
    ("/usr/bin/cc", "shim"),
    ("shim", "/usr/bin/cc"),
    (str(HELPER), "/usr/bin/cc"),
    (str(HELPER), "shim"),
])
def test_cmake_compiler_toggle_requires_fresh_configure(tmp_path, cached, current):
    native = tmp_path / "native"
    native.mkdir()
    shim = tmp_path / "shim"
    shim.write_text("#!/bin/sh\nexec /usr/bin/cc \"$@\"\n")
    shim.chmod(0o755)
    cached = str(shim) if cached == "shim" else cached
    current = str(shim) if current == "shim" else current
    (native / "CMakeCache.txt").write_text(f'CMAKE_C_COMPILER:FILEPATH={cached}\n')
    result = subprocess.run(["bash", "-c", f'set -eu; source "{HELPER}"; '
                             'cuteafd_compiler_cache_check_cmake_compilers "$1"', "test", str(native)],
                            env={**os.environ, "CC": current}, text=True, capture_output=True)
    assert result.returncode == 1
    assert "rerun with fresh configure" in result.stderr
    assert cached in result.stderr and current in result.stderr
    assert result.stdout == ""


def test_cmake_unchanged_compilers_allow_reconfigure(tmp_path):
    import shutil
    native = tmp_path / "native"
    native.mkdir()
    (native / "CMakeCache.txt").write_text(''.join(
        f'CMAKE_{language}_COMPILER:FILEPATH={Path(shutil.which(compiler)).resolve()}\n'
        for language, compiler in (("C", "cc"), ("CXX", "c++"))))
    result = run_setup(tmp_path, {"CC": "cc", "CXX": "c++"},
                       f'cuteafd_compiler_cache_check_cmake_compilers "{native}"')
    assert result.stdout == result.stderr == ""


def test_cmake_cxx_toggle_is_detected_and_build_scripts_check_it(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "CMakeCache.txt").write_text(f'CMAKE_CXX_COMPILER:FILEPATH={HELPER}\n'
                                          'CMAKE_CXX_COMPILER_ARG1:STRING= c++\n')
    result = subprocess.run(["bash", "-c", f'source "{HELPER}"; '
                             'cuteafd_compiler_cache_check_cmake_compilers "$1"', "test", str(native)],
                            env={**os.environ, "CXX": "c++"}, text=True, capture_output=True)
    assert result.returncode == 1 and "CMake CXX compiler changed" in result.stderr
    for name in ("build-wip-artifacts.sh", "build-release-artifacts.sh"):
        text = (ROOT / "scripts/build" / name).read_text()
        assert text.index("cuteafd_compiler_cache_check_cmake_compilers") < text.index("cuteafd_host_isa_cargo_build ")


def test_docker_missing_remote_emits_no_partial_mounts(tmp_path):
    env = {**os.environ, "CUTEAFD_KACHE": fake_kache(tmp_path),
           "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
           "CUTEAFD_KACHE_REMOTE": str(tmp_path / "missing-remote")}
    result = subprocess.run(["bash", "-c", f'set -eu; source "{HELPER}"; cuteafd_compiler_cache_docker_args'],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["-e", "CUTEAFD_KACHE_REQUESTED=1"]
    assert result.stderr.count("warning:") == 1


def test_cpu_dev_shell_has_no_gpu_or_cache_requirement(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUTEAFD_KACHE")}
    env["PATH"] = str(binary) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", str(ROOT / "scripts/build/cuteafd-dev.sh"), "cpu", "--", "true"],
                            env=env, text=True, capture_output=True, check=True)
    assert "--gpus" not in result.stdout
    assert "/opt/cuteafd-kache" not in result.stdout
    assert result.stdout.splitlines()[-2:] == ["cuteafd-coordinator-dev", "true"]


def test_cpu_dev_shell_mounts_opt_in_target_and_cache(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "CUTEAFD_KACHE": fake_kache(tmp_path), "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
           "CUTEAFD_DEV_TARGET_DIR": str(tmp_path / "target")}
    env.pop("CUTEAFD_KACHE_REMOTE", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/build/cuteafd-dev.sh"), "cpu", "--", "true"],
                            env=env, text=True, capture_output=True, check=True)
    assert "--gpus" not in result.stdout
    assert "dst=/opt/cuteafd-target" in result.stdout
    assert "CARGO_TARGET_DIR=/opt/cuteafd-target" in result.stdout
    assert "CUTEAFD_KACHE=/opt/cuteafd-kache" in result.stdout
    assert "cuteafd_compiler_cache_setup" in result.stdout


def test_cpu_dev_shell_mounts_baseline_source_read_only(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    baseline = tmp_path / "baseline"
    (baseline / "third_party").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUTEAFD_KACHE")}
    env.update(PATH=str(binary) + os.pathsep + env["PATH"], CUTEAFD_DEV_BASELINE_SOURCE=str(baseline))
    run = lambda: subprocess.run(["bash", str(ROOT / "scripts/build/cuteafd-dev.sh"), "cpu", "--", "true"],
                                 env=env, text=True, capture_output=True)
    # Not an engine checkout with a pinned SparkInfer tree: refused before docker runs.
    result = run()
    assert result.returncode == 2
    assert "pinned SparkInfer tree" in result.stderr
    assert "cuteafd-coordinator-dev" not in result.stdout
    (baseline / "third_party/sparkinfer.lock.json").write_text("{}")
    result = run()
    assert result.returncode == 0, result.stderr
    real = baseline.resolve()
    assert f"type=bind,src={real},dst={real},readonly" in result.stdout.splitlines()
    assert f"CUTEAFD_DEV_BASELINE_SOURCE={real}" in result.stdout.splitlines()
    env.pop("CUTEAFD_DEV_BASELINE_SOURCE")
    assert str(real) not in run().stdout


def test_provenance_records_fallback_and_hash(tmp_path):
    import hashlib
    import json
    (tmp_path / "BUILD_IDENTITY.json").write_text('{"commit":"abc123","dirty":true}')
    binary = tmp_path / "cuteafd"
    binary.write_bytes(b"test binary")
    warning = tmp_path / "warned"
    warning.mkdir()
    destination = tmp_path / "seal.json"
    subprocess.run(["python3", str(ROOT / "scripts/build/write-compiler-provenance.py"),
                    str(tmp_path), str(binary), str(destination)],
                   env={**os.environ, "CUTEAFD_KACHE_MODE": "enabled", "CUTEAFD_KACHE_WARNING_DIR": str(warning)},
                   check=True)
    seal = json.loads(destination.read_text())
    assert seal["source"] == {"commit": "abc123", "dirty": True}
    assert seal["kache_mode"] == "fallback"
    assert seal["binary_sha256"] == hashlib.sha256(b"test binary").hexdigest()
    assert seal["rustc"] and seal["cargo"]


def test_docker_unset_is_noop(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUTEAFD_KACHE")}
    result = subprocess.run(["bash", "-c", f'source "{HELPER}"; cuteafd_compiler_cache_docker_args'],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout == result.stderr == ""


@pytest.mark.parametrize("kache", [False, True])
def test_sccache_selects_only_cuda(tmp_path, kache):
    binary = tmp_path / "sccache"
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    extra = {"CUTEAFD_SCCACHE_CUDA": "1", "SCCACHE_DIR": str(tmp_path / "cuda-cache"),
             "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    if kache:
        extra.update(CUTEAFD_KACHE=fake_kache(tmp_path), CUTEAFD_KACHE_CACHE_DIR=str(tmp_path / "cache"))
    result = run_setup(tmp_path, extra, 'cuteafd_compiler_cache_cmake_args "$1/native"; env | sort')
    assert f"CMAKE_CUDA_COMPILER_LAUNCHER={binary}" in result.stdout
    assert f"-DCMAKE_CUDA_COMPILER_LAUNCHER={binary}" in result.stdout
    assert (f"RUSTC_WRAPPER={HELPER}" in result.stdout) == kache
    assert (f"CMAKE_C_COMPILER_LAUNCHER={HELPER}" in result.stdout) == kache
    assert result.stderr == ""


def test_image_cache_tools_need_no_host_binary(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": "1", "CUTEAFD_SCCACHE_CUDA": "1",
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "kache-cache"),
                                "CUTEAFD_SCCACHE_CACHE_DIR": str(tmp_path / "sccache-cache")},
                       'cuteafd_compiler_cache_docker_args')
    assert "dst=/opt/cuteafd-kache,readonly" not in result.stdout
    assert "dst=/opt/cuteafd-kache-cache" in result.stdout
    assert "dst=/opt/cuteafd-sccache-cache" in result.stdout
    assert "CUTEAFD_SCCACHE_CUDA=1" in result.stdout


@pytest.mark.parametrize("language", ["C", "CXX", "CUDA"])
def test_managed_cache_preserves_other_launchers(tmp_path, language):
    binary = tmp_path / "sccache"
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    result = run_setup(tmp_path, {"CUTEAFD_SCCACHE_CUDA": "1", "CUTEAFD_KACHE": fake_kache(tmp_path),
                                 "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
                                 f"CMAKE_{language}_COMPILER_LAUNCHER": "/custom/launcher",
                                 "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
                       'cuteafd_compiler_cache_cmake_args "$1/native"')
    assert f"-DCMAKE_{language}_COMPILER_LAUNCHER=/custom/launcher" in result.stdout
