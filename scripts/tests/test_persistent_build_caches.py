"""CPU-only Docker argv and filesystem-admission gates for persistent caches."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "scripts/build/build-cache-plan.py"


def fixture(tmp_path, kind="ext4", device="nvme0n1p2"):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, text in {
        "findmnt": '#!/bin/sh\nprintf \'%s\\n\' \'{"filesystems":[{"source":"/dev/' + device +
                    '","fstype":"' + kind + '","options":"rw"}]}\'\n',
        "lsblk": '#!/bin/sh\nprintf "%s\\n" "' + device + '"\n',
        "docker": '#!/usr/bin/env python3\nimport json,os,sys\n'
                  'with open(os.environ["DOCKER_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                  'if sys.argv[1:2] == ["run"]:\n'
                  '    from pathlib import Path\n'
                  '    mounts=[dict(part.split("=",1) for part in sys.argv[i+1].split(",") if "=" in part) for i,arg in enumerate(sys.argv[:-1]) if arg=="--mount"]\n'
                  '    for mount in mounts:\n'
                  '        source=Path(mount["src"]); assert source.exists(), source\n'
                  '        assert source.stat().st_uid == os.getuid(), source\n'
                  '        for parent in mounts:\n'
                  '            dst=Path(mount["dst"]); home=Path(parent["dst"])\n'
                  '            if dst != home and dst.is_relative_to(home):\n'
                  '                target=Path(parent["src"])/dst.relative_to(home)\n'
                  '                assert target.is_dir(), target\n'
                  '                assert target.stat().st_uid == os.getuid(), target\n'
                  'if sys.argv[1:3] == ["container","inspect"]: sys.exit(1)\n'
                  'if "inspect" in sys.argv: print("image-id")\n',
        "nvidia-smi": '#!/bin/sh\nexit 0\n',
    }.items():
        path = binary / name
        path.write_text(text)
        path.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CUTEAFD_", "CARGO_", "KACHE_", "SCCACHE_"))}
    env.update(HOME=str(tmp_path / "home"), PATH=str(binary) + os.pathsep + env["PATH"],
               DOCKER_LOG=str(tmp_path / "docker.log"))
    return env


def plan(tmp_path, env, arch="x86_64", tc="abcd", mode="prepare"):
    return subprocess.run(["python3", str(PLAN), "--build-root", str(tmp_path / "build"),
                           "--container-home", "/container/home", "--toolchain", tc,
                           "--arch", arch, "--mode", mode], env=env,
                          text=True, capture_output=True)


def pairs(tokens, option):
    return [tokens[i + 1] for i, value in enumerate(tokens[:-1]) if value == option]


def assert_plan(tokens, env, arch="x86_64", home="/container/home"):
    mounts = pairs(tokens, "--mount")
    assert any(mount.endswith(f",dst={home}") for mount in mounts)
    for name in ("registry", "git"):
        assert f"type=bind,src={env['HOME']}/.cache/cuteafd/cargo-home/{arch}/{name},dst={home}/cargo/{name}" in mounts
    for name, var in (("triton", "TRITON_CACHE_DIR"), ("torchinductor", "TORCHINDUCTOR_CACHE_DIR"),
                      ("torch-extensions", "TORCH_EXTENSIONS_DIR"), ("xdg", "XDG_CACHE_HOME"),
                      ("roce", "B12X_ROCE_CACHE_DIR")):
        assert any(f"/jit/abcd/{arch}/{name},dst={home}/{name}" in mount for mount in mounts)
        assert f"{var}={home}/{name}" in pairs(tokens, "-e")
    for name in ("kache", "sccache"):
        assert f"type=bind,src={env['HOME']}/.cache/cuteafd/{name}/{arch},dst=/opt/cuteafd-{name}-cache/{arch}" in mounts


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_architecture_cache_mounts_and_warm_state(tmp_path, arch):
    env = fixture(tmp_path)
    first = plan(tmp_path, env, arch)
    assert first.returncode == 0, first.stderr
    assert_plan(first.stdout.splitlines(), env, arch)
    path = Path(env["HOME"]) / f".cache/cuteafd/jit/abcd/{arch}/triton"
    (path / "cached").write_text("warm")
    second = plan(tmp_path, env, arch)
    assert f"build cache triton: warm {path}" in second.stderr
    assert "cold" in plan(tmp_path, env, arch, tc="bcde").stderr


def test_off_ignores_all_explicit_compiler_cache_settings(tmp_path):
    env = fixture(tmp_path)
    env.update(CUTEAFD_BUILD_CACHES="off", CUTEAFD_KACHE="/missing/tool", CUTEAFD_SCCACHE_CUDA="1",
               CUTEAFD_KACHE_REMOTE="/remote")
    result = plan(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert "/.cache/cuteafd/cargo-home" not in result.stdout
    assert "/.cache/cuteafd/jit" not in result.stdout
    assert "/opt/cuteafd-kache" not in result.stdout
    assert "CUTEAFD_SCCACHE_CUDA=1" not in result.stdout
    assert "CUTEAFD_KACHE_REMOTE=" in result.stdout
    assert "/build/cache/cold-" in result.stdout


@pytest.mark.parametrize("kind,device", [("ext4", "sda1"), ("nfs", "nvme0n1p2"), ("ntfs3", "nvme0n1p2")])
def test_unsafe_or_non_nvme_persistent_path_falls_back(tmp_path, kind, device):
    env = fixture(tmp_path)
    # A symlinked cache root must not bypass admission; its separate mount is unsafe.
    (tmp_path / "bin/findmnt").write_text(
        '#!/usr/bin/env python3\nimport json,sys\n'
        f'kind={kind!r} if "/.cache/cuteafd/" in " ".join(sys.argv) else "ext4"\n'
        f'device={device!r} if "/.cache/cuteafd/" in " ".join(sys.argv) else "nvme0n1p2"\n'
        'print(json.dumps({"filesystems":[{"source":"/dev/"+device,"fstype":kind,"options":"rw"}]}))\n')
    # findmnt checks an existing parent, so create the candidates first.
    for name in ("cargo-home/x86_64/registry", "cargo-home/x86_64/git", "jit/abcd/x86_64",
                 "kache/x86_64", "sccache/x86_64"):
        (Path(env["HOME"]) / ".cache/cuteafd" / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin/lsblk").write_text('#!/bin/sh\nprintf "%s\\n" "$5"\n')
    result = plan(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert "warning:" in result.stderr and "using per-build directory" in result.stderr
    assert f"src={tmp_path}/build/cache/cargo-registry," in result.stdout
    assert "src=" + env["HOME"] + "/.cache/cuteafd/" not in result.stdout


def test_dry_plan_has_no_side_effects(tmp_path):
    env = fixture(tmp_path)
    result = plan(tmp_path, env, mode="dry")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "build").exists()
    assert not Path(env["HOME"]).exists()
    assert "cold" in result.stderr


@pytest.mark.parametrize("entry", ["dev", "release", "wip", "remote-release", "remote-wip"])
@pytest.mark.parametrize("enabled", [True, False])
def test_docker_mounts_reach_each_entrypoint(tmp_path, entry, enabled):
    env = fixture(tmp_path)
    env["CUTEAFD_BUILD_CACHES"] = "on" if enabled else "off"
    env["HF_HOME"] = str(tmp_path / "hf")
    if entry == "dev":
        result = subprocess.run(["bash", str(ROOT / "scripts/build/cuteafd-dev.sh"), "cpu", "--", "true"],
                                env=env, capture_output=True, text=True)
    else:
        text = (ROOT / ("build.sh" if "release" in entry else "wip.sh")).read_text()
        prefix = 'set -euo pipefail\n'
        prefix += f'source "{ROOT}/scripts/build/build-caches.sh"\n'
        prefix += f'source "{ROOT}/scripts/build/compiler-cache.sh"\n'
        prefix += 'cuteafd_build_cache_defaults\n'
        if entry == "release":
            prefix += f'repo_root="{ROOT}"; release_build_root="{tmp_path}/build"; release_source_parent="$release_build_root"; release_toolchain_hash=abcd\n'
            prefix += 'release_die() { exit 2; }; release_build_container_home() { printf /container/home; }\n'
            body = text[text.index('compiler_cache_args=()\nmapfile'):] if 'compiler_cache_args=()\nmapfile' in text else text[text.index('compiler_cache_args=()\ncache_plan='):]
            body = body[:body.index('timeout "$export_timeout"')]
            prefix += body + '\ndocker run "${compiler_cache_args[@]}" image true\n'
        elif entry == "wip":
            prefix += f'coordinator_container=test; COORDINATOR_DOCKER_DEV=image; RELEASE_COORDINATOR_GPU_UUID=uuid; hf_home="{tmp_path}/hf"; wip_mount_root="{tmp_path}/build"; wip_toolchain_hash=abcd; WIP_ROOT="$wip_mount_root"\n'
            prefix += 'release_die() { printf "%s\\n" "$*" >&2; exit 2; }; mkdir -p "$wip_mount_root"\n'
            body = text[text.index('ensure_local_container() {'):text.index('\nensure_remote_container()')]
            prefix += body + '\nensure_local_container\n'
        elif entry == "remote-wip":
            body = text.split('"$wip_toolchain_hash" <<\'REMOTE\'\n', 1)[1].split('\nREMOTE', 1)[0]
            # Execute the real remote heredoc with the staged helper and host-native identity.
            args = ['container', 'image', '1', '__unset__', '__unset__', str(ROOT), '__none__', '__none__',
                    __import__('base64').b64encode((ROOT/'scripts/build/assert-build-filesystem.py').read_bytes()).decode(),
                    '1', 'on' if enabled else 'off', 'abcd']
            result = subprocess.run(['bash', '-s', '--', *args], input=body, env=env, text=True, capture_output=True)
        else:
            body = text.split('"${CUTEAFD_BUILD_CACHES:-on}" <<\'REMOTE\'\n', 1)[1].split('\nREMOTE', 1)[0]
            # Cache setup + exact Docker export argv, omitting GPU admission and supervision.
            begin = body.index('container_home=/tmp/cuteafd-home')
            end = body.index('spark_export_container_pid=$!')
            body = body[begin:end]
            prefix += f'cd "{ROOT}"; release_build_root="{tmp_path}/build"; remote_dir="{tmp_path}/remote"; export_container=test; dev_image=image\n'
            prefix += 'compiler_cache_args=(); native_build_env_args=(); release_build_root_args=(); exl3_paired_tp4=0; spark_tp_roles=; expert_families=; bf16_families=; audio_aot=OFF\n'
            prefix += body + '\nwait\n'
        if entry != "remote-wip":
            result = subprocess.run(['bash', '-c', prefix], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    runs = [json.loads(line) for line in Path(env["DOCKER_LOG"]).read_text().splitlines()]
    argv = next(tokens for tokens in runs if tokens[0] == 'run')
    assert f'{os.getuid()}:{os.getgid()}' in pairs(argv, '--user') or entry == 'release'
    assert 'CUTEAFD_BUILD_CACHES=' + ('on' if enabled else 'off') in pairs(argv, '-e')
    mounts = pairs(argv, '--mount')
    assert bool([m for m in mounts if '/cargo-home/' in m]) == enabled
    assert bool([m for m in mounts if '/jit/' in m]) == enabled
    if entry != 'dev':
        assert bool([m for m in mounts if '/opt/cuteafd-kache-cache/' in m]) == enabled
        assert bool([m for m in mounts if '/opt/cuteafd-sccache-cache/' in m]) == enabled
    if 'wip' in entry:
        assert 'HOME=/wip/home' in pairs(argv, '-e')
        assert any(value.endswith(':/wip') for value in pairs(argv, '-v'))


def test_legacy_root_container_refused_without_mutation(tmp_path):
    env = fixture(tmp_path)
    docker = tmp_path / 'bin/docker'
    docker.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                      'with open(os.environ["DOCKER_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                      'print("" if "{{.Config.User}}" in sys.argv else "image-id")\n')
    text = (ROOT / 'wip.sh').read_text()
    body = text[text.index('ensure_local_container() {'):text.index('\nensure_remote_container()')]
    script = 'set -euo pipefail\ncoordinator_container=legacy; COORDINATOR_DOCKER_DEV=image\n'
    script += 'release_die() { printf "%s\\n" "$*" >&2; exit 2; }\n' + body + '\nensure_local_container\n'
    result = subprocess.run(['bash', '-c', script], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'legacy root WIP container' in result.stderr and 'run.sh --wip still works' in result.stderr
    calls = [json.loads(line) for line in Path(env['DOCKER_LOG']).read_text().splitlines()]
    assert all('inspect' in call for call in calls)


def test_cargo_offline_probe_and_locked_builds(tmp_path):
    env = fixture(tmp_path)
    cargo = tmp_path / 'bin/cargo'
    cargo.write_text('#!/bin/sh\nexit 0\n')
    cargo.chmod(0o755)
    script = f'source "{ROOT}/scripts/build/build-caches.sh"; cuteafd_build_cache_cargo_offline fake/Cargo.toml; printf "%s" "${{CARGO_NET_OFFLINE:-unset}}"'
    result = subprocess.run(['bash', '-c', script], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.endswith('true')
    cargo.write_text('#!/bin/sh\nexit 1\n')
    result = subprocess.run(['bash', '-c', script], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.endswith('unset')
    for name in ('build-release-artifacts.sh', 'build-wip-artifacts.sh'):
        assert 'cargo build \\\n  --locked' in (ROOT/'scripts/build'/name).read_text()


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("enabled", [True, False])
def test_every_planned_directory_and_nested_target_precreated(tmp_path, arch, enabled):
    env = fixture(tmp_path)
    env["CUTEAFD_BUILD_CACHES"] = "on" if enabled else "off"
    result = plan(tmp_path, env, arch)
    assert result.returncode == 0, result.stderr
    mounts = [dict(part.split("=", 1) for part in value.split(",") if "=" in part)
              for value in pairs(result.stdout.splitlines(), "--mount")]
    host_home = tmp_path / "build/container-home"
    for mount in mounts:
        source = Path(mount["src"])
        assert source.is_dir() and source.stat().st_uid == os.getuid()
        destination = Path(mount["dst"])
        if destination.is_relative_to("/container/home"):
            target = host_home / destination.relative_to("/container/home")
            assert target.is_dir() and target.stat().st_uid == os.getuid()


def test_foreign_owned_nested_target_names_path_and_manual_fix(tmp_path):
    import importlib.util
    from unittest.mock import patch

    spec = importlib.util.spec_from_file_location("cache_plan_owner", PLAN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = fixture(tmp_path)
    target = tmp_path / "build/container-home/cargo/git"
    target.mkdir(parents=True)
    real_stat = Path.stat

    def stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == target:
            values = list(result)
            values[4] = os.getuid() + 1
            return os.stat_result(values)
        return result

    with patch.dict(os.environ, env, clear=True), patch.object(Path, "stat", stat), patch.object(
        module, "nvme_cache"
    ), patch.object(module.filesystem, "check_path"), patch.object(
        __import__("sys"), "argv", [str(PLAN), "--build-root", str(tmp_path / "build"),
                                  "--container-home", "/container/home", "--toolchain", "abcd"]
    ):
        with pytest.raises(ValueError) as error:
            module.main()
    assert str(target) in str(error.value)
    assert "fresh WIP instance" in str(error.value)
    assert "agent-sudo chown" in str(error.value)
