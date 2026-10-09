#!/usr/bin/env python3
"""Admit host-local NVMe caches and render Docker bind/env arguments."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

spec = importlib.util.spec_from_file_location("build_filesystem", Path(__file__).with_name("assert-build-filesystem.py"))
filesystem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filesystem)


def nvme_cache(path: Path) -> None:
    filesystem.check_path(str(path))
    existing = path.resolve()
    while not existing.exists():
        existing = existing.parent
    mount = json.loads(subprocess.check_output(
        ["findmnt", "--json", "--target", str(existing), "--output", "SOURCE,FSTYPE"], text=True
    ))["filesystems"][0]
    if mount["fstype"] not in {"ext4", "xfs", "btrfs", "zfs"}:
        raise ValueError(f"not a local NVMe filesystem: {mount['fstype']}")
    devices = subprocess.check_output(
        ["lsblk", "--noheadings", "--inverse", "--output", "NAME", mount["source"].split("[")[0]], text=True
    )
    if "nvme" not in devices:
        raise ValueError("cache is not backed by NVMe")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--container-home", required=True)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--mode", choices=("prepare", "dry"), default="prepare")
    parser.add_argument("--arch", default=platform.machine())
    args = parser.parse_args()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(args.arch, args.arch)
    if arch not in {"x86_64", "aarch64"}:
        parser.error(f"unsupported build architecture: {arch}")
    if not args.toolchain or any(c not in "0123456789abcdef" for c in args.toolchain):
        parser.error("toolchain must be a hexadecimal hash")
    enabled = os.environ.get("CUTEAFD_BUILD_CACHES", "on") != "off"
    print(f"build caches: {'on' if enabled else 'off (cold per-build inputs, plain compilers)'}", file=sys.stderr)
    home = Path.home()
    root = home / ".cache/cuteafd"
    fallback = Path(args.build_root) / "cache"
    if not enabled:
        fallback /= f"cold-{os.getpid()}"
    container_home = Path(args.container_home)
    argv = ["-e", f"CUTEAFD_BUILD_CACHES={'on' if enabled else 'off'}"]

    host_home = Path(args.build_root) / "container-home"

    def directory(path: Path) -> None:
        filesystem.check_path(str(path))
        if path.exists() and (path.stat().st_uid != os.getuid() or not os.access(path, os.W_OK)):
            raise ValueError(
                f"directory not owned/writable by host uid {os.getuid()}: {path}; "
                "choose a fresh WIP instance, or have the user run agent-sudo chown "
                f"-R {os.getuid()}:{os.getgid()} -- {path} (no automatic chown)"
            )
        if args.mode == "prepare":
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"pre-create before docker run (host uid {os.getuid()}): mkdir -p -- {path}", file=sys.stderr)

    def bind_directory(source: Path, destination: Path) -> None:
        directory(source)
        # Nested mount targets live inside the host bind of container_home.
        # Docker otherwise creates these directories as root in that host tree.
        if destination != container_home and destination.is_relative_to(container_home):
            directory(host_home / destination.relative_to(container_home))
        argv.extend(["--mount", f"type=bind,src={source.resolve()},dst={destination}"])

    def leaf(name: str, persistent: Path, destination: Path, env: str | None = None) -> None:
        source = persistent if enabled else fallback / name
        if enabled:
            try:
                nvme_cache(source)
                if "," in str(source):
                    raise ValueError("Docker bind path contains a comma")
                if source.exists() and (source.stat().st_uid != os.getuid() or not os.access(source, os.W_OK)):
                    raise ValueError(f"cache is not writable/owned by host uid {os.getuid()}")
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                print(f"warning: {name} cache refused ({source}): {error}; using per-build directory", file=sys.stderr)
                source = fallback / name
        filesystem.check_path(str(source))
        warm = source.is_dir() and any(source.iterdir())
        print(f"build cache {name}: {'warm' if warm else 'cold'} {source} -> {destination}", file=sys.stderr)
        bind_directory(source, destination)
        if env:
            argv.extend(["-e", f"{env}={destination}"])

    # Docker creates missing mount parents as root. Bind a host-owned home first
    # so Cargo can write its lock/config alongside the separately mounted inputs.
    bind_directory(host_home, container_home)
    directory(host_home / "cargo")
    # Cargo binaries remain image-owned; only its downloadable inputs are shared.
    argv.extend(["-e", f"CARGO_HOME={container_home / 'cargo'}"])
    for name in ("registry", "git"):
        leaf(f"cargo-{name}", root / "cargo-home" / arch / name, container_home / "cargo" / name)
    for name, env in (("triton", "TRITON_CACHE_DIR"), ("torchinductor", "TORCHINDUCTOR_CACHE_DIR"),
                      ("torch-extensions", "TORCH_EXTENSIONS_DIR"), ("xdg", "XDG_CACHE_HOME"),
                      ("roce", "B12X_ROCE_CACHE_DIR")):
        leaf(name, root / "jit" / args.toolchain / arch / name, container_home / name, env)
    # Cache-off must override the dev image and explicit compiler-cache settings.
    argv.extend(["-e", "CUTEAFD_KACHE=", "-e", "CUTEAFD_SCCACHE_CUDA=0", "-e", "CUTEAFD_KACHE_REMOTE="])
    if enabled and os.environ.get("CUTEAFD_KACHE", "1"):
        requested = os.environ.get("CUTEAFD_KACHE", "1")
        wrapper = "/opt/cuteafd-kache"
        available = True
        if requested != "1":
            executable = shutil.which(requested)
            available = executable is not None
            if available:
                argv.extend(["--mount", f"type=bind,src={Path(executable).resolve()},dst={wrapper},readonly"])
            else:
                print(f"warning: kache not found ({requested}); using plain compilers", file=sys.stderr)
        if available:
            leaf("kache", Path(os.environ.get("CUTEAFD_KACHE_CACHE_DIR", str(root / 'kache'))) / arch,
                 Path("/opt/cuteafd-kache-cache") / arch)
            argv.extend(["-e", f"CUTEAFD_KACHE={wrapper}", "-e", "CUTEAFD_KACHE_REQUESTED=1",
                         "-e", "CUTEAFD_KACHE_CACHE_DIR=/opt/cuteafd-kache-cache"])
    if enabled and os.environ.get("CUTEAFD_SCCACHE_CUDA", "1") == "1":
        leaf("sccache", Path(os.environ.get("CUTEAFD_SCCACHE_CACHE_DIR", str(root / 'sccache'))) / arch,
             Path("/opt/cuteafd-sccache-cache") / arch)
        argv.extend(["-e", "CUTEAFD_SCCACHE_CUDA=1", "-e", "SCCACHE_DIR=/opt/cuteafd-sccache-cache"])
    print("\n".join(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"error: build cache plan: {error}", file=sys.stderr)
        raise SystemExit(2)
