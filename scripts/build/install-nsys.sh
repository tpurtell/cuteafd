#!/usr/bin/env bash
set -euo pipefail
# Official CLI packages, pinned independently of the NGC base's older CUDA copy.
[[ "$NSYS_VERSION" == 2026.6.1.251-266139333969v0 ]]
case "$(uname -m)" in
  x86_64)
    arch=amd64
    target=target-linux-x64
    package=NsightSystems-linux-cli-public-2026.6.1.251-39333969.deb
    sha=3455b6832266542fcf3b5a4c60a3abcdd1340444c7307c9ad20731efbeb0686a
    ;;
  aarch64)
    arch=arm64
    target=target-linux-sbsa-armv8
    package=nsight-systems-cli-2026.6.1_2026.6.1.251-1_arm64.deb
    sha=0404d3a3db9f3930dc12118438124fcfbfe2812e78eb1cef06ad3d8589cd675f
    ;;
  *) printf 'unsupported Nsight Systems architecture\n' >&2; exit 1 ;;
esac
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
curl --retry 3 -fsSL "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/$arch/$package" -o "$work/nsys.deb"
printf '%s  %s\n' "$sha" "$work/nsys.deb" | sha256sum --check --strict
[[ "$(dpkg-deb -f "$work/nsys.deb" Version)" == "$NSYS_VERSION" ]]
apt-get update
apt-get install -y --no-install-recommends "$work/nsys.deb"
rm -rf /var/lib/apt/lists/*
# wip-cards uses CUDA's absolute path to control the profiling session; PATH
# resolution and the launcher's /usr/local/bin preference must agree with it.
bin="/opt/nvidia/nsight-systems-cli/2026.6.1/$target/nsys"
test -x "$bin"
ln -sf "$bin" /usr/local/cuda/bin/nsys
nsys --version
/usr/local/bin/nsys --version
/usr/local/cuda/bin/nsys --version
