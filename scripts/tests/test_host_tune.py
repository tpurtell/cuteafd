"""Host policy fixtures: no host writes, privileges, or hardware required."""
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "host/tune.sh"
TCP = {
    "net.core.default_qdisc": "fq", "net.ipv4.tcp_congestion_control": "bbr",
    "net.core.rmem_max": "134217728", "net.core.wmem_max": "134217728",
    "net.ipv4.tcp_rmem": "4096 87380 134217728",
    "net.ipv4.tcp_wmem": "4096 65536 134217728",
    "net.core.netdev_max_backlog": "500000", "net.core.somaxconn": "65535",
}
SYSCTL_CONFIG = ("# TJ 2026-10-23: background proactive compaction on GB10 unified memory stalls serving.\n"
                 "vm.compaction_proactiveness = 0\nvm.zone_reclaim_mode = 0\n") + "\n".join(
    f"{key} = {value}" for key, value in TCP.items())


def put(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(value) + "\n")
    return target


def fixture(root, role):
    policy = "devices/system/cpu/cpufreq/policy0"
    put(root, f"{policy}/scaling_governor", "powersave")
    put(root, f"{policy}/energy_performance_preference", "balance_performance")
    # Names deliberately do not encode ordering or the latency policy.
    latencies = [0, 1, 49, 50, 100] if role == "coordinator" else [1, 165, 199, 200, 231, 433]
    for cpu in (0, 1):
        for index, latency in enumerate(latencies):
            state = f"devices/system/cpu/cpu{cpu}/cpuidle/state{index}"
            for name, value in {"name": f"arbitrary-{index}", "latency": latency,
                                "disable": 0, "usage": 123}.items():
                put(root, f"{state}/{name}", value)
    for key, value in TCP.items():
        put(root, "proc/sys/" + key.replace(".", "/"), value)
    put(root, "proc/sys/net/ipv4/tcp_available_congestion_control", "reno cubic bbr")
    put(root, "etc/sysctl.d/90-cuteafd.conf", SYSCTL_CONFIG)
    put(root, "etc/modules-load.d/90-cuteafd.conf", "tcp_bbr")
    put(root, "proc/sys/vm/compaction_proactiveness", 20)
    put(root, "proc/sys/vm/swappiness", 60)
    put(root, "proc/sys/vm/zone_reclaim_mode", 0)
    put(root, "swapon.txt", "")
    put(root, "swap-units.txt", "swap.img.swap")
    put(root, "etc/fstab", "# /swap.img none swap sw 0 0")
    units = root / "etc/systemd/system"
    units.mkdir(parents=True)
    (units / "swap.img.swap").symlink_to("/dev/null")
    put(root, "lscpu.csv", "# CPU,MAXMHZ\n0,2800.0000\n1,3900.0000")
    for name in ("enabled", "defrag"):
        put(root, f"kernel/mm/transparent_hugepage/{name}", "always [madvise] never")
    put(root, "class/infiniband/mlx5_0/ports/1/rate", "200 Gb/sec (4X HDR)")
    put(root, "class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/0", "roce0")
    put(root, "class/net/roce0/device/msi_irqs/42", "")
    put(root, "proc/irq/42/smp_affinity_list", "0-3")
    return latencies


def run(root, *args, unprivileged=False):
    env = dict(os.environ, TUNE_SYSFS_ROOT=str(root))
    # A caller must not be able to forge bash's readonly EUID through env.
    env["EUID"] = "0"
    kwargs = {}
    if unprivileged and os.geteuid() == 0:
        kwargs.update(user=65534, group=65534, extra_groups=[])
    return subprocess.run(["bash", str(SCRIPT), *args], env=env,
                          text=True, capture_output=True, timeout=15, **kwargs)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("role,threshold", [("coordinator", 50), ("spark", 200)])
def test_report_and_dry_run_select_by_latency(tmp_path, role, threshold):
    latencies = fixture(tmp_path, role)
    before = snapshot(tmp_path)
    report = run(tmp_path, "report", "--role", role)
    assert report.returncode == 0, report.stderr
    for cpu in (0, 1):
        for index, latency in enumerate(latencies):
            expected = int(latency >= threshold)
            marker = "differs" if expected else "pass"
            assert (f"[{marker}] cpu{cpu}/state{index} "
                    f"name=arbitrary-{index} latency_us={latency} usage=123 disable: 0 "
                    f"| target={expected}") in report.stdout
    assert "[pass] kernel/mm/transparent_hugepage/enabled: madvise | target=madvise" in report.stdout
    assert "[pass] proc/sys/vm/zone_reclaim_mode: 0 | target=0" in report.stdout
    assert "[pass] active swap (swapon --show): none | target=none" in report.stdout
    assert "[pass] fstab active swap entries: none | target=none" in report.stdout
    assert "[pass] swap unit swap.img.swap: masked | target=masked" in report.stdout
    if role == "spark":
        assert "P=1 (3900MHz) E=0 (2800MHz)" in report.stdout
    assert "200 Gb/sec" in report.stdout
    assert "IRQs 42 affinity_list: 0-3" in report.stdout
    assert "[unknown] roce0 PFC" in report.stdout
    assert "[unknown] GPU performance / clocks-event reasons" in report.stdout

    dry = run(tmp_path, "apply", "--role", role, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    writes = {line.split(" <- ")[0].removeprefix("DRY-RUN write ")
              for line in dry.stdout.splitlines() if line.startswith("DRY-RUN write ")}
    expected = {str(tmp_path / "devices/system/cpu/cpufreq/policy0/scaling_governor"),
                str(tmp_path / "proc/sys/vm/compaction_proactiveness")}
    if role == "coordinator":
        expected.add(str(tmp_path / "devices/system/cpu/cpufreq/policy0/energy_performance_preference"))
    for cpu in (0, 1):
        for index, latency in enumerate(latencies):
            if latency >= threshold:
                expected.add(str(tmp_path / f"devices/system/cpu/cpu{cpu}/cpuidle/state{index}/disable"))
    assert writes == expected
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("role", ["coordinator", "spark"])
def test_idempotent_preview_and_shallow_state_preserved(tmp_path, role):
    latencies = fixture(tmp_path, role)
    threshold = 50 if role == "coordinator" else 200
    put(tmp_path, "devices/system/cpu/cpufreq/policy0/scaling_governor", "performance")
    put(tmp_path, "devices/system/cpu/cpufreq/policy0/energy_performance_preference", "performance")
    put(tmp_path, "proc/sys/vm/compaction_proactiveness", 0)
    put(tmp_path, "proc/sys/vm/swappiness", 10)
    for cpu in (0, 1):
        for index in range(len(latencies)):
            put(tmp_path, f"devices/system/cpu/cpu{cpu}/cpuidle/state{index}/disable", 1)
    result = run(tmp_path, "apply", "--role", role, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "write " not in result.stdout
    assert f"idle_disable_latency_us>={threshold}" in result.stdout


def test_refuses_apply_without_root(tmp_path):
    result = run(tmp_path, "apply", "--role", "spark", unprivileged=True)
    assert result.returncode == 1
    assert "apply requires root (EUID 0)" in result.stderr
    assert not list(tmp_path.iterdir())


def test_fixture_report_never_runs_host_probes(tmp_path):
    fixture(tmp_path, "spark")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "host-probe-ran"
    for name in ("lscpu", "ps", "taskset", "ethtool", "dcb", "rdma", "nvidia-smi", "systemctl", "swapon", "swapoff"):
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
        stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               TUNE_SYSFS_ROOT=str(tmp_path))
    result = subprocess.run(["bash", str(SCRIPT), "report", "--role", "spark"],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_swap_disable_preview_and_default_enforcement(tmp_path):
    fixture(tmp_path, "spark")
    put(tmp_path, "swapon.txt", "/swap.img file 16G 0B -2")
    put(tmp_path, "etc/fstab", "/dev/root / ext4 defaults 0 1\n/swap.img none swap sw 0 0\n# /old none swap sw 0 0")
    (tmp_path / "etc/systemd/system/swap.img.swap").unlink()
    for name in ("enabled", "defrag"):
        put(tmp_path, f"kernel/mm/transparent_hugepage/{name}", "[always] madvise never")
    put(tmp_path, "proc/sys/vm/zone_reclaim_mode", 1)
    before = snapshot(tmp_path)
    result = run(tmp_path, "apply", "--role", "spark", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN command swapoff -a" in result.stdout
    assert f"DRY-RUN backup {tmp_path}/etc/fstab" in result.stdout
    assert f"DRY-RUN comment active swap entries in {tmp_path}/etc/fstab" in result.stdout
    assert "DRY-RUN command systemctl mask -- swap.img.swap" in result.stdout
    assert f"{tmp_path}/proc/sys/vm/zone_reclaim_mode <- 0" in result.stdout
    for name in ("enabled", "defrag"):
        assert f"{tmp_path}/kernel/mm/transparent_hugepage/{name} <- madvise" in result.stdout
    assert "swappiness" not in result.stdout
    assert snapshot(tmp_path) == before


def test_tcp_policy_preview_and_persistence(tmp_path):
    fixture(tmp_path, "spark")
    for key in TCP:
        put(tmp_path, "proc/sys/" + key.replace(".", "/"), "0")
    put(tmp_path, "proc/sys/net/ipv4/tcp_available_congestion_control", "reno cubic")
    (tmp_path / "etc/sysctl.d/90-cuteafd.conf").unlink()
    (tmp_path / "etc/modules-load.d/90-cuteafd.conf").unlink()
    before = snapshot(tmp_path)
    result = run(tmp_path, "apply", "--role", "spark", "--dry-run")
    assert result.returncode == 0, result.stderr
    for key, value in TCP.items():
        assert f"{tmp_path}/proc/sys/{key.replace('.', '/')} <- {value}" in result.stdout
    assert "check tcp_bbr availability (modinfo); command modprobe tcp_bbr" in result.stdout
    assert f"DRY-RUN write config {tmp_path}/etc/sysctl.d/90-cuteafd.conf" in result.stdout
    assert f"DRY-RUN write config {tmp_path}/etc/modules-load.d/90-cuteafd.conf <- tcp_bbr" in result.stdout
    assert snapshot(tmp_path) == before


def test_unknown_latency_is_never_written(tmp_path):
    fixture(tmp_path, "coordinator")
    state = "devices/system/cpu/cpu0/cpuidle/state4"
    put(tmp_path, f"{state}/latency", "unavailable")
    result = run(tmp_path, "apply", "--role", "coordinator", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert f"{state}/disable" not in result.stdout


def test_missing_interfaces_report_unknown(tmp_path):
    result = run(tmp_path, "--role", "spark")
    assert result.returncode == 0, result.stderr
    for label in ("CPU governor", "CPU energy_performance_preference", "cpuidle",
                  "kernel/mm/transparent_hugepage/enabled", "rdma link", "RoCE port rate", "RoCE NIC diagnostics",
                  "memlock", "irqbalance", "GB10 P/E cores", "active swap", "fstab active swap", "swap units"):
        assert f"[unknown] {label}" in result.stdout


def test_persist_prints_only_unit(tmp_path):
    result = run(tmp_path, "persist", "--role", "spark")
    assert result.returncode == 0, result.stderr
    assert "[Unit]" in result.stdout
    assert "ExecStart=/usr/local/sbin/cuteafd-host-tune apply --role spark" in result.stdout
    assert str(tmp_path) not in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("args", [("--role", "invalid"), ("--role",), ("report", "apply"),
                                   ("report", "--dry-run"), ("--unknown",)])
def test_invalid_arguments(tmp_path, args):
    assert run(tmp_path, *args).returncode == 2
