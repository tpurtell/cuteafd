# Host tuning

One inventory for host policy: `scripts/host/tune.sh`. Run outside containers.
Auto-detection maps x86-64 to coordinator and ARM64 to Spark; other sites
should pass `--role coordinator|spark` explicitly. ARM64 is not proof of GB10.

## Applied policy

| Setting | Role | Recommended value | Reason |
| --- | --- | --- | --- |
| CPU governor | Both | `performance` | Avoid frequency ramp-up on short host phases. |
| Swap | Both | Off; fstab swap entries commented, swap units masked | Avoid reclaim-to-disk stalls in serving. Leave the backing file intact. |
| THP `enabled` / `defrag` | Both | `madvise` | Keep hugepages opt-in and avoid unsolicited compaction. |
| `vm.zone_reclaim_mode` | Both | `0` | Avoid local-zone reclaim stalls. |
| Energy-performance preference (EPP) | Coordinator, where exposed | `performance` | Prefer responsiveness over energy saving. |
| cpuidle `disable` | Coordinator | `1` when exit `latency >= 50` microseconds | Avoid deep-idle wake latency between waves; retain shallow states. |
| cpuidle `disable` | Spark | `1` when exit `latency >= 200` microseconds | Avoid deep GB10 wake latency; retain shallow states. |
| `vm.compaction_proactiveness` | Both | `0` | TJ's 2026-10-23 Spark policy, extended to both roles: stop proactive background compaction competing with serving. This does not disable allocation-time compaction. |

The cutoffs are variables at the top of the script. Selection uses each
state's measured sysfs `latency`, never its name. On raptor this excludes
C2 (100 microseconds), retaining POLL/C1. On GB10 it excludes LPI-2
(231 microseconds) and LPI-3 (433 microseconds), retaining LPI-0/LPI-1.
The script only disables deep states; it never re-enables a shallow state
an operator deliberately disabled. Governor and EPP apply per cpufreq policy.

**Measured evidence:** [#717](https://github.com/tpurtell/cuteafd/issues/717)
profiles identify per-wave host gaps: V4.1 had 11.8 seconds host-late in a
25.2-second window, with 9.1 seconds in gaps at least 200 microseconds.
Other families show route/exchange waits. These establish latency-sensitive
host work, not that idle exit caused every gap.

**Recommended, not measured speedups:** the idle cutoffs are TJ's
2026-10-25 decision, approved without an A/B. They trade idle power for
wake responsiveness. Governor/EPP performance and Spark compaction zero
were already cluster policy; the script enforces them. No throughput gain
is claimed for this host-tuning change.

## Read-only inventory

- CPU: governor/EPP and every CPU's idle-state name, exit latency, disable
  bit and usage counter. `[pass]` matches a recommendation, `[differs]`
  does not, `[unknown]` means unavailable or no universal recommendation.
- THP `enabled`/`defrag` `madvise` and `vm.zone_reclaim_mode=0`: likely
  Spark/DGX OS defaults, already set here; apply enforces them nonetheless.
- Unlimited memlock and inactive irqbalance: established cluster settings,
  not assumed universal OS defaults; report only. Memlock is the report process's limit;
  a running expert worker's limits are also shown. Service/container limits
  can differ from an interactive shell. For a human-installed
  `/etc/security/limits.d/90-cuteafd.conf`, use `* - memlock unlimited`
  (add `root - memlock unlimited` if needed); systemd services need
  `LimitMEMLOCK=infinity`, containers `--ulimit memlock=-1:-1`. New login or
  service restart is required. Review IRQ placement before disabling an
  active irqbalance service; the script does not stop it.
- RoCE netdev IRQ CPU lists and `ethtool -c` coalescing: report only.
  Affinities are grouped by identical CPU list, retaining every IRQ ID.
  Choose IRQ/NUMA placement and coalescing against the site's topology and
  traffic; this script does not set them.
- `rdma link` and each RDMA port's sysfs `rate`: negotiated rates, not
  assumed hardware maxima. Check the actual rails before interpreting runs.
- PFC and DCB ETS/application/DCBX state via `dcb`: report only. Missing
  tools, unsupported drivers or insufficient privileges yield unknown.
  No generic PFC/DCB/ECN recommendation; coordinate with the fabric owner.
- `nvidia-smi -q -d PERFORMANCE`: GPU clocks-event reasons, read-only.
  GPU power is gpu-powerd's job on our cluster. Never change GPU limits or
  clocks, switch settings, NIC firmware, DCB, ECN or PFC with this script.

## GB10 core placement

GB10 has 10 Cortex-X925 performance cores (3.9 GHz) and 10 Cortex-A725
efficiency cores (about 2.8 GHz; ostrich reports 2808 MHz). Report derives the P/E sets from `lscpu`
`MAXMHZ`, never fixed CPU ranges. Missing or uniform frequencies yield
unknown. Recommend pinning the Spark expert worker to the reported P
cores; do not copy another machine's CPU IDs. The
[NVIDIA discussion](https://forums.developer.nvidia.com/t/fine-tune-kernel-level/372645)
is guidance, not a cuteafd performance measurement.

For a visible `cuteafd expertd`, report lists thread IDs, last CPUs and
`taskset -p` masks. It labels the highest lifetime-%CPU thread's placement
P/E/unknown. This is a snapshot heuristic, not proof of which thread is
currently hot or where it spends all its time. Pinning is a recommendation
only; this PR does not change launcher affinity.

## Considered and not applied

- `nohz_full`/`rcu_nocbs`/`isolcpus`: reduce tick/RCU-callback jitter on
  dedicated cores, but require boot parameters/reboot, one hot task per
  core, and can slightly slow syscalls. Try P-core placement first; use
  only if a profile shows tick jitter on a spinning hot thread. Device-side
  decode exchange removes most engine-thread spinning.
- `processor.max_cstate=1`: needs a reboot; use the latency-selected sysfs
  idle policy instead.
- NCCL: not used by design; serving uses our verbs/P2P collectives.

## GB10 memory

CUDA allocations reclaim page cache. Size from `/proc/meminfo`'s
`MemAvailable`, not `MemFree` or `nvidia-smi`'s unavailable GB10 memory
reading. Never drop caches. Keep headroom and bound simultaneous CUDA
context creation: [#730](https://github.com/tpurtell/cuteafd/issues/730)
records context-creation OOMs despite high MemAvailable; reclaimable bytes
are not a guarantee that parallel contexts can allocate immediately.

## TCP policy (both roles)

Recommended, not measured requirements: these serve HTTP, SparkNest and a
possible future 10G TCP expert transport for an eight-node cluster. They do
not affect today's RDMA verbs path. Apply checks BBR availability, loads
`tcp_bbr` when needed, and writes `/etc/modules-load.d/90-cuteafd.conf`.
The VM and TCP values share `/etc/sysctl.d/90-cuteafd.conf`:

```ini
# TJ 2026-10-23: background proactive compaction on GB10 unified memory stalls serving.
vm.compaction_proactiveness = 0
vm.zone_reclaim_mode = 0
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 500000
net.core.somaxconn = 65535
```

`fq` is required for BBR pacing. Changing `default_qdisc` does not replace
an existing interface's qdisc; inspect `tc qdisc show` and arrange interface
recreation/reboot in a maintenance window. The script never runs `tc`.
BBR v1 can crowd out CUBIC flows on shared links: acceptable on our own
cluster, but reconsider on a shared network. Buffer maxima are ceilings,
not per-socket allocations. Existing configuration files get a one-time
`.cuteafd.bak` backup. Changes are reversible: restore prior sysctl values
and backups/remove these snippets, then reload sysctls; remove the module
snippet if no longer wanted. Do not unload a module in active use.

## Swap policy

Report checks `swapon --show`, active swap lines in `/etc/fstab`, and
installed/generated/loaded swap-unit masks. Apply runs `swapoff -a` only
when needed, backs up fstab once to `/etc/fstab.cuteafd.bak`, comments active
swap entries, and masks discovered swap units. It never deletes backing
files or uses force to overwrite a local unit. A conflicting local swap
unit must be reviewed by the human. Fstab must be readable and systemd
inventory available; unknown inventory refuses swap apply. Swapoff may
need substantial memory headroom: review report and available RAM first.
Apply with serving stopped: masking units can trigger a systemd reload,
which can invalidate running GPU containers on our cluster.

On raptor/ostrich/kakapo, `/swap.img` is commented in fstab,
`swap.img.swap` is masked and no swap is active. Ostrich still has its
16 GB `/swap.img`; an operator may remove it after confirming swap is off.
`vm.swappiness` is reported for context but never set: it is irrelevant
once swap is disabled. Disable any external zram/swap generator separately
if it recreates units after boot. No swap file is deleted by this script.

## Run and persist

```bash
scripts/host/tune.sh report --role coordinator
scripts/host/tune.sh apply --role coordinator --dry-run
# Human operator only, after reviewing the proposed writes:
sudo scripts/host/tune.sh apply --role coordinator
scripts/host/tune.sh persist --role coordinator
```

Use `--role spark` on GB10. Default action is `report`; no root needed.
`apply` refuses non-root callers; `--dry-run` is an unprivileged preview
that prints only proposed writes and changes nothing. Repeated apply
skips values already correct. Writes fail visibly rather than claiming
success when the kernel rejects them.

`persist` only prints a systemd unit and installation instructions. A human
installs the reviewed script at `/usr/local/sbin/cuteafd-host-tune`, owned by
root and mode 0755, and the unit at
`/etc/systemd/system/cuteafd-host-tune.service`, then enables it. Idle-state
`disable` resets on reboot. Coordinate with any other governor/tuning
service; the unit runs after `ondemand.service` where present. CPU hotplug
or later policy changes may require reapplying; this is a boot oneshot,
not a watchdog. No service or host setting is installed by `persist`.

Tests use `TUNE_SYSFS_ROOT=PATH`: sysfs files live under PATH, proc files
under PATH/proc, and external hardware probes are suppressed. Do not set
this fixture override for real host tuning.
