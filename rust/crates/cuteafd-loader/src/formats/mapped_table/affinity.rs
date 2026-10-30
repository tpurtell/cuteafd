//! The CPUs the table read reactor (and its ring's io-uring workers) runs on.
//!
//! While every mapped table file is local, reads are confined to [`READER_CPUS`]
//! cores of one last-level cache: on raptor the real Engram table reads 479K
//! rows/s there against 359K spread over all 64 CPUs, and 8 cores spread over
//! three L3 domains lose most of the gain. The domain is the one taking the
//! fewest network and NVMe interrupts, never the one holding CPU 0 while another
//! is allowed, and never a CPU another thread of the process is pinned to.
//!
//! Once a remote table file is mapped the reader is spread again: through the
//! SparkNest 0.7.1 FUSE mount remote reads go faster over every CPU (173K vs
//! 156K rows/s). (On SparkNest 0.6 the pin had made remote reads 2.5x faster by
//! sidestepping a mutex the daemon took per read; 0.7 removed it.)
use std::collections::BTreeSet;

/// Cores the reader uses (one CPU each).
pub(super) const READER_CPUS: usize = 8;

/// Where a CPU sits: its physical core and its last-level cache domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Topology { pub core: usize, pub l3: usize }

/// Reader CPUs from `allowed` (the process's CPUs): CPUs another thread is
/// pinned to (`pinned`) and CPU 0's core are left out; of the L3 domains left,
/// the one with the least interrupt load (`irqs`: network and NVMe interrupts
/// per L3 domain), not CPU 0's while another has CPUs, and ties to the higher
/// domain; in it one CPU per core (its lowest), at most [`READER_CPUS`].
/// Empty when nothing qualifies (the caller then leaves the affinity alone).
pub(super) fn reader_cpus(allowed: &[usize], pinned: &BTreeSet<usize>, at: impl Fn(usize) -> Topology,
    irqs: &std::collections::HashMap<usize, u64>) -> Vec<usize> {
    let home = at(0);
    let mut domains: std::collections::BTreeMap<usize, std::collections::BTreeMap<usize, usize>> = Default::default();
    for &cpu in allowed {
        let place = at(cpu);
        if place.core == home.core || pinned.contains(&cpu) { continue; }
        domains.entry(place.l3).or_default().entry(place.core)
            .and_modify(|first| *first = (*first).min(cpu)).or_insert(cpu);
    }
    let others = domains.keys().any(|&l3| l3 != home.l3);
    let Some((_, cores)) = domains.into_iter().filter(|&(l3, _)| !others || l3 != home.l3)
        .min_by_key(|&(l3, _)| (irqs.get(&l3).copied().unwrap_or(0), std::cmp::Reverse(l3))) else { return Vec::new() };
    let mut cpus: Vec<usize> = cores.into_values().take(READER_CPUS).collect();
    cpus.sort_unstable();
    cpus
}

fn topology(cpu: usize) -> Topology {
    let read = |path: String| std::fs::read_to_string(path).ok().and_then(|text| text.trim().parse().ok());
    let base = format!("/sys/devices/system/cpu/cpu{cpu}");
    Topology { core: read(format!("{base}/topology/core_id")).unwrap_or(cpu),
        l3: read(format!("{base}/cache/index3/id")).unwrap_or(0) }
}

/// Each L3 domain's network (mlx5) and NVMe interrupt count from /proc/interrupts.
fn domain_irqs(at: impl Fn(usize) -> Topology) -> std::collections::HashMap<usize, u64> {
    let mut load = std::collections::HashMap::new();
    let Ok(text) = std::fs::read_to_string("/proc/interrupts") else { return load };
    let mut lines = text.lines();
    let cpus = lines.next().map_or(0, |header| header.split_whitespace().count());
    for line in lines {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() <= cpus || !fields[0].trim_end_matches(':').chars().all(|c| c.is_ascii_digit()) { continue; }
        let name = fields[cpus + 1..].join(" ");
        if !(name.contains("mlx5") || name.contains("nvme")) { continue; }
        for (cpu, count) in fields[1..=cpus].iter().enumerate() {
            if let Ok(count) = count.parse::<u64>() { *load.entry(at(cpu).l3).or_insert(0) += count; }
        }
    }
    load
}

fn affinity_of(tid: libc::pid_t) -> Option<Vec<usize>> {
    // SAFETY: a zeroed cpu_set_t is valid; sched_getaffinity fills it for `tid`.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `set` is a valid, sized cpu_set_t owned by this frame.
    if unsafe { libc::sched_getaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } != 0 { return None; }
    // SAFETY: CPU_ISSET reads the initialized set within its fixed size.
    Some((0..libc::CPU_SETSIZE as usize).filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) }).collect())
}

/// Whether the reader is pinned: only while every mapped table file is local.
/// Reads through a remote FUSE mount (SparkNest 0.7.1) are faster spread over
/// every CPU (173K vs 156K reads/s on raptor); local reads are faster on one
/// L3 domain (479K vs 359K, real Engram table).
pub(super) fn pin_wanted(remote_files: u64) -> bool { remote_files == 0 }

fn set_of(cpus: &[usize]) -> libc::cpu_set_t {
    // SAFETY: a zeroed cpu_set_t is valid; CPU_SET writes indices below CPU_SETSIZE.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &cpu in cpus.iter().filter(|&&cpu| cpu < libc::CPU_SETSIZE as usize) { unsafe { libc::CPU_SET(cpu, &mut set) }; }
    set
}

/// The reader CPUs for this process now.
fn reader_set() -> Option<(Vec<usize>, Vec<usize>)> {
    let allowed = affinity_of(0)?;
    // CPUs other threads of this process are pinned to (a narrower set than ours).
    let mut pinned = BTreeSet::new();
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            let Some(tid) = task.file_name().to_str().and_then(|t| t.parse::<libc::pid_t>().ok()) else { continue };
            if let Some(cpus) = affinity_of(tid).filter(|cpus| cpus.len() < allowed.len()) { pinned.extend(cpus); }
        }
    }
    let cpus = reader_cpus(&allowed, &pinned, topology, &domain_irqs(topology));
    (!cpus.is_empty()).then_some((cpus, allowed))
}

/// The reactor thread's CPUs, kept in step with the mapped tables' placement.
pub(super) struct Pinning { remote_seen: Option<u64>, allowed: Option<Vec<usize>>, pinned: bool }

impl Pinning {
    pub(super) fn new() -> Self { Self { remote_seen: None, allowed: affinity_of(0), pinned: false } }

    /// Before the ring exists: pin the thread (its io-uring workers inherit it)
    /// when every table mapped so far is local.
    pub(super) fn before_ring(&mut self) { self.update(None); }

    /// When the mapped tables' placement changed: pin or unpin the thread and
    /// the ring's io-uring workers.
    pub(super) fn update(&mut self, ring: Option<&io_uring::IoUring>) {
        let remote = super::placement::remote_files();
        if self.remote_seen == Some(remote) { return; }
        self.remote_seen = Some(remote);
        let want = pin_wanted(remote);
        if want == self.pinned { return; }
        let cpus = if want { match reader_set() { Some((cpus, _)) => cpus, None => return } }
            else { self.allowed.clone().unwrap_or_default() };
        if cpus.is_empty() { return; }
        let set = set_of(&cpus);
        // SAFETY: applies a valid set to the calling (reactor) thread only.
        let thread = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) } == 0;
        let workers = ring.map(|ring| if want { ring.submitter().register_iowq_aff(&set) }
            else { ring.submitter().unregister_iowq_aff() }.is_ok());
        self.pinned = want && thread;
        tracing::info!(pinned = self.pinned, cpus = ?(if want { cpus } else { Vec::new() }), remote_files = remote, ?workers,
            "table read reactor CPUs");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn pinned_only_while_every_table_is_local() {
        assert!(pin_wanted(0));
        assert!(!pin_wanted(1));
        assert!(!pin_wanted(14));
    }

    /// raptor: 32 cores in four L3 domains of 8; CPU n and n + 32 are one core.
    fn raptor(cpu: usize) -> Topology { Topology { core: cpu % 32, l3: (cpu % 32) / 8 } }

    #[test]
    fn the_quietest_l3_domain_other_than_cpu0s() {
        let all: Vec<usize> = (0..64).collect();
        // raptor 2026-10-31: mlx5 + NVMe interrupts per L3 domain (millions).
        let irqs: HashMap<usize, u64> = [(0, 844), (1, 587), (2, 572), (3, 585)].into_iter().collect();
        assert_eq!(reader_cpus(&all, &BTreeSet::new(), raptor, &irqs), (16..24).collect::<Vec<_>>());
        // Without interrupt data: the highest domain.
        assert_eq!(reader_cpus(&all, &BTreeSet::new(), raptor, &HashMap::new()), (24..32).collect::<Vec<_>>());
    }

    #[test]
    fn cpu0s_domain_only_when_nothing_else_is_allowed() {
        let quiet0: HashMap<usize, u64> = [(0, 1), (1, 900)].into_iter().collect();
        assert_eq!(reader_cpus(&(0..16).collect::<Vec<_>>(), &BTreeSet::new(), raptor, &quiet0), (8..16).collect::<Vec<_>>());
        // A cpuset inside domain 0: its cores but CPU 0's.
        assert_eq!(reader_cpus(&[0, 1, 2, 3, 33], &BTreeSet::new(), raptor, &HashMap::new()), vec![1, 2, 3]);
        assert!(reader_cpus(&[0, 32], &BTreeSet::new(), raptor, &HashMap::new()).is_empty());
    }

    #[test]
    fn pinned_cpus_are_avoided_and_cores_are_not_split() {
        let all: Vec<usize> = (0..64).collect();
        let pinned: BTreeSet<usize> = (28..32).collect();
        // Domain 3 keeps cores 28..31 through their siblings 60..63.
        assert_eq!(reader_cpus(&all, &pinned, raptor, &HashMap::new()), vec![24, 25, 26, 27, 60, 61, 62, 63]);
        // A cpuset of SMT siblings only: one CPU per core.
        assert_eq!(reader_cpus(&[4, 36, 5, 37], &BTreeSet::new(), raptor, &HashMap::new()), vec![4, 5]);
    }
}
