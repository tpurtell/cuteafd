//! The CPUs the table read reactor (and so its io-uring workers, which inherit
//! the creating thread's affinity) runs on.
//!
//! Spread over every CPU, reads through a FUSE mount make the daemon's per-CPU
//! request threads contend on one futex hash bucket: on raptor, 64 CPUs read
//! 57K rows/s at 2 ms a read while the daemon spends ~48 cores spinning; 8 CPUs
//! read 144K rows/s at 0.7 ms with the daemon under one core. Local tables gain
//! too (+10% to +35%), and 16 CPUs already bring the contention back, so reads
//! are confined to [`READER_CPUS`] cores of one last-level cache (the
//! daemon's request threads then share it: 8 cores spread over three L3
//! domains read half as fast as 8 in one), chosen from the process's own CPUs:
//! the L3 domain taking the fewest network and NVMe interrupts, never the one
//! holding CPU 0 when another exists, nor a CPU another thread is pinned to.
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

/// Pin the calling thread to the reader CPUs. Call before creating the ring.
pub(super) fn pin_reader() {
    let Some(allowed) = affinity_of(0) else { return };
    // CPUs other threads of this process are pinned to (a narrower set than ours).
    let mut pinned = BTreeSet::new();
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            let Some(tid) = task.file_name().to_str().and_then(|t| t.parse::<libc::pid_t>().ok()) else { continue };
            if let Some(cpus) = affinity_of(tid).filter(|cpus| cpus.len() < allowed.len()) { pinned.extend(cpus); }
        }
    }
    let cpus = reader_cpus(&allowed, &pinned, topology, &domain_irqs(topology));
    if cpus.is_empty() { return; }
    // SAFETY: as above; the set is built from valid CPU indices below CPU_SETSIZE.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &cpu in &cpus { unsafe { libc::CPU_SET(cpu, &mut set) }; }
    // SAFETY: applies a valid set to the calling thread only.
    let applied = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) } == 0;
    tracing::info!(?cpus, applied, allowed = allowed.len(), pinned_elsewhere = pinned.len(), "table read reactor CPUs");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
