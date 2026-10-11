//! Startup-only pinned prefix sizing. GPU KV capacity and active request
//! admission are separate: these bytes store exact inactive snapshots.
use super::parse_bytes;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::FamilyLayout;
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostBudget {
    Auto,
    Bytes(u64),
}

impl FromStr for HostBudget {
    type Err = String;
    fn from_str(text: &str) -> std::result::Result<Self, Self::Err> {
        match text.trim() {
            "auto" => Ok(Self::Auto),
            value => parse_bytes(value).map(Self::Bytes),
        }
    }
}

impl HostBudget {
    pub fn enabled(self) -> bool {
        !matches!(self, Self::Bytes(0))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HostMemory {
    pub total: u64,
    pub available: u64,
}

/// Bound against live available RAM and every readable cgroup ancestor's
/// remaining hard limit. Already-used mapped weights and staging are counted
/// by the operating system, rather than subtracted from total RAM twice.
pub(super) fn host_memory() -> Result<HostMemory> {
    let text = std::fs::read_to_string("/proc/meminfo").context("reading host memory for automatic prefix budget")?;
    let mut memory = meminfo(&text)?;
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").context("reading prefix-cache cgroup membership")?;
    for line in cgroups.lines() {
        let parts: Vec<_> = line.splitn(3, ':').collect();
        let [_, controllers, relative] = parts.as_slice() else { continue };
        let (root, limit_file, current_file) = if controllers.is_empty() {
            (Path::new("/sys/fs/cgroup"), "memory.max", "memory.current")
        } else if controllers.split(',').any(|c| c == "memory") {
            (Path::new("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")
        } else { continue };
        let member = root.join(relative.trim_start_matches('/'));
        // The namespace may expose its current cgroup directly at the mount
        // root instead of reproducing the host membership path.
        for path in member.ancestors().take_while(|p| p.starts_with(root)).chain(std::iter::once(root)) {
            let limit = std::fs::read_to_string(path.join(limit_file));
            let current = std::fs::read_to_string(path.join(current_file));
            if let (Ok(limit), Ok(current)) = (limit, current) {
                apply_cgroup(&mut memory, &limit, &current)?;
            }
        }
    }
    Ok(memory)
}

/// Host RAM total for host-side caches sized from it (`EmbeddingCache::default_budget`).
pub(crate) fn host_total() -> Result<u64> { Ok(host_memory()?.total) }

fn meminfo(text: &str) -> Result<HostMemory> {
    let field = |name: &str| -> Result<u64> {
        let value = text.lines().find_map(|line| line.strip_prefix(name))
            .with_context(|| format!("/proc/meminfo lacks {name}"))?;
        let kib: u64 = value.split_whitespace().next().context("memory field without a value")?.parse()?;
        kib.checked_mul(1024).context("host memory field overflows bytes")
    };
    let (total, available) = (field("MemTotal:")?, field("MemAvailable:")?);
    ensure!(total > 0 && available <= total, "invalid MemTotal/MemAvailable for automatic prefix budget");
    Ok(HostMemory { total, available })
}

fn apply_cgroup(memory: &mut HostMemory, limit: &str, current: &str) -> Result<()> {
    if limit.trim() == "max" { return Ok(()) }
    let (limit, current): (u64, u64) = (limit.trim().parse()?, current.trim().parse()?);
    memory.total = memory.total.min(limit);
    memory.available = memory.available.min(limit.saturating_sub(current));
    Ok(())
}

/// Two retained banks plus one incoming snapshot. Count slab fragmentation
/// separately per class, as the pinned pool does; duplicated GPU copies do
/// not create extra logical retained entries.
pub(super) fn retained_bytes(layout: FamilyLayout, entries: usize, max_context: usize, chunk: u64) -> Result<u64> {
    ensure!(layout.page_rows > 0 && layout.page_bytes > 0 && max_context > 0 && chunk > 0,
        "invalid family geometry for automatic host prefix budget");
    let snapshots = (entries as u64).checked_mul(2).and_then(|v| v.checked_add(1))
        .context("prefix snapshot count overflow")?;
    let pages = (max_context as u64).div_ceil(layout.page_rows as u64).checked_mul(snapshots)
        .context("prefix snapshot page count overflow")?;
    let mut chunks = 0u64;
    for (slabs, bytes) in [(pages, layout.page_bytes as u64), (snapshots, layout.mark_bytes.max(1) as u64),
        (snapshots, layout.draft_bytes as u64)] {
        if bytes == 0 { continue }
        let per_chunk = chunk / bytes;
        ensure!(per_chunk > 0, "prefix slab {bytes} exceeds pinned chunk {chunk}");
        chunks = chunks.checked_add(slabs.div_ceil(per_chunk)).context("prefix chunk count overflow")?;
    }
    chunks.checked_mul(chunk).context("automatic host prefix budget overflow")
}

pub(super) fn automatic_bytes(required: u64, chunk: u64, memory: HostMemory, headroom: u64) -> u64 {
    let available = memory.available.saturating_sub(headroom.max(memory.total / 10));
    let ceiling = available / chunk * chunk;
    required.min(ceiling)
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    #[test]
    fn auto_respects_available_memory_headroom_and_nested_cgroup_pressure() -> Result<()> {
        let mut memory = meminfo("MemTotal: 134217728 kB\nMemAvailable: 100663296 kB\n")?;
        assert_eq!(automatic_bytes(200 * GIB, GIB, memory, 8 * GIB), 83 * GIB);
        apply_cgroup(&mut memory, &(64 * GIB).to_string(), &(40 * GIB).to_string())?;
        assert_eq!(automatic_bytes(200 * GIB, GIB, memory, 8 * GIB), 16 * GIB);
        apply_cgroup(&mut memory, &(48 * GIB).to_string(), &(44 * GIB).to_string())?;
        assert_eq!(automatic_bytes(200 * GIB, GIB, memory, 8 * GIB), 0);
        apply_cgroup(&mut memory, "max", "0")?;
        assert_eq!(memory.available, 4 * GIB);
        assert!(meminfo("MemTotal: 100 kB\n").is_err());
        assert!(apply_cgroup(&mut memory, "broken", "0").is_err());
        Ok(())
    }

    #[test]
    fn auto_never_allocates_more_than_full_retention_or_a_fractional_chunk() -> Result<()> {
        let layout = FamilyLayout { page_rows: 64, pages: 100, page_bytes: 512, mark_bytes: 140,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT, mark_store: Default::default(), page_owners: Default::default() };
        // Five snapshots: 10 page slabs in five chunks; five marks in one.
        assert_eq!(retained_bytes(layout, 2, 65, 1024)?, 6 * 1024);
        let memory = HostMemory { total: 10000, available: 9900 };
        assert_eq!(automatic_bytes(6 * 1024, 1024, memory, 0), 6 * 1024);
        assert_eq!(automatic_bytes(100 * 1024, 1024, memory, 0), 8 * 1024);
        assert!(retained_bytes(layout, usize::MAX, usize::MAX, 1024).is_err());
        Ok(())
    }
}
