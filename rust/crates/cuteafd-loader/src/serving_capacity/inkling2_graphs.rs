//! Measured lag-8 graph working sets, shared by Inkling admission and runtime.
use crate::placement::graph_reserve::{self, Error, Resolved};

const MIB: u64 = 1 << 20;
const GRAPH_CHUNK: u64 = 2 * MIB;

fn working_set(rank: usize, ranks: usize) -> Option<u64> {
    // EXL3 K4 MC original, 1 RTX + 4 Sparks / 2 RTX + 6 Sparks (#2643).
    match (ranks, rank) {
        (1, 0) => Some(302 * MIB),
        (2, 0) => Some(358 * MIB),
        (2, 1) => Some(144 * MIB),
        _ => None,
    }
}

fn resolve(rank: usize, ranks: usize, enabled: bool, total: Option<&str>,
    extra: Option<&str>, cap: Option<&str>) -> Result<Resolved, Error> {
    if !enabled { return Ok(graph_reserve::disabled(rank)); }
    let Some(base) = working_set(rank, ranks) else {
        return graph_reserve::resolve_envelope(total, extra, cap, rank,
            graph_reserve::envelope_bytes(graph_reserve::GraphFamily::Inkling2, ranks, rank));
    };
    let mut policy = graph_reserve::resolve_with_extra(total, extra, cap, rank, false, base, 0)?;
    // Keep explicit totals and the shared override/selector rules. An implicit
    // extra has a cap (the working set by default); use the census margin
    // instead of the shared 500 MB default. No separate allocation tail.
    if let Some(cap_bytes) = policy.cap_bytes.filter(|_| policy.extra_source == policy.base_source) {
        let margin = base.div_ceil(5).max(100_000_000).next_multiple_of(GRAPH_CHUNK);
        policy.source = "Inkling measured working set plus margin";
        policy.extra_bytes = margin;
        policy.effective_extra_bytes = margin.min(cap_bytes);
        policy.bytes = policy.base_bytes + policy.effective_extra_bytes;
    }
    Ok(policy)
}

/// `decode_graphs` is `--decode-graphs` (default on); off charges nothing.
pub fn policy(rank: usize, ranks: usize, decode_graphs: bool) -> Result<Resolved, Error> {
    if !decode_graphs {
        return resolve(rank, ranks, false, None, None, None);
    }
    let read = |key: &'static str| match std::env::var(format!("CUTEAFD_{key}")) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::Environment { key }),
    };
    resolve(rank, ranks, true, read("GPU_GRAPH_RESERVE")?.as_deref(),
        read("GPU_GRAPH_RESERVE_EXTRA")?.as_deref(), read("GPU_GRAPH_RESERVE_EXTRA_CAP")?.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_working_sets_charge_one_margin() {
        for (ranks, rank, base, total) in [(1, 0, 302, 398), (2, 0, 358, 454), (2, 1, 144, 240)] {
            let policy = resolve(rank, ranks, true, None, None, None).unwrap();
            assert_eq!(policy.base_bytes, base * MIB);
            assert_eq!(policy.effective_extra_bytes, 96 * MIB);
            assert_eq!(policy.bytes, total * MIB);
        }
    }

    #[test]
    fn graph_off_and_user_overrides_keep_shared_semantics() {
        let pick = |rank, enabled, total, extra, cap| resolve(rank, 2, enabled, total, extra, cap).unwrap().bytes / MIB;
        assert_eq!(pick(0, false, Some("4GiB"), None, None), 0);
        assert_eq!(pick(0, true, Some("0"), None, None), 0);
        assert_eq!(pick(1, true, Some("gpu0=0,gpu1=62MiB"), None, None), 62);
        assert_eq!(pick(0, true, Some("gpu1=62MiB"), None, None), 454);
        assert_eq!(pick(0, true, None, Some("gpu1=20MiB"), Some("10MiB")), 368);
        assert_eq!(pick(1, true, None, Some("20MiB"), Some("15MiB")), 159);
        assert!(resolve(0, 1, true, Some("0"), Some("30MiB"), None).is_err());
        // A rank count without a census charges the family envelope, never 0.
        assert_eq!(resolve(2, 3, true, None, None, None).unwrap().bytes, 454 * MIB);
    }
}
