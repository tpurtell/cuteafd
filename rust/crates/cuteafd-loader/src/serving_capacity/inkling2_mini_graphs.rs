//! Shared #2697 working-set soft-cap graph reserve for Inkling 2 Mini (TP1).
//!
//! Soft cap = `W1 + max(20% W1, 100 MB, 2 * step_growth)`, rounded up to 2 MiB;
//! hard cap = `soft + 2 * capture_growth` (clamped to <= soft).
use crate::placement::graph_reserve::{self, Error, Resolved, WorkingSet};

const MIB: u64 = 1 << 20;

/// Measured per-key working set from the boot-`CONCURRENCY=64` RTX PRO 6000
/// census (`inkling2_mini.step`, monolithic 48-visit kernel-only graph,
/// `GPU_GRAPH_RESERVE=2GiB`, shipping default `SPECULATOR=dflash2`, `128/128`
/// keys prewarmed, `0` evictions, `0` misses/eager runs at C1 and C48):
/// - Decode `1..=64` (64 keys): `508 MiB` (`7.9375 MiB/key`, 3,037–3,125 nodes)
/// - Verify `1..=64` (64 keys): `506 MiB` (`7.90625 MiB/key`, 3,037–3,125 nodes)
/// - Exact 128-key union: `1,014 MiB` (`1,063,256,064 B` = `8,306,688 B/key`)
/// - Largest single capture and step growth: `10 MiB` (`10,485,760 B`; 20 of
///   128 keys crossed five 2 MiB slabs, 83 crossed four, 25 crossed three).
pub const PER_KEY_BYTES: u64 = 1_063_256_064 / 128;
pub const STEP_GROWTH_BYTES: u64 = 10 * MIB;
pub const CAPTURE_GROWTH_BYTES: u64 = 10 * MIB;

/// Resolved exact-row monolithic graph key space for Inkling 2 Mini.
///
/// Shared by admission (`cuteafd plan` and serving startup) and startup prewarm
/// so the reserve scales with `CONCURRENCY` and active verify rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySpace {
    /// Plain decode row keys `1..=decode_keys` (`CONCURRENCY` clamped to `1..=decode_rows`).
    pub decode_keys: usize,
    /// Speculative verify row keys `1..=verify_keys` (`decode_rows` with drafter, `0` without).
    pub verify_keys: usize,
}

impl KeySpace {
    pub const fn for_config(max_sequences: u64, decode_rows: usize, has_draft: bool) -> Self {
        let cap = if decode_rows == 0 { 1 } else { decode_rows };
        let decode_keys = if max_sequences == 0 {
            1
        } else if max_sequences > cap as u64 {
            cap
        } else {
            max_sequences as usize
        };
        Self {
            decode_keys,
            verify_keys: if has_draft { decode_rows } else { 0 },
        }
    }

    pub const fn total_keys(self) -> usize {
        self.decode_keys + self.verify_keys
    }
}

/// Derive the `#2697` working-set seed from the resolved key space, chunk-aligned
/// to `GRAPH_CHUNK` (`2 MiB`).
pub fn working_set(keys: KeySpace) -> WorkingSet {
    let total = keys.total_keys().max(1) as u64;
    let w1 = total
        .saturating_mul(PER_KEY_BYTES)
        .next_multiple_of(graph_reserve::GRAPH_CHUNK);
    WorkingSet {
        w1,
        step_growth: STEP_GROWTH_BYTES,
        capture_growth: CAPTURE_GROWTH_BYTES,
    }
}

fn resolve(
    rank: usize,
    keys: KeySpace,
    enabled: bool,
    total: Option<&str>,
    extra: Option<&str>,
    cap: Option<&str>,
) -> Result<Resolved, Error> {
    if !enabled {
        return graph_reserve::resolve(Some("0"), rank, false, 0, 0);
    }
    graph_reserve::resolve_working_set(total, extra, cap, rank, working_set(keys))
}

pub fn policy(rank: usize, keys: KeySpace) -> Result<Resolved, Error> {
    let enabled = !std::env::var_os("CUTEAFD_INKLING2_MINI_GRAPHS").is_some_and(|on| on == "0");
    if !enabled {
        return resolve(rank, keys, false, None, None, None);
    }
    let read = |key: &'static str| match std::env::var(format!("CUTEAFD_{key}")) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::Environment { key }),
    };
    resolve(
        rank,
        keys,
        true,
        read("GPU_GRAPH_RESERVE")?.as_deref(),
        read("GPU_GRAPH_RESERVE_EXTRA")?.as_deref(),
        read("GPU_GRAPH_RESERVE_EXTRA_CAP")?.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_working_set_scales_with_concurrency_and_verify_rows() {
        // Every measured and scaled configuration uses #2697 soft = ceil2M(W1 + max(20% W1, 100 MB, 20 MiB)),
        // hard = soft + 20 MiB (capture_growth = 10 MiB).
        for (sequences, decode_rows, has_draft, total_keys, w1_mib, soft_mib, hard_mib) in [
            (1, 64, false, 1, 8, 104, 124),
            (16, 64, false, 16, 128, 224, 244),
            (64, 64, false, 64, 508, 610, 630),
            (1, 64, true, 65, 516, 620, 640),
            (16, 64, true, 80, 634, 762, 782),
            (32, 64, true, 96, 762, 916, 936),
            (48, 64, true, 112, 888, 1066, 1086),
            (64, 64, true, 128, 1014, 1218, 1238),
            (64, 128, true, 192, 1522, 1828, 1848),
        ] {
            let keys = KeySpace::for_config(sequences, decode_rows, has_draft);
            assert_eq!(keys.total_keys(), total_keys);
            let ws = working_set(keys);
            assert_eq!(ws.w1, w1_mib * MIB, "seq={sequences} rows={decode_rows} draft={has_draft}");
            assert_eq!(ws.step_growth, 10 * MIB);
            assert_eq!(ws.capture_growth, 10 * MIB);
            let p = resolve(0, keys, true, None, None, None).unwrap();
            assert_eq!(p.base_bytes, soft_mib * MIB, "seq={sequences} rows={decode_rows} draft={has_draft}");
            assert_eq!(p.growth_headroom, Some(20 * MIB));
            assert_eq!(p.bytes, hard_mib * MIB, "seq={sequences} rows={decode_rows} draft={has_draft}");
        }
    }

    #[test]
    fn mini_working_set_clamps_sequences_and_honors_overrides() {
        assert_eq!(KeySpace::for_config(0, 64, true), KeySpace { decode_keys: 1, verify_keys: 64 });
        assert_eq!(KeySpace::for_config(128, 64, true), KeySpace { decode_keys: 64, verify_keys: 64 });
        let keys = KeySpace::for_config(64, 64, true);
        assert_eq!(resolve(0, keys, false, Some("2GiB"), None, None).unwrap().bytes, 0);
        assert_eq!(resolve(0, keys, true, Some("0"), None, None).unwrap().bytes, 0);
        assert_eq!(resolve(0, keys, true, Some("gpu0=256MiB"), None, None).unwrap().bytes, 256 * MIB);
        assert!(resolve(0, keys, true, None, Some("10MB"), None).is_err());
    }
}
