//! Canonical serving decode shapes and exact-key graph ownership.
mod bank;
mod owner;
mod native;
mod watch;
pub(crate) use watch::CaptureWatch;
pub(crate) use native::{LayerGraphs, RowGraphs};
pub(crate) use bank::{GraphBank, GraphDecision, GraphPolicy, GraphStats};
pub(crate) use owner::{fatal_drain, GraphOwner};

pub(crate) fn fatal_drain(result: anyhow::Result<()>, site: &str) {
    if let Err(error) = result {
        tracing::error!(%error, site, "graph storage drain failed; aborting before storage release");
        // A failed drain cannot prove captured pointers are no longer in use.
        std::process::abort();
    }
}

pub(crate) const ROW_BUCKETS: [usize; 4] = [1, 4, 16, 64];

pub(crate) fn row_bucket(rows: usize) -> usize {
    ROW_BUCKETS.into_iter().find(|&bucket| bucket >= rows).unwrap_or(rows)
}

/// A projection's inclusive skinny-row crossover from the pinned kernel source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProjectionThreshold {
    pub name: &'static str,
    pub skinny_rows: usize,
}

/// Padding must never move a real row batch across a registered arithmetic route.
pub(crate) fn check_bucket_thresholds(buckets: &[usize], projections: &[ProjectionThreshold]) -> anyhow::Result<()> {
    anyhow::ensure!(!buckets.is_empty(), "decode bucket set is empty");
    let mut previous = 0;
    for &bucket in buckets {
        anyhow::ensure!(bucket > previous, "decode buckets must be positive and strictly increasing");
        for projection in projections {
            let threshold = projection.skinny_rows;
            anyhow::ensure!(!(previous < threshold && threshold < bucket),
                "projection {} skinny threshold {threshold} straddled by bucket {previous}..{bucket}", projection.name);
        }
        previous = bucket;
    }
    Ok(())
}

/// No KV/index/KDA storage writes, no MLA keys, and an independent conv sequence.
/// Families append these sentinels to their own tables and ignore padded logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaskedRow {
    pub position: i64,
    pub kv_slot: i64,
    pub pool_slot: i64,
    pub state_slot: i32,
    pub seq_first: i32,
    pub cache_length: i32,
}

pub(crate) fn masked_row(row: usize) -> MaskedRow {
    MaskedRow { position: -1, kv_slot: -1, pool_slot: -1, state_slot: -1,
        seq_first: row as i32, cache_length: 0 }
}

/// Real-row expert work must finish before clearing the padding tail: quantizers
/// may borrow the output as scratch. Families own the buffers and stream ordering.
pub(crate) fn real_row_moe(
    real: usize, bucket: usize,
    run: impl FnOnce(usize) -> anyhow::Result<()>,
    clear: impl FnOnce(std::ops::Range<usize>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    anyhow::ensure!(real > 0 && real <= bucket, "invalid real-row MoE extent {real}/{bucket}");
    run(real)?;
    if real < bucket { clear(real..bucket)?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn skinny_thresholds_fail_closed_with_projection_name() {
        use super::*;
        let projections = [ProjectionThreshold { name: "ple.kv", skinny_rows: 24 },
            ProjectionThreshold { name: "attention.in", skinny_rows: 8 },
            ProjectionThreshold { name: "hc.down", skinny_rows: 160 }];
        check_bucket_thresholds(&[1, 4, 8, 16], &projections).unwrap();
        check_bucket_thresholds(&[2, 4, 8, 16, 24, 32, 64], &projections).unwrap();
        assert!(check_bucket_thresholds(&[1, 4, 16], &projections).unwrap_err().to_string().contains("attention.in"));
        assert!(check_bucket_thresholds(&[2, 4, 8, 16, 32, 64], &projections).unwrap_err().to_string().contains("ple.kv"));
        for buckets in [&[][..], &[0][..], &[4, 4][..], &[8, 4][..]] {
            assert!(check_bucket_thresholds(buckets, &[]).is_err());
        }
        check_bucket_thresholds(&[8, 24, 64], &projections).unwrap();
        check_bucket_thresholds(&[1, 4, 16], &[]).unwrap();
    }

    #[test]
    fn real_row_experts_precede_tail_clear() {
        let events = std::cell::RefCell::new(Vec::new());
        super::real_row_moe(17, 32, |rows| { events.borrow_mut().push(("run", rows, rows)); Ok(()) },
            |tail| { events.borrow_mut().push(("clear", tail.start, tail.end)); Ok(()) }).unwrap();
        assert_eq!(*events.borrow(), [("run", 17, 17), ("clear", 17, 32)]);
        super::real_row_moe(8, 8, |_| Ok(()), |_| panic!("no tail")).unwrap();
        assert!(super::real_row_moe(0, 8, |_| panic!("invalid"), |_| panic!("invalid")).is_err());
        assert!(super::real_row_moe(9, 8, |_| panic!("invalid"), |_| panic!("invalid")).is_err());
        assert!(super::real_row_moe(3, 4, |_| anyhow::bail!("expert failure"),
            |_| panic!("cannot clear failed work")).is_err());
    }

    #[test]
    fn canonical_rows_and_mask() {
        use super::*;
        for (rows, bucket) in [(1, 1), (3, 4), (4, 4), (10, 16), (16, 16), (17, 64), (64, 64)] {
            assert_eq!(row_bucket(rows), bucket);
        }
        let row = masked_row(10);
        assert_eq!((row.position, row.kv_slot, row.pool_slot, row.state_slot, row.cache_length), (-1, -1, -1, -1, 0));
        assert_eq!(row.seq_first, 10);
    }
}
