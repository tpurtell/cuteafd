//! Canonical serving decode shapes and exact-key graph ownership.

pub(crate) use cuteafd_loader::serving_capacity::decode_graph_key::{
    GraphGeometry, MIN_PAGE_STRIDE, decode_page_stride, decode_strides,
};

#[derive(Clone, Copy, Default)]
struct Options { no_prewarm: bool, donor_always: bool, donor_cold_lag: u64 }
impl Options {
    fn validate(self) -> anyhow::Result<Self> {
        anyhow::ensure!(self.donor_always || self.donor_cold_lag == 0,
            "--graph-donor-cold-lag requires --graph-donor-always when non-zero");
        Ok(self)
    }
}
static OPTIONS: std::sync::OnceLock<Options> = std::sync::OnceLock::new();
pub(crate) fn configure(no_prewarm: bool, donor_always: bool, donor_cold_lag: u64) -> anyhow::Result<()> {
    let options = Options { no_prewarm, donor_always, donor_cold_lag }.validate()?;
    assert!(OPTIONS.set(options).is_ok(), "graph policy configured twice");
    Ok(())
}
pub(crate) fn prewarm_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = testing::PREWARM.with(std::cell::Cell::get) { return enabled; }
    !OPTIONS.get().copied().unwrap_or_default().no_prewarm
}

#[cfg(test)]
pub(crate) mod testing {
    use std::cell::Cell;
    thread_local! { pub(super) static PREWARM: Cell<Option<bool>> = const { Cell::new(None) }; }
    /// Restores this thread's previous prewarm policy on drop.
    pub(crate) struct PrewarmOverride(Option<bool>);
    impl Drop for PrewarmOverride {
        fn drop(&mut self) { PREWARM.with(|prewarm| prewarm.set(self.0)); }
    }
    /// `GRAPH_PREWARM=off` for this test thread only; the process policy is global.
    pub(crate) fn prewarm_off_on_this_thread() -> PrewarmOverride {
        PrewarmOverride(PREWARM.with(|prewarm| prewarm.replace(Some(false))))
    }
}
fn donor_always() -> bool { OPTIONS.get().copied().unwrap_or_default().donor_always }
fn donor_cold_lag() -> u64 { OPTIONS.get().copied().unwrap_or_default().donor_cold_lag }

/// Object-safe control surface for a family's heterogeneous graph owners.
/// The caller drains the complete submitted step before invoking `drain`.
pub(crate) trait CacheAttached {
    fn attach(&mut self, library: &cuteafd_ffi::NativeLibrary, device: i32, reserve: u64, site: device::Site) -> anyhow::Result<()>;
    fn cycle(&mut self, cycle: device::Cycle);
    fn needs_boundary(&self) -> bool;
    fn measured(&mut self, bytes: u64);
    fn tick(&mut self);
    fn lane_owner(&mut self, lane: usize);
    /// # Safety
    /// Pins released by owner drop must not own captured allocations. Their
    /// persistent backing storage outlives every in-flight graph launch.
    unsafe fn evict_persistent(&mut self);
    fn drain(&mut self, drain: &mut dyn FnMut() -> anyhow::Result<()>,
        measure: &mut dyn FnMut() -> anyhow::Result<device::ReleaseMeasurement>) -> anyhow::Result<()>;
}

// Adapters delegate control only; model geometry and device placement stay in
// the family's visitor. Both concrete and trait-object paths share bank state.
macro_rules! cache_attached_methods {
    ($field:ident) => {
        fn attach(&mut self, library: &cuteafd_ffi::NativeLibrary, device: i32, reserve: u64, site: super::device::Site) -> anyhow::Result<()> {
            self.$field.attach_device(library, device, reserve, site)
        }
        fn cycle(&mut self, cycle: super::device::Cycle) { self.$field.cache_cycle(cycle); }
        fn needs_boundary(&self) -> bool { self.$field.cache_needs_boundary() }
        fn measured(&mut self, bytes: u64) { self.$field.cache_measured(bytes); }
        fn tick(&mut self) { self.$field.cache_tick(); }
        fn lane_owner(&mut self, lane: usize) { self.$field.cache_lane_owner(lane); }
        unsafe fn evict_persistent(&mut self) {
            // SAFETY: forwarded persistent-storage contract from the visitor.
            unsafe { self.$field.cache_evict_persistent(); }
        }
        fn drain(&mut self, drain: &mut dyn FnMut() -> anyhow::Result<()>,
            measure: &mut dyn FnMut() -> anyhow::Result<super::device::ReleaseMeasurement>) -> anyhow::Result<()> {
            self.$field.cache_drain(drain, measure)
        }
    };
}
mod bank;
pub(crate) mod device;
pub(crate) mod census;
pub(crate) mod stats;
mod owner;
pub(crate) mod sites;
mod native;
mod binding;
pub(crate) use binding::BindingGraphs;
mod watch;
pub(crate) use watch::CaptureWatch;
pub(crate) use native::{LayerGraphs, RowGraphs};
pub(crate) use bank::{missing_variants, note_eager, CaptureTicket, GraphBank};
pub(crate) use owner::GraphOwner;

/// Recording runs host code without executing device work. Restore the complete
/// host checkpoint on failure before an eager retry. The recorder must end
/// capture and destroy unaccepted executables on errors; rollback changes no
/// device storage or cache policy.
pub(crate) fn capture_transaction<S: ?Sized, C, T>(
    state: &mut S, checkpoint: C,
    record: impl FnOnce(&mut S) -> anyhow::Result<T>,
    rollback: impl FnOnce(&mut S, C),
) -> anyhow::Result<T> {
    let result = record(state);
    if result.is_err() { rollback(state, checkpoint); }
    result
}

/// # Safety
/// The ticket admits this capture; the caller owns the stream and pins every
/// recorded operand through the joined drain of all consuming launches.
pub(crate) unsafe fn begin_capture(library: &cuteafd_ffi::NativeLibrary,
    stream: *mut std::ffi::c_void, ticket: &CaptureTicket) -> anyhow::Result<()> {
    ticket.require_active()?;
    // SAFETY: forwarded stream ownership and captured-storage contract.
    unsafe { library.cuda_graph_begin_capture(stream) }
}

/// # Safety
/// The caller owns capture and retains all recorded operands through replay.
pub(crate) unsafe fn end_capture(library: &cuteafd_ffi::NativeLibrary,
    stream: *mut std::ffi::c_void, ticket: &mut CaptureTicket)
    -> anyhow::Result<(*mut std::ffi::c_void, usize, usize, bool, Option<String>)> {
    // SAFETY: forwarded stream capture and operand lifetime contract.
    let source = unsafe { library.cuda_graph_capture_source(stream) }?;
    ticket.complete_source(library, source, stream)
}

/// # Safety
/// The executable comes from an attached bank. Its captured allocations and
/// stream remain live until every queued consumer has drained.
pub(crate) unsafe fn launch(library: &cuteafd_ffi::NativeLibrary,
    graph: *mut std::ffi::c_void, stream: *mut std::ffi::c_void,
    tally: Option<&device::BankLaunches>) -> anyhow::Result<()> {
    // SAFETY: forwarded executable, stream and captured-storage contract.
    unsafe { library.cuda_graph_launch(graph, stream)?; }
    if let Some(tally) = tally { tally.record(); }
    Ok(())
}

/// Asynchronously uploads an executable on `stream` ahead of its next launch.
///
/// # Safety
/// The executable comes from an attached bank owned on the current executor
/// thread, and `stream` is a live CUDA stream (or null for stream 0) on the
/// same device.
#[allow(dead_code)]
pub(crate) unsafe fn upload(library: &cuteafd_ffi::NativeLibrary,
    graph: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> anyhow::Result<()> {
    // SAFETY: forwarded executable and stream ownership contract.
    unsafe { library.cuda_graph_upload(graph, stream) }
}

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
    fn capture_transaction_commits_success_and_restores_partial_errors() {
        let mut flags = [1, 2, 3];
        let checkpoint = flags;
        let value = super::capture_transaction(&mut flags, checkpoint,
            |state| { state[0] = 9; Ok(42) }, |state, saved| *state = saved).unwrap();
        assert_eq!((value, flags), (42, [9, 2, 3]));
        let checkpoint = flags;
        let error = super::capture_transaction(&mut flags, checkpoint,
            |state| -> anyhow::Result<()> { state[1] = 8; anyhow::bail!("partial recording failed") },
            |state, saved| *state = saved).unwrap_err();
        assert_eq!(error.to_string(), "partial recording failed");
        assert_eq!(flags, checkpoint);
    }

    #[test]
    fn donor_cold_lag_requires_donor_always() {
        use super::Options;
        for donor_always in [false, true] {
            assert!(Options { no_prewarm: false, donor_always, donor_cold_lag: 0 }.validate().is_ok());
            let result = Options { no_prewarm: false, donor_always, donor_cold_lag: 16 }.validate();
            assert_eq!(result.is_ok(), donor_always);
            if let Err(error) = result {
                assert!(error.to_string().contains("--graph-donor-cold-lag requires --graph-donor-always"));
            }
        }
    }

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
