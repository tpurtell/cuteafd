//! L2 prefetch of the next layer's decode weights while a one-lane decode or
//! verify step waits for its routed experts (after Hugh Madden's glm53f-afd
//! L2 prefetch, MIT, v1.1.0 13f682f).
//!
//! A decode step's MoE layer sends its rows to the expert ranks, queues the
//! shared expert, and waits: the GPU idles for most of the exchange. The
//! next layer's attention is bound by its weight reads. With `--l2-prefetch`
//! the engine queues, after the shared expert and on the step's stream, one
//! kernel that touches one byte of every 128-byte line of the first `budget`
//! bytes of the next layer's weights in the order its decode programs read
//! them (at most 16 ranges); the next layer then finds them in L2. Nothing is
//! written, so every output is bit-identical with it on or off. The launch
//! sits between captured graph segments (the exchange is never captured), so
//! graphs are unchanged. Prefill steps and passes with several lanes never
//! prefetch (other work fills the exchange there).
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

/// Ranges one launch takes.
pub(crate) const MAX_RANGES: usize = 16;
/// Bytes between touches (one per 128-byte line; each load asks for 256 B).
const STRIDE: usize = 128;

/// A device range: its first byte and length.
pub(crate) type Range = (*const c_void, usize);

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct L2PrefetchArgs {
    /// L2 prefetch of the next layer's weights during a decode step's expert
    /// exchange: off, auto (3/4 of the device L2) or a budget in MiB. The
    /// default is the family's: [`GLM_DEFAULT`] for GLM 5.3, GLM 5.3 Flash,
    /// Hy4 and Hy4-flash, [`OTHER_DEFAULT`] for MiMo, Qwen and DeepSeek V4.1.
    #[arg(long, env = "CUTEAFD_L2_PREFETCH")]
    pub l2_prefetch: Option<String>,
}

/// Default `"auto"` (3/4 of the device L2 via `cudaDevAttrL2CacheSize`):
/// GLM 5.3, GLM 5.3 Flash, Hy4 and Hy4-flash. On GLM 5.3 / GLM 5.3 Flash
/// (1 RTX PRO 6000 at 325 W + TP4, DFlash2, 3 interleaved rounds): GLM 5.3
/// weighted mix 38.53 -> 40.21 tok/s (+4.4%, every round and case); GLM 5.3
/// Flash C1 counting / code / topic 157.5 / 118.0 / 99.0 -> 180.8 / 125.3 / 106.4.
pub(crate) const GLM_DEFAULT: &str = "auto";
/// Families not yet measured on real Spark exchanges keep it off.
pub(crate) const OTHER_DEFAULT: &str = "off";

/// The prefetch plan: per MoE layer, the ranges to touch while its experts
/// are out (the next layer's, or the head's after the last).
pub(crate) struct L2Prefetch {
    pub budget: usize,
    blocks: usize,
    plans: Vec<Vec<Range>>,
}

impl L2PrefetchArgs {
    /// The budget in bytes (None: off).
    pub fn budget(&self, library: &NativeLibrary, default: &str) -> Result<Option<usize>> {
        match self.l2_prefetch.as_deref().unwrap_or(default) {
            "off" | "0" => Ok(None),
            "auto" => {
                let l2 = library.l2_cache_bytes()?;
                anyhow::ensure!(l2 > 0, "--l2-prefetch auto: the device reports no L2 size");
                Ok(Some(l2 / 4 * 3))
            }
            mib => Ok(Some(mib.parse::<usize>().context("--l2-prefetch takes off, auto or MiB")? << 20)),
        }
    }
}

/// Ranges below this are left to DRAM (they would spend a launch's range slots on a few lines).
const MIN_RANGE: usize = 64 << 10;

/// The ranges of `names` in order (via `range`), each name's E4M3 copy and
/// scales (`{name}_fp8`, `{name}_scale`) in its place when the layer has them;
/// missing names are skipped.
pub(crate) fn operands(names: &[&str], range: impl Fn(&str) -> Option<Range>) -> Vec<Range> {
    let mut out = Vec::new();
    for name in names {
        match range(&format!("{name}_fp8")) {
            Some(fp8) => out.extend([Some(fp8), range(&format!("{name}_scale"))].into_iter().flatten()),
            None => out.extend(range(name)),
        }
    }
    out
}

/// The first `budget` bytes of `ranges` (the last one cut short), at most
/// [`MAX_RANGES`] ranges, skipping ranges under 64 KiB.
pub(crate) fn prefix(ranges: &[Range], budget: usize) -> Vec<Range> {
    let mut out = Vec::with_capacity(MAX_RANGES);
    let mut left = budget;
    for &(ptr, bytes) in ranges {
        if left == 0 || out.len() == MAX_RANGES {
            break;
        }
        if bytes < MIN_RANGE || ptr.is_null() {
            continue;
        }
        let take = bytes.min(left);
        out.push((ptr, take));
        left -= take;
    }
    out
}

impl L2Prefetch {
    /// `order[i]`: the ranges a decode step reads after MoE layer `i`'s
    /// experts, in read order.
    pub fn new(library: &NativeLibrary, budget: usize, order: &[Vec<Range>]) -> Result<Self> {
        let blocks = library.sm_count().unwrap_or(128);
        let plans: Vec<Vec<Range>> = order.iter().map(|r| prefix(r, budget)).collect();
        let covered: usize = plans.iter().map(|p| p.iter().map(|r| r.1).sum::<usize>()).max().unwrap_or(0);
        tracing::info!(budget_mib = budget >> 20, layers = plans.len(), largest_mib = covered >> 20,
            "L2 prefetch of the next layer's weights during decode exchanges");
        Ok(Self { budget, blocks, plans })
    }

    /// Queues the prefetch planned after MoE layer `index` on `stream`.
    pub fn issue(&self, library: &NativeLibrary, index: usize, stream: *mut c_void) -> Result<()> {
        let Some(plan) = self.plans.get(index).filter(|p| !p.is_empty()) else { return Ok(()) };
        // SAFETY: the ranges are the engine's resident weights, live while it runs.
        unsafe { library.l2_prefetch(plan, STRIDE, self.blocks, stream) }
    }
}

/// Benchmark only (`CUTEAFD_EMULATE_EXCHANGE_US`): microseconds a decode
/// step with local experts waits after the shared expert, as a Spark
/// exchange would, before its experts run.
pub(crate) fn emulated_exchange_us() -> u64 {
    static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *US.get_or_init(|| std::env::var("CUTEAFD_EMULATE_EXCHANGE_US").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// With CUTEAFD_EMULATE_EXCHANGE_US (benchmarks), a decode step with no
/// real exchange waits as a Spark exchange would: [`exchange_mark`] after the
/// shared expert is queued (before any prefetch), [`exchange_wait`] once the
/// prefetch is queued (the host waits for the mark, then spins).
pub(crate) struct ExchangeMark(*mut c_void);

pub(crate) fn exchange_mark(library: &NativeLibrary, stream: *mut c_void) -> Result<Option<ExchangeMark>> {
    if emulated_exchange_us() == 0 {
        return Ok(None);
    }
    mark(library, stream).map(Some)
}

/// Records the process's ordering event on `stream` (one engine thread uses it at a time).
pub(crate) fn mark(library: &NativeLibrary, stream: *mut c_void) -> Result<ExchangeMark> {
    static EVENT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let event = match EVENT.get() {
        Some(&e) => e,
        None => *EVENT.get_or_init(|| library.cuda_event_create_ordering().map_or(0, |e| e as usize)),
    };
    anyhow::ensure!(event != 0, "L2 prefetch ordering event");
    // SAFETY: records a live event on the caller's live stream.
    unsafe { library.cuda_event_record(event as *mut c_void, stream)? };
    Ok(ExchangeMark(event as *mut c_void))
}

/// Waits (host) for the queue up to `mark`.
pub(crate) fn reached(library: &NativeLibrary, mark: ExchangeMark) -> Result<()> {
    // SAFETY: the event was recorded by `mark`.
    unsafe { library.cuda_event_synchronize(mark.0) }
}

pub(crate) fn exchange_wait(library: &NativeLibrary, mark: Option<ExchangeMark>) -> Result<()> {
    if let Some(mark) = mark {
        reached(library, mark)?;
        spin(emulated_exchange_us());
    }
    Ok(())
}

/// Spins `us` microseconds (sleep is too coarse for sub-millisecond waits).
pub(crate) fn spin(us: u64) {
    let until = std::time::Instant::now() + std::time::Duration::from_micros(us);
    while std::time::Instant::now() < until {
        std::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(offset: usize) -> *const c_void {
        (0x1000_0000usize + offset) as *const c_void
    }

    #[test]
    fn operands_prefer_the_fp8_copies_in_order() {
        let have = |n: &str| match n {
            "a" => Some((at(0), 1 << 20)),
            "b" => Some((at(1 << 24), 2 << 20)),
            "b_fp8" => Some((at(2 << 24), 1 << 20)),
            "b_scale" => Some((at(3 << 24), 128 << 10)),
            _ => None,
        };
        let got = operands(&["a", "missing", "b"], have);
        assert_eq!(got, vec![(at(0), 1 << 20), (at(2 << 24), 1 << 20), (at(3 << 24), 128 << 10)]);
    }

    #[test]
    fn prefix_cuts_at_the_budget_skips_small_ranges_and_caps_the_count() {
        let ranges = vec![(at(0), 4 << 10), (at(1 << 20), 3 << 20), (at(8 << 20), 3 << 20)];
        assert_eq!(prefix(&ranges, 4 << 20), vec![(at(1 << 20), 3 << 20), (at(8 << 20), 1 << 20)]);
        let many: Vec<Range> = (0..40).map(|i| (at(i << 20), 1 << 20)).collect();
        assert_eq!(prefix(&many, 1 << 30).len(), MAX_RANGES);
    }
}
