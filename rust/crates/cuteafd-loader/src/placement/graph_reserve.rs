//! A graph arena is a named owner, independent of GPU_RESERVE and KV policy.
use crate::serving_capacity::glm_graphs::GlmPrewarmFamily;
use cuteafd_core::memory_policy::{Size, SizeError};

/// Measured C1 working-set inputs for GLM 5.3 / ML4 exact-row rank-local decode graphs.
pub fn glm_default(family: GlmPrewarmFamily, ranks: usize, rank: usize) -> WorkingSet {
    crate::serving_capacity::glm_graphs::glm_working_set(family, ranks, rank)
}
/// Per-device V4.1 measured C1 working set under shipping settings (rank-local
/// prewarm intact), scaled by the number of decode lanes sharing the device.
/// Passed directly to `resolve_working_set` / `current_working_set` with no
/// family-local margin or hard-extra arithmetic.
///
/// Measured from `v41-strided-lanes-ab` reconciliation logs + C1 bands and the
/// five Nov 18 V4.1 graph traces (`v41-width7-layout1`, `v41-width7-layout2`,
/// `v41-history-final`; `largest_bytes = 4 MiB` across all five traces):
/// - 1x RTX GPU 0: `W1_lane = 984 MiB` (944 MiB prewarm/lane + 38 MiB C1
///   `backbone.shared` + 2 MiB 32K index), `shared = 80 MiB` (14 MiB ready
///   shared + 66 MiB `first_request` prefill), `step_growth = 27 MiB/lane`.
/// - 2x RTX GPU 0: `W1_lane = 592 MiB` (590 MiB prewarm/lane + 2 MiB 32K index),
///   `shared = 12 MiB`, `step_growth = 11 MiB/lane`.
/// - 2x RTX GPU 1: `W1_lane = 594 MiB` (592 MiB prewarm/lane + 2 MiB 32K index),
///   `shared = 14 MiB` (incl. 6 MiB spanning `dspark_chain`), `step_growth = 11 MiB/lane`.
/// An unknown layout returns `None` so it never borrows another layout's inventory.
/// For 128 verify rows, the Nov 25 no-churn StridedPLE census (#2460,
/// `v41-r128-nochurn-traces`) measured one-lane charged high-water of
/// 3286 / 1348 / 1356 MiB (TP1 / TP2 GPU0 / GPU1). Use that retained
/// working set, not the C1-only boundary, to prevent churn at C8/C16.
/// Remove the existing shared allowance before scaling per-lane storage;
/// scale the step estimate by the same measured ratio. Interpolate between
/// the two measured row bands, extrapolate above 128 (not a measured band).
/// Single-capture high-water is still 4 MiB on all three devices.
pub fn v41_working_set(ranks: usize, rank: usize, lanes: u8, decode_rows: u32) -> Option<WorkingSet> {
    const MIB: u64 = 1 << 20;
    let lanes = u64::from(lanes.max(1));
    let (lane64, shared, step64, total128) = match (ranks, rank) {
        (1, 0) => (984u64, 80, 27, 3286),
        (2, 0) => (592, 12, 11, 1348),
        (2, 1) => (594, 14, 11, 1356),
        _ => return None,
    };
    let additional = (total128 - shared - lane64) * u64::from(decode_rows.saturating_sub(64));
    let lane = lane64 * MIB + (additional * MIB).div_ceil(64);
    let step = (step64 * lane).div_ceil(lane64);
    Some(WorkingSet { w1: lanes * lane + shared * MIB, step_growth: lanes * step, capture_growth: 4 * MIB })
}

/// Resolve V4.1's reserve through the shared `current_working_set` rule for
/// measured layouts, or `current(rank, false, 0, 0)` for unknown layouts.
pub fn v41_current(ranks: usize, rank: usize, lanes: u8, decode_rows: u32) -> Result<Resolved, Error> {
    match v41_working_set(ranks, rank, lanes, decode_rows) {
        Some(measured) => current_working_set(rank, measured),
        None => current(rank, false, 0, 0),
    }
}

/// Kimi K3 selects its arena by checkpoint core format and coordinator rank count: TP2 captures a
/// different inventory than a single coordinator.
pub fn kimi_default(core: crate::serving_capacity::kimi_prewarm::KimiCoreFormat, ranks: usize, rank: usize) -> (u64, u64) {
    crate::serving_capacity::kimi_prewarm::kimi_reserve(core, ranks, rank)
}
pub fn source(id: &str) -> String { format!("graph reserve: {id}") }
pub fn is_source(source: &str) -> bool { source.starts_with("graph reserve: ") || source.starts_with("startup graphs: ") }

const FAMILY: &str = "family default";
const USER: &str = "user override";
const DEFAULT_EXTRA: u64 = 500_000_000;
pub const GRAPH_CHUNK: u64 = 2 << 20;
const WORKING_SET_SLACK_MIN_BYTES: u128 = 100_000_000;

/// One chunk or twice the largest single capture, rounded up once. The caller
/// clamps the result to half the hard reserve.
pub fn capture_growth_headroom(growth: u64, chunk: u64) -> u64 {
    let chunk = u128::from(chunk);
    u64::try_from((2 * u128::from(growth)).div_ceil(chunk).max(1) * chunk).unwrap_or(u64::MAX)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("{key}: {error}")]
    Size { key: &'static str, error: SizeError },
    #[error("{key}: invalid graph reserve selector {selector:?}; expected gpuN=value")]
    Selector { key: &'static str, selector: String },
    #[error("{key}: duplicate graph reserve selector {selector}")]
    Duplicate { key: &'static str, selector: String },
    #[error("SPARK_GRAPH_RESERVE accepts one size for every worker")]
    SparkSelector,
    #[error("{key}: graph reserve overflows")]
    Overflow { key: &'static str },
    #[error("{key}: environment value is not UTF-8")]
    Environment { key: &'static str },
    #[error("GPU_GRAPH_RESERVE is a user total and conflicts with {key}")]
    Conflict { key: &'static str },
    #[error("{key} is not supported with working-set graph sizing; use GPU_GRAPH_RESERVE for the total")]
    WorkingSetExtra { key: &'static str },
}

/// The admission charge and the exact inputs that produced it. `extra_bytes` is
/// the requested headroom; `effective_extra_bytes` is the headroom after capping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Resolved {
    #[serde(rename = "charged_bytes")]
    pub bytes: u64,
    #[serde(rename = "charged_source")]
    pub source: &'static str,
    pub base_bytes: u64,
    pub base_source: &'static str,
    pub extra_bytes: u64,
    pub extra_source: &'static str,
    pub cap_bytes: Option<u64>,
    pub cap_source: &'static str,
    pub effective_extra_bytes: u64,
    /// Census seed for the runtime's adaptive margin; absent for prewarm reserves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub growth_headroom: Option<u64>,
}

#[derive(Clone, Copy)]
enum Value { Bytes(u64), Percent(u64) }
impl Value {
    fn parse(key: &'static str, text: &str, percentage: bool) -> Result<Self, Error> {
        if let Some(number) = text.trim().strip_suffix('%').filter(|_| percentage) {
            // Size's decimal grammar gives millionths without floating-point rounding.
            let parts = format!("{}MB", number.trim()).parse::<Size>()
                .map_err(|error| Error::Size { key, error })?.0;
            Ok(Self::Percent(parts))
        } else {
            text.parse::<Size>().map(|size| Self::Bytes(size.0))
                .map_err(|error| Error::Size { key, error })
        }
    }
    fn bytes(self, key: &'static str, base: u64) -> Result<u64, Error> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Percent(parts) => u64::try_from(u128::from(base) * u128::from(parts) / 100_000_000)
                .map_err(|_| Error::Overflow { key }),
        }
    }
}

/// A scalar applies independently to every GPU; unspecified selectors retain
/// that device's default. Validate every entry, including unselected devices.
fn select(text: Option<&str>, rank: usize, spark: bool, key: &'static str,
    percentage: bool) -> Result<Option<Value>, Error> {
    let Some(text) = text.filter(|text| !text.trim().is_empty()) else { return Ok(None) };
    if !text.contains('=') { return Value::parse(key, text, percentage).map(Some) }
    if spark { return Err(Error::SparkSelector) }
    let mut seen = std::collections::HashSet::new();
    let mut selected = None;
    for item in text.split(',') {
        let invalid = || Error::Selector { key, selector: item.into() };
        let (selector, value) = item.trim().split_once('=').ok_or_else(invalid)?;
        let index = selector.strip_prefix("gpu").and_then(|n| n.parse::<usize>().ok()).ok_or_else(invalid)?;
        if selector != format!("gpu{index}") { return Err(invalid()) }
        if !seen.insert(index) { return Err(Error::Duplicate { key, selector: selector.into() }) }
        let value = Value::parse(key, value, percentage)?;
        if index == rank { selected = Some(value) }
    }
    Ok(selected)
}

/// Resolve the trace-derived base plus shared bounded runtime headroom. Spark
/// workers retain the scalar reserve policy and never acquire an extra owner.
pub fn resolve_with_extra(text: Option<&str>, extra_text: Option<&str>, cap_text: Option<&str>,
    rank: usize, spark: bool, base: u64, tail: u64) -> Result<Resolved, Error> {
    let key = if spark { "SPARK_GRAPH_RESERVE" } else { "GPU_GRAPH_RESERVE" };
    let selected = select(text, rank, spark, key, false)?;
    let base_bytes = match selected {
        Some(value) => value.bytes(key, 0)?,
        None => base.checked_add(tail).ok_or(Error::Overflow { key })?,
    };
    let base_source = if selected.is_some() { USER } else { FAMILY };
    let extra = if spark { None } else { select(extra_text, rank, false, "GPU_GRAPH_RESERVE_EXTRA", true)? };
    let cap = if spark { None } else { select(cap_text, rank, false, "GPU_GRAPH_RESERVE_EXTRA_CAP", true)? };
    if selected.is_some() && !spark {
        for (key, value) in [("GPU_GRAPH_RESERVE_EXTRA", extra), ("GPU_GRAPH_RESERVE_EXTRA_CAP", cap)] {
            if value.is_some() { return Err(Error::Conflict { key }) }
        }
    }
    if selected.is_some() || spark {
        return Ok(Resolved { bytes: base_bytes, source: if selected.is_some() { "user total" } else { FAMILY },
            base_bytes, base_source, extra_bytes: 0, extra_source: "not applicable", cap_bytes: None,
            cap_source: "not applicable", effective_extra_bytes: 0, growth_headroom: None });
    }
    let extra_bytes = extra.unwrap_or(Value::Bytes(DEFAULT_EXTRA))
        .bytes("GPU_GRAPH_RESERVE_EXTRA", base_bytes)?;
    let cap_bytes = match cap {
        Some(value) => Some(value.bytes("GPU_GRAPH_RESERVE_EXTRA_CAP", base_bytes)?),
        None if extra.is_none() => Some(base_bytes),
        None => None,
    };
    let effective_extra_bytes = cap_bytes.map_or(extra_bytes, |cap| extra_bytes.min(cap));
    let bytes = base_bytes.checked_add(effective_extra_bytes).ok_or(Error::Overflow { key })?;
    Ok(Resolved { bytes, source: "family default with extra",
        base_bytes, base_source, extra_bytes, extra_source: if extra.is_some() { USER } else { FAMILY },
        cap_bytes, cap_source: if cap.is_some() { USER } else { FAMILY }, effective_extra_bytes,
        growth_headroom: None })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkingSet {
    pub w1: u64,
    pub step_growth: u64,
    pub capture_growth: u64,
}

/// Start donor pressure at `W1 + max(20% of W1, 100 MB, 2x step growth)`, rounded
/// up to 2 MiB. Charge that soft cap plus the single-capture growth headroom.
pub fn resolve_working_set(text: Option<&str>, extra_text: Option<&str>, cap_text: Option<&str>,
    rank: usize, measured: WorkingSet) -> Result<Resolved, Error> {
    let key = "GPU_GRAPH_RESERVE";
    let selected = select(text, rank, false, key, false)?;
    for (key, text) in [("GPU_GRAPH_RESERVE_EXTRA", extra_text), ("GPU_GRAPH_RESERVE_EXTRA_CAP", cap_text)] {
        if select(text, rank, false, key, true)?.is_some() {
            return Err(Error::WorkingSetExtra { key });
        }
    }
    let headroom = capture_growth_headroom(measured.capture_growth, GRAPH_CHUNK);
    if let Some(value) = selected {
        let bytes = value.bytes(key, 0)?;
        let margin = headroom.min(bytes / 2);
        return Ok(Resolved {
            bytes, source: "user total", base_bytes: bytes - margin,
            base_source: USER, extra_bytes: 0, extra_source: "not applicable",
            cap_bytes: None, cap_source: "not applicable", effective_extra_bytes: 0,
            growth_headroom: Some(margin),
        });
    }
    let w1 = u128::from(measured.w1);
    let chunk = u128::from(GRAPH_CHUNK);
    let slack = w1.div_ceil(5).max(WORKING_SET_SLACK_MIN_BYTES).max(2 * u128::from(measured.step_growth));
    let soft = u64::try_from((w1 + slack).div_ceil(chunk) * chunk).map_err(|_| Error::Overflow { key })?;
    // Margin <= hard/2 is equivalent to margin <= soft, before adding it.
    let margin = headroom.min(soft);
    let bytes = soft.checked_add(margin).ok_or(Error::Overflow { key })?;
    Ok(Resolved {
        bytes, source: "working-set reserve", base_bytes: soft, base_source: FAMILY,
        extra_bytes: headroom, extra_source: FAMILY, cap_bytes: Some(bytes / 2), cap_source: FAMILY,
        effective_extra_bytes: margin, growth_headroom: Some(margin),
    })
}

pub fn current_working_set(rank: usize, measured: WorkingSet) -> Result<Resolved, Error> {
    let text = read("GPU_GRAPH_RESERVE")?;
    let extra = read("GPU_GRAPH_RESERVE_EXTRA")?;
    let cap = read("GPU_GRAPH_RESERVE_EXTRA_CAP")?;
    resolve_working_set(text.as_deref(), extra.as_deref(), cap.as_deref(), rank, measured)
}

pub fn resolve(text: Option<&str>, rank: usize, spark: bool, base: u64, tail: u64) -> Result<Resolved, Error> {
    resolve_with_extra(text, None, None, rank, spark, base, tail)
}

/// Physical capture floor, independent of graph ownership and GPU_RESERVE.
pub fn physical_floor(text: Option<&str>, rank: usize, memory_padding: u64) -> Result<u64, Error> {
    let key = "GRAPH_DISABLE_LOW_MEM";
    select(text, rank, false, key, false)?.map_or(Ok(memory_padding / 2), |value| value.bytes(key, 0))
}

pub fn current(rank: usize, spark: bool, base: u64, tail: u64) -> Result<Resolved, Error> {
    let text = read(if spark { "SPARK_GRAPH_RESERVE" } else { "GPU_GRAPH_RESERVE" })?;
    let extra = if spark { None } else { read("GPU_GRAPH_RESERVE_EXTRA")? };
    let cap = if spark { None } else { read("GPU_GRAPH_RESERVE_EXTRA_CAP")? };
    resolve_with_extra(text.as_deref(), extra.as_deref(), cap.as_deref(), rank, spark, base, tail)
}

fn read(key: &'static str) -> Result<Option<String>, Error> {
    match std::env::var(format!("CUTEAFD_{key}")) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::Environment { key }),
    }
}

/// Select whole prewarm entries against every device's base. The caller supplies
/// cumulative arena estimates so shared allocator tails are charged only once.
/// Pinned entries are indivisible as a set; the remaining list is a prefix.
pub fn prewarm_selection(pinned: &[bool], bases: &[u64],
    mut fits: impl FnMut(&[usize], &[u64]) -> bool) -> Vec<usize> {
    let mut selected: Vec<_> = pinned.iter().enumerate().filter_map(|(index, pin)| pin.then_some(index)).collect();
    if bases.iter().any(|base| *base == 0) || !fits(&selected, bases) { return Vec::new() }
    for (index, pin) in pinned.iter().enumerate() {
        if *pin { continue }
        selected.push(index);
        if !fits(&selected, bases) { selected.pop(); break }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_floor_defaults_selectors_and_disable() {
        assert_eq!(physical_floor(None, 0, 100 << 20).unwrap(), 50 << 20);
        assert_eq!(physical_floor(None, 0, 0).unwrap(), 0);
        assert_eq!(physical_floor(Some("75MB"), 1, 100).unwrap(), 75_000_000);
        assert_eq!(physical_floor(Some("gpu0=0,gpu1=25MiB"), 0, 100).unwrap(), 0);
        assert_eq!(physical_floor(Some("gpu0=0,gpu1=25MiB"), 1, 100).unwrap(), 25 << 20);
        assert_eq!(physical_floor(Some("gpu0=0,gpu1=25MiB"), 2, 100).unwrap(), 50);
        for invalid in ["10%", "gpu0=1,gpu0=2", "gpu1=no", "spark=1", "gpu01=1"] {
            assert!(physical_floor(Some(invalid), 0, 100).is_err(), "{invalid}");
        }
    }

    #[test]
    fn defaults_bound_shared_headroom_by_the_trace_base() {
        for (base, charged) in [(109_000_000, 218_000_000), (5_100_000_000, 5_600_000_000)] {
            let policy = resolve(None, 0, false, base, 0).unwrap();
            assert_eq!(policy.bytes, charged);
            assert_eq!(policy.extra_bytes, 500_000_000);
            assert_eq!(policy.cap_bytes, Some(base));
            assert_eq!(policy.source, "family default with extra");
        }
        let measured = glm_default(GlmPrewarmFamily::Glm5, 1, 0);
        let policy = resolve_working_set(None, None, None, 0, measured).unwrap();
        let headroom = policy.growth_headroom.unwrap();
        assert!(headroom >= GRAPH_CHUNK && headroom % GRAPH_CHUNK == 0);
        assert!(policy.base_bytes > measured.w1 && policy.base_bytes % GRAPH_CHUNK == 0);
        assert_eq!(policy.bytes, policy.base_bytes + headroom);
    }
    #[test]
    fn ml4_layout_bases_get_working_set_margin() {
        for ranks in [1, 2] {
            for rank in 0..ranks {
                let measured = glm_default(GlmPrewarmFamily::MistralLarge4, ranks, rank);
                let policy = resolve_working_set(None, None, None, rank, measured).unwrap();
                let headroom = policy.growth_headroom.unwrap();
                assert!(headroom >= GRAPH_CHUNK && headroom % GRAPH_CHUNK == 0);
                assert!(policy.base_bytes > measured.w1 && policy.base_bytes % GRAPH_CHUNK == 0);
                assert_eq!(policy.bytes, policy.base_bytes + headroom);
            }
        }
    }
    #[test]
    fn explicit_size_or_percentage_has_no_implicit_cap() {
        for (extra, bytes) in [("500MB", 600_000_000), ("250%", 350_000_000), ("12.5%", 112_500_000)] {
            let policy = resolve_with_extra(None, Some(extra), None, 0, false, 100_000_000, 0).unwrap();
            assert_eq!(policy.bytes, bytes);
            assert_eq!(policy.cap_bytes, None);
            assert_eq!(policy.extra_source, USER);
            assert_eq!(policy.source, "family default with extra");
        }
    }
    #[test]
    fn explicit_cap_accepts_sizes_and_percentages() {
        for (cap, expected) in [("20MB", 120_000_000), ("150%", 250_000_000), ("0B", 100_000_000)] {
            let policy = resolve_with_extra(None, Some("500MB"), Some(cap), 0, false, 100_000_000, 0).unwrap();
            assert_eq!(policy.bytes, expected);
            assert_eq!(policy.cap_source, USER);
        }
        assert_eq!(resolve_with_extra(None, None, Some("50%"), 0, false, 100, 0).unwrap().bytes, 150);
    }
    #[test]
    fn selectors_and_conditional_caps_resolve_per_device() {
        let inputs = (None, Some("gpu1=200%"), Some("gpu0=100MB"));
        let first = resolve_with_extra(inputs.0, inputs.1, inputs.2, 0, false, 100_000_000, 0).unwrap();
        assert_eq!(first.bytes, 200_000_000);
        assert_eq!(first.extra_source, FAMILY);
        let second = resolve_with_extra(inputs.0, inputs.1, inputs.2, 1, false, 100_000_000, 0).unwrap();
        assert_eq!(second.bytes, 300_000_000);
        assert_eq!(second.cap_bytes, None);
        assert_eq!(resolve_with_extra(inputs.0, inputs.1, inputs.2, 2, false, 100, 20).unwrap().bytes, 240);
        assert_eq!(resolve(Some("gpu0=3GiB,gpu1=2GiB"), 1, false, 100, 20).unwrap().bytes, 2 << 30);
        assert_eq!(resolve(Some("gpu0=3GiB,gpu1=2GiB"), 2, false, 100, 20).unwrap().bytes, 240);
        for rank in [0, 1, 7] {
            assert_eq!(resolve(Some("2GiB"), rank, false, 0, 0).unwrap().bytes, 2 << 30);
            assert_eq!(resolve(Some("1GiB"), rank, true, 0, 0).unwrap().bytes, 1 << 30);
        }
        assert_eq!(resolve(Some("0B"), 0, false, 100, 20).unwrap().bytes, 0);
        assert_eq!(resolve(None, 0, true, 100, 20).unwrap().bytes, 120);
    }
    #[test]
    fn user_total_is_exact_and_conflicting_extra_is_rejected() {
        let policy = resolve(Some("128MB"), 0, false, 100, 20).unwrap();
        assert_eq!(policy.bytes, 128_000_000);
        assert_eq!(policy.source, "user total");
        assert_eq!(policy.extra_bytes, 0);
        for (extra, cap, key) in [(Some("10MB"), None, "GPU_GRAPH_RESERVE_EXTRA"),
            (None, Some("100%"), "GPU_GRAPH_RESERVE_EXTRA_CAP")] {
            assert_eq!(resolve_with_extra(Some("1GB"), extra, cap, 0, false, 100, 20), Err(Error::Conflict { key }));
        }
        // Only selected user totals conflict; other devices still use family defaults.
        assert!(resolve_with_extra(Some("gpu0=1GB"), Some("500MB"), None, 1, false, 100, 20).is_ok());
        assert_eq!(resolve(None, 0, false, 0, 0).unwrap().bytes, 0);
        assert_eq!(resolve_with_extra(None, Some("10MB"), None, 0, false, 0, 0).unwrap().bytes, 10_000_000);
    }
    #[test]
    fn prewarm_keeps_pinned_first_then_a_list_prefix() {
        let pins = [false, true, false, true, false];
        let sizes = [20, 10, 30, 10, 5];
        let fit = |selected: &[usize], bases: &[u64]| selected.iter().map(|index| sizes[*index]).sum::<u64>() <= bases[0];
        assert_eq!(prewarm_selection(&pins, &[50], fit), vec![1, 3, 0]);
        assert_eq!(prewarm_selection(&pins, &[19], fit), Vec::<usize>::new());
        assert_eq!(prewarm_selection(&pins, &[0], fit), Vec::<usize>::new());
        assert_eq!(prewarm_selection(&pins, &[100], fit), vec![1, 3, 0, 2, 4]);
        let lazy = resolve_with_extra(None, Some("10MB"), None, 0, false, 0, 0).unwrap();
        assert!(lazy.bytes > 0);
        assert!(prewarm_selection(&pins, &[lazy.base_bytes], fit).is_empty());
    }
    #[test]
    fn invalid_values_are_typed_errors() {
        for text in ["gpu0=1GiB,gpu0=2GiB", "spark=512MiB", "gpu01=1MB", "gpu0=1B,broken", "gpu0=1B,gpu1=nope", "-1MB", "10%"] {
            assert!(resolve(Some(text), 0, false, 0, 0).is_err(), "{text}");
        }
        assert!(matches!(resolve(Some("gpu0=1GiB"), 0, true, 0, 0), Err(Error::SparkSelector)));
        for text in ["-1%", "NaN%", "1.2345678%", "100MB%"] {
            assert!(resolve_with_extra(None, Some(text), None, 0, false, 100, 0).is_err(), "{text}");
        }
        assert!(matches!(resolve(None, 0, false, u64::MAX, 1), Err(Error::Overflow { .. })));
        assert!(matches!(resolve_with_extra(None, Some("100%"), None, 0, false, u64::MAX, 0), Err(Error::Overflow { .. })));
        assert!(resolve_with_extra(None, None, Some("invalid"), 0, false, 100, 0).is_err());
    }

    #[test]
    fn working_set_soft_cap_uses_maximum_of_three_branches() {
        for (w1, step_growth, soft) in [(94, 10, 190), (600, 10, 720), (200, 70, 340)] {
            let policy = resolve_working_set(None, None, None, 0, WorkingSet {
                w1: w1 << 20, step_growth: step_growth << 20, capture_growth: (4 << 20) + 1,
            }).unwrap();
            assert_eq!(policy.base_bytes, soft << 20);
            assert_eq!(policy.growth_headroom, Some(10 << 20));
            assert_eq!(policy.bytes, (soft + 10) << 20);
        }
        let tiny_growth = WorkingSet { w1: 94 << 20, step_growth: 0, capture_growth: 0 };
        let policy = resolve_working_set(None, None, None, 0, tiny_growth).unwrap();
        assert_eq!(policy.bytes, 192 << 20);
        assert_eq!(policy.growth_headroom, Some(GRAPH_CHUNK));
        let overflow = WorkingSet { w1: u64::MAX, ..tiny_growth };
        assert!(matches!(resolve_working_set(None, None, None, 0, overflow), Err(Error::Overflow { .. })));
    }

    #[test]
    fn working_set_user_total_subtracts_census_headroom_clamped_to_half() {
        let measured = WorkingSet { w1: 94 << 20, step_growth: 10 << 20, capture_growth: 4 << 20 };
        for (rank, hard, soft, headroom) in [(0, 256, 248, 8), (1, 4, 2, 2), (2, 0, 0, 0), (3, 198, 190, 8)] {
            let user = resolve_working_set(Some("gpu0=256MiB,gpu1=4MiB,gpu2=0"), None, None, rank, measured).unwrap();
            assert_eq!((user.bytes, user.base_bytes, user.growth_headroom), (hard << 20, soft << 20, Some(headroom << 20)));
        }
        for (extra, cap, key) in [(Some("10MB"), None, "GPU_GRAPH_RESERVE_EXTRA"),
            (None, Some("10%"), "GPU_GRAPH_RESERVE_EXTRA_CAP")] {
            assert_eq!(resolve_working_set(None, extra, cap, 0, measured), Err(Error::WorkingSetExtra { key }));
            assert_eq!(resolve_working_set(Some("256MiB"), extra, cap, 0, measured), Err(Error::WorkingSetExtra { key }));
        }
        assert!(resolve_working_set(None, Some("gpu1=1MB"), None, 0, measured).is_ok());
    }

    #[test]
    fn v41_working_set_uses_shared_resolver_across_layouts_and_lanes() {
        const MIB: u64 = 1 << 20;
        for (ranks, rank, lanes, w1_mib, soft_mib, hard_mib) in [
            (1, 0, 1, 1_064, 1_278, 1_286),
            (1, 0, 2, 2_048, 2_458, 2_466),
            (2, 0, 1, 604, 726, 734),
            (2, 0, 2, 1_196, 1_436, 1_444),
            (2, 1, 1, 608, 730, 738),
            (2, 1, 2, 1_202, 1_444, 1_452),
        ] {
            let measured = v41_working_set(ranks, rank, lanes, 64).unwrap();
            assert_eq!(measured.w1, w1_mib * MIB, "ranks={ranks} rank={rank} lanes={lanes}");
            let policy = resolve_working_set(None, None, None, rank, measured).unwrap();
            assert_eq!(policy.base_bytes, soft_mib * MIB);
            assert_eq!(policy.growth_headroom, Some(8 * MIB));
            assert_eq!(policy.bytes, hard_mib * MIB);
        }
        assert!(v41_working_set(3, 0, 1, 64).is_none());
    }
    #[test]
    fn v41_wide_rows_cover_no_churn_census_and_scale_only_lane_bytes() {
        const MIB: u64 = 1 << 20;
        for (ranks, rank, shared, w128) in [(1, 0, 80, 3286), (2, 0, 12, 1348), (2, 1, 14, 1356)] {
            let base = v41_working_set(ranks, rank, 1, 64).unwrap();
            let wide = v41_working_set(ranks, rank, 1, 128).unwrap();
            let two = v41_working_set(ranks, rank, 2, 128).unwrap();
            assert_eq!(wide.w1, w128 * MIB);
            assert_eq!(two.w1, 2 * wide.w1 - shared * MIB);
            assert_eq!(two.step_growth, 2 * wide.step_growth);
            assert!(wide.step_growth > base.step_growth);
            assert_eq!(wide.capture_growth, base.capture_growth);
            assert!(v41_working_set(ranks, rank, 1, 256).unwrap().w1 > wide.w1);
            let resolved = resolve_working_set(None, None, None, rank, wide).unwrap();
            assert!(resolved.base_bytes > wide.w1);
            let override_ = resolve_working_set(Some("6GiB"), None, None, rank, wide).unwrap();
            assert_eq!(override_.bytes, 6 << 30);
        }
    }

}
