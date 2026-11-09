//! A graph arena is a named owner, independent of GPU_RESERVE and KV policy.
use crate::serving_capacity::glm_graphs::GlmPrewarmFamily;
use cuteafd_core::memory_policy::{Size, SizeError};

/// Trace-derived prewarm arena and its allocation tail. Together these are the base.
pub fn glm_default(family: GlmPrewarmFamily, ranks: usize) -> (u64, u64) {
    let list = crate::serving_capacity::glm_graphs::glm_prewarm(family, ranks);
    (list.reserve_base_bytes, list.reserve_extra_bytes)
}
/// Kimi K3 selects its arena by coordinator layout (rank count): TP2 captures a
/// different inventory than a single coordinator.
pub fn kimi_default(ranks: usize) -> (u64, u64) {
    crate::serving_capacity::kimi_prewarm::kimi_reserve(ranks)
}
pub fn source(id: &str) -> String { format!("graph reserve: {id}") }
pub fn is_source(source: &str) -> bool { source.starts_with("graph reserve: ") || source.starts_with("startup graphs: ") }

const FAMILY: &str = "family default";
const USER: &str = "user override";
const DEFAULT_EXTRA: u64 = 500_000_000;

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
            cap_source: "not applicable", effective_extra_bytes: 0 });
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
        cap_bytes, cap_source: if cap.is_some() { USER } else { FAMILY }, effective_extra_bytes })
}

pub fn resolve(text: Option<&str>, rank: usize, spark: bool, base: u64, tail: u64) -> Result<Resolved, Error> {
    resolve_with_extra(text, None, None, rank, spark, base, tail)
}

pub fn current(rank: usize, spark: bool, base: u64, tail: u64) -> Result<Resolved, Error> {
    let read = |key: &'static str| match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::Environment { key }),
    };
    let text = read(if spark { "CUTEAFD_SPARK_GRAPH_RESERVE" } else { "CUTEAFD_GPU_GRAPH_RESERVE" })?;
    let extra = if spark { None } else { read("CUTEAFD_GPU_GRAPH_RESERVE_EXTRA")? };
    let cap = if spark { None } else { read("CUTEAFD_GPU_GRAPH_RESERVE_EXTRA_CAP")? };
    resolve_with_extra(text.as_deref(), extra.as_deref(), cap.as_deref(), rank, spark, base, tail)
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
    fn defaults_bound_shared_headroom_by_the_trace_base() {
        for (base, charged) in [(109_000_000, 218_000_000), (5_100_000_000, 5_600_000_000)] {
            let policy = resolve(None, 0, false, base, 0).unwrap();
            assert_eq!(policy.bytes, charged);
            assert_eq!(policy.extra_bytes, 500_000_000);
            assert_eq!(policy.cap_bytes, Some(base));
            assert_eq!(policy.source, "family default with extra");
        }
        let (base, tail) = glm_default(GlmPrewarmFamily::Glm5, 1);
        assert_eq!(resolve(None, 0, false, base, tail).unwrap().bytes, 675_282_944);
    }
    #[test]
    fn ml4_layout_bases_get_shared_headroom() {
        for (ranks, expected_base) in [(1, 715_128_832), (2, 914_358_272)] {
            let (base, tail) = glm_default(GlmPrewarmFamily::MistralLarge4, ranks);
            let policy = resolve(None, 0, false, base, tail).unwrap();
            assert_eq!(policy.base_bytes, expected_base);
            assert_eq!(policy.bytes, expected_base + 500_000_000);
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
}
