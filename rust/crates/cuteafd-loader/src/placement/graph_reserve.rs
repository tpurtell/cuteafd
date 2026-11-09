//! A graph arena is a named owner, independent of GPU_RESERVE and KV policy.
use cuteafd_core::memory_policy::Size;

/// Per-device prewarm base plus extra capture headroom from the family list.
pub fn glm_default() -> (u64, u64) {
    let list = crate::serving_capacity::glm_graphs::glm_prewarm_list();
    (list.reserve_base_bytes, list.reserve_extra_bytes)
}
/// V4.1 bootstrap: the #1635 census10 measured lazy graph growth is the arena
/// base; the extra is one capture quantum until the trace card measures the
/// largest single capture on this family's qualification config.
pub fn v41_default() -> (u64, u64) {
    let list = crate::serving_capacity::v41_graphs::v41_prewarm_list();
    (list.reserve_base_bytes, list.reserve_extra_bytes)
}
pub fn source(id: &str) -> String { format!("graph reserve: {id}") }
pub fn is_source(source: &str) -> bool { source.starts_with("graph reserve: ") || source.starts_with("startup graphs: ") }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved { pub bytes: u64, pub source: &'static str }

/// A bare size overrides each coordinator device, never splits across ranks.
/// GPU selectors override `gpu0`, `gpu1`, ...; Spark overrides are scalar.
pub fn resolve(text: Option<&str>, rank: usize, spark: bool, base: u64, extra: u64) -> Result<Resolved, String> {
    let default = || base.checked_add(extra).map(|bytes| Resolved { bytes, source: "family default" })
        .ok_or_else(|| "graph reserve overflows".into());
    let Some(text) = text.filter(|text| !text.trim().is_empty()) else { return default() };
    if !text.contains('=') {
        let size = text.parse::<Size>().map_err(|e| e.to_string())?.0;
        return Ok(Resolved { bytes: size, source: "user override" });
    }
    if spark { return Err("SPARK_GRAPH_RESERVE accepts one size for every worker".into()); }
    let target = format!("gpu{rank}");
    let mut seen = std::collections::HashSet::new();
    let mut selected = None;
    for item in text.split(',') {
        let (key, value) = item.trim().split_once('=').ok_or("expected graph reserve selector=size")?;
        if key.strip_prefix("gpu").and_then(|n| n.parse::<usize>().ok()).is_none() {
            return Err(format!("invalid graph reserve selector {key}"));
        }
        if !seen.insert(key) { return Err(format!("duplicate graph reserve selector {key}")); }
        let size = value.parse::<Size>().map_err(|e| e.to_string())?.0;
        if key == target { selected = Some(Resolved { bytes: size, source: "user override" }); }
    }
    selected.map_or_else(default, Ok)
}

pub fn current(rank: usize, spark: bool, base: u64, extra: u64) -> Result<Resolved, String> {
    resolve(std::env::var(if spark { "CUTEAFD_SPARK_GRAPH_RESERVE" } else { "CUTEAFD_GPU_GRAPH_RESERVE" }).ok().as_deref(), rank, spark, base, extra)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reserve_selectors_are_per_device_and_independent_of_gpu_reserve() {
        assert_eq!(resolve(None, 0, false, 100, 20).unwrap().bytes, 120);
        let text = Some("gpu0=3GiB,gpu1=2GiB");
        assert_eq!(resolve(text, 0, false, 100, 20).unwrap().bytes, 3 << 30);
        assert_eq!(resolve(text, 1, false, 100, 20).unwrap().bytes, 2 << 30);
        assert_eq!(resolve(Some("512MiB"), 9, true, 100, 20).unwrap().bytes, 512 << 20);
        assert_eq!(resolve(Some("0B"), 1, false, 100, 20).unwrap().bytes, 0);
        assert_eq!(resolve(Some("1GiB"), 1, true, 100, 20).unwrap().bytes, 1 << 30);
        assert!(resolve(Some("gpu0=1GiB,gpu0=2GiB"), 0, false, 0, 0).is_err());
        assert!(resolve(Some("spark=512MiB"), 0, false, 0, 0).is_err());
        assert!(resolve(text, 0, true, 0, 0).is_err());
        assert_eq!(resolve(text, 2, false, 300, 20).unwrap().bytes, 320);
        for rank in [0, 1, 7] {
            assert_eq!(resolve(Some("2GiB"), rank, false, 0, 0).unwrap().bytes, 2 << 30);
            assert_eq!(resolve(Some("1GiB"), rank, true, 0, 0).unwrap().bytes, 1 << 30);
        }
    }
}
