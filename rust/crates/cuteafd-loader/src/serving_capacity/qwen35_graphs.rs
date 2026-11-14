//! Dense Qwen's single-RTX decode working set; never borrow another checkpoint's trace.
use crate::families::{qwen3_5::Qwen35Config, qwen4::Qwen4KvCache};
use serde::Deserialize;

pub const BOOTSTRAP_BYTES: u64 = 2 << 20;
pub const TAIL_BYTES: u64 = 2 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub struct Geometry {
    pub rows: usize,
    pub table_stride: usize,
    pub spec: bool,
    pub pin: bool,
}

#[derive(Debug, Deserialize)]
pub struct Layout {
    pub variant: String,
    pub traced: bool,
    pub reserve_base_bytes: u64,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<Geometry>,
}

#[derive(Deserialize)]
struct Prewarm { schema_version: u32, family: String, layouts: Vec<Layout> }

fn prewarm() -> &'static Prewarm {
    static LIST: std::sync::OnceLock<Prewarm> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list: Prewarm = serde_json::from_str(include_str!("qwen35_prewarm.json")).expect("Dense Qwen prewarm JSON");
        assert_eq!(list.schema_version, 1);
        assert_eq!(list.family, "qwen3_5");
        for layout in &list.layouts {
            assert!(layout.traced || layout.entries.iter().all(|entry| !entry.pin));
            assert!(layout.entries.iter().all(|entry| (1..=64).contains(&entry.rows)
                && entry.table_stride.is_power_of_two()));
        }
        list
    })
}

pub fn variant(cfg: &Qwen35Config, fp8: bool, kv: Qwen4KvCache) -> String {
    format!("h{}-l{}-{}-kv{}-rtx1", cfg.hidden, cfg.attention.len(),
        if fp8 { "fp8" } else { "bf16" }, kv)
}

pub fn enabled() -> bool {
    std::env::var("CUTEAFD_QWEN35_GRAPHS").ok().as_deref() == Some("1")
}

pub fn configured_reserve(variant: &str, enabled: bool) -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    if enabled { reserve(variant) } else {
        crate::placement::graph_reserve::resolve_with_extra(Some("0"), None, None, 0, false, 0, 0)
    }
}

pub fn layout(variant: &str) -> Option<&'static Layout> {
    prewarm().layouts.iter().find(|layout| layout.variant == variant)
}

/// Before tracing, no prewarm or pins; 64 MiB is a provisional lazy-capture
/// bootstrap (32 two-MiB driver pages), not a measured claim. Trace with an
/// explicit large reserve and replace it with the selected working set.
pub fn reserve(variant: &str) -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    let (base, tail) = layout(variant).map_or((64 << 20, TAIL_BYTES),
        |layout| (layout.reserve_base_bytes, layout.reserve_extra_bytes));
    let rank = if variant.ends_with("-layers-rtx2-rank1") { 1 } else { 0 };
    crate::placement::graph_reserve::current(rank, false, base, tail)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn untraced_layouts_never_pin_or_borrow_a_working_set() {
        assert!(layout("unknown-rtx2").is_none());
        for layout in &prewarm().layouts {
            assert!(layout.traced || layout.entries.iter().all(|entry| !entry.pin));
        }
    }
}
