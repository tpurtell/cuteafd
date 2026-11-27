//! Dense Qwen's single-RTX decode working set; never borrow another checkpoint's trace.
use crate::families::{qwen3_5::Qwen35Config, qwen4::Qwen4KvCache};
use serde::Deserialize;

pub const BOOTSTRAP_BYTES: u64 = 2 << 20;

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

/// `--decode-graphs` (default on): off charges nothing.
pub fn configured_reserve(variant: &str, enabled: bool) -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    if enabled { reserve(variant) } else { Ok(crate::placement::graph_reserve::disabled(rank_of(variant))) }
}

fn rank_of(variant: &str) -> usize { if variant.ends_with("-layers-rtx2-rank1") { 1 } else { 0 } }

pub fn layout(variant: &str) -> Option<&'static Layout> {
    prewarm().layouts.iter().find(|layout| layout.variant == variant)
}

/// A traced variant charges its measured arena plus the shared extra; any other
/// variant (another checkpoint, KV format or a two-rank layout) has no prewarm or
/// pins and charges the dense Qwen × ranks envelope (2x borrows 1x per rank).
pub fn reserve(variant: &str) -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    use crate::placement::graph_reserve::{self, GraphFamily};
    let rank = rank_of(variant);
    match layout(variant) {
        Some(layout) => graph_reserve::current(rank, false, layout.reserve_base_bytes, layout.reserve_extra_bytes),
        None => graph_reserve::current_envelope(GraphFamily::Qwen35,
            if variant.contains("-layers-rtx2-") { 2 } else { 1 }, rank),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn variants_without_trace_charge_the_envelope_and_off_charges_nothing() {
        use crate::placement::graph_reserve::{envelope_bytes, GraphFamily};
        cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
            for (variant, ranks, rank) in [("h4096-l32-bf16-kvfp8-rtx1", 1, 0),
                ("h5120-l64-bf16-kvfp8-rtx1-layers-rtx2-rank0", 2, 0),
                ("h5120-l64-bf16-kvfp8-rtx1-layers-rtx2-rank1", 2, 1)] {
                assert_eq!(reserve(variant).unwrap().bytes, envelope_bytes(GraphFamily::Qwen35, ranks, rank));
                assert_eq!(configured_reserve(variant, false).unwrap().bytes, 0);
            }
        });
    }

    #[test]
    fn untraced_layouts_never_pin_or_borrow_a_working_set() {
        assert!(layout("unknown-rtx2").is_none());
        for layout in &prewarm().layouts {
            assert!(layout.traced || layout.entries.iter().all(|entry| !entry.pin));
        }
    }
}
