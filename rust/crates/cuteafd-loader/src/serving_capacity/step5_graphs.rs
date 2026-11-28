//! Trace-owned Step-5 row geometries shared by admission and startup.
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Step5GraphLayout {
    #[serde(rename = "fp8-2rtx-6spark")] Fp8Max,
    #[serde(rename = "nvfp4-1rtx-4spark")] Nvfp4Min,
    #[serde(rename = "exl3-k4-core-k6-1rtx-3spark")] Exl3Min,
    #[serde(rename = "exl3-k4-core-k6-2rtx-6spark")] Exl3Max,
    /// Any other EXL3 K4 core (the shipping K4-MC views) at 1 RTX + 3 Sparks /
    /// 2 RTX + 6 Sparks: no trace inventory, sized by its working-set census.
    #[serde(rename = "exl3-k4-mc-1rtx-3spark")] Exl3McMin,
    #[serde(rename = "exl3-k4-mc-2rtx-6spark")] Exl3McMax,
}
impl Step5GraphLayout {
    pub fn resolve(catalog: &crate::OfficialV41Catalog,
        contract: &crate::plan::families::step5::coordinator::CoordinatorContract,
        ranks: usize, sparks: usize) -> Option<Self> {
        use crate::formats::fp8_experts::ExpertFormat;
        if let Some(exl3) = catalog.exl3() {
            return Self::exl3_layout(exl3.decoder_tiers(), &contract.core, ranks, sparks);
        }
        match (catalog.fp8()?.format(), ranks, sparks) {
            (ExpertFormat::Fp8Block128, 2, 6) => Some(Self::Fp8Max),
            (ExpertFormat::Nvfp4, 1, 4) => Some(Self::Nvfp4Min),
            _ => None,
        }
    }
    fn exl3_layout(tiers: &[usize], core: &crate::plan::families::step5::exl3::CoreInventory,
        ranks: usize, sparks: usize) -> Option<Self> {
        // Native HQA and compact attention share geometries. Check the K6
        // target FFN/head recipe, not the optional quantized attention or K4 MTP.
        let target = core.projections.values().filter(|p| p.stem == "lm_head"
            || p.stem.strip_prefix("model.layers.").and_then(|r| r.split('.').next())
                .and_then(|l| l.parse::<usize>().ok()).is_some_and(|l| l < 92));
        let k6_core = core.get("lm_head").is_some()
            && target.clone().any(|p| !p.stem.contains(".self_attn."))
            && target.filter(|p| !p.stem.contains(".self_attn.")).all(|p| p.bits == 6);
        match (tiers, k6_core, ranks, sparks) {
            ([4, 6], true, 1, 3) => Some(Self::Exl3Min),
            ([4, 6], true, 2, 6) => Some(Self::Exl3Max),
            ([4, ..], false, 1, 3) => Some(Self::Exl3McMin),
            ([4, ..], false, 2, 6) => Some(Self::Exl3McMax),
            _ => None,
        }
    }
    fn ranks(self) -> usize {
        match self { Self::Fp8Max | Self::Exl3Max | Self::Exl3McMax => 2,
            Self::Nvfp4Min | Self::Exl3Min | Self::Exl3McMin => 1 }
    }
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphSection {
    pub layer: usize,
    pub section: String,
    pub capture_bytes: u64,
    pub replay_hits: u64,
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphRow {
    pub rows: usize,
    pub pin: bool,
    pub replay_hits: u64,
    pub capture_bytes: u64,
    pub sections: Vec<Step5GraphSection>,
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphRank {
    pub rank: usize,
    pub reserve_base_bytes: u64,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<Step5GraphRow>,
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphTrace {
    pub schema_version: u32,
    pub family: String,
    pub layout: Step5GraphLayout,
    pub ranks: Vec<Step5GraphRank>,
}
#[derive(Deserialize)]
struct List { layouts: Vec<Step5GraphTrace> }
pub fn prewarm(layout: Option<Step5GraphLayout>) -> Option<&'static Step5GraphTrace> {
    static LIST: OnceLock<List> = OnceLock::new();
    let layout = layout?;
    LIST.get_or_init(|| {
        let list: List = serde_json::from_str(include_str!("step5_prewarm.json"))
            .expect("Step-5 trace inventory");
        for trace in &list.layouts {
            assert_eq!(trace.schema_version, 1);
            assert_eq!(trace.family, "step5");
            assert_eq!(trace.ranks.len(), trace.layout.ranks());
            for (rank, data) in trace.ranks.iter().enumerate() {
                assert_eq!(data.rank, rank);
                assert_eq!(data.reserve_extra_bytes, 2 << 20);
                assert!(!data.entries.is_empty());
                for row in &data.entries {
                    assert!(row.rows > 0 && row.replay_hits > 0);
                    assert_eq!(row.capture_bytes, row.sections.iter().map(|s| s.capture_bytes).sum::<u64>());
                }
                assert_eq!(data.reserve_base_bytes,
                    data.entries.iter().map(|e| e.capture_bytes).sum::<u64>().div_ceil(2 << 20) * (2 << 20));
                assert_eq!(data.entries.iter().map(|e| (e.rows, e.pin)).collect::<Vec<_>>(),
                    trace.ranks[0].entries.iter().map(|e| (e.rows, e.pin)).collect::<Vec<_>>());
            }
        }
        list
    }).layouts.iter().find(|trace| trace.layout == layout)
}
pub fn reserve(layout: Option<Step5GraphLayout>, rank: usize) -> Option<(u64, u64)> {
    prewarm(layout).and_then(|trace| trace.ranks.get(rank))
        .map(|data| (data.reserve_base_bytes, data.reserve_extra_bytes))
}

/// Full-card working-set census of the shipping K4-MC original cards (cand
/// acc0704eb6, 2026-11-28, MTP depth 3, `baseline` + C1/C16 + post-load C1,
/// `GPU_GRAPH_RESERVE=4608MiB`, `GRAPH_PREWARM=off`, pressure-only donors, zero
/// evictions; 48 live row counts at ~52 MiB/row on 1 RTX): W1 is the full-card
/// charged high-water (the Hy4 #2803 rule), growth is half the observed margin.
/// - 1 RTX + 3 Sparks: W1 2,506 MiB, growth 56 MiB -> soft/hard 3,008/3,120 MiB.
/// - 2 RTX + 6 Sparks: W1 1,900/1,688 MiB, growth 44/40 MiB -> 2,280/2,368 and 2,026/2,106 MiB.
const CENSUS: &[(Step5GraphLayout, usize, crate::placement::graph_reserve::WorkingSet)] = {
    use crate::placement::graph_reserve::WorkingSet;
    &[
        (Step5GraphLayout::Exl3McMin, 0, WorkingSet { w1: 2506 << 20, step_growth: 56 << 20, capture_growth: 56 << 20 }),
        (Step5GraphLayout::Exl3McMax, 0, WorkingSet { w1: 1900 << 20, step_growth: 44 << 20, capture_growth: 44 << 20 }),
        (Step5GraphLayout::Exl3McMax, 1, WorkingSet { w1: 1688 << 20, step_growth: 40 << 20, capture_growth: 40 << 20 }),
    ]
};

/// The measured working set, else the traced layout's arena plus the shared
/// extra, else the Step-5 × ranks envelope.
pub fn policy(layout: Option<Step5GraphLayout>, ranks: usize, rank: usize)
    -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    use crate::placement::graph_reserve::{self, GraphFamily};
    if let Some(measured) = graph_reserve::census_working_set(rank, layout, CENSUS) {
        return graph_reserve::current_working_set(rank, measured);
    }
    match reserve(layout, rank) {
        Some((base, tail)) => graph_reserve::current(rank, false, base, tail),
        None => graph_reserve::current_envelope(GraphFamily::Step5, ranks, rank),
    }
}
/// Select complete rows against all devices; never borrow an untraced layout.
pub fn selected_rows(layout: Option<Step5GraphLayout>, max_rows: usize) -> anyhow::Result<Vec<(usize, bool)>> {
    let Some(trace) = prewarm(layout) else { return Ok(Vec::new()) };
    let candidates = trace.ranks[0].entries.iter().enumerate()
        .filter_map(|(i, row)| (row.rows <= max_rows).then_some(i)).collect::<Vec<_>>();
    let bases = trace.ranks.iter().map(|rank| {
        crate::placement::graph_reserve::current(rank.rank, false,
            rank.reserve_base_bytes, rank.reserve_extra_bytes).map(|p| p.base_bytes)
    }).collect::<Result<Vec<_>, _>>()?;
    let pins = candidates.iter().map(|&i| trace.ranks[0].entries[i].pin).collect::<Vec<_>>();
    let selected = crate::placement::graph_reserve::prewarm_selection(&pins, &bases, |selected, bases| {
        trace.ranks.iter().zip(bases).all(|(rank, &base)| {
            let bytes = selected.iter().map(|&i| rank.entries[candidates[i]].capture_bytes).sum::<u64>();
            bytes.div_ceil(2 << 20) * (2 << 20) + rank.reserve_extra_bytes <= base
        })
    });
    Ok(selected.into_iter().map(|i| {
        let row = &trace.ranks[0].entries[candidates[i]];
        (row.rows, row.pin)
    }).collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exl3_default_and_compact_attention_share_only_their_own_layout() {
        use crate::{formats::{exl3_storage::Exl3Codebook, trellis_core::{CoreProjection, CoreSplit}},
            plan::families::step5::exl3::CoreInventory};
        use Step5GraphLayout::*;
        let projection = |stem: &str, bits| CoreProjection { stem: stem.into(), bits,
            codebook: Exl3Codebook::Mcg, shape: [16, 16], stored_outputs: 16,
            split: CoreSplit::Whole, source_bytes: 256 };
        let mut core = CoreInventory::default();
        for stem in ["lm_head", "model.layers.0.mlp.gate_proj"] {
            core.projections.insert(stem.into(), projection(stem, 6));
        }
        // Native HQA has no attention trellis; compact mode adds K6 ones.
        for compact in [false, true] {
            if compact {
                let stem = "model.layers.0.self_attn.q_proj";
                core.projections.insert(stem.into(), projection(stem, 6));
            }
            assert_eq!(Step5GraphLayout::exl3_layout(&[4, 6], &core, 1, 3), Some(Exl3Min));
            assert_eq!(Step5GraphLayout::exl3_layout(&[4, 6], &core, 2, 6), Some(Exl3Max));
            assert_eq!(Step5GraphLayout::exl3_layout(&[4, 5], &core, 1, 3), None);
            assert_eq!(Step5GraphLayout::exl3_layout(&[4, 6], &core, 1, 4), None);
        }
        core.projections.get_mut("lm_head").unwrap().bits = 5;
        assert_eq!(Step5GraphLayout::exl3_layout(&[4, 6], &core, 1, 3), Some(Exl3McMin));
        assert_eq!(Step5GraphLayout::exl3_layout(&[4, 6], &core, 2, 6), Some(Exl3McMax));
        assert_eq!(Step5GraphLayout::exl3_layout(&[4], &CoreInventory::default(), 1, 3), Some(Exl3McMin));
        assert_eq!(Exl3Min.ranks(), 1);
        assert_eq!(Exl3Max.ranks(), 2);
    }
    #[test]
    fn shipped_hqa_inventories_retain_exact_rank_bases_and_default_rows() {
        cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
            for (layout, bases, rows, pins) in [
                (Step5GraphLayout::Exl3Min, vec![3_407_872_000], 58, vec![46]),
                (Step5GraphLayout::Exl3Max, vec![2_835_349_504, 2_629_828_608], 59, vec![]),
            ] {
                let trace = prewarm(Some(layout)).unwrap();
                for (rank, base) in bases.into_iter().enumerate() {
                    assert_eq!(reserve(Some(layout), rank), Some((base, 2 << 20)));
                    assert_eq!(trace.ranks[rank].entries.len(), rows);
                }
                let selected = selected_rows(Some(layout), 64).unwrap();
                assert_eq!(selected.len(), rows);
                assert_eq!(selected.iter().filter_map(|&(row, pin)| pin.then_some(row))
                    .collect::<Vec<_>>(), pins);
            }
            for (layout, rank, soft, hard) in [(Step5GraphLayout::Exl3McMin, 0, 3008, 3120),
                (Step5GraphLayout::Exl3McMax, 0, 2280, 2368), (Step5GraphLayout::Exl3McMax, 1, 2026, 2106)] {
                assert!(prewarm(Some(layout)).is_none());
                let charged = policy(Some(layout), layout.ranks(), rank).unwrap();
                assert_eq!((charged.base_bytes, charged.bytes), (soft << 20, hard << 20), "{layout:?} rank {rank}");
            }
            for layout in [Step5GraphLayout::Fp8Max, Step5GraphLayout::Nvfp4Min] {
                assert!(prewarm(Some(layout)).is_none());
                assert_eq!(reserve(Some(layout), 0), None);
            }
        });
    }
    #[test]
    fn layout_without_trace_has_no_pins_or_rows_and_charges_the_envelope() {
        use crate::placement::graph_reserve::{envelope_bytes, GraphFamily};
        assert!(prewarm(None).is_none());
        assert_eq!(reserve(None, 0), None);
        assert!(selected_rows(None, 4096).unwrap().is_empty());
        cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
            for ranks in [1, 2] {
                for rank in 0..ranks {
                    assert_eq!(policy(None, ranks, rank).unwrap().bytes, envelope_bytes(GraphFamily::Step5, ranks, rank));
                }
            }
        });
    }
}
