//! Persistent rank-local decode sections on the shared bounded device graph cache (#2591 C2).
//!
//! Host joins (Engram staging H2D/P2P, DSA indexer pending D2D + score/topk,
//! TP2 peer reductions, Spark expert exchange, and TP2 head gather) stay eager
//! outside capture so every captured section is strictly rank-local and depends
//! only on `(layer, rows, section)`.
use crate::shared::{
    decode_graph::{
        device::{self, Basis, Cycle, Preference, Site},
        GraphBank, GraphOwner,
    },
    memory::device::Stream,
};
use anyhow::Result;
use cuteafd_loader::serving_capacity::hy4_graphs::{self, Hy4GraphLayout};
use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Section {
    DraftFusion,
    DraftLayer,
    DraftCorrection,
    /// TP1 local SWA layer: optional prev-FFN finish + optional Engram compute + MLA SWA + post-attn + Dense/MoE+EXL3+post-FFN.
    SwaLocal,
    /// TP1 Spark-routed single-row SWA layer: optional prev-FFN finish + optional Engram compute + MLA SWA + post-attn + router.
    SwaSpark,
    /// SWA attention section (TP2 before peer reduce).
    SwaAttn,
    /// Multi-row SWA pre-attention before eager `mla_swa_extend`.
    SwaExtendPre,
    /// Multi-row SWA post-attention after eager `mla_swa_extend` (TP1 local or TP2).
    SwaExtendPost,
    /// Multi-row TP1 Spark-routed SWA post-attention + post-attn iHC + router after eager `mla_swa_extend`.
    SwaExtendSpark,
    /// DSA pre-indexer (short context <=2048, or TP1 local / TP2): optional prev-FFN finish + optional Engram compute + MLA Q/KV/absorb + indexer_wk + rms_norm_128.
    DsaPre,
    /// TP1 Spark-routed DSA pre-indexer at >2048 context: `DsaPre` + `dsa_index_q` (eliminates separate `DsaIndexQ` launch).
    DsaPreScored,
    /// DSA >2048 indexer query projections + RoPE (3 kernels; TP1 local / TP2).
    DsaIndexQ,
    /// 2-GPU ContextSplit DSA dual partials (h0 + h16 over local pages).
    DsaPartial,
    /// TP1 local DSA post-indexer: `mla_dsa_decode` + sink/UV + o_proj + post-attn + MoE+EXL3+post-FFN.
    DsaLocalPost,
    /// TP1 Spark-routed DSA post-indexer: `mla_dsa_decode` + sink/UV + o_proj + post-attn + router.
    DsaSparkPost,
    /// DSA attention output only (TP2 before peer reduce): `mla_dsa_decode` + sink/UV + o_proj.
    DsaAttnOut,
    /// Post-attn iHC + Dense/MoE frontend (TP2 local/dense; includes TP2 local EXL3 when onboarded).
    FfnFrontend,
    /// Spark-routed post-attention and router, before request staging (TP2).
    SparkRouter,
    /// Shared expert FFN launched after staging, ahead of the landed wait (TP2; TP1 runs eagerly in the RDMA shadow).
    SparkShared,
    /// Post-FFN combine + norm + iHC post (non-deferred fallback).
    FfnFinish,
    /// Rank-0 final layer post-FFN combine + norm + iHC post + terminal `hc_head`.
    Tail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    pub(crate) layer: usize,
    pub(crate) rows: usize,
    pub(crate) section: Section,
    /// True when the preceding layer's deferred FFN finish is fused into this section's start.
    pub(crate) prev_ffn: bool,
}

impl Key {
    pub(crate) const fn new(layer: usize, rows: usize, section: Section) -> Self {
        Self { layer, rows, section, prev_ffn: false }
    }

    pub(crate) const fn with_prev_ffn(layer: usize, rows: usize, section: Section, prev_ffn: bool) -> Self {
        Self { layer, rows, section, prev_ffn }
    }
}

/// Per-layer graph sections emitted on GPU 0 for a TP1 Spark-routed decode layer.
///
/// The previous layer's `FfnFinish` fuses into the start of the first section
/// (`prev_ffn: true` for every layer after a Spark-routed predecessor), `SparkRouter`
/// fuses into the end of the last section, and `shared_ffn` runs eagerly between
/// `waves.send` and `waves.land` in the Spark RDMA shadow (matching GLM 5.3).
pub(crate) fn tp1_spark_layer_sections(
    is_sparse: bool,
    all_single_row: bool,
    need_score: bool,
) -> &'static [Section] {
    if !is_sparse {
        if all_single_row {
            &[Section::SwaSpark]
        } else {
            &[Section::SwaExtendPre, Section::SwaExtendSpark]
        }
    } else if need_score {
        &[Section::DsaPreScored, Section::DsaSparkPost]
    } else {
        &[Section::DsaPre, Section::DsaSparkPost]
    }
}

pub(crate) struct Hy4Graphs<'a> {
    bank: RefCell<GraphBank<Key, GraphOwner<'a, ()>>>,
    // A section found to contain fewer than two native nodes always stays eager.
    eager: RefCell<HashSet<Key>>,
    touched: RefCell<HashSet<Key>>,
    device: i32,
    first_touch_pending: Cell<bool>,
}

impl<'a> Hy4Graphs<'a> {
    pub(crate) fn new() -> Self {
        Self {
            bank: RefCell::new(GraphBank::with_family("hy4_flash")),
            eager: RefCell::new(HashSet::new()),
            touched: RefCell::new(HashSet::new()),
            device: -1,
            first_touch_pending: Cell::new(false),
        }
    }

    pub(crate) fn attach(
        &mut self,
        library: &'a cuteafd_ffi::NativeLibrary,
        device: i32,
        rank: usize,
        layout: Option<Hy4GraphLayout>,
        tp2_dfly2: bool,
    ) -> Result<()> {
        let policy = hy4_graphs::reserve_policy_for(layout, rank, tp2_dfly2)?;
        self.bank.get_mut().attach_device(
            library,
            device,
            policy.bytes,
            Site {
                name: "hy4_flash.section",
                cycle: Cycle::Other,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: super::graph_sites::spans("hy4_flash.section", false),
            },
        )?;
        device::set_reserve_policy(device, policy)?;
        self.device = device;
        Ok(())
    }

    pub(crate) fn attached(&self) -> bool {
        self.bank.borrow().cache_attached()
    }

    pub(crate) fn cycle(&self, cycle: Cycle) {
        self.bank
            .borrow_mut()
            .cache_cycle(cycle);
    }

    pub(crate) fn clear_drained(&self) {
        self.bank.borrow_mut().cache_clear_drained();
    }

    pub(super) fn retained_draft_rows(&self, rows: usize) -> usize {
        self.bank.borrow().count(|key| {
            key.rows == rows
                && matches!(
                    key.section,
                    Section::DraftFusion | Section::DraftLayer | Section::DraftCorrection
                )
        })
    }

    pub(crate) fn retained_rows(&self, rows: usize) -> usize {
        self.bank.borrow().count(|key| key.rows == rows)
    }

    pub(crate) fn needs_boundary(&self) -> bool {
        self.first_touch_pending.get() || self.bank.borrow().cache_needs_boundary()
    }

    pub(crate) fn observed(&self) {
        self.first_touch_pending.set(false);
    }

    pub(crate) fn tick(&self) {
        self.bank.borrow_mut().cache_tick();
    }

    pub(crate) fn visit(&self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(self.device, "hy4_flash.section", false, &mut *self.bank.borrow_mut())
    }

    pub(super) fn draft(
        &self, rows: usize, stage: usize, section: Section, stream: &Stream<'a>,
        work: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        self.run(Key::new(stage, rows, section), stream, work)
    }

    pub(crate) fn run(
        &self,
        key: Key,
        stream: &Stream<'a>,
        mut work: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        if !self.attached() || self.eager.borrow().contains(&key) {
            return work();
        }
        // First use initializes modules/workspaces eagerly; the outer step drains
        // before a later use can reserve capture growth for this geometry.
        if self.touched.borrow_mut().insert(key) {
            self.first_touch_pending.set(true);
            return work();
        }
        let cached = self.bank.borrow_mut().cached_launch(&key).map(|graph| graph.raw);
        if let Some(raw) = cached {
            return stream.device.run(|| {
                // SAFETY: containing Hy4Engine retains weights, cache, scratch, and EXL3 owners;
                // all rank streams drain before terminal clear or storage drop.
                unsafe { crate::shared::decode_graph::launch(&stream.device.library, raw, stream.raw, self.bank.borrow().launch_tally()) }
            });
        }
        let Some(mut ticket) = self
            .bank
            .borrow_mut()
            .cache_ticket(&key, false, None)
        else {
            return work();
        };
        stream.device.run(|| {
            let library = stream.device.library;
            // SAFETY: sections contain only ordered rank-local launches on this stream over persistent storage.
            if let Err(error) = unsafe { crate::shared::decode_graph::begin_capture(&library, stream.raw, &ticket) } {
                if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Hy4-flash capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: always end this stream's capture, including a failed section.
            let captured = unsafe { crate::shared::decode_graph::end_capture(&library, stream.raw, &mut ticket) };
            if let Err(error) = result {
                self.bank
                    .borrow_mut()
                    .cache_ticket_failed(ticket, captured.as_ref().err().unwrap_or(&error));
                if let Ok((raw, ..)) = captured {
                    // SAFETY: failed capture has never launched.
                    unsafe {
                        let _ = library.cuda_graph_exec_destroy(raw);
                    }
                }
                return Err(error);
            }
            let (raw, nodes, width, spans_devices, inspection_error) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                        tracing::warn!(%error, ?key, "Hy4-flash instantiate unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: exclusive executable ownership; all captured allocations belong to Hy4Engine,
            // not the cache entry, and terminate() drains + clears banks before dropping storage.
            let graph = unsafe { GraphOwner::new(library, stream.device.id, raw, ())? };
            if !retain(nodes, inspection_error.is_none()) {
                self.eager.borrow_mut().insert(key);
                drop(graph);
                drop(ticket);
                return work();
            }
            self.bank
                .borrow_mut()
                .cache_commit(key, graph, ticket, nodes as u64, width, spans_devices);
            // SAFETY: committed executable retains the same persistent pointer contract.
            unsafe { crate::shared::decode_graph::launch(&library, raw, stream.raw, self.bank.borrow().launch_tally()) }
        })
    }
}

fn retain(nodes: usize, inspected: bool) -> bool {
    inspected && nodes >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_never_retain_single_node_or_unknown_topology() {
        assert!(!retain(0, true));
        assert!(!retain(1, true));
        assert!(!retain(10, false));
        assert!(retain(2, true));
    }

    #[test]
    fn untraced_zero_reserve_runs_every_section_eagerly_without_capture() {
        let policy = cuteafd_loader::placement::graph_reserve::resolve(None, 0, false, 0, 0).unwrap();
        assert_eq!(policy.bytes, 0);
        let mut bank: GraphBank<Key, ()> = GraphBank::with_family("hy4_flash");
        bank.attach_device_for_test(
            -2591,
            policy.bytes,
            Site {
                name: "hy4_flash.section",
                cycle: Cycle::Decode,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: false,
            },
        )
        .unwrap();
        for cycle in [Cycle::Decode, Cycle::Verify, Cycle::Prefill] {
            bank.cache_cycle(cycle);
            for layer in 0..42 {
                for section in [
                    Section::SwaLocal,
                    Section::SwaSpark,
                    Section::SwaAttn,
                    Section::SwaExtendPre,
                    Section::SwaExtendPost,
                    Section::SwaExtendSpark,
                    Section::DsaPre,
                    Section::DsaPreScored,
                    Section::DsaIndexQ,
                    Section::DsaPartial,
                    Section::DsaLocalPost,
                    Section::DsaSparkPost,
                    Section::DsaAttnOut,
                    Section::FfnFrontend,
                    Section::SparkRouter,
                    Section::SparkShared,
                    Section::FfnFinish,
                    Section::Tail,
                ] {
                    for rows in [1, 4, 16, 64] {
                        assert!(bank
                            .cache_ticket(&Key::new(layer, rows, section), false, None)
                            .is_none());
                    }
                }
            }
            bank.cache_tick();
            assert!(!bank.cache_needs_boundary());
        }
        assert!(bank
            .cached_launch(&Key::new(0, 1, Section::SwaLocal))
            .is_none());
    }

    #[test]
    fn keys_are_geometry_only() {
        let key = Key::new(8, 16, Section::SwaLocal);
        assert_eq!(key, Key::new(8, 16, Section::SwaLocal));
        assert_ne!(key, Key { rows: 4, ..key });
        assert_ne!(key, Key { section: Section::DsaPre, ..key });
        assert_ne!(key, Key { prev_ffn: true, ..key });
    }

    #[test]
    fn tp1_spark_layer_section_sequence_is_minimal() {
        // Single-row SWA Spark layer: 1 segment (prev FFN finish + Engram + SWA + post-attn + router).
        assert_eq!(tp1_spark_layer_sections(false, true, false), &[Section::SwaSpark]);
        assert_eq!(tp1_spark_layer_sections(false, true, true), &[Section::SwaSpark]);
        // Multi-row verify SWA Spark layer: 2 segments around eager mla_swa_extend.
        assert_eq!(
            tp1_spark_layer_sections(false, false, false),
            &[Section::SwaExtendPre, Section::SwaExtendSpark]
        );
        // Short-context (<=2048) DSA Spark layer: 2 segments around eager pending D2D + topk.
        assert_eq!(
            tp1_spark_layer_sections(true, true, false),
            &[Section::DsaPre, Section::DsaSparkPost]
        );
        // Long-context (>2048) DSA Spark layer: 2 segments (dsa_index_q folded into DsaPreScored).
        assert_eq!(
            tp1_spark_layer_sections(true, true, true),
            &[Section::DsaPreScored, Section::DsaSparkPost]
        );
    }
}
