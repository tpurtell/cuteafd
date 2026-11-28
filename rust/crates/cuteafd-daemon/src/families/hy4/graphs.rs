//! Persistent rank-local decode sections on the shared bounded device graph cache (#2550).
//!
//! Host joins (DSA index top-k, TP2 peer/context exchanges, Spark expert exchange,
//! and TP2 head handoff) stay eager outside capture so every captured section is
//! strictly rank-local and depends only on `(layer, rows, section)`.
use crate::shared::{
    decode_graph::{
        device::{self, Basis, Cycle, Preference, Site},
        GraphBank, GraphOwner,
    },
    memory::device::Stream,
};
use anyhow::Result;
use std::{
    cell::RefCell,
    collections::HashSet,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Section {
    DraftFusion,
    DraftLayer,
    DraftCorrection,
    /// Full IndexCache layer pre-topk (TP1 local / TP2): optional prev-FFN iHC + MLA Q/KV/absorb + indexer projections & index_post.
    DsaFullPre,
    /// Shared IndexCache layer pre-attention (TP2): optional prev-FFN iHC + MLA Q/KV/absorb (zero indexer compute).
    DsaSharedPre,
    /// TP1 Spark-routed layer (Full and Shared IndexCache): optional prev-FFN iHC + MLA Q/KV/absorb
    /// + optional full-indexer `index_post` & `index_topk_decode` + `mla_dsa_decode` + sink/UV + o_proj
    /// + post-attn iHC + router.
    DsaSpark,
    /// TP1 local DSA post-indexer: `mla_dsa_decode` + sink/UV + o_proj + post-attn + Dense/MoE+EXL3+post-FFN.
    DsaLocalPost,
    /// DSA attention output only (TP2 before peer reduce).
    DsaAttnOut,
    /// Post-attn iHC + Dense/MoE frontend (TP2 local/dense; includes TP2 local EXL3 when onboarded).
    FfnFrontend,
    /// Spark-routed post-attention + router on rank 0, or post-attention + shared FFN on rank 1 (TP2).
    SparkRouter,
    /// Post-FFN combine + iHC post (non-deferred fallback).
    FfnFinish,
    /// Rank-0 final layer post-FFN combine + iHC post + terminal `hc_head`.
    Tail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    pub(crate) layer: usize,
    pub(crate) rows: usize,
    pub(crate) section: Section,
    /// Bucketed decode page-table width/stride (`decode_page_stride`) when this
    /// section captures `index_topk_decode`, or `0` when it does not read the page table.
    pub(crate) table_width: usize,
    /// True when the preceding layer's deferred FFN finish is fused into this section's start.
    pub(crate) prev_ffn: bool,
}

impl Key {
    pub(crate) const fn new(layer: usize, rows: usize, section: Section) -> Self {
        Self { layer, rows, section, table_width: 0, prev_ffn: false }
    }

    pub(crate) const fn with_prev_ffn(layer: usize, rows: usize, section: Section, prev_ffn: bool) -> Self {
        Self { layer, rows, section, table_width: 0, prev_ffn }
    }

    pub(crate) const fn with_table(
        layer: usize,
        rows: usize,
        section: Section,
        table_width: usize,
        prev_ffn: bool,
    ) -> Self {
        Self { layer, rows, section, table_width, prev_ffn }
    }
}

/// Per-layer graph sections emitted on GPU 0 for a TP1 Spark-routed decode layer.
///
/// Both Full IndexCache layers (with bucketed `pt_stride` captured inside `index_topk_decode`)
/// and Shared IndexCache layers (with `table_width = 0`) execute as a single `DsaSpark`
/// segment from the previous layer's deferred FFN finish through this layer's router.
/// `shared_ffn` runs eagerly between `waves.send` and `waves.land` in the Spark RDMA shadow.
pub(crate) fn tp1_spark_layer_sections(can_capture_topk: bool, is_full_indexer: bool) -> &'static [Section] {
    if !is_full_indexer || can_capture_topk {
        &[Section::DsaSpark]
    } else {
        &[Section::DsaFullPre, Section::DsaAttnOut, Section::SparkRouter]
    }
}

pub(crate) struct Hy4Graphs<'a> {
    bank: RefCell<GraphBank<Key, GraphOwner<'a, ()>>>,
    eager: RefCell<HashSet<Key>>,
    touched: RefCell<HashSet<Key>>,
    device: i32,
}

impl<'a> Hy4Graphs<'a> {
    pub(crate) fn new() -> Self {
        Self {
            bank: RefCell::new(GraphBank::with_family("hy4")),
            eager: RefCell::new(HashSet::new()),
            touched: RefCell::new(HashSet::new()),
            device: -1,
        }
    }

    pub(crate) fn attach(
        &mut self,
        library: &'a cuteafd_ffi::NativeLibrary,
        device: i32,
        policy: cuteafd_loader::placement::graph_reserve::Resolved,
    ) -> Result<()> {
        self.bank.get_mut().attach_device(
            library,
            device,
            policy.bytes,
            Site {
                name: "hy4.section",
                cycle: Cycle::Other,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: super::graph_sites::spans("hy4.section", false),
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
        self.bank.borrow_mut().cache_cycle(cycle);
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

    pub(super) fn draft(
        &self,
        rows: usize,
        stage: usize,
        section: Section,
        stream: &Stream<'a>,
        work: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        self.run(Key::new(stage, rows, section), stream, work)
    }

    pub(crate) fn tick(&self) {
        self.bank.borrow_mut().cache_tick();
    }

    pub(crate) fn visit(&self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(
            self.device,
            "hy4.section",
            false,
            &mut *self.bank.borrow_mut(),
        )
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
        if self.touched.borrow_mut().insert(key) {
            return work();
        }
        let cached = self
            .bank
            .borrow_mut()
            .cached_launch(&key)
            .map(|graph| graph.raw);
        if let Some(raw) = cached {
            return stream.device.run(|| {
                // SAFETY: containing Hy4Engine retains weights, cache, scratch, and EXL3 owners;
                // all rank streams drain before terminal clear or storage drop.
                unsafe { crate::shared::decode_graph::launch(&stream.device.library, raw, stream.raw, self.bank.borrow().launch_tally()) }
            });
        }
        let Some(mut ticket) = self.bank.borrow_mut().cache_ticket(&key, false, None) else {
            return work();
        };
        stream.device.run(|| {
            let library = stream.device.library;
            // SAFETY: sections contain only ordered rank-local launches on this stream over persistent storage.
            if let Err(error) =
                unsafe { crate::shared::decode_graph::begin_capture(&library, stream.raw, &ticket) }
            {
                if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Hy4 capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: always end this stream's capture, including a failed section.
            let captured =
                unsafe { crate::shared::decode_graph::end_capture(&library, stream.raw, &mut ticket) };
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
                        tracing::warn!(%error, ?key, "Hy4 instantiate unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: exclusive executable ownership; terminate() drains + clears banks before drop.
            let graph = unsafe { GraphOwner::new(library, stream.device.id, raw, ())? };
            if !retain(nodes, inspection_error.is_none()) {
                self.eager.borrow_mut().insert(key);
                drop(graph);
                drop(ticket);
                return work();
            }
            self.bank.borrow_mut().cache_commit(
                key,
                graph,
                ticket,
                nodes as u64,
                width,
                spans_devices,
            );
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
        let policy =
            cuteafd_loader::placement::graph_reserve::resolve(None, 0, false, 0, 0).unwrap();
        assert_eq!(policy.bytes, 0);
        let mut bank: GraphBank<Key, ()> = GraphBank::with_family("hy4");
        bank.attach_device_for_test(
            -2550,
            policy.bytes,
            Site {
                name: "hy4.section",
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
            for layer in 0..78 {
                for section in [
                    Section::DsaFullPre,
                    Section::DsaSharedPre,
                    Section::DsaSpark,
                    Section::DsaLocalPost,
                    Section::DsaAttnOut,
                    Section::FfnFrontend,
                    Section::SparkRouter,
                    Section::FfnFinish,
                    Section::Tail,
                ] {
                    assert!(bank
                        .cache_ticket(&Key::new(layer, 1, section), false, None)
                        .is_none());
                }
            }
            bank.cache_tick();
            assert!(!bank.cache_needs_boundary());
        }
    }

    #[test]
    fn tp1_spark_layer_section_sequence_is_one_segment_per_layer() {
        // Both Full IndexCache and Shared IndexCache Spark-routed layers execute in 1 graph segment.
        assert_eq!(tp1_spark_layer_sections(true, true), &[Section::DsaSpark]);
        assert_eq!(tp1_spark_layer_sections(true, false), &[Section::DsaSpark]);
        assert_eq!(tp1_spark_layer_sections(false, false), &[Section::DsaSpark]);
        // Keys normalize table_width=0 for shared-indexer layers and distinguish prev_ffn/table_width.
        let shared = Key::with_table(2, 1, Section::DsaSpark, 0, true);
        let full_4k = Key::with_table(1, 1, Section::DsaSpark, 64, false);
        let full_8k = Key::with_table(1, 1, Section::DsaSpark, 128, false);
        assert_ne!(full_4k, full_8k);
        assert_ne!(shared, Key { prev_ffn: false, ..shared });
    }

    #[test]
    fn decode_graphs_off_resolves_zero_reserve_and_selects_eager_route() {
        use cuteafd_loader::families::hy4::graph_reserve::{reserve_policy, Layout};
        use cuteafd_loader::placement::{graph_reserve::disabled, AttentionPlacement};
        for (gpus, sparks, attention) in [(1, 4, AttentionPlacement::Heads), (2, 6, AttentionPlacement::Context)] {
            let layout = Layout::resolve(gpus, sparks, attention);
            let policies: Vec<_> = (0..gpus)
                .map(|rank| reserve_policy(layout, gpus, rank, false).unwrap())
                .collect();
            assert!(policies.iter().enumerate().all(|(rank, p)| *p == disabled(rank)));
            assert!(!policies.iter().any(|p| p.bytes > 0), "configure_graphs disables graphs_enabled and EventCycles");
            for (rank, policy) in policies.iter().enumerate() {
                let mut bank: GraphBank<Key, ()> = GraphBank::with_family("hy4");
                let device = -2560 - (gpus as i32) * 10 - rank as i32;
                bank.attach_device_for_test(device, policy.bytes, Site {
                    name: "hy4.section",
                    cycle: Cycle::Decode,
                    preference: Preference::Adaptive,
                    bootstrap: 2 << 20,
                    basis: Basis::Bytes,
                    spans_devices: false,
                }).unwrap();
                for cycle in [Cycle::Decode, Cycle::Verify] {
                    bank.cache_cycle(cycle);
                    for section in [Section::DsaFullPre, Section::DsaSharedPre, Section::DsaSpark,
                        Section::DsaLocalPost, Section::DsaAttnOut,
                        Section::FfnFrontend, Section::SparkRouter, Section::FfnFinish, Section::Tail] {
                        assert!(bank.cache_ticket(&Key::new(0, 1, section), false, None).is_none());
                    }
                    bank.cache_tick();
                    assert!(!bank.cache_needs_boundary());
                }
            }
        }
    }
}
