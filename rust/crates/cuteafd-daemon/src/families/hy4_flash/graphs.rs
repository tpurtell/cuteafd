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
    /// TP1 local SWA layer: optional Engram compute + MLA SWA + post-attn + Dense/MoE+EXL3+post-FFN.
    SwaLocal,
    /// SWA attention section (TP2 before peer reduce, or Spark-routed before FFN frontend).
    SwaAttn,
    /// Multi-row SWA pre-attention before eager `mla_swa_extend`.
    SwaExtendPre,
    /// Multi-row SWA post-attention after eager `mla_swa_extend`.
    SwaExtendPost,
    /// DSA pre-indexer: optional Engram compute + MLA Q/KV/absorb + indexer_wk + rms_norm_128.
    DsaPre,
    /// DSA >2048 indexer query projections + RoPE (3 kernels).
    DsaIndexQ,
    /// 2-GPU ContextSplit DSA dual partials (h0 + h16 over local pages).
    DsaPartial,
    /// TP1 local DSA post-indexer: `mla_dsa_decode` + sink/UV + o_proj + post-attn + MoE+EXL3+post-FFN.
    DsaLocalPost,
    /// DSA attention output only (TP2 before peer reduce, or Spark-routed): `mla_dsa_decode` + sink/UV + o_proj.
    DsaAttnOut,
    /// Post-attn iHC + Dense/MoE frontend (TP2 or Spark-routed; includes TP2 local EXL3 when onboarded).
    FfnFrontend,
    /// Spark-routed post-attention and router, before request staging.
    SparkRouter,
    /// Shared expert FFN launched after staging, ahead of the landed wait.
    SparkShared,
    /// Onboarded TP2 EXL3 routed experts under `layers` placement (runs in parallel on both ranks).
    Tp2Expert,
    /// Post-FFN combine + norm + iHC post (TP2 or Spark-routed).
    FfnFinish,
    /// Rank-0 final layer post-FFN combine + norm + iHC post + terminal `hc_head`.
    Tail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    pub(crate) layer: usize,
    pub(crate) rows: usize,
    pub(crate) section: Section,
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
        self.run(Key { layer: stage, rows, section }, stream, work)
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
        if let Some(raw) = self.bank.borrow_mut().cached_launch(&key).map(|graph| graph.raw) {
            return stream.device.run(|| {
                // SAFETY: containing Hy4Engine retains weights, cache, scratch, and EXL3 owners;
                // all rank streams drain before terminal clear or storage drop.
                unsafe { crate::shared::decode_graph::launch(&stream.device.library, raw, stream.raw) }
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
            unsafe { crate::shared::decode_graph::launch(&library, raw, stream.raw) }
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
                    Section::SwaAttn,
                    Section::SwaExtendPre,
                    Section::SwaExtendPost,
                    Section::DsaPre,
                    Section::DsaIndexQ,
                    Section::DsaPartial,
                    Section::DsaLocalPost,
                    Section::DsaAttnOut,
                    Section::FfnFrontend,
                    Section::SparkRouter,
                    Section::SparkShared,
                    Section::Tp2Expert,
                    Section::FfnFinish,
                ] {
                    for rows in [1, 4, 16, 64] {
                        assert!(bank
                            .cache_ticket(&Key { layer, rows, section }, false, None)
                            .is_none());
                    }
                }
            }
            bank.cache_tick();
            assert!(!bank.cache_needs_boundary());
        }
        assert!(bank
            .cached_launch(&Key {
                layer: 0,
                rows: 1,
                section: Section::SwaLocal,
            })
            .is_none());
    }

    #[test]
    fn keys_are_geometry_only() {
        let key = Key {
            layer: 8,
            rows: 16,
            section: Section::SwaLocal,
        };
        assert_eq!(
            key,
            Key {
                layer: 8,
                rows: 16,
                section: Section::SwaLocal,
            }
        );
        assert_ne!(key, Key { rows: 4, ..key });
        assert_ne!(
            key,
            Key {
                section: Section::DsaPre,
                ..key
            }
        );
    }
}
