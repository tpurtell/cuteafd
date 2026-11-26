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
    /// Full IndexCache layer pre-topk: optional prev-FFN iHC + MLA Q/KV/absorb + indexer projections & index_post.
    DsaFullPre,
    /// Shared IndexCache layer pre-attention: optional prev-FFN iHC + MLA Q/KV/absorb (zero indexer compute).
    DsaSharedPre,
    /// 2-GPU ContextSplit DSA dual partials (h0 + h32 over parity-owned local pages).
    DsaPartial,
    /// TP1 local DSA post-indexer: `mla_dsa_decode` + sink/UV + o_proj + post-attn + Dense/MoE+EXL3+post-FFN.
    DsaLocalPost,
    /// DSA attention output only (TP2 before peer reduce, or Spark-routed).
    DsaAttnOut,
    /// Post-attn iHC + Dense/MoE frontend (TP2 or Spark-routed; includes TP2 local EXL3 when onboarded).
    FfnFrontend,
    /// Spark-routed post-attention and router, before request staging.
    SparkRouter,
    /// Shared expert FFN launched after staging, ahead of the landed wait.
    SparkShared,
    /// Post-FFN combine + iHC post (TP2 or Spark-routed; no sandwich norm in Hy4).
    FfnFinish,
    /// Rank-0 final layer post-FFN combine + iHC post + terminal `hc_head`.
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
        rank: usize,
    ) -> Result<()> {
        let policy = cuteafd_loader::placement::graph_reserve::current(rank, false, 0, 0)?;
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
        self.run(Key { layer: stage, rows, section }, stream, work)
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
        if let Some(raw) = self
            .bank
            .borrow_mut()
            .cached_launch(&key)
            .map(|graph| graph.raw)
        {
            return stream.device.run(|| {
                // SAFETY: containing Hy4Engine retains weights, cache, scratch, and EXL3 owners;
                // all rank streams drain before terminal clear or storage drop.
                unsafe { crate::shared::decode_graph::launch(&stream.device.library, raw, stream.raw) }
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
                    Section::DsaPartial,
                    Section::DsaLocalPost,
                    Section::DsaAttnOut,
                    Section::FfnFrontend,
                    Section::SparkRouter,
                    Section::SparkShared,
                    Section::FfnFinish,
                    Section::Tail,
                ] {
                    assert!(bank
                        .cache_ticket(&Key { layer, rows: 1, section }, false, None)
                        .is_none());
                }
            }
            bank.cache_tick();
            assert!(!bank.cache_needs_boundary());
        }
    }
}
