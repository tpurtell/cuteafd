//! Heads arithmetic uses the shared paired projection protocol on two FIFO streams.
use super::{layer::UnitOwner, owners::Rank};
use crate::shared::{experts::rtx::pair::submit_scoped,
    peer_split::{attention::{self, HeadKernels, HeadMerge, HeadPrecision, HeadProjection, HeadSequence},
        PeerExchange, RankDevice, TerminalState}};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::minimax_m3::execution::HeadsLayout;
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reduction { Attention, Ffn { shared: bool } }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tag {
    pub unit: usize,
    pub layer: usize,
    pub rows: usize,
    pub kind: Reduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Publication { ordinal: u64, tag: Tag }

/// Fixed-size host guard: no allocation, device query or synchronization. Both
/// projections publish the same ordinal/tag before either peer push is issued.
#[derive(Default)]
struct Ordering {
    ordinal: Cell<u64>,
    projected: Cell<[Option<Publication>; 2]>,
    merged: Cell<[bool; 2]>,
}
impl Ordering {
    fn begin(&self) -> Result<u64> {
        ensure!(self.projected.get() == [None; 2] && self.merged.get() == [false; 2],
            "MiniMax Heads publication is already active");
        let ordinal = self.ordinal.get();
        ensure!(ordinal < u64::MAX, "MiniMax Heads publication ordinal exhausted");
        Ok(ordinal)
    }
    fn project(&self, rank: usize, publication: Publication) -> Result<()> {
        ensure!(rank < 2 && publication.ordinal == self.ordinal.get(), "MiniMax Heads projection ordinal differs");
        let mut projected = self.projected.get();
        ensure!(projected[rank].is_none() && projected[1 - rank].is_none_or(|other| other == publication),
            "MiniMax Heads ranks called reductions in different order");
        projected[rank] = Some(publication);
        self.projected.set(projected);
        Ok(())
    }
    fn merge(&self, rank: usize, publication: Publication) -> Result<()> {
        ensure!(rank < 2 && self.projected.get() == [Some(publication); 2]
            && publication.ordinal == self.ordinal.get(), "MiniMax Heads merge publication differs");
        let mut merged = self.merged.get();
        ensure!(!merged[rank], "MiniMax Heads rank merged twice");
        merged[rank] = true;
        self.merged.set(merged);
        Ok(())
    }
    fn finish(&self) -> Result<()> {
        ensure!(self.merged.get() == [true; 2], "MiniMax Heads publication is incomplete");
        self.ordinal.set(self.ordinal.get() + 1);
        self.projected.set([None; 2]);
        self.merged.set([false; 2]);
        Ok(())
    }
}

pub(crate) struct Heads<'a> {
    exchange: PeerExchange<'a>,
    sequence: RefCell<HeadSequence>,
    ordering: Ordering,
}
impl<'a> Heads<'a> {
    pub fn new(ranks: &[Rank<'a>], decode: u32, prefill: u32) -> Result<Self> {
        ensure!(ranks.len() == 2, "MiniMax Heads requires two physical ranks");
        let endpoints = [0, 1].map(|gpu| {
            let stream = &ranks[gpu].decode.borrow().stream;
            RankDevice { device: stream.device.id, stream: stream.raw }
        });
        let library = ranks[0].decode.borrow().stream.device.library;
        for (gpu, rank) in ranks.iter().enumerate() {
            ensure!(rank.prefill.iter().all(|unit| {
                let unit = unit.borrow();
                unit.stream.raw == endpoints[gpu].stream && unit.stream.device.id == endpoints[gpu].device
                    && std::ptr::eq(unit.stream.device.library, library)
            }), "MiniMax Heads global sequence requires one stream order per rank");
        }
        let rows = HeadsLayout::serialized_peer_rows(decode, prefill)?;
        let sizes = rows.map(|rows| rows as usize * 6144 * 4);
        let exchange = PeerExchange::new_abortable_sized(library, endpoints, &sizes, 6144 * 4 * 64)?;
        Ok(Self { exchange, sequence: RefCell::new(HeadSequence::new(0)?), ordering: Ordering::default() })
    }

    pub fn require_live(&self) -> Result<()> { self.exchange.require_live() }
    pub fn publish_abort(&self) -> Result<()> {
        if self.exchange.terminal_state() == TerminalState::Active { self.exchange.publish_abort() }
        else { Ok(()) }
    }
    pub fn cancel(&self) -> Result<()> {
        self.publish_abort()?;
        if self.exchange.terminal_state() == TerminalState::Aborting {
            let result = self.exchange.drain_compute();
            self.exchange.finish_terminal(result.is_ok());
            result
        } else { Ok(()) }
    }

    pub fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        for (rank, name, buffer) in self.exchange.allocation_buffers() {
            let plane = name.strip_prefix("peer.").context("MiniMax Heads shared peer receipt namespace")?;
            inventory.device(&format!("heads.peer.{rank}.{plane}"), buffer)?;
        }
        Ok(())
    }

    /// This publication is deliberately eager: the persistent parity may flip
    /// between calls, so no captured graph bakes either receive-slot pointer.
    pub fn reduce(&self, ranks: &[Rank<'a>], units: [&UnitOwner<'a>; 2], tags: [Tag; 2]) -> Result<()> {
        let ordinal = self.ordering.begin()?;
        struct Kernels<'s, 'a> {
            ranks: &'s [Rank<'a>],
            units: [&'s UnitOwner<'a>; 2],
            ordering: &'s Ordering,
            ordinal: u64,
            tags: [Tag; 2],
        }
        impl HeadKernels for Kernels<'_, '_> {
            fn project(&mut self, rank: usize) -> Result<HeadProjection> {
                self.ordering.project(rank, Publication { ordinal: self.ordinal, tag: self.tags[rank] })?;
                Ok(HeadProjection { partial: self.units[rank].plane("reduction")?, rounded: None })
            }
            fn merge(&mut self, rank: usize, values: HeadMerge) -> Result<()> {
                let tag = self.tags[rank];
                self.ordering.merge(rank, Publication { ordinal: self.ordinal, tag })?;
                let HeadMerge::F32(values) = values else { anyhow::bail!("MiniMax Heads requires FP32 native partials") };
                // SAFETY: shared head_partials ordered both producers/publications
                // before either wait; the global FIFO sequence retains the views.
                unsafe { self.units[rank].merge_heads(&self.ranks[rank].kernels, values, tag.rows,
                    tag.kind == Reduction::Attention, matches!(tag.kind, Reduction::Ffn { shared: true })) }
            }
        }
        submit_scoped(|| self.require_live(), || self.cancel(), |scope| {
            attention::head_partials(scope, &self.exchange, &mut self.sequence.borrow_mut(), tags[0].rows * 6144,
                HeadPrecision::F32, &mut Kernels { ranks, units, ordering: &self.ordering, ordinal, tags })?;
            self.ordering.finish()
        })
    }

    /// The next paired reduction orders the receiving stream's last read before
    /// the same per-unit destination can be reused. No new peer planes exist.
    pub fn replicate(&self, units: [&UnitOwner<'a>; 2], plane: &str, bytes: usize) -> Result<()> {
        submit_scoped(|| self.require_live(), || self.cancel(), |_| {
            attention::replicate_terminal(&self.exchange, units[0].plane(plane)?, units[1].plane(plane)?, bytes)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(layer: usize) -> Tag { Tag { unit: 0, layer, rows: 16, kind: Reduction::Attention } }

    #[test]
    fn rank_call_order_or_ordinal_mismatch_is_rejected_before_merge() -> Result<()> {
        let order = Ordering::default();
        let first = Publication { ordinal: order.begin()?, tag: tag(3) };
        order.project(0, first)?;
        assert!(order.project(1, Publication { tag: tag(17), ..first }).is_err());
        assert!(order.merge(0, first).is_err());
        assert!(order.project(1, Publication { ordinal: 1, ..first }).is_err());
        order.project(1, first)?;
        order.merge(0, first)?;
        assert!(order.finish().is_err());
        order.merge(1, first)?;
        order.finish()?;
        assert_eq!(order.begin()?, 1);
        assert!(order.project(0, first).is_err());
        Ok(())
    }

    #[test]
    fn one_sequence_orders_decode_and_prefill_without_per_step_storage() -> Result<()> {
        let order = Ordering::default();
        for (ordinal, (unit, layer, kind)) in [(usize::MAX, 0, Reduction::Attention),
            (0, 0, Reduction::Ffn { shared: false }), (1, 3, Reduction::Attention),
            (usize::MAX, 3, Reduction::Ffn { shared: true })].into_iter().enumerate() {
            assert_eq!(order.begin()?, ordinal as u64);
            let publication = Publication { ordinal: ordinal as u64, tag: Tag { unit, layer, rows: 1, kind } };
            for rank in 0..2 { order.project(rank, publication)?; }
            for rank in 0..2 { order.merge(rank, publication)?; }
            order.finish()?;
        }
        assert_eq!(std::mem::size_of::<Ordering>(), std::mem::size_of_val(&order));
        Ok(())
    }
}
