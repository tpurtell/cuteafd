//! Routed TP2 halves broadcast from and reduce onto the whole-layer owner.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

#[derive(Debug, Clone, Copy)]
struct Tp2Rows {
    input: usize,
    ids: usize,
    weights: usize,
    bytes: usize,
    partial_bytes: usize,
    reduction_bytes: usize,
    exchange_slot_bytes: usize,
}

impl Tp2Rows {
    fn transfer_bytes(&self) -> usize { self.bytes.max(self.partial_bytes) }

    fn new(hidden: usize, topk: usize, rows: usize, wire: bool,
        partial: crate::shared::experts::rtx::PartialDtype) -> Result<Self> {
        use crate::shared::experts::rtx::PartialDtype;
        ensure!(matches!((wire, partial), (true, PartialDtype::F32) | (false, PartialDtype::Bf16)),
            "owner TP2 wire and partial ABI disagree");
        let layout = cuteafd_loader::serving_capacity::OwnerTp2Rows::new(
            hidden as u64, topk as u64, rows as u64, wire)?;
        Ok(Self { input: usize::try_from(layout.input)?, ids: usize::try_from(layout.ids)?,
            weights: usize::try_from(layout.weights)?, bytes: usize::try_from(layout.bytes)?,
            partial_bytes: usize::try_from(layout.partial_bytes)?,
            reduction_bytes: usize::try_from(layout.reduction_bytes)?,
            exchange_slot_bytes: usize::try_from(layout.exchange_slot_bytes)? })
    }
}

/// TP2 routed halves reduce only onto the whole-attention owner. The next
/// broadcast cannot overtake that owner's sum, so both rank workspaces are
/// serialized without a host synchronization on the request path.
pub(crate) struct OwnerTp2<'a> {
    library: &'a NativeLibrary,
    ranks: [Device<'a>; 2],
    pub(crate) experts: std::mem::ManuallyDrop<[Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>; 2]>,
    pub(crate) exchange: std::mem::ManuallyDrop<crate::shared::peer_split::PeerExchange<'a>>,
    combine: [cuteafd_ffi::RtxTp2Combine<'a>; 2],
    send: [Allocation<'a>; 2],
    reduced: [Allocation<'a>; 2],
    hidden: usize,
    topk: usize,
    max_rows: usize,
    pub(crate) wire: bool,
}

impl<'a> OwnerTp2<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, ranks: [crate::shared::peer_split::RankDevice; 2],
        experts: [Box<dyn crate::shared::experts::rtx::RtxExpertLayer + 'a>; 2],
        hidden: usize, topk: usize, max_rows: usize, decode_rows: usize, wire: bool, tag_prefix: &str) -> Result<Self> {
        use crate::shared::experts::rtx::{PartialDtype, RtxShard};
        ensure!(!tag_prefix.is_empty() && decode_rows > 0 && decode_rows <= max_rows,
            "invalid owner TP2 tag or decode extent");
        ensure!(experts.iter().all(|expert| expert.uses_fp8_input() == wire),
            "owner TP2 input ABI differs from selected experts");
        ensure!(experts[0].partial() == experts[1].partial()
            && experts[0].layers() == experts[1].layers(), "owner TP2 halves disagree");
        for rank in 0..2 {
            ensure!(experts[rank].shard() == RtxShard::Tp2 { rank: rank as u8 }
                && experts[rank].device() == ranks[rank].device, "owner TP2 rank ownership mismatch");
        }
        let layout = Tp2Rows::new(hidden, topk, max_rows, wire, experts[0].partial())?;
        let devices = ranks.map(|rank| Device { library, id: rank.device });
        let zero = |rank: usize, bytes, tag| -> Result<Allocation<'a>> {
            let allocation = Allocation::new(devices[rank], bytes, tag)?;
            devices[rank].run(|| library.cuda_zero_bytes(allocation.buffer, bytes))?;
            Ok(allocation)
        };
        let _scope = devices[0].enter()?;
        let exchange = crate::shared::peer_split::PeerExchange::new_abortable(library, ranks, 2,
            layout.exchange_slot_bytes, {
                let decode = Tp2Rows::new(hidden, topk, decode_rows, wire, experts[0].partial())?;
                decode.transfer_bytes()
            })?;
        let combine = [devices[0].run(|| library.rtx_tp2_combine())?, devices[1].run(|| library.rtx_tp2_combine())?];
        // Local wire-packing scratch for this rank's routed rows, and the reduce output of both
        // ranks' partials. The exchange's own landing slots are the transport backing.
        let send_tag = crate::shared::memory::tag(crate::shared::memory::MemoryKind::Workspace,
            &format!("{tag_prefix}/tp2-send"));
        let reduce_tag = crate::shared::memory::tag(crate::shared::memory::MemoryKind::Workspace,
            &format!("{tag_prefix}/tp2-reduce"));
        let send = [zero(0, layout.bytes, send_tag)?, zero(1, layout.bytes, send_tag)?];
        let reduced = [zero(0, layout.reduction_bytes, reduce_tag)?, zero(1, layout.reduction_bytes, reduce_tag)?];
        let mut tp2 = Self { library, ranks: devices, experts: std::mem::ManuallyDrop::new(experts),
            exchange: std::mem::ManuallyDrop::new(exchange), combine, send, reduced,
            hidden, topk, max_rows, wire };
        // Prime before any peer wait: CUDA LAZY initialization may synchronize
        // the device, which would deadlock after a not-yet-published peer flag.
        for rank in 0..2 {
            let input = tp2.input(tp2.send[rank].buffer.ptr);
            let routes = tp2.routes(tp2.send[rank].buffer.ptr, layout);
            let stream = tp2.exchange.stream(rank);
            let dtype = match tp2.experts[rank].partial() { PartialDtype::F32 => cuteafd_ffi::RtxPartialDtype::F32,
                PartialDtype::Bf16 => cuteafd_ffi::RtxPartialDtype::Bf16 };
            devices[rank].run(|| {
                // SAFETY: zeroed input/routes cover max_rows, and no peer waits exist yet.
                unsafe { tp2.experts[rank].prime(input, routes, stream)?; }
                let partial = cuteafd_ffi::CuteafdDeviceBuffer { ptr: tp2.experts[rank].output(),
                    bytes: layout.partial_bytes, device_id: devices[rank].id, ..Default::default() };
                library.cuda_zero_bytes(partial, partial.bytes)?;
                // Before waits exist, an admitted receive slot is idle scratch.
                let shared = tp2.exchange.recv(rank, 0)?;
                // SAFETY: initialize geometry-generic sum/add on disjoint retained buffers.
                unsafe {
                    tp2.combine[rank].sum(partial.ptr, partial.ptr, tp2.reduced[rank].buffer.ptr.cast(),
                        max_rows * hidden, dtype, stream)?;
                    library.peer_add_bf16(tp2.reduced[rank].buffer.ptr, tp2.reduced[rank].buffer.ptr,
                        shared, max_rows * hidden, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                Ok(())
            })?;
        }
        Ok(tp2)
    }

    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        prefix: &str) -> Result<()> {
        for rank in 0..2 {
            self.experts[rank].allocation_receipts(inventory, &format!("{prefix}.{rank}.experts"))?;
            self.send[rank].inventory(inventory, &format!("{prefix}.{rank}.send"))?;
            self.reduced[rank].inventory(inventory, &format!("{prefix}.{rank}.reduced"))?;
        }
        for (rank, name, buffer) in self.exchange.allocation_buffers() {
            inventory.device(&format!("{prefix}.{rank}.{name}"), buffer)?;
        }
        Ok(())
    }

    fn input(&self, ptr: *mut c_void) -> crate::shared::experts::rtx::ExpertInput {
        if self.wire { crate::shared::experts::rtx::ExpertInput::Fp8K32(ptr) }
        else { crate::shared::experts::rtx::ExpertInput::Bf16(ptr) }
    }

    fn routes(&self, ptr: *mut c_void, layout: Tp2Rows) -> crate::shared::experts::rtx::Routes {
        crate::shared::experts::rtx::Routes { ids: ptr.cast::<u8>().wrapping_add(layout.ids).cast(),
            weights: ptr.cast::<u8>().wrapping_add(layout.weights).cast() }
    }

    /// # Safety
    /// Input/routes/shared/output belong to owner, cover rows, are disjoint
    /// except the read-only operands, and stay live until both streams drain.
    pub(crate) unsafe fn enqueue(&mut self, owner: usize, layer: usize, rows: usize, input: *const c_void,
        routes: crate::shared::experts::rtx::Routes, shared: *const c_void, output: *mut c_void) -> Result<()> {
        self.exchange.require_live()?;
        // SAFETY: forwards the retained input/output and ordering contract.
        let result = unsafe { self.enqueue_live(owner, layer, rows, input, routes, shared, output) };
        if result.is_err() {
            // Publish independently, but the whole-engine boundary owns drainage.
            let aborted = match self.exchange.terminal_state() {
                crate::shared::peer_split::TerminalState::Aborting => Ok(()),
                _ => self.exchange.publish_abort(),
            };
            crate::shared::decode_graph::fatal_drain(aborted, "owner RTX enqueue abort");
        }
        result
    }

    unsafe fn enqueue_live(&mut self, owner: usize, layer: usize, rows: usize, input: *const c_void,
        routes: crate::shared::experts::rtx::Routes, shared: *const c_void, output: *mut c_void) -> Result<()> {
        use crate::shared::experts::rtx::PartialDtype;
        ensure!(owner < 2 && (1..=self.max_rows).contains(&rows), "invalid owner TP2 owner/rows");
        let peer = 1 - owner;
        let extent = Tp2Rows::new(self.hidden, self.topk, rows, self.wire, self.experts[owner].partial())?;
        let ptr = self.send[owner].buffer.ptr;
        let stream = self.exchange.stream(owner);
        self.ranks[owner].run(|| {
            for (source, offset, bytes) in [(input, 0, extent.input), (routes.ids.cast_const(), extent.ids, rows * self.topk * 4),
                (routes.weights.cast_const(), extent.weights, rows * self.topk * 4)] {
                let dst = cuteafd_ffi::CuteafdDeviceBuffer { ptr: ptr.cast::<u8>().wrapping_add(offset).cast(),
                    bytes, device_id: self.ranks[owner].id, ..Default::default() };
                let src = cuteafd_ffi::CuteafdDeviceBuffer { ptr: source.cast_mut(), ..dst };
                // SAFETY: documented owner inputs are live and destination spans are nonoverlapping.
                unsafe { crate::shared::memory::device::copy_device_ordered(self.library, dst, src, bytes, stream)?; }
            }
            Ok(())
        })?;
        self.exchange.push(owner, 0, ptr, extent.bytes)?;
        self.exchange.wait(peer, 0)?;
        let local_input = self.input(ptr);
        let local_routes = self.routes(ptr, extent);
        let peer_ptr = self.exchange.recv(peer, 0)?;
        let peer_input = self.input(peer_ptr);
        let peer_routes = self.routes(peer_ptr, extent);
        // SAFETY: broadcast is ordered before the peer and owner input packing before local work.
        // A failed rank launch publishes both abort words from the home device
        // before returning, so no queued peer wait outlives its producer.
        let exchange = &*self.exchange;
        let home = &self.ranks[0];
        let mut abort = || crate::shared::decode_graph::fatal_drain(home.run(|| exchange.publish_abort()),
            "owner TP2 RTX enqueue abort");
        unsafe {
            self.experts[owner].enqueue(layer, rows, local_input, local_routes, stream, &mut abort)?;
            self.experts[peer].enqueue(layer, rows, peer_input, peer_routes, exchange.stream(peer), &mut abort)?;
        }
        self.exchange.push(peer, 1, self.experts[peer].output(), extent.partial_bytes)?;
        self.exchange.wait(owner, 1)?;
        let mine = self.experts[owner].output();
        let theirs = self.exchange.recv(owner, 1)?;
        let [rank0, rank1] = if owner == 0 { [mine, theirs] } else { [theirs, mine] };
        let dtype = match self.experts[owner].partial() { PartialDtype::F32 => cuteafd_ffi::RtxPartialDtype::F32,
            PartialDtype::Bf16 => cuteafd_ffi::RtxPartialDtype::Bf16 };
        self.ranks[owner].run(|| {
            // SAFETY: rank partials arrived on owner; reduction/output storage is disjoint and retained.
            unsafe {
                self.combine[owner].sum(rank0, rank1, self.reduced[owner].buffer.ptr.cast(), rows * self.hidden, dtype, stream)?;
                self.library.peer_add_bf16(self.reduced[owner].buffer.ptr, shared, output, rows * self.hidden, stream)
            }
        })
    }
}

impl Drop for OwnerTp2<'_> {
    fn drop(&mut self) {
        use crate::shared::peer_split::TerminalState;
        let complete = match self.exchange.terminal_state() {
            TerminalState::Drained => true,
            TerminalState::Active => {
                // Publish abort before draining: a failed enqueue can leave a
                // peer wait queued whose producer was never reached.
                let aborted = self.exchange.publish_abort();
                let drained = self.exchange.drain_compute();
                let complete = aborted.is_ok() && drained.is_ok();
                self.exchange.finish_terminal(complete);
                if !complete { tracing::error!(?aborted, ?drained, "owner TP2 streams failed to drain"); }
                complete
            }
            _ => false,
        };
        if !complete {
            self.library.quarantine_module_after_failed_drain();
            return;
        }
        // SAFETY: both compute streams drained; each retained component drops exactly once.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.experts);
            std::mem::ManuallyDrop::drop(&mut self.exchange);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp2_payloads_pad_routes_and_check_extents() -> Result<()> {
        use crate::shared::experts::rtx::PartialDtype;
        for wire in [false, true] {
            for rows in [1, 3, 64, 4096] {
                let partial = if wire { PartialDtype::F32 } else { PartialDtype::Bf16 };
                let p = Tp2Rows::new(2560, 10, rows, wire, partial)?;
                let planned = cuteafd_loader::serving_capacity::OwnerTp2Rows::new(
                    2560, 10, rows as u64, wire)?;
                assert_eq!(p.exchange_slot_bytes as u64, planned.exchange_slot_bytes);
                assert_eq!(p.reduction_bytes as u64, planned.reduction_bytes);
                assert_eq!(p.ids % 16, 0);
                assert_eq!(p.weights % 16, 0);
                assert_eq!(p.bytes % 16, 0);
                assert!(p.ids >= p.input && p.weights >= p.ids + rows * 40);
                assert_eq!(p.partial_bytes, rows * 2560 * if wire { 4 } else { 2 });
            }
        }
        assert!(Tp2Rows::new(usize::MAX, 10, 64, false, PartialDtype::F32).is_err());
        assert!(Tp2Rows::new(2561, 10, 64, true, PartialDtype::F32).is_err());
        assert!(Tp2Rows::new(2560, 10, 64, false, PartialDtype::F32).is_err());
        assert!(Tp2Rows::new(2560, 10, 64, true, PartialDtype::Bf16).is_err());
        Ok(())
    }

    #[test]
    fn decode_extent_matches_admitted_owner_transport() -> Result<()> {
        use crate::shared::experts::rtx::PartialDtype;
        for wire in [true, false] {
            for hidden in [4096, 5120, 6144] {
                for rows in [1, 16, 64] {
                    let partial = if wire { PartialDtype::F32 } else { PartialDtype::Bf16 };
                    let actual = Tp2Rows::new(hidden, 8, rows, wire, partial)?;
                    let planned = cuteafd_loader::serving_capacity::OwnerTp2Rows::new(
                        hidden as u64, 8, rows as u64, wire)?;
                    assert_eq!(actual.transfer_bytes() as u64, planned.bytes.max(planned.partial_bytes));
                    assert!(actual.transfer_bytes() as u64 <= planned.exchange_slot_bytes);
                }
            }
        }
        Ok(())
    }
}
