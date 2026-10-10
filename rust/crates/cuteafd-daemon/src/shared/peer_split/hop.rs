//! The hop primitive (placement P3): one residual move between the two GPUs
//! at an ownership boundary, sized and placed by the solver's
//! [`Placement::hops`](cuteafd_loader::placement::Placement).
//!
//! A hop has two phases, and the executor queues them where it chooses:
//! - [`HopLink::send`] queues the push on the source GPU's stream and returns
//!   a [`HopTicket`].
//! - [`HopLink::land`] queues the wait on the destination's stream and
//!   returns the receive buffer.
//!
//! Splitting the phases is what lets the executor own the order across
//! exchanges: a wait for a peer's result is never queued inside a unit while
//! that peer can be blocked on something this GPU pushes later.
//! [`order::check`](super::order::check) proves a schedule free of such
//! cycles on the CPU.
//!
//! Receive buffers: [`HOP_SLOTS`] per lane on each GPU that hops land on,
//! alternating by the hop's ordinal among hops into that GPU. The bytes are
//! exactly the solver's `residual hops` demand.
use super::{on_device, End, RankDevice};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::placement::{hop_buffer_bytes, Hop, HopSpec, HOP_SLOTS};
use std::ffi::c_void;

/// Both GPUs' hop receive slots and control words.
pub(crate) struct HopLink<'a> {
    library: &'a NativeLibrary,
    ranks: [RankDevice; 2],
    ends: [End<'a>; 2],
    spec: HopSpec,
    /// The placement's hops; a hop's index here keys its slot.
    hops: Vec<Hop>,
    /// Receive slots per lane on each GPU (`min(hops into it, HOP_SLOTS)`).
    per_lane: [usize; 2],
}

/// A queued push whose matching wait is still owed on the destination.
#[must_use = "every sent hop must land on its destination stream"]
#[derive(Debug)]
pub(crate) struct HopTicket {
    to: usize,
    flag: usize,
    recv: usize,
}

impl<'a> HopLink<'a> {
    /// Allocates the receive slots `hops` need on each destination (exactly
    /// [`hop_buffer_bytes`]) and the control words on both GPUs; `ranks[0]`'s
    /// device must be current, and is again on return.
    pub fn new(library: &'a NativeLibrary, ranks: [RankDevice; 2], hops: &[Hop], spec: HopSpec) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("residual-hops");
        ensure!(ranks[0].device != ranks[1].device, "a hop moves between two GPUs");
        ensure!(spec.row_bytes % 16 == 0 && spec.lanes > 0, "hop rows must be 16-byte multiples over at least one lane");
        ensure!(hops.iter().all(|h| h.from != h.to && h.from < 2 && h.to < 2 && h.row_bytes == spec.row_bytes),
            "hops must move between GPU0 and GPU1 at the spec's row bytes");
        hop_buffer_bytes(hops, &spec, 2).context("hop buffer bytes overflow")?;
        let slot_bytes = usize::try_from(spec.rows * spec.row_bytes)?;
        let flags = usize::try_from(spec.lanes * HOP_SLOTS)?;
        let per_lane = [0u8, 1].map(|gpu| hops.iter().filter(|h| h.charged() && h.to == gpu).count()
            .min(HOP_SLOTS as usize));
        let end = |rank: usize| -> Result<End<'a>> {
            on_device(library, ranks[rank].device, ranks[0].device, || {
                library.cuda_enable_peer(ranks[1 - rank].device)?;
                library.peer_exchange_initialize()?;
                let zeroed = |bytes: usize| -> Result<DeviceAllocation<'a>> {
                    let allocation = DeviceAllocation::new(library, bytes.max(256))?;
                    library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
                    Ok(allocation)
                };
                let slots = usize::try_from(spec.lanes)? * per_lane[rank];
                Ok(End {
                    recv: (0..slots).map(|_| zeroed(slot_bytes)).collect::<Result<_>>()?,
                    control: zeroed((flags + 1) * 16)?,
                })
            })
        };
        Ok(Self { library, ranks, ends: [end(0)?, end(1)?], spec, hops: hops.to_vec(), per_lane })
    }

    /// Receive bytes held on `rank` (the solver charged exactly this).
    pub fn receive_bytes(&self, rank: usize) -> u64 {
        self.ends[rank].recv.iter().map(|r| r.buffer.bytes as u64).sum()
    }

    /// The slot (per lane) hop `index` lands in: its ordinal among the hops
    /// into the same GPU, by parity.
    fn slot_of(&self, index: usize) -> Result<(usize, usize)> {
        let hop = self.hops.get(index).with_context(|| format!("hop {index} of {}", self.hops.len()))?;
        ensure!(hop.charged(), "the entry hop lands in the step input, not a hop slot");
        let ordinal = self.hops[..index].iter().filter(|h| h.charged() && h.to == hop.to).count();
        let to = usize::from(hop.to);
        Ok((to, ordinal % self.per_lane[to]))
    }

    /// Queues hop `index` of `rows` rows for `lane` on its source stream:
    /// `source` (final on that stream) into the destination's slot, then the
    /// slot's next sequence. The caller lands the ticket on the destination.
    pub fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<HopTicket> {
        let (to, slot) = self.slot_of(index)?;
        ensure!((lane as u64) < self.spec.lanes, "hop lane {lane} of {}", self.spec.lanes);
        ensure!(rows as u64 <= self.spec.rows, "hop of {rows} rows past {}", self.spec.rows);
        let (flag, recv) = (lane * HOP_SLOTS as usize + slot, lane * self.per_lane[to] + slot);
        let from = 1 - to;
        let destination = self.ends[to].recv.get(recv).context("hop receive slot")?.buffer.ptr;
        let bytes = rows * usize::try_from(self.spec.row_bytes)?;
        let (mine, theirs) = (&self.ends[from], &self.ends[to]);
        // SAFETY: peer access is enabled both ways (new); `source` is final on
        // the sender's stream and unchanged until later work on it; the
        // destination slot is read only after its matching wait, and the
        // executor reuses a slot only after the destination consumed it.
        on_device(self.library, self.ranks[from].device, self.ranks[0].device, || unsafe {
            self.library.peer_push_signal(destination, source, bytes, theirs.flag(flag), mine.send_state(flag), 0,
                self.ranks[from].stream)
        })?;
        Ok(HopTicket { to, flag, recv })
    }

    /// Queues the wait for `ticket` on its destination stream; returns the
    /// receive buffer, valid on that stream after the wait.
    pub fn land(&self, ticket: HopTicket) -> Result<*mut c_void> {
        let end = &self.ends[ticket.to];
        // SAFETY: this GPU's control words; the matching push was queued by `send`.
        on_device(self.library, self.ranks[ticket.to].device, self.ranks[0].device, || unsafe {
            self.library.peer_wait(end.flag(ticket.flag), end.recv_state(ticket.flag), self.ranks[ticket.to].stream)
        })?;
        Ok(end.recv.get(ticket.recv).context("hop receive slot")?.buffer.ptr)
    }
}

#[cfg(test)]
mod cuda_tests {
    use super::*;
    use cuteafd_loader::placement::{plan_hops, FfnMode, LayerMode};

    /// Layer ranges 0-1 on GPU0, 2-3 on GPU1, head on GPU0: one boundary hop
    /// each way per lane, landing byte-exact, with receive bytes equal to the
    /// solver's charge.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and two CUDA devices with peer access"]
    fn ranges_hop_both_ways_byte_exact_at_the_charged_bytes() -> Result<()> {
        // SAFETY: every CUDA owner below is dropped before the library.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let (w0, w1) = (LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner });
        let spec = HopSpec { row_bytes: 4 * 64 * 2, rows: 32, lanes: 2, entry_gpu: 0, head_gpu: 0 };
        let hops = plan_hops(&[w0, w0, w1, w1], &spec);
        assert_eq!(hops.len(), 2);
        library.cuda_set_device(0)?;
        let mut ranks = [RankDevice { device: 0, stream: std::ptr::null_mut() }, RankDevice { device: 1, stream: std::ptr::null_mut() }];
        for rank in &mut ranks {
            rank.stream = on_device(&library, rank.device, 0, || library.cuda_stream_create())?;
        }
        let result = (|| -> Result<()> {
            let link = HopLink::new(&library, ranks, &hops, spec)?;
            let charged = hop_buffer_bytes(&hops, &spec, 2).unwrap();
            assert_eq!([link.receive_bytes(0), link.receive_bytes(1)], [charged[0], charged[1]]);
            let rows = 17;
            let bytes = rows * spec.row_bytes as usize;
            for lane in 0..2 {
                for (index, hop) in hops.iter().enumerate() {
                    let from = usize::from(hop.from);
                    let pattern: Vec<u8> = (0..bytes).map(|i| (i * 7 + lane * 13 + index * 31) as u8).collect();
                    let source = on_device(&library, from as i32, 0, || {
                        let source = DeviceAllocation::new(&library, bytes)?;
                        library.copy_h2d(source.buffer, &pattern)?;
                        Ok(source)
                    })?;
                    let ticket = link.send(index, lane, source.buffer.ptr, rows)?;
                    let landed = link.land(ticket)?;
                    let to = usize::from(hop.to);
                    let mut out = vec![0u8; bytes];
                    on_device(&library, to as i32, 0, || {
                        // SAFETY: the destination stream waited for the push; the source drains with its stream.
                        unsafe { library.cuda_stream_synchronize(ranks[to].stream)? };
                        unsafe { library.cuda_stream_synchronize(ranks[from].stream) }.ok();
                        library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { ptr: landed, bytes,
                            device_id: to as i32, flags: 0 })
                    })?;
                    assert_eq!(out, pattern, "lane {lane} hop {index}");
                    on_device(&library, from as i32, 0, || { drop(source); Ok(()) })?;
                }
            }
            Ok(())
        })();
        for rank in ranks {
            on_device(&library, rank.device, 0, || {
                // SAFETY: nothing else uses the test streams.
                unsafe { library.cuda_stream_synchronize(rank.stream)?; library.cuda_stream_destroy(rank.stream) }
            })?;
        }
        result
    }
}
