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
use crate::shared::memory::device::{Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::placement::{hop_buffer_bytes, Hop, HopSpec, HopStorage, HOP_SLOTS};
use std::ffi::c_void;

/// One directed transfer stream per request lane. No intermediate device or
/// host buffer is needed: copies land in the destination's existing inputs.
pub(crate) struct DirectInputHop<'a> {
    source: Device<'a>,
    contract: HopSpec,
    destination: Stream<'a>,
    /// Direct-input storage is V4.1-only today; v41_peer_copy is its FFI entry point.
    /// Device-ordered passes: an SM copy on the
    /// destination GPU ordered by the stage chain, with no host wait.
    sm: Option<cuteafd_ffi::V41PeerCopy<'a>>,
}
impl<'a> DirectInputHop<'a> {
    pub fn new(source: Device<'a>, destination: Device<'a>, spec: HopSpec) -> Result<Self> {
        validate_direct_input_contract(source.id, destination.id, &[], spec)?;
        source.run(|| source.library.cuda_enable_peer(destination.id))?;
        destination.run(|| destination.library.cuda_enable_peer(source.id))?;
        let sm = crate::shared::memory::chain::device_enabled()
            .then(|| destination.run(|| destination.library.v41_peer_copy())).transpose()?;
        Ok(Self {
            source,
            contract: spec,
            destination: Stream::new(destination)?,
            sm,
        })
    }
    /// Admission checks the planned endpoints against the required geometry.
    pub fn bind(&mut self, hops: &[Hop], spec: HopSpec) -> Result<()> {
        validate_direct_input_contract(self.source.id, self.destination.device.id, hops, spec)?;
        ensure!(self.contract == spec, "direct-input hop geometry differs from construction");
        Ok(())
    }

    pub fn destination(&self) -> Device<'a> { self.destination.device }

    #[cfg(test)]
    pub fn stream_raw(&self) -> *mut c_void { self.destination.raw }

    /// # Safety
    /// Sources are complete and all buffers remain exclusively owned until the
    /// copy finishes or cancellation drains. Never queue behind an unready
    /// producer event: doing so can stall otherwise independent peer DMA work.
    pub async unsafe fn copy<const N: usize>(
        &mut self,
        sources: [CuteafdDeviceBuffer; N],
        destinations: [CuteafdDeviceBuffer; N],
    ) -> Result<()> {
        ensure!(
            sources
                .iter()
                .zip(destinations)
                .all(|(s, d)| s.device_id == self.source.id
                    && d.device_id == self.destination.device.id
                    && s.bytes <= d.bytes),
            "block transfer devices or extents differ"
        );
        validate_direct_input_extent(self.contract, sources.iter().map(|source| source.bytes))?;
        let device = self.destination.device;
        if let Some(sm) = self.sm.as_ref().filter(|_| crate::shared::memory::chain::deferred()) {
            // The SM copy follows the chain head (the producers on the source
            // GPU) on the device and becomes the new head: no host wait.
            return device.run(|| unsafe {
                crate::shared::memory::chain::join(device.library, self.destination.raw)?;
                for (source, destination) in sources.into_iter().zip(destinations) {
                    sm.launch(destination, source, source.bytes, self.destination.raw)?;
                }
                crate::shared::memory::chain::finish(device.library, self.destination.raw)
            });
        }
        // Peer DMA never waits on an unresolved event; settle chained producers.
        crate::shared::memory::chain::settle(device.library)?;
        let queued = device.run(|| {
            for (source, destination) in sources.into_iter().zip(destinations) {
                unsafe {
                    device.library.copy_peer_async(
                        destination,
                        source,
                        source.bytes,
                        self.destination.raw,
                    )?;
                }
            }
            Ok(())
        });
        if let Err(error) = queued {
            self.destination.drain()?;
            return Err(error);
        }
        // Stream::wait retains and drains the stream on failure/cancellation.
        self.destination.wait().await
    }
}

fn validate_direct_input_extent(spec: HopSpec, extents: impl IntoIterator<Item = usize>) -> Result<()> {
    let bytes = extents.into_iter().try_fold(0usize, |sum, bytes| sum.checked_add(bytes))
        .context("direct-input hop extent overflow")?;
    let row_bytes = usize::try_from(spec.row_bytes)?;
    ensure!(row_bytes > 0 && bytes > 0 && bytes % row_bytes == 0 && (bytes / row_bytes) as u64 <= spec.rows,
        "direct-input hop exceeds admitted row geometry");
    Ok(())
}

fn validate_direct_input_contract(source: i32, destination: i32, hops: &[Hop], spec: HopSpec) -> Result<()> {
    ensure!(spec.storage == HopStorage::DirectInput && spec.row_bytes > 0 && spec.row_bytes % 16 == 0
        && spec.rows > 0 && spec.lanes > 0 && source != destination,
        "invalid direct-input hop contract");
    ensure!(hops.iter().all(|hop| hop.storage == spec.storage && hop.row_bytes == spec.row_bytes
        && i32::from(hop.from) == source && i32::from(hop.to) == destination),
        "direct-input hop endpoints or storage differ from admission");
    Ok(())
}

/// Both GPUs' hop receive slots and control words.
pub(crate) struct HopLink<'a> {
    library: &'a NativeLibrary,
    ranks: [RankDevice; 2],
    ends: [End<'a>; 2],
    spec: HopSpec,
    /// A private instance retains logical plan identity but owns only physical lane zero.
    private_lane: Option<usize>,
    /// The placement's hops; a hop's index here keys its slot.
    hops: Vec<Hop>,
    /// Receive slots per lane on each GPU (`min(hops into it, HOP_SLOTS)`).
    per_lane: [usize; 2],
    abort: Option<[*const u32; 2]>,
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
        ensure!(spec.storage == cuteafd_loader::placement::HopStorage::ReceiveSlots,
            "HopLink requires receive-slot storage; direct-input hops use the destination's own buffers");
        ensure!(hops.iter().all(|hop| hop.storage == spec.storage), "hop storage differs from spec");
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
                    progress: None,
                })
            })
        };
        Ok(Self { library, ranks, ends: [end(0)?, end(1)?], spec, private_lane: None,
            hops: hops.to_vec(), per_lane, abort: None })
    }

    /// Independent streams carry one logical lane, so allocate only its slots.
    pub(crate) fn new_private_lane(library: &'a NativeLibrary, ranks: [RankDevice; 2],
        hops: &[Hop], spec: HopSpec, lane: usize) -> Result<Self> {
        let allocation = private_lane_spec(spec, lane)?;
        let mut link = Self::new(library, ranks, hops, allocation)?;
        link.spec = spec;
        link.private_lane = Some(lane);
        Ok(link)
    }

    pub(crate) fn validate_dispatch(&self, lane: usize, shared: bool) -> Result<()> {
        ensure!((lane as u64) < self.spec.lanes, "hop dispatch lane outside admission");
        ensure!(self.private_lane.is_none_or(|owner| owner == lane && !shared),
            "private hop instance differs from pipeline lane");
        Ok(())
    }

    /// Startup-only borrowing of the routed exchange's abort words. The
    /// engine must keep that exchange alive until every hop consumer drains.
    pub(crate) fn attach_abort(&mut self, abort: [*const u32; 2]) -> Result<()> {
        ensure!(abort.iter().all(|word| !word.is_null()), "hop abort words must be live");
        self.library.peer_abort_available()?;
        self.abort = Some(abort);
        Ok(())
    }

    /// Serving layer executors borrow the exact same terminal owner as entry
    /// and FFN exchanges; a legacy non-abortable hop cannot be bound to them.
    pub(crate) fn validate_binding(&self, plan: &cuteafd_loader::placement::LayerExecutionPlan,
        peer: &super::PeerExchange<'_>) -> Result<()> {
        ensure!(self.spec == plan.spec() && self.hops == plan.hops(), "hop storage differs from layer plan");
        let words = [peer.abort_word(0)?.context("rank 0 exchange is not abortable")?,
            peer.abort_word(1)?.context("rank 1 exchange is not abortable")?];
        ensure!(self.abort == Some(words), "hops must borrow the layer exchange's terminal owner");
        ensure!((0..2).all(|rank| self.ranks[rank].device == peer.device(rank)
            && self.ranks[rank].stream == peer.stream(rank)), "hop and FFN streams differ");
        peer.require_live()
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
        let physical_lane = physical_lane(self.spec, self.private_lane, lane)?;
        let (flag, recv) = (physical_lane * HOP_SLOTS as usize + slot, physical_lane * self.per_lane[to] + slot);
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
            match self.abort {
                Some(abort) => self.library.peer_wait_abortable(end.flag(ticket.flag), end.recv_state(ticket.flag),
                    abort[ticket.to], self.ranks[ticket.to].stream),
                None => self.library.peer_wait(end.flag(ticket.flag), end.recv_state(ticket.flag), self.ranks[ticket.to].stream),
            }
        })?;
        Ok(end.recv.get(ticket.recv).context("hop receive slot")?.buffer.ptr)
    }
}

fn private_lane_spec(spec: HopSpec, lane: usize) -> Result<HopSpec> {
    ensure!((lane as u64) < spec.lanes, "private hop lane outside admission");
    Ok(HopSpec { lanes: 1, ..spec })
}

fn physical_lane(spec: HopSpec, private: Option<usize>, lane: usize) -> Result<usize> {
    ensure!((lane as u64) < spec.lanes, "hop lane outside admission");
    if let Some(owner) = private {
        ensure!(owner == lane, "private hop instance belongs to another lane");
        Ok(0)
    } else { Ok(lane) }
}

#[cfg(test)]
mod cuda_tests {
    use super::*;
    use cuteafd_loader::placement::{plan_hops, FfnMode, LayerMode};

    #[test]
    fn private_hop_instances_charge_one_lane_and_sum_to_the_admitted_forecast() -> Result<()> {
        let (w0, w1) = (LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
            LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner });
        for lanes in [1, 2] {
            for modes in [vec![w0, w0], vec![w0, w1], vec![w0, w1, w0, w1, w0, w1]] {
                let spec = HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: 16384,
                    rows: 640, lanes, entry_gpu: 0, head_gpu: 0,
                    terminal: cuteafd_loader::placement::TerminalHome::AttentionOwner };
                let hops = plan_hops(&modes, &spec)?;
                let admitted = hop_buffer_bytes(&hops, &spec, 2).unwrap();
                let single = hop_buffer_bytes(&hops, &HopSpec { lanes: 1, ..spec }, 2).unwrap();
                let mut total = [0u64; 2];
                for lane in 0..lanes as usize {
                    let allocation = private_lane_spec(spec, lane)?;
                    assert_eq!(allocation.lanes, 1);
                    let bytes = hop_buffer_bytes(&hops, &allocation, 2).unwrap();
                    assert_eq!(bytes, single);
                    assert_eq!(physical_lane(spec, Some(lane), lane)?, 0);
                    assert_eq!(physical_lane(spec, None, lane)?, lane);
                    for rank in 0..2 { total[rank] += bytes[rank]; }
                    assert!(physical_lane(spec, Some(lane), 1 - lane).is_err());
                }
                assert_eq!(total.as_slice(), admitted);
                assert!(private_lane_spec(spec, lanes as usize).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn direct_input_binding_rejects_receive_storage_and_wrong_endpoints() -> Result<()> {
        let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(2);
        let modes: Vec<_> = (0..40).map(|layer| LayerMode::Whole {
            gpu: u8::from(layer >= 20), ffn: FfnMode::Owner,
        }).collect();
        let hops = plan_hops(&modes, &spec)?;
        validate_direct_input_contract(0, 1, &hops, spec)?;
        validate_direct_input_contract(1, 0, &[], spec)?;
        assert!(validate_direct_input_contract(1, 0, &hops, spec).is_err());
        assert!(validate_direct_input_contract(0, 0, &[], spec).is_err());
        for bad in [HopSpec { storage: HopStorage::ReceiveSlots, ..spec },
            HopSpec { row_bytes: 0, ..spec }, HopSpec { row_bytes: 17, ..spec },
            HopSpec { rows: 0, ..spec }, HopSpec { lanes: 0, ..spec }] {
            assert!(validate_direct_input_contract(0, 1, &hops, bad).is_err());
        }
        let mut bad = hops;
        bad[0].storage = HopStorage::ReceiveSlots;
        assert!(validate_direct_input_contract(0, 1, &bad, spec).is_err());
        Ok(())
    }

    #[test]
    fn direct_input_extent_always_enforces_admitted_geometry() -> Result<()> {
        let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(2);
        let row_bytes = usize::try_from(spec.row_bytes)?;
        for rows in [1, spec.rows as usize] {
            validate_direct_input_extent(spec, [rows * 40960, rows * 16])?;
        }
        for extents in [vec![], vec![0], vec![row_bytes - 1], vec![row_bytes + 1],
            vec![(spec.rows as usize + 1) * row_bytes], vec![usize::MAX, 1]] {
            assert!(validate_direct_input_extent(spec, extents).is_err());
        }
        Ok(())
    }

    /// Layer ranges 0-1 on GPU0, 2-3 on GPU1, head on GPU0: one boundary hop
    /// each way per lane, landing byte-exact, with receive bytes equal to the
    /// solver's charge.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and two CUDA devices with peer access"]
    fn ranges_hop_both_ways_byte_exact_at_the_charged_bytes() -> Result<()> {
        // SAFETY: every CUDA owner below is dropped before the library.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let (w0, w1) = (LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner });
        let spec = HopSpec { storage: cuteafd_loader::placement::HopStorage::ReceiveSlots, row_bytes: 4 * 64 * 2, rows: 32, lanes: 2, entry_gpu: 0, head_gpu: 0, terminal: cuteafd_loader::placement::TerminalHome::AttentionOwner };
        let hops = plan_hops(&[w0, w0, w1, w1], &spec).unwrap();
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
            let private = (0..2).map(|lane| HopLink::new_private_lane(&library, ranks, &hops, spec, lane))
                .collect::<Result<Vec<_>>>()?;
            let single = hop_buffer_bytes(&hops, &HopSpec { lanes: 1, ..spec }, 2).unwrap();
            for rank in 0..2 {
                for instance in &private { assert_eq!(instance.receive_bytes(rank), single[rank]); }
                assert_eq!(private.iter().map(|instance| instance.receive_bytes(rank)).sum::<u64>(), charged[rank]);
            }
            let rows = 17;
            let bytes = rows * spec.row_bytes as usize;
            for lane in 0..2 {
                let link = &private[lane];
                assert!(link.validate_dispatch(1 - lane, false).is_err());
                assert!(link.validate_dispatch(lane, true).is_err());
                link.validate_dispatch(lane, false)?;
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
