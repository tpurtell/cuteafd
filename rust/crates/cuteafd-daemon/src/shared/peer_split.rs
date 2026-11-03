//! Two-GPU head split plumbing shared by the families that split attention
//! heads over the coordinator's two GPUs (MiMo V2.6 Pro, GLM 5.x): the ranks'
//! devices and streams, and both ends of the exchange between them.
//!
//! Exchange: size-routed pushes into the other GPU's receive buffers with a release
//! flag, and a spinning acquire wait on the receiving stream (the
//! `peer_exchange` kernels; device-side sequence numbers, so graphs replay
//! them). Every slot has its own flag and sequences: each push into a slot is
//! matched, in order, by one wait on that slot on the other GPU, whatever the
//! interleaving of slots (prefill lanes); the host queues a push before it
//! waits on either stream. Callers pick slots by layer parity (and lane), so a
//! push never lands on rows the other GPU may still read. One extra flag
//! (`DIRECT`) serves pushes into buffers outside the slots.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};

// K3 (GLM 5.3 `context`) is the first executor; until it lands nothing outside tests uses this.
#[allow(dead_code)]
pub(crate) mod context;
// Family bindings qualify these shared executors incrementally.
#[allow(dead_code)]
pub(crate) mod attention;
pub(crate) mod hop;
pub(crate) mod order;
use cuteafd_ffi::peer_exchange::{PeerDiagnostic, PeerProgress};
use cuteafd_ffi::NativeLibrary;
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

static PEER_ISSUE: AtomicU64 = AtomicU64::new(1);

/// Probe both directions before admitting or loading a two-GPU layout. Cards
/// without peer access (including GeForce) retain the single-device path.
pub(crate) fn probed_device(library: &NativeLibrary, home: i32, requested: Option<i32>) -> Result<Option<i32>> {
    let Some(peer) = requested else { return Ok(None) };
    ensure!(home != peer, "--split-device must differ from --device");
    let result = if std::env::var("CUTEAFD_P2P_PROBE").is_ok_and(|v| v == "unavailable") {
        Err(anyhow::anyhow!("P2P probe forced unavailable"))
    } else {
        crate::shared::memory::device::Device { library, id: home }.run(|| library.cuda_enable_peer(peer))
            .and_then(|()| crate::shared::memory::device::Device { library, id: peer }.run(|| library.cuda_enable_peer(home)))
    };
    // A failed peer capability must not leave subsequent allocations on the peer.
    library.cuda_set_device(home)?;
    Ok(peer_probe_result(peer, result))
}

fn peer_probe_result(peer: i32, result: Result<()>) -> Option<i32> {
    match result {
        Ok(()) => Some(peer),
        Err(error) => {
            tracing::warn!(peer, reason = %format!("{error:#}"), "P2P unavailable; serving from one GPU");
            None
        }
    }
}

#[cfg(test)]
mod probe_tests {
    #[test]
    fn unavailable_peer_falls_back_before_loading() {
        assert_eq!(super::peer_probe_result(1, Err(anyhow::anyhow!("unavailable"))), None);
        assert_eq!(super::peer_probe_result(1, Ok(())), Some(1));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalState {
    Active,
    Aborting,
    Drained,
    Retained,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum TerminalPeerError {
    #[error("terminal head-split engine cannot be reused ({0:?})")]
    Closed(TerminalState),
    #[error("terminal head-split abort/drain failed; queued storage must be retained: {0}")]
    Retain(String),
    #[error("head split was not constructed with terminal cancellation ownership")]
    NotAbortable,
}

struct AbortSide<'a> {
    library: &'a NativeLibrary,
    device: i32,
    word: DeviceAllocation<'a>,
    stream: *mut c_void,
}

impl Drop for AbortSide<'_> {
    fn drop(&mut self) {
        // No blocked wait is ever enqueued on this independent stream. The
        // complete engine is quarantined instead of dropped after a failed drain.
        if let Err(error) = on_device(self.library, self.device, || {
            // SAFETY: this owned stream only publishes into the still-live word.
            unsafe {
                self.library.cuda_stream_synchronize(self.stream)?;
                self.library.cuda_stream_destroy(self.stream)
            }
        }) {
            tracing::error!(error = %format!("{error:#}"), "destroying terminal-abort stream");
        }
    }
}

struct TerminalAbort<'a> {
    sides: [AbortSide<'a>; 2],
    state: Cell<TerminalState>,
}

/// One GPU of a head split: its device and the stream its work runs on.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RankDevice {
    pub device: i32,
    pub stream: *mut c_void,
}

fn check_cooperative_wait_stream(waiter: RankDevice, producer: RankDevice) -> Result<()> {
    ensure!(waiter.device == producer.device && waiter.stream == producer.stream && !producer.stream.is_null(),
        "resident peer wait is not serialized with its cooperative producer");
    Ok(())
}

#[cfg(test)]
mod cooperative_wait_tests {
    use super::*;

    #[test]
    fn independent_resident_wait_stream_requires_an_event_edge() {
        let producer = RankDevice { device: 1, stream: 1usize as *mut c_void };
        assert!(check_cooperative_wait_stream(producer, producer).is_ok());
        assert!(check_cooperative_wait_stream(RankDevice { stream: 2usize as *mut c_void, ..producer }, producer).is_err());
        assert!(check_cooperative_wait_stream(RankDevice { device: 0, ..producer }, producer).is_err());
        let default = RankDevice { stream: std::ptr::null_mut(), ..producer };
        assert!(check_cooperative_wait_stream(default, default).is_err());
    }
}

fn run_on_rank<T>(library: &NativeLibrary, rank: RankDevice, body: impl FnOnce() -> Result<T>) -> Result<T> {
    on_device(library, rank.device, body)
}

/// Runs synchronous work on `device`, restoring the caller's device on every exit.
pub(crate) fn on_device<T>(
    library: &NativeLibrary,
    device: i32,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    crate::shared::memory::device::Device { library, id: device }.run(body)
}

/// One GPU's end: receive buffers and, per flag (slot, plus [`DIRECT`]), four
/// control words: `[0]` the sequence the other GPU publishes, `[1..3]` this
/// GPU's send state, `[3]` its receive state.
struct End<'a> {
    recv: Vec<DeviceAllocation<'a>>,
    control: DeviceAllocation<'a>,
    progress: Option<DeviceAllocation<'a>>,
}

impl End<'_> {
    fn word(&self, flag: usize, word: usize) -> *mut u32 {
        debug_assert!((flag * 4 + word) * 4 < self.control.buffer.bytes);
        // SAFETY: four words per flag inside the zeroed control allocation (sized in `new`).
        unsafe { self.control.buffer.ptr.cast::<u32>().add(flag * 4 + word) }
    }

    fn flag(&self, flag: usize) -> *mut u32 {
        self.word(flag, 0)
    }

    fn send_state(&self, flag: usize) -> *mut u32 {
        self.word(flag, 1)
    }

    fn recv_state(&self, flag: usize) -> *mut u32 {
        self.word(flag, 3)
    }
}

/// The flag of pushes into buffers outside the receive slots.
pub(crate) const DIRECT: usize = usize::MAX;

pub(crate) fn validate_slot_sizes(devices: [i32; 2], sizes: &[usize]) -> Result<()> {
    ensure!(devices[0] != devices[1],
        "head split requires two different GPUs, got devices {} and {}", devices[0], devices[1]);
    ensure!(!sizes.is_empty(), "head split requires at least one receive slot");
    for (slot, &bytes) in sizes.iter().enumerate() {
        ensure!(bytes > 0 && bytes % 16 == 0,
            "head split receive slot {slot} has invalid size {bytes} bytes; expected nonzero 16-byte rows");
    }
    Ok(())
}

/// Both ends of the exchange between ranks 0 and 1.
pub(crate) struct PeerExchange<'a> {
    library: &'a NativeLibrary,
    ranks: [RankDevice; 2],
    ends: [End<'a>; 2],
    abort: Option<TerminalAbort<'a>>,
    decode_lane_bytes: usize,
    trace_name: Cell<&'static str>,
    host_ordinals: RefCell<Vec<[[u64; 2]; 2]>>,
}

/// Decode copies stay on SMs; wider transfers use the sender's copy engine.
fn validate_decode_extent(bytes: usize) -> Result<()> {
    ensure!(bytes > 0 && bytes % 16 == 0, "decode peer extent must contain aligned rows");
    Ok(())
}

impl<'a> PeerExchange<'a> {
    /// Physical retained owners, never reconstructed from requested slot extents.
    pub(crate) fn allocation_buffers(&self) -> Vec<(usize, String, cuteafd_ffi::CuteafdDeviceBuffer)> {
        let mut out = Vec::new();
        for (rank, end) in self.ends.iter().enumerate() {
            for (slot, recv) in end.recv.iter().enumerate() {
                out.push((rank, format!("peer.recv.{slot}"), recv.buffer));
            }
            out.push((rank, "peer.control".into(), end.control.buffer));
            if let Some(progress) = &end.progress {
                out.push((rank, "peer.progress".into(), progress.buffer));
            }
            if let Some(abort) = &self.abort {
                out.push((rank, "peer.abort".into(), abort.sides[rank].word.buffer));
            }
        }
        out
    }

    /// Ordinary host backing retained by opt-in progress diagnostics.
    pub(crate) fn diagnostic_host_backing(&self) -> Result<Option<(u64, u64)>> {
        let ordinals = self.host_ordinals.try_borrow()?;
        if ordinals.capacity() == 0 { return Ok(None); }
        let bytes = ordinals.capacity().checked_mul(std::mem::size_of::<[[u64; 2]; 2]>())
            .context("peer diagnostic host backing overflow")?;
        Ok(Some((ordinals.as_ptr() as u64, u64::try_from(bytes)?)))
    }

    /// Enables peer access both ways and allocates `slots` zeroed receive
    /// buffers of `slot_bytes` and the control words on each rank; `ranks[0]`'s
    /// device must be current (and is again on return).
    pub fn new(
        library: &'a NativeLibrary,
        ranks: [RankDevice; 2],
        slots: usize,
        slot_bytes: usize,
        decode_lane_bytes: usize,
    ) -> Result<Self> {
        Self::new_sized(library, ranks, &vec![slot_bytes; slots], decode_lane_bytes)
    }

    fn new_sized(library: &'a NativeLibrary, ranks: [RankDevice; 2], sizes: &[usize], decode_lane_bytes: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("peer-split");
        validate_decode_extent(decode_lane_bytes)?;
        let slots = sizes.len();
        validate_slot_sizes(ranks.map(|rank| rank.device), sizes)?;
        let diagnostics = std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1");
        let end = |rank: usize| -> Result<End<'a>> {
            on_device(library, ranks[rank].device, || {
                library.cuda_enable_peer(ranks[1 - rank].device)?;
                // Lazy native module loading may synchronize behind an unmatched
                // peer wait. Force every registered helper before any wait exists.
                let started = std::time::Instant::now();
                library.tp2_native_initialize()?;
                tracing::info!(device = ranks[rank].device,
                    elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                    "TP2 native kernels loaded before peer waits");
                library.peer_exchange_initialize()?;
                if diagnostics { library.peer_diagnostic_initialize()?; }
                let zeroed = |bytes: usize| -> Result<DeviceAllocation<'a>> {
                    let allocation = DeviceAllocation::new(library, bytes.max(256))?;
                    library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
                    Ok(allocation)
                };
                Ok(End {
                    recv: sizes.iter()
                        .map(|&bytes| zeroed(bytes))
                        .collect::<Result<_>>()?,
                    control: zeroed((slots + 1) * 16)?,
                    progress: if diagnostics { Some(zeroed((slots + 1) * std::mem::size_of::<PeerProgress>())?) } else { None },
                })
            })
        };
        Ok(Self {
            library,
            ranks,
            ends: [end(0)?, end(1)?],
            abort: None,
            decode_lane_bytes,
            trace_name: Cell::new("unnamed"),
            host_ordinals: RefCell::new(if diagnostics { vec![[[0; 2]; 2]; slots + 1] } else { Vec::new() }),
        })
    }

    /// Optional MiMo terminal ownership. Other families retain the legacy ABI.
    pub fn new_abortable(
        library: &'a NativeLibrary,
        ranks: [RankDevice; 2],
        slots: usize,
        slot_bytes: usize,
        decode_lane_bytes: usize,
    ) -> Result<Self> {
        Self::new_abortable_sized(library, ranks, &vec![slot_bytes; slots], decode_lane_bytes)
    }

    /// Receive extents are admitted per slot, including private decode slots.
    pub fn new_abortable_sized(library: &'a NativeLibrary, ranks: [RankDevice; 2], sizes: &[usize], decode_lane_bytes: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("peer-split");
        validate_decode_extent(decode_lane_bytes)?;
        library.peer_abort_available()?;
        let mut exchange = Self::new_sized(library, ranks, sizes, decode_lane_bytes)?;
        let side = |rank: usize| {
            on_device(library, ranks[rank].device, || {
                let probe = |stage| crate::shared::inventory::startup_probe(
                    library, &[ranks[rank].device], stage);
                probe("peer-abort-before-preload")?;
                library.peer_abort_initialize()?;
                probe("peer-abort-after-preload")?;
                let word = DeviceAllocation::new(library, 256)?;
                library.cuda_zero_bytes(word.buffer, word.buffer.bytes)?;
                probe("peer-abort-after-word")?;
                let stream = library.cuda_stream_create()?;
                probe("peer-abort-after-publisher-stream")?;
                Ok(AbortSide {
                    library,
                    device: ranks[rank].device,
                    word,
                    stream,
                })
            })
        };
        exchange.abort = Some(TerminalAbort {
            sides: [side(0)?, side(1)?],
            state: Cell::new(TerminalState::Active),
        });
        Ok(exchange)
    }

    /// Startup-only pointer map for failure-only CUDA watchdog messages. Do
    /// not download control words or add synchronization to the exchange path.
    pub fn trace_identity(&self, name: &'static str) {
        self.trace_name.set(name);
        if self.ends[0].progress.is_none() { return }
        for rank in 0..2 {
            let end = &self.ends[rank];
            for flag in 0..=end.recv.len() {
                let abort = self.abort.as_ref().map(|a| a.sides[rank].word.buffer.ptr);
                tracing::info!(exchange = name, rank, device = self.ranks[rank].device,
                    stream = ?self.ranks[rank].stream, flag_index = flag,
                    direct = flag == end.recv.len(), flag = ?end.flag(flag),
                    send_state = ?end.send_state(flag), receive_state = ?end.recv_state(flag),
                    abort = ?abort, "peer wait watchdog pointer map");
            }
        }
    }

    fn diagnostic(&self, rank: usize, slot: usize, operation: &'static str) -> Option<PeerDiagnostic> {
        let local = self.ends[rank].progress.as_ref()?;
        let peer = self.ends[1 - rank].progress.as_ref()?;
        let issue = PEER_ISSUE.fetch_add(1, Ordering::Relaxed);
        let op = usize::from(operation == "wait");
        let ordinal = {
            let mut ordinals = self.host_ordinals.borrow_mut();
            ordinals[slot][rank][op] += 1;
            ordinals[slot][rank][op]
        };
        // Graph replay retains issue/ordinal, but advances the actual device
        // sequence. Never label this host ordinal as a publication sequence.
        tracing::info!(exchange = self.trace_name.get(), plane = ?self.ends[0].control.buffer.ptr,
            rank, slot, operation, issue, host_ordinal = ordinal, seq = "device-runtime",
            stream = ?self.stream(rank), "peer host issue begin");
        Some(PeerDiagnostic {
            local: local.buffer.ptr.cast(), peer: peer.buffer.ptr.cast(),
            local_control: self.ends[rank].control.buffer.ptr.cast(),
            peer_control: self.ends[1 - rank].control.buffer.ptr.cast(),
            issue, host_stream: self.stream(rank) as usize as u64,
            rank: rank as u32, slot: slot as u32, slots: (self.ends[rank].recv.len() + 1) as u32,
        })
    }

    fn diagnostic_return(&self, diagnostic: Option<PeerDiagnostic>, result: &Result<()>) {
        if let Some(d) = diagnostic {
            tracing::info!(exchange = self.trace_name.get(), issue = d.issue,
                rank = d.rank, slot = d.slot, stream = d.host_stream,
                ok = result.is_ok(), "peer host issue returned");
        }
    }

    pub fn terminal_state(&self) -> TerminalState {
        self.abort
            .as_ref()
            .map_or(TerminalState::Active, |abort| abort.state.get())
    }

    pub fn require_live(&self) -> Result<()> {
        let state = self.terminal_state();
        if state != TerminalState::Active {
            return Err(TerminalPeerError::Closed(state).into());
        }
        Ok(())
    }

    /// Terminal only: publish both abort words before draining either compute
    /// stream. A failure requires the complete engine/native owners to be
    /// quarantined; this object must then never be dropped or reused.
    pub fn publish_abort(&self) -> Result<()> {
        let abort = self.abort.as_ref().ok_or(TerminalPeerError::NotAbortable)?;
        match abort.state.get() {
            TerminalState::Drained => return Ok(()),
            TerminalState::Active => abort.state.set(TerminalState::Aborting),
            state => return Err(TerminalPeerError::Closed(state).into()),
        }
        let mut failures = Vec::new();
        for (rank, side) in abort.sides.iter().enumerate() {
            // SAFETY: independent owned stream, preloaded publisher, persistent
            // word shared by every wait including both lanes and graph replay.
            if let Err(error) = self.on(rank, || unsafe {
                self.library
                    .peer_abort_publish(side.word.buffer.ptr.cast(), side.stream)
            }) {
                failures.push(format!("publish rank {rank}: {error:#}"));
            }
        }
        for (rank, side) in abort.sides.iter().enumerate() {
            // SAFETY: no compute dependency or wait is queued on this stream.
            if let Err(error) = self.on(rank, || unsafe {
                self.library.cuda_stream_synchronize(side.stream)
            }) {
                failures.push(format!("publish drain rank {rank}: {error:#}"));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            abort.state.set(TerminalState::Retained);
            Err(TerminalPeerError::Retain(failures.join("; ")).into())
        }
    }

    /// External QPs must quiesce while all owners remain live between abort
    /// publication and this compute drain. Only the full engine can prove all
    /// intake copies also drained and publish the final ownership state.
    pub fn drain_compute(&self) -> Result<()> {
        ensure!(
            self.terminal_state() == TerminalState::Aborting,
            "compute drain requires visible terminal abort words"
        );
        let mut failures = Vec::new();
        for rank in 0..2 {
            // SAFETY: abort words are visible; every graph/plane owner remains
            // live until the engine also drains the external intake streams.
            if let Err(error) = self.on(rank, || unsafe {
                self.library
                    .cuda_stream_synchronize(self.ranks[rank].stream)
            }) {
                failures.push(format!("compute drain rank {rank}: {error:#}"));
            }
        }
        ensure!(
            failures.is_empty(),
            "terminal compute drainage failed: {}",
            failures.join("; ")
        );
        Ok(())
    }

    pub fn finish_terminal(&self, drained: bool) {
        if let Some(abort) = &self.abort {
            abort.state.set(if drained {
                TerminalState::Drained
            } else {
                TerminalState::Retained
            });
        }
    }

    #[cfg(test)]
    pub(crate) fn fault_fixture_control(&self, rank: usize) -> Result<Vec<u8>> {
        self.on(rank, || {
            let buffer = self.ends[rank].control.buffer;
            let mut bytes = vec![0; buffer.bytes];
            self.library.copy_d2h(&mut bytes, buffer)?;
            Ok(bytes)
        })
    }

    /// External fixture cleanup only: publish the already-owned words without
    /// changing terminal state or allowing reuse, then drain the two streams.
    #[cfg(test)]
    pub(crate) fn fault_fixture_external_drain(&self) -> Result<()> {
        self.library.terminal_fault_fixture_clear()?;
        let abort = self.abort.as_ref().context("fixture has no abort words")?;
        for (rank, side) in abort.sides.iter().enumerate() {
            self.on(rank, || {
                // SAFETY: live fixture word and independent publisher stream;
                // injection is cleared and no state/counters are rewritten.
                unsafe {
                    self.library
                        .peer_abort_publish(side.word.buffer.ptr.cast(), side.stream)?;
                    self.library.cuda_stream_synchronize(side.stream)
                }
            })?;
        }
        for rank in 0..2 {
            self.on(rank, || {
                // SAFETY: persistent abort word is visible; fixture owners live.
                unsafe {
                    self.library
                        .cuda_stream_synchronize(self.ranks[rank].stream)
                }
            })?;
        }
        Ok(())
    }

    /// Rank `rank`'s terminal abort word (None: a legacy, non-abortable
    /// exchange). Other exchanges on the same streams (the context exchange)
    /// spin on it too, so this exchange's publish/drain covers their waits;
    /// they must be dropped before this exchange.
    #[allow(dead_code)] // first used by K3's context executor
    pub fn abort_word(&self, rank: usize) -> Result<Option<*const u32>> {
        ensure!(rank < 2, "abort word of rank {rank}");
        Ok(self.abort.as_ref().map(|abort| abort.sides[rank].word.buffer.ptr.cast_const().cast()))
    }

    /// Startup check for exchanges sharing a full-grid cooperative producer.
    /// A different wait stream needs an explicit producer-event prerequisite
    /// instead (as the remote draft bridge enforces at every context wave).
    pub(crate) fn require_cooperative_producer_stream(&self, rank: usize, producer: RankDevice) -> Result<()> {
        ensure!(rank < 2, "invalid cooperative peer-wait rank {rank}");
        check_cooperative_wait_stream(self.ranks[rank], producer)
    }

    pub fn stream(&self, rank: usize) -> *mut c_void {
        self.ranks[rank].stream
    }

    pub fn device(&self, rank: usize) -> i32 {
        self.ranks[rank].device
    }

    /// Runs `body` with rank `rank`'s device current.
    pub fn on<T>(&self, rank: usize, body: impl FnOnce() -> Result<T>) -> Result<T> {
        run_on_rank(self.library, self.ranks[rank], body)
    }

    /// Receive buffer `slot` of rank `rank`.
    pub fn recv(&self, rank: usize, slot: usize) -> Result<*mut c_void> {
        Ok(self.ends[rank]
            .recv
            .get(slot)
            .with_context(|| format!("exchange slot {slot}"))?
            .buffer
            .ptr)
    }

    /// Bytes of each receive buffer.
    pub fn slot_bytes(&self) -> usize {
        self.ends[0].recv[0].buffer.bytes
    }

    /// Control index of `slot` ([`DIRECT`]: the one after the slots).
    fn flag_index(&self, slot: usize) -> Result<usize> {
        let slots = self.ends[0].recv.len();
        if slot == DIRECT {
            return Ok(slots);
        }
        ensure!(slot < slots, "exchange slot {slot} of {slots}");
        Ok(slot)
    }

    /// Queues on `from`'s stream: push `bytes` of `source` (rows final on that
    /// stream) into the other GPU's receive slot `slot` and publish that slot's
    /// next sequence.
    pub fn push(
        &self,
        from: usize,
        slot: usize,
        source: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let capacity = self.ends.get(1 - from).and_then(|end| end.recv.get(slot))
            .context("peer receive slot")?.buffer.bytes;
        ensure!(bytes <= capacity, "push of {bytes} bytes past the {capacity}-byte exchange slot");
        self.push_to(from, slot, source, self.recv(1 - from, slot)?, bytes)
    }

    /// [`Self::push`] into any `destination` on the other GPU, signalling flag
    /// `slot` (usually [`DIRECT`]): the destination's reader waits on that flag.
    pub fn push_to(
        &self,
        from: usize,
        slot: usize,
        source: *const c_void,
        destination: *mut c_void,
        bytes: usize,
    ) -> Result<()> {
        self.require_live()?;
        let flag = self.flag_index(slot)?;
        tracing::trace!(from, flag, bytes, "peer push");
        let (mine, theirs) = (&self.ends[from], &self.ends[1 - from]);
        let diagnostic = self.diagnostic(from, flag, "push");
        // SAFETY: peer access is enabled both ways (new); the source rows are final on
        // this stream and untouched until later work on it; the destination is a live
        // buffer of the other GPU that it reads only after its matching wait.
        let result = self.on(from, || unsafe {
            let bytes = if crate::shared::memory::device::sm_peer_copy_for(true, bytes, self.decode_lane_bytes) {
                bytes
            } else {
                let buffer = |ptr, device_id| cuteafd_ffi::CuteafdDeviceBuffer { ptr, bytes, device_id, flags: 0 };
                self.library.copy_d2d_async(buffer(destination, self.device(1 - from)),
                    buffer(source.cast_mut(), self.device(from)), bytes, self.stream(from))?;
                // Publication follows the copy on the same issuing stream.
                0
            };
            if let Some(d) = &diagnostic {
                return self.library.peer_push_signal_diagnostic(destination, source, bytes,
                    theirs.flag(flag), mine.send_state(flag), 0, self.stream(from), d);
            }
            self.library.peer_push_signal(
                destination,
                source,
                bytes,
                theirs.flag(flag),
                mine.send_state(flag),
                0,
                self.ranks[from].stream,
            )
        });
        self.diagnostic_return(diagnostic, &result);
        result
    }

    /// Push strided planes on the source rank's stream, then publish one
    /// sequence. As with `push_to`, the caller orders destination reuse after
    /// its previous consumer, not merely after the previous publication.
    pub fn push_planes(&self, from: usize, slot: usize, planes: &[cuteafd_ffi::peer_exchange::PeerPlane]) -> Result<()> {
        self.require_live()?;
        ensure!(from < 2, "peer plane source rank {from}");
        let flag = self.flag_index(slot)?;
        let (mine, theirs) = (&self.ends[from], &self.ends[1 - from]);
        let diagnostic = self.diagnostic(from, flag, "push_planes");
        // SAFETY: callers keep every plane live and immutable through the
        // source stream's copy; the destination reads only after its wait.
        let result = self.on(from, || unsafe {
            if let Some(d) = &diagnostic {
                return self.library.peer_push_planes_diagnostic(planes,
                    theirs.flag(flag), mine.send_state(flag), 0, self.stream(from), d);
            }
            self.library.peer_push_planes(planes,
                Some((theirs.flag(flag), mine.send_state(flag))), 0, self.stream(from))
        });
        self.diagnostic_return(diagnostic, &result);
        result
    }

    /// Publish completion of earlier work without copying payload bytes. This
    /// acknowledgement orders the peer's next write after this rank's reader.
    pub fn signal(&self, from: usize, slot: usize) -> Result<()> {
        ensure!(from < 2, "peer signal source rank {from}");
        self.push_to(from, slot, self.ends[from].control.buffer.ptr.cast_const(),
            self.ends[1 - from].control.buffer.ptr, 0)
    }

    /// Queues on `rank`'s stream: `out = bf16(a + b)` over `count` elements (the same
    /// bits whichever order the two partials come in).
    pub fn add(
        &self,
        rank: usize,
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        count: usize,
    ) -> Result<()> {
        // SAFETY: callers pass live, disjoint [count] BF16 buffers of rank `rank`'s GPU whose
        // producers are ordered before on its stream.
        self.on(rank, || unsafe {
            self.library
                .peer_add_bf16(a, b, out, count, self.ranks[rank].stream)
        })
    }

    /// Queues on `at`'s stream: wait for the other GPU's next push on flag `slot`.
    pub fn wait(&self, at: usize, slot: usize) -> Result<()> {
        self.require_live()?;
        let flag = self.flag_index(slot)?;
        tracing::trace!(at, flag, "peer wait");
        let end = &self.ends[at];
        let diagnostic = self.diagnostic(at, flag, "wait");
        // SAFETY: the flag and state are this GPU's control words; every wait is matched by a
        // push the host queues before it waits on either stream.
        let result = self.on(at, || unsafe {
            if let Some(d) = &diagnostic {
                let aborted = self.abort.as_ref().map_or(std::ptr::null(),
                    |a| a.sides[at].word.buffer.ptr.cast_const().cast());
                return self.library.peer_wait_diagnostic(end.flag(flag), end.recv_state(flag),
                    aborted, self.stream(at), d);
            }
            match &self.abort {
                Some(abort) => self.library.peer_wait_abortable(
                    end.flag(flag),
                    end.recv_state(flag),
                    abort.sides[at].word.buffer.ptr.cast(),
                    self.ranks[at].stream,
                ),
                None => self.library.peer_wait(
                    end.flag(flag),
                    end.recv_state(flag),
                    self.ranks[at].stream,
                ),
            }
        });
        self.diagnostic_return(diagnostic, &result);
        result
    }
}

/// How a head split slices a 2-D weight: by output rows or by input columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    Rows,
    Cols,
}

/// `part` of `ranks` equal slices of a row-major `[rows, cols]` tensor of
/// `elem`-byte elements along `axis`, as contiguous bytes.
pub(crate) fn slice_2d(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    elem: usize,
    axis: Axis,
    part: usize,
    ranks: usize,
) -> Vec<u8> {
    match axis {
        Axis::Rows => {
            let n = rows / ranks;
            bytes[part * n * cols * elem..(part + 1) * n * cols * elem].to_vec()
        }
        Axis::Cols => {
            let n = cols / ranks;
            let mut out = Vec::with_capacity(rows * n * elem);
            for row in bytes.chunks_exact(cols * elem) {
                out.extend_from_slice(&row[part * n * elem..(part + 1) * n * elem]);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn rank_callback_restores_remote_ambient_device_on_error_and_unwind() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        library.cuda_set_device(1)?;
        let target = RankDevice { device: 0, stream: std::ptr::null_mut() };
        let remote = RankDevice { device: 1, stream: std::ptr::null_mut() };
        assert_eq!(on_device(&library, target.device, || library.cuda_get_device())?, 0);
        assert_eq!(library.cuda_get_device()?, 1);
        assert_eq!(on_device(&library, remote.device, || library.cuda_get_device())?, 1);
        assert_eq!(library.cuda_get_device()?, 1);
        let result: Result<()> = run_on_rank(&library, target, || {
            assert_eq!(library.cuda_get_device()?, 0);
            assert_eq!(run_on_rank(&library, remote, || library.cuda_get_device())?, 1);
            assert_eq!(library.cuda_get_device()?, 0);
            anyhow::bail!("injected callback failure")
        });
        assert!(result.is_err());
        assert_eq!(library.cuda_get_device()?, 1);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<()> = run_on_rank(&library, target, || panic!("injected callback unwind"));
        }));
        assert!(panic.is_err());
        assert_eq!(library.cuda_get_device()?, 1);
        Ok(())
    }

    #[test]
    fn decode_extent_validation_and_copy_boundary() {
        for invalid in [0, 1, 17, usize::MAX] { assert!(validate_decode_extent(invalid).is_err()); }
        let decode = 64 * 6144 * 2;
        assert!(validate_decode_extent(decode).is_ok());
        for bytes in [16, decode - 16, decode] {
            assert!(crate::shared::memory::device::sm_peer_copy_for(true, bytes, decode));
        }
        assert!(!crate::shared::memory::device::sm_peer_copy_for(true, decode + 16, decode));
    }

    #[test]
    fn invalid_slot_sizes_name_the_slot_and_extent() {
        for bytes in [0, 17] {
            let error = validate_slot_sizes([0, 1], &[256, bytes]).unwrap_err().to_string();
            assert!(error.contains("slot 1") && error.contains(&format!("{bytes} bytes")), "{error}");
        }
        assert!(validate_slot_sizes([0, 0], &[256]).unwrap_err().to_string().contains("devices 0 and 0"));
        assert!(validate_slot_sizes([0, 1], &[]).is_err());
        assert!(validate_slot_sizes([0, 1], &[16, 256, 1024]).is_ok());
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn heterogeneous_slots_keep_decode_and_prefill_extents_separate() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let ranks = [RankDevice { device: 0, stream: std::ptr::null_mut() },
            RankDevice { device: 1, stream: std::ptr::null_mut() }];
        let sizes = [256, 256, 1024, 1024, 1024];
        let exchange = PeerExchange::new_sized(&library, ranks, &sizes, 256)?;
        for rank in 0..2 {
            for (slot, &bytes) in sizes.iter().enumerate() {
                assert_eq!(exchange.ends[rank].recv[slot].buffer.bytes, bytes);
                for other in 0..sizes.len() {
                    assert_eq!(exchange.recv(rank, slot)? == exchange.recv(rank, other)?, slot == other);
                }
            }
        }
        let source = DeviceAllocation::new(&library, 1024)?;
        assert!(exchange.push(0, 0, source.buffer.ptr, 272).is_err());
        assert!(exchange.push(0, sizes.len(), source.buffer.ptr, 16).is_err());
        assert!(PeerExchange::new_sized(&library, ranks, &[0], 256).is_err());
        assert!(PeerExchange::new_sized(&library, ranks, &[17], 256).is_err());
        Ok(())
    }

    #[test]
    fn sender_stream_wide_push_uses_p2p_d2d_entry_point() {
        let source = include_str!("peer_split.rs");
        let push = source.split("    pub fn push_to(").nth(1).unwrap()
            .split("    /// Push strided planes").next().unwrap();
        assert!(push.contains("self.on(from, || unsafe"));
        assert!(push.contains("self.library.copy_d2d_async(buffer(destination, self.device(1 - from)),"));
        assert!(!push.contains("copy_peer_async"));
        assert!(push.contains("buffer(source.cast_mut(), self.device(from)), bytes, self.stream(from))?"));
        assert!(push.find("copy_d2d_async").unwrap() < push.find("peer_push_signal(").unwrap());
    }

    #[test]
    fn peer_scopes_restore_the_caller_instead_of_a_fixed_home() {
        let source = include_str!("peer_split.rs");
        let helper = source.split("pub(crate) fn on_device<T>").nth(1).unwrap()
            .split("/// One GPU's end").next().unwrap();
        assert!(helper.contains("Device { library, id: device }.run(body)"));
        assert!(!helper.contains("home"));
        let mimo = include_str!("../families/mimo_v2/engine.rs");
        let rank_scope = mimo.split("pub(crate) fn on<T>").nth(1).unwrap()
            .split("pub(crate) fn is_terminal").next().unwrap();
        assert!(rank_scope.contains("Device { library: self.library, id: device }.run(body)"));
        assert!(!rank_scope.contains("cuda_set_device"));
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs"]
    fn peer_scopes_preserve_terminal_device_for_blocking_stream_waits() -> Result<()> {
        use crate::shared::memory::{device::{Device, Event, Stream}, BlockingWait};
        use cuteafd_transport::wait_policy::{set_wait_policy, HostWait, WaitPolicy};
        set_wait_policy(WaitPolicy { wait: HostWait::Block, ..Default::default() })?;
        // SAFETY: all streams and events are destroyed before the loaded library.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
        let streams = devices.map(Stream::new).into_iter().collect::<Result<Vec<_>>>()?;
        let mut landed = BlockingWait::new(&library);
        for terminal in [1usize, 0] {
            library.cuda_set_device(terminal as i32)?;
            let mut produced = Event::new(devices[terminal])?;
            for rank in [0usize, 1, 0] {
                on_device(&library, rank as i32, || {
                    assert_eq!(library.cuda_get_device()?, rank as i32);
                    Ok(())
                })?;
                assert_eq!(library.cuda_get_device()?, terminal as i32);
                produced.record(&streams[terminal])?;
                // SAFETY: this terminal-device stream remains owned through the wait.
                unsafe { landed.wait(streams[terminal].raw)?; }
            }
            let failure: Result<()> = on_device(&library, 1 - terminal as i32,
                || anyhow::bail!("injected peer failure"));
            assert!(failure.is_err());
            assert_eq!(library.cuda_get_device()?, terminal as i32);
            // SAFETY: failed peer work restored the same live terminal-device stream.
            unsafe { landed.wait(streams[terminal].raw)?; }
        }
        Ok(())
    }

    #[test]
    fn slices_cover_the_tensor() {
        let bytes: Vec<u8> = (0..24).collect();
        assert_eq!(
            slice_2d(&bytes, 4, 6, 1, Axis::Rows, 1, 2),
            (12..24).collect::<Vec<u8>>()
        );
        assert_eq!(
            slice_2d(&bytes, 4, 6, 1, Axis::Cols, 0, 2),
            vec![0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]
        );
        assert_eq!(
            slice_2d(&bytes, 2, 6, 2, Axis::Cols, 1, 3),
            vec![4, 5, 6, 7, 16, 17, 18, 19]
        );
    }
}
