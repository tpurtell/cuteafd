//! One device-driven bridge for a drafter that lives on the other GPU from
//! its target (#65, #106). Families bind it to their drafter: the target rank
//! (0) keeps its single head, embedding and tap producer; the draft rank (1)
//! owns the drafter, a dedicated stream and an ordering event. Context, input
//! rows, hidden rows and logits move GPU to GPU over one abortable
//! [`PeerExchange`]; the host only enqueues.
//!
//! **Ordering invariant (#94).** A resident peer wait on the draft GPU must not
//! steal the SMs a full-grid cooperative target launch on that same GPU needs.
//! Every draft-side context wait is queued behind the peer target producer's
//! completion event ([`RemoteBridge::publish_context`]), never a host sync.
use crate::shared::memory::device::{Device, Event, Stream};
use crate::shared::peer_split::{PeerExchange, RankDevice, TerminalState};
use anyhow::{ensure, Result};
use cuteafd_ffi::peer_exchange::PeerPlane;
use cuteafd_ffi::NativeLibrary;
use std::cell::Cell;

/// Context rows (taps), draft input rows, draft hidden rows, target logits.
pub(crate) const CONTEXT: usize = 0;
pub(crate) const INPUT: usize = 1;
pub(crate) const HIDDEN: usize = 2;
pub(crate) const LOGITS: usize = 3;
const SLOTS: usize = cuteafd_loader::placement::draft::BRIDGE_SLOTS as usize;

/// The bridge's sampled controls on each GPU: four zero-payload receive slots
/// (256 B each), the control block and the abort word. The solver charges the
/// same bytes as the bridge's exchange demand on both GPUs.
pub(crate) const EXCHANGE_BYTES: u64 = cuteafd_loader::placement::draft::BRIDGE_EXCHANGE_BYTES;

/// Created before admission so its controls are in the measured sample; it is
/// dropped unused when the solver keeps the drafter beside the target.
pub(crate) struct Prepared<'a> {
    pub exchange: PeerExchange<'a>,
    pub stream: Stream<'a>,
    target_peer: RankDevice,
    target_ready: Event<'a>,
}

impl<'a> Prepared<'a> {
    pub fn new(library: &'a NativeLibrary, target: RankDevice, target_peer: RankDevice,
        identity: &'static str) -> Result<Self> {
        ensure!(target.device != target_peer.device && !target_peer.stream.is_null(),
            "remote draft requires the peer target producer stream");
        let device = Device { library, id: target_peer.device };
        let stream = Stream::new(device)?;
        let target_ready = Event { device, raw: device.run(|| library.cuda_event_create_ordering())? };
        let exchange = PeerExchange::new_abortable(library,
            [target, RankDevice { device: target_peer.device, stream: stream.raw }], SLOTS, 0)?;
        exchange.trace_identity(identity);
        Ok(Self { exchange, stream, target_peer, target_ready })
    }

    /// Sampled before admission but charged only for the selected remote candidate.
    pub fn sampled_bytes(&self) -> u64 { EXCHANGE_BYTES }
}

pub(crate) struct RemoteBridge<'a> {
    pub exchange: PeerExchange<'a>,
    target_peer: RankDevice,
    target_ready: Event<'a>,
    // The exchange and event drop before the stream they borrow.
    pub stream: Stream<'a>,
    update_pending: Cell<bool>,
}

impl<'a> RemoteBridge<'a> {
    pub fn new(prepared: Prepared<'a>) -> Self {
        Self { exchange: prepared.exchange, target_peer: prepared.target_peer,
            target_ready: prepared.target_ready, stream: prepared.stream, update_pending: Cell::new(false) }
    }

    pub fn library(&self) -> &'a NativeLibrary { self.stream.device.library }
    pub fn target_device(&self) -> i32 { self.exchange.device(0) }

    pub fn require_live(&self) -> Result<()> { self.exchange.require_live() }

    pub fn publish_abort(&self) -> Result<()> {
        match self.exchange.terminal_state() {
            TerminalState::Aborting | TerminalState::Drained => Ok(()),
            _ => self.exchange.publish_abort(),
        }
    }

    /// No owner is released on an unproved drain; the engine's fatal-drain
    /// boundary also retains every target graph and peer-exchange operand.
    pub fn synchronize(&self) -> Result<()> {
        self.finish_request()?;
        self.stream.drain()?;
        if self.exchange.terminal_state() == TerminalState::Aborting { self.update_pending.set(false); }
        Ok(())
    }

    /// Consume the final acknowledgement before any request slot is reused.
    pub fn finish_request(&self) -> Result<()> {
        if self.update_pending.get() && self.exchange.terminal_state() == TerminalState::Active {
            self.exchange.wait(0, CONTEXT)?;
            self.exchange.on(0, || {
                // SAFETY: request owners remain live until this acknowledgement retires.
                unsafe { self.library().cuda_stream_synchronize(self.exchange.stream(0)) }
            })?;
            self.update_pending.set(false);
        }
        Ok(())
    }

    pub fn finish_terminal(&self, drained: bool) {
        if self.exchange.terminal_state() == TerminalState::Aborting {
            self.exchange.finish_terminal(drained);
        }
    }

    /// Run `work`; any failure publishes the abort so neither side waits forever.
    pub fn checked<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        self.require_live()?;
        let result = work();
        if result.is_err() {
            crate::shared::decode_graph::fatal_drain(self.publish_abort(), "remote draft bridge abort publication");
        }
        result
    }

    pub(crate) fn validate_target_peer(&self, producer: RankDevice) -> Result<()> {
        ensure!(self.target_peer.device == producer.device && self.target_peer.stream == producer.stream
            && self.target_ready.device.id == producer.device
            && self.stream.device.id == producer.device && self.exchange.stream(1) == self.stream.raw,
            "resident remote peer waits must be ordered behind the cooperative target producer");
        Ok(())
    }

    /// Publish final context rows to the drafter and wait for them on the
    /// draft stream, ordered behind the peer target producer. Call inside the
    /// drafter's own context update, after its metadata uploads; the
    /// drafter's readers then run, and [`Self::acknowledge_context`] follows.
    pub fn publish_context(&self, planes: &[PeerPlane]) -> Result<()> {
        if self.update_pending.get() { self.exchange.wait(0, CONTEXT)?; }
        self.exchange.push_planes(0, CONTEXT, planes)?;
        // Both target ranks have submitted this prefill chunk or decode
        // round. A resident waiter must not steal the SM needed by the
        // peer target's full-grid cooperative EXL3 launch. Wait on its
        // completion event before starting the context spinner, with no host
        // sync. GPU0 publishes context N before its chunk-N+1 waits, so this
        // spinner retires before GPU1 needs its next cooperative grid.
        self.validate_target_peer(self.target_peer)?;
        self.stream.device.run(|| unsafe {
            // SAFETY: both same-device streams and the reusable ordering
            // event survive every queued update; stream order protects reuse.
            self.library().cuda_event_record(self.target_ready.raw, self.target_peer.stream)?;
            self.library().cuda_stream_wait_event(self.stream.raw, self.target_ready.raw)
        })?;
        self.exchange.wait(1, CONTEXT)
    }

    /// Acknowledgement means all context readers are finished, not just that
    /// the incoming rows arrived. The next overwrite waits for it.
    pub fn acknowledge_context(&self) -> Result<()> {
        self.exchange.signal(1, CONTEXT)?;
        self.update_pending.set(true);
        Ok(())
    }

    /// Target-side rows into the drafter's input on the draft stream.
    pub fn send_input(&self, source: *const std::ffi::c_void, destination: *mut std::ffi::c_void, bytes: usize) -> Result<()> {
        self.exchange.push_to(0, INPUT, source, destination, bytes)?;
        self.exchange.wait(1, INPUT)
    }

    /// Drafter hidden rows to the target head, then the target's logits back.
    pub fn round_trip_head(&self, hidden: *const std::ffi::c_void, target_hidden: *mut std::ffi::c_void,
        hidden_bytes: usize, head: impl FnOnce() -> Result<()>, target_logits: *const std::ffi::c_void,
        logits: *mut std::ffi::c_void, logits_bytes: usize) -> Result<()> {
        self.exchange.push_to(1, HIDDEN, hidden, target_hidden, hidden_bytes)?;
        self.exchange.wait(0, HIDDEN)?;
        head()?;
        self.exchange.push_to(0, LOGITS, target_logits, logits, logits_bytes)?;
        self.exchange.wait(1, LOGITS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::order::{self, Schedule};

    fn update(schedule: &mut Schedule, pending: bool) {
        if pending { schedule.wait(0, "draft", CONTEXT, "previous context readers"); }
        schedule.push(0, "draft", CONTEXT, "target context final");
        schedule.wait(1, "draft", CONTEXT, "context arrived");
        schedule.push(1, "draft", CONTEXT, "context readers finished");
    }

    fn draft(schedule: &mut Schedule) {
        schedule.push(0, "draft", INPUT, "target input gathered");
        schedule.wait(1, "draft", INPUT, "draft input arrived");
        schedule.push(1, "draft", HIDDEN, "draft hidden normalized");
        schedule.wait(0, "draft", HIDDEN, "target head input arrived");
        schedule.push(0, "draft", LOGITS, "target head finished");
        schedule.wait(1, "draft", LOGITS, "draft selector logits arrived");
    }

    #[test]
    fn repeated_context_and_head_bridges_have_no_wait_cycle() {
        for rounds in [1, 2, 8, 32] {
            let mut schedule = Schedule::default();
            for round in 0..rounds {
                update(&mut schedule, round > 0);
                // Target-only rounds still update the drafter's context.
                if round % 3 != 1 { draft(&mut schedule); }
            }
            schedule.wait(0, "draft", CONTEXT, "request boundary context readers");
            schedule.end_request();
            assert_eq!(order::check(&schedule), Ok(()));
        }
    }

    #[test]
    fn head_wait_before_input_publication_is_detected() {
        let mut schedule = Schedule::default();
        schedule.wait(0, "draft", HIDDEN, "head before input");
        draft(&mut schedule);
        assert!(order::check(&schedule).is_err());
    }
}
