//! Routed-only family adapter over the shared Spark, whole-layer and TP2 owners.
use crate::shared::{experts::{rtx::{ExpertInput, PartialDtype, PartialLayout, Routes,
    RtxExpertLayer, RtxShard, owner_tp2::OwnerTp2}, staging::RouteStaging},
    memory::{device::{Allocation, Device, Event, Stream}, receipts::Inventory},
    spark_intake::SparkLink, spark_waves::{self, WaveTicket}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind, expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use std::{cell::{Cell, RefCell}, collections::VecDeque, ffi::c_void, rc::Rc};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum DispatchError {
    #[error("MiniMax routed output is already leased")]
    Busy,
    #[error("MiniMax routed source copy has not been staged")]
    NotStaged,
    #[error("MiniMax routed output has not landed")]
    NotLanded,
    #[error("MiniMax routed owner was terminally cancelled")]
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase { Idle, CopyPending, Staged, Landed, Terminal }

/// Events are allocated once with the unit, not on the request path. All
/// handles alias existing physical owners; only UnitOwner inventories them.
pub(crate) struct Boundary<'a> {
    phase: Cell<Phase>,
    producer: Event<'a>,
    staged: Event<'a>,
    ready: Event<'a>,
}
impl<'a> Boundary<'a> {
    pub(crate) fn new(device: Device<'a>) -> Result<Self> {
        Ok(Self { phase: Cell::new(Phase::Idle), producer: Event::new(device)?,
            staged: Event::new(device)?, ready: Event::new(device)? })
    }
    pub(crate) fn idle(&self) -> Result<()> {
        match self.phase.get() {
            Phase::Idle => Ok(()),
            Phase::Terminal => Err(DispatchError::Terminal.into()),
            _ => Err(DispatchError::Busy.into()),
        }
    }
    fn acquire(&self) -> Result<()> {
        self.idle()?;
        self.phase.set(Phase::CopyPending);
        Ok(())
    }
    fn record(event: &Event<'_>, stream: *mut c_void) -> Result<()> {
        // SAFETY: event and stream belong to the retained unit/exchange device.
        event.device.run(|| unsafe { event.device.library.cuda_event_record(event.raw, stream) })
    }
    fn wait(event: &Event<'_>, stream: *mut c_void) -> Result<()> {
        // SAFETY: a record precedes every wait; retained events survive drainage.
        event.device.run(|| unsafe { event.device.library.cuda_stream_wait_event(stream, event.raw) })
    }
    pub(crate) fn before_ffn(&self, stream: *mut c_void) -> Result<()> {
        match self.phase.get() {
            Phase::Idle => Ok(()), // Cold golden still uses its direct ordered dispatch.
            Phase::Staged | Phase::Landed => Self::wait(&self.staged, stream),
            Phase::CopyPending => Err(DispatchError::NotStaged.into()),
            Phase::Terminal => Err(DispatchError::Terminal.into()),
        }
    }
    pub(crate) fn before_join(&self, stream: *mut c_void) -> Result<()> {
        match self.phase.get() {
            Phase::Idle => Ok(()),
            Phase::Landed => Self::wait(&self.ready, stream),
            Phase::Terminal => Err(DispatchError::Terminal.into()),
            _ => Err(DispatchError::NotLanded.into()),
        }
    }
    pub(crate) fn joined(&self) {
        if self.phase.get() == Phase::Landed { self.phase.set(Phase::Idle); }
    }
}

/// Retained aliases permit shared FFN mutation without borrowing UnitOwner.
/// The ticket borrows this handle until land; the boundary blocks buffer reuse.
#[derive(Clone)]
pub(crate) struct RoutedRows<'a> {
    activation: Rc<Allocation<'a>>,
    ids: Rc<Allocation<'a>>,
    weights: Rc<Allocation<'a>>,
    output: Rc<Allocation<'a>>,
    stream: Rc<Stream<'a>>,
    boundary: Rc<Boundary<'a>>,
    rows: usize,
}
impl<'a> RoutedRows<'a> {
    pub(crate) fn new(activation: Rc<Allocation<'a>>, ids: Rc<Allocation<'a>>,
        weights: Rc<Allocation<'a>>, output: Rc<Allocation<'a>>, stream: Rc<Stream<'a>>,
        boundary: Rc<Boundary<'a>>, rows: usize) -> Result<Self> {
        ensure!(rows > 0, "MiniMax routed rows are empty");
        for (owner, bytes) in [(&activation, rows.checked_mul(6144 * 2)),
            (&ids, rows.checked_mul(4 * 4)), (&weights, rows.checked_mul(4 * 4)),
            (&output, rows.checked_mul(6144 * 2))] {
            ensure!(bytes.is_some_and(|bytes| bytes <= owner.buffer.bytes)
                && owner.device.id == stream.device.id
                && std::ptr::eq(owner.device.library, stream.device.library),
                "MiniMax routed owner geometry/device differs");
        }
        let pointers = [activation.buffer.ptr, ids.buffer.ptr, weights.buffer.ptr, output.buffer.ptr];
        ensure!((0..4).all(|i| (i+1..4).all(|j| pointers[i] != pointers[j])),
            "MiniMax routed inputs/output overlap");
        Ok(Self { activation, ids, weights, output, stream, boundary, rows })
    }
    fn routes(&self) -> Routes { Routes { ids: self.ids.buffer.ptr, weights: self.weights.buffer.ptr } }
    fn sources(&self, layout: RouteStaging) -> [CuteafdDeviceBuffer; 3] {
        [CuteafdDeviceBuffer { bytes: layout.route_bytes, ..self.ids.buffer },
            CuteafdDeviceBuffer { bytes: layout.route_bytes, ..self.weights.buffer },
            CuteafdDeviceBuffer { bytes: layout.wire_bytes, ..self.activation.buffer }]
    }
}

/// Row identities come from the shared packer or decode request table, never a
/// synthesized single-request identity or a contiguous-position assumption.
pub(crate) struct DispatchRows {
    pub request_id: u64,
    pub placement_version: u64,
    pub rows: Vec<ExpertProtocolV2RowDescriptor>,
}
impl DispatchRows {
    fn validate(&self, rows: usize) -> Result<()> {
        ensure!(self.request_id != 0 && self.rows.len() == rows, "MiniMax dispatch identity/rows differ");
        let mut identities = std::collections::BTreeSet::new();
        for (index, row) in self.rows.iter().enumerate() {
            ensure!(row.source_request_id != 0 && identities.insert(row.row_id)
                && row.route_count == 4 && row.route_offset as usize == index * 4,
                "MiniMax dispatch row identity/route span differs");
        }
        Ok(())
    }
    fn combined_reply(&self) -> bool {
        spark_waves::combined_reply_eligible(self.rows.iter().map(|row|
            (row.source_request_id, row.source_kind == ExpertV2SourceKind::Prefill)))
    }
    fn request(self, layer: usize, routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>,
        wire: bytes::Bytes) -> Result<ExpertProtocolV2Request> {
        let mut request = ExpertProtocolV2Request::new_bytes(self.request_id, self.placement_version,
            u32::try_from(layer)?, 6144, ExpertV2Dtype::Bf16, self.rows, routes, wire)?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        Ok(request)
    }
}

enum Backend<'a> {
    Closed,
    Spark(SparkLink<'a>),
    Whole(Box<dyn RtxExpertLayer + 'a>),
    Tp2 { owner: usize, executor: OwnerTp2<'a> },
}
struct Pending<'a> {
    id: u64,
    rows: RoutedRows<'a>,
    wave: Option<WaveTicket>,
    land: bool,
}
struct Inner<'a> {
    backend: Backend<'a>,
    terminal: bool,
    next: u64,
    pending: VecDeque<Pending<'a>>,
    last: Option<RoutedRows<'a>>,
    owners: std::collections::BTreeMap<usize, RoutedRows<'a>>,
}
/// Owns only the already-admitted shared executor, never a family layer loop.
pub(crate) struct Dispatcher<'a> { inner: Rc<RefCell<Inner<'a>>> }
impl<'a> Dispatcher<'a> {
    fn new(backend: Backend<'a>) -> Self {
        Self { inner: Rc::new(RefCell::new(Inner { backend, terminal: false, next: 1,
            pending: VecDeque::new(), last: None, owners: std::collections::BTreeMap::new() })) }
    }
    /// The caller supplies its prepared, bootstrapped, terminal-owned SparkLink.
    pub(crate) fn spark(mut link: SparkLink<'a>) -> Result<Self> {
        link.exchange()?;
        Ok(Self::new(Backend::Spark(link)))
    }
    pub(crate) fn whole(experts: Box<dyn RtxExpertLayer + 'a>) -> Result<Self> {
        ensure!(experts.shard() == RtxShard::Whole && experts.partial() == PartialDtype::Bf16
            && !experts.uses_fp8_input(), "MiniMax whole experts require native BF16 routed sums");
        Ok(Self::new(Backend::Whole(experts)))
    }
    pub(crate) fn tp2(owner: usize, executor: OwnerTp2<'a>) -> Result<Self> {
        ensure!(owner < 2 && !executor.wire && executor.experts.iter().all(|expert|
            expert.partial() == PartialDtype::Bf16), "MiniMax TP2 requires native BF16 routed partials");
        Ok(Self::new(Backend::Tp2 { owner, executor }))
    }
    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        match &self.inner.borrow().backend {
            Backend::Closed => Err(DispatchError::Terminal.into()),
            Backend::Spark(link) => link.allocation_receipts(inventory, prefix),
            Backend::Whole(experts) => experts.allocation_receipts(inventory, prefix),
            Backend::Tp2 { executor, .. } => executor.allocation_receipts(inventory, prefix),
        }
    }
    pub(crate) fn begin<'t>(&self, layer: usize, rows: &'t RoutedRows<'a>,
        identities: DispatchRows) -> Result<Ticket<'t, 'a>> {
        identities.validate(rows.rows)?;
        let id = {
            let mut inner = self.inner.borrow_mut();
            ensure!(!inner.terminal, DispatchError::Terminal);
            ensure!(matches!(inner.backend, Backend::Spark(_)) || inner.pending.is_empty(), DispatchError::Busy);
            if let Some(previous) = &inner.last { previous.boundary.idle()?; }
            let device = rows.stream.device;
            if let Some(previous) = inner.owners.values().next() {
                ensure!(previous.stream.device.id == device.id
                    && std::ptr::eq(previous.stream.device.library, device.library),
                    "MiniMax dispatcher units must share one device/library");
            }
            let id = inner.next;
            let next = inner.next.checked_add(1).context("MiniMax dispatch identity exhausted")?;
            rows.boundary.acquire()?;
            inner.next = next;
            // Retain every participating unit through terminal stream drainage,
            // including units whose land already returned before cancellation.
            inner.owners.insert(Rc::as_ptr(&rows.boundary) as usize, rows.clone());
            inner.pending.push_back(Pending { id, rows: rows.clone(), wave: None, land: false });
            id
        };
        let mut ticket = Ticket { inner: self.inner.clone(), rows, id, complete: false };
        ticket.enqueue(layer, identities)?;
        Ok(ticket)
    }
}
impl Inner<'_> {
    fn cancel(&mut self) {
        self.terminal = true;
        for rows in self.owners.values() { rows.boundary.phase.set(Phase::Terminal); }
        let result = (|| {
            match &mut self.backend {
                Backend::Closed => (),
                Backend::Spark(link) => link.terminal_quiesce()?,
                Backend::Tp2 { executor, .. } => {
                    use crate::shared::peer_split::TerminalState;
                    if executor.exchange.terminal_state() == TerminalState::Active {
                        executor.exchange.publish_abort()?;
                    }
                    executor.exchange.drain_compute()?;
                }
                Backend::Whole(_) => (),
            }
            for rows in self.owners.values() { rows.stream.drain()?; }
            match &mut self.backend {
                Backend::Closed => (),
                Backend::Spark(link) => { link.terminal_drain_copies()?; link.terminal_release()?; }
                Backend::Tp2 { executor, .. } => executor.exchange.finish_terminal(true),
                Backend::Whole(_) => (),
            }
            Ok(())
        })();
        crate::shared::decode_graph::fatal_drain(result, "MiniMax routed ticket cancellation");
        self.pending.clear();
    }
}
impl Drop for Inner<'_> {
    fn drop(&mut self) {
        // Terminal teardown always precedes alias/stream release, even if every
        // ticket landed but some consumer work is still queued.
        if !self.terminal { self.cancel(); }
    }
}

#[must_use = "land the routed ticket before joining the shared branch"]
pub(crate) struct Ticket<'t, 'a> {
    inner: Rc<RefCell<Inner<'a>>>,
    rows: &'t RoutedRows<'a>,
    id: u64,
    complete: bool,
}
impl Ticket<'_, '_> {
    fn enqueue(&mut self, layer: usize, identities: DispatchRows) -> Result<()> {
        let rows = self.rows;
        let device = rows.stream.device;
        let mut inner = self.inner.borrow_mut();
        if let Some(previous) = &inner.last {
            Boundary::record(&previous.boundary.producer, previous.stream.raw)?;
            Boundary::wait(&previous.boundary.producer, rows.stream.raw)?;
        }
        Boundary::record(&rows.boundary.producer, rows.stream.raw)?;
        let producer = rows.boundary.producer.raw;
        let mut wave = None;
        let stream = match &mut inner.backend {
            Backend::Closed => return Err(DispatchError::Terminal.into()),
            Backend::Spark(link) => {
                ensure!(link.device() == device.id, "MiniMax Spark owner device differs");
                let exchange = link.exchange()?;
                let layout = RouteStaging::new(rows.rows, 4, 6144 * 2)?;
                let combined = identities.combined_reply();
                wave = Some(device.run(|| {
                    // SAFETY: the borrowed handle retains all source owners;
                    // staging is before the source-ready event and shared FFN.
                    unsafe { exchange.send(rows.rows, exchange.compact_payload(rows.rows), layout,
                        rows.sources(layout), rows.stream.raw, combined,
                        Box::new(move |routes, wire| identities.request(layer, routes, wire))) }
                })?);
                rows.stream.raw
            }
            Backend::Whole(experts) => {
                ensure!(experts.device() == device.id && experts.partial_layout(rows.rows)? == PartialLayout::Tokens,
                    "MiniMax whole expert device/layout differs");
                device.run(|| {
                    // SAFETY: input/routes and backend output stay live on the
                    // producer stream; no shared addition is performed here.
                    unsafe { experts.enqueue(layer, rows.rows, ExpertInput::Bf16(rows.activation.buffer.ptr),
                        rows.routes(), rows.stream.raw, &mut || ())?; }
                    let source = CuteafdDeviceBuffer { ptr: experts.output(), bytes: rows.rows * 6144 * 2,
                        device_id: device.id, ..Default::default() };
                    // SAFETY: native BF16 routed sum and distinct unit output.
                    unsafe { device.library.copy_d2d_async(rows.output.buffer, source, source.bytes, rows.stream.raw) }
                })?;
                rows.stream.raw
            }
            Backend::Tp2 { owner, executor } => {
                ensure!(executor.experts[*owner].device() == device.id, "MiniMax TP2 source owner differs");
                let stream = executor.exchange.stream(*owner);
                device.run(|| {
                    // SAFETY: producer record orders router and normalized input
                    // before the shared executor copies them into its own send plane.
                    unsafe { device.library.cuda_stream_wait_event(stream, producer)?;
                        executor.enqueue_routed(*owner, layer, rows.rows, rows.activation.buffer.ptr,
                            rows.routes(), rows.output.buffer.ptr) }
                })?;
                stream
            }
        };
        inner.pending.iter_mut().find(|pending| pending.id == self.id)
            .context("MiniMax dispatch lost pending owner")?.wave = wave;
        Boundary::record(&rows.boundary.staged, stream)?;
        rows.boundary.phase.set(Phase::Staged);
        Ok(())
    }
    /// An out-of-order request is retained until earlier tickets land. Its
    /// join returns NotLanded in the meantime; no CUDA wait is queued early.
    pub(crate) fn land(mut self) -> Result<()> {
        {
            let mut inner = self.inner.borrow_mut();
            ensure!(!inner.terminal, DispatchError::Terminal);
            inner.pending.iter_mut().find(|pending| pending.id == self.id)
                .context("MiniMax routed ticket no longer pending")?.land = true;
            let device = self.rows.stream.device;
            device.run(|| inner.flush())?;
        }
        self.complete = true;
        Ok(())
    }
}
fn flush_fifo<'a>(pending: &mut VecDeque<Pending<'a>>,
    mut land: impl FnMut(&mut Pending<'a>) -> Result<*mut c_void>) -> Result<Option<RoutedRows<'a>>> {
    let mut last = None;
    while pending.front().is_some_and(|pending| pending.land) {
        // Retain the entry until every fallible native submission succeeds.
        let front = pending.front_mut().expect("checked pending entry");
        let stream = land(front)?;
        Boundary::record(&front.rows.boundary.ready, stream)?;
        front.rows.boundary.phase.set(Phase::Landed);
        last = pending.pop_front().map(|pending| pending.rows);
    }
    Ok(last)
}
impl Inner<'_> {
    fn flush(&mut self) -> Result<()> {
        let backend = &mut self.backend;
        let last = flush_fifo(&mut self.pending, |pending| {
            let rows = &pending.rows;
            let device = rows.stream.device;
            match backend {
                Backend::Closed => Err(DispatchError::Terminal.into()),
                Backend::Spark(link) => {
                    let exchange = link.exchange()?;
                    let wave = pending.wave.take().context("MiniMax Spark ticket lacks its retained wave")?;
                    let slot = wave.slot();
                    device.run(|| {
                        exchange.land(wave, rows.stream.raw)?;
                        // SAFETY: FIFO land orders planes before a routed-only
                        // reduction into the retained, distinct unit output.
                        unsafe { exchange.reduce(slot, std::ptr::null(), rows.output.buffer.ptr.cast(),
                            rows.rows, rows.stream.raw) }
                    })?;
                    Ok(rows.stream.raw)
                }
                Backend::Whole(_) => Ok(rows.stream.raw),
                Backend::Tp2 { owner, executor } => Ok(executor.exchange.stream(*owner)),
            }
        })?;
        if !matches!(self.backend, Backend::Spark(_)) && last.is_some() { self.last = last; }
        Ok(())
    }
}
impl Drop for Ticket<'_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            // Drain outside the RefCell borrow. Other tickets see a closed
            // dispatcher before terminal quiescence can block this thread.
            let mut retained = {
                let mut inner = self.inner.borrow_mut();
                std::mem::replace(&mut *inner, Inner { backend: Backend::Closed, terminal: true,
                    next: 0, pending: VecDeque::new(), last: None, owners: Default::default() })
            };
            retained.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_core::memory_layout::{Category, Tag};
    use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};

    fn rows(device: Device<'_>) -> Result<RoutedRows<'_>> {
        let allocation = |bytes| Allocation::new(device, bytes, Tag::workspace("minimax/test/routed")).map(Rc::new);
        RoutedRows::new(allocation(6144 * 2)?, allocation(16)?, allocation(16)?, allocation(6144 * 2)?,
            Rc::new(Stream::new(device)?), Rc::new(Boundary::new(device)?), 1)
    }
    fn identities(id: u64) -> DispatchRows {
        DispatchRows { request_id: id, placement_version: 23,
            rows: vec![ExpertProtocolV2RowDescriptor { row_id: id + 100,
                source_kind: ExpertV2SourceKind::Decode, source_request_id: id + 200,
                token_position: 937 + id, route_offset: 0, route_count: 4 }] }
    }
    fn is_error(result: Result<()>, expected: DispatchError) {
        assert_eq!(result.unwrap_err().downcast_ref::<DispatchError>(), Some(&expected));
    }
    fn fences(trace: &str) -> String { trace.chars().filter(|ch| "RW".contains(*ch)).collect() }

    #[test]
    fn source_copy_event_is_required_before_shared_ffn_reuses_activation() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let rows = rows(Device { library: &library, id: 0 })?;
        rows.boundary.acquire()?;
        is_error(rows.boundary.before_ffn(rows.stream.raw), DispatchError::NotStaged);
        let before = fixture.events()?.len();
        Boundary::record(&rows.boundary.staged, rows.stream.raw)?;
        rows.boundary.phase.set(Phase::Staged);
        rows.boundary.before_ffn(rows.stream.raw)?;
        assert_eq!(fences(&fixture.events()?[before..]), "RW");
        Ok(())
    }

    #[test]
    fn output_ready_event_is_required_before_residual_join() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let rows = rows(Device { library: &library, id: 0 })?;
        rows.boundary.acquire()?;
        rows.boundary.phase.set(Phase::Staged);
        is_error(rows.boundary.before_join(rows.stream.raw), DispatchError::NotLanded);
        let before = fixture.events()?.len();
        Boundary::record(&rows.boundary.ready, rows.stream.raw)?;
        rows.boundary.phase.set(Phase::Landed);
        rows.boundary.before_join(rows.stream.raw)?;
        assert_eq!(fences(&fixture.events()?[before..]), "RW");
        rows.boundary.joined();
        rows.boundary.idle()?;
        Ok(())
    }

    struct Whole<'a> { output: Allocation<'a>, count: Rc<Cell<usize>> }
    impl RtxExpertLayer for Whole<'_> {
        fn shard(&self) -> RtxShard { RtxShard::Whole }
        fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
        fn uses_fp8_input(&self) -> bool { false }
        fn layers(&self) -> &[usize] { &[2] }
        fn device(&self) -> i32 { self.output.device.id }
        fn workspace_bytes(&self) -> usize { self.output.buffer.bytes }
        fn output(&self) -> *mut c_void { self.output.buffer.ptr }
        unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput,
            _routes: Routes, stream: *mut c_void, _abort: &mut dyn FnMut()) -> Result<()> {
            ensure!(layer == 2 && rows == 1, "test backend geometry differs");
            let ExpertInput::Bf16(input) = input else { anyhow::bail!("test expects BF16"); };
            self.count.set(self.count.get() + 1);
            let source = CuteafdDeviceBuffer { ptr: input, ..self.output.buffer };
            // SAFETY: adapter retains source/output and orders their producer.
            unsafe { self.output.device.library.copy_d2d_async(self.output.buffer, source, source.bytes, stream) }
        }
    }
    fn dispatcher<'a>(device: Device<'a>) -> Result<(Dispatcher<'a>, Rc<Cell<usize>>)> {
        let count = Rc::new(Cell::new(0));
        Ok((Dispatcher::whole(Box::new(Whole { output: Allocation::new(device, 6144 * 2,
            Tag::workspace("minimax/test/whole"))?, count: count.clone() }))?, count))
    }

    #[test]
    fn outstanding_ticket_and_unjoined_output_reuse_are_typed_busy() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let rows = rows(device)?;
        let other = self::rows(device)?;
        let (dispatcher, count) = dispatcher(device)?;
        let ticket = dispatcher.begin(2, &rows, identities(1))?;
        for handle in [&rows, &other] {
            let error = dispatcher.begin(2, handle, identities(2)).err().expect("busy backend");
            assert_eq!(error.downcast_ref::<DispatchError>(), Some(&DispatchError::Busy));
        }
        rows.boundary.before_ffn(rows.stream.raw)?;
        ticket.land()?;
        is_error(rows.boundary.idle(), DispatchError::Busy);
        assert!(dispatcher.begin(2, &other, identities(2)).is_err());
        rows.boundary.before_join(rows.stream.raw)?;
        rows.boundary.joined();
        let ticket = dispatcher.begin(2, &other, identities(2))?;
        ticket.land()?;
        other.boundary.before_join(other.stream.raw)?;
        other.boundary.joined();
        assert_eq!(count.get(), 2);
        Ok(())
    }

    #[test]
    fn whole_backend_stages_before_overwrite_and_only_writes_routed_sum() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let rows = rows(device)?;
        device.run(|| library.copy_h2d(rows.activation.buffer, &vec![37; 6144 * 2]))?;
        let (dispatcher, _) = dispatcher(device)?;
        let ticket = dispatcher.begin(2, &rows, identities(1))?;
        rows.boundary.before_ffn(rows.stream.raw)?;
        device.run(|| library.copy_h2d(rows.activation.buffer, &vec![91; 6144 * 2]))?;
        ticket.land()?;
        rows.boundary.before_join(rows.stream.raw)?;
        let mut output = vec![0; 6144 * 2];
        device.run(|| library.copy_d2h(&mut output, rows.output.buffer))?;
        assert!(output.iter().all(|&value| value == 37));
        rows.boundary.joined();
        Ok(())
    }

    #[test]
    fn out_of_order_spark_unit_lands_flush_only_in_admission_order() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let first = rows(device)?;
        let second = rows(device)?;
        for rows in [&first, &second] {
            rows.boundary.acquire()?;
            Boundary::record(&rows.boundary.staged, rows.stream.raw)?;
            rows.boundary.phase.set(Phase::Staged);
        }
        let mut pending = VecDeque::from([
            Pending { id: 1, rows: first.clone(), wave: None, land: false },
            Pending { id: 2, rows: second.clone(), wave: None, land: true }]);
        let mut delivered = Vec::new();
        flush_fifo(&mut pending, |item| { delivered.push(item.id); Ok(item.rows.stream.raw) })?;
        assert!(delivered.is_empty());
        is_error(second.boundary.before_join(second.stream.raw), DispatchError::NotLanded);
        pending[0].land = true;
        flush_fifo(&mut pending, |item| { delivered.push(item.id); Ok(item.rows.stream.raw) })?;
        assert_eq!(delivered, [1, 2]);
        assert!(pending.is_empty());
        first.boundary.before_join(first.stream.raw)?;
        second.boundary.before_join(second.stream.raw)?;
        first.boundary.joined();
        second.boundary.joined();
        Ok(())
    }

    #[test]
    fn dropped_ticket_terminally_drains_before_releasing_owners() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let rows = rows(device)?;
        let (dispatcher, _) = dispatcher(device)?;
        let ticket = dispatcher.begin(2, &rows, identities(1))?;
        drop(ticket);
        is_error(rows.boundary.idle(), DispatchError::Terminal);
        assert!(dispatcher.inner.borrow().terminal);
        let error = dispatcher.begin(2, &rows, identities(2)).err().expect("terminal backend");
        assert_eq!(error.downcast_ref::<DispatchError>(), Some(&DispatchError::Terminal));
        Ok(())
    }

    #[test]
    fn row_identities_and_positions_survive_protocol_construction() -> Result<()> {
        let mut origins = identities(41);
        origins.rows.push(ExpertProtocolV2RowDescriptor { row_id: 500,
            source_kind: ExpertV2SourceKind::Prefill, source_request_id: 999,
            token_position: 71, route_offset: 4, route_count: 4 });
        origins.validate(2)?;
        let expected = origins.rows.clone();
        let routes = (0..8).map(|index| cuteafd_transport::ExpertProtocolV2RouteEntry {
            row_index: index / 4, expert_id: index % 4, gate_weight: 0.25 }).collect();
        let request = origins.request(2, routes, bytes::Bytes::from(vec![0; 2 * 6144 * 2]))?;
        assert_eq!(request.rows, expected);
        let mut bad = identities(1);
        bad.rows[0].route_offset = 4;
        assert!(bad.validate(1).is_err());
        Ok(())
    }

    #[test]
    fn retained_activation_aliases_add_no_physical_allocation_or_receipt() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let rows = rows(device)?;
        let count = fixture.events()?.matches('D').count();
        let alias = rows.clone();
        assert!(Rc::ptr_eq(&rows.activation, &alias.activation));
        let devices = [0];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "test", Lifetime::TARGET,
            Category::Workspace, "routed activation");
        rows.activation.inventory(&mut inventory, "activation")?;
        assert_eq!(inventory.receipts.len(), 1);
        assert_eq!(fixture.events()?.matches('D').count(), count);
        Ok(())
    }
}
