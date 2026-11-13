//! Request shapes on shared packing and causal unit execution; no serve loop.
use super::{cache::{Append, Metadata, PageBank, Placement},
    experts::{DispatchRows, Dispatcher, RoutedRows, Ticket}, head::Head, serve::Storage};
use crate::shared::{prefill_packing::{self, Limits},
    prefill_pipeline::{prefill_units, run_prefill_units, FfnRoute, PrefillHooks, Unit, UnitSegment},
    token_io::DeviceLogits};
use cuteafd_engine::prefix::PrefixFamily;
use cuteafd_loader::families::minimax_m3::MiniMaxM3Config;
use cuteafd_transport::{ExpertProtocolV2RowDescriptor, ExpertV2SourceKind};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::placement::families::minimax_m3::RequestCapacity;
use std::{collections::BTreeSet, ops::Range};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestRows {
    pub slot: usize,
    pub start: usize,
    pub rows: usize,
}

/// Only geometry from admission, not a family-local packing or wave policy.
pub(crate) struct Requests {
    capacity: RequestCapacity,
    context: usize,
    layers: usize,
    waves: usize,
}

impl Requests {
    pub fn new(capacity: RequestCapacity, context: usize, layers: usize, waves: usize) -> Result<Self> {
        ensure!(RequestCapacity::new(u64::from(capacity.decode.rows), u64::from(capacity.prefill.rows))? == capacity,
            "MiniMax request capacity differs from admission");
        ensure!(context > 0 && layers > 0, "MiniMax request geometry must be positive");
        super::admission::serve_admission::validate_prefill(cuteafd_loader::placement::planner::PrefillShape {
            rows: u64::from(capacity.prefill.rows), waves: u64::try_from(waves)?,
        })?;
        Ok(Self { capacity, context, layers, waves })
    }

    fn validate(&self, requests: &[RequestRows]) -> Result<()> {
        ensure!(!requests.is_empty() && requests.len() <= self.capacity.decode.requests as usize,
            "MiniMax request batch exceeds admitted concurrency");
        let mut slots = BTreeSet::new();
        for request in requests {
            ensure!(request.slot < self.capacity.decode.requests as usize && slots.insert(request.slot),
                "MiniMax duplicate or unadmitted request slot");
            ensure!(request.rows > 0 && request.start.checked_add(request.rows)
                .is_some_and(|end| end <= self.context), "MiniMax request exceeds admitted context");
        }
        Ok(())
    }

    pub fn decode_rows(&self, requests: &[RequestRows]) -> Result<usize> {
        self.validate(requests)?;
        ensure!(requests.iter().all(|request| request.rows == 1),
            "MiniMax native decode requires one row per request");
        ensure!(requests.len() <= self.capacity.execution_rows() as usize,
            "MiniMax native decode exceeds admitted execution rows");
        Ok(requests.len())
    }

    /// Multi-token verify is ordinary packed prefill. Both cutting and grouping
    /// use the shared planners; repeated pieces of a request cannot share a unit.
    pub fn verify_units(&self, requests: &[RequestRows]) -> Result<Vec<Unit>> {
        self.validate(requests)?;
        let limits = Limits { rows: self.capacity.prefill.rows as usize,
            chunk_rows: self.capacity.prefill.rows as usize, max_context: self.context, taps: None };
        let mut chunks = Vec::new();
        for (sequence, request) in requests.iter().enumerate() {
            for rows in prefill_units(request.rows, limits.rows, 1, 1)? {
                let start = request.start.checked_add(rows.start).context("MiniMax verify position overflow")?;
                chunks.push((sequence, (start, rows.len())));
            }
        }
        let fits = |range: Range<usize>| {
            let members = &chunks[range];
            members.len() <= self.capacity.prefill.requests as usize
                && members.iter().map(|(sequence, _)| sequence).collect::<BTreeSet<_>>().len() == members.len()
                && prefill_packing::plan(&members.iter().map(|(_, chunk)| *chunk).collect::<Vec<_>>(), &limits).is_ok()
        };
        let mut offset = 0usize;
        prefill_packing::groups(chunks.len(), fits).into_iter().enumerate().map(|(index, range)| {
            let members = &chunks[range];
            let packed = prefill_packing::plan(&members.iter().map(|(_, chunk)| *chunk).collect::<Vec<_>>(), &limits)?;
            let rows = packed.last().context("MiniMax verify unit is empty")?.end_row();
            ensure!(members.len() <= self.capacity.prefill.requests as usize
                && rows <= self.capacity.execution_rows() as usize,
                "MiniMax verify unit exceeds admitted execution capacity");
            let first = offset;
            offset = offset.checked_add(rows).context("MiniMax verify row overflow")?;
            let segments = members.iter().zip(packed).map(|((sequence, _), segment)| UnitSegment {
                sequence: *sequence, rows: segment.first..segment.end_row(),
                tokens: segment.start..segment.start + segment.rows,
                prompt_end: false, snapshot: false, head: true,
            }).collect();
            Ok(Unit { index, sequence: members[0].0, segments, rows: first..offset,
                layers: 0..self.layers, slot: index % self.waves, barrier: false })
        }).collect()
    }

    /// end_unit consumes bounded head tiles before commit frees a slot. The
    /// arithmetic hooks retain all storage and own abort/drain on submission error.
    pub fn verify<H: PrefillHooks>(&self, requests: &[RequestRows], hooks: &mut H) -> Result<()> {
        run_prefill_units(hooks, &self.verify_units(requests)?, self.waves)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum ServingError {
    #[error("MiniMax packed serving awaits per-owner two-RTX transport qualification")]
    TwoRtxNotQualified,
    #[error("MiniMax packed serving requires one ordered decode/head/prefill stream")]
    DifferentStreams,
}

pub(crate) struct ServingRequest<'p> {
    pub identity: u64,
    pub kind: ExpertV2SourceKind,
    pub placement: &'p mut Placement,
    /// Full token positions, including any restored prefix.
    pub tokens: &'p [u32],
}

fn ordered_rank(streams: &[Vec<usize>]) -> Result<()> {
    ensure!(streams.len() == 1, ServingError::TwoRtxNotQualified);
    let first = streams[0].first().context("MiniMax packed serving has no stream")?;
    ensure!(streams[0].iter().all(|stream| stream == first), ServingError::DifferentStreams);
    Ok(())
}

fn appends<'p>(unit: &Unit, requests: &'p [ServingRequest<'_>]) -> Result<Vec<Placement>> {
    unit.segments.iter().map(|segment| {
        let request = requests.get(segment.sequence).context("MiniMax packed sequence missing")?;
        ensure!(request.identity != 0 && request.placement.len <= segment.tokens.start
            && segment.tokens.end <= request.tokens.len(), "MiniMax packed request frontier/tokens differ");
        Ok(Placement { len: segment.tokens.start, ..request.placement.clone() })
    }).collect()
}

fn prepare_unit<'a>(bank: &PageBank<'a>, metadata: &mut Metadata<'a>, unit: &Unit,
    requests: &[ServingRequest<'_>]) -> Result<()> {
    let placements = appends(unit, requests)?;
    let spans = unit.segments.iter().zip(&placements).map(|(segment, placement)|
        Append { first_row: segment.rows.start, placement, rows: segment.rows.len() }).collect::<Vec<_>>();
    bank.prepare_segments(metadata, &spans)
}

fn dispatch_rows(unit: &Unit, requests: &[ServingRequest<'_>], identity: u64, version: u64) -> Result<DispatchRows> {
    ensure!(identity != 0 && version != 0, "MiniMax dispatch identity/version must be nonzero");
    let mut rows = Vec::with_capacity(unit.rows.len());
    for segment in &unit.segments {
        let request = requests.get(segment.sequence).context("MiniMax dispatch sequence missing")?;
        for (row, position) in segment.rows.clone().zip(segment.tokens.clone()) {
            rows.push(ExpertProtocolV2RowDescriptor { row_id: u64::try_from(unit.rows.start + row)?,
                source_kind: request.kind, source_request_id: request.identity,
                token_position: u64::try_from(position)?, route_offset: u32::try_from(row * 4)?, route_count: 4 });
        }
    }
    Ok(DispatchRows { request_id: identity, placement_version: version, rows })
}

/// One primitive per hook, not a second executor. The CPU model supplies these
/// primitives explicitly; production always uses retained native arithmetic.
trait SlotArithmetic {
    fn begin(&mut self, unit: &Unit, requests: &[ServingRequest<'_>]) -> Result<()>;
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()>;
    fn router(&mut self, layer: usize, unit: &Unit) -> Result<()>;
    fn ffn(&mut self, layer: usize, unit: &Unit, shared: bool) -> Result<()>;
    fn join(&mut self, unit: &Unit) -> Result<()>;
    fn head(&mut self, unit: &Unit) -> Result<()>;
    fn commit(&mut self, unit: &Unit, requests: &mut [ServingRequest<'_>]) -> Result<()>;
}

struct NativeSlots<'s, 'a, F> {
    storage: &'s Storage<'a>,
    cfg: &'s MiniMaxM3Config,
    consume: F,
}
impl<F: FnMut(usize, Range<usize>, DeviceLogits) -> Result<()>> SlotArithmetic for NativeSlots<'_, '_, F> {
    fn begin(&mut self, unit: &Unit, requests: &[ServingRequest<'_>]) -> Result<()> {
        let mut state = self.storage.requests.borrow_mut();
        let rank = &mut state.ranks[0];
        prepare_unit(&rank.bank, &mut rank.prefill[unit.slot], unit, requests)?;
        let owner = self.storage.ranks[0].prefill[unit.slot].borrow();
        let tokens = unit.segments.iter().flat_map(|segment|
            requests[segment.sequence].tokens[segment.tokens.clone()].iter().copied()).collect::<Vec<_>>();
        let capacity = state.capacity.prefill.rows as usize;
        Head::embed_tokens(self.cfg, capacity, &self.storage.embedding, &tokens, &owner.stream,
            super::cache::view(owner.plane("route_ids")?, 0, capacity * 4)?,
            owner.dense.producer_destination(), owner.plane("residual")?)
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let state = self.storage.requests.borrow();
        let request = &state.ranks[0];
        let segments = request.bank.segments(&request.prefill[unit.slot], layer)?;
        let rank = &self.storage.ranks[0];
        // SAFETY: the retained bank and metadata cover disjoint physical append
        // slots; the shared executor orders causal units on one issuing stream.
        unsafe { rank.prefill[unit.slot].borrow_mut().attention_segments(&rank.kernels,
            &rank.weights, &rank.layout, layer, &segments) }
    }
    fn router(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let rank = &self.storage.ranks[0];
        rank.prefill[unit.slot].borrow_mut().router(&rank.kernels, &rank.weights, layer, unit.rows.len())
    }
    fn ffn(&mut self, layer: usize, unit: &Unit, shared: bool) -> Result<()> {
        let rank = &self.storage.ranks[0];
        rank.prefill[unit.slot].borrow_mut().ffn(&rank.kernels, &rank.weights, &rank.layout,
            layer, unit.rows.len(), shared)
    }
    fn join(&mut self, unit: &Unit) -> Result<()> {
        let rank = &self.storage.ranks[0];
        rank.prefill[unit.slot].borrow().join_routed(&rank.kernels, unit.rows.len())
    }
    fn head(&mut self, unit: &Unit) -> Result<()> {
        let rank = &self.storage.ranks[0];
        let residual = rank.prefill[unit.slot].borrow().plane("residual")?;
        for segment in &unit.segments {
            if !segment.head { continue; }
            let input = super::cache::view(residual, segment.rows.start * self.cfg.hidden * 4,
                segment.rows.len() * self.cfg.hidden * 4)?;
            self.storage.head.project_spans(&rank.weights, input, segment.rows.len(), &mut |offset, logits| {
                let first = segment.tokens.start + offset;
                (self.consume)(segment.sequence, first..first + logits.rows, logits)
            })?;
        }
        Ok(())
    }
    fn commit(&mut self, unit: &Unit, requests: &mut [ServingRequest<'_>]) -> Result<()> {
        for segment in &unit.segments {
            let request = &mut requests[segment.sequence];
            ensure!(request.placement.len == segment.tokens.start, "MiniMax packed commit is out of order");
            let mut next = Placement { len: segment.tokens.end, ..request.placement.clone() };
            self.storage.prefix.bind(&mut next).map_err(|error| anyhow::anyhow!(error.to_string()))?;
            *request.placement = next;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RoutedBinding {
    pub dispatcher: usize,
    pub route: FfnRoute,
}

struct ServingHooks<'s, 'p, 't, 'a, A> {
    arithmetic: &'s mut A,
    sources: &'t [RoutedRows<'a>],
    dispatchers: &'s [Dispatcher<'a>],
    bindings: &'s [Option<RoutedBinding>],
    requests: &'s mut [ServingRequest<'p>],
    next_identity: &'s mut u64,
    version: u64,
}
impl<'t, 'a, A: SlotArithmetic> PrefillHooks for ServingHooks<'_, '_, 't, 'a, A> {
    type Ticket = Ticket<'t, 'a>;
    fn route(&self, layer: usize) -> FfnRoute {
        self.bindings[layer].map_or(FfnRoute::Local, |binding| binding.route)
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> { self.arithmetic.begin(unit, self.requests) }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> { self.arithmetic.attention(layer, unit) }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        if self.bindings[layer].is_some() {
            let ticket = self.send_experts(layer, unit)?;
            self.land_experts(layer, unit, ticket)
        } else { self.arithmetic.ffn(layer, unit, false) }
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        self.arithmetic.router(layer, unit)?;
        let identity = *self.next_identity;
        *self.next_identity = identity.checked_add(1).context("MiniMax packed dispatch identity exhausted")?;
        let identities = dispatch_rows(unit, self.requests, identity, self.version)?;
        let source: &'t RoutedRows<'a> = &self.sources[unit.index];
        let dispatcher = &self.dispatchers[self.bindings[layer].context("MiniMax routed layer has no owner")?.dispatcher];
        let ticket = dispatcher.begin(layer, source, identities)?;
        // Dispatcher stages the normalized activation before shared FFN reuses it.
        self.arithmetic.ffn(layer, unit, true)?;
        Ok(ticket)
    }
    fn land_experts(&mut self, _: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        ticket.land()?;
        self.arithmetic.join(unit)
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> { self.arithmetic.head(unit) }
    fn commit_unit(&mut self, unit: &Unit) -> Result<()> { self.arithmetic.commit(unit, self.requests) }
}

/// Explicit layer-to-dispatcher membership; no last-rank or all-Spark inference.
/// A failed submission retires every dispatcher before draining arithmetic.
pub(crate) struct ServingPrefill<'a> {
    dispatchers: Vec<Dispatcher<'a>>,
    bindings: Vec<Option<RoutedBinding>>,
    next_identity: u64,
    version: u64,
    terminal: bool,
}
impl<'a> ServingPrefill<'a> {
    pub fn new(cfg: &MiniMaxM3Config, dispatchers: Vec<Dispatcher<'a>>, bindings: Vec<Option<RoutedBinding>>,
        version: u64) -> Result<Self> {
        ensure!(version != 0 && bindings.len() == cfg.layers
            && bindings.iter().zip(&cfg.moe).all(|(binding, moe)| binding.is_some() == *moe)
            && bindings.iter().flatten().all(|binding| binding.dispatcher < dispatchers.len())
            && (0..dispatchers.len()).all(|index| bindings.iter().flatten().any(|binding| binding.dispatcher == index)),
            "MiniMax routed bindings differ from admitted layers/owners");
        Ok(Self { dispatchers, bindings, next_identity: 1, version, terminal: false })
    }
    pub fn execute(&mut self, storage: &Storage<'a>, cfg: &MiniMaxM3Config,
        schedule: &Requests, requests: &mut [ServingRequest<'_>],
        consume: impl FnMut(usize, Range<usize>, DeviceLogits) -> Result<()>) -> Result<()> {
        ensure!(!self.terminal, "MiniMax packed serving is terminal");
        ordered_rank(&storage.ranks.iter().map(|rank| std::iter::once(rank.decode.borrow().stream.raw as usize)
            .chain(rank.prefill.iter().map(|owner| owner.borrow().stream.raw as usize)).collect()).collect::<Vec<_>>())?;
        ensure!(schedule.capacity == storage.requests.borrow().capacity && schedule.layers == cfg.layers
            && schedule.waves == storage.ranks[0].prefill.len() && self.bindings.len() == cfg.layers,
            "MiniMax packed schedule differs from retained owners");
        let rows = requests.iter().map(|request| RequestRows { slot: request.placement.slot,
            start: request.placement.len, rows: request.tokens.len().saturating_sub(request.placement.len) }).collect::<Vec<_>>();
        let units = schedule.verify_units(&rows)?;
        let mut identities = BTreeSet::new();
        let mut writable = BTreeSet::new();
        for request in requests.iter() {
            ensure!(request.identity != 0 && identities.insert(request.identity), "MiniMax duplicate request identity");
            storage.requests.borrow().ranks[0].bank.validate(request.placement, request.tokens.len())?;
            for position in request.placement.len..request.tokens.len() {
                ensure!(writable.insert((request.placement.units[position / super::cache::PAGE_ROWS],
                    position % super::cache::PAGE_ROWS)), "MiniMax requests alias writable cache slots");
            }
        }
        let sources = units.iter().map(|unit|
            storage.ranks[0].prefill[unit.slot].borrow().routed_rows(unit.rows.len())).collect::<Result<Vec<_>>>()?;
        let mut arithmetic = NativeSlots { storage, cfg, consume };
        let result = run_prefill_units(&mut ServingHooks { arithmetic: &mut arithmetic, sources: &sources,
            dispatchers: &self.dispatchers, bindings: &self.bindings, requests,
            next_identity: &mut self.next_identity, version: self.version }, &units, schedule.waves);
        if result.is_err() {
            self.terminal = true;
            self.dispatchers.clear();
        }
        // Pending ticket drops terminally quiesce exchange before these drains.
        for unit in &storage.ranks[0].prefill {
            unit.borrow().stream.drain_or_abort("MiniMax packed serving submission");
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::FfnRoute;
    use std::collections::BTreeMap;

    use crate::shared::{experts::rtx::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard},
        memory::device::{Allocation, Device, Stream}};
    use cuteafd_core::memory_layout::Tag;
    use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, native_library_lifetime_fixture::Fixture};
    use cuteafd_loader::families::minimax_m3::execution::{MsaCache, MsaRope};
    use std::{cell::Cell, ffi::c_void, rc::Rc};

    struct ModelSlot<'a> {
        activation: Rc<Allocation<'a>>,
        ids: Rc<Allocation<'a>>,
        weights: Rc<Allocation<'a>>,
        output: Rc<Allocation<'a>>,
        boundary: Rc<super::super::experts::Boundary<'a>>,
        values: Vec<u8>,
    }
    impl<'a> ModelSlot<'a> {
        fn new(device: Device<'a>, rows: usize) -> Result<Self> {
            let allocate = |bytes| Allocation::new(device, bytes, Tag::workspace("minimax/test/packed")).map(Rc::new);
            Ok(Self { activation: allocate(rows * 6144 * 2)?, ids: allocate(rows * 16)?,
                weights: allocate(rows * 16)?, output: allocate(rows * 6144 * 2)?,
                boundary: Rc::new(super::super::experts::Boundary::new(device)?), values: Vec::new() })
        }
        fn source(&self, stream: Rc<Stream<'a>>, rows: usize) -> Result<RoutedRows<'a>> {
            RoutedRows::new(self.activation.clone(), self.ids.clone(), self.weights.clone(),
                self.output.clone(), stream, self.boundary.clone(), rows)
        }
    }
    struct ModelExperts<'a> { output: Allocation<'a>, calls: Rc<Cell<usize>> }
    impl RtxExpertLayer for ModelExperts<'_> {
        fn shard(&self) -> RtxShard { RtxShard::Whole }
        fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
        fn uses_fp8_input(&self) -> bool { false }
        fn layers(&self) -> &[usize] { &[1] }
        fn device(&self) -> i32 { self.output.device.id }
        fn workspace_bytes(&self) -> usize { self.output.buffer.bytes }
        fn output(&self) -> *mut c_void { self.output.buffer.ptr }
        unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput,
            _: Routes, stream: *mut c_void, _: &mut dyn FnMut()) -> Result<()> {
            ensure!(layer == 1, "model expert layer differs");
            let ExpertInput::Bf16(input) = input else { anyhow::bail!("model requires BF16"); };
            self.calls.set(self.calls.get() + 1);
            let source = Buffer { ptr: input, bytes: rows * 6144 * 2, ..self.output.buffer };
            // SAFETY: the real Dispatcher retains input and output and supplies
            // the ordered stream; this explicit CPU model only copies bytes.
            unsafe { self.output.device.library.copy_d2d_async(self.output.buffer, source, source.bytes, stream) }
        }
    }
    struct ModelSlots<'a> {
        bank: PageBank<'a>,
        metadata: Vec<Metadata<'a>>,
        slots: Vec<ModelSlot<'a>>,
        live: BTreeMap<usize, usize>,
        writes: BTreeMap<(usize, usize, usize), u8>,
        heads: Vec<(usize, Range<usize>, Vec<u8>)>,
        ended: BTreeSet<usize>,
        commits: Vec<usize>,
    }
    impl ModelSlots<'_> {
        fn read(&self, buffer: Buffer) -> Result<Vec<u8>> {
            self.bank.stream.drain()?;
            self.bank.stream.device.run(|| {
                let mut out = vec![0; buffer.bytes];
                self.bank.stream.device.library.copy_d2h(&mut out, buffer)?;
                Ok(out)
            })
        }
    }
    // Arithmetic is an explicit model: no native launch is stubbed successful.
    // PageBank, ordered uploads, Dispatcher, boundaries and executor are real.
    impl SlotArithmetic for ModelSlots<'_> {
        fn begin(&mut self, unit: &Unit, requests: &[ServingRequest<'_>]) -> Result<()> {
            ensure!(self.live.insert(unit.slot, unit.index).is_none(), "model slot reused before commit");
            prepare_unit(&self.bank, &mut self.metadata[unit.slot], unit, requests)?;
            self.slots[unit.slot].values = unit.segments.iter().flat_map(|segment|
                requests[segment.sequence].tokens[segment.tokens.clone()].iter().map(|&token| token as u8)).collect();
            Ok(())
        }
        fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            let segments = self.bank.segments(&self.metadata[unit.slot], layer)?;
            for (segment, prepared) in unit.segments.iter().zip(segments) {
                let positions = self.read(prepared.request.positions)?;
                let slots = self.read(prepared.request.slots)?;
                let causal = self.read(prepared.request.causal_lengths)?;
                let pages = self.read(prepared.request.pages)?;
                for (local, position) in segment.tokens.clone().enumerate() {
                    let actual = i64::from_le_bytes(positions[local * 8..local * 8 + 8].try_into()?);
                    ensure!(actual == position as i64, "model absolute position differs");
                    let length = i32::from_le_bytes(causal[local * 4..local * 4 + 4].try_into()?);
                    ensure!(length == position as i32 + 1, "model causal length differs");
                    let physical = i64::from_le_bytes(slots[local * 8..local * 8 + 8].try_into()?) as usize;
                    let page = i32::from_le_bytes(pages[position / 64 * 4..position / 64 * 4 + 4].try_into()?) as usize;
                    ensure!(physical == page * 64 + position % 64, "model writes wrong physical page");
                    // A later unit must see every prior write for this sequence
                    // at this layer; the current unit may attend earlier rows.
                    if local == 0 && self.writes.keys().any(|&(sequence, _, _)| sequence == segment.sequence) {
                        let prior = self.writes.keys().filter(|&&(sequence, l, _)|
                            sequence == segment.sequence && l == layer).map(|&(_, _, p)| p).max();
                        if let Some(prior) = prior { ensure!(prior + 1 == position, "model causal predecessor missing"); }
                    }
                    let value = self.slots[unit.slot].values[segment.rows.start + local];
                    ensure!(self.writes.insert((segment.sequence, layer, position), value).is_none(), "duplicate cache write");
                    for (plane, stride) in super::super::cache::PAGE_PLANES {
                        let destination = super::super::cache::view(self.bank.plane(&format!("layer.{layer}.{plane}"))?,
                            page * stride + (position % 64) * 128, 1)?;
                        self.bank.stream.device.run(|| self.bank.stream.device.library.copy_h2d(destination, &[value]))?;
                    }
                }
            }
            Ok(())
        }
        fn router(&mut self, _: usize, unit: &Unit) -> Result<()> {
            let slot = &self.slots[unit.slot];
            slot.boundary.idle()?;
            let bytes = slot.values.iter().flat_map(|&value| std::iter::repeat_n(value, 6144 * 2)).collect::<Vec<_>>();
            self.bank.stream.device.run(|| self.bank.stream.device.library.copy_h2d(slot.activation.buffer, &bytes))
        }
        fn ffn(&mut self, _: usize, unit: &Unit, shared: bool) -> Result<()> {
            if shared {
                let slot = &self.slots[unit.slot];
                slot.boundary.before_ffn(self.bank.stream.raw)?;
                self.bank.stream.device.run(|| self.bank.stream.device.library.copy_h2d(slot.activation.buffer,
                    &vec![251; unit.rows.len() * 6144 * 2]))?;
            }
            Ok(())
        }
        fn join(&mut self, unit: &Unit) -> Result<()> {
            let slot = &self.slots[unit.slot];
            slot.boundary.before_join(self.bank.stream.raw)?;
            let output = self.read(super::super::cache::view(slot.output.buffer, 0, unit.rows.len() * 6144 * 2)?)?;
            for (row, &expected) in slot.values.iter().enumerate() {
                ensure!(output[row * 6144 * 2..(row + 1) * 6144 * 2].iter().all(|&value| value == expected),
                    "shared FFN overwrote unstaged routed input");
            }
            slot.boundary.joined();
            Ok(())
        }
        fn head(&mut self, unit: &Unit) -> Result<()> {
            for segment in &unit.segments {
                for tile in crate::shared::probe::head_spans(segment.rows.len(), 3)? {
                    let first = segment.tokens.start + tile.start;
                    self.heads.push((segment.sequence, first..first + tile.len(),
                        self.slots[unit.slot].values[segment.rows.start + tile.start..segment.rows.start + tile.end].to_vec()));
                }
            }
            ensure!(self.ended.insert(unit.index), "model head consumed twice");
            Ok(())
        }
        fn commit(&mut self, unit: &Unit, requests: &mut [ServingRequest<'_>]) -> Result<()> {
            ensure!(self.ended.contains(&unit.index), "model slot released before head");
            for segment in &unit.segments {
                let request = &mut requests[segment.sequence];
                ensure!(request.placement.len == segment.tokens.start, "model frontier commit order");
                request.placement.len = segment.tokens.end;
                self.bank.install(request.placement, self.bank.stream.raw)?;
            }
            ensure!(self.live.remove(&unit.slot) == Some(unit.index), "model freed wrong slot");
            self.commits.push(unit.index);
            Ok(())
        }
    }

    #[test]
    fn serving_hooks_write_packed_causal_slots_consume_head_tiles_and_dispatch_once_per_unit() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 64 << 20)?;
        let device = Device { library: &library, id: 1 };
        let stream = Rc::new(Stream::new(device)?);
        let capacity = RequestCapacity::new(4, 64)?;
        let mut cfg = MiniMaxM3Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json"))?)?;
        cfg.layers = 2;
        cfg.moe = vec![false, true];
        let bank = PageBank::new(vec![stream.clone()], &cfg, capacity, MsaCache::new(2, 16, 4, 320)?,
            MsaRope::new(&cfg, 320)?, 0..2)?;
        bank.reserve_tables(stream.raw)?;
        let mut placements = [Placement { units: vec![7, 2, 9, 14, 15], slot: 3, len: 63 },
            Placement { units: vec![4, 11, 1], slot: 0, len: 128 },
            Placement { units: vec![8, 6], slot: 2, len: 1 }];
        for placement in &placements { bank.install(placement, stream.raw)?; }
        let tokens = [vec![37; 270], vec![71; 133], vec![113; 2]];
        let starts = placements.iter().map(|placement| placement.len).collect::<Vec<_>>();
        let mut requests = placements.iter_mut().zip(&tokens).enumerate().map(|(index, (placement, tokens))|
            ServingRequest { identity: 19 + index as u64, kind: ExpertV2SourceKind::Prefill, placement, tokens }).collect::<Vec<_>>();
        let schedule = Requests::new(capacity, 320, 2, 3)?;
        let units = schedule.verify_units(&requests.iter().map(|request| RequestRows {
            slot: request.placement.slot, start: request.placement.len,
            rows: request.tokens.len() - request.placement.len }).collect::<Vec<_>>())?;
        ensure!(units.len() > 3 && units.iter().any(|unit| unit.segments.len() == 3), "model lacks packing/slot reuse");
        let slots = (0..3).map(|_| ModelSlot::new(device, 64)).collect::<Result<Vec<_>>>()?;
        let sources = units.iter().map(|unit| slots[unit.slot].source(stream.clone(), unit.rows.len())).collect::<Result<Vec<_>>>()?;
        let calls = Rc::new(Cell::new(0));
        let dispatchers = vec![Dispatcher::whole(Box::new(ModelExperts {
            output: Allocation::new(device, 64 * 6144 * 2, Tag::workspace("minimax/test/experts"))?, calls: calls.clone() }))?];
        let bindings = vec![None, Some(RoutedBinding { dispatcher: 0, route: FfnRoute::Local })];
        let mut arithmetic = ModelSlots { bank, metadata: (0..3).map(|_| Metadata::new_prefill(stream.clone(), capacity))
            .collect::<Result<_>>()?, slots, live: BTreeMap::new(), writes: BTreeMap::new(),
            heads: Vec::new(), ended: BTreeSet::new(), commits: Vec::new() };
        let mut identity = 1;
        run_prefill_units(&mut ServingHooks { arithmetic: &mut arithmetic, sources: &sources,
            dispatchers: &dispatchers, bindings: &bindings, requests: &mut requests,
            next_identity: &mut identity, version: 23 }, &units, 3)?;
        assert_eq!(calls.get(), units.len());
        assert_eq!(identity, units.len() as u64 + 1);
        assert_eq!(arithmetic.commits, (0..units.len()).collect::<Vec<_>>());
        assert!(arithmetic.live.is_empty());
        for (index, request) in requests.iter().enumerate() {
            assert_eq!(request.placement.len, tokens[index].len());
            let consumed = arithmetic.heads.iter().filter(|(sequence, _, _)| *sequence == index)
                .flat_map(|(_, range, values)| range.clone().zip(values.iter().copied())).collect::<Vec<_>>();
            assert_eq!(consumed, (starts[index]..tokens[index].len())
                .map(|position| (position, tokens[index][position] as u8)).collect::<Vec<_>>());
            for layer in 0..2 {
                for position in starts[index]..tokens[index].len() {
                    let page = request.placement.units[position / 64] as usize;
                    for (plane, stride) in super::super::cache::PAGE_PLANES {
                        let buffer = super::super::cache::view(arithmetic.bank.plane(&format!("layer.{layer}.{plane}"))?,
                            page * stride + position % 64 * 128, 1)?;
                        assert_eq!(arithmetic.read(buffer)?, [tokens[index][position] as u8]);
                    }
                }
            }
        }
        let descriptors = dispatch_rows(units.last().unwrap(), &requests, 88, 23)?;
        assert_eq!(descriptors.rows.len(), units.last().unwrap().rows.len());
        assert_eq!(descriptors.rows.iter().map(|row| row.source_request_id).collect::<BTreeSet<_>>(), BTreeSet::from([19, 20, 21]));
        assert!(descriptors.rows.iter().enumerate().all(|(row, descriptor)| descriptor.route_offset == (row * 4) as u32));
        drop(dispatchers);
        Ok(())
    }

    #[test]
    fn unqualified_two_rtx_and_unordered_streams_fail_with_typed_reasons() {
        assert_eq!(ordered_rank(&[vec![1], vec![2]]).unwrap_err().downcast_ref::<ServingError>(), Some(&ServingError::TwoRtxNotQualified));
        assert_eq!(ordered_rank(&[vec![1, 2, 1]]).unwrap_err().downcast_ref::<ServingError>(), Some(&ServingError::DifferentStreams));
        assert!(ordered_rank(&[vec![7, 7, 7, 7]]).is_ok());
    }

    fn requests(concurrency: usize, rows: usize) -> Vec<RequestRows> {
        (0..concurrency).map(|slot| RequestRows { slot, start: 63 + slot, rows }).collect()
    }

    #[test]
    fn native_decode_is_one_token_per_unique_admitted_request() -> Result<()> {
        for concurrency in [1, 16, 64] {
            let schedule = Requests::new(RequestCapacity::new(concurrency, 64)?, 4096, 60, 3)?;
            let mut batch = requests(concurrency as usize, 1);
            assert_eq!(schedule.decode_rows(&batch)?, concurrency as usize);
            batch.push(RequestRows { slot: concurrency as usize, start: 0, rows: 1 });
            assert!(schedule.decode_rows(&batch).is_err());
            batch.pop();
            batch[0].rows = 2;
            assert!(schedule.decode_rows(&batch).is_err());
        }
        Ok(())
    }

    #[test]
    fn verify_uses_shared_units_with_bounded_rows_members_and_causal_positions() -> Result<()> {
        for concurrency in [1, 16, 64] {
            for unit_rows in [64, 128] {
                let capacity = RequestCapacity::new(concurrency, unit_rows)?;
                let schedule = Requests::new(capacity, 4096, 60, 3)?;
                for rows in [1, 17, 65, 257] {
                    let batch = requests(concurrency as usize, rows);
                    let units = schedule.verify_units(&batch)?;
                    let mut next = batch.iter().map(|request| request.start).collect::<Vec<_>>();
                    let mut flattened = 0;
                    for (index, unit) in units.iter().enumerate() {
                        assert_eq!((unit.index, unit.slot, unit.layers.clone()), (index, index % 3, 0..60));
                        assert_eq!(unit.rows.start, flattened);
                        assert!(unit.rows.len() <= unit_rows as usize);
                        assert!(unit.rows.len() <= capacity.execution_rows() as usize);
                        assert!(unit.segments.len() <= capacity.prefill.requests as usize);
                        let mut local = 0;
                        let mut seen = BTreeSet::new();
                        for segment in &unit.segments {
                            assert!(seen.insert(segment.sequence));
                            assert_eq!(segment.rows.start, local);
                            assert_eq!(segment.rows.len(), segment.tokens.len());
                            assert_eq!(segment.tokens.start, next[segment.sequence]);
                            assert!(segment.head && !segment.snapshot && !segment.prompt_end);
                            next[segment.sequence] = segment.tokens.end;
                            local = segment.rows.end;
                        }
                        assert_eq!(local, unit.rows.len());
                        flattened = unit.rows.end;
                    }
                    assert_eq!(flattened, concurrency as usize * rows);
                    assert_eq!(next, batch.iter().map(|request| request.start + rows).collect::<Vec<_>>());
                    if flattened > unit_rows as usize { assert!(units.len() > 1); }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn draft_depth_three_keeps_twenty_and_sixty_four_live_rows_unpadded() -> Result<()> {
        for concurrency in [5, 16] {
            let schedule = Requests::new(RequestCapacity::new(concurrency, 64)?, 4096, 60, 3)?;
            let units = schedule.verify_units(&requests(concurrency as usize, 4))?;
            assert_eq!(units.len(), 1);
            assert_eq!(units[0].rows.len(), concurrency as usize * 4);
            assert_eq!(units[0].segments.len(), concurrency as usize);
            assert!(units[0].segments.iter().all(|segment| segment.rows.len() == 4));
        }
        Ok(())
    }

    #[test]
    fn duplicate_slots_and_invalid_geometry_fail_before_submission() -> Result<()> {
        let capacity = RequestCapacity::new(64, 64)?;
        let schedule = Requests::new(capacity, 4096, 60, 3)?;
        let mut batch = requests(2, 1);
        batch[1].slot = batch[0].slot;
        assert!(schedule.decode_rows(&batch).is_err());
        assert!(schedule.verify_units(&batch).is_err());
        batch[1].slot = 64;
        assert!(schedule.verify_units(&batch).is_err());
        batch[1].slot = 1;
        batch[1].start = usize::MAX;
        assert!(schedule.verify_units(&batch).is_err());
        assert!(schedule.verify_units(&[]).is_err());
        assert!(Requests::new(capacity, 0, 60, 3).is_err());
        assert!(Requests::new(capacity, 4096, 60, 0).is_err());
        assert!(Requests::new(RequestCapacity::new(16, 32)?, 4096, 60, 3).is_err());
        let mut invalid = capacity;
        invalid.prefill.requests = 1;
        assert!(Requests::new(invalid, 4096, 60, 3).is_err());
        Ok(())
    }

    #[derive(Default)]
    struct Hooks {
        live: BTreeMap<usize, usize>,
        heads: BTreeSet<usize>,
        committed: Vec<usize>,
        attended: BTreeMap<(usize, usize), usize>,
    }
    impl PrefillHooks for Hooks {
        type Ticket = usize;
        fn route(&self, layer: usize) -> FfnRoute {
            if layer == 1 { FfnRoute::Spark } else { FfnRoute::Local }
        }
        fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
            assert!(self.live.insert(unit.slot, unit.index).is_none(), "slot reused before commit");
            Ok(())
        }
        fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            for segment in &unit.segments {
                let next = self.attended.entry((segment.sequence, layer)).or_insert(segment.tokens.start);
                assert_eq!(*next, segment.tokens.start, "later causal unit attended first");
                *next = segment.tokens.end;
            }
            Ok(())
        }
        fn local_ffn(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
        fn send_experts(&mut self, _: usize, unit: &Unit) -> Result<usize> { Ok(unit.index) }
        fn land_experts(&mut self, _: usize, unit: &Unit, ticket: usize) -> Result<()> {
            assert_eq!(unit.index, ticket);
            Ok(())
        }
        fn end_unit(&mut self, unit: &Unit) -> Result<()> {
            assert!(self.heads.insert(unit.index), "head consumed twice");
            Ok(())
        }
        fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
            assert!(self.heads.contains(&unit.index), "slot freed before head consumption");
            assert_eq!(self.live.remove(&unit.slot), Some(unit.index));
            self.committed.push(unit.index);
            Ok(())
        }
    }

    #[test]
    fn shared_executor_preserves_causality_and_consumes_heads_before_slot_reuse() -> Result<()> {
        let schedule = Requests::new(RequestCapacity::new(16, 64)?, 4096, 4, 3)?;
        let batch = requests(16, 257);
        let units = schedule.verify_units(&batch)?;
        let mut hooks = Hooks::default();
        schedule.verify(&batch, &mut hooks)?;
        assert!(hooks.live.is_empty());
        assert_eq!(hooks.committed, (0..units.len()).collect::<Vec<_>>());
        assert_eq!(hooks.heads.len(), units.len());
        for (sequence, request) in batch.iter().enumerate() {
            for layer in 0..4 { assert_eq!(hooks.attended[&(sequence, layer)], request.start + request.rows); }
        }
        Ok(())
    }
}
