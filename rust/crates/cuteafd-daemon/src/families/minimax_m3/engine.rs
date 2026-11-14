//! Request shapes on shared packing and causal unit execution; no serve loop.
use super::{cache::{Append, Metadata, PageBank, Placement},
    experts::{DispatchRows, Dispatcher, RoutedRows, Ticket}, head::Head, owners::Storage,
    graphs::{Graphs, Key, Op}};
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
    /// Absolute first position of this append; tokens contain no dummy history.
    pub start: usize,
    pub tokens: &'p [u32],
    /// Absolute rows whose heads the caller consumes before scratch reuse.
    pub heads: Range<usize>,
}

impl ServingRequest<'_> {
    fn end(&self) -> Result<usize> {
        self.start.checked_add(self.tokens.len()).context("MiniMax append extent overflow")
    }
    fn tokens_at(&self, span: Range<usize>) -> Result<&[u32]> {
        ensure!(span.start >= self.start && span.end <= self.end()? && span.start <= span.end,
            "MiniMax append slice differs from absolute positions");
        Ok(&self.tokens[span.start - self.start..span.end - self.start])
    }
}

// Each channel owns one peer sequence: flag, two send words, receive word.
// All offsets address the unit's already-admitted 256-byte split control.
const SPLIT_CONTROL_CHANNELS: [(&str, usize, usize); 4] = [
    ("partial", 0, 16),
    ("routes", 16, 16),
    ("routed-sum", 32, 16),
    ("release", 48, 16),
];

pub(crate) struct SplitExecution<'a> {
    hops: crate::shared::peer_split::hop::HopLink<'a>,
    owners: Vec<usize>,
    hop_indices: Vec<usize>,
    onboarded: Vec<bool>,
    abort_streams: [crate::shared::memory::device::Stream<'a>; 2],
    compute_streams: [std::rc::Rc<crate::shared::memory::device::Stream<'a>>; 2],
    abort_words: [Vec<cuteafd_ffi::CuteafdDeviceBuffer>; 2],
    terminal: std::cell::Cell<bool>,
}

impl<'a> SplitExecution<'a> {
    pub fn new(inputs: &cuteafd_loader::placement::families::minimax_m3::MiniMaxInputs,
        admitted: &cuteafd_loader::placement::families::minimax_m3::AdmittedMiniMax,
        ranks: &[super::owners::Rank<'a>]) -> Result<Self> {
        use crate::shared::peer_split::{hop::HopLink, RankDevice};
        use cuteafd_loader::placement::{ExpertHome, HopSpec, HopStorage, TerminalHome};
        ensure!(inputs.split() && ranks.len() == 2 && inputs.split_wire.is_some(),
            "MiniMax split execution needs two admitted ranks and a selected wire");
        let endpoints = [0, 1].map(|gpu| {
            let owner = ranks[gpu].decode.borrow();
            RankDevice { device: owner.stream.device.id, stream: owner.stream.raw }
        });
        let library = ranks[0].decode.borrow().stream.device.library;
        for gpu in 0..2 {
            let rank = &ranks[gpu];
            ensure!(rank.prefill.iter().all(|unit| unit.borrow().stream.raw == endpoints[gpu].stream),
                "MiniMax split prefill differs from its HopLink stream");
            rank.decode.borrow().stream.device.run(|| {
                library.peer_exchange_initialize()?;
                library.peer_abort_initialize()
            })?;
        }
        let peak = 1 + cuteafd_loader::push_load::COPY_STREAMS as u64 + 1 + 1;
        for gpu in 0..2 {
            let expected = format!("minimax/driver-stream-slabs/gpu{gpu} ({peak} peak live streams: model + push copies + rotary; split abort 1)");
            ensure!(admitted.solved.facts.allowances.iter().any(|allowance| allowance.source == expected),
                "MiniMax split abort stream is not charged at its exact peak count");
        }
        let abort_streams = [crate::shared::memory::device::Stream::new(ranks[0].decode.borrow().stream.device)?,
            crate::shared::memory::device::Stream::new(ranks[1].decode.borrow().stream.device)?];
        let spec = HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: inputs.cfg.hidden as u64 * 4,
            rows: inputs.prefill.rows.max(inputs.sequences), lanes: 1, entry_gpu: 0, head_gpu: 1,
            terminal: TerminalHome::Replicated };
        let mut hops = HopLink::new(library, endpoints, &admitted.placement.hops, spec,
            inputs.cfg.hidden * 4 * 64)?;
        let abort = [0, 1].map(|gpu| ranks[gpu].decode.borrow().split_plane("abort")
            .map(|buffer| buffer.ptr.cast::<u32>() as *const u32));
        let [abort0, abort1] = abort;
        hops.attach_abort([abort0?, abort1?])?;
        let hop_indices = (0..inputs.cfg.layers).map(|layer| admitted.placement.hops.iter()
            .position(|hop| hop.at == cuteafd_loader::placement::HopPoint::AfterAttention(layer))
            .context("MiniMax split residual broadcast missing from admission")).collect::<Result<_>>()?;
        let compute_streams = [0, 1].map(|gpu| std::rc::Rc::clone(&ranks[gpu].decode.borrow().stream));
        let abort_words = [0, 1].map(|gpu| std::iter::once(&ranks[gpu].decode).chain(&ranks[gpu].prefill)
            .map(|unit| unit.borrow().split_plane("abort")).collect::<Result<Vec<_>>>());
        let [words0, words1] = abort_words;
        Ok(Self { hops, owners: inputs.owners.iter().map(|&gpu| usize::from(gpu)).collect(), hop_indices,
            onboarded: admitted.placement.layers.iter().map(|layer| layer.experts == ExpertHome::RtxTp2).collect(),
            abort_streams, compute_streams, abort_words: [words0?, words1?], terminal: std::cell::Cell::new(false) })
    }

    pub fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        for (name, buffer) in self.hops.backing_inventory("hops") {
            inventory.device(&name, buffer)?;
        }
        Ok(())
    }

    fn publish_abort(&self) -> Result<()> {
        self.terminal.set(true);
        // Never publish behind a possibly blocked compute wait. The independent
        // retained streams have no dependency on either rank's compute stream.
        let mut first_error = None;
        for gpu in 0..2 {
            let stream = &self.abort_streams[gpu];
            let result = stream.device.run(|| {
                for word in &self.abort_words[gpu] {
                    // SAFETY: charged persistent abort words and independent
                    // startup stream; words are terminal and never reset.
                    unsafe { stream.device.library.peer_abort_publish(word.ptr.cast(), stream.raw)?; }
                }
                Ok(())
            }).and_then(|()| stream.drain());
            if first_error.is_none() { first_error = result.err(); }
        }
        match first_error { Some(error) => Err(error), None => Ok(()) }
    }

    fn cancel(&self) -> Result<()> {
        let aborted = self.publish_abort();
        let drained = drain_split_pair(|gpu| self.compute_streams[gpu].drain());
        aborted.and(drained)
    }

    fn owner(&self, layer: usize) -> Result<usize> {
        ensure!(!self.terminal.get(), "MiniMax split transport is terminal");
        self.owners.get(layer).copied().context("MiniMax split attention layer missing")
    }

    fn residual(&self, units: [&super::layer::UnitOwner<'a>; 2], layer: usize, rows: usize) -> Result<()> {
        let from = self.owner(layer)?;
        let index = self.hop_indices[layer];
        let source = units[from].plane("residual")?;
        let destination = units[1 - from].plane("residual")?;
        let ticket = self.hops.send_into(index, 0, source.ptr, rows, Some(destination))?;
        self.hops.land(ticket)?;
        Ok(())
    }

    fn word(buffer: cuteafd_ffi::CuteafdDeviceBuffer, channel: usize, word: usize) -> Result<*mut u32> {
        let (_, offset, bytes) = *SPLIT_CONTROL_CHANNELS.get(channel).context("MiniMax split channel missing")?;
        ensure!(bytes == 16 && word < 4 && offset + bytes <= buffer.bytes
            && buffer.ptr as usize % 16 == 0, "MiniMax split control extent/alignment differs");
        Ok(buffer.ptr.wrapping_byte_add(offset + word * 4).cast())
    }

    fn send(units: [&super::layer::UnitOwner<'a>; 2], from: usize, channel: usize,
        regions: &[(cuteafd_ffi::CuteafdDeviceBuffer, cuteafd_ffi::CuteafdDeviceBuffer, usize)]) -> Result<usize> {
        let to = 1 - from;
        let source = units[from];
        let target = units[to];
        let local = source.split_plane("control")?;
        let peer = target.split_plane("control")?;
        let flag = Self::word(peer, channel, 0)?;
        let sequence = Self::word(local, channel, 1)?;
        source.stream.device.run(|| {
            for &(src, dst, bytes) in regions {
                ensure!(src.device_id == source.stream.device.id && dst.device_id == target.stream.device.id
                    && bytes > 0 && bytes <= src.bytes && bytes <= dst.bytes,
                    "MiniMax split transfer differs from charged destination");
                // SAFETY: admitted unit planes stay leased through the matching
                // wait and release handshake; source FIFO orders all producers.
                unsafe { source.stream.device.library.copy_d2d_async(dst, src, bytes, source.stream.raw)?; }
            }
            // SAFETY: disjoint persistent 16-byte channel, with payload copies
            // preceding its system-scope signal on this source stream.
            unsafe { source.stream.device.library.peer_push_signal(peer.ptr, local.ptr,
                0, flag, sequence, 0, source.stream.raw)?; }
            Ok(to)
        })
    }

    fn land(units: [&super::layer::UnitOwner<'a>; 2], to: usize, channel: usize) -> Result<()> {
        let unit = units[to];
        let control = unit.split_plane("control")?;
        let flag = Self::word(control, channel, 0)?;
        let sequence = Self::word(control, channel, 3)?;
        let abort = unit.split_plane("abort")?;
        unit.stream.device.run(|| {
            // SAFETY: the matching send has already been queued, never merely
            // promised. All words remain live until both ranks drain.
            unsafe { unit.stream.device.library.peer_wait_abortable(flag, sequence,
                abort.ptr.cast(), unit.stream.raw) }
        })
    }

    fn routes(&self, units: [&super::layer::UnitOwner<'a>; 2], layer: usize, rows: usize) -> Result<()> {
        if !self.onboarded[layer] { return Ok(()); }
        let regions = ["route_ids", "route_weights"].map(|name| Ok((units[0].plane(name)?,
            units[1].plane(name)?, rows * 16))).into_iter().collect::<Result<Vec<_>>>()?;
        let ticket = Self::send(units, 0, 1, &regions)?;
        Self::land(units, ticket, 1)
    }

    fn partials(&self, units: [&super::layer::UnitOwner<'a>; 2], rows: usize) -> Result<()> {
        send_split_pair(|from| {
            let local = units[from].split_plane("local")?;
            let receive = units[1 - from].split_plane("receive")?;
            let bytes = local.bytes / units[from].rows() as usize * rows;
            Self::send(units, from, 0, &[(local, receive, bytes)])
        }, |to| Self::land(units, to, 0))
    }

    fn routed(&self, units: [&super::layer::UnitOwner<'a>; 2], rows: usize) -> Result<()> {
        let ticket = Self::send(units, 0, 2, &[(units[0].plane("routed")?, units[1].plane("routed")?, rows * 6144 * 2)])?;
        Self::land(units, ticket, 2)
    }

    fn stage(&self, storage: &Storage<'a>, units: [&mut super::layer::UnitOwner<'a>; 2],
        layer: usize, rows: usize, shared: bool) -> Result<()> {
        if shared { self.routes([&*units[0], &*units[1]], layer, rows)?; }
        for (gpu, unit) in units.into_iter().enumerate() {
            let rank = &storage.ranks[gpu];
            unit.partial_ffn(&rank.kernels, &rank.weights, &rank.layout, layer, rows, shared)?;
        }
        Ok(())
    }

    fn meet(&self, storage: &Storage<'a>, units: [&super::layer::UnitOwner<'a>; 2],
        rows: usize, shared: bool) -> Result<()> {
        self.partials(units, rows)?;
        // SAFETY: both partial sends precede their waits. Rank0's routed
        // boundary explicitly waits for ticket.land's ready event here.
        unsafe { units[0].meet_ffn(&storage.ranks[0].kernels, rows, shared)?; }
        // The rank0 FIFO now orders ready before this broadcast, including
        // TP2 experts whose producer/ready event is on an exchange stream.
        if shared { self.routed(units, rows)?; }
        // SAFETY: rank1's partial and final BF16 routed sum have landed; its
        // idle routed boundary is not an invented second routed ticket.
        unsafe { units[1].meet_ffn(&storage.ranks[1].kernels, rows, shared)?; }
        self.release(units)
    }

    fn release(&self, units: [&super::layer::UnitOwner<'a>; 2]) -> Result<()> {
        // Both consumers finish their meeting before either side can overwrite
        // the peer's retained receive/routed/route planes on the next layer.
        send_split_pair(|from| Self::send(units, from, 3, &[]), |to| Self::land(units, to, 3))
    }
}

fn send_split_pair<T>(mut send: impl FnMut(usize) -> Result<T>,
    mut land: impl FnMut(T) -> Result<()>) -> Result<()> {
    let rank0 = send(0)?;
    let rank1 = send(1)?;
    land(rank0)?;
    land(rank1)
}

fn drain_split_pair(mut drain: impl FnMut(usize) -> Result<()>) -> Result<()> {
    let rank0 = drain(0);
    let rank1 = drain(1);
    rank0.and(rank1)
}

fn validate_execution_streams(storage: &Storage<'_>) -> Result<()> {
    let streams = storage.ranks.iter().map(|rank| std::iter::once(rank.decode.borrow().stream.raw as usize)
        .chain(rank.prefill.iter().map(|owner| owner.borrow().stream.raw as usize)).collect()).collect::<Vec<_>>();
    if storage.split.is_some() {
        ensure!(streams.len() == 2, "MiniMax split execution needs two ranks");
        for rank in streams { ordered_rank(&[rank])?; }
        Ok(())
    } else { ordered_rank(&streams) }
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
            && request.start <= segment.tokens.start && segment.tokens.end <= request.end()?, "MiniMax packed request frontier/tokens differ");
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
    fn head(&mut self, unit: &Unit, requests: &[ServingRequest<'_>]) -> Result<()>;
    fn commit(&mut self, unit: &Unit, requests: &mut [ServingRequest<'_>]) -> Result<()>;
    fn abort_split(&self) {}
}

struct NativeSlots<'s, 'a, F> {
    storage: &'s Storage<'a>,
    cfg: &'s MiniMaxM3Config,
    consume: F,
    publish: bool,
    taps: Option<&'s mut super::prefill::LayerTaps>,
}
impl<F: FnMut(usize, Range<usize>, DeviceLogits) -> Result<()>> SlotArithmetic for NativeSlots<'_, '_, F> {
    fn abort_split(&self) {
        if let Some(split) = &self.storage.split {
            crate::shared::decode_graph::fatal_drain(split.publish_abort(), "MiniMax split hook before ticket retirement");
        }
    }
    fn begin(&mut self, unit: &Unit, requests: &[ServingRequest<'_>]) -> Result<()> {
        let mut state = self.storage.requests.borrow_mut();
        if self.storage.split.is_some() {
            for rank in &mut state.ranks {
                prepare_unit(&rank.bank, &mut rank.prefill[unit.slot], unit, requests)?;
            }
        } else {
            let rank = &mut state.ranks[0];
            prepare_unit(&rank.bank, &mut rank.prefill[unit.slot], unit, requests)?;
        }
        let owner = self.storage.ranks[0].prefill[unit.slot].borrow();
        let tokens = unit.segments.iter().map(|segment|
            requests[segment.sequence].tokens_at(segment.tokens.clone())).collect::<Result<Vec<_>>>()?
            .into_iter().flatten().copied().collect::<Vec<_>>();
        let capacity = state.capacity.prefill.rows as usize;
        Head::embed_tokens(self.cfg, capacity, &self.storage.embedding, &tokens, &owner.stream,
            super::cache::view(owner.plane("route_ids")?, 0, capacity * 4)?,
            owner.dense.producer_destination(), owner.plane("residual")?)
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let state = self.storage.requests.borrow();
        let gpu = self.storage.split.as_ref().map(|split| split.owner(layer)).transpose()?.unwrap_or(0);
        let request = &state.ranks[gpu];
        let segments = request.bank.segments(&request.prefill[unit.slot], layer)?;
        let rank = &self.storage.ranks[gpu];
        if let Some(taps) = &mut self.taps {
            taps.write(layer, "input", "residual", 4, unit, &rank.prefill[unit.slot].borrow())?;
        }
        // SAFETY: the retained bank and metadata cover disjoint physical append
        // slots; the shared executor orders causal units on one issuing stream.
        unsafe { rank.prefill[unit.slot].borrow_mut().attention_segments(&rank.kernels,
            &rank.weights, &rank.layout, layer, &segments)?; }
        if let Some(split) = &self.storage.split {
            let a = self.storage.ranks[0].prefill[unit.slot].borrow();
            let b = self.storage.ranks[1].prefill[unit.slot].borrow();
            split.residual([&a, &b], layer, unit.rows.len())?;
            if let Some(taps) = &mut self.taps {
                for (gpu, owner) in [a, b].iter().enumerate() {
                    taps.write(layer, &format!("rank{gpu}.post_attention"), "residual", 4, unit, owner)?;
                }
            }
        }
        if let Some(taps) = &mut self.taps {
            let owner = rank.prefill[unit.slot].borrow();
            taps.write(layer, "attention_projected.bf16", "ffn", 2, unit, &owner)?;
            taps.write(layer, "post_attention", "residual", 4, unit, &owner)?;
        }
        Ok(())
    }
    fn router(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let rank = &self.storage.ranks[0];
        let owner = rank.prefill[unit.slot].borrow();
        owner.router(&rank.kernels, &rank.weights, layer, unit.rows.len())?;
        if let Some(taps) = &mut self.taps { taps.router(layer, &rank.weights, unit, &owner)?; }
        if let Some(usage) = &self.storage.usage {
            let tally = usage.routes(owner.stream.device.id, layer,
                cuteafd_ffi::expert_usage::UsagePhase::Prefill)?;
            // SAFETY: router initialized retained route ids on this tally's
            // ordered owner stream; the shared executor pins the unit.
            owner.stream.device.run(|| unsafe { tally.record(owner.plane("route_ids")?.ptr,
                unit.rows.len(), None, u32::MAX, true, owner.stream.raw) })?;
        }
        Ok(())
    }
    fn ffn(&mut self, layer: usize, unit: &Unit, shared: bool) -> Result<()> {
        if let Some(split) = &self.storage.split {
            let mut a = self.storage.ranks[0].prefill[unit.slot].borrow_mut();
            let mut b = self.storage.ranks[1].prefill[unit.slot].borrow_mut();
            split.stage(self.storage, [&mut a, &mut b], layer, unit.rows.len(), shared)?;
            if let Some(taps) = &mut self.taps {
                for (gpu, owner) in [&*a, &*b].into_iter().enumerate() {
                    taps.ffn_inputs(layer, &format!("rank{gpu}."), shared,
                        &self.storage.ranks[gpu].layout, unit, owner)?;
                    let buffer = owner.split_plane("local")?;
                    let width = buffer.bytes / owner.rows() as usize / 6144;
                    taps.buffer(layer, &format!("rank{gpu}.partial"), buffer, width, unit, owner)?;
                }
            }
            if !shared {
                split.meet(self.storage, [&a, &b], unit.rows.len(), false)?;
                if let Some(taps) = &mut self.taps {
                    for (gpu, owner) in [&*a, &*b].into_iter().enumerate() {
                        taps.buffer(layer, &format!("rank{gpu}.meeting"), owner.split_plane("meeting")?, 4, unit, owner)?;
                        taps.write(layer, &format!("rank{gpu}.mlp.bf16"), "ffn", 2, unit, owner)?;
                        taps.write(layer, &format!("rank{gpu}.residual"), "residual", 4, unit, owner)?;
                    }
                }
            }
            Ok(())
        } else {
            let rank = &self.storage.ranks[0];
            rank.prefill[unit.slot].borrow_mut().ffn(&rank.kernels, &rank.weights, &rank.layout,
                layer, unit.rows.len(), shared)
        }
    }
    fn join(&mut self, unit: &Unit) -> Result<()> {
        if let Some(split) = &self.storage.split {
            let a = self.storage.ranks[0].prefill[unit.slot].borrow();
            let b = self.storage.ranks[1].prefill[unit.slot].borrow();
            split.meet(self.storage, [&a, &b], unit.rows.len(), true)
        } else {
            let rank = &self.storage.ranks[0];
            rank.prefill[unit.slot].borrow().join_routed(&rank.kernels, unit.rows.len())
        }
    }
    fn head(&mut self, unit: &Unit, requests: &[ServingRequest<'_>]) -> Result<()> {
        let rank = &self.storage.ranks[if self.storage.split.is_some() { 1 } else { 0 }];
        let residual = rank.prefill[unit.slot].borrow().plane("residual")?;
        for segment in &unit.segments {
            if !segment.head { continue; }
            let requested = &requests[segment.sequence].heads;
            let span = segment.tokens.start.max(requested.start)..segment.tokens.end.min(requested.end);
            if span.is_empty() { continue; }
            let row = segment.rows.start + span.start - segment.tokens.start;
            let input = super::cache::view(residual, row * self.cfg.hidden * 4, span.len() * self.cfg.hidden * 4)?;
            self.storage.head.project_spans(&rank.weights, input, span.len(), &mut |offset, logits| {
                let first = span.start + offset;
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
            if self.publish {
                self.storage.prefix.bind(&mut next).map_err(|error| anyhow::anyhow!(error.to_string()))?;
            }
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

fn abort_hook_error<T>(arithmetic: &impl SlotArithmetic, result: Result<T>) -> Result<T> {
    if result.is_err() { arithmetic.abort_split(); }
    result
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
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let result = self.arithmetic.begin(unit, self.requests);
        abort_hook_error(self.arithmetic, result)
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let result = self.arithmetic.attention(layer, unit);
        abort_hook_error(self.arithmetic, result)
    }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        if self.bindings[layer].is_some() {
            let ticket = self.send_experts(layer, unit)?;
            self.land_experts(layer, unit, ticket)
        } else {
            let result = self.arithmetic.ffn(layer, unit, false);
            abort_hook_error(self.arithmetic, result)
        }
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        let setup = (|| {
            self.arithmetic.router(layer, unit)?;
            let identity = *self.next_identity;
            *self.next_identity = identity.checked_add(1).context("MiniMax packed dispatch identity exhausted")?;
            let identities = dispatch_rows(unit, self.requests, identity, self.version)?;
            let source: &'t RoutedRows<'a> = &self.sources[unit.index];
            let dispatcher = &self.dispatchers[self.bindings[layer].context("MiniMax routed layer has no owner")?.dispatcher];
            dispatcher.begin_guarded(layer, source, identities, || self.arithmetic.abort_split())
        })();
        let ticket = abort_hook_error(self.arithmetic, setup)?;
        // Abort before this draining ticket drops, not after the shared driver
        // unwinds its outstanding-ticket vector.
        let result = self.arithmetic.ffn(layer, unit, true);
        abort_hook_error(self.arithmetic, result)?;
        Ok(ticket)
    }
    fn land_experts(&mut self, _: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        ticket.land_guarded(|| self.arithmetic.abort_split())?;
        let result = self.arithmetic.join(unit);
        abort_hook_error(self.arithmetic, result)
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let result = self.arithmetic.head(unit, self.requests);
        abort_hook_error(self.arithmetic, result)
    }
    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        let result = self.arithmetic.commit(unit, self.requests);
        abort_hook_error(self.arithmetic, result)
    }
}

fn decode_section(captured: bool, bucket: usize, live: usize,
    graph: impl FnOnce(usize) -> Result<()>, eager: impl FnOnce(usize) -> Result<()>) -> Result<()> {
    if captured { graph(bucket) } else { eager(live) }
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
        schedule: &Requests, requests: &mut [ServingRequest<'_>], publish: bool,
        consume: impl FnMut(usize, Range<usize>, DeviceLogits) -> Result<()>) -> Result<()> {
        self.execute_with_taps(storage, cfg, schedule, requests, publish, consume, None)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_taps(&mut self, storage: &Storage<'a>, cfg: &MiniMaxM3Config,
        schedule: &Requests, requests: &mut [ServingRequest<'_>], publish: bool,
        consume: impl FnMut(usize, Range<usize>, DeviceLogits) -> Result<()>,
        taps: Option<&mut super::prefill::LayerTaps>) -> Result<()> {
        ensure!(!self.terminal, "MiniMax packed serving is terminal");
        validate_execution_streams(storage)?;
        ensure!(schedule.capacity == storage.requests.borrow().capacity && schedule.layers == cfg.layers
            && schedule.waves == storage.ranks[0].prefill.len() && self.bindings.len() == cfg.layers,
            "MiniMax packed schedule differs from retained owners");
        let rows = requests.iter().map(|request| {
            ensure!(request.start == request.placement.len && request.heads.start >= request.start
                && request.heads.end <= request.end()? && request.heads.start <= request.heads.end,
                "MiniMax append/head frontier differs from placement");
            Ok(RequestRows { slot: request.placement.slot, start: request.start, rows: request.tokens.len() })
        }).collect::<Result<Vec<_>>>()?;
        let units = schedule.verify_units(&rows)?;
        let mut identities = BTreeSet::new();
        let mut writable = BTreeSet::new();
        for request in requests.iter() {
            ensure!(request.identity != 0 && identities.insert(request.identity), "MiniMax duplicate request identity");
            storage.requests.borrow().ranks[0].bank.validate(request.placement, request.end()?)?;
            for position in request.start..request.end()? {
                ensure!(writable.insert((request.placement.units[position / super::cache::PAGE_ROWS],
                    position % super::cache::PAGE_ROWS)), "MiniMax requests alias writable cache slots");
            }
        }
        let sources = units.iter().map(|unit|
            storage.ranks[0].prefill[unit.slot].borrow().routed_rows(unit.rows.len())).collect::<Result<Vec<_>>>()?;
        let mut arithmetic = NativeSlots { storage, cfg, consume, publish, taps };
        let result = run_prefill_units(&mut ServingHooks { arithmetic: &mut arithmetic, sources: &sources,
            dispatchers: &self.dispatchers, bindings: &self.bindings, requests,
            next_identity: &mut self.next_identity, version: self.version }, &units, schedule.waves);
        if result.is_err() {
            self.terminal = true;
            if let Some(split) = &storage.split {
                crate::shared::decode_graph::fatal_drain(split.cancel(), "MiniMax split prefill cancellation");
            }
            self.dispatchers.clear();
        }
        // Pending ticket drops terminally quiesce exchange before these drains.
        if let Some(split) = &storage.split {
            for gpu in 0..2 {
                if let Err(error) = storage.ranks[gpu].decode.borrow().stream.drain() {
                    self.terminal = true;
                    crate::shared::decode_graph::fatal_drain(split.cancel(), "MiniMax split prefill async fault");
                    self.dispatchers.clear();
                    return result.and(Err(error));
                }
            }
        } else {
            for unit in &storage.ranks[0].prefill {
                unit.borrow().stream.drain_or_abort("MiniMax packed serving submission");
            }
        }
        result
    }

    /// Attention consumes only live cache metadata. Independent FFN/router/join
    /// sections use bounded row keys with synthetic tails outside logical state.
    pub fn decode(&mut self, storage: &Storage<'a>, cfg: &MiniMaxM3Config,
        schedule: &Requests, requests: &[ServingRequest<'_>], graphs: &mut Graphs<'a, Storage<'a>>,
        captured: bool)
        -> Result<DeviceLogits> {
        ensure!(!self.terminal, "MiniMax decode serving is terminal");
        validate_execution_streams(storage)?;
        let rows = schedule.decode_rows(&requests.iter().map(|request| RequestRows {
            slot: request.placement.slot, start: request.start, rows: request.tokens.len(),
        }).collect::<Vec<_>>())?;
        if storage.split.is_some() { return self.decode_split(storage, cfg, schedule, requests, rows); }
        let rank = &storage.ranks[0];
        let graph_rows = if captured { graphs.row_key(rows)? } else { rows };
        graphs.cycle(crate::shared::decode_graph::device::Cycle::Decode);
        let result = (|| {
            let mut state = storage.requests.borrow_mut();
            ensure!(state.capacity == schedule.capacity, "MiniMax decode capacity differs from storage");
            let mut identities = BTreeSet::new();
            for request in requests {
                ensure!(request.start == request.placement.len && request.identity != 0
                    && identities.insert(request.identity), "MiniMax decode identity/frontier differs");
            }
            let placements = requests.iter().map(|request| &*request.placement).collect::<Vec<_>>();
            let cache = &mut state.ranks[0];
            cache.bank.prepare_decode(&mut cache.decode, &placements)?;
            let ids = state.tokens.upload(&requests.iter().map(|request| request.tokens[0]).collect::<Vec<_>>())?;
            {
                let owner = rank.decode.borrow();
                // SAFETY: dedicated retained ids and scratch are ordered on the
                // single decode stream; all supplied token rows are initialized.
                unsafe { Head::embed_device_ids(cfg, state.capacity.decode.rows as usize,
                    &storage.embedding, rows, &owner.stream, ids, owner.dense.producer_destination(),
                    owner.plane("residual")?)?; }
            }
            let source = rank.decode.borrow().routed_rows(rows)?;
            for layer in 0..cfg.layers {
                let cache = &state.ranks[0];
                let request = cache.bank.decode_request(&cache.decode, layer)?;
                // SAFETY: metadata covers private writable rows on this ordered
                // retained stream; the backend validates actual shared-pool refs.
                unsafe { rank.decode.borrow_mut().decode(&rank.kernels, &rank.weights,
                    &rank.layout, layer, request)?; }
                if graph_rows > rows {
                    let owner = rank.decode.borrow();
                    let tail = super::cache::view(owner.plane("residual")?, rows * cfg.hidden * 4,
                        (graph_rows - rows) * cfg.hidden * 4)?;
                    owner.stream.device.run(|| {
                        // SAFETY: padded rows are private admitted scratch, never
                        // cache slots; row-independent sections cannot affect live rows.
                        unsafe { owner.stream.device.library.cuda_zero_bytes_async(tail, tail.bytes, owner.stream.raw) }
                    })?;
                }
                if let Some(binding) = self.bindings[layer] {
                    rank.decode.borrow().router_boundary()?;
                    // SAFETY: the exact row key retains all weights/scratch; host
                    // boundary transitions run outside recording and every replay.
                    decode_section(captured, graph_rows, rows, |section_rows| {
                        // SAFETY: retained exact-row section operands described above.
                        unsafe { graphs.run(Key { rank: 0, layer, rows: section_rows, op: Op::Router }, None,
                            || rank.decode.borrow().router_enqueue(&rank.kernels, &rank.weights, layer, section_rows)) }
                    }, |live_rows| rank.decode.borrow().router_enqueue(&rank.kernels, &rank.weights, layer, live_rows))?;
                    if let Some(usage) = &storage.usage {
                        let owner = rank.decode.borrow();
                        let tally = usage.routes(owner.stream.device.id, layer,
                            cuteafd_ffi::expert_usage::UsagePhase::Decode)?;
                        // SAFETY: router initialized retained route ids; the tally
                        // and this arithmetic owner share the same ordered stream.
                        owner.stream.device.run(|| unsafe { tally.record(owner.plane("route_ids")?.ptr,
                            rows, None, u32::MAX, true, owner.stream.raw) })?;
                    }
                    let identity = self.next_identity;
                    self.next_identity = identity.checked_add(1).context("MiniMax dispatch identity exhausted")?;
                    let dispatch = DispatchRows { request_id: identity, placement_version: self.version,
                        rows: requests.iter().enumerate().map(|(row, request)| ExpertProtocolV2RowDescriptor {
                            row_id: row as u64, source_kind: request.kind, source_request_id: request.identity,
                            token_position: request.start as u64, route_offset: (row * 4) as u32, route_count: 4,
                        }).collect() };
                    let ticket = self.dispatchers[binding.dispatcher].begin(layer, &source, dispatch)?;
                    rank.decode.borrow().ffn_boundary(true)?;
                    // SAFETY: immutable geometry and retained unit owner; routed
                    // staging event wait was enqueued before entering this section.
                    decode_section(captured, graph_rows, rows, |section_rows| {
                        // SAFETY: retained exact-row section operands described above.
                        unsafe { graphs.run(Key { rank: 0, layer, rows: section_rows, op: Op::Ffn }, None,
                            || rank.decode.borrow_mut().ffn_enqueue(&rank.kernels, &rank.weights,
                                &rank.layout, layer, section_rows, true)) }
                    }, |live_rows| rank.decode.borrow_mut().ffn_enqueue(&rank.kernels, &rank.weights,
                        &rank.layout, layer, live_rows, true))?;
                    ticket.land()?;
                    if graph_rows > rows {
                        let owner = rank.decode.borrow();
                        let tail = super::cache::view(owner.plane("routed")?, rows * cfg.hidden * 2,
                            (graph_rows - rows) * cfg.hidden * 2)?;
                        owner.stream.device.run(|| {
                            // SAFETY: dispatch writes only live rows; initialize
                            // graph padding without reading an uninitialized routed tail.
                            unsafe { owner.stream.device.library.cuda_zero_bytes_async(tail, tail.bytes, owner.stream.raw) }
                        })?;
                    }
                    rank.decode.borrow().join_boundary()?;
                    // SAFETY: landed routed output and fixed row key remain live;
                    // joined() is a host transition performed on every replay.
                    decode_section(captured, graph_rows, rows, |section_rows| {
                        // SAFETY: retained landed routed output and exact row key.
                        unsafe { graphs.run(Key { rank: 0, layer, rows: section_rows, op: Op::JoinRouted }, None,
                            || rank.decode.borrow().join_routed_enqueue(&rank.kernels, section_rows)) }
                    }, |live_rows| rank.decode.borrow().join_routed_enqueue(&rank.kernels, live_rows))?;
                    rank.decode.borrow().joined();
                } else {
                    rank.decode.borrow().ffn_boundary(false)?;
                    // SAFETY: row-independent dense FFN with retained fixed pointers.
                    decode_section(captured, graph_rows, rows, |section_rows| {
                        // SAFETY: retained row-independent dense section operands.
                        unsafe { graphs.run(Key { rank: 0, layer, rows: section_rows, op: Op::Ffn }, None,
                            || rank.decode.borrow_mut().ffn_enqueue(&rank.kernels, &rank.weights,
                                &rank.layout, layer, section_rows, false)) }
                    }, |live_rows| rank.decode.borrow_mut().ffn_enqueue(&rank.kernels, &rank.weights,
                        &rank.layout, layer, live_rows, false))?;
                }
            }
            let residual = rank.decode.borrow().plane("residual")?;
            storage.head.project_device(&rank.weights, residual, rows)
        })();
        if result.is_err() {
            self.terminal = true;
            self.dispatchers.clear();
        }
        rank.decode.borrow().stream.drain_or_abort("MiniMax eager decode submission");
        result
    }

    fn decode_split(&mut self, storage: &Storage<'a>, cfg: &MiniMaxM3Config,
        schedule: &Requests, requests: &[ServingRequest<'_>], rows: usize) -> Result<DeviceLogits> {
        let split = storage.split.as_ref().context("MiniMax split owners missing")?;
        let result = (|| {
            let mut state = storage.requests.borrow_mut();
            ensure!(state.capacity == schedule.capacity, "MiniMax split decode capacity differs");
            let mut identities = BTreeSet::new();
            for request in requests {
                ensure!(request.start == request.placement.len && request.identity != 0
                    && identities.insert(request.identity), "MiniMax split decode identity/frontier differs");
            }
            let placements = requests.iter().map(|request| &*request.placement).collect::<Vec<_>>();
            for cache in &mut state.ranks { cache.bank.prepare_decode(&mut cache.decode, &placements)?; }
            let ids = state.tokens.upload(&requests.iter().map(|request| request.tokens[0]).collect::<Vec<_>>())?;
            {
                let owner = storage.ranks[0].decode.borrow();
                // SAFETY: dedicated entry-rank ids and retained decode stream.
                unsafe { Head::embed_device_ids(cfg, state.capacity.decode.rows as usize,
                    &storage.embedding, rows, &owner.stream, ids, owner.dense.producer_destination(),
                    owner.plane("residual")?)?; }
            }
            let source = storage.ranks[0].decode.borrow().routed_rows(rows)?;
            for layer in 0..cfg.layers {
                let gpu = split.owner(layer)?;
                let rank = &storage.ranks[gpu];
                let cache = &state.ranks[gpu];
                let request = cache.bank.decode_request(&cache.decode, layer)?;
                // SAFETY: owner-only KV and dedicated metadata cover private
                // append rows, ordered on this rank's one compute stream.
                unsafe { rank.decode.borrow_mut().decode(&rank.kernels, &rank.weights,
                    &rank.layout, layer, request)?; }
                {
                    let a = storage.ranks[0].decode.borrow();
                    let b = storage.ranks[1].decode.borrow();
                    split.residual([&a, &b], layer, rows)?;
                }
                let ticket = if let Some(binding) = self.bindings[layer] {
                    let rank0 = &storage.ranks[0];
                    let owner = rank0.decode.borrow();
                    owner.router(&rank0.kernels, &rank0.weights, layer, rows)?;
                    if let Some(usage) = &storage.usage {
                        let tally = usage.routes(owner.stream.device.id, layer,
                            cuteafd_ffi::expert_usage::UsagePhase::Decode)?;
                        // SAFETY: the only router/tally producer is logical GPU0.
                        owner.stream.device.run(|| unsafe { tally.record(owner.plane("route_ids")?.ptr,
                            rows, None, u32::MAX, true, owner.stream.raw) })?;
                    }
                    let identity = self.next_identity;
                    self.next_identity = identity.checked_add(1).context("MiniMax split dispatch identity exhausted")?;
                    let dispatch = DispatchRows { request_id: identity, placement_version: self.version,
                        rows: requests.iter().enumerate().map(|(row, request)| ExpertProtocolV2RowDescriptor {
                            row_id: row as u64, source_kind: request.kind, source_request_id: request.identity,
                            token_position: request.start as u64, route_offset: (row * 4) as u32, route_count: 4,
                        }).collect() };
                    Some(self.dispatchers[binding.dispatcher].begin_guarded(layer, &source, dispatch,
                        || crate::shared::decode_graph::fatal_drain(split.publish_abort(), "MiniMax split decode begin before ticket drop"))?)
                } else { None };
                let shared = ticket.is_some();
                {
                    let mut a = storage.ranks[0].decode.borrow_mut();
                    let mut b = storage.ranks[1].decode.borrow_mut();
                    if let Err(error) = split.stage(storage, [&mut a, &mut b], layer, rows, shared) {
                        crate::shared::decode_graph::fatal_drain(split.publish_abort(), "MiniMax split decode stage before ticket drop");
                        return Err(error);
                    }
                }
                if let Some(ticket) = ticket {
                    ticket.land_guarded(|| crate::shared::decode_graph::fatal_drain(split.publish_abort(),
                        "MiniMax split decode land before ticket drop"))?;
                }
                {
                    let a = storage.ranks[0].decode.borrow();
                    let b = storage.ranks[1].decode.borrow();
                    split.meet(storage, [&a, &b], rows, shared)?;
                }
            }
            let rank = &storage.ranks[1];
            let residual = rank.decode.borrow().plane("residual")?;
            storage.head.project_device(&rank.weights, residual, rows)
        })();
        if result.is_err() {
            self.terminal = true;
            crate::shared::decode_graph::fatal_drain(split.cancel(), "MiniMax split decode cancellation");
            self.dispatchers.clear();
        }
        for gpu in 0..2 {
            if let Err(error) = storage.ranks[gpu].decode.borrow().stream.drain() {
                self.terminal = true;
                crate::shared::decode_graph::fatal_drain(split.cancel(), "MiniMax split decode async fault");
                self.dispatchers.clear();
                return result.and(Err(error));
            }
        }
        result
    }

    pub fn drain(&self, storage: &Storage<'a>) -> Result<()> {
        for rank in &storage.ranks {
            rank.decode.borrow().stream.drain()?;
            for owner in &rank.prefill { owner.borrow().stream.drain()?; }
        }
        storage.prefix.drain().map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::FfnRoute;
    use std::collections::BTreeMap;

    #[test]
    fn ungraphed_sections_never_call_the_bank_and_keep_live_rows() -> Result<()> {
        use std::cell::Cell;
        for bucket in [1, 4, 16, 64] {
            for live in 1..=bucket {
                let graph_calls = Cell::new(0);
                let eager_rows = Cell::new(0);
                for _ in 0..4 {
                    decode_section(false, bucket, live, |_| {
                        graph_calls.set(graph_calls.get() + 1); Ok(())
                    }, |rows| { eager_rows.set(rows); Ok(()) })?;
                }
                assert_eq!(graph_calls.get(), 0);
                assert_eq!(eager_rows.get(), live);
                decode_section(true, bucket, live, |rows| {
                    assert_eq!(rows, bucket); graph_calls.set(graph_calls.get() + 1); Ok(())
                }, |_| panic!("captured path used eager closure"))?;
                assert_eq!(graph_calls.get(), 1);
            }
        }
        let backend = include_str!("backend.rs");
        let ungraphed = backend.split_once("    fn verify_device_ungraphed(").unwrap().1
            .split_once("    fn verify_bucket_rows(").unwrap().0;
        assert!(ungraphed.contains("self.decode(sequences, pool, true, false)"));
        assert!(ungraphed.contains("self.decode(sequences, pool, false, false)"));
        let decode = include_str!("engine.rs").split_once("    pub fn decode(").unwrap().1
            .split_once("    pub fn drain(").unwrap().0;
        assert_eq!(decode.matches("decode_section(captured,").count(), 4);
        assert!(decode.contains("if captured { graphs.row_key(rows)? } else { rows }"));
        Ok(())
    }

    use crate::shared::{experts::rtx::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard},
        memory::device::{Allocation, Device, Stream}};
    use cuteafd_core::memory_layout::Tag;
    use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, native_library_lifetime_fixture::Fixture};
    use cuteafd_loader::families::minimax_m3::execution::{MsaCache, MsaRope};
    use std::{cell::{Cell, RefCell}, ffi::c_void, rc::Rc};

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
            self.slots[unit.slot].values = unit.segments.iter().map(|segment|
                requests[segment.sequence].tokens_at(segment.tokens.clone())).collect::<Result<Vec<_>>>()?
                .into_iter().flatten().map(|&token| token as u8).collect();
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
        fn head(&mut self, unit: &Unit, _: &[ServingRequest<'_>]) -> Result<()> {
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
            ServingRequest { identity: 19 + index as u64, kind: ExpertV2SourceKind::Prefill,
                start: placement.len, tokens: &tokens[placement.len..], heads: placement.len..tokens.len(), placement }).collect::<Vec<_>>();
        let schedule = Requests::new(capacity, 320, 2, 3)?;
        let units = schedule.verify_units(&requests.iter().map(|request| RequestRows {
            slot: request.placement.slot, start: request.placement.len,
            rows: request.tokens.len() }).collect::<Vec<_>>())?;
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
    fn actual_split_transport_cpu_model_orders_partials_routed_sum_and_fault_abort() -> Result<()> {
        use super::super::{owners::{self, LoadedRank, Owners}, weights::TargetWeights, cache::RequestStreams};
        use cuteafd_loader::{families::minimax_m3::execution::WirePrecision,
            placement::{families::minimax_m3 as canonical, planner::{Context as ModelContext, Operation}},
            families::minimax_m3::weight_layout::SplitFfnWeights};
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.extent_only(&library)?;
        fixture.peer_ordering_model(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        for id in 0..2 { fixture.configure_memory(&library, id, 96 << 30)?; }
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let (checkpoint, cfg) = owners::tests::snapshot(dir.path())?;
        for wire in [WirePrecision::Bf16, WirePrecision::F32] {
            let mut inputs = owners::tests::inputs(&checkpoint, &cfg, &[0, 1], 1)?;
            inputs.split_wire = Some(wire);
            let request = canonical::request(ModelContext::automatic(inputs.context.maximum), Some(129 * 64),
                cuteafd_loader::placement::Onboard::Layers(0), inputs.prefill,
                Operation::Serving, None, Default::default());
            let admitted = inputs.admit(request)?;
            fixture.register_programs(&library, inputs.manifest["programs"].as_array().unwrap())?;
            let manifest = dir.path().join("PROGRAMS.json");
            std::fs::write(&manifest, serde_json::to_vec(&*inputs.manifest)?)?;
            let programs = library.programs()?.with_manifest(&manifest)?;
            let layouts = SplitFfnWeights::plan(&checkpoint, &cfg,
                &inputs.owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>(),
                cuteafd_loader::placement::planner::EmbeddingHome::Device(0))?;
            let mut loaded = Vec::new();
            let mut embedding = None;
            for gpu in 0..2 {
                let device = Device { library: &library, id: gpu as i32 };
                let (weights, records) = crate::shared::experts::rtx::dense::recording(|| {
                    if gpu == 0 {
                        let (weights, table) = TargetWeights::load_split_ffn_with_embedding(device, &checkpoint,
                            &layouts[gpu], crate::shared::token_io::EmbedPlacement::Gpu)?;
                        embedding = Some(table);
                        Ok(weights)
                    } else { TargetWeights::load_split_ffn(device, &checkpoint, &layouts[gpu]) }
                })?;
                drop(records);
                let decode = Rc::new(Stream::new(device)?);
                loaded.push(LoadedRank { weights, streams: RequestStreams {
                    decode: Rc::clone(&decode), prefill: vec![decode; inputs.prefill.waves as usize] } });
            }
            let owners = Owners::new(&inputs, &admitted, &programs, loaded, embedding.unwrap())?;
            let storage = &owners.storage;
            let split = storage.split.as_ref().unwrap();
            {
                let a = storage.ranks[0].decode.borrow();
                let b = storage.ranks[1].decode.borrow();
                let units = [&*a, &*b];
                let local = a.split_plane("local")?;
                let receive = b.split_plane("receive")?;
                let wrong_copy = a.stream.device.run(|| {
                    // SAFETY: buffers/stream are live; deliberately wrong current
                    // device for the destination-stream-only peer-copy API.
                    unsafe { library.copy_peer_async(receive, local, 16, a.stream.raw) }
                });
                assert!(wrong_copy.is_err(), "CPU model must reject source-stream copy_peer_async");
                let first = fixture.events()?.len();
                split.partials(units, 1)?;
                let trace = fixture.events()?[first..].chars().filter(|ch| "pw".contains(*ch)).collect::<String>();
                assert_eq!(trace, "ppww", "both actual sends must precede either wait");
                a.stream.drain()?;
                b.stream.drain()?;
                // Route flags are absent at count0, but final routed output is
                // always retained on both ranks, including Spark-backed layers.
                assert!(b.plane("route_ids").is_err());
                let first = fixture.events()?.len();
                split.routed(units, 1)?;
                assert_eq!(fixture.events()?[first..].chars().filter(|ch| "pw".contains(*ch)).collect::<String>(), "pw");
                a.stream.drain()?;
                b.stream.drain()?;
                fixture.peer_stream_fault(&library, a.stream.raw)?;
                split.partials(units, 1)?;
                assert!(a.stream.drain().is_err(), "asynchronous producer fault must surface");
            }
            let first = fixture.events()?.len();
            split.cancel()?;
            let trace = &fixture.events()?[first..];
            assert!(trace.contains('a'), "independent abort publication missing");
            assert!(trace.matches('s').count() >= 4, "both abort and compute ranks must drain");
            assert!(split.owner(0).is_err(), "terminal split transport must reject reuse");
            fixture.peer_stream_fault(&library, std::ptr::null_mut())?;
            drop(owners);
        }
        Ok(())
    }

    #[test]
    fn split_routed_broadcast_uses_existing_payload_and_disjoint_charged_controls() -> Result<()> {
        use cuteafd_loader::families::minimax_m3::execution::{SplitLayout, WirePrecision};
        for rows in [1, 16, 64, 128] {
            for wire in [WirePrecision::Bf16, WirePrecision::F32] {
                let layout = SplitLayout::new(rows, wire)?;
                let buffers = layout.buffers().into_iter().collect::<BTreeMap<_, _>>();
                let control = buffers["control"] as usize;
                let mut ranges = Vec::new();
                let mut names = BTreeSet::new();
                for &(name, offset, bytes) in &SPLIT_CONTROL_CHANNELS {
                    assert!(names.insert(name));
                    let end = offset.checked_add(bytes).expect("control extent overflow");
                    assert_eq!(offset % 16, 0);
                    assert_eq!(bytes, 16);
                    assert!(end <= control);
                    assert!(ranges.iter().all(|&(a, b)| end <= a || b <= offset));
                    ranges.push((offset, end));
                }
                assert_eq!(names, BTreeSet::from(["partial", "routes", "routed-sum", "release"]));
                assert_eq!(buffers["abort"], 256);
                assert_eq!(layout.events(), 2, "no extra transport event allocation");
            }
        }
        Ok(())
    }

    #[test]
    fn hook_error_publishes_split_abort_before_draining_ticket_drop() -> Result<()> {
        struct FaultSlots<'a> { abort: Allocation<'a>, stream: Rc<Stream<'a>> }
        impl SlotArithmetic for FaultSlots<'_> {
            fn begin(&mut self, _: &Unit, _: &[ServingRequest<'_>]) -> Result<()> { Ok(()) }
            fn attention(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
            fn router(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
            fn ffn(&mut self, _: usize, _: &Unit, _: bool) -> Result<()> { anyhow::bail!("injected mid-unit hook error") }
            fn join(&mut self, _: &Unit) -> Result<()> { Ok(()) }
            fn head(&mut self, _: &Unit, _: &[ServingRequest<'_>]) -> Result<()> { Ok(()) }
            fn commit(&mut self, _: &Unit, _: &mut [ServingRequest<'_>]) -> Result<()> { Ok(()) }
            fn abort_split(&self) {
                self.stream.device.run(|| {
                    // SAFETY: test-owned word and independent abort stream.
                    unsafe { self.stream.device.library.peer_abort_publish(self.abort.buffer.ptr.cast(), self.stream.raw)?; }
                    Ok(())
                }).unwrap();
                self.stream.drain().unwrap();
            }
        }
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.peer_ordering_model(&library)?;
        let device = Device { library: &library, id: 0 };
        let stream = Rc::new(Stream::new(device)?);
        let slot = ModelSlot::new(device, 1)?;
        let source = slot.source(Rc::clone(&stream), 1)?;
        let calls = Rc::new(Cell::new(0));
        let dispatcher = Dispatcher::whole(Box::new(ModelExperts { output: Allocation::new(device,
            6144 * 2, Tag::workspace("minimax/test/fault-output"))?, calls }))?;
        let mut arithmetic = FaultSlots { abort: Allocation::new(device, 256,
            Tag::workspace("minimax/test/fault-abort"))?, stream: Rc::new(Stream::new(device)?) };
        device.run(|| library.cuda_zero_bytes(arithmetic.abort.buffer, 256))?;
        fixture.peer_stream_fault(&library, stream.raw)?;
        let mut placement = Placement { units: vec![0], slot: 0, len: 0 };
        let mut requests = [ServingRequest { identity: 1, kind: ExpertV2SourceKind::Prefill,
            placement: &mut placement, start: 0, tokens: &[7], heads: 0..1 }];
        let unit = Unit { index: 0, sequence: 0, segments: vec![UnitSegment { sequence: 0,
            rows: 0..1, tokens: 0..1, prompt_end: false, snapshot: false, head: true }],
            rows: 0..1, layers: 1..2, slot: 0, barrier: false };
        let mut identity = 1;
        let first = fixture.events()?.len();
        let result = run_prefill_units(&mut ServingHooks { arithmetic: &mut arithmetic,
            sources: &[source], dispatchers: &[dispatcher], bindings: &[None,
                Some(RoutedBinding { dispatcher: 0, route: FfnRoute::Spark })],
            requests: &mut requests, next_identity: &mut identity, version: 23 }, &[unit], 1);
        assert!(result.unwrap_err().to_string().contains("injected mid-unit"));
        let trace = fixture.events()?;
        let trace = &trace[first..];
        assert!(trace.find('a').unwrap() < trace.find('s').unwrap(), "abort before ticket drain: {trace}");
        assert!(!trace.contains('w'), "no unmatched peer wait may be invented: {trace}");
        fixture.peer_stream_fault(&library, std::ptr::null_mut())?;
        Ok(())
    }

    #[test]
    fn split_pair_sends_both_partials_before_either_destination_waits() -> Result<()> {
        let trace = RefCell::new(Vec::new());
        send_split_pair(|rank| {
            trace.borrow_mut().push(("send", rank));
            Ok(rank)
        }, |ticket| {
            assert_eq!(trace.borrow().iter().filter(|(op, _)| *op == "send").count(), 2);
            trace.borrow_mut().push(("land", ticket));
            Ok(())
        })?;
        assert_eq!(*trace.borrow(), [("send", 0), ("send", 1), ("land", 0), ("land", 1)]);
        Ok(())
    }

    #[test]
    fn split_pair_failed_send_never_queues_an_unmatched_peer_wait() {
        for failure in 0..2 {
            let trace = RefCell::new(Vec::new());
            let result = send_split_pair(|rank| {
                trace.borrow_mut().push(("send", rank));
                ensure!(rank != failure, "injected send failure");
                Ok(rank)
            }, |ticket| {
                trace.borrow_mut().push(("land", ticket));
                Ok(())
            });
            assert!(result.is_err());
            assert!(!trace.borrow().iter().any(|(op, _)| *op == "land"));
        }
    }

    #[test]
    fn split_pair_drains_both_ranks_even_if_the_first_drain_fails() {
        let drained = RefCell::new(Vec::new());
        let result = drain_split_pair(|rank| {
            drained.borrow_mut().push(rank);
            ensure!(rank != 0, "injected first-rank drain failure");
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(*drained.borrow(), [0, 1]);
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
