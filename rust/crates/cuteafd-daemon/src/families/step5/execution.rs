//! Explicit one-lane native binding for shared-pipeline qualification.
use super::{cache::CacheRank, csa::CsaLayer, ffn::{DenseFfn, DensePlan},
    rope::TargetTables, routed::{RoutedFfn, RoutedPlan}, swa::SwaLayer,
    transaction::{Staging, Step}};
use crate::shared::{experts::{rtx::pair::{self, SubmissionScope}, staging::RouteStaging},
    memory::{HostAllocation, device::{Allocation, Device, DeviceOwner}},
    peer_split::{PeerExchange, RankDevice, attention::{self, AttentionStateOwners,
        HeadKernels, HeadMerge, HeadPrecision, HeadProjection, HeadSequence,
        LaneOrder, LayerKernels, PipelineKernels}, hop::HopLink}, spark_intake::SparkLane};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::Programs};
use cuteafd_hostcache::copy::CopyEngine;
use cuteafd_loader::{placement::{Hop, HopPoint, LayerMode}, plan::{checkpoint::Checkpoint,
    families::step5::{coordinator::{CoordinatorContract, RankWeightLayout}, ownership::ExecutionOwnership}}};
use cuteafd_transport::ExpertProtocolV2Request;
use std::{cell::RefCell, ffi::c_void, rc::Rc, time::Duration};

const H: usize = 4096;

enum Attention<'a> { Sliding(SwaLayer<'a>), Sparse(CsaLayer<'a>) }
enum Ffn<'a> { Dense(DenseFfn<'a>), Routed(RoutedFfn<'a>) }
struct Layer<'a> { attention: Attention<'a>, ffn: Ffn<'a> }
impl Layer<'_> {
    fn input(&self) -> CuteafdDeviceBuffer {
        match &self.attention { Attention::Sliding(l) => l.input(), Attention::Sparse(l) => l.input() }
    }
    fn post(&self) -> CuteafdDeviceBuffer {
        match &self.attention { Attention::Sliding(l) => l.attention_output(), Attention::Sparse(l) => l.attention_output() }
    }
    fn output(&self) -> CuteafdDeviceBuffer {
        match &self.ffn { Ffn::Dense(l) => l.output(), Ffn::Routed(l) => l.output() }
    }
    fn ffn_input(&self) -> CuteafdDeviceBuffer {
        match &self.ffn { Ffn::Dense(l) => l.input(), Ffn::Routed(l) => l.input() }
    }
}

/// Mutable native objects are deliberately admitted for ONE lane only. Sharing
/// immutable weights with separate lane workspaces is a prerequisite to serving.
pub(super) struct Execution<'e, 'a> {
    ownership: ExecutionOwnership,
    ranks: [Vec<Option<Layer<'a>>>; 2],
    caches: [&'e RefCell<CacheRank<'a>>; 2],
    stages: [&'e Staging<'a>; 2],
    peer: PeerExchange<'a>,
    hops: HopLink<'a>,
    sequence: RefCell<HeadSequence>,
    lanes: [RefCell<Option<DeviceOwner<'a, SparkLane<'a>>>>; 2],
    pinned: [RefCell<HostAllocation<'a>>; 2],
    rows: usize,
    inventory: Vec<Inventory>,
}

struct Inventory { weights: Vec<u8>, kv: Vec<u8>, indexer: Option<Vec<u8>>, compressor: Option<Vec<u8>> }

impl<'e, 'a> Execution<'e, 'a> {
    pub fn required_bytes(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, rank: usize, rows: usize) -> Result<u64> {
        Self::required_bytes_count(programs, checkpoint, ownership, rank, rows, 92)
    }

    pub(super) fn required_bytes_count(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, rank: usize, rows: usize, count: usize) -> Result<u64> {
        ensure!(matches!(count, 1 | 92), "Step-5 diagnostic layer count");
        ensure!(ownership.ranks() == 2 && ownership.execution().spec().lanes == 1,
            "Step-5 shared golden explicitly admits exactly one lane and two ranks");
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let layout = RankWeightLayout::for_execution(checkpoint, rank, ownership)?;
        let mut bytes = layout.operands.iter().filter(|w| (0..count).any(|layer| w.name.starts_with(&format!("model.layers.{layer}."))))
            .try_fold(contract.rope_tables.allocation_bytes, |n, w| n.checked_add(w.allocation_bytes))
            .context("Step-5 native rank weight overflow")?;
        for layer in 0..count {
            if ownership.layer(layer, rank)?.query_heads == 0 { continue; }
            let attention = if contract.layers[layer].sliding.is_some() {
                SwaLayer::workspace_for_heads(rows, ownership.layer(layer, rank)?.query_heads)? }
                else { CsaLayer::workspace_bytes(rows)? };
            let ffn = if contract.layers[layer].dense_activation.is_some() {
                DensePlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?.workspace_bytes()
            } else {
                RoutedPlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?.workspace_bytes()
            };
            bytes = bytes.checked_add(attention).and_then(|n| n.checked_add(ffn))
                .context("Step-5 native rank workspace overflow")?;
        }
        Ok(bytes)
    }

    pub fn load(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: ExecutionOwnership, caches: [&'e RefCell<CacheRank<'a>>; 2],
        stages: [&'e Staging<'a>; 2], rows: usize, available: [u64; 2],
        reservations: [u64; 2], lanes: [Option<DeviceOwner<'a, SparkLane<'a>>>; 2]) -> Result<Self> {
        Self::load_count(programs, checkpoint, ownership, caches, stages, rows, available,
            reservations, lanes, 92)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_count(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: ExecutionOwnership, caches: [&'e RefCell<CacheRank<'a>>; 2],
        stages: [&'e Staging<'a>; 2], rows: usize, available: [u64; 2],
        reservations: [u64; 2], lanes: [Option<DeviceOwner<'a, SparkLane<'a>>>; 2],
        count: usize) -> Result<Self> {
        ensure!(matches!(count, 1 | 92), "Step-5 execution count must be layer-zero diagnostic or full target");
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        for rank in 0..2 {
            let required = Self::required_bytes_count(programs, checkpoint, &ownership, rank, rows, count)?;
            ensure!(required.checked_add(reservations[rank]).is_some_and(|n| n <= available[rank]),
                "Step-5 native rank {rank} exceeds preallocation admission");
            ensure!(rows as u64 <= ownership.execution().spec().rows,
                "Step-5 live rows exceed execution plan");
            let cache = caches[rank].borrow();
            ensure!(cache.rank() == rank && cache.layout().ranks.len() == 2,
                "Step-5 cache rank inventory");
            ensure!(lanes[rank].is_some() == (0..count).any(|l|
                ownership.layer(l, rank).is_ok_and(|local| local.routed_wave)),
                "Step-5 routed-wave transport inventory");
        }
        let devices = [caches[0].borrow().stream.device, caches[1].borrow().stream.device];
        ensure!(devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "Step-5 rank devices/library differ");
        let native_ranks = std::array::from_fn(|rank| RankDevice { device: devices[rank].id,
            stream: caches[rank].borrow().stream.raw });
        let peer = devices[0].run(|| PeerExchange::new_abortable(devices[0].library, native_ranks, 8, rows * H * 4))?;
        let mut hops = devices[0].run(|| HopLink::new(devices[0].library, native_ranks,
            ownership.execution().hops(), ownership.execution().spec()))?;
        hops.attach_abort([peer.abort_word(0)?.context("Step-5 lead abort")?,
            peer.abort_word(1)?.context("Step-5 peer abort")?])?;
        hops.validate_binding(ownership.execution(), &peer)?;
        let mut ranks: [Vec<Option<Layer<'a>>>; 2] = [Vec::new(), Vec::new()];
        for rank in 0..2 {
            let device = devices[rank];
            let rope = Rc::new(TargetTables::load(device, programs, &contract, "step5_", available[rank], &[])?);
            for layer in 0..count {
                if ownership.layer(layer, rank)?.query_heads == 0 { ranks[rank].push(None); continue; }
                let attention = if contract.layers[layer].sliding.is_some() {
                    Attention::Sliding(SwaLayer::for_execution(device, programs, checkpoint, layer,
                        rows, Some(rope.clone()), available[rank], &[], rank, &ownership)?)
                } else {
                    Attention::Sparse(CsaLayer::for_execution(device, programs, checkpoint, layer,
                        rows, Some(rope.clone()), available[rank], &[], rank, &ownership)?)
                };
                let ffn = if contract.layers[layer].dense_activation.is_some() {
                    let plan = DensePlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, &ownership)?;
                    Ffn::Dense(DenseFfn::load_plan(device, programs, checkpoint, plan)?)
                } else {
                    let plan = RoutedPlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, &ownership)?;
                    Ffn::Routed(RoutedFfn::load(device, programs, checkpoint, layer, plan)?)
                };
                ranks[rank].push(Some(Layer { attention, ffn }));
            }
        }
        let bytes = RouteStaging::new(rows, 8, H * 2)?.bytes;
        let pinned = [RefCell::new(HostAllocation::new(devices[0].library, bytes)?),
            RefCell::new(HostAllocation::new(devices[1].library, bytes)?)];
        let mut inventory = Vec::new();
        for layer in 0..count {
            let mut state = Inventory { weights: Vec::new(), kv: Vec::new(),
                indexer: contract.layers[layer].sliding.is_none().then(Vec::new),
                compressor: contract.layers[layer].sliding.is_none().then(Vec::new) };
            for rank in 0..2 {
                if let Some(local) = &ranks[rank][layer] {
                    match &local.attention {
                        Attention::Sliding(owner) => owner.validate_live_weights(devices[rank])?,
                        Attention::Sparse(owner) => owner.validate_live_weights(devices[rank])?,
                    }
                    match &local.ffn {
                        Ffn::Dense(owner) => owner.validate_live_weights(devices[rank])?,
                        Ffn::Routed(owner) => owner.validate_live_weights(devices[rank])?,
                    }
                    state.weights.push(rank as u8);
                }
                let cache = caches[rank].borrow();
                let layout = &cache.layout().ranks[rank];
                if let Some(ring) = layout.sliding.iter().find(|ring| ring.layer == layer && !ring.mtp) {
                    let key = cache.request_component(ring.key, 0)?;
                    let value = cache.request_component(ring.value, 0)?;
                    ensure!(!key.ptr.is_null() && !value.ptr.is_null() && key.bytes > 0 && value.bytes > 0,
                        "Step-5 live SWA state missing");
                    state.kv.push(rank as u8);
                }
                if let Some(region) = layout.regions.iter().find(|region| region.layer == layer) {
                    let (key, value) = cache.csa_planes(layer)?;
                    ensure!(!key.ptr.is_null() && !value.ptr.is_null() && key.bytes > 0 && value.bytes > 0,
                        "Step-5 live CSA pages missing");
                    state.kv.push(rank as u8);
                    let summary = cache.request_component(region.summaries.context("Step-5 CSA summary owner")?, 0)?;
                    ensure!(!summary.ptr.is_null() && summary.bytes > 0, "Step-5 live indexer state missing");
                    state.indexer.as_mut().context("Step-5 unexpected indexer")?.push(rank as u8);
                    ensure!(stages[rank].owns_compressor(&cache, layer), "Step-5 live compressor missing");
                    state.compressor.as_mut().context("Step-5 unexpected compressor")?.push(rank as u8);
                }
            }
            for owners in [Some(state.weights.as_slice()), Some(state.kv.as_slice()),
                state.indexer.as_deref(), state.compressor.as_deref()].into_iter().flatten() {
                ownership.execution().validate_state_owners(layer, owners)?;
            }
            inventory.push(state);
        }
        let execution = Self { ownership, ranks, caches, stages, peer, hops,
            sequence: RefCell::new(HeadSequence::new(0)?), lanes: lanes.map(RefCell::new), pinned, rows, inventory };
        if count == 92 { execution.validate_replications()?; }
        Ok(execution)
    }

    fn layer(&self, rank: usize, layer: usize) -> Result<&Layer<'a>> {
        self.ranks[rank].get(layer).and_then(Option::as_ref).context("Step-5 missing live layer owner")
    }
    pub fn output(&self, layer: usize) -> Result<(Device<'a>, CuteafdDeviceBuffer)> {
        let rank = match self.ownership.execution().mode(layer)? {
            LayerMode::Whole { gpu, .. } => gpu as usize, LayerMode::HeadSplit => 0,
            LayerMode::ContextSplit => anyhow::bail!("Step-5 context output unsupported"),
        };
        Ok((self.caches[rank].borrow().stream.device, self.layer(rank, layer)?.output()))
    }
    pub(super) fn diagnostic_layer0(&self, rank: usize) -> Result<(&SwaLayer<'a>, &DenseFfn<'a>)> {
        let layer = self.layer(rank, 0)?;
        let (Attention::Sliding(attention), Ffn::Dense(ffn)) = (&layer.attention, &layer.ffn)
            else { anyhow::bail!("Step-5 layer zero diagnostic ownership"); };
        Ok((attention, ffn))
    }

    pub(super) fn enqueue_layer0<'s, E: CopyEngine>(&'s self, step: &mut Step<'s, 'a, E>,
        input: &'s Allocation<'a>, directory: &'s std::path::Path) -> Result<()> {
        ensure!(self.ranks.iter().all(|layers| layers.len() == 1), "layer-zero diagnostic must load only layer zero");
        step.validate_binding(&self.caches, &self.stages, self.rows)?;
        pair::submit_scoped(|| self.peer.require_live(), || self.shutdown(), |scope| {
            step.in_submission(scope, |step| {
                step.compose(0, input, |_, _| Ok(()))?;
                attention::replicate_terminal(&self.peer, input.buffer, self.layer(1, 0)?.input(), self.rows * H * 4)?;
                let mut binding = Bindings { execution: self, step, scope,
                    current: [None, Some(self.layer(1, 0)?.input())], input: input.buffer,
                    request: super::golden::request, layer: 0, diagnostic: Some(directory) };
                binding.heads(0)?;
                binding.ffn(0, None)?;
                for cache in &self.caches { cache.borrow().stream.drain()?; }
                Ok(())
            })
        })
    }

    fn validate_replications(&self) -> Result<()> {
        let mut copies = self.ownership.routed_replications(0)?;
        for (layer, copy) in copies.iter_mut().enumerate() {
            if let Some(copy) = copy {
                let Ffn::Routed(source) = &self.layer(copy.source as usize, layer)?.ffn else { anyhow::bail!("Step-5 routed source executor"); };
                let Ffn::Routed(destination) = &self.layer(1 - copy.source as usize, layer)?.ffn else { anyhow::bail!("Step-5 routed receive executor"); };
                let (source, destination) = (source.routed_output(), destination.routed_output());
                ensure!(!source.ptr.is_null() && !destination.ptr.is_null()
                    && source.bytes >= self.rows * H * 2 && destination.bytes >= self.rows * H * 2
                    && source.device_id == self.peer.device(0) && destination.device_id == self.peer.device(1),
                    "Step-5 routed replication live storage/device");
                copy.storage = destination.ptr as usize;
            }
        }
        self.ownership.execution().validate_routed_replications(&copies)?;
        Ok(())
    }
    fn shutdown(&self) -> Result<()> {
        if self.peer.terminal_state() == crate::shared::peer_split::TerminalState::Drained { return Ok(()); }
        let library = self.caches[0].borrow().stream.device.library;
        let result = pair::teardown(&mut pair::Teardown {
            abort: || self.peer.publish_abort(), ranks: || self.peer.drain_compute(),
            links: || { for lane in &self.lanes { if let Some(lane) = lane.borrow_mut().as_mut() {
                let device = lane.device;
                device.run(|| lane.get_mut().terminal_quiesce())?;
            }} Ok(()) },
            copies: || { for lane in &self.lanes { if let Some(lane) = lane.borrow().as_ref() {
                lane.device.run(|| lane.intake.terminal_drain_copies())?;
            }}
                for lane in &self.lanes { if let Some(lane) = lane.borrow_mut().as_mut() {
                    let device = lane.device;
                    device.run(|| lane.get_mut().terminal_release())?;
                }}
                Ok(()) },
            graphs: || Ok(()),
            finish: |drained| {
                self.peer.finish_terminal(drained);
                if !drained { library.quarantine_module_after_failed_drain(); }
            },
        });
        // This qualification owner has no quarantine for borrowed cache/staging
        // streams. A failed teardown must not return and release their storage.
        crate::shared::decode_graph::fatal_drain(result, "Step-5 pipeline terminal owners");
        Ok(())
    }
    pub fn enqueue<'s, E: CopyEngine>(&'s self, step: &mut Step<'s, 'a, E>,
        input: &'s Allocation<'a>, request: Request) -> Result<()> {
        step.validate_binding(&self.caches, &self.stages, self.rows)?;
        self.validate_replications()?;
        ensure!(!input.buffer.ptr.is_null() && input.buffer.device_id == self.peer.device(self.ownership.execution().spec().entry_gpu as usize)
            && input.buffer.bytes >= self.rows * H * 4, "Step-5 live input allocation");
        pair::submit_scoped(|| self.peer.require_live(), || self.shutdown(), |scope| {
            step.in_submission(scope, |step| {
                step.compose(self.ownership.execution().spec().entry_gpu as usize, input, |_, _| Ok(()))?;
                let mut bindings = [Bindings { execution: self, step, scope, current: [None, None], input: input.buffer, request, layer: 0, diagnostic: None }];
                attention::layer_pipeline(scope, self.ownership.execution(), &self.peer, &self.hops,
                    LaneOrder::Independent, &[self.rows], &mut bindings)?;
                for cache in &self.caches { cache.borrow().stream.drain()?; }
                Ok(())
            })
        })
    }
}
impl Drop for Execution<'_, '_> {
    fn drop(&mut self) { if let Err(error) = self.shutdown() {
        tracing::error!(%error, "Step-5 native pipeline teardown retained queued owners");
    }}
}

type Request = fn(usize, usize, Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, bytes::Bytes) -> Result<ExpertProtocolV2Request>;
struct Bindings<'s, 'b, 'e, 'a, 'g, E: CopyEngine> {
    execution: &'s Execution<'e, 'a>, step: &'b mut Step<'s, 'a, E>,
    scope: &'g SubmissionScope<'g>, current: [Option<CuteafdDeviceBuffer>; 2],
    input: CuteafdDeviceBuffer, request: Request, layer: usize,
    diagnostic: Option<&'s std::path::Path>,
}
impl<'s, E: CopyEngine> Bindings<'s, '_, '_, '_, '_, E> {
    fn copy(&mut self, rank: usize, destination: CuteafdDeviceBuffer, source: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(destination.device_id == self.execution.peer.device(rank)
            && source.device_id == destination.device_id && !source.ptr.is_null() && !destination.ptr.is_null()
            && source.bytes >= self.execution.rows * H * 4 && destination.bytes >= self.execution.rows * H * 4,
            "Step-5 residual copy live rank extent");
        if source.ptr == destination.ptr { return Ok(()); }
        self.step.compose(rank, self.execution, |_, meta| meta.stream.device.run(|| {
            // SAFETY: the transaction borrows the complete execution owner; admitted disjoint rank planes survive drainage.
            unsafe { meta.stream.device.library.copy_d2d_async(destination, source, meta.rows * H * 4, meta.stream.raw) }
        }))
    }
    fn attention(&mut self, rank: usize, layer: usize) -> Result<()> {
        let owner = self.execution.layer(rank, layer)?;
        let source = self.current[rank].unwrap_or(self.input);
        self.copy(rank, owner.input(), source)?;
        match &owner.attention {
            Attention::Sliding(owner) => owner.enqueue_attention_step(self.step, rank),
            Attention::Sparse(owner) => owner.enqueue_attention_step(self.step, rank),
        }
    }
}
impl<E: CopyEngine> LayerKernels for Bindings<'_, '_, '_, '_, '_, E> {
    fn pack(&mut self, hop: Hop) -> Result<*const c_void> {
        Ok(self.current[hop.from as usize].unwrap_or(self.input).ptr.cast_const())
    }
    fn entry_destination(&mut self, rank: usize) -> Result<*mut c_void> {
        // Entry hops land directly in the first layer's owned input.
        Ok(self.execution.layer(rank, 0)?.input().ptr)
    }
    fn adopt(&mut self, hop: Hop, residual: *mut c_void) -> Result<()> {
        ensure!(matches!(hop.at, HopPoint::Entry | HopPoint::BeforeLayer(_) | HopPoint::Exit),
            "Step-5 whole native FFN never broadcasts after attention");
        self.current[hop.to as usize] = Some(CuteafdDeviceBuffer { ptr: residual,
            bytes: self.execution.rows * H * 4, device_id: self.execution.peer.device(hop.to as usize), flags: 0 });
        Ok(())
    }
    fn heads(&mut self, layer: usize) -> Result<()> {
        self.layer = layer;
        self.attention(0, layer)?;
        self.attention(1, layer)?;
        if let Some(directory) = self.diagnostic {
            // Capture before merge reuses the normalized input plane.
            for rank in 0..2 {
                self.execution.caches[rank].borrow().stream.drain()?;
                let (owner, _) = self.execution.diagnostic_layer0(rank)?;
                let device = self.execution.caches[rank].borrow().stream.device;
                super::golden::dump_plane(directory, &format!("rank{rank}-attention_input"), device,
                    owner.diagnostic_norm(), self.execution.rows, true)?;
                super::golden::dump_plane(directory, &format!("rank{rank}-o_partial"), device,
                    owner.projection_partial()?, self.execution.rows, false)?;
            }
        }
        let execution = self.execution;
        attention::head_partials(self.scope, &execution.peer, &mut execution.sequence.borrow_mut(),
            execution.rows * H, HeadPrecision::F32, self)
    }
    fn context(&mut self, _layer: usize, _ordinal: usize) -> Result<()> {
        anyhow::bail!("Step-5 context split has no qualified binding")
    }
    fn owned(&mut self, layer: usize, rank: usize) -> Result<()> {
        self.layer = layer;
        self.attention(rank, layer)
    }
    fn ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<()> {
        let ticket = self.begin_ffn(layer, owner)?;
        self.finish_ffn(layer, ticket)
    }
}

impl<E: CopyEngine> PipelineKernels for Bindings<'_, '_, '_, '_, '_, E> {
    type FfnTicket = Option<usize>;
    fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
        let state = self.execution.inventory.get(layer).context("Step-5 live state inventory")?;
        Ok(AttentionStateOwners { weights: &state.weights, kv: &state.kv,
            indexer: state.indexer.as_deref(), compressor: state.compressor.as_deref() })
    }
    fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<Self::FfnTicket> {
        let execution = self.execution;
        let mut wave = None;
        for rank in 0..2 {
            if owner.is_some_and(|owner| owner != rank) { continue; }
            let local = execution.layer(rank, layer)?;
            self.copy(rank, local.ffn_input(), local.post())?;
            match &local.ffn {
                Ffn::Dense(ffn) => self.step.compose(rank, ffn, |ffn, meta| ffn.enqueue_on(meta.rows, meta.stream))?,
                Ffn::Routed(ffn) => {
                    let dispatch = execution.ownership.layer(layer, rank)?.routed_wave;
                    self.step.compose(rank, ffn, |ffn, meta| {
                        if dispatch { ffn.frontend_on(meta.rows, meta.stream) }
                        else { ffn.normalize_on(meta.rows, meta.stream) }
                    })?;
                    if dispatch {
                        ensure!(wave.is_none(), "Step-5 duplicate routed Spark wave");
                        let layout = RouteStaging::new(execution.rows, 8, H * 2)?;
                        let host = execution.pinned[rank].borrow();
                        self.step.compose(rank, ffn, |ffn, meta| {
                            meta.stream.device.run(|| {
                                // SAFETY: pinned arena belongs to this lane; next write follows wait and consumed-event drainage.
                                unsafe { layout.enqueue(meta.stream.device.library, host.buffer, ffn.wave_inputs(), meta.stream.raw) }
                            })?;
                            // Both head publications were queued before either merge. No future
                            // publication is needed for this stream's staging completion.
                            meta.stream.drain()
                        })?;
                        let routes = layout.routes(host.bytes())?;
                        let wire = bytes::Bytes::copy_from_slice(layout.wire(host.bytes())?);
                        let request = self.request;
                        let rows = execution.rows;
                        let mut lane = execution.lanes[rank].borrow_mut();
                        let lane = lane.as_mut().context("Step-5 routed wave lane missing")?;
                        let device = lane.device;
                        device.run(|| lane.get_mut().submit(execution.rows,
                            Box::new(move || request(layer, rows, routes, wire))))?;
                        wave = Some(rank);
                    }
                    self.step.compose(rank, ffn, |ffn, meta| ffn.shared_on(meta.rows, meta.stream))?;
                }
            }
            self.current[rank] = Some(local.output());
        }
        Ok(wave)
    }
    fn finish_ffn(&mut self, layer: usize, ticket: Self::FfnTicket) -> Result<()> {
        let execution = self.execution;
        let Some(rank) = ticket else { return Ok(()); };
        let Ffn::Routed(ffn) = &execution.layer(rank, layer)?.ffn else { anyhow::bail!("Step-5 routed completion owner"); };
        self.step.compose(rank, ffn, |ffn, meta| {
            let mut lane = execution.lanes[rank].borrow_mut();
            let lane = lane.as_mut().context("Step-5 routed completion lane")?;
            let device = lane.device;
            device.run(|| lane.get_mut().wait(meta.rows, Duration::from_secs(120), meta.stream.raw))?;
            meta.stream.device.run(|| {
                // SAFETY: waited intake planes and routed output are disjoint [rows,4096]
                // BF16 owners. Runtime expert geometry is set before library construction.
                unsafe { meta.stream.device.library.v41_compact_reducer()?.reduce_planes(
                    lane.intake.pointers(), lane.world_size() as u32, std::ptr::null(),
                    ffn.routed_output().ptr.cast(), meta.rows as u32, meta.stream.raw) }
            })?;
            meta.stream.device.run(|| lane.intake.consumed(meta.stream.raw))
        })?;
        if execution.ownership.execution().mode(layer)? == LayerMode::HeadSplit {
            let Ffn::Routed(destination) = &execution.layer(1 - rank, layer)?.ffn else { anyhow::bail!("Step-5 routed replica missing"); };
            let binding = execution.ownership.routed_replications(0)?[layer].context("Step-5 routed publication binding")?;
            attention::replicate_result(self.scope, &execution.peer, rank, binding.slot,
                ffn.routed_output(), destination.routed_output(), execution.rows * H * 2)?;
            for at in 0..2 {
                let Ffn::Routed(ffn) = &execution.layer(at, layer)?.ffn else { anyhow::bail!("Step-5 native routed join missing"); };
                self.step.compose(at, ffn, |ffn, meta| ffn.finish_on(meta.rows, meta.stream))?;
            }
        } else {
            self.step.compose(rank, ffn, |ffn, meta| ffn.finish_on(meta.rows, meta.stream))?;
        }
        Ok(())
    }
}

impl<E: CopyEngine> HeadKernels for Bindings<'_, '_, '_, '_, '_, E> {
    fn project(&mut self, rank: usize) -> Result<HeadProjection> {
        // heads() has queued both projections through the cache transaction.
        let layer = self.layer;
        let Attention::Sliding(owner) = &self.execution.layer(rank, layer)?.attention else { anyhow::bail!("Step-5 head split requires SWA"); };
        Ok(HeadProjection { partial: owner.projection_partial()?, rounded: None })
    }
    fn merge(&mut self, rank: usize, values: HeadMerge) -> Result<()> {
        let layer = self.layer;
        let Attention::Sliding(owner) = &self.execution.layer(rank, layer)?.attention else { anyhow::bail!("Step-5 head split requires SWA"); };
        let HeadMerge::F32(values) = values else { anyhow::bail!("Step-5 head merge requires F32"); };
        let capture = self.diagnostic.map(|directory| (directory, rank));
        self.step.compose(rank, owner, |owner, meta| owner.merge_partial_capture_on(meta.rows, values, meta.stream, capture))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::{attention::{LayerProtocol, layer_pipeline_with},
        order::{check_submission_faults, host::{HostSchedule, check_host}}};
    use cuteafd_loader::plan::families::step5::Config;
    use std::cell::Cell;

    struct Trace {
        ownership: ExecutionOwnership,
        host: RefCell<HostSchedule>,
        heads: Cell<usize>,
        waves: Cell<usize>,
    }
    impl Trace {
        fn record(&self, from: usize, exchange: &'static str, slot: usize) {
            let mut host = self.host.borrow_mut();
            host.push(from, exchange, slot, "publication");
            host.wait(1 - from, exchange, slot, "receive");
        }
    }
    impl LayerProtocol for Trace {
        type Ticket = (usize, usize);
        fn require_live(&self) -> Result<()> { Ok(()) }
        fn push_entry(&self, from: usize, _: *const c_void, _: *mut c_void, _: usize) -> Result<()> {
            self.host.borrow_mut().push(from, "entry", 0, "entry push"); Ok(())
        }
        fn wait_entry(&self, to: usize) -> Result<()> {
            self.host.borrow_mut().wait(to, "entry", 0, "entry receive"); Ok(())
        }
        fn send(&self, index: usize, _: usize, _: *const c_void, _: usize) -> Result<Self::Ticket> {
            let hop = self.ownership.execution().hops()[index];
            self.host.borrow_mut().push(hop.from as usize, "hop", index, "hop push");
            Ok((hop.to as usize, index))
        }
        fn land(&self, (to, index): Self::Ticket) -> Result<*mut c_void> {
            self.host.borrow_mut().wait(to, "hop", index, "hop receive");
            Ok(std::ptr::null_mut())
        }
    }
    struct Recorder<'a>(&'a Trace);
    impl LayerKernels for Recorder<'_> {
        fn pack(&mut self, _: Hop) -> Result<*const c_void> { Ok(std::ptr::null()) }
        fn entry_destination(&mut self, _: usize) -> Result<*mut c_void> { Ok(std::ptr::null_mut()) }
        fn adopt(&mut self, _: Hop, _: *mut c_void) -> Result<()> { Ok(()) }
        fn heads(&mut self, _: usize) -> Result<()> {
            let ordinal = self.0.heads.get();
            let mut host = self.0.host.borrow_mut();
            // Mirrors head_partials: both projections/publications precede waits.
            for rank in 0..2 { host.push(rank, "heads", ordinal % 2, "head push"); }
            for rank in 0..2 { host.wait(rank, "heads", ordinal % 2, "native head merge"); }
            self.0.heads.set(ordinal + 1); Ok(())
        }
        fn context(&mut self, _: usize, _: usize) -> Result<()> { anyhow::bail!("no context binding") }
        fn owned(&mut self, _: usize, _: usize) -> Result<()> { Ok(()) }
        fn ffn(&mut self, _: usize, _: Option<usize>) -> Result<()> { unreachable!() }
    }
    impl PipelineKernels for Recorder<'_> {
        type FfnTicket = Option<usize>;
        fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
            let owners: &[u8] = match self.0.ownership.execution().mode(layer)? {
                LayerMode::Whole { gpu: 0, .. } => &[0],
                LayerMode::Whole { gpu: 1, .. } => &[1],
                LayerMode::HeadSplit => &[0, 1],
                _ => unreachable!(),
            };
            let sparse = layer % 4 == 3;
            Ok(AttentionStateOwners { weights: owners, kv: owners,
                indexer: sparse.then_some(owners), compressor: sparse.then_some(owners) })
        }
        fn begin_ffn(&mut self, layer: usize, _: Option<usize>) -> Result<Self::FfnTicket> {
            let mut wave = None;
            for rank in 0..2 {
                if self.0.ownership.layer(layer, rank)?.routed_wave {
                    assert!(wave.is_none(), "duplicate routed dispatch");
                    self.0.host.borrow_mut().sync(rank, "route D2H before Spark dispatch");
                    self.0.waves.set(self.0.waves.get() + 1);
                    wave = Some(rank);
                }
            }
            Ok(wave)
        }
        fn finish_ffn(&mut self, layer: usize, wave: Option<usize>) -> Result<()> {
            if let Some(rank) = wave {
                if let Some(copy) = self.0.ownership.routed_replications(0)?[layer] {
                    assert_eq!(rank, copy.source as usize);
                    self.0.record(rank, "routed", copy.slot);
                }
            }
            Ok(())
        }
    }
    fn config() -> Config {
        Config { norm_eps: 1e-5, hidden: H, vocab: 128896, layers: 92, mtp: 3,
            dense: 13824, experts: 352, intermediate: 1536, shared: 1536, topk: 8,
            heads: 64, groups: 4, dim: 192, window: 512, max_context: 1 << 20,
            full: (0..95).map(|l| l < 92 && l % 4 == 3).collect(),
            moe: (0..95).map(|l| (3..91).contains(&l)).collect(),
            theta: vec![], rotary: vec![], rope_scaling: None, scale_full: false,
            scale_sliding: false, activation: vec![], shared_activation: vec![] }
    }
    #[test]
    fn one_lane_native_plan_orders_host_syncs_hops_and_routed_replies_across_odd_head_passes() -> Result<()> {
        let trace = Trace { ownership: super::super::golden::split_ownership(&config(), 64)?,
            host: RefCell::new(HostSchedule::default()), heads: Cell::new(0), waves: Cell::new(0) };
        for pass in 0..2 {
            pair::submit_scoped(|| Ok(()), || Ok(()), |scope| {
                layer_pipeline_with(scope, trace.ownership.execution(), &trace,
                    LaneOrder::Independent, &[64], &mut [Recorder(&trace)])
            })?;
            assert_eq!(trace.heads.get(), (pass + 1) * 69);
            assert_eq!(trace.waves.get(), (pass + 1) * 88);
            check_host(&trace.host.borrow()).map_err(anyhow::Error::msg)?;
        }
        check_submission_faults(&trace.host.borrow().streams()).map_err(anyhow::Error::msg)?;
        Ok(())
    }
}
