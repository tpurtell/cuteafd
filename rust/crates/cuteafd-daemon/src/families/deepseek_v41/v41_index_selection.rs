//! Bounded, captured index selection and snapshot-checked candidate sharing.
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::families::deepseek_v41::v41_compressor::{IndexBinding, IndexProposal};
use crate::families::deepseek_v41::v41_index_query::IndexQueryOutput;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LayerStaging, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41CandidateBlocks, V41IndexScores, V41IndexTopK,
};
use std::{collections::{BTreeSet}, cell::Cell, ffi::c_void, marker::PhantomData, rc::Rc};
const WIDTH: usize = 16384;
const BLOCKS: usize = WIDTH / 8;
// Index fingerprints include request layouts; retain a bounded recent set.
// Per-layer projection graphs have a separate residency limit.
const MAX_RETAINED_DECODE_GRAPHS: usize = 512;
const DECODE_WIDTH_BUCKET: usize = 512;
const AUDIT_WIDTH: usize = 2048;
const AUDIT_SOURCE_BYTES: usize = 2048 * 68;

struct AsyncAuditRow<'a> {
    host: HostAllocation<'a>,
    pending: Option<(usize, usize, usize, String)>,
}
struct AsyncAudit<'a> {
    metadata: DeviceAllocation<'a>,
    scores: DeviceAllocation<'a>,
    layers: Vec<AsyncAuditRow<'a>>,
    score_bytes: usize,
}
impl<'a> AsyncAudit<'a> {
    fn new(library: &'a NativeLibrary, capacity: usize) -> Result<Option<Self>> {
        if !std::env::var_os("CUTEAFD_V41_INDEX_AUDIT").is_some_and(|v| v == "async") {
            return Ok(None);
        }
        let score_bytes = capacity * AUDIT_WIDTH * 4;
        Ok(Some(Self {
            metadata: DeviceAllocation::new(library, capacity * 48)?,
            scores: DeviceAllocation::new(library, score_bytes)?,
            layers: (0..8).map(|_| Ok(AsyncAuditRow {
                host: HostAllocation::new(library, score_bytes * 2 + capacity * 48
                    + AUDIT_SOURCE_BYTES * 2 + 32768)?, pending: None,
            })).collect::<Result<_>>()?,
            score_bytes,
        }))
    }
}

fn tile_width(max_length: usize, bucket: usize) -> usize {
    max_length.max(1).div_ceil(bucket).saturating_mul(bucket).min(WIDTH)
}

#[derive(Debug, PartialEq, Eq)]
struct ScoreGroup {
    request: usize,
    offset: usize,
    rows: usize,
    binding: [usize; 10],
}

fn extend_score_groups(groups: &mut Vec<ScoreGroup>, request: usize, rows: usize, binding: [usize; 10]) {
    if let Some(last) = groups.last_mut() {
        if last.binding == binding { last.rows += rows; return; }
    }
    let offset = groups.last().map_or(0, |last| last.offset + last.rows);
    groups.push(ScoreGroup { request, offset, rows, binding });
}

pub(crate) struct SelectionRequest<'a> {
    pub proposal: &'a IndexProposal<'a>,
    /// Increasing query token positions, in the query producer's row order.
    pub positions: &'a [u64],
}
pub(crate) struct IndexSelectionOutput<'a> {
    origin: Option<QueryBinding>,
    /// Logical compressed row IDs, sorted ascending, padded with -1. No window offset.
    pub selected: CuteafdDeviceBuffer,
    pub rows: usize,
    layer: usize,
    blocks: Option<CuteafdDeviceBuffer>,
    bindings: &'a [(IndexBinding, u64)],
    _sources: PhantomData<&'a ()>,
}
struct Ready {
    origin: Option<QueryBinding>,
    layer: usize,
    rows: usize,
    bindings: Vec<(IndexBinding, u64)>,
}
impl IndexSelectionOutput<'_> {
    /// Reuse logical row selections on a peer without changing their snapshot
    /// identity. Candidate blocks remain private to the original index producer.
    /// # Safety
    /// Selected contains an exact copy of this output on the consumer device.
    /// Order publication before consumption and retain its allocation until all
    /// consumers drain, including cold graph preparation and cancellation.
    pub unsafe fn peer_attention(&self, selected: CuteafdDeviceBuffer) -> Result<IndexSelectionOutput<'_>> {
        ensure!(selected.device_id >= 0 && selected.device_id != self.selected.device_id
            && !selected.ptr.is_null() && selected.flags == 0
            && self.rows.checked_mul(2048) == Some(selected.bytes),
            "peer selection storage differs");
        Ok(IndexSelectionOutput { origin: self.origin, selected, rows: self.rows,
            layer: self.layer, blocks: None, bindings: self.bindings, _sources: PhantomData })
    }
    #[cfg(test)]
    pub(crate) fn candidate_blocks(&self) -> Option<CuteafdDeviceBuffer> { self.blocks }
    pub fn validate_query(&self, query: QueryBinding) -> Result<()> {
        let origin = self.origin.context("selection has no query origin")?;
        ensure!(
            origin.layer() == self.layer && (query.layer() != self.layer || origin == query),
            "selection query snapshot differs"
        );
        Ok(())
    }

    /// Bind an attention row to the exact source execution and token used by
    /// selection. Intermediate layers reuse their nearest index producer.
    pub fn validate_attention(&self, layer: usize, bindings: &[(IndexBinding, u64)]) -> Result<()> {
        let producer = [2, 8, 14, 20, 24, 28, 32, 36]
            .into_iter()
            .rev()
            .find(|&n| n <= layer);
        ensure!(
            layer < 40
                && producer == Some(self.layer)
                && self.bindings == bindings
                && self.rows == bindings.len(),
            "attention selection layer, snapshot or row order differs"
        );
        Ok(())
    }
}
pub(crate) struct IndexSelectionWave<'a> {
    stream: LoadStream<'a>,
    buffers: Vec<Rc<DeviceAllocation<'a>>>,
    shared_scratch_busy: Option<Rc<Cell<bool>>>,
    /// Per-layer pinned metadata and lengths (see `LayerStaging`).
    staging: LayerStaging<'a>,
    capacity: usize,
    score: V41IndexScores<'a>,
    top: V41IndexTopK<'a>,
    candidates: V41CandidateBlocks<'a>,
    graph: Option<(*mut c_void, Vec<usize>)>,
    graphs: crate::shared::decode_graph::GraphBank<Vec<usize>, crate::shared::decode_graph::GraphOwner<'a, ()>>,
    retain_decode_graphs: bool,
    fixed_captures: BTreeSet<Vec<usize>>,
    fixed_recaptures: usize,
    ready: Option<Ready>,
    pending: Option<Ready>,
    in_flight: bool,
    audit: bool,
    async_audit: Option<AsyncAudit<'a>>,
}
fn slice(mut b: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
impl<'a> IndexSelectionWave<'a> {
    fn sizes(capacity: usize) -> Result<[usize; 13]> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid index selection capacity"
        );
        Ok([
            48,
            8,
            8,
            WIDTH * 8,
            WIDTH * 4,
            BLOCKS * 4,
            BLOCKS * 8,
            4096,
            V41IndexTopK::scratch_bytes(1, WIDTH)?,
            2048,
            16384,
            V41IndexTopK::block_scratch_bytes(1, BLOCKS)?,
            8192,
        ]
        .map(|n| n * capacity))
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(Self::sizes(capacity)?.iter().sum())
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "index selection exceeds budget"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            buffers: Self::sizes(capacity)?
                .into_iter()
                .map(|n| DeviceAllocation::new(library, n).map(Rc::new))
                .collect::<Result<_>>()?,
            shared_scratch_busy: None,
            staging: LayerStaging::new(library, capacity * 56, 40)?,
            capacity,
            score: library.v41_index_scores()?,
            top: library.v41_index_topk()?,
            candidates: library.v41_candidate_blocks()?,
            graph: None, graphs: crate::shared::decode_graph::GraphBank::new(None), retain_decode_graphs: false, fixed_captures: BTreeSet::new(), fixed_recaptures: 0,
            ready: None,
            pending: None,
            in_flight: false,
            audit: std::env::var_os("CUTEAFD_V41_INDEX_AUDIT").is_some_and(|v| v == "1"),
            async_audit: AsyncAudit::new(library, capacity)?,
        })
    }
    /// Only selected IDs and source candidate blocks survive another stage.
    pub fn shared_device_bytes(capacity: usize) -> Result<usize> {
        let sizes = Self::sizes(capacity)?;
        Ok(sizes[9] + sizes[12])
    }
    /// Create a follower in the same lane. Both owners retain independent
    /// outputs/graphs; their temporary scores and top-k workspace are shared.
    /// An outstanding queued selection excludes the other owner until polled
    /// complete or explicitly drained. No cross-lane storage is involved.
    pub fn sharing_scratch(source: &mut Self, budget: usize) -> Result<Self> {
        ensure!(!source.in_flight && source.shared_scratch_busy.is_none(), "index scratch already shared or pending");
        source.stream.require_complete()?;
        let capacity = source.capacity;
        ensure!(Self::shared_device_bytes(capacity)? <= budget, "shared index outputs exceed budget");
        let library = source.stream.library;
        ensure!(library.cuda_get_device()? == source.b(0).device_id, "shared index workspace device differs");
        let sizes = Self::sizes(capacity)?;
        let buffers = source.buffers.iter().enumerate().map(|(i, allocation)| {
            if [9, 12].contains(&i) { DeviceAllocation::new(library, sizes[i]).map(Rc::new) }
            else { Ok(allocation.clone()) }
        }).collect::<Result<Vec<_>>>()?;
        let busy = Rc::new(Cell::new(false));
        let value = Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? }, buffers,
            shared_scratch_busy: Some(busy.clone()), staging: LayerStaging::new(library, capacity * 56, 40)?,
            capacity, score: library.v41_index_scores()?, top: library.v41_index_topk()?,
            candidates: library.v41_candidate_blocks()?, graph: None,
            graphs: crate::shared::decode_graph::GraphBank::new(None), retain_decode_graphs: false, fixed_captures: BTreeSet::new(), fixed_recaptures: 0, ready: None, pending: None, in_flight: false,
            audit: source.audit,
            async_audit: AsyncAudit::new(library, capacity)?,
        };
        source.shared_scratch_busy = Some(busy);
        Ok(value)
    }
    fn b(&self, i: usize) -> CuteafdDeviceBuffer {
        self.buffers[i].buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    pub fn enable_small_graph_shapes(&mut self) {
        if !self.retain_decode_graphs {
            self.retain_decode_graphs = true;
        }
    }
    /// Invalidate published selections without discarding immutable launch shapes.
    pub fn restart(&mut self) -> Result<()> {
        if !self.retain_decode_graphs && super::graph_policy::fixed_binding_limit().is_none() {
            return self.clear_graph();
        }
        ensure!(!self.in_flight, "index selection pending");
        self.stream.require_complete()?;
        self.ready = None;
        Ok(())
    }
    fn select_graph(&mut self, fingerprint: &[usize]) -> Result<bool> {
        // Eight indexed layers share this owner; source identities remain exact.
        let policy = super::graph_policy::fixed_binding_limit().map(|limit| (
            super::graph_policy::captures_shape(fingerprint[1] as u32), limit * 8));
        self.select_graph_with_policy(fingerprint, policy)
    }
    fn select_graph_with_policy(&mut self, fingerprint: &[usize], policy: Option<(bool, usize)>) -> Result<bool> {
        if self.graph.as_ref().is_some_and(|(_, key)| key == fingerprint) { return Ok(true); }
        let fixed_limit = policy.map(|(_, limit)| limit);
        let key = fingerprint.to_vec();
        let cached = self.graphs.get(&key).map(|owner| owner.raw);
        if let Some(limit) = fixed_limit {
            ensure!(!self.in_flight, "index selection pending");
            self.stream.require_complete()?;
            self.ready = None;
            if !fixed_binding_admitted(cached.is_some(), policy.unwrap().0, self.graphs.len(), limit) {
                tracing::debug!(target: "cuteafd::target_step", rows=fingerprint[1],
                    retained=self.graphs.len(), "V4.1 index selection eager binding");
                return Ok(false);
            }
        } else if !self.retain_decode_graphs {
            self.clear_graph()?;
            return Ok(true);
        } else { self.restart()?; }
        if let Some((_, old)) = self.graph.take() {
            if fixed_limit.is_none() && old[1] > cuteafd_core::V41_VERIFY_ROWS {
                self.graphs.retire(&old);
            }
        }
        if cached.is_some() {
            self.graphs.launch(&key);
        } else if fixed_limit.is_none() && self.graphs.len() > MAX_RETAINED_DECODE_GRAPHS {
            self.graphs.retire_oldest(|_| true);
        }
        drop(self.graphs.drain_retired(|| Ok(()))?);
        self.graph = cached.map(|raw| (raw, key));
        if self.graph.is_none() {
            tracing::debug!(target: "cuteafd::graph_capture", site="index_selection",
                retained=self.graphs.len(), retain=self.retain_decode_graphs,
                key=?fingerprint, "graph cache miss");
        }
        Ok(true)
    }
    fn insert_graph(&mut self, graph: *mut c_void, fingerprint: Vec<usize>) -> Result<()> {
        let device = self.stream.library.cuda_get_device()?;
        // SAFETY: request/cache proposals and wave storage remain pinned by the lane.
        let owner = unsafe { crate::shared::decode_graph::GraphOwner::new(self.stream.library, device, graph, ())? };
        tracing::debug!(target: "cuteafd::graph_capture", site="index_selection", device,
            bank=self as *const Self as usize, key=?fingerprint, "native binding graph captured");
        self.graphs.insert(fingerprint.clone(), owner, None);
        self.graph = Some((graph, fingerprint));
        Ok(())
    }
    fn record_fixed_capture(&mut self, fingerprint: &[usize]) {
        if !self.fixed_captures.insert(fingerprint.to_vec()) {
            self.fixed_recaptures += 1;
            tracing::warn!(target: "cuteafd::graph_capture", site="index_selection",
                recaptures=self.fixed_recaptures, key=?fingerprint,
                "fixed index selection graph recaptured");
        }
        tracing::debug!(target: "cuteafd::graph_capture", site="index_selection",
            captures=self.fixed_captures.len(), recaptures=self.fixed_recaptures,
            retained=self.graphs.len(),
            "fixed index selection graph bank");
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        ensure!(!self.in_flight, "index selection pending");
        self.ready = None;
        self.stream.require_complete()?;
        self.graph = None;
        self.graphs.retire_all();
        drop(self.graphs.drain_retired(|| Ok(()))?);
        Ok(())
    }
    unsafe fn enqueue(
        &self,
        query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>],
        groups: &[ScoreGroup],
        shared: Option<&IndexSelectionOutput<'_>>,
        rows: usize,
        tiles: usize,
        width: usize,
        use_candidates: bool,
    ) -> Result<()> {
        let source = query.layer == 20;
        let follower = use_candidates;
        let stream = self.stream.raw;
        let passes = if follower { 1 } else { tiles };
        for tile in 0..passes {
            unsafe {
                if follower {
                    self.candidates.expand(
                        shared
                            .context("candidate source missing")?
                            .blocks
                            .context("candidate blocks missing")?,
                        self.b(1),
                        self.b(3),
                        rows,
                        stream,
                    )?;
                } else {
                    self.candidates.tile(
                        self.b(3),
                        self.b(2),
                        rows,
                        width,
                        (tile * width) as u64,
                        stream,
                    )?;
                }
                for group in groups {
                    let (offset, n) = (group.offset, group.rows);
                    let p = requests[group.request].proposal;
                    let slots = p.cache.pool_rows.bytes / 8;
                    let stride = p.cache.pool_pages.bytes / (slots * 4);
                    self.score.execute_overlay(
                        slice(query.packed, offset * 2048, n * 2048),
                        slice(query.scales, offset * 128, n * 128),
                        slice(query.head_weights, offset * 64, n * 64),
                        p.cache.packed,
                        p.cache.scales,
                        p.cache.pool_pages,
                        p.cache.pool_rows,
                        slice(self.b(0), offset * 48, n * 48),
                        slice(self.b(3), offset * width * 8, n * width * 8),
                        slice(self.b(4), offset * width * 4, n * width * 4),
                        p.packed,
                        p.scales,
                        p.capacity,
                        n,
                        width,
                        slots,
                        stride,
                        p.cache.packed.bytes / 64,
                        stream,
                    )?;
                }
                self.top.update(
                    self.b(4),
                    self.b(3),
                    self.b(7),
                    self.b(8),
                    self.b(9),
                    rows,
                    width,
                    tile == 0,
                    stream,
                )?;
                if source {
                    self.candidates.maxima(
                        self.b(4),
                        self.b(2),
                        self.b(1),
                        self.b(5),
                        self.b(6),
                        rows,
                        width,
                        stream,
                    )?;
                    self.top.update_blocks(
                        self.b(5),
                        self.b(6),
                        self.b(10),
                        self.b(11),
                        self.b(12),
                        rows,
                        width / 8,
                        tile == 0,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// # Safety
    /// Query rows are finite and correspond exactly to the supplied request/token
    /// order; their producers have drained. Proposal borrows are from the matching
    /// source's current wave. No external writes race inputs, caches or outputs.
    /// Results borrow the proposal views until their consumers have drained.
    pub unsafe fn execute<'s>(
        &'s mut self,
        query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>],
        shared: Option<&IndexSelectionOutput<'_>>,
    ) -> Result<IndexSelectionOutput<'s>> {
        unsafe { self.execute_inner(query, requests, shared, false)?; }
        self.output()
    }
    /// # Safety
    /// Retain query, proposals and optional source candidates through polling or abort.
    pub unsafe fn enqueue_selection(&mut self, query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>], shared: Option<&IndexSelectionOutput<'_>>) -> Result<()> {
        let result = unsafe { self.execute_inner(query, requests, shared, true) };
        if result.is_err() { self.abort_pending()?; }
        result
    }
    pub fn poll_pending(&mut self) -> Result<bool> {
        ensure!(self.in_flight && self.pending.is_some(), "no pending index selection");
        let ready = if crate::shared::memory::chain::active() {
            unsafe { crate::shared::memory::chain::finish(self.stream.library, self.stream.raw) }.map(|()| true)
        } else { unsafe { self.stream.library.cuda_stream_query(self.stream.raw) } };
        match ready {
            Ok(false) => Ok(false),
            Ok(true) => { self.ready = self.pending.take(); self.in_flight = false;
                if let Some(busy) = &self.shared_scratch_busy { busy.set(false); }
                Ok(true) }
            Err(error) => { self.abort_pending()?; Err(error) }
        }
    }
    pub fn abort_pending(&mut self) -> Result<()> {
        if self.in_flight {
            self.synchronize()?;
            self.in_flight = false; self.pending = None; self.ready = None;
            if let Some(busy) = &self.shared_scratch_busy { busy.set(false); }
        }
        Ok(())
    }
    unsafe fn execute_inner(&mut self, query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>], shared: Option<&IndexSelectionOutput<'_>>,
        defer: bool) -> Result<()> {
        ensure!(!self.in_flight, "index selection pending");
        ensure!(!self.shared_scratch_busy.as_ref().is_some_and(|busy| busy.get()), "shared index scratch is pending");
        self.ready = None;
        ensure!(
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&query.layer),
            "invalid selection producer layer"
        );
        ensure!(
            !requests.is_empty() && requests.len() <= 16,
            "invalid selection request count"
        );
        let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "selection rows exceed capacity"
        );
        ensure!(
            query.packed.bytes == rows * 2048
                && query.scales.bytes == rows * 128
                && query.head_weights.bytes == rows * 64,
            "query output row count differs"
        );
        ensure!(
            query.packed.device_id == self.b(0).device_id,
            "selection query device differs"
        );
        let expected_source = if query.layer >= 20 { 20 } else { query.layer };
        let mut bindings = Vec::with_capacity(rows);
        let mut max_length = 0u64;
        let mut fingerprint = vec![
            query.layer,
            rows,
            query.packed.ptr as usize,
            query.scales.ptr as usize,
            query.head_weights.ptr as usize,
        ];
        let mut metadata = Vec::with_capacity(rows);
        let mut groups = Vec::new();
        for (i, r) in requests.iter().enumerate() {
            ensure!(
                !r.positions.is_empty() && r.positions.windows(2).all(|x| x[0] < x[1]),
                "empty or unordered selection request"
            );
            ensure!(
                r.proposal.source_layer == expected_source,
                "index key source layer differs"
            );
            ensure!(
                requests[..i]
                    .iter()
                    .all(|x| !x.proposal.binding().same_request(r.proposal.binding())),
                "duplicate selection request"
            );
            ensure!(
                requests[0]
                    .proposal
                    .binding()
                    .same_pool(r.proposal.binding()),
                "selection state owner differs"
            );
            let p = r.proposal;
            ensure!(p.cache.pool_rows.bytes > 0 && p.cache.slot < p.cache.pool_rows.bytes / 8
                && p.cache.pool_pages.bytes == (p.cache.pool_rows.bytes / 8) * p.cache.device_pages.bytes,
                "index pool table geometry differs");
            extend_score_groups(&mut groups, i, r.positions.len(), [
                p.cache.packed.ptr as usize,
                p.cache.scales.ptr as usize,
                p.cache.pool_pages.ptr as usize,
                p.cache.pool_rows.ptr as usize,
                p.cache.pool_pages.bytes,
                p.cache.pool_rows.bytes,
                p.cache.packed.bytes,
                p.packed.ptr as usize,
                p.scales.ptr as usize,
                p.capacity,
            ]);
            for &position in r.positions {
                let mut m = p.metadata(position)?;
                m[0] = p.cache.slot as u64;
                max_length = max_length.max(m[1]);
                metadata.push(m);
                bindings.push((p.binding(), position));
            }
        }
        // Request partitions and slots live in per-row metadata, not executable
        // geometry. Only distinct proposal/table owners require another launch.
        for group in &groups {
            fingerprint.push(group.rows);
            fingerprint.extend(group.binding);
        }
        if query.origin().is_some() {
            ensure!(
                bindings
                    .iter()
                    .map(|(_, p)| *p)
                    .eq(query.bound_tokens()?.iter().copied()),
                "selection token order differs from query producer"
            );
        }
        if query.layer > 20 {
            let s = shared.context("later index layer requires source candidates")?;
            ensure!(
                s.layer == 20 && s.rows == rows && s.bindings == bindings && s.blocks.is_some(),
                "candidate source snapshot or row order differs"
            );
            fingerprint.push(s.blocks.unwrap().ptr as usize);
        } else {
            ensure!(
                shared.is_none(),
                "candidate source is unexpected for this layer"
            );
        }
        // Causal scoring masks padded candidates to -inf; top-k excludes them.
        // Width buckets keep context growth inside the bounded decode graph bank.
        let width = tile_width(max_length as usize, DECODE_WIDTH_BUCKET);
        let use_candidates = query.layer > 20 && max_length > WIDTH as u64;
        let tiles = (max_length as usize).div_ceil(width).max(1);
        fingerprint.extend([width, usize::from(use_candidates)]);
        fingerprint.push(if query.layer > 20 { 1 } else { tiles });
        let (host, staging) = self.staging.region(query.layer);
        for (i, m) in metadata.iter().enumerate() {
            for (j, value) in m.iter().enumerate() {
                staging[i * 48 + j * 8..i * 48 + j * 8 + 8].copy_from_slice(&value.to_ne_bytes());
            }
            staging[rows * 48 + i * 8..rows * 48 + i * 8 + 8].copy_from_slice(&m[1].to_ne_bytes());
        }
        // Prefill tails (and any other uncached large shape) bind eagerly.
        let forced_eager = crate::shared::decode_graph::census::eager("index")
            || !super::graph_policy::captures_shape(rows as u32);
        let capture = if forced_eager { false } else { self.select_graph(&fingerprint)? };
        if defer {
            self.in_flight = true;
            if let Some(busy) = &self.shared_scratch_busy { busy.set(true); }
            unsafe {
                crate::shared::memory::chain::join(self.stream.library, self.stream.raw)?;
                self.stream.library.copy_host_buffer_h2d_async(self.b(0), host, rows * 48, self.stream.raw)?;
                let mut lengths = host;
                lengths.ptr = host.ptr.cast::<u8>().add(rows * 48).cast(); lengths.bytes = rows * 8;
                self.stream.library.copy_host_buffer_h2d_async(self.b(1), lengths, rows * 8, self.stream.raw)?;
            }
        } else {
            let (metadata, lengths) = (self.b(0), self.b(1));
            let (_, staged) = self.staging.region(query.layer);
            self.stream.library.copy_h2d(metadata, &staged[..rows * 48])?;
            self.stream.library.copy_h2d(lengths, &staged[rows * 48..rows * 56])?;
        }
        if capture && self.graph.is_none() {
            unsafe {
                self.stream
                    .library
                    .cuda_graph_begin_capture(self.stream.raw)?;
            }
            let launched = unsafe {
                self.enqueue(query, requests, &groups, shared, rows, tiles, width, use_candidates)
            };
            let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
            match (launched, captured) {
                (Ok(()), Ok(g)) => {
                    if super::graph_policy::fixed_binding_limit().is_some() {
                        self.record_fixed_capture(&fingerprint);
                    }
                    self.insert_graph(g, fingerprint)?;
                },
                (Err(e), Ok(g)) => {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(g)?;
                    }
                    return Err(e);
                }
                (Err(e), Err(_)) | (Ok(()), Err(e)) => return Err(e),
            }
        }
        let dispatch = || unsafe {
            if capture {
                let g = self.graph.as_ref().context("selection graph missing")?.0;
                self.stream.library.cuda_graph_launch(g, self.stream.raw)
            } else { self.enqueue(query, requests, &groups, shared, rows, tiles, width, use_candidates) }
        };
        let launched = if crate::shared::decode_graph::census::enabled() {
            unsafe { self.stream.library.cuda_graph_census_call(self.stream.raw, c"index", rows, capture,
                crate::shared::decode_graph::census::arm(), dispatch) }
        } else { dispatch() };
        if self.audit && launched.is_ok() {
            // SAFETY: execute_inner retains its inputs; the audit drains before reading scratch.
            unsafe { self.audit_scores(query, requests, rows, width)?; }
        }
        if self.async_audit.is_some() && launched.is_ok() {
            // SAFETY: snapshots are retained until the completed-step hook.
            unsafe { self.enqueue_async_audit(query, requests, &metadata, rows, width)?; }
        }
        let ready = Ready { origin: query.origin(), layer: query.layer, rows, bindings };
        if defer {
            launched?;
            self.pending = Some(ready);
        } else {
            launched.and(self.synchronize())?;
            self.ready = Some(ready);
        }
        Ok(())
    }
    /// Inspect retained snapshots only after the target step has completed.
    pub fn finish_audit(&mut self) -> Result<()> {
        let Some(audit) = self.async_audit.as_mut() else { return Ok(()) };
        // SAFETY: querying does not drain or change stream ordering.
        ensure!(unsafe { self.stream.library.cuda_stream_query(self.stream.raw)? },
            "async index audit inspected before target completion");
        for row in &mut audit.layers {
            let Some((layer, bytes, source_end, description)) = row.pending.take() else { continue };
            let data = row.host.bytes();
            let pooled = &data[..bytes];
            let old = &data[audit.score_bytes..audit.score_bytes + bytes];
            let mismatch = pooled.chunks_exact(4).zip(old.chunks_exact(4)).position(|(a, b)| a != b);
            let hash = |data: &[u8]| data.iter().fold(0xcbf29ce484222325u64,
                |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100000001b3));
            tracing::info!(target: "cuteafd::index_audit", layer, %description,
                pooled_hash = hash(pooled), old_hash = hash(old),
                source_hash = hash(&data[audit.score_bytes * 2..source_end]), ?mismatch,
                "asynchronous live index scorer audit after target step");
            ensure!(mismatch.is_none(), "asynchronous index audit mismatch at layer {layer} element {mismatch:?}");
        }
        Ok(())
    }
    /// # Safety
    /// Inputs and pinned destinations remain live until finish_audit after step completion.
    unsafe fn enqueue_async_audit(&mut self, query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>], metadata: &[[u64; 6]], rows: usize, width: usize) -> Result<()> {
        ensure!(requests.len() == 1 && width <= AUDIT_WIDTH,
            "async index audit is limited to one short-context golden request");
        let p = requests[0].proposal;
        ensure!(p.cache.rows <= 2048 && (p.packed.bytes + p.scales.bytes <= AUDIT_SOURCE_BYTES
            || metadata[0][3] == 0), "async index audit source exceeds diagnostic bounds");
        let layer = [2, 8, 14, 20, 24, 28, 32, 36].iter().position(|&l| l == query.layer).unwrap();
        let positions = slice(self.b(3), 0, rows * width * 8);
        let pooled = slice(self.b(4), 0, rows * width * 4);
        let audit = self.async_audit.as_mut().unwrap();
        let saved = &mut audit.layers[layer];
        ensure!(saved.pending.is_none(), "async index audit slot reused before target completion");
        let library = self.stream.library;
        let stream = self.stream.raw;
        let mut host = saved.host.buffer;
        let metadata_offset = audit.score_bytes * 2;
        for (i, values) in metadata.iter().enumerate() {
            for (j, &value) in values.iter().enumerate() {
                saved.host.bytes_mut()[metadata_offset + i * 48 + j * 8..metadata_offset + i * 48 + j * 8 + 8]
                    .copy_from_slice(&(if j == 0 { 0 } else { value }).to_ne_bytes());
            }
        }
        // SAFETY: destinations are disjoint bounded regions of retained pinned storage.
        unsafe {
            library.copy_d2h_host_buffer_async(host, pooled, pooled.bytes, stream)?;
            host.ptr = saved.host.buffer.ptr.cast::<u8>().add(metadata_offset).cast();
            host.bytes = rows * 48;
            library.copy_host_buffer_h2d_async(audit.metadata.buffer, host, rows * 48, stream)?;
            self.score.execute_overlay(query.packed, query.scales, query.head_weights,
                p.cache.packed, p.cache.scales, p.cache.device_pages, p.cache.device_rows,
                slice(audit.metadata.buffer, 0, rows * 48), positions,
                slice(audit.scores.buffer, 0, rows * width * 4), p.packed, p.scales,
                p.capacity, rows, width, 1, p.cache.device_pages.bytes / 4,
                p.cache.packed.bytes / 64, stream)?;
            host = saved.host.buffer;
            host.ptr = host.ptr.cast::<u8>().add(audit.score_bytes).cast();
            host.bytes = audit.score_bytes;
            library.copy_d2h_host_buffer_async(host, audit.scores.buffer, pooled.bytes, stream)?;
        }
        let mut cursor = metadata_offset + rows * 48;
        let mut snapshot = |buffer: CuteafdDeviceBuffer| -> Result<()> {
            if buffer.bytes == 0 { return Ok(()) }
            ensure!(cursor + buffer.bytes <= saved.host.buffer.bytes, "async audit snapshot capacity exceeded");
            let mut host = saved.host.buffer;
            // SAFETY: cursor is bounded above and retained until stream completion.
            unsafe {
                host.ptr = host.ptr.cast::<u8>().add(cursor).cast(); host.bytes = buffer.bytes;
                library.copy_d2h_host_buffer_async(host, buffer, buffer.bytes, stream)?;
            }
            cursor += buffer.bytes;
            Ok(())
        };
        snapshot(p.cache.device_rows)?;
        snapshot(slice(p.cache.device_pages, 0, p.cache.rows.div_ceil(256) * 4))?;
        for (i, &page) in p.cache.pages.iter().take(p.cache.rows.div_ceil(256)).enumerate() {
            let count = (p.cache.rows - i * 256).min(256);
            snapshot(slice(p.cache.packed, page as usize * 256 * 64, count * 64))?;
            snapshot(slice(p.cache.scales, page as usize * 256 * 4, count * 4))?;
        }
        if metadata[0][3] > 0 {
            snapshot(p.packed)?;
            snapshot(p.scales)?;
        }
        saved.pending = Some((query.layer, pooled.bytes, cursor, format!(
            "first={} last={} rows={rows} width={width} slot={} old_pages={:?} pool_pages={:?} keys={:?} proposals={:?} initialized={} descriptor={:?} stream={:?}",
            requests[0].positions[0], requests[0].positions[rows - 1], p.cache.slot,
            p.cache.device_pages.ptr, p.cache.pool_pages.ptr, p.cache.packed.ptr, p.packed.ptr,
            p.cache.rows, metadata[0], stream)));
        Ok(())
    }
    /// Compare live pooled addressing with the original per-request ABI after dispatch.
    /// # Safety
    /// Same input lifetimes as execute_inner; this diagnostic drains its own stream.
    unsafe fn audit_scores(&self, query: &IndexQueryOutput<'_>, requests: &[SelectionRequest<'_>],
        rows: usize, width: usize) -> Result<()> {
        self.synchronize()?;
        let library = self.stream.library;
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes];
            if !bytes.is_empty() { library.copy_d2h(&mut bytes, buffer)?; }
            Ok(bytes)
        };
        let hash = |bytes: &[u8]| bytes.iter().fold(0xcbf29ce484222325u64,
            |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100000001b3));
        let pooled = read(slice(self.b(4), 0, rows * width * 4))?;
        let mut metadata = read(slice(self.b(0), 0, rows * 48))?;
        for row in metadata.chunks_exact_mut(48) { row[..8].fill(0); }
        let local = DeviceAllocation::new(library, metadata.len())?;
        library.copy_h2d(local.buffer, &metadata)?;
        let mut offset = 0;
        for (request, r) in requests.iter().enumerate() {
            let p = r.proposal;
            let n = r.positions.len();
            let slots = p.cache.pool_rows.bytes / 8;
            let stride = p.cache.device_pages.bytes / 4;
            let page_bytes = p.cache.rows.div_ceil(256) * 4;
            let local_pages = read(slice(p.cache.device_pages, 0, page_bytes))?;
            let pool_pages = read(slice(p.cache.pool_pages, p.cache.slot * stride * 4, page_bytes))?;
            let local_length = read(p.cache.device_rows)?;
            let pool_length = read(slice(p.cache.pool_rows, p.cache.slot * 8, 8))?;
            let mut key_hash = 0u64;
            let mut scale_hash = 0u64;
            for (i, page) in pool_pages.chunks_exact(4).enumerate() {
                let page = u32::from_ne_bytes(page.try_into().unwrap()) as usize;
                let count = (p.cache.rows - i * 256).min(256);
                key_hash ^= hash(&read(slice(p.cache.packed, page * 256 * 64, count * 64))?);
                scale_hash ^= hash(&read(slice(p.cache.scales, page * 256 * 4, count * 4))?);
            }
            let descriptor = p.metadata(r.positions[0])?;
            let mut proposal_hash = 0u64;
            let mut proposal_scale_hash = 0u64;
            if descriptor[3] > 0 {
                let packed = read(p.packed)?;
                let scales = read(p.scales)?;
                for row in 0..descriptor[3] as usize {
                    let physical = descriptor[4] as usize + row * descriptor[5] as usize;
                    proposal_hash ^= hash(&packed[physical * 64..(physical + 1) * 64]);
                    proposal_scale_hash ^= hash(&scales[physical * 4..(physical + 1) * 4]);
                }
            }
            tracing::info!(target: "cuteafd::index_audit", layer = query.layer, request,
                first = r.positions[0], last = r.positions[n - 1], rows = n, width,
                old_slot = 0, pool_slot = p.cache.slot, slots, stride,
                old_pages = ?p.cache.device_pages.ptr, pool_pages = ?p.cache.pool_pages.ptr,
                keys = ?p.cache.packed.ptr, proposals = ?p.packed.ptr,
                host_initialized_rows = p.cache.rows, published_rows = u64::from_ne_bytes(pool_length[..8].try_into().unwrap()),
                start = descriptor[2], count = descriptor[3], proposal_offset = descriptor[4], step = descriptor[5],
                key_hash, scale_hash, proposal_hash, proposal_scale_hash,
                table_equal = local_pages == pool_pages, length_equal = local_length == pool_length,
                "live index scorer source audit");
            ensure!(local_pages == pool_pages && local_length == pool_length,
                "index audit table mismatch at layer {} request {request}", query.layer);
            // SAFETY: all original inputs are drained and retained; only scratch scores are overwritten.
            unsafe {
                self.score.execute_overlay(
                    slice(query.packed, offset * 2048, n * 2048),
                    slice(query.scales, offset * 128, n * 128),
                    slice(query.head_weights, offset * 64, n * 64),
                    p.cache.packed, p.cache.scales, p.cache.device_pages, p.cache.device_rows,
                    slice(local.buffer, offset * 48, n * 48),
                    slice(self.b(3), offset * width * 8, n * width * 8),
                    slice(self.b(4), offset * width * 4, n * width * 4),
                    p.packed, p.scales, p.capacity, n, width, 1, stride,
                    p.cache.packed.bytes / 64, self.stream.raw)?;
            }
            offset += n;
        }
        self.synchronize()?;
        let old = read(slice(self.b(4), 0, rows * width * 4))?;
        let mismatch = pooled.chunks_exact(4).zip(old.chunks_exact(4)).position(|(a, b)| a != b);
        tracing::info!(target: "cuteafd::index_audit", layer = query.layer, rows, width,
            pooled_hash = hash(&pooled), old_hash = hash(&old), ?mismatch,
            "live index scorer bitwise audit");
        ensure!(mismatch.is_none(), "index scorer audit mismatch at layer {} row {:?}",
            query.layer, mismatch.map(|i| (i / width, i % width)));
        Ok(())
    }
    /// Borrow completed selection storage. Consumers validate its retained source
    /// bindings against live cache proposals before using these logical row IDs.
    pub fn output(&self) -> Result<IndexSelectionOutput<'_>> {
        let r = self.ready.as_ref().context("index selection output unpublished")?;
        let rows = r.rows;
        Ok(IndexSelectionOutput {
            origin: r.origin,
            selected: slice(self.b(9), 0, rows * 2048),
            rows: r.rows,
            layer: r.layer,
            blocks: (r.layer == 20).then(|| slice(self.b(12), 0, rows * 8192)),
            bindings: &r.bindings,
            _sources: PhantomData,
        })
    }
}
fn fixed_binding_admitted(cached: bool, allowed_shape: bool, count: usize, limit: usize) -> bool {
    cached || (allowed_shape && count < limit)
}
impl Drop for IndexSelectionWave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.abort_pending() { crate::shared::decode_graph::fatal_drain(Err(error), "pending index selection"); }
        if let Err(error) = self.clear_graph() {
            crate::shared::decode_graph::fatal_drain(Err(error), "index selection");
        }
        if self.async_audit.is_some() {
            if let Err(error) = self.synchronize().and_then(|()| self.finish_audit()) {
                tracing::error!(%error, "final asynchronous index audit failed");
            }
        }
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn score_groups_ignore_request_partition_but_preserve_launch_bindings() {
        let binding = [17; 10];
        for partition in [vec![1; 16], vec![8, 8], vec![1, 15], vec![15, 1], vec![16]] {
            let mut groups = Vec::new();
            for (request, rows) in partition.into_iter().enumerate() {
                extend_score_groups(&mut groups, request, rows, binding);
            }
            assert_eq!(groups, vec![ScoreGroup { request: 0, offset: 0, rows: 16, binding }]);
        }
        for field in 0..binding.len() {
            let mut other = binding;
            other[field] += 1;
            let mut groups = Vec::new();
            extend_score_groups(&mut groups, 0, 8, binding);
            extend_score_groups(&mut groups, 1, 3, other);
            extend_score_groups(&mut groups, 2, 5, binding);
            assert_eq!(groups.iter().map(|g| (g.request, g.offset, g.rows)).collect::<Vec<_>>(),
                vec![(0, 0, 8), (1, 8, 3), (2, 11, 5)]);
        }
    }

    #[test]
    fn fixed_selection_bank_keeps_hits_and_never_captures_overflow() {
        for limit in [192, 256] {
            assert!(fixed_binding_admitted(false, true, limit - 1, limit));
            assert!(!fixed_binding_admitted(false, true, limit, limit));
            assert!(fixed_binding_admitted(true, true, limit, limit));
            assert!(fixed_binding_admitted(true, false, limit, limit));
            assert!(!fixed_binding_admitted(false, false, 0, limit));
        }
    }

    #[test]
    fn tile_buckets_cover_causal_rows_without_unaligned_tiles() {
        for bucket in [8, DECODE_WIDTH_BUCKET] {
            for length in 0..=1_048_576usize {
                let width = tile_width(length, bucket);
                assert!((bucket..=WIDTH).contains(&width) && width % 8 == 0);
                assert!(width >= length.min(WIDTH));
                if length <= WIDTH { assert!(width - length < bucket || length == 0); }
                else { assert_eq!(width, WIDTH); }
            }
        }
    }

    #[test]
    fn peer_selection_preserves_query_snapshot_validation() -> Result<()> {
        let origin=QueryBinding::new(2)?;
        // Only metadata is inspected; these pointers are never dereferenced.
        let buffer=CuteafdDeviceBuffer { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            bytes: 2048, device_id: 0, flags: 0 };
        let original=IndexSelectionOutput { origin: Some(origin), selected: buffer,
            rows: 1, layer: 2, blocks: Some(buffer), bindings: &[], _sources: PhantomData };
        let mut destination=buffer; destination.device_id=1;
        let peer=unsafe { original.peer_attention(destination)? };
        peer.validate_query(origin)?;
        assert!(peer.validate_query(QueryBinding::new(2)?).is_err());
        // Intermediate layers may reuse this producer, as on the original GPU.
        peer.validate_query(QueryBinding::new(3)?)?;
        assert!(peer.blocks.is_none());
        assert!(unsafe { original.peer_attention(buffer) }.is_err());
        destination.bytes-=1;
        assert!(unsafe { original.peer_attention(destination) }.is_err());
        Ok(())
    }

    #[test]
    fn cuda_selection_shapes_replay_current_inputs_after_restart() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_INDEX_GRAPH_LIBRARY") else {
            eprintln!("skip selection graph lifetime test: CUTEAFD_INDEX_GRAPH_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        library.cuda_set_device(0)?;
        let mut wave = IndexSelectionWave::new(&library, 80, IndexSelectionWave::device_bytes(80)?)?;
        wave.enable_small_graph_shapes();
        let input = slice(wave.b(0), 0, 320);
        let output = slice(wave.b(1), 0, 320);
        let mut handles = std::collections::BTreeMap::new();
        for (iteration, (layer, rows)) in [(2, 4), (8, 2), (2, 4), (8, 2), (2, 80), (2, 4), (8, 2)]
            .into_iter().enumerate() {
            wave.restart()?;
            let fingerprint = vec![layer, rows, input.ptr as usize, output.ptr as usize];
            assert!(wave.select_graph_with_policy(&fingerprint, None)?);
            if wave.graph.is_none() {
                unsafe {
                    library.cuda_graph_begin_capture(wave.stream.raw)?;
                    library.copy_d2d_async(output, input, rows * 4, wave.stream.raw)?;
                    let graph = library.cuda_graph_end_capture(wave.stream.raw)?;
                    wave.insert_graph(graph, fingerprint)?;
                }
            }
            let graph = wave.graph.as_ref().unwrap().0;
            if rows <= 64 {
                assert_eq!(*handles.entry((layer, rows)).or_insert(graph), graph,
                    "restart or large-prefill transition discarded a decode graph");
            }
            let value = iteration as u8 + 1;
            library.copy_h2d(input, &vec![value; 320])?;
            library.copy_h2d(output, &[0; 320])?;
            unsafe { library.cuda_graph_launch(graph, wave.stream.raw)?; }
            wave.synchronize()?;
            let mut actual = vec![0; 320];
            library.copy_d2h(&mut actual, output)?;
            assert_eq!(&actual[..rows * 4], vec![value; rows * 4]);
            assert!(actual[rows * 4..].iter().all(|&x| x == 0));
        }
        wave.clear_graph()?;
        assert!(wave.graph.is_none() && wave.graphs.len() == 0);
        // Exercise actual CUDA handle destruction at the retention limit, then
        // replay current data through each newly captured graph. Distinct final
        // fingerprint fields stand in for changing request/source layouts.
        for key in 0..MAX_RETAINED_DECODE_GRAPHS + 3 {
            wave.restart()?;
            let fingerprint = vec![2, 4, input.ptr as usize, output.ptr as usize, key];
            assert!(wave.select_graph_with_policy(&fingerprint, None)?);
            assert!(wave.graph.is_none());
            unsafe {
                library.cuda_graph_begin_capture(wave.stream.raw)?;
                library.copy_d2d_async(output, input, 16, wave.stream.raw)?;
                let graph = library.cuda_graph_end_capture(wave.stream.raw)?;
                wave.insert_graph(graph, fingerprint)?;
            }
            assert!(wave.graphs.len() <= MAX_RETAINED_DECODE_GRAPHS + 1);
            let value = (key % 251 + 1) as u8;
            library.copy_h2d(input, &vec![value; 320])?;
            library.copy_h2d(output, &[0; 320])?;
            unsafe { library.cuda_graph_launch(wave.graph.as_ref().unwrap().0, wave.stream.raw)?; }
            wave.synchronize()?;
            let mut actual = vec![0; 320];
            library.copy_d2h(&mut actual, output)?;
            assert_eq!(&actual[..16], &[value; 16]);
            assert!(actual[16..].iter().all(|&x| x == 0));
        }
        assert_eq!(wave.graphs.len(), MAX_RETAINED_DECODE_GRAPHS + 1);
        assert!(wave.graphs.count(|key| key[4] == 0) == 0);
        wave.clear_graph()?;
        assert!(wave.graph.is_none() && wave.graphs.len() == 0);
        // A full fixed bank replays exact hits and leaves both handles intact
        // while overflow or excluded shapes execute directly on the same stream.
        let mut fixed_handles = std::collections::BTreeMap::new();
        for (iteration, (key, rows, allowed)) in
            [(0, 4, true), (1, 4, true), (2, 4, true), (3, 80, false), (0, 4, true), (1, 4, true)]
                .into_iter().enumerate() {
            wave.restart()?;
            let fingerprint = vec![2, rows, input.ptr as usize, output.ptr as usize, key];
            let capture = wave.select_graph_with_policy(&fingerprint, Some((allowed, 2)))?;
            assert_eq!(capture, key < 2);
            if capture && wave.graph.is_none() {
                // SAFETY: both allocations outlive the bank and the stream is drained.
                let graph = unsafe {
                    library.cuda_graph_begin_capture(wave.stream.raw)?;
                    library.copy_d2d_async(output, input, rows * 4, wave.stream.raw)?;
                    library.cuda_graph_end_capture(wave.stream.raw)?
                };
                wave.insert_graph(graph, fingerprint.clone())?;
                wave.record_fixed_capture(&fingerprint);
            }
            let value = iteration as u8 + 1;
            library.copy_h2d(input, &vec![value; 320])?;
            library.copy_h2d(output, &[0; 320])?;
            // SAFETY: inputs/outputs are live, shape-exact, and exclusive until synchronize.
            unsafe {
                if capture {
                    let graph = wave.graph.as_ref().unwrap().0;
                    assert_eq!(*fixed_handles.entry(key).or_insert(graph), graph);
                    library.cuda_graph_launch(graph, wave.stream.raw)?;
                } else {
                    library.copy_d2d_async(output, input, rows * 4, wave.stream.raw)?;
                }
            }
            wave.synchronize()?;
            let mut actual = vec![0; 320];
            library.copy_d2h(&mut actual, output)?;
            assert_eq!(&actual[..rows * 4], vec![value; rows * 4]);
            assert!(actual[rows * 4..].iter().all(|&x| x == 0));
            assert_eq!(wave.fixed_recaptures, 0);
            assert!(wave.graphs.len() <= 2);
        }
        assert_eq!(wave.fixed_captures.len(), 2);
        wave.clear_graph()?;
        Ok(())
    }
}
