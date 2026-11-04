//! Backbone FP8 window KV production, private proposals and accepted ring writes.
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::shared::memory::row_window::RowWindow;
use crate::shared::decode_graph::{fatal_drain, GraphBank, GraphOwner};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Plan, V41Kv};
use cuteafd_loader::OfficialV41Catalog;
use std::{
    ffi::c_void,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
mod prefix;
pub(crate) mod replica;
mod commit;
use commit::PendingCommit;
pub(crate) use commit::BatchedWindowCommit;
pub(crate) use prefix::WINDOW_PREFIX_BYTES;
static NEXT_SNAPSHOT: AtomicU64 = AtomicU64::new(1);
fn next(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| anyhow::anyhow!("window IDs exhausted"))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowLease {
    owner: u64,
    slot: usize,
    generation: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct WindowChunk {
    pub lease: WindowLease,
    pub position: u64,
    pub tokens: u32,
}
#[derive(Clone, Copy, Default)]
struct Slot {
    request: Option<u64>,
    generation: u64,
    version: u64,
    end: u64,
    begin: u64,
}
pub(crate) struct WindowCacheView<'a> {
    pub values: CuteafdDeviceBuffer,
    pub scales: CuteafdDeviceBuffer,
    pub device_end: CuteafdDeviceBuffer,
    pub end: u64,
    /// Earliest initialized logical row; preceding ring bytes are inaccessible.
    pub begin: u64,
    _owner: PhantomData<&'a ()>,
}
fn slice(mut b: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
pub(crate) struct WindowState<'a> {
    values: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    ends: DeviceAllocation<'a>,
    slots: [Slot; cuteafd_core::V41_MAX_REQUESTS],
    writing: std::rc::Rc<std::cell::Cell<u64>>,
    slot_count: usize,
    layer: usize,
    owner: u64,
    replica: Option<replica::WindowStateReplica<'a>>,
}
impl<'a> WindowState<'a> {
    pub fn device_bytes(layer: usize, slots: usize) -> Result<usize> {
        ensure!(
            layer < 40 && (1..=cuteafd_core::V41_MAX_REQUESTS).contains(&slots),
            "invalid backbone window state shape"
        );
        Ok(slots * (128 * 528 + 8))
    }
    pub fn new(
        library: &'a NativeLibrary,
        layer: usize,
        slots: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(layer, slots)? <= budget,
            "window state exceeds budget"
        );
        let value = Self {
            values: DeviceAllocation::attention_plane(library, slots * 128, 512, "window cache values")?,
            scales: DeviceAllocation::attention_plane(library, slots * 128, 16, "window cache scales")?,
            ends: DeviceAllocation::new(library, slots * 8)?,
            slots: [Slot::default(); cuteafd_core::V41_MAX_REQUESTS],
            writing: Default::default(),
            slot_count: slots,
            layer,
            owner: next(&NEXT_OWNER)?,
            replica: None,
        };
        library.copy_h2d(value.ends.buffer, &vec![0; slots * 8])?;
        Ok(value)
    }
    pub fn begin_request(&mut self, slot: usize, request: u64) -> Result<WindowLease> {
        ensure!(
            slot < self.slot_count && self.slots[slot].request.is_none()
                && self.writing.get() & (1 << slot) == 0,
            "window slot unavailable"
        );
        ensure!(
            !self.slots.iter().any(|s| s.request == Some(request)),
            "duplicate window request"
        );
        let generation = self.slots[slot]
            .generation
            .checked_add(1)
            .context("window generation exhausted")?;
        self.ends
            .library
            .copy_h2d(slice(self.ends.buffer, slot * 8, 8), &[0; 8])?;
        if let Some(replica)=&self.replica { replica.storage.reset_end(slot,0)?; }
        self.slots[slot] = Slot {
            request: Some(request),
            generation,
            ..Slot::default()
        };
        Ok(WindowLease {
            owner: self.owner,
            slot,
            generation,
        })
    }
    /// Start a fresh decoder window at the retained encoder suffix. No ring rows
    /// become valid: consumers must enforce the returned view's lower bound.
    /// Existing consumers must be drained; version advancement revokes proposals.
    pub fn begin_replay(&mut self, lease: WindowLease, position: u64) -> Result<()> {
        ensure!(self.layer >= 20, "invalid decoder replay layer");
        self.begin_empty_at(lease, position)
    }
    pub fn begin_encoder_replay(&mut self, lease: WindowLease, position: u64) -> Result<()> {
        ensure!(self.layer < 20, "invalid encoder replay layer");
        self.begin_empty_at(lease, position)
    }
    fn begin_empty_at(&mut self, lease: WindowLease, position: u64) -> Result<()> {
        let slot = self.validate(lease)?;
        ensure!(position <= 1048576, "invalid window replay start");
        ensure!(
            self.slots[slot].end == 0 && self.slots[slot].version == 0,
            "decoder replay requires a fresh window lease"
        );
        let written=self.ends.library.copy_h2d(
            slice(self.ends.buffer, slot * 8, 8), &position.to_ne_bytes()).and_then(|()| {
                if let Some(replica)=&self.replica { replica.storage.reset_end(slot,position)?; }
                Ok(())
            });
        if let Err(error) = written {
            self.slots[slot].request = None;
            return Err(error);
        }
        self.slots[slot].begin = position;
        self.slots[slot].end = position;
        self.slots[slot].version = 1;
        Ok(())
    }
    fn validate(&self, lease: WindowLease) -> Result<usize> {
        let slot = self.validate_identity(lease)?;
        ensure!(self.writing.get() & (1 << slot) == 0, "window write is unpublished");
        Ok(slot)
    }
    fn validate_identity(&self, lease: WindowLease) -> Result<usize> {
        ensure!(
            lease.owner == self.owner && lease.slot < self.slot_count,
            "foreign window lease"
        );
        let s = self.slots[lease.slot];
        ensure!(
            s.request.is_some() && s.generation == lease.generation,
            "stale window lease"
        );
        Ok(lease.slot)
    }
    pub fn end(&self, lease: WindowLease) -> Result<u64> {
        Ok(self.slots[self.validate_identity(lease)?].end)
    }
    /// The ring's initialized span `(begin, end)` (host bookkeeping only).
    pub fn span(&self, lease: WindowLease) -> Result<(u64, u64)> {
        let slot = self.slots[self.validate_identity(lease)?];
        Ok((slot.begin, slot.end))
    }
    pub fn request_id(&self, lease: WindowLease) -> Result<u64> {
        self.slots[self.validate_identity(lease)?]
            .request
            .context("window request missing")
    }
    pub fn view(&self, lease: WindowLease) -> Result<WindowCacheView<'_>> {
        let slot = self.validate(lease)?;
        Ok(WindowCacheView {
            values: slice(self.values.buffer, slot * 128 * 512, 128 * 512),
            scales: slice(self.scales.buffer, slot * 128 * 16, 128 * 16),
            device_end: slice(self.ends.buffer, slot * 8, 8),
            end: self.slots[slot].end,
            begin: self.slots[slot].begin,
            _owner: PhantomData,
        })
    }
    pub(crate) fn ensure_not_writing(&self, lease: WindowLease) -> Result<()> {
        self.validate(lease).map(|_| ())
    }
    /// Consumers must drain before release; generation checks revoke old views.
    pub fn release(&mut self, lease: WindowLease) -> Result<()> {
        let slot = self.validate(lease)?;
        self.slots[slot].request = None;
        // No consumer may use a revoked lease. Admission resets device_end before
        // publishing its replacement, so clearing it here only adds a host wait.
        Ok(())
    }
    pub fn invalidate(&mut self, leases: &[WindowLease]) -> Result<()> {
        let slots = leases
            .iter()
            .map(|&l| self.validate(l))
            .collect::<Result<Vec<_>>>()?;
        for &slot in &slots {
            self.slots[slot].request = None;
        }
        let mut result = Ok(());
        for slot in slots {
            result = result.and(
                self.ends
                    .library
                    .copy_h2d(slice(self.ends.buffer, slot * 8, 8), &[0; 8]),
            );
        }
        result
    }
}
pub(crate) struct WindowWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 3],
    tensors: NativeRtxTensors<'a>,
    scales: DeviceAllocation<'a>,
}
impl<'a> WindowWeights<'a> {
    fn names(layer: usize) -> Result<[String; 3]> {
        ensure!(layer < 40, "invalid backbone window layer");
        Ok([
            format!("layers.{layer}.attn.wkv.weight"),
            format!("layers.{layer}.attn.wkv.scale"),
            format!("layers.{layer}.attn.kv_norm.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let resident = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(resident == 2_625_024, "unexpected window tensor sizes");
        Ok(resident
            + library
                .v41_fp8_matrix_info(1, 5120, 512)?
                .packed_weight_scale_bytes as usize)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "window weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging)?;
        let fp8 = library.v41_fp8_matrix_kernel(1, 5120, 512)?;
        let scales = DeviceAllocation::new(library, fp8.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let launched =
            unsafe { crate::shared::experts::rtx::dense::v41_pack_scales(&fp8, tensors.get(&names[1])?, scales.buffer, stream.raw) };
        launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
        })
    }
    /// A wave with its own storage: rows, unit and capacity are equal
    /// (references, tests, decode).
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<WindowWave<'_, 'a>> {
        ensure!(
            WindowWave::device_bytes(self.library, capacity)? <= budget,
            "window wave exceeds budget"
        );
        let rows = capacity as usize;
        self.on_storage(WindowStorage::new(self.library, rows, rows, &[self.layer])?)
    }
    /// This layer's wave on `storage` (same device): its proposal plane is the
    /// storage's plane for this layer, windowed by `select_window`.
    pub fn on_storage(&self, storage: std::rc::Rc<WindowStorage<'a>>) -> Result<WindowWave<'_, 'a>> {
        let plane = storage.planes.iter().position(|p| p.layer == self.layer)
            .context("window storage has no plane for this layer")?;
        ensure!(storage.input.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "window storage device differs");
        let unit = storage.unit;
        let device = crate::shared::memory::device::Device {
            library: self.library, id: storage.input.buffer.device_id,
        };
        let ordering = || -> Result<_> { Ok(crate::shared::memory::device::Event {
            device, raw: device.run(|| self.library.cuda_event_create_ordering())?,
        }) };
        Ok(WindowWave {
            produced: ordering()?,
            committed: ordering()?,
            #[cfg(test)]
            own_stream: PRIVATE_STREAM.with(|private| private.get()).then(|| -> Result<_> { Ok(LoadStream {
                library: self.library, raw: self.library.cuda_stream_create()? }) }).transpose()?,
            storage,
            plane,
            window: RowWindow::NONE,
            norm: self.library.v41_attention_ops()?,
            kv: self.library.v41_kv()?,
            weights: self,
            staging: HostAllocation::new(self.library, unit * 8)?,
            commit_staging: HostAllocation::new(self.library, commit::staging_bytes(unit))?,
            capacity: unit,
            graph: None,
            graphs: GraphBank::new(None),
            ready: None,
            pending_query: None,
            pending_commit: None,
            unsettled: Vec::new(),
            replica: None,
        })
    }
}
#[cfg(test)]
thread_local! {
    /// Test-only negative control: waves built while set issue on their own
    /// stream instead of their storage's shared one.
    pub(crate) static PRIVATE_STREAM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
struct Prepared {
    snapshot: u64,
    owner: u64,
    chunks: Vec<WindowChunk>,
    offsets: Vec<usize>,
    versions: Vec<u64>,
    rows: usize,
}
pub(crate) struct WindowOutput<'a> {
    pub values: CuteafdDeviceBuffer,
    pub scales: CuteafdDeviceBuffer,
    pub projected: CuteafdDeviceBuffer,
    pub normalized: CuteafdDeviceBuffer,
    pub frequencies: CuteafdDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowBinding {
    snapshot: u64,
    lease: WindowLease,
}
impl WindowBinding {
    pub fn same_pool(self, other: Self) -> bool {
        self.lease.owner == other.lease.owner
    }
}
pub(crate) struct WindowProposal<'a> {
    pub cache: WindowCacheView<'a>,
    pub values: CuteafdDeviceBuffer,
    pub scales: CuteafdDeviceBuffer,
    pub capacity: usize,
    pub request: u64,
    pub layer: usize,
    pub binding: WindowBinding,
    first: u64,
    tokens: usize,
    offset: usize,
    _wave: PhantomData<&'a ()>,
}
impl WindowProposal<'_> {
    /// # Safety
    /// Producer writes are complete or ordered before the replica's stream.
    /// Retain both plane owners until all peer consumers drain.
    pub unsafe fn copy_peer(&self,replica:&crate::shared::memory::proposal_replica::ProposalReplica<'_>,
        stream:*mut c_void)->Result<()> {
        ensure!(replica.format()==crate::shared::memory::proposal_replica::ProposalFormat::WindowFp8,
            "window proposal replica format differs");
        unsafe { replica.copy_rows(self.values,self.scales,self.offset,self.tokens,1,stream) }
    }
    /// Committed end, proposal offset/count and query position. Ring consumers
    /// use positions max(cache.begin,query+1-128)..=query; rows >=end use proposals.
    pub fn metadata(&self, position: u64) -> Result<[u64; 4]> {
        ensure!(
            position >= self.first && position < self.first + self.tokens as u64,
            "query outside window proposal"
        );
        Ok([self.first, self.offset as u64, self.tokens as u64, position])
    }
}
/// One layer's proposal planes in a [`WindowStorage`].
pub(crate) struct WindowPlanes<'a> {
    layer: usize,
    values: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    destinations: DeviceAllocation<'a>,
}
/// The window producers' storage on one GPU (`shared::memory::row_window`
/// contract).
///
/// | Buffer | Class | Why |
/// |---|---|---|
/// | proposal `values`/`scales` (per layer) | row plane | attention reads them on its stream; the commit stores them, in a later hook for decode/verify |
/// | commit `destinations` (per layer) | row plane | the commit's H2D lands there and its store reads them (a later hook for decode/verify) |
/// | `input`, `positions`, `frequencies`, `projected`, `normalized`, GEMM `scratch`/`alpha` | plain scratch at the unit cap | written and consumed by this producer's stream inside one hook; nothing else reads them (`WindowOutput` exposes them to tests only, valid until the next production) |
/// | position `staging`, `commit_staging` | per slot (on the wave) | pinned and rewritten on the host for each unit (#815) |
///
/// Plain scratch is ordered by the storage's one persistent stream. Decode
/// layers share it within a pass; prefill slots share it across their units.
/// Each layer retains its own graph and proposal plane. The pass orders layer
/// L+1's producer after layer L's attention on the chain or a host drain.
/// A separate persistent commit stream stores each layer's proposal after its
/// producer event; a completion event orders that layer's next plane reuse.
/// Independent passes retain separate producer/commit stream pairs.
pub(crate) struct WindowStorage<'a> {
    stream: LoadStream<'a>,
    commit_stream: LoadStream<'a>,
    fp8: V41Fp8Plan<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    input: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    normalized: DeviceAllocation<'a>,
    positions: DeviceAllocation<'a>,
    frequencies: DeviceAllocation<'a>,
    planes: Vec<WindowPlanes<'a>>,
    /// Rows the planes span (the reserve); scratch and plans run up to `unit`.
    rows: usize,
    unit: usize,
}
impl<'a> WindowStorage<'a> {
    /// Plain scratch bytes for units of up to `unit` rows.
    pub fn scratch_bytes(library: &NativeLibrary, unit: u32) -> Result<usize> {
        Ok(library.v41_fp8_matrix_plan_info(unit, 5120, 512)?.scratch_bytes as usize
            + 4 + unit as usize * (10240 + 1024 + 1024 + 8 + 256))
    }
    /// One layer's planes over `rows` rows.
    pub fn plane_bytes(rows: usize) -> usize { rows * (512 + 16 + 8) }
    pub fn device_bytes(library: &NativeLibrary, rows: usize, unit: u32, layers: usize) -> Result<usize> {
        ensure!(unit >= 1 && unit as usize <= rows, "window unit {unit} outside {rows} rows");
        Ok(Self::scratch_bytes(library, unit)? + layers * Self::plane_bytes(rows))
    }
    /// Planes for `layers` over `rows` reserve rows, scratch for units of up to
    /// `unit` rows. Every wave on this storage issues on its one stream.
    pub fn new(library: &'a NativeLibrary, rows: usize, unit: usize, layers: &[usize])
        -> Result<std::rc::Rc<Self>> {
        ensure!(unit >= 1 && unit <= rows && rows <= crate::shared::memory::row_window::MAX_RESERVE_ROWS.max(4096)
            && layers.iter().all(|&l| l < 40), "invalid window storage of {rows} rows, {unit}-row units, layers {layers:?}");
        let fp8 = library.v41_fp8_matrix_plan(unit as u32, 5120, 512)?;
        let value = Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            commit_stream: LoadStream { library, raw: library.cuda_stream_create()? },
            scratch: DeviceAllocation::new(library, fp8.info().scratch_bytes as usize)?,
            alpha: DeviceAllocation::new(library, 4)?,
            input: DeviceAllocation::new(library, unit * 10240)?,
            projected: DeviceAllocation::new(library, unit * 1024)?,
            normalized: DeviceAllocation::new(library, unit * 1024)?,
            positions: DeviceAllocation::new(library, unit * 8)?,
            frequencies: DeviceAllocation::new(library, unit * 256)?,
            planes: layers.iter().map(|&layer| Ok(WindowPlanes {
                layer,
                values: DeviceAllocation::attention_plane(library, rows, 512, "window proposal values")?,
                scales: DeviceAllocation::attention_plane(library, rows, 16, "window proposal scales")?,
                destinations: DeviceAllocation::new(library, rows * 8)?,
            })).collect::<Result<Vec<_>>>()?,
            fp8,
            rows,
            unit,
        };
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        unsafe { value.fp8.initialize_scratch(value.scratch.buffer, value.alpha.buffer, stream.raw)?; }
        unsafe { library.cuda_stream_synchronize(stream.raw)?; }
        Ok(std::rc::Rc::new(value))
    }
    pub fn rows(&self) -> usize { self.rows }
    pub fn unit(&self) -> usize { self.unit }
}
impl Drop for WindowStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: owned stream; its queued work reads only this storage.
        fatal_drain(unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }, "window storage");
        // SAFETY: the persistent commit stream reads this storage's proposal planes.
        fatal_drain(unsafe { self.commit_stream.library.cuda_stream_synchronize(self.commit_stream.raw) }, "window commits");
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only negative control: the commit stages through the producer's
    /// position staging (the pre-#815 bug).
    pub(crate) static SHARED_COMMIT_STAGING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(crate) struct WindowWave<'w, 'a> {
    produced: crate::shared::memory::device::Event<'a>,
    committed: crate::shared::memory::device::Event<'a>,
    /// Negative control only: bypass the shared scratch ordering stream.
    #[cfg(test)]
    own_stream: Option<LoadStream<'a>>,
    storage: std::rc::Rc<WindowStorage<'a>>,
    /// This layer's planes in `storage`.
    plane: usize,
    /// This unit's rows in the planes (`RowWindow::NONE`: from row 0).
    window: RowWindow,
    norm: V41AttentionOps<'a>,
    kv: V41Kv<'a>,
    weights: &'w WindowWeights<'a>,
    /// Per slot: position staging rewritten on the host for each unit.
    staging: HostAllocation<'a>,
    /// Commit destinations and slot ends: their own pinned buffer, never the
    /// producer's position staging, which a queued H2D may still read (#815).
    commit_staging: HostAllocation<'a>,
    capacity: usize,
    graph: Option<(usize, u64)>,
    // Exact cache-owner bindings; containing wave pins workspace and weights.
    graphs: GraphBank<(usize, u64), GraphOwner<'a, ()>>,
    ready: Option<Prepared>,
    pending_query: Option<(Prepared, bool)>,
    pending_commit: Option<PendingCommit>,
    /// Leases a device-ordered direct commit advanced on the host before its
    /// queued write completed: abort drains and revokes them (hunt C #2);
    /// the next commit (after the unit's drain) settles them.
    unsettled: Vec<WindowLease>,
    replica: Option<(std::rc::Rc<replica::WindowReplica<'a>>,
        crate::shared::memory::peer_publication::PeerPublication<'a>)>,
}
impl<'a> WindowWave<'_, 'a> {
    fn stream(&self) -> &LoadStream<'a> {
        #[cfg(test)]
        if let Some(stream) = &self.own_stream { return stream; }
        &self.storage.stream
    }
}
impl WindowWave<'_, '_> {
    /// A wave with its own storage.
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        WindowStorage::device_bytes(library, capacity as usize, capacity, 1)
    }
    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.storage.input.buffer
    }
    fn values(&self) -> CuteafdDeviceBuffer { self.window.apply(self.storage.planes[self.plane].values.buffer, 512) }
    fn scales(&self) -> CuteafdDeviceBuffer { self.window.apply(self.storage.planes[self.plane].scales.buffer, 16) }
    fn destinations(&self) -> CuteafdDeviceBuffer {
        self.window.apply(self.storage.planes[self.plane].destinations.buffer, 8)
    }
    /// Selects this slot's unit rows in the storage's planes, at rest
    /// (`shared::memory::row_window`). Windowed production runs eagerly.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.ready.is_some() || self.pending_query.is_some() || self.pending_commit.is_some(),
            "window producer", "a proposal or commit is pending")?;
        if let Some(range) = window.range() {
            ensure!(range.end <= self.storage.rows && range.len() <= self.capacity,
                "window producer window {range:?} outside {} rows or wider than the {}-row unit",
                self.storage.rows, self.capacity);
        }
        self.window = window;
        Ok(())
    }
    fn synchronize(&self) -> Result<()> {
        // SAFETY: both streams and their referenced storage belong to this wave.
        let produced = unsafe { self.stream().library.cuda_stream_synchronize(self.stream().raw) };
        let committed = unsafe { self.stream().library.cuda_stream_synchronize(self.storage.commit_stream.raw) };
        produced.and(committed)
    }
    fn prepare(&mut self, state: &WindowState<'_>, chunks: &[WindowChunk]) -> Result<Prepared> {
        ensure!(self.pending_query.is_none(), "cache query production pending");
        ensure!(self.pending_commit.is_none(), "window commit pending");
        self.ready = None;
        ensure!(
            state.layer == self.weights.layer
                && state.values.buffer.device_id == self.storage.input.buffer.device_id,
            "window state binding differs"
        );
        ensure!(
            !chunks.is_empty() && chunks.len() <= state.slot_count,
            "invalid window chunks"
        );
        let mut p = Prepared {
            snapshot: next(&NEXT_SNAPSHOT)?,
            owner: state.owner,
            chunks: chunks.to_vec(),
            offsets: vec![],
            versions: vec![],
            rows: 0,
        };
        let mut seen = [false; cuteafd_core::V41_MAX_REQUESTS];
        for c in chunks {
            let slot = state.validate(c.lease)?;
            ensure!(
                !seen[slot] && c.tokens > 0,
                "duplicate or empty window chunk"
            );
            seen[slot] = true;
            ensure!(
                c.position == state.slots[slot].end
                    && c.position <= 1048576
                    && u64::from(c.tokens) <= 1048576 - c.position,
                "window position differs or exceeds context"
            );
            p.offsets.push(p.rows);
            p.versions.push(state.slots[slot].version);
            p.rows += c.tokens as usize;
            ensure!(p.rows <= self.capacity, "window rows exceed capacity");
        }
        ensure!(self.window.admits(p.rows), "window rows differ from the unit's window");
        Ok(p)
    }
    fn upload(&mut self, p: &Prepared) -> Result<()> {
        // Reuse of this layer's proposal plane follows its last off-stream store.
        // Keep this after graph selection's at-rest check and outside capture.
        // SAFETY: events/streams share a device and live through both drains;
        // CUDA treats an event not yet recorded (the first production) as complete.
        unsafe { self.stream().library.cuda_stream_wait_event(self.stream().raw, self.committed.raw)?; }
        let bytes = self.staging.bytes_mut();
        for (i, c) in p.chunks.iter().enumerate() {
            for j in 0..c.tokens as usize {
                let row = p.offsets[i] + j;
                bytes[row * 8..row * 8 + 8].copy_from_slice(&(c.position + j as u64).to_ne_bytes());
            }
        }
        unsafe {
            self.stream().library.copy_host_buffer_h2d_async(
                self.storage.positions.buffer, self.staging.buffer, p.rows * 8, self.stream().raw,
            )
        }
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.storage.positions.buffer,
                self.storage.frequencies.buffer,
                rows as u32,
                self.weights.layer as u32,
                self.stream().raw,
            )?;
            self.storage.fp8.launch(
                self.storage.input.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales.buffer,
                self.storage.scratch.buffer,
                self.storage.alpha.buffer,
                self.storage.projected.buffer,
                rows as u32,
                self.stream().raw,
            )?;
            self.norm.norm(
                self.storage.projected.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                None,
                self.storage.normalized.buffer,
                rows as u32,
                512,
                self.stream().raw,
            )?;
            self.kv.pack(
                self.storage.normalized.buffer,
                Some(self.storage.frequencies.buffer),
                self.values(),
                self.scales(),
                rows,
                self.stream().raw,
            )?;
        }
        Ok(())
    }
    pub(crate) fn capture_startup(&mut self, state: &WindowState<'_>, rows: &[usize]) -> Result<()> {
        ensure!(self.ready.is_none() && self.pending_query.is_none() && self.pending_commit.is_none()
            && !self.window.is_windowed() && state.layer == self.weights.layer,
            "window startup requires an idle matching owner");
        self.stream().require_complete()?;
        let keys: Vec<_> = rows.iter().map(|&row| (row, state.owner)).collect();
        self.graphs.declare_startup(&keys)?;
        for &row in rows {
            ensure!(row > 0 && row <= self.capacity, "window startup row exceeds capacity");
            let library = self.stream().library;
            let stream = self.stream().raw;
            // SAFETY: capture records the exact proposal operations on owned
            // buffers; it neither executes them nor commits cache storage.
            unsafe { library.cuda_graph_begin_capture(stream)?; }
            let queued = unsafe { self.enqueue(row) };
            let captured = unsafe { library.cuda_graph_end_capture(stream) };
            match (queued, captured) {
                (Ok(()), Ok(graph)) => self.insert_graph(row, state.owner, graph)?,
                (Err(error), Ok(graph)) => {
                    unsafe { library.cuda_graph_exec_destroy(graph)?; }
                    return Err(error);
                }
                (Err(error), Err(_)) | (Ok(()), Err(error)) => return Err(error),
            }
        }
        self.graphs.finish_startup()?;
        self.graph = None;
        Ok(())
    }
    /// # Safety
    /// Query and admitted state slots stay alive and immutable until poll_query
    /// completes or abort_query drains. Peer work may use only disjoint slots.
    pub unsafe fn enqueue_query(&mut self, state: &WindowState<'_>, chunks: &[WindowChunk],
        query: &crate::families::deepseek_v41::v41_attention_query::AttentionQueryOutput<'_>) -> Result<()> {
        let prepared = self.prepare(state, chunks)?;
        ensure!(query.binding()?.layer() == self.weights.layer && query.layer == self.weights.layer
            && query.rows == prepared.rows && query.hidden.bytes == prepared.rows * 10240
            && query.hidden.device_id == self.storage.input.buffer.device_id
            && query.tokens()?.iter().copied().eq(chunks.iter().flat_map(|c|
                c.position..c.position + u64::from(c.tokens))), "queued cache query differs");
        let forced_eager = crate::shared::decode_graph::census::eager("window")
            || !super::graph_policy::captures_shape(prepared.rows as u32) || self.window.is_windowed()
            || (!self.graphs.contains(&(prepared.rows, state.owner))
                && !self.graphs.capture_allowed(&(prepared.rows, state.owner)));
        if !forced_eager { self.select_graph(prepared.rows, state.owner, false)?; }
        let capture = !forced_eager && self.graph.is_none();
        let result = (|| -> Result<()> {
            // Reads only the normalized layer input: overlap the query projections.
            unsafe { crate::shared::memory::chain::join_fork(self.stream().library, self.stream().raw)?; }
            unsafe { self.stream().library.copy_d2d_async(self.storage.input.buffer, query.hidden,
                query.hidden.bytes, self.stream().raw)?; }
            self.upload(&prepared)?;
            let dispatch = || unsafe {
                if capture || forced_eager { self.enqueue(prepared.rows) }
                else { self.stream().library.cuda_graph_launch(self.current_graph().unwrap().0, self.stream().raw) }
            };
            if crate::shared::decode_graph::census::enabled() {
                unsafe { self.stream().library.cuda_graph_census_call(self.stream().raw, c"window", prepared.rows,
                    !capture && !forced_eager, crate::shared::decode_graph::census::arm(), dispatch) }
            } else { dispatch() }
        })();
        self.pending_query = Some((prepared, capture));
        if let Err(error) = result { self.abort_query()?; return Err(error); }
        Ok(())
    }
    /// # Safety
    /// Preserve the enqueue_query ownership contract. Capture never suspends.
    pub unsafe fn poll_query(&mut self, state: &WindowState<'_>) -> Result<bool> {
        let result = (|| -> Result<bool> {
            let (prepared, capture) = self.pending_query.as_ref().context("no queued cache query")?;
            ensure!(prepared.owner == state.owner, "queued cache owner differs");
            let chained = crate::shared::memory::chain::active();
            if !chained && !unsafe { self.stream().library.cuda_stream_query(self.stream().raw)? } { return Ok(false); }
            if *capture {
                unsafe { self.stream().library.cuda_graph_begin_capture(self.stream().raw)?; }
                let queued = unsafe { self.enqueue(prepared.rows) };
                let captured = unsafe { self.stream().library.cuda_graph_end_capture(self.stream().raw) };
                let graph = match (queued, captured) {
                    (Ok(()), Ok(graph)) => graph,
                    (Err(error), Ok(graph)) => {
                        unsafe { self.stream().library.cuda_graph_exec_destroy(graph)?; }
                        return Err(error);
                    }
                    (Err(error), Err(_)) | (Ok(()), Err(error)) => return Err(error),
                };
                self.insert_graph(prepared.rows, state.owner, graph)?;
                // Eager proposal is complete. Capture records future work without
                // committing cache state; publish the existing result below.
            }
            // Inside a stage chain the eager/replayed proposal is ordered, not complete.
            if chained { unsafe { crate::shared::memory::chain::finish(self.stream().library, self.stream().raw)?; } }
            self.ready = Some(self.pending_query.take().unwrap().0);
            self.output(state)?;
            Ok(true)
        })();
        if result.is_err() { self.abort_query()?; }
        result
    }
    pub fn abort_query(&mut self) -> Result<()> {
        if self.pending_query.is_some() { self.synchronize()?; }
        self.pending_query = None;
        self.ready = None;
        Ok(())
    }
    /// Consume the exact normalized hidden rows used by a bound attention query.
    /// # Safety
    /// Query producers have completed; no external writes race query or cache
    /// storage. The batch's request identities correspond to these hidden rows.
    pub unsafe fn execute_query<'s>(
        &'s mut self,
        state: &'s WindowState<'_>,
        chunks: &[WindowChunk],
        query: &crate::families::deepseek_v41::v41_attention_query::AttentionQueryOutput<'_>,
    ) -> Result<WindowOutput<'s>> {
        self.ready = None;
        let prepared = self.prepare(state, chunks)?;
        ensure!(query.binding()?.layer() == self.weights.layer
            && query.layer == self.weights.layer
            && query.rows == prepared.rows
            && query.hidden.bytes == prepared.rows * 10240
            && query.hidden.device_id == self.storage.input.buffer.device_id
            && query.tokens()?.iter().copied().eq(chunks.iter().flat_map(|c|
                c.position..c.position + u64::from(c.tokens))),
            "window query layer, rows or positions differ");
        let executed = (|| -> Result<()> {
            unsafe {
                // Reads only the normalized layer input: overlap the query projections.
                crate::shared::memory::chain::join_fork(self.stream().library, self.stream().raw)?;
                self.stream().library.copy_d2d_async(
                    self.storage.input.buffer, query.hidden, query.hidden.bytes, self.stream().raw,
                )?;
            }
            if !super::graph_policy::captures_shape(prepared.rows as u32) || self.window.is_windowed()
                || (!self.graphs.contains(&(prepared.rows, state.owner))
                    && !self.graphs.capture_allowed(&(prepared.rows, state.owner))) {
                // An uncached prefill tail or a windowed unit (graphs bake
                // row-0 plane pointers): run it once eagerly, never capture it.
                // Launch and return: the chain orders consumers (outside a
                // chain scope `finish` drains on the host as before).
                let launched = self.upload(&prepared).and_then(|()| unsafe { self.enqueue(prepared.rows) });
                launched.and(unsafe { crate::shared::memory::chain::finish(self.stream().library, self.stream().raw) })?;
                self.ready = Some(prepared);
                return Ok(());
            }
            self.select_graph(prepared.rows, state.owner, true)?;
            if self.graph.is_none() {
                unsafe { self.capture(state, chunks)?; }
            }
            unsafe { self.replay(state, chunks)?; }
            Ok(())
        })();
        if let Err(error) = executed {
            self.synchronize()?;
            self.ready = None;
            return Err(error);
        }
        self.output(state)
    }
    /// # Safety
    /// Finite attention-input hidden rows follow chunk order on this device,
    /// with producer writes complete. No external writes race this wave/state.
    pub unsafe fn execute<'s>(
        &'s mut self,
        state: &'s WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<WindowOutput<'s>> {
        let p = self.prepare(state, chunks)?;
        let launched = self.upload(&p).and_then(|()| unsafe { self.enqueue(p.rows) });
        launched.and(self.synchronize())?;
        self.ready = Some(p);
        self.output(state)
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained and unpublished after capture.
    pub unsafe fn capture(
        &mut self,
        state: &WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<()> {
        self.ready = None;
        ensure!(self.graph.is_none() && !self.window.is_windowed(), "window graph already captured or windowed");
        unsafe {
            self.execute(state, chunks)?;
        }
        let p = self.prepare(state, chunks)?;
        unsafe {
            self.stream()
                .library
                .cuda_graph_begin_capture(self.stream().raw)?;
        }
        let launched = unsafe { self.enqueue(p.rows) };
        let captured = unsafe { self.stream().library.cuda_graph_end_capture(self.stream().raw) };
        match (launched, captured) {
            (Ok(()), Ok(g)) => self.insert_graph(p.rows, state.owner, g)?,
            (Err(e), Ok(g)) => {
                unsafe {
                    self.stream().library.cuda_graph_exec_destroy(g)?;
                }
                return Err(e);
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => return Err(e),
        }
        Ok(())
    }
    /// # Safety
    /// Same input contract as execute; capture fixes owner and live row count.
    pub unsafe fn replay<'s>(
        &'s mut self,
        state: &'s WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<WindowOutput<'s>> {
        let p = self.prepare(state, chunks)?;
        ensure!(!self.window.is_windowed(), "windowed window production is eager");
        let (g, rows, owner) = self.current_graph().context("window graph missing")?;
        ensure!(
            rows == p.rows && owner == state.owner,
            "window capture binding differs"
        );
        let launched = self.upload(&p).and_then(|()| {
            let eager = crate::shared::decode_graph::census::eager("window");
            let dispatch = || unsafe {
                if eager { self.enqueue(p.rows) } else { self.stream().library.cuda_graph_launch(g, self.stream().raw) }
            };
            if crate::shared::decode_graph::census::enabled() {
                unsafe { self.stream().library.cuda_graph_census_call(self.stream().raw, c"window", p.rows, !eager,
                    crate::shared::decode_graph::census::arm(), dispatch) }
            } else { dispatch() }
        });
        launched.and(unsafe { crate::shared::memory::chain::finish(self.stream().library, self.stream().raw) })?;
        self.ready = Some(p);
        self.output(state)
    }
    fn validate_ready(&self, state: &WindowState<'_>) -> Result<&Prepared> {
        let p = self.ready.as_ref().context("window output unpublished")?;
        ensure!(p.owner == state.owner, "window output owner differs");
        for (i, c) in p.chunks.iter().enumerate() {
            let slot = state.validate(c.lease)?;
            ensure!(
                state.slots[slot].version == p.versions[i] && state.slots[slot].end == c.position,
                "stale window output"
            );
        }
        Ok(p)
    }
    pub fn output<'s>(&'s self, state: &'s WindowState<'_>) -> Result<WindowOutput<'s>> {
        let rows = self.validate_ready(state)?.rows;
        Ok(WindowOutput {
            values: slice(self.values(), 0, rows * 512),
            scales: slice(self.scales(), 0, rows * 16),
            projected: slice(self.storage.projected.buffer, 0, rows * 1024),
            normalized: slice(self.storage.normalized.buffer, 0, rows * 1024),
            frequencies: slice(self.storage.frequencies.buffer, 0, rows * 256),
            _owner: PhantomData,
        })
    }
    pub fn proposal<'s>(
        &'s self,
        state: &'s WindowState<'_>,
        lease: WindowLease,
    ) -> Result<WindowProposal<'s>> {
        let p = self.validate_ready(state)?;
        let i = p
            .chunks
            .iter()
            .position(|c| c.lease == lease)
            .context("request absent from window wave")?;
        let c = p.chunks[i];
        let o = self.output(state)?;
        Ok(WindowProposal {
            cache: state.view(lease)?,
            values: o.values,
            scales: o.scales,
            capacity: p.rows,
            request: state.request_id(lease)?,
            layer: self.weights.layer,
            binding: WindowBinding {
                snapshot: p.snapshot,
                lease,
            },
            first: c.position,
            tokens: c.tokens as usize,
            offset: p.offsets[i],
            _wave: PhantomData,
        })
    }
    /// Validate an enclosing cache transaction's exact request order/proposal.
    pub(crate) fn validate_batch(
        &self,
        state: &WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<()> {
        let prepared = if self.pending_commit.is_some() { self.validate_pending_commit(state)? }
            else { self.validate_ready(state)? };
        ensure!(
            prepared.chunks.len() == chunks.len()
                && prepared.chunks.iter().zip(chunks).all(|(a, b)|
                    a.lease == b.lease && a.position == b.position && a.tokens == b.tokens),
            "window transaction proposal differs"
        );
        Ok(())
    }
    /// Switch only after prior launches finish. Graphs bind the cache owner and
    /// lane workspace, while current request descriptors are uploaded on replay.
    fn select_graph(&mut self, rows: usize, owner: u64, drain: bool) -> Result<()> {
        if self.graph == Some((rows, owner)) { return Ok(()); }
        ensure!(self.pending_query.is_none() && self.pending_commit.is_none(),
            "cannot switch a pending cache producer graph");
        if self.graph.is_some_and(|(_, o)| o != owner)
            || self.graphs.count(|(_, o)| *o != owner) > 0 {
            return self.clear_graph_inner(drain);
        }
        self.ready = None;
        if drain { self.synchronize()?; } else { self.stream().require_complete()?; }
        self.graph = None;
        if rows > 64 && self.graphs.count(|&(r, _)| r > 64 && r != rows) >= crate::shared::decode_graph::LARGE_SHAPES {
            // Keep the full prefill chunk and decoder replay across decode rounds.
            self.graphs.retire_oldest(|&(r, _)| r > 64 && r != rows);
        }
        drop(self.graphs.drain_retired(|| Ok(()))?);
        self.graph = self.graphs.launch(&(rows, owner)).map(|_| (rows, owner));
        if self.graph.is_none() {
            tracing::debug!(target: "cuteafd::graph_capture", site="window", layer=self.weights.layer,
                rows, owner, device=self.storage.input.buffer.device_id, bank=self as *const Self as usize,
                retained=self.graphs.len(), "graph cache miss");
        }
        Ok(())
    }
    fn current_graph(&self) -> Option<(*mut c_void, usize, u64)> {
        let (rows, owner) = self.graph?;
        Some((self.graphs.get(&(rows, owner))?.raw, rows, owner))
    }
    fn insert_graph(&mut self, rows: usize, owner: u64, raw: *mut c_void) -> Result<()> {
        let device = self.storage.input.buffer.device_id;
        // SAFETY: the wave pins weights/workspace; Drop drains before releasing fields.
        let graph = unsafe { GraphOwner::new(self.stream().library, device, raw, ())? };
        self.graphs.insert((rows, owner), graph, None);
        self.graph = Some((rows, owner));
        tracing::debug!(target: "cuteafd::graph_capture", site="window", layer=self.weights.layer,
            rows, owner, device, bank=self as *const Self as usize, "window graph captured");
        Ok(())
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.clear_graph_inner(true)
    }
    fn clear_graph_inner(&mut self, drain: bool) -> Result<()> {
        ensure!(self.pending_query.is_none(), "cannot clear pending cache query");
        ensure!(self.pending_commit.is_none(), "cannot reset a pending window commit");
        self.ready = None;
        if drain { self.synchronize()?; } else { self.stream().require_complete()?; }
        self.graph = None;
        self.graphs.retire_all();
        drop(self.graphs.drain_retired(|| Ok(()))?);
        Ok(())
    }
}
impl<'w, 'a> WindowWave<'w, 'a> {
    /// The commit's pinned host staging: its own buffer, never the
    /// producer's position staging, which a queued H2D may still read (#815).
    /// The test-only negative control reuses the producer's, as before the fix.
    fn commit_host(&mut self) -> &mut HostAllocation<'a> {
        #[cfg(test)]
        if SHARED_COMMIT_STAGING.with(|shared| shared.get()) { return &mut self.staging; }
        &mut self.commit_staging
    }
}
impl Drop for WindowWave<'_, '_> {
    fn drop(&mut self) {
        fatal_drain(self.synchronize(), "pending window writes");
        self.pending_query = None;
        self.pending_commit = None;
        fatal_drain(self.clear_graph(), "window graph");
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    #[test]
    fn decoder_replay_window_lease_boundaries() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_WINDOW_REPLAY_LIBRARY") else {
            eprintln!("skip GPU window replay test: CUTEAFD_WINDOW_REPLAY_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let mut state = WindowState::new(&library, 20, 16, WindowState::device_bytes(20, 16)?)?;
        let mut encoder = WindowState::new(&library, 19, 1, WindowState::device_bytes(19, 1)?)?;
        let encoder_lease = encoder.begin_request(0, 999)?;
        assert!(encoder.begin_replay(encoder_lease, 128).is_err());
        assert!(state.begin_replay(encoder_lease, 128).is_err());
        for slot in 0..16 {
            let lease = state.begin_request(slot, slot as u64 + 1)?;
            let before = state.slots[slot].version;
            let position = [0, 1, 127, 128, 16384, 1048576][slot % 6];
            assert!(state.begin_replay(lease, 1048577).is_err());
            assert_eq!(state.end(lease)?, 0);
            state.begin_replay(lease, position)?;
            assert_eq!(state.slots[slot].version, before + 1);
            assert!(state.begin_replay(lease, position).is_err());
            let view = state.view(lease)?;
            assert_eq!((view.begin, view.end), (position, position));
            let mut bytes = [0; 8];
            library.copy_d2h(&mut bytes, view.device_end)?;
            assert_eq!(u64::from_ne_bytes(bytes), position);
            state.release(lease)?;
            assert!(state.begin_replay(lease, 0).is_err());
            let replacement = state.begin_request(slot, slot as u64 + 101)?;
            assert!(state.view(lease).is_err());
            assert_eq!((state.view(replacement)?.begin, state.end(replacement)?), (0, 0));
            state.release(replacement)?;
        }
        eprintln!("PASS 16 decoder replay slots: bounds, device ends, versions, stale leases, encoder/foreign guards and reuse");
        Ok(())
    }
}

#[cfg(test)]
impl WindowWeights<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.tensors.visit(prefix, f);
        f(format!("{prefix}packed_scale"), self.scales.buffer);
    }
}
