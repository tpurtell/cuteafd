//! One lane's learned index query and retained source-20 candidates.
use crate::families::deepseek_v41::v41_attention_query::AttentionQueryOutput;
use crate::families::deepseek_v41::v41_backbone_cache::{CacheAttention, CachePlacement};
use crate::shared::memory::device::{Device, DeviceOwner};
mod placement;
use crate::families::deepseek_v41::v41_index_query::{IndexQueryStorage, IndexQueryWave, IndexQueryWeights};
use crate::families::deepseek_v41::v41_index_selection::{IndexSelectionOutput, IndexSelectionStorage, IndexSelectionWave};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::OfficialV41Catalog;
use std::rc::Rc;
use crate::shared::memory::row_window::RowWindow;

const LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
pub(crate) struct IndexLaneWeights<'a> {
    library: &'a NativeLibrary,
    weights: Vec<DeviceOwner<'a, IndexQueryWeights<'a>>>,
    placement: Option<CachePlacement>,
}
impl<'a> IndexLaneWeights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        for weights in &self.weights { weights.inventory(inventory)?; }
        Ok(())
    }
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        LAYERS.into_iter().try_fold(0usize, |total, layer| {
            total
                .checked_add(IndexQueryWeights::device_bytes(library, catalog, layer)?)
                .context("index weight budget overflow")
        })
    }
    #[cfg(test)]
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "index lane weights exceed budget"
        );
        Self::load_placed(library, catalog, staging, None)
    }
}

/// One per GPU: selected IDs and source-20 candidate blocks span the reserve
/// because later hooks reuse them. Query outputs are guarded unit-cap scratch.
/// Source/reindex share a selection stream and guard; stream order protects
/// their plain scratch within a unit, release/acquire protects fork readers
/// across units. Bindings, phases and pinned metadata stay per slot.
pub(crate) struct IndexLaneStorage<'a> {
    query: Rc<IndexQueryStorage<'a>>,
    source: Rc<IndexSelectionStorage<'a>>,
    reindex: Option<Rc<IndexSelectionStorage<'a>>>,
    device: Option<i32>,
}
pub(crate) struct IndexLane<'w, 'a> {
    weights: &'w IndexLaneWeights<'a>,
    source: IndexSelectionWave<'a>,
    reindex: Option<IndexSelectionWave<'a>>,
    query: IndexQueryWave<'w, 'a>,
    next: usize,
    device: Option<i32>,
    ready: Option<usize>,
    invalid: bool,
}
impl<'w, 'a> IndexLane<'w, 'a> {
    pub(crate) fn startup_selections(&self) -> Vec<Option<cuteafd_ffi::CuteafdDeviceBuffer>> {
        (0..40).map(|layer| {
            if layer < 2 { None }
            else if layer < 24 { Some(self.source.startup_outputs().0) }
            else { self.reindex.as_ref().map(|wave| wave.startup_outputs().0) }
        }).collect()
    }
    pub(crate) fn capture_startup(&mut self, rows: &[usize],
        mut source: impl FnMut(usize, usize) -> Result<crate::families::deepseek_v41::v41_compressor::IndexScoreGeometry>) -> Result<()> {
        ensure!(!self.invalid && self.ready.is_none(), "startup index lane requires idle owner");
        let layers: Vec<_> = (0..LAYERS.len()).filter(|&i|
            self.device.is_none_or(|gpu| self.weights.weights[i].device.id == gpu)).collect();
        let original = self.next;
        let result = (|| -> Result<()> {
            let shared = self.source.startup_outputs().1;
            for &i in &layers {
                self.query.rebind(&self.weights.weights[i])?;
                for &row in rows {
                    self.query.capture_startup(row as u32)?;
                    let query = self.query.startup_geometry(row)?;
                    let geometry = source(if LAYERS[i] >= 20 { 20 } else { LAYERS[i] }, row)?;
                    // SAFETY: closure supplies real stable source planes; recording never publishes a query.
                    unsafe {
                        if LAYERS[i] > 20 {
                            self.reindex.as_mut().context("startup follower absent")?
                                .capture_startup(&query, geometry, Some(shared), row)?;
                        } else { self.source.capture_startup(&query, geometry, None, row)?; }
                    }
                }
            }
            Ok(())
        })();
        let restored = if original < LAYERS.len() { self.query.rebind(&self.weights.weights[original]) } else { Ok(()) };
        result.and(restored)
    }
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<[usize; 3]> {
        Ok([
            IndexQueryWave::device_bytes(library, capacity)?,
            IndexSelectionWave::device_bytes(capacity as usize)?,
            // The reindex wave shares the source's scores and top-k scratch
            // (they run one after the other in a lane), as the placed lanes do.
            IndexSelectionWave::shared_device_bytes(capacity as usize)?,
        ])
    }
    pub fn new(weights: &'w IndexLaneWeights<'a>, capacity: u32, budget: usize) -> Result<Self> {
        Self::new_inner(weights, capacity, budget, None)
    }
    fn new_inner(weights: &'w IndexLaneWeights<'a>, capacity: u32, budget: usize,
        device: Option<i32>) -> Result<Self> {
        let bytes = match device {
            Some(gpu) => Self::placed_workspace_bytes(weights.library,
                weights.placement.context("placed index weights absent")?,capacity,gpu as usize)?,
            None => Self::workspace_bytes(weights.library, capacity)?,
        };
        let total = bytes.iter().try_fold(0usize, |n, &b| {
            n.checked_add(b).context("index workspace budget overflow")
        })?;
        ensure!(total <= budget, "index lane workspace exceeds budget");
        ensure!(
            weights.weights.len() == 8,
            "index lane requires all eight weight owners"
        );
        ensure!(weights.placement.is_some() == device.is_some(), "index workspace placement differs from weights");
        let first = Self::next_owned(weights, device, 0);
        ensure!(first < LAYERS.len(), "GPU has no index producers");
        Self::on_storage(weights, Self::storage_inner(weights, capacity, capacity, device, &[weights.library.cuda_get_device()?])?)
    }
    pub fn storage_bytes(library: &NativeLibrary, rows: u32, unit: u32) -> Result<[usize; 3]> {
        Ok([IndexQueryWave::storage_bytes(library, rows, unit)?,
            IndexSelectionWave::storage_bytes(rows as usize, unit as usize)?,
            IndexSelectionWave::follower_storage_bytes(rows as usize, unit as usize)?])
    }
    pub fn storage(weights: &'w IndexLaneWeights<'a>, rows: u32, unit: u32, readers: &[i32]) -> Result<Rc<IndexLaneStorage<'a>>> {
        ensure!(weights.placement.is_none(), "unplaced index storage requires unplaced weights");
        Self::storage_inner(weights, rows, unit, None, readers)
    }
    fn storage_inner(weights: &'w IndexLaneWeights<'a>, rows: u32, unit: u32,
        device: Option<i32>, readers: &[i32]) -> Result<Rc<IndexLaneStorage<'a>>> {
        ensure!(weights.weights.len() == LAYERS.len() && weights.placement.is_some() == device.is_some(),
            "index storage weight placement differs");
        let first = Self::next_owned(weights, device, 0);
        ensure!(first < LAYERS.len(), "GPU has no index producers");
        let source = IndexSelectionWave::storage(weights.library, rows as usize, unit as usize, readers)?;
        let reindex = if device.is_none_or(|gpu| weights.placement.is_some_and(|p| p.attention(20).ok() == Some(gpu as usize))) {
            Some(IndexSelectionWave::follower_storage(&source)?)
        } else { None };
        Ok(Rc::new(IndexLaneStorage {
            query: weights.weights[first].storage(rows, unit, readers)?, source, reindex, device,
        }))
    }
    pub fn on_storage(weights: &'w IndexLaneWeights<'a>, storage: Rc<IndexLaneStorage<'a>>) -> Result<Self> {
        ensure!(weights.weights.len() == LAYERS.len() && weights.placement.is_some() == storage.device.is_some(),
            "index lane storage placement differs");
        let first = Self::next_owned(weights, storage.device, 0);
        ensure!(first < LAYERS.len(), "GPU has no index producers");
        Ok(Self {
            weights, query: weights.weights[first].on_storage(storage.query.clone())?,
            source: IndexSelectionWave::on_storage(storage.source.clone())?,
            reindex: storage.reindex.as_ref().map(|s| IndexSelectionWave::on_storage(s.clone())).transpose()?,
            next: first, device: storage.device, ready: None, invalid: false,
        })
    }
    /// Call after the attention hook's final fork reader joined the chain,
    /// possibly on the peer GPU. Source/reindex share one release event.
    pub(crate) fn release_scratch(&self) -> Result<()> {
        self.query.release_scratch()?;
        self.source.release_scratch()
    }
    /// Select at rest, after the previous unit and all its readers have drained.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.invalid || self.ready.is_some(), "index lane", "a restart is incomplete or an output unconsumed")?;
        self.query.select_window(window.clone())?;
        self.source.select_window(window.clone())?;
        if let Some(reindex) = &mut self.reindex { reindex.select_window(window)?; }
        Ok(())
    }
    fn next_owned(weights: &IndexLaneWeights<'_>, device: Option<i32>, start: usize) -> usize {
        if device.is_none() { return start; }
        (start..LAYERS.len()).find(|&i| device.is_none_or(|id| weights.weights[i].device.id == id))
            .unwrap_or(LAYERS.len())
    }
    fn advance(&mut self) {
        self.next = Self::next_owned(self.weights, self.device, self.next + 1);
    }
    /// Begin a new batch after all consumers finish; also recovers failed work.
    pub fn enable_small_graph_shapes(&mut self) {}
    pub fn restart(&mut self) -> Result<()> {
        self.restart_at(0)
    }
    /// Replay begins at decoder source 20; earlier encoder selections are absent.
    pub fn restart_decoder(&mut self) -> Result<()> {
        self.restart_at(3)
    }
    fn restart_at(&mut self, first: usize) -> Result<()> {
        self.invalid = true;
        self.ready = None;
        self.source.finish_audit()?;
        if let Some(reindex) = &mut self.reindex { reindex.finish_audit()?; }
        self.source.restart()?;
        if let Some(reindex) = &mut self.reindex { reindex.restart()?; }
        let first = Self::next_owned(self.weights, self.device, first);
        if first < LAYERS.len() { self.query.rebind(&self.weights.weights[first])?; }
        self.next = first;
        self.invalid = false;
        Ok(())
    }
    /// # Safety
    /// Query and cache belong to the same admitted batch. All producers have
    /// completed and no external writes race these owners. Call once per index
    /// producer in layer order; intermediate attention layers reuse output().
    pub unsafe fn select(
        &mut self,
        query: &AttentionQueryOutput<'_>,
        cache: &CacheAttention<'_>,
    ) -> Result<()> {
        if crate::shared::memory::chain::active() {
            // Stream-ordered: the queued projection and selection finish on the
            // chain immediately, so the host never waits here.
            unsafe { self.enqueue_projection(query)?; }
            ensure!(self.poll_projection()?, "chained index projection still pending");
            unsafe { self.enqueue_selection(cache)?; }
            ensure!(self.poll_selection()?, "chained index selection still pending");
            return Ok(());
        }
        let valid = !std::mem::replace(&mut self.invalid, true);
        self.ready = None;
        ensure!(
            valid && LAYERS.get(self.next) == Some(&query.layer)
                && self.device.is_none_or(|id| query.hidden.device_id == id),
            "index producer order differs; restart lane"
        );
        // The direct index path drains its own streams and uses legacy-stream
        // uploads; order it after any chained producers on the host.
        crate::shared::memory::chain::settle(self.weights.library)?;
        self.query.rebind(&self.weights.weights[self.next])?;
        let requests = cache.selection_requests()?;
        let projected = unsafe { self.query.execute_attention(query)? };
        if query.layer <= 20 {
            unsafe {
                self.source.execute(&projected, &requests, None)?;
            }
        } else {
            let candidates = self.source.output()?;
            unsafe {
                self.reindex.as_mut().context("decoder reindex workspace absent")?
                    .execute(&projected, &requests, Some(&candidates))?;
            }
        }
        self.ready = Some(query.layer);
        self.advance();
        self.invalid = false;
        Ok(())
    }
    /// # Safety
    /// Retain main query/cache inputs until selection completes or abort drains.
    pub unsafe fn enqueue_projection(&mut self, query: &AttentionQueryOutput<'_>) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        self.ready = None;
        ensure!(valid && LAYERS.get(self.next) == Some(&query.layer)
                && self.device.is_none_or(|id| query.hidden.device_id == id), "queued index order differs");
        self.query.rebind(&self.weights.weights[self.next])?;
        unsafe { self.query.enqueue_attention(query) }
    }
    pub fn poll_projection(&mut self) -> Result<bool> { self.query.poll_pending() }
    /// # Safety
    /// Preserve completed projection and cache proposal storage through selection.
    pub unsafe fn enqueue_selection(&mut self, cache: &CacheAttention<'_>) -> Result<()> {
        ensure!(self.invalid, "queued index projection absent");
        let query = self.query.output()?;
        ensure!(LAYERS.get(self.next) == Some(&query.layer), "queued selection layer differs");
        let requests = cache.selection_requests()?;
        if query.layer <= 20 {
            unsafe { self.source.enqueue_selection(&query, &requests, None) }
        } else {
            let shared = self.source.output()?;
            unsafe { self.reindex.as_mut().context("decoder reindex workspace absent")?.enqueue_selection(&query, &requests, Some(&shared)) }
        }
    }
    pub fn poll_selection(&mut self) -> Result<bool> {
        ensure!(self.invalid, "queued index selection absent");
        let layer = *LAYERS.get(self.next).context("queued index order exhausted")?;
        let ready = if layer <= 20 { self.source.poll_pending()? } else { self.reindex.as_mut().context("decoder reindex workspace absent")?.poll_pending()? };
        if ready { self.ready = Some(layer); self.advance(); self.invalid = false; }
        Ok(ready)
    }
    pub fn abort_pending(&mut self) -> Result<()> {
        // Drain consumers before their projected inputs can be recycled.
        let reindex = self.reindex.as_mut().map_or(Ok(()), |wave| wave.abort_pending());
        let result = reindex.and(self.source.abort_pending()).and(self.query.abort_pending());
        self.invalid = true; self.ready = None;
        result
    }
    /// Revalidate exact proposal snapshots and row order for the attention layer.
    pub fn output(
        &self,
        layer: usize,
        cache: &CacheAttention<'_>,
    ) -> Result<IndexSelectionOutput<'_>> {
        ensure!(!self.invalid, "index lane requires restart");
        let producer = self.ready.context("index lane output unavailable")?;
        let output = if producer <= 20 {
            self.source.output()?
        } else {
            self.reindex.as_ref().context("decoder reindex workspace absent")?.output()?
        };
        let requests = cache.selection_requests()?;
        let bindings = requests
            .iter()
            .flat_map(|r| r.positions.iter().map(|&p| (r.proposal.binding(), p)))
            .collect::<Vec<_>>();
        output.validate_attention(layer, &bindings)?;
        Ok(output)
    }
}

impl Drop for IndexLane<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.abort_pending() { tracing::error!(%error, "draining index lane"); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_index_lane_budgets_reject_before_allocation() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LANE_PLAN_LIBRARY") else {
            eprintln!("skip index planning test: CUTEAFD_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("CUTEAFD_LANE_PLAN_MODEL")
            .context("CUTEAFD_LANE_PLAN_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model),
        )?;
        let bytes = IndexLaneWeights::device_bytes(&library, &catalog)?;
        assert_eq!(bytes, 45_916_160);
        assert!(
            IndexLaneWeights::load(&library, &catalog, bytes - 1, 1024 * 1024)
                .err()
                .unwrap()
                .to_string()
                .contains("weights exceed budget")
        );
        let empty = IndexLaneWeights {
            library: &library,
            weights: Vec::new(),
            placement: None,
        };
        for capacity in [1, 80, 4096] {
            let groups = IndexLane::workspace_bytes(&library, capacity)?;
            let total = groups.iter().sum::<usize>();
            assert!(IndexLane::new(&empty, capacity, total - 1)
                .err()
                .unwrap()
                .to_string()
                .contains("workspace exceeds budget"));
            assert!(IndexLane::new(&empty, capacity, total)
                .err()
                .unwrap()
                .to_string()
                .contains("all eight weight owners"));
            eprintln!("index lane capacity={capacity} workspace_groups={groups:?} total={total}");
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(IndexLane::workspace_bytes(&library, capacity).is_err());
        }
        Ok(())
    }
}

#[cfg(test)]
impl IndexLaneWeights<'_> {
    pub(crate) fn visit(&self, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        for (i, w) in self.weights.iter().enumerate() { w.visit(&format!("gpu{}.index{i}.", w.device.id), f); }
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::families::deepseek_v41::v41_attention_query::AttentionQueryWeights;
    use crate::families::deepseek_v41::v41_compressor::{CompressorChunk, CompressorState, CompressorWeights};
    use crate::families::deepseek_v41::{v41_index_query, v41_index_selection};
    use crate::families::deepseek_v41::v41_index_selection::SelectionRequest;
    use crate::shared::memory::{chain, DeviceAllocation};
    use crate::shared::memory::device::Stream;
    use cuteafd_ffi::CuteafdDeviceBuffer;

    #[derive(Clone, Copy, Debug)]
    enum Scratch { Query, Selection }
    impl Scratch {
        fn skip(self, skip: bool) {
            match self {
                Self::Query => v41_index_query::SKIP_ACQUIRE.with(|value| value.set(skip)),
                Self::Selection => v41_index_selection::SKIP_ACQUIRE.with(|value| value.set(skip)),
            }
        }
        fn buffers(self, lane: &IndexLane<'_, '_>) -> Result<Vec<CuteafdDeviceBuffer>> {
            Ok(match self {
                Self::Query => {
                    let q = lane.query.output()?;
                    vec![q.packed, q.scales, q.head_weights]
                }
                Self::Selection => vec![if lane.query.output()?.layer == 20 {
                    lane.source.output()?.selected
                } else { lane.reindex.as_ref().unwrap().output()?.selected }],
            })
        }
        fn stream(self, lane: &IndexLane<'_, '_>) -> *mut std::ffi::c_void {
            match self {
                Self::Query => lane.query.scratch_stream(),
                Self::Selection => lane.source.scratch_stream(),
            }
        }
    }
    struct ResetSkip(Scratch);
    impl Drop for ResetSkip { fn drop(&mut self) { self.0.skip(false); } }

    // Each arm exercises two independent chains, widths 13/11, source20 and
    // reindex24. Query and selection guards are tested independently: the other
    // guard must not accidentally serialize the negative control's writer.
    fn run(focus: Scratch, guarded: bool, peer: bool) -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(
            &std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices: &[i32] = if peer { &[0, 1] } else { &[0] };
        let reader_device = Device { library: &library, id: i32::from(peer) };
        if peer {
            library.cuda_enable_peer(1)?;
            reader_device.run(|| library.cuda_enable_peer(0))?;
        }
        let weights = if peer {
            let placement = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer < 20)))?;
            IndexLaneWeights::load_distributed(&library, &catalog, placement,
                IndexLaneWeights::distributed_device_bytes(&library, &catalog, placement)?, 1 << 20)?
        } else { IndexLaneWeights::load(&library, &catalog, usize::MAX, 1 << 20)? };
        let shared = if peer { IndexLane::storage_on_device(&weights, 48, 16, 0, devices)? }
            else { Device { library: &library, id: 0 }.own(|| IndexLane::storage(&weights, 48, 16, devices))? };
        let mut slots = (0..2).map(|_| {
            if peer { IndexLane::on_storage_on_device(&weights, shared.get().clone()) }
            else { Device { library: &library, id: 0 }.own(|| IndexLane::on_storage(&weights, shared.get().clone())) }
        }).collect::<Result<Vec<_>>>()?;
        let counts = [13usize, 11];
        let context = [1024usize, 896];
        let tokens = std::array::from_fn::<_, 2, _>(|i|
            ((context[i] - counts[i]) as u64..context[i] as u64).collect::<Vec<_>>());
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::rows(i * 16..i * 16 + counts[i]))?;
        }
        let input = |seed: usize, rows: usize| -> Vec<u8> {
            (0..rows * 5120).flat_map(|i| {
                let x = ((i * 7 + seed * 11) % 29) as f32 / 32.0 - 0.4;
                ((x.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect()
        };
        let cw = CompressorWeights::load(&library, &catalog, 20, usize::MAX, 1 << 20)?;
        let mut state = CompressorState::new(&library, 20, 2, 2, usize::MAX)?;
        let leases = [state.begin_request(0, 1)?, state.begin_request(1, 2)?];
        let mut compressors = [cw.wave(context[0], usize::MAX)?, cw.wave(context[1], usize::MAX)?];
        for i in 0..2 {
            library.copy_h2d(compressors[i].input(), &input(i, context[i]))?;
            // SAFETY: private initialized input and retained state; execution drains.
            unsafe { compressors[i].execute(&state, &[CompressorChunk {
                lease: leases[i], position: 0, tokens: context[i] as u32,
            }])?; }
        }
        let proposals = [compressors[0].index_proposal(&state, leases[0])?,
            compressors[1].index_proposal(&state, leases[1])?];
        let query_weights = [20usize, 24].into_iter().map(|layer| {
            AttentionQueryWeights::load(&library, &catalog, layer, usize::MAX, 1 << 20)
        }).collect::<Result<Vec<_>>>()?;
        let mut queries = Vec::new();
        for (layer, weight) in query_weights.iter().enumerate() {
            let mut units = Vec::new();
            for i in 0..2 {
                let mut query = weight.wave(16, usize::MAX)?;
                library.copy_h2d(query.input(), &input(i + layer * 7, counts[i]))?;
                // SAFETY: finite private input; tokens match rows.
                unsafe { query.execute_tokens(&tokens[i])?; }
                units.push(query);
            }
            queries.push(units);
        }
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; library.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        // Private references include each source/follower output and the source
        // blocks that must survive both units' later reindex hooks.
        let mut expected = Vec::new();
        let mut blocks = Vec::new();
        for i in 0..2 {
            let mut private = if peer { IndexLane::new_on_device(&weights, 16, usize::MAX, 0)? }
                else { Device { library: &library, id: 0 }.own(|| IndexLane::new(&weights, 16, usize::MAX))? };
            let requests = [SelectionRequest { proposal: &proposals[i], positions: &tokens[i] }];
            let mut layers = Vec::new();
            for layer in 0..2 {
                private.query.rebind(&weights.weights[3 + layer])?;
                // SAFETY: completed input, exclusive private storage.
                unsafe { private.query.execute_attention(&queries[layer][i].output()?)?; }
                if layer == 0 {
                    let lane = private.get_mut();
                    unsafe { lane.source.execute(&lane.query.output()?, &requests, None)?; }
                    blocks.push(read(private.source.output()?.candidate_blocks().unwrap())?);
                } else {
                    let lane = private.get_mut();
                    unsafe { lane.reindex.as_mut().unwrap().execute(&lane.query.output()?,
                        &requests, Some(&lane.source.output()?))?; }
                }
                layers.push(focus.buffers(&private)?.into_iter().map(&read).collect::<Result<Vec<_>>>()?);
            }
            expected.push(layers);
        }
        let reader = Stream::new(reader_device)?;
        let copies = expected[0][0].iter().map(|bytes|
            reader_device.own(|| DeviceAllocation::new(&library, bytes.len(), crate::shared::memory::TEST)))
            .collect::<Result<Vec<_>>>()?;
        let copy_outputs = |buffers: Vec<CuteafdDeviceBuffer>| -> Result<()> {
            reader_device.run(|| {
                // SAFETY: all allocations remain live through both chains; the
                // reader joins its producer and publishes its completion.
                unsafe {
                    chain::join(&library, reader.raw)?;
                    for (copy, source) in copies.iter().zip(buffers) {
                        if peer { library.copy_peer_async(copy.buffer, source, source.bytes, reader.raw)?; }
                        else { library.copy_d2d_async(copy.buffer, source, source.bytes, reader.raw)?; }
                    }
                    chain::finish(&library, reader.raw)
                }
            })
        };
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Warm exact enqueue/poll, rebind, snapshot, and peer-copy calls on the
        // very same plans/streams and both widths before any gate exists.
        eprintln!("index guard {focus:?} guarded={guarded} peer={peer}: warm exact calls");
        for i in 0..2 {
            let warm = chain::StageChain::on_devices(&library, devices)?;
            for layer in 0..2 {
                runtime.block_on(warm.handle().scope(async {
                    let slot = slots[i].get_mut();
                    slot.query.rebind(&weights.weights[3 + layer])?;
                    unsafe { slot.query.enqueue_attention(&queries[layer][i].output()?)?; }
                    ensure!(slot.query.poll_pending()?);
                    let requests = [SelectionRequest { proposal: &proposals[i], positions: &tokens[i] }];
                    if layer == 0 {
                        unsafe { slot.source.enqueue_selection(&slot.query.output()?, &requests, None)?; }
                        ensure!(slot.source.poll_pending()?);
                    } else {
                        let source = slot.source.output()?;
                        let follower = slot.reindex.as_mut().unwrap();
                        unsafe { follower.enqueue_selection(&slot.query.output()?, &requests, Some(&source))?; }
                        ensure!(follower.poll_pending()?);
                    }
                    copy_outputs(focus.buffers(slot)?)?;
                    slot.release_scratch()
                }))?;
                warm.drain()?;
            }
            slots[i].restart()?;
        }
        let chains = [chain::StageChain::on_devices(&library, devices)?,
            chain::StageChain::on_devices(&library, devices)?];
        for layer in 0..2 {
            let gate = reader_device.run(|| unsafe {
                cuteafd_ffi::test_support::CudaStreamGate::new(&library, reader.raw)
            })?;
            eprintln!("index guard {focus:?}: gate closed, enqueue A layer={layer}");
            runtime.block_on(chains[0].handle().scope(async {
                let slot = slots[0].get_mut();
                let i = 0;
                slot.query.rebind(&weights.weights[3 + layer])?;
                // SAFETY: queries/proposals persist through both units and readers.
                unsafe { slot.query.enqueue_attention(&queries[layer][i].output()?)?; }
                ensure!(slot.query.poll_pending()?, "projection blocked on host");
                let requests = [SelectionRequest { proposal: &proposals[i], positions: &tokens[i] }];
                if layer == 0 {
                    unsafe { slot.source.enqueue_selection(&slot.query.output()?, &requests, None)?; }
                    ensure!(slot.source.poll_pending()?, "selection blocked on host");
                } else {
                    let source = slot.source.output()?;
                    let follower = slot.reindex.as_mut().unwrap();
                    unsafe { follower.enqueue_selection(&slot.query.output()?, &requests, Some(&source))?; }
                    ensure!(follower.poll_pending()?, "reindex blocked on host");
                }

                // In the selection-only test the query's final reader is the
                // selection stream, not the held sparse-attention fork. Release
                // it now so it cannot hide a missing selection guard.
                if matches!(focus, Scratch::Selection) { slot.query.release_scratch()?; }
                copy_outputs(focus.buffers(slot)?)?;
                if matches!(focus, Scratch::Query) { slot.query.release_scratch()?; }
                slot.source.release_scratch()
            }))?;
            let reset = ResetSkip(focus);
            focus.skip(!guarded);
            eprintln!("index guard {focus:?}: enqueue B layer={layer}");
            runtime.block_on(chains[1].handle().scope(async {
                let slot = slots[1].get_mut();
                let i = 1;
                slot.query.rebind(&weights.weights[3 + layer])?;
                // SAFETY: queries/proposals persist through both units and readers.
                unsafe { slot.query.enqueue_attention(&queries[layer][i].output()?)?; }
                ensure!(slot.query.poll_pending()?, "projection blocked on host");
                let requests = [SelectionRequest { proposal: &proposals[i], positions: &tokens[i] }];
                if layer == 0 {
                    unsafe { slot.source.enqueue_selection(&slot.query.output()?, &requests, None)?; }
                    ensure!(slot.source.poll_pending()?, "selection blocked on host");
                } else {
                    let source = slot.source.output()?;
                    let follower = slot.reindex.as_mut().unwrap();
                    unsafe { follower.enqueue_selection(&slot.query.output()?, &requests, Some(&source))?; }
                    ensure!(follower.poll_pending()?, "reindex blocked on host");
                }
                anyhow::Ok(())
            }))?;
            drop(reset);
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_secs(if guarded { 2 } else { 10 });
            let mut done = false;
            while std::time::Instant::now() < deadline {
                if unsafe { library.cuda_stream_query(focus.stream(&slots[1]))? } { done = true; break; }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            eprintln!("index guard {focus:?} guarded={guarded} peer={peer} layer={layer}: before release writer_done={done}");
            assert_eq!(done, !guarded, "negative control must run; ordered writer must wait");
            drop(gate);
            for chain in &chains { chain.drain()?; }
            let got = copies.iter().map(|copy| reader_device.run(|| read(copy.buffer)))
                .collect::<Result<Vec<_>>>()?;
            if guarded { assert_eq!(got, expected[0][layer], "A fork reader was clobbered"); }
            else { assert_ne!(got, expected[0][layer], "negative control failed to expose clobber"); }
            let actual_b = focus.buffers(&slots[1])?.into_iter().map(&read).collect::<Result<Vec<_>>>()?;
            assert_eq!(actual_b, expected[1][layer], "B differs from private execution");
            for i in 0..2 {
                assert_eq!(read(slots[i].source.output()?.candidate_blocks().unwrap())?, blocks[i],
                    "source20 candidates must survive later reindex hooks");
            }
            runtime.block_on(chains[1].handle().scope(async { slots[1].release_scratch() }))?;
            chains[1].drain()?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires native library, snapshot and a GPU"]
    fn windowed_index_units_match_private_runs_behind_a_gated_stream() -> Result<()> {
        run(Scratch::Query, true, false)?; run(Scratch::Selection, true, false)
    }
    #[test]
    #[ignore = "requires native library, snapshot and a GPU"]
    fn unguarded_index_query_clobbers_a_gated_fork_reader() -> Result<()> { run(Scratch::Query, false, false) }
    #[test]
    #[ignore = "requires native library, snapshot and a GPU"]
    fn unguarded_index_selection_clobbers_a_gated_fork_reader() -> Result<()> { run(Scratch::Selection, false, false) }
    #[test]
    #[ignore = "requires native library, snapshot and two GPUs"]
    fn placed_index_units_match_private_runs_with_a_peer_fork_reader() -> Result<()> {
        run(Scratch::Query, true, true)?; run(Scratch::Selection, true, true)
    }
    #[test]
    #[ignore = "requires native library, snapshot and two GPUs"]
    fn unguarded_placed_index_query_clobbers_a_peer_fork_reader() -> Result<()> { run(Scratch::Query, false, true) }
    #[test]
    #[ignore = "requires native library, snapshot and two GPUs"]
    fn unguarded_placed_index_selection_clobbers_a_peer_fork_reader() -> Result<()> { run(Scratch::Selection, false, true) }
}
