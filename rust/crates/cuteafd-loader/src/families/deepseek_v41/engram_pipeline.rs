//! Request-owned early engram I/O and accepted-prefix history transactions.
use crate::{
    EngramGatherLease, EngramGatherPoll, EngramGatherTicket, EngramGatherer, EngramTable,
    EngramTokenMap, OfficialV41Catalog,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{EngramBatch, EngramHistory, ENGRAM_LAYERS};
use std::sync::Arc;

pub struct EngramRequestTokens<'a> {
    pub history: &'a EngramHistory,
    pub token_ids: &'a [u32],
    pub image_mask: Option<&'a [bool]>,
}
enum LayerIo {
    AwaitingAdmission,
    Queued(EngramGatherTicket),
    Ready(EngramGatherLease),
    Cancelled,
    Consumed,
}
pub struct EngramWave {
    pipeline: Arc<()>,
    gatherer: Arc<EngramGatherer>,
    tables: [Arc<EngramTable>; 2],
    batches: Vec<Arc<EngramBatch>>,
    layers: [LayerIo; 2],
    finished: bool,
    waiting: [Option<std::time::Instant>; 2],
    decode: bool,
}
impl EngramWave {
    pub fn copy_capacity_stalled(&self, layer: usize) -> Result<bool> {
        Ok(matches!(self.layers.get(layer), Some(LayerIo::AwaitingAdmission))
            && self.gatherer.copy_capacity_stalled()?)
    }
    pub fn batches(&self) -> &[Arc<EngramBatch>] {
        &self.batches
    }
    pub fn retain_copy(&self, lease: EngramGatherLease) { self.gatherer.retain_copy(lease); }
    pub fn reclaim_copies(&self) -> Result<()> { self.gatherer.reclaim_copies() }
    /// Await admission and I/O using only wave-owned state. The caller must
    /// revalidate request generations through `EngramPipeline::poll` afterwards.
    pub async fn wait_gather(&mut self, layer: usize) -> Result<()> {
        ensure!(!self.finished, "engram wave is already finished");
        ensure!(layer < self.layers.len(), "invalid engram layer index");
        self.waiting[layer].get_or_insert_with(std::time::Instant::now);
        let result = self.wait_layer(layer).await;
        if result.is_err() { self.cancel(); }
        result
    }
    async fn wait_layer(&mut self, layer: usize) -> Result<()> {
        if matches!(self.layers[layer], LayerIo::AwaitingAdmission) {
            self.gatherer.reclaim_copies()?;
            let ticket = self.gatherer.submit(Arc::clone(&self.tables[layer]),
                &self.batches, layer).await?;
            self.layers[layer] = LayerIo::Queued(ticket);
        }
        let result = match &mut self.layers[layer] {
            LayerIo::Queued(ticket) => ticket.wait().await?,
            LayerIo::Ready(_) | LayerIo::Cancelled => return Ok(()),
            LayerIo::Consumed => anyhow::bail!("engram layer result already consumed"),
            LayerIo::AwaitingAdmission => unreachable!("admission completed above"),
        };
        self.layers[layer] = match result {
            EngramGatherPoll::Ready(lease) => LayerIo::Ready(lease),
            EngramGatherPoll::Cancelled => LayerIo::Cancelled,
            EngramGatherPoll::Pending => anyhow::bail!("engram gather wait resolved before completion"),
        };
        Ok(())
    }
    fn admit(&mut self, layer: usize) -> Result<()> {
        if matches!(self.layers[layer], LayerIo::AwaitingAdmission) {
            if let Some(ticket) = self.gatherer.try_submit(Arc::clone(&self.tables[layer]),
                &self.batches, layer)? {
                self.layers[layer] = LayerIo::Queued(ticket);
            }
        }
        Ok(())
    }
    /// Cancel early I/O without committing any token history.
    pub fn cancel(&mut self) {
        self.finished = true;
        self.layers = [LayerIo::Consumed, LayerIo::Consumed];
    }
    /// Preflight history publication before an enclosing device-cache commit.
    pub fn validate_commit(&self, histories: &[&EngramHistory], accepted: &[usize]) -> Result<()> {
        ensure!(!self.finished, "engram wave is already finished");
        ensure!(
            histories.len() == self.batches.len() && accepted.len() == self.batches.len(),
            "engram wave acceptance dimensions differ"
        );
        for ((history, batch), &count) in histories.iter().zip(&self.batches).zip(accepted) {
            history.validate_commit(batch, count)?;
        }
        Ok(())
    }
    /// Commit accepted prefixes only after validating the complete physical wave.
    /// Rejected speculative rows never enter history; remaining I/O is cancelled.
    pub fn commit(
        &mut self,
        histories: &mut [&mut EngramHistory],
        accepted: &[usize],
    ) -> Result<()> {
        self.validate_commit(
            &histories.iter().map(|h| &**h).collect::<Vec<_>>(),
            accepted,
        )?;
        // Exclusive history borrows prevent changes between validation and commit.
        for ((history, batch), &count) in histories.iter_mut().zip(&self.batches).zip(accepted) {
            history.commit(batch, count)?;
        }
        self.cancel();
        Ok(())
    }
}

pub struct EngramPipeline {
    identity: Arc<()>,
    tables: [Arc<EngramTable>; 2],
    token_map: EngramTokenMap,
    gatherer: Arc<EngramGatherer>,
    decode_gatherer: Arc<EngramGatherer>,
    capacity: usize,
    placement: crate::formats::mapped_preload::MappedPlacement,
    shape: crate::EngramStagingShape,
    /// Opt-in background page-cache warm; dropping the pipeline stops it.
    warm: Option<crate::formats::mapped_warm::TableWarm>,
}
impl EngramPipeline {
    /// # Safety
    /// Official checkpoint files must stay immutable while any mapping/job is alive.
    pub unsafe fn new(
        library: &cuteafd_ffi::NativeLibrary,
        catalog: &OfficialV41Catalog,
        token_map: EngramTokenMap,
        shape: crate::EngramStagingShape,
        scale_preload_cap: u64,
    ) -> Result<Self> {
        let capacity = shape.rows();
        // SAFETY: forwarded from this constructor's immutable checkpoint contract.
        let mut tables = unsafe { [catalog.map_engram(ENGRAM_LAYERS[0] as usize)?,
            catalog.map_engram(ENGRAM_LAYERS[1] as usize)?] };
        use crate::formats::mapped_preload::{host_budget_bytes, MappedPlacement};
        let host_budget = host_budget_bytes().unwrap_or_else(|error| {
            tracing::warn!(%error, "cannot admit mapped scale preload; keeping all scales mapped");
            0
        });
        let bytes = tables.iter().map(|table| table.scales().bytes()).sum();
        let mut placement = MappedPlacement::resolve(4, 2, bytes, scale_preload_cap, host_budget);
        let started = std::time::Instant::now();
        if placement.preloaded > 0 {
            let [first, second] = &mut tables;
            if let Err(error) = crate::MappedTable::preload_all(&mut [first.scales_mut(), second.scales_mut()]) {
                placement.fallback(&error);
            }
        }
        tracing::info!(placement = %placement.label(), scale_bytes = bytes,
            preloaded_bytes = placement.preloaded_bytes, cap_bytes = scale_preload_cap,
            host_budget_bytes = host_budget, reason = %placement.reason,
            elapsed_ms = started.elapsed().as_millis(), "mapped scale preload admission");
        let tables = tables.map(Arc::new);
        Ok(Self {
            identity: Arc::new(()),
            tables,
            token_map,
            gatherer: Arc::new(EngramGatherer::new_pinned(library, shape)?),
            decode_gatherer: Arc::new(EngramGatherer::new_decode(library, shape)?),
            capacity,
            placement, shape,
            warm: None,
        })
    }
    pub fn placement(&self) -> &crate::formats::mapped_preload::MappedPlacement { &self.placement }
    pub fn shape(&self) -> crate::EngramStagingShape { self.shape }
    pub fn reclaim_copies(&self) -> Result<()> {
        self.gatherer.reclaim_copies()?; self.decode_gatherer.reclaim_copies()
    }
    /// Opt-in background page-cache warm of both tables
    /// ([`TableWarm::start`](crate::formats::mapped_warm::TableWarm::start)):
    /// once `ready` holds and available host memory admits them, `readers`
    /// threads read the scale tables (every row read needs one, at a fraction
    /// of the weight bytes), then the weights. Resident (preloaded) scales are
    /// skipped. Dropping the pipeline stops it between chunks.
    pub fn start_warm(&mut self, readers: usize, ready: impl Fn() -> bool + Send + 'static) -> Result<()> {
        ensure!(self.warm.is_none(), "engram warm already started");
        self.warm = Some(crate::formats::mapped_warm::TableWarm::start("engram", self.tables.clone(),
            |[first, second]| vec![first.scales(), second.scales(), first.weights(), second.weights()],
            readers, ready)?);
        Ok(())
    }
    pub fn new_history(&self) -> Result<EngramHistory> {
        Ok(EngramHistory::new(self.token_map.pad_id())?)
    }
    /// Normalize only the bounded suffix needed to resume Engram at a retained
    /// compression boundary. Tokens are chronological, immediately before it.
    pub fn history_at(&self, position: u64, recent: &[u32], image_mask: Option<&[u8]>) -> Result<EngramHistory> {
        ensure!(recent.len() == position.min(3) as usize
            && image_mask.is_none_or(|mask| mask.len() == recent.len()),
            "engram resume token or image lookback differs");
        let compressed = recent.iter().enumerate().map(|(i, &token)|
            self.token_map.compress(token, image_mask.is_some_and(|mask| mask[i] != 0)))
            .collect::<Result<Vec<_>>>()?;
        Ok(EngramHistory::from_recent(self.token_map.pad_id(), position, &compressed)?)
    }
    /// Call as soon as decode, prefill or verification token IDs and image spans are known.
    /// Prepare all hashes before starting I/O, preserving committed request histories.
    pub fn prepare(&self, requests: &[EngramRequestTokens<'_>]) -> Result<EngramWave> {
        self.prepare_on(requests, false)
    }
    pub fn prepare_decode(&self, requests: &[EngramRequestTokens<'_>]) -> Result<EngramWave> {
        self.prepare_on(requests, true)
    }
    fn prepare_on(&self, requests: &[EngramRequestTokens<'_>], decode: bool) -> Result<EngramWave> {
        self.reclaim_copies()?;
        ensure!(
            !requests.is_empty() && requests.len() <= 16,
            "engram wave requires 1..16 requests"
        );
        let total = requests.iter().try_fold(0usize, |sum, request| {
            ensure!(
                !request.token_ids.is_empty(),
                "engram request batch is empty"
            );
            sum.checked_add(request.token_ids.len())
                .context("engram wave row overflow")
        })?;
        ensure!(
            total <= if decode { self.shape.decode_rows() } else { self.capacity },
            "engram wave exceeds pipeline capacity"
        );
        // Reject repeated histories before any work is admitted; one request owns
        // exactly one contiguous batch in a physical wave.
        for (index, request) in requests.iter().enumerate() {
            ensure!(
                !requests[..index]
                    .iter()
                    .any(|prior| std::ptr::eq(prior.history, request.history)),
                "engram wave repeats a request history"
            );
        }
        let batches: Result<Vec<_>> = requests
            .iter()
            .map(|request| {
                self.token_map
                    .prepare_batch(
                        request.history,
                        request.history.position(),
                        request.token_ids,
                        request.image_mask,
                        self.capacity,
                    )
                    .map(Arc::new)
            })
            .collect();
        let mut wave = EngramWave {
            pipeline: Arc::clone(&self.identity),
            gatherer: Arc::clone(if decode { &self.decode_gatherer } else { &self.gatherer }),
            tables: self.tables.clone(),
            batches: batches?,
            layers: [LayerIo::AwaitingAdmission, LayerIo::AwaitingAdmission],
            finished: false,
            waiting: [None; 2],
            decode,
        };
        // Both layers' host gathers start now (L14 right after L1, for prefill
        // and decode alike); the gather worker runs every queued L1 before any
        // L14, so an early L14 never delays a unit's L1 deadline. Each layer's
        // upload stays host-issued after its gate (never parked on a copy engine).
        wave.admit(0)?;
        wave.admit(1)?;
        Ok(wave)
    }
    /// Retry bounded admission and poll without waiting for mapped pages.
    /// A returned lease must be uploaded or dropped before its staging can be reused.
    pub fn poll(
        &self,
        wave: &mut EngramWave,
        histories: &[&EngramHistory],
        layer: usize,
    ) -> Result<EngramGatherPoll> {
        wave.poll_gather(&self.identity, histories, layer)
    }
}

impl EngramWave {
    fn poll_gather(&mut self, pipeline: &Arc<()>, histories: &[&EngramHistory],
        layer: usize) -> Result<EngramGatherPoll> {
        let wave = self;
        ensure!(!wave.finished, "engram wave is already finished");
        ensure!(layer < 2, "invalid engram layer index");
        let result = (|| {
            ensure!(
                Arc::ptr_eq(pipeline, &wave.pipeline),
                "engram wave belongs to another pipeline"
            );
            ensure!(
                histories.len() == wave.batches.len(),
                "engram wave history count differs"
            );
            for (history, batch) in histories.iter().zip(&wave.batches) {
                history.validate_commit(batch, 0)?;
            }
            wave.gatherer.reclaim_copies()?;
            wave.admit(layer)?;
            let result = match &mut wave.layers[layer] {
                LayerIo::AwaitingAdmission => {
                    wave.waiting[layer].get_or_insert_with(std::time::Instant::now);
                    return Ok(EngramGatherPoll::Pending);
                },
                LayerIo::Queued(ticket) => ticket.poll()?,
                LayerIo::Ready(_) => {
                    let LayerIo::Ready(lease) = std::mem::replace(&mut wave.layers[layer], LayerIo::Consumed)
                        else { unreachable!("matched ready lease") };
                    EngramGatherPoll::Ready(lease)
                }
                LayerIo::Cancelled => EngramGatherPoll::Cancelled,
                LayerIo::Consumed => anyhow::bail!("engram layer result already consumed"),
            };
            if let EngramGatherPoll::Ready(lease) = &result {
                ensure!(
                    lease.batches().len() == wave.batches.len()
                        && lease
                            .batches()
                            .iter()
                            .zip(&wave.batches)
                            .all(|(a, b)| Arc::ptr_eq(a, b)),
                    "engram gather result belongs to a different wave"
                );
            }
            if matches!(result, EngramGatherPoll::Pending) {
                wave.waiting[layer].get_or_insert_with(std::time::Instant::now);
            } else if matches!(result, EngramGatherPoll::Ready(_)) {
                let elapsed = wave.waiting[layer].take().map_or(std::time::Duration::ZERO, |t| t.elapsed());
                record_layer_wait(&wave.tables[layer], elapsed, wave.decode);
            }
            if !matches!(result, EngramGatherPoll::Pending) {
                wave.layers[layer] = LayerIo::Consumed;
            }
            Ok(result)
        })();
        if result.is_err() {
            wave.cancel();
        }
        result
    }
}

// A lease contains both weight and scale rows. Its consumer wait belongs to
// the layer/pair, not each tensor; store it on weights so totals never double.
fn record_layer_wait(table: &EngramTable, elapsed: std::time::Duration, decode: bool) {
    table.weights().stats().record_stall(elapsed);
    if decode { table.weights().stats().record_decode_stall(elapsed); }
}

#[cfg(test)]
mod wait_tests {
    use super::*;
    use std::{future::Future, task::{Context as TaskContext, Poll, Wake, Waker}, time::{Duration, Instant}};

    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) { self.0.unpark(); }
        fn wake_by_ref(self: &Arc<Self>) { self.0.unpark(); }
    }
    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let mut cx = TaskContext::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Poll::Ready(result) = future.as_mut().poll(&mut cx) { return result; }
            assert!(Instant::now() < deadline, "engram fixture completion timed out");
            std::thread::park_timeout(deadline.saturating_duration_since(Instant::now()));
        }
    }
    fn table_fixture(selected: &[u64]) -> Result<Arc<EngramTable>> {
        use std::os::unix::fs::FileExt;
        let weights = tempfile::NamedTempFile::new()?;
        let scales = tempfile::NamedTempFile::new()?;
        let rows = cuteafd_core::ENGRAM_ROWS[0];
        weights.as_file().set_len(rows * 256)?;
        scales.as_file().set_len(rows * 8)?;
        for &row in selected {
            let weight: Vec<_> = (0..256).map(|column| row.wrapping_mul(37).wrapping_add(column) as u8).collect();
            let scale: Vec<_> = (0..8).map(|column| row.wrapping_mul(19).wrapping_add(column + 1) as u8).collect();
            weights.as_file().write_all_at(&weight, row * 256)?;
            scales.as_file().write_all_at(&scale, row * 8)?;
        }
        // SAFETY: immutable sparse files remain mapped after their names are removed.
        let map = |file: &tempfile::NamedTempFile, width| unsafe {
            crate::MappedTable::single(file.path(), 0, rows,
                crate::RowFormat { dtype: cuteafd_core::DType::U8, width, row_bytes: width })
        };
        Ok(Arc::new(EngramTable::new(map(&weights, 256)?, map(&scales, 8)?)?))
    }
    fn wave_fixture(table: Arc<EngramTable>, gatherer: Arc<EngramGatherer>, batch: Arc<EngramBatch>) -> EngramWave {
        EngramWave { pipeline: Arc::new(()), gatherer, tables: [table.clone(), table],
            batches: vec![batch],
            layers: [LayerIo::AwaitingAdmission, LayerIo::Consumed], finished: false,
            waiting: [None; 2], decode: false }
    }

    #[test]
    fn awaited_gather_rejects_request_reuse_and_changed_generation() -> Result<()> {
        let table = table_fixture(&[])?;
        let budget = crate::EngramBatchStaging::storage_bytes(1)?;
        let gatherer = Arc::new(EngramGatherer::new(1, 1, budget)?);
        for reused in [false, true] {
            let mut history = EngramHistory::new(2)?;
            let batch = Arc::new(history.prepare(0, &[Some(11)], 1)?);
            let mut blocker = gatherer.try_submit(table.clone(), std::slice::from_ref(&batch), 0)?.unwrap();
            let EngramGatherPoll::Ready(lease) = block_on(blocker.wait())?
                else { anyhow::bail!("fixture gather did not complete") };
            let mut wave = wave_fixture(table.clone(), gatherer.clone(), batch.clone());
            let identity = wave.pipeline.clone();
            {
                let wait = wave.wait_gather(0);
                let mut wait = std::pin::pin!(wait);
                let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
                assert!(wait.as_mut().poll(&mut TaskContext::from_waker(&waker)).is_pending());
                if reused { history = EngramHistory::new(2)?; }
                else { history.commit(&batch, 0)?; }
                drop(lease);
                block_on(wait)?;
            }
            assert!(matches!(wave.layers[0], LayerIo::Ready(_)));
            let error = wave.poll_gather(&identity, &[&history], 0).err().context("stale gather was consumed")?;
            assert!(error.to_string().contains("stale or foreign engram batch"));
            assert!(wave.finished);
            let mut recycled = block_on(gatherer.submit(table.clone(), std::slice::from_ref(&batch), 0))?;
            assert!(matches!(block_on(recycled.wait())?, EngramGatherPoll::Ready(_)));
        }
        Ok(())
    }

    #[test]
    fn awaited_gather_matches_direct_rows_and_consumes_once() -> Result<()> {
        let history = EngramHistory::new(2)?;
        let batch = Arc::new(history.prepare(0, &[Some(11), None, Some(13)], 3)?);
        let table = table_fixture(&batch.prefetch_rows(0)?)?;
        let gatherer = Arc::new(EngramGatherer::new(1, 3, crate::EngramBatchStaging::storage_bytes(3)?)?);
        let mut direct = crate::EngramBatchStaging::new(3)?;
        let expected = direct.gather(&table, &[batch.as_ref()], 0)?;
        assert!(expected.weights.iter().any(|&value| value != 0));
        assert!(expected.scales.iter().any(|&value| value != 0));
        let mut wave = wave_fixture(table.clone(), gatherer, batch);
        let identity = wave.pipeline.clone();
        block_on(wave.wait_gather(0))?;
        // Waiting twice must preserve the ready lease, not consume it twice.
        block_on(wave.wait_gather(0))?;
        let EngramGatherPoll::Ready(lease) = wave.poll_gather(&identity, &[&history], 0)?
            else { anyhow::bail!("awaited fixture gather was not ready") };
        let actual = lease.view()?;
        assert_eq!(actual.weights, expected.weights);
        assert_eq!(actual.scales, expected.scales);
        assert_eq!(actual.text_mask, expected.text_mask);
        assert_eq!((actual.rows, actual.layer_index), (expected.rows, expected.layer_index));
        assert!(block_on(wave.wait_gather(0)).is_err());
        assert!(wave.finished);
        assert_eq!(table.weights().stats().snapshot().consumer_steps, 1);
        Ok(())
    }

    #[test]
    fn cancelled_wave_never_consumes_queued_or_ready_gathers() -> Result<()> {
        let table = table_fixture(&[])?;
        let gatherer = Arc::new(EngramGatherer::new(1, 1, crate::EngramBatchStaging::storage_bytes(1)?)?);
        let history = EngramHistory::new(2)?;
        for ready in [false, true] {
            let batch = Arc::new(history.prepare(0, &[Some(11)], 1)?);
            let mut wave = wave_fixture(table.clone(), gatherer.clone(), batch);
            wave.admit(0)?;
            if ready { block_on(wave.wait_gather(0))?; }
            let identity = wave.pipeline.clone();
            wave.cancel();
            assert!(block_on(wave.wait_gather(0)).is_err());
            assert!(wave.poll_gather(&identity, &[&history], 0).is_err());
        }
        Ok(())
    }

    #[test]
    fn layer_wait_is_recorded_once_for_weight_scale_pair() -> Result<()> {
        let weights = tempfile::NamedTempFile::new()?;
        let scales = tempfile::NamedTempFile::new()?;
        weights.as_file().set_len(cuteafd_core::ENGRAM_ROWS[0] * 256)?;
        scales.as_file().set_len(cuteafd_core::ENGRAM_ROWS[0] * 8)?;
        // SAFETY: owned immutable sparse fixtures remain live through the maps.
        let map = |file: &tempfile::NamedTempFile, width| unsafe {
            crate::MappedTable::single(file.path(), 0, cuteafd_core::ENGRAM_ROWS[0],
                crate::RowFormat { dtype: cuteafd_core::DType::U8, width, row_bytes: width })
        };
        let table = EngramTable::new(map(&weights, 256)?, map(&scales, 8)?)?;
        record_layer_wait(&table, std::time::Duration::from_micros(7), true);
        record_layer_wait(&table, std::time::Duration::from_micros(3), false);
        let weight = table.weights().stats().snapshot();
        let scale = table.scales().stats().snapshot();
        assert_eq!((weight.consumer_steps, weight.stall_ns), (2, 10_000));
        assert_eq!((weight.decode_steps, weight.decode_stall_ns), (1, 7_000));
        assert_eq!((scale.consumer_steps, scale.stall_ns, scale.decode_steps, scale.decode_stall_ns), (0, 0, 0, 0));
        Ok(())
    }
}
