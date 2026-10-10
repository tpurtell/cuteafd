//! Host snapshot cache binding: the glue between the
//! engine's retained snapshots (`Saved`) and `cuteafd_hostcache::HostCache`. Design of record:
//! recipes `dsv41-flash-tp4-engram/research/afd-hostcache-design.md` §4.6. Everything here runs
//! on the scheduler thread; the GPU copies asynchronously on two dedicated streams.
//!
//! A retained snapshot's device bytes are its compressor source pages (each page's rows in four
//! device buffers), its backbone tail in an arena slot and its dSpark rings. Everything else in
//! a `Saved` is small host data and travels as the cache payload (`HostSaved`), so restoring
//! is: allocate fresh pages and arena slots, copy the bytes back, rebuild the `Saved` from parts
//! and let the engine's own restore logic run unchanged.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::BackbonePrefix;
use crate::families::deepseek_v41::v41_compressor::CompressorPrefix;
use crate::families::deepseek_v41::v41_dspark_cache::DsparkPrefix;
use crate::shared::memory::SnapshotStorage;
use crate::families::deepseek_v41::v41_window::WindowPrefix;
use cuteafd_core::EngramHistory;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use crate::shared::prefix::{cuda_copy::range, CudaCopyEngine};
use cuteafd_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget,
    StoreOutcome, StoreTicket,
};
use cuteafd_hostcache::config::Config;
use cuteafd_hostcache::copy::Stream;
use cuteafd_hostcache::metrics::Snapshot as MetricsSnapshot;
use cuteafd_hostcache::pool::Layout;
use cuteafd_hostcache::snapshot::{DevicePageId, Hit, SnapshotMeta};
use cuteafd_hostcache::COMPRESSORS;

/// The engine-side descriptors of a snapshot: everything in a `Saved` that is not device bytes.
/// `images` pins the snapshot's image key ids for as long as the host copy can be looked up.
pub(super) struct HostSaved {
    session: Option<String>,
    images: ImageKeys,
    history: EngramHistory,
    next: TokenScores,
    owner: u64,
    end: u64,
    /// `WindowPrefix` parts per backbone window.
    windows: Vec<(u64, u64, u64)>,
    /// Per compressor: owner, end, page count, rows.
    sources: Vec<(u64, u64, usize, usize)>,
    /// Per dSpark window: owner, end, ring bytes.
    draft: Option<Vec<(u64, u64, usize)>>,
}

/// The glue: builds `DeviceSnapshot`s from `Saved`s and `Saved`s from restored bytes.
pub(crate) struct HostCacheBinding<'a> {
    cache: HostCache<CudaCopyEngine<'a>, HostSaved>,
}

impl<'a> HostCacheBinding<'a> {
    /// `None` when the cache is disabled: the engine's paths stay untouched.
    pub fn new(
        library: &'a NativeLibrary,
        config: Config,
        template: CuteafdDeviceBuffer,
    ) -> Result<Option<Self>> {
        config.validate()?;
        if !config.enabled() {
            return Ok(None);
        }
        let engine = CudaCopyEngine::new(library, template)?;
        let cache = HostCache::with_rule(config, Layout::engine(0), engine,
            cuteafd_core::prefix::ReuseRule::V41, cuteafd_hostcache::snapshot::EvictionOrder::LeastRecent)?;
        tracing::info!(target: "cuteafd::host_cache", config = ?cache.config(), "host snapshot cache enabled");
        Ok(Some(Self { cache }))
    }

    fn describe(
        &self,
        kind: SnapshotKind,
        keys: &[u32],
        saved: &Saved<'a>,
        requests: &Requests<'a>,
    ) -> Result<(DeviceSnapshot, HostSaved)> {
        let (backbone, history) = saved.target.parts();
        let (owner, end, tail, windows, sources) = backbone.parts();
        let caches = requests.cache().sources();
        let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
        let mut source_parts = Vec::with_capacity(COMPRESSORS);
        for (c, prefix) in sources.iter().enumerate() {
            let (source_owner, source_end, source) = prefix.parts();
            let cache = caches[c].get().source_cache();
            pages[c] = source
                .pages()
                .iter()
                .map(|&page| DevicePage {
                    id: DevicePageId {
                        compressor: c as u8,
                        page,
                        generation: cache.page_generation(page),
                    },
                    segments: cache.page_segments(page).into_iter().map(range).collect(),
                })
                .collect();
            source_parts.push((
                source_owner,
                source_end,
                source.pages().len(),
                source.rows(),
            ));
        }
        let draft = saved.draft.as_ref().map(|d| {
            d.parts()
                .iter()
                .map(|p| p.parts())
                .map(|(o, e, ring)| ((o, e, ring.buffer.bytes), range(ring.buffer)))
                .unzip::<_, _, Vec<_>, Vec<_>>()
        });
        let (draft_parts, draft_ranges) = match draft {
            Some((parts, ranges)) => (Some(parts), Some(ranges)),
            None => (None, None),
        };
        let snapshot = DeviceSnapshot {
            meta: SnapshotMeta {
                kind,
                tokens: keys.to_vec(),
                end: end as u32,
                has_draft: draft_ranges.is_some(),
            },
            pages,
            tail: vec![range(tail.buffer)],
            draft: draft_ranges,
            scores: vec![],
        };
        let payload = HostSaved {
            session: saved.session.clone(),
            images: saved._images.through(end as usize),
            history: history.fork()?,
            next: saved.next.clone(),
            owner,
            end,
            windows: windows.iter().map(WindowPrefix::parts).collect(),
            sources: source_parts,
            draft: draft_parts,
        };
        Ok((snapshot, payload))
    }

    /// Issue the write-behind copy of a freshly retained snapshot; the ticket lives in the `Saved`.
    pub(super) fn store(
        &mut self,
        kind: SnapshotKind,
        keys: &[u32],
        saved: &Saved<'a>,
        requests: &Requests<'a>,
    ) -> Result<Option<StoreTicket>> {
        let (snapshot, payload) = self.describe(kind, keys, saved, requests)?;
        Ok(match self.cache.store(&snapshot, payload) {
            StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) => Some(ticket),
            StoreOutcome::Skipped(_) => None,
        })
    }
    /// The key-space tokens a resident host snapshot is keyed by.
    pub(super) fn snapshot_tokens(&self, key: cuteafd_hostcache::snapshot::Key) -> Option<Vec<u32>> {
        self.cache.snapshot_tokens(key).map(<[u32]>::to_vec)
    }
    pub(super) fn tick(&mut self) {
        self.cache.tick();
    }
    /// A host hit whose restore could not be carried out (for example no device pages for it):
    /// counted with the copy failures so `/v1/stats` shows every abandoned restore; the request
    /// prefills instead.
    pub(super) fn count_abandoned_restore(&mut self) {
        self.cache.metrics_mut().get_mut().restore_failures += 1;
    }
    /// The engine is dropping a `Saved`: let its copy finish within budget or count the loss.
    pub(super) fn before_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        self.cache.before_device_evict(ticket)
    }
    pub(super) fn release_barrier(&mut self) -> Result<()> {
        let store = self.cache.engine_mut().synchronize(Stream::Store);
        let restore = self.cache.engine_mut().synchronize(Stream::Restore);
        store.and(restore)
    }
    /// Delegates to [`HostCache::prefill_hold`]; the full contract lives there. The
    /// scheduler's wrapper observes the store stream (`tick`) on every prefill chunk
    /// regardless of `store_pace_ns` before calling this hold.
    pub(super) fn prefill_hold(&mut self) -> anyhow::Result<()> {
        self.cache.prefill_hold()
    }
    /// The effective configuration, exported with the metrics.
    pub(super) fn config(&self) -> &Config {
        self.cache.config()
    }
    pub(super) fn lookup(&mut self, keys: &[u32]) -> Option<Hit> {
        self.cache.lookup(keys)
    }
    /// Rebuild a `Saved` from the host copy. `Ok(None)` when the restore timed out or failed (the
    /// caller prefills); allocations are released only after the restore stream drained.
    pub(super) fn restore<C: crate::families::deepseek_v41::v41_native_serve::speculative::DraftChain<'a>>(
        &mut self,
        hit: &Hit,
        requests: &Requests<'a>,
        draft: Option<&DraftRuntime<'_, 'a, C>>,
    ) -> Result<Option<Saved<'a>>> {
        // Take what the rebuilt `Saved` needs out of the payload before the mutable restore call.
        let (owner, end, windows, source_parts, draft_parts, history, next, images, session) = {
            let payload = self
                .cache
                .payload(hit.key)
                .context("host cache hit without payload")?;
            (
                payload.owner,
                payload.end,
                payload.windows.clone(),
                payload.sources.clone(),
                payload.draft.clone(),
                payload.history.fork()?,
                payload.next.clone(),
                payload.images.through(payload.end as usize),
                payload.session.clone(),
            )
        };
        let backbone = requests.cache();
        let caches = backbone.sources();
        ensure!(
            draft_parts.is_some() == draft.is_some(),
            "host snapshot execution mode differs"
        );
        let mut sources = Vec::with_capacity(COMPRESSORS);
        let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
        for (c, &(owner, end, count, rows)) in source_parts.iter().enumerate() {
            let cache = caches[c].get().source_cache();
            let source = cache.allocate_prefix(count, rows)?;
            pages[c] = source
                .pages()
                .iter()
                .map(|&page| DevicePage {
                    id: DevicePageId {
                        compressor: c as u8,
                        page,
                        generation: cache.page_generation(page),
                    },
                    segments: cache.page_segments(page).into_iter().map(range).collect(),
                })
                .collect();
            sources.push(CompressorPrefix::from_parts(owner, end, source));
        }
        let tail = SnapshotStorage::new(
            backbone.prefix_library(),
            BackbonePrefix::device_bytes(),
            backbone.prefix_pool(),
        )?;
        let rings = match (&draft_parts, draft) {
            (Some(parts), Some(runtime)) => Some(
                parts
                    .iter()
                    .zip(runtime.windows())
                    .map(|(&(_, _, bytes), window)| {
                        window.device().run(||
                            SnapshotStorage::new(window.library(), bytes, window.prefix_pool()))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => None,
        };
        let target = RestoreTarget {
            pages,
            tail: vec![range(tail.buffer)],
            draft: rings
                .as_ref()
                .map(|rings| rings.iter().map(|r| range(r.buffer)).collect()),
            scores: vec![],
        };
        match self.cache.restore(hit.key, &target) {
            RestoreOutcome::Done { .. } => {}
            outcome => {
                tracing::warn!(target: "cuteafd::host_cache", ?outcome, "host restore did not complete; prefilling");
                if let Err(error) = self.cache.engine_mut().synchronize(Stream::Restore) {
                    // A failed release barrier must not return pages/arena
                    // slots while an upload may still write them.
                    std::mem::forget((sources, tail, rings));
                    return Err(error);
                }
                return Ok(None);
            }
        }
        for (cache,prefix) in caches.iter().zip(&sources) {
            // RestoreOutcome::Done means the host upload completed. Publish FP4
            // replica pages before the rebuilt prefix can enter the GPU cache.
            unsafe { cache.get().source_cache().publish_restored_prefix(prefix.parts().2)?; }
        }
        let windows = windows
            .iter()
            .map(|&(o, e, b)| WindowPrefix::from_parts(o, e, b))
            .collect();
        let target_prefix = RequestPrefix::from_parts(
            BackbonePrefix::from_parts(owner, end, tail, windows, sources),
            history,
        );
        let draft = match (draft_parts.as_ref(), rings) {
            (Some(parts), Some(rings)) => Some(DraftPrefix::from_parts(backbone.prefix_library(),
                parts
                    .iter()
                    .zip(rings)
                    .map(|(&(o, e, _), ring)| DsparkPrefix::from_parts(o, e, ring))
                    .collect(),
            )?),
            _ => None,
        };
        Ok(Some(Saved {
            session,
            _images: images,
            target: target_prefix,
            draft,
            next,
            ticket: None,
        }))
    }
    pub(super) fn metrics(&self) -> MetricsSnapshot {
        self.cache.metrics()
    }
}
