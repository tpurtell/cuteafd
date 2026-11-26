//! The backbone's part of a prefix snapshot: every window's ring rows and every compressed
//! source's carry, copied into a positional mark the prefix engine's arena holds
//! ([`BackboneMark::BYTES`]), described by plain data ([`BackboneMark`]). Global source rows live
//! in pages the engine forks; nothing here owns device memory.
use crate::shared::memory::Tag;
use super::*;
use crate::families::deepseek_v41::v41_compressor::COMPRESSOR_PREFIX_BYTES;
use crate::families::deepseek_v41::v41_window::WINDOW_PREFIX_BYTES;
use cuteafd_ffi::CuteafdDeviceBuffer;
use serde::{Deserialize, Serialize};

/// What a backbone mark holds besides its bytes: the snapshot's frontier and each window's
/// initialized span. Plain data, valid for any backbone of the same geometry.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BackboneMark {
    pub end: u64,
    /// Per window layer: `(begin, end)` of its ring.
    pub windows: Vec<(u64, u64)>,
}

impl BackboneMark {
    pub const BYTES: usize = 40 * WINDOW_PREFIX_BYTES + 4 * COMPRESSOR_PREFIX_BYTES;
}

fn slice(mut buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
    buffer.bytes = bytes;
    buffer
}

impl<'a> BackboneCache<'a> {
    /// Canonical initialized state for the golden gate. Physical page ids and unused
    /// ring/carry/page padding are deliberately excluded from the comparison.
    pub fn diagnostic_state(&mut self, lease: CacheLease) -> Result<(BackboneMark, Vec<u8>)> {
        use crate::shared::memory::device::{Allocation, Device};
        let library = self.prefix_stream.library;
        let device = Device { library, id: library.cuda_get_device()? };
        let mark = Allocation::new(device, BackboneMark::BYTES, Tag::prefix("v41/prefix-marks"))?;
        let meta = self.capture_mark(lease, mark.buffer)?;
        let mut bytes = Vec::new();
        let mut read = |buffer: CuteafdDeviceBuffer| -> Result<()> {
            let start = bytes.len();
            bytes.resize(start + buffer.bytes, 0);
            self.prefix_stream.copy_d2h(&mut bytes[start..], buffer)
        };
        for (layer, &(begin, end)) in meta.windows.iter().enumerate() {
            let first = begin.max(end.saturating_sub(128));
            let count = (end - first) as usize;
            let offset = first as usize % 128;
            let initial = count.min(128 - offset);
            for (offset, rows) in [(offset, initial), (0, count - initial)] {
                if rows == 0 { continue; }
                for (base, width) in [(0, 512), (128 * 512, 16)] {
                    read(slice(mark.buffer, layer * WINDOW_PREFIX_BYTES + base + offset * width, rows * width))?;
                }
            }
            read(slice(mark.buffer, layer * WINDOW_PREFIX_BYTES + 128 * 528, 8))?;
        }
        if meta.end % 2 == 1 {
            // Only the three ratio-two sources have a live compressor carry.
            for source in 0..3 {
                read(slice(mark.buffer, 40 * WINDOW_PREFIX_BYTES + source * COMPRESSOR_PREFIX_BYTES,
                    COMPRESSOR_PREFIX_BYTES))?;
            }
        }
        let request = self.request_identity(lease)?;
        for (state, &source_lease) in self.sources.iter().zip(&request.sources) {
            let (pages, rows) = state.binding(source_lease)?;
            ensure!(rows == meta.end as usize / state.ratio(), "diagnostic source frontier differs");
            state.device.run(|| {
                for (index, &page) in pages.iter().take(rows.div_ceil(PAGE_ROWS)).enumerate() {
                    let count = (rows - index * PAGE_ROWS).min(PAGE_ROWS);
                    for segment in state.source_cache().page_segments(page) {
                        read(slice(segment, 0, count * (segment.bytes / PAGE_ROWS)))?;
                    }
                }
                Ok(())
            })?;
        }
        Ok((meta, bytes))
    }

    /// Start a fresh request at the complete-group frontier `end` of source pages the prefix
    /// engine forked and bound, with empty encoder windows at the bounded replay start
    /// `start` (the media-safe frontier chosen by the family plan). Decoder windows remain fresh. Replay reads the shared source rows without
    /// producing them. Returns the replay start.
    pub fn restore_encoder_prefix(&mut self, lease: CacheLease, end: u64, start: u64, prompt_end: u64) -> Result<u64> {
        let request = self.request(lease)?;
        ensure!(
            end > 0
                && start < end
                && end % 2 == 0
                && end <= prompt_end
                && prompt_end <= 1048576
                && request.end == 0
                && request.version == 0
                && request.phase == CachePhase::Full
                && request.publication.is_empty(),
            "invalid encoder prefix restore frontier"
        );
        let windows = request.windows;
        let sources = request.sources;
        let result = (|| -> Result<()> {
            for layer in 0..20 {
                self.windows[layer].begin_encoder_replay(windows[layer], start)?;
            }
            for (state, lease) in self.sources.iter_mut().zip(sources) {
                state.restore_compressed_prefix(lease, end)?;
            }
            Ok(())
        })();
        result?; // The shared prefix engine drains and rolls back failed admissions.
        // Replay reads forked tail rows on compute streams. Single-device
        // publication need not drain the prefix copy stream on its own.
        self.prefix_drain()?;
        let request = self.requests[lease.slot]
            .as_mut()
            .expect("validated fresh admission");
        request.end = end;
        request.phase = CachePhase::EncoderReplay {
            prefix: end,
            target: prompt_end,
            end: start,
        };
        request.version = 1;
        self.committed_end(lease)?;
        Ok(start)
    }

    /// Copy a fully committed request's windows and carry into `mark` (synchronously). Only a
    /// complete frontier is captured, after all producer/consumer streams drained.
    pub fn capture_mark(&mut self, lease: CacheLease, mark: CuteafdDeviceBuffer) -> Result<BackboneMark> {
        let stream = self.prefix_stream.raw;
        let saved = self.copy_mark(lease, mark, stream)?;
        unsafe { self.prefix_stream.library.cuda_stream_synchronize(stream)?; }
        Ok(saved)
    }
    /// [`BackboneCache::capture_mark`] queued on capture lane `lane`'s own stream; the request
    /// cannot run or be released until [`BackboneCache::mark_ready`] or `abort_mark`.
    ///
    /// The lane's copies follow everything queued on the prefix stream so far (a snapshot's
    /// tail copy) through a device event, not a host drain: another request's queued restore
    /// or snapshot on the prefix stream never stalls this one's admission.
    pub fn queue_mark(&mut self, lane: usize, lease: CacheLease, mark: CuteafdDeviceBuffer) -> Result<BackboneMark> {
        let copies = self.prefix_copies.get(lane).context("invalid snapshot lane")?;
        ensure!(copies.pending.is_none(), "backbone snapshot lane is occupied");
        let stream = copies.stream.raw;
        // SAFETY: both streams and the event belong to this cache and are live.
        unsafe {
            self.prefix_stream.library.cuda_event_record(self.prefix_queued.0, self.prefix_stream.raw)?;
            self.prefix_stream.library.cuda_stream_wait_event(stream, self.prefix_queued.0)?;
        }
        let saved = self.copy_mark(lease, mark, stream)?;
        self.prefix_copies[lane].pending = Some(lease);
        Ok(saved)
    }
    /// Whether lane `lane`'s queued mark copies landed; the lane frees once they have.
    pub fn mark_ready(&mut self, lane: usize) -> Result<bool> {
        let copies = self.prefix_copies.get(lane).context("invalid snapshot lane")?;
        ensure!(copies.pending.is_some(), "no backbone snapshot is pending on this lane");
        let ready = copies.ready()?;
        if ready { self.prefix_copies[lane].pending = None; }
        Ok(ready)
    }
    /// Drain lane `lane`'s queued mark copies and free the lane.
    pub fn abort_mark(&mut self, lane: usize) -> Result<()> {
        self.prefix_copies.get_mut(lane).context("invalid snapshot lane")?.abort()
    }
    fn copy_mark(&mut self, lease: CacheLease, mark: CuteafdDeviceBuffer, stream: *mut std::ffi::c_void)
        -> Result<BackboneMark> {
        let end = self.committed_end(lease)?;
        let request = self.request(lease)?;
        ensure!(
            end > 0 && request.phase == CachePhase::Full && request.publication.is_empty(),
            "prefix retention requires a complete request frontier"
        );
        ensure!(mark.bytes == BackboneMark::BYTES, "backbone mark storage size differs");
        let result = (|| -> Result<_> {
            let windows = self.windows.iter().zip(request.windows).enumerate()
                .map(|(i, (state, lease))| unsafe {
                    state.retain_prefix(lease, slice(mark, i * WINDOW_PREFIX_BYTES, WINDOW_PREFIX_BYTES), stream)
                })
                .collect::<Result<Vec<_>>>()?;
            for (i, (state, lease)) in self.sources.iter().zip(request.sources).enumerate() {
                let saved = unsafe {
                    state.retain_prefix(lease,
                        slice(mark, 40 * WINDOW_PREFIX_BYTES + i * COMPRESSOR_PREFIX_BYTES, COMPRESSOR_PREFIX_BYTES),
                        stream)?
                };
                ensure!(saved == end, "source frontier differs from the request's");
            }
            Ok(windows)
        })();
        // No await occurs here. On a partial enqueue failure, drain before the mark storage can
        // be handed out again.
        match result {
            Ok(windows) => Ok(BackboneMark { end, windows }),
            Err(error) => {
                unsafe { self.prefix_stream.library.cuda_stream_synchronize(stream)?; }
                Err(error)
            }
        }
    }

    /// Restore a snapshot into a fresh request whose source pages hold its rows through
    /// `saved.end`. On error the caller must reset or release the drained request; a partially
    /// applied restore must never execute.
    pub fn restore_mark(&mut self, lease: CacheLease, saved: &BackboneMark, mark: CuteafdDeviceBuffer) -> Result<()> {
        self.restore_retained_phase(lease, saved, mark, CachePhase::Full)
    }

    /// A new suffix covering the whole decoder window needs only the saved encoder rings and
    /// the global sources. Preserve odd compressor carry while leaving decoder rings fresh for
    /// the final-window replay.
    pub fn restore_encoder_continuation(&mut self, lease: CacheLease, saved: &BackboneMark,
        mark: CuteafdDeviceBuffer, prompt_end: u64) -> Result<()> {
        ensure!(prompt_end <= 1048576 && prompt_end.saturating_sub(saved.end) >= 128,
            "encoder continuation must cover the final decoder window");
        self.restore_retained_phase(lease, saved, mark, CachePhase::Encoder { target: prompt_end })
    }

    fn restore_retained_phase(&mut self, lease: CacheLease, saved: &BackboneMark, mark: CuteafdDeviceBuffer,
        phase: CachePhase) -> Result<()> {
        let request = self.request(lease)?;
        ensure!(
            request.end == 0
                && request.version == 0
                && request.phase == CachePhase::Full
                && request.publication.is_empty()
                && saved.windows.len() == self.windows.len()
                && mark.bytes == BackboneMark::BYTES,
            "nonfresh admission or foreign backbone mark"
        );
        let windows = request.windows;
        let sources = request.sources;
        let result = (|| -> Result<()> {
            for (i, ((state, lease), &(begin, end))) in self.windows.iter_mut().zip(windows).zip(&saved.windows)
                .take(phase.stage().windows().end).enumerate() {
                unsafe {
                    state.restore_prefix(lease, begin, end,
                        slice(mark, i * WINDOW_PREFIX_BYTES, WINDOW_PREFIX_BYTES), self.prefix_stream.raw)?;
                }
            }
            for (i, (state, lease)) in self.sources.iter_mut().zip(sources).enumerate() {
                unsafe {
                    state.restore_prefix(lease, saved.end,
                        slice(mark, 40 * WINDOW_PREFIX_BYTES + i * COMPRESSOR_PREFIX_BYTES, COMPRESSOR_PREFIX_BYTES),
                        self.prefix_stream.raw)?;
                }
            }
            Ok(())
        })();
        let drained = self.prefix_drain();
        result.and(drained)?;
        let request = self.requests[lease.slot]
            .as_mut()
            .expect("validated prefix admission");
        request.end = saved.end;
        request.phase = phase;
        request.version = 1;
        self.committed_end(lease)?;
        Ok(())
    }

    /// Reset failed prefix state without changing the outer lease. Component leases are
    /// replaced so neither partial positional restores nor source page bindings survive.
    pub fn reset_prefix(&mut self, lease: CacheLease) -> Result<()> {
        self.prefix_drain()?;
        let request = self.request(lease)?;
        let (id, windows, sources) = (request.id, request.windows, request.sources);
        let reset = (|| -> Result<_> {
            let mut new_windows = Vec::with_capacity(windows.len());
            let mut new_sources = Vec::with_capacity(sources.len());
            for (state, old) in self.windows.iter_mut().zip(windows) {
                let device = state.device;
                new_windows.push(device.run(|| {
                    state.release(old)?;
                    state.begin_request(lease.slot, id)
                })?);
            }
            for (state, old) in self.sources.iter_mut().zip(sources) {
                let device = state.device;
                new_sources.push(device.run(|| {
                    state.release(old)?;
                    state.begin_request(lease.slot, id)
                })?);
            }
            Ok((new_windows, new_sources))
        })();
        let (windows, sources) = match reset {
            Ok(reset) => reset,
            Err(error) => { self.poison(&error); return Err(error); }
        };
        let request = self.requests[lease.slot].as_mut().expect("validated prefix request");
        request.windows = windows.try_into().ok().expect("40 windows");
        request.sources = sources.try_into().ok().expect("four sources");
        request.end = 0;
        request.version = 0;
        request.phase = CachePhase::Full;
        request.publication.clear();
        Ok(())
    }

    /// A complete request's mark metadata, from host bookkeeping only (no copy).
    pub fn mark_meta(&self, lease: CacheLease) -> Result<BackboneMark> {
        let end = self.committed_end(lease)?;
        let request = self.request(lease)?;
        ensure!(end > 0 && request.phase == CachePhase::Full && request.publication.is_empty(),
            "prefix retention requires a complete request frontier");
        let windows = self.windows.iter().zip(request.windows).map(|(state, lease)| state.span(lease))
            .collect::<Result<Vec<_>>>()?;
        Ok(BackboneMark { end, windows })
    }

    /// Wait for every snapshot copy and restore queued on the prefix stream (never a compute
    /// stream: prefix copies run on their own stream).
    pub fn prefix_drain(&self) -> Result<()> {
        // SAFETY: this stream belongs to the backbone and every queued owner remains live.
        let result = unsafe { self.prefix_stream.library.cuda_stream_synchronize(self.prefix_stream.raw) };
        if let Err(error) = &result {
            // Ring/carry writes may still be in flight: page quarantine alone cannot make
            // the request slot safe to reuse. Require process restart, not a cold retry.
            self.poison(&anyhow::anyhow!("V4.1 prefix copy drain failed; restart required: {error:#}"));
        }
        result
    }

    /// Physical pages of prefix-engine `units` (512 tokens each) in every source, row order: a
    /// ratio-two source's unit `u` is its page `u`, the ratio-one source's pages `2u, 2u+1`.
    pub fn unit_pages(&self, units: &[u32]) -> [Vec<u32>; 4] {
        std::array::from_fn(|i| {
            let per_unit = UNIT_TOKENS / (self.sources[i].ratio() * PAGE_ROWS);
            units.iter().flat_map(|&u| (0..per_unit as u32).map(move |k| u * per_unit as u32 + k)).collect()
        })
    }
    /// Initialized source rows of a request `tokens` long, per source.
    pub fn source_rows(&self, tokens: usize) -> [usize; 4] {
        std::array::from_fn(|i| tokens / self.sources[i].ratio())
    }
    /// Units every source can hold.
    pub fn unit_capacity(&self) -> usize {
        (0..4).map(|i| {
            let (ratio, pages) = (self.sources[i].ratio(), self.sources[i].source_cache().page_count());
            pages / (UNIT_TOKENS / (ratio * PAGE_ROWS))
        }).min().unwrap_or(0)
    }
    /// Enqueue on the prefix stream the copy of the first `tokens` tokens of unit `from` into unit
    /// `to`, in every source (a forked snapshot's partial tail).
    pub fn copy_unit_rows(&self, from: u32, to: u32, tokens: usize) -> Result<()> {
        ensure!(tokens <= UNIT_TOKENS, "unit tail copy exceeds a unit");
        let (from, to) = (self.unit_pages(&[from]), self.unit_pages(&[to]));
        for (i, source) in self.sources.iter().enumerate() {
            let mut rows = tokens / source.ratio();
            for (&f, &t) in from[i].iter().zip(&to[i]) {
                let count = rows.min(PAGE_ROWS);
                // SAFETY: `to` belongs to the caller's fresh fork and no reader; both pages stay
                // allocated until the prefix stream drains (the engine drains before release).
                unsafe { source.source_cache().copy_page_rows(f, t, count, self.prefix_stream.raw)?; }
                rows -= count;
            }
        }
        Ok(())
    }
    /// Publish `units`' payload to every source's peer replica (replicated layouts): after a
    /// host restore or a tail copy wrote them. Drains the prefix stream first.
    pub fn publish_units(&self, units: &[u32]) -> Result<()> {
        if !self.sources.iter().any(|s| s.source_cache().replicated()) { return Ok(()); }
        self.prefix_drain()?;
        let pages = self.unit_pages(units);
        for (source, pages) in self.sources.iter().zip(&pages) {
            // SAFETY: the pages' writes completed (drained above or host restore done) and the
            // caller owns them exclusively until they are published.
            unsafe { source.source_cache().publish_restored_pages(pages)?; }
        }
        Ok(())
    }
    /// Every source's page buffers (packed index, index scales, KV values, KV scales), for the
    /// host tier's copy engine to register.
    pub fn source_buffers(&self) -> Vec<CuteafdDeviceBuffer> {
        self.sources.iter().flat_map(|source| {
            let cache = source.source_cache();
            [cache.packed.buffer, cache.scales.buffer, cache.kv_values.buffer, cache.kv_scales.buffer]
        }).collect()
    }
    /// The device segments holding unit `unit` in every source (four per physical page), in
    /// the order the host tier stores them.
    pub fn unit_segments(&self, unit: u32) -> Vec<CuteafdDeviceBuffer> {
        let pages = self.unit_pages(&[unit]);
        self.sources.iter().zip(&pages)
            .flat_map(|(source, pages)| pages.iter().flat_map(|&p| source.source_cache().page_segments(p)))
            .collect()
    }
}

/// Model tokens in one prefix-engine unit: one page of each ratio-two source, two of layer 20.
pub(crate) const UNIT_TOKENS: usize = 512;
/// Bytes of one unit across the four sources (five pages).
pub(crate) const UNIT_BYTES: usize = 5 * PAGE_ROWS * crate::families::deepseek_v41::v41_compressor::SOURCE_ROW_BYTES;
use crate::families::deepseek_v41::v41_compressor::PAGE_ROWS;

/// Component tests: a retained snapshot as the prefix engine would keep it (its mark, metadata
/// and a copy of every source's initialized pages, standing in for the engine's shared units),
/// restored into a fresh slot's own pages.
#[cfg(test)]
pub(crate) struct TestSnapshot<'a> {
    mark: crate::shared::memory::DeviceAllocation<'a>,
    meta: BackboneMark,
    /// Per source: its rows and the copied page bytes (four planes per page, `page_segments`
    /// order).
    sources: Vec<(usize, Vec<Vec<u8>>)>,
}
#[cfg(test)]
impl TestSnapshot<'_> {
    pub fn end(&self) -> u64 { self.meta.end }
}
#[cfg(test)]
impl<'a> BackboneCache<'a> {
    fn test_snapshot(&self, lease: CacheLease, mark: crate::shared::memory::DeviceAllocation<'a>, meta: BackboneMark)
        -> Result<TestSnapshot<'a>> {
        let library = self.prefix_stream.library;
        let request = self.request_identity(lease)?;
        let mut sources = Vec::new();
        for (state, &lease) in self.sources.iter().zip(&request.sources) {
            let (pages, rows) = state.binding(lease)?;
            let device = state.device;
            let copied = device.run(|| pages[..rows.div_ceil(PAGE_ROWS)].iter().map(|&page| {
                state.source_cache().page_segments(page).iter().map(|&segment| {
                    let mut bytes = vec![0; segment.bytes];
                    library.copy_d2h_quiescent(&mut bytes, segment)?;
                    Ok(bytes)
                }).collect::<Result<Vec<_>>>().map(|planes| planes.concat())
            }).collect::<Result<Vec<_>>>())?;
            sources.push((rows, copied));
        }
        Ok(TestSnapshot { mark, meta, sources })
    }
    fn test_fork(&mut self, lease: CacheLease, saved: &TestSnapshot<'a>) -> Result<()> {
        let library = self.prefix_stream.library;
        let sources = self.request(lease)?.sources;
        for (((state, lease), (rows, pages)), i) in self.sources.iter_mut().zip(sources).zip(&saved.sources).zip(0..) {
            let rows = *rows;
            let (bound, _) = state.binding(lease)?;
            let bound = bound.to_vec();
            let device = state.device;
            device.run(|| {
                for (page, bytes) in bound.iter().zip(pages).take(rows.div_ceil(PAGE_ROWS)) {
                    let mut offset = 0;
                    for segment in state.source_cache().page_segments(*page) {
                        library.copy_h2d(segment, &bytes[offset..offset + segment.bytes])?;
                        offset += segment.bytes;
                    }
                }
                Ok(())
            })?;
            let _ = i;
            let device = state.device;
            device.run(|| state.bind(lease, &bound, rows))?;
        }
        Ok(())
    }
    pub fn retain_prefix(&mut self, lease: CacheLease, budget: usize) -> Result<TestSnapshot<'a>> {
        ensure!(budget >= BackboneMark::BYTES, "retained backbone tail exceeds budget");
        let mark = crate::shared::memory::DeviceAllocation::new(self.prefix_stream.library, BackboneMark::BYTES, Tag::prefix("v41/prefix-marks"))?;
        let meta = self.capture_mark(lease, mark.buffer)?;
        self.test_snapshot(lease, mark, meta)
    }
    pub fn queue_prefix(&mut self, lane: usize, lease: CacheLease, budget: usize) -> Result<()> {
        ensure!(budget >= BackboneMark::BYTES, "retained backbone tail exceeds budget");
        let mark = crate::shared::memory::DeviceAllocation::new(self.prefix_stream.library, BackboneMark::BYTES, Tag::prefix("v41/prefix-marks"))?;
        let meta = self.queue_mark(lane, lease, mark.buffer)?;
        self.test_queued[lane] = Some((lease, mark, meta));
        Ok(())
    }
    pub fn prefix_ready(&mut self, lane: usize, lease: CacheLease) -> Result<bool> {
        ensure!(self.test_queued.get(lane).and_then(Option::as_ref).is_some_and(|(l, _, _)| *l == lease),
            "backbone snapshot owner differs");
        self.request_identity(lease)?;
        let copies = self.prefix_copies.get(lane).context("invalid snapshot lane")?;
        copies.ready()
    }
    pub fn finish_prefix(&mut self, lane: usize, lease: CacheLease) -> Result<TestSnapshot<'a>> {
        ensure!(self.prefix_ready(lane, lease)?, "backbone snapshot copies are incomplete");
        ensure!(self.mark_ready(lane)?, "backbone snapshot copies are incomplete");
        let (_, mark, meta) = self.test_queued[lane].take().context("no queued snapshot")?;
        self.test_snapshot(lease, mark, meta)
    }
    pub fn abort_prefix(&mut self, lane: usize) -> Result<()> {
        let aborted = self.abort_mark(lane);
        self.test_queued[lane] = None;
        aborted
    }
    pub fn restore_prefix(&mut self, lease: CacheLease, saved: &TestSnapshot<'a>) -> Result<()> {
        self.test_fork(lease, saved)?;
        self.restore_mark(lease, &saved.meta, saved.mark.buffer)
    }
    pub fn restore_continuation(&mut self, lease: CacheLease, saved: &TestSnapshot<'a>, prompt_end: u64)
        -> Result<()> {
        ensure!(prompt_end <= 1048576 && prompt_end.saturating_sub(saved.meta.end) >= 128,
            "encoder continuation must cover the final decoder window");
        self.test_fork(lease, saved)?;
        self.restore_encoder_continuation(lease, &saved.meta, saved.mark.buffer, prompt_end)
    }
}

#[cfg(test)]
mod rollback_tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    #[test]
    fn poisoned_backbone_marks_idle_health_unavailable_without_an_admission() -> Result<()> {
        const CHILD: &str = "CUTEAFD_V41_POISON_HEALTH_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "families::deepseek_v41::v41_backbone_cache::prefix::rollback_tests::poisoned_backbone_marks_idle_health_unavailable_without_an_admission", "--nocapture"])
                .env(CHILD, "1").status()?;
            ensure!(status.success(), "isolated poison/health regression failed");
            return Ok(());
        }
        let fixture = Fixture::build_for_tests()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        let mut bank = BackboneCache::new(&lib, 1, [4, 4, 4, 8], usize::MAX)?;
        tokio::runtime::Runtime::new()?.block_on(async {
            use axum::{body::Body, http::{Request, StatusCode}};
            use tower::ServiceExt;
            let (queue, mut receive) = tokio::sync::mpsc::channel(1);
            let mut profile = cuteafd_api::openai::ModelProfile::default();
            profile.engine_health = Some(cuteafd_api::openai::health::HealthWitness(
                std::sync::Arc::new(cuteafd_transport::health::failure_reason)));
            let app = cuteafd_api::openai::router_for_model(queue, Default::default(),
                std::sync::Arc::new(std::sync::Mutex::new(serde_json::Value::Null)),
                std::time::Duration::from_secs(1), cuteafd_api::openai::ConsoleHub::disabled(), profile);
            let get = || Request::get("/health").body(Body::empty()).unwrap();
            assert_eq!(app.clone().oneshot(get()).await?.status(), StatusCode::OK);
            fixture.configure_pack(&lib, 0, 1)?;
            let lease = bank.begin_request(0, 7)?;
            assert!(bank.prefix_drain().is_err());
            assert!(bank.poisoned.get());
            assert!(bank.bind_sources(lease, [&[], &[], &[], &[]], [0; 4]).is_err());
            assert!(bank.begin_request(0, 8).is_err(), "queued second request cannot reuse the slot");
            let response = app.oneshot(get()).await?;
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            let body = axum::body::to_bytes(response.into_body(), 4096).await?;
            assert!(String::from_utf8_lossy(&body).contains("prefix copy drain failed"));
            fixture.configure_pack(&lib, 0, 0)?;
            assert!(receive.try_recv().is_err(), "health must not require another admission");
            anyhow::Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn bounded_replay_drains_forked_tail_before_publishing() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let lib = fixture.load()?;
        // This fixture defers device-to-device copies until their stream drains.
        let mut bank = BackboneCache::new(&lib, 1, [4, 4, 4, 8], usize::MAX)?;
        bank.use_engine_pages()?;
        let lease = bank.begin_request(0, 7)?;
        let from = bank.unit_pages(&[0]);
        let to = bank.unit_pages(&[1]);
        for (i, state) in bank.sources.iter().enumerate() {
            for (page, byte) in [(from[i][0], 0x2a), (to[i][0], 0x59)] {
                for segment in state.source_cache().page_segments(page) {
                    lib.copy_h2d(segment, &vec![byte; segment.bytes])?;
                }
            }
        }
        bank.copy_unit_rows(0, 1, 200)?;
        let tail = bank.sources[0].source_cache().page_segments(to[0][0])[0];
        let mut bytes = vec![0; tail.bytes];
        lib.copy_d2h_quiescent(&mut bytes, tail)?;
        assert_eq!(bytes[0], 0x59, "fork is not visible before the copy drains");
        bank.bind_sources(lease, [&to[0], &to[1], &to[2], &to[3]], bank.source_rows(200))?;
        assert_eq!(bank.restore_encoder_prefix(lease, 200, 72, 225)?, 72);
        lib.copy_d2h_quiescent(&mut bytes, tail)?;
        assert_eq!(bytes[0], 0x2a, "replay can read the copied tail at publication");
        assert_eq!(bank.history_end(lease)?, 72);
        assert_eq!(bank.committed_end(lease)?, 200);
        let request = bank.request(lease)?;
        for (i, (state, &window)) in bank.windows.iter().zip(&request.windows).enumerate() {
            let start = if i < 20 { 72 } else { 0 };
            assert_eq!(state.span(window)?, (start, start));
        }
        bank.release(&[lease])?;
        Ok(())
    }

    #[test]
    fn replay_upload_failure_at_layer_k_keeps_the_lease_for_cold_rebind() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        let mut bank = BackboneCache::new(&lib, 1, [4, 4, 4, 8], usize::MAX)?;
        bank.use_engine_pages()?;
        let lease = bank.begin_request(0, 7)?;
        let pages = bank.unit_pages(&[0]);
        bank.bind_sources(lease, [&pages[0], &pages[1], &pages[2], &pages[3]], bank.source_rows(200))?;
        fixture.fail_h2d_after(&lib, 7)?;
        assert!(bank.restore_encoder_prefix(lease, 200, 72, 225).is_err());
        let request = bank.request(lease)?;
        assert_eq!(bank.windows[6].span(request.windows[6])?, (72, 72));
        assert_eq!(bank.windows[7].span(request.windows[7])?, (0, 0));
        bank.reset_prefix(lease)?;
        let pages = bank.unit_pages(&[1]);
        bank.bind_sources(lease, [&pages[0], &pages[1], &pages[2], &pages[3]], [0; 4])?;
        assert_eq!(bank.request_id(lease)?, 7);
        assert_eq!(bank.history_end(lease)?, 0);
        bank.begin_encoder(lease, 225)?;
        bank.release(&[lease])?;
        Ok(())
    }

    #[test]
    fn failed_restore_keeps_lease_and_cold_rebinds_after_partial_layers() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        let mut bank = BackboneCache::new(&lib, 2, [4, 4, 4, 8], usize::MAX)?;
        bank.use_engine_pages()?;
        let lease = bank.begin_request(0, 7)?;
        let fresh = bank.begin_request(1, 8)?;
        let mark = crate::shared::memory::DeviceAllocation::new(&lib, BackboneMark::BYTES,
            crate::shared::memory::TEST)?;
        let mut metadata = BackboneMark { end: 1173, windows: vec![(1045, 1173); 40] };
        // Fail only after twenty windows have already been installed.
        metadata.windows[20] = (1174, 1173);
        let pages = bank.unit_pages(&[0, 1, 2]);
        let refs = [&pages[0][..], &pages[1][..], &pages[2][..], &pages[3][..]];
        bank.bind_sources(lease, refs, bank.source_rows(1173))?;
        assert!(bank.restore_mark(lease, &metadata, mark.buffer).is_err());
        assert_eq!(bank.request_id(lease)?, 7, "restore errors cannot release the outer lease");
        bank.reset_prefix(lease)?;
        // Different pages, fewer initialized rows: the stale binding used to reject this.
        let pages = bank.unit_pages(&[3]);
        let refs = [&pages[0][..], &pages[1][..], &pages[2][..], &pages[3][..]];
        bank.bind_sources(lease, refs, [0; 4])?;
        let a = bank.request(lease)?;
        let b = bank.request(fresh)?;
        assert_eq!((a.end, a.phase), (b.end, b.phase));
        for ((state, &x), &y) in bank.windows.iter().zip(&a.windows).zip(&b.windows) {
            assert_eq!(state.span(x)?, state.span(y)?);
            let mut x_end = [0; 8]; let mut y_end = [0; 8];
            lib.copy_d2h_quiescent(&mut x_end, state.view(x)?.device_end)?;
            lib.copy_d2h_quiescent(&mut y_end, state.view(y)?.device_end)?;
            assert_eq!(x_end, y_end);
        }
        for (state, &lease) in bank.sources.iter().zip(&a.sources) {
            assert_eq!(state.committed_end(lease)?, 0);
            assert_eq!(state.binding(lease)?.1, 0);
        }
        bank.begin_encoder(lease, 1173)?;
        bank.release(&[lease, fresh])?;
        Ok(())
    }
}

/// An ordering event owned by the backbone cache (destroyed with it).
pub(crate) struct PrefixEvent<'a>(pub(crate) *mut std::ffi::c_void, &'a NativeLibrary);
impl<'a> PrefixEvent<'a> {
    pub(crate) fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self(library.cuda_event_create_ordering()?, library))
    }
}
impl Drop for PrefixEvent<'_> {
    fn drop(&mut self) {
        // SAFETY: created here; the streams that wait on it are drained by their owners first
        // (the cache's prefix stream and snapshot lanes drop before or with this event).
        if let Err(error) = unsafe { self.1.cuda_event_destroy(self.0) } {
            tracing::error!(%error, "destroying V4.1 prefix ordering event");
        }
    }
}
