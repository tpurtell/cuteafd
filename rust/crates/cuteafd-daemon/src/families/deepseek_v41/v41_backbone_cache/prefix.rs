//! The backbone's part of a prefix snapshot: every window's ring rows and every compressed
//! source's carry, copied into a positional mark the prefix engine's arena holds
//! ([`BackboneMark::BYTES`]), described by plain data ([`BackboneMark`]). Global source rows live
//! in pages the engine forks; nothing here owns device memory.
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
    /// Start a fresh request at the complete-group frontier `end` of source pages the prefix
    /// engine forked and bound, with empty encoder windows at the bounded replay start
    /// `end - 128`. Decoder windows remain fresh. Replay reads the shared source rows without
    /// producing them. Returns the replay start.
    pub fn restore_encoder_prefix(&mut self, lease: CacheLease, end: u64, prompt_end: u64) -> Result<u64> {
        let request = self.request(lease)?;
        ensure!(
            end > 0
                && end % 2 == 0
                && end <= prompt_end
                && prompt_end <= 1048576
                && request.end == 0
                && request.version == 0
                && request.phase == CachePhase::Full
                && request.publication.is_empty(),
            "invalid encoder prefix restore frontier"
        );
        let start = end.saturating_sub(128);
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
        if let Err(error) = result {
            if let Err(cleanup) = self.release(&[lease]) {
                tracing::error!(%cleanup, "releasing failed encoder prefix restore");
            }
            return Err(error);
        }
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
    pub fn queue_mark(&mut self, lane: usize, lease: CacheLease, mark: CuteafdDeviceBuffer) -> Result<BackboneMark> {
        let copies = self.prefix_copies.get(lane).context("invalid snapshot lane")?;
        ensure!(copies.pending.is_none(), "backbone snapshot lane is occupied");
        let stream = copies.stream.raw;
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
    /// `saved.end`. Any partially applied failure revokes the whole request after draining; no
    /// mixed-layer frontier is observable.
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
        let drained = unsafe { self.prefix_stream.library.cuda_stream_synchronize(self.prefix_stream.raw) };
        if let Err(error) = result.and(drained) {
            if let Err(cleanup) = self.release(&[lease]) {
                tracing::error!(%cleanup, "releasing failed prefix restore");
            }
            return Err(error);
        }
        let request = self.requests[lease.slot]
            .as_mut()
            .expect("validated prefix admission");
        request.end = saved.end;
        request.phase = phase;
        request.version = 1;
        self.committed_end(lease)?;
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

    /// Wait for every snapshot copy and restore queued on the prefix stream.
    pub fn drain_prefix_stream(&self) -> Result<()> {
        unsafe { self.prefix_stream.library.cuda_stream_synchronize(self.prefix_stream.raw) }
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
        self.drain_prefix_stream()?;
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
                    library.copy_d2h(&mut bytes, segment)?;
                    Ok(bytes)
                }).collect::<Result<Vec<_>>>().map(|planes| planes.concat())
            }).collect::<Result<Vec<_>>>())?;
            sources.push((rows, copied));
        }
        Ok(TestSnapshot { mark, meta, sources })
    }
    fn test_fork(&mut self, lease: CacheLease, saved: &TestSnapshot<'a>, end: Option<u64>) -> Result<()> {
        let library = self.prefix_stream.library;
        let sources = self.request(lease)?.sources;
        for (((state, lease), (rows, pages)), i) in self.sources.iter_mut().zip(sources).zip(&saved.sources).zip(0..) {
            let rows = end.map_or(*rows, |end| end as usize / state.ratio());
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
        let mark = crate::shared::memory::DeviceAllocation::new(self.prefix_stream.library, BackboneMark::BYTES)?;
        let meta = self.capture_mark(lease, mark.buffer)?;
        self.test_snapshot(lease, mark, meta)
    }
    pub fn queue_prefix(&mut self, lane: usize, lease: CacheLease, budget: usize) -> Result<()> {
        ensure!(budget >= BackboneMark::BYTES, "retained backbone tail exceeds budget");
        let mark = crate::shared::memory::DeviceAllocation::new(self.prefix_stream.library, BackboneMark::BYTES)?;
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
        self.test_fork(lease, saved, None)?;
        self.restore_mark(lease, &saved.meta, saved.mark.buffer)
    }
    pub fn restore_continuation(&mut self, lease: CacheLease, saved: &TestSnapshot<'a>, prompt_end: u64)
        -> Result<()> {
        ensure!(prompt_end <= 1048576 && prompt_end.saturating_sub(saved.meta.end) >= 128,
            "encoder continuation must cover the final decoder window");
        self.test_fork(lease, saved, None)?;
        self.restore_encoder_continuation(lease, &saved.meta, saved.mark.buffer, prompt_end)
    }
    pub fn restore_encoder_prefix_from(&mut self, lease: CacheLease, saved: &TestSnapshot<'a>, end: u64,
        prompt_end: u64) -> Result<u64> {
        ensure!(end <= saved.meta.end && end % 2 == 0, "encoder prefix past the snapshot or unaligned");
        self.test_fork(lease, saved, Some(end))?;
        self.restore_encoder_prefix(lease, end, prompt_end)
    }
}
