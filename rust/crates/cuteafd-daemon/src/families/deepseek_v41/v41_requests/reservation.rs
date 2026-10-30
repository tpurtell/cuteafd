//! Bind reserved prompt rows to a private Engram preparation frontier.
use super::*;
impl Requests<'_> {
    /// # Safety
    /// Retain query, cache and producer owners; abort queued writes on cancellation.
    pub unsafe fn poll_distributed_encoder_production(&mut self, batch: &RequestBatch,
        pending: &mut crate::families::deepseek_v41::v41_backbone_execution::PendingDistributedProduction<'_, '_, '_, '_>) -> Result<bool> {
        self.validate(batch)?;
        unsafe { pending.poll_encoder(&mut self.cache, batch.cache()?) }
    }

    /// Device-ordered prefill units: publish `layer`'s window as a chained
    /// direct commit (see `DistributedExecution::publish_unit_window`).
    pub fn publish_distributed_unit_window(&mut self, batch: &RequestBatch,
        execution: &mut crate::families::deepseek_v41::v41_backbone_execution::DistributedExecution<'_, '_>,
        layer: usize) -> Result<()> {
        self.validate(batch)?;
        execution.publish_unit_window(&mut self.cache, batch.cache()?, layer)
    }

    pub fn finish_distributed_encoder_publication(&mut self, batch: &RequestBatch,
        execution: &mut crate::families::deepseek_v41::v41_backbone_execution::DistributedExecution<'_, '_>) -> Result<()> {
        self.validate(batch)?;
        execution.finish_encoder_publication(&mut self.cache, batch.cache()?)
    }

    /// # Safety
    /// The prepared layer-20 boundary and execution owner belong to this batch.
    pub unsafe fn publish_encoder_boundary(&mut self, batch: &RequestBatch,
        execution: &mut BackboneExecution<'_, '_>, lane: &BackboneLane<'_, '_>) -> Result<()> {
        self.validate(batch)?;
        unsafe { execution.publish_encoder_boundary(&mut self.cache, batch.cache()?, lane) }
    }

    /// # Safety
    /// The lane/query/index owners identify this batch and have completed their
    /// producers. Poll and complete the returned owner on the CUDA thread.
    pub unsafe fn prepare_encoder_layer<'l, 'lw, 'la>(&mut self, batch: &RequestBatch,
        execution: &mut BackboneExecution<'_, '_>, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut crate::families::deepseek_v41::v41_index_lane::IndexLane<'_, '_>)
        -> Result<crate::families::deepseek_v41::v41_backbone_execution::PreparedLayer<'l, 'lw, 'la>> {
        self.validate(batch)?;
        unsafe { execution.prepare_encoder_layer(&mut self.cache, batch.cache()?, lane, index) }
    }

    /// Start the entire mandatory burst before the executor can begin it.
    /// A private cursor hashes later known chunks without publishing history.
    /// Image requests intentionally prepare at reservation instead.
    pub fn prefetch_encoder(&mut self, lease: CacheLease, chunks: &[&[u32]]) -> Result<()> {
        self.pipeline.reclaim_copies()?;
        if chunks.is_empty() || self.image_requests > 0 { return Ok(()); }
        let live = self.request(lease)?;
        let mut cursor = live.prefill.as_ref().map_or(&live.history, |p| p.history()).prefill_cursor();
        for tokens in chunks.iter().take(self.pipeline.shape().prefetch_units()) {
            let key = prefetch::PrefetchKey { lease, position: cursor.history().position(), tokens: tokens.to_vec() };
            if let Some(wave) = self.prefetched.matching(&key) {
                cursor.advance_full(&wave.batches()[0])?;
            } else {
                let wave = self.pipeline.prepare(&[EngramRequestTokens {
                    history: cursor.history(), token_ids: tokens, image_mask: None, l14_gate: self.l14_gate(lease) }])?;
                cursor.advance_full(&wave.batches()[0])?;
                self.prefetched.hold(key, wave);
            }
        }
        Ok(())
    }

    pub fn reserve_encoder(&mut self, requests: &[RequestTokens<'_>]) -> Result<RequestBatch> {
        ensure!(!requests.is_empty() && requests.len() <= 16, "invalid encoder request count");
        let masks = self.resolved_masks(requests, true)?;
        let mut prepared = Vec::with_capacity(requests.len());
        let mut work = Vec::with_capacity(requests.len());
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        for (i, r) in requests.iter().enumerate() {
            ensure!(!requests[..i].iter().any(|p| p.lease == r.lease), "duplicate encoder request");
            ensure!(r.kind == ExpertV2SourceKind::Prefill && !r.tokens.is_empty()
                && r.image_mask.is_none_or(|m| m.len() == r.tokens.len()),
                "invalid encoder tokens, kind or image mask");
            let live = self.request(r.lease)?;
            ensure!(live.history.position() == self.cache.committed_end(r.lease)?,
                "encoder accepted histories differ");
            let history = live.prefill.as_ref().map_or(&live.history, |p| p.history());
            ensure!(history.position() == self.cache.encoder_prepared_end(r.lease)?,
                "encoder preparation frontiers differ");
            prepared.push(history.prefill_cursor());
            work.push(CacheWork { lease: r.lease, tokens: u32::try_from(r.tokens.len())?, kind: r.kind });
            tokens.extend_from_slice(r.tokens);
            let image_mask = masks.as_ref().map_or(r.image_mask, |m| m[i].as_deref());
            mask.extend((0..r.tokens.len()).map(|i| u8::from(image_mask.is_some_and(|m| m[i]))));
        }
        let inputs = requests.iter().zip(&prepared).enumerate().map(|(i, (r, p))| EngramRequestTokens {
            history: p.history(), token_ids: r.tokens,
            image_mask: masks.as_ref().map_or(r.image_mask, |m| m[i].as_deref()),
            l14_gate: self.l14_gate(r.lease),
        }).collect::<Vec<_>>();
        // Adopt this reservation's exact wave, or cancel only its participants'
        // stale lookahead; unrelated requests retain their mandatory burst.
        let key = (requests.len() == 1 && masks.is_none()).then(|| prefetch::PrefetchKey {
            lease: requests[0].lease, position: prepared[0].history().position(), tokens: requests[0].tokens.to_vec() });
        let adopted = match &key {
            Some(key) => self.prefetched.take_matching(key),
            None => {
                for request in requests { self.prefetched.cancel_owner(&request.lease); }
                None
            }
        };
        let mut engram = match adopted { Some(wave) => wave, None => self.pipeline.prepare(&inputs)? };
        let advanced = prepared.iter().zip(engram.batches()).map(|(p, b)| {
            let mut next = p.history().prefill_cursor();
            next.advance_full(b)?;
            Ok(next)
        }).collect::<Result<Vec<_>>>()?;
        // No request preparation frontier changes unless every cache reservation
        // succeeds. Rejected capacity/extent cancels the speculative I/O work.
        let cache = match self.cache.reserve_encoder(&work) {
            Ok(cache) => cache,
            Err(error) => { engram.cancel(); return Err(error); }
        };
        for (r, next) in requests.iter().zip(advanced) {
            self.slots.iter_mut().flatten().find(|live| live.lease == r.lease)
                .expect("validated encoder request").prefill = Some(next);
        }
        Ok(RequestBatch { cache, prepared, engram: Some(engram),
            leases: requests.iter().map(|r| r.lease).collect(), tokens, image_mask: mask, finished: false })
    }
}

