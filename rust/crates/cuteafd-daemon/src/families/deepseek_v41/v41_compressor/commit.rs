//! Lane-owned compressed source publication; page claims survive GPU work.
use super::*;
pub(super) struct PendingCommit {
    prepared: Prepared,
    plan: source_cache::IndexPlan,
    completed: Vec<CompressorLatentRow>,
    accepted: Vec<u32>,
}
impl<'a> CompressorWave<'_, 'a> {
    /// Configure once before production. Each lane owns its publication events;
    /// replicas share the authoritative pool's physical page identities.
    pub fn enable_replica(&mut self, state: &CompressorState<'a>,
        replica: std::rc::Rc<SourceReplica<'a>>) -> Result<()> {
        ensure!(self.ready.is_none() && self.pending_query.is_none() && self.pending_commit.is_none()
            && self.replica.is_none(), "source replica requires an unused wave");
        replica.validate_owner(&state.index)?;
        ensure!(state.layer==self.weights.layer && state.index.kv_values.buffer.device_id==self.kv_values().device_id
            && std::ptr::eq(state.index.kv_values.library,self.stream().library), "source replica producer differs");
        let source=crate::shared::memory::device::Device { library:self.stream().library,id:self.kv_values().device_id };
        let publication=crate::shared::memory::peer_publication::PeerPublication::new(source,replica.device())?;
        self.replica=Some((replica,publication)); Ok(())
    }
    /// # Safety
    /// Proposal consumers are complete. Keep wave and state alive and drain on
    /// every error before dropping page claims or releasing participating slots.
    pub unsafe fn enqueue_commit(&mut self, state: &CompressorState<'_>, accepted: &[u32]) -> Result<()> {
        let (lib, raw) = (self.stream().library, self.stream().raw);
        let storage = self.storage.clone();
        let (cache_destinations, index_packed, index_scales, kv_values, kv_scales, projected, scores) = (self.cache_destinations(), self.index_packed(), self.index_scales(), self.kv_values(), self.kv_scales(), self.projected(), self.scores());
        ensure!(self.pending_commit.is_none(), "compressor commit already pending");
        if let Some(configured)=&state.index.replica {
            ensure!(self.replica.as_ref().is_some_and(|(storage,_)|
                std::rc::Rc::ptr_eq(storage,&configured.storage)),"compressor producer replica is not configured");
        }
        if let Some((replica,_))=&self.replica { replica.validate_owner(&state.index)?; }
        let prepared = self
            .ready
            .take()
            .context("compressor has no proposal to commit")?;
        ensure!(
            prepared.owner == state.owner && accepted.len() == prepared.chunks.len(),
            "compressor commit binding differs"
        );
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            let slot = state.validate(chunk.lease)?;
            ensure!(
                state.slots[slot].version == prepared.versions[i]
                    && state.slots[slot].end == chunk.position,
                "stale compressor proposal"
            );
            ensure!(
                accepted[i] <= chunk.tokens,
                "compressor acceptance exceeds proposal"
            );
            state.slots[slot]
                .version
                .checked_add(1)
                .context("compressor version exhausted")?;
        }
        // A ratio-two carry is read from this layer's unit-cap scratch: valid
        // while no later production of the layer was queued on this stream.
        if state.pending.is_some() && !self.carry_intact() {
            self.ready = Some(prepared);
            anyhow::bail!("compressor carry of layer {} overwritten by a later production", self.weights.layer);
        }
        let ratio = self.weights.ratio;
        let mut completed = vec![];
        let mut appends = vec![];
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            let end = prepared.offsets[i] + accepted[i] as usize;
            completed.extend(
                prepared
                    .completed
                    .iter()
                    .filter(|r| {
                        r.source_row as usize >= prepared.offsets[i]
                            && (r.source_row as usize) < end
                    })
                    .copied(),
            );
            appends.push((
                chunk.lease.slot,
                chunk.position as usize / ratio,
                (chunk.position as usize + accepted[i] as usize) / ratio,
            ));
        }
        let plan = state.index.reserve(&appends)?;
        let mut destinations = vec![u64::MAX; prepared.rows];
        for row in &completed {
            destinations[row.source_row as usize] =
                state
                    .index
                    .destination(&plan, row.lease.slot, row.position as usize / ratio)?;
        }
        // Commit is outside the proposal graph because acceptance is only known
        // afterward; its destinations use their own pinned buffer.
        for (dst, value) in self.commit_destinations.bytes_mut()[..prepared.rows * 8]
            .chunks_exact_mut(8)
            .zip(&destinations)
        {
            dst.copy_from_slice(&value.to_ne_bytes());
        }
        let slice = |b: CuteafdDeviceBuffer, row: usize| CuteafdDeviceBuffer {
            ptr: unsafe { b.ptr.cast::<u8>().add(row * 2048).cast() },
            bytes: 2048,
            ..b
        };

        self.pending_commit = Some(PendingCommit { prepared, plan, completed, accepted: accepted.to_vec() });
        let pending = self.pending_commit.as_ref().unwrap();
        let prepared = &pending.prepared;
        let plan = &pending.plan;
        let completed = &pending.completed;
        (|| -> Result<()> {
            if !completed.is_empty() {
                unsafe {
                    lib.copy_h2d_async(
                        cache_destinations,
                        &self.commit_destinations.bytes_mut()[..prepared.rows * 8],
                        raw,
                    )?;
                    storage.kernel.index_store(
                        index_packed,
                        index_scales,
                        cache_destinations,
                        state.index.packed.buffer,
                        state.index.scales.buffer,
                        prepared.rows,
                        state.index.capacity,
                        raw,
                    )?;
                    self.kv.store(
                        kv_values,
                        kv_scales,
                        cache_destinations,
                        state.index.kv_values.buffer,
                        state.index.kv_scales.buffer,
                        prepared.rows,
                        state.index.capacity,
                        raw,
                    )?;
                }
            }
            if let Some(pending) = &state.pending {
                for (i, chunk) in prepared.chunks.iter().enumerate() {
                    let count = accepted[i] as usize;
                    if count > 0 && (chunk.position + count as u64) % 2 == 1 {
                        let row = prepared.offsets[i] + count - 1;
                        for (dst, src) in [
                            (pending[0].buffer, projected),
                            (pending[1].buffer, scores.unwrap()),
                        ] {
                            unsafe {
                                lib.copy_d2d_async(
                                    slice(dst, chunk.lease.slot),
                                    slice(src, row),
                                    2048,
                                    raw,
                                )?;
                            }
                        }
                    }
                }
            }
            unsafe {
                state.index.upload_staged(plan, self.commit_staging.bytes_mut(), raw)?;
                if let Some((replica,publication))=self.replica.as_mut() {
                    publication.enqueue(raw,|stream|replica.copy_append(&state.index,plan,stream))?;
                }
            }
            Ok(())
        })()

    }
    pub fn poll_commit(&self) -> Result<bool> {
        if self.pending_commit.is_none() { return Ok(true); }
        unsafe { self.stream().library.cuda_stream_query(self.stream().raw) }
    }
    pub(super) fn validate_pending_commit(&self, state: &CompressorState<'_>) -> Result<&Prepared> {
        let pending = self.pending_commit.as_ref().context("source commit absent")?;
        let p = &pending.prepared;
        ensure!(p.owner == state.owner, "queued source owner differs");
        state.index.validate_plan(&pending.plan)?;
        for (i, chunk) in p.chunks.iter().enumerate() {
            let slot = state.validate_identity(chunk.lease)?;
            ensure!(state.slots[slot].version == p.versions[i] && state.slots[slot].end == chunk.position,
                "queued source binding changed");
        }
        Ok(p)
    }
    pub fn commit(&mut self, state: &mut CompressorState<'_>, accepted: &[u32]) -> Result<Vec<CompressorLatentRow>> {
        let direct = self.pending_commit.is_none();
        if direct {
            // A direct (prompt publication) commit inside a device-ordered
            // chain follows every chained reader of the private wave (#815
            // units); outside a chain they already completed (no-op).
            // SAFETY: this wave's live stream on the chain's device.
            unsafe { crate::shared::memory::chain::join(self.stream().library, self.stream().raw)?; }
            let launched = unsafe { self.enqueue_commit(state, accepted) };
            // Device-ordered prefill units (#815 option (i)) record the chain
            // head instead of a host wait: every later reader of this source's
            // published rows joins it (the same unit's later stages, and the
            // next unit through its publication event). Its host staging is
            // this pass's own and is reused only after the unit's drained commit.
            let completed = if crate::shared::memory::chain::deferred() {
                unsafe { crate::shared::memory::chain::finish(self.stream().library, self.stream().raw) }
            } else { self.synchronize() };
            if let Err(error) = launched.and(completed) {
                if let Err(cleanup) = self.abort_commit(state) { tracing::error!(%cleanup, "revoking failed source commit"); }
                return Err(error);
            }
        }
        self.validate_pending_commit(state)?;
        ensure!(self.pending_commit.as_ref().unwrap().accepted == accepted, "queued source acceptance changed");
        ensure!(direct || self.poll_commit()?, "source commit incomplete");
        let pending = self.pending_commit.take().unwrap();
        self.unsettled.clear();
        if direct && crate::shared::memory::chain::deferred() {
            self.unsettled.extend(pending.prepared.chunks.iter().map(|c| c.lease.slot));
        }
        state.index.apply(pending.plan);
        for (chunk, &count) in pending.prepared.chunks.iter().zip(accepted) {
            state.slots[chunk.lease.slot].end = chunk.position + u64::from(count);
            state.slots[chunk.lease.slot].version += 1;
        }
        Ok(pending.completed)
    }
    pub fn abort_commit(&mut self, state: &mut CompressorState<'_>) -> Result<()> {
        if self.pending_commit.is_none() {
            // A device-ordered direct commit already advanced the host ends:
            // drain its queued write and revoke the slots (hunt C #2).
            if self.unsettled.is_empty() { return Ok(()); }
            let drained = self.synchronize();
            let mut revoked = Ok(());
            for slot in std::mem::take(&mut self.unsettled) {
                if state.slots[slot].request.take().is_some() {
                    if let Err(error) = state.index.release(slot) { revoked = Err(error); }
                }
            }
            return drained.and(revoked);
        }
        let drained = self.synchronize();
        let pending = self.pending_commit.take().unwrap();
        let slots = pending.prepared.chunks.iter().map(|c| c.lease.slot).collect::<Vec<_>>();
        drop(pending); // Drain before returning claimed pages and write slots.
        let mut revoked = Ok(());
        for slot in slots {
            state.slots[slot].request = None;
            if let Err(error) = state.index.release(slot) { revoked = Err(error); }
        }
        drained.and(revoked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(state: &CompressorState<'_>, slot: usize, rows: usize, lib: &NativeLibrary) -> Result<Vec<u8>> {
        let index = state.index.view(slot, rows);
        let kv = state.index.kv_view(slot, rows);
        let mut bytes = Vec::new();
        for row in 0..rows {
            let physical = index.pages[row / 256] as usize*256 + row % 256;
            for (buffer, width) in [(index.packed, 64), (index.scales, 4),
                (kv.values, V41Kv::COMPRESSED_VALUE_BYTES), (kv.scales, V41Kv::COMPRESSED_SCALE_BYTES)] {
                let start = bytes.len(); bytes.resize(start+width, 0);
                let source = CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(physical*width).cast() }, bytes: width, ..buffer };
                lib.copy_d2h_quiescent(&mut bytes[start..], source)?;
            }
        }
        Ok(bytes)
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn native_independent_source_commits_match_direct_values_carry_and_abort() -> Result<()> {
        source_commits(false)
    }
    #[test]
    #[ignore = "requires SM peer copy, CUTEAFD_CACHE_COMMIT_MODEL and two CUDA GPUs"]
    fn native_independent_source_commits_publish_peer_payloads_and_metadata() -> Result<()> {
        source_commits(true)
    }
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Shared { Ordered, PrivateStreams, LateCommit, LateCommitUnchecked }
    /// row_window contract, compressor storage: two units (slots 0 and 1) on
    /// one shared storage of ratio-two source layer 2, windows 0..13 and
    /// 16..29 of a 48-row reserve, each producing and committing 13 tokens (so
    /// the commit copies the carry row 12 from the unit-cap scratch).
    /// - `Ordered`: the gate holds the shared stream after unit A wrote its
    ///   scratch input and before its kernels; A then commits, B produces and
    ///   commits. Every host staging write happens before any of it runs, B
    ///   must stay pending behind A, and both units' cache rows and carries
    ///   equal private runs.
    /// - `PrivateStreams` (negative control for the shared stream): each wave
    ///   has its own stream, B runs while A is held and clobbers A's scratch.
    /// - `LateCommit`: B produces before A commits; A's commit must refuse
    ///   (its carry scratch was overwritten).
    /// - `LateCommitUnchecked` (negative control for the carry check): the
    ///   refusal is skipped and A's carry row differs from its private run.
    fn shared_compressor_storage(mode: Shared) -> Result<()> {
        use crate::shared::memory::row_window::RowWindow;
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let model = std::env::var("CUTEAFD_CACHE_COMMIT_MODEL")?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model))?;
        let layer = 2;
        let weights = CompressorWeights::load(&lib, &catalog, layer,
            CompressorWeights::device_bytes(&catalog, layer)?, 1024*1024)?;
        let (unit, reserve, rows) = (16usize, 48usize, 13usize);
        let input = |seed: usize| -> Vec<u8> { (0..rows*5120).flat_map(|i| {
            let value = (((i * 7 + seed * 11) % 19) as f32 / 19.0 - 0.3).to_bits();
            ((value >> 16) as u16).to_ne_bytes()
        }).collect() };
        let chunk = |lease| CompressorChunk { lease, position: 0, tokens: rows as u32 };
        let carry = |state: &CompressorState<'_>, slot: usize| -> Result<Vec<u8>> {
            let mut bytes = vec![0; 4096];
            for (k, pending) in state.pending.as_ref().context("ratio-two carry")?.iter().enumerate() {
                lib.copy_d2h_quiescent(&mut bytes[k*2048..(k+1)*2048], CuteafdDeviceBuffer {
                    ptr: unsafe { pending.buffer.ptr.cast::<u8>().add(slot * 2048).cast() }, bytes: 2048, ..pending.buffer })?;
            }
            Ok(bytes)
        };
        let cached = rows / 2;
        // Private runs: expected rows, and an ungated warm-up of every kernel
        // and copy the gated run issues (lazy module loading would otherwise
        // synchronize behind the gate).
        let mut expected = Vec::new();
        for seed in 0..2 {
            let mut reference = CompressorState::new(&lib, layer, 1, 2, usize::MAX)?;
            let lease = reference.begin_request(0, 11)?;
            let mut private = weights.wave(unit, usize::MAX)?;
            lib.copy_h2d(private.input(), &input(seed))?;
            unsafe { private.execute(&reference, &[chunk(lease)])?; }
            private.commit(&mut reference, &[rows as u32])?;
            expected.push((read(&reference, 0, cached, &lib)?, carry(&reference, 0)?));
        }
        let sources = [input(0), input(1)].map(|host| -> Result<DeviceAllocation<'_>> {
            let buffer = DeviceAllocation::new(&lib, host.len(), crate::shared::memory::TEST)?;
            lib.copy_h2d(buffer.buffer, &host)?;
            Ok(buffer)
        }).into_iter().collect::<Result<Vec<_>>>()?;
        let storage = CompressorStorage::new(&lib, reserve, unit, &[layer])?;
        let private_streams = mode == Shared::PrivateStreams;
        crate::families::deepseek_v41::v41_compressor::PRIVATE_STREAM.with(|p| p.set(private_streams));
        let built = (|| -> Result<Vec<_>> { Ok(vec![weights.on_storage(storage.clone())?, weights.on_storage(storage.clone())?]) })();
        crate::families::deepseek_v41::v41_compressor::PRIVATE_STREAM.with(|p| p.set(false));
        let mut slots = built?;
        assert_eq!(std::ptr::eq(slots[0].stream(), slots[1].stream()), !private_streams);
        slots[0].select_window(RowWindow::within(0..rows, reserve)?)?;
        slots[1].select_window(RowWindow::within(unit..unit + rows, reserve)?)?;
        let mut state = CompressorState::new(&lib, layer, 2, 4, usize::MAX)?;
        let leases = [state.begin_request(0, 11)?, state.begin_request(1, 22)?];
        // Stage the scratch input; the gate (unit A only) goes before the kernels.
        let stage = |wave: &mut CompressorWave<'_, '_>, seed: usize, state: &CompressorState<'_>| -> Result<Prepared> {
            let prepared = wave.prepare(state, &[chunk(leases[seed])])?;
            unsafe { lib.copy_d2d_async(wave.input(), sources[seed].buffer, rows * 10240, wave.stream().raw)?; }
            wave.upload_queued(&prepared)?;
            Ok(prepared)
        };
        let launch = |wave: &mut CompressorWave<'_, '_>, prepared: Prepared, state: &CompressorState<'_>| -> Result<()> {
            unsafe { wave.enqueue(state, prepared.rows)?; }
            wave.ready = Some(prepared);
            Ok(())
        };
        let late = matches!(mode, Shared::LateCommit | Shared::LateCommitUnchecked);
        let prepared = stage(&mut slots[0], 0, &state)?;
        let mut gate = Some(unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&lib, slots[0].stream().raw)? });
        launch(&mut slots[0], prepared, &state)?;
        if !late { unsafe { slots[0].enqueue_commit(&state, &[rows as u32])?; } }
        let prepared = stage(&mut slots[1], 1, &state)?;
        launch(&mut slots[1], prepared, &state)?;
        unsafe { slots[1].enqueue_commit(&state, &[rows as u32])?; }
        if late {
            crate::families::deepseek_v41::v41_compressor::SKIP_CARRY_CHECK
                .with(|skip| skip.set(mode == Shared::LateCommitUnchecked));
            let refused = unsafe { slots[0].enqueue_commit(&state, &[rows as u32]) };
            crate::families::deepseek_v41::v41_compressor::SKIP_CARRY_CHECK.with(|skip| skip.set(false));
            if mode == Shared::LateCommit {
                let error = refused.expect_err("a commit after another production of its layer must refuse");
                assert!(error.to_string().contains("carry"), "{error:#}");
                drop(gate.take());
                slots[1].synchronize()?;
                slots[1].commit(&mut state, &[rows as u32])?;
                assert_eq!((read(&state, 1, cached, &lib)?, carry(&state, 1)?), expected[1]);
                eprintln!("PASS shared compressor storage {mode:?}: stale carry refused");
                return Ok(());
            }
            refused?;
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(if private_streams { 10 } else { 2 });
        let mut b_done = false;
        while std::time::Instant::now() < deadline {
            if unsafe { lib.cuda_stream_query(slots[1].stream().raw)? } { b_done = true; break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(b_done, private_streams, "unit B {} while unit A was held mid-production",
            if b_done { "completed" } else { "stayed pending" });
        drop(gate.take());
        for wave in &slots { wave.synchronize()?; }
        for wave in slots.iter_mut() { wave.commit(&mut state, &[rows as u32])?; }
        let got = (0..2).map(|seed| Ok((read(&state, seed, cached, &lib)?, carry(&state, seed)?))).collect::<Result<Vec<_>>>()?;
        assert_eq!(got[1], expected[1], "unit B differs from its private run");
        match mode {
            Shared::Ordered => assert_eq!(got[0], expected[0], "unit A differs from its private run on shared compressor storage"),
            Shared::PrivateStreams => assert_ne!(got[0], expected[0],
                "without the shared stream unit B should clobber A's scratch: the test cannot detect a race"),
            _ => {
                assert_eq!(got[0].0, expected[0].0, "unit A's planes are its own rows");
                assert_ne!(got[0].1, expected[0].1, "an unchecked late commit should copy B's carry: the check is untested");
            }
        }
        eprintln!("PASS shared compressor storage {mode:?}: unit B {} while A was held",
            if b_done { "completed" } else { "stayed pending" });
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn shared_compressor_storage_units_match_private_runs_behind_a_gated_stream() -> Result<()> {
        shared_compressor_storage(Shared::Ordered)
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn private_streams_on_shared_compressor_storage_clobber_scratch() -> Result<()> {
        shared_compressor_storage(Shared::PrivateStreams)
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn compressor_commit_refuses_a_carry_overwritten_by_a_later_production() -> Result<()> {
        shared_compressor_storage(Shared::LateCommit)
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn unchecked_late_compressor_commit_copies_a_clobbered_carry() -> Result<()> {
        shared_compressor_storage(Shared::LateCommitUnchecked)
    }
    fn source_commits(replicated: bool) -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let model = std::env::var("CUTEAFD_CACHE_COMMIT_MODEL")?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model))?;
        for layer in [2, 20] {
            let weights = CompressorWeights::load(&lib, &catalog, layer,
                CompressorWeights::device_bytes(&catalog, layer)?, 1024*1024)?;
            let mut state = CompressorState::new(&lib, layer, 2, 4, usize::MAX)?;
            let mut reference = CompressorState::new(&lib, layer, 2, 4, usize::MAX)?;
            let replica=if replicated {
                Some(state.enable_replica(
                    crate::shared::memory::device::Device { library:&lib,id:1-lib.cuda_get_device()? })?)
            } else { None };
            let leases = [state.begin_request(0, 11)?, state.begin_request(1, 22)?];
            let originals = [reference.begin_request(0, 11)?, reference.begin_request(1, 22)?];
            let mut waves = [weights.wave(16, usize::MAX)?, weights.wave(16, usize::MAX)?];
            if let Some(replica)=&replica {
                for wave in &mut waves { wave.enable_replica(&state,replica.clone())?; }
            }
            let mut direct = weights.wave(16, usize::MAX)?;
            let chunk = |lease, position| CompressorChunk { lease, position, tokens: 5 };
            for i in 0..2 {
                let input: Vec<u8> = (0..5*5120).flat_map(|j| {
                    let value = ((j % 19) as f32 * (i+1) as f32 / 19.0).to_bits();
                    ((value >> 16) as u16).to_ne_bytes()
                }).collect();
                lib.copy_h2d(waves[i].input(), &input)?; lib.copy_h2d(direct.input(), &input)?;
                unsafe {
                    waves[i].execute(&state, &[chunk(leases[i], 0)])?;
                    direct.execute(&reference, &[chunk(originals[i], 0)])?;
                }
                direct.commit(&mut reference, &[i as u32+3])?;
            }
            unsafe {
                waves[0].enqueue_commit(&state, &[3])?;
                waves[1].enqueue_commit(&state, &[4])?;
            }
            assert!(state.release(leases[0]).is_err());
            assert!(state.release(leases[1]).is_err());
            assert_eq!(state.committed_end(leases[0])?, 0);
            assert!(waves[0].clear_graph().is_err());
            waves[1].synchronize()?;
            waves[1].commit(&mut state, &[4])?;
            assert_eq!(state.committed_end(leases[1])?, 4);
            assert_eq!(state.committed_end(leases[0])?, 0);
            waves[0].synchronize()?;
            assert!(waves[0].commit(&mut state, &[2]).is_err());
            waves[0].commit(&mut state, &[3])?;
            for i in 0..2 {
                assert_eq!(read(&state, i, (i+3)/ratio(layer)?, &lib)?,
                    read(&reference, i, (i+3)/ratio(layer)?, &lib)?);
                if let Some(replica)=&replica {
                    let original=state.kv_cache(leases[i])?;
                    let peer=unsafe { replica.view(&state.index,i,original.rows)? };
                    for row in 0..original.rows {
                        let physical=original.pages[row/256] as usize*256+row%256;
                        for (a,b,width) in [(original.values,peer.values,V41Kv::COMPRESSED_VALUE_BYTES),
                            (original.scales,peer.scales,V41Kv::COMPRESSED_SCALE_BYTES)] {
                            let slice=|buffer:CuteafdDeviceBuffer| CuteafdDeviceBuffer {
                                ptr:unsafe { buffer.ptr.cast::<u8>().add(physical*width).cast() },bytes:width,..buffer };
                            let mut left=vec![0;width];let mut right=left.clone();
                            lib.copy_d2h_quiescent(&mut left,slice(a))?;
                            replica.device().run(||lib.copy_d2h_quiescent(&mut right,slice(b)))?;
                            assert_eq!(left,right);
                        }
                    }
                    let mut rows=[0;8];
                    replica.device().run(||lib.copy_d2h_quiescent(&mut rows,peer.device_rows))?;
                    assert_eq!(u64::from_ne_bytes(rows),original.rows as u64);
                    let mut pages=vec![0;original.pages.len()*4];
                    let page_bytes=pages.len();
                    replica.device().run(||lib.copy_d2h_quiescent(&mut pages,
                        CuteafdDeviceBuffer { bytes:page_bytes,..peer.device_pages }))?;
                    assert_eq!(pages,original.pages.iter().flat_map(|p|p.to_ne_bytes()).collect::<Vec<_>>());
                }
            }
            if let (Some(actual), Some(expected)) = (&state.pending, &reference.pending) {
                for (a, b) in actual.iter().zip(expected) {
                    let mut left = vec![0; 2048]; let mut right = left.clone();
                    lib.copy_d2h_quiescent(&mut left, CuteafdDeviceBuffer { bytes: 2048, ..a.buffer })?;
                    lib.copy_d2h_quiescent(&mut right, CuteafdDeviceBuffer { bytes: 2048, ..b.buffer })?;
                    assert_eq!(left, right);
                }
            }
            unsafe {
                waves[0].execute(&state, &[chunk(leases[0], 3)])?;
                waves[0].enqueue_commit(&state, &[2])?;
            }
            let prefix = state.committed_proposal(leases[0], 0..3, reserve_source_snapshot()?)?;
            assert_eq!(prefix.cache.rows, 3 / ratio(layer)?);
            assert_eq!(prefix.metadata(2)?[1], 3 / ratio(layer)? as u64);
            assert!(state.committed_proposal(leases[0], 0..4, reserve_source_snapshot()?).is_err());
            if let Some(replica)=&replica {
                let peer=unsafe { prefix.peer_attention(&state,replica,prefix.kv_values,prefix.kv_scales)? };
                assert_eq!(peer.binding(),prefix.binding());
                assert_eq!(peer.metadata(2)?,prefix.metadata(2)?);
                assert_eq!(peer.kv_cache.rows,prefix.kv_cache.rows);
                assert_eq!(peer.kv_values.device_id,replica.device().id);
                assert_eq!(peer.kv_scales.device_id,replica.device().id);
                assert!(peer.metadata(3).is_err());
                // Ordinary mutable/current-proposal views still reject the writer.
                assert!(unsafe { replica.view(&state.index,0,prefix.kv_cache.rows) }.is_err());
            }
            drop(prefix);
            assert!(state.index_cache(leases[0]).is_err());
            assert!(state.release(leases[0]).is_err());
            waves[0].abort_commit(&mut state)?;
            assert!(state.request_id(leases[0]).is_err());
            assert_eq!(state.committed_end(leases[1])?, 4);
            assert_eq!(read(&state, 1, 4/ratio(layer)?, &lib)?, read(&reference, 1, 4/ratio(layer)?, &lib)?);
        }
        Ok(())
    }
}
