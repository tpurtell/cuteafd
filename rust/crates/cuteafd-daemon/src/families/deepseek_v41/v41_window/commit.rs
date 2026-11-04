//! Queued accepted SWA writes retain their lane's producer and upload storage.
use super::*;

pub(super) fn staging_bytes(rows: usize) -> usize {
    (rows + cuteafd_core::V41_MAX_REQUESTS) * 8
}

fn stage_commit(staging: &mut [u8], destinations: &[u64], ends: &[u64]) -> Result<()> {
    ensure!(!ends.is_empty() && ends.len() <= cuteafd_core::V41_MAX_REQUESTS
        && ends.len() <= destinations.len(), "invalid window commit request count");
    let bytes = (destinations.len() + ends.len()) * 8;
    ensure!(staging.len() >= bytes, "window commit staging too small");
    for (dst, value) in staging[..bytes].chunks_exact_mut(8).zip(destinations.iter().chain(ends)) {
        dst.copy_from_slice(&value.to_ne_bytes());
    }
    Ok(())
}

pub(super) struct WriteReservation {
    flags: std::rc::Rc<std::cell::Cell<u64>>,
    mask: u64,
}
impl Drop for WriteReservation {
    fn drop(&mut self) { self.flags.set(self.flags.get() & !self.mask); }
}
pub(super) struct PendingCommit {
    prepared: Prepared,
    ends: Vec<u64>,
    accepted: Vec<u32>,
    /// A batched multi-layer store runs here instead of the storage's commit stream.
    external: Option<*mut c_void>,
    _reservation: WriteReservation,
}
/// One layer's part of a batched window commit (see `stage_batched_commit`).
pub(crate) struct BatchedWindowCommit {
    pub destinations: Vec<u64>,
    pub ends: Vec<(u64, u64)>,
    pub layer: cuteafd_ffi::V41KvStoreLayer,
    pub device: i32,
}
impl<'a> WindowWave<'_, 'a> {
    /// Configure before production. Each wave owns independent publication
    /// events, while waves using disjoint request slots share replica storage.
    pub fn enable_replica(&mut self, state: &WindowState<'a>,
        replica: std::rc::Rc<super::replica::WindowReplica<'a>>) -> Result<()> {
        ensure!(self.ready.is_none() && self.pending_query.is_none() && self.pending_commit.is_none()
            && self.replica.is_none(), "window replica requires an unused wave");
        replica.validate_owner(state)?;
        ensure!(state.layer==self.weights.layer && state.values.buffer.device_id==self.values().device_id
            && std::ptr::eq(state.ends.library,self.stream().library),
            "window replica producer differs");
        let source=crate::shared::memory::device::Device { library:self.stream().library,id:self.values().device_id };
        let publication=crate::shared::memory::peer_publication::PeerPublication::new(source,replica.device())?;
        self.replica=Some((replica,publication)); Ok(())
    }
    /// # Safety
    /// All consumers of this proposal have drained. Keep state and wave alive
    /// through completion; on any failure drain before invalidating its requests.
    pub unsafe fn enqueue_commit(&mut self, state: &WindowState<'_>, accepted: &[u32]) -> Result<()> {
        unsafe { self.enqueue_commit_on(state, accepted, None).map(|_| ()) }
    }
    /// Host-side part of enqueue_commit for a batched multi-layer store: the
    /// caller launches one store for every layer on `stream`, and this wave's
    /// completion is then observed on that stream. Replicated windows keep the
    /// per-layer path (their publication is ordered on the commit stream).
    /// # Safety
    /// Same contract as enqueue_commit; the caller enqueues the store on `stream`
    /// before any poll/commit and drains `stream` before releasing this state.
    pub(crate) unsafe fn stage_batched_commit(&mut self, state: &WindowState<'_>, accepted: &[u32],
        stream: *mut c_void) -> Result<BatchedWindowCommit> {
        ensure!(self.replica.is_none() && state.replica.is_none(), "replicated windows commit per layer");
        unsafe { self.enqueue_commit_on(state, accepted, Some(stream)) }?
            .context("batched window commit staged no rows")
    }
    unsafe fn enqueue_commit_on(&mut self, state: &WindowState<'_>, accepted: &[u32],
        external: Option<*mut c_void>) -> Result<Option<BatchedWindowCommit>> {
        let (lib, raw) = (self.stream().library, self.storage.commit_stream.raw);
        let (values, scales, device_destinations) = (self.values(), self.scales(), self.destinations());
        ensure!(self.pending_commit.is_none(), "window commit already pending");
        if let Some(configured)=&state.replica {
            ensure!(self.replica.as_ref().is_some_and(|(storage,_)|
                std::rc::Rc::ptr_eq(storage,&configured.storage)),"window producer replica is not configured");
        }
        if let Some((replica,_))=&self.replica { replica.validate_owner(state)?; }
        self.validate_ready(state)?;
        let p = self.ready.take().context("window output unpublished")?;
        ensure!(accepted.len() == p.chunks.len(), "window acceptance count differs");
        let mut destinations = vec![u64::MAX; p.rows];
        let mut ends = vec![];
        for (i, c) in p.chunks.iter().enumerate() {
            ensure!(
                accepted[i] <= c.tokens,
                "window accepted prefix exceeds proposal"
            );
            state.slots[c.lease.slot]
                .version
                .checked_add(1)
                .context("window version exhausted")?;
            let n = accepted[i] as usize;
            for j in n.saturating_sub(128)..n {
                destinations[p.offsets[i] + j] =
                    (c.lease.slot * 128) as u64 + (c.position + j as u64) % 128;
            }
            ends.push(c.position + n as u64);
        }
        stage_commit(self.commit_host().bytes_mut(), &destinations, &ends)?;

        let mask = p.chunks.iter().fold(0u64, |mask, c| mask | (1 << c.lease.slot));
        ensure!(state.writing.get() & mask == 0, "window slot is being committed");
        state.writing.set(state.writing.get() | mask);
        let batched = external.map(|_| {
            let pending_ends = p.chunks.iter().zip(&ends).map(|(c, &end)| (c.lease.slot as u64, end)).collect();
            BatchedWindowCommit {
                destinations: destinations.clone(),
                ends: pending_ends,
                layer: cuteafd_ffi::V41KvStoreLayer {
                    values: values.ptr as u64,
                    scales: scales.ptr as u64,
                    cache: state.values.buffer.ptr as u64,
                    cache_scales: state.scales.buffer.ptr as u64,
                    ends: state.ends.buffer.ptr as u64,
                    capacity: (state.slot_count * 128) as u64,
                    // Filled in by the batch owner once its arena is laid out.
                    destinations: 0,
                    end_pairs: 0,
                },
                device: values.device_id,
            }
        });
        self.pending_commit = Some(PendingCommit { prepared: p, ends, accepted: accepted.to_vec(),
            external, _reservation: WriteReservation { flags: state.writing.clone(), mask } });
        if batched.is_some() { return Ok(batched); }
        // SAFETY: the wave owns both events and keeps its proposal planes alive;
        // the commit stream must not read them before their producer completes.
        unsafe {
            lib.cuda_event_record(self.produced.raw, self.stream().raw)?;
            lib.cuda_stream_wait_event(raw, self.produced.raw)?;
        }
        // The H2D source: the commit staging written above (raw view, so the
        // closure below can borrow the wave).
        let host = self.commit_host().buffer;
        let p = &self.pending_commit.as_ref().unwrap().prepared;
        (|| -> Result<()> {
            unsafe {
                lib.copy_host_buffer_h2d_async(
                    device_destinations, host, p.rows * 8, raw,
                )?;
                self.kv.store(
                    values,
                    scales,
                    device_destinations,
                    state.values.buffer,
                    state.scales.buffer,
                    p.rows,
                    state.slot_count * 128,
                    raw,
                )?;
                for (i, c) in p.chunks.iter().enumerate() {
                    let mut end = host;
                    end.ptr = host.ptr.cast::<u8>().wrapping_add(p.rows * 8 + i * 8).cast();
                    end.bytes = 8;
                    lib.copy_host_buffer_h2d_async(
                        slice(state.ends.buffer, c.lease.slot * 8, 8), end, 8, raw,
                    )?;
                }
                if let Some((replica,publication)) = self.replica.as_mut() {
                    let pending=self.pending_commit.as_ref().unwrap();
                    publication.enqueue(raw,|stream| {
                        for (chunk,&end) in pending.prepared.chunks.iter().zip(&pending.ends) {
                            replica.copy_commit(state,chunk.lease,end,stream)?;
                        }
                        Ok(())
                    })?;
                }
                lib.cuda_event_record(self.committed.raw, raw)?;
            }
            Ok(())
        })().map(|()| None)
    }
    pub(crate) fn has_pending_commit(&self) -> bool { self.pending_commit.is_some() }
    pub(crate) fn has_replica(&self) -> bool { self.replica.is_some() }
    pub fn poll_commit(&self) -> Result<bool> {
        let Some(pending) = &self.pending_commit else { return Ok(true) };
        // SAFETY: a pending commit retains the selected stream and its storage.
        unsafe { self.stream().library.cuda_stream_query(pending.external.unwrap_or(self.storage.commit_stream.raw)) }
    }
    pub(super) fn validate_pending_commit(&self, state: &WindowState<'_>) -> Result<&Prepared> {
        let pending = self.pending_commit.as_ref().context("window commit absent")?;
        let p = &pending.prepared;
        ensure!(p.owner == state.owner, "queued window owner differs");
        for (i, c) in p.chunks.iter().enumerate() {
            let slot = state.validate_identity(c.lease)?;
            ensure!(state.writing.get() & (1 << slot) != 0
                && state.slots[slot].version == p.versions[i] && state.slots[slot].end == c.position,
                "queued window binding changed");
        }
        Ok(p)
    }
    /// Complete either an already-queued decode write or a direct prefill/C1 write.
    pub fn commit(&mut self, state: &mut WindowState<'_>, accepted: &[u32]) -> Result<()> {
        let direct = self.pending_commit.is_none();
        if direct {
            // A direct (prompt publication) commit inside a device-ordered
            // chain follows every chained reader of the private wave (#815
            // units); outside a chain they already completed (no-op).
            // SAFETY: this wave's live stream on the chain's device.
            unsafe { crate::shared::memory::chain::join(self.stream().library, self.stream().raw)?; }
            let launched = unsafe { self.enqueue_commit(state, accepted) };
            // Device-ordered prefill units (#815 option (i)) record the chain
            // head instead of a host wait: every later reader of this window's
            // published rows joins it (the same unit's later stages, and the
            // next unit through its publication event). Its host staging is
            // this pass's own and is reused only after the unit's drained commit.
            let completed = if crate::shared::memory::chain::deferred() {
                // SAFETY: this storage owns the commit stream through the chain's drain.
                unsafe { crate::shared::memory::chain::finish(self.stream().library, self.storage.commit_stream.raw) }
            } else { self.synchronize() };
            if let Err(error) = launched.and(completed) {
                self.abort_commit(state)?;
                return Err(error);
            }
        }
        self.validate_pending_commit(state)?;
        ensure!(self.pending_commit.as_ref().unwrap().accepted == accepted, "queued window acceptance changed");
        ensure!(direct || self.poll_commit()?, "window commit incomplete");
        let pending = self.pending_commit.take().unwrap();
        // Device-ordered: the write is queued, not complete. Remember its
        // leases until a later commit (after the unit's drain) settles them.
        self.unsettled.clear();
        if direct && crate::shared::memory::chain::deferred() {
            self.unsettled.extend(pending.prepared.chunks.iter().map(|c| c.lease));
        }
        for (c, end) in pending.prepared.chunks.iter().zip(&pending.ends) {
            state.slots[c.lease.slot].end = *end;
            state.slots[c.lease.slot].version += 1;
        }
        Ok(())
    }
    /// Drain before returning write slots to the bank; revoke any touched request.
    pub fn abort_commit(&mut self, state: &mut WindowState<'_>) -> Result<()> {
        if self.pending_commit.is_none() {
            // A device-ordered direct commit already advanced the host ends:
            // drain its queued write and revoke the leases (hunt C #2).
            if self.unsettled.is_empty() { return Ok(()); }
            let drained = self.synchronize();
            let leases = std::mem::take(&mut self.unsettled);
            let live = leases.into_iter().filter(|&l| state.validate(l).is_ok()).collect::<Vec<_>>();
            return drained.and(state.invalidate(&live));
        }
        let mut drained = self.synchronize();
        if let Some(stream) = self.pending_commit.as_ref().unwrap().external {
            drained = drained.and(unsafe { self.stream().library.cuda_stream_synchronize(stream) });
        }
        let pending = self.pending_commit.take().unwrap();
        let leases = pending.prepared.chunks.iter().map(|c| c.lease).collect::<Vec<_>>();
        drop(pending);
        let revoked = state.invalidate(&leases);
        drained.and(revoked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn commit_staging_covers_c64_and_refuses_old_arena() -> Result<()> {
        for count in [1, 16, 33, 64] {
            for rows in [count, 2752] {
                let destinations: Vec<_> = (0..rows as u64).collect();
                let ends: Vec<_> = (0..count).map(|slot| 1000 + slot as u64).collect();
                let capacity = staging_bytes(rows);
                let mut bytes = vec![0xa5; capacity + 16];
                stage_commit(&mut bytes[..capacity], &destinations, &ends)?;
                let expected: Vec<_> = destinations.iter().chain(&ends).flat_map(|n| n.to_ne_bytes()).collect();
                assert_eq!(&bytes[..expected.len()], &expected);
                assert!(bytes[expected.len()..].iter().all(|&b| b == 0xa5));
                let mut short = vec![0xa5; expected.len() - 1];
                assert!(stage_commit(&mut short, &destinations, &ends).is_err());
                assert!(short.iter().all(|&b| b == 0xa5));
            }
        }
        assert_eq!(staging_bytes(2752), 22528);
        let mut old = vec![0xa5; 22144];
        assert!(stage_commit(&mut old, &vec![0; 2752], &vec![0; 17]).is_err());
        assert!(old.iter().all(|&b| b == 0xa5));
        assert!(stage_commit(&mut vec![0; staging_bytes(65)], &vec![0; 65], &vec![0; 65]).is_err());
        Ok(())
    }
    #[test]
    fn write_reservations_release_only_their_slots() {
        let flags = std::rc::Rc::new(std::cell::Cell::new(0b1010));
        let first = WriteReservation { flags: flags.clone(), mask: 0b0010 };
        let peer = WriteReservation { flags: flags.clone(), mask: 0b1000 };
        drop(first);
        assert_eq!(flags.get(), 0b1000);
        drop(peer);
        assert_eq!(flags.get(), 0);
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn native_queued_windows_match_direct_and_keep_peer_slots_independent() -> Result<()> {
        queued_windows(false)
    }
    #[test]
    #[ignore = "requires SM peer copy, CUTEAFD_CACHE_COMMIT_MODEL and two CUDA GPUs"]
    fn native_queued_windows_publish_replicas_before_commit_completion() -> Result<()> {
        queued_windows(true)
    }
    /// #815: a device-ordered commit stages its destinations on the host while
    /// the producer's queued position upload has not run yet. With the stream
    /// gated, both are queued before either executes; the result must still
    /// match the drained reference (the commit must not reuse the producer's
    /// pinned staging).
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn native_commit_staging_survives_a_queued_producer_upload() -> Result<()> { commit_behind_queued_upload(false) }
    /// Negative control: staging the commit through the producer's position
    /// staging (the pre-#815 layout) must clobber the queued upload.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn native_shared_commit_staging_clobbers_a_queued_producer_upload() -> Result<()> { commit_behind_queued_upload(true) }
    fn commit_behind_queued_upload(shared_staging: bool) -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let model = std::env::var("CUTEAFD_CACHE_COMMIT_MODEL")?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model))?;
        let weights = WindowWeights::load(&lib, &catalog, 0,
            WindowWeights::device_bytes(&lib, &catalog, 0)?, 1024*1024)?;
        let mut state = WindowState::new(&lib, 0, 1, usize::MAX)?;
        let mut reference = WindowState::new(&lib, 0, 1, usize::MAX)?;
        let lease = state.begin_request(0, 11)?;
        let original = reference.begin_request(0, 11)?;
        let mut gated = weights.wave(16, usize::MAX)?;
        let mut direct = weights.wave(16, usize::MAX)?;
        let rows = 13;
        let input: Vec<u8> = (0..rows*5120).flat_map(|i| {
            let value = ((i % 23) as f32 / 23.0).to_bits();
            ((value >> 16) as u16).to_ne_bytes()
        }).collect();
        // Positions start past the 128-row window so the commit's destinations
        // ((300 + j) % 128) differ from the producer's positions (300 + j):
        // at position 0 both are 0..rows, and a clobber would be invisible.
        let chunk = |lease| WindowChunk { lease, position: 300, tokens: rows as u32 };
        // Advance both windows to position 300 (ungated; also warms every kernel).
        for start in (0..300u64).step_by(15) {
            let tokens = (300 - start).min(15) as u32;
            let warm: Vec<u8> = (0..tokens as usize * 5120).flat_map(|i| {
                let value = ((i % 19) as f32 / 19.0 - 0.3).to_bits();
                ((value >> 16) as u16).to_ne_bytes()
            }).collect();
            lib.copy_h2d(direct.input(), &warm)?;
            for (window, held) in [(&mut reference, original), (&mut state, lease)] {
                unsafe { direct.execute(window, &[WindowChunk { lease: held, position: start, tokens }])?; }
                direct.commit(window, &[tokens])?;
            }
        }
        lib.copy_h2d(direct.input(), &input)?;
        unsafe { direct.execute(&reference, &[chunk(original)])?; }
        direct.commit(&mut reference, &[rows as u32])?;
        lib.copy_h2d(gated.input(), &input)?;
        {
            let gate = unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&lib, gated.stream().raw)? };
            let prepared = gated.prepare(&state, &[chunk(lease)])?;
            gated.upload(&prepared)?;
            unsafe { gated.enqueue(prepared.rows)?; }
            gated.ready = Some(prepared);
            // The commit stages on the host while the upload above is still queued.
            crate::families::deepseek_v41::v41_window::SHARED_COMMIT_STAGING.with(|shared| shared.set(shared_staging));
            let staged = unsafe { gated.enqueue_commit(&state, &[rows as u32]) };
            crate::families::deepseek_v41::v41_window::SHARED_COMMIT_STAGING.with(|shared| shared.set(false));
            staged?;
            drop(gate);
        }
        gated.synchronize()?;
        // The bug's direct observable: the positions the queued upload delivered.
        // (The packed window values do not change with these positions at this
        // layer, so they alone cannot detect the clobber.)
        let mut device = vec![0u8; rows * 8];
        lib.copy_d2h(&mut device, slice(gated.storage.positions.buffer, 0, rows * 8))?;
        let delivered: Vec<u64> = device.chunks_exact(8).map(|b| u64::from_ne_bytes(b.try_into().unwrap())).collect();
        let intended: Vec<u64> = (300..300 + rows as u64).collect();
        gated.commit(&mut state, &[rows as u32])?;
        let (actual, expected) = (state.view(lease)?, reference.view(original)?);
        let mut equal = true;
        for (a, b, width) in [(actual.values, expected.values, 512), (actual.scales, expected.scales, 16)] {
            let mut left = vec![0; rows * width];
            let mut right = left.clone();
            lib.copy_d2h(&mut left, slice(a, 0, rows * width))?;
            lib.copy_d2h(&mut right, slice(b, 0, rows * width))?;
            equal &= left == right;
        }
        if shared_staging {
            assert_ne!(delivered, intended,
                "sharing the producer's staging did not clobber the queued upload: the test cannot detect #815");
        } else {
            assert_eq!(delivered, intended, "queued producer positions were overwritten by commit staging");
            assert!(equal, "committed window differs from the direct run");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn decode_layers_share_stream_and_keep_graphs_and_passes_independent() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let model = std::env::var("CUTEAFD_CACHE_COMMIT_MODEL")?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model))?;
        let weights = [0, 1].map(|layer| WindowWeights::load(&lib, &catalog, layer,
            WindowWeights::device_bytes(&lib, &catalog, layer)?, 1024 * 1024))
            .into_iter().collect::<Result<Vec<_>>>()?;
        let rows = 4usize;
        let storage = WindowStorage::new(&lib, rows, rows, &[0, 1])?;
        let states = || (0..2).map(|layer| WindowState::new(&lib, layer, 1, usize::MAX))
            .collect::<Result<Vec<_>>>();
        let mut actual = states()?;
        let mut expected = states()?;
        let actual_leases = actual.iter_mut().map(|state| state.begin_request(0, 11))
            .collect::<Result<Vec<_>>>()?;
        let expected_leases = expected.iter_mut().map(|state| state.begin_request(0, 11))
            .collect::<Result<Vec<_>>>()?;
        // Waves drain before their state, including on gated assertion failures.
        let mut waves = weights.iter().map(|weights| weights.on_storage(storage.clone()))
            .collect::<Result<Vec<_>>>()?;
        let other_pass = weights[0].on_storage(WindowStorage::new(&lib, rows, rows, &[0, 1])?)?;
        assert!(std::ptr::eq(waves[0].stream(), waves[1].stream()));
        assert_ne!(waves[0].stream().raw, other_pass.stream().raw);
        assert_ne!(storage.stream.raw, storage.commit_stream.raw);
        assert_ne!(storage.commit_stream.raw, other_pass.storage.commit_stream.raw);
        let mut references = weights.iter().map(|weights| weights.wave(rows as u32, usize::MAX))
            .collect::<Result<Vec<_>>>()?;
        for round in 0..3 {
            for layer in 0..2 {
                let input: Vec<_> = (0..rows * 5120).flat_map(|i| {
                    let value = (((i * 7 + layer * 11 + round * 13) % 23) as f32 / 23.0 - 0.3).to_bits();
                    ((value >> 16) as u16).to_ne_bytes()
                }).collect();
                lib.copy_h2d(waves[layer].input(), &input)?;
                lib.copy_h2d(references[layer].input(), &input)?;
                let chunk = |lease| WindowChunk { lease, position: (round * rows) as u64, tokens: rows as u32 };
                // Capture once per layer on the shared stream; later rounds replay
                // those distinct graphs without replacing storage or bindings.
                if round == 0 {
                    unsafe { waves[layer].capture(&actual[layer], &[chunk(actual_leases[layer])])?; }
                }
                unsafe {
                    waves[layer].replay(&actual[layer], &[chunk(actual_leases[layer])])?;
                    references[layer].execute(&expected[layer], &[chunk(expected_leases[layer])])?;
                }
                waves[layer].commit(&mut actual[layer], &[rows as u32])?;
                references[layer].commit(&mut expected[layer], &[rows as u32])?;
                assert_eq!(waves[layer].graphs.len(), 1);
                let read = |state: &WindowState<'_>, lease| -> Result<Vec<u8>> {
                    let view = state.view(lease)?;
                    let count = (round + 1) * rows;
                    let mut bytes = vec![0; count * 528];
                    lib.copy_d2h(&mut bytes[..count * 512], slice(view.values, 0, count * 512))?;
                    lib.copy_d2h(&mut bytes[count * 512..], slice(view.scales, 0, count * 16))?;
                    Ok(bytes)
                };
                assert_eq!(read(&actual[layer], actual_leases[layer])?,
                    read(&expected[layer], expected_leases[layer])?, "layer {layer}, round {round}");
            }
        }
        assert_ne!(waves[0].current_graph().unwrap().0, waves[1].current_graph().unwrap().0);
        // Holding stores must not stall another layer's producer on the shared
        // compute stream. Both stores still wait for their own proposal writes.
        let inputs = (0..2).map(|layer| -> Result<_> {
            let input: Vec<_> = (0..rows * 5120).flat_map(|i| {
                let value = (((i * 7 + layer * 11 + 2 * 13) % 23) as f32 / 23.0 - 0.3).to_bits();
                ((value >> 16) as u16).to_ne_bytes()
            }).collect();
            let source = DeviceAllocation::new(&lib, input.len())?;
            lib.copy_h2d(source.buffer, &input)?;
            Ok(source)
        }).collect::<Result<Vec<_>>>()?;
        let gate = unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&lib, storage.commit_stream.raw)? };
        for layer in 0..2 {
            let chunk = WindowChunk { lease: actual_leases[layer], position: 12, tokens: rows as u32 };
            let prepared = waves[layer].prepare(&actual[layer], &[chunk])?;
            // SAFETY: these preloaded inputs outlive the gated production.
            unsafe { lib.copy_d2d_async(waves[layer].input(), inputs[layer].buffer,
                rows * 10240, storage.stream.raw)?; }
            waves[layer].upload(&prepared)?;
            unsafe { waves[layer].enqueue(prepared.rows)?; }
            waves[layer].ready = Some(prepared);
            unsafe { waves[layer].enqueue_commit(&actual[layer], &[rows as u32])?; }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        // SAFETY: the live producer must complete independently of gated stores.
        while !unsafe { lib.cuda_stream_query(storage.stream.raw)? } {
            ensure!(std::time::Instant::now() < deadline, "window producer waited for an unrelated layer's store");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!waves[0].poll_commit()?);
        assert!(!waves[1].poll_commit()?);
        drop(gate);
        for layer in 0..2 {
            waves[layer].synchronize()?;
            waves[layer].commit(&mut actual[layer], &[rows as u32])?;
            // Shared scratch still holds layer 1's input; rebuild each layer's
            // final-round input before running the independent eager reference.
            let input: Vec<_> = (0..rows * 5120).flat_map(|i| {
                let value = (((i * 7 + layer * 11 + 2 * 13) % 23) as f32 / 23.0 - 0.3).to_bits();
                ((value >> 16) as u16).to_ne_bytes()
            }).collect();
            lib.copy_h2d(references[layer].input(), &input)?;
            unsafe { references[layer].execute(&expected[layer], &[WindowChunk {
                lease: expected_leases[layer], position: 12, tokens: rows as u32,
            }])?; }
            references[layer].commit(&mut expected[layer], &[rows as u32])?;
            let read = |state: &WindowState<'_>, lease| -> Result<Vec<u8>> {
                let view = state.view(lease)?;
                let mut bytes = vec![0; 16 * 528];
                lib.copy_d2h(&mut bytes[..16 * 512], slice(view.values, 0, 16 * 512))?;
                lib.copy_d2h(&mut bytes[16 * 512..], slice(view.scales, 0, 16 * 16))?;
                Ok(bytes)
            };
            assert_eq!(read(&actual[layer], actual_leases[layer])?, read(&expected[layer], expected_leases[layer])?);
        }
        drop(waves.remove(0));
        waves[0].synchronize()?;
        assert_eq!(waves[0].graphs.len(), 1);
        Ok(())
    }
    /// row_window contract, window storage: two units (slots 0 and 1) on one
    /// shared storage, windows 0..13 and 16..29 of a 48-row reserve, produce
    /// and commit layer 0. The gate holds the shared stream after unit A wrote
    /// its scratch input and before A's kernels, so every host staging write
    /// (both units' positions and commit destinations) happens before any of
    /// that work runs. Unit B must stay queued behind it: the shared stream is
    /// what orders the plain scratch. Both units' ring rows must equal private
    /// runs. `ordered == false` is the negative control: each wave gets its
    /// own stream (`PRIVATE_STREAM`), B runs while A is held and overwrites
    /// the shared scratch input, so A's ring rows must differ.
    fn shared_window_storage(ordered: bool) -> Result<()> {
        use crate::shared::memory::row_window::RowWindow;
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let model = std::env::var("CUTEAFD_CACHE_COMMIT_MODEL")?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model))?;
        let weights = WindowWeights::load(&lib, &catalog, 0,
            WindowWeights::device_bytes(&lib, &catalog, 0)?, 1024*1024)?;
        let (unit, reserve, rows) = (16usize, 48usize, 13usize);
        let input = |seed: usize| -> Vec<u8> { (0..rows*5120).flat_map(|i| {
            let value = (((i * 7 + seed * 11) % 23) as f32 / 23.0 - 0.3).to_bits();
            ((value >> 16) as u16).to_ne_bytes()
        }).collect() };
        let chunk = |lease| WindowChunk { lease, position: 0, tokens: rows as u32 };
        let read = |state: &WindowState<'_>, lease| -> Result<Vec<u8>> {
            let view = state.view(lease)?;
            let mut bytes = vec![0; rows * 528];
            lib.copy_d2h(&mut bytes[..rows*512], slice(view.values, 0, rows*512))?;
            lib.copy_d2h(&mut bytes[rows*512..], slice(view.scales, 0, rows*16))?;
            Ok(bytes)
        };
        // Private runs: the expected rows, and an ungated warm-up of every
        // kernel and copy the gated run issues (lazy module loading would
        // otherwise synchronize behind the gate).
        let mut expected = Vec::new();
        for seed in 0..2 {
            let mut reference = WindowState::new(&lib, 0, 1, usize::MAX)?;
            let lease = reference.begin_request(0, 11)?;
            let mut private = weights.wave(unit as u32, usize::MAX)?;
            lib.copy_h2d(private.input(), &input(seed))?;
            unsafe { private.execute(&reference, &[chunk(lease)])?; }
            private.commit(&mut reference, &[rows as u32])?;
            expected.push(read(&reference, lease)?);
        }
        let sources = [input(0), input(1)].map(|host| -> Result<DeviceAllocation<'_>> {
            let buffer = DeviceAllocation::new(&lib, host.len())?;
            lib.copy_h2d(buffer.buffer, &host)?;
            Ok(buffer)
        }).into_iter().collect::<Result<Vec<_>>>()?;
        let storage = WindowStorage::new(&lib, reserve, unit, &[0])?;
        crate::families::deepseek_v41::v41_window::PRIVATE_STREAM.with(|private| private.set(!ordered));
        let built = (|| -> Result<Vec<_>> { Ok(vec![weights.on_storage(storage.clone())?, weights.on_storage(storage.clone())?]) })();
        crate::families::deepseek_v41::v41_window::PRIVATE_STREAM.with(|private| private.set(false));
        let mut slots = built?;
        assert_eq!(std::ptr::eq(slots[0].stream(), slots[1].stream()), ordered);
        slots[0].select_window(RowWindow::within(0..rows, reserve)?)?;
        slots[1].select_window(RowWindow::within(unit..unit + rows, reserve)?)?;
        let mut state = WindowState::new(&lib, 0, 2, usize::MAX)?;
        let leases = [state.begin_request(0, 11)?, state.begin_request(1, 22)?];
        let produce = |wave: &mut WindowWave<'_, '_>, seed: usize, gate: bool, state: &WindowState<'_>|
            -> Result<Option<cuteafd_ffi::test_support::CudaStreamGate<'_>>> {
            let prepared = wave.prepare(state, &[chunk(leases[seed])])?;
            let stream = wave.stream().raw;
            unsafe { lib.copy_d2d_async(wave.input(), sources[seed].buffer, rows * 10240, stream)?; }
            wave.upload(&prepared)?;
            let gate = gate.then(|| unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&lib, stream) }).transpose()?;
            unsafe { wave.enqueue(prepared.rows)?; }
            wave.ready = Some(prepared);
            unsafe { wave.enqueue_commit(state, &[rows as u32])?; }
            Ok(gate)
        };
        let gate = produce(&mut slots[0], 0, true, &state)?;
        produce(&mut slots[1], 1, false, &state)?;
        // B is queued: unordered (negative control) it completes while A is
        // held; ordered it stays pending behind A on the shared stream.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(if ordered { 2 } else { 10 });
        let mut b_done = false;
        while std::time::Instant::now() < deadline {
            if unsafe { lib.cuda_stream_query(slots[1].stream().raw)? } { b_done = true; break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(b_done, !ordered, "unit B {} while unit A was held mid-production",
            if b_done { "completed" } else { "stayed pending" });
        drop(gate);
        for (seed, wave) in slots.iter_mut().enumerate() {
            wave.synchronize()?;
            wave.commit(&mut state, &[rows as u32])?;
            let got = read(&state, leases[seed])?;
            if ordered || seed == 1 {
                assert_eq!(got, expected[seed], "unit {seed} differs from its private run on shared window storage");
            } else {
                assert_ne!(got, expected[seed], "without the shared stream unit B should clobber A's scratch: the test cannot detect a race");
            }
        }
        eprintln!("PASS shared window storage ordered={ordered}: unit B {} while A was held",
            if b_done { "completed" } else { "stayed pending" });
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn shared_window_storage_units_match_private_runs_behind_a_gated_stream() -> Result<()> {
        shared_window_storage(true)
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_CACHE_COMMIT_MODEL and CUDA"]
    fn private_streams_on_shared_window_storage_clobber_scratch() -> Result<()> {
        shared_window_storage(false)
    }
    fn queued_windows(replicated: bool) -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let model = std::env::var("CUTEAFD_CACHE_COMMIT_MODEL")?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model))?;
        let weights = WindowWeights::load(&lib, &catalog, 0,
            WindowWeights::device_bytes(&lib, &catalog, 0)?, 1024*1024)?;
        let mut state = WindowState::new(&lib, 0, 2, usize::MAX)?;
        let mut reference = WindowState::new(&lib, 0, 2, usize::MAX)?;
        let replica=if replicated {
            Some(state.enable_replica(
                crate::shared::memory::device::Device { library:&lib,id:1-lib.cuda_get_device()? })?)
        } else { None };
        let first = state.begin_request(0, 11)?;
        let second = state.begin_request(1, 22)?;
        let originals = [reference.begin_request(0, 11)?, reference.begin_request(1, 22)?];
        // Waves drop/drain before either state, including on assertion/error paths.
        let mut waves = [weights.wave(16, usize::MAX)?, weights.wave(16, usize::MAX)?];
        if let Some(replica)=&replica {
            for wave in &mut waves { wave.enable_replica(&state,replica.clone())?; }
        }
        let mut direct = weights.wave(16, usize::MAX)?;
        let chunk = |lease, position| WindowChunk { lease, position, tokens: 3 };
        let input: Vec<u8> = (0..3*5120).flat_map(|i| {
            let value = ((i % 19) as f32 / 19.0).to_bits();
            ((value >> 16) as u16).to_ne_bytes()
        }).collect();
        for (i, lease) in [first, second].into_iter().enumerate() {
            lib.copy_h2d(waves[i].input(), &input)?;
            lib.copy_h2d(direct.input(), &input)?;
            unsafe {
                waves[i].execute(&state, &[chunk(lease, 0)])?;
                direct.execute(&reference, &[chunk(originals[i], 0)])?;
            }
            direct.commit(&mut reference, &[i as u32+2])?;
        }
        unsafe { waves[0].enqueue_commit(&state, &[2])?; }
        assert_eq!(state.end(first)?, 0);
        assert!(state.view(first).is_err());
        assert!(state.release(first).is_err());
        state.view(second)?;
        unsafe { waves[1].enqueue_commit(&state, &[3])?; }
        assert!(state.view(second).is_err());
        waves[0].synchronize()?;
        assert!(waves[0].commit(&mut state, &[3]).is_err());
        waves[0].commit(&mut state, &[2])?;
        assert_eq!(state.end(first)?, 2);
        assert_eq!(state.end(second)?, 0);
        assert!(state.view(second).is_err());
        waves[1].synchronize()?;
        waves[1].commit(&mut state, &[3])?;
        for (i, lease) in [first, second].into_iter().enumerate() {
            let actual = state.view(lease)?;
            let expected = reference.view(originals[i])?;
            assert_eq!(actual.end, expected.end);
            for (a, b, width) in [(actual.values, expected.values, 512), (actual.scales, expected.scales, 16)] {
                let mut left = vec![0; (i+2)*width];
                let mut right = left.clone();
                lib.copy_d2h(&mut left, slice(a, 0, (i+2)*width))?;
                lib.copy_d2h(&mut right, slice(b, 0, (i+2)*width))?;
                assert_eq!(left, right);
            }
            if let Some(replica)=&replica {
                let peer=unsafe { replica.view(&state,lease)? };
                for (a,b,width) in [(actual.values,peer.values,512),(actual.scales,peer.scales,16)] {
                    let mut left=vec![0;(i+2)*width]; let mut right=left.clone();
                    lib.copy_d2h(&mut left,slice(a,0,(i+2)*width))?;
                    replica.device().run(||lib.copy_d2h(&mut right,slice(b,0,(i+2)*width)))?;
                    assert_eq!(left,right);
                }
                let mut end=[0;8];
                replica.device().run(||lib.copy_d2h(&mut end,peer.device_end))?;
                assert_eq!(u64::from_ne_bytes(end),actual.end);
            }
        }
        unsafe {
            waves[0].execute(&state, &[chunk(first, 2)])?;
            waves[0].enqueue_commit(&state, &[2])?;
        }
        waves[0].abort_commit(&mut state)?;
        assert!(state.request_id(first).is_err());
        assert_eq!(state.end(second)?, 3);
        let device_end = state.view(second)?.device_end;
        state.release(second)?;
        assert!(state.view(second).is_err());
        let mut end_bytes = [0; 8];
        lib.copy_d2h(&mut end_bytes, device_end)?;
        assert_eq!(u64::from_ne_bytes(end_bytes), 3); // Release needs no GPU clear.
        let replacement = state.begin_request(1, 33)?;
        lib.copy_d2h(&mut end_bytes, state.view(replacement)?.device_end)?;
        assert_eq!(u64::from_ne_bytes(end_bytes), 0);
        assert_eq!(state.end(replacement)?, 0);
        Ok(())
    }
}
