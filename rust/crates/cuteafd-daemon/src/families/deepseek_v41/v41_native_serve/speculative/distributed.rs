//! GPU1-owned request/cache lifecycle with independent distributed draft chains.
use super::*;
use crate::families::deepseek_v41::v41_experts::dspark::DistributedDsparkChain;
use crate::shared::memory::device::{Device, DeviceOwner};
use crate::families::deepseek_v41::v41_tensors::VocabularyShard;

impl<'w, 'a> DraftRuntime<'w, 'a, DistributedDsparkChain<'w, 'a>> {
    pub(crate) fn capture_startup(&mut self) -> Result<()> {
        ensure!(self.pending.iter().all(Option::is_none),
            "draft startup requires no pending proposals");
        match &mut self.backend {
            DraftBackend::Dspark { chains, windows, requests, .. } => {
                ensure!(requests.is_empty(), "draft startup requires no admitted requests");
                let windows = windows.each_ref();
                for chain in chains { chain.capture_startup(windows)?; }
                Ok(())
            }
            DraftBackend::Dfly2 { .. } => Ok(()),
        }
    }
    /// The returned owner scopes destruction to GPU1. Synchronous runtime methods
    /// select their execution device internally and restore it before returning;
    /// no device scope survives a scheduler yield.
    pub fn with_distributed_requests(devices: [Device<'a>; 2], weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>, shards: [&'w VocabularyShard<'a>; 2],
        capacity: u32, decode_rows: u32, requests: u32, lanes: u8) -> Result<DeviceOwner<'a, Self>> {
        let (lane_count, lane_requests) = crate::families::deepseek_v41::v41_experts::dspark::draft_lanes(requests, lanes)?;
        let lane_count = lane_count as usize;
        devices[1].own(|| {
            let lib = devices[1].library;
            let lane_bytes = DistributedDsparkChain::device_bytes(weights, lane_requests, shards[0].tokens().end)?;
            let mut mains = vec![weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?];
            if lane_count > 1 { mains.push(weights.main_context(decode_rows, DsparkMainContext::device_bytes(lib, decode_rows)?)?); }
            let chains = (0..lane_count).map(|_| DistributedDsparkChain::new(devices, weights, table,
                shards, lane_requests, lane_bytes)).collect::<Result<Vec<_>>>()?;
            let window = || DsparkWindow::new(lib, requests as usize, capacity,
                DsparkWindow::device_bytes(requests as usize, capacity)?);
            tracing::info!(lanes=lane_count, lane_requests, draft_width=weights.draft_width(), gpu0_bytes_per_lane=lane_bytes[0],
                gpu1_bytes_per_lane=lane_bytes[1], "distributed lane-local dSpark draft workspaces");
            Ok(Self {
                backend: DraftBackend::Dspark {
                    mains,
                    chains,
                    windows: [window()?, window()?, window()?],
                    requests: Default::default(),
                },
                pending_commit_ids: vec![Vec::new(); lane_count], pending_prefix_ids: [None, None],
                pending: vec![None; lane_count],
                request_limit: requests as usize, verify_rows: decode_rows as usize, draft_limit: weights.draft_width(), draft_width: weights.draft_width(),
                confidence_trace: Default::default(), neural_lengths: Default::default(), length: Default::default(), policy: None,
                lanes: vec![LaneDraft::default(); lane_count],
            })
        })
    }

    pub fn with_dfly2_distributed(
        devices: [Device<'a>; 2],
        weights: &'w crate::families::deepseek_v41::v41_experts::dfly2::Dfly2Weights<'a>,
        table: &'w NativeRtxTensors<'a>,
        shards: [&'w VocabularyShard<'a>; 2],
        decode_rows: u32,
        requests: u32,
        lanes: u8,
    ) -> Result<DeviceOwner<'a, Self>> {
        let (lane_count, lane_requests) = crate::families::deepseek_v41::v41_experts::dspark::draft_lanes(requests, lanes)?;
        let lane_count = lane_count as usize;
        devices[1].own(|| {
            let chains = (0..lane_count)
                .map(|_| crate::families::deepseek_v41::v41_experts::dfly2::Dfly2Chain::new(
                    weights,
                    table,
                    crate::families::deepseek_v41::v41_experts::dfly2::Dfly2Vocab::Split(shards),
                    lane_requests as usize,
                ))
                .collect::<Result<Vec<_>>>()?;
            let taps = crate::families::deepseek_v41::v41_experts::dfly2::Dfly2TapStore::new(devices[1], requests as usize)?;
            let commit_streams = (0..lane_count).map(|_| Stream::new(devices[1])).collect::<Result<Vec<_>>>()?;
            let commit_done = (0..lane_count).map(|_| Event::new(devices[1])).collect::<Result<Vec<_>>>()?;
            tracing::info!(
                lanes = lane_count,
                lane_requests,
                block_size = weights.block_size(),
                draft_width = weights.draft_width(),
                tp2_experts = weights.is_tp2(),
                "distributed lane-local V4.1 DFly2 draft workspaces"
            );
            Ok(Self {
                backend: DraftBackend::Dfly2 {
                    chains,
                    taps,
                    commit_streams,
                    commit_done,
                    pending_updates: vec![Vec::new(); lane_count],
                    failure: crate::families::deepseek_v41::v41_experts::dspark::DraftFailure::default(),
                },
                pending_commit_ids: vec![Vec::new(); lane_count],
                pending_prefix_ids: [None, None],
                pending: vec![None; lane_count],
                request_limit: requests as usize,
                verify_rows: decode_rows as usize,
                draft_limit: weights.draft_width(),
                draft_width: weights.draft_width(),
                confidence_trace: Default::default(),
                neural_lengths: Default::default(),
                length: Default::default(),
                policy: None,
                lanes: vec![LaneDraft::default(); lane_count],
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::v41_dspark_cache::WindowChunk;
    fn append<'a, C: DraftChain<'a>>(runtime: &mut DraftRuntime<'_, 'a, C>, lib: &NativeLibrary, cycle: usize) -> Result<u64> {
        let DraftBackend::Dspark { windows, requests, .. } = &mut runtime.backend else {
            anyhow::bail!("expected dSpark backend");
        };
        let position = if cycle == 0 { 0 } else { 4 + cycle as u64 };
        let tokens = if cycle == 0 { 5 } else { 1 };
        for stage in 0..3 {
            let values: Vec<u8> = (0..80 * 512).flat_map(|i| {
                let v = ((i * 7 + stage * 11 + cycle * 13) % 63) as f32 / 64. - 0.5;
                ((v.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            lib.copy_h2d(windows[stage].source(), &values)?;
            let chunks: Vec<_> = requests.values().map(|r| WindowChunk {
                lease: r.leases[stage], position, source_row: r.slot as u32 * tokens, tokens,
            }).collect();
            unsafe { windows[stage].write(&chunks)?; }
        }
        Ok(position + u64::from(tokens))
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and two CUDA GPUs"]
    fn distributed_draft_runtime_matches_single_and_restores_prefixes() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let names = ["embed.weight".to_string()];
        let embedding = devices[0].own(|| NativeRtxTensors::load(&lib, &catalog, &names,
            NativeRtxTensors::plan(&catalog, &names)?, 16 << 20))?;
        let shards = [devices[0].own(|| VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?,
            devices[1].own(|| VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?];
        let full = devices[1].own(|| VocabularyHead::load(&lib, &catalog, 2 << 30, 16 << 20))?;
        let weights = devices[1].own(|| DsparkWeights::load_serving_with_width(&lib, &catalog,
            80, 16, 2, 32usize << 30, 16 << 20, 5))?;
        let mut actual = DraftRuntime::with_distributed_requests(devices, &weights, &embedding,
            [&shards[0], &shards[1]], 80, cuteafd_core::V41_VERIFY_ROWS as u32, 16, 2)?;
        let mut reference = devices[1].own(|| DraftRuntime::with_requests(&lib, &weights, &embedding, &full, 80, cuteafd_core::V41_VERIFY_ROWS as u32, 16, 2))?;
        devices[1].run(|| actual.get_mut().capture_startup())?;
        assert!(matches!(&actual.backend, DraftBackend::Dspark { requests, .. } if requests.is_empty()));
        assert!(actual.pending.iter().all(Option::is_none));
        devices[1].run(|| {
            actual.bind_policy(0); reference.bind_policy(0);
            for id in 1000..1016 { actual.admit(id)?; reference.admit(id)?; }
            Ok(())
        })?;
        for (cycle, count) in [1, 3, 8, 3].into_iter().enumerate() {
            let end = devices[1].run(|| {
                let end = append(actual.get_mut(), &lib, cycle)?;
                ensure!(end == append(reference.get_mut(), &lib, cycle)?, "fixture cache frontier differs");
                Ok(end)
            })?;
            let inputs: [Vec<_>; 2] = std::array::from_fn(|lane| (0..count).map(|row| {
                let id = 1000 + (lane * 8 + row) as u64;
                (id, (42 + lane * 101 + row * 31 + cycle * 17) as u32, end, 12)
            }).collect());
            let mut expected = [None, None];
            let mut observed = [None, None];
            // Admit both lanes, then finish lane 1 while lane 0 is left unpolled.
            for lane in 0..2 {
                observed[lane] = actual.poll_propose(lane, &inputs[lane])?;
                assert_eq!(lib.cuda_get_device()?, 0);
                expected[lane] = devices[1].run(|| reference.poll_propose(lane, &inputs[lane]))?;
            }
            for lane in [1, 0] {
                while observed[lane].is_none() || expected[lane].is_none() {
                    if observed[lane].is_none() { observed[lane] = actual.poll_propose(lane, &inputs[lane])?; }
                    assert_eq!(lib.cuda_get_device()?, 0);
                    if expected[lane].is_none() {
                        expected[lane] = devices[1].run(|| reference.poll_propose(lane, &inputs[lane]))?;
                    }
                    std::thread::yield_now();
                }
                ensure!(observed[lane].as_ref().unwrap().0 == expected[lane].as_ref().unwrap().0, "runtime proposal differs");
                for &(id, _, _, _) in &inputs[lane] {
                    ensure!(actual.confidence_trace(id) == reference.confidence_trace(id), "runtime confidence differs");
                }
                if lane == 1 { assert!(actual.pending[0].is_some()); }
            }
            eprintln!("PASS distributed runtime requests_per_lane={count}: identical proposals/confidence and independent polling");
        }
        let saved = devices[1].run(|| crate::shared::memory::DeviceAllocation::new(&lib,
            crate::families::deepseek_v41::v41_native_serve::speculative::DRAFT_MARK_BYTES, crate::shared::memory::TEST))?;
        actual.queue_mark(0, 1000, 8, saved.buffer)?;
        while !actual.mark_ready(0)? { std::thread::yield_now(); }
        actual.release(1000)?;
        actual.admit(2000)?;
        actual.restore_mark(2000, 8, saved.buffer)?;
        actual.validate_position(2000, 8)?;
        for id in 1001..1016 { actual.release(id)?; }
        actual.release(2000)?;
        assert_eq!(lib.cuda_get_device()?, 0);
        drop(actual); // Retained snapshot storage outlives the producing runtime.
        assert_eq!(lib.cuda_get_device()?, 0);
        drop(saved);
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("PASS distributed runtime queued prefix, release/re-admission, restore and device restoration");
        Ok(())
    }
}
