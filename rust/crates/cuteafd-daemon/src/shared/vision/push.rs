//! Explicit startup contribution to the family's existing coordinator push group.
use super::{pcm_audio::PcmTowerSpec, ReadTransform, TowerSpec};
use crate::shared::experts::rtx::{dense::{Job, Operand, Records, Rect, Step}, push::PushStartupError};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::{sync::{atomic::{AtomicBool, Ordering}, mpsc, Arc, Mutex}, thread::JoinHandle};
use crate::shared::gpu_turn::GpuTurn;

pub(crate) struct OwnerThread {
    thread: Mutex<Option<JoinHandle<()>>>,
    cancel: mpsc::Sender<bool>,
    media: Option<(Arc<AtomicBool>, Option<Arc<GpuTurn>>)>,
}
impl OwnerThread {
    #[cfg(test)]
    pub(super) fn mock(thread: JoinHandle<()>) -> Arc<Self> {
        Arc::new(Self::new(thread, mpsc::channel().0))
    }
    pub(super) fn new(thread: JoinHandle<()>, cancel: mpsc::Sender<bool>) -> Self {
        Self { thread: Mutex::new(Some(thread)), cancel, media: None }
    }
    pub(super) fn coordinator(thread: JoinHandle<()>, cancel: mpsc::Sender<bool>, healthy: Arc<AtomicBool>,
        turn: Option<Arc<GpuTurn>>) -> Self {
        Self { thread: Mutex::new(Some(thread)), cancel, media: Some((healthy, turn)) }
    }
    pub(crate) fn is_finished(&self) -> bool {
        self.thread.lock().unwrap().as_ref().is_none_or(JoinHandle::is_finished)
    }
}
impl Drop for OwnerThread {
    fn drop(&mut self) {
        let _turn = self.media.as_ref().and_then(|(healthy, turn)| {
            healthy.store(false, Ordering::Release);
            turn.as_deref().map(GpuTurn::joining)
        });
        let _ = self.cancel.send(false);
        if let Some(thread) = self.thread.get_mut().unwrap().take() { let _ = thread.join(); }
    }
}

/// Address-only operand descriptors of [`Records`], movable between threads.
#[allow(clippy::type_complexity)]
pub(crate) struct SentRecords(Vec<(i32, usize, usize, u64, usize, Vec<(Rect, usize)>)>);

struct Completion {
    commit: mpsc::Sender<bool>,
    acknowledgement: mpsc::Receiver<super::Result<()>>,
    _owner: Arc<OwnerThread>,
}

/// Owns a reference to each pending media thread, including on startup errors.
/// Records are passed explicitly; no thread-local registry or separate group.
#[derive(Default)]
pub(crate) struct Contribution {
    records: Records,
    completions: Vec<Completion>,
}
impl Contribution {
    pub(super) fn prepare(device: i32, vision: Option<(&TowerSpec, usize)>,
        pcm: Option<(&PcmTowerSpec, usize)>, mimo: Option<(&super::audio::AudioTowerSpec, usize)>,
        completion: Option<(mpsc::Sender<bool>, mpsc::Receiver<super::Result<()>>, Arc<OwnerThread>)>) -> Result<Self> {
        let mut out = Self::default();
        if let Some((commit, acknowledgement, owner)) = completion {
            out.completions.push(Completion { commit, acknowledgement, _owner: owner });
        }
        let mut reserved = Vec::new();
        if let Some((spec, address)) = vision {
            spec.validate_constants()?;
            for region in spec.constant_regions() {
                let start = usize::try_from(region.offset)?;
                let end = start.checked_add(region.bytes.len()).context("vision constants overflow")?;
                ensure!(end <= spec.native.weight_bytes as usize, "vision constants outside admitted arena");
                reserved.push((device, address.checked_add(start).context("vision constants address overflow")?,
                    address.checked_add(end).context("vision constants address overflow")?));
            }
            for read in &spec.reads {
                append(&mut out.records, device, address, spec.native.weight_bytes as usize,
                    &read.path, &read.metadata, read.destination, read.transform)?;
            }
        }
        if let Some((spec, address)) = pcm {
            for read in spec.push_reads() {
                append(&mut out.records, device, address, spec.native_spec().weight_bytes as usize,
                    &read.path, &read.metadata, usize::try_from(read.destination)?, ReadTransform::Direct)?;
            }
        }
        if let Some((spec, address)) = mimo {
            for read in spec.plan().reads().values() {
                append(&mut out.records, device, address, spec.native().weight_bytes as usize,
                    &read.path, &read.metadata, usize::try_from(read.destination)?,
                    if read.metadata.dtype == cuteafd_core::DType::Bf16 {ReadTransform::Bf16ToF32}else{ReadTransform::Direct})?;
            }
        }
        check_destinations(&out.records, reserved)?;
        Ok(out)
    }
    /// [`Records`] as plain addresses: a family media thread's recorded owners
    /// cross to the startup thread (one process, one primary context per device).
    pub(crate) fn send_records(records: Records) -> Result<SentRecords> {
        ensure!(records.steps.is_empty() && records.finishes.is_empty() && records.peer_copies.is_empty(),
            "sent media records carry only direct operands");
        Ok(SentRecords(records.operands.into_iter().map(|operand| (operand.device, operand.buffer.ptr as usize,
            operand.buffer.bytes, operand.buffer.flags, operand.bytes, operand.lands)).collect()))
    }
    /// A family-owned media thread's recorded owners (its own mount-mode loader
    /// under the shared recorder, payload unread) with the same commit protocol:
    /// `commit` releases the owner only after the family group has committed.
    pub(crate) fn recorded(records: SentRecords, commit: mpsc::Sender<bool>,
        acknowledgement: mpsc::Receiver<super::Result<()>>, thread: JoinHandle<()>) -> Result<(Self, Arc<OwnerThread>)> {
        let records = Records { operands: records.0.into_iter().map(|(device, ptr, buffer_bytes, flags, bytes, lands)|
            Operand { device, buffer: CuteafdDeviceBuffer { ptr: ptr as *mut _, bytes: buffer_bytes, device_id: device, flags },
                lands, bytes }).collect(), ..Default::default() };
        check_destinations(&records, Vec::new())?;
        let owner = Arc::new(OwnerThread::new(thread, commit.clone()));
        Ok((Self { records, completions: vec![Completion { commit, acknowledgement, _owner: owner.clone() }] }, owner))
    }
    pub(crate) fn extend(&mut self, mut other: Self) {
        self.records.extend(std::mem::take(&mut other.records));
        self.completions.append(&mut other.completions);
    }
    pub(crate) fn records(&self) -> Records { self.records.clone() }
    /// Invoke only after the family's existing receive has committed/drained.
    pub(crate) fn commit(mut self) -> Result<()> {
        while !self.completions.is_empty() {
            let completion = self.completions.remove(0);
            completion.commit.send(true).context("media owner disappeared before push commit")
                .map_err(PushStartupError::Delivery)?;
            // Timeout fails startup, not the native commit: the service retains its owner,
            // whose drop joins the thread before releasing its arenas/library.
            completion.acknowledgement.recv_timeout(std::time::Duration::from_secs(120)).context("media owner commit acknowledgement")
                .and_then(|result| result.map_err(Into::into)).map_err(PushStartupError::Delivery)?;
        }
        Ok(())
    }
}
impl Drop for Contribution {
    fn drop(&mut self) {
        for completion in &self.completions { let _ = completion.commit.send(false); }
    }
}

/// Include native-initialized regions so push records cannot overwrite them or each other.
fn check_destinations(records: &Records, mut ranges: Vec<(i32, usize, usize)>) -> Result<()> {
    let mut add = |buffer: &CuteafdDeviceBuffer| -> Result<()> {
        let start = buffer.ptr as usize;
        let end = start.checked_add(buffer.bytes).context("media destination overflow")?;
        ranges.push((buffer.device_id, start, end));
        Ok(())
    };
    for operand in &records.operands { add(&operand.buffer)?; }
    for step in &records.steps {
        for job in &step.jobs {
            match job {
                Job::Widen { dst, .. } | Job::NarrowF32Bf16 { dst, .. } => add(dst)?,
                _ => anyhow::bail!("unexpected media transform"),
            }
        }
    }
    ranges.sort_unstable();
    ensure!(ranges.windows(2).all(|pair| pair[0].0 != pair[1].0 || pair[0].2 <= pair[1].1),
        "overlapping media destinations or native constants");
    Ok(())
}

/// One BF16 media vector's widen windows (one staged step each), shared by the
/// recorded contribution and the header-only host admission of its staging.
pub(crate) fn transform_windows(path: &std::path::Path, metadata: &cuteafd_loader::SafetensorsTensorMetadata,
    transform:ReadTransform) -> Vec<Rect> {
    let bytes = metadata.byte_length as usize;
    let window = crate::shared::experts::rtx::dense::TRANSFORM_BYTES & !(transform.source_alignment()-1);
    (0..bytes).step_by(window).map(|at| {
        let count = (bytes - at).min(window);
        Rect { name: metadata.name.clone(), path: path.into(), base: metadata.byte_offset + at as u64,
            row_bytes: count, rows: 0..1, cols: 0..count }
    }).collect()
}

fn append(records: &mut Records, device: i32, address: usize, arena_bytes: usize,
    path: &std::path::Path, metadata: &cuteafd_loader::SafetensorsTensorMetadata,
    destination: usize, transform:ReadTransform) -> Result<()> {
    let bytes = usize::try_from(metadata.byte_length)?;
    let resident = transform.resident_bytes(metadata)?;
    ensure!(bytes > 0 && address != 0 && address.checked_add(arena_bytes).is_some()
        && metadata.byte_offset.checked_add(metadata.byte_length).is_some()
        && destination.checked_add(resident).is_some_and(|end| end <= arena_bytes),
        "media tensor outside admitted arena: {}", metadata.name);
    let buffer = |offset, len| CuteafdDeviceBuffer {
        ptr: (address as *mut u8).wrapping_add(offset).cast(), bytes: len, device_id: device, flags: 0,
    };
    if transform!=ReadTransform::Direct {
        let mut at=0;
        for rect in transform_windows(path,metadata,transform) {
            let count=rect.bytes();
            let (offset,resident)=match transform {
                ReadTransform::Bf16ToF32=>(at*2,count*2),
                ReadTransform::F32ToBf16=>(at/2,count/2),
                ReadTransform::Direct=>unreachable!(),
            };
            let dst=buffer(destination+offset,resident);
            let job=match transform {
                ReadTransform::Bf16ToF32=>Job::Widen {at:0,bytes:count,dst},
                ReadTransform::F32ToBf16=>Job::NarrowF32Bf16 {at:0,bytes:count,dst},
                ReadTransform::Direct=>unreachable!(),
            };
            records.steps.push(Step {device,rects:vec![rect],jobs:vec![job]});
            at+=count;
        }
    } else {
        let rect = Rect { name: metadata.name.clone(), path: path.into(), base: metadata.byte_offset,
            row_bytes: bytes, rows: 0..1, cols: 0..bytes };
        records.operands.push(Operand { device, buffer: buffer(destination, resident),
            lands: vec![(rect, 0)], bytes: resident });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    /// A tower's header-only contribution at a synthetic arena address on `device`.
    pub(crate) fn vision_records(device: i32, spec: &TowerSpec) -> Result<Records> {
        Ok(Contribution::prepare(device, Some((spec, 1 << 40)), None, None, None)?.records())
    }
    pub(crate) fn assert_vision_plan(snapshot: &std::path::Path, spec: &TowerSpec) {
        let contribution = Contribution::prepare(0, Some((spec, 4096)), None, None, None).unwrap();
        let records = contribution.records();
        let source: u64 = spec.reads.iter().map(|read| read.metadata.byte_length).sum();
        let resident: usize = spec.reads.iter().map(|read|
            read.transform.resident_bytes(&read.metadata).unwrap()).sum();
        assert_records(snapshot, records, source, resident);
        assert!(spec.reads.iter().all(|read| read.destination + read.transform.resident_bytes(&read.metadata).unwrap() <= spec.native.weight_bytes as usize));
    }
    fn assert_records(snapshot: &std::path::Path, records: Records, source: u64, resident: usize) {
        let direct: usize = records.operands.iter().map(|operand| operand.bytes).sum();
        let widened: usize = records.steps.iter().flat_map(|step| &step.jobs).map(|job| match job {
            Job::Widen { bytes, dst, .. } => { assert_eq!(dst.bytes, bytes * 2); dst.bytes },
            Job::NarrowF32Bf16 { bytes, dst, .. } => {assert_eq!(dst.bytes,bytes/2);dst.bytes},
            _ => panic!("unexpected media transform"),
        }).sum();
        assert_eq!(direct + widened, resident);
        let rank = crate::shared::experts::rtx::dense::RankLayout::new(records.operands, records.steps, 0).unwrap();
        let checked = rank.checked(snapshot, 1, 0).unwrap();
        assert_eq!(checked.totals.rank_bytes, vec![source]);
        assert_eq!(checked.totals.file_bytes.iter().sum::<u64>(), source);
    }
    fn seed_payloads<'a>(reads: impl IntoIterator<Item = (&'a std::path::Path, &'a cuteafd_loader::SafetensorsTensorMetadata)>) {
        use std::os::unix::fs::FileExt;
        // Signed zero, infinities, NaN payloads and finite BF16 values must survive bitwise.
        let pattern: Vec<u8> = [0u16, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0x0001]
            .into_iter().flat_map(u16::to_le_bytes).collect();
        for (path, metadata) in reads {
            let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
            let count = pattern.len().min(metadata.byte_length as usize);
            file.write_all_at(&pattern[..count], metadata.byte_offset).unwrap();
            file.write_all_at(&pattern[..count], metadata.byte_offset + metadata.byte_length - count as u64).unwrap();
        }
    }

    fn assert_replayed_arena(snapshot: &std::path::Path, mounted: &[u8], constants: Option<(usize, &[u8])>,
        prepare: impl FnOnce(usize) -> Result<Records>) -> Result<()> {
        use crate::shared::{experts::rtx::dense, memory::{device::Device, DeviceAllocation}};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let root = std::path::PathBuf::from(std::env::var_os("HOME").context("HOME")?)
            .join(".cache/cuteafd/builds/media-tower-arena-replay-fixture");
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 16usize << 30)?;
        fixture.immediate_copies(&library)?;
        fixture.quiet_events(&library, true)?;
        let device = Device { library: &library, id: 0 };
        let allocation = DeviceAllocation::new(&library, mounted.len(), crate::shared::memory::TEST)?;
        let mut actual = vec![0; mounted.len()];
        library.copy_h2d(allocation.buffer, &actual)?;
        // The real vision prepare initializes this region independently of pushed payloads.
        if let Some((at, bytes)) = constants {
            let mut dst = allocation.buffer;
            dst.ptr = dst.ptr.wrapping_byte_add(at);
            dst.bytes = bytes.len();
            library.copy_h2d(dst, bytes)?;
        }
        let records = prepare(allocation.buffer.ptr as usize)?;
        let layout = dense::RankLayout::new(records.operands, records.steps, 0)?;
        dense::testing::replay(device, layout, snapshot, 1, 0)?;
        library.copy_d2h_quiescent(&mut actual, allocation.buffer)?;
        assert_eq!(actual.len(), mounted.len());
        assert!(actual == mounted, "pushed arena differs at byte {:?}",
            actual.iter().zip(mounted).position(|(a, b)| a != b));
        Ok(())
    }

    #[test]
    fn muse_vision_pcm_share_a_checked_group_without_changing_resident_bytes() {
        let root = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::write_muse_media_snapshot(root.path(), &[0]);
        let vision = TowerSpec::muse(root.path(), 4096).unwrap();
        super::super::constants::assert_original_table(&vision);
        assert_vision_plan(root.path(), &vision);
        let pcm = PcmTowerSpec::from_snapshot(root.path(), 19200).unwrap();
        let records = Contribution::prepare(0, Some((&vision, 4096)),
            Some((&pcm, 4096 + vision.native.weight_bytes as usize)), None, None).unwrap().records();
        let source = vision.reads.iter().map(|read| read.metadata.byte_length).sum::<u64>()
            + pcm.push_reads().map(|read| read.metadata.byte_length).sum::<u64>();
        let resident = vision.reads.iter().map(|read| read.transform.resident_bytes(&read.metadata).unwrap()).sum::<usize>()
            + pcm.push_reads().map(|read| read.metadata.byte_length as usize).sum::<usize>();
        assert_records(root.path(), records, source, resident);
    }
    #[test]
    fn pushed_vision_and_pcm_arenas_match_mounted_bytes() -> Result<()> {
        let root = tempfile::tempdir()?;
        let shard = root.path().join("tower.safetensors");
        cuteafd_loader::plan::testing::write_safetensors(&shard, &[
            cuteafd_loader::plan::testing::t("matrix", "BF16", &[8, 16]),
            cuteafd_loader::plan::testing::t("vector", "BF16", &[8]),
        ]);
        let headers = cuteafd_loader::read_safetensors_metadata(&shard)?;
        let reads = headers.into_iter().enumerate().map(|(index, metadata)| super::super::TensorRead {
            path: shard.clone(), metadata, destination: index * 256, transform: if index == 1 {ReadTransform::Bf16ToF32}else{ReadTransform::Direct},
        }).collect::<Vec<_>>();
        let vision = TowerSpec { native: cuteafd_ffi::vision::VisionSpec {
            abi_version: 2, head_dim: 8, inv_freq: 512, weight_bytes: 520, ..Default::default()
        }, reads, constants:vec![super::super::ConstantRegion::fp32("rotary.inv_freq",512,
            [1.0,0.01],"two FP32 constants","fixture")], family: None };
        seed_payloads(vision.reads.iter().map(|read| (read.path.as_path(), &read.metadata)));
        let constants = &vision.constants[0].bytes;
        assert_replayed_arena(root.path(), &vision.load_weights()?,
            Some((vision.native.inv_freq as usize, &constants)), |address|
                Ok(Contribution::prepare(0, Some((&vision, address)), None, None, None)?.records()))?;
        // The PCM branch is direct-copy-only; the full-shape ledger test above
        // already validates every real PCM source and owner. Replay a private
        // miniature of those exact records instead of a second 236 MiB buffer.
        let mut pcm = vision.clone();
        for read in &mut pcm.reads { read.transform = ReadTransform::Direct; }
        let expected = pcm.load_weights()?;
        assert_replayed_arena(root.path(), &expected,
            Some((pcm.constants[0].offset as usize, &pcm.constants[0].bytes)), |address| {
                let mut records = Records::default();
                for read in &pcm.reads {
                    append(&mut records, 0, address, pcm.native.weight_bytes as usize,
                        &read.path, &read.metadata, read.destination, ReadTransform::Direct)?;
                }
                Ok(records)
            })?;
        Ok(())
    }
    #[test]
    fn vision_push_rejects_native_constant_and_record_overlaps() {
        let root = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::write_muse_media_snapshot(root.path(), &[0]);
        let spec = TowerSpec::muse(root.path(), 4096).unwrap();
        let prepare = |spec: &TowerSpec| Contribution::prepare(0, Some((spec, 4096)), None, None, None);
        assert!(prepare(&spec).is_ok());
        let mut constants = spec.clone();
        // Isolate the vector so only its widened tail can intersect native constants.
        let mut vector = constants.reads[0].clone();
        vector.transform = ReadTransform::Bf16ToF32;
        vector.metadata.byte_length = 2;
        vector.metadata.shape = vec![1];
        vector.destination = constants.native.inv_freq as usize - 3;
        constants.reads = vec![vector];
        assert!(prepare(&constants).err().unwrap().to_string().contains("overlapping"));
        let mut duplicate = spec.clone();
        duplicate.reads.push(duplicate.reads[0].clone());
        assert!(prepare(&duplicate).err().unwrap().to_string().contains("overlapping"));
        let records = prepare(&spec).unwrap().records();
        let first = &records.operands[0].buffer;
        let start = first.ptr as usize;
        assert!(check_destinations(&records, vec![(first.device_id + 1, start, start + first.bytes)]).is_ok());
        assert!(check_destinations(&records, vec![(first.device_id, start + first.bytes - 1,
            start + first.bytes + 1)]).is_err());
    }
    #[test]
    fn mimo_fp32_audio_plan_pushes_codebooks_and_widens_bounded_matrices() {
        use cuteafd_loader::plan::testing::{write_safetensors, Tensor};
        use serde_json::json;
        let root = tempfile::tempdir().unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/tests/fixtures/mimo-qualified-pro.json"))).unwrap();
        std::fs::write(root.path().join("config.json"), fixture["target_config"].to_string()).unwrap();
        let tokenizer = root.path().join("audio_tokenizer");
        std::fs::create_dir(&tokenizer).unwrap();
        let books = [1024,1024,256,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128];
        std::fs::write(tokenizer.join("config.json"), json!({
            "max_audio_seconds":300,"stride_size":2,"avg_pooler":2,"d_model":1024,"kernel_size":3,
            "encoder_layers":24,"encoder_skip_layer_id":3,"encoder_attention_heads":16,"encoder_ffn_dim":4096,
            "nfft":960,"n_mels":128,"sampling_rate":24000,"hop_length":240,"window_size":960,"fmin":0,
            "num_quantizers":20,"rope_theta":10000,"encoder_causal":true,"hybrid_attention":true,
            "swa_per_block":2,"scale_embedding":false,"activation_function":"gelu","position_embedding_type":"rope",
            "rope_type":"default","ln_type":"LayerNorm","encoder_attn_window_size":[128,0],"codebook_size":books,"fmax":null
        }).to_string()).unwrap();
        let mut tensors: Vec<Tensor> = Vec::new();
        let mut put = |name: String, shape: Vec<usize>| tensors.push((name, "BF16", shape));
        for (name, shape) in [
            ("conv1.weight",vec![1024,128,3]),("conv1.bias",vec![1024]),
            ("conv2.weight",vec![1024,1024,3]),("conv2.bias",vec![1024]),
            ("down_sample_layer.0.weight",vec![1024,1024,2]),("layer_norm.weight",vec![1024]),
            ("layer_norm.bias",vec![1024]),("down_sample_norm.weight",vec![1024]),("down_sample_norm.bias",vec![1024]),
        ] { put(format!("encoder.{name}"), shape); }
        for i in 0..24 {
            for (name, shape) in [
                ("self_attn.q_proj.weight",vec![1024,1024]),("self_attn.q_proj.bias",vec![1024]),
                ("self_attn.k_proj.weight",vec![1024,1024]),("self_attn.v_proj.weight",vec![1024,1024]),
                ("self_attn.v_proj.bias",vec![1024]),("self_attn.out_proj.weight",vec![1024,1024]),
                ("self_attn.out_proj.bias",vec![1024]),("self_attn_layer_norm.weight",vec![1024]),
                ("self_attn_layer_norm.bias",vec![1024]),("final_layer_norm.weight",vec![1024]),
                ("final_layer_norm.bias",vec![1024]),("fc1.weight",vec![4096,1024]),("fc1.bias",vec![4096]),
                ("fc2.weight",vec![1024,4096]),("fc2.bias",vec![1024]),
            ] { put(format!("encoder.layers.{i}.{name}"), shape); }
        }
        for i in 0..20 { put(format!("speech_embeddings.{i}.weight"),vec![1280,1024]); }
        for i in 0..6 {
            for (name, shape) in [
                ("input_layernorm.weight",vec![1024]),("post_attention_layernorm.weight",vec![1024]),
                ("self_attn.q_proj.weight",vec![1024,1024]),("self_attn.q_proj.bias",vec![1024]),
                ("self_attn.k_proj.weight",vec![1024,1024]),("self_attn.k_proj.bias",vec![1024]),
                ("self_attn.v_proj.weight",vec![1024,1024]),("self_attn.v_proj.bias",vec![1024]),
                ("self_attn.o_proj.weight",vec![1024,1024]),("mlp.gate_proj.weight",vec![4096,1024]),
                ("mlp.up_proj.weight",vec![4096,1024]),("mlp.down_proj.weight",vec![1024,4096]),
            ] { put(format!("audio_encoder.input_local_transformer.layers.{i}.{name}"), shape); }
        }
        put("audio_encoder.input_local_transformer.norm.weight".into(),vec![1024]);
        put("audio_encoder.projection.mlp.0.weight".into(),vec![16384,4096]);
        put("audio_encoder.projection.mlp.2.weight".into(),vec![6144,16384]);
        drop(put);
        for (i, book) in books.into_iter().enumerate() {
            tensors.push((format!("encoder.quantizer.vq.layers.{i}._codebook.embed"),"F32",vec![book,1024]));
        }
        let (codec, lm): (Vec<_>, Vec<_>) = tensors.into_iter().partition(|tensor| tensor.0.starts_with("encoder."));
        write_safetensors(&tokenizer.join("model.safetensors"), &codec);
        write_safetensors(&root.path().join("model.safetensors"), &lm);
        let spec = super::super::audio::AudioTowerSpec::from_snapshot(root.path(), 24000).unwrap();
        let before = spec.plan().weight_bytes();
        let records = Contribution::prepare(0, None, None, Some((&spec,4096)), None).unwrap().records();
        assert_eq!(records.operands.len(), 20);
        assert!(records.steps.iter().all(|step| step.rects.iter().map(Rect::bytes).sum::<usize>()
            <= crate::shared::experts::rtx::dense::TRANSFORM_BYTES));
        let source = spec.plan().reads().values().map(|read| read.metadata.byte_length).sum();
        let resident = spec.plan().reads().values().map(|read| read.metadata.byte_length as usize
            * if read.metadata.dtype == cuteafd_core::DType::Bf16 { 2 } else { 1 }).sum();
        assert_records(root.path(), records, source, resident);
        assert_eq!(spec.native().weight_bytes,before);
        seed_payloads(spec.plan().reads().values().map(|read| (read.path.as_path(), &read.metadata)));
        let mounted = spec.plan().load_weights(before).unwrap();
        assert_replayed_arena(root.path(), &mounted, None, |address|
            Ok(Contribution::prepare(0, None, None, Some((&spec, address)), None)?.records())).unwrap();
    }
    #[test]
    fn widening_is_bounded_and_rejects_invalid_extents() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("weights");
        let bytes = crate::shared::experts::rtx::dense::TRANSFORM_BYTES * 2 + 2;
        std::fs::File::create(&path).unwrap().set_len(bytes as u64).unwrap();
        let mut metadata = cuteafd_loader::SafetensorsTensorMetadata {
            name: "audio.matrix".into(), dtype: cuteafd_core::DType::Bf16,
            shape: vec![bytes / 2], byte_offset: 0, byte_length: bytes as u64,
        };
        let mut records = Records::default();
        append(&mut records, 0, 4096, bytes * 2, &path, &metadata, 0, ReadTransform::Bf16ToF32).unwrap();
        assert_eq!(records.steps.len(), 3);
        assert_records(root.path(), records, bytes as u64, bytes * 2);
        metadata.byte_length = 3;
        assert!(append(&mut Records::default(), 0, 4096, 6, &path, &metadata, 0, ReadTransform::Bf16ToF32).is_err());
        assert!(append(&mut Records::default(), 0, usize::MAX, 6, &path, &metadata, 0, ReadTransform::Direct).is_err());
    }
    #[test]
    fn abandoned_contribution_cancels_owner_without_publishing() {
        let (commit, committed) = mpsc::channel();
        let (ack, acknowledgement) = mpsc::sync_channel(1);
        let (cancel, _) = mpsc::channel();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed = cancelled.clone();
        let thread = std::thread::spawn(move || {
            observed.store(committed.recv() == Ok(false), std::sync::atomic::Ordering::Release);
            drop(ack);
        });
        let owner = Arc::new(OwnerThread::new(thread, cancel));
        let contribution = Contribution::prepare(0, None, None, None, Some((commit, acknowledgement, owner))).unwrap();
        drop(contribution);
        assert!(cancelled.load(std::sync::atomic::Ordering::Acquire));
    }
    #[test]
    fn commit_waits_for_owner_ack_and_propagates_failure() {
        let (commit, committed) = mpsc::channel();
        let (ack, acknowledgement) = mpsc::sync_channel(1);
        let (cancel, _) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            assert_eq!(committed.recv(), Ok(true));
            ack.send(Err(super::super::VisionError::Unavailable)).unwrap();
        });
        let owner = Arc::new(OwnerThread::new(thread, cancel));
        let error = Contribution::prepare(0, None, None, None, Some((commit, acknowledgement, owner))).unwrap().commit().unwrap_err();
        assert!(error.downcast_ref::<PushStartupError>().is_some());
    }
}

#[cfg(test)]
mod narrow_tests {
    use super::*;
    use cuteafd_core::DType;
    #[test]
    fn narrowing_is_a_bounded_transient_step_not_a_second_resident_operand() -> Result<()> {
        let metadata=cuteafd_loader::SafetensorsTensorMetadata {name:"patch".into(),dtype:DType::F32,
            shape:vec![8],byte_offset:16,byte_length:32};
        let mut records=Records::default();
        append(&mut records,0,4096,256,std::path::Path::new("not-read.safetensors"),&metadata,0,ReadTransform::F32ToBf16)?;
        assert!(records.operands.is_empty());assert_eq!(records.steps.len(),1);
        let step=&records.steps[0];assert_eq!(step.rects[0].bytes(),32);
        match &step.jobs[0] {
            Job::NarrowF32Bf16 {at,bytes,dst}=> {assert_eq!((*at,*bytes,dst.bytes),(0,32,16));assert_eq!(dst.ptr as usize,4096);}
            _=>panic!("wrong conversion job"),
        }
        check_destinations(&records,vec![])?;
        assert!(append(&mut Records::default(),0,4096,8,std::path::Path::new("x"),&metadata,0,ReadTransform::F32ToBf16).is_err());
        Ok(())
    }
}
