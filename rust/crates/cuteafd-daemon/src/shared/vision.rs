//! Cold-path tower description, resident runtime and bounded local owner service.
//! MiMo arithmetic/order is ported from Hugh Madden's mimo26f-afd v1.3.0,
//! crates/mimo26-coordinator/src/vision.rs; weights are resident, never transient.
mod glm_flash;
mod qwen;
pub mod local;
pub mod remote;
pub mod worker;

use cuteafd_core::DType;
use cuteafd_ffi::vision::{NativeVision, VisionBlock, VisionLedger, VisionSpec, NO_VISION_OFFSET};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
};

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("vision unsupported: {0}")]
    Unsupported(String),
    #[error("vision I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("vision config: {0}")]
    Json(#[from] serde_json::Error),
    #[error("vision native: {0}")]
    Native(#[from] cuteafd_ffi::vision::VisionError),
    #[error("vision encoder queue is full")]
    QueueFull,
    #[error("vision encoder unavailable")]
    Unavailable,
    #[error("vision encoding cancelled")]
    Cancelled,
}
type Result<T> = std::result::Result<T, VisionError>;

#[derive(Clone, Debug, Deserialize)]
struct MimoConfig {
    model_type: String,
    hidden_size: u32,
    vision_config: MimoVisionConfig,
}
#[derive(Clone, Debug, Deserialize)]
struct MimoVisionConfig {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    out_hidden_size: u32,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    #[serde(default = "default_head_dim")]
    qk_channels: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    hidden_act: String,
    fullatt_block_indexes: Vec<usize>,
    vit_window_attn_types: Vec<i32>,
    visual_token_window_size: i32,
    use_sink: bool,
}
fn default_head_dim() -> usize {
    64
}
fn default_eps() -> f32 {
    1e-6
}
impl MimoConfig {
    fn validate(&self, max_tokens: usize) -> Result<()> {
        let v = &self.vision_config;
        let fixed = [
            ("depth", v.depth, 28),
            ("hidden_size", v.hidden_size, 1280),
            ("intermediate_size", v.intermediate_size, 4608),
            ("num_heads", v.num_heads, 32),
            ("num_key_value_heads", v.num_key_value_heads, 8),
            ("qk_channels", v.qk_channels, 64),
            ("patch_size", v.patch_size, 16),
            ("temporal_patch_size", v.temporal_patch_size, 2),
            ("spatial_merge_size", v.spatial_merge_size, 2),
        ];
        for (name, actual, want) in fixed {
            if actual != want {
                return Err(VisionError::Unsupported(format!(
                    "vision_config.{name}={actual}, need {want}; add a tower exporter/kernel"
                )));
            }
        }
        if self.model_type != "mimo_v2"
            || ![4096, 6144].contains(&v.out_hidden_size)
            || self.hidden_size != v.out_hidden_size
            || max_tokens == 0
            || max_tokens > 4096
            || v.rms_norm_eps != 1e-6
            || v.hidden_act != "silu"
            || v.vit_window_attn_types.len() != 28
            || v.vit_window_attn_types.last() == Some(&1)
            || v.vit_window_attn_types
                .iter()
                .any(|t| ![-1, 0, 1].contains(t))
            || v.fullatt_block_indexes.iter().any(|i| *i >= 28)
            || !(1..=64).contains(&v.visual_token_window_size)
        {
            return Err(VisionError::Unsupported(
                "MiMo tower config/merger geometry or image capacity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct TensorRead {
    path: PathBuf,
    metadata: SafetensorsTensorMetadata,
    destination: usize,
    vector: bool,
}
/// Header-only plan. It touches no tensor data until weight/scratch admission.
#[derive(Clone, Debug)]
pub struct TowerSpec {
    pub native: VisionSpec,
    reads: Vec<TensorRead>,
}
fn tower_catalog(snapshot: &Path, prefix: &str) -> Result<BTreeMap<String, (PathBuf, SafetensorsTensorMetadata)>> {
    let mut files = BTreeSet::new();
    let index_path = snapshot.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value = serde_json::from_reader(File::open(index_path)?)?;
        let map = index["weight_map"]
        .as_object()
        .ok_or_else(|| VisionError::Unsupported("safetensors weight_map absent".into()))?;
        for (name, file) in map {
        if name.starts_with(prefix) {
            let file = file.as_str().ok_or_else(|| {
            VisionError::Unsupported(format!("invalid file for {name}"))
            })?;
            let path = Path::new(file);
            if path.is_absolute()
            || path
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
            return Err(VisionError::Unsupported(format!(
                "invalid shard path {file}"
            )));
            }
            files.insert(snapshot.join(path));
        }
        }
    } else {
        for entry in std::fs::read_dir(snapshot)? {
        let path = entry?.path();
        if path.extension().is_some_and(|s| s == "safetensors") {
            files.insert(path);
        }
        }
    }
    let mut tensors = BTreeMap::new();
    for path in files {
        for metadata in read_safetensors_metadata(&path)
        .map_err(|e| VisionError::Unsupported(e.to_string()))?
        {
        if metadata.name.starts_with(prefix) {
            let name = metadata.name.clone();
            if tensors
            .insert(name.clone(), (path.clone(), metadata))
            .is_some()
            {
            return Err(VisionError::Unsupported(format!("duplicate {name}")));
            }
        }
        }
    }
    Ok(tensors)
}

impl TowerSpec {
    pub fn from_snapshot(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_reader(File::open(snapshot.join("config.json"))?)?;
        match cfg["model_type"].as_str() {
            Some("mimo_v2") => Self::mimo(snapshot, max_tokens),
            Some("qwen4_exp") => Self::qwen(snapshot, max_tokens),
            Some("glm5_next") => Self::glm_flash(snapshot, max_tokens),
            kind => Err(VisionError::Unsupported(format!("tower model_type {kind:?}; add a tower exporter/kernel"))),
        }
    }
    pub fn image_family(&self) -> cuteafd_loader::media::ImageFamily {
        use cuteafd_loader::media::ImageFamily;
        match self.native.reserved {
            2 => ImageFamily::Qwen,
            3 => ImageFamily::GlmFlash,
            _ => ImageFamily::Mimo,
        }
    }
    pub fn mimo(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let cfg: MimoConfig = serde_json::from_reader(File::open(snapshot.join("config.json"))?)?;
        cfg.validate(max_tokens)?;
        let tensors = tower_catalog(snapshot, "visual.")?;
        for bias in [
            "visual.merger.ln_q.bias",
            "visual.merger.mlp.0.bias",
            "visual.merger.mlp.2.bias",
        ] {
            if tensors.contains_key(bias) {
                return Err(VisionError::Unsupported(format!("{bias} present; this MiMo merger assumes the official missing zero-initialized bias")));
            }
        }
        let mut reads = Vec::new();
        let mut cursor = 0usize;
        let mut put = |name: &str, shape: &[usize], vector: bool, align: bool| -> Result<u64> {
            let full = format!("visual.{name}");
            let (path, metadata) = tensors
                .get(&full)
                .ok_or_else(|| VisionError::Unsupported(format!("missing {full}")))?;
            if metadata.dtype != DType::Bf16
                || metadata.shape != shape
                || metadata.byte_length != (shape.iter().product::<usize>() * 2) as u64
            {
                return Err(VisionError::Unsupported(format!(
                    "{full}: {:?} {:?}; need BF16 {shape:?}",
                    metadata.dtype, metadata.shape
                )));
            }
            if align {
                cursor = (cursor + 255) & !255;
            }
            let destination = cursor;
            cursor += metadata.byte_length as usize * if vector { 2 } else { 1 };
            reads.push(TensorRead {
                path: path.clone(),
                metadata: metadata.clone(),
                destination,
                vector,
            });
            Ok(destination as u64)
        };
        let mut native = VisionSpec {
            abi_version: 1,
            max_tokens: max_tokens as u32,
            output_width: cfg.hidden_size,
            ..Default::default()
        };
        native.patch = put(
            "patch_embed.proj.weight",
            &[1280, 3, 2, 16, 16],
            false,
            true,
        )?;
        let v = cfg.vision_config;
        for (i, block) in native.blocks.iter_mut().enumerate() {
            let mut field = |suffix: &str, shape: &[usize], vector: bool, align: bool| {
                put(&format!("blocks.{i}.{suffix}"), shape, vector, align)
            };
            *block = VisionBlock {
                qkv: field("attn.qkv.weight", &[3072, 1280], false, true)?,
                qkv_bias: field("attn.qkv.bias", &[3072], true, true)?,
                proj: field("attn.proj.weight", &[1280, 2048], false, true)?,
                proj_bias: field("attn.proj.bias", &[1280], true, true)?,
                gate_up: field("mlp.gate_proj.weight", &[4608, 1280], false, true)?,
                gate_up_bias: 0,
                down: 0,
                down_bias: 0,
                norm1: 0,
                norm2: 0,
                key0_bias: NO_VISION_OFFSET,
                window: if v.fullatt_block_indexes.contains(&i) {
                    0
                } else {
                    v.visual_token_window_size
                },
                column_order: i32::from(v.vit_window_attn_types[i] == 1),
            };
            field("mlp.up_proj.weight", &[4608, 1280], false, false)?;
            block.gate_up_bias = field("mlp.gate_proj.bias", &[4608], true, true)?;
            field("mlp.up_proj.bias", &[4608], true, false)?;
            block.down = field("mlp.down_proj.weight", &[1280, 4608], false, true)?;
            block.down_bias = field("mlp.down_proj.bias", &[1280], true, true)?;
            block.norm1 = field("norm1.weight", &[1280], true, true)?;
            block.norm2 = field("norm2.weight", &[1280], true, true)?;
            if v.use_sink && block.window != 0 {
                block.key0_bias = field("attn.sinks", &[32], true, true)?;
            }
        }
        native.merger_norm = put("merger.ln_q.weight", &[1280], true, true)?;
        native.merger_fc1 = put("merger.mlp.0.weight", &[5120, 5120], false, true)?;
        native.merger_fc2 = put(
            "merger.mlp.2.weight",
            &[cfg.hidden_size as usize, 5120],
            false,
            true,
        )?;
        drop(put);
        cursor = (cursor + 255) & !255;
        native.inv_freq = cursor as u64;
        native.weight_bytes = (cursor + 64) as u64;
        Ok(Self { native, reads })
    }
    /// Canonical header extents, sorted by tensor name; no tower payload is read.
    pub fn encoder_id(&self, revision: &str, sm: u32) -> cuteafd_loader::media::EncoderId {
        let headers = self.reads.iter().map(|read| {
            let m = &read.metadata;
            (m.name.clone(), serde_json::json!({"dtype": format!("{:?}", m.dtype),
                "shape": m.shape, "byte_offset": m.byte_offset, "byte_length": m.byte_length}))
        }).collect();
        let family = match self.image_family() {
            cuteafd_loader::media::ImageFamily::Mimo => "mimo_v2",
            cuteafd_loader::media::ImageFamily::Qwen => "qwen4",
            cuteafd_loader::media::ImageFamily::GlmFlash => "glm5_flash",
        };
        cuteafd_loader::media::EncoderId::derive(family, revision, &headers, 1, sm)
    }

    fn load_weights(&self) -> Result<Vec<u8>> {
        let mut weights = vec![0u8; self.native.weight_bytes as usize];
        for read in &self.reads {
            let m = &read.metadata;
            let mut file = File::open(&read.path)?;
            file.seek(SeekFrom::Start(m.byte_offset))?;
            let start = read.destination;
            if read.vector {
                let mut raw = vec![0u8; m.byte_length as usize];
                file.read_exact(&mut raw)?;
                for (i, bytes) in raw.chunks_exact(2).enumerate() {
                    let value =
                        f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16);
                    weights[start + i * 4..start + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
            } else {
                file.read_exact(&mut weights[start..start + m.byte_length as usize])?;
            }
        }
        let frequencies = if self.native.abi_version == 2 { self.native.head_dim as usize / 4 } else { 16 };
        for i in 0..frequencies {
            let value = 1.0f32 / 10000f32.powf(i as f32 / frequencies as f32);
            let start = self.native.inv_freq as usize + i * 4;
            weights[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(weights)
    }
}

/// Matches the CPU reference's f64 rescale followed by f32 normalize, without FMA.
pub fn normalization_lut(config: &cuteafd_loader::media::ProcessorConfig) -> Arc<[f32; 768]> {
    Arc::new(std::array::from_fn(|i| {
        let c = i / 256;
        let pixel = ((i % 256) as f64 * (1.0 / 255.0)) as f32;
        (pixel - config.mean[c] as f32) / config.std[c] as f32
    }))
}

pub struct VitRuntime {
    native: NativeVision,
    pub spec: TowerSpec,
}
impl VitRuntime {
    /// `admitted_bytes` comes from the placement memory ledger, not cudaMemGetInfo.
    pub fn load(spec: TowerSpec, library: &Path, device: i32, admitted_bytes: u64) -> Result<Self> {
        let required = NativeVision::required(library, &spec.native)?;
        if required.total_bytes() > admitted_bytes {
            return Err(VisionError::Unsupported(format!(
                "tower admission needs {} bytes, got {admitted_bytes}",
                required.total_bytes()
            )));
        }
        let weights = spec.load_weights()?;
        let native = NativeVision::load(library, &spec.native, device, admitted_bytes, &weights)?;
        Ok(Self { native, spec })
    }
    pub fn encode_into(
        &mut self,
        rgb: &[u8],
        grid: [usize; 2],
        lut: &[f32; 768],
        output: &mut [u16],
    ) -> Result<()> {
        self.native.encode_into(rgb, grid, lut, output)?;
        Ok(())
    }
    pub fn ledger(&self) -> Result<VisionLedger> {
        Ok(self.native.ledger()?)
    }
}

pub struct EncodeJob {
    pub rgb: Arc<[u8]>,
    pub grid: [usize; 2],
    pub lut: Arc<[f32; 768]>,
    pub output: Vec<u16>,
}
pub struct EncoderTicket {
    result: mpsc::Receiver<Result<Vec<u16>>>,
    cancelled: Arc<AtomicBool>,
}
impl EncoderTicket {
    pub fn poll(&self) -> Result<Option<Vec<u16>>> {
        match self.result.try_recv() {
            Ok(result) => result.map(Some),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(VisionError::Unavailable),
        }
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
impl Drop for EncoderTicket {
    fn drop(&mut self) {
        self.cancel();
    }
}
struct Work {
    job: EncodeJob,
    reply: mpsc::SyncSender<Result<Vec<u16>>>,
    cancelled: Arc<AtomicBool>,
}
struct OwnerHealth(Arc<AtomicBool>);
impl Drop for OwnerHealth {
    fn drop(&mut self) { self.0.store(false, Ordering::Release); }
}

fn terminal_encode_failure(result: &Result<Vec<u16>>) -> bool {
    matches!(result, Err(VisionError::Unavailable | VisionError::Native(
        cuteafd_ffi::vision::VisionError::Native(_) | cuteafd_ffi::vision::VisionError::Runtime(_))))
}

fn reply_encode(healthy: &AtomicBool, reply: &mpsc::SyncSender<Result<Vec<u16>>>, result: Result<Vec<u16>>) -> bool {
    let terminal = terminal_encode_failure(&result);
    if terminal { healthy.store(false, Ordering::Release); }
    let _ = reply.send(result);
    terminal
}

/// Bounded queue, one image at a time. Every CUDA call stays on this thread.
/// Dropping the service closes admission, drains accepted jobs and joins owner.
pub struct EncoderService {
    queue: Option<mpsc::SyncSender<Work>>,
    owner: Option<JoinHandle<()>>,
    pub ledger: VisionLedger,
    healthy: Arc<AtomicBool>,
}
impl EncoderService {
    pub fn start(
        spec: TowerSpec,
        library: PathBuf,
        device: i32,
        admitted_bytes: u64,
    ) -> Result<Self> {
        let (queue, jobs) = mpsc::sync_channel::<Work>(2);
        let (ready, readiness) = mpsc::sync_channel(1);
        let healthy = Arc::new(AtomicBool::new(false));
        let owner_health = healthy.clone();
        let owner = thread::Builder::new()
            .name("vision-owner".into())
            .spawn(move || {
                let _health = OwnerHealth(owner_health.clone());
                let mut runtime = match VitRuntime::load(spec, &library, device, admitted_bytes) {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                let ledger = match runtime.ledger() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                owner_health.store(true, Ordering::Release);
                if ready.send(Ok(ledger)).is_err() {
                    return;
                }
                while let Ok(mut work) = jobs.recv() {
                    let result = if work.cancelled.load(Ordering::Acquire) {
                        Err(VisionError::Cancelled)
                    } else {
                        runtime
                            .encode_into(
                                &work.job.rgb,
                                work.job.grid,
                                &work.job.lut,
                                &mut work.job.output,
                            )
                            .and_then(|()| {
                                if work.cancelled.load(Ordering::Acquire) {
                                    Err(VisionError::Cancelled)
                                } else {
                                    Ok(work.job.output)
                                }
                            })
                    };
                    if let Err(ref error) = result {
                        tracing::debug!(%error, "vision job failed");
                    }
                    // CUDA errors poison this owner: queued tickets fail closed.
                    if reply_encode(&owner_health, &work.reply, result) { break; }
                }
            })?;
        match readiness.recv() {
            Ok(Ok(ledger)) => Ok(Self {
                queue: Some(queue),
                owner: Some(owner),
                ledger,
                healthy,
            }),
            Ok(Err(error)) => {
                drop(queue);
                let _ = owner.join();
                Err(error)
            }
            Err(_) => {
                drop(queue);
                let _ = owner.join();
                Err(VisionError::Unavailable)
            }
        }
    }
    pub fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire) && self.owner.as_ref().is_some_and(|owner| !owner.is_finished())
    }
    pub fn health_handle(&self) -> Arc<AtomicBool> { self.healthy.clone() }
    pub fn submit(&self, job: EncodeJob) -> Result<EncoderTicket> {
        if !self.healthy() { return Err(VisionError::Unavailable); }
        let (reply, result) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let work = Work {
            job,
            reply,
            cancelled: cancelled.clone(),
        };
        self.queue
            .as_ref()
            .ok_or(VisionError::Unavailable)?
            .try_send(work)
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => VisionError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => VisionError::Unavailable,
            })?;
        Ok(EncoderTicket { result, cancelled })
    }
}
impl Drop for EncoderService {
    fn drop(&mut self) {
        self.queue.take();
        if let Some(owner) = self.owner.take() {
            let _ = owner.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owner_exit_and_panic_publish_health_without_scheduler_polling() {
        for panic in [false, true] {
            let health = Arc::new(AtomicBool::new(true));
            let observed = health.clone();
            let owner = thread::spawn(move || {
                let _health = OwnerHealth(health);
                if panic { panic!("injected vision owner failure"); }
            });
            assert_eq!(owner.join().is_err(), panic);
            assert!(!observed.load(Ordering::Acquire));
        }
    }
    #[test]
    fn only_terminal_native_errors_poison_owner_before_reply() {
        for error in [VisionError::QueueFull, VisionError::Cancelled,
            VisionError::Native(cuteafd_ffi::vision::VisionError::InvalidInput("bad grid"))] {
            assert!(!terminal_encode_failure(&Err(error)));
        }
        for error in [VisionError::Unavailable,
            VisionError::Native(cuteafd_ffi::vision::VisionError::Native(1)),
            VisionError::Native(cuteafd_ffi::vision::VisionError::Runtime("CUDA failure".into()))] {
            assert!(terminal_encode_failure(&Err(error)));
        }
    }
    #[test]
    fn terminal_reply_publishes_unhealthy_and_disconnects_queued_tickets() {
        let healthy = Arc::new(AtomicBool::new(true));
        let owner_health = healthy.clone();
        let (reply, result) = mpsc::sync_channel(0);
        let (queued_reply, queued_result) = mpsc::sync_channel(1);
        let owner = thread::spawn(move || {
            let _health = OwnerHealth(owner_health.clone());
            assert!(reply_encode(&owner_health, &reply, Err(VisionError::Native(
                cuteafd_ffi::vision::VisionError::Native(1)))));
            drop(queued_reply);
        });
        assert!(matches!(result.recv().unwrap(), Err(VisionError::Native(_))));
        assert!(!healthy.load(Ordering::Acquire));
        let queued = EncoderTicket { result: queued_result, cancelled: Arc::new(AtomicBool::new(false)) };
        owner.join().unwrap();
        assert!(matches!(queued.poll(), Err(VisionError::Unavailable)));

        for error in [VisionError::QueueFull, VisionError::Cancelled] {
            healthy.store(true, Ordering::Release);
            let (reply, result) = mpsc::sync_channel(1);
            assert!(!reply_encode(&healthy, &reply, Err(error)));
            assert!(result.recv().unwrap().is_err());
            assert!(healthy.load(Ordering::Acquire));
        }
    }
    #[test]
    fn reject_unimplemented_geometry_before_weights() {
        let cfg = MimoConfig {
            model_type: "mimo_v2".into(),
            hidden_size: 4096,
            vision_config: MimoVisionConfig {
                depth: 28,
                hidden_size: 1280,
                intermediate_size: 4608,
                num_heads: 32,
                num_key_value_heads: 8,
                out_hidden_size: 4096,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                qk_channels: 64,
                rms_norm_eps: 1e-6,
                hidden_act: "silu".into(),
                fullatt_block_indexes: vec![0, 9, 18, 27],
                vit_window_attn_types: vec![0; 28],
                visual_token_window_size: 64,
                use_sink: true,
            },
        };
        assert!(cfg.validate(4096).is_ok());
        assert!(cfg.validate(4097).is_err());
        let mut pro = cfg.clone();
        pro.hidden_size = 6144;
        pro.vision_config.out_hidden_size = 6144;
        assert!(pro.validate(4096).is_ok());
        pro.vision_config.qk_channels = 72;
        assert!(pro
            .validate(4096)
            .unwrap_err()
            .to_string()
            .contains("qk_channels"));
    }
    fn sparse_snapshot(width: usize) -> tempfile::TempDir {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({"model_type":"mimo_v2","hidden_size":width,"vision_config":{
            "depth":28,"hidden_size":1280,"intermediate_size":4608,"num_heads":32,"num_key_value_heads":8,
            "out_hidden_size":width,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
            "hidden_act":"silu","fullatt_block_indexes":[0,9,18,27],"vit_window_attn_types":[-1,0,0,0,0,1,1,1,1,-1,0,0,0,0,1,1,1,1,-1,0,0,0,0,1,1,1,1,-1],
            "visual_token_window_size":64,"use_sink":true}});
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        let mut add = |name: String, shape: Vec<usize>| {
            let start = end;
            end += shape.iter().product::<usize>() as u64 * 2;
            header.insert(
                format!("visual.{name}"),
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,end]}),
            );
        };
        add("patch_embed.proj.weight".into(), vec![1280, 3, 2, 16, 16]);
        for i in 0..28 {
            for (name, shape) in [
                ("attn.qkv.weight", vec![3072, 1280]),
                ("attn.qkv.bias", vec![3072]),
                ("attn.proj.weight", vec![1280, 2048]),
                ("attn.proj.bias", vec![1280]),
                ("mlp.gate_proj.weight", vec![4608, 1280]),
                ("mlp.up_proj.weight", vec![4608, 1280]),
                ("mlp.gate_proj.bias", vec![4608]),
                ("mlp.up_proj.bias", vec![4608]),
                ("mlp.down_proj.weight", vec![1280, 4608]),
                ("mlp.down_proj.bias", vec![1280]),
                ("norm1.weight", vec![1280]),
                ("norm2.weight", vec![1280]),
            ] {
                add(format!("blocks.{i}.{name}"), shape);
            }
            if ![0, 9, 18, 27].contains(&i) {
                add(format!("blocks.{i}.attn.sinks"), vec![32]);
            }
        }
        add("merger.ln_q.weight".into(), vec![1280]);
        add("merger.mlp.0.weight".into(), vec![5120, 5120]);
        add("merger.mlp.2.weight".into(), vec![width, 5120]);
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(dir.path().join("vision.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + end).unwrap();
        let mut map = serde_json::Map::new();
        for name in header.keys() {
            map.insert(name.clone(), serde_json::json!("vision.safetensors"));
        }
        map.insert(
            "model.unread_lm.weight".into(),
            serde_json::json!("absent-lm-shard.safetensors"),
        );
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
        )
        .unwrap();
        dir
    }
    #[test]
    fn tower_plan_fuses_gate_up_and_never_opens_lm_payload() {
        for width in [4096, 6144] {
            let snapshot = sparse_snapshot(width);
            let tower = TowerSpec::mimo(snapshot.path(), 4096).unwrap();
            assert_eq!(tower.native.output_width, width as u32);
            assert_eq!(
                tower.native.weight_bytes,
                1_458_170_944 + (width as u64 - 4096) * 5120 * 2
            );
            assert_eq!(tower.native.blocks[0].key0_bias, NO_VISION_OFFSET);
            assert_eq!(tower.native.blocks[5].column_order, 1);
            for i in 0..28 {
                let gate = tower
                    .reads
                    .iter()
                    .find(|r| r.metadata.name == format!("visual.blocks.{i}.mlp.gate_proj.weight"))
                    .unwrap();
                let up = tower
                    .reads
                    .iter()
                    .find(|r| r.metadata.name == format!("visual.blocks.{i}.mlp.up_proj.weight"))
                    .unwrap();
                assert_eq!(
                    up.destination,
                    gate.destination + gate.metadata.byte_length as usize
                );
                assert_eq!(tower.native.blocks[i].gate_up, gate.destination as u64);
            }
            let config = snapshot.path().join("config.json");
            let mut cfg: serde_json::Value =
                serde_json::from_reader(File::open(&config).unwrap()).unwrap();
            cfg["vision_config"]["patch_size"] = serde_json::json!(14);
            std::fs::write(config, serde_json::to_vec(&cfg).unwrap()).unwrap();
            assert!(TowerSpec::mimo(snapshot.path(), 4096)
                .unwrap_err()
                .to_string()
                .contains("patch_size"));
        }
    }
    #[test]
    fn vector_loading_uses_absolute_extent_and_fp32_bits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.bin");
        std::fs::write(&path, [0, 0, 0, 0, 0x80, 0x3f, 0x00, 0xc0]).unwrap();
        let spec = TowerSpec {
            native: VisionSpec {
                weight_bytes: 256,
                inv_freq: 128,
                ..Default::default()
            },
            reads: vec![TensorRead {
                path,
                metadata: SafetensorsTensorMetadata {
                    name: "test".into(),
                    dtype: DType::Bf16,
                    shape: vec![2],
                    byte_offset: 4,
                    byte_length: 4,
                },
                destination: 0,
                vector: true,
            }],
        };
        let arena = spec.load_weights().unwrap();
        assert_eq!(&arena[0..4], &1.0f32.to_le_bytes());
        assert_eq!(&arena[4..8], &(-2.0f32).to_le_bytes());
        assert_eq!(&arena[128..132], &1.0f32.to_le_bytes());
    }
    #[test]
    #[ignore = "requires matching CUDA container, VISION_SNAPSHOT and VISION_LIBRARY"]
    fn resident_service_load_encode_cancel_and_drain() {
        let snapshot = PathBuf::from(std::env::var("VISION_SNAPSHOT").unwrap());
        let library = PathBuf::from(std::env::var("VISION_LIBRARY").unwrap());
        let spec = TowerSpec::mimo(&snapshot, 256).unwrap();
        let ledger = NativeVision::required(&library, &spec.native).unwrap();
        // Admission is rejected before reading any of the 1.4 GiB payload.
        assert!(matches!(
            VitRuntime::load(spec.clone(), &library, 0, ledger.total_bytes() - 1),
            Err(VisionError::Unsupported(_))
        ));
        let width = spec.native.output_width as usize;
        let service = EncoderService::start(spec, library, 0, ledger.total_bytes()).unwrap();
        assert_eq!(service.ledger.device_allocations, 2);
        let lut = Arc::new(std::array::from_fn(|i| (i % 256) as f32 / 127.5 - 1.0));
        let job = |grid: [usize; 2]| EncodeJob {
            rgb: vec![113; grid[0] * grid[1] * 768].into(),
            grid,
            lut: lut.clone(),
            output: vec![0; grid[0] * grid[1] / 4 * width],
        };
        let wait = |ticket: &EncoderTicket| {
            let start = std::time::Instant::now();
            loop {
                if let Some(output) = ticket.poll().unwrap() {
                    break output;
                }
                assert!(start.elapsed() < std::time::Duration::from_secs(30));
                std::thread::yield_now();
            }
        };
        let baseline = wait(&service.submit(job([4, 4])).unwrap());
        assert!(baseline.iter().any(|v| *v != 0));
        for _ in 0..3 {
            wait(&service.submit(job([8, 4])).unwrap());
            assert_eq!(wait(&service.submit(job([4, 4])).unwrap()), baseline);
        }
        let ticket = service.submit(job([8, 4])).unwrap();
        ticket.cancel();
        // Cancellation may race completion, but owner teardown must always drain.
        drop(ticket);
        drop(service);
    }
    #[test]
    fn cancellation_is_shared_and_disconnect_is_not_pending() {
        let (send, recv) = mpsc::sync_channel(1);
        let flag = Arc::new(AtomicBool::new(false));
        let ticket = EncoderTicket {
            result: recv,
            cancelled: flag.clone(),
        };
        assert!(ticket.poll().unwrap().is_none());
        ticket.cancel();
        assert!(flag.load(Ordering::Acquire));
        drop(send);
        assert!(matches!(ticket.poll(), Err(VisionError::Unavailable)));
    }
}
