//! V4.1 edge adapter: canonical BF16 patches in, complete BF16 spans out.
use super::v41_vision::VisionRuntime;
use crate::shared::vision::remote::{EncoderHandshake, EncoderServer, RemoteEncoder};
use anyhow::{ensure, Context, Result};
use cuteafd_core::ImageKey;
use cuteafd_engine::media::{EncodeJob, EncodeOutput, EncoderClient, EncoderTicket, MediaError};
use cuteafd_loader::{OfficialV41Catalog, V41Image, V41ImageGrid};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

const CAPACITY: usize = 9216;
fn media_error(error: impl std::fmt::Display) -> MediaError {
    MediaError::Encoder(error.to_string())
}

pub(crate) fn handshake(
    catalog: &OfficialV41Catalog,
    revision: &str,
    sm: u32,
    plan_hash: [u8; 32],
) -> Result<EncoderHandshake> {
    let headers = VisionRuntime::names()
        .iter()
        .map(|name| {
            let m = &catalog.tensor(name)?.metadata;
            Ok((
                name.clone(),
                serde_json::json!({"dtype":format!("{:?}",m.dtype),"shape":m.shape,
            "byte_offset":m.byte_offset,"byte_length":m.byte_length}),
            ))
        })
        .collect::<Result<_>>()?;
    Ok(EncoderHandshake {
        encoder_id: cuteafd_loader::media::EncoderId::derive(
            "deepseek_v41",
            revision,
            &headers,
            1,
            sm,
        ),
        max_patches: CAPACITY as u32,
        output_width: 5120,
        patch_size: 14,
        merge_size: 3,
        plan_hash,
    })
}

fn validate_job(job: &EncodeJob) -> std::result::Result<V41ImageGrid, MediaError> {
    job.validate()?;
    let ([t, h, w], patches) = job.image_input()?;
    let grid = V41ImageGrid {
        pixel_height: h as usize * 14,
        pixel_width: w as usize * 14,
        vit_height: h as usize,
        vit_width: w as usize,
        llm_height: (h as usize).div_ceil(3),
        llm_width: (w as usize).div_ceil(3),
    };
    let count = u64::from(h) * u64::from(w);
    if t != 1
        || job.hidden_width != 5120
        || h == 0
        || w == 0
        || count > CAPACITY as u64
        || grid.tokens() > 1024
        || grid.tokens() != job.tokens
        || patches.len() as u64 != count * 588 * 2
    {
        return Err(media_error(
            "invalid V4.1 BF16 patches or complete span geometry",
        ));
    }
    Ok(grid)
}

struct Work {
    job: EncodeJob,
    reply: mpsc::SyncSender<std::result::Result<EncodeOutput, MediaError>>,
    cancelled: Arc<AtomicBool>,
}
struct Pending {
    result: mpsc::Receiver<std::result::Result<EncodeOutput, MediaError>>,
    cancelled: Arc<AtomicBool>,
}
struct Health(Arc<AtomicBool>);
impl Drop for Health {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
/// Queue storage stays owned until CUDA drains, including after cancellation.
pub(crate) struct LocalEncoder {
    queue: Option<mpsc::SyncSender<Work>>,
    owner: Option<thread::JoinHandle<()>>,
    pending: HashMap<EncoderTicket, Pending>,
    next: u64,
    healthy: Arc<AtomicBool>,
    pub bytes: usize,
}
impl LocalEncoder {
    pub fn start(snapshot: PathBuf, library: PathBuf, device: i32, budget: usize) -> Result<Self> {
        let (queue, jobs) = mpsc::sync_channel::<Work>(2);
        let (ready, readiness) = mpsc::sync_channel(1);
        let healthy = Arc::new(AtomicBool::new(false));
        let health = healthy.clone();
        let owner = thread::Builder::new()
            .name("v41-vision-owner".into())
            .spawn(move || {
                let _health = Health(health.clone());
                let run = (|| -> Result<()> {
                    // SAFETY: selected architecture-matching native library; all CUDA owners live on this thread.
                    let lib = unsafe { cuteafd_ffi::NativeLibrary::load(&library) }?;
                    lib.cuda_set_device(device)?;
                    let catalog = cuteafd_loader::read_official_v41_catalog(
                        cuteafd_loader::OFFICIAL_V41_MODEL_ID,
                        &snapshot,
                    )?;
                    let bytes = VisionRuntime::device_bytes(&catalog, CAPACITY)?;
                    let info = lib.cuda_device_info(device)?;
                    let sm = info.compute_capability_major * 10 + info.compute_capability_minor;
                    let overhead = if sm == 121 {
                        cuteafd_loader::plan::encoder::V41_SPARK_CUDA_OVERHEAD_BYTES as usize
                    } else {
                        0
                    };
                    let admitted = bytes
                        .checked_add(overhead)
                        .context("vision admission overflow")?;
                    ensure!(
                        admitted <= budget,
                        "V4.1 vision needs {admitted} admitted bytes, budget {budget}"
                    );
                    let free_before = lib.cuda_physical_memory_info()?.0;
                    let available_before = if sm == 121 {
                        crate::shared::memory_report::unified_available_bytes()?
                    } else {
                        lib.cuda_memory_info()?.0
                    };
                    ensure!(
                        admitted <= available_before,
                        "V4.1 vision needs {admitted} bytes, available {available_before}"
                    );
                    let _scope = cuteafd_ffi::memory_ledger::scope("v41/vision");
                    let mut runtime = VisionRuntime::new(&lib, &catalog, CAPACITY, bytes)?;
                    // Cache reclamation changes raw CUDA free independently of
                    // allocations on GB10; retain the delta as telemetry only.
                    let free_after = lib.cuda_physical_memory_info()?.0;
                    tracing::info!(
                        device,
                        resident_bytes = bytes,
                        admitted_bytes = admitted,
                        cuda_overhead_bytes = overhead,
                        available_before,
                        free_before,
                        free_after,
                        observed_device_delta = free_before.saturating_sub(free_after),
                        "V4.1 vision owner admitted and allocated"
                    );
                    health.store(true, Ordering::Release);
                    ready.send(Ok(admitted)).ok();
                    while let Ok(work) = jobs.recv() {
                        if work.cancelled.load(Ordering::Acquire) {
                            continue;
                        }
                        let started = Instant::now();
                        let result = (|| -> Result<EncodeOutput> {
                            let grid = validate_job(&work.job).map_err(anyhow::Error::new)?;
                            let (_, patches) =
                                work.job.image_input().map_err(anyhow::Error::new)?;
                            let features = runtime.encode_patches(patches, grid, None)?;
                            let mut bytes = vec![0; features.bytes];
                            lib.copy_d2h(&mut bytes, features)?;
                            Ok(EncodeOutput {
                                key: work.job.key,
                                features: bytes.into(),
                                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                            })
                        })()
                        .map_err(media_error);
                        if result.is_err() {
                            health.store(false, Ordering::Release);
                        }
                        if !work.cancelled.load(Ordering::Acquire) {
                            let _ = work.reply.send(result);
                        }
                        if !health.load(Ordering::Acquire) {
                            break;
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = run {
                    let _ = ready.send(Err(format!("{error:#}")));
                }
            })?;
        match readiness.recv() {
            Ok(Ok(bytes)) => Ok(Self {
                queue: Some(queue),
                owner: Some(owner),
                pending: HashMap::new(),
                next: 0,
                healthy,
                bytes,
            }),
            failed => {
                drop(queue);
                let _ = owner.join();
                anyhow::bail!("V4.1 vision owner startup failed: {failed:?}")
            }
        }
    }
    pub fn health_handle(&self) -> Arc<AtomicBool> {
        self.healthy.clone()
    }
}
impl EncoderClient for LocalEncoder {
    fn submit(&mut self, job: EncodeJob) -> std::result::Result<EncoderTicket, MediaError> {
        validate_job(&job)?;
        if !self.healthy.load(Ordering::Acquire) {
            return Err(media_error("V4.1 vision unavailable"));
        }
        let id = EncoderTicket(self.next);
        self.next = self.next.checked_add(1).ok_or(MediaError::QueueFull)?;
        let (reply, result) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        self.queue
            .as_ref()
            .ok_or_else(|| media_error("V4.1 vision unavailable"))?
            .try_send(Work {
                job,
                reply,
                cancelled: cancelled.clone(),
            })
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => MediaError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => media_error("V4.1 owner stopped"),
            })?;
        self.pending.insert(id, Pending { result, cancelled });
        Ok(id)
    }
    fn poll(&mut self, id: EncoderTicket) -> Option<std::result::Result<EncodeOutput, MediaError>> {
        let result = match self.pending.get(&id)?.result.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => Err(media_error("V4.1 owner stopped")),
        };
        self.pending.remove(&id);
        Some(result)
    }
    fn cancel(&mut self, id: EncoderTicket) {
        if let Some(p) = self.pending.remove(&id) {
            p.cancelled.store(true, Ordering::Release);
        }
    }
}
impl Drop for LocalEncoder {
    fn drop(&mut self) {
        self.queue.take();
        if let Some(owner) = self.owner.take() {
            let _ = owner.join();
        }
    }
}

pub(crate) enum Encoder {
    Off,
    Local(LocalEncoder),
    Remote(RemoteEncoder),
    Pending {
        peers: Vec<std::net::SocketAddr>,
        handshake: EncoderHandshake,
    },
}
impl Encoder {
    pub fn load(args: &crate::cli::NativeServeArgs, catalog: &OfficialV41Catalog) -> Result<Self> {
        if !cuteafd_api::openai::vision_input_enabled() {
            ensure!(
                args.vision_peers.is_none(),
                "VISION=off refuses encoder peers"
            );
            return Ok(Self::Off);
        }
        if let Some(peers) = &args.vision_peers {
            let hash = crate::shared::vision::worker::parse_plan_hash(
                args.encoder_plan_hash
                    .as_deref()
                    .context("remote vision requires plan hash")?,
            )?;
            let revision = args
                .encoder_revision
                .as_deref()
                .context("remote vision requires revision")?;
            let peers = peers
                .split(',')
                .map(|peer| peer.parse())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            return Ok(Self::Pending {
                peers,
                handshake: handshake(catalog, revision, 121, hash)?,
            });
        }
        Ok(Self::Local(LocalEncoder::start(
            args.snapshot.clone(),
            args.native_lib.clone(),
            0,
            VisionRuntime::device_bytes(catalog, CAPACITY)?,
        )?))
    }
    /// Connect only after the launcher has consumed the expert-boundary handoff.
    pub fn connect(&mut self) -> Result<()> {
        if let Self::Pending { peers, handshake } = self {
            let remote =
                RemoteEncoder::connect(peers.clone(), handshake.clone(), Duration::from_secs(60))
                    .map_err(anyhow::Error::new)?;
            *self = Self::Remote(remote);
        }
        Ok(())
    }
    pub fn health_handle(&self) -> Option<Arc<AtomicBool>> {
        match self {
            Self::Off | Self::Pending { .. } => None,
            Self::Local(e) => Some(e.health_handle()),
            Self::Remote(e) => Some(e.health_handle()),
        }
    }
    pub fn image_job(image: &V41Image) -> EncodeJob {
        let grid = image.grid();
        EncodeJob::image(
            ImageKey(*image.identity()),
            [1, grid.vit_height as u32, grid.vit_width as u32],
            image.patches().to_vec().into(),
            grid.tokens(),
            5120,
        )
    }

}

impl EncoderClient for Encoder {
    fn submit(&mut self, job: EncodeJob) -> std::result::Result<EncoderTicket, MediaError> {
        match self {
            Self::Local(owner) => owner.submit(job),
            Self::Remote(remote) => remote.submit(job),
            Self::Off | Self::Pending { .. } => Err(media_error("vision encoder unavailable")),
        }
    }
    fn poll(
        &mut self,
        ticket: EncoderTicket,
    ) -> Option<std::result::Result<EncodeOutput, MediaError>> {
        match self {
            Self::Local(owner) => owner.poll(ticket),
            Self::Remote(remote) => remote.poll(ticket),
            Self::Off | Self::Pending { .. } => {
                Some(Err(media_error("vision encoder unavailable")))
            }
        }
    }
    fn cancel(&mut self, ticket: EncoderTicket) {
        match self {
            Self::Local(owner) => owner.cancel(ticket),
            Self::Remote(remote) => remote.cancel(ticket),
            Self::Off | Self::Pending { .. } => (),
        }
    }
}

pub(crate) fn start_worker(
    config: &crate::shared::vision::worker::EncoderWorkerConfig,
    snapshot: &Path,
    library: PathBuf,
    budget: u64,
) -> Result<(EncoderServer, u64)> {
    ensure!(
        config.max_tokens == 1024,
        "V4.1 worker requires 1024 complete image spans"
    );
    let sm = crate::shared::vision::worker::architecture(&library)?;
    ensure!(sm == 121, "V4.1 Spark worker requires SM121, got SM{sm}");
    let catalog =
        cuteafd_loader::read_official_v41_catalog(cuteafd_loader::OFFICIAL_V41_MODEL_ID, snapshot)?;
    let mut encoder =
        LocalEncoder::start(snapshot.to_owned(), library, 0, usize::try_from(budget)?)?;
    let bytes = encoder.bytes as u64;
    let health = encoder.health_handle();
    let server = EncoderServer::start_image_backend(
        config.listen.parse()?,
        handshake(&catalog, &config.revision, sm, config.plan_hash)?,
        Duration::from_secs(60),
        move || health.load(Ordering::Acquire),
        move |job| {
            let id = encoder.submit(job.clone())?;
            loop {
                if let Some(result) = encoder.poll(id) {
                    return result;
                }
                thread::sleep(Duration::from_millis(1));
            }
        },
    )
    .map_err(anyhow::Error::new)?;
    tracing::info!(bytes, address=%server.address, "V4.1 vision encoder ready");
    Ok((server, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires matching CUDA image and official checkpoint; qualifies real owner and wire"]
    fn native_owner_wire_features_and_loss() -> Result<()> {
        let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_VISION_MODEL").context("model")?);
        let library = PathBuf::from(std::env::var_os("CUTEAFD_VISION_LIBRARY").context("library")?);
        let output = PathBuf::from(std::env::var_os("CUTEAFD_VISION_OUTPUT").context("output")?);
        ensure!(!output.exists(), "preserve qualification evidence");
        std::fs::create_dir_all(&output)?;
        let catalog = cuteafd_loader::read_official_v41_catalog(
            cuteafd_loader::OFFICIAL_V41_MODEL_ID,
            &snapshot,
        )?;
        let sm = crate::shared::vision::worker::architecture(&library)?;
        let identity = handshake(
            &catalog,
            "dba1be0a40aa45a94ad051997016db3960a90277",
            sm,
            [7; 32],
        )?;
        let bytes = VisionRuntime::device_bytes(&catalog, CAPACITY)?;
        let budget = bytes
            + if sm == 121 {
                cuteafd_loader::plan::encoder::V41_SPARK_CUDA_OVERHEAD_BYTES as usize
            } else {
                0
            };
        let mut owner = LocalEncoder::start(snapshot.clone(), library.clone(), 0, budget)?;
        let mut cases = Vec::new();
        let mut jobs = Vec::new();
        for (i, (h, w)) in [(5u32, 7u32), (45, 45), (93, 93)].into_iter().enumerate() {
            let patches: Vec<u8> = (0..h as usize * w as usize * 588)
                .flat_map(|j| {
                    let v = ((j * 17 + j / 97 * 29) % 257) as f32 / 128.0 - 1.0;
                    let b = v.to_bits();
                    (((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16).to_le_bytes()
                })
                .collect();
            let job = EncodeJob::image(
                ImageKey([i as u8; 32]),
                [1, h, w],
                patches.into(),
                h.div_ceil(3) as usize * (w.div_ceil(3) as usize + 1) + 2,
                5120,
            );
            jobs.push(job);
        }
        fn wait(client: &mut dyn EncoderClient, job: EncodeJob) -> Result<EncodeOutput> {
            let ticket = client.submit(job)?;
            let deadline = Instant::now() + Duration::from_secs(120);
            loop {
                if let Some(result) = client.poll(ticket) {
                    return Ok(result?);
                }
                ensure!(Instant::now() < deadline, "qualification timeout");
                thread::sleep(Duration::from_millis(1));
            }
        }
        let mut expected = Vec::new();
        for job in &jobs {
            let first = wait(&mut owner, job.clone())?;
            let warm = wait(&mut owner, job.clone())?;
            ensure!(first.features == warm.features, "local repeat differs");
            expected.push(warm);
        }
        let health = owner.health_handle();
        let server = EncoderServer::start_image_backend(
            "127.0.0.1:0".parse()?,
            identity.clone(),
            Duration::from_secs(120),
            move || health.load(Ordering::Acquire),
            move |job| wait(&mut owner, job.clone()).map_err(media_error),
        )?;
        let mut remote =
            RemoteEncoder::connect(vec![server.address], identity, Duration::from_secs(120))?;
        for (i, (job, local)) in jobs.into_iter().zip(expected).enumerate() {
            let _ = wait(&mut remote, job.clone())?;
            let started = Instant::now();
            let actual = wait(&mut remote, job.clone())?;
            let roundtrip_ms = started.elapsed().as_secs_f64() * 1000.0;
            ensure!(
                actual.key == local.key && actual.features == local.features,
                "wire features differ"
            );
            std::fs::write(output.join(format!("case-{i}-span.bin")), &actual.features)?;
            cases.push(
                serde_json::json!({"tokens":job.tokens,"local_ms":local.elapsed_ms,
                "owner_ms":actual.elapsed_ms,"loopback_ms":roundtrip_ms,"byte_exact":true}),
            );
        }
        drop(server);
        let deadline = Instant::now() + Duration::from_secs(5);
        while remote.health_handle().load(Ordering::Acquire) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        ensure!(
            !remote.health_handle().load(Ordering::Acquire),
            "loss not visible"
        );
        std::fs::write(
            output.join("report.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"sm":sm,"resident_bytes":bytes,"cases":cases,"loss_visible":true}),
            )?,
        )?;
        Ok(())
    }

    #[test]
    fn bad_geometry_is_refused_before_owner_submission() {
        let job = |h, w, tokens| {
            EncodeJob::image(
                ImageKey([1; 32]),
                [1, h, w],
                vec![0; h as usize * w as usize * 588 * 2].into(),
                tokens,
                5120,
            )
        };
        assert_eq!(validate_job(&job(93, 93, 994)).unwrap().tokens(), 994);
        assert!(validate_job(&job(96, 96, 1058)).is_err());
        assert!(validate_job(&job(3, 3, 1)).is_err());
        let mut encoder = LocalEncoder {
            queue: None,
            owner: None,
            pending: HashMap::new(),
            next: 0,
            healthy: Arc::new(AtomicBool::new(true)),
            bytes: 0,
        };
        assert!(encoder.submit(job(96, 96, 1058)).is_err());
        assert!(encoder.healthy.load(Ordering::Acquire));
        assert_eq!(encoder.next, 0);
        assert!(encoder.pending.is_empty());
    }
}
