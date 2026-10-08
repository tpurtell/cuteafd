//! GLM 5.x (glm_moe_dsa) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod dflash;
pub(crate) mod dflash_policy;
pub(crate) mod engine;
pub(crate) mod prefix;
pub(crate) mod serve;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::glm5::GlmDsaConfig;
use crate::shared::spark_intake::{SparkLane, SparkLink};
use std::ffi::c_void;
use std::path::PathBuf;
use std::time::Instant;

/// What every GLM command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot (the EXL3 publication carries the dense weights).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Split every layer's attention heads (q_b, kv_b, o_proj; 32 of 64 each)
    /// and the dense / shared-expert MLPs over --device and this second GPU,
    /// with the small latent projection, latent cache and DSA indexer
    /// replicated; the partial sums meet over peer memory. Router, routed
    /// experts, LM head and drafter stay on --device.
    #[arg(long)]
    pub split_device: Option<i32>,
    /// Run only the first N layers (dense layers need no experts).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence; 0 selects checkpoint full, bounded by compiled support and the admitted pool.
    #[arg(long, default_value_t = 0)]
    pub max_context: usize,
    /// Tokens the shared latent/index cache pool holds across sequences; 0
    /// sizes it from what every GPU has left after weights and the planner's
    /// workspace, graph and drafter costs (up to the common 2M-token target).
    #[arg(long, default_value_t = 262_144)]
    pub pool_tokens: usize,
    #[arg(long, default_value_t = 4096)]
    pub prefill_rows: usize,
    /// Spark expert ranks in TP order (HOST:PORT,...), for MoE layers.
    #[arg(long)]
    pub peers: Option<String>,
    /// DFlash2 drafter snapshot (incoai/GLM-5.3-DFlash2); taps the target
    /// layers and drafts on the coordinator GPU.
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Maximum members of one draft batch, independent of context slots.
    #[arg(long, default_value_t = 16)]
    pub draft_sequences: usize,
    /// Context slots (default max(20, draft_sequences)); target head is shared.
    #[arg(long)]
    pub draft_context_slots: Option<usize>,
    /// Explicit calibration-free E4M3 quantization of own drafter GEMMs.
    /// Unset/true: E4M3 single copy (measured faster); false keeps checkpoint BF16.
    #[arg(long, action = clap::ArgAction::Set)]
    pub draft_fp8: Option<bool>,
    /// Scale rule of the FP8 copies made from BF16 weights at load: amax /
    /// 448, the smallest power of two >= it (pow2), or per block whichever of
    /// the two leaves the smaller error (best).
    #[arg(long, value_enum, default_value_t = crate::shared::fp8_linear::Fp8Scales::Amax)]
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    #[command(flatten)]
    pub l2: crate::shared::l2_prefetch::L2PrefetchArgs,
    /// Keep every prefill row's logits (glm-golden --nll; 2.5 GiB at 4096 rows).
    #[arg(long, hide = true)]
    pub full_prefill_logits: bool,
    /// Benchmarks only: MoE layers skip the Spark exchange (routed experts
    /// contribute zero) but keep the route/wire download, the host sync, the
    /// shared expert, the plane uploads and the reduce, timing the
    /// coordinator alone. Outputs are not the model's.
    #[arg(long, hide = true)]
    pub skip_routed_experts: bool,
    /// Prefill projections over the checkpoint's FP8 weights run W8A16 (the
    /// weights widened to BF16 in-kernel: the former BF16 prefill bit for bit)
    /// instead of W8A8 (E4M3 activations per row and 128-K block with FP32
    /// scales, the official FP8 release's served numerics).
    #[arg(long)]
    pub prefill_w8a16: bool,
    #[command(flatten)]
    pub token_io: crate::shared::token_io::TokenIoArgs,
}

#[cfg(test)]
mod draft_cli_tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Parse {
        #[command(flatten)]
        engine: EngineArgs,
    }

    #[test]
    fn drafter_defaults_are_checkpoint_bf16_with_independent_capacities() {
        let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"])
            .unwrap().engine;
        assert_eq!(parsed.draft_fp8, None);
        assert_eq!(parsed.draft_sequences, 16);
        assert_eq!(parsed.draft_context_slots.unwrap_or(20.max(parsed.draft_sequences)), 20);
    }

    #[test]
    fn explicit_fp8_option_and_context_batch_limits_are_forwarded() {
        for (value, expected) in [("true", true), ("false", false)] {
            let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--draft-fp8", value, "--draft-context-slots", "20", "--draft-sequences", "16"])
                .unwrap().engine;
            assert_eq!(parsed.draft_fp8, Some(expected));
            assert_eq!((parsed.draft_context_slots, parsed.draft_sequences), (Some(20), 16));
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--draft-fp8", "auto"]).is_err());
    }
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/families/glm5/golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Prefill only the first N tokens, then feed the rest through decode
    /// steps of --step-rows rows (teacher-forced), comparing their rows.
    #[arg(long)]
    pub prefill: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub step_rows: usize,
    /// Compare only logits during decode (decode then runs its captured graphs).
    #[arg(long)]
    pub no_layer_compare: bool,
    /// With --draft: after the prefill, decode N greedy tokens one row per
    /// step, drafting before each, and report how many drafts the target
    /// reproduced (without it, drafts are scored against tokens.bin).
    #[arg(long)]
    pub generate: Option<usize>,
    /// With --draft: run only the drafter on the golden taps and compare with
    /// python/reference/families/glm5/dflash2/reference.py's output directory.
    #[arg(long)]
    pub draft_oracle: Option<PathBuf>,
    /// With --draft: replay the drafter alone on the golden taps, drafting
    /// after every token from this position on, BF16 and FP8 (acceptance
    /// against the text and the golden greedy picks, draft time).
    #[arg(long)]
    pub draft_replay: Option<usize>,
    /// Teacher-force this many copies of the sequence together (prefilled to
    /// staggered lengths), --step-rows rows each per step, and compare every
    /// copy's logits with the golden logits.
    #[arg(long)]
    pub sequences: Option<usize>,
    /// Score every prefill row's logits against the golden (mean NLL, top-1,
    /// KL); the prefill runs without layer downloads (Spark lanes).
    #[arg(long)]
    pub nll: bool,
    /// With --nll: also write tokens.bin and every row's logits.bin (F32) to
    /// this directory, a golden for A/B runs between builds or numerics (e.g.
    /// W8A8 against --prefill-w8a16 with --skip-routed-experts).
    #[arg(long, hide = true)]
    pub save_logits: Option<PathBuf>,
    /// Time this many prefills of --bench-prefill-tokens tokens (the golden
    /// prompt repeated, in chunks of the prefill capacity) on fresh sequences.
    #[arg(long, default_value_t = 0)]
    pub bench_prefill: usize,
    #[arg(long, default_value_t = 4096)]
    pub bench_prefill_tokens: usize,
    /// Time teacher-forced verify steps of 1, 2, 4, ... up to N rows of one
    /// sequence after --prefill tokens (default 512) of the golden prompt
    /// (median of 7 after 2 warm-ups; the sequence rewinds between steps),
    /// and with --draft the drafter step for 1, 2, 4, ... sequences.
    #[arg(long)]
    pub bench_verify: Option<usize>,
    /// With --bench-verify: time only these row counts (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub bench_rows: Vec<usize>,
    /// With --bench-verify and --draft: time only these sequence counts.
    #[arg(long, value_delimiter = ',')]
    pub bench_sequences: Vec<usize>,
    /// With --bench-verify: write each row count's last logits (F32) and,
    /// with --draft, each sequence count's drafts to this directory (A/B
    /// numerics checks between builds).
    #[arg(long)]
    pub bench_dump: Option<PathBuf>,
    /// With --bench-verify: prefill this many tokens (the golden prompt
    /// repeated) instead of --prefill, for long-context step costs.
    #[arg(long)]
    pub bench_context: Option<usize>,
    /// Prefix-cache restore check at token P: prefill the first P tokens (of --prefill, default
    /// all), fork their pages into a second sequence (shared full pages, the copied tail page),
    /// continue both (prefill in --prefill-chunk rows, then --resume-decode greedy steps) and
    /// compare every layer's rows, the logits and the paged rows byte for byte.
    #[arg(long)]
    pub resume_at: Option<usize>,
    /// Prefill chunk rows of --resume-at (default the engine's prefill rows).
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    #[arg(long, default_value_t = 4)]
    pub resume_decode: usize,
    /// With --resume-at: prefill the second sequence cold on its own pages instead of
    /// restoring it (the floor: what the kernels themselves vary).
    #[arg(long, hide = true)]
    pub resume_cold: bool,
    /// --resume-at attempts on fresh sequences (all must be byte-identical).
    #[arg(long, default_value_t = 1)]
    pub resume_repeat: usize,
    /// Token I/O gate after the golden prompt (--prefill tokens of it), then
    /// stop: the resident embedding table against the shard, device against
    /// host greedy selection over this many decode steps, device against host sampling.
    #[arg(long)]
    pub token_check: Option<usize>,
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub snapshot: PathBuf,
    pub catalog: cuteafd_loader::OfficialV41Catalog,
    pub cfg: GlmDsaConfig,
    pub library: NativeLibrary,
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    cuteafd_core::set_expert_geometry(catalog.routed_experts().geometry()?)
        .map_err(|g| anyhow::anyhow!("geometry already {g:?}"))?;
    let cfg = GlmDsaConfig::read(&args.snapshot)?;
    if let Some(snapshot) = &args.draft {
        dflash::check_snapshot_target_bf16_head(&args.snapshot,
            cfg.hidden, cfg.vocab_size, false)?;
        dflash::check_checkpoint(snapshot, args.draft_fp8, args.draft_context_slots, args.draft_sequences)?;
    }
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    Ok(Opened { snapshot: args.snapshot.clone(), catalog, cfg, library })
}

impl Opened {
    /// Builds the engine (and the Spark transport when peers are given) and
    /// hands them to `body`.
    pub fn with_engine<T>(&self, args: &EngineArgs,
        body: impl FnOnce(&engine::GlmEngine<'_>, Option<&mut SparkLink<'_>>, &tokio::runtime::Runtime) -> Result<T>)
        -> Result<T> {
        let automatic_context = args.max_context == 0;
        let mut context_args = args.clone();
        context_args.max_context = crate::shared::context::checkpoint_context(
            &args.snapshot, &args.manifest, "glm5", args.max_context)?;
        let args = &context_args;
        let programs = self.library.programs()?.with_manifest(&args.manifest)?;
        programs.capacities().require_context("glm5", args.max_context)?;
        programs.load_all()?;
        let stream = self.library.cuda_stream_create()?;
        // The head split's second GPU and its stream (load kernels, then the engine's).
        // A head split needs its share's programs (`glm2`) in this build.
        let split_device = match args.split_device {
            Some(device) if programs.spec("glm2_o_m64").is_ok() => Some(device),
            Some(device) => {
                tracing::info!(device, "no head-split programs (glm2) in this build; serving from --device alone");
                None
            }
            None => None,
        };
        let split_device = crate::shared::peer_split::probed_device(&self.library, args.device, split_device)?;
        let peer_stream = match split_device {
            Some(device) => {
                ensure!(device != args.device, "--split-device must differ from --device");
                self.library.cuda_enable_peer(device)?;
                self.library.cuda_set_device(device)?;
                let stream = self.library.cuda_enable_peer(args.device).and_then(|()| self.library.cuda_stream_create());
                self.library.cuda_set_device(args.device)?;
                Some((device, stream?))
            }
            None => None,
        };
        let draft_file = args.draft.as_deref().map(dflash::prefetch);
        let started = Instant::now();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let loader = weights::GlmLoader { library: &self.library, catalog: &self.catalog, stream, device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect() };
        let (embedding, (model, mut shares)) = crate::shared::token_io::TokenEmbedding::load(&self.library, self.embed_source()?,
            args.token_io.embed_placement, || { let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights"); loader.model(&self.cfg, layers) })?;
        let bytes: usize = model.layers.iter().map(|l| l.bytes()).sum::<usize>() + model.norm.buffer.bytes
            + model.head.buffer.bytes;
        let peer_bytes: usize = shares.iter().flatten().map(|l| l.bytes()).sum();
        tracing::info!(layers, elapsed_ms = started.elapsed().as_millis() as u64,
            gib = format!("{:.2}", bytes as f64 / (1u64 << 30) as f64),
            split_gib = format!("{:.2}", peer_bytes as f64 / (1u64 << 30) as f64), "GLM coordinator weights resident");
        let pool_tokens = if args.full_prefill_logits || args.pool_tokens == 0 || cuteafd_ffi::coordinator_gpu_budget().is_some() {
            // The planner's GLM costs stay free on each GPU; records fill the rest.
            let devices: Vec<i32> = std::iter::once(args.device).chain(peer_stream.map(|(d, _)| d)).collect();
            // GLM 5.3's pages are its whole state: no recurrent slots, no mark arena.
            crate::shared::memory_report::planned_pool_tokens_with_reserves(&self.library, &args.snapshot, &devices,
                args.draft.as_deref(), args.prefill_rows, 0, 0,
                (args.pool_tokens > 0).then_some(args.pool_tokens as u64), 0,
                &crate::shared::memory_report::lead_reserves(devices.len(),
                    if args.full_prefill_logits { cuteafd_loader::plan::layout::full_prefill_logits_bytes_with_lanes(
                        "glm5", args.prefill_rows as u64, self.cfg.vocab_size as u64,
                        if args.peers.is_some() { engine::configured_lanes() } else { 1 }) } else { 0 }),
                Default::default(), 4, cuteafd_loader::serving_capacity::GLMF_DECODE_ROWS)?
        } else {
            args.pool_tokens
        };
        let max_context = crate::shared::context::pool_context("glm5", args.max_context, automatic_context, pool_tokens, 256)?;
        let pages = pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::GlmEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            max_context, args.prefill_rows, pages, embedding)?;
        engine.full_prefill_logits = args.full_prefill_logits;
        if let Some((device, stream)) = peer_stream {
            engine.attach_peer(device, stream, shares.pop().context("head-split shares")?)?;
        }
        engine.prefill_w8a8 = !args.prefill_w8a16;
        if args.skip_routed_experts {
            engine.skip_routed_experts()?;
        }
        if let Some(snapshot) = &args.draft {
            let started = Instant::now();
            let cfg = dflash::DflashConfig::read(snapshot)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "the DFlash2 drafter does not fit this target");
            let mask = engine.embedding.host_rows(&[cfg.mask_token])?;
            let file = draft_file.context("drafter prefetch")?.join()
                .map_err(|_| anyhow::anyhow!("drafter prefetch panicked"))??;
            let representation = cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation
                ::from_fp8_option(args.draft_fp8);
            let drafter = dflash::GlmDrafter::load(&self.library, snapshot, file, stream,
                args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences,
                mask, false, representation, args.fp8_scales)?;
            engine.drafter = Some(drafter);
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "DFlash2 drafter resident");
        }
        let ranks = |peers: &str| -> Result<(Vec<std::net::SocketAddr>, Vec<u64>)> {
            let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let executors: Vec<u64> = (0..peers.len())
                .map(|rank| cuteafd_transport::expert::v41_spark_executor_id(peers.len(), rank))
                .collect::<Result<_>>()?;
            Ok((peers, executors))
        };
        let config = cuteafd_transport::TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 };
        let row_bytes = self.cfg.hidden * 2;
        crate::shared::memory_report::release_load_staging(&self.library);
        let mut transport = args.peers.as_deref().map(|peers| -> Result<SparkLink<'_>> {
            let (peers, executors) = ranks(peers)?;
            SparkLink::new(&self.library, &peers, &executors, 4096, config.clone(), row_bytes)
        }).transpose()?;
        // One transport thread per prefill lane (their waves fly beside each other's).
        let mut lanes = match args.peers.as_deref() {
            Some(peers) => (0..engine::configured_lanes()).filter(|_| engine::configured_lanes() > 1).map(|_| {
                let (peers, executors) = ranks(peers)?;
                SparkLane::new(&self.library, peers, executors, 4096, config.clone(), row_bytes)
            }).collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        // Connect every rank and register full-size buffers now: the first
        // request otherwise pays seconds of connection setup.
        let (rows, h, topk, experts, layer) =
            (args.prefill_rows, self.cfg.hidden, self.cfg.topk, self.cfg.experts, self.cfg.first_moe_layer as u32);
        let warm = move || -> Result<cuteafd_transport::ExpertProtocolV2Request> {
            let routes = (0..rows * topk).map(|i| cuteafd_transport::ExpertProtocolV2RouteEntry {
                row_index: (i / topk) as u32, expert_id: (i % experts) as u32, gate_weight: 0.0,
            }).collect();
            let mut request = cuteafd_transport::ExpertProtocolV2Request::new(1, 17, layer,
                h as u32, cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                (0..rows as u32).map(|row| cuteafd_transport::ExpertProtocolV2RowDescriptor {
                    row_id: u64::from(row), source_kind: cuteafd_transport::ExpertV2SourceKind::Prefill,
                    source_request_id: 1, token_position: u64::from(row), route_offset: row * topk as u32,
                    route_count: topk as u32,
                }).collect(),
                routes, vec![0; rows * (h + h / 32)])?;
            request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            Ok(request)
        };
        let started = Instant::now();
        if let Some(transport) = transport.as_mut() {
            let request = warm()?;
            runtime.block_on(async {
                let wave = transport.dispatch(&request)?;
                transport.receive(wave, rows, stream).await
            })?;
        }
        for lane in &mut lanes {
            lane.submit(rows, Box::new(warm))?;
        }
        for lane in &mut lanes {
            lane.wait(rows, std::time::Duration::from_secs(120), stream)?;
        }
        // SAFETY: the engine's stream; the warm waves' intake waits drain here.
        unsafe { self.library.cuda_stream_synchronize(stream)? };
        if transport.is_some() {
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, lanes = lanes.len(),
                "Spark expert transports warm");
        }
        *engine.lanes.borrow_mut() = lanes;
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::GLM_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
            if engine.ranks() > 1 {
                engine.attach_peer_l2(budget)?;
            }
        }
        // Every decode graph shape serving replays, captured now (CUTEAFD_GLM_WARM_GRAPHS=0 skips it).
        if engine.weights.layers.len() == self.cfg.layers
            && std::env::var("CUTEAFD_GLM_WARM_GRAPHS").map_or(true, |v| v != "0")
            && (transport.is_some() || args.skip_routed_experts) {
            engine.warm_decode_graphs(transport.as_mut().map(|t| (t, &runtime)))?;
        }
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        let result = body(&engine, transport.as_mut(), &runtime);
        drop(engine);
        drop(transport);
        // SAFETY: the engine that used the streams is gone.
        unsafe {
            self.library.cuda_stream_destroy(stream)?;
            if let Some((device, stream)) = peer_stream {
                self.library.cuda_set_device(device)?;
                let destroyed = self.library.cuda_stream_destroy(stream);
                self.library.cuda_set_device(args.device)?;
                destroyed?;
            }
        }
        result
    }
}

/// Vocabulary head of `rows` rows: 2..=16 rows on the few-row FP32 kernel
/// (one read of the head per 8 rows), others on the pedantic cuBLAS head;
/// CUTEAFD_GLM_HEAD=cublas keeps every row count on cuBLAS.
///
/// # Safety
/// `x` [rows, width] BF16, `weight` [vocab, width] BF16 and `logits` [rows,
/// vocab] FP32 are live device buffers on `stream`'s device; `head` was
/// created for at least `rows` rows of this width and vocabulary.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_head(library: &NativeLibrary, head: &cuteafd_ffi::programs::VocabularyHead<'_>,
    x: *const c_void, weight: *const c_void, logits: *mut f32, rows: usize, width: usize, vocab: usize,
    stream: *mut c_void) -> Result<()> {
    static CUBLAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let cublas = *CUBLAS.get_or_init(|| std::env::var("CUTEAFD_GLM_HEAD").is_ok_and(|v| v == "cublas"));
    if !cublas && (2..=cuteafd_ffi::VOCAB_HEAD_ROWS_MAX).contains(&rows) {
        // SAFETY: the caller's contract.
        return unsafe { library.vocab_head_rows(x, weight, logits, rows, width, vocab, stream) };
    }
    // SAFETY: the caller's contract.
    unsafe { head.launch(x.cast(), weight.cast(), logits, rows as u32, stream) }
}

impl Opened {
    /// The checkpoint's `model.embed_tokens.weight` (BF16 [vocab, hidden]).
    fn embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let tensor = self.catalog.tensor("model.embed_tokens.weight")?;
        crate::shared::token_io::EmbedSource::new(self.catalog.snapshot(), &tensor.shard, &tensor.metadata,
            self.cfg.hidden)
    }
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn similarity(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE), diff.sqrt() / nb.sqrt().max(f64::MIN_POSITIVE))
}

pub(crate) async fn run_golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.full_prefill_logits |= args.nll || args.resume_at.is_some();
    tokio::task::spawn_blocking(move || golden(args)).await?
}

/// Mean KL(golden || engine) of logits rows, in float64.
fn mean_kl(logits: &[f32], golden: &[f32], vocab: usize) -> f64 {
    let log_softmax = |l: &[f32]| -> Vec<f64> {
        let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln();
        l.iter().map(|&x| x as f64 - lse).collect()
    };
    let rows = logits.len() / vocab;
    (0..rows).map(|r| {
        let (p, q) = (log_softmax(&golden[r * vocab..][..vocab]), log_softmax(&logits[r * vocab..][..vocab]));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum::<f64>()
    }).sum::<f64>() / rows.max(1) as f64
}

/// --nll: the golden prompt through prefill chunks of the engine's capacity
/// (Spark lanes when available), every row's logits scored against the
/// golden: mean NLL of the next token, top-1 agreement, KL.
fn nll_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let vocab = opened.cfg.vocab_size;
    let mut placement = engine::PageAllocator::new(engine.pages).admit(tokens.len())?;
    let started = Instant::now();
    let mut logits = Vec::new();
    for chunk in tokens.chunks(engine.prefill_capacity()) {
        logits.extend(engine.prefill_rows_logits(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)),
            None, chunk.len())?.context("the prefill needs every layer")?);
    }
    let seconds = started.elapsed().as_secs_f64();
    if let Some(dir) = &args.save_logits {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("tokens.bin"), tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())?;
        std::fs::write(dir.join("logits.bin"), logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let nll_of = |l: &[f32], next: u32| {
        let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln() - l[next as usize] as f64
    };
    let Ok(golden) = std::fs::read(args.golden.join("logits.bin")) else {
        // No golden logits (a token file alone): the engine's NLL of the text and a digest of
        // every logit (bitwise A/B between builds and runs).
        use std::hash::{Hash, Hasher};
        let mut digest = std::collections::hash_map::DefaultHasher::new();
        logits.iter().for_each(|v| v.to_bits().hash(&mut digest));
        let nll: f64 = (0..tokens.len() - 1).map(|r| nll_of(&logits[r * vocab..][..vocab], tokens[r + 1])).sum();
        println!("prefill logits: {} tokens in {seconds:.2} s | mean NLL engine {:.4} | logits digest {:016x}",
            tokens.len(), nll / (tokens.len() - 1) as f64, digest.finish());
        return Ok(());
    };
    let golden: Vec<f32> = golden.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let rows = tokens.len();
    let (mut agree, mut nll, mut golden_nll) = (0usize, 0f64, 0f64);
    for r in 0..rows {
        let (ours, theirs) = (&logits[r * vocab..][..vocab], &golden[r * vocab..][..vocab]);
        agree += usize::from(argmax(ours) == argmax(theirs));
        if let Some(&next) = tokens.get(r + 1) {
            nll += nll_of(ours, next);
            golden_nll += nll_of(theirs, next);
        }
    }
    println!("prefill logits: {rows} tokens in {seconds:.2} s | top-1 agreement {:.2}% | mean NLL engine {:.4} golden \
        {:.4} | mean KL(golden||engine) {:.5}", 100.0 * agree as f64 / rows as f64, nll / (rows - 1) as f64,
        golden_nll / (rows - 1) as f64, mean_kl(&logits, &golden[..logits.len()], vocab));
    Ok(())
}

/// --bench-verify: verify-step cost by rows (the DFlash2 policy's step
/// table) with the host's phase split, then the drafter step by sequences.
fn bench_verify(args: &GoldenArgs, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime, max_rows: usize) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(max_rows >= 1 && max_rows <= engine::DECODE_ROWS, "--bench-verify takes 1..={} rows", engine::DECODE_ROWS);
    ensure!(transport.is_some() || engine.skip_routed(), "--bench-verify needs Spark peers or --skip-routed-experts");
    let prefill = args.bench_context.or(args.prefill).unwrap_or(512);
    let tokens: Vec<u32> = tokens.iter().copied().cycle().take(prefill + max_rows).collect();
    let mut placement = engine::PageAllocator::new(engine.pages).admit(prefill + max_rows)?;
    for chunk in tokens[..prefill].chunks(engine.prefill_capacity()) {
        engine.prefill(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), None)?;
    }
    let start = placement.len;
    println!("verify cost after {prefill} tokens (teacher-forced, one sequence, median of 7; host phases per step: \
        GPU wait before each exchange, exchange, logits download){}",
        if engine.skip_routed() { " [routed experts skipped]" } else { "" });
    let mut counts: Vec<usize> = (0..).map(|i| 1usize << i).take_while(|&r| r < max_rows).chain([max_rows]).collect();
    if !args.bench_rows.is_empty() {
        ensure!(args.bench_rows.iter().all(|&r| r >= 1 && r <= max_rows), "--bench-rows past --bench-verify");
        counts = args.bench_rows.clone();
    }
    for rows in counts {
        let mut times = Vec::new();
        for round in 0..9 {
            placement.len = start;
            if round == 2 {
                *engine.profile.borrow_mut() = [0.0; 3];
            }
            let started = Instant::now();
            let logits = engine.verify(&mut [(&mut placement, rows)], &tokens[start..start + rows],
                transport.as_deref_mut().map(|t| (t, runtime)), None)?;
            if round >= 2 {
                times.push(started.elapsed().as_secs_f64() * 1e3);
            }
            if let (Some(dir), Some(logits), 8) = (&args.bench_dump, logits, round) {
                std::fs::create_dir_all(dir)?;
                std::fs::write(dir.join(format!("logits-{rows}.bin")),
                    logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        let n = times.len() as f64;
        times.sort_by(f64::total_cmp);
        println!("  {rows} rows: {:.2} ms (min {:.2}) | gpu wait {:.2} exchange {:.2} logits {:.2} ms", times[times.len() / 2],
            times[0], 1e3 * phases[0] / n, 1e3 * phases[1] / n, 1e3 * phases[2] / n);
    }
    if let Some(drafter) = engine.drafter.as_ref() {
        // The drafter reads its ring context of the prefilled taps; anchors at the prefill's end.
        let n = start.min(dflash::TAP_ROWS);
        drafter.update(&(0..n).map(|r| dflash::ContextRow { tap_row: r, slot: 0, position: start - n + r })
            .collect::<Vec<_>>())?;
        println!("drafter step (anchors at position {start}, median of 7):");
        let max_batch = dflash::ReplayDrafter::sequences(drafter);
        let mut counts: Vec<usize> = (0..).map(|i| 1usize << i).take_while(|&s| s < max_batch)
            .chain([max_batch]).collect();
        if !args.bench_sequences.is_empty() {
            ensure!(args.bench_sequences.iter().all(|&s| s >= 1 && s <= max_batch), "--bench-sequences past the draft batch capacity");
            counts = args.bench_sequences.clone();
        }
        for sequences in counts {
            let seqs: Vec<dflash::DraftSeq> = (0..sequences)
                .map(|slot| dflash::DraftSeq { slot, anchor: tokens[start], position: start, valid_from: 0 }).collect();
            let mut times = Vec::new();
            for round in 0..9 {
                let started = Instant::now();
                let drafts = drafter.draft_device(&seqs, &engine.embedding,
                    dflash::TargetHead::Bf16(&engine.weights.head))?;
                if round >= 2 {
                    times.push(started.elapsed().as_secs_f64() * 1e3);
                }
                if let (Some(dir), 8) = (&args.bench_dump, round) {
                    std::fs::write(dir.join(format!("drafts-{sequences}.txt")),
                        drafts.iter().map(|d| format!("{:?}\n", d.tokens)).collect::<String>())?;
                }
            }
            times.sort_by(f64::total_cmp);
            println!("  {sequences} sequences: {:.2} ms (min {:.2})", times[times.len() / 2], times[0]);
        }
    }
    Ok(())
}

/// --bench-prefill: fresh sequences of --bench-prefill-tokens tokens.
fn bench_prefill(args: &GoldenArgs, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let n = args.bench_prefill_tokens;
    let long: Vec<u32> = tokens.iter().copied().cycle().take(n).collect();
    let mut allocator = engine::PageAllocator::new(engine.pages);
    let mut times = Vec::new();
    for round in 0..=args.bench_prefill {
        if round == 1 {
            *engine.profile.borrow_mut() = [0.0; 3];
        }
        let mut placement = allocator.admit(n)?;
        let started = Instant::now();
        for chunk in long.chunks(engine.prefill_capacity()) {
            engine.prefill(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), None)?;
        }
        // Round 0 warms the workspaces.
        if round > 0 {
            times.push(started.elapsed().as_secs_f64());
        }
        allocator.release(placement);
    }
    times.sort_by(f64::total_cmp);
    let (median, runs) = (times[times.len() / 2], times.len() as f64);
    let phases = *engine.profile.borrow();
    let host = *engine.exchange_host.borrow();
    println!("prefill bench host per prefill: request build + post {:.1} ms, partial copies {:.1} ms (measured runs \
        and warm-up; build {:.1} ms, post {:.1} ms)", 1e3 * host[0] / (runs + 1.0), 1e3 * host[1] / (runs + 1.0),
        1e3 * host[2] / (runs + 1.0), 1e3 * host[3] / (runs + 1.0));
    println!("prefill bench: {n} tokens, median {:.1} ms ({:.0} tok/s), min {:.1} ms; per prefill GPU wait {:.1} ms, \
        Spark wait {:.1} ms; runs {:?} ms", 1e3 * median, n as f64 / median, 1e3 * times[0], 1e3 * phases[0] / runs,
        1e3 * phases[1] / runs, times.iter().map(|t| (t * 1e3).round() as u32).collect::<Vec<_>>());
    Ok(())
}

fn golden(args: GoldenArgs) -> Result<()> {
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine, transport, runtime| golden_run(&args, &opened, engine, transport, runtime))
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    if let Some(dir) = &args.draft_oracle {
        return draft_oracle(args, opened, engine, dir);
    }
    if let Some(start) = args.draft_replay {
        let drafter = engine.drafter.as_ref().context("--draft-replay needs --draft")?;
        let (tokens, greedy) = dflash::golden_sequence(&args.golden, opened.cfg.vocab_size)?;
        let hidden = opened.cfg.hidden;
        let layers: Vec<Vec<u8>> = drafter.cfg.taps.iter()
            .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
        let row = hidden * 2;
        let taps = |first: usize, n: usize| -> Result<Vec<u8>> {
            let mut taps = vec![0u8; n * layers.len() * row];
            for r in 0..n {
                for (i, layer) in layers.iter().enumerate() {
                    taps[(r * layers.len() + i) * row..][..row].copy_from_slice(&layer[(first + r) * row..][..row]);
                }
            }
            Ok(taps)
        };
        return dflash::replay(drafter, &tokens, &greedy, &taps, &|t| engine.embedding.host_rows(t),
            engine.weights.head.buffer.ptr, start);
    }
    if let Some(rows) = args.bench_verify {
        return bench_verify(args, engine, transport, runtime, rows);
    }
    if let Some(at) = args.resume_at {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return prefix::resume_check(engine, &tokens, at, n,
            args.prefill_chunk.unwrap_or(engine.prefill_rows), args.resume_decode, args.resume_cold,
            args.resume_repeat, transport.map(|t| (t, runtime)));
    }
    if let Some(steps) = args.token_check {
        return token_check(args, opened, engine, transport, runtime, steps);
    }
    if engine.drafter.is_some() {
        return draft_run(args, opened, engine, transport, runtime);
    }
    if args.nll || args.bench_prefill > 0 {
        ensure!(transport.is_some() || engine.skip_routed(),
            "--nll and --bench-prefill need Spark peers or --skip-routed-experts");
        if args.nll {
            nll_run(args, opened, engine, transport.as_deref_mut(), runtime)?;
        }
        if args.bench_prefill > 0 {
            bench_prefill(args, engine, transport, runtime)?;
        }
        return Ok(());
    }
    if let Some(copies) = args.sequences {
        return multi_run(args, opened, engine, copies, transport, runtime);
    }
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut placement = engine::PageAllocator::new(engine.pages).admit(tokens.len())?;
    let row = cfg.hidden * 2;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    let started = Instant::now();
    // Rows [first, first + n) of layer `layer`'s golden output.
    let compare = |layer: usize, first: usize, stream: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        if let Ok(golden) = std::fs::read(&path) {
            ensure!(golden.len() >= (first * row + stream.len()), "golden layer {layer} is short");
            let golden = &golden[first * row..][..stream.len()];
            let (cosine, rel) = similarity(&bf16s(stream), &bf16s(golden));
            if worst.len() <= layer {
                worst.resize(layer + 1, 1.0);
            }
            worst[layer] = worst[layer].min(cosine);
            if first == 0 {
                println!("layer {layer:2}: cosine {cosine:.6} rel_l2 {rel:.3e}");
            }
        }
        Ok(())
    };
    let mut worst = Vec::new();
    let logits = engine.prefill(&mut placement, &tokens[..prefill], transport.as_deref_mut().map(|t| (t, runtime)),
        Some(&mut |layer, stream| compare(layer, 0, stream, &mut worst)))?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let mut layer_compare = |layer: usize, stream: &[u8]| compare(layer, first, stream, &mut decode_worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.no_layer_compare { None } else { Some(&mut layer_compare) };
        if let Some(logits) = engine.verify(&mut [(&mut placement, n)], &tokens[position..position + n],
            transport.as_deref_mut().map(|t| (t, runtime)), on_layer)? {
            decode_logits.extend(logits);
        }
        position += n;
    }
    if prefill < tokens.len() {
        println!("decode: {} rows in steps of {} in {:.2} s; worst row-block cosine per layer {:?}", tokens.len() - prefill,
            args.step_rows, started.elapsed().as_secs_f64(),
            decode_worst.iter().map(|c| format!("{c:.6}")).collect::<Vec<_>>());
    }
    if !decode_logits.is_empty() {
        let golden: Vec<f32> = std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let vocab = cfg.vocab_size;
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        let (mut agree, mut next_ok, mut golden_next) = (0usize, 0usize, 0usize);
        let rows = decode_logits.len() / vocab;
        for r in 0..rows {
            let (ours, theirs) = (&decode_logits[r * vocab..][..vocab], &golden[(prefill + r) * vocab..][..vocab]);
            agree += usize::from(argmax(ours) == argmax(theirs));
            if let Some(&next) = tokens.get(prefill + r + 1) {
                next_ok += usize::from(argmax(ours) == next as usize);
                golden_next += usize::from(argmax(theirs) == next as usize);
            }
        }
        println!("decode logits: top-1 agreement {:.1}% over {rows} rows | next-token accuracy engine {:.1}% golden {:.1}%",
            100.0 * agree as f64 / rows as f64, 100.0 * next_ok as f64 / (rows - 1).max(1) as f64,
            100.0 * golden_next as f64 / (rows - 1).max(1) as f64);
    }
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s");
    if let Some(logits) = logits {
        let golden: Vec<f32> = std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let last = &golden[(prefill - 1) * cfg.vocab_size..][..cfg.vocab_size];
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).context("empty");
        let (cosine, _) = similarity(&logits, last);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(&logits)?, argmax(last)?);
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// Prefills the golden prompt, then decodes one row per step (teacher-forced
/// on tokens.bin, or greedy with --generate), drafting with DFlash2 before
/// every step; reports the accepted prefix per step against the sequence.
fn draft_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft")?;
    let mut sequence: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let prefill = args.prefill.unwrap_or(sequence.len() / 2).min(sequence.len() - 1);
    let end = match args.generate {
        Some(n) => {
            sequence.truncate(prefill);
            prefill + n
        }
        None => sequence.len(),
    };
    let mut placement = engine::PageAllocator::new(engine.pages).admit(end + 1)?;
    let started = Instant::now();
    let logits = engine.prefill(&mut placement, &sequence[..prefill],
        transport.as_deref_mut().map(|t| (t, runtime)), None)?.context("drafting needs every target layer")?;
    let n = prefill.min(dflash::TAP_ROWS);
    drafter.update(&(0..n).map(|r| dflash::ContextRow { tap_row: r, slot: 0, position: prefill - n + r })
        .collect::<Vec<_>>())?;
    if args.generate.is_some() {
        sequence.push(argmax(&logits));
    }
    println!("prefill: {prefill} tokens in {:.2} s", started.elapsed().as_secs_f64());
    let (mut drafts, mut draft_seconds) = (Vec::new(), 0f64);
    let started = Instant::now();
    for position in prefill..end {
        let anchor = sequence[position];
        let timer = Instant::now();
        let draft = drafter.draft_device(&[dflash::DraftSeq { slot: 0, anchor, position, valid_from: 0 }],
            &engine.embedding, dflash::TargetHead::Bf16(&engine.weights.head))?;
        draft_seconds += timer.elapsed().as_secs_f64();
        drafts.push((position, draft.into_iter().next().context("draft")?));
        let logits = engine.verify(&mut [(&mut placement, 1)], &[anchor], transport.as_deref_mut().map(|t| (t, runtime)),
            None)?.context("decode needs every layer")?;
        drafter.update(&[dflash::ContextRow { tap_row: 0, slot: 0, position }])?;
        if args.generate.is_some() {
            sequence.push(argmax(&logits));
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let block = drafter.cfg.drafts();
    let mut histogram = vec![0usize; block + 1];
    let mut first_rank = [0usize; 16];
    for (position, draft) in &drafts {
        let truth = &sequence[position + 1..];
        if truth.len() < block {
            continue;
        }
        let accepted = draft.tokens.iter().zip(truth).take_while(|(d, t)| d == t).count();
        histogram[accepted] += 1;
        first_rank[draft.features[0][3] as usize] += 1;
    }
    let steps: usize = histogram.iter().sum();
    let accepted: usize = histogram.iter().enumerate().map(|(i, c)| i * c).sum();
    println!("drafts: {steps} steps, {accepted} drafted tokens accepted as a prefix ({:.2} per step, {:.1}% of {}), \
        histogram {histogram:?}, first-draft selector rank {first_rank:?}, {:.2} ms/draft, {:.1} ms/step",
        accepted as f64 / steps.max(1) as f64, 100.0 * accepted as f64 / (steps * block).max(1) as f64, steps * block,
        draft_seconds * 1e3 / drafts.len().max(1) as f64, seconds * 1e3 / drafts.len().max(1) as f64);
    if args.generate.is_some() {
        let text = cuteafd_loader::LoadedTokenizer::from_snapshot(&opened.snapshot)?
            .decode_ids(&sequence[prefill..], false).map(|d| d.text).unwrap_or_default();
        println!("generated: {text:?}");
    }
    Ok(())
}

/// Runs the drafter alone on the golden taps at reference.py's anchor
/// positions and compares tokens, selector features and final-norm rows.
fn draft_oracle(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, dir: &std::path::Path) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-oracle needs --draft")?;
    let (hidden, block, drafts_per) = (opened.cfg.hidden, drafter.cfg.block, drafter.cfg.drafts());
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let positions: Vec<usize> = meta["positions"].as_array().context("positions")?.iter()
        .map(|p| p.as_u64().map(|p| p as usize).context("position")).collect::<Result<_>>()?;
    let words = |name: &str| -> Result<Vec<u32>> {
        Ok(std::fs::read(dir.join(name))?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let (ref_tokens, ref_features) = (words("drafts.bin")?, words("features.bin")?);
    let ref_hidden = bf16s(&std::fs::read(dir.join("hidden.bin"))?);
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let layers: Vec<Vec<u8>> = drafter.cfg.taps.iter()
        .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
    let row = hidden * 2;
    let width = layers.len() * row;
    let (mut done, mut exact, mut first, mut matched, mut worst, mut feature_error) = (0usize, 0, 0, 0, 1f64, 0f64);
    let mut draft_seconds = 0f64;
    for (index, &position) in positions.iter().enumerate() {
        // Context rows done..position through the tap buffer.
        while done < position {
            let n = (position - done).min(dflash::TAP_ROWS);
            let mut taps = vec![0u8; n * width];
            for r in 0..n {
                for (i, layer) in layers.iter().enumerate() {
                    taps[r * width + i * row..][..row].copy_from_slice(&layer[(done + r) * row..][..row]);
                }
            }
            drafter.put_taps(&taps)?;
            drafter.update(&(0..n).map(|r| dflash::ContextRow { tap_row: r, slot: 0, position: done + r })
                .collect::<Vec<_>>())?;
            // SAFETY: the oracle owns the stream; a following chunk reuses
            // the taps and context metadata read by this update.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream)? };
            done += n;
        }
        let anchor = tokens[position];
        let timer = Instant::now();
        let draft = drafter.draft_device(&[dflash::DraftSeq { slot: 0, anchor, position, valid_from: 0 }],
            &engine.embedding, dflash::TargetHead::Bf16(&engine.weights.head))?.remove(0);
        draft_seconds += timer.elapsed().as_secs_f64();
        let reference = &ref_tokens[index * drafts_per..][..drafts_per];
        exact += usize::from(draft.tokens == reference);
        first += usize::from(draft.tokens[0] == reference[0]);
        matched += draft.tokens.iter().zip(reference).take_while(|(a, b)| a == b).count();
        let (cosine, _) = similarity(&bf16s(&drafter.last_hidden(1)?), &ref_hidden[index * block * hidden..][..block * hidden]);
        worst = worst.min(cosine);
        if draft.tokens[0] == reference[0] {
            let theirs = f32::from_bits(ref_features[index * drafts_per * 4]);
            feature_error = feature_error.max(f64::from((draft.features[0][0] - theirs).abs()));
        }
        if draft.tokens != reference {
            println!("position {position}: engine {:?} reference {reference:?}", draft.tokens);
        }
    }
    let n = positions.len();
    println!("draft oracle: {n} anchors, identical drafts {exact}/{n}, first draft {first}/{n}, matching prefix \
        {:.2} of {drafts_per}, worst final-norm cosine {worst:.6}, first-margin max error {feature_error:.4}, \
        {:.2} ms/draft", matched as f64 / n as f64, draft_seconds * 1e3 / n as f64);
    Ok(())
}

/// Several copies of the golden sequence verified in the same decode steps.
fn multi_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, copies: usize,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    let vocab = opened.cfg.vocab_size;
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let golden: Vec<f32> = std::fs::read(args.golden.join("logits.bin"))?
        .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let prefill = args.prefill.unwrap_or(512);
    let mut pages = engine::PageAllocator::new(engine.pages);
    let mut placements = Vec::new();
    for copy in 0..copies {
        let mut placement = pages.admit(tokens.len())?;
        let n = prefill - 37 * copy;
        engine.prefill(&mut placement, &tokens[..n],
            transport.as_deref_mut().map(|t| (t, runtime)), None)?;
        placements.push(placement);
    }
    let (mut agree, mut total) = (vec![0usize; copies], vec![0usize; copies]);
    loop {
        let counts: Vec<usize> = placements.iter().map(|p| args.step_rows.min(tokens.len() - p.len)).collect();
        if counts.contains(&0) {
            break;
        }
        let mut rows = Vec::new();
        for (p, &n) in placements.iter().zip(&counts) {
            rows.extend_from_slice(&tokens[p.len..p.len + n]);
        }
        let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let mut step: Vec<(&mut engine::GlmPlacement, usize)> = placements.iter_mut().zip(&counts).map(|(p, &n)| (p, n)).collect();
        let logits = engine.verify(&mut step, &rows,
            transport.as_deref_mut().map(|t| (t, runtime)), None)?.context("decode needs every layer")?;
        let mut offset = 0;
        for (copy, (&start, &n)) in starts.iter().zip(&counts).enumerate() {
            for r in 0..n {
                let ours = &logits[(offset + r) * vocab..][..vocab];
                let theirs = &golden[(start + r) * vocab..][..vocab];
                agree[copy] += usize::from(argmax(ours) == argmax(theirs));
                total[copy] += 1;
            }
            offset += n;
        }
    }
    println!("multi: {copies} copies, {} rows each per step: top-1 agreement per copy {:?}", args.step_rows,
        agree.iter().zip(&total).map(|(a, t)| format!("{:.1}% of {t}", 100.0 * *a as f64 / *t as f64)).collect::<Vec<_>>());
    Ok(())
}

/// `--token-check N`: prefill the golden prompt (--prefill tokens of it), then
/// [`crate::shared::token_io::gate`] over N greedy decode steps.
fn token_check(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime, steps: usize) -> Result<()> {
    ensure!(transport.is_some() || engine.skip_routed(), "--token-check needs Spark peers or --skip-routed-experts");
    let mut tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(p) = args.prefill {
        tokens.truncate(p);
    }
    let mut allocator = engine::PageAllocator::new(engine.pages);
    let mut placement = allocator.admit(tokens.len() + steps + 1)?;
    let mut last = None;
    for chunk in tokens.chunks(engine.prefill_capacity()) {
        last = engine.prefill_device(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)))?;
    }
    let last = last.context("--token-check needs every layer")?.row_host(&opened.library, 0)?;
    let first = cuteafd_core::TargetSamplingParams::greedy().select_token(&last, None, 0)? as u32;
    let result = crate::shared::token_io::gate(&opened.library, &engine.embedding, first, steps, |token| {
        engine.verify_device(&mut [(&mut placement, 1)], &[token], transport.as_deref_mut().map(|t| (t, runtime)))?
            .context("decode needs every layer")
    });
    allocator.release(placement);
    result
}
