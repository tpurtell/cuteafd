//! Production owners for the 20/20 attention split and bottom-up TP2 experts.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::CachePlacement;
use crate::families::deepseek_v41::v41_backbone_execution::DistributedExecution;
use crate::families::deepseek_v41::v41_backbone_shared::tp2::Weights as SharedWeights;
use crate::families::deepseek_v41::v41_engram::placement::{PlacedEngram, PlacedEngramWeights};
use crate::families::deepseek_v41::v41_experts::{tp2_ffn, ExpertLayer, ExpertWeights};
use crate::shared::memory::device::Device;
use crate::families::deepseek_v41::v41_target_head::distributed_target::DistributedTargetHead;
use crate::families::deepseek_v41::v41_target_pass::{DistributedTargetPass, TargetTapWave};
use std::rc::Rc;

pub(super) fn worker(mut args: crate::cli::NativeServeArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: &mut Option<oneshot::Sender<std::result::Result<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>, String>>>,
    stats: std::sync::Arc<std::sync::Mutex<serde_json::Value>>) -> Result<()> {
    if let Some(directory)=args.placement_directory.as_deref() {
        if let Some(layers)=PlacementHandoff::<PlacementPlan>::resumed(directory)?.map(|plan| plan.rtx_expert_layers) {
            if let admission::LocalLayers::Count(requested)=args.rtx_expert_layers {
                ensure!(requested==layers,"restart layer request differs from acknowledged workers");
            }
            args.rtx_expert_layers=admission::LocalLayers::Count(layers);
        }
    }
    let minimum_expert_layers = match args.rtx_expert_layers {
        admission::LocalLayers::Auto => 1,
        admission::LocalLayers::Count(count) => {
            ensure!((1..=40).contains(&count), "dual RTX expert layers must be 1..=40");
            count as usize
        }
    };
    prefill_capacity(args.prefill_batch_tokens)?;
    // AOT kernels keep their full scratch; row storage follows live capacity.
    let capacity = cuteafd_core::coordinator_programs::v41_live_rows(args.prefill_batch_tokens)
        .context("invalid V4.1 prefill batch")?;
    let lib = unsafe { NativeLibrary::load(&args.native_lib)? };
    lib.cuda_set_device(0)?;
    let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
    for device in devices { device.run(|| lib.cuda_enable_peer(1 - device.id))?; }
    let memory_checkpoint = |stage: &str| -> Result<()> {
        let occupied = devices.map(|d| d.run(|| { let (free, total) = lib.cuda_memory_info()?; Ok(total - free) }))
            .into_iter().collect::<Result<Vec<_>>>()?;
        tracing::info!(stage, occupied_bytes=?occupied, "dual RTX startup memory");
        cuteafd_ffi::memory_ledger::relabel_other(cuteafd_ffi::memory_ledger::intern(&format!("v41/{stage}")));
        Ok(())
    };
    memory_checkpoint("CUDA contexts and peer access")?;
    let catalog = cuteafd_loader::read_official_v41_catalog(cuteafd_loader::OFFICIAL_V41_MODEL_ID, &args.snapshot)?;
    let topology = crate::shared::spark_topology::resolve(
        args.spark_tp,
        args.spark_ep,
        args.peers.len(),
        "serve-native",
    )?;
    // Reject an explicit non-native checkpoint before any expert allocation,
    // KV reservation or readiness publication.
    crate::shared::spark_topology::require_native(topology, &catalog)?;
    if let Some(topology) = topology {
        // Fail before weights are loaded when the library cannot reduce this
        // physical-rank count.
        lib.v41_compact_reducer()?
            .require_rank_count(topology.world_size() as u32)?;
    } else if matches!(args.peers.len(), 3 | 6) {
        lib.v41_compact_reducer()?.require_rank_count(args.peers.len() as u32)?;
    }
    ensure!(args.peers.len() == 4 || topology.is_some() || catalog.exl3().is_some(),
        "implicit non-TP4 Spark layouts require an EXL3 checkpoint");
    let paired_profile = crate::families::deepseek_v41::v41_experts::paired::PairedProfile::for_serving(&catalog, args.exl3_paired_tp4)?;
    let map = CachePlacement::encoder_decoder();
    let started = Instant::now();
    // Sum resident storage plus the largest transient loading excess.
    let format = crate::families::deepseek_v41::v41_experts::ExpertFormat::of(&catalog);
    let compressed = format.is_exl3();
    let exl3_tiers: &[usize] = catalog.exl3().map(|m| m.decoder_tiers()).unwrap_or(&[]);
    let exl3_directory = crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(&args.native_lib, exl3_tiers, "rtx-tp2");
    if compressed {
        let per_lane = crate::shared::experts::rtx::pair::OwnerPair::exl3_device_bytes(&exl3_directory, capacity, cuteafd_core::ExpertGeometry::DEEPSEEK_V41)?;
        tracing::info!(per_gpu_per_lane_bytes=per_lane, capacity, "EXL3 TP2 expert workspace plan");
    } else if format.is_nvfp4() {
        let per_lane = crate::shared::experts::rtx::pair::OwnerPair::device_bytes(&lib, capacity, true, cuteafd_core::ExpertGeometry::DEEPSEEK_V41)?;
        tracing::info!(per_gpu_per_lane_bytes=per_lane, capacity, "NVFP4 W4A4 TP2 expert workspace plan");
    }
    let mut exl3_modules = [None, None];
    let inventory = crate::shared::inventory::RuntimeInventory::measure(&lib,&[0,1], |gpu| {
        admission::preload(&lib,capacity,2,catalog.nvfp4().is_some(),compressed)?;
        if compressed {
            exl3_modules[gpu as usize] = Some(admission::preload_exl3(&lib, &exl3_directory, capacity, 2)?);
        }
        Ok(())
    })?;
    let [mut decode_modules, mut prefill_modules] = [0, 1].map(|_| [None, None]);
    for (rank, modules) in exl3_modules.into_iter().enumerate() {
        if let Some([decode, prefill]) = modules {
            decode_modules[rank] = Some(decode);
            prefill_modules[rank] = Some(prefill);
        }
    }
    let mut lane_modules = [decode_modules, prefill_modules].into_iter();
    let expert_demands=admission::expert_demands(&lib,&catalog,capacity,2)?;
    let weights = BackboneLaneWeights::load_distributed_with_split(
        &lib,
        &catalog,
        map,
        BackboneLaneWeights::distributed_device_bytes_with_split(&lib, &catalog, map,args.tp2_query_projection,args.tp2_output_projection)?,
        16 << 20,
        args.tp2_query_projection,
        args.tp2_output_projection,
    )?;
    let query_shards=if args.tp2_query_projection {
        use crate::families::deepseek_v41::v41_projection_tp2::{Kind,Weights};
        Some([0usize,1].map(|rank| (0..40).map(|layer|
            Weights::load(devices[rank],&catalog,layer,Kind::QueryB,rank,
                Weights::load_peak_device_bytes(Kind::QueryB))).collect::<Result<Vec<_>>>())
            .into_iter().collect::<Result<Vec<_>>>()?)
    } else { None };
    let output_shards=if args.tp2_output_projection {
        use crate::families::deepseek_v41::v41_projection_tp2::{Kind,Weights};
        Some([0usize,1].map(|rank| (0..40).map(|layer|
            Weights::load(devices[rank],&catalog,layer,Kind::OutputB,rank,
                Weights::load_peak_device_bytes(Kind::OutputB))).collect::<Result<Vec<_>>>())
            .into_iter().collect::<Result<Vec<_>>>()?)
    } else { None };
    memory_checkpoint("attention mHC router weights")?;
    let producers = CacheProducerWeights::load_distributed(
        &lib,
        &catalog,
        map,
        CacheProducerWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    memory_checkpoint("cache producer weights")?;
    let iw = IndexLaneWeights::load_distributed(
        &lib,
        &catalog,
        map,
        IndexLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    memory_checkpoint("index query weights")?;
    let ew = PlacedEngramWeights::load(
        &lib,
        &catalog,
        map,
        PlacedEngramWeights::device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    memory_checkpoint("Engram weights")?;
    let table = devices[0].own(|| {
        let (_, total) = lib.cuda_memory_info()?;
        NativeRtxTensors::load_embedding(&lib, &catalog,
            super::admission::embedding_placement(args.embedding_placement, total))
    })?;
    memory_checkpoint("embedding weights")?;
    let vocab = [
        devices[0].own(|| crate::families::deepseek_v41::v41_tensors::VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?,
        devices[1].own(|| crate::families::deepseek_v41::v41_tensors::VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?,
    ];
    memory_checkpoint("vocabulary weights")?;
    let hw = devices[1].own(|| {
        TargetHeadWeights::load(
            &lib,
            &catalog,
            TargetHeadWeights::device_bytes(&catalog)?,
            16 << 20,
        )
    })?;
    memory_checkpoint("head normalization weights")?;
    let shared: [Rc<Vec<SharedWeights<'_>>>; 2] = [0, 1]
        .map(|r| {
            (0..40)
                .map(|l| {
                    SharedWeights::load(
                        devices[r],
                        &catalog,
                        l,
                        SharedWeights::load_peak_device_bytes(),
                    )
                })
                .collect::<Result<Vec<_>>>()
                .map(Rc::new)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .ok()
        .expect("two ranks");
    memory_checkpoint("weights")?;
    tracing::info!(capacity, backbone_lane_bytes=BackboneLane::placed_workspace_bytes_with_split(&lib, capacity,args.tp2_query_projection,args.tp2_output_projection)?,
        producer_bytes=?crate::families::deepseek_v41::v41_backbone_execution::PlacedProducerWaves::device_bytes(&lib, map, capacity)?,
        "dual RTX per-lane workspace plan");
    let make_pass = |decoder_capacity: u32| -> Result<DistributedTargetPass<'_, '_>> {
        let mut lanes = [
            BackboneLane::new_on_device(
                &weights,
                capacity,
                BackboneLane::placed_workspace_bytes_with_split(&lib, capacity,args.tp2_query_projection,args.tp2_output_projection)?,
                0,
            )?,
            BackboneLane::new_on_device(
                &weights,
                capacity,
                BackboneLane::placed_workspace_bytes_with_split(&lib, capacity,args.tp2_query_projection,args.tp2_output_projection)?,
                1,
            )?,
        ];
        if let Some(shards)=&query_shards {
            use crate::families::deepseek_v41::v41_projection_tp2::{Kind,Wave};
            for (owner,lane) in lanes.iter_mut().enumerate() {
                let budgets=Wave::device_bytes(&lib,Kind::QueryB,capacity,owner)?;
                lane.enable_tp2_query([&shards[0],&shards[1]],budgets)?;
                tracing::info!(owner,capacity,per_gpu_bytes=?budgets,"TP2 query projection workspace");
            }
        }
        if let Some(shards)=&output_shards {
            use crate::families::deepseek_v41::v41_projection_tp2::{Kind,Wave};
            for (owner,lane) in lanes.iter_mut().enumerate() {
                let budgets=Wave::device_bytes(&lib,Kind::OutputB,capacity,owner)?;
                lane.enable_tp2_output([&shards[0],&shards[1]],budgets)?;
                tracing::info!(owner,capacity,per_gpu_bytes=?budgets,"TP2 output projection workspace");
            }
        }
        let indices = [
            Some(IndexLane::new_on_device(
                &iw,
                capacity,
                IndexLane::placed_workspace_bytes(&lib, map, capacity, 0)?
                    .iter()
                    .sum(),
                0,
            )?),
            Some(IndexLane::new_on_device(
                &iw,
                decoder_capacity,
                IndexLane::placed_workspace_bytes(&lib, map, decoder_capacity, 1)?
                    .iter()
                    .sum(),
                1,
            )?),
        ];
        DistributedTargetPass::new(
            map,
            devices[0].own(|| {
                TargetEmbeddingWave::new(
                    &lib,
                    &table,
                    capacity as usize,
                    TargetEmbeddingWave::device_bytes(capacity as usize)?,
                )
            })?,
            lanes,
            indices,
            DistributedExecution::new(
                &producers,
                capacity,
                crate::families::deepseek_v41::v41_backbone_execution::PlacedProducerWaves::device_bytes(
                    &lib, map, capacity,
                )?,
            )?,
            PlacedEngram::new(&ew, capacity as usize, PlacedEngram::device_bytes(&lib, map, capacity as usize)?)?,
            DistributedTargetHead::new(devices, &hw, [&vocab[0], &vocab[1]], cuteafd_core::V41_VERIFY_ROWS,
                DistributedTargetHead::device_bytes(cuteafd_core::V41_VERIFY_ROWS, 64640)?)?,
            devices[1]
                .own(|| TargetTapWave::new(&lib, decoder_capacity as usize, TargetTapWave::device_bytes(decoder_capacity as usize)?))?,
            Duration::from_secs(120),
        )
    };
    let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        cuteafd_core::V41_VERIFY_ROWS, args.dspark.then_some(usize::from(args.dspark_draft_limit)),
        args.attention_plan_rows)?;
    let mut pass = make_pass(capacity).context("constructing first distributed target lane")?;
    if args.tp2_attention { pass.enable_dual_attention()?; }
    memory_checkpoint("first target lane")?;
    // Both lanes also produce source 20 from every encoder row on GPU1, so
    // backbone/query and producer buffers retain full prefill capacity there.
    // Only the first pass handles decoder replay and full cached continuation;
    // lane 1's GPU1 index selection and taps serve at most 64 verify rows.
    // TP2 expert workspaces remain full capacity on both GPUs and lanes.
    let mut second = make_pass(80).context("constructing second distributed target lane")?;
    if args.tp2_attention { second.enable_dual_attention()?; }
    pass.install_attention_plan(attention_plan)?;
    second.install_attention_plan(attention_plan)?;
    memory_checkpoint("second target lane")?;
    let draft_weights = if args.dspark {
        let width=if args.dspark_draft_limit>5 {7} else {5};
        Some(if args.tp2_dspark_experts {
            let budgets=[devices[0].run(||Ok(lib.cuda_memory_info()?.0.min(32usize<<30)))?,
                devices[1].run(||Ok(lib.cuda_memory_info()?.0.min(32usize<<30)))?];
            devices[1].own(||crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::load_serving_tp2(&lib,&catalog,
                capacity,args.concurrency,budgets,16<<20,width))?
        } else {devices[1].own(|| crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::load_serving_with_width(&lib, &catalog,
            capacity, args.concurrency, 32 << 30, 16 << 20, width,
            Some(&crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(&args.native_lib, exl3_tiers, "dspark"))))?})
    } else { None };
    memory_checkpoint("draft weights")?;
    let mut draft = draft_weights.as_ref().map(|weights| DraftRuntime::with_distributed_requests(
        devices, weights, &table, [&vocab[0], &vocab[1]], capacity, args.concurrency)).transpose()?;
    if let Some(draft) = &mut draft {
        draft.set_draft_limit(args.dspark_draft_limit)?;
        draft.set_fixed(args.dspark_fixed);
    }
    attention_plan.validate_loaded(draft.as_ref().map(|draft| draft.attention_drafts()))?;
    tracing::info!(plan_rows=attention_plan.rows(), capacity=attention_plan.capacity_rows(),
        "V4.1 fixed BF16 AOT compressed/native FP32 window attention plan");
    memory_checkpoint("draft runtime")?;
    // Vision and target snapshot copies use GPU0. These allocations precede KV sizing.
    let mut vision = crate::families::deepseek_v41::v41_vision_encoder::Encoder::load(&args, &catalog)?;
    memory_checkpoint("vision")?;
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    let snapshot_slots = if args.prefix_cache_entries == 0 { 0 } else { 2 * args.prefix_cache_entries as usize + 2 };
    let prefix_arenas = super::prefix::Arenas::new(devices[0],
        draft.as_ref().map(|d| d.mark_device()), snapshot_slots)?;
    let snapshot_bytes = super::prefix::Arenas::device_bytes(snapshot_slots, draft.is_some());
    memory_checkpoint("snapshot arenas")?;
    // Every other persistent owner is now live. Reserve both transport lanes,
    // minimum routed weights, KV and setup headroom before filling extra layers.
    let per_lane = match format {
        crate::families::deepseek_v41::v41_experts::ExpertFormat::Exl3 => {
            tp2_ffn::Wave::exl3_device_bytes(&exl3_directory, &lib, capacity)?
        }
        crate::families::deepseek_v41::v41_experts::ExpertFormat::Nvfp4 => {
            tp2_ffn::Wave::nvfp4_device_bytes(&lib, capacity)?
        }
        crate::families::deepseek_v41::v41_experts::ExpertFormat::Native => tp2_ffn::Wave::device_bytes(&lib, capacity)?,
    };
    // Both Spark transport lanes live on GPU1 and reserve one wave each; a
    // six-rank replicated layout needs its own larger per-wave buffer count.
    let spark_ranks = args.peers.len();
    let wave_bytes = NativeTp4Wave::device_bytes_for(capacity, spark_ranks)?;
    let memory=admission::admission_memory(&lib,&inventory)?;
    let deferred=[admission::EXPERT_SETUP_HEADROOM as u64,
        (admission::EXPERT_SETUP_HEADROOM + 2*wave_bytes) as u64];
    let pool=admission::resolve(&args,&memory,expert_demands, (2*per_lane) as u64,&deferred,minimum_expert_layers)?;
    let expert_layers=pool.placement.onboard_layers;
    let admitted=pool.placement.tp2.context("dual RTX admission has no routed arena")?;
    let rank_budgets=std::array::from_fn(|gpu| admitted.peak_bytes[gpu].saturating_sub((2*per_lane) as u64) as usize);
    tracing::info!(expert_layers,rank_peak_bytes=?rank_budgets,cache_bytes=?pool.cache_bytes,
        pool_tokens=pool.placement.pool_tokens,deferred_bytes=?deferred,"shared dual RTX admission");
    let placement_handoff=args.placement_directory.as_deref().map(|directory|
        PlacementHandoff::publish(std::path::Path::new(directory), PlacementPlan::new(args.rtx_gpus, expert_layers))).transpose()?;
    eprintln!("loading bottom {expert_layers} expert layers as TP2");
    let routed = tp2_ffn::load_routed_weights(devices, &catalog, expert_layers, capacity,
        rank_budgets, &exl3_directory)?;
    if let Some(handoff)=placement_handoff {
        handoff.wait_ready(Duration::from_secs(900))?;
    }
    let mut make_transport = || {
        let timing = crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing();
        let mut transport = devices[1].own(|| {
            let roce = super::spark_transport(&args.peers, capacity, timing, topology)?;
            NativeTp4Wave::new(&lib, roce, wave_bytes)
        })?;
        if let Some(profile) = &paired_profile { transport.install_paired(profile.clone())?; }
        transport.install_tp2(tp2_ffn::Wave::new(routed.clone(), shared.clone(), expert_layers, capacity, lane_modules.next().context("missing admitted TP2 lane")?)?)?;
        if crate::shared::memory::chain::device_exchange_enabled() && paired_profile.is_none() && topology.is_none()
            && format == crate::families::deepseek_v41::v41_experts::ExpertFormat::Native {
            // Verification waves (up to 80 rows) on the device-driven exchange.
            devices[1].run(|| transport.get_mut().install_device_link(&args.peers, 80.min(capacity as usize), 39,
                cuteafd_transport::TcpTransportConfig { timing, timeout: Duration::from_secs(120),
                    max_frame_bytes: 64 << 20 }))?;
        }
        Ok::<_, anyhow::Error>(transport)
    };
    let mut transport = make_transport()?;
    let mut second_transport = make_transport()?;
    memory_checkpoint("TP2 transports")?;
    if let Some(draft) = &mut draft { draft.configure_policy(&transport, catalog.nvfp4().is_some())?; }
    // Allocation must consume exactly the admitted pages; do not run a second
    // sizing policy after loading conditional owners.
    for gpu in 0..2 {
        let (free,_) = devices[gpu].run(|| lib.cuda_memory_info())?;
        ensure!(pool.cache_bytes[gpu] <= free,"TP2 setup exceeded shared cache admission on GPU {gpu}");
    }
    tracing::info!(tp2_attention=args.tp2_attention,source_pages=?pool.pages,global_bytes=pool.global_bytes,
        cache_bytes=?pool.cache_bytes,reservation_bytes=?pool.reservation_bytes,snapshot_slots,snapshot_bytes,
        rtx_expert_layers=expert_layers,"dual RTX cache reservation after fixed allocations");
    let cache_bytes: [usize;2]=pool.cache_bytes.clone().try_into().map_err(|_|anyhow::anyhow!("dual cache budgets"))?;
    let token_map = cuteafd_loader::EngramTokenMap::from_file(&args.snapshot.join("tokenizer.json"))?;
    let rows = capacity as usize;
    let pipeline = unsafe { cuteafd_loader::EngramPipeline::new(&catalog, token_map, rows,
        2 * cuteafd_core::ENGRAM_LAYERS.len(), rows * 64 * 1024)? };
    let mut requests = if args.tp2_attention {
        Requests::new_replicated(&lib,pipeline,args.concurrency as usize,pool.pages,map,cache_bytes)?
    } else { Requests::new_distributed(&lib,pipeline,args.concurrency as usize,pool.pages,map,cache_bytes)? };
    pass.configure_cache_replicas(requests.cache())?;
    second.configure_cache_replicas(requests.cache())?;
    requests.use_engine_pages()?;
    memory_checkpoint("allocated KV cache")?;
    crate::shared::memory_report::release_load_staging(&lib);
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let prefixes = scheduler::prepare_prefix_cache(&lib, &args, &requests, prefix_arenas, draft.is_some())?;
    scheduler::publish_capacity(&requests, &prefixes);
    tracing::info!(elapsed_ms=started.elapsed().as_millis(), "dual RTX serving owners ready");
    vision.connect()?;
    ready.take().context("startup readiness missing")?.send(Ok(vision.health_handle()))
        .map_err(|_| anyhow::anyhow!("API startup cancelled"))?;
    scheduler::serve(&lib, &args, &runtime, &mut receive, &mut pass, &mut second, &mut requests,
        &mut transport, &mut second_transport, draft.as_mut().map(|d| d.get_mut()), &mut vision, stats, prefixes)
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT, CUTEAFD_DUAL_PEERS, two GPUs and live Sparks"]
    fn distributed_worker_serves_text_requests() -> Result<()> {
        use clap::{Args, FromArgMatches};
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
        let snapshot = std::env::var("CUTEAFD_SNAPSHOT")?;
        let native = std::env::var("CUTEAFD_NATIVE_LIB")?;
        let peers = std::env::var("CUTEAFD_DUAL_PEERS")?;
        let batch = std::env::var("CUTEAFD_WORKER_PREFILL").unwrap_or_else(|_| "80".into());
        let matches = crate::cli::NativeServeArgs::augment_args(clap::Command::new("fixture"))
            .try_get_matches_from(["fixture", "--snapshot", &snapshot, "--native-lib", &native,
                "--peers", &peers, "--dspark", "--rtx-gpus", "2", "--prefill-batch-tokens", &batch])?;
        let args = crate::cli::NativeServeArgs::from_arg_matches(&matches)?;
        let (send, receive) = mpsc::channel(16);
        let mut outputs = Vec::new();
        let arithmetic = if std::env::var_os("CUTEAFD_WORKER_LONG").is_some() {
            format!("{}\nWhat is two plus two? Answer briefly.", "This is background context.\n".repeat(600))
        } else { "What is two plus two? Answer briefly.".into() };
        for text in [arithmetic.as_str(), "Write a Python function that adds two numbers.", arithmetic.as_str()] {
            let (events, output) = mpsc::unbounded_channel();
            send.blocking_send(NativeRequest { prompt: format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>"),
                constraint: None, images: Vec::new(), media: Vec::new(), audio: Vec::new(), max_tokens: 8, sampling: Default::default(), stop_token_ids: vec![cuteafd_api::openai::DEEPSEEK_EOS_TOKEN_ID], events, usage: None, probe: None })?;
            outputs.push(output);
        }
        drop(send);
        let (ready, mut readiness) = oneshot::channel();
            let stats = std::sync::Arc::new(std::sync::Mutex::new(serde_json::Value::Null));
            super::super::worker(args, receive, &mut Some(ready), stats)?;
        readiness.try_recv()?.map_err(anyhow::Error::msg)?;
        for (i, mut output) in outputs.into_iter().enumerate() {
            let mut ready = 0;
            let mut finish = 0;
            let mut response = String::new();
            while let Ok(chunk) = output.try_recv() {
                match chunk {
                    Ok(InferenceChunk::Ready { prompt_usage, .. }) => {
                        ready += 1;
                        if i == 2 { ensure!(prompt_usage.prompt_cache_hit_tokens == prompt_usage.prompt_tokens,
                            "repeated worker prompt did not restore the exact prefix"); }
                    },
                    Ok(InferenceChunk::Text { content, .. }) => response.push_str(&content),
                    Ok(InferenceChunk::Finish { .. }) => finish += 1,
                    Err(error) => anyhow::bail!("worker request {i} failed: {error:?}"),
                    _ => (),
                }
            }
            ensure!(ready == 1 && finish == 1 && !response.is_empty(), "worker request {i} did not complete");
            eprintln!("PASS distributed worker request {i}: {response:?}");
        }
        Ok(())
    }
}
