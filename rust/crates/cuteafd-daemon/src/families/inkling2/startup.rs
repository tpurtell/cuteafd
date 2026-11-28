//! Native startup facts on the same shared solver as `cuteafd plan`.
//! Weight delivery stays unpublished until its admitted push transaction lands.
use super::layers::Kernels;
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory}, prefix::{budget::HostBudget, PrefixArgs}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::inkling2::InklingEncoding, ModelEncoding};
use cuteafd_ffi::{programs::Programs, NativeLibrary};
use cuteafd_loader::{families::inkling2::{admission::{gather_with_weights, Coordinator, FactsRequest, SelectedWeights}, Inkling2Config},
    placement::{families::inkling2::{self, AdmittedInkling, InklingInputs}, planner::PrefillShape,
        AttentionPlacement, Onboard, ResidentWeights}, plan::Checkpoint};
use std::{net::SocketAddr, path::Path, sync::Arc};

#[derive(Clone, Copy)]
pub(crate) struct Options<'a> {
    pub device: i32,
    pub split_device: Option<i32>,
    pub peers: &'a [SocketAddr],
    pub manifest: &'a Path,
    pub native_lib: &'a Path,
    pub prefill: PrefillShape,
    pub decode_rows: u32,
    pub sequences: u64,
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub kv_format: cuteafd_loader::families::inkling2::InklingKvCache,
    pub drafter:Option<cuteafd_loader::families::qwen4::Qwen4KvCache>,
    pub onboard: Onboard,
    pub prefix: &'a PrefixArgs,
    pub resident_weights: ResidentWeights,
    pub expert_usage: bool,
    pub vision: &'a crate::shared::vision::still::Options,
    /// `--decode-graphs` (default on): off charges no graph reserve and never captures.
    pub decode_graphs: bool,
}

/// Prepared owner facts, not readiness. No target/cache/transport allocation
/// escapes this edge, and no measured sample is taken after loading weights.
pub(crate) struct Prepared<'a> {
    pub kernels: Vec<Kernels<'a>>,
    pub encoding: ModelEncoding,
    pub eos: Vec<u32>,
    pub inputs: InklingInputs,
    pub admitted: AdmittedInkling,
    pub runtime: RuntimeInventory,
    /// Expert code belongs to the sampled baseline until its runtime binds.
    pub module_pins: Vec<cuteafd_ffi::fp8_moe::Fp8MoeModule>,
    pub exl3_modules: Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>,
}

fn validate(options: &Options<'_>, cfg: &Inkling2Config) -> Result<()> {
    ensure!(options.device >= 0 && (2..=8).contains(&options.peers.len()),
        "Inkling distributed startup requires 2..8 Spark endpoints");
    ensure!(options.peers.iter().collect::<std::collections::BTreeSet<_>>().len() == options.peers.len(),
        "Inkling Spark endpoints must be distinct");
    cuteafd_core::concurrency::validate("inkling2", options.sequences)?;
    let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(cfg, true, 1)?.ring_rows;
    cuteafd_loader::families::inkling2::verify::verify_ring(cfg.window, ring, inkling2::DECODE_ROWS as usize)?;
    ensure!(options.context_tokens == 0 || options.context_tokens <= cfg.max_position as u64,
        "Inkling context exceeds checkpoint");
    ensure!(options.prefill.rows > 0 && options.prefill.rows <= 4096 && options.prefill.rows % 64 == 0
        && (1..=3).contains(&options.prefill.waves), "Inkling prefill differs from shared export/wave capacity");
    ensure!((1..=inkling2::MAX_DECODE_ROWS).contains(&options.decode_rows),
        "Inkling decode_rows {} exceeds supported 1..={}", options.decode_rows, inkling2::MAX_DECODE_ROWS);
    ensure!(options.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
        "Inkling requires exact captured SWA/convolution frontiers");
    Ok(())
}

pub(crate) fn prepare<'a>(library: &'a NativeLibrary, programs: &'a Programs<'a>, checkpoint: &Checkpoint,
    cfg: &Inkling2Config, options: Options<'_>) -> Result<Prepared<'a>> {
    validate(&options, cfg)?;
    cfg.release_geometry()?;
    // Checkpoint-native chat resolution happens before CUDA or target loading.
    let encoding = ModelEncoding::Inkling(Arc::new(InklingEncoding::from_snapshot(&checkpoint.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(eos == [200006], "Inkling checkpoint EOS differs from its native sampling-end marker");
    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::INKLING2)
        .map_err(|geometry| anyhow::anyhow!("Inkling requires isolated process geometry, found {geometry:?}"))?;
    let catalog = match &checkpoint.view {
        Some(view) => cuteafd_loader::read_expert_catalog_view(view)?,
        None => cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?,
    };
    let manifest = Arc::new(serde_json::from_slice(&std::fs::read(options.manifest)?)?);
    let mut module_pins = Vec::new();
    let mut exl3_modules = Vec::new();
    let peers = options.split_device.map(|peer| {
        crate::shared::peer_split::probed_device(library, options.device, Some(peer))?.context("Inkling heads split requires bidirectional GPU peer access")
    }).transpose()?;
    let devices = std::iter::once(options.device).chain(peers).collect::<Vec<_>>();
    let gpus = devices.len();
    let selected_weights = SelectedWeights::new(checkpoint, cfg, gpus, AttentionPlacement::Heads, options.drafter.is_some())?;
    let body_rows = options.prefill.rows.max(u64::from(options.decode_rows)) as usize;
    let replay_cap = cuteafd_loader::families::inkling2::execution::replay_capacity(options.decode_rows as usize);
    let fir_commit = if replay_cap == 64 { "fir_history_commit" } else { "fir_history_commit_m128" };
    let compact_programs = (0..gpus).map(|rank|
        selected_weights.compact_programs(&manifest, rank, body_rows))
        .collect::<Result<Vec<_>>>()?;
    let mut kernels = Vec::with_capacity(gpus);
    let runtime = RuntimeInventory::measure(library, &devices, InventoryPhase::PreWeight, |id| {
        // Static load finish is explicitly checked before any push owner exists.
        // Missing pin/export is a named startup error, never a host fallback.
        programs.program("inkling2_padded_transpose_bf16_r16_c1024_p64", &["source", "output"])?;
        kernels.push(Kernels::bind_rank_rows(programs, options.kv_format, gpus, options.decode_rows as usize)?);
        let rank = devices.iter().position(|&device| device == id).context("Inkling rank not in the prepared device list")?;
        let names = &compact_programs[rank];
        let (loaded, _) = programs.load_matching(|name| names.contains(name))?;
        ensure!(loaded == names.len(), "Inkling compact pre-weight program inventory differs from manifest");
        if options.drafter.is_some(){
            programs.load_matching(|name|name.starts_with("inkling_draft_")||name.starts_with("draft_bf16_")||name==fir_commit)?;
        }
        if let Some(tensors) = catalog.fp8() {
            let layers = (0..cfg.layers).collect::<Vec<_>>();
            for (view, _) in tensors.homogeneous_layers(&layers)? {
                let directory = crate::shared::experts::fp8::package_directory(options.native_lib, gpus, view.format());
                // SAFETY: trusted installed static expert package; retained
                // across sampling and final backend binding, no weight storage.
                module_pins.push(unsafe { cuteafd_ffi::fp8_moe::Fp8MoeModule::load(&directory)? });
            }
        } else if let Some(exl3) = catalog.exl3() {
            let directory = options.native_lib.parent().context("Inkling native library directory absent")?
                .join("exl3").join(exl3.package_name("inkling2")).join(format!("rtx-tp{gpus}"));
            let preload = if gpus == 2 { crate::shared::experts::rtx::exl3::Exl3Tp2::preload_tp2_modules }
                else { crate::shared::experts::rtx::exl3::Exl3Tp2::preload_whole_modules };
            exl3_modules.extend(preload(
                crate::shared::memory::device::Device { library, id }, &directory,
                body_rows, cuteafd_core::ExpertGeometry::INKLING2,
                crate::shared::experts::rtx::RtxSelectionPolicy::CanonicalDecode)?);
        }
        Ok(())
    })?;
    let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
    let samples = runtime.gpus.iter().map(|sample| (sample.total_bytes,
        policy.gpu_cap_bytes(sample.total_bytes), sample.free_bytes)).collect();
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    let host_cache_bytes = match options.prefix.host_cache_bytes {
        HostBudget::Auto => None, HostBudget::Bytes(bytes) => Some(bytes),
    };
    let request = FactsRequest { target_pool_tokens: None, vision: options.vision.vision,
        vision_replicas: options.vision.vision_replicas,
        media_cache_bytes: options.vision.media_cache_bytes.map(|bytes| bytes.0 as u64), kv_format: options.kv_format, drafter: options.drafter.map(|kv_cache|cuteafd_loader::families::inkling2::draft::Selection{
        kv_cache,sms:library.sm_count().unwrap_or(1) as usize}), checkpoint, catalog: &catalog, cfg, manifest_path: Some(options.manifest), manifest: Some(manifest),
        coordinator: Coordinator::Measured { samples, loaded_code: vec![0; gpus] }, attention: AttentionPlacement::Heads,
        spark_ranks: options.peers.len(), spark_bytes: cuteafd_loader::serving_capacity::spark_memory::runtime_bytes(&options.peers.iter().map(ToString::to_string).collect::<Vec<_>>())?.into_iter().max().unwrap_or(0),
        prefill: options.prefill, decode_graphs: options.decode_graphs, decode_rows: options.decode_rows, context_tokens: options.context_tokens, pool_tokens: options.pool_tokens,
        sequences: options.sequences, host, host_cache_bytes,
        prefix_entries: options.prefix.prefix_cache_entries as u64,
        prefix_mark_budget: options.prefix.prefix_cache_mark_mib as u64 * (1 << 20),
        replies: cuteafd_core::spark_reply::requested(),
        head: crate::shared::target_head::TargetHead::configured().blas_head(true),
        expert_usage: options.expert_usage, resident_weight_limit: options.resident_weights };
    let mut inputs = gather_with_weights(&request, selected_weights)?;
    cuteafd_loader::serving_capacity::spark_memory::apply_rank_budgets(&mut inputs.budgets,
        &cuteafd_loader::serving_capacity::spark_memory::runtime_bytes(&options.peers.iter().map(ToString::to_string).collect::<Vec<_>>())?)?;
    let admitted = inputs.admit(inkling2::request(inputs.context, options.pool_tokens, options.onboard,
        options.prefill, cuteafd_loader::placement::planner::Operation::Serving,
        inputs.resident_weight_limit, request.replies))?;
    let sampled_free = runtime.gpus.iter().map(|gpu| gpu.free_bytes).collect::<Vec<_>>();
    for (rank, sample) in runtime.gpus.iter().enumerate() {
        let baseline = admitted.solved.budgets.iter().find(|budget| budget.locations.contains(
            &cuteafd_loader::placement::planner::Location::coordinator(rank as u8)))
            .context("Inkling admission missing measured coordinator budget")?;
        ensure!(matches!(baseline.baseline, cuteafd_loader::placement::planner::Baseline::Sampled { free, .. }
            if free == sample.free_bytes) && inputs.code[rank] == 0,
            "Inkling admission must charge preloaded compact code in rank {rank}'s sampled baseline exactly once");
        tracing::info!(rank, selected = compact_programs[rank].len(), sampled_free = sample.free_bytes,
            module_bytes = sample.module_bytes, "Inkling compact programs included in admitted pre-weight sample");
    }
    runtime.record("inkling2", &sampled_free);
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    cuteafd_bench::context::set_resolved("max-context-tokens", &admitted.solved.served_context().to_string());
    cuteafd_bench::context::set_resolved("decode-rows", &options.decode_rows.to_string());
    Ok(Prepared { kernels, encoding, eos, inputs, admitted, runtime, module_pins, exl3_modules })
}

/// Admitted grouped delivery entry, still not a public serving executor. The
/// endpoint is selected before target allocation and native views remain private
/// until both push participants and Spark workers commit their exact assignment.
pub(crate) fn deliver<'a>(library: &'a NativeLibrary,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView, checkpoint: &Checkpoint,
    prepared: &Prepared<'a>, options: &Options<'_>, media: crate::shared::vision::push::Contribution)
    -> Result<(Vec<super::weights::TargetWeights<'a>>, Vec<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>,
        Vec<crate::shared::speculation::recurrent_weights::Weights<'a>>)> {
    use crate::shared::{memory::device::Device, placement::{spark_rings, SparkAdmission}};
    use cuteafd_core::{ExpertGeometry, SparkEndpoint};
    use cuteafd_loader::placement::SparkWorkingSet;
    use cuteafd_transport::{worker_selection::ModelAdmission, TcpTransportConfig};
    let catalog = cuteafd_loader::read_expert_catalog_view(view)?;
    let inputs = &prepared.inputs;
    let admitted = &prepared.admitted;
    let working = SparkWorkingSet { rtx_layers: inputs.routed[..admitted.onboard].to_vec(),
        spark_layers: inputs.routed[admitted.onboard..].to_vec(), spark_ranks: inputs.spark_ranks,
        prefill_lanes: inkling2::LANES };
    let ingress = inkling2::ingress(inputs.experts.first().and_then(|layer| layer.storage));
    let ingress = if inputs.experts.iter().all(|layer| inkling2::ingress(layer.storage) == ingress) { ingress }
        else { cuteafd_core::SparkIngress::Native };
    let endpoint = SparkEndpoint { ring_depth: None, capacity: u32::try_from(inputs.prefill.rows)?,
        ingress, partial_reply_bytes: None };
    let config = TcpTransportConfig::default();
    let model = ModelAdmission::for_clients(ExpertGeometry::INKLING2, endpoint,
        spark_rings(config.max_frame_bytes)?, 1,
        inkling2::spark_executions_for_rows(inputs.prefill, catalog.exl3().is_some(), inputs.decode_rows)?)?;
    let spark = SparkAdmission::plan_view(view, ExpertGeometry::INKLING2, Some(&catalog),
        &working, inputs.spark_ranks, model, None)?;
    let devices = std::iter::once(options.device).chain(options.split_device).map(|id| Device { library, id }).collect();
    let (target, membership) = super::weights::PendingTarget::prepare(devices,
        view, checkpoint, &catalog, inputs, admitted, media)?;
    let pending = spark.start_with(options.peers, Some(membership))?;
    target.finish(&catalog, pending)
}

/// Construct the execution after exact shared admission and grouped delivery.
pub(crate) fn whole<'a>(library: &'a Arc<NativeLibrary>, programs: &'a Programs<'a>,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView, checkpoint: &Checkpoint,
    cfg: &Inkling2Config, options: Options<'_>) -> Result<(super::engine::Execution<'a>, InklingInputs, AdmittedInkling, ModelEncoding, crate::shared::vision::still::Ready)> {
    use crate::shared::{memory::device::{Device, Stream}, spark_intake::SparkLink};
    use cuteafd_transport::{expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
        ExpertV2Dtype, SparkSessionShape, TcpTransportConfig};
    let phase = crate::shared::memory::load_timing::Split::start();
    let decode_graphs = options.decode_graphs;
    let prepared = prepare(library, programs, checkpoint, cfg, Options { ..options })?;
    phase.log("inkling2 code and admission");
    cuteafd_bench::context::phase("Inkling code and admission");
    let phase = crate::shared::memory::load_timing::Split::start();
    let physical = std::iter::once(options.device).chain(options.split_device).collect::<Vec<_>>();
    let mut media = crate::shared::vision::still::Ready::load(options.vision, &checkpoint.snapshot,
        options.native_lib, library, &physical, prepared.inputs.vision.as_ref()
            .and_then(|vision| vision.admitted(&prepared.admitted.solved)),
        cuteafd_loader::media::ImageFamily::Inkling2, prepared.inputs.vision.as_ref().map_or(0, |vision| vision.media_cache))?;
    let (targets, pushed, draft_weights) = deliver(library, view, checkpoint, &prepared, &options, std::mem::take(&mut media.push))?;
    phase.log("inkling2 grouped weight delivery");
    cuteafd_bench::context::phase("Inkling grouped weight delivery");
    let phase = crate::shared::memory::load_timing::Split::start();
    let Prepared { kernels, encoding, inputs, admitted, module_pins, exl3_modules, .. } = prepared;
    let device = Device { library, id: options.device };
    let stream = std::rc::Rc::new(Stream::new(device)?);
    let requests = super::owners::RequestOwners::rank(stream.clone(), &inputs, &admitted, 0)?;
    let layout = &inputs.targets[0];

    let catalog = cuteafd_loader::read_expert_catalog_view(view)?;
    let backings = match &inputs.local {
        inkling2::LocalPackage::Installed { backings } => backings[admitted.onboard].as_slice(),
        _ if admitted.onboard == 0 => &[],
        _ => anyhow::bail!("Inkling admitted resident backend absent"),
    };
    let decode_rows = inputs.decode_rows as usize;
    let body_rows = usize::try_from(inputs.prefill.rows.max(u64::from(inputs.decode_rows)))?;
    let peer_stream = options.split_device.map(|id| Stream::new(Device { library, id }).map(std::rc::Rc::new)).transpose()?;
    let head = if let Some(peer) = &peer_stream {
        super::head::Head::pair(stream.clone(),peer.clone(),cfg,decode_rows,[layout.stored_head(), inputs.targets[1].stored_head()], !draft_weights.is_empty())?
    } else { super::head::Head::new(stream.clone(),cfg,decode_rows,layout.stored_head())? };
    let local = super::experts::serving_dispatchers(stream.clone(), peer_stream.clone(), &catalog,
        options.native_lib, backings, body_rows, pushed, cfg, &admitted.solved)?;
    ensure!(local.len() == admitted.onboard, "Inkling resident layer handle count differs");
    let geometry = inputs.links.first().context("Inkling admitted Spark link absent")?.clone();
    let mut link = device.run(|| SparkLink::prepared(library, SparkWorld::canonical(options.peers.len())?,
        options.peers, TcpTransportConfig::default(), geometry))?;
    link.enable_terminal_ownership(Arc::clone(library))?;
    let mut session = SparkSessionShape::new(u32::try_from(inputs.prefill.rows)?, u32::try_from(cfg.hidden)?, u32::try_from(cfg.topk)?,
        if inputs.experts.iter().all(|layer| inkling2::ingress(layer.storage) == cuteafd_core::SparkIngress::Fp8K32) {
            ExpertV2Dtype::Fp8E4m3Ue8m0K32
        } else { ExpertV2Dtype::Bf16 })?;
    session.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    device.run(|| link.bootstrap(&session))?;
    let mut targets = targets.into_iter();
    let mut kernels = kernels.into_iter();
    let mut lead_kernels = kernels.next().context("Inkling lead kernels absent")?;
    lead_kernels.bind_compact(programs, stream.clone(), layout, body_rows, decode_rows.min(64), inputs.target_fp8_scratch[0])?;
    if peer_stream.is_some() && !draft_weights.is_empty() && layout.stored_head().is_none() {
        let width = cuteafd_loader::shared::vocabulary::shard(cfg.vocab_size,128,0)?.len();
        lead_kernels.draft_head=Some(device.run(|| Ok(cuteafd_ffi::draft_projection::DraftProjections::load(programs,&[(cfg.hidden,width)],64)?))?);
    }
    let mut execution = super::engine::Execution::new(link, requests, head,
        targets.next().context("Inkling lead target absent")?, lead_kernels,
        local, cfg.clone(),
        inputs.experts.iter().map(|layer| inkling2::ingress(layer.storage)).collect())?;
    if let Some(peer_stream) = &peer_stream {
        use crate::shared::peer_split::{PeerExchange, RankDevice};
        let ranks = [&stream, peer_stream].map(|stream| RankDevice { device: stream.device.id, stream: stream.raw });
        let requests = super::owners::RequestOwners::rank(peer_stream.clone(), &inputs, &admitted, 1)?;
        let exchange = PeerExchange::new_abortable(library, ranks, 1,
            body_rows * cfg.hidden * 4, decode_rows * cfg.hidden * 4)?;
        let mut peer_kernels = kernels.next().context("Inkling peer kernels absent")?;
        peer_kernels.bind_compact(programs, peer_stream.clone(), &inputs.targets[1],
            body_rows, decode_rows.min(64), inputs.target_fp8_scratch[1])?;
        if !draft_weights.is_empty() && inputs.targets[1].stored_head().is_none() {
            let width = cuteafd_loader::shared::vocabulary::shard(cfg.vocab_size,128,1)?.len();
            peer_kernels.draft_head=Some(peer_stream.device.run(|| Ok(cuteafd_ffi::draft_projection::DraftProjections::load(programs,&[(cfg.hidden,width)],64)?))?);
        }
        execution.peer = Some(super::engine::Peer { graphs: None, exchange, requests,
            target: targets.next().context("Inkling peer target absent")?, kernels: peer_kernels });
    }
    ensure!(targets.next().is_none() && kernels.next().is_none(), "Inkling rank inventory not fully bound");
    if !draft_weights.is_empty() {
        let facts=inputs.drafter.as_ref().context("delivered drafter has no admission facts")?;
        let mut layout=facts.layout.clone();
        for rank in &mut layout.runtime{
            let projection = rank.buffers.iter().filter(|(name, _)| matches!(name.as_str(), "projection.trellis" | "projection.fp8" | "step.partial_f32")).cloned().collect::<Vec<_>>();
            *rank=cuteafd_loader::formats::block_dspark::recurrent::runtime::RecurrentRuntimeLayout::new(&layout.config,
                rank.slots,rank.sequences,rank.context_rows,usize::try_from(inputs.context.maximum)?,
                usize::try_from(admitted.solved.pool_tokens)?,library.sm_count()? as usize,rank.kv_cache,
                rank.rank.rank,rank.rank.ranks,rank.rank.lead)?;
            rank.buffers.extend(projection);
        }
        let shape = layout.runtime.first().context("Inkling admitted draft ranks")?.rank;
        crate::shared::draft::report::publish(drafter_report(shape.ranks, shape.lead, &physical, &admitted.solved)?);
        let streams=std::iter::once(stream.clone()).chain(peer_stream.clone()).collect();
        execution.draft=Some(std::rc::Rc::new(super::draft::Binding::new(programs,streams,draft_weights,layout,
            &admitted.solved,execution.target.projection("model.llm.unembed.weight")?,
            &execution.head, &execution.kernels, execution.peer.as_ref().map(|peer| super::head::PeerHead { kernels: &peer.kernels, target: &peer.target }),
            usize::try_from(inputs.prefill.rows)?,usize::try_from(inputs.prefill.waves)?,decode_rows,inputs.prefix_mark_slots)?));
    }
    let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?;
    let mut weights = crate::shared::memory::receipts::Inventory::new(devices, "inkling_target",
        cuteafd_loader::placement::planner::Lifetime::RESIDENT, cuteafd_core::memory_layout::Category::Weights, "Inkling target");
    execution.target.inventory(&mut weights, "lead")?;
    if let Some(peer) = &execution.peer { peer.target.inventory(&mut weights, "peer")?; }
    for receipt in weights.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()) {
        let actual = &receipt.allocation;
        let planned = admitted.solved.facts.allocations.iter().find(|owner| owner.id == actual.id)
            .context("Inkling target arena backing not admitted")?;
        ensure!(actual.bytes == planned.bytes && actual.location == planned.location,
            "Inkling target arena backing differs: {}", actual.id);
    }
    ensure!(weights.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).count() == inputs.gpus,
        "Inkling one target arena per physical rank differs");
    let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "retained",
        cuteafd_loader::placement::planner::Lifetime::TARGET, cuteafd_core::memory_layout::Category::Workspace, "Inkling requests");
    let mut census = crate::shared::memory::receipts::Census::default();
    execution.requests.allocation_receipts(&mut inventory, &mut census, devices)?;
    execution.head.inventory(&mut inventory, "inkling2.head")?;
    execution.kernels.compact_inventory(&mut inventory, "inkling2.target.0")?;
    if let Some(peer) = &execution.peer {
        peer.requests.allocation_receipts(&mut inventory, &mut census, devices)?;
        peer.kernels.compact_inventory(&mut inventory, "inkling2.target.1")?;
        for (rank, name, buffer) in peer.exchange.allocation_buffers() {
            inventory.device(&format!("inkling2.attention.{rank}.{name}"), buffer)?;
        }
    }
    super::admission::reconcile_requests(&admitted.solved, &inputs, &inventory.receipts)?;
    // Attach after every persistent target/draft owner is resident, so the
    // graph controller starts from the same non-graph allocation baseline.
    use super::graphs::DecodeGraphs;
    use cuteafd_loader::serving_capacity::inkling2_graphs;
    let attach = |rank, stream: &crate::shared::memory::device::Stream<'a>| {
        let policy = inkling2_graphs::policy(rank, inputs.gpus, decode_graphs)?;
        ensure!(inputs.graph_reserve[rank] == policy.bytes, "Inkling graph cache disagrees with admission");
        DecodeGraphs::new(stream, rank, policy)
    };
    execution.graphs = attach(0, &execution.requests.decode.stream)?;
    if let Some(peer) = &mut execution.peer { peer.graphs = attach(1, &peer.requests.decode.stream)?; }
    execution.enable_graph_cycles()?;
    // Runtime backends now retain the same code sampled before weights.
    drop(module_pins); drop(exl3_modules);
    phase.log("inkling2 request owners and transport bootstrap");
    cuteafd_bench::context::phase("Inkling request owners and transport bootstrap");
    Ok((execution, inputs, admitted, encoding, media))
}

#[cfg(test)]
mod tests {
    #[test]
    fn startup_uses_native_chat_and_preweight_shared_fact_gather_without_readiness() {
        let source = include_str!("startup.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("InklingEncoding::from_snapshot(&checkpoint.snapshot)"));
        assert!(source.contains("eos == [200006]"));
        assert!(source.contains("InventoryPhase::PreWeight"));
        assert!(source.contains("RtxSelectionPolicy::CanonicalDecode"));
        assert!(!source.contains("RtxSelectionPolicy::FirstCapacity"));
        let resident=include_str!("experts.rs").split("#[cfg(test)]").next().unwrap();
        assert!(resident.contains("RtxSelectionPolicy::CanonicalDecode"));
        assert!(!resident.contains("RtxSelectionPolicy::FirstCapacity"));
        assert!(source.contains("Coordinator::Measured"));
        assert_eq!(source.matches("SelectedWeights::new(").count(), 1);
        assert!(source.contains("gather_with_weights(&request, selected_weights)"));
        assert!(source.find("programs.load_matching(|name| names.contains(name))").unwrap()
            < source.find("Coordinator::Measured { samples").unwrap());
        assert!(source.contains("inputs.admit(inkling2::request("));
        assert!(!source.contains("ready_tx"));
        assert!(!source.contains("TargetWeights::load"));
        assert!(source.contains("SparkAdmission::plan_view("));
        assert!(source.contains("PendingTarget::prepare("));
        assert!(source.contains("spark.start_with(options.peers, Some(membership))"));
        assert!(source.contains("target.finish(&catalog, pending)"));
    }
}

fn drafter_report(ranks: usize, lead: usize, devices: &[i32], plan: &cuteafd_loader::placement::planner::ResolvedPlan) -> Result<crate::shared::draft::report::DrafterLayout> {
    use crate::shared::draft::report::{self, Kind, Placement};
    let placement = if ranks == 1 { Placement::Whole } else { Placement::Split };
    report::admitted(Kind::Dspark, placement, u8::try_from(lead)?, devices, plan)
}

#[cfg(test)]
#[test]
fn inkling_reports_recurrent_draft_rank_shape() -> Result<()> {
    use crate::shared::draft::report::{Kind, Placement, Rank, tests::{owner, plan}};
    let admitted = plan(vec![owner(0, 2048), owner(1, 1024)]);
    let split = drafter_report(2, 0, &[4, 6], &admitted)?;
    assert_eq!((split.kind, split.placement, split.lead), (Kind::Dspark, Some(Placement::Split), Some(Rank::CudaRank(4))));
    assert_eq!(split.arena_bytes, [2048, 1024]);
    assert_eq!(drafter_report(1, 0, &[4], &admitted)?.placement, Some(Placement::Whole));
    Ok(())
}
