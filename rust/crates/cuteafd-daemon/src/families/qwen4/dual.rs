//! Spark-free Qwen caller: solve both owners before any model allocation.
use super::{admission, attention_plan, engine, ple, required_fp8_programs, weights, EngineArgs, Opened};
use anyhow::{ensure, Context, Result};
use crate::shared::memory::device::{Device, Stream};
use crate::shared::experts::rtx::{exl3::Exl3Tp2, fp8moe::Fp8MoeTp2, RtxExpertLayer};
use cuteafd_loader::placement::{families::qwen4, GpuBudget, LayerMode, Onboard};

impl Opened {
    pub(super) fn with_dual_engine<T>(&self, args: &EngineArgs, peer: i32, context: cuteafd_loader::placement::planner::Context,
        media: crate::shared::experts::rtx::dense::Records,
        body: impl FnOnce(&engine::Qwen4Engine<'_>) -> Result<T>) -> Result<T> {
        ensure!(args.local_experts && args.peers.is_none() && !args.shared_only && args.expert_window.is_none(),
            "Qwen dual requires Spark-free resident local experts");
        ensure!(peer != args.device, "Qwen dual devices must differ");
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let mtp = args.mtp > 0 && layers == self.cfg.layers;
        cuteafd_bench::context::set_resolved("mtp", &if mtp { args.mtp.min(engine::DECODE_ROWS - 1) } else { 0 }.to_string());
        let devices = [args.device, peer].map(|id| Device { library: &self.library, id });
        let _entry = devices[0].enter()?;
        for rank in 0..2 { devices[rank].run(|| self.library.cuda_enable_peer(devices[1-rank].id))?; }
        let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "qwen4", split_family: None };
        let plan = attention_plan(args, mtp)?;
        let mut programs = Vec::with_capacity(2);
        for device in devices {
            programs.push(device.run(|| self.library.programs()?.with_manifest(&args.manifest)?
                .with_attention_plan(plan, |name| selected.contains(name)))?);
        }
        for program in &programs {
            program.capacities().require_context("qwen4", args.max_context)?;
            super::programs::validate_all(&self.cfg, program, args.max_context)?;
            for name in required_fp8_programs(args) { program.spec(&name)?; }
        }
        let inventory = crate::shared::inventory::RuntimeInventory::measure(&self.library,
            &[args.device, peer], crate::shared::inventory::InventoryPhase::PreWeight, |id| {
                let rank = usize::from(id == peer);
                programs[rank].load_matching(|name| selected.contains(name))?;
                self.library.peer_exchange_initialize()?;
                self.library.peer_abort_initialize()?;
                if std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1") {
                    self.library.peer_diagnostic_initialize()?;
                }
                Ok(())
            })?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
        let sms = [devices[0].run(|| self.library.sm_count())?, devices[1].run(|| self.library.sm_count())?];
        let manifests = qwen4::dual_program_manifests(&manifest,
            [u32::try_from(sms[0])?, u32::try_from(sms[1])?], plan)?;
        let catalog = self.experts.as_ref().context("Qwen TP2 needs the selected expert catalog")?;
        let experts = admission::dual_experts(args, catalog, layers)?;
        let weights = admission::dual_weights(args, &self.checkpoint, &self.cfg, layers, mtp)?;
        let mut fixed = weights.fixed;
        fixed.extend(admission::loaded_media_weights(&[args.device, peer], args.planner_media_weights)?);
        fixed.extend(experts.transport);
        fixed.extend(qwen4::dual_peer_diagnostics(
            std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1")));
        let draft = if mtp { qwen4::dual_draft_demands(catalog, &args.native_lib, args.prefill_rows as u64)? }
            else { Vec::new() };
        let draft_budget = usize::try_from(draft.iter().map(|d| d.bytes).sum::<u64>())?;
        fixed.extend(draft);
        let ple_info = self.checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
            .map(|t| (weights.resident.ple_row_bytes, t.meta.dtype == cuteafd_core::DType::F8E4M3));
        let inputs = admission::inputs(args, &self.cfg, layers, mtp, Some(&manifests[0]), ple_info, 0)?;
        let geometry = cuteafd_loader::serving_capacity::qwen_cache_geometry(&self.cfg, layers, mtp, args.kv_format)?;
        let mark_bytes = geometry.ranks[0].retained_mark_bytes;
        ensure!(mark_bytes > 0 && inputs.mark_bytes % mark_bytes == 0, "Qwen prefix reservation must name complete marks");
        let code = if catalog.exl3().is_some() { "exl3" } else { "fp8" };
        let mut pending = [0; 2];
        for rank in 0..2 {
            pending[rank] = crate::shared::inventory::pending_code(&self.library, devices[rank].id,
                rank, true, "qwen4", code,
                &cuteafd_loader::placement::ProgramSet::from_manifest(&manifests[rank], &["qwen4"]))?;
        }
        let baselines = inventory.baselines(&[]);
        let enabled = engine::startup_graphs_enabled(std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref(),
            std::env::var("CUTEAFD_QWEN4_STARTUP_GRAPHS").ok().as_deref());
        // Namespace identity alone selects push: both GPUs' dense records (and a
        // coordinator tower's) form one standalone group, admitted from headers here.
        use crate::shared::experts::rtx::push::{CoordinatorPush, PushPolicy};
        let push_policy = PushPolicy::for_snapshot(&args.snapshot)?;
        let dense_envelope = push_policy.plan(|| {
            let owners = qwen4::dual_owners(layers);
            let staging = admission::dual_dense_staging(&self.checkpoint, &self.cfg, &owners, mtp,
                args.fp8_decode, args.mtp_fp8_head, &media, [args.device, peer])?;
            admission::dual_push_envelope(staging)
        }).context("Qwen dual coordinator push admission")?;
        let host_embedding = if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host {
            u64::try_from(cuteafd_loader::families::qwen4::weight_layout::embedding_table_bytes(&self.checkpoint)?)?
        } else { 0 };
        let host = admission::host_owners(args, &self.cfg, layers, mtp, 2, 0, false,
            ple_info.map(|(row_bytes, _)| row_bytes), host_embedding,
            dense_envelope.iter().map(|envelope| ("qwen.host.push.dual", envelope.clone())).collect())?;
        let (placement, solved) = qwen4::dual_admitted(&qwen4::QwenDualInputs {
        blas_head: crate::shared::target_head::TargetHead::configured().blas_head(!args.mtp_fp8_head),
        resident_weight_limit: Some(crate::shared::memory::budget::resident_weight_limit()), context,
            owner_manifests: manifests.each_ref().map(Some),
            mark_slots: inputs.mark_bytes / mark_bytes, admission: inputs,
            gpus: [0, 1].map(|rank| GpuBudget { capacity_bytes: baselines[rank].0,
                baseline: baselines[rank].1, reserve_bytes: cuteafd_loader::placement::reserve::gpu("qwen4") }),
            arenas: &weights.arenas, layer_experts: &experts.costs,
            layer_extra: &weights.layer_extra, fixed, tp2_workspace: experts.workspace, pending_code_bytes: pending,
            max_context: args.max_context as u64, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
            spark_ranks: 0,
            onboard: args.onboard.map(|setting| setting.resolve(catalog.routed_experts().routed_layers()))
                .unwrap_or(Onboard::Auto),
            startup_graph_modes: args.planner_graph_modes.filter(|_| enabled),
            host: host.clone(), encoders: Default::default(),
        })?;
        tracing::info!(ledger = %serde_json::to_string(&cuteafd_loader::plan::SolvedPhaseLedger::from(&solved))?,
            "Qwen dual admitted phase ledger");
        cuteafd_loader::placement::reserve::log_plan(&solved, "qwen4");
        inventory.record("qwen4", &inventory.gpus.iter().map(|g| g.free_bytes).collect::<Vec<_>>());
        cuteafd_bench::context::ResolvedSettings::placement(&placement).publish();
        tracing::info!(placement = %placement.summary(), pool_tokens = placement.pool_tokens,
            "Qwen dual shared solver admission before allocation");
        let owners = placement.layers.iter().map(|layer| match layer.mode {
            LayerMode::Whole { gpu, .. } => Ok(gpu as usize), _ => anyhow::bail!("Qwen dual owner mode")
        }).collect::<Result<Vec<_>>>()?;
        qwen4::check_dual_layer_owners(&owners)?;
        // The admitted arenas were planned at the fixed cut; allocate exactly those.
        ensure!(owners == qwen4::dual_owners(layers),
            "Qwen dual solved owners differ from the planned arena cut");
        let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let loader = weights::Qwen4Loader { library: &self.library, checkpoint: &self.checkpoint,
            fp8_decode: args.fp8_decode, fp8_scales: args.fp8_scales, stream: streams[0].raw,
            dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(&self.library)), arenas: None };
        let ranks = [(devices[0], streams[0].raw), (devices[1], streams[1].raw)];
        let loader = loader.with_arenas(&self.cfg, &owners, &ranks, mtp, args.mtp_fp8_head,
            args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu)?;
        let (embedding, model) = match dense_envelope {
            Some(envelope) => {
                let load = std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::push(&self.library));
                let planning = weights::Qwen4Loader { dense: load.clone(), ..loader.clone() };
                let model = {
                    let _scope = cuteafd_ffi::memory_ledger::scope("weights");
                    planning.model_placed(&self.cfg, &owners, &ranks, mtp, args.mtp_fp8_head)?
                };
                let mut records = load.sink.as_ref().context("push sink")?.take_all();
                records.extend(media);
                let (landed, operands, staged, steps, largest, finishes) = records.summary();
                tracing::info!(landed, operands, staged, steps, largest_step_bytes = largest, finishes,
                    "Qwen dual coordinator dense records planned");
                let planned = crate::shared::experts::rtx::dense::plan_records(&args.snapshot, records, &devices, 0);
                let push = CoordinatorPush::prepare_alone_admitted(&args.snapshot, None, 2, None, envelope)
                    .map_err(crate::shared::experts::rtx::push::PushStartupError::Planning)?;
                let split = crate::shared::memory::load_timing::Split::start();
                let (pushed, embedding) = push.receive_with(devices.to_vec(), None, 0, Some(planned), ||
                    crate::shared::token_io::TokenEmbedding::load_with(&self.library, self.embed_source()?,
                        args.token_io.embed_placement, || Ok(()), |name, bytes| loader.embedding_allocation(name, bytes)));
                let pushed = push_policy.committed(pushed).context("Qwen dual coordinator push delivery")?;
                let (embedding, ()) = embedding?;
                tracing::info!(pushed = pushed.is_some(), "Qwen dual coordinator dense operands landed");
                split.log("qwen4 dual dense push");
                (embedding, model)
            }
            None => {
                anyhow::ensure!(media.operands.is_empty() && media.steps.is_empty(),
                    "Qwen dual media records exist only under a push namespace");
                crate::shared::token_io::TokenEmbedding::load_with(&self.library, self.embed_source()?,
                    args.token_io.embed_placement, || {
                        let _scope = cuteafd_ffi::memory_ledger::scope("weights");
                        loader.model_placed(&self.cfg, &owners, &ranks, mtp, args.mtp_fp8_head)
                    }, |name, bytes| loader.embedding_allocation(name, bytes))?
            }
        };
        let (projections, head) = model.check_single_residency(args.fp8_decode, args.mtp_fp8_head)?;
        let expected = cuteafd_loader::families::qwen4::resident::resident_bytes(&self.cfg, layers, mtp,
            cuteafd_loader::families::qwen4::resident::Qwen4Representation {
                fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head });
        ensure!(projections == expected.projections && head == expected.head,
            "Qwen dual selectable weight residency disagrees with admission");
        let ple = match self.cfg.ple_layers.first().copied().filter(|&layer| layer < layers) {
            Some(layer) => Some(devices[owners[layer]].run(|| ple::PleTable::load(&self.library, &self.checkpoint,
                &self.cfg, layer, args.prefill_rows, 1, engine::DECODE_ROWS))?), None => None,
        };
        weights::arena::probe(&self.library, &[devices[0].id, devices[1].id], &model, &embedding, ple.as_ref())?;
        // Load the independent full-width draft before the target halves, with
        // its admitted arena rather than a post-allocation free-memory guess.
        let draft = if mtp { Some(self.local_mtp_experts(args, layers, streams[0].raw, draft_budget)?) } else { None };
        let budget = placement.tp2.context("Qwen dual solver did not admit TP2 experts")?.peak_bytes;
        let budgets = [usize::try_from(budget[0])?, usize::try_from(budget[1])?];
        let rows = args.prefill_rows.max(engine::DECODE_ROWS);
        let pair: [Box<dyn RtxExpertLayer + '_>; 2] = if catalog.exl3().is_some() {
            Exl3Tp2::load_pair_with_packages(devices, catalog, cuteafd_core::ExpertGeometry::QWEN4,
                experts.packages.each_ref().map(|p| p.as_path()), (0..layers).collect(), rows, budgets)?.map(|e| Box::new(e) as _)
        } else {
            let tensors = catalog.fp8().context("Qwen native TP2 tensors")?;
            let directory = args.fp8_package.clone().unwrap_or_else(||
                crate::shared::experts::fp8::package_directory(&args.native_lib, 2, tensors.format()));
            Fp8MoeTp2::load_pair(devices, tensors, &directory, (0..layers).collect(), rows, budgets)?.map(|e| Box::new(e) as _)
        };
        for rank in 0..2 { ensure!(pair[rank].workspace_bytes() as u64 == experts.workspace[rank],
            "Qwen allocated TP2 workspace disagrees with admission"); }
        let pool = usize::try_from(placement.pool_tokens)?;
        let context = crate::shared::context::pool_context("qwen4", context, pool)?;
        let engine_ranks = [(devices[0], &programs[0], streams[0].raw), (devices[1], &programs[1], streams[1].raw)];
        let mut engine = engine::Qwen4Engine::new_placed(&self.library, self.cfg.clone(), model, ple, &owners,
            &engine_ranks, context, args.prefill_rows, pool.div_ceil(engine::PAGE_ROWS), args.slots, args.kv_format, embedding)?;
        engine.attention_plan = plan;
        engine.spark_bf16 = self.checkpoint.spark_input_is_bf16();
        engine.w8a8_prefill = args.fp8_prefill_w8a8;
        if args.planner_graph_modes.is_some() { engine.enable_startup_graphs(); }
        engine.set_tp2(pair, experts.wire, draft, &placement, cuteafd_loader::placement::HopSpec { storage: cuteafd_loader::placement::HopStorage::ReceiveSlots,
            row_bytes: self.cfg.hc_width() as u64 * 2, rows: rows as u64, lanes: 1, entry_gpu: 0, head_gpu: 0, terminal: cuteafd_loader::placement::TerminalHome::AttentionOwner })?;
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::OTHER_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
        }
        engine.admitted_host = Some(host);
        engine.admit_host_memory(args)?;
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        // Weights and both expert halves are resident: release the load's
        // transient pinned upload high-water before serving.
        crate::shared::memory_report::release_load_staging(&self.library)?;
        let result = body(&engine);
        engine.drain_state()?;
        drop(engine);
        for stream in &streams { stream.drain()?; }
        result
    }
}
