//! Spark-free Qwen caller: solve both owners before any model allocation.
use super::{admission, attention_plan, engine, ple, required_fp8_programs, weights, EngineArgs, Opened};
use anyhow::{ensure, Context, Result};
use crate::shared::memory::device::{Device, Stream};
use crate::shared::experts::rtx::{exl3::Exl3Tp2, fp8moe::Fp8MoeTp2, RtxExpertLayer};
use cuteafd_loader::placement::{families::qwen4, GpuBudget, LayerMode, Onboard};

impl Opened {
    pub(super) fn with_dual_engine<T>(&self, args: &EngineArgs, peer: i32, automatic_context: bool,
        body: impl FnOnce(&engine::Qwen4Engine<'_>) -> Result<T>) -> Result<T> {
        ensure!(args.local_experts && args.peers.is_none() && !args.shared_only && args.expert_window.is_none(),
            "Qwen dual requires Spark-free resident local experts");
        ensure!(peer != args.device, "Qwen dual devices must differ");
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let mtp = args.mtp > 0 && layers == self.cfg.layers;
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
            for name in required_fp8_programs(args) { program.spec(&name)?; }
        }
        let inventory = crate::shared::inventory::RuntimeInventory::measure(&self.library,
            &[args.device, peer], crate::shared::inventory::InventoryPhase::PreWeight, |id| {
                let rank = usize::from(id == peer);
                programs[rank].load_matching(|name| selected.contains(name))?;
                self.library.tp2_native_initialize()?;
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
        let placement = qwen4::dual_placement(&qwen4::QwenDualInputs { automatic_context: automatic_context,
            owner_manifests: manifests.each_ref().map(Some),
            mark_slots: inputs.mark_bytes / mark_bytes, admission: inputs,
            gpus: [0, 1].map(|rank| GpuBudget { capacity_bytes: baselines[rank].0,
                baseline: baselines[rank].1, headroom_bytes: cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes.max(3 << 30) }),
            layer_weights: &weights.resident.layer_bytes, layer_experts: &experts.costs,
            layer_extra: &weights.layer_extra, fixed, tp2_workspace: experts.workspace, pending_code_bytes: pending,
            max_context: args.max_context as u64, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
            spark_ranks: 0, onboard: args.onboard.unwrap_or(Onboard::Auto), startup_graph_modes: args.planner_graph_modes.filter(|_| enabled),
        })?;
        inventory.record("qwen4", &inventory.gpus.iter().map(|g| g.free_bytes).collect::<Vec<_>>());
        tracing::info!(placement = %placement.summary(), pool_tokens = placement.pool_tokens,
            "Qwen dual shared solver admission before allocation");
        let owners = placement.layers.iter().map(|layer| match layer.mode {
            LayerMode::Whole { gpu, .. } => Ok(gpu as usize), _ => anyhow::bail!("Qwen dual owner mode")
        }).collect::<Result<Vec<_>>>()?;
        qwen4::check_dual_layer_owners(&owners)?;
        let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let loader = weights::Qwen4Loader { library: &self.library, checkpoint: &self.checkpoint,
            fp8_decode: args.fp8_decode, fp8_scales: args.fp8_scales, stream: streams[0].raw,
            dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(&self.library)) };
        let ranks = [(devices[0], streams[0].raw), (devices[1], streams[1].raw)];
        let (embedding, model) = crate::shared::token_io::TokenEmbedding::load(&self.library, self.embed_source()?,
            args.token_io.embed_placement, || {
                let _scope = cuteafd_ffi::memory_ledger::scope("weights");
                loader.model_placed(&self.cfg, &owners, &ranks, mtp, args.mtp_fp8_head)
            })?;
        let (projections, head) = model.check_single_residency(args.fp8_decode, args.mtp_fp8_head)?;
        let expected = cuteafd_loader::families::qwen4::resident::resident_bytes(&self.cfg, layers, mtp,
            cuteafd_loader::families::qwen4::resident::Qwen4Representation {
                fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head });
        ensure!(projections == expected.projections && head == expected.head,
            "Qwen dual selectable weight residency disagrees with admission");
        let ple = match self.cfg.ple_layers.first().copied().filter(|&layer| layer < layers) {
            Some(layer) => Some(devices[owners[layer]].run(|| ple::PleTable::load(&self.library, &self.checkpoint,
                &self.cfg, layer, &args.table, args.prefill_rows.max(engine::DECODE_ROWS)))?), None => None,
        };
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
        let context = crate::shared::context::pool_context("qwen4", args.max_context, automatic_context, pool, 256)?;
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
        engine.admit_host_memory(args)?;
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        let result = body(&engine);
        engine.drain_state()?;
        drop(engine);
        for stream in &streams { stream.drain()?; }
        result
    }
}
