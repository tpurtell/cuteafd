//! serve-dsv4 admission's CPU half: the `placement` request from the
//! engine's arguments, its loaded config and one sample per GPU. The CUDA
//! half (samples, native workspace) is the caller's, so the equality test
//! drives this exact path with fake samples.
use super::EngineArgs;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;
use cuteafd_loader::placement::{families::deepseek_v4 as v4, planner::Pin, Baseline, Onboard};

pub(crate) struct Inputs<'a> {
    pub cfg: &'a DeepseekV4Config,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub manifest: &'a serde_json::Value,
    /// `dsv4f` / `dsv4p`.
    pub family: &'a str,
    /// (total bytes, baseline) per GPU: `--device` then `--split-device`.
    pub gpus: Vec<(u64, Baseline)>,
    /// Per GPU, selected-program code still to load by ready, measured from the
    /// same pre-weight sample (`pending_code`). Charged as the named Runtime
    /// `loaded code` owner, never subtracted from the free sample (#1614).
    pub pending_code: Vec<u64>,
    /// False before uploads. After uploads, measured free bytes are rebased by
    /// the retained arenas (embedding included), which remain explicit demands.
    pub weights_resident: bool,
    /// dSpark stages the loaded drafter owns caches for.
    pub cache_stages: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub max_context: usize,
    pub prefix: Option<&'a crate::shared::prefix::PrefixArgs>,
    /// The local expert arena's workspace; `None` when this build lacks the
    /// coordinator expert kernels.
    pub expert_workspace: Option<u64>,
    /// Actual per-rank TP2 expert workspace probe; None means no TP2 package.
    pub tp2_workspace: Option<[u64; 2]>,
    pub exchange_f32: bool,
    /// Each GPU's physical capacity beside its logical ceiling in `gpus`.
    pub physical: Vec<u64>,
    /// Host RAM (total, available) sampled before any allocation.
    pub host_memory: (u64, u64),
}

#[cfg(test)]
mod spark_tests {

    fn planned(family: &str, backend: cuteafd_loader::plan::spark_executions::SparkBackend, decode: Option<u32>,
        unit: u64, waves: u64) -> cuteafd_core::SparkExecutions {
        let options = cuteafd_loader::plan::layout::LayoutOptions { prefill_unit_rows: Some(unit),
            prefill_waves: waves, ..Default::default() };
        cuteafd_loader::plan::layout::planned_spark_executions(family, backend, decode, &options).unwrap()
    }

    #[test]
    fn planner_and_serve_admit_identical_spark_owners() -> anyhow::Result<()> {
        use cuteafd_loader::plan::spark_executions::SparkBackend;
        for backend in [SparkBackend::Fp8, SparkBackend::Native, SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest)] {
            for (waves, unit) in [(1usize, 4096usize), (2, 4096), (3, 2752), (4, 2048)] {
                assert_eq!(super::spark_model(backend, 64, unit, waves)?.executions,
                    planned("deepseek_v4", backend, Some(64), unit as u64, waves as u64));
            }
        }
        Ok(())
    }

    #[test]
    fn provenance_names_the_admitted_layout_and_graph_arena() {
        let sample = cuteafd_loader::placement::RuntimeSample { device: 0, total_bytes: 96 << 30,
            free_bytes: 80 << 30, context_bytes: 0, module_bytes: 0, sms: 188,
            arch: "sm_120".into(), driver: "test".into() };
        let provenance = super::Provenance { layout: "snapshot/x; context/1048576".into(),
            native_sha256: "native".into(), programs_sha256: "programs".into(), gpus: vec![sample],
            spark_world: 2, onboard_layers: vec![vec![0, 1]], prefill: cuteafd_loader::placement::planner::PrefillShape {
                rows: 2752, waves: 3 }, graph_reserves: vec![1 << 20, 2 << 20], graph_bootstraps: vec![1 << 19, 1 << 19] };
        let value = provenance.value();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["family"], "deepseek_v4");
        assert_eq!(value["devices"][0]["arch"], "sm_120");
        assert_eq!(value["onboard_layers"][0], serde_json::json!([0, 1]));
        assert_eq!(value["graph_reserve_bytes"], serde_json::json!([1 << 20, 2 << 20]));
        assert_eq!(value["graph_bootstrap_bytes"], serde_json::json!([1 << 19, 1 << 19]));
        provenance.register();
        let registered = cuteafd_ffi::memory_ledger::fault::inputs();
        assert_eq!(registered["memory_reconciliation_provenance"]["layout"], "snapshot/x; context/1048576");
        assert_eq!(registered["memory_reconciliation_provenance"]["programs_sha256"], "programs");
    }

    #[test]
    fn v4_spark_endpoints_match_the_decode_and_prefill_links() -> anyhow::Result<()> {
        use cuteafd_loader::plan::spark_executions::SparkBackend;
        for (decode_rows, compiled) in [(16, 16), (64, 80), (80, 80)] {
            let model = super::spark_model(SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest), decode_rows, 2752, 3)?;
            model.validate(cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH)?;
            // The static prefill owner holds the selected unit; EXL3 exports m2752.
            assert_eq!((model.capacity(), model.compiled_capacity()), (2752, 2752));
            assert_eq!(model.executions.decode, [cuteafd_core::DecodeExecution {
                capacity: decode_rows as u32, compiled_capacity: compiled }]);
            assert_eq!((model.executions.prefill.unit_rows, model.executions.prefill.buffered_units), (2752, 3));
            assert_eq!(model.endpoints.len(), 2);
            assert_eq!(model.endpoints[0].capacity, decode_rows as u32);
            assert_eq!(model.endpoints[1].capacity, 2752);
            assert!(model.endpoints.iter().all(|endpoint| endpoint.ingress == cuteafd_core::SparkIngress::Fp8K32));
        }
        Ok(())
    }
}

pub(crate) fn spark_model(backend: cuteafd_loader::plan::spark_executions::SparkBackend,
    decode_rows: usize, unit_rows: usize, waves: usize) -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    use cuteafd_core::{SparkEndpoint, SparkIngress};
    // The installed coordinator manifest fixes this build's decode rows.
    let executions = cuteafd_loader::plan::spark_executions::admit_family("deepseek_v4", &backend.exports(),
        Some(u32::try_from(decode_rows)?), unit_rows as u64, waves as u64)?;
    let endpoints = vec![
        SparkEndpoint { ring_depth: None, capacity: u32::try_from(decode_rows)?, ingress: SparkIngress::Fp8K32, partial_reply_bytes: None },
        SparkEndpoint { ring_depth: None, capacity: u32::try_from(unit_rows.max(decode_rows))?, ingress: SparkIngress::Fp8K32, partial_reply_bytes: None },
    ];
    cuteafd_transport::worker_selection::ModelAdmission::new(endpoints,
        crate::shared::placement::spark_rings(64 << 20)?, executions)
}

/// The retained prefix config serve-dsv4 builds for exactly the admitted quota.
/// Admission already charged it with every pinned table and host owner.
pub(crate) fn host_config(prefix: Option<&crate::shared::prefix::PrefixArgs>,
    layout: cuteafd_engine::prefix::FamilyLayout, max_context: usize, sequences: usize,
    admitted: cuteafd_loader::placement::planner::serving::HostPrefixOwners)
    -> Result<Option<cuteafd_hostcache::config::Config>> {
    let (Some(prefix), true) = (prefix, admitted.quota_bytes > 0) else { return Ok(None) };
    let mut exact = prefix.clone();
    exact.host_cache_bytes = crate::shared::prefix::budget::HostBudget::Bytes(admitted.quota_bytes);
    let config = exact.resolve_host_config(layout, max_context, sequences)?;
    ensure!(config.as_ref().is_some_and(|c| c.bytes == admitted.quota_bytes
        && c.chunk_bytes == admitted.chunk_bytes),
        "V4 host prefix config differs from the admitted quota {admitted:?}");
    Ok(config)
}

/// The graph arena admission charged on each coordinator rank, and the
/// estimate a device uses until its first measured capture.
pub(crate) struct GraphCache {
    pub reserves: Vec<u64>,
    pub bootstraps: Vec<u64>,
}

/// One admission: what was placed, what it charged, and the decisions
/// `/v1/stats` reconciliation cites.
pub(crate) struct Admission {
    pub placement: cuteafd_loader::placement::Placement,
    pub host_prefix: cuteafd_loader::placement::planner::serving::HostPrefixOwners,
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub graphs: GraphCache,
    pub decisions: cuteafd_loader::placement::planner::Decisions,
}

/// Admit one placement from these inputs: one solve over every device and
/// host owner, before any allocation.
pub(crate) fn admit(inputs: &v4::V4Inputs) -> Result<Admission> {
    let prepared = v4::prepare(inputs).map_err(anyhow::Error::msg)?;
    let prefix = prepared.prefix().owners;
    let expert_usage = prepared.expert_usage().cloned();
    for note in prepared.prefix().notes(inputs.max_context, inputs.sequences) { tracing::info!("{note}"); }
    let request = v4::production_request(inputs);
    let admitted = prepared.admit(request).map_err(anyhow::Error::msg)?;
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "deepseek_v4");
    tracing::info!(solve_us = admitted.solve_us, admissions = 1,
        "V4 admission prepare + solve/execution (excludes checkpoint inventory)");
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    Ok(Admission { placement: admitted.placement, host_prefix: prefix, expert_usage,
        graphs: GraphCache { reserves: admitted.graph_reserves, bootstraps: admitted.graph_bootstraps },
        decisions: admitted.solved.decisions })
}

/// The admitted layout identity `/v1/stats` Ready and first_request memory
/// reconciliation rows cite. Pure data, so the CPU half can test it.
pub(crate) struct Provenance {
    pub layout: String,
    pub native_sha256: String,
    pub programs_sha256: String,
    pub gpus: Vec<cuteafd_loader::placement::RuntimeSample>,
    pub spark_world: usize,
    pub onboard_layers: Vec<Vec<usize>>,
    pub prefill: cuteafd_loader::placement::planner::PrefillShape,
    pub graph_reserves: Vec<u64>,
    pub graph_bootstraps: Vec<u64>,
}

impl Provenance {
    pub fn value(&self) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1, "family": "deepseek_v4",
            "source_commit": env!("CUTEAFD_BUILD_COMMIT"), "source_dirty": env!("CUTEAFD_BUILD_DIRTY"),
            "toolchain": env!("CUTEAFD_BUILD_TOOLCHAIN"),
            "native_sha256": self.native_sha256, "programs_sha256": self.programs_sha256,
            "layout": self.layout, "devices": &self.gpus, "spark_world": self.spark_world,
            "onboard_layers": &self.onboard_layers, "prefill": self.prefill,
            "graph_basis": "per-site byte estimates; no calibrated node rates for this family",
            "graph_reserve_bytes": &self.graph_reserves, "graph_bootstrap_bytes": &self.graph_bootstraps
        })
    }

    /// Register with the memory ledger so every reconciliation row names it.
    pub fn register(&self) {
        cuteafd_ffi::memory_ledger::fault::set_inputs(
            serde_json::json!({ "memory_reconciliation_provenance": self.value() }));
    }
}

/// The family facts serve-dsv4 admits: the same inputs `plan --layout` builds.
pub(crate) fn inputs(args: &EngineArgs, inputs: &Inputs<'_>) -> Result<v4::V4Inputs> {
    let gpus = inputs.gpus.len();
    let weights = cuteafd_loader::families::deepseek_v4::resident::WeightInventory::from_headers(
        inputs.cfg, gpus, args.dspark, args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu,
        inputs.catalog.tensors().iter().map(|t| &t.metadata))?;
    let mut baselines = inputs.gpus.clone();
    if inputs.weights_resident {
        let loaded = weights.loaded_bytes()?;
        for ((total, baseline), bytes) in baselines.iter_mut().zip(loaded) {
            *baseline = match *baseline {
                Baseline::Measured { free_bytes } => {
                    let free_bytes = free_bytes.checked_add(bytes).context("V4 measured weight rebase overflow")?;
                    ensure!(free_bytes <= *total, "V4 measured sample and resident weights exceed total memory");
                    Baseline::Measured { free_bytes }
                }
                Baseline::Planned { .. } => anyhow::bail!("V4 loaded modes require a measured sample"),
            };
        }
    }
    // The loader's legacy config requests one nextn stage, but a dSpark
    // checkpoint loads all three concrete stages. Describe those actual
    // cache owners rather than the caller's nextn configuration.
    let mut cache_cfg = inputs.cfg.clone();
    cache_cfg.n_mtp_layers = inputs.cache_stages;
    let shape = args.prefill_shape(inputs.manifest["capacities"]["prefill_rows"]
        .as_u64().context("V4 manifest prefill_rows")?, None)?;
    let (prefill, decode) = (shape.unit_rows, inputs.decode_rows as u64);
    ensure!(inputs.prefill_rows as u64 == prefill,
        "V4 runtime unit rows {} disagree with admitted shape {prefill}", inputs.prefill_rows);
    let context = v4::manifest_context(inputs.manifest, inputs.family, inputs.cfg.n_heads, decode);
    let attention = cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?;
    let resolved = attention.unwrap_or(if gpus == 2 && context.is_some() {
        cuteafd_loader::placement::AttentionPlacement::Context
    } else { cuteafd_loader::placement::AttentionPlacement::Heads });
    let host_marks = v4::host_marks_enabled(gpus, resolved != cuteafd_loader::placement::AttentionPlacement::Heads,
        inputs.prefix.is_some_and(|p| p.host_cache_bytes.enabled()));
    let mark_slots = v4::mark_slots(&cache_cfg, gpus, prefill, inputs.cache_stages, args.max_sequences as u64,
        inputs.prefix.map_or(0, |p| p.prefix_cache_entries) as u64,
        (inputs.prefix.map_or(0, |p| p.prefix_cache_mark_mib) as u64) << 20, host_marks)?;
    let scratch = cuteafd_loader::serving_capacity::deepseek_v4_workspace_scratch(
        inputs.manifest, inputs.family, gpus == 2, prefill, decode)?;
    let workspace = cuteafd_loader::serving_capacity::deepseek_v4_workspace_geometry(&cache_cfg, prefill, decode,
        cuteafd_loader::serving_capacity::compiled_c128_width(inputs.manifest, inputs.family)? * 128, gpus, shape.waves, scratch)?;
    let onboard = args.onboard(inputs.catalog.routed_experts().routed_layers());
    let stages = if args.dspark && !args.skip_routed_experts { inputs.cache_stages } else { 0 };
    let routed = *inputs.catalog.routed_experts();
    let (mut experts, draft) = if args.skip_routed_experts { (Vec::new(), Vec::new()) }
        else { v4::expert_costs(inputs.catalog, stages)? };
    let expert_workspace = match inputs.expert_workspace {
        Some(bytes) => bytes,
        // TP2 backbone executors do not need the TP1/dSpark package.
        None if gpus == 2 && (stages == 0 || args.tp2_dspark_experts) => 0,
        None => {
            ensure!(stages == 0 && !args.fixed_onboard(),
                "V4 local expert kernels are missing: export the matching rtx_backbone package for explicit local layers/dSpark");
            tracing::warn!("V4 local expert kernels unavailable; routing all backbone layers to Sparks");
            experts.clear();
            0
        }
    };
    let (experts_half, tp2_workspace) = if gpus == 2 && !args.skip_routed_experts {
        match inputs.tp2_workspace {
            Some(workspace) => (v4::expert_half_costs(inputs.catalog)?, workspace),
            None => {
                ensure!(!args.tp2_dspark_experts && (onboard == Onboard::Auto || onboard.layers(routed.layers - routed.first_layer) == Some(0)), "V4 TP2 expert kernels are missing: export the matching rtx_tp2 package for explicit local layers");
                tracing::warn!("V4 TP2 expert kernels unavailable; routing all backbone layers to Sparks");
                (Vec::new(), [0; 2])
            }
        }
    } else { (Vec::new(), [0; 2]) };
    let spark_ranks = if args.skip_routed_experts { 0 } else { args.peers.split(',').filter(|p| !p.is_empty()).count() };
    Ok(v4::V4Inputs {
        expert_usage: args.expert_usage_enabled()?,
        resident_weight_limit: Some(crate::shared::memory::budget::resident_weight_limit()), spark_reply_format: cuteafd_core::SparkReplyFormat::resolve(cuteafd_core::spark_reply::requested().prefill, false), serving_context: cuteafd_loader::placement::planner::Context::resolve(inputs.max_context as u64, (args.max_context > 0).then_some(args.max_context as u64)), cfg: cache_cfg, weights, cache_stages: inputs.cache_stages,
        gpus: baselines, spark_ranks,
        sequences: args.max_sequences as u64, prefill_rows: prefill, prefill_waves: shape.waves, decode_rows: decode,
        max_context: inputs.max_context as u64, mark_slots,
        workspace: Some(workspace), draft_half: if args.tp2_dspark_experts { v4::draft_half_costs(inputs.catalog, stages)? } else { Vec::new() },
        experts, draft, expert_workspace, first_routed: routed.first_layer,
        experts_half, tp2_workspace, exchange_f32: inputs.exchange_f32,
        requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        onboard: if args.skip_routed_experts { Onboard::Auto } else { onboard },
        pending_code: inputs.pending_code.clone(),
        context, attention, extra: Vec::new(), physical_bytes: inputs.physical.clone(),
        host: Some(v4::V4Host { total: inputs.host_memory.0, available: inputs.host_memory.1,
            headroom: inputs.prefix.map_or(8 << 30, |p| p.host_cache_headroom_bytes).max(inputs.host_memory.0 / 10),
            cache_bytes: match inputs.prefix.map(|p| p.host_cache_bytes) {
                Some(crate::shared::prefix::budget::HostBudget::Auto) => None,
                Some(crate::shared::prefix::budget::HostBudget::Bytes(bytes)) => Some(bytes),
                None => Some(0),
            },
            prefix_entries: inputs.prefix.map_or(0, |p| p.prefix_cache_entries as u64),
            span_rows: shape.span_rows,
            embedding_bytes: if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host {
                inputs.catalog.tensor("embed.weight")?.metadata.byte_length } else { 0 },
            c128_width: cuteafd_loader::serving_capacity::compiled_c128_width(inputs.manifest, inputs.family)? }) })
}
