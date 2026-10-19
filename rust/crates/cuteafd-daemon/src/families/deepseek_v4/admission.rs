//! serve-dsv4 admission's CPU half: the `placement` request from the
//! engine's arguments, its loaded config and one sample per GPU. The CUDA
//! half (samples, native workspace) is the caller's, so the equality test
//! drives this exact path with fake samples.
use super::EngineArgs;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;
use cuteafd_loader::placement::{families::deepseek_v4 as v4, Baseline, Onboard, PlacementRequest};

pub(crate) struct Inputs<'a> {
    pub cfg: &'a DeepseekV4Config,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub manifest: &'a serde_json::Value,
    /// `dsv4f` / `dsv4p`.
    pub family: &'a str,
    /// (total bytes, baseline) per GPU: `--device` then `--split-device`.
    pub gpus: Vec<(u64, Baseline)>,
    /// None before uploads; exact frozen modes after uploads. Rebase measured
    /// free bytes by their non-embedding weights, which remain explicit demands.
    pub loaded_modes: Option<&'a [cuteafd_loader::placement::LayerMode]>,
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
}

#[cfg(test)]
mod spark_tests {
    #[test]
    fn v4_spark_endpoints_match_prefill_links_and_optional_decode_link() -> anyhow::Result<()> {
        for decode_rows in [16, 64, 80] {
            for device in [false, true] {
                let model = super::spark_model(decode_rows, device)?;
                model.validate(cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH)?;
                assert_eq!((model.capacity, model.compiled_capacity), (4096, 4096));
                assert_eq!(model.endpoints.len(), super::super::engine::PREFILL_LANES + usize::from(device));
                assert!(model.endpoints[..super::super::engine::PREFILL_LANES].iter()
                    .all(|endpoint| endpoint.capacity == 4096));
                if device { assert_eq!(model.endpoints.last().unwrap().capacity, decode_rows as u32); }
                assert!(model.endpoints.iter().all(|endpoint| endpoint.ingress == cuteafd_core::SparkIngress::Fp8K32));
            }
        }
        Ok(())
    }
}

pub(crate) fn spark_model(decode_rows: usize, device_exchange: bool)
    -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    use cuteafd_core::{SparkEndpoint, SparkIngress};
    let mut endpoints = vec![SparkEndpoint { capacity: 4096, ingress: SparkIngress::Fp8K32 };
        super::engine::PREFILL_LANES];
    if device_exchange {
        endpoints.push(SparkEndpoint { capacity: u32::try_from(decode_rows)?, ingress: SparkIngress::Fp8K32 });
    }
    Ok(cuteafd_transport::worker_selection::ModelAdmission {
        capacity: 4096, compiled_capacity: 4096, endpoints,
        rings: crate::shared::placement::spark_rings(64 << 20)?,
    })
}

pub(crate) fn request(args: &EngineArgs, inputs: &Inputs<'_>) -> Result<PlacementRequest> {
    let gpus = inputs.gpus.len();
    let weights = cuteafd_loader::families::deepseek_v4::resident::WeightInventory::from_headers(
        inputs.cfg, gpus, args.dspark, inputs.catalog.tensors().iter().map(|t| &t.metadata))?;
    let mut baselines = inputs.gpus.clone();
    if let Some(modes) = inputs.loaded_modes {
        let loaded = weights.loaded_bytes(modes, gpus)?;
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
    let (prefill, decode) = (inputs.prefill_rows as u64, inputs.decode_rows as u64);
    let mark_slots = match inputs.prefix.map_or(0, |p| p.prefix_cache_entries) {
        0 => 0,
        entries => v4::mark_slots(&cache_cfg, gpus, prefill, inputs.cache_stages, args.max_sequences as u64,
            entries as u64, (inputs.prefix.map_or(0, |p| p.prefix_cache_mark_mib) as u64) << 20)?,
    };
    let scratch = cuteafd_loader::serving_capacity::deepseek_v4_workspace_scratch(
        inputs.manifest, inputs.family, gpus == 2, prefill, decode)?;
    let workspace = cuteafd_loader::serving_capacity::deepseek_v4_workspace_geometry(&cache_cfg, prefill, decode,
        cuteafd_loader::serving_capacity::compiled_c128_width(inputs.manifest, inputs.family)? * 128, gpus, scratch)?;
    let onboard = args.onboard()?;
    let stages = if args.dspark && !args.skip_routed_experts { inputs.cache_stages } else { 0 };
    let routed = *inputs.catalog.routed_experts();
    let (mut experts, draft) = if args.skip_routed_experts { (Vec::new(), Vec::new()) }
        else { v4::expert_costs(inputs.catalog, stages)? };
    let expert_workspace = match inputs.expert_workspace {
        Some(bytes) => bytes,
        // TP2 backbone executors do not need the TP1/dSpark package.
        None if gpus == 2 && stages == 0 => 0,
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
                ensure!(onboard == Onboard::Auto || onboard.layers(routed.layers - routed.first_layer) == Some(0), "V4 TP2 expert kernels are missing: export the matching rtx_tp2 package for explicit local layers");
                tracing::warn!("V4 TP2 expert kernels unavailable; routing all backbone layers to Sparks");
                (Vec::new(), [0; 2])
            }
        }
    } else { (Vec::new(), [0; 2]) };
    let spark_ranks = if args.skip_routed_experts { 0 } else { args.peers.split(',').filter(|p| !p.is_empty()).count() };
    let request = v4::request(&v4::V4Inputs { automatic_context: args.max_context == 0, cfg: &cache_cfg, weights, cache_stages: inputs.cache_stages,
        gpus: baselines, headroom_floor: 0, spark_ranks,
        sequences: args.max_sequences as u64, prefill_rows: prefill, decode_rows: decode,
        max_context: inputs.max_context as u64, reserve_bytes: (args.reserve_gib as u64) << 30, mark_slots,
        workspace: Some(workspace), experts, draft, expert_workspace, first_routed: routed.first_layer,
        experts_half, tp2_workspace, exchange_f32: inputs.exchange_f32,
        requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        onboard: if args.skip_routed_experts { Onboard::Auto } else { onboard },
        full_prefill_logits: 0, code_bytes: v4::code_bytes(inputs.cfg.dim, v4::code_experts(inputs.catalog), gpus),
        context: v4::manifest_context(inputs.manifest, inputs.family, inputs.cfg.n_heads, decode),
        attention: cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)? })?;
    Ok(request)
}
