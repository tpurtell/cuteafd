//! serve-dsv4 admission's CPU half: the `placement` request from the
//! engine's arguments, its loaded config and one sample per GPU. The CUDA
//! half (samples, native workspace) is the caller's, so the equality test
//! drives this exact path with fake samples.
use super::EngineArgs;
use anyhow::{ensure, Result};
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
    /// dSpark stages the loaded drafter owns caches for.
    pub cache_stages: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub max_context: usize,
    pub prefix: Option<&'a crate::shared::prefix::PrefixArgs>,
    /// The local expert arena's workspace; `None` when this build lacks the
    /// coordinator expert kernels.
    pub expert_workspace: Option<u64>,
}

pub(crate) fn request(args: &EngineArgs, inputs: &Inputs<'_>) -> Result<PlacementRequest> {
    let gpus = inputs.gpus.len();
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
    let onboard = args.onboard(gpus)?;
    let stages = if args.dspark && !args.skip_routed_experts { inputs.cache_stages } else { 0 };
    let routed = *inputs.catalog.routed_experts();
    let (mut experts, draft) = if args.skip_routed_experts { (Vec::new(), Vec::new()) }
        else { v4::expert_costs(inputs.catalog, stages)? };
    let expert_workspace = match inputs.expert_workspace {
        Some(bytes) => bytes,
        None => {
            ensure!(stages == 0 && !args.fixed_onboard(),
                "V4 local expert kernels are missing: export the matching rtx_backbone package for explicit local layers/dSpark");
            tracing::warn!("V4 local expert kernels unavailable; routing all backbone layers to Sparks");
            experts.clear();
            0
        }
    };
    let spark_ranks = if args.skip_routed_experts { 0 } else { args.peers.split(',').filter(|p| !p.is_empty()).count() };
    let request = v4::request(&v4::V4Inputs { cfg: &cache_cfg, cache_stages: inputs.cache_stages,
        gpus: inputs.gpus.clone(), headroom_floor: 0, spark_ranks,
        sequences: args.max_sequences as u64, prefill_rows: prefill, decode_rows: decode,
        max_context: inputs.max_context as u64, reserve_bytes: (args.reserve_gib as u64) << 30, mark_slots,
        workspace: Some(workspace), experts, draft, expert_workspace, first_routed: routed.first_layer,
        peer_experts: v4::peer_experts(inputs.catalog, args.peer_expert_ranges),
        requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        onboard: if args.skip_routed_experts { Onboard::Auto } else { onboard },
        full_prefill_logits: 0, code_bytes: v4::code_bytes(inputs.cfg.dim, v4::code_experts(inputs.catalog), gpus) })?;
    Ok(request)
}
