//! Qwen automatic KV admission after weights/PLE and exact expert ownership are established.
use super::EngineArgs;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::qwen4::Qwen4Config;
use cuteafd_loader::serving_capacity::qwen_cache_geometry;

use cuteafd_loader::serving_capacity::qwen_graphs::{qwen_admission, qwen_graph_pool, qwen_startup_graphs,
    QwenAdmissionInputs, QWEN_GRAPH_BYTES_PER_GRAPH};

/// The shared admission inputs (`serving_capacity::qwen_graphs::qwen_admission`) at serve's arguments.
#[allow(clippy::too_many_arguments)]
pub(super) fn inputs<'a>(args: &EngineArgs, cfg: &'a Qwen4Config, layers: usize, mtp: bool,
    manifest: Option<&'a serde_json::Value>, ple: Option<(u64, bool)>, future_expert_bytes: u64) -> Result<QwenAdmissionInputs<'a>> {
    let geometry = qwen_cache_geometry(cfg, layers, mtp)?;
    let costs = cuteafd_loader::plan::layout::family_costs("qwen4");
    let marks = args.planner_prefix_bytes.unwrap_or(geometry.ranks[0].retained_mark_bytes * costs.mark_slots);
    let logits = if args.full_prefill_logits { cuteafd_loader::plan::layout::full_prefill_logits_bytes(
        "qwen4", args.prefill_rows as u64, cfg.vocab_size as u64) } else { 0 };
    Ok(QwenAdmissionInputs { cfg, layers, mtp, manifest, prefill_rows: args.prefill_rows as u64,
        slots: args.slots as u64, mark_bytes: marks, full_prefill_logits: logits, ple, future_expert_bytes,
        headroom: cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes.max(3 << 30) })
}

/// Qwen's KV pool from one sample of free memory after weights, the PLE table and resident experts:
/// the shared admission's fixed items and, with startup graphs, the largest pool whose own graph set
/// fits beside them.
pub(super) fn pool_tokens(library: &NativeLibrary, args: &EngineArgs, inputs: &QwenAdmissionInputs<'_>) -> Result<usize> {
    let admission = qwen_admission(inputs)?;
    let target = cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS;
    let requested = (args.pool_tokens > 0).then_some(args.pool_tokens as u64);
    let enabled = super::engine::startup_graphs_enabled(
        std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref(),
        std::env::var("CUTEAFD_QWEN4_STARTUP_GRAPHS").ok().as_deref());
    let costs = cuteafd_loader::plan::layout::family_costs("qwen4");
    let (requested, graph_bytes) = if let Some((sequences, speculation)) = args.planner_graph_modes.filter(|_| enabled) {
        ensure!(sequences <= 16, "Qwen startup graphs support at most 16 concurrent sequences");
        let current = library.cuda_get_device()?;
        library.cuda_set_device(args.device)?;
        let sample = library.cuda_memory_info();
        library.cuda_set_device(current)?;
        let (available, _) = sample?;
        let (tokens, set) = qwen_graph_pool(available as u64, admission.fixed(), admission.per_token, target, requested,
            |tokens| qwen_startup_graphs(args.max_context, tokens as usize, inputs.cfg.dense_context(), sequences,
                speculation, inputs.layers)).map_err(anyhow::Error::msg)?;
        tracing::info!(pool_tokens = tokens, graphs = set.ranks[0].executables, graph_reserve_bytes = set.bytes(0),
            bytes_per_graph = QWEN_GRAPH_BYTES_PER_GRAPH, headroom_bytes = admission.headroom,
            items = ?admission.items, "Qwen graph-aware KV admission before allocation");
        (Some(tokens), set.bytes(0))
    } else { (requested, costs.graph_bytes[0]) };
    let tokens = crate::shared::memory_report::admitted_pool_tokens(library,
        &[crate::shared::memory_report::KvDevice { device: args.device,
            bytes_per_token: admission.per_token, reserve_bytes: admission.fixed().checked_add(graph_bytes)
                .context("Qwen admission reserve overflow")? }], 256, target, requested)?;
    Ok(usize::try_from(tokens)?)
}

#[cfg(test)]
mod tests {
    use cuteafd_loader::serving_capacity::qwen_graphs::*;

    fn set(count: u64) -> Option<cuteafd_loader::placement::GraphSet> {
        Some(cuteafd_loader::placement::GraphSet::new(&[count], QWEN_GRAPH_BYTES_PER_GRAPH, QWEN_GRAPH_MARGIN_PERCENT,
            QWEN_GRAPH_MARGIN_BYTES, 0, cuteafd_loader::placement::Lifetime::Startup))
    }

    #[test]
    fn auto_admission_subtracts_measured_graph_reserve_and_separate_headroom() {
        let (available, headroom, per_token) = (4_u64 << 30, 1_u64 << 30, 4096);
        let (tokens, graphs) = qwen_graph_pool(available, headroom, per_token, 2_097_152, None, |_| set(12397)).unwrap();
        let reserve = graphs.bytes(0);
        assert_eq!(reserve, 12397 * QWEN_GRAPH_BYTES_PER_GRAPH + QWEN_GRAPH_MARGIN_BYTES);
        assert_eq!(tokens, (available - headroom - reserve) / per_token / 256 * 256);
        assert!(tokens * per_token + headroom + reserve <= available);
    }

    #[test]
    fn admission_recounts_geometry_after_pool_shrinks_and_rejects_fixed_overcommit() {
        let count = |tokens| set(if tokens >= 32768 { 21560 } else { 7840 });
        let (tokens, graphs) = qwen_graph_pool(4 << 30, 512 << 20, 65536, 65536, None, count).unwrap();
        assert!(tokens < 32768);
        assert_eq!(graphs.ranks[0].executables, 7840);
        assert!(tokens * 65536 + (512 << 20) + graphs.bytes(0) <= 4 << 30);
        assert!(qwen_graph_pool(4 << 30, 512 << 20, 65536, 65536, Some(32768), count).is_err());
    }

    #[test]
    fn margin_and_fixed_pool_rounding_do_not_spend_headroom() {
        let measured = 21560 * QWEN_GRAPH_BYTES_PER_GRAPH;
        assert_eq!(set(21560).unwrap().bytes(0), measured + measured.div_ceil(10));
        let (tokens, graphs) = qwen_graph_pool(8 << 30, 3 << 30, 28422, 2097152, Some(73727), |_| set(12397)).unwrap();
        assert_eq!(tokens, 73728);
        assert!(tokens * 28422 + graphs.bytes(0) + (3 << 30) <= 8 << 30);
        assert!(qwen_graph_pool(256 << 20, 0, 28422, 2097152, None, |_| set(12397)).is_err());
    }

    #[test]
    fn graph_counts_match_actual_pool_context_layers_and_serving_modes() {
        use super::super::engine::serving_graph_count;
        assert_eq!(serving_graph_count(8192, 73728, 2051, 16, true, 48).unwrap(), 12397);
        assert_eq!(serving_graph_count(8192, 2097152, 2051, 16, true, 48).unwrap(), 12397);
        assert_eq!(serving_graph_count(8192, 73728, 2051, 16, false, 48).unwrap(), 4508);
        assert_eq!(serving_graph_count(32768, 32768, 2051, 16, true, 48).unwrap(), 21560);
        assert!(serving_graph_count(32768, 4096, 2051, 16, true, 48).unwrap() < 21560);
        assert_eq!(qwen_startup_graphs(8192, 73728, 2051, 16, true, 48).unwrap().ranks[0].executables, 12397);
    }
}
