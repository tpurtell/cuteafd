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
                assert_eq!(model.endpoints.len(), 2 + usize::from(device));
                assert_eq!(model.endpoints[0].capacity, decode_rows as u32);
                assert_eq!(model.endpoints[1].capacity, 4096);
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
    let mut endpoints = vec![
        SparkEndpoint { ring_depth: None, capacity: u32::try_from(decode_rows)?, ingress: SparkIngress::Fp8K32 },
        SparkEndpoint { ring_depth: None, capacity: 4096, ingress: SparkIngress::Fp8K32 },
    ];
    if device_exchange {
        endpoints.push(SparkEndpoint { ring_depth: None, capacity: u32::try_from(decode_rows)?, ingress: SparkIngress::Fp8K32 });
    }
    Ok(cuteafd_transport::worker_selection::ModelAdmission {
        capacity: 4096, compiled_capacity: 4096, endpoints,
        rings: crate::shared::placement::spark_rings(64 << 20)?, execution_slots: 1
    })
}

/// Future pinned owners in the coordinator process, summed across its RTX ranks.
/// `units` includes sequence-tail padding and context parity rounding from PoolShape.
pub(crate) struct PinnedTables {
    pub units: u64,
    pub prefill_rows: u64,
    pub decode_rows: u64,
    pub waves: u64,
    pub ranks: u64,
    pub c128_width: u64,
    pub hidden: u64,
    pub topk: u64,
    pub host_embedding: bool,
    /// Compiled context extent before the pool-limited user context is clamped.
    pub context_extent: Option<u64>,
}

impl PinnedTables {
    pub fn named_bytes(&self) -> Result<Vec<(&'static str, u64)>> {
        let mul = |a: u64, b: u64| a.checked_mul(b).context("V4 pinned table multiplication overflow");
        let add = |a: u64, b: u64| a.checked_add(b).context("V4 pinned table addition overflow");
        ensure!(self.units > 0 && self.prefill_rows > 0 && self.decode_rows > 0
            && (1..=3).contains(&self.waves) && (1..=2).contains(&self.ranks)
            && self.c128_width > 0 && self.hidden > 0 && self.topk > 0,
            "invalid V4 pinned table geometry");
        let mut tables = Vec::new();
        for (prefill, rows, slots) in [(false, self.decode_rows, 1), (true, self.prefill_rows, self.waves)] {
            let copies = mul(slots, self.ranks)?;
            let ints = mul(rows, 4)?;
            // Match reserve_slot_uploads/step_buffers, with a floor per destination.
            let mut bounded = [ints, mul(rows, 8)?, mul(rows, 8)?, mul(ints, 128)?, ints,
                ints, ints, mul(ints, self.c128_width)?, ints].into_iter()
                .try_fold(0, |sum, bytes| add(sum, bytes.max(256)))?;
            bounded = add(bounded, mul(mul(add(rows, 2)?, 4)?.max(256), 18)?)?;
            tables.push((if prefill { "deepseek_v4/ordered-h2d/prefill-tables" }
                else { "deepseek_v4/ordered-h2d/decode-tables" }, mul(bounded, copies)?));
            let lists = if prefill { 1 } else { rows };
            tables.push((if prefill { "deepseek_v4/c4-page-table/prefill" }
                else { "deepseek_v4/c4-page-table/decode" },
                mul(mul(mul(self.units, lists)?, 4)?.max(256), copies)?));
            if self.host_embedding {
                tables.push((if prefill { "deepseek_v4/ordered-h2d/prefill-embedding" }
                    else { "deepseek_v4/ordered-h2d/decode-embedding" },
                    mul(mul(mul(rows, self.hidden)?, 8)?.max(256), slots)?));
            }
            // One HostAllocation per workspace, not per prefill slot; peer is a 256-byte dummy.
            let router_stride = add(mul(self.topk, 8)?, add(self.hidden, self.hidden / 32)?)?;
            tables.push((if prefill { "deepseek_v4/prefill-router-host" }
                else { "deepseek_v4/decode-router-host" },
                add(mul(rows, router_stride)?, mul(self.ranks - 1, 256)?)?));
        }
        if let Some(extent) = self.context_extent {
            ensure!(self.ranks == 2 && extent > 0, "V4 context tables need two RTX ranks and nonzero extent");
            let half = extent.div_ceil(256).div_ceil(2);
            let copies = mul(self.ranks, self.waves)?;
            // Each owner subview is reserved at its exact extent, without the device floor.
            tables.push(("deepseek_v4/ordered-h2d/c4-context-gather", mul(mul(half, 8)?, copies)?));
            let commits = add(self.prefill_rows.div_ceil(256), 1)?.div_ceil(2);
            tables.push(("deepseek_v4/ordered-h2d/c4-context-commit", mul(mul(commits, 96)?.max(256), copies)?));
        }
        tables.iter().try_fold(0, |sum, (_, bytes)| add(sum, *bytes))?;
        Ok(tables)
    }
}

#[cfg(test)]
mod pinned_tests {
    use super::PinnedTables;

    fn geometry() -> PinnedTables {
        PinnedTables { units: 8200, prefill_rows: 4096, decode_rows: 64, waves: 2,
            ranks: 2, c128_width: 8192, hidden: 4096, topk: 8,
            host_embedding: false, context_extent: Some(1_048_576) }
    }

    #[test]
    fn named_pinned_tables_match_startup_destinations() -> anyhow::Result<()> {
        for ranks in [1, 2] {
            for waves in [1, 2, 3] {
                for prefill_rows in [512, 2048, 4096] {
                    for decode_rows in [1, 64, 80] {
                        for host_embedding in [false, true] {
                            let g = PinnedTables { ranks, waves, prefill_rows, decode_rows, host_embedding,
                                context_extent: (ranks == 2).then_some(1_048_576), ..geometry() };
                            let named = g.named_bytes()?;
                            let sum: u64 = named.iter().map(|(_, bytes)| bytes).sum();
                            let mut expected = 0;
                            for (rows, slots, lists) in [(decode_rows, 1, decode_rows), (prefill_rows, waves, 1)] {
                                let buffers = [rows*4, rows*8, rows*8, rows*128*4, rows*4,
                                    rows*4, rows*4, rows*g.c128_width*4, rows*4, lists*g.units*4];
                                expected += ranks*slots*(buffers.into_iter().map(|b| b.max(256)).sum::<u64>()
                                    + 18*((rows+2)*4).max(256));
                                expected += rows*(g.topk*8+g.hidden+g.hidden/32)+(ranks-1)*256;
                                if host_embedding { expected += slots*(rows*g.hidden*8).max(256); }
                            }
                            if ranks == 2 {
                                expected += ranks*waves*(2*2048*4 + (96*(prefill_rows.div_ceil(256)+1).div_ceil(2)).max(256));
                            }
                            assert_eq!(sum, expected);
                            assert_eq!(named.iter().find(|(n, _)| *n == "deepseek_v4/c4-page-table/prefill").unwrap().1,
                                ranks*waves*8200*4);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn named_pinned_tables_preserve_small_destination_floors() -> anyhow::Result<()> {
        let g = PinnedTables { units: 1, decode_rows: 1, prefill_rows: 1, c128_width: 1,
            context_extent: Some(1), ..geometry() };
        let tables = g.named_bytes()?;
        let get = |name| tables.iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(get("deepseek_v4/c4-page-table/decode"), 2*256);
        assert_eq!(get("deepseek_v4/c4-page-table/prefill"), 4*256);
        assert_eq!(get("deepseek_v4/ordered-h2d/c4-context-gather"), 4*2*4);
        assert_eq!(get("deepseek_v4/ordered-h2d/c4-context-commit"), 4*256);
        Ok(())
    }

    #[test]
    fn named_pinned_tables_refuse_overflow_and_invalid_geometry() {
        for g in [PinnedTables { units: u64::MAX, ..geometry() },
            PinnedTables { hidden: u64::MAX, ..geometry() },
            PinnedTables { c128_width: u64::MAX, ..geometry() },
            PinnedTables { prefill_rows: u64::MAX, ..geometry() },
            PinnedTables { ranks: 0, ..geometry() }, PinnedTables { waves: 0, ..geometry() },
            PinnedTables { ranks: 1, ..geometry() }] {
            assert!(g.named_bytes().is_err());
        }
    }
}

pub(crate) fn host_memory(engine: &super::engine::Engine<'_>, args: &EngineArgs,
    prefix: Option<&crate::shared::prefix::PrefixArgs>, context_extent: usize)
    -> Result<Option<cuteafd_hostcache::config::Config>> {
    let layout = super::prefix::Dsv4Prefix::layout_of(engine)?;
    let host_enabled = v4::host_marks_enabled(engine.ranks(), engine.context_kv(), true);
    let config = if host_enabled {
        prefix.map(|p| p.resolve_host_config(layout, engine.max_context, engine.shape.sequences))
            .transpose()?.flatten()
    } else { None };
    let tables = PinnedTables { units: engine.shape.units as u64,
        prefill_rows: engine.prefill_rows as u64, decode_rows: engine.decode_rows as u64,
        waves: engine.prefill_waves as u64, ranks: engine.ranks() as u64,
        c128_width: engine.c128_width as u64, hidden: engine.cfg.dim as u64,
        topk: engine.cfg.n_activated_experts as u64, host_embedding: !engine.embedding.device_gather(),
        context_extent: engine.context_kv().then_some(context_extent as u64) }.named_bytes()?;
    let scoring_rows = if args.full_prefill_logits { engine.prefill_capacity() } else { 64 };
    let logits = cuteafd_core::memory_layout::host_logits_bytes(scoring_rows as u64,
        engine.prefill_rows as u64, engine.cfg.vocab_size as u64).context("V4 host logits overflow")?;
    crate::shared::memory::budget::admit_host_memory(config.as_ref().map_or(0, |c| c.bytes),
        &tables, logits, prefix.map_or(0, |p| p.host_cache_headroom_bytes))?;
    Ok(config)
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
        cuteafd_loader::serving_capacity::compiled_c128_width(inputs.manifest, inputs.family)? * 128, gpus, args.prefill_waves as u64, scratch)?;
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
        sequences: args.max_sequences as u64, prefill_rows: prefill, prefill_waves: args.prefill_waves as u64, decode_rows: decode,
        max_context: inputs.max_context as u64, reserve_bytes: (args.reserve_gib as u64) << 30, mark_slots,
        workspace: Some(workspace), experts, draft, expert_workspace, first_routed: routed.first_layer,
        experts_half, tp2_workspace, exchange_f32: inputs.exchange_f32,
        requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        onboard: if args.skip_routed_experts { Onboard::Auto } else { onboard },
        full_prefill_logits: 0, code_bytes: v4::code_bytes(inputs.cfg.dim, v4::code_experts(inputs.catalog), gpus),
        context, attention })?;
    Ok(request)
}
