//! Pure allocation contract; native scratch is injected as metadata queries.
use anyhow::{ensure, Context, Result};
use cuteafd_core::serving_capacity::MemoryReservation;
use super::{MimoV2Config, MimoKvFormats, MimoPrefillOutput};

/// One shared selector for planner and live admission; no family/topology depth.
pub fn prefill_shape(spark_ranks: usize, rows: Option<u64>) -> Result<crate::serving_capacity::PrefillShape> {
    use crate::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
    let selected = prefill_shape(spark_ranks, PrefillLimits::default(), PrefillOverrides {
        unit_rows: rows, ..Default::default()
    })?;
    Ok(selected.units())
}

pub fn workspace_shapes(rows: usize, lead: bool, _spark: bool, _waves: usize, _output: MimoPrefillOutput, _mtp: usize)
    -> Vec<(&'static str, bool, usize, bool)> {
    vec![("prefill", false, rows, lead), ("decode", true, 64, lead)]
}

pub fn tensor_bytes(label: &str, dimensions: &[usize]) -> Result<u64> {
    dimensions
        .iter()
        .try_fold(1u64, |bytes, &n| bytes.checked_mul(n as u64))
        .ok_or_else(|| anyhow::anyhow!("{label}: allocation size overflows"))
}

pub fn allocation_bytes(label: &str, dimensions: &[usize]) -> Result<u64> {
    Ok(tensor_bytes(label, dimensions)?.max(256))
}

pub struct DraftReservations {
    pub steady: Vec<MemoryReservation>,
    /// Selected weights and FP8 scratch coexist with one drained source
    /// matrix during packing, after target KV but before target workspaces.
    /// Empty for BF16-only, which needs no conversion staging.
    pub packing: Vec<MemoryReservation>,
}

/// Validate source tensor headers, then reserve the immutable selected
/// representation from the same pure layout consumed by the actual loader.
/// Metadata/native scratch queries do not allocate tensor storage or modules.
#[derive(Debug, Clone, Copy)]
pub enum DraftScratch {
    Fp8 {
        rows: usize,
        k: usize,
        n: usize,
    },
    Attention {
        sequences: usize,
        heads: usize,
        kv_heads: usize,
        block: usize,
        keys: usize,
    },
    Topk {
        rows: usize,
    },
}

pub fn draft_reservations_with(
    cfg: &super::draft_config::DflashConfig,
    headers: &[crate::SafetensorsTensorMetadata],
    mode: crate::families::mimo_v2::draft_representation::MimoDraftRepresentation,
    capacity: crate::families::mimo_v2::draft_representation::MimoDraftCapacity,
    mut scratch: impl FnMut(DraftScratch) -> Result<usize>,
) -> Result<DraftReservations> {
    use super::draft_config::{RING, TAP_ROWS};
    use cuteafd_core::DType;
    const VOCABULARY_HEAD_WORKSPACE: usize = 4 << 20;
    use crate::families::mimo_v2::draft_representation::MimoDraftCapacity;
    let slots = capacity.context_slots;
    let max_sequences = capacity.max_batch_sequences;
    let expected = MimoDraftCapacity { kv_cache: capacity.kv_cache,
        ..MimoDraftCapacity::new(slots, max_sequences, cfg.block)? };
    ensure!(
        capacity == expected,
        "DFlash capacity does not match its configured block extent"
    );
    ensure!(
        slots > 0
            && max_sequences > 0
            && max_sequences <= slots
            && cfg.block > 1
            && cfg.hidden > 0
            && cfg.intermediate > 0
            && cfg.layers > 0
            && cfg.heads > 0
            && cfg.kv_heads > 0
            && cfg.head_dim == 128
            && cfg.vocab > 0
            && !cfg.taps.is_empty(),
        "invalid DFlash admission geometry"
    );
    let headers: std::collections::HashMap<_, _> =
        headers.iter().map(|t| (t.name.as_str(), t)).collect();
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    let width = |heads: usize| {
        heads
            .checked_mul(128)
            .ok_or_else(|| anyhow::anyhow!("DFlash head width overflows"))
    };
    let (attention, kv) = (width(cfg.heads)?, width(cfg.kv_heads)?);
    let two_kv = kv.checked_mul(2).context("DFlash KV width")?;
    let qkv = attention.checked_add(two_kv).context("DFlash QKV width")?;
    let taps = cfg.taps.len().checked_mul(h).context("DFlash tap width")?;
    let source = |name: &str, shape: &[usize]| -> Result<()> {
        let t = headers
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("DFlash missing tensor {name}"))?;
        let bytes = tensor_bytes(name, &shape.iter().copied().chain([2]).collect::<Vec<_>>())?;
        ensure!(
            t.dtype == DType::Bf16 && t.shape == shape && t.byte_length == bytes,
            "DFlash {name}: expected BF16 {shape:?} ({bytes}B), found {:?} {:?} ({}B)",
            t.dtype,
            t.shape,
            t.byte_length
        );
        Ok(())
    };
    source("fc.weight", &[h, taps])?;
    source("hidden_norm.weight", &[h])?;
    source("norm.weight", &[h])?;
    for layer in 0..cfg.layers {
        let p = format!("layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            source(&format!("{p}.{norm}.weight"), &[h])?;
        }
        for (part, rows) in [("q", attention), ("k", kv), ("v", kv)] {
            source(&format!("{p}.self_attn.{part}_proj.weight"), &[rows, h])?;
        }
        for part in ["q", "k"] {
            source(&format!("{p}.self_attn.{part}_norm.weight"), &[128])?;
        }
        if cfg.sinks {
            source(&format!("{p}.self_attn.attention_sink_bias"), &[cfg.heads])?;
        }
        source(&format!("{p}.self_attn.o_proj.weight"), &[h, attention])?;
        for part in ["gate", "up"] {
            source(&format!("{p}.mlp.{part}_proj.weight"), &[inter, h])?;
        }
        source(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?;
    }
    let layout = cfg.runtime_layout(mode, capacity)?;
    layout.weights.loading_peak_bytes()?;
    let mut costs = [
        ("bf16_values", layout.weights.bf16_values),
        ("fp8_values", layout.weights.fp8_values),
        ("fp8_scales", layout.weights.fp8_scales),
        ("small_bf16", layout.weights.small_bf16),
        ("head_fp8_values", layout.weights.head_fp8_values),
        ("head_fp8_scales", layout.weights.head_fp8_scales),
    ]
    .into_iter()
    .filter(|(_, bytes)| *bytes > 0)
    .map(|(name, bytes)| MemoryReservation {
        name: format!("draft.weights.{name}"),
        bytes,
    })
    .collect::<Vec<_>>();
    let mut packing = costs.clone();
    if let Some(fp8) = &layout.fp8_scratch {
        let bytes = fp8
            .shapes
            .iter()
            .map(|shape| {
                scratch(DraftScratch::Fp8 {
                    rows: fp8.rows,
                    k: usize::try_from(shape.k)?,
                    n: usize::try_from(shape.n)?,
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0)
            .max(256) as u64;
        let cost = MemoryReservation {
            name: "draft.fp8_scratch".into(),
            bytes,
        };
        costs.push(cost.clone());
        packing.push(cost);
    }
    if layout.weights.max_load_staging > 0 {
        packing.push(MemoryReservation {
            name: "loading.draft_source_matrix".into(),
            bytes: layout.weights.max_load_staging,
        });
    } else {
        packing.clear();
    }
    let mut reserve = |name: String, dims: &[usize]| -> Result<()> {
        costs.push(MemoryReservation {
            bytes: allocation_bytes(&name, dims)?,
            name,
        });
        Ok(())
    };
    for layer in 0..cfg.layers {
        for kind in ["k", "v"] {
            reserve(
                format!("draft.layer{layer}.{kind}_ring"),
                &[slots, RING, capacity.kv_cache.plane_record_bytes(cfg.kv_heads, cfg.head_dim)],
            )?;
        }
    }
    for (name, dims) in [
        ("taps", vec![TAP_ROWS, taps, 2]),
        ("fused", vec![TAP_ROWS, h, 2]),
        ("fused_norm", vec![TAP_ROWS, h, 2]),
        ("context_kv", vec![TAP_ROWS, 2, kv, 2]),
        ("context_positions", vec![TAP_ROWS, 8]),
        ("context_slots", vec![TAP_ROWS, 4]),
    ] {
        reserve(format!("draft.{name}"), &dims)?;
    }
    let rows = capacity.block_rows;
    let drafted = max_sequences
        .checked_mul(cfg.block - 1)
        .context("DFlash candidate rows")?;
    for (name, dims) in [
        ("head_workspace", vec![VOCABULARY_HEAD_WORKSPACE]),
        ("h", vec![rows, h, 2]),
        ("n", vec![rows, h, 2]),
        ("qkv", vec![rows, qkv, 2]),
        ("q", vec![rows, attention, 2]),
        ("k", vec![rows, kv, 2]),
        ("v", vec![rows, kv, 2]),
        ("attn", vec![rows, attention, 2]),
        ("delta", vec![rows, h, 2]),
        ("gate_up", vec![rows, 2, inter, 2]),
        ("act", vec![rows, inter, 2]),
        ("logits", vec![rows, cfg.vocab, 4]),
        ("unary", vec![drafted, 16, 4]),
        ("candidates", vec![drafted, 16, 4]),
        ("positions", vec![rows, 8]),
        ("tables", vec![3, max_sequences, 4]),
        ("ids", vec![rows, 4]),
        (
            "attention_workspace",
            vec![scratch(DraftScratch::Attention {
                sequences: max_sequences,
                heads: cfg.heads,
                kv_heads: cfg.kv_heads,
                block: cfg.block,
                keys: RING.checked_add(cfg.block).context("DFlash key extent")?,
            })?],
        ),
        (
            "topk_workspace",
            vec![scratch(DraftScratch::Topk { rows: drafted })?],
        ),
    ] {
        reserve(format!("draft.workspace.{name}"), &dims)?;
    }
    drop(reserve);
    costs
        .iter()
        .try_fold(0u64, |sum, r| sum.checked_add(r.bytes))
        .context("DFlash reservation sum")?;
    Ok(DraftReservations {
        steady: costs,
        packing,
    })
}

pub fn workspace_scratch(
    cfg: &MimoV2Config,
    ranks: usize,
    rank: usize,
    decode: bool,
    kv: MimoKvFormats,
    fp8_output: bool,
    mut query: impl FnMut(&str) -> Result<u64>,
) -> Result<u64> {
    ensure!(
        [1, 2].contains(&ranks) && rank < ranks,
        "invalid MiMo workspace rank {rank}/{ranks}"
    );
    let family = cfg.program_family()?;
    let split_family = cfg.head_split(ranks)?.program_family()?;
    let lead = rank == 0;
    let mut scratch = query(&format!("{family}_router_scores"))?;
    let families: &[&str] = match (ranks, lead) {
        (1, _) => &[family],
        (_, true) => &[family, split_family],
        (_, false) => &[split_family],
    };
    let (cap, mode) = if decode {
        ("m64", "decode")
    } else {
        ("m4096", "prefill")
    };
    for family in families {
        let swa = kv.sliding.program_tag();
        let kv = kv.full.program_tag();
        for name in [
            format!("{family}_full_producer{kv}_{cap}"),
            format!("{family}_swa_producer{swa}_{cap}"),
            format!("{family}_full_attention{kv}_{mode}_{cap}"),
            format!("{family}_swa_attention{swa}_{mode}_{cap}"),
            format!("{family}_ffn_{cap}"),
        ] {
            scratch = scratch.max(query(&name)?);
        }
        if fp8_output {
            scratch = scratch.max(query(&format!("{family}_o_w8_{cap}"))?);
        }
    }
    Ok(scratch)
}

/// Peer exchange slots follow the same lanes as the selected expert backend.
pub fn peer_slots(waves: usize) -> usize { 4 + 4 * waves }

pub fn peer_slot_bytes(rows: usize, waves: usize, hidden: usize) -> Result<Vec<usize>> {
    let bytes = |rows| usize::try_from(allocation_bytes("MiMo peer slot", &[rows, hidden, 2])?)
        .context("MiMo peer slot exceeds process size");
    let decode = bytes(super::decode_graph::MIMO_DECODE_ROWS)?;
    let prefill = bytes(rows)?;
    Ok(std::iter::repeat_n(decode, 4).chain(std::iter::repeat_n(prefill, 4 * waves)).collect())
}

/// The 97% occupancy ceiling and Spark's drained 64 MiB startup probe.
pub fn headroom_bytes(total: u64, spark_probe: bool) -> u64 {
    (total - total * 97 / 100 + if spark_probe { 64 << 20 } else { 0 })
        .max(cuteafd_core::serving_capacity::small_card_headroom_bytes(total))
}

pub fn spark_intake_probe_bytes(spark: bool, configured: Option<&str>) -> Result<u64> {
    if !spark { return Ok(0); }
    match configured {
        None | Some("auto" | "gpu") => Ok(64 << 20),
        Some("host" | "pinned") => Ok(0),
        Some(other) => anyhow::bail!("CUTEAFD_SPARK_INTAKE={other:?} is not auto, gpu, pinned or host"),
    }
}

pub fn attention_workspace_geometry(rank: usize, ranks: usize, decode: bool,
    all_target_layers_split: bool) -> super::MimoAttentionWorkspace {
    match (rank, ranks, decode, all_target_layers_split) {
        (1, 2, _, _) | (0, 2, false, true) => super::MimoAttentionWorkspace::PartitionedHeads { ranks: 2 },
        _ => super::MimoAttentionWorkspace::Global,
    }
}

pub fn workspace_options(cfg: &MimoV2Config, ranks: usize, rank: usize, rows: usize,
    context: usize, lanes: usize, spark: bool, expert_input: super::MimoExpertInput, kv: MimoKvFormats, mtp: usize,
    output: MimoPrefillOutput, fp8_output: bool,
    mut query: impl FnMut(&str) -> Result<u64>) -> Result<Vec<(String, super::MimoWorkspaceOptions)>> {
    workspace_shapes(rows, rank == 0, spark, lanes, output, mtp).into_iter()
        .map(|(name, decode, rows, with_head)| Ok((name.into(), super::MimoWorkspaceOptions {
            rows: rows as u64, waves: if decode { 1 } else { lanes as u64 }, decode, lead: rank == 0, with_head, spark: spark && rank == 0, expert_input,
            max_context: context as u64, pool_pages: 0, kv_cache: kv,
            attention: attention_workspace_geometry(rank, ranks, decode, true), prefill_output: output,
            native_scratch_bytes: workspace_scratch(cfg, ranks, rank, decode, kv, fp8_output, &mut query)?,
            head_workspace_bytes: 4 << 20,
        }))).collect()
}

/// Immutable endpoints: decode is one sequential m64 wave; prefill keeps the
/// selected unit capacity and the shared transport's full ring depth.
pub fn spark_endpoints(rows: usize, input: super::MimoExpertInput) -> Result<[cuteafd_core::SparkEndpoint; 2]> {
    use cuteafd_core::{SparkEndpoint, SparkIngress};
    let decode = u32::try_from(super::decode_graph::MIMO_DECODE_ROWS)?;
    let decode_ingress = if input.bf16(true) { SparkIngress::Bf16 } else { SparkIngress::Fp8K32 };
    let prefill_ingress = if input.bf16(false) { SparkIngress::Bf16 } else { SparkIngress::Fp8K32 };
    Ok([SparkEndpoint { capacity: decode, ingress: decode_ingress, ring_depth: Some(1), partial_reply_bytes: None },
        SparkEndpoint { capacity: u32::try_from(rows.max(decode as usize))?, ingress: prefill_ingress, ring_depth: None, partial_reply_bytes: None }])
}

pub fn transport_reservations(cfg: &MimoV2Config, ranks: usize, rank: usize,
    rows: usize, lanes: usize, spark_ranks: usize) -> Result<Vec<MemoryReservation>> {
    let mut costs = Vec::new();
    if ranks == 2 {
        costs.push(MemoryReservation { name: "transport.peer_receive_slots".into(),
            bytes: peer_slot_bytes(rows, lanes, cfg.hidden)?.into_iter().try_fold(0u64, |sum, bytes|
                sum.checked_add(bytes as u64).context("MiMo peer receive slots overflow"))? });
        costs.push(MemoryReservation { name: "transport.peer_control".into(), bytes: ((peer_slots(lanes) + 1) * 16).max(256) as u64 + 256 });
    }
    if rank == 0 && spark_ranks > 0 {
        costs.push(MemoryReservation { name: "transport.spark_intake_planes".into(),
            bytes: tensor_bytes("MiMo prefill intake planes", &[lanes, spark_ranks, rows.max(super::decode_graph::MIMO_DECODE_ROWS), cfg.hidden, 2])?
                .checked_add(tensor_bytes("MiMo decode intake planes", &[super::decode_graph::MIMO_DECODE_ROWS, spark_ranks, cfg.hidden, 2])?)
                .context("MiMo intake planes overflow")? });
    }
    Ok(costs)
}

pub fn draft_prefix_reservations(enabled: bool, mark: u64, slots: u64, rings: u64) -> Result<Vec<MemoryReservation>> {
    if !enabled { return Ok(Vec::new()); }
    Ok(vec![MemoryReservation { name: "prefix.dflash_context_marks".into(),
        bytes: mark.checked_mul(slots).context("DFlash prefix marks overflow")? },
        MemoryReservation { name: "prefix.dflash_valid_floor_transfer".into(),
            bytes: rings.checked_mul(8).context("DFlash ring metadata overflow")? }])
}

/// Pure counterparts of the handwritten draft kernels' workspace queries.
/// SM count is physical, not the planner's memory-cap override.
pub fn draft_scratch_bytes(shape: DraftScratch, sm_count: usize) -> Result<usize> {
    let align = |n: usize| n.div_ceil(256) * 256;
    Ok(match shape {
        DraftScratch::Fp8 { rows, k, n } => {
            ensure!(sm_count > 0 && n >= 16 && k >= 128, "invalid FP8 scratch geometry");
            let tiles = n / 16;
            let kbs = k / 128;
            let wanted = (sm_count * 16).div_ceil(tiles).clamp(1, kbs);
            let splits = kbs.div_ceil(kbs.div_ceil(wanted));
            align(2 * 64 * 4) + align(k.div_ceil(256) * 64 * 4) + align(k * 64 * 2)
                + if splits > 1 { align(splits * rows.min(64) * n * 4) } else { 0 }
        }
        DraftScratch::Attention { sequences, heads, kv_heads, block, keys } => {
            ensure!(kv_heads > 0 && heads > 0 && heads % kv_heads == 0, "invalid draft attention geometry");
            let queries = (heads / kv_heads).checked_mul(block).context("draft query extent")?;
            usize::try_from(crate::serving_capacity::draft_attention_workspace_bytes(
                sequences as u64, kv_heads as u64, queries.div_ceil(32) as u64, keys as u64,
                128, sm_count as u64)?)?
        }
        DraftScratch::Topk { rows } => usize::try_from(tensor_bytes("DFlash topk scratch", &[rows, 64, 16, 8])?)?,
    })
}

/// Still-lazy coordinator host owners, shared by live and offline admission.
/// Intake/egress and synchronous staging are safe upper bounds; table extents
/// use the same checked layouts the engine allocates.
pub fn future_host_workspaces(cfg: &MimoV2Config, rows: usize, waves: usize, ranks: usize,
    pool_tokens: u64, context: usize, spark_ranks: usize, expert_input: super::MimoExpertInput,
    kv: MimoKvFormats, output: MimoPrefillOutput, serving: bool,
    diagnostic_rows: usize, diagnostic_layers: usize) -> Result<Vec<(String, u64)>> {
    let mut demands = vec![("mimo/mtp-staging".into(), 1 << 20)];
    let mut sync = 64u64 << 20; // embedding loader's pinned chunk / synchronous copy upper bound
    for rank in 0..ranks {
        for (name, mut options) in workspace_options(cfg, ranks, rank, rows, context, waves,
            spark_ranks > 0, expert_input, kv, 0, output, false, |_| Ok(0))? {
            options.pool_pages = pool_tokens.div_ceil(64);
            let layout = super::MimoWorkspaceLayout::new(cfg, options)?;
            let label = format!("mimo/owner-{rank}/{name}");
            demands.push((format!("{label}.ordered-tables"), layout.ordered_host_bytes(options.waves)?));
            sync = sync.max(layout.page_table).max(layout.step_inputs).max(layout.ids);
        }
    }
    demands.push(("mimo/staging/sync-h2d-upper-bound".into(), sync));
    if diagnostic_layers > 0 {
        demands.push(("mimo/router-diagnostics".into(), tensor_bytes("MiMo route checks",
            &[diagnostic_rows.max(1), cfg.topk, 8, ranks, diagnostic_layers])?));
    }
    if serving {
        let mask = allocation_bytes("MiMo sampling mask", &[64, cfg.vocab_size.div_ceil(32), 4])?;
        demands.push(("mimo/sampling".into(), mask.checked_add(64 * (64 + 5 * 4 + 12))
            .context("MiMo sampling host overflow")?));
    }
    if spark_ranks > 0 {
        use cuteafd_core::{ExpertGeometry, SparkRings};
        let geometry = ExpertGeometry { hidden: u32::try_from(cfg.hidden)?, experts: u32::try_from(cfg.experts)?,
            topk: u32::try_from(cfg.topk)?, intermediate: u32::try_from(cfg.moe_intermediate)?, layers: u32::try_from(cfg.layers)? };
        let endpoints = spark_endpoints(rows, expert_input)?;
        let mut ring = 0usize;
        let mut prefix = 0usize;
        for endpoint in endpoints {
            let (request, response) = endpoint.wire_bytes(geometry).context("MiMo host wire overflow")?;
            let rings = SparkRings { max_frame_bytes: (64 << 20).max(request).max(response), depth: 8,
                slot_bytes: 8 << 20, alignment: 4096 };
            ring = ring.checked_add(endpoint.ring_bytes(geometry, rings).context("MiMo host rings overflow")?)
                .context("MiMo host rings overflow")?;
            prefix = prefix.checked_add(endpoint.response_prefix_bytes().context("MiMo response prefix")?)
                .context("MiMo response prefixes overflow")?;
        }
        demands.push(("mimo/transport/rdma-rings".into(), tensor_bytes("MiMo coordinator rings", &[ring, spark_ranks])?));
        demands.push(("mimo/transport/response-prefix".into(), tensor_bytes("MiMo response metadata", &[prefix, spark_ranks])?));
        let decode_rows = endpoints[0].capacity as usize;
        let prefill_rows = endpoints[1].capacity as usize;
        let intake_rows = prefill_rows.checked_mul(waves).and_then(|n| n.checked_add(decode_rows))
            .context("MiMo host intake rows overflow")?;
        demands.push(("mimo/spark-intake-upper-bound".into(), tensor_bytes("MiMo host intake", &[intake_rows, cfg.hidden, 2, spark_ranks])?));
        let request_bytes = |rows: usize, decode: bool| -> Result<usize> {
            Ok(geometry.compact_wire_bytes(u32::try_from(rows)?, expert_input.bf16(decode))
                .context("MiMo registered egress geometry overflow")?.0)
        };
        let decode_egress = request_bytes(decode_rows, true)?;
        let egress = request_bytes(prefill_rows, false)?.checked_mul(waves)
            .and_then(|n| n.checked_add(decode_egress))
            .context("MiMo registered egress slots overflow")?;
        demands.push(("mimo/transport/egress".into(), egress as u64));
    }
    Ok(demands)
}

pub fn sampling_wave_bytes(capacity: usize, vocab: usize) -> usize {
    capacity * (64 + 64 + 2 * vocab.div_ceil(32) * 4 + 2048 * 4 + 256 * 12 + 8 * 4)
}
pub fn sampling_reservations(vocab: usize) -> Result<Vec<MemoryReservation>> {
    Ok(vec![MemoryReservation { name: "sampling.selector_output".into(), bytes: 64 * 12 },
        MemoryReservation { name: "sampling.target_wave".into(), bytes: sampling_wave_bytes(64, vocab) as u64 }])
}
