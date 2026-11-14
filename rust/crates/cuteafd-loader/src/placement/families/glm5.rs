//! GLM 5.3 coordinator admission shared by the planner and runtime. On two
//! GPUs the executor runs `heads` (replicated MLA/DSA KV, head-split
//! attention) or `context` (K3: the same head split for the projections, KV
//! pages owned by logical-page parity, a q/candidate/partial exchange per
//! layer), or explicit `layers` (whole attention and owner-local KV, with
//! split FFNs). `context` remains the default where its programs exist
//! ([`GlmContext`]); owned attention follows the shared prefill wave policy.
use crate::families::glm5::{names::Operand, GlmDsaConfig, GlmIndexer};
use crate::placement::*;
use crate::serving_capacity::{Exl3Epilogue};
use cuteafd_core::memory_layout::{Basis, Category};

pub mod admission;
pub mod buffers;
pub mod exchanges;
pub mod hooks;
pub mod local_physical;
pub mod weights;

pub const PAGE_ROWS: u64 = 64;
pub const DECODE_ROWS: u64 = 64;
pub const DEFAULT_PREFILL_LANES: u64 = 3;
pub const SKIP_RANKS: usize = 4;

/// Immutable phase/row pairs avoid pinned-source reuse and host synchronization.
pub const USAGE_MAX_ROWS: usize = 4096;
pub const USAGE_LOOKUP_BYTES: usize = 2 * (USAGE_MAX_ROWS + 1) * 8;

pub fn usage_metadata_components(waves: u64, maximum: u64)
    -> Result<Vec<planner::Component>, planner::Refusal> {
    use planner::*;
    if !(1..=4).contains(&waves) { return Err(Refusal::InvalidFact("GLM usage waves".into())); }
    let mut owners = vec![Allocation::buffer("glm.usage.lookup", Location::PINNED,
        USAGE_LOOKUP_BYTES as u64, Lifetime::TARGET, Category::Staging, "expert usage replay metadata")];
    for (kind, slots) in [("decode", 1), ("prefill", waves)] {
        for slot in 0..slots {
            owners.push(Allocation::buffer(format!("glm.usage.{kind}.{slot}"), Location::coordinator(0),
                256, Lifetime::TARGET, Category::Workspace, "expert usage replay metadata"));
        }
    }
    Ok(owners.into_iter().map(|owner| { let bytes = owner.bytes;
        Component::allocation(owner, ByteCurve::fixed(bytes, maximum), false, maximum) }).collect())
}

const FLOOR: u64 = 256;
const GIB: u64 = 1 << 30;

/// Scratch retained separately by each decode/prefill workspace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmScratch {
    pub programs: u64,
    pub topk: u64,
}

/// The token-split (`context`) programs of a build and their scratch, from
/// PROGRAMS.json or the loaded table (the planner and runtime pass the same).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmContext {
    /// `glm_scored_index_topk_decode_m64` scratch.
    pub topk_scratch: u64,
    /// Largest `glm_sparse_mla_partial_h*_s*_m64` scratch.
    pub partial_scratch: u64,
}

/// Programs `context` needs in the variant's namespace (`glm_*`, `ml4_*`), beside the split
/// partials (any `_s<N>` variant) and one staging gather per record contract.
pub const CONTEXT_PROGRAMS: [&str; 4] = ["scored_index_topk_decode_m64", "dsa_candidate_merge",
    "lse_combine2", "paged_staging_gather_index"];

/// `cfg`'s staging gather of record contract `kind` (`kv`, NoPE `kv_nope`).
pub fn context_gather(cfg: &GlmDsaConfig, kind: &str) -> String {
    format!("{}_paged_staging_gather_{}", cfg.program_family(),
        if kind == "sparse_mla_nope" { "kv_nope" } else { "kv" })
}

/// The `context` programs of `cfg`, for the fail-closed reason when a build's
/// manifest lacks them (issue #202).
pub fn context_program_list(cfg: &GlmDsaConfig) -> String {
    let family = cfg.program_family();
    let mut names: Vec<String> = CONTEXT_PROGRAMS.iter().map(|n| format!("{family}_{n}")).collect();
    for kind in cfg.attention_kinds() {
        names.push(context_gather(cfg, kind));
        names.push(format!("{family}_{kind}_partial_h0_s*_m64"));
        names.push(format!("{family}_{kind}_partial_h{}_s*_m64", cfg.heads / 2));
    }
    names.join(", ")
}

/// `cfg`'s `context` programs and scratch `lookup` (name -> scratch bytes) finds,
/// or None when this build lacks any of them (explicit `context` fails closed;
/// standard auto uses whole-layer owners without replicating the records).
pub fn context_programs<'n>(cfg: &GlmDsaConfig, names: impl IntoIterator<Item = &'n str> + Clone,
    lookup: impl Fn(&str) -> Option<u64>) -> Option<GlmContext> {
    let family = cfg.program_family();
    let partial = |kind: &str, begin: usize| names.clone().into_iter()
        .filter(|n| n.starts_with(&format!("{family}_{kind}_partial_h{begin}_s")) && n.ends_with("_m64"))
        .filter_map(&lookup).max();
    CONTEXT_PROGRAMS.iter().all(|n| lookup(&format!("{family}_{n}")).is_some()).then_some(())?;
    let mut partial_scratch = 0;
    for kind in cfg.attention_kinds() {
        lookup(&context_gather(cfg, kind))?;
        partial_scratch = partial_scratch.max(partial(kind, 0)?).max(partial(kind, cfg.heads / 2)?);
    }
    Some(GlmContext { topk_scratch: lookup(&format!("{family}_{}", CONTEXT_PROGRAMS[0]))?, partial_scratch })
}

/// Index records per page (64 keys) and their bytes (64 x 128 E4M3, 64 FP32 scales).
pub const INDEX_PAGE_BYTES: u64 = PAGE_ROWS * 132;
/// Latent record pages of GLM 5.x (64 x 656 B); [`record_page_bytes`] per variant.
pub const RECORD_PAGE_BYTES: u64 = PAGE_ROWS * 656;

/// The widest latent record page over `cfg`'s layers (staging reused across layers).
pub fn record_page_bytes(cfg: &GlmDsaConfig) -> u64 { PAGE_ROWS * cfg.max_record_bytes() as u64 }

/// Most pages one GPU commits per prefill lane of `rows` rows (a range of
/// rows spans one more page than it fills; every second page is this GPU's).
pub fn context_commit_pages(rows: u64) -> u64 {
    (rows.div_ceil(PAGE_ROWS) + 1).div_ceil(2)
}

/// Every allocation the `context` executor adds on one GPU beside the
/// exchange and the staging ([`ContextBuffers`]), in order; `glm5::engine`
/// allocates exactly this list (each with its 256-byte floor) and the solver
/// charges [`context_workspace_bytes`]. Decode reuses the step's own tables:
/// its page table holds the compact half table, its cache lengths the shard
/// lengths, its indices/lengths the merged local selection.
pub fn context_workspace(cfg: &GlmDsaConfig, extent: u64, prefill_rows: u64, lanes: u64, context: GlmContext)
    -> Vec<(&'static str, u64)> {
    let (rows, k, half_heads) = (DECODE_ROWS, cfg.index_topk as u64, cfg.heads as u64 / 2);
    let partial = rows * half_heads * (512 * 2 + 4);
    // History pages of either parity, as compact local-page lists.
    let half = extent.div_ceil(PAGE_ROWS).div_ceil(2);
    // Per lane: source offsets, destination offsets and sizes of each owned
    // page's records (one plane per record contract), index keys and index scales.
    let contracts = cfg.attention_kinds().len() as u64;
    let commit = 3 * (contracts + 2) * context_commit_pages(prefill_rows) * 8;
    let mut out = vec![("candidates", rows * k * 8), ("own-head partial", partial), ("peer-head partial", partial),
        ("scored top-k scratch", context.topk_scratch), ("partial scratch", context.partial_scratch),
        ("gather tables", 2 * half * 4)];
    out.extend((0..lanes).map(|_| ("commit blocks", commit)));
    out
}

/// The `context` buffers of a GLM 5.3 engine with `extent` compiled context and
/// `lanes` prefill lanes of `prefill_rows`: the solver's demand and the
/// engine's allocation (`ContextExchange` and staging) both come from this.
pub fn context_buffers(cfg: &GlmDsaConfig, extent: u64, prefill_rows: u64, lanes: u64, context: GlmContext)
    -> ContextBuffers {
    ContextBuffers { staging_unit_bytes: record_page_bytes(cfg) + INDEX_PAGE_BYTES, staging_unit_rows: PAGE_ROWS,
        // Own heads' absorbed BF16 query at the widest latent (576 with RoPE); a partial of
        // 32 heads and its LSE.
        query_row_bytes: cfg.heads as u64 / 2 * cfg.max_latent_dim() as u64 * 2,
        partial_row_bytes: cfg.heads as u64 / 2 * (512 * 2 + 4),
        candidate_row_bytes: cfg.index_topk as u64 * 8, compiled_extent: extent, decode_rows: DECODE_ROWS,
        // The gather-route prefill never exchanges; decode runs one lane.
        lanes: 1, workspace_bytes: context_workspace_bytes(cfg, extent, prefill_rows, lanes, context) }
}

/// The `context` programs a manifest carries (PROGRAMS.json), as [`context_programs`].
pub fn manifest_context(cfg: &GlmDsaConfig, manifest: &serde_json::Value, sms: usize,
    plan: &cuteafd_core::coordinator_programs::AttentionReductionPlan) -> Option<GlmContext> {
    let programs = manifest["programs"].as_array()?;
    let lookup = |name: &str| programs.iter().find(|p| p["name"].as_str() == Some(name))
        .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0));
    let names = programs.iter().filter_map(|p| p["name"].as_str());
    let mut context = context_programs(cfg, names, lookup)?;
    context.partial_scratch = cfg.attention_kinds().iter().flat_map(|kind| [0, cfg.heads / 2].map(|begin| (*kind, begin)))
        .map(|(kind, begin)| {
            let name = cuteafd_core::coordinator_programs::attention_partial_program_of(
                manifest, cfg.program_family(), kind, begin, DECODE_ROWS as usize, sms, plan).ok()?;
            lookup(&name)
        }).collect::<Option<Vec<_>>>()?.into_iter().max()?;
    Some(context)
}

/// Resolve context recipes and scratch on every participating card before choosing operand owners.
pub fn manifest_context_for_cards(cfg: &GlmDsaConfig, manifest: &serde_json::Value, sms: &[usize],
    plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> anyhow::Result<Option<GlmContext>> {
    let mut context: Option<GlmContext> = None;
    for &sm_count in sms {
        let mut selected = manifest.clone();
        cuteafd_core::coordinator_programs::select_card_plans(&mut selected, sm_count.try_into()?)?;
        let family = cuteafd_core::coordinator_programs::CoordinatorPrograms {
            family: cfg.program_family(), split_family: cfg.split_program_family() };
        cuteafd_core::coordinator_programs::select_core_view_plans(&mut selected, plan,
            |name| family.contains(name))?;
        let Some(card) = manifest_context(cfg, &selected, sm_count, &plan) else { return Ok(None) };
        context = Some(match context {
            Some(previous) => GlmContext { topk_scratch: previous.topk_scratch.max(card.topk_scratch),
                partial_scratch: previous.partial_scratch.max(card.partial_scratch) },
            None => card,
        });
    }
    Ok(context)
}

/// [`context_workspace`]'s bytes with the engine's allocation floor.
pub fn context_workspace_bytes(cfg: &GlmDsaConfig, extent: u64, prefill_rows: u64, lanes: u64, context: GlmContext)
    -> u64 {
    context_workspace(cfg, extent, prefill_rows, lanes, context).iter().map(|&(_, b)| b.max(FLOOR)).sum()
}

/// Program scratch selection mirrors `glm5::engine::workspace_here`. Optional
/// BF16/per-tensor programs participate when present, even in a mixed checkpoint.
pub fn step_scratch(cfg: &GlmDsaConfig, split: bool, decode: bool,
    lookup: impl Fn(&str) -> Option<u64>) -> anyhow::Result<GlmScratch> {
    let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
    if !cfg.variant.is_glm() {
        // Attention through the variant's stage programs (K6 core only): the sparse
        // MLA of both record kinds is the only scratch owner, whole or a head split's share.
        let family = cfg.program_family();
        let mut programs = 0;
        for tag in ["", "_nope"] {
            let name = format!("{family}_sparse_mla{tag}_{mode}_{cap}");
            programs = programs.max(lookup(&name).ok_or_else(|| anyhow::anyhow!("{family} step needs program {name}"))?);
            // Two GPUs share the step arena between whole-layer owners and, where the build
            // carries them, the head split's share programs (heads and context require them
            // through their own program checks).
            if let (true, Some(split)) = (split, cfg.split_program_family()) {
                programs = programs.max(lookup(&format!("{split}_sparse_mla{tag}_{mode}_{cap}")).unwrap_or(0));
            }
        }
        // The FP8 indexer query (a checkpoint keeping wq_b as shipped) shares the arena.
        programs = programs.max(lookup(&format!("{family}_index_q_{cap}")).unwrap_or(0));
        let name = format!("{family}_index_topk_{mode}_{cap}");
        return Ok(GlmScratch { programs,
            topk: lookup(&name).ok_or_else(|| anyhow::anyhow!("{family} step needs program {name}"))? });
    }
    let prefix = if split { "glm2" } else { "glm" };
    let divisor = if split { 2 } else { 1 };
    let (moe, dense) = (cfg.moe_intermediate / divisor, cfg.dense_intermediate / divisor);
    let mut programs = 0;
    for name in [format!("glm_index_producer_{cap}"), format!("{prefix}_producer_{cap}"),
        format!("{prefix}_sparse_mla_{mode}_{cap}"), format!("{prefix}_o_{cap}"),
        format!("{prefix}_ffn_i{moe}_{cap}"), format!("{prefix}_ffn_i{dense}_{cap}")] {
        programs = programs.max(lookup(&name).ok_or_else(|| anyhow::anyhow!("GLM step needs program {name}"))?);
    }
    let optional = [format!("glm_index_producer_bf16_{cap}"), format!("{prefix}_producer_bf16_{cap}"),
        format!("{prefix}_o_bf16_{cap}"), format!("{prefix}_ffn_i{moe}_bf16_{cap}"),
        format!("{prefix}_ffn_i{dense}_bf16_{cap}"),
        format!("{prefix}_ffn_i{dense}_pt_{cap}")];
    for name in optional { programs = programs.max(lookup(&name).unwrap_or(0)); }
    let name = format!("glm_index_topk_{mode}_{cap}");
    Ok(GlmScratch { programs, topk: lookup(&name).ok_or_else(|| anyhow::anyhow!("GLM step needs program {name}"))? })
}

/// Per-row buffers of a GLM step workspace, in allocation order: (name, bytes
/// per row). `lead` is rank 0 (rank 1 owns no router, wire or logits rows
/// unless a TP2 pair runs its local router). Page tables and logits are sized
/// separately ([`step_rows`]).
pub fn step_row_widths(cfg: &GlmDsaConfig, lead: bool, split: bool, router: bool) -> [(&'static str, u64); 19] {
    let h = cfg.hidden as u64;
    let heads = cfg.heads as u64 / if split { 2 } else { 1 };
    let routed = |width: u64| if lead || router { width } else { 0 };
    [("h", h * 2), ("x", h * 2), ("query", heads * 576 * 2), ("q_resid", cfg.q_lora_rank as u64 * 2),
        ("q_fp8", cfg.index_heads as u64 * cfg.index_head_dim as u64), ("head_weights", cfg.index_heads as u64 * 4),
        ("indices", cfg.index_topk as u64 * 4), ("lengths", 4), ("attn", heads * 512 * 2), ("delta", h * 2),
        ("positions", 8), ("slots", 8), ("cache_lengths", 4), ("router_logits", routed(cfg.experts as u64 * 4)),
        ("route_ids", routed(cfg.topk as u64 * 4)), ("route_weights", routed(cfg.topk as u64 * 4)),
        ("wire", routed(h + h / 32)), ("shared", h * 2), ("ids", 4)]
}

/// Logits rows a step workspace keeps for `rows`-row calls: every row for
/// decode or full logits, else at most [`DECODE_ROWS`].
pub fn logits_rows(rows: u64, decode: bool) -> u64 {
    if decode { rows } else { rows.min(DECODE_ROWS) }
}

/// Device bytes of a GLM step workspace's row buffers for `rows` rows (each
/// with the engine's 256-byte floor), its `select` buffer, and `logits` rows
/// of logits (rank 0).
pub fn step_rows(cfg: &GlmDsaConfig, rows: u64, lead: bool, split: bool, router: bool,
    logits: u64) -> Result<u64, PlacementError> {
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(PlacementError::Overflow("GLM step buffers"));
    let mut sum = 0u64;
    for (_, width) in step_row_widths(cfg, lead, split, router) {
        sum = sum.checked_add(if width == 0 { FLOOR } else { mul(rows, width)?.max(FLOOR) })
            .ok_or(PlacementError::Overflow("GLM step buffers"))?;
    }
    let logits = if lead { mul(logits, cfg.vocab_size as u64 * 4)? } else { FLOOR };
    [logits, mul(rows, 8)?].into_iter().try_fold(sum, |sum, bytes| sum.checked_add(bytes.max(FLOOR))
        .ok_or(PlacementError::Overflow("GLM step buffers")))
}

/// Device bytes a step workspace shares across its rows: program and top-k
/// scratch and the LM head workspace (rank 0).
pub fn head_workspace_bytes(lead: bool, native_head: bool) -> Option<u64> {
    if lead && native_head { None } else { Some(if lead { 4 << 20 } else { FLOOR }) }
}

pub fn step_shared(lead: bool, native_head: bool, scratch: GlmScratch) -> u64 {
    scratch.programs.max(FLOOR) + scratch.topk.max(FLOOR)
        + head_workspace_bytes(lead, native_head).unwrap_or(0)
}

/// Device bytes beside a step's pool-dependent page table. Every allocation
/// has the engine's 256-byte floor; pinned router staging is not GPU storage.
/// Peer local-router rows are a separate `local_extra_workspace` backend delta.
pub fn step_bytes(cfg: &GlmDsaConfig, rows: u64, lead: bool, split: bool,
    decode: bool, native_head: bool, scratch: GlmScratch) -> Result<u64, PlacementError> {
    step_rows(cfg, rows, lead, split, false, logits_rows(rows, decode))?
        .checked_add(step_shared(lead, native_head, scratch))
        .ok_or(PlacementError::Overflow("GLM step buffers"))
}

/// A GLM prefill's shape on one GPU: `slots` units of `unit_rows` in flight,
/// slot views into per-field arenas of [`prefill_span`] rows, and the
/// hook-local storage (program/top-k scratch, head) shared once; its
/// logits rows serve one unit at a time (`launch_head` per unit). Its bytes:
/// one step workspace's rows at the reserve.
pub fn prefill_bytes(cfg: &GlmDsaConfig, unit_rows: u64, slots: u64, lead: bool, split: bool, router: bool,
    native_head: bool, scratch: GlmScratch) -> Result<u64, PlacementError> {
    let span = prefill_span(unit_rows, slots);
    step_rows(cfg, span, lead, split, router, logits_rows(unit_rows, false))?
        .checked_add(step_shared(lead, native_head, scratch)).ok_or(PlacementError::Overflow("GLM prefill buffers"))
}

/// Rows of every per-slot prefill arena: `slots` units of `unit_rows`.
pub fn prefill_span(unit_rows: u64, slots: u64) -> u64 { slots.max(1) * unit_rows }

/// The engine's prefill slot views: slot `s` at rows `s * unit_rows` inside
/// the arenas of [`prefill_span`] rows.
pub fn slot_views(unit_rows: u64, slots: u64) -> Vec<std::ops::Range<u64>> {
    (0..slots.max(1)).map(|slot| slot * unit_rows..(slot + 1) * unit_rows).collect()
}

/// Whole-attention/indexer groups follow the solver's pure alternating rule.
pub fn attention_owners(cfg: &GlmDsaConfig, layers: usize) -> Result<Vec<u8>, PlacementError> {
    if layers == 0 || layers > cfg.layers || layers > cfg.indexers.len() {
        return Err(PlacementError::Inventory("GLM attention owner layer extent"));
    }
    super::super::attention::alternating_owners(
        colocate_groups(cfg, layers).into_iter().map(|group| (true, Some(group))), 0)
}

/// The colocation group of each layer: a layer joins the full indexer layer
/// whose top-k it reuses and (Mistral Large 4) the layer whose latent KV pages
/// it reads. Groups are the least layer of their union; GLM's are exactly its
/// index sources.
pub fn colocate_groups(cfg: &GlmDsaConfig, layers: usize) -> Vec<u16> {
    let mut parent: Vec<usize> = (0..layers).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; }
        x
    }
    for layer in 0..layers {
        for other in [cfg.index_source(layer), cfg.kv_source(layer)] {
            if other >= layers { continue; }
            let (a, b) = (root(&mut parent, layer), root(&mut parent, other));
            parent[a.max(b)] = a.min(b);
        }
    }
    (0..layers).map(|layer| root(&mut parent, layer) as u16).collect()
}

/// The attention placement a request resolves to before admission: the
/// request, except that auto on two GPUs for Mistral Large 4 means context
/// after its correctness and interleaved throughput qualification. Explicit
/// requests are preserved. GLM is unchanged (auto stays auto; the solver
/// picks context on two GPUs).
pub fn resolve_attention(cfg: &GlmDsaConfig, requested: Option<AttentionPlacement>, gpus: usize)
    -> Option<AttentionPlacement> {
    if requested.is_none() && gpus == 2 && !cfg.variant.is_glm() {
        Some(AttentionPlacement::Context)
    } else { requested }
}

/// Resolve auto before choosing owner-local operands; explicit requests never fall back.
pub fn resolve_context_attention(cfg: &GlmDsaConfig, requested: Option<AttentionPlacement>, gpus: usize,
    context_available: bool, layers_supported: bool) -> Result<Option<AttentionPlacement>, PlacementError> {
    if gpus == 2 && requested.is_none() && !context_available && cfg.split_program_family().is_some() {
        let programs = context_program_list(cfg);
        if !layers_supported {
            return Err(PlacementError::MissingContextPrograms { family: cfg.family_id(), programs });
        }
        tracing::warn!(family = cfg.family_id(), %programs, fallback = "layers",
            "auto context programs missing; falling back to whole-layer attention without KV replication");
        return Ok(Some(AttentionPlacement::Layers));
    }
    Ok(resolve_attention(cfg, requested, gpus))
}

/// Header-only draft storage shared by plan and serve (mode 0 is GLM's W8A16).
pub fn drafter_residency(snapshot: &std::path::Path, slots: usize, sequences: usize,
    sms: u64, bf16: bool, kv_cache: crate::families::qwen4::Qwen4KvCache) -> anyhow::Result<(u64, ResidentWeights)> {
    use crate::families::glm5::draft_representation::{draft_resident_bytes_with_mode, GlmDraftRepresentation};
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let residency = draft_resident_bytes_with_mode(&config, slots, sequences, sms,
        if bf16 { GlmDraftRepresentation::Bf16Only } else { GlmDraftRepresentation::Fp8Only }, 0, kv_cache)?;
    Ok((residency.owned.checked_add(residency.scratch)
        .ok_or_else(|| anyhow::anyhow!("GLM draft storage overflow"))?, residency.weights))
}

/// Largest transient BF16 source retained while packing one draft projection.
/// Mistral Large 4's bundled mtp_dspark on the shared MLA block-draft path: its
/// immutable runtime layout from the checkpoint's own params.json and text
/// config (every draft layer windowed, so the target pool does not size it).
/// Bytes ahead of a draft-window prefix mark (its captured frontier; the GLM prefix family).
pub const DRAFT_MARK_HEADER: usize = 16;

/// One prefix mark of a window-ring drafter: its header and one slot's draft windows.
pub fn draft_window_mark_bytes(layout: &crate::formats::dflash::MlaDraftRuntimeLayout) -> anyhow::Result<usize> {
    Ok(DRAFT_MARK_HEADER + layout.window_mark_bytes()?)
}

/// The mtp_dspark drafter's resident bytes beside `marks` prefix marks of its draft windows
/// (one slot's rings each; the warm state a prefix restore copies back).
pub fn ml4_draft_resident_bytes(layout: &crate::formats::dflash::MlaDraftRuntimeLayout, marks: usize)
    -> anyhow::Result<u64> {
    let windows = (draft_window_mark_bytes(layout)? as u64).checked_mul(marks as u64)
        .ok_or_else(|| anyhow::anyhow!("draft window marks overflow"))?;
    layout.resident_bytes()?.checked_add(windows).ok_or_else(|| anyhow::anyhow!("drafter bytes overflow"))
}

pub fn ml4_draft_layout(snapshot: &std::path::Path, slots: usize, sequences: usize, compiled_context: usize,
    sms: usize) -> anyhow::Result<crate::formats::dflash::MlaDraftRuntimeLayout> {
    use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation};
    use crate::families::mistral_large4::{dspark::Ml4DsparkConfig, Ml4Config};
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let target = Ml4Config::from_hf(&config)?;
    let params = crate::plan::checkpoint::read_json(&snapshot.join("params.json"))?;
    let dspark = Ml4DsparkConfig::from_params(&params, target.layers)?;
    let draft = crate::formats::dflash::MlaDflashConfig::ml4(&dspark, &target)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, dspark.depth)?;
    Ok(draft.runtime_layout(GlmDraftRepresentation::Bf16Only, capacity, compiled_context, compiled_context, sms, 0,
        cuteafd_core::MlaCacheFormat::Fp8)?)
}

pub fn drafter_staging(snapshot: &std::path::Path, slots: usize, sequences: usize,
    bf16: bool) -> anyhow::Result<u64> {
    use crate::families::glm5::draft_representation::{draft_geometry, GlmDraftCapacity,
        GlmDraftRepresentation, GlmDraftRuntimeLayout};
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let (geometry, block) = draft_geometry(&config)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, usize::try_from(block)?)?;
    Ok(GlmDraftRuntimeLayout::new(geometry,
        if bf16 { GlmDraftRepresentation::Bf16Only } else { GlmDraftRepresentation::Fp8Only },
        capacity, 2048)?.weights.max_load_staging)
}

pub fn manifest_scratch(cfg: &GlmDsaConfig, split: bool, manifest: &serde_json::Value)
    -> anyhow::Result<[GlmScratch; 2]> {
    let programs = manifest["programs"].as_array().ok_or_else(|| anyhow::anyhow!("program manifest has no programs"))?;
    let lookup = |name: &str| programs.iter().find(|p| p["name"].as_str() == Some(name))
        .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0));
    Ok([step_scratch(cfg, split, true, lookup)?, step_scratch(cfg, split, false, lookup)?])
}

/// Local package arithmetic, including its input and routed-partial representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmLocalFormat {
    Exl3,
    Exlr8,
    Nvfp4 { w4a4: bool },
}

/// Header-only local inventory. Runtime must attest the native ABI before execution.
#[derive(Debug, Clone)]
pub struct LocalInventory {
    pub package: std::path::PathBuf,
    pub format: GlmLocalFormat,
    pub experts: Vec<ExpertCost>,
    pub backend_workspace: u64,
    pub extra_workspace: [u64; 2],
    pub world: usize,
    pub unit_limit: usize,
}

impl LocalInventory {
    pub fn workspace(&self) -> [u64; 2] {
        std::array::from_fn(|rank| if rank < self.world {
            self.extra_workspace[rank] + self.backend_workspace
        } else { 0 })
    }

    pub fn workspace_for(&self, cfg: &GlmDsaConfig, rows: u64, lanes: u64,
        exchange_f32: bool) -> [u64; 2] {
        if self.world == 1 { return [self.backend_workspace, 0]; }
        local_extra_workspace(cfg.hidden as u64, cfg.experts as u64, cfg.topk as u64,
            rows, lanes, exchange_f32).map(|bytes| bytes + self.backend_workspace)
    }
}

fn local_extra_workspace(h: u64, experts: u64, topk: u64, rows: u64, lanes: u64,
    exchange_f32: bool) -> [u64; 2] {
    let dtype = if exchange_f32 { 4 } else { 2 };
    // Workspace rows: one arena whose per-slot views serve every admitted depth (#762).
    let span = prefill_span(rows, lanes);
    let payload = (DECODE_ROWS * h * dtype).max(FLOOR) + (span * h * dtype).max(FLOOR);
    let exchange_rows = rows.max(DECODE_ROWS);
    let routes = 2 * lanes * (exchange_rows * topk * 8).next_multiple_of(16)
        + ((2 * lanes + 1) * 16).max(FLOOR) + FLOOR;
    let wide_exchange = if exchange_f32 { 4 * lanes * exchange_rows * h * 2 } else { 0 };
    // Rank 1 normally owns 256-byte placeholders for logits, ids, weights and wire.
    let peer_routes = [experts * 4, topk * 4, topk * 4, h + h / 32].into_iter().map(|width|
        (DECODE_ROWS * width).max(FLOOR) - FLOOR
            + ((span * width).max(FLOOR) - FLOOR)).sum::<u64>();
    // These optional deltas open with the RTX arena; shared step/attention rows stay Always.
    let rows = [("local expert payloads", [payload; 2]), ("canonical route exchange", [routes; 2]),
        ("FP32 exchange widening", [wide_exchange; 2]),
        ("peer local router rows", [0, peer_routes])];
    let mut extra = [0; 2];
    for (_, bytes) in rows {
        for (rank, bytes) in bytes.into_iter().enumerate() { extra[rank] += bytes; }
    }
    extra
}

fn native_layer_resident(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors,
    ranks: usize, rank: usize, slicing: crate::formats::fp8_experts::Slicing) -> anyhow::Result<u64> {
    use crate::formats::fp8_experts::Fp8Projection;
    Fp8Projection::ALL.iter().try_fold(0u64, |sum, &projection| {
        let (weights, _) = tensors.slice_bytes_with(projection, ranks, rank, slicing)?;
        let weights = tensors.shape().experts.checked_mul(weights)
            .ok_or_else(|| anyhow::anyhow!("GLM expert weights overflow"))?.max(FLOOR as usize) as u64;
        let scales = tensors.scale_region_bytes_with(projection, ranks, rank, slicing)?
            .max(FLOOR as usize) as u64;
        sum.checked_add(weights).and_then(|n| n.checked_add(scales))
            .ok_or_else(|| anyhow::anyhow!("GLM expert resident storage overflow"))
    })
}

/// Physical Spark weight arenas for the selected routed suffix. The caller
/// supplies the qualified package's actual slicing, not a proportional tensor
/// share or an assumption based on the configured peer count.
pub fn spark_resident_bytes(catalog: &crate::OfficialV41Catalog, layers: &[usize],
    ranks: usize, slicing: crate::formats::fp8_experts::Slicing) -> anyhow::Result<Vec<u64>> {
    use crate::formats::fp8_experts::{ExpertFormat, Slicing};
    if layers.is_empty() {
        anyhow::ensure!(ranks == 0, "GLM empty Spark suffix must select zero ranks");
        return Ok(Vec::new());
    }
    // The routed geometry is the checkpoint's own (its config read by its family's
    // catalog); only the suffix and rank extents are checked here.
    let shape = catalog.routed_experts();
    anyhow::ensure!((1..=8).contains(&ranks)
        && shape.layers > shape.first_layer && layers.last() == Some(&(shape.layers - 1))
        && layers.iter().all(|&layer| (shape.first_layer..shape.layers).contains(&layer))
        && layers.windows(2).all(|w| w[0] + 1 == w[1]), "GLM Spark rank/selected-suffix geometry");
    if let Some(exl3) = catalog.exl3() {
        anyhow::ensure!(slicing == Slicing::Blocks(128), "GLM EXL3 needs disjoint whole-128-row Spark slices");
        return (0..ranks).map(|rank| layers.iter().try_fold(0u64, |sum, &layer| {
            let arena = exl3.balanced_residency(crate::V41Exl3Layer::Backbone(layer), ranks, rank)?
                .device_arena_layout()?.1 as u64;
            sum.checked_add(arena).ok_or_else(|| anyhow::anyhow!("GLM Spark EXL3 residency overflow"))
        })).collect();
    }
    let tensors = catalog.fp8().filter(|t| matches!(t.format(), ExpertFormat::Fp8Block128 | ExpertFormat::Nvfp4))
        .ok_or_else(|| anyhow::anyhow!("GLM Spark inventory needs EXL3, FP8 or native NVFP4 headers"))?;
    for &layer in layers { tensors.validate_layer(layer)?; }
    (0..ranks).map(|rank| layers.iter().try_fold(0u64, |sum, &layer| {
        let view = tensors.for_layer_metadata(layer)?;
        sum.checked_add(native_layer_resident(&view, ranks, rank, slicing)?)
            .ok_or_else(|| anyhow::anyhow!("GLM Spark residency overflow"))
    })).collect()
}

pub fn local_inventory(cfg: &GlmDsaConfig, catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    selected_layers: usize, rows: u64, lanes: u64, exchange_f32: bool) -> anyhow::Result<LocalInventory> {
    local_inventory_selected(cfg, catalog, manifest, selected_layers, rows, lanes, exchange_f32, true)
}

/// Explicit activation selection keeps package admission deterministic in tests.
/// Geometry, the package namespace and the router names are the checkpoint's
/// own: `cfg` (its config and variant) and the catalog read from its headers.
#[allow(clippy::too_many_arguments)]
pub fn local_inventory_selected(cfg: &GlmDsaConfig, catalog: &crate::OfficialV41Catalog,
    manifest: &std::path::Path, selected_layers: usize, rows: u64, lanes: u64, exchange_f32: bool, a4: bool)
    -> anyhow::Result<LocalInventory> {
    local_inventory_world(cfg, catalog, manifest, selected_layers, 2, rows, lanes, exchange_f32, a4)
}

#[allow(clippy::too_many_arguments)]
fn local_inventory_world(cfg: &GlmDsaConfig, catalog: &crate::OfficialV41Catalog,
    manifest: &std::path::Path, selected_layers: usize, world: usize, rows: u64, lanes: u64,
    exchange_f32: bool, a4: bool) -> anyhow::Result<LocalInventory> {
    let shape = catalog.routed_experts();
    anyhow::ensure!(shape.hidden == cfg.hidden && shape.intermediate == cfg.moe_intermediate
        && shape.experts == cfg.experts && shape.topk == cfg.topk && shape.first_layer == cfg.first_moe_layer,
        "routed expert catalog {shape:?} differs from the checkpoint config");
    anyhow::ensure!((1..=2).contains(&world) && shape.intermediate % world == 0 && (1..=4096).contains(&rows) && (1..=4).contains(&lanes)
        && (shape.first_layer..=shape.layers).contains(&selected_layers), "GLM TP2 geometry/layer extent");
    let Some(exl3) = catalog.exl3() else {
        anyhow::ensure!(world == 2, "GLM native NVFP4 local experts require two GPUs");
        anyhow::ensure!(cfg.variant.is_glm(), "{} routed experts serve TP2 only from its EXL3 publication",
            cfg.program_family());
        return nvfp4_inventory(catalog, manifest, selected_layers, rows, lanes, exchange_f32, a4);
    };
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let stem = format!("exl3-{}-k{tiers}", cfg.program_family());
    let layout = format!("rtx-tp{world}");
    let parent = manifest.parent().ok_or_else(|| anyhow::anyhow!("GLM program manifest parent"))?;
    let package = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)]
        .into_iter().map(|p| p.join(&layout)).find(|p| p.is_dir())
        .ok_or_else(|| anyhow::Error::new(MissingLocalPackage(format!("{stem}/{layout}"))))?;
    let capacities = cuteafd_core::EXL3_CAPACITIES.map(u64::from);
    let max_rows = rows.max(DECODE_ROWS);
    let compiled = capacities.into_iter().find(|&c| c >= max_rows).unwrap();
    let manifests = capacities.into_iter().filter(|&c| c <= compiled).map(|capacity| {
        let directory = package.join(format!("m{capacity}"));
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        anyhow::ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
            && value["intermediate"].as_u64() == Some(shape.intermediate as u64 / world as u64)
            && value["experts"].as_u64() == Some(shape.experts as u64)
            && value["top_k"].as_u64() == Some(shape.topk as u64)
            && value["capacity"].as_u64() == Some(capacity)
            && value["output_dtype"].as_str() == Some("fp32")
            && value["bits"] == serde_json::json!(exl3.decoder_tiers()),
            "GLM TP2 EXL3 capacity/geometry/tiers/output mismatch in {}", directory.display());
        value["trellis_lut"]["file"].as_str()
            .ok_or_else(|| anyhow::anyhow!("GLM TP2 missing trellis LUT asset name"))?;
        if value["direct"].as_bool() != Some(true) {
            value["route_preparation"]["manifest"].as_str()
                .ok_or_else(|| anyhow::anyhow!("GLM TP2 missing packed route export manifest"))?;
        }
        // Header-only planning needs geometry, not binaries. Exl3Execution opens
        // the libraries and attests ELF geometry and LUT hashes before execution.
        Ok(value)
    }).collect::<anyhow::Result<Vec<_>>>()?;
    let backend_workspace = crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::Sum)?
        + max_rows * shape.hidden as u64 * 4;
    let mut experts = Vec::new();
    for layer in shape.first_layer..selected_layers {
        let whole = exl3.residency(crate::V41Exl3Layer::Backbone(layer), 1, 0)?.device_arena_layout()?.1 as u64;
        let mut half = [Bytes2::default(); 2];
        for (rank, bytes) in half.iter_mut().enumerate() {
            bytes.resident = exl3.residency(crate::V41Exl3Layer::Backbone(layer), 2, rank)?.device_arena_layout()?.1 as u64;
        }
        // Rank 1 owns the router (and its bias, on variants that have one).
        for operand in [Operand::Router, Operand::RouterBias] {
            if let Some(name) = cfg.weight(layer, operand) {
                half[1].resident += catalog.tensor(&name)?.metadata.byte_length.max(FLOOR);
            }
        }
        experts.push(ExpertCost { whole: Bytes2 { resident: whole, staging: 0 }, half,
            tp2: true, whole_supported: true, spark_ok: true });
    }
    Ok(LocalInventory { package, format: GlmLocalFormat::Exl3, experts, backend_workspace, world, unit_limit: 0,
        extra_workspace: if world == 1 { [0; 2] } else {
            local_extra_workspace(shape.hidden as u64, shape.experts as u64,
                shape.topk as u64, rows, lanes, exchange_f32)
        } })
}

fn nvfp4_inventory(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    selected_layers: usize, rows: u64, lanes: u64, exchange_f32: bool, a4: bool)
    -> anyhow::Result<LocalInventory> {
    use crate::formats::fp8_experts::{ExpertFormat, Slicing};
    let tensors = catalog.fp8().filter(|t| t.format() == ExpertFormat::Nvfp4)
        .ok_or_else(|| anyhow::anyhow!("GLM RTX TP2 needs EXL3 or native ModelOpt NVFP4 expert tensors"))?;
    anyhow::ensure!(!exchange_f32,
        "GLM NVFP4 TP2 emits BF16 routed partials; use BF16 exchange (no BF16-to-FP32 partial adapter)");
    let parent = manifest.parent().ok_or_else(|| anyhow::anyhow!("GLM program manifest parent"))?;
    let roots = [parent.join("fp8"), parent.join("../lib/fp8")];
    let find = |name: &str| roots.iter().map(|root| root.join(name).join("tp2")).find(|p| p.is_dir());
    anyhow::ensure!(a4, "checkpoint-declared GLM NVFP4 requires W4A4 at every row count");
    let package = find("fp8-glm-nvfp4a4").ok_or_else(|| anyhow::Error::new(
        MissingLocalPackage("calibrated GLM NVFP4 W4A4 tp2".into())))?;
    let w4a4 = true;
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
        package.parent().unwrap().join("manifest.json"))?)?;
    let shape = catalog.routed_experts();
    let layout = &value["layouts"]["tp2"];
    let geometry = if w4a4 { "glm_nvfp4a4" } else { "glm_nvfp4" };
    anyhow::ensure!(value["schema"].as_str() == Some("cuteafd.fp8moe-package.v1")
        && value["role"].as_str() == Some("coordinator") && value["geometry"].as_str() == Some(geometry)
        && layout["tp"].as_u64() == Some(2) && layout["hidden"].as_u64() == Some(shape.hidden as u64)
        && layout["intermediate"].as_u64() == Some(shape.intermediate as u64)
        && layout["slice"].as_u64() == Some(shape.intermediate as u64 / 2)
        && layout["experts"].as_u64() == Some(shape.experts as u64)
        && layout["top_k"].as_u64() == Some(shape.topk as u64)
        && layout["input"].as_str() == Some("bf16") && layout["weights"].as_str() == Some("nvfp4")
        && layout["swiglu_limit"].as_f64() == Some(0.0),
        "GLM TP2 NVFP4 package geometry/input/format mismatch in {}", package.display());
    // NVFP4 has one program per capacity. Non-default forms need the ABI's
    // largest-form scratch query, which this header-only inventory cannot infer.
    anyhow::ensure!(layout["prefill_forms"].as_object().is_some_and(|forms|
        forms.values().all(|v| v.as_array().is_some_and(Vec::is_empty))),
        "GLM TP2 NVFP4 non-default prefill forms need largest-form scratch metadata");
    let capacities = layout["capacities"].as_array()
        .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 capacities missing"))?
        .iter().map(|c| -> anyhow::Result<(u64, u64)> {
            Ok((c["capacity"].as_u64().filter(|&n| n > 0)
                .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 capacity is invalid"))?,
                c["scratch_bytes"].as_u64()
                    .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 scratch is invalid"))?))
        }).collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!capacities.is_empty() && capacities.windows(2).all(|w| w[0].0 < w[1].0),
        "GLM TP2 NVFP4 capacities must be strictly increasing");
    let max_rows = rows.max(DECODE_ROWS);
    let scratch = capacities.iter().find(|&&(capacity, _)| capacity >= max_rows)
        .map(|&(_, bytes)| bytes.max(FLOOR)).ok_or_else(|| anyhow::anyhow!(
            "GLM TP2 NVFP4 package has no capacity for {max_rows} rows"))?;
    let library = package.join("libcuteafd_fp8moe.so");
    anyhow::ensure!(library.is_file(), "GLM TP2 missing binary {}", library.display());
    let backend_workspace = scratch.checked_add(max_rows * shape.hidden as u64 * 2)
        .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 workspace overflow"))?;
    let experts = (shape.first_layer..selected_layers).map(|layer| -> anyhow::Result<ExpertCost> {
        let view = tensors.for_coordinator().for_layer(layer)?;
        let supported = view.format() == tensors.format();
        let resident = |tp, rank| native_layer_resident(&view, tp, rank, Slicing::Blocks(128));
        let whole = Bytes2 { resident: resident(1, 0)?, staging: 0 };
        let mut half = [resident(2, 0)?, resident(2, 1)?].map(|resident| Bytes2 { resident, staging: 0 });
        for suffix in ["weight", "e_score_correction_bias"] {
            half[1].resident += catalog.tensor(&format!("model.layers.{layer}.mlp.gate.{suffix}"))?
                .metadata.byte_length.max(FLOOR);
        }
        Ok(ExpertCost { whole, half, tp2: supported, whole_supported: supported, spark_ok: true })
    }).collect::<anyhow::Result<Vec<_>>>()?;
    Ok(LocalInventory { package, format: GlmLocalFormat::Nvfp4 { w4a4 }, experts, backend_workspace, world: 2, unit_limit: 0,
        extra_workspace: local_extra_workspace(shape.hidden as u64, shape.experts as u64,
            shape.topk as u64, rows, lanes, false) })
}

#[cfg(test)]
fn exlr8_expert_costs(snapshot: &std::path::Path, selected: &str, first: usize, layers: usize,
    router: &[crate::plan::exlr8::dense::DenseRecord]) -> anyhow::Result<Vec<ExpertCost>> {
    exlr8_expert_costs_of(&crate::plan::exlr8::residency::MetadataInput::open(snapshot, selected)?, first, layers, router)
}

/// Exact checkpoint-native TP2 costs; rotated ownership forbids proportional halves.
fn exlr8_expert_costs_of(input: &crate::plan::exlr8::residency::MetadataInput, first: usize, layers: usize,
    router: &[crate::plan::exlr8::dense::DenseRecord]) -> anyhow::Result<Vec<ExpertCost>> {
    let mut costs = crate::plan::exlr8::routed_expert_costs(input, &(first..layers).collect::<Vec<_>>())?;
    // GLM keeps the lead router in its baseline; only a local TP2 layer needs a peer replica.
    for (layer, cost) in (first..layers).zip(&mut costs) {
        for record in router.iter().filter(|record| record.tensor.starts_with(&format!("model.layers.{layer}.mlp.gate."))) {
            for component in &record.components {
                cost.half[1].resident = cost.half[1].resident.checked_add((component.source.bytes as u64).max(FLOOR))
                    .ok_or_else(|| anyhow::anyhow!("EXLR8 local router storage overflow"))?;
            }
        }
    }
    Ok(costs)
}

/// Native manifest selection is mandatory; never route EXLR8 through EXL3.
pub fn exlr8_inventory(snapshot: &std::path::Path, selected: &str, manifest: &std::path::Path,
    selected_layers: usize, world: usize, rows: u64, lanes: u64, exchange_f32: bool,
    activations: crate::plan::exlr8::PrefillActivations) -> anyhow::Result<LocalInventory> {
    let input = crate::plan::exlr8::residency::MetadataInput::open(snapshot, selected)?;
    exlr8_inventory_with_metadata(snapshot, manifest, selected_layers, world, rows, lanes, exchange_f32, activations, &input)
}

pub(crate) fn exlr8_inventory_with_metadata(snapshot: &std::path::Path, manifest: &std::path::Path,
    selected_layers: usize, world: usize, rows: u64, lanes: u64, exchange_f32: bool,
    activations: crate::plan::exlr8::PrefillActivations, input: &crate::plan::exlr8::residency::MetadataInput)
    -> anyhow::Result<LocalInventory> {
    anyhow::ensure!((1..=2).contains(&world) && !exchange_f32,
        "GLM EXLR8 uses one or two ranks and BF16 routed output");
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let cfg = GlmDsaConfig::from_hf(&config)?;
    anyhow::ensure!(cfg.hidden == 6144 && cfg.moe_intermediate == 2048 && cfg.experts == 256
        && cfg.topk == 8 && (cfg.first_moe_layer..=cfg.layers).contains(&selected_layers)
        && (1..=4).contains(&lanes), "GLM EXLR8 TP2 geometry/layer extent");
    let programs = crate::plan::checkpoint::read_json(manifest)?;
    let mut unit_limit = 0;
    for layer in cfg.first_moe_layer..selected_layers { for rank in 0..world {
        unit_limit = unit_limit.max(input.admission_geometry(layer, world, rank,
            crate::plan::exlr8::residency::ExpertSet::Routed)?.1);
    } }
    let backend_workspace = crate::plan::exlr8::routed_workspace(&programs,
        cuteafd_ffi::exlr8::KernelFamily::Glm, rows, unit_limit, activations)?;
    let records = crate::plan::exlr8::dense::DenseRecord::inventory(snapshot)?;
    for layer in cfg.first_moe_layer..selected_layers {
        for suffix in ["weight", "e_score_correction_bias"] {
            let name = format!("model.layers.{layer}.mlp.gate.{suffix}");
            anyhow::ensure!(records.iter().any(|record| record.tensor == name), "EXLR8 missing local router {name}");
        }
    }
    let experts = exlr8_expert_costs_of(input, cfg.first_moe_layer, selected_layers, &records)?;
    Ok(LocalInventory { package: manifest.to_path_buf(), format: GlmLocalFormat::Exlr8, experts,
        backend_workspace, world, unit_limit,
        extra_workspace: if world == 1 { [0; 2] } else {
            local_extra_workspace(cfg.hidden as u64, cfg.experts as u64, cfg.topk as u64, rows, lanes, false)
        } })
}

#[derive(Debug, thiserror::Error)]
#[error("missing {0} package")]
struct MissingLocalPackage(String);

/// Shared planner/runtime activation policy. Auto remains remote when the build lacks local exports.
#[allow(clippy::too_many_arguments)]
pub fn select_local_inventory(cfg: &GlmDsaConfig, catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    layers: usize, gpus: usize, rows: u64, lanes: u64, exchange_f32: bool,
    onboard: Onboard, skip: bool, exlr8: Option<(&str, crate::plan::exlr8::PrefillActivations)>)
    -> anyhow::Result<Option<LocalInventory>> {
    if skip || onboard.layers(layers.saturating_sub(catalog.routed_experts().first_layer)) == Some(0)
        || layers <= catalog.routed_experts().first_layer { return Ok(None); }
    if catalog.format() == crate::plan::checkpoint::CheckpointFormat::Exlr8 {
        let (selected, activations) = exlr8.ok_or_else(|| anyhow::anyhow!("EXLR8 local inventory requires the selected native manifest"))?;
        return exlr8_inventory(catalog.snapshot(), selected, manifest, layers, gpus, rows, lanes, exchange_f32, activations).map(Some);
    }
    if gpus != 2 && !(gpus == 1 && catalog.exl3().is_some()) {
        anyhow::ensure!(onboard == Onboard::Auto,
            "GLM local experts require a two-GPU TP2 executor");
        return Ok(None);
    }
    anyhow::ensure!(exlr8.is_none(), "EXLR8 manifest supplied for non-native checkpoint");
    let automatic = onboard == Onboard::Auto;
    let supported = catalog.exl3().is_some() || catalog.fp8().is_some_and(|t|
        t.format() == crate::formats::fp8_experts::ExpertFormat::Nvfp4);
    if automatic && !supported { return Ok(None); }
    match local_inventory_world(cfg, catalog, manifest, layers, gpus, rows, lanes, exchange_f32, true) {
        Ok(local) => Ok(Some(local)),
        Err(error) if automatic && error.is::<MissingLocalPackage>() => {
            tracing::info!(reason = %error, "GLM RTX TP2 package unavailable; automatic placement uses Sparks");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

pub fn local_exchange_f32() -> anyhow::Result<bool> {
    match std::env::var("CUTEAFD_TP2_EXCHANGE").ok().as_deref() {
        None | Some("") | Some("auto") | Some("bf16") => Ok(false),
        Some("f32") => Ok(true),
        Some(value) => anyhow::bail!("CUTEAFD_TP2_EXCHANGE must be auto, bf16 or f32, got {value:?}"),
    }
}

pub fn default_onboard() -> Onboard { Onboard::Auto }

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GlmDsaConfig {
        let mut indexers = vec![GlmIndexer::Full; 3];
        indexers.extend((3..78).map(|l| if (l - 2) % 4 == 0 { GlmIndexer::Full } else { GlmIndexer::Shared }));
        GlmDsaConfig { vocab_size: 154880, hidden: 6144, layers: 78, heads: 64, q_lora_rank: 2048,
            kv_lora_rank: 512, qk_nope_head_dim: 192, qk_rope_head_dim: 64, v_head_dim: 256,
            index_heads: 32, index_head_dim: 128, index_topk: 2048, indexers, first_moe_layer: 3,
            dense_intermediate: 12288, experts: 256, topk: 8, moe_intermediate: 2048,
            shared_experts: 1, routed_scale: 2.5, rope_theta: 8000000.0, rms_norm_eps: 1e-5,
            mtp_layers: 1, eos_tokens: vec![154820], variant: crate::families::glm5::GlmVariant::Glm }
    }



    #[test]
    fn whole_workspace_has_no_pair_routes_payloads_or_peer_router_rows() {
        let cfg = config();
        let local = LocalInventory { package: Default::default(), format: GlmLocalFormat::Exlr8,
            experts: Vec::new(), backend_workspace: 17 << 20, extra_workspace: [0; 2], world: 1, unit_limit: 16 };
        assert_eq!(local.workspace(), [17 << 20, 0]);
        for rows in [1, 64, 2752, 4096] { for lanes in 1..=4 {
            assert_eq!(local.workspace_for(&cfg, rows, lanes, false), [17 << 20, 0]);
        } }
    }


    #[test]
    fn exlr8_costs_use_rotated_native_residency_and_peer_router_records() {
        use crate::plan::{exlr8::{dense::{DenseRecord, DenseEncoding, DenseComponent, DenseRange, ShardSide},
            residency::{Residency, ExpertSet}}, testing::exlr8::fixture};
        let root = tempfile::tempdir().unwrap();
        fixture(root.path(), false);
        let records = ["weight", "e_score_correction_bias"].into_iter().zip([768usize, 8]).map(|(suffix, bytes)| DenseRecord {
            ownership: crate::plan::exlr8::dense::DenseOwnership::new(crate::plan::exlr8::dense::DenseExecution::RankZero),
            tensor: format!("model.layers.3.mlp.gate.{suffix}"), encoding: DenseEncoding::Bf16,
            shape: vec![bytes / 2], side: ShardSide::Replicated, channels_per_shard: 0,
            rotation_id: 0, rotation: None, hessian_solved: false, native_blocks: 0,
            components: vec![DenseComponent { window: None, suffix: suffix.into(), dtype: "BF16".into(), shape: vec![bytes / 2],
                source: DenseRange { path: root.path().join("router"), offset: 0, bytes } }],
        }).collect::<Vec<_>>();
        let original = std::fs::read(root.path().join("manifests/exlr8-manifest-mixed.json")).unwrap();
        for selected in ["up1", "bit325"] {
            let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
            manifest["name"] = serde_json::json!(selected);
            std::fs::write(root.path().join(format!("manifests/exlr8-manifest-{selected}.json")), manifest.to_string()).unwrap();
            let costs = exlr8_expert_costs(root.path(), selected, 3, 4, &records).unwrap();
            assert_eq!(costs.len(), 1);
            let cost = costs[0];
            assert_eq!(cost.whole.resident, Residency::plan(root.path(), selected, 3, 1, 0, ExpertSet::Routed).unwrap().resident_bytes as u64);
            for rank in 0..2 {
                assert_eq!(cost.half[rank].resident, Residency::plan(root.path(), selected, 3, 2, rank, ExpertSet::Routed)
                    .unwrap().resident_bytes as u64 + if rank == 1 { 768 + 256 } else { 0 });
                assert_eq!(cost.half[rank].staging, 0);
            }
            assert!(cost.tp2 && cost.spark_ok);
            let residents = crate::plan::exlr8::spark_resident_bytes(root.path(), selected, &[3], 2).unwrap();
            assert_eq!(residents[0], cost.half[0].resident);
            assert_eq!(residents[1] + 1024, cost.half[1].resident);
        }
        assert!(exlr8_expert_costs(root.path(), "missing", 3, 4, &records).is_err());
        assert_eq!(crate::plan::exlr8::spark_resident_bytes(root.path(), "up1", &[], 0).unwrap(), Vec::<u64>::new());
        assert!(crate::plan::exlr8::spark_resident_bytes(root.path(), "up1", &[], 2).is_err());
    }




    #[test]
    fn native_head_removes_only_rank_zero_vocabulary_workspaces() {
        let cfg = config();
        let scratch = GlmScratch { programs: 1000, topk: 2000 };
        for lead in [false, true] {
            let ordinary = step_bytes(&cfg, 64, lead, true, true, false, scratch).unwrap()
                + prefill_bytes(&cfg, 2752, 3, lead, true, false, false, scratch).unwrap();
            let native = step_bytes(&cfg, 64, lead, true, true, true, scratch).unwrap()
                + prefill_bytes(&cfg, 2752, 3, lead, true, false, true, scratch).unwrap();
            assert_eq!(ordinary - native, if lead { 2 * (4 << 20) } else { 0 });
            assert_eq!(head_workspace_bytes(lead, true), if lead { None } else { Some(FLOOR) });
            assert_eq!(head_workspace_bytes(lead, false), Some(if lead { 4 << 20 } else { FLOOR }));
        }
    }

    /// GLM K4 max (two GPUs, six Sparks, 3 x 2752): the runtime's prefill
    /// storage is one arena per `step_row_widths` field at the span plus the
    /// shared step storage, and each slot is a view inside it: slot 0 at
    /// offset 0, slots contiguous and disjoint. The planner charges exactly
    /// that, the shared storage once rather than per slot.
    #[test]
    fn prefill_slots_are_views_and_the_planner_charges_one_span() {
        let cfg = config();
        let (unit, slots) = (2752u64, 3u64);
        let span = prefill_span(unit, slots);
        assert_eq!(span, 8256);
        let views = slot_views(unit, slots);
        assert_eq!(views.len(), 3);
        assert_eq!(views[0].start, 0);
        for pair in views.windows(2) { assert_eq!(pair[0].end, pair[1].start); }
        assert!(views.iter().all(|r| r.end - r.start == unit));
        assert_eq!(views.last().unwrap().end, span);
        let scratch = GlmScratch { programs: 380 << 20, topk: 742 << 20 };
        for (lead, router, native_head) in [(true, false, false), (true, false, true), (false, false, false), (false, true, true)] {
            // The runtime's allocation list: each arena floored at 256 bytes.
            let arenas: u64 = step_row_widths(&cfg, lead, true, router).iter()
                .map(|&(_, width)| (span * width).max(FLOOR)).sum();
            let logits = if lead { DECODE_ROWS * cfg.vocab_size as u64 * 4 } else { FLOOR };
            let runtime = arenas + logits + span * 8 + step_shared(lead, native_head, scratch);
            assert_eq!(prefill_bytes(&cfg, unit, slots, lead, true, router, native_head, scratch).unwrap(), runtime);
            // Compare identical router ownership; step_bytes excludes the local-backend delta.
            let per_slot = slots * (step_rows(&cfg, unit, lead, true, router,
                logits_rows(unit, false)).unwrap() + step_shared(lead, native_head, scratch));
            assert!(runtime + (slots - 1) * (scratch.programs + scratch.topk) <= per_slot,
                "lead {lead}, router {router}, native head {native_head}");
        }
        assert_eq!(prefill_span(4096, 2), 8192);
        assert_eq!(prefill_span(4096, 1), 4096);
        assert_eq!(slot_views(4096, 1), vec![0..4096]);
    }


    /// The engine builds its prefill arenas from the planner's field list and
    /// carves the planner's views; decode keeps its own workspace.
    #[test]
    fn glm_engine_prefill_storage_follows_the_planner_lists() {
        let engine = include_str!("../../../../cuteafd-daemon/src/families/glm5/engine.rs");
        assert!(engine.contains("glm5::step_row_widths(&self.cfg, lead"));
        assert!(engine.contains("glm5::slot_views(unit as u64, slots as u64)"));
        assert!(engine.contains("glm5::prefill_span(unit as u64, slots as u64)"));
        assert!(engine.contains("*self.decode_workspace.borrow_mut() = Some(self.workspace(DECODE_ROWS, true)?)"));
    }










    fn glm_cfg() -> GlmDsaConfig { GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap() }

    fn local_fixture() -> (tempfile::TempDir, crate::OfficialV41Catalog, std::path::PathBuf) {
        use crate::plan::testing::{exl3, exl3_compact, glm5_config, glm5_tensors, write_snapshot};
        use serde_json::json;
        let dir = tempfile::tempdir().unwrap();
        let mut config = glm5_config();
        config["quantization_config"] = exl3_compact(4);
        let mut tensors = glm5_tensors(|name, n, k| exl3(name, n, k, 4));
        for expert in 1..256 {
            for (projection, n, k) in [("gate_proj", 2048, 6144), ("up_proj", 2048, 6144),
                ("down_proj", 6144, 2048)] {
                tensors.extend(exl3(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k, 4));
            }
        }
        let snapshot = dir.path().join("snapshot");
        write_snapshot(&snapshot, &config, &tensors, None);
        let catalog = crate::read_expert_catalog(&snapshot).unwrap();
        let manifest = dir.path().join("PROGRAMS.json");
        std::fs::write(&manifest, b"{\"programs\":[]}").unwrap();
        for capacity in [1, 16, 80] {
            let package = dir.path().join(format!("exl3/exl3-glm-k45/rtx-tp2/m{capacity}"));
            std::fs::create_dir_all(&package).unwrap();
            let value = json!({"hidden": 6144, "intermediate": 1024, "experts": 256,
                "top_k": 8, "capacity": capacity, "bits": [4, 5], "output_dtype": "fp32",
                "input_format": "e4m3_k32", "direct": false,
                "route_preparation": {"manifest": "routes/v41_exl3_routes.json"},
                "trellis_lut": {"file": "lut.bin", "bytes": 32}, "buffers": {
                    "scratch": {"allocation": "scratch", "bytes": capacity * 64,
                        "dtype": "f32", "zero_on_create": false},
                    "state": {"allocation": "state", "bytes": 0,
                        "dtype": "i32", "zero_on_create": true}}});
            std::fs::write(package.join("v41_exl3.json"), serde_json::to_vec(&value).unwrap()).unwrap();
            // Header-only fixtures do not attest or execute these stand-in assets.
            std::fs::create_dir_all(package.join("routes")).unwrap();
            for file in ["libcuteafd_exl3.so", "lut.bin", "routes/v41_exl3_routes.json", "routes/libv41_exl3_routes.so"] {
                std::fs::write(package.join(file), b"").unwrap();
            }
        }
        (dir, catalog, manifest)
    }

    #[test]
    fn activation_policy_uses_exports_and_refuses_malformed_packages() {
        let (dir, catalog, manifest) = local_fixture();
        let cfg = glm_cfg();
        for onboard in [Onboard::Auto, Onboard::Layers(1)] {
            assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 2, 64, 3, false, onboard, false, None)
                .unwrap().is_some());
        }
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 2, 64, 3, false, Onboard::Layers(0), false, None)
            .unwrap().is_none());
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 2, 64, 3, false, Onboard::Layers(1), true, None)
            .unwrap().is_none());
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 1, 64, 3, false, Onboard::Auto, false, None)
            .unwrap().is_none());
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 1, 64, 3, false, Onboard::Layers(1), false, None).is_err());
        let package = dir.path().join("exl3/exl3-glm-k45/rtx-tp2");
        std::fs::remove_file(package.join("m16/v41_exl3.json")).unwrap();
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 2, 64, 3, false, Onboard::Auto, false, None).is_err(),
            "present but incomplete exports must fail closed");
        std::fs::remove_dir_all(package).unwrap();
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 2, 64, 3, false, Onboard::Auto, false, None)
            .unwrap().is_none());
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 2, 64, 3, false, Onboard::Layers(1), false, None).is_err());
    }


    #[test]
    fn world_one_exl3_uses_whole_package_and_exact_physical_owners() {
        use super::local_physical::{LocalPhysicalInput, LocalPhysicalSource, LocalStorage};
        let (dir, catalog, manifest) = local_fixture();
        let cfg = glm_cfg();
        let root = dir.path().join("exl3/exl3-glm-k45");
        for capacity in [1, 16, 80] {
            let source = root.join(format!("rtx-tp2/m{capacity}/v41_exl3.json"));
            let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
            value["intermediate"] = serde_json::json!(cfg.moe_intermediate);
            let package = root.join(format!("rtx-tp1/m{capacity}"));
            std::fs::create_dir_all(&package).unwrap();
            std::fs::write(package.join("v41_exl3.json"), value.to_string()).unwrap();
        }
        for onboard in [Onboard::Auto, Onboard::Layers(1)] {
            let local = select_local_inventory(&cfg, &catalog, &manifest, 2, 1, 64, 3, false,
                onboard, false, None).unwrap().unwrap();
            assert_eq!(local.world, 1);
            assert!(local.package.ends_with("rtx-tp1"));
            assert_eq!(local.extra_workspace, [0; 2]);
            assert_eq!(local.workspace(), [local.backend_workspace, 0]);
            let resident = catalog.exl3().unwrap().residency(crate::V41Exl3Layer::Backbone(1), 1, 0)
                .unwrap().device_arena_layout().unwrap().1 as u64;
            assert_eq!(local.experts[0].whole.resident, resident);
            for storage in [LocalStorage::Push, LocalStorage::Mount] {
                let input = LocalPhysicalInput { layers: vec![1], max_rows: 64, lanes: 3, storage,
                    route_checks: true, peer_diagnostics: true,
                    activations: crate::plan::exlr8::PrefillActivations::A16 };
                let physical = LocalPhysicalSource::read(&cfg, &catalog, &local, &input, None)
                    .unwrap().inventory(&input).unwrap();
                assert_eq!(physical.backend_bytes(0).unwrap(), resident + local.backend_workspace);
                assert_eq!(physical.backend_bytes(1).unwrap(), 0);
                assert!(!physical.owners.iter().any(|owner| owner.id.contains("peer") || owner.id.contains("checks")));
            }
        }
        let path = root.join("rtx-tp1/m16/v41_exl3.json");
        let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["intermediate"] = serde_json::json!(cfg.moe_intermediate / 2);
        std::fs::write(path, value.to_string()).unwrap();
        assert!(select_local_inventory(&cfg, &catalog, &manifest, 2, 1, 64, 3, false,
            Onboard::Auto, false, None).is_err(), "a TP2 package cannot masquerade as Whole");
    }

    #[test]
    fn ml4_world_one_exl3_uses_its_own_namespace_and_geometry() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("snapshot");
        std::fs::create_dir_all(&snapshot).unwrap();
        crate::plan::testing::write_mistral_large4_snapshot(&snapshot, None, true);
        let catalog = crate::read_expert_catalog(&snapshot).unwrap();
        let cfg = GlmDsaConfig::from_hf(&crate::plan::checkpoint::read_json(&snapshot.join("config.json")).unwrap()).unwrap();
        let tiers = catalog.exl3().unwrap().decoder_tiers().iter().map(usize::to_string).collect::<String>();
        let root = dir.path().join(format!("exl3/exl3-ml4-k{tiers}/rtx-tp1"));
        for capacity in [1, 16, 80] {
            let package = root.join(format!("m{capacity}"));
            std::fs::create_dir_all(&package).unwrap();
            std::fs::write(package.join("v41_exl3.json"), serde_json::json!({
                "hidden": cfg.hidden, "intermediate": cfg.moe_intermediate, "experts": cfg.experts,
                "top_k": cfg.topk, "capacity": capacity, "bits": catalog.exl3().unwrap().decoder_tiers(),
                "output_dtype": "fp32", "input_format": "e4m3_k32", "direct": true,
                "trellis_lut": {"file": "lut.bin", "bytes": 32},
                "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * 64,
                    "dtype": "f32", "zero_on_create": false}}
            }).to_string()).unwrap();
        }
        let local = select_local_inventory(&cfg, &catalog, &dir.path().join("PROGRAMS.json"),
            2, 1, 64, 3, false, Onboard::Auto, false, None).unwrap().unwrap();
        assert_eq!(local.package, root);
        assert_eq!(local.world, 1);
        assert_eq!((cfg.moe_intermediate, cfg.experts, cfg.topk), (6144, 128, 4));
        assert_eq!(local.experts[0].whole.resident, catalog.exl3().unwrap()
            .residency(crate::V41Exl3Layer::Backbone(1), 1, 0).unwrap()
            .device_arena_layout().unwrap().1 as u64);
    }

    #[test]
    fn local_inventory_charges_all_retained_capacities_and_exact_rank_storage() {
        let (_dir, catalog, manifest) = local_fixture();
        let cfg = glm_cfg();
        let local = local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, false).unwrap();
        assert_eq!(local.backend_workspace, 80 * 64 + 3 * (32 + 16) + 64 * 6144 * 4);
        assert_eq!(local.experts.len(), 1);
        let exl3 = catalog.exl3().unwrap();
        let layer = crate::V41Exl3Layer::Backbone(1);
        assert_eq!(local.experts[0].whole.resident, exl3.residency(layer, 1, 0).unwrap()
            .device_arena_layout().unwrap().1 as u64);
        for rank in 0..2 {
            let router = if rank == 1 { 256 * 6144 * 2 + 256 * 4 } else { 0 };
            assert_eq!(local.experts[0].half[rank].resident, exl3.residency(layer, 2, rank).unwrap()
                .device_arena_layout().unwrap().1 as u64 + router);
            assert_eq!(local.experts[0].half[rank].staging, 0);
        }
        let wide = local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, true).unwrap();
        let widening = (DECODE_ROWS + 3 * 64) * 6144 * 2 + 4 * 3 * 64 * 6144 * 2;
        for rank in 0..2 {
            assert_eq!(wide.workspace()[rank] - local.workspace()[rank], widening);
        }
        assert!(local_inventory(&cfg, &catalog, &manifest, 0, 64, 3, false).is_err());
        assert!(local_inventory(&cfg, &catalog, &manifest, 3, 64, 3, false).is_err());
    }

    #[test]
    fn local_inventory_keeps_decode_capacity_with_short_prefill_rows() {
        let (_dir, catalog, manifest) = local_fixture();
        let cfg = glm_cfg();
        for rows in [1, 16, 63] {
            for lanes in [1, 3, 4] {
                for f32 in [false, true] {
                    let local = local_inventory(&cfg, &catalog, &manifest, 2, rows, lanes, f32).unwrap();
                    assert_eq!(local.backend_workspace, 80 * 64 + 3 * (32 + 16) + DECODE_ROWS * 6144 * 4);
                    let dtype = if f32 { 4 } else { 2 };
                    // One payload arena whose slot views serve every admitted depth (#762).
                    let span = prefill_span(rows, lanes);
                    let payloads = (DECODE_ROWS * 6144 * dtype).max(FLOOR)
                        + (span * 6144 * dtype).max(FLOOR);
                    let route_slots = 2 * lanes * (DECODE_ROWS * 8 * 8)
                        + ((2 * lanes + 1) * 16).max(FLOOR) + FLOOR;
                    let widened_exchange = if f32 { 4 * lanes * DECODE_ROWS * 6144 * 2 } else { 0 };
                    assert_eq!(local.extra_workspace[0], payloads + route_slots + widened_exchange);
                    let peer_routes = [256 * 4, 8 * 4, 8 * 4, 6144 + 6144 / 32].into_iter().map(|width|
                        (DECODE_ROWS * width).max(FLOOR) - FLOOR
                            + ((span * width).max(FLOOR) - FLOOR)).sum::<u64>();
                    assert_eq!(local.extra_workspace[1] - local.extra_workspace[0], peer_routes);
                }
            }
        }
    }

    #[test]
    fn local_inventory_refuses_missing_assets_and_mismatched_specializations() {
        use serde_json::json;
        let (dir, catalog, manifest) = local_fixture();
        let cfg = glm_cfg();
        let package = dir.path().join("exl3/exl3-glm-k45/rtx-tp2/m16");
        let path = package.join("v41_exl3.json");
        let original: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (field, value) in [("hidden", json!(5120)), ("intermediate", json!(2048)),
            ("experts", json!(128)), ("top_k", json!(6)), ("capacity", json!(80)),
            ("output_dtype", json!("bf16")), ("bits", json!([4, null, 5]))] {
            let mut changed = original.clone();
            changed[field] = value;
            std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            let error = local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, false).unwrap_err();
            assert!(error.to_string().contains("capacity/geometry/tiers/output mismatch"), "{field}: {error:#}");
        }
        std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
        for asset in ["libcuteafd_exl3.so", "lut.bin", "routes/libv41_exl3_routes.so", "routes/v41_exl3_routes.json"] {
            std::fs::remove_file(package.join(asset)).unwrap();
            assert!(local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, false).is_ok(),
                "header-only planning does not load {asset}");
        }
        std::fs::remove_dir_all(dir.path().join("exl3")).unwrap();
        let error = local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, false).unwrap_err();
        assert!(error.to_string().contains("missing exl3-glm-k45/rtx-tp2 package"), "{error:#}");
    }

    #[test]
    fn local_inventory_resolves_declared_route_assets() {
        use serde_json::json;
        let (dir, catalog, manifest) = local_fixture();
        let cfg = glm_cfg();
        let package = dir.path().join("exl3/exl3-glm-k45/rtx-tp2/m16");
        let path = package.join("v41_exl3.json");
        let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        std::fs::rename(package.join("routes"), package.join("packed")).unwrap();
        value["route_preparation"]["manifest"] = json!("packed/v41_exl3_routes.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, false).is_ok());

        value["route_preparation"] = serde_json::Value::Null;
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, false).unwrap_err();
        assert!(error.to_string().contains("missing packed route export manifest"), "{error:#}");

        value["direct"] = json!(true);
        std::fs::remove_dir_all(package.join("packed")).unwrap();
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(local_inventory(&cfg, &catalog, &manifest, 2, 64, 3, false).is_ok());
    }

    fn nvfp4_checkpoint_tensors() -> Vec<crate::plan::testing::Tensor> {
        use crate::plan::testing::{glm5_tensors, nvfp4};
        let mut tensors = glm5_tensors(nvfp4);
        for expert in 1..256 {
            for (projection, n, k) in [("gate_proj", 2048, 6144), ("up_proj", 2048, 6144),
                ("down_proj", 6144, 2048)] {
                tensors.extend(nvfp4(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k));
            }
        }
        tensors
    }

    fn nvfp4_fixture() -> (tempfile::TempDir, crate::OfficialV41Catalog, std::path::PathBuf) {
        use crate::plan::testing::{glm5_config, write_snapshot};
        let dir = tempfile::tempdir().unwrap();
        let mut config = glm5_config();
        config["quantization_config"] = serde_json::json!({"quant_method":"modelopt", "quant_algo":"NVFP4",
            "config_groups":{"group_0":{"weights":{"num_bits":4,"type":"float","group_size":16}}}});
        let tensors = nvfp4_checkpoint_tensors();
        let snapshot = dir.path().join("snapshot");
        write_snapshot(&snapshot, &config, &tensors, None);
        let catalog = crate::read_expert_catalog(&snapshot).unwrap();
        let share = dir.path().join("share");
        std::fs::create_dir_all(&share).unwrap();
        let manifest = share.join("PROGRAMS.json");
        std::fs::write(&manifest, b"{\"programs\":[]}").unwrap();
        for (geometry, offset) in [("glm_nvfp4", 1000), ("glm_nvfp4a4", 2000)] {
            let name = geometry.replace('_', "-");
            let package = dir.path().join(format!("lib/fp8/fp8-{name}"));
            std::fs::create_dir_all(package.join("tp2")).unwrap();
            std::fs::write(package.join("manifest.json"), serde_json::json!({
                "schema":"cuteafd.fp8moe-package.v1", "role":"coordinator", "geometry":geometry,
                "layouts":{"tp2":{"tp":2, "hidden":6144, "intermediate":2048, "slice":1024,
                    "experts":256, "top_k":8, "input":"bf16", "weights":"nvfp4", "swiglu_limit":0.0,
                    "prefill_forms":{"w8a16":[], "w8a8":[]},
                    "capacities":([1,16,80,256,1024,4096].map(|capacity|
                        serde_json::json!({"capacity":capacity,"scratch_bytes":offset + capacity * 64})))}}
            }).to_string()).unwrap();
            // Header fixtures do not execute or attest their stand-in native binary.
            std::fs::write(package.join("tp2/libcuteafd_fp8moe.so"), b"").unwrap();
        }
        (dir, catalog, manifest)
    }

    fn extend_routed_fixture(catalog: &crate::OfficialV41Catalog, mut tensors: Vec<crate::plan::testing::Tensor>)
        -> crate::OfficialV41Catalog {
        let mut config = crate::plan::checkpoint::read_json(&catalog.snapshot().join("config.json")).unwrap();
        config["num_hidden_layers"] = serde_json::json!(4);
        config["indexer_types"] = serde_json::json!(["full", "shared", "shared", "shared"]);
        config["mlp_layer_types"] = serde_json::json!(["dense", "sparse", "sparse", "sparse"]);
        let routed: Vec<_> = tensors.iter().filter(|(name, ..)| name.starts_with("model.layers.1."))
            .cloned().collect();
        for layer in 2..4 {
            tensors.extend(routed.iter().cloned().map(|(name, dtype, shape)|
                (name.replacen("model.layers.1.", &format!("model.layers.{layer}."), 1), dtype, shape)));
        }
        crate::plan::testing::write_snapshot(catalog.snapshot(), &config, &tensors, None);
        crate::read_expert_catalog(catalog.snapshot()).unwrap()
    }

    #[test]
    fn spark_nvfp4_residency_charges_selected_suffix_and_replicated_scalars() {
        use crate::formats::fp8_experts::Slicing;
        let (_dir, catalog, _manifest) = nvfp4_fixture();
        let cfg = glm_cfg();
        let catalog = extend_routed_fixture(&catalog, nvfp4_checkpoint_tensors());
        let scalars = 3 * 256 * 8;
        let bytes = |width| 3 * 256 * 6144 * width * (8 + 1) / 16 + scalars;
        for ranks in 1..=8 {
            for slicing in [Slicing::Blocks(16), Slicing::Blocks(128), Slicing::Padded] {
                let whole_suffix = spark_resident_bytes(&catalog, &[1, 2, 3], ranks, slicing).unwrap();
                let selected = spark_resident_bytes(&catalog, &[2, 3], ranks, slicing).unwrap();
                for rank in 0..ranks {
                    let expected = |layers: &[usize]| layers.iter().map(|&layer| {
                        let width = match slicing {
                            Slicing::Padded => (128usize.div_ceil(ranks) * 16).div_ceil(128) * 128,
                            Slicing::Blocks(g) => {
                                let owner = (rank + ranks - layer % ranks) % ranks;
                                let units = 2048 / g;
                                (units / ranks + usize::from(owner < units % ranks)) * g
                            }
                        };
                        bytes(width as u64)
                    }).sum::<u64>();
                    assert_eq!(selected[rank], expected(&[2, 3]));
                    assert_eq!(whole_suffix[rank], expected(&[1, 2, 3]));
                }
            }
        }
        let exact = spark_resident_bytes(&catalog, &[3], 6, Slicing::Blocks(128)).unwrap();
        let padded = spark_resident_bytes(&catalog, &[3], 6, Slicing::Padded).unwrap();
        assert_eq!(exact, [bytes(384), bytes(256), bytes(256), bytes(384), bytes(384), bytes(384)]);
        assert_eq!(padded, vec![bytes(384); 6]);
        // Per-expert alpha/input scales replicate; they must not shrink with the slice.
        assert_ne!(exact[0] * 2, exact[1] * 3);
        assert_eq!(exact.iter().sum::<u64>(), bytes(2048) + 5 * scalars);
        assert!(spark_resident_bytes(&catalog, &[], 0, Slicing::Padded).unwrap().is_empty());
        for (layers, ranks) in [(vec![], 4), (vec![1], 4), (vec![3, 2], 4),
            (vec![1, 3], 4), (vec![2, 3], 0), (vec![2, 3], 9), (vec![4], 4)] {
            assert!(spark_resident_bytes(&catalog, &layers, ranks, Slicing::Padded).is_err());
        }
    }

    #[test]
    fn spark_exl3_residency_uses_selected_layer_arenas_not_checkpoint_shares() {
        use crate::formats::fp8_experts::Slicing;
        use crate::plan::testing::{exl3, glm5_tensors};
        let (_dir, catalog, _manifest) = local_fixture();
        let cfg = glm_cfg();
        let mut tensors = glm5_tensors(|name, n, k| exl3(name, n, k, 4));
        for expert in 1..256 {
            for (projection, n, k) in [("gate_proj", 2048, 6144), ("up_proj", 2048, 6144),
                ("down_proj", 6144, 2048)] {
                tensors.extend(exl3(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k, 4));
            }
        }
        let catalog = extend_routed_fixture(&catalog, tensors);
        for ranks in [2, 3, 4, 6] {
            let selected = spark_resident_bytes(&catalog, &[2, 3], ranks, Slicing::Blocks(128)).unwrap();
            for (rank, &bytes) in selected.iter().enumerate() {
                let expected: u64 = [2, 3].into_iter().map(|layer|
                    catalog.exl3().unwrap().balanced_residency(crate::V41Exl3Layer::Backbone(layer), ranks, rank)
                        .unwrap().device_arena_layout().unwrap().1 as u64).sum();
                assert_eq!(bytes, expected);
                assert_eq!(bytes % (2 << 20), 0);
            }
            let single = spark_resident_bytes(&catalog, &[3], ranks, Slicing::Blocks(128)).unwrap();
            for (rank, actual) in single.into_iter().enumerate() {
                let expected = catalog.exl3().unwrap().balanced_residency(
                    crate::V41Exl3Layer::Backbone(3), ranks, rank).unwrap();
                assert_eq!(actual, expected.device_arena_layout().unwrap().1 as u64);
            }
        }
        assert!(spark_resident_bytes(&catalog, &[2, 3], 6, Slicing::Padded).is_err());
    }

    #[test]
    fn nvfp4_inventory_retains_native_scales_and_selects_matching_activation_package() {
        let (dir, catalog, manifest) = nvfp4_fixture();
        let cfg = glm_cfg();
        let whole = 3 * (256 * 6144 * 2048 / 2 + 256 * 6144 * 2048 / 16 + 256 * 8);
        let half = 3 * (256 * 6144 * 1024 / 2 + 256 * 6144 * 1024 / 16 + 256 * 8);
        for rows in [1u64, 16, 63, 64, 81, 4096] {
            assert!(local_inventory_selected(&cfg, &catalog, &manifest, 2, rows, 3, false, false).is_err());
            for a4 in [true] {
                let local = local_inventory_selected(&cfg, &catalog, &manifest, 2, rows, 3, false, a4).unwrap();
                assert_eq!(local.format, GlmLocalFormat::Nvfp4 { w4a4: a4 });
                assert_eq!(local.experts.len(), 1);
                assert_eq!(local.experts[0].whole, Bytes2 { resident: whole, staging: 0 });
                assert_eq!(local.experts[0].half[0], Bytes2 { resident: half, staging: 0 });
                assert_eq!(local.experts[0].half[1], Bytes2 {
                    resident: half + 256 * 6144 * 2 + 256 * 4, staging: 0 });
                assert!(local.experts[0].tp2);
                let max_rows = rows.max(DECODE_ROWS);
                let capacity = [1,16,80,256,1024,4096].into_iter().find(|&n| n >= max_rows).unwrap();
                let scratch = if a4 { 2000 } else { 1000 };
                assert_eq!(local.backend_workspace, scratch + capacity * 64 + max_rows * 6144 * 2);
                assert_eq!(local.extra_workspace, local_extra_workspace(6144, 256, 8, rows, 3, false));
                assert_eq!(local.workspace_for(&GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap(), rows, 1, false),
                    local_extra_workspace(6144, 256, 8, rows, 1, false).map(|b| b + local.backend_workspace));
            }
        }
        std::fs::remove_dir_all(dir.path().join("lib/fp8/fp8-glm-nvfp4a4/tp2")).unwrap();
        assert!(local_inventory_selected(&cfg, &catalog, &manifest, 2, 64, 1, false, true).is_err());
    }

    #[test]
    fn nvfp4_inventory_validates_every_selected_expert_header() {
        use crate::plan::testing::{t, write_snapshot};
        let (_dir, catalog, manifest) = nvfp4_fixture();
        let cfg = glm_cfg();
        let config = crate::plan::checkpoint::read_json(&catalog.snapshot().join("config.json")).unwrap();
        let snapshot = catalog.snapshot().to_path_buf();
        let input = "model.layers.1.mlp.experts.255.up_proj.input_scale";
        let down = "model.layers.1.mlp.experts.255.down_proj.weight_scale";
        for (name, replacement) in [(input, None), (input, Some(t(input, "BF16", &[]))),
            (down, Some(t(down, "F8_E4M3", &[6144, 1])))] {
            let mut tensors = nvfp4_checkpoint_tensors();
            let index = tensors.iter().position(|(n, ..)| n == name).unwrap();
            if let Some(replacement) = replacement { tensors[index] = replacement; }
            else { tensors.remove(index); }
            write_snapshot(&snapshot, &config, &tensors, None);
            // The first gate still establishes NVFP4; the later expert must not escape admission validation.
            let error = crate::read_expert_catalog(&snapshot)
                .and_then(|catalog| local_inventory_selected(&cfg, &catalog, &manifest, 2, 64, 3, false, true))
                .unwrap_err();
            assert!(format!("{error:#}").contains(name), "{name}: {error:#}");
        }
    }


    #[test]
    fn nvfp4_inventory_refuses_unimplemented_exchange_and_bad_package_contracts() {
        use serde_json::json;
        let (dir, catalog, manifest) = nvfp4_fixture();
        let cfg = glm_cfg();
        let error = local_inventory_selected(&cfg, &catalog, &manifest, 2, 64, 3, true, true).unwrap_err();
        assert!(error.to_string().contains("use BF16 exchange"), "{error:#}");
        let package = dir.path().join("lib/fp8/fp8-glm-nvfp4a4");
        let path = package.join("manifest.json");
        let original: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (field, value) in [("tp", json!(1)), ("hidden", json!(5120)), ("slice", json!(2048)),
            ("top_k", json!(6)), ("input", json!("wire")), ("weights", json!("mxfp4")),
            ("swiglu_limit", json!(7.0))] {
            let mut changed = original.clone();
            changed["layouts"]["tp2"][field] = value;
            std::fs::write(&path, changed.to_string()).unwrap();
            let error = local_inventory_selected(&cfg, &catalog, &manifest, 2, 64, 3, false, true).unwrap_err();
            assert!(error.to_string().contains("geometry/input/format mismatch"), "{field}: {error:#}");
        }
        for (field, value) in [
            ("capacities", json!([{"capacity":80,"scratch_bytes":null}])),
            ("capacities", json!([{"capacity":80,"scratch_bytes":1},{"capacity":16,"scratch_bytes":2}])),
            ("capacities", json!([{"capacity":16,"scratch_bytes":1}])),
            ("prefill_forms", json!({"w8a16":[80],"w8a8":[]})),
        ] {
            let mut changed = original.clone();
            changed["layouts"]["tp2"][field] = value;
            std::fs::write(&path, changed.to_string()).unwrap();
            assert!(local_inventory_selected(&cfg, &catalog, &manifest, 2, 64, 3, false, true).is_err(), "{field}");
        }
        std::fs::write(&path, original.to_string()).unwrap();
        std::fs::remove_file(package.join("tp2/libcuteafd_fp8moe.so")).unwrap();
        let error = local_inventory_selected(&cfg, &catalog, &manifest, 2, 64, 3, false, true).unwrap_err();
        assert!(error.to_string().contains("missing binary"), "{error:#}");
        std::fs::remove_dir_all(dir.path().join("lib/fp8")).unwrap();
        let error = local_inventory_selected(&cfg, &catalog, &manifest, 2, 64, 3, false, true).unwrap_err();
        assert!(error.to_string().contains("missing calibrated GLM NVFP4 W4A4 tp2 package"), "{error:#}");
    }



    #[test]
    fn colocation_groups_are_glm_index_sources_and_join_mistral_latent_readers() {
        let cfg = config();
        assert_eq!(colocate_groups(&cfg, cfg.layers),
            (0..cfg.layers).map(|layer| cfg.index_source(layer) as u16).collect::<Vec<_>>());
        let ml4 = GlmDsaConfig::from_hf(&crate::families::mistral_large4::config::tests::official()).unwrap();
        let groups = colocate_groups(&ml4, ml4.layers);
        // Layer 35 owns the latent that 39, 43, ..., 71 read; their indexer groups join it.
        for layer in (39..72).step_by(4) {
            assert_eq!(ml4.kv_source(layer), 35);
            assert_eq!(groups[layer], groups[35], "layer {layer}");
            assert_eq!(groups[layer], groups[ml4.index_source(layer)]);
        }
        let owners = attention_owners(&ml4, ml4.layers).unwrap();
        assert!(owners.contains(&0) && owners.contains(&1));
        for layer in 0..ml4.layers {
            assert_eq!(owners[layer], owners[ml4.kv_source(layer)]);
            assert_eq!(owners[layer], owners[ml4.index_source(layer)]);
        }
    }








    const CONTEXT_SCRATCH: GlmContext = GlmContext { topk_scratch: 3 << 20, partial_scratch: 9 << 20 };





    #[test]
    fn context_programs_need_every_primitive() {
        let names = ["glm_scored_index_topk_decode_m64", "glm_dsa_candidate_merge", "glm_lse_combine2",
            "glm_paged_staging_gather_kv", "glm_paged_staging_gather_index", "glm_sparse_mla_partial_h0_s2_m64",
            "glm_sparse_mla_partial_h32_s2_m64", "glm_sparse_mla_partial_h32_s4_m64"];
        let scratch = |name: &str| names.contains(&name).then(|| if name.ends_with("s4_m64") { 900 }
            else if name.contains("topk") { 70 } else { 100 });
        let cfg = config();
        assert_eq!(context_programs(&cfg, names, scratch), Some(GlmContext { topk_scratch: 70, partial_scratch: 900 }));
        for missing in names[..5].iter().chain(&names[5..6]) {
            let kept: Vec<_> = names.iter().copied().filter(|n| n != missing).collect();
            assert_eq!(context_programs(&cfg, kept.iter().copied(), |n| kept.contains(&n).then_some(1)), None, "{missing}");
        }
    }


    #[test]
    fn context_manifest_scratch_matches_runtime_partial_recipe_for_every_plan() {
        use cuteafd_core::coordinator_programs::{attention_partial_program, AttentionReductionPlan};
        let mut programs: Vec<_> = CONTEXT_PROGRAMS.iter().map(|name| format!("glm_{name}"))
            .chain(["glm_paged_staging_gather_kv".to_string()]).map(|name|
            serde_json::json!({"name": name, "scratch_bytes_at_capacity": {"scratch": 70}})).collect();
        for begin in [0, 32] {
            for (splits, caps170, caps188, bytes) in [(2, vec![64], vec![64], 100),
                (4, vec![1, 8], vec![1], 900), (8, vec![], vec![8], 1800)] {
                programs.push(serde_json::json!({
                    "name": format!("glm_sparse_mla_partial_h{begin}_s{splits}_m64"),
                    "params": {"device_sm_counts": [170, 188],
                        "row_caps": {"170": caps170, "188": caps188}},
                    "scratch_bytes_at_capacity": {"scratch": bytes}}));
            }
        }
        let manifest = serde_json::json!({"programs": programs});
        for sms in [170, 188] {
            for rows in 1..=64 {
                let plan = AttentionReductionPlan::new(64, None, Some(rows)).unwrap();
                let selected = manifest_context(&config(), &manifest, sms, &plan).unwrap();
                let runtime_scratch = [0, 32].into_iter().map(|begin| {
                    let name = attention_partial_program(&manifest, "glm", begin, 64, sms, &plan).unwrap();
                    manifest["programs"].as_array().unwrap().iter().find(|p| p["name"] == name).unwrap()
                        ["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap()
                }).max().unwrap();
                assert_eq!(selected, GlmContext { topk_scratch: 70, partial_scratch: runtime_scratch });
                assert_eq!(selected.partial_scratch, if rows == 1 || (rows <= 8 && sms == 170) { 900 }
                    else if rows <= 8 { 1800 } else { 100 });
            }
        }
    }

    #[test]
    fn scratch_selection_includes_optional_programs_and_refuses_missing_required() {
        let cfg = config();
        let lookup = |name: &str| Some(if name.contains("bf16") { 500 } else if name.contains("topk") { 80 } else { 100 });
        assert_eq!(step_scratch(&cfg, true, false, lookup).unwrap(), GlmScratch { programs: 500, topk: 80 });
        for split in [false, true] {
            for decode in [false, true] {
                let lookup = |name: &str| Some(if name.contains("_pt_") { 900 } else { 100 });
                assert_eq!(step_scratch(&cfg, split, decode, lookup).unwrap().programs, 900);
            }
        }
        assert!(step_scratch(&cfg, false, true, |_| None).is_err());
        assert_eq!(default_onboard(), Onboard::Auto);
    }
}
