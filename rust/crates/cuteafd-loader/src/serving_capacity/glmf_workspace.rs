//! Header/geometry-only GLM 5.3 Flash step workspaces: the device bytes of a
//! GPU's decode workspace and prefill lanes. The engine sizes every buffer from
//! these functions (`glm5_flash::engine::StepPlan`), so what it allocates and
//! what the planner charges agree by construction.
//!
//! A step workspace is a lane's own buffers (the mHC streams with their post
//! and comb weights, the step's input, attention and shared-expert rows, the
//! router, route and wire rows, the step tables and the token ids) over the
//! temporaries of one attention call (the MLA queries and latents, the DSA
//! index rows, the programs' scratch, the logits and the vocabulary head).
//! Prefill lanes run on one stream and use the temporaries only inside one
//! unit's attention call, so a GPU's lanes share one set; decode keeps its own,
//! since its captured graphs hold its pointers. Allocations are floored at 256
//! bytes, as in the engine.
use crate::families::glm5_flash::GlmNextConfig;

/// Rows of the decode-route programs (`_m64`).
pub const GLMF_DECODE_ROWS: u64 = 64;
/// Rows of the wide decode-route programs (`_m128`): with `--decode-rows 128` a decode or verify
/// step of more than [`GLMF_DECODE_ROWS`] rows runs them; fewer rows keep the `_m64` programs.
pub const GLMF_WIDE_DECODE_ROWS: u64 = 128;
/// Selected-slot row width of the sparse MLA programs (2048 + 3, padded to 64).
pub const GLMF_SPARSE_TOPK: u64 = 2112;
/// The vocabulary head's cuBLAS workspace (`cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE`).
pub const GLMF_HEAD_WORKSPACE: u64 = 4 << 20;
/// Prefill lanes by default (the engine's `--prefill-lanes`).
pub const GLMF_DEFAULT_PREFILL_LANES: u64 = 2;
/// mHC streams per row.
const HC: u64 = 4;
/// Tokens per DSA index pool.
const KPOOL: u64 = 4;
/// Smallest device allocation the engine makes.
const FLOOR: u64 = 256;

/// What a GPU's step buffers depend on besides their rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmfStepShape {
    /// Rank 0: the vocabulary head, the logits and the router, route and wire rows live there.
    pub lead: bool,
    /// A two-GPU head split: both ranks hold the partials' sum, rank 1 zero rows.
    pub split: bool,
    /// Routed experts on this GPU (the FP8 package): their partial rows.
    pub local_experts: bool,
    /// Routed experts on Spark ranks: pinned staging of the routes and wire rows.
    pub spark: bool,
    /// Bytes per element of the attention partial: 2, or 4 with FP32 KDA partials.
    pub partial_bytes: u64,
    /// The KDA output shard's joined heads (`rows * hidden * 4`) follow the programs' scratch.
    pub output_shard: bool,
    /// Every prefill row's logits (golden scoring); otherwise at most the decode rows'.
    pub full_prefill_logits: bool,
    /// Columns of one row's MLA page table and of its pool-page table.
    pub table_pages: u64,
    pub table_pool_pages: u64,
}

/// Which programs a GPU's steps launch, for the scratch they share.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmfScratchOptions {
    /// The layers are head-split shares (`glmf2_*` programs).
    pub split: bool,
    /// Some layer keeps FP8-only KDA projections (`glmf_kda_w8_*`).
    pub kda_w8: bool,
    pub kda_fp32_partials: bool,
    pub kda_output_shard: bool,
    pub kda_prefill_expanded: bool,
    /// The compact DSA index cache (`--index-cache compact`): the `glmf_index_producer_c_*`
    /// producers, whose scratch also holds the step's key | gate rows.
    pub index_compact: bool,
    /// The KDA recurrent state (`--kda-state`): which KDA programs the steps launch.
    pub kda_state: GlmfKdaState,
}

/// The storage of GLM 5.3 Flash's KDA recurrent state (`--kda-state`). Every KDA program computes
/// in FP32; a BF16 state has programs of its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlmfKdaState {
    #[default]
    F32,
    /// BF16, rounded where each chunked-prefill window stores it.
    Bf16,
    /// BF16, the chunked prefill rounded after every 16-row tile.
    Bf16Tile,
}

impl GlmfKdaState {
    /// Bytes of one recurrent-state element: 4 (FP32) or 2 (BF16).
    pub fn bytes(self) -> u64 {
        if self == Self::F32 { 4 } else { 2 }
    }

    /// The `glmf_kda_*` program (without the prefix) of capacity `cap` (`m64`, `m4096`): the FP32
    /// state's, or the BF16 state's `kda_s16_*` (`kda_s16t_m4096` for the tile-rounded prefill).
    pub fn program(self, cap: &str) -> String {
        match (self, cap) {
            (Self::F32, _) => format!("kda_{cap}"),
            (Self::Bf16Tile, "m4096") => format!("kda_s16t_{cap}"),
            _ => format!("kda_s16_{cap}"),
        }
    }
}

/// A step's scratch: the largest of its programs' (with the KDA output shard's
/// joined heads after it), and the index top-k's own, which stays zeroed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmfScratch {
    pub programs: u64,
    pub topk: u64,
}

/// A program a step launches that the build lacks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmfMissingProgram(pub String);

impl std::fmt::Display for GlmfMissingProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GLM 5.3 Flash steps need program {}, which this build lacks", self.0)
    }
}

impl std::error::Error for GlmfMissingProgram {}

/// The scratch of steps of up to `rows` rows (`decode`: the `_m64` programs, and past
/// [`GLMF_DECODE_ROWS`] rows the wide `_m128` ones too, which run over the same decode workspace;
/// else the `_m4096` ones). `lookup(name)` is None when the build lacks program `name`, else its
/// scratch bytes at capacity (0 without scratch).
pub fn glmf_step_scratch(lookup: impl Fn(&str) -> Option<u64>, cfg: &GlmNextConfig, options: GlmfScratchOptions,
    rows: u64, decode: bool) -> Result<GlmfScratch, GlmfMissingProgram> {
    if !decode {
        return glmf_cap_scratch(&lookup, cfg, options, rows, "m4096", "prefill");
    }
    let narrow = glmf_cap_scratch(&lookup, cfg, options, rows, "m64", "decode")?;
    if rows <= GLMF_DECODE_ROWS {
        return Ok(narrow);
    }
    let wide = glmf_cap_scratch(&lookup, cfg, options, rows, &format!("m{GLMF_WIDE_DECODE_ROWS}"), "decode")?;
    Ok(GlmfScratch { programs: narrow.programs.max(wide.programs), topk: narrow.topk.max(wide.topk) })
}

/// [`glmf_step_scratch`] of the programs of one capacity `cap` (`m64`, `m128`, `m4096`) and route
/// `mode` (`decode`, `prefill`).
fn glmf_cap_scratch(lookup: &impl Fn(&str) -> Option<u64>, cfg: &GlmNextConfig, options: GlmfScratchOptions,
    rows: u64, cap: &str, mode: &str) -> Result<GlmfScratch, GlmfMissingProgram> {
    let decode = mode == "decode";
    let required = |name: String| lookup(&name).ok_or(GlmfMissingProgram(name));
    let mut scratch = 0;
    for name in [format!("mhc_post_pre_{cap}"), options.kda_state.program(cap), format!("mla_producer_{cap}"),
        format!("sparse_mla_{mode}_{cap}"), format!("o_{cap}"), format!("ffn_i2048_{cap}"),
        format!("ffn_i12288_{cap}"), format!("index_producer_{cap}"), "mhc_pre".into()] {
        scratch = scratch.max(required(format!("glmf_{name}"))?);
    }
    if options.index_compact {
        // The step's key | gate rows follow the compact producer's query and projection scratch.
        scratch = scratch.max(required(format!("glmf_index_producer_c_{cap}"))?);
    }
    if options.split {
        // The head split's share programs (those this build has).
        for name in [format!("kda_{cap}"), format!("mla_producer_{cap}"), format!("sparse_mla_{mode}_{cap}"),
            format!("o_{cap}"), format!("ffn_i{}_{cap}", cfg.moe_intermediate / 2),
            format!("ffn_i{}_{cap}", cfg.dense_intermediate / 2), format!("kda_w8_{cap}")] {
            scratch = scratch.max(lookup(&format!("glmf2_{name}")).unwrap_or(0));
        }
    }
    if options.kda_w8 {
        scratch = scratch.max(required(format!("glmf_kda_w8_{cap}"))?);
    }
    if options.kda_fp32_partials || options.kda_output_shard || options.kda_prefill_expanded {
        let dtype = if options.kda_output_shard { "_norm" } else if options.kda_fp32_partials { "_f32" } else { "" };
        let expanded = if options.kda_prefill_expanded && !decode { "_expanded" } else { "" };
        // Joined head activations occupy a fixed tail after the program's scratch
        // and stay live through the output token-row projection.
        let output = if options.kda_output_shard { rows * cfg.hidden as u64 * 4 } else { 0 };
        scratch = scratch.max(required(format!("glmf2_kda_w8{dtype}{expanded}_{cap}"))? + output);
        if options.kda_output_shard {
            scratch = scratch.max(required(format!("glmf2_kda_output_rows{expanded}_{cap}"))? + output);
        }
    }
    Ok(GlmfScratch { programs: scratch, topk: required(format!("glmf_index_topk_{mode}_{cap}"))? })
}

/// `glmf_step_scratch`'s lookup over an exported `PROGRAMS.json`.
pub fn glmf_manifest_scratch(manifest: &serde_json::Value) -> impl Fn(&str) -> Option<u64> + '_ {
    move |name| {
        manifest["programs"].as_array()?.iter().find(|p| p["name"].as_str() == Some(name))
            .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0))
    }
}

/// One lane's own buffers: bytes before the allocation floor, None where none is allocated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmfLaneBytes {
    /// Zero rows: rank 1's partial of a dense MLP rank 0 runs whole (head split only).
    pub zero: Option<u64>,
    /// The sum of a head split's two partials.
    pub sum: Option<u64>,
    /// Each of the two mHC stream buffers.
    pub streams: u64,
    pub post: u64,
    pub comb: u64,
    pub x: u64,
    pub delta: u64,
    pub shared: u64,
    /// The routed experts' partial (local experts only).
    pub routed: Option<u64>,
    pub positions: u64,
    pub kv_slots: u64,
    pub kda_slots: u64,
    pub seq_first: u64,
    pub pool_slots: u64,
    pub cache_lengths: u64,
    pub page_table: u64,
    pub pool_table: u64,
    pub ids: u64,
    pub select: u64,
    pub router_logits: u64,
    pub route_ids: u64,
    pub route_weights: u64,
    pub wire: u64,
    /// Pinned host staging of the routes and wire rows (Spark experts); not device memory.
    pub router_host: u64,
}

impl GlmfLaneBytes {
    /// Device bytes, every allocation floored as the engine allocates it.
    pub fn device_bytes(&self) -> u64 {
        [self.streams, self.streams, self.post, self.comb, self.x, self.delta, self.shared, self.positions,
            self.kv_slots, self.kda_slots, self.seq_first, self.pool_slots, self.cache_lengths, self.page_table,
            self.pool_table, self.ids, self.select, self.router_logits, self.route_ids, self.route_weights, self.wire]
            .into_iter().chain([self.zero, self.sum, self.routed].into_iter().flatten())
            .map(|bytes| bytes.max(FLOOR)).sum()
    }
}

/// The temporaries of one attention call (shared by a GPU's prefill lanes).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmfTemporaryBytes {
    pub query: u64,
    pub q_resid: u64,
    pub latent: u64,
    pub q_fp8: u64,
    pub head_weights: u64,
    pub pools: u64,
    pub indices: u64,
    pub lengths: u64,
    pub scratch: u64,
    pub topk_scratch: u64,
    pub logits: u64,
    pub head_workspace: u64,
}

impl GlmfTemporaryBytes {
    /// Device bytes, every allocation floored as the engine allocates it.
    pub fn device_bytes(&self) -> u64 {
        [self.query, self.q_resid, self.latent, self.q_fp8, self.head_weights, self.pools, self.indices,
            self.lengths, self.scratch, self.topk_scratch, self.logits, self.head_workspace]
            .into_iter().map(|bytes| bytes.max(FLOOR)).sum()
    }
}

/// A lane's own buffers for steps of up to `rows` rows.
pub fn glmf_lane_bytes(cfg: &GlmNextConfig, rows: u64, decode: bool, shape: &GlmfStepShape) -> GlmfLaneBytes {
    let (t, h, topk) = (rows, cfg.hidden as u64, cfg.topk as u64);
    let lead = |bytes: u64| if shape.lead { bytes } else { 0 };
    // Decode steps carry one padded table row per step row; prefill steps one shared row.
    let table_rows = if decode { t } else { 1 };
    GlmfLaneBytes {
        zero: (shape.split && !shape.lead).then_some(t * h * 2),
        sum: shape.split.then_some(t * h * 2),
        streams: t * HC * h * 2,
        post: t * HC * 4,
        comb: t * HC * HC * 4,
        x: t * h * 2,
        delta: t * h * shape.partial_bytes,
        shared: t * h * 2,
        routed: (shape.lead && shape.local_experts).then_some(t * h * 2),
        positions: t * 8,
        kv_slots: t * 8,
        kda_slots: t * 4,
        seq_first: t * 4,
        pool_slots: t * 8,
        cache_lengths: t * 4,
        page_table: table_rows * shape.table_pages * 4,
        pool_table: table_rows * shape.table_pool_pages * 4,
        ids: t * 4,
        select: t * 8,
        router_logits: lead(t * cfg.experts as u64 * 4),
        route_ids: lead(t * topk * 4),
        route_weights: lead(t * topk * 4),
        wire: lead(t * (h + h / 32)),
        router_host: if shape.lead && shape.spark { t * (topk * 8 + h + h / 32) } else { FLOOR },
    }
}

/// The temporaries of steps of up to `rows` rows over `scratch` (`glmf_step_scratch`).
pub fn glmf_temporary_bytes(cfg: &GlmNextConfig, rows: u64, decode: bool, shape: &GlmfStepShape,
    scratch: GlmfScratch) -> GlmfTemporaryBytes {
    let (t, n, latent) = (rows, cfg.heads as u64, cfg.kv_lora_rank as u64);
    let lead = |bytes: u64| if shape.lead { bytes } else { 0 };
    let logit_rows = if decode || shape.full_prefill_logits { t } else { t.min(GLMF_DECODE_ROWS) };
    GlmfTemporaryBytes {
        query: t * n * latent * 2,
        q_resid: t * cfg.q_lora_rank as u64 * 2,
        latent: t * n * latent * 2,
        q_fp8: t * 32 * 128,
        head_weights: t * 32 * 4,
        pools: t * (cfg.index_topk as u64 / KPOOL) * 4,
        indices: t * GLMF_SPARSE_TOPK * 4,
        lengths: t * 4,
        scratch: scratch.programs,
        topk_scratch: scratch.topk,
        logits: lead(logit_rows * cfg.vocab_size as u64 * 4),
        head_workspace: lead(GLMF_HEAD_WORKSPACE),
    }
}

/// Columns of a step row's MLA page table and pool-page table: the pages of one sequence of
/// `max_context` tokens (whole 256-token units), rounded up to a power of two as decode strides
/// are. Known before the pool is sized; positions past `max_context` are never stepped.
pub fn glmf_table_pages(max_context: u64) -> (u64, u64) {
    let units = max_context.div_ceil(256).max(1);
    ((units * 4).next_power_of_two(), units.next_power_of_two())
}

/// One GPU's step workspaces: a decode workspace, and `lanes` prefill lanes over
/// one shared set of temporaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmfStepWorkspaces {
    /// The decode workspace: its lane buffers and its own temporaries.
    pub decode: u64,
    /// One prefill lane's own buffers.
    pub lane: u64,
    pub lanes: u64,
    /// The prefill lanes' shared temporaries.
    pub prefill_temporaries: u64,
}

impl GlmfStepWorkspaces {
    pub fn device_bytes(&self) -> u64 {
        self.decode + self.lanes * self.lane + self.prefill_temporaries
    }
}

/// The token selector of decode steps of up to `rows` rows (the daemon's `TokenSelector`): its
/// greedy outputs (ids, statuses and log-probabilities, 12 B a row) and the GPU sampler the
/// start-up reserves for up to 128 of them (`TargetSamplingWave::device_bytes`: per row 64 B of
/// parameters and 64 B of scratch, two vocabulary bit masks, a 2,048-bucket histogram, 256
/// rank-ordered ids with their 8-byte staging, and 32 B of row outputs).
pub fn glmf_selector_bytes(rows: u64, vocab: u64) -> u64 {
    let capacity = rows.clamp(1, 128);
    let mask = capacity * vocab.div_ceil(32) * 4;
    rows * 12 + capacity * (64 + 64 + 2048 * 4 + 256 * (4 + 8) + 32) + 2 * mask
}

/// The step workspaces of `lanes` prefill lanes of `lane_rows` rows and the decode workspace of
/// `decode_rows` rows (the engine's `--decode-rows`), over the scratch of each step shape.
pub fn glmf_step_workspaces(cfg: &GlmNextConfig, lanes: usize, lane_rows: u64, decode_rows: u64,
    shape: &GlmfStepShape, decode_scratch: GlmfScratch, prefill_scratch: GlmfScratch) -> GlmfStepWorkspaces {
    let rows = decode_rows;
    GlmfStepWorkspaces {
        decode: glmf_lane_bytes(cfg, rows, true, shape).device_bytes()
            + glmf_temporary_bytes(cfg, rows, true, shape, decode_scratch).device_bytes(),
        lane: glmf_lane_bytes(cfg, lane_rows, false, shape).device_bytes(),
        lanes: lanes as u64,
        prefill_temporaries: glmf_temporary_bytes(cfg, lane_rows, false, shape, prefill_scratch).device_bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GLM programs' scratch at capacity of our first own build (slot `base`,
    /// `work/p0` c3e5de13 exported on an RTX 5090): `glmf_*` single-GPU programs.
    const BASE_SCRATCH: [(&str, u64); 21] = [
        ("glmf_mhc_pre", 26_214_400), ("glmf_index_producer_m64", 561_152),
        ("glmf_index_topk_decode_m64", 8_653_824), ("glmf_mhc_post_pre_m64", 409_600),
        ("glmf_kda_m64", 10_526_720), ("glmf_kda_w8_m64", 10_526_720), ("glmf_mla_producer_m64", 2_359_296),
        ("glmf_o_m64", 2_097_152), ("glmf_sparse_mla_decode_m64", 8_404_992), ("glmf_ffn_i2048_m64", 786_432),
        ("glmf_ffn_i12288_m64", 4_718_592), ("glmf_index_producer_m4096", 35_913_728),
        ("glmf_index_topk_prefill_m4096", 558_007_296), ("glmf_mhc_post_pre_m4096", 26_214_400),
        ("glmf_kda_m4096", 782_236_672), ("glmf_kda_w8_m4096", 782_236_672),
        ("glmf_mla_producer_m4096", 168_296_448), ("glmf_o_m4096", 203_423_744),
        ("glmf_sparse_mla_prefill_m4096", 1_048_576), ("glmf_ffn_i2048_m4096", 67_633_152),
        ("glmf_ffn_i12288_m4096", 353_894_400),
    ];

    fn lookup(name: &str) -> Option<u64> {
        BASE_SCRATCH.iter().find(|(n, _)| *n == name).map(|&(_, bytes)| bytes)
    }

    fn glm53_flash() -> GlmNextConfig {
        let mut config = crate::plan::testing::glm5_flash_config(45);
        config["text_config"]["vocab_size"] = 154_880.into();
        GlmNextConfig::from_hf(&config).unwrap()
    }

    /// One RTX + Sparks, BF16 KDA, at the base run's 118,528-token pool (1,852 MLA pages).
    fn spark_shape() -> GlmfStepShape {
        GlmfStepShape { lead: true, split: false, local_experts: false, spark: true, partial_bytes: 2,
            output_shard: false, full_prefill_logits: false, table_pages: 1852, table_pool_pages: 463 }
    }

    fn scratch(cfg: &GlmNextConfig, rows: u64, decode: bool) -> GlmfScratch {
        glmf_step_scratch(lookup, cfg, GlmfScratchOptions::default(), rows, decode).unwrap()
    }

    #[test]
    fn the_scratch_selection_takes_the_largest_program_of_each_step_shape() {
        let cfg = glm53_flash();
        // mhc_pre's prefill-capacity scratch is the largest a decode step launches.
        assert_eq!(scratch(&cfg, 64, true), GlmfScratch { programs: 26_214_400, topk: 8_653_824 });
        // The KDA programs' (their chunked recurrence workspace at 4,096 rows) for prefill,
        // whatever the lane's rows: the programs lay it out for their capacity.
        for rows in [2048, 4096] {
            assert_eq!(scratch(&cfg, rows, false), GlmfScratch { programs: 782_236_672, topk: 558_007_296 });
        }
        let error = glmf_step_scratch(|name| lookup(name).filter(|_| name != "glmf_o_m4096"), &cfg,
            GlmfScratchOptions::default(), 4096, false).unwrap_err();
        assert_eq!(error, GlmfMissingProgram("glmf_o_m4096".into()));
        // A head split's share programs count only where the build has them.
        let split = GlmfScratchOptions { split: true, ..Default::default() };
        assert_eq!(glmf_step_scratch(lookup, &cfg, split, 4096, false).unwrap(), scratch(&cfg, 4096, false));
        let manifest = serde_json::json!({"programs": [{"name": "glmf_o_m64", "scratch_bytes_at_capacity":
            {"scratch": 2_097_152}}, {"name": "glmf_add", "scratch_bytes_at_capacity": {}}]});
        let from_manifest = glmf_manifest_scratch(&manifest);
        assert_eq!((from_manifest("glmf_o_m64"), from_manifest("glmf_add"), from_manifest("glmf_head")),
            (Some(2_097_152), Some(0), None));
    }

    #[test]
    fn the_formula_reproduces_the_measured_two_lane_workspaces() {
        // Before the lanes shared their temporaries, each lane held a whole workspace, and every
        // workspace its head-split and local-expert rows: the base ledger's `workspace` scope at
        // 16 sequences, 5,069,385,816 bytes (decode + two lanes of 4,096 rows).
        let cfg = glm53_flash();
        let shape = spark_shape();
        let unshared = |rows: u64, decode: bool| {
            let mut lane = glmf_lane_bytes(&cfg, rows, decode, &shape);
            lane.zero = Some(rows * 4096 * 2);
            lane.sum = Some(rows * 4096 * 2);
            lane.routed = Some(rows * 4096 * 2);
            lane.device_bytes() + glmf_temporary_bytes(&cfg, rows, decode, &shape, scratch(&cfg, rows, decode))
                .device_bytes()
        };
        assert_eq!(unshared(64, true) + 2 * unshared(4096, false), 5_069_385_816);
    }

    #[test]
    fn prefill_lanes_share_one_set_of_temporaries() {
        let cfg = glm53_flash();
        let shape = spark_shape();
        let workspaces = |lanes: usize, rows: u64| glmf_step_workspaces(&cfg, lanes, rows, GLMF_DECODE_ROWS, &shape,
            scratch(&cfg, 64, true), scratch(&cfg, rows, false));
        let two = workspaces(2, 4096);
        assert_eq!((two.decode, two.lane, two.prefill_temporaries), (94_955_520, 391_914_540, 1_993_850_880));
        assert_eq!(two.device_bytes(), 2_872_635_480);
        // Four lanes of 2,048 rows hold the same rows in flight; the per-row temporaries halve,
        // the programs' scratch (laid out for 4,096 rows) does not.
        let four = workspaces(4, 2048);
        assert_eq!((four.lane, four.prefill_temporaries), (195_961_900, 1_688_969_216));
        assert_eq!(four.device_bytes(), 2_567_772_336);
        assert_eq!(workspaces(1, 4096).device_bytes(), two.device_bytes() - two.lane);
    }

    #[test]
    fn a_bf16_kda_state_charges_its_programs_and_needs_them() {
        let cfg = glm53_flash();
        let bf16 = GlmfScratchOptions { kda_state: GlmfKdaState::Bf16, ..Default::default() };
        let tile = GlmfScratchOptions { kda_state: GlmfKdaState::Bf16Tile, ..Default::default() };
        // A build whose BF16-state programs take the FP32 programs' scratch: the workspaces do not move.
        let with = |name: &str| match name {
            "glmf_kda_s16_m64" => lookup("glmf_kda_m64"),
            "glmf_kda_s16_m4096" | "glmf_kda_s16t_m4096" => lookup("glmf_kda_m4096"),
            _ => lookup(name),
        };
        assert_eq!(glmf_step_scratch(with, &cfg, bf16, 64, true).unwrap(), scratch(&cfg, 64, true));
        assert_eq!(glmf_step_scratch(with, &cfg, bf16, 4096, false).unwrap(), scratch(&cfg, 4096, false));
        assert_eq!(glmf_step_scratch(with, &cfg, tile, 4096, false).unwrap(), scratch(&cfg, 4096, false));
        // The steps get the scratch of the program they launch: window and tile rounding apart.
        let larger = |name: &str| if name == "glmf_kda_s16t_m4096" { Some(900_000_000) } else { with(name) };
        assert_eq!(glmf_step_scratch(larger, &cfg, tile, 4096, false).unwrap().programs, 900_000_000);
        assert_eq!(glmf_step_scratch(larger, &cfg, bf16, 4096, false).unwrap(), scratch(&cfg, 4096, false));
        assert_eq!(glmf_step_scratch(larger, &cfg, GlmfScratchOptions::default(), 4096, false).unwrap(),
            scratch(&cfg, 4096, false));
        // A build without them cannot run a BF16 state; an FP32 one never asks for them.
        assert_eq!(glmf_step_scratch(lookup, &cfg, bf16, 64, true).unwrap_err(),
            GlmfMissingProgram("glmf_kda_s16_m64".into()));
        assert_eq!(glmf_step_scratch(lookup, &cfg, tile, 4096, false).unwrap_err(),
            GlmfMissingProgram("glmf_kda_s16t_m4096".into()));
        assert_eq!([GlmfKdaState::F32, GlmfKdaState::Bf16, GlmfKdaState::Bf16Tile].map(GlmfKdaState::bytes), [4, 2, 2]);
    }

    #[test]
    fn the_compact_index_cache_charges_its_producers_and_needs_them() {
        let cfg = glm53_flash();
        let compact = GlmfScratchOptions { index_compact: true, ..Default::default() };
        // `index_producer_c_scratch_bytes`: the producer's scratch, then the step's key | gate rows.
        let with = |name: &str| match name {
            "glmf_index_producer_c_m64" => Some(561_152 + 64 * 512),
            "glmf_index_producer_c_m4096" => Some(35_913_728 + 4096 * 512),
            _ => lookup(name),
        };
        // Below the largest program's scratch at either capacity: the workspace bytes do not move.
        assert_eq!(glmf_step_scratch(with, &cfg, compact, 64, true).unwrap(), scratch(&cfg, 64, true));
        assert_eq!(glmf_step_scratch(with, &cfg, compact, 4096, false).unwrap(), scratch(&cfg, 4096, false));
        // A larger producer scratch is the step's: the lanes' shared temporaries hold it.
        let larger = |name: &str| if name == "glmf_index_producer_c_m4096" { Some(900_000_000) } else { with(name) };
        assert_eq!(glmf_step_scratch(larger, &cfg, compact, 4096, false).unwrap().programs, 900_000_000);
        assert_eq!(glmf_step_scratch(larger, &cfg, GlmfScratchOptions::default(), 4096, false).unwrap(),
            scratch(&cfg, 4096, false));
        // A build without the compact producers cannot run the compact cache.
        assert_eq!(glmf_step_scratch(lookup, &cfg, compact, 64, true).unwrap_err(),
            GlmfMissingProgram("glmf_index_producer_c_m64".into()));
    }

    #[test]
    fn page_tables_hold_one_sequence_of_the_context() {
        // 131,072 tokens: 512 units, 2,048 MLA pages; 1M tokens: 16,384 pages.
        assert_eq!(glmf_table_pages(131_072), (2048, 512));
        assert_eq!(glmf_table_pages(1_048_576), (16_384, 4096));
        // Decode strides are the next power of two of a sequence's pages.
        assert_eq!(glmf_table_pages(65_537), (2048, 512));
        assert_eq!(glmf_table_pages(1), (4, 1));
        // A decode workspace's tables at 1M tokens: 64 rows of 16,384 + 4,096 columns.
        let shape = GlmfStepShape { table_pages: 16_384, table_pool_pages: 4096, ..spark_shape() };
        let lane = glmf_lane_bytes(&glm53_flash(), 64, true, &shape);
        assert_eq!(lane.page_table + lane.pool_table, 64 * (16_384 + 4096) * 4);
    }

    #[test]
    fn single_gpu_lanes_allocate_no_split_or_local_expert_rows() {
        let cfg = glm53_flash();
        let lane = glmf_lane_bytes(&cfg, 4096, false, &spark_shape());
        assert_eq!((lane.zero, lane.sum, lane.routed), (None, None, None));
        assert_eq!(lane.router_host, 4096 * (8 * 8 + 4096 + 128));
        let local = glmf_lane_bytes(&cfg, 4096, false,
            &GlmfStepShape { local_experts: true, spark: false, ..spark_shape() });
        assert_eq!((local.routed, local.router_host), (Some(4096 * 4096 * 2), FLOOR));
        // A head split: both ranks keep the sum, rank 1 the zero rows and none of the lead's buffers.
        let lead = glmf_lane_bytes(&cfg, 64, true, &GlmfStepShape { split: true, ..spark_shape() });
        assert_eq!((lead.zero, lead.sum), (None, Some(64 * 4096 * 2)));
        let peer_shape = GlmfStepShape { lead: false, split: true, spark: false, ..spark_shape() };
        let peer = glmf_lane_bytes(&cfg, 64, true, &peer_shape);
        assert_eq!((peer.zero, peer.sum, peer.router_logits, peer.wire), (Some(64 * 4096 * 2), Some(64 * 4096 * 2), 0, 0));
        let temps = glmf_temporary_bytes(&cfg, 64, true, &peer_shape, scratch(&cfg, 64, true));
        assert_eq!((temps.logits, temps.head_workspace), (0, 0));
        // Decode tables hold a padded row per step row; prefill tables one shared row.
        assert_eq!((lead.page_table, lane.page_table), (64 * 1852 * 4, 1852 * 4));
    }

    /// The wide decode programs' scratch at capacity in the offline SM 12.0 export of SparkInfer
    /// 1ce62d27 at 170 SMs (the sparse MLA with `full_launch_splits=1`), and the same export's
    /// BF16-state and compact-index decode programs of 64 rows.
    const WIDE_SCRATCH: [(&str, u64); 14] = [
        ("glmf_index_producer_m128", 1_122_304), ("glmf_index_producer_c_m128", 1_187_840),
        ("glmf_index_topk_decode_m128", 17_304_576), ("glmf_mhc_post_pre_m128", 819_200),
        ("glmf_kda_m128", 21_053_440), ("glmf_kda_w8_m128", 21_053_440), ("glmf_kda_s16_m128", 21_053_440),
        ("glmf_mla_producer_m128", 4_718_592), ("glmf_o_m128", 4_194_304),
        ("glmf_sparse_mla_decode_m128", 16_809_984), ("glmf_ffn_i2048_m128", 1_572_864),
        ("glmf_ffn_i12288_m128", 9_437_184), ("glmf_kda_s16_m64", 10_526_720), ("glmf_index_producer_c_m64", 593_920),
    ];

    fn wide_lookup(name: &str) -> Option<u64> {
        WIDE_SCRATCH.iter().find(|(n, _)| *n == name).map(|&(_, bytes)| bytes).or_else(|| lookup(name))
    }

    /// A step of more than 64 rows runs the `_m128` programs over the decode workspace that also
    /// runs the `_m64` ones: its scratch is the larger of each program set's, and its top-k scratch
    /// the wide top-k's. Up to 64 rows the wide programs are not consulted.
    #[test]
    fn a_wide_decode_step_runs_both_program_sets() {
        let cfg = glm53_flash();
        let wide = |rows| glmf_step_scratch(wide_lookup, &cfg, GlmfScratchOptions::default(), rows, true);
        // mhc_pre's 26,214,400 B stays the largest program scratch (the one-split sparse MLA is
        // 16,809,984 B); the wide top-k's 17,304,576 B replaces the 64-row one's 8,653,824 B.
        assert_eq!(wide(128).unwrap(), GlmfScratch { programs: 26_214_400, topk: 17_304_576 });
        assert_eq!(wide(64).unwrap(), scratch(&cfg, 64, true));
        assert_eq!(glmf_step_scratch(lookup, &cfg, GlmfScratchOptions::default(), 64, true).unwrap(),
            scratch(&cfg, 64, true));
        // A build without the wide programs cannot take a step past 64 rows, and names the first it lacks.
        assert_eq!(glmf_step_scratch(lookup, &cfg, GlmfScratchOptions::default(), 128, true).unwrap_err(),
            GlmfMissingProgram("glmf_mhc_post_pre_m128".into()));
        // The state and index cache select their wide programs as they select the 64-row ones.
        let compact = GlmfScratchOptions { index_compact: true, kda_state: GlmfKdaState::Bf16, kda_w8: true,
            ..Default::default() };
        assert_eq!(glmf_step_scratch(wide_lookup, &cfg, compact, 128, true).unwrap(),
            GlmfScratch { programs: 26_214_400, topk: 17_304_576 });
        let without = |name: &str| wide_lookup(name).filter(|_| name != "glmf_index_producer_c_m128");
        assert_eq!(glmf_step_scratch(without, &cfg, compact, 128, true).unwrap_err(),
            GlmfMissingProgram("glmf_index_producer_c_m128".into()));
        // The 33-split sparse MLA a planner without `full_launch_splits` gives at 170 SMs would be the
        // step's largest scratch.
        let split = |name: &str| if name == "glmf_sparse_mla_decode_m128" { Some(554_729_472) } else { wide_lookup(name) };
        assert_eq!(glmf_step_scratch(split, &cfg, GlmfScratchOptions::default(), 128, true).unwrap().programs,
            554_729_472);
    }

    /// The decode workspace of 128 rows against 64, on one RTX with Spark experts at 131,072 tokens
    /// (2,048 + 512 table columns a step row): the lane buffers and temporaries double with the rows,
    /// the top-k scratch is the wide top-k's, the program scratch stays mhc_pre's. The prefill lanes
    /// do not move. The selector and sampler of 128 rows add 3,209,984 B.
    #[test]
    fn a_wide_decode_workspace_doubles_its_rows() {
        let cfg = glm53_flash();
        let shape = GlmfStepShape { table_pages: 2048, table_pool_pages: 512, ..spark_shape() };
        let wide = |rows| glmf_step_scratch(wide_lookup, &cfg, GlmfScratchOptions::default(), rows, true).unwrap();
        let at = |decode_rows: u64| glmf_step_workspaces(&cfg, 2, 4096, decode_rows, &shape, wide(decode_rows),
            scratch(&cfg, 4096, false));
        let (narrow, broad) = (at(GLMF_DECODE_ROWS), at(GLMF_WIDE_DECODE_ROWS));
        assert_eq!((broad.lane, broad.lanes, broad.prefill_temporaries), (narrow.lane, narrow.lanes, narrow.prefill_temporaries));
        let rows = |decode_rows| glmf_lane_bytes(&cfg, decode_rows, true, &shape).device_bytes()
            + glmf_temporary_bytes(&cfg, decode_rows, true, &shape, GlmfScratch { programs: 0, topk: 0 }).device_bytes()
            - 2 * FLOOR;
        assert_eq!(rows(128) - rows(64), 55_955_712);
        assert_eq!(broad.decode - narrow.decode, 55_955_712 + (17_304_576 - 8_653_824));
        assert_eq!(broad.device_bytes() - narrow.device_bytes(), 64_606_464);
        assert_eq!(glmf_selector_bytes(64, 154_880), 3_209_984);
        assert_eq!(glmf_selector_bytes(128, 154_880) - glmf_selector_bytes(64, 154_880), 3_209_984);
    }
}
