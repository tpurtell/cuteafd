//! Programs belonging to one coordinator, including unsplit draft/target work.

/// An immutable numerical reduction recipe hint, not an execution row limit.
/// Choose before admission/capture and reuse it for serial decode and verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionReductionPlan {
    capacity_rows: usize,
    active_drafts: Option<usize>,
    override_rows: Option<usize>,
    rows: usize,
}

impl AttentionReductionPlan {
    pub fn new(capacity_rows: usize, active_drafts: Option<usize>, override_rows: Option<usize>)
        -> Result<Self, AttentionPlanError> {
        if capacity_rows == 0 { return Err(AttentionPlanError::ZeroCapacity); }
        let automatic = match active_drafts {
            Some(drafts) => drafts.checked_add(1).ok_or(AttentionPlanError::DraftOverflow)?.min(capacity_rows),
            None => 1,
        };
        let rows = match override_rows {
            Some(rows) if rows == 0 || rows > capacity_rows =>
                return Err(AttentionPlanError::InvalidOverride { rows, capacity_rows }),
            Some(rows) => rows,
            None => automatic,
        };
        Ok(Self { capacity_rows, active_drafts, override_rows, rows })
    }

    pub fn rows(self) -> usize { self.rows }
    pub fn capacity_rows(self) -> usize { self.capacity_rows }
    pub fn active_drafts(self) -> Option<usize> { self.active_drafts }

    /// Alignment is a load contract, never a reason to change the reduction.
    pub fn sparse_recipe(self, compressed: bool) -> SparseAttentionRecipe {
        SparseAttentionRecipe { parts: if compressed { 10 } else { 2 },
            backend: if compressed { SparseAttentionBackend::Bf16Aot } else { SparseAttentionBackend::NativeFp32 } }
    }

    /// A row without indexed records must retain the window-only fold even when
    /// another row has records. Only mixed steps need more than one launch.
    pub fn mixed_indexed_rows(self, lengths: &[i32]) -> bool {
        lengths.first().is_some_and(|first|
            lengths.iter().any(|length| (*length > 0) != (*first > 0)))
    }

    pub fn indexed_ranges(self, lengths: &[i32])
        -> impl Iterator<Item = IndexedAttentionRange> + '_ {
        let mut start = 0;
        std::iter::from_fn(move || {
            let compressed = *lengths.get(start)? > 0;
            let rows = lengths[start..].iter().take_while(|length| (**length > 0) == compressed).count();
            let range = IndexedAttentionRange { start, rows, compressed };
            start += rows;
            Some(range)
        })
    }

    /// Admission may use validated checkpoint geometry before loading the drafter.
    /// Check raw geometry too: different widths can clamp to the same plan.
    pub fn validate_loaded(self, active_drafts: Option<usize>) -> Result<Self, AttentionPlanError> {
        let loaded = Self::new(self.capacity_rows, active_drafts, self.override_rows)?;
        if loaded != self {
            return Err(AttentionPlanError::DrafterMismatch {
                provisional: self.active_drafts, loaded: active_drafts,
            });
        }
        Ok(loaded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedAttentionRange {
    pub start: usize,
    pub rows: usize,
    pub compressed: bool,
}

/// A load-time dense output reduction. Live rows select storage, never arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputReductionPlan {
    capacity_rows: usize,
    expected_rows: usize,
    split_slices: usize,
}

impl OutputReductionPlan {
    pub fn new(capacity_rows: usize, expected_rows: usize, split_slices: usize) -> Result<Self, OutputPlanError> {
        if capacity_rows == 0 || expected_rows == 0 || !matches!(split_slices, 1 | 2 | 4) {
            return Err(OutputPlanError::InvalidRecipe);
        }
        Ok(Self { capacity_rows, expected_rows, split_slices })
    }
    pub fn expected_rows(self) -> usize { self.expected_rows }
    pub fn split_slices(self) -> usize { self.split_slices }
    pub fn validate(self, capacity: usize, expected_rows: usize, split_slices: usize,
        all_rows: bool) -> Result<(), OutputPlanError> {
        if capacity >= self.capacity_rows && expected_rows == self.expected_rows
            && split_slices == self.split_slices && all_rows { Ok(()) }
        else { Err(OutputPlanError::IncompatibleExport) }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OutputPlanError {
    #[error("invalid dense output reduction recipe")]
    InvalidRecipe,
    #[error("dense output export does not honor the immutable reduction plan")]
    IncompatibleExport,
}

/// Native rotary MLA drafting contract, shared by admission and manifest validation.
/// All exports process at most 64 rows; larger batches must be chunked by the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaDraftProgramLayout {
    pub context: usize,
    pub splits: usize,
    pub chunks_per_split: usize,
    pub partial_output_bytes: usize,
    pub partial_lse_bytes: usize,
    /// Producer, attention, ungated output, respectively. These phases are ordered.
    pub scratch: [usize; 3],
}

/// Which native block-draft program set a drafter runs: the DFlash2 MLA draft
/// (`mla_draft_*`, q-LoRA 1536 at hidden 7168) or Mistral Large 4's bundled
/// mtp_dspark (`ml4_draft_*`: no q-LoRA, hidden 6144, block-anchored window).
/// Both share the attention geometry (64 heads, 576/512 latent) and the
/// ungated UV/o output; they differ in the producer and its scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlaDraftKind { Dflash2, Ml4Dspark }

impl MlaDraftKind {
    pub fn hidden(self) -> usize { match self { Self::Dflash2 => 7168, Self::Ml4Dspark => 6144 } }

    /// Producer scratch at `rows`: DFlash2 [rows, 2112] q|kv_a then [rows, 64, 192] q;
    /// mtp_dspark [rows, 64, 128] q then [rows, 576] kv_a.
    fn producer_scratch(self, rows: usize) -> usize {
        let align = |n: usize| n.div_ceil(1024) * 1024;
        match self {
            Self::Dflash2 => align(rows * 2112 * 2) + align(rows * MlaDraftProgramLayout::HEADS * 192 * 2),
            Self::Ml4Dspark => align(rows * MlaDraftProgramLayout::HEADS * 128 * 2) + align(rows * 576 * 2),
        }
    }
}

impl MlaDraftProgramLayout {
    pub const ROWS: usize = 64;
    pub const HEADS: usize = 64;
    pub const PAGE: usize = 64;

    pub fn new(context: usize) -> Result<Self, MlaDraftLayoutError> {
        Self::for_kind(MlaDraftKind::Dflash2, context)
    }

    pub fn for_kind(kind: MlaDraftKind, context: usize) -> Result<Self, MlaDraftLayoutError> {
        if !(1..=1_048_576).contains(&context) { return Err(MlaDraftLayoutError); }
        let pages = context.div_ceil(Self::PAGE);
        let splits = pages.min(16);
        let align = |n: usize| n.div_ceil(1024) * 1024;
        let partial_output_bytes = if splits > 1 { align(Self::ROWS * Self::HEADS * splits * 512 * 2) } else { 0 };
        let partial_lse_bytes = if splits > 1 { align(Self::ROWS * Self::HEADS * splits * 4) } else { 0 };
        Ok(Self { context, splits, chunks_per_split: pages.div_ceil(splits),
            partial_output_bytes, partial_lse_bytes,
            scratch: [kind.producer_scratch(Self::ROWS),
                partial_output_bytes + partial_lse_bytes, align(Self::ROWS * Self::HEADS * 128 * 2)] })
    }

    /// One arena can cover all three phase-disjoint native scratch planes.
    pub fn scratch_bytes(self) -> usize { self.scratch.into_iter().max().unwrap_or(0) }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("MLA draft program context must be in 1..=1048576")]
pub struct MlaDraftLayoutError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseAttentionBackend { NativeFp32, Bf16Aot }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseAttentionRecipe {
    pub parts: usize,
    pub backend: SparseAttentionBackend,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AttentionPlanError {
    #[error("attention numerical plan capacity must be positive")]
    ZeroCapacity,
    #[error("active drafter width overflows verifier rows")]
    DraftOverflow,
    #[error("attention plan rows {rows} outside numerical domain 1..={capacity_rows}")]
    InvalidOverride { rows: usize, capacity_rows: usize },
    #[error("loaded attention drafter geometry {loaded:?} differs from admission geometry {provisional:?}")]
    DrafterMismatch { provisional: Option<usize>, loaded: Option<usize> },
}

// Literal registry consumed by release-common.sh as well as the serving owners.
pub const V41_PREFILL_MIN_ROWS: u32 = 80;
pub const V41_PREFILL_MAX_ROWS: u32 = 4096;
pub const V41_LIVE_MIN_ROWS: u32 = 256;
pub const V41_DECODER_REPLAY_ROWS: u32 = 128;
pub const V41_SERVING_AOT_ROWS: [u32; 3] = [256, 1024, 4096];

pub fn v41_live_rows(chunk: u32) -> Option<u32> {
    (V41_PREFILL_MIN_ROWS..=V41_PREFILL_MAX_ROWS).contains(&chunk)
        .then_some(chunk.max(V41_LIVE_MIN_ROWS).max(V41_DECODER_REPLAY_ROWS))
}

pub fn v41_aot_rows(chunk: u32) -> Option<u32> {
    let live = v41_live_rows(chunk)?;
    V41_SERVING_AOT_ROWS.into_iter().find(|&rows| rows >= live)
}
#[derive(Debug, Clone, Copy)]
pub struct CoordinatorPrograms<'a> {
    pub family: &'a str,
    pub split_family: Option<&'a str>,
}

/// Compound namespaces precede the ordinary first-underscore boundary. Keep
/// runtime selection and manifest admission on the same namespace contract.
pub fn program_namespace(name: &str) -> Option<&str> {
    for family in ["glm_stage", "glm2_stage"] {
        if name.strip_prefix(family).is_some_and(|suffix| suffix.starts_with('_') && suffix.len() > 1) {
            return Some(&name[..family.len()]);
        }
    }
    name.split_once('_').and_then(|(family, suffix)| (!suffix.is_empty()).then_some(family))
}

impl CoordinatorPrograms<'_> {
    /// Namespace boundaries matter: `glmf` must not select `glmf2` or `glm`.
    pub fn contains(self, name: &str) -> bool {
        program_namespace(name).is_some_and(|family| {
            family == self.family || self.split_family == Some(family)
        })
    }

    /// Shared arenas exclude index top-k, which owns a separate persistent buffer.
    pub fn shared_scratch<'a>(self, sizes: impl IntoIterator<Item = (&'a str, u64)>) -> u64 {
        sizes.into_iter().filter(|(name, _)| self.contains(name) && !name.contains("index_topk"))
            .map(|(_, bytes)| bytes).max().unwrap_or(0)
    }

    /// Unshaped helpers are shared; shaped exports belong only to the arena
    /// with their compiled row capacity and prefill/decode dispatch mode.
    pub fn workspace_program(self, name: &str, rows: u64, prefill: bool) -> bool {
        self.contains(name)
            && !name.contains(if prefill { "_decode_" } else { "_prefill_" })
            && name.rsplit_once("_m").and_then(|(_, suffix)| suffix.parse::<u64>().ok())
                .is_none_or(|capacity| capacity == rows)
    }

    pub fn validate_v4(self, prefill_rows: u64, decode_rows: u64,
        available: impl IntoIterator<Item = impl AsRef<str>>) -> Result<(), MissingCoordinatorProgram> {
        let available: std::collections::HashSet<String> = available.into_iter()
            .map(|name| name.as_ref().to_string()).collect();
        for name in self.v4_required(prefill_rows, decode_rows) {
            if !self.contains(&name) || !available.contains(&name) {
                return Err(MissingCoordinatorProgram(name));
            }
        }
        Ok(())
    }

    /// All V4 dispatch routes, including the unsplit dSpark target in split mode.
    /// Validate this list before weights, arenas or graph captures are allocated.
    pub fn v4_required(self, prefill_rows: u64, decode_rows: u64) -> Vec<String> {
        let mut names = Vec::new();
        let mut add = |family: &str, suffix: String| names.push(format!("{family}_{suffix}"));
        for suffix in ["mhc_pre", "mhc_post", "mhc_head", "router_scores", "expert_input_quant",
            "block_fp8_scale_prep"] {
            add(self.family, suffix.to_string());
        }
        for ratio in [4, 128] {
            for mode in ["decode", "prefill", "continuation"] {
                add(self.family, format!("compressor_{mode}_c{ratio}"));
            }
        }
        for (mode, rows) in [("decode", decode_rows), ("prefill", prefill_rows)] {
            for suffix in [format!("mhc_post_pre_m{rows}"), format!("index_producer_m{rows}"),
                format!("index_topk_{mode}_m{rows}")] {
                add(self.family, suffix);
            }
            for family in std::iter::once(self.family).chain(self.split_family) {
                for suffix in ["producer", "wo", "shared_ffn"] {
                    add(family, format!("{suffix}_m{rows}"));
                }
                for attention in ["win", "c4", "c128"] {
                    add(family, format!("sparse_mla_{mode}_{attention}_m{rows}"));
                }
            }
        }
        names
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingCoordinatorProgram(pub String);

impl std::fmt::Display for MissingCoordinatorProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V4 startup requires selected program {}", self.0)
    }
}

impl std::error::Error for MissingCoordinatorProgram {}

/// The `{family}_sparse_mla_partial_h{begin}_s{N}_m{rows}` variants a GPU with `sms` SMs runs,
/// as (row cap, program) ascending: the exporter's split plan per (SM count, row cap)
/// (`row_caps`; a variant without it serves the full `rows` of its `device_sm_counts`), else
/// the one variant. These caps describe numerical plan alternatives, not live row limits.
pub fn attention_partial_programs(manifest: &serde_json::Value, family: &str, begin: usize, rows: usize, sms: usize)
    -> Result<Vec<(usize, String)>, AttentionPartialError> {
    let programs = manifest["programs"].as_array().ok_or(AttentionPartialError::InvalidManifest("program manifest"))?;
    let prefix = format!("{family}_sparse_mla_partial_h{begin}_s");
    let suffix = format!("_m{rows}");
    let variants: Vec<&serde_json::Value> = programs.iter()
        .filter(|p| p["name"].as_str().is_some_and(|n| n.starts_with(&prefix) && n.ends_with(&suffix))).collect();
    let mut out: Vec<(usize, String)> = Vec::new();
    for p in &variants {
        let name = p["name"].as_str().ok_or(AttentionPartialError::InvalidManifest("program name"))?.to_string();
        let caps: Vec<usize> = match p["params"]["row_caps"].get(sms.to_string()) {
            Some(caps) => caps.as_array().ok_or(AttentionPartialError::InvalidManifest("row caps"))?.iter().filter_map(|c| c.as_u64().map(|c| c as usize)).collect(),
            None if p["params"]["device_sm_counts"].as_array()
                .is_some_and(|sm| sm.iter().any(|s| s.as_u64() == Some(sms as u64))) => vec![rows],
            None => Vec::new(),
        };
        out.extend(caps.into_iter().map(|cap| (cap, name.clone())));
    }
    if out.is_empty() && variants.len() == 1 {
        out.push((rows, variants[0]["name"].as_str().ok_or(AttentionPartialError::InvalidManifest("program name"))?.to_string()));
    }
    out.sort();
    out.dedup_by_key(|(cap, _)| *cap);
    if !out.last().is_some_and(|(cap, _)| *cap >= rows) {
        return Err(AttentionPartialError::MissingVariant { prefix, suffix, sms, rows });
    }
    Ok(out)
}

/// Select once at load, using numerical plan rows rather than live execution rows.
/// Every exported alternative has the same full execution capacity in its name.
pub fn attention_partial_program(manifest: &serde_json::Value, family: &str, begin: usize,
    execution_capacity_rows: usize, sms: usize,
    plan: &AttentionReductionPlan) -> Result<String, AttentionPartialError> {
    if plan.capacity_rows() > execution_capacity_rows { return Err(AttentionPartialError::NumericalDomain); }
    let partials = attention_partial_programs(manifest, family, begin, execution_capacity_rows, sms)?;
    partials.into_iter().find(|(cap, _)| *cap >= plan.rows()).map(|(_, name)| name)
        .ok_or(AttentionPartialError::NoRecipe)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AttentionPartialError {
    #[error("invalid attention partial {0}")]
    InvalidManifest(&'static str),
    #[error("no {prefix}*{suffix} variant planned for {sms} SMs at {rows} rows")]
    MissingVariant { prefix: String, suffix: String, sms: usize, rows: usize },
    #[error("attention numerical domain exceeds execution capacity")]
    NumericalDomain,
    #[error("attention plan has no partial reduction recipe")]
    NoRecipe,
}

/// Charge only the selected numerical recipe at full execution capacity.
/// Export geometry owns recipe scratch; this helper is independent of family names.
pub fn attention_plan_scratch(geometry: &serde_json::Value, plan: AttentionReductionPlan)
    -> Result<Option<u64>, AttentionScratchError> {
    let Some(recipes) = geometry.get("attention_plan_scratch") else { return Ok(None) };
    let invalid = || AttentionScratchError;
    let capacity = geometry["attention_plan_capacity_rows"].as_u64().ok_or_else(invalid)?;
    if capacity != plan.capacity_rows() as u64 { return Err(invalid()); }
    let recipes = recipes.as_array().ok_or_else(invalid)?;
    let mut previous = 0;
    let mut selected = None;
    for recipe in recipes {
        let cap = recipe["rows_cap"].as_u64().ok_or_else(invalid)?;
        let bytes = recipe["scratch"].as_u64().ok_or_else(invalid)?;
        if cap <= previous || cap > capacity { return Err(invalid()); }
        previous = cap;
        if selected.is_none() && cap >= plan.rows() as u64 { selected = Some(bytes); }
    }
    if previous != capacity { return Err(invalid()); }
    selected.map(Some).ok_or_else(invalid)
}

/// Narrow a parsed program's scratch using the immutable numerical recipe.
pub fn select_attention_scratch(geometry: &serde_json::Value,
    scratch: &mut std::collections::HashMap<String, u64>, plan: AttentionReductionPlan)
    -> Result<(), AttentionScratchError> {
    if let Some(bytes) = attention_plan_scratch(geometry, plan)? {
        scratch.insert("scratch".into(), bytes);
    }
    Ok(())
}

/// Planner counterpart of runtime program selection; card plans must be selected first.
pub fn select_attention_plans(manifest: &mut serde_json::Value, plan: AttentionReductionPlan,
    keep: impl Fn(&str) -> bool) -> Result<(), AttentionScratchError> {
    let Some(programs) = manifest["programs"].as_array_mut() else { return Ok(()) };
    for program in programs {
        if !program["name"].as_str().is_some_and(&keep)
            || program["geometry"].get("attention_plan_scratch").is_none() { continue; }
        let mut scratch = serde_json::from_value(program["scratch_bytes_at_capacity"].clone())
            .map_err(|_| AttentionScratchError)?;
        select_attention_scratch(&program["geometry"], &mut scratch, plan)?;
        program["scratch_bytes_at_capacity"] = serde_json::to_value(scratch)
            .map_err(|_| AttentionScratchError)?;
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid attention recipe scratch contract or numerical domain mismatch")]
pub struct AttentionScratchError;

/// Select the same 170/188-SM program recipe as native load, before admission.
/// The retained plans allow another device to select its own scratch envelope.
pub fn select_card_plans(manifest: &mut serde_json::Value, sms: u32) -> Result<(), CardPlanError> {
    let Some(programs) = manifest["programs"].as_array_mut() else { return Ok(()) };
    for program in programs {
        if program.get("card_plans").is_none() { continue; }
        let name = program["name"].as_str().unwrap_or("<unnamed>").to_string();
        let invalid = || CardPlanError { name: name.clone() };
        let plans = program["card_plans"].as_array().ok_or_else(invalid)?;
        if sms == 0 || plans.len() != 2 { return Err(invalid()); }
        let selected_sms = if sms >= 188 { 188 } else { 170 };
        let mut selected = None;
        for expected in [170, 188] {
            let matching: Vec<_> = plans.iter().filter(|p| p["sm_count"].as_u64() == Some(expected)).collect();
            if matching.len() != 1 { return Err(invalid()); }
            let plan = matching[0];
            let scratch = plan["scratch_bytes_at_capacity"].as_object().ok_or_else(invalid)?;
            if scratch.values().any(|v| v.as_u64().is_none()) || !plan["geometry"].is_object() {
                return Err(invalid());
            }
            if expected == selected_sms { selected = Some(plan.clone()); }
        }
        let plan = selected.ok_or_else(invalid)?;
        program["scratch_bytes_at_capacity"] = plan["scratch_bytes_at_capacity"].clone();
        program["geometry"] = plan["geometry"].clone();
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("program {name} has an invalid 170/188-SM card-plan contract")]
pub struct CardPlanError {
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_draft_scratch_preserves_export_capacity_and_ungated_output() {
        for context in [1, 63, 64, 65, 1024, 1025, 262144, 1048576] {
            let layout = MlaDraftProgramLayout::new(context).unwrap();
            let pages = context.div_ceil(64);
            assert_eq!(layout.splits, pages.min(16));
            assert_eq!(layout.chunks_per_split, pages.div_ceil(layout.splits));
            assert_eq!(layout.scratch[0], 1_843_200);
            assert_eq!(layout.scratch[2], 1_048_576, "ungated output has exactly one value plane");
            let partial = if layout.splits > 1 { 4_194_304 * layout.splits } else { 0 };
            let lse = if layout.splits > 1 { 16_384 * layout.splits } else { 0 };
            assert_eq!(layout.partial_output_bytes, partial);
            assert_eq!(layout.partial_lse_bytes, lse);
            assert_eq!(layout.scratch[1], partial + lse);
            assert_eq!(layout.scratch_bytes(), layout.scratch.iter().copied().max().unwrap());
        }
        assert_eq!(MlaDraftProgramLayout::new(262144).unwrap().scratch_bytes(), 67_371_008);
        assert_eq!(MlaDraftProgramLayout::new(262144).unwrap().scratch,
            MlaDraftProgramLayout::new(1048576).unwrap().scratch);
        for context in [0, 1048577, usize::MAX] {
            assert_eq!(MlaDraftProgramLayout::new(context), Err(MlaDraftLayoutError));
        }
    }

    #[test]
    fn output_recipe_is_immutable_and_rejects_partial_row_exports() {
        let plan = OutputReductionPlan::new(64, 16, 1).unwrap();
        for _rows in 1..=64 {
            assert_eq!((plan.expected_rows(), plan.split_slices()), (16, 1));
            assert!(plan.validate(80, 16, 1, true).is_ok());
        }
        assert!(plan.validate(16, 16, 1, true).is_err());
        assert!(plan.validate(80, 1, 2, true).is_err());
        assert!(plan.validate(80, 16, 1, false).is_err());
        assert!(OutputReductionPlan::new(0, 16, 1).is_err());
        assert!(OutputReductionPlan::new(64, 0, 1).is_err());
        assert!(OutputReductionPlan::new(64, 16, 3).is_err());
    }

    #[test]
    fn sparse_recipe_is_immutable_across_decode_verify_widths() {
        for plan_rows in 1..=64 {
            let plan = AttentionReductionPlan::new(64, Some(5), Some(plan_rows)).unwrap();
            for compressed in [false, true] {
                let expected = plan.sparse_recipe(compressed);
                for _live_rows in 1..=64 {
                    assert_eq!(plan.sparse_recipe(compressed), expected);
                    assert_eq!(plan.validate_loaded(Some(5)).unwrap(), plan);
                }
                assert_eq!(expected.parts, if compressed { 10 } else { 2 });
                assert_eq!(expected.backend, if compressed { SparseAttentionBackend::Bf16Aot }
                    else { SparseAttentionBackend::NativeFp32 });
            }
            assert!(plan.validate_loaded(Some(7)).is_err());
            assert_eq!(plan.rows(), plan_rows);
        }
    }

    #[test]
    fn indexed_rows_keep_their_serial_recipe_across_boundaries_and_requests() {
        let plan = AttentionReductionPlan::new(64, Some(5), None).unwrap();
        for lengths in [vec![], vec![0], vec![1], vec![0; 6], vec![1; 6],
            vec![0, 0, 1, 1, 1, 1], vec![1, 0, 0, 1, 0, 2]] {
            let ranges: Vec<_> = plan.indexed_ranges(&lengths).collect();
            assert_eq!(ranges.iter().map(|r| r.rows).sum::<usize>(), lengths.len());
            assert_eq!(plan.mixed_indexed_rows(&lengths), ranges.len() > 1);
            for range in ranges {
                assert!(range.rows > 0);
                for &length in &lengths[range.start..range.start + range.rows] {
                    assert_eq!(range.compressed, length > 0);
                    assert_eq!(range.compressed, plan.indexed_ranges(&[length]).next().unwrap().compressed);
                }
            }
        }
        assert_eq!(plan.indexed_ranges(&[0, 0, 1, 1]).collect::<Vec<_>>(), vec![
            IndexedAttentionRange { start: 0, rows: 2, compressed: false },
            IndexedAttentionRange { start: 2, rows: 2, compressed: true }]);
        // Several requests flatten into one verify. The route depends only on
        // this position's visible records, not a later row or request.
        for starts in [[93, 123, 129], [122, 126, 127]] {
            let lengths: Vec<_> = starts.into_iter().flat_map(|start|
                (start..start + 6).map(|position| ((position + 1) / 128) as i32)).collect();
            let ranges: Vec<_> = plan.indexed_ranges(&lengths).collect();
            for (row, length) in lengths.iter().enumerate() {
                let range = ranges.iter().find(|r| (r.start..r.start + r.rows).contains(&row)).unwrap();
                assert_eq!(range.compressed, *length > 0);
            }
        }
    }

    #[test]
    fn attention_plan_uses_active_drafts_not_execution_width() {
        for (drafts, rows) in [(None, 1), (Some(0), 1), (Some(7), 8), (Some(63), 64), (Some(127), 64)] {
            let plan = AttentionReductionPlan::new(64, drafts, None).unwrap();
            assert_eq!(plan.rows(), rows);
            assert_eq!(plan.validate_loaded(drafts).unwrap(), plan);
        }
        assert_eq!(AttentionReductionPlan::new(64, Some(7), Some(1)).unwrap().rows(), 1);
        assert_eq!(AttentionReductionPlan::new(64, None, Some(8)).unwrap().rows(), 8);
    }

    #[test]
    fn attention_plan_rejects_invalid_geometry_and_unclamped_overrides() {
        assert_eq!(AttentionReductionPlan::new(0, None, None), Err(AttentionPlanError::ZeroCapacity));
        assert_eq!(AttentionReductionPlan::new(64, Some(usize::MAX), Some(1)), Err(AttentionPlanError::DraftOverflow));
        for rows in [0, 65, usize::MAX] {
            assert_eq!(AttentionReductionPlan::new(64, None, Some(rows)),
                Err(AttentionPlanError::InvalidOverride { rows, capacity_rows: 64 }));
        }
    }

    #[test]
    fn attention_plan_checks_raw_loaded_geometry_even_when_clamped_or_overridden() {
        for override_rows in [None, Some(8)] {
            let plan = AttentionReductionPlan::new(64, Some(80), override_rows).unwrap();
            assert_eq!(plan.validate_loaded(Some(81)), Err(AttentionPlanError::DrafterMismatch {
                provisional: Some(80), loaded: Some(81),
            }));
        }
        assert!(AttentionReductionPlan::new(64, None, None).unwrap().validate_loaded(Some(0)).is_err());
    }

    #[test]
    fn attention_scratch_reserves_selected_recipe_at_full_capacity() {
        let geometry = serde_json::json!({"attention_plan_capacity_rows": 64,
            "attention_plan_scratch": [{"rows_cap": 1, "scratch": 67371008},
                {"rows_cap": 8, "scratch": 23158784}, {"rows_cap": 64, "scratch": 8421376}]});
        for (rows, bytes) in [(1, 67371008), (8, 23158784), (9, 8421376), (64, 8421376)] {
            let plan = AttentionReductionPlan::new(64, None, Some(rows)).unwrap();
            assert_eq!(attention_plan_scratch(&geometry, plan).unwrap(), Some(bytes));
        }
        let plan = AttentionReductionPlan::new(64, Some(7), None).unwrap();
        assert_eq!(attention_plan_scratch(&serde_json::json!({}), plan).unwrap(), None);
        for defect in 0..4 {
            let mut bad = geometry.clone();
            match defect {
                0 => bad["attention_plan_capacity_rows"] = 128.into(),
                1 => bad["attention_plan_scratch"][1]["rows_cap"] = 1.into(),
                2 => bad["attention_plan_scratch"][2]["rows_cap"] = 63.into(),
                _ => bad["attention_plan_scratch"][0]["scratch"] = (-1).into(),
            }
            assert!(attention_plan_scratch(&bad, plan).is_err());
        }
    }

    #[test]
    fn planner_and_runtime_attention_scratch_match_every_plan_on_both_cards() {
        let geometry = |scale: u64| serde_json::json!({"attention_plan_capacity_rows": 64,
            "attention_plan_scratch": [{"rows_cap": 1, "scratch": 134742016 * scale},
                {"rows_cap": 8, "scratch": 46317568 * scale},
                {"rows_cap": 64, "scratch": 8421376 * scale}]});
        for sms in [170, 188] {
            for rows in 1..=64 {
                let plan = AttentionReductionPlan::new(64, None, Some(rows)).unwrap();
                let mut manifest = serde_json::json!({"programs": [{"name": "glm_sparse_mla_decode_m64",
                    "geometry": geometry(1), "scratch_bytes_at_capacity": {"scratch": 134742016},
                    "card_plans": [
                        {"sm_count": 170, "geometry": geometry(2), "scratch_bytes_at_capacity": {"scratch": 269484032}},
                        {"sm_count": 188, "geometry": geometry(1), "scratch_bytes_at_capacity": {"scratch": 134742016}}
                    ]}, {"name": "other_attention", "geometry": geometry(1),
                        "scratch_bytes_at_capacity": {"scratch": 99}}]});
                select_card_plans(&mut manifest, sms).unwrap();
                let program = &manifest["programs"][0];
                let mut runtime = serde_json::from_value(program["scratch_bytes_at_capacity"].clone()).unwrap();
                select_attention_scratch(&program["geometry"], &mut runtime, plan).unwrap();
                select_attention_plans(&mut manifest, plan, |name| name.starts_with("glm_")).unwrap();
                assert_eq!(manifest["programs"][0]["scratch_bytes_at_capacity"], serde_json::to_value(runtime).unwrap());
                assert_eq!(manifest["programs"][1]["scratch_bytes_at_capacity"]["scratch"], 99);
            }
        }
    }

    fn dual_manifest() -> serde_json::Value {
        serde_json::json!({"programs": [{"name": "dsv4p_sparse_mla_decode_c128_m64",
            "scratch_bytes_at_capacity": {"scratch": 1094778880}, "geometry": {},
            "card_plans": [
                {"sm_count": 170, "scratch_bytes_at_capacity": {"scratch": 1094778880}, "geometry": {"num_splits": 130}},
                {"sm_count": 188, "scratch_bytes_at_capacity": {"scratch": 8421376}, "geometry": {"num_splits": 1}}
            ]}, {"name": "dsv4p_wo", "scratch_bytes_at_capacity": {"scratch": 9}}]})
    }

    #[test]
    fn card_plan_scratch_tracks_native_selection_without_losing_the_other_plan() {
        let mut manifest = dual_manifest();
        for (sms, bytes, splits) in [(188, 8421376, 1), (170, 1094778880, 130), (188, 8421376, 1)] {
            select_card_plans(&mut manifest, sms).unwrap();
            assert_eq!(manifest["programs"][0]["scratch_bytes_at_capacity"]["scratch"], bytes);
            assert_eq!(manifest["programs"][0]["geometry"]["num_splits"], splits);
            assert_eq!(manifest["programs"][1]["scratch_bytes_at_capacity"]["scratch"], 9);
        }
    }

    #[test]
    fn malformed_card_plans_fail_closed_instead_of_using_the_capacity_envelope() {
        for defect in 0..5 {
            let mut manifest = dual_manifest();
            match defect {
                0 => manifest["programs"][0]["card_plans"] = serde_json::json!([]),
                1 => manifest["programs"][0]["card_plans"][1]["sm_count"] = 170.into(),
                2 => manifest["programs"][0]["card_plans"][1]["scratch_bytes_at_capacity"]["scratch"] = (-1).into(),
                3 => manifest["programs"][0]["card_plans"][1]["geometry"] = serde_json::Value::Null,
                _ => {},
            }
            let error = select_card_plans(&mut manifest, if defect == 4 { 0 } else { 188 }).unwrap_err();
            assert_eq!(error.name, "dsv4p_sparse_mla_decode_c128_m64");
        }
    }

    #[test]
    fn v41_serving_rows_cover_live_chunks_and_decoder_replay() {
        for chunk in V41_PREFILL_MIN_ROWS..=V41_PREFILL_MAX_ROWS {
            let live = v41_live_rows(chunk).unwrap();
            let aot = v41_aot_rows(chunk).unwrap();
            assert!(live >= chunk && live >= V41_DECODER_REPLAY_ROWS && live >= V41_LIVE_MIN_ROWS);
            assert!(aot >= live && V41_SERVING_AOT_ROWS.contains(&aot));
        }
        for invalid in [0, 79, 4097, u32::MAX] {
            assert_eq!(v41_live_rows(invalid), None);
            assert_eq!(v41_aot_rows(invalid), None);
        }
    }

    #[test]
    fn namespace_selection_keeps_only_serving_and_split_families() {
        let families = ["dsv4f", "dsv4p", "glm", "glmf", "mimo", "mimof", "mimop", "qwen4"];
        for family in families {
            let split_family = format!("{family}2");
            for split in [false, true] {
                let selected = CoordinatorPrograms { family, split_family: split.then_some(split_family.as_str()) };
                assert!(selected.contains(&format!("{family}_target")));
                assert_eq!(selected.contains(&format!("{family}2_target")), split);
                for other in families.into_iter().filter(|other| *other != family) {
                    assert!(!selected.contains(&format!("{other}_target")));
                    assert!(!selected.contains(&format!("{other}2_target")));
                }
                assert!(!selected.contains(family));
            }
        }
    }

    #[test]
    fn workspace_selection_keeps_only_its_shape_and_shared_helpers() {
        let selected = CoordinatorPrograms { family: "dsv4f", split_family: Some("dsv4f2") };
        for family in ["dsv4f", "dsv4f2"] {
            for prefill in [false, true] {
                let rows = if prefill { 4096 } else { 64 };
                let mode = if prefill { "prefill" } else { "decode" };
                assert!(selected.workspace_program(&format!("{family}_producer_m{rows}"), rows, prefill));
                assert!(selected.workspace_program(&format!("{family}_sparse_mla_{mode}_c128_m{rows}"), rows, prefill));
                assert!(selected.workspace_program(&format!("{family}_compressor_continuation_c4"), rows, prefill));
                assert!(!selected.workspace_program(&format!("{family}_producer_m{}", if prefill { 64 } else { 4096 }), rows, prefill));
                assert!(!selected.workspace_program("unrelated_producer_m64", rows, prefill));
                let other = if prefill { "decode" } else { "prefill" };
                assert!(!selected.workspace_program(&format!("{family}_index_topk_{other}_m{rows}"), rows, prefill));
            }
        }
    }

    #[test]
    fn native_stages_are_not_plain_glm_programs_or_scratch() {
        for split in [false, true] {
            let selected = CoordinatorPrograms { family: "glm", split_family: split.then_some("glm2") };
            assert!(!selected.contains("glm_stage_rank_pack_m64"));
            assert!(!selected.contains("glm2_stage_swiglu_i1024"));
            assert_eq!(selected.shared_scratch([("glm_producer_m64", 7),
                ("glm_stage_rank_pack_m64", 900), ("glm2_stage_swiglu_i1024", 999)]), 7);
        }
        let stages = CoordinatorPrograms { family: "glm_stage", split_family: Some("glm2_stage") };
        assert!(stages.contains("glm_stage_rank_pack_m64"));
        assert!(stages.contains("glm2_stage_swiglu_i1024"));
        assert!(!stages.contains("glm_producer_m64"));
    }

    #[test]
    fn split_v4_keeps_unsplit_draft_programs() {
        for family in ["dsv4f", "dsv4p"] {
            let split_family = format!("{family}2");
            let selected = CoordinatorPrograms { family, split_family: Some(&split_family) };
            let names = selected.v4_required(4096, 64);
            assert!(names.iter().all(|name| selected.contains(name)));
            for suffix in ["producer_m64", "producer_m4096", "wo_m64", "sparse_mla_decode_win_m64"] {
                assert!(names.contains(&format!("{family}_{suffix}")));
                assert!(names.contains(&format!("{split_family}_{suffix}")));
            }
        }
    }
}
