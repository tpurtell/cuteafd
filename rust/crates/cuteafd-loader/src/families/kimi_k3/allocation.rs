//! CPU allocation layouts shared by Kimi checkpoint planning and native owners.
use super::config::{KimiAttention, KimiK3Config};
use crate::placement::{Demand, GraphRank, GraphSet, Lifetime};
use crate::serving_capacity::Exl3Epilogue;
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeScratch {
    pub kda: usize,
    pub mla: [usize; 3],
}

impl NativeScratch {
    pub fn combine(kda: [usize; 2], mla: impl IntoIterator<Item = [usize; 3]>) -> Result<Self> {
        let mut scratch = Self { kda: *kda.iter().max().unwrap(), mla: [0; 3] };
        for requirements in mla {
            for (current, required) in scratch.mla.iter_mut().zip(requirements) {
                *current = (*current).max(required);
            }
        }
        scratch.bytes()?;
        Ok(scratch)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.mla.iter().try_fold(self.kda, |sum, bytes| sum.checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("Kimi native scratch extent overflow")))
    }
}

/// Most rows one kernel launch (one prefill unit) handles.
pub const UNIT_ROWS_MAX: usize = 4096;
/// Largest shared live span: three page-aligned units of 2752 rows.
/// Each lane allocates its exact admitted span, never this validation ceiling.
pub const LANE_ROWS_MAX: usize = (crate::serving_capacity::prefill_waves::RESERVE_ROWS + 64) as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLaneLayout {
    /// Lane rows: the call's row planes, each unit owning its rows of them.
    pub rows: usize,
    /// Unit rows: hook-local scratch (the phase arena and K6 stages)
    /// serves one unit's kernel launch at a time.
    pub unit: usize,
    pub max_context: usize,
    pub operands: BTreeMap<&'static str, usize>,
    pub arena_offsets: BTreeMap<&'static str, usize>,
    pub arena_bytes: usize,
}

pub const SCRATCH_PHASES: [&[&str]; 3] = [
    &["KDA normalized", "KDA scratch"],
    &["MLA query", "MLA q residual", "MLA attention", "MLA LSE",
        "MLA producer scratch", "MLA attention scratch", "MLA output scratch"],
    &["gate-up", "activated"],
];
/// Live operands at consecutive rank-local kernel boundaries. Projection scratch
/// is a separate owner; all these phases execute on the captured lane stream.
pub const CORE_PHASES: &[&[&str]] = &[
    &["K6 q", "K6 k", "K6 v", "K6 gate"],
    &["K6 q", "K6 k", "K6 v", "K6 gate", "K6 normalized", "K6 KDA scratch"],
    &["K6 normalized"],
    &["K6 q_a", "K6 kv_a", "MLA q residual", "K6 MLA scratch"],
    &["K6 q_b", "MLA q residual", "MLA query"],
    &["MLA query", "MLA attention", "MLA LSE", "MLA attention scratch"],
    &["MLA attention", "K6 gate", "K6 normalized"],
    &["K6 FFN gate", "K6 FFN up"],
    &["K6 FFN gate", "K6 FFN up", "gate-up"],
    &["gate-up", "activated"],
    &["activated"],
];
pub const CORE_MEET_PLANES: [&str; 3] = ["K6 attention partial", "K6 FFN partial", "K6 reduced"];
pub const ARENA_ALIGNMENT: usize = 1024;
pub const ATTENTION_METADATA: [&str; 6] = ["KDA slots", "KDA first", "MLA slots",
    "MLA page table", "MLA lengths", "MLA query offsets"];
/// Consumed entirely on the lane stream inside route/latent_up. Neither peer
/// exchange nor a pending Spark wave retains these intermediates.
pub const HOOK_SCRATCH: [&str; 3] = ["router logits", "latent input", "latent normalized"];
/// Index metadata survives every layer of its unit, including independently
/// addressed full-history index pages when MLA uses compact Context pages.
pub const DSA_METADATA: [&str; 6] = ["DSA positions", "DSA slots", "DSA page table",
    "DSA lengths", "DSA query offsets", "DSA selected lengths"];
/// Unit-local index temporaries are not lane row planes. Keep their backing
/// distinct from projected-core scratch until all index and attention readers end.
pub const DSA_SCRATCH: [&str; 6] = ["DSA query", "DSA head weights", "DSA indices", "DSA scores",
    "DSA producer scratch", "DSA selector scratch"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaScratch {
    pub producer: usize,
    pub selector: usize,
}

impl DsaScratch {
    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self> {
        let mut required = Self { producer: 0, selector: 0 };
        for rows in [64, 4096] {
            required.producer = required.producer.max(program_scratch(manifest,
                &format!("kimik31_index_producer_m{rows}"), rows)?);
            let mode = if rows == 64 { "decode" } else { "prefill" };
            for stem in ["index_topk", "scored_index_topk"] {
                required.selector = required.selector.max(program_scratch(manifest,
                    &format!("kimik31_{stem}_{mode}_m{rows}"), rows)?);
            }
        }
        ensure!(required.producer > 0 && required.selector > 0, "Kimi DSA scratch must be declared");
        Ok(required)
    }
}

/// Read the K3.1 compiled descriptor and validate every sparse specialization.
pub fn sparse_mla_chunk(manifest: &serde_json::Value) -> Result<usize> {
    let programs = manifest["programs"].as_array().context("Kimi program inventory missing")?;
    let mut policy = None;
    for rows in [64, 4096] {
        let name = format!("kimik31_index_producer_m{rows}");
        let entries = programs.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
        ensure!(entries.len() == 1, "Kimi requires one {name} descriptor");
        let chunk = cuteafd_ffi::KimiSparseMla::descriptor_chunk_size(&entries[0]["geometry"])?;
        ensure!(policy.is_none_or(|p| p == chunk), "Kimi mixed sparse MLA descriptor chunks");
        policy = Some(chunk);
    }
    let chunk = policy.context("Kimi sparse MLA descriptor absent")?;
    for rows in [64usize, 4096] { for cache in ["", "_fp8"] {
        for suffix in ["", "_h96", "_partial_h0", "_partial_h48"] {
            let name = if let Some(head) = suffix.strip_prefix("_partial_") {
                format!("kimik31_sparse_mla_partial{cache}_{head}_m{rows}")
            } else { format!("kimik31_sparse_mla{suffix}{cache}_m{rows}") };
            let entries = programs.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
            ensure!(entries.len() == 1, "Kimi requires one {name} chunk inventory");
            let entry = entries[0];
            ensure!(entry["geometry"]["row_chunk_size"].as_u64() == Some(rows.min(chunk) as u64),
                "Kimi {name} chunk disagrees with descriptor");
            let partial = suffix.starts_with("_partial");
            let heads = if suffix.is_empty() { 48 } else { 96 };
            let expected = cuteafd_ffi::KimiSparseMla::scratch_bytes_for(rows.min(chunk), heads, partial)?;
            ensure!(program_scratch(manifest, &name, rows)? == expected,
                "Kimi {name} scratch disagrees with chunk");
        }
    }}
    Ok(chunk)
}

fn program_scratch(manifest: &serde_json::Value, name: &str, rows: usize) -> Result<usize> {
    let entries = manifest["programs"].as_array().context("Kimi program inventory missing")?;
    let mut found = entries.iter().filter(|entry| entry["name"] == name);
    let entry = found.next().with_context(|| format!("Kimi missing {name} scratch inventory"))?;
    ensure!(found.next().is_none(), "Kimi duplicate {name} scratch inventory");
    ensure!(entry["capacity_rows"].as_u64() == Some(rows as u64), "Kimi {name} scratch capacity differs");
    let scratch = entry["scratch_bytes_at_capacity"].as_object().context("Kimi scratch inventory missing")?;
    ensure!(scratch.len() == 1, "Kimi {name} scratch inventory has undeclared owners");
    usize::try_from(scratch.get("scratch").and_then(serde_json::Value::as_u64).context("Kimi scratch byte extent missing")?)
        .context("Kimi scratch byte extent overflow")
}

impl NativeLaneLayout {
    fn metadata_banks_for(rows: usize, unit: usize) -> usize {
        if rows <= 64 { return 1; }
        rows.div_ceil(unit)
    }

    pub fn metadata_banks(&self) -> usize { Self::metadata_banks_for(self.rows, self.unit) }

    /// Each bank has the decode-capable extent; unit tails can be <=64 rows.
    pub fn metadata_extent(&self, name: &str, bank: usize) -> Result<(usize, usize)> {
        ensure!((ATTENTION_METADATA.contains(&name) || DSA_METADATA.contains(&name)) && bank < self.metadata_banks(),
            "Kimi attention metadata bank outside admission");
        let bytes = self.operands.get(name).context("Kimi metadata plane absent")?;
        ensure!(bytes % self.metadata_banks() == 0, "Kimi metadata bank stride");
        let stride = bytes / self.metadata_banks();
        ensure!(stride >= 16 && stride % 16 == 0, "Kimi metadata bank alignment");
        Ok((bank * stride, stride))
    }

    /// `rows` lane rows hold units of at most `unit` rows; one-unit lanes pass `unit == rows`.
    pub fn new(cfg: &KimiK3Config, rows: usize, unit: usize, max_context: usize,
        scratch: NativeScratch) -> Result<Self> {
        Self::for_ranks(cfg, rows, unit, max_context, scratch, 2)
    }

    pub fn for_ranks(cfg: &KimiK3Config, rows: usize, unit: usize, max_context: usize,
        scratch: NativeScratch, ranks: usize) -> Result<Self> {
        cfg.release_geometry()?;
        ensure!(matches!(ranks, 1 | 2), "Kimi native lane physical rank count");
        ensure!((1..=LANE_ROWS_MAX).contains(&rows) && (1..=UNIT_ROWS_MAX.min(rows)).contains(&unit)
            && (1..=1_048_576).contains(&max_context), "Kimi lane row/unit/context capacity");
        let mut operands = BTreeMap::new();
        let mut plane = |name, shape: &[usize]| -> Result<()> {
            let bytes = shape.iter().try_fold(1usize, |n, size| n.checked_mul(*size)
                .ok_or_else(|| anyhow::anyhow!("Kimi lane {name} extent overflow")))?;
            ensure!(operands.insert(name, bytes.max(16)).is_none(), "duplicate Kimi lane operand");
            Ok(())
        };
        let hidden = [rows, cfg.hidden, 2];
        plane("residual arena", &[9, rows, cfg.hidden, 2])?;
        for name in ["delta", "normalized", "attention output", "ffn output", "shared output", "routed output", "combined output"] {
            plane(name, &hidden)?;
        }
        let intermediate = (cfg.dense_intermediate / ranks).max(cfg.moe_intermediate * cfg.shared_experts / ranks);
        plane("gate-up", &[unit, 2, intermediate, 2])?;
        plane("activated", &[unit, intermediate, 2])?;
        plane("expert combined", &[rows, cfg.latent, 2])?;
        for name in ["latent input", "latent normalized"] { plane(name, &[unit, cfg.latent, 2])?; }
        plane("router logits", &[unit, cfg.experts, 4])?;
        for name in ["route ids", "route weights"] { plane(name, &[rows, cfg.topk, 4])?; }
        plane("expert wire", &[rows, (3584 + 3584 / 32)])?;
        // Phase-arena operands serve one unit's rows. Metadata has one bank
        // per possible call unit (sized below), never rewritten per layer.
        plane("KDA normalized", &[unit, cfg.kda_heads / ranks, cfg.kda_head_dim, 2])?;
        for name in ["KDA slots", "KDA first"] { plane(name, &[unit, 4])?; }
        plane("MLA slots", &[unit, 8])?;
        plane("MLA query", &[unit, cfg.heads, 576, 2])?;
        plane("MLA q residual", &[unit, cfg.q_lora_rank, 2])?;
        plane("MLA attention", &[unit, cfg.heads, 512, 2])?;
        plane("MLA LSE", &[unit, cfg.heads, 4])?;
        // Decode/verify permits one request per row, at most the 64-row
        // specialization. Wide prefill is one sequence. Cache pages live outside.
        let requests = unit.min(64);
        plane("MLA page table", &[requests, max_context.div_ceil(64), 4])?;
        plane("MLA lengths", &[requests, 4])?;
        plane("MLA query offsets", &[requests + 1, 4])?;
        plane("KDA scratch", &[scratch.kda])?;
        for (name, bytes) in ["MLA producer scratch", "MLA attention scratch", "MLA output scratch"]
            .into_iter().zip(scratch.mla) { plane(name, &[bytes])?; }
        // A call stages each unit once. Metadata survives all its layers, so
        // in-flight units require disjoint banks, including a <=64-row tail.
        // Native kernels require every bank pointer, not just its base, aligned.
        let banks = Self::metadata_banks_for(rows, unit);
        for name in ATTENTION_METADATA {
            let bytes = operands.get_mut(name).expect("attention metadata plane");
            let stride = bytes.checked_add(15).context("Kimi metadata alignment overflow")? / 16 * 16;
            *bytes = stride.checked_mul(banks).context("Kimi metadata banks overflow")?;
        }
        let mut arena_offsets = BTreeMap::new();
        let mut arena_bytes = 0;
        for phase in SCRATCH_PHASES {
            let mut end = 0usize;
            for &name in phase {
                let offset = end.checked_add(ARENA_ALIGNMENT - 1)
                    .map(|n| n / ARENA_ALIGNMENT * ARENA_ALIGNMENT)
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase alignment overflow"))?;
                ensure!(arena_offsets.insert(name, offset).is_none(), "duplicate Kimi arena operand");
                end = offset.checked_add(operands[name])
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase extent overflow"))?;
            }
            arena_bytes = arena_bytes.max(end);
        }
        let layout = Self { rows, unit, max_context, operands, arena_offsets, arena_bytes };
        layout.bytes()?;
        Ok(layout)
    }

    /// Sparse MLA uses runtime row offsets. Each placed layer's head width
    /// shares this one serialized attention scratch owner on its rank.
    pub fn with_sparse_mla_rows(mut self, heads: usize, chunk: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&chunk), "Kimi sparse MLA chunk outside exported bounds");
        let bytes = cuteafd_ffi::KimiSparseMla::scratch_bytes_for(self.unit.max(64).min(chunk), heads, false)?;
        let old = self.operands.insert("MLA attention scratch", bytes).context("Kimi MLA attention scratch absent")?;
        ensure!(old >= bytes, "Kimi sparse attention scratch exceeds validated exported capacity");
        self.arena_offsets.clear();
        self.arena_bytes = 0;
        for phase in SCRATCH_PHASES {
            let mut end = 0usize;
            for &name in phase {
                let offset = end.next_multiple_of(ARENA_ALIGNMENT);
                self.arena_offsets.insert(name, offset);
                end = offset.checked_add(self.operands[name]).context("Kimi sparse phase extent overflow")?;
            }
            self.arena_bytes = self.arena_bytes.max(end);
        }
        self.bytes()?;
        Ok(self)
    }

    /// Projection offsets use live rows; sparse selection still retains its
    /// AOT-baked selector and recurrence workspaces at their declared capacity.
    pub fn with_live_projection_rows(mut self, heads: usize, ranks: usize) -> Result<Self> {
        let rows = self.unit.max(64);
        for (name, stride) in [("MLA query", heads * 576 * 2), ("MLA attention", heads * 512 * 2),
            ("MLA LSE", heads * 4)] {
            self.operands.insert(name, self.unit * stride);
        }
        let [producer, output] = cuteafd_ffi::KimiMlaProjection::scratch_bytes_for(rows, heads)?;
        for (name, bytes) in [("MLA producer scratch", producer), ("MLA output scratch", output),
            ("DSA producer scratch", cuteafd_ffi::KimiIndexProducer::scratch_bytes_for(rows)?)] {
            let old = self.operands.insert(name, bytes).context("Kimi projection owner absent")?;
            ensure!(old >= bytes, "Kimi live projection exceeds exported scratch");
        }
        let old = self.operands["KDA scratch"];
        self.operands.insert("KDA scratch", cuteafd_ffi::KimiKda::scratch_bytes_for(rows, 4096, 96 / ranks, old)?);
        self.arena_offsets.clear();
        self.arena_bytes = 0;
        for phase in SCRATCH_PHASES {
            let mut end = 0usize;
            for &name in phase {
                let offset = end.next_multiple_of(ARENA_ALIGNMENT);
                self.arena_offsets.insert(name, offset);
                end = offset.checked_add(self.operands[name]).context("Kimi live phase extent overflow")?;
            }
            self.arena_bytes = self.arena_bytes.max(end);
        }
        self.bytes()?;
        Ok(self)
    }

    /// Add exact index owners without changing any existing BF16 or core phase
    /// offsets. Context candidate exchange/partial owners are admitted separately.
    pub fn with_dsa(mut self, cfg: &KimiK3Config, scratch: DsaScratch) -> Result<Self> {
        ensure!(cfg.dsa.is_some_and(|index| (index.heads, index.head_dim, index.topk) == (32, 128, 2048)),
            "Kimi DSA lane requires the native 32x128/top2048 indexer");
        ensure!(scratch.producer > 0 && scratch.selector > 0, "Kimi DSA scratch missing");
        let rows = self.unit;
        let requests = rows.min(64);
        for (name, bytes) in [("DSA query", rows * 4096), ("DSA head weights", rows * 32 * 4),
            ("DSA indices", rows * 2048 * 4), ("DSA scores", rows * 2048 * 4),
            ("DSA producer scratch", scratch.producer),
            ("DSA selector scratch", scratch.selector)] {
            ensure!(self.operands.insert(name, bytes).is_none(), "duplicate Kimi DSA scratch owner {name}");
        }
        for (name, bytes) in [("DSA positions", rows * 8), ("DSA slots", rows * 8),
            ("DSA page table", requests * self.max_context.div_ceil(64) * 4),
            ("DSA lengths", rows * 4), ("DSA query offsets", (rows + 1) * 4),
            ("DSA selected lengths", rows * 4)] {
            let stride = bytes.checked_add(15).context("Kimi DSA metadata alignment overflow")? / 16 * 16;
            let bytes = stride.checked_mul(self.metadata_banks()).context("Kimi DSA metadata banks overflow")?;
            ensure!(self.operands.insert(name, bytes).is_none(), "duplicate Kimi DSA metadata owner {name}");
        }
        self.bytes()?;
        Ok(self)
    }

    /// Page-table banks follow the served extent; kernel phase offsets stay fixed.
    pub fn with_served_context(&self, context: usize) -> Result<Self> {
        ensure!((1..=self.max_context).contains(&context), "Kimi lane context exceeds prepared extent");
        let mut layout = self.clone();
        for name in ["MLA page table", "DSA page table"] {
            if let Some(bytes) = layout.operands.get_mut(name) {
                *bytes = (self.unit.min(64) * context.div_ceil(64) * 4)
                    .max(16).next_multiple_of(16) * self.metadata_banks();
            }
        }
        layout.max_context = context;
        layout.bytes()?;
        Ok(layout)
    }

    /// Preserve every BF16 offset. Place transient K6 operands into holes whose
    /// previous reader has completed on the same lane stream. Peer meet outputs
    /// remain separate allocations because the exchange stream reads them.
    pub fn with_core_stages(mut self, stages: BTreeMap<&'static str, usize>) -> Result<Self> {
        let mut transient = stages.iter().filter(|(name, _)| !CORE_MEET_PLANES.contains(name)
            && !matches!(**name, "native attention F32 partial" | "native attention FP8 scratch"))
            .map(|(&name, &bytes)| (name, bytes)).collect::<Vec<_>>();
        transient.sort_by_key(|&(name, bytes)| (std::cmp::Reverse(bytes), name));
        ensure!(stages.keys().all(|name| !self.operands.contains_key(name)), "duplicate Kimi core arena operand");
        self.operands.extend(stages);
        let phases = self.core_phases();
        for (name, bytes) in transient {
            let conflicts = phases.iter().map(Vec::as_slice).chain(SCRATCH_PHASES)
                .filter(|phase| phase.contains(&name)).flat_map(|phase| phase.iter().copied())
                .filter(|other| *other != name && self.arena_offsets.contains_key(other))
                .collect::<std::collections::BTreeSet<_>>();
            ensure!(CORE_PHASES.iter().any(|phase| phase.contains(&name)), "Kimi stage lacks a live phase: {name}");
            let mut candidates = vec![0];
            for other in &conflicts {
                candidates.push(self.arena_offsets[other].checked_add(self.operands[other])
                    .context("Kimi core arena candidate overflow")?);
            }
            candidates.sort_unstable();
            let mut selected = None;
            for start in candidates {
                let start = start.checked_add(ARENA_ALIGNMENT - 1)
                    .map(|n| n / ARENA_ALIGNMENT * ARENA_ALIGNMENT).context("Kimi core arena alignment overflow")?;
                let end = start.checked_add(bytes).context("Kimi core arena extent overflow")?;
                if conflicts.iter().all(|other| end <= self.arena_offsets[other]
                    || start >= self.arena_offsets[other] + self.operands[other]) {
                    selected = Some((start, end)); break;
                }
            }
            let (start, end) = selected.context("Kimi core arena placement absent")?;
            self.arena_offsets.insert(name, start);
            self.arena_bytes = self.arena_bytes.max(end);
        }
        self.bytes()?;
        Ok(self)
    }

    /// Resolve optional larger split scratch to the fused scratch fallback used
    /// by the runtime. Every listed operand is live through that kernel boundary.
    pub fn core_phases(&self) -> Vec<Vec<&'static str>> {
        CORE_PHASES.iter().map(|phase| phase.iter().map(|&name| match name {
            "K6 KDA scratch" if !self.operands.contains_key(name) => "KDA scratch",
            "K6 MLA scratch" if !self.operands.contains_key(name) => "MLA producer scratch",
            _ => name,
        }).collect()).collect()
    }

    pub fn bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(name, _)| !self.arena_offsets.contains_key(*name))
            .try_fold(self.arena_bytes, |sum, (_, bytes)| sum.checked_add(*bytes).context("Kimi lane extent overflow"))
    }
}

impl NativeScratch {
    pub fn from_manifest_for_config(manifest: &serde_json::Value, cfg: &KimiK3Config,
        ranks: usize) -> Result<Self> {
        if cfg.dsa.is_none() { return Self::from_manifest_for_ranks(manifest, ranks); }
        sparse_mla_chunk(manifest)?;
        ensure!(matches!(ranks, 1 | 2), "Kimi native scratch physical rank count");
        let mut mla = Vec::new();
        for rows in [64, 4096] { for suffix in ["", "_h96"] { for cache in ["", "_fp8"] {
            let output = program_scratch(manifest, &format!("kimik31_mla_output{suffix}_m{rows}"), rows)?;
            let producer = program_scratch(manifest, &format!("kimik31_mla_producer{suffix}{cache}_m{rows}"), rows)?;
            mla.push([producer, program_scratch(manifest,
                &format!("kimik31_sparse_mla{suffix}{cache}_m{rows}"), rows)?, output]);
            if suffix == "_h96" { for begin in [0, 48] {
                mla.push([producer, program_scratch(manifest,
                    &format!("kimik31_sparse_mla_partial{cache}_h{begin}_m{rows}"), rows)?, output]);
            }}
        }}}
        let prefix = if ranks == 1 { "kda_fused_h96" } else { "kimik31_kda" };
        Self::combine([program_scratch(manifest, &format!("{prefix}_m64"), 64)?,
            program_scratch(manifest, &format!("{prefix}_m4096"), 4096)?], mla)
    }

    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self> {
        Self::from_manifest_for_ranks(manifest, 2)
    }

    pub fn from_manifest_for_ranks(manifest: &serde_json::Value, ranks: usize) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2), "Kimi native scratch physical rank count");
        let scratch = |name: &str| -> Result<usize> {
            let found = manifest["programs"].as_array().context("Kimi program inventory missing")?
                .iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
            ensure!(found.len() == 1, "Kimi requires one {name} scratch inventory");
            usize::try_from(found[0]["scratch_bytes_at_capacity"]["scratch"].as_u64()
                .with_context(|| format!("Kimi missing {name} scratch extent"))?).context("Kimi scratch overflow")
        };
        let mut mla = Vec::new();
        for suffix in ["", "_h96"] {
            for (mode, rows) in [("decode", 64), ("verify", 64), ("extend", 4096)] {
                mla.push([scratch(&format!("kimik3_mla_producer{suffix}_m{rows}"))?,
                    scratch(&format!("kimik3_dense_mla_{mode}{suffix}_m{rows}"))?,
                    scratch(&format!("kimik3_mla_output{suffix}_m{rows}"))?]);
            }
        }
        let prefix = if ranks == 1 { "kda_fused_h96" } else { "kimik3_kda" };
        Self::combine([scratch(&format!("{prefix}_m64"))?, scratch(&format!("{prefix}_m4096"))?], mla)
    }
}

pub const GRAPH_RESERVE: &str = "graph reserve";
/// Header-only plans calibrate the CUDA context and loaded modules as one
/// Estimated owner; the runtime replaces it with the measured post-module
/// baseline under `CONTEXT_MEASURED`. Neither is a single driver request.
pub const CONTEXT_ESTIMATE: &str = "CUDA context (estimated)";
pub const CONTEXT_MEASURED: &str = "CUDA context and loaded modules";
/// Measured SM120 startup capture for the Kimi K3 HQA min config on one
/// coordinator GPU (2026-11-09 census): 530,579,456 B over 3,961 executables.
/// The estimate scales that observation by executable count with no padding;
/// the synchronized capture at Ready is the exact owner.
pub const GRAPH_CENSUS_BYTES: u64 = 530_579_456;
pub const GRAPH_CENSUS_EXECUTABLES: u64 = 3_961;
/// The same capture on two coordinator GPUs (2026-11-09 2x census), as
/// `(bytes, executables)` per rank. A rank's kernels differ from the other
/// rank's, so each rank keeps its own observation; one shared ratio would
/// under-reserve the higher rank.
pub const GRAPH_CENSUS_TP2: [(u64, u64); 2] = [(761_266_176, 5_367), (763_363_328, 5_377)];

/// The measured `(bytes, executables)` startup capture for `rank` of a
/// `ranks`-GPU coordinator layout.
pub fn graph_census(ranks: usize, rank: usize) -> (u64, u64) {
    match (ranks, rank) {
        (2, 0) => GRAPH_CENSUS_TP2[0],
        (2, _) => GRAPH_CENSUS_TP2[1],
        _ => (GRAPH_CENSUS_BYTES, GRAPH_CENSUS_EXECUTABLES),
    }
}
pub const GRAPH_ROWS: [usize; 5] = [1, 4, 16, 64, 4096];

/// Speculative verify rows replay at their exact flattened batch width (no
/// padding: every row writes KDA replay records and MLA cache slots). Each of
/// `slots` requests can supply `verify` rows, capped by the native 64-row step.
/// Zero disables speculation; one slot retains the diagnostic verify widths.
pub fn verify_graph_rows(verify: usize, slots: usize, capacity: usize) -> std::ops::RangeInclusive<usize> {
    2..=verify.saturating_mul(slots).min(64).min(capacity)
}

/// Every captured row count: the plain buckets plus each exact verify width.
/// Plain attention stays on the plain buckets; the added widths capture
/// speculative attention and FFN sections. Residuals have no standalone keys.
pub fn graph_rows(verify: usize, capacity: usize) -> Vec<usize> {
    let mut rows = GRAPH_ROWS.into_iter().chain(verify_graph_rows(verify, 1, capacity))
        .filter(|&r| r <= capacity).collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    rows
}

pub fn graphs(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    resident: impl Fn(usize, usize) -> bool) -> Result<GraphSet> {
    graphs_for_ranks(cfg, lanes, capacity, slots, verify, 2, resident)
}

pub fn graphs_for_ranks(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    ranks: usize, resident: impl Fn(usize, usize) -> bool) -> Result<GraphSet> {
    graphs_with_context(cfg, lanes, capacity, slots, verify, ranks, resident, |_| false, None)
}

#[allow(clippy::too_many_arguments)]
pub fn graphs_with_context(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    ranks: usize, resident: impl Fn(usize, usize) -> bool, context: impl Fn(usize) -> bool,
    decode: Option<&super::context::DecodeBuckets>) -> Result<GraphSet> {
    ensure!(matches!(ranks, 1 | 2) && (1..=2).contains(&lanes) && (1..=UNIT_ROWS_MAX).contains(&capacity)
        && (1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots) && verify <= 64, "Kimi graph geometry");
    ensure!((0..cfg.layers).all(|layer| !context(layer)
        || (ranks == 2 && cfg.attention[layer] == KimiAttention::Mla && (0..2).all(|rank| resident(rank, layer)))),
        "Kimi context graph residency differs from two-rank MLA");
    ensure!(!(0..cfg.layers).any(&context) || decode.is_some(),
        "Kimi context graph inventory needs package-derived decode buckets");
    let mut counts = vec![0; ranks];
    for (rank, count) in counts.iter_mut().enumerate() {
        for rows in graph_rows(verify, capacity) {
            *count += cfg.layers as u64;
            for layer in 0..cfg.layers {
                if resident(rank, layer) && !context(layer) {
                    if GRAPH_ROWS.contains(&rows) {
                        *count += 1;
                    }
                    if verify_graph_rows(verify, 1, capacity).contains(&rows) { *count += 1; }
                }
            }
        }
        if let Some(decode) = decode {
            for rows in decode.graph_rows(verify, slots, capacity) {
                let variants = u64::from(decode.plain().contains(&rows))
                    + u64::from(verify_graph_rows(verify, slots, capacity).contains(&rows));
                *count += (0..cfg.layers).filter(|&layer| context(layer) && resident(rank, layer)).count() as u64
                    * variants;
            }
        }
        *count *= lanes as u64;
    }
    let inventory = counts.iter().enumerate().map(|(rank, &count)| {
        let (observed_bytes, observed_executables) = graph_census(ranks, rank);
        GraphRank::estimate(count, observed_bytes, observed_executables).map_err(|error| anyhow::anyhow!(error))
    }).collect::<Result<Vec<_>>>()?;
    Ok(GraphSet { ranks: inventory, lifetime: Lifetime::Startup, shapes: counts.iter().sum() })
}

pub fn graph_demand(set: &GraphSet, rank: usize) -> Demand {
    let mut demand = set.demand(rank);
    demand.group = GRAPH_RESERVE.into();
    demand.basis = Basis::Estimated;
    demand
}

/// The configured graph arena, one Demand per physical rank. Admission owns
/// the whole reserve before KV sizing; the bounded device cache grows into it
/// lazily and evicts, rather than re-planning memory, when it is full.
pub fn graph_reserve_demands(core: crate::serving_capacity::kimi_prewarm::KimiCoreFormat, ranks: usize) -> anyhow::Result<Vec<Demand>> {
    (0..ranks).map(|rank| {
        let (base, extra) = crate::placement::graph_reserve::kimi_default(core, ranks, rank);
        let reserve = crate::placement::graph_reserve::current(rank, false, base, extra)?;
        tracing::info!(rank, reserve = reserve.bytes, source = %reserve.source,
            "Kimi GPU_GRAPH_RESERVE");
        Ok(Demand::new(rank as u8, Category::Runtime, GRAPH_RESERVE, reserve.bytes, Basis::Estimated))
    }).collect()
}

pub fn kda_demands(cfg: &KimiK3Config, slots: usize, marks: usize, lanes: usize,
    rank: u8) -> Result<[Demand; 4]> {
    kda_demands_for_layers(cfg.kda_layers(), slots, marks, lanes, rank)
}

pub fn kda_demands_for_layers(layers: usize, slots: usize, marks: usize, lanes: usize,
    rank: u8) -> Result<[Demand; 4]> {
    kda_demands_for_heads(layers, slots, marks, lanes, rank, 48)
}

pub fn kda_demands_for_heads(layers: usize, slots: usize, marks: usize, lanes: usize,
    rank: u8, heads: usize) -> Result<[Demand; 4]> {
    ensure!((1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots) && (1..=2).contains(&lanes)
        && rank < 2 && matches!(heads, 48 | 96), "Kimi KDA state geometry");
    let state = heads * 128 * 128 * 4 + 3 * 3 * heads * 128 * 2;
    let replay = 64 * heads * (3 * 128 * 4 + 4 + 3 * 128 * 2);
    Ok([(Category::Kv, "Kimi KDA active state", layers * slots * state),
        (Category::Workspace, "Kimi KDA replay", layers * lanes * replay),
        (Category::Prefix, "Kimi exact KDA marks", layers * marks * state),
        (Category::Workspace, "Kimi KDA commit tables", lanes * slots * 3 * 4)]
        .map(|(category, group, bytes)| Demand::new(rank, category, group, bytes as u64, Basis::Exact)))
}

pub fn io_extents(vocab: usize, rank: usize, rows: usize, core_head: bool) -> [usize; 3] {
    io_extents_for_ranks(vocab, rank, rows, core_head, 2).expect("valid pair token IO rank")
}

pub fn io_extents_for_ranks(vocab: usize, rank: usize, rows: usize, core_head: bool,
    ranks: usize) -> Result<[usize; 3]> {
    io_extents_with_logits(vocab, rank, rows, rows.min(64), core_head, ranks)
}

pub fn io_extents_with_logits(vocab: usize, rank: usize, rows: usize, logit_rows: usize,
    core_head: bool, ranks: usize) -> Result<[usize; 3]> {
    ensure!(matches!(ranks, 1 | 2) && rank < ranks, "Kimi token IO physical rank count");
    ensure!((1..=rows).contains(&logit_rows), "Kimi token IO logit rows exceed lane");
    Ok([if rank == 0 { rows * 4 } else { 0 },
        if rank == ranks - 1 { logit_rows * vocab * 4 } else { 0 },
        if rank == ranks - 1 && !core_head { 4 << 20 } else { 0 }])
}

/// Return slots for the prefill units, from every admitted rank: `shape.waves`
/// slots of `shape.unit_rows` each.
pub fn spark_demand(shape: crate::serving_capacity::PrefillShape, lane: usize, ranks: usize,
    format: cuteafd_core::SparkReplyFormat) -> Result<Demand> {
    spark_demand_on(shape, lane, ranks, 1, format)
}

pub fn spark_demand_on(shape: crate::serving_capacity::PrefillShape, lane: usize, ranks: usize,
    owner: u8, format: cuteafd_core::SparkReplyFormat) -> Result<Demand> {
    Ok(Demand::new(owner, Category::Workspace, format!("Kimi Spark lane {lane} latent intake"),
        crate::serving_capacity::prefill_return_bytes_with_format(shape.waves, shape.unit_rows, ranks as u64, 3584, format)?, Basis::Exact))
}

/// Row planes hold exactly the admitted units; scratch and return slots serve
/// one unit each. Diagnostic callers normalize their shape before allocation.
pub fn prefill_shape_with(rows: usize, waves: usize, unit: Option<usize>)
    -> Result<crate::serving_capacity::PrefillShape> {
    use crate::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
    ensure!((1..=crate::serving_capacity::prefill_waves::MAX_WAVES as usize).contains(&waves)
        && (1..=LANE_ROWS_MAX).contains(&rows) && rows >= 64 * waves,
        "Kimi lane row/wave capacity: {rows} rows x {waves} waves");
    let unit = unit.unwrap_or(rows / waves) as u64;
    let plan = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
        unit_rows: Some(unit), waves: Some(waves as u64), span_rows: Some(rows as u64), reserve_rows: None,
    })?;
    Ok(crate::serving_capacity::PrefillShape { unit_rows: plan.unit_rows, waves: plan.waves })
}

pub fn exchange_demand(rows: usize, rank: u8, progress: usize) -> Result<Demand> {
    let recv = super::core::exchange_recv_bytes(rows)?;
    let bytes = recv + 256 + 256 + progress;
    Ok(Demand::new(rank, Category::Workspace, "Kimi peer/abort controls", bytes as u64, Basis::Exact))
}

pub const CODE_OWNER: &str = "Kimi selected-program loaded code";

pub fn code_demand(rank: usize, pending: u64, ranks: usize) -> Result<Demand> {
    ensure!(matches!(ranks, 1 | 2) && rank < ranks, "Kimi code physical rank geometry");
    Ok(Demand::new(rank as u8, Category::Runtime, CODE_OWNER, pending, Basis::Estimated))
}

/// Exact FirstCapacity owner, using the shared EXL3 capacity arenas.
pub fn expert_workspace(package: &std::path::Path, rows: usize) -> Result<u64> {
    expert_workspace_for_ranks(package, rows, 2)
}

pub fn expert_workspace_for_ranks(package: &std::path::Path, rows: usize, ranks: usize) -> Result<u64> {
    expert_workspace_for_ranks_with(package, rows, ranks, |path| {
        Ok(std::sync::Arc::new(serde_json::from_slice(&std::fs::read(path)
            .with_context(|| format!("read {}", path.display()))?)?))
    })
}

pub(crate) fn expert_workspace_for_ranks_with(package: &std::path::Path, rows: usize, ranks: usize,
    read: impl Fn(&std::path::Path) -> Result<std::sync::Arc<serde_json::Value>>) -> Result<u64> {
    ensure!(matches!(ranks, 1 | 2), "Kimi expert workspace physical rank count");
    let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
        .capacities(rows)?;
    let manifests = capacities.into_iter().map(|capacity| -> Result<std::sync::Arc<serde_json::Value>> {
        let path = package.join(format!("m{capacity}/v41_exl3.json"));
        let manifest = read(&path)?;
        ensure!(manifest["hidden"] == 3584 && manifest["intermediate"] == 3072 / ranks
            && manifest["experts"] == 896 && manifest["top_k"] == 16
            && manifest["capacity"] == capacity && manifest["output_dtype"] == "fp32",
            "Kimi expert workspace geometry/capacity mismatch");
        Ok(manifest)
    }).collect::<Result<Vec<_>>>()?;
    Ok(crate::serving_capacity::exl3_workspace_bytes_from_refs(manifests.iter().map(|m| m.as_ref()), true, Exl3Epilogue::Sum)? + (rows * 3584 * 4) as u64)
}

pub fn expert_package(manifest: &std::path::Path) -> Result<std::path::PathBuf> {
    expert_package_for_ranks(manifest, 2)
}

pub fn expert_package_for_ranks(manifest: &std::path::Path, ranks: usize) -> Result<std::path::PathBuf> {
    ensure!(matches!(ranks, 1 | 2), "Kimi expert package physical rank count");
    let parent = manifest.parent().context("Kimi manifest has no parent")?;
    let role = format!("exl3/exl3-kimik3-k23/rtx-tp{ranks}");
    [parent.join(&role), parent.join("../lib").join(&role)].into_iter()
        .find(|path| path.is_dir()).context("Kimi installed expert workspace manifests missing")
}

/// Lane row planes plus `shape.waves` return slots of `shape.unit_rows` each.
pub fn baseline_demands(cfg: &KimiK3Config, manifest: &serde_json::Value,
    rows: usize, shape: crate::serving_capacity::PrefillShape, context: usize, slots: usize, verify: usize,
    core: &super::core::CoreInventory, heads: bool, ranks: usize,
    format: cuteafd_core::SparkReplyFormat) -> Result<Vec<Demand>> {
    baseline_demands_for_ranks(cfg, manifest, rows, shape, context, slots, verify, core, heads, ranks, 2, format, if heads { 48 } else { 96 })
}

#[allow(clippy::too_many_arguments)]
pub fn baseline_demands_for_ranks(cfg: &KimiK3Config, manifest: &serde_json::Value,
    rows: usize, shape: crate::serving_capacity::PrefillShape, context: usize, slots: usize, verify: usize,
    core: &super::core::CoreInventory, heads: bool, spark_ranks: usize, physical_ranks: usize,
    format: cuteafd_core::SparkReplyFormat, attention_heads: usize) -> Result<Vec<Demand>> {
    ensure!(matches!(physical_ranks, 1 | 2), "Kimi baseline physical owner count");
    let shape = shape.validate()?;
    let unit = usize::try_from(shape.unit_rows)?;
    prefill_shape_with(rows, usize::try_from(shape.waves)?, Some(unit))?;
    let mut layout = NativeLaneLayout::for_ranks(cfg, rows, unit, context,
        NativeScratch::from_manifest_for_config(manifest, cfg, physical_ranks)?, physical_ranks)?;
    if cfg.dsa.is_some() {
        layout = layout.with_sparse_mla_rows(attention_heads, sparse_mla_chunk(manifest)?)?
            .with_dsa(cfg, DsaScratch::from_manifest(manifest)?)?
            .with_live_projection_rows(attention_heads, physical_ranks)?;
    }
    let owners = crate::placement::attention::alternating_owners(
        cfg.attention.iter().map(|kind| (*kind == KimiAttention::Mla, None)), 0)?;
    // Enumerate and validate every capturable key; memory is the configured
    // arena below, not this census row.
    graphs_for_ranks(cfg, 1, unit, slots, verify, physical_ranks,
        |rank, layer| physical_ranks == 1 || cfg.attention[layer] == KimiAttention::Kda
            || heads || rank == usize::from(owners[layer]))?;
    let mut demands = vec![spark_demand_on(shape, 0, spark_ranks, (physical_ranks - 1) as u8, format)?];
    demands.extend(graph_reserve_demands(crate::serving_capacity::kimi_prewarm::KimiCoreFormat::from_inventory(core), physical_ranks)?);
    for rank in 0..physical_ranks {
        demands.push(Demand::new(rank as u8, Category::Workspace, "Kimi lane 0 native operands",
            layout.bytes()? as u64, Basis::Exact));
        demands.push(Demand::new(rank as u8, Category::Workspace, "Kimi lane 0 token IO",
            io_extents_for_ranks(cfg.vocab_size, rank, rows, core.get_weight("language_model.lm_head").is_some(), physical_ranks)?
                .into_iter().sum::<usize>() as u64, Basis::Exact));
        if physical_ranks == 2 {
            demands.push(exchange_demand(unit, rank as u8, 0)?);
            if core.native_attention_split(physical_ranks) {
                demands.push(core.native_attention_receive_demand(unit, rank as u8, physical_ranks)?);
            }
        }
        // Active state and retained marks are charged by the caller's mark policy.
        demands.extend(kda_demands_for_heads(cfg.kda_layers(), slots, 1, 1, rank as u8,
            cfg.kda_heads / physical_ranks)?.into_iter()
            .filter(|d| d.category == Category::Workspace));
    }
    Ok(demands)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KimiK3Config {
        KimiK3Config::from_hf(&crate::plan::testing::kimi_k3_config()).unwrap()
    }

    #[test]
    fn k31_live_lane_inventory_uses_placed_heads_and_unit_rows() -> Result<()> {
        let mut cfg = cfg();
        cfg.dsa = Some(super::super::config::KimiDsaConfig { heads: 32, head_dim: 128, topk: 2048 });
        let scratch = NativeScratch { kda: 629_464_064,
            mla: [168_296_448, 6_467_616_768, 201_326_592] };
        for heads in [48, 96] {
            let old = NativeLaneLayout::new(&cfg, 8256, 2752, 1_048_576, scratch)?
                .with_sparse_mla_rows(heads, 4096)?.with_dsa(&cfg,
                    DsaScratch { producer: 34_865_152, selector: 742_556_672 })?;
            let live = old.clone().with_live_projection_rows(heads, 2)?;
            for name in ["MLA query", "MLA attention", "MLA LSE"] {
                assert_eq!(live.operands[name] * 96, old.operands[name] * heads);
            }
            assert_eq!(live.operands["MLA attention scratch"], 2752 * heads * 16 * 1028);
            assert_eq!(live.operands["DSA selector scratch"], 742_556_672);
            assert_eq!(live.operands["DSA producer scratch"],
                cuteafd_ffi::KimiIndexProducer::scratch_bytes_for(2752)?);
            assert!(live.bytes()? < old.bytes()?);
            println!("h{heads}: before={} after={} arena={}", old.bytes()?, live.bytes()?, live.arena_bytes);
            for (name, bytes) in &live.operands {
                println!("h{heads} {name}: {bytes} arena_offset={:?}", live.arena_offsets.get(name));
            }
        }
        Ok(())
    }

    #[test]
    fn k31_sparse_mla_owner_matches_live_rows_and_placed_heads() -> Result<()> {
        let mut cfg = cfg();
        cfg.dsa = Some(super::super::config::KimiDsaConfig { heads: 32, head_dim: 128, topk: 2048 });
        let maximum = cuteafd_ffi::KimiSparseMla::scratch_bytes_for(4096, 96, false)?;
        for unit in [64, 1024, 2752, 4096] {
            for heads in [48, 96] {
                let baseline = NativeLaneLayout::new(&cfg, unit, unit, 1_048_576,
                    NativeScratch { kda: 4096, mla: [4096, maximum, 4096] })?;
                let exact = baseline.clone().with_sparse_mla_rows(heads, 4096)?;
                let bytes = cuteafd_ffi::KimiSparseMla::scratch_bytes_for(unit, heads, false)?;
                assert_eq!(exact.operands["MLA attention scratch"], bytes);
                assert_eq!(baseline.bytes()? - exact.bytes()?, maximum - bytes);
                let offset = exact.arena_offsets["MLA attention scratch"];
                assert!(offset + bytes <= exact.arena_bytes);
                assert_eq!(exact, baseline.with_sparse_mla_rows(heads, 4096)?, "runtime and planner use the same owner layout");
            }
        }
        // Isolate this owner's effect in the real pool solver: fixed BF16
        // weights/other owners cancel, so reclaimed scratch must become KV.
        use crate::placement::planner as p;
        struct Recipe { bytes: u64 }
        impl p::Recipe for Recipe {
            fn prepare(&self, _: &p::Request) -> std::result::Result<p::Contract, p::Refusal> {
                let maximum = 32768;
                let owner = |id, bytes, category| p::Allocation::buffer(id, p::Location::coordinator(0), bytes,
                    p::Lifetime::RESIDENT, category, id);
                let facts = p::Facts { phase_order: p::PhaseOrder::TARGET_THEN_DRAFT,
                    allocations: vec![owner("Kimi lane 0 native operands", self.bytes, Category::Workspace),
                        owner("records", 64 * 12 * (656 + 132), Category::Kv)], resident_weights: vec![],
                    deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
                    driver_page_tables: vec![], allocation_granularity: vec![] };
                let mut contract = p::Contract::fixed(64, maximum, 1, facts);
                contract.set_byte_extent("records", p::ByteCurve::linear(64 * 12 * (656 + 132),
                    64 * 12 * (656 + 132), maximum))?;
                contract.variables = vec![p::Variable::Attention { layer: 0,
                    values: vec![crate::placement::LayerMode::Whole { gpu: 0, ffn: crate::placement::FfnMode::Owner }] },
                    p::Variable::Prefill(vec![p::PrefillShape { rows: 2752, waves: 1 }]), p::Variable::Lanes(vec![1])];
                Ok(contract)
            }
        }
        let old = NativeLaneLayout::new(&cfg, 2752, 2752, 1_048_576,
            NativeScratch { kda: 4096, mla: [4096, maximum, 4096] })?;
        let exact = old.clone().with_sparse_mla_rows(96, 4096)?;
        let request = p::Request { context: p::Context::resolve(1_048_576, None), pool: p::Pin::Auto,
            onboard: p::Pin::Auto, prefill: p::Pin::Auto, operation: p::Operation::Serving,
            resident_weight_limit: None, spark_reply_formats: Default::default() };
        let capacity = old.bytes()? as u64 + (4 << 30);
        let budget = p::Budget { id: "rtx0".into(), locations: vec![p::Location::coordinator(0)],
            physical_bytes: capacity, logical_bytes: capacity, headroom: 0,
            baseline: p::Baseline::Forecast { occupied: 0, source: "K3.1 exact scratch regression".into(), assumed: false } };
        let before = p::solve(request.clone(), vec![budget.clone()], &Recipe { bytes: old.bytes()? as u64 })?;
        let after = p::solve(request, vec![budget], &Recipe { bytes: exact.bytes()? as u64 })?;
        assert!(after.pool_tokens > before.pool_tokens);
        assert_eq!(after.facts.allocations.iter().find(|owner| owner.id == "Kimi lane 0 native operands").unwrap().bytes,
            exact.bytes()? as u64, "solver charges the exact runtime lane allocation");
        Ok(())
    }

    fn k31_manifest() -> serde_json::Value {
        let mut programs = Vec::new();
        let mut add = |name: String, rows, bytes| programs.push(serde_json::json!({
            "name": name, "capacity_rows": rows, "scratch_bytes_at_capacity": {"scratch": bytes}}));
        for rows in [64, 4096] {
            for name in ["kimik31_kda", "kda_fused_h96"] { add(format!("{name}_m{rows}"), rows, rows); }
            add(format!("kimik31_index_producer_m{rows}"), rows, rows * 8512);
            let mode = if rows == 64 { "decode" } else { "prefill" };
            for (stem, bytes) in [("index_topk", rows * 128), ("scored_index_topk", rows * 256)] {
                add(format!("kimik31_{stem}_{mode}_m{rows}"), rows, bytes);
            }
            for suffix in ["", "_h96"] {
                add(format!("kimik31_mla_output{suffix}_m{rows}"), rows, rows * 1024);
                for cache in ["", "_fp8"] {
                    add(format!("kimik31_mla_producer{suffix}{cache}_m{rows}"), rows, rows * 2048);
                    add(format!("kimik31_sparse_mla{suffix}{cache}_m{rows}"), rows, rows * 4096);
                    if suffix == "_h96" { for begin in [0, 48] {
                        add(format!("kimik31_sparse_mla_partial{cache}_h{begin}_m{rows}"), rows, rows * 3072);
                    }}
                }
            }
        }
        for program in &mut programs {
            let name = program["name"].as_str().unwrap().to_owned();
            if name.starts_with("kimik31_index_producer_") {
                program["geometry"] = serde_json::json!({"sparse_mla_row_chunk_size": 64});
            }
            if name.starts_with("kimik31_sparse_mla") {
                let heads = if name.contains("_h96_") || name.contains("_partial") { 96 } else { 48 };
                program["geometry"] = serde_json::json!({"row_chunk_size": 64});
                program["scratch_bytes_at_capacity"]["scratch"] = serde_json::json!(
                    cuteafd_ffi::KimiSparseMla::scratch_bytes_for(64, heads, name.contains("_partial")).unwrap());
            }
        }
        serde_json::json!({"programs": programs})
    }

    #[test]
    fn k31_chunked_owner_is_bounded_for_every_live_unit_and_head_width() -> Result<()> {
        let cfg = cfg();
        let maximum = cuteafd_ffi::KimiSparseMla::scratch_bytes_for(4096, 96, false)?;
        for unit in [1, 6, 64, 512, 513, 2752, 4096] { for heads in [48, 96] {
            let base = NativeLaneLayout::new(&cfg, unit.max(64), unit.max(64), 1_048_576,
                NativeScratch { kda: 4096, mla: [4096, maximum, 4096] })?;
            let chunked = base.clone().with_sparse_mla_rows(heads, 64)?;
            let expected = cuteafd_ffi::KimiSparseMla::scratch_bytes_for(64, heads, false)?;
            assert_eq!(chunked.operands["MLA attention scratch"], expected);
            assert_eq!(base.bytes()? - chunked.bytes()?, maximum - expected);
            for invalid in [0, 4097] { assert!(base.clone().with_sparse_mla_rows(heads, invalid).is_err()); }
        }}
        Ok(())
    }

    #[test]
    fn k31_sparse_chunk_descriptor_refuses_missing_mixed_and_wrong_scratch() -> Result<()> {
        let manifest = k31_manifest();
        assert_eq!(sparse_mla_chunk(&manifest)?, 64);
        for key in ["sparse_mla_row_chunk_size", "row_chunk_size"] {
            for value in [serde_json::Value::Null, serde_json::json!(0), serde_json::json!(512)] {
                let mut wrong = manifest.clone();
                let entry = wrong["programs"].as_array_mut().unwrap().iter_mut()
                    .find(|p| p["geometry"].get(key).is_some()).unwrap();
                entry["geometry"][key] = value;
                assert!(sparse_mla_chunk(&wrong).is_err(), "{key}");
            }
        }
        let mut wrong = manifest.clone();
        let entry = wrong["programs"].as_array_mut().unwrap().iter_mut()
            .find(|p| p["geometry"].get("row_chunk_size").is_some()).unwrap();
        entry["scratch_bytes_at_capacity"]["scratch"] = serde_json::json!(1);
        assert!(sparse_mla_chunk(&wrong).is_err());
        Ok(())
    }

    #[test]
    fn k31_scratch_uses_selected_manifests_not_k3_dense_fallback() -> Result<()> {
        let mut cfg = cfg();
        cfg.dsa = Some(super::super::config::KimiDsaConfig { heads: 32, head_dim: 128, topk: 2048 });
        let manifest = k31_manifest();
        for ranks in [1, 2] {
            let scratch = NativeScratch::from_manifest_for_config(&manifest, &cfg, ranks)?;
            assert_eq!(scratch, NativeScratch { kda: 4096, mla: [4096 * 2048, 64 * 96 * 16 * 1028, 4096 * 1024] });
        }
        assert_eq!(DsaScratch::from_manifest(&manifest)?, DsaScratch { producer: 4096 * 8512, selector: 4096 * 256 });
        assert!(NativeScratch::from_manifest_for_ranks(&manifest, 2).is_err());
        for index in 0..manifest["programs"].as_array().unwrap().len() {
            let mut missing = manifest.clone(); missing["programs"].as_array_mut().unwrap().remove(index);
            assert!(NativeScratch::from_manifest_for_config(&missing, &cfg, 1).is_err()
                || NativeScratch::from_manifest_for_config(&missing, &cfg, 2).is_err()
                || DsaScratch::from_manifest(&missing).is_err(), "missing program {index}");
        }
        for bad in [serde_json::json!({}), serde_json::json!({"extra":16}), serde_json::json!({"scratch":"4096"})] {
            let mut wrong = manifest.clone(); wrong["programs"][2]["scratch_bytes_at_capacity"] = bad;
            assert!(DsaScratch::from_manifest(&wrong).is_err());
        }
        let mut duplicate = manifest.clone();
        duplicate["programs"].as_array_mut().unwrap().push(manifest["programs"][2].clone());
        assert!(DsaScratch::from_manifest(&duplicate).is_err());
        Ok(())
    }

    #[test]
    fn k31_index_lane_owners_preserve_phase_offsets_and_bank_causal_frontiers() -> Result<()> {
        let mut cfg = cfg();
        cfg.dsa = Some(super::super::config::KimiDsaConfig { heads: 32, head_dim: 128, topk: 2048 });
        let manifest = k31_manifest();
        let scratch = NativeScratch::from_manifest_for_config(&manifest, &cfg, 2)?;
        let index = DsaScratch::from_manifest(&manifest)?;
        for unit in [64, 2752, 4096] {
            let base = NativeLaneLayout::new(&cfg, unit * 2, unit, 1_048_576, scratch)?;
            let lane = base.clone().with_dsa(&cfg, index)?;
            assert_eq!(lane.arena_offsets, base.arena_offsets);
            assert_eq!(lane.arena_bytes, base.arena_bytes);
            assert_eq!(lane.operands["DSA indices"], unit * 2048 * 4);
            for name in DSA_METADATA {
                let (first, stride) = lane.metadata_extent(name, 0)?;
                assert_eq!(first, 0);
                assert_eq!(stride % 16, 0);
                assert_eq!(lane.metadata_extent(name, 1)?, (stride, stride));
            }
            assert_eq!(lane.metadata_extent("DSA lengths", 0)?.1, unit * 4);
            for context in [1usize, 65, 8192, 997_376, 1_048_576] {
                let resolved = lane.with_served_context(context)?;
                assert_eq!(resolved.arena_offsets, lane.arena_offsets);
                assert_eq!(resolved.arena_bytes, lane.arena_bytes);
                for name in ["MLA page table", "DSA page table"] {
                    assert_eq!(resolved.operands[name],
                        (unit.min(64) * context.div_ceil(64) * 4).max(16).next_multiple_of(16) * lane.metadata_banks());
                }
                for name in DSA_SCRATCH { assert_eq!(resolved.operands[name], lane.operands[name]); }
            }
            let extra = DSA_METADATA.into_iter().chain(DSA_SCRATCH).map(|name| lane.operands[name]).sum::<usize>();
            assert_eq!(lane.bytes()? - base.bytes()?, extra);
            assert!(lane.with_dsa(&cfg, index).is_err());
        }
        Ok(())
    }

    #[test]
    fn expert_packages_match_actual_owner_width_and_shared_capacity_census() -> Result<()> {
        let root = tempfile::tempdir()?;
        let manifest = root.path().join("PROGRAMS.json");
        let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .capacities(128)?;
        for ranks in [1usize, 2] {
            let package = root.path().join(format!("exl3/exl3-kimik3-k23/rtx-tp{ranks}"));
            let mut inventory = Vec::new();
            for &capacity in &capacities {
                let directory = package.join(format!("m{capacity}"));
                std::fs::create_dir_all(&directory)?;
                let value = serde_json::json!({"hidden": 3584, "intermediate": 3072 / ranks,
                    "experts": 896, "top_k": 16, "capacity": capacity, "output_dtype": "fp32",
                    "input_format": "e4m3_k32", "trellis_lut": {"bytes": 32}, "buffers": {
                        "scratch": {"allocation": "scratch", "bytes": u64::from(capacity) * 64 * ranks as u64,
                            "dtype": "f32", "zero_on_create": false},
                        "state": {"allocation": "state", "bytes": 16,
                            "dtype": "i32", "zero_on_create": true}}});
                std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(&value)?)?;
                inventory.push(value);
            }
            assert_eq!(expert_package_for_ranks(&manifest, ranks)?, package);
            assert_eq!(expert_workspace_for_ranks(&package, 128, ranks)?,
                crate::serving_capacity::exl3_workspace_bytes(&inventory, true, Exl3Epilogue::Sum)? + 128 * 3584 * 4);
            assert!(expert_workspace_for_ranks(&package, 128, 3 - ranks).is_err());
            if ranks == 2 {
                assert_eq!(expert_package(&manifest)?, package);
                assert_eq!(expert_workspace(&package, 128)?, expert_workspace_for_ranks(&package, 128, ranks)?);
            }
        }
        assert!(expert_package_for_ranks(&manifest, 0).is_err());
        assert!(expert_workspace_for_ranks(root.path(), 128, 3).is_err());
        Ok(())
    }

    #[test]
    fn startup_graph_census_keeps_a_separate_observation_per_rank() -> Result<()> {
        assert_eq!(graph_census(1, 0), (GRAPH_CENSUS_BYTES, GRAPH_CENSUS_EXECUTABLES));
        assert_eq!(graph_census(2, 0), GRAPH_CENSUS_TP2[0]);
        assert_eq!(graph_census(2, 1), GRAPH_CENSUS_TP2[1]);
        // The first 2x launch scaled the 1x ratio by both ranks' executable
        // count and under-reserved rank 0 by 5.9%; each rank now keeps its own.
        let (bytes, executions) = graph_census(2, 0);
        let shared = GraphRank::estimate(executions, GRAPH_CENSUS_BYTES, GRAPH_CENSUS_EXECUTABLES)
            .map_err(|error| anyhow::anyhow!(error))?;
        let observed = GraphRank::estimate(executions, bytes, executions)
            .map_err(|error| anyhow::anyhow!(error))?;
        assert!(shared.bytes < bytes, "shared ratio under-reserves the lead rank");
        assert_eq!(observed.bytes, bytes);
        assert_eq!(observed.basis, Basis::Estimated);
        Ok(())
    }

    #[test]
    fn single_owner_baseline_charges_full_state_io_and_no_peer_controls() -> Result<()> {
        let cfg = cfg();
        let mut programs = Vec::new();
        let mut add = |name: String, bytes| programs.push(serde_json::json!({
            "name": name, "scratch_bytes_at_capacity": {"scratch": bytes}}));
        for (prefix, bytes) in [("kimik3_kda", 1024), ("kda_fused_h96", 2048)] {
            for rows in [64, 4096] { add(format!("{prefix}_m{rows}"), bytes); }
        }
        for suffix in ["", "_h96"] {
            for (mode, rows) in [("decode", 64), ("verify", 64), ("extend", 4096)] {
                for name in [format!("kimik3_mla_producer{suffix}_m{rows}"),
                    format!("kimik3_dense_mla_{mode}{suffix}_m{rows}"),
                    format!("kimik3_mla_output{suffix}_m{rows}")] { add(name, 512); }
            }
        }
        programs.sort_by_key(|program| program["name"].as_str().unwrap().to_owned());
        programs.dedup_by(|left, right| left["name"] == right["name"]);
        let manifest = serde_json::json!({"programs": programs});
        let shape = prefill_shape_with(8256, 3, Some(2752))?;
        let core = super::super::core::CoreInventory::default();
        let single = baseline_demands_for_ranks(&cfg, &manifest, 8256, shape, 262144, 16, 8,
            &core, false, 6, 1, cuteafd_core::SparkReplyFormat::Bf16, 96)?;
        let pair = baseline_demands(&cfg, &manifest, 8256, shape, 262144, 16, 8, &core, false, 6, cuteafd_core::SparkReplyFormat::Bf16)?;
        assert!(single.iter().all(|d| d.gpu == 0));
        assert!(!single.iter().any(|d| d.group == "Kimi peer/abort controls"));
        assert_eq!(pair.iter().filter(|d| d.group == "Kimi peer/abort controls").count(), 2);
        let bytes = |demands: &[Demand], group: &str| demands.iter().filter(|d| d.group == group)
            .map(|d| d.bytes).sum::<u64>();
        for group in ["Kimi lane 0 token IO", "Kimi KDA replay", "Kimi Spark lane 0 latent intake"] {
            assert_eq!(bytes(&single, group), bytes(&pair, group), "{group}");
        }
        let lane = NativeLaneLayout::for_ranks(&cfg, 8256, 2752, 262144,
            NativeScratch::from_manifest_for_ranks(&manifest, 1)?, 1)?;
        assert_eq!(bytes(&single, "Kimi lane 0 native operands"), lane.bytes()? as u64);
        assert_eq!(code_demand(0, 100, 1)?.bytes, 100);
        assert_eq!(code_demand(0, 100, 2)?.basis, Basis::Estimated);
        assert_eq!(code_demand(1, 100, 2)?.bytes, 100);
        assert!(code_demand(1, 0, 1).is_err());
        assert!(baseline_demands_for_ranks(&cfg, &manifest, 8256, shape, 262144, 16, 8,
            &core, false, 6, 0, cuteafd_core::SparkReplyFormat::Bf16, 96).is_err());
        Ok(())
    }

    #[test]
    fn two_unit_lane_doubles_row_planes_and_keeps_unit_scratch() -> Result<()> {
        let cfg = cfg();
        let scratch = NativeScratch { kda: 1024, mla: [512, 0, 1024] };
        let one = NativeLaneLayout::new(&cfg, 4096, 4096, 1_048_576, scratch)?;
        let two = NativeLaneLayout::new(&cfg, 8192, 4096, 1_048_576, scratch)?;
        assert_eq!(two.arena_bytes, one.arena_bytes, "hook-local scratch serves one unit");
        assert_eq!(two.arena_offsets, one.arena_offsets);
        for name in ["residual arena", "delta", "normalized", "ffn output", "expert combined",
            "route ids", "expert wire"] {
            assert_eq!(two.operands[name], 2 * one.operands[name], "{name} spans the lane");
        }
        for name in ["gate-up", "activated", "KDA normalized", "MLA query", "MLA attention"]
            .into_iter().chain(HOOK_SCRATCH) {
            assert_eq!(two.operands[name], one.operands[name], "{name} is one unit's");
        }
        for name in ATTENTION_METADATA {
            assert_eq!(two.metadata_extent(name, 0)?.1, one.metadata_extent(name, 0)?.1);
            assert_eq!(two.operands[name], 2 * one.operands[name], "{name} has two unit banks");
        }
        // Units never exceed the kernels' 4096 rows nor the admitted lane.
        for (rows, unit) in [(LANE_ROWS_MAX + 1, 4096), (8192, 4097), (2048, 4096), (4096, 0)] {
            assert!(NativeLaneLayout::new(&cfg, rows, unit, 1_048_576, scratch).is_err(), "{rows}/{unit}");
        }
        Ok(())
    }

    #[test]
    fn shared_wave_shapes_allocate_exact_row_planes_and_unit_scratch() -> Result<()> {
        use crate::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
        let cfg = cfg();
        let scratch = NativeScratch { kda: 1024, mla: [512, 0, 1024] };
        for (waves, unit, rows) in [(2, 4096, 8192), (3, 2752, 8256), (4, 2048, 8192)] {
            let plan = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                waves: Some(waves), ..Default::default()
            })?;
            assert_eq!((plan.unit_rows, plan.span_rows), (unit, rows));
            let shape = prefill_shape_with(rows as usize, waves as usize, Some(unit as usize))?;
            assert_eq!(shape, plan.units());
            let one = NativeLaneLayout::new(&cfg, unit as usize, unit as usize, 1_048_576, scratch)?;
            let lane = NativeLaneLayout::new(&cfg, rows as usize, unit as usize, 1_048_576, scratch)?;
            assert_eq!(lane.metadata_banks(), waves as usize, "banks follow the admitted depth");
            assert_eq!(lane.arena_bytes, one.arena_bytes);
            for name in ["residual arena", "delta", "expert combined", "route ids", "expert wire"] {
                assert_eq!(lane.operands[name], waves as usize * one.operands[name], "{name}");
            }
            for name in HOOK_SCRATCH { assert_eq!(lane.operands[name], one.operands[name], "{name}"); }
            assert_eq!(lane.metadata_extent("KDA slots", 0)?.1, unit as usize * 4);
            assert_eq!(lane.metadata_extent("MLA slots", 0)?.1, unit as usize * 8);
            for ranks in 1..=8 {
                assert_eq!(spark_demand(shape, 0, ranks, cuteafd_core::SparkReplyFormat::Bf16)?.bytes, ranks as u64 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(rows * 3584 * 2)?);
            }
            assert!(prefill_shape_with(rows as usize - 64, waves as usize, Some(unit as usize)).is_err());
        }
        Ok(())
    }

    #[test]
    fn hook_scratch_is_unit_sized_but_pending_wave_planes_keep_the_span() -> Result<()> {
        let cfg = cfg();
        let layout = NativeLaneLayout::new(&cfg, 8256, 2752, 1_048_576,
            NativeScratch { kda: 16, mla: [16; 3] })?;
        for name in HOOK_SCRATCH {
            let stride = if name == "router logits" { cfg.experts * 4 } else { cfg.latent * 2 };
            assert_eq!(layout.operands[name], layout.unit * stride);
            assert_ne!(layout.operands[name], layout.rows * stride, "negative control: span-sized scratch");
        }
        assert_eq!(layout.operands["expert combined"], layout.rows * cfg.latent * 2);
        for name in ["normalized", "shared output", "routed output", "delta"] {
            assert_eq!(layout.operands[name], layout.rows * cfg.hidden * 2);
        }
        Ok(())
    }

    #[test]
    fn metadata_banks_are_disjoint_and_reject_undersized_or_unknown_planes() -> Result<()> {
        let layout = NativeLaneLayout::new(&cfg(), 8256, 2752, 1_048_576,
            NativeScratch { kda: 16, mla: [16; 3] })?;
        assert_eq!(layout.metadata_banks(), 3);
        assert_ne!(layout.metadata_banks(), 4, "negative control: no maximum-depth padding");
        for rows in [1, 16, 64] {
            let decode = NativeLaneLayout::new(&cfg(), rows, rows, 1_048_576,
                NativeScratch { kda: 16, mla: [16; 3] })?;
            assert_eq!(decode.metadata_banks(), 1, "decode rows {rows}");
        }
        for name in ATTENTION_METADATA {
            let mut end = 0;
            for bank in 0..layout.metadata_banks() {
                let (offset, bytes) = layout.metadata_extent(name, bank)?;
                assert_eq!(offset, end);
                assert!(bytes >= 16);
                assert_eq!(offset % 16, 0, "{name} bank {bank} offset");
                assert_eq!(bytes % 16, 0, "{name} bank {bank} stride");
                end += bytes;
            }
            assert_eq!(end, layout.operands[name]);
            assert!(layout.metadata_extent(name, layout.metadata_banks()).is_err());
        }
        assert!(layout.metadata_extent("delta", 0).is_err());
        let mut bad = layout.clone();
        *bad.operands.get_mut("KDA slots").unwrap() -= 1;
        assert!(bad.metadata_extent("KDA slots", 0).is_err());
        Ok(())
    }

    #[test]
    fn every_metadata_bank_is_aligned_even_for_odd_unit_and_context_extents() -> Result<()> {
        for (rows, unit, context) in [(768, 256, 1_048_576), (8256, 2752, 1_048_576),
            (195, 65, 65), (195, 65, 129), (9, 3, 65), (51, 17, 129)] {
            let layout = NativeLaneLayout::new(&cfg(), rows, unit, context,
                NativeScratch { kda: 16, mla: [16; 3] })?;
            let requests = unit.min(64);
            for (name, raw) in [("KDA slots", unit * 4), ("KDA first", unit * 4),
                ("MLA slots", unit * 8), ("MLA page table", requests * context.div_ceil(64) * 4),
                ("MLA lengths", requests * 4), ("MLA query offsets", (requests + 1) * 4)] {
                let stride = raw.max(16).div_ceil(16) * 16;
                assert_eq!(layout.operands[name], stride * layout.metadata_banks());
                for bank in 0..layout.metadata_banks() {
                    assert_eq!(layout.metadata_extent(name, bank)?, (bank * stride, stride));
                    assert_eq!((bank * stride) % 16, 0, "{name} bank {bank}");
                }
            }
        }
        // Negative control: the former 65-i32 stride misaligned banks 1 and 2.
        let mut old = NativeLaneLayout::new(&cfg(), 768, 256, 1_048_576,
            NativeScratch { kda: 16, mla: [16; 3] })?;
        let old_stride = 65 * 4;
        old.operands.insert("MLA query offsets", old_stride * old.metadata_banks());
        assert_eq!(old_stride, 260);
        assert_ne!(old_stride % 16, 0);
        for bank in 0..old.metadata_banks() {
            assert!(old.metadata_extent("MLA query offsets", bank).is_err());
        }
        Ok(())
    }

    #[test]
    fn prefill_shapes_derive_units_up_to_4096() -> Result<()> {
        let unit = |rows, waves, explicit| prefill_shape_with(rows, waves, explicit).map(|s| s.unit_rows);
        assert_eq!(unit(8192, 2, None)?, 4096);
        assert_eq!(unit(8192, 2, Some(4096))?, 4096);
        assert_eq!(unit(4096, 2, None)?, 2048);
        assert_eq!(unit(4096, 1, None)?, 4096);
        assert!(unit(8192, 1, None).is_err(), "a unit is at most 4096 rows");
        assert!(unit(8192, 2, Some(8192)).is_err(), "a unit is at most 4096 rows");
        assert!(unit(4096, 2, Some(4096)).is_err(), "the lane holds both units");
        assert!(unit(8192, 2, Some(2048)).is_err(), "the lane holds exactly its live units");
        assert!(unit(4097, 2, None).is_err(), "the lane cannot hide a rounded tail");
        assert!(unit(8256, 2, Some(2048)).is_err(), "the reserve of 2048-row slots is 8192 rows");
        Ok(())
    }
}
