//! Per-rank coordinator weight layout of one Muse Spark layer and the
//! globals: the operand inventory and byte extents the device packs and the
//! planner charges (one definition, so plan == runtime). Across a head-split
//! pair, attention heads split (64 query / 8 KV heads per GPU) and the MoE
//! projections split over E (pre_expert by outputs, post_expert by inputs);
//! the router, norms and gates are whole on each GPU. K6 projections keep their payload + H128 vectors;
//! the residual gates are packed as host-computed F32 alpha/beta.
use super::{core::{CoreInventory, CoreSplit, NativeLayout, PREFIX}, MuseSparkConfig};
use crate::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// Final target storage on one physical GPU; conversion inputs and the
/// independently selected drafter/onboarding scopes are not retained here.
pub fn coordinator_arena(rank: usize, ranks: usize, global: &GlobalLayout, layers: &[&LayerLayout])
    -> Result<crate::serving_capacity::WeightArenaLayout> {
    ensure!(matches!(ranks, 1 | 2) && rank < ranks, "Muse arena rank {rank} of {ranks}");
    let mut tensors = vec![("global.final_norm".to_owned(), global.final_norm)];
    if rank == 0 { tensors.push(("embedding.table".into(), global.embedding)); }
    if rank + 1 == ranks {
        if let Some(head) = &global.head { tensors.extend(head.arena_tensors(&head.stem)?); }
        else if let Some(head) = &global.head_native {
            let extents = head.allocation_extents()?;
            tensors.push((format!("{}.values", head.stem), usize::try_from(extents[0])?));
            if head.fp8 { tensors.push((format!("{}.scales", head.stem), usize::try_from(extents[1])?)); }
        } else { tensors.push(("global.head".into(), global.head_bf16)); }
    }
    for (layer, layout) in layers.iter().enumerate() {
        tensors.extend(layout.operands.iter().map(|(name, bytes)| (format!("layers.{layer}.{name}"), *bytes)));
        for core in layout.core.values() { tensors.extend(core.arena_tensors(&core.stem)?); }
        for native in layout.native.values() {
            let [n, k] = native.shape;
            tensors.push((format!("{}.values", native.stem), n.checked_mul(k)
                .and_then(|bytes| bytes.checked_mul(if native.fp8 { 1 } else { 2 }))
                .context("Muse native arena values overflow")?));
            if native.fp8 {
                tensors.push((format!("{}.scales", native.stem), n.div_ceil(128).checked_mul(k.div_ceil(128))
                    .and_then(|bytes| bytes.checked_mul(4)).context("Muse native arena scales overflow")?));
            }
        }
    }
    crate::serving_capacity::WeightArenaLayout::new(format!("muse/coordinator/{rank}"), tensors)
}

/// One rank's operands of one layer: plain operand bytes by name, and K6 core
/// layouts by name. Names are the device binding names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerLayout {
    pub operands: BTreeMap<&'static str, usize>,
    pub core: BTreeMap<&'static str, CoreWeightLayout>,
    pub native: BTreeMap<&'static str, NativeLayout>,
}

impl LayerLayout {
    pub fn bytes(&self) -> Result<usize> {
        let native = self.native.values().map(|layout| layout.bytes()
            .and_then(|bytes| usize::try_from(bytes).context("Muse native weight extent")))
            .collect::<Result<Vec<_>>>()?;
        self.operands.values().copied().chain(self.core.values().map(|c| c.bytes)).chain(native)
            .try_fold(0usize, |n, b| n.checked_add(b).context("Muse layer weight extent overflow"))
    }

    /// The layout of every rank (one RTX: `ranks` 1; a head-split pair: 2).
    pub fn ranks(cfg: &MuseSparkConfig, layer: usize, ranks: usize, core: &CoreInventory) -> Result<Vec<Self>> {
        ensure!(layer < cfg.layers && matches!(ranks, 1 | 2) && cfg.heads % ranks == 0 && cfg.kv_heads % ranks == 0,
            "Muse layer layout: layer {layer} of {}, ranks {ranks}", cfg.layers);
        let (h, d, e) = (cfg.hidden, cfg.head_dim, cfg.expert_hidden);
        let (q, kv) = (cfg.heads * d / ranks, cfg.kv_heads * d / ranks);
        let root = format!("{PREFIX}{layer}");
        (0..ranks).map(|rank| {
            let mut operands = BTreeMap::new();
            let mut cores = BTreeMap::new();
            let mut native = BTreeMap::new();
            let mut matrix = |name: &'static str, stem: &str, n: usize, k: usize, split: Option<CoreSplit>| -> Result<()> {
                match core.projections.get(&format!("{root}.{stem}")) {
                    Some(projection) => {
                        // One source for TP2 ownership: the inventory's contract (`core::split`).
                        ensure!(Some(projection.split) == split, "{root}.{stem}: inventory split {:?} differs from the layout's {split:?}",
                            projection.split);
                        let side = match (split, ranks) {
                            (Some(CoreSplit::Input), 2) => TrellisSide::Input,
                            _ => TrellisSide::Output,
                        };
                        let (r, w) = if split.is_some() && ranks == 2 { (rank, 2) } else { (0, 1) };
                        cores.insert(name, CoreWeightLayout::new(projection, side, r, w)?);
                    }
                    None => {
                        if let Some(projection) = core.native.get(&format!("{root}.{stem}")) {
                            ensure!(Some(projection.split) == split, "Muse native projection split differs from layout");
                            native.insert(name, NativeLayout { stem: format!("{root}.{stem}"), shape: [n, k],
                                fp8: projection.fp8, rank, ranks, split: projection.split });
                        } else { operands.insert(name, n * k * 2); }
                    }
                }
                Ok(())
            };
            // Head-split attention (TP2 output units, o_proj input units).
            matrix("attn.q", "self_attn.q_proj", q, h, Some(CoreSplit::Output))?;
            matrix("attn.k", "self_attn.k_proj", kv, h, Some(CoreSplit::Output))?;
            matrix("attn.v", "self_attn.v_proj", kv, h, Some(CoreSplit::Output))?;
            matrix("attn.o", "self_attn.o_proj", h, q, Some(CoreSplit::Input))?;
            matrix("attn.gate", "self_attn.gate_proj", q, h, Some(CoreSplit::Output))?;
            // MoE frontend/backend split too (single residency): pre_expert by
            // output columns over E, post_expert by input columns over E (K-split,
            // rank-ordered F32 reduction). Router and vectors stay whole.
            matrix("moe.pre_expert", "mlp.pre_expert_proj", e / ranks, h, Some(CoreSplit::Output))?;
            matrix("moe.post_expert", "mlp.post_expert_proj", h, e / ranks, Some(CoreSplit::Input))?;
            operands.insert("moe.router", cfg.experts * h * 4);
            operands.insert("moe.bias", cfg.experts * 4);
            operands.insert("moe.pre_expert_norm", e * 2);
            // The routed experts' one post_expert_norm, applied by the coordinator combine.
            operands.insert("moe.post_expert_norm", e * 2);
            for norm in ["norm.input", "norm.pre_ffn", "norm.post_ffn"] { operands.insert(norm, h * 2); }
            // Residual gates as host-computed F32 (alpha, beta).
            for gate in ["gate.attention.alpha", "gate.attention.beta", "gate.ffn.alpha", "gate.ffn.beta"] {
                operands.insert(gate, h * 4);
            }
            Ok(Self { operands, core: cores, native })
        }).collect()
    }
}

/// Load-only native reader bounds. Projections share one drained runner per
/// rank; BF16 source rows also bound a checkpoint that stores FP8 directly.
pub fn native_mount_load<'a>(layouts: impl IntoIterator<Item = &'a NativeLayout>)
    -> Result<crate::shared::dense_read::MountLoad> {
    use crate::shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES};
    let mut mount = MountLoad::default();
    for layout in layouts {
        let [n, k] = layout.shape;
        let source_k = k.checked_mul(if layout.split == CoreSplit::Input { layout.ranks } else { 1 })
            .context("Muse native source width overflow")?;
        let batch = ReadBatch::new(n, source_k.checked_mul(2).context("Muse native row overflow")?,
            k.checked_mul(2).context("Muse native packed row overflow")?, STEP_BYTES)?;
        let mut load = MountLoad::read(batch);
        if layout.fp8 {
            load.pinned_tables = cuteafd_ffi::programs::fp8_load_table_bytes(batch.rows, 1)? as u64;
        }
        mount.merge(load);
    }
    Ok(mount)
}

/// Globals: the embedding (on the entry GPU), the final norm (each GPU) and the
/// head (on the head GPU; K6, stored N128-padded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalLayout {
    pub embedding: usize,
    pub final_norm: usize,
    pub head: Option<CoreWeightLayout>,
    pub head_native: Option<NativeLayout>,
    pub head_bf16: usize,
}

impl GlobalLayout {
    pub fn new(cfg: &MuseSparkConfig, core: &CoreInventory) -> Result<Self> {
        let head = core.get_weight(super::core::HEAD).map(|p| CoreWeightLayout::new(p, TrellisSide::Output, 0, 1))
            .transpose()?;
        let head_native = core.native.get(super::core::HEAD).map(|p| NativeLayout {
            stem: super::core::HEAD.into(), shape: p.shape, fp8: p.fp8,
            rank: 0, ranks: 1, split: CoreSplit::Whole });
        Ok(Self { embedding: cfg.vocab_size * cfg.hidden * 2, final_norm: cfg.hidden * 2,
            head_bf16: if head.is_some() || head_native.is_some() { 0 }
                else { cfg.vocab_size * cfg.hidden * 2 }, head, head_native })
    }

    pub fn head_bytes(&self) -> usize {
        self.head.as_ref().map(|h| h.bytes)
            .or_else(|| self.head_native.as_ref().map(|h| h.bytes().expect("validated head extent") as usize))
            .unwrap_or(self.head_bf16)
    }

    pub fn stored_head_outputs(&self, logical: usize) -> usize {
        self.head.as_ref().map(|h| h.shape[0])
            .or_else(|| self.head_native.as_ref().map(|h| h.shape[0])).unwrap_or(logical)
    }
}

/// A K6 projection program's package geometry: (K, N stored, side 1/2, codebook).
pub type ProjectionKey = (usize, usize, usize, &'static str, usize);

/// Output dtype a K6 geometry runs with: BF16 rows, or F32 (the head, and the
/// TP2 side-2 o_proj partial whose F32 core result is reduced across GPUs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProjectionOutput { Bf16, F32 }

pub fn projection_key(layout: &CoreWeightLayout) -> ProjectionKey {
    (layout.shape[1], layout.shape[0], match layout.side { TrellisSide::Output => 1, TrellisSide::Input => 2 },
        match layout.codebook { crate::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
            crate::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1" }, layout.bits)
}

/// One concrete native owner, including the caller's reduction output contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRequest {
    pub layout: NativeLayout,
    pub output: ProjectionOutput,
}

impl NativeRequest {
    pub fn new(layout: NativeLayout) -> Self {
        let output = if layout.ranks > 1 && layout.split == CoreSplit::Input {
            ProjectionOutput::F32
        } else { ProjectionOutput::Bf16 };
        Self { layout, output }
    }
}

/// Every projection one coordinator rank binds. Native owners retain their own
/// row ceiling, independent of whether this rank has K6 scratch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankProjections {
    pub programs: BTreeMap<(ProjectionKey, ProjectionOutput), usize>,
    pub scratch: Option<crate::formats::trellis_core::ProjectionScratchLayout>,
    pub native: Vec<NativeRequest>,
    pub native_rows: usize,
    pub native_head: Option<NativeLayout>,
    pub native_programs: crate::placement::planner::native_attention::NativePrograms,
}

impl RankProjections {
    /// Generic native exports are charged only when this rank selected them.
    pub fn include_native_programs(&self, programs: &mut crate::placement::inventory::ProgramSet) {
        if self.native_programs.programs.is_empty() { return; }
        programs.names.extend(self.native_programs.programs.iter().cloned());
        programs.names.sort();
        programs.names.dedup();
    }

    /// Check actual owners against admission without resolving programs again.
    pub fn validate_native_owners(&self, actual: &[NativeRequest], rows: usize)
        -> Result<(), crate::placement::planner::Refusal> {
        let invalid = |detail: &str| crate::placement::planner::Refusal::InvalidFact(
            format!("Muse native owner contract: {detail}"));
        if rows != self.native_rows { return Err(invalid("row ceiling differs from admission")); }
        let by_stem = |owners: &[NativeRequest]| -> Result<BTreeMap<String, NativeRequest>> {
            let mut map = BTreeMap::new();
            for owner in owners {
                ensure!(map.insert(owner.layout.stem.clone(), owner.clone()).is_none(),
                    "duplicate Muse native owner {}", owner.layout.stem);
            }
            Ok(map)
        };
        let expected = by_stem(&self.native).map_err(|_| invalid("duplicate admitted owner"))?;
        let actual = by_stem(actual).map_err(|_| invalid("duplicate actual owner"))?;
        if expected != actual { return Err(invalid("shape, storage, output or ownership differs from admission")); }
        Ok(())
    }
}

/// Per rank: the K6 geometries of every layer plus the head on the head rank
/// (`ranks - 1`), with the scratch planned at `rows` (head at most 64 rows:
/// the decode/verify width; wider prefill/drafter scoring uses head microbatches).
pub fn projections(cfg: &MuseSparkConfig, core: &CoreInventory, ranks: usize, rows: usize)
    -> Result<Vec<RankProjections>> {
    projections_with_manifest(cfg, core, ranks, rows, None)
}

pub fn projections_with_manifest(cfg: &MuseSparkConfig, core: &CoreInventory, ranks: usize, rows: usize,
    manifest: Option<&crate::placement::inventory::ProgramsManifest>) -> Result<Vec<RankProjections>> {
    use crate::placement::planner::native_attention::{NativeProgram, NativePrograms};
    ensure!(rows > 0 && rows <= i32::MAX as usize, "Muse projection row ceiling outside signed launch domain");
    let mut out = vec![RankProjections { programs: BTreeMap::new(), scratch: None,
        native: Vec::new(), native_rows: rows, native_head: None,
        native_programs: NativePrograms::resolve(&[], rows, manifest)? }; ranks];
    for layer in 0..cfg.layers {
        for (rank, layout) in LayerLayout::ranks(cfg, layer, ranks, core)?.into_iter().enumerate() {
            for weight in layout.native.values() {
                out[rank].native.push(NativeRequest::new(weight.clone()));
            }
            for weight in layout.core.values() {
                // The package's K-split H128 egress is F32; narrow only after
                // summing full-width partials and applying that one egress.
                let dtype = if weight.side == TrellisSide::Input { ProjectionOutput::F32 } else { ProjectionOutput::Bf16 };
                *out[rank].programs.entry((projection_key(weight), dtype)).or_insert(rows) = rows;
            }
        }
    }
    let global = GlobalLayout::new(cfg, core)?;
    if let Some(head) = global.head {
        out[ranks - 1].programs.insert((projection_key(&head), ProjectionOutput::F32), rows.min(64));
    }
    out[ranks - 1].native_head = global.head_native;
    for rank in &mut out {
        let requests = rank.native.iter().map(|request| NativeProgram {
            layout: &request.layout, output: match request.output {
                ProjectionOutput::Bf16 => cuteafd_ffi::draft_projection::ProjectionOutput::Bf16,
                ProjectionOutput::F32 => cuteafd_ffi::draft_projection::ProjectionOutput::F32,
            },
        }).collect::<Vec<_>>();
        rank.native_programs = NativePrograms::resolve(&requests, rows, manifest)?;
        if let Some(head) = &rank.native_head {
            let selected = NativePrograms::resolve(&[NativeProgram { layout: head,
                output: cuteafd_ffi::draft_projection::ProjectionOutput::F32 }], 64, manifest)?;
            rank.native_programs.programs.extend(selected.programs);
            rank.native_programs.programs.sort();
            rank.native_programs.programs.dedup();
            rank.native_programs.fp8_scratch_bytes = rank.native_programs.fp8_scratch_bytes.max(selected.fp8_scratch_bytes);
        }
        let mut merged: Option<crate::formats::trellis_core::ProjectionScratchLayout> = None;
        for (&((k, n, _, _, _), _), &program_rows) in &rank.programs {
            let mut layout = crate::formats::trellis_core::ProjectionScratchLayout::new(k, n, program_rows)?;
            // One scratch set serves the body's full row capacity.
            layout.rows = rows;
            match &mut merged { Some(m) => m.merge(&layout)?, None => merged = Some(layout) }
        }
        rank.scratch = merged;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "reads the installed Muse checkpoint headers"]
    fn installed_coordinator_arena_inventory() -> Result<()> {
        let hub = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/mnt/sparknest/hf-home"));
        let snapshot = hub.join("hub/models--wrldsuksgo2mars--Muse-Spark-Open-EXL3-K4-Core-K6/snapshots/89998fb177acd9a862ac8bd4fb79bfd50e940c97");
        let checkpoint = crate::plan::Checkpoint::inventory(&snapshot)?;
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        let global = GlobalLayout::new(&cfg, &core)?;
        for ranks in [1, 2] {
            let layers = (0..cfg.layers).map(|layer| LayerLayout::ranks(&cfg, layer, ranks, &core))
                .collect::<Result<Vec<_>>>()?;
            for rank in 0..ranks {
                let local = layers.iter().map(|per_rank| &per_rank[rank]).collect::<Vec<_>>();
                let arena = coordinator_arena(rank, ranks, &global, &local)?;
                let large = arena.views().values().filter(|view| view.bytes() > 1 << 20).collect::<Vec<_>>();
                let rounding = |bytes: usize| bytes.div_ceil(2 << 20) * (2 << 20) - bytes;
                let expected = match (ranks, rank) {
                    (1, 0) => (23_557_636_864, 1_768_704),
                    (2, 0) => (13_082_126_336, 2_004_992),
                    (2, 1) => (11_013_968_640, 273_664),
                    _ => unreachable!(),
                };
                assert_eq!((arena.bytes(), rounding(arena.bytes())), expected);
                use crate::placement::planner::{weight_arena_components, ComponentKind, Lifetime, Location};
                let components = weight_arena_components(&arena, Location::coordinator(rank as u8),
                    Lifetime::RESIDENT, 1, cuteafd_core::memory_layout::MemoryKind::Weights)?;
                assert_eq!(components.iter().filter(|component| matches!(component.kind,
                    ComponentKind::Allocation { .. })).count(), 1);
                assert_eq!(components.iter().filter(|component| matches!(component.kind,
                    ComponentKind::AllocationGranularity(_))).map(|component| component.bytes.at(1).unwrap())
                    .sum::<u64>(), expected.1 as u64);
                eprintln!("Muse installed inventory TP{ranks} rank{rank}: N={} payload={} large_N={} large_bytes={} old_large_slack={} arena={} padding={} final_tail={}",
                    arena.views().len(), arena.payload_bytes(), large.len(),
                    large.iter().map(|view| view.bytes()).sum::<usize>(),
                    large.iter().map(|view| rounding(view.bytes())).sum::<usize>(), arena.bytes(),
                    arena.padding_bytes(), rounding(arena.bytes()));
            }
        }
        Ok(())
    }

    #[test]
    fn coordinator_arenas_cover_final_tp1_and_tp2_formats() -> Result<()> {
        use super::super::core::{CoreSplit, NativeProjection};
        let dir = tempfile::tempdir()?;
        crate::plan::testing::write_muse_spark_snapshot(dir.path(), &[]);
        let checkpoint = crate::plan::Checkpoint::inventory(dir.path())?;
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let release = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        let mut native = release.clone();
        for (stem, projection) in &release.projections {
            if !stem.contains(".self_attn.") { continue; }
            native.projections.remove(stem);
            native.native.insert(stem.clone(), NativeProjection { shape: projection.shape, fp8: true,
                split: projection.split });
        }
        // Attention format varies independently of the selected target head.
        let mut native_bf16 = native.clone();
        for projection in native_bf16.native.values_mut() { projection.fp8 = false; }
        for (format, core) in [("K6", release), ("native-fp8", native), ("native-bf16", native_bf16),
            ("bf16", CoreInventory::default())] {
            let global = GlobalLayout::new(&cfg, &core)?;
            for ranks in [1, 2] {
                let layers = (0..cfg.layers).map(|layer| LayerLayout::ranks(&cfg, layer, ranks, &core))
                    .collect::<Result<Vec<_>>>()?;
                for rank in 0..ranks {
                    let local = layers.iter().map(|per_rank| &per_rank[rank]).collect::<Vec<_>>();
                    let arena = coordinator_arena(rank, ranks, &global, &local)?;
                    let payload = local.iter().map(|layer| layer.bytes()).collect::<Result<Vec<_>>>()?
                        .iter().sum::<usize>() + global.final_norm + if rank == 0 { global.embedding } else { 0 }
                        + if rank + 1 == ranks { global.head_bytes() } else { 0 };
                    assert_eq!(arena.payload_bytes(), payload);
                    let mut ordered = arena.views().values().copied().collect::<Vec<_>>();
                    ordered.sort_by_key(|view| view.offset());
                    assert!(ordered.windows(2).all(|pair| pair[0].offset() + pair[0].bytes() <= pair[1].offset()));
                    assert!(ordered.iter().all(|view| view.offset() % 256 == 0));
                    assert_eq!(ordered.last().map(|view| view.offset() + view.bytes()), Some(arena.bytes()));
                    assert_eq!(arena.view("embedding.table").is_some(), rank == 0);
                    assert_eq!(arena.padding_bytes(), arena.bytes() - payload);
                    let large = ordered.iter().filter(|view| view.bytes() > 1 << 20).collect::<Vec<_>>();
                    let rounding = |bytes: usize| bytes.div_ceil(2 << 20) * (2 << 20) - bytes;
                    eprintln!("Muse arena inventory {format} TP{ranks} rank{rank}: N={} payload={} large_N={} large_bytes={} old_large_slack={} arena={} padding={} final_tail={}",
                        ordered.len(), payload, large.len(), large.iter().map(|view| view.bytes()).sum::<usize>(),
                        large.iter().map(|view| rounding(view.bytes())).sum::<usize>(), arena.bytes(),
                        arena.padding_bytes(), rounding(arena.bytes()));
                    for native in local.iter().flat_map(|layer| layer.native.values()) {
                        assert!(matches!(native.split, CoreSplit::Input | CoreSplit::Output));
                        assert!(arena.view(&format!("{}.values", native.stem)).is_some());
                        assert_eq!(arena.view(&format!("{}.scales", native.stem)).is_some(), native.fp8);
                        assert!(arena.view(&format!("{}.weight", native.stem)).is_none(), "no conversion source residency");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn release_layer_layout_on_one_and_two_gpus() {
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::write_muse_spark_snapshot(dir.path(), &[]);
        let checkpoint = crate::plan::Checkpoint::inventory(dir.path()).unwrap();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config).unwrap();
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint).unwrap();
        let whole = LayerLayout::ranks(&cfg, 3, 1, &core).unwrap().remove(0);
        let pair = LayerLayout::ranks(&cfg, 3, 2, &core).unwrap();
        // K6 q/k/v/o and pre/post expert; gate_proj BF16 in this publication.
        assert_eq!(whole.core.len(), 6);
        assert_eq!(whole.operands["attn.gate"], 8192 * 8192 * 2);
        assert_eq!(pair[0].operands["attn.gate"], 4096 * 8192 * 2);
        // Head-split attention halves across the pair; whole operands repeat.
        for rank in &pair {
            assert_eq!(rank.core["attn.q"].shape, [4096, 8192]);
            assert_eq!(rank.core["attn.o"].shape, [8192, 4096]);
            assert_eq!(rank.core["moe.pre_expert"].shape, [2048, 8192]);
            assert_eq!(rank.core["moe.post_expert"].shape, [8192, 2048]);
            assert_eq!(rank.operands["moe.router"], 256 * 8192 * 4);
        }
        // No projection is held twice: the MoE halves sum to the whole (plus one vector set).
        for name in ["moe.pre_expert", "moe.post_expert"] {
            let halves = pair[0].core[name].bytes + pair[1].core[name].bytes;
            assert!(halves >= whole.core[name].bytes && halves - whole.core[name].bytes < 1 << 20, "{name}");
        }
        let global = GlobalLayout::new(&cfg, &core).unwrap();
        assert_eq!(global.head.as_ref().unwrap().shape, [202112, 8192]);
    }

    #[test]
    fn exact_native_resolution_names_only_selected_stage_and_charges_scratch() -> Result<()> {
        use crate::placement::{inventory::ProgramsManifest, planner::native_attention::NativeResolution};
        let cfg = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config())?;
        let mut core = CoreInventory::default();
        core.native.insert(format!("{PREFIX}0.self_attn.o_proj"),
            super::super::core::NativeProjection { shape: [8192, 8192], fp8: true, split: CoreSplit::Input });
        for rows in [64, 2752] {
            let stage = if rows <= 64 { 64 } else { 4096 };
            let name = format!("native_attention_n8192_k4096_m{stage}_f32");
            let manifest = ProgramsManifest::from_manifest(&serde_json::json!({"programs":[{
                "name":name,"capacity_rows":stage,
                "pointers":[{"name":"x"},{"name":"weight"},{"name":"scales"},{"name":"out"},{"name":"scratch"}],
                "scalars":[{"type":"int32"}],"scratch_bytes_at_capacity":{"scratch":2048},
                "geometry":{"input_features":4096,"output_features":8192,"output_dtype":"torch.float32",
                    "max_rows":stage,"decode_rows":64,"scale_block":128,"storage":"row_major_e4m3",
                    "reduction":"canonical_fp8_gemv64_w8a16"}
            }]}))?;
            let exact = projections_with_manifest(&cfg, &core, 2, rows, Some(&manifest))?;
            for rank in &exact {
                assert_eq!(rank.native_programs.programs, [name.clone()]);
                assert_eq!(rank.native_programs.fp8_scratch_bytes, 2048);
                assert_eq!(rank.native_programs.resolution, NativeResolution::Exact);
                let mut programs = crate::placement::inventory::ProgramSet::from_names(&["muse"], ["muse_z", "muse_a"]);
                rank.include_native_programs(&mut programs);
                assert_eq!(programs.names, ["muse_a", "muse_z", name.as_str()]);
            }
            let assumed = projections(&cfg, &core, 2, rows)?;
            assert_eq!(assumed[0].native_programs.fp8_scratch_bytes, 256);
            assert!(projections_with_manifest(&cfg, &core, 2, rows,
                Some(&ProgramsManifest::default())).is_err());
        }
        let compact = projections(&cfg, &CoreInventory::default(), 2, 2752)?;
        let mut programs = crate::placement::inventory::ProgramSet::from_names(&["muse"], ["muse_z", "muse_a", "muse_z"]);
        let old = programs.clone();
        compact[0].include_native_programs(&mut programs);
        assert_eq!(programs, old, "no native exports must preserve Compact program accounting verbatim");
        Ok(())
    }

    #[test]
    fn native_owner_contract_keeps_rows_outputs_and_every_layer() -> Result<()> {
        let cfg = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config())?;
        let mut core = CoreInventory::default();
        for layer in 0..cfg.layers {
            for (leaf, split) in [("q_proj", CoreSplit::Output), ("o_proj", CoreSplit::Input)] {
                core.native.insert(format!("{PREFIX}{layer}.self_attn.{leaf}"),
                    super::super::core::NativeProjection { shape: [8192, 8192], fp8: true, split });
            }
        }
        for ranks in [1, 2] {
            for rank in projections(&cfg, &core, ranks, 2752)? {
                assert!(rank.scratch.is_none(), "native capacity must not depend on K6 scratch");
                assert_eq!(rank.native_rows, 2752);
                assert_eq!(rank.native.len(), cfg.layers * 2, "same-shaped owners must not disappear");
                assert!(rank.native.iter().all(|request| request.output ==
                    if ranks == 2 && request.layout.split == CoreSplit::Input {
                        ProjectionOutput::F32
                    } else { ProjectionOutput::Bf16 }));
                let mut actual = rank.native.clone();
                actual.reverse();
                rank.validate_native_owners(&actual, 2752)?;
                assert!(rank.validate_native_owners(&actual, 64).is_err());
                for field in 0..7 {
                    let mut wrong = actual.clone();
                    match field {
                        0 => wrong[0].layout.shape[1] /= 2,
                        1 => wrong[0].layout.fp8 = false,
                        2 => wrong[0].output = if wrong[0].output == ProjectionOutput::Bf16 {
                            ProjectionOutput::F32
                        } else { ProjectionOutput::Bf16 },
                        3 => wrong[0].layout.rank += 1,
                        4 => wrong[0].layout.ranks += 1,
                        5 => { wrong.pop(); },
                        _ => wrong.push(wrong[0].clone()),
                    }
                    assert!(matches!(rank.validate_native_owners(&wrong, 2752),
                        Err(crate::placement::planner::Refusal::InvalidFact(_))), "field {field}");
                }
            }
        }
        assert!(projections(&cfg, &core, 2, 0).is_err());
        Ok(())
    }

    #[test]
    fn k6_programs_per_rank_and_their_shared_scratch() {
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::write_muse_spark_snapshot(dir.path(), &[]);
        let checkpoint = crate::plan::Checkpoint::inventory(dir.path()).unwrap();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config).unwrap();
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint).unwrap();
        let whole = projections(&cfg, &core, 1, 2752).unwrap();
        let keys = |rank: &RankProjections| rank.programs.keys().copied().collect::<Vec<_>>();
        use ProjectionOutput::*;
        assert_eq!(keys(&whole[0]), [((4096, 8192, 1, "mcg", 6), Bf16), ((8192, 1024, 1, "mcg", 6), Bf16),
            ((8192, 4096, 1, "mcg", 6), Bf16), ((8192, 8192, 1, "mcg", 6), Bf16), ((8192, 202112, 1, "mcg", 6), F32)]);
        let pair = projections(&cfg, &core, 2, 2752).unwrap();
        // q/k/v halve as output units; o_proj halves its input and returns an F32 partial.
        assert!(pair[0].programs.contains_key(&((4096, 8192, 2, "mcg", 6), F32)));
        assert!(pair[0].programs.contains_key(&((8192, 512, 1, "mcg", 6), Bf16)));
        // The MoE split: pre_expert halves its output, post_expert its input (an F32 K-split core).
        assert!(pair[0].programs.contains_key(&((8192, 2048, 1, "mcg", 6), Bf16)));
        assert!(pair[0].programs.contains_key(&((2048, 8192, 2, "mcg", 6), F32)));
        assert!(!pair[0].programs.keys().any(|(key, _)| key.1 == 202112), "head on the last rank only");
        assert!(pair[1].programs.contains_key(&((8192, 202112, 1, "mcg", 6), F32)));
        let scratch = pair[1].scratch.as_ref().unwrap();
        assert_eq!(scratch.rows, 2752);
        // The head's accumulator dominates at 64 rows; the body's rotated input at 2752.
        assert!(scratch.extents["accumulator"] >= 64 * 202112 * 4);
        assert_eq!(scratch.extents["rotated"], 2752 * 8192 * 2);
    }
}
