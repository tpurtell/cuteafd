//! Checkpoint-native target operands, packed once per physical coordinator.
//! Layer ownership is supplied by placement; this module does not choose it.
//! Native BF16 and selected MC K8/K6 projections use their own header-derived
//! layouts; compressed payloads remain on the shared dense-trellis path.
use super::Inkling2Config;
use crate::placement::LayerMode;
use crate::plan::families::inkling2::TEXT;
use crate::{plan::checkpoint::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use crate::formats::{dense_trellis::TrellisSide,
    native_attention::{Fp8Role, NativeLayout},
    trellis_core::{CoreContract, CoreSplit, CoreWeightLayout}};
use std::collections::BTreeMap;

/// Checkpoint rectangle for a native arena view; column ranges are in bytes.
pub struct NativeSource<'a> {
    pub name: &'a str,
    pub rows: std::ops::Range<usize>,
    pub columns: std::ops::Range<usize>,
    pub row_bytes: usize,
}

impl<'a> NativeSource<'a> {
    pub fn new(checkpoint: &Checkpoint, view: &'a str, rank: usize) -> Result<Self> {
        let (name, split) = match view.split_once("#rank") {
            Some((name, owner)) => {
                ensure!(rank < 2 && owner.parse::<usize>()? == rank, "Inkling native view rank differs: {view}");
                let axis = usize::from(name.ends_with(".attn.wo_ud.weight") || name.ends_with("_sconv.static_w"));
                (name, Some(axis))
            }
            None => (view, None),
        };
        let header = checkpoint.header(name).with_context(|| format!("Inkling native view lacks {name}"))?;
        let count = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
        let bytes = usize::try_from(header.byte_length)?;
        ensure!(count > 0 && bytes.is_multiple_of(count), "Inkling native view has invalid source rows: {name}");
        let row_bytes = bytes / count;
        let mut rows = 0..count;
        let mut columns = 0..row_bytes;
        if let Some(axis) = split {
            if name == HEAD_WEIGHT {
                rows = crate::shared::vocabulary::shard(count, 128, rank)?;
                return Ok(Self { name, rows, columns, row_bytes });
            }
            ensure!(header.shape.len() >= 2 && header.shape[axis].is_multiple_of(2),
                "Inkling native view cannot split source axis {axis}: {name}");
            let range = if axis == 0 { &mut rows } else { &mut columns };
            let half = range.len() / 2;
            *range = rank * half..(rank + 1) * half;
        }
        Ok(Self { name, rows, columns, row_bytes })
    }
}

/// A target core's immutable views. Expert onboarding, media and the bundled
/// drafter are independently retired scopes and must not be added here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinatorWeights {
    pub rank: usize,
    pub layers: Vec<usize>,
    pub arena: WeightArenaLayout,
    pub core: BTreeMap<String, CoreWeightLayout>,
    pub fp8: BTreeMap<String, NativeLayout>,
}

pub const HEAD: &str = "model.llm.unembed";
const HEAD_WEIGHT: &str = "model.llm.unembed.weight";

pub fn fp8_role(name: &str) -> Fp8Role {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    if stem == HEAD { return Fp8Role::padded("vocabulary head", true, 128); }
    let attn = stem.strip_prefix("model.llm.layers.").and_then(|s| s.split_once('.'))
        .is_some_and(|(layer, role)| layer.parse::<usize>().is_ok() && matches!(role,
            "attn.gate_attn" | "attn.wq_du" | "attn.wr_du" | "attn.wk_dv" | "attn.wv_dv" | "attn.wo_ud"));
    Fp8Role::exact("target projection", attn)
}

pub fn projection_prefix(stem: &str, rank: usize, ranks: usize) -> String {
    if ranks > 1 { format!("{stem}#rank{rank}") } else { stem.to_owned() }
}

impl CoordinatorWeights {
    #[cfg(feature = "test-support")]
    pub fn head_only(checkpoint: &Checkpoint, cfg: &Inkling2Config) -> Result<Self> {
        let mut builder = Builder { checkpoint, tensors: Vec::new(), core: BTreeMap::new(), fp8: BTreeMap::new() };
        builder.projection(format!("{HEAD}.weight"), &[cfg.vocab_size, cfg.hidden], 0, None)?;
        Ok(Self { rank: 0, layers: vec![], core: builder.core, fp8: builder.fp8,
            arena: WeightArenaLayout::new("inkling2/recurrent-head-bench", builder.tensors)? })
    }

    /// Layer modes come from placement. A whole layer lives on its owner; a
    /// head- or context-split layer keeps half its heads' projections on each
    /// rank and replicates the per-layer norms, convolutions, router and
    /// residual queries (the residual itself is replicated under a split). The
    /// entry GPU owns the embedding and its norm; the final GPU owns the
    /// unembedding, the final norm and the output residual query.
    pub fn plan(checkpoint: &Checkpoint, cfg: &Inkling2Config, ranks: usize, modes: &[LayerMode]) -> Result<Vec<Self>> {
        ensure!(matches!(ranks, 1 | 2), "Inkling 2 coordinator ranks must be 1 or 2");
        ensure!(modes.len() == cfg.layers, "Inkling 2 target layer modes must cover every layer exactly once");
        ensure!(Inkling2Config::from_hf(&checkpoint.config)? == *cfg, "Inkling 2 arena configuration differs from checkpoint");
        // The recurrent drafter owns MTP source validation and allocation.
        checkpoint.validate_fp8_roles_where("Inkling 2", |name| !name.starts_with("mtp."), fp8_role)?;
        (0..ranks).map(|rank| {
            let layers = modes.iter().enumerate().filter_map(|(layer, mode)| match mode {
                LayerMode::Whole { gpu, .. } => (usize::from(*gpu) == rank).then_some(layer),
                LayerMode::HeadSplit | LayerMode::ContextSplit => Some(layer),
            }).collect::<Vec<_>>();
            let mut builder = Builder { checkpoint, tensors: Vec::new(), core: BTreeMap::new(), fp8: BTreeMap::new() };
            let (h, v) = (cfg.hidden, cfg.vocab_size);
            if rank == 0 {
                builder.plain(format!("{TEXT}embed.weight"), DType::Bf16, &[v, h])?;
                builder.plain(format!("{TEXT}embed_norm.weight"), DType::Bf16, &[h])?;
            }
            let head = if modes.iter().all(|mode| *mode == LayerMode::HeadSplit) { 0 } else { ranks - 1 };
            if rank == head || ranks == 2 {
                builder.projection(format!("{TEXT}unembed.weight"), &[v, h], 0, (ranks == 2).then_some(rank))?;
            }
            if rank == head {
                builder.plain(format!("{TEXT}norm.weight"), DType::Bf16, &[h])?;
                builder.plain(format!("{TEXT}attn_res.output_query"), DType::F32, &[h])?;
            }
            for &layer in &layers {
                let split = !matches!(modes[layer], LayerMode::Whole { .. });
                ensure!(!split || ranks == 2, "Inkling 2 split attention needs two coordinator ranks");
                builder.layer(cfg, layer, if split { Some(rank) } else { None })?;
            }
            Ok(Self { rank, layers, arena: WeightArenaLayout::new(format!("inkling2/target/{rank}"), builder.tensors)?,
                core: builder.core, fp8: builder.fp8 })
        }).collect()
    }

    /// Cold golden retains globals and exactly the selected layers. This is
    /// a physical subset, not a change to the checkpoint's model geometry.
    pub fn golden(checkpoint: &Checkpoint, cfg: &Inkling2Config, layers: &[usize]) -> Result<Self> {
        ensure!(!layers.is_empty() && layers.windows(2).all(|pair| pair[0] < pair[1])
            && layers.iter().all(|&layer| layer < cfg.layers), "Inkling golden layer inventory invalid");
        let modes = vec![LayerMode::Whole { gpu: 0, ffn: crate::placement::FfnMode::Owner }; cfg.layers];
        let whole = Self::plan(checkpoint, cfg, 1, &modes)?.remove(0);
        let selected = whole.arena.views().iter().filter(|(name, _)| {
            if let Some(rest) = name.strip_prefix(&format!("{TEXT}layers.")) {
                return rest.split('.').next().and_then(|n| n.parse::<usize>().ok()).is_some_and(|layer| layers.contains(&layer));
            }
            if let Some(rest) = name.strip_prefix(&format!("{TEXT}attn_res.mlp_queries.")) {
                return rest.parse::<usize>().ok().is_some_and(|layer| layers.contains(&layer));
            }
            if let Some(rest) = name.strip_prefix(&format!("{TEXT}attn_res.attn_queries.")) {
                return rest.parse::<usize>().ok().is_some_and(|query| layers.contains(&(query+1)));
            }
            true
        }).map(|(name, view)| (name.clone(), view.bytes())).collect::<Vec<_>>();
        let keep_projection = |stem: &str| {
            stem.strip_prefix(&format!("{TEXT}layers.")).and_then(|rest| rest.split('.').next())
                .and_then(|layer| layer.parse::<usize>().ok()).is_none_or(|layer| layers.contains(&layer))
        };
        let core = whole.core.into_iter().filter(|(stem, _)| keep_projection(stem)).collect();
        let fp8 = whole.fp8.into_iter().filter(|(stem, _)| keep_projection(stem)).collect();
        Ok(Self { rank: 0, layers: layers.to_vec(), arena: WeightArenaLayout::new("inkling2/golden/target", selected)?, core, fp8 })
    }

    /// Reuse the mount buffers across native copies, trellis tile packing,
    /// and selected BF16-to-block-FP8 transforms.
    pub fn mount_load(&self, checkpoint: &Checkpoint) -> Result<crate::shared::dense_read::MountLoad> {
        use crate::shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES};
        let mut load = MountLoad::default();
        for (view_name, view) in self.arena.views() {
            if self.is_specialized_view(view_name) { continue; }
            let source = NativeSource::new(checkpoint, view_name, self.rank)?;
            let (rows, width) = (source.rows.len(), source.columns.len());
            ensure!(rows.checked_mul(width) == Some(view.bytes()), "Inkling 2 mount {view_name}: admitted bytes differ");
            load.merge(MountLoad::read(ReadBatch::new(rows, source.row_bytes, width, STEP_BYTES)?));
        }
        for layout in self.core.values() {
            for suffix in ["trellis", "suh", "svh"] {
                let header = checkpoint.header(&format!("{}.{}", layout.stem, suffix))
                    .context("Inkling compact mount lacks validated member")?;
                let rows = header.shape.first().copied().unwrap_or(1);
                let row_bytes = usize::try_from(header.byte_length)? / rows;
                let width = if suffix == "trellis" && layout.side == TrellisSide::Output { layout.shape[0] / 16 * 32 * layout.bits }
                    else { row_bytes };
                load.merge(MountLoad::read(ReadBatch::new(rows, row_bytes, width, STEP_BYTES)?));
            }
        }
        for layout in self.fp8.values() {
            let header = checkpoint.header(&format!("{}.weight", layout.stem)).context("Inkling FP8 source missing")?;
            let width = layout.shape[1] * 2;
            let rows = (cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES / (width * 128)).max(1) * 128;
            load.merge(MountLoad::read(ReadBatch::new(rows.min(header.shape[0]), header.shape[1] * 2, width, STEP_BYTES)?));
            load.pinned_tables = load.pinned_tables.max(cuteafd_ffi::programs::fp8_load_table_bytes(rows, 128)? as u64);
        }
        Ok(load)
    }

    pub fn is_specialized_view(&self, view_name: &str) -> bool {
        let Some((prefix, _)) = view_name.rsplit_once('.') else { return false; };
        let stem = prefix.split_once('#').map_or(prefix, |(stem, _)| stem);
        self.core.contains_key(stem) || self.fp8.contains_key(stem)
    }

    pub fn trellis_scratch(&self, rows: usize, head_rows: usize)
        -> Result<Option<crate::formats::trellis_core::ProjectionScratchLayout>> {
        use crate::formats::trellis_core::ProjectionScratchLayout;
        let mut scratch = None::<ProjectionScratchLayout>;
        for layout in self.core.values() {
            let active = if layout.stem == HEAD { head_rows } else { rows };
            let mut need = ProjectionScratchLayout::for_weight(layout, active)?;
            need.rows = rows.max(head_rows);
            if let Some(scratch) = &mut scratch { scratch.merge(&need)?; } else { scratch = Some(need); }
        }
        Ok(scratch)
    }

    pub fn trellis_programs(&self, manifest: &serde_json::Value, rows: usize, head_rows: usize)
        -> Result<std::collections::BTreeSet<String>> {
        use cuteafd_ffi::dense_trellis::{validate_package_geometry_bits_with_egress, TrellisDType};
        use crate::formats::exl3_storage::Exl3Codebook;
        let mut names = std::collections::BTreeSet::new();
        let mut shapes = std::collections::BTreeSet::new();
        for l in self.core.values() {
            let [n, k] = l.shape;
            let side = if l.side == TrellisSide::Input { 2 } else { 1 };
            let book = if l.codebook == Exl3Codebook::Mcg { "mcg" } else { "mul1" };
            let active = if l.stem == HEAD { head_rows.min(64) } else { rows };
            if shapes.insert((k, n, side, l.bits, book, active)) {
                names.extend(validate_package_geometry_bits_with_egress(manifest, k, n, side, l.bits, book,
                    active, TrellisDType::F32, Some(&format!("dense_trellis_k6_trellis_output_n{n}_f32")))
                    .with_context(|| format!("{}.weight: Inkling trellis program missing or incompatible", l.stem))?);
            }
        }
        Ok(names)
    }

    pub fn fp8_programs(&self, manifest: &serde_json::Value, rows: usize, head_rows: usize)
        -> Result<BTreeMap<String, usize>> {
        use cuteafd_ffi::{draft_projection::ProjectionOutput, native_attention::validate_package_geometry};
        let mut programs = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        for p in self.fp8.values() {
            let active = if p.stem == HEAD { head_rows.min(64) } else { rows };
            for live in [active, active.min(64)] {
                if seen.insert((p.shape, live)) {
                    let (name, bytes) = validate_package_geometry(manifest, p.shape[1], p.shape[0], live, ProjectionOutput::F32)
                        .with_context(|| format!("{}.weight: Inkling FP8 program missing or incompatible", p.stem))?;
                    programs.insert(name, bytes);
                }
            }
        }
        Ok(programs)
    }

    pub fn fp8_scratch(&self, manifest: &serde_json::Value, rows: usize, head_rows: usize) -> Result<usize> {
        Ok(self.fp8_programs(manifest, rows, head_rows)?.into_values().max().unwrap_or(0))
    }

    pub fn stored_head(&self) -> Option<usize> {
        self.core.get(HEAD).map(|l| l.shape[0]).or_else(|| self.fp8.get(HEAD).map(|l| l.shape[0]))
    }

    /// Native ordered-K16 relative projection consumes a padded transpose in
    /// the same target backing, in addition to the unchanged checkpoint view.
    pub fn execution_arena(&self) -> Result<WeightArenaLayout> {
        let mut tensors = self.arena.views().iter().map(|(name, view)| (name.clone(), view.bytes())).collect::<Vec<_>>();
        for &layer in &self.layers {
            tensors.push((format!("{TEXT}layers.{layer}.attn.rel_logits_proj.padded"), 1024*64*2));
        }
        WeightArenaLayout::new(self.arena.id(), tensors)
    }

    /// Charge one arena backing and its one driver-allocation tail. Tensor
    /// views are aliases, not additional resident-weight charges.
    pub fn weight_components(&self, maximum_pages: u64)
        -> std::result::Result<Vec<crate::placement::planner::Component>, crate::placement::planner::Refusal> {
        use crate::placement::planner::{weight_arena_components, Lifetime, Location, Refusal};
        let rank = u8::try_from(self.rank).map_err(|_| Refusal::Overflow("Inkling 2 coordinator rank"))?;
        let arena = self.execution_arena().map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        weight_arena_components(&arena, Location::coordinator(rank), Lifetime::RESIDENT,
            maximum_pages, cuteafd_core::memory_layout::MemoryKind::Weights)
    }
}

struct Builder<'a> {
    checkpoint: &'a Checkpoint,
    tensors: Vec<(String, usize)>,
    core: BTreeMap<String, CoreWeightLayout>,
    fp8: BTreeMap<String, NativeLayout>,
}

impl Builder<'_> {
    fn projection(&mut self, name: String, shape: &[usize], axis: usize, split: Option<usize>) -> Result<()> {
        let stem = name.strip_suffix(".weight").context("Inkling projection suffix")?;
        if self.checkpoint.converts_to_fp8(&name) {
            let header = self.checkpoint.header(&name).with_context(|| format!("Inkling FP8 source missing {name}"))?;
            ensure!(header.dtype == DType::Bf16 && header.shape == shape,
                "{name}: Inkling FP8 source must be BF16 {shape:?}");
            let mut stored = fp8_role(&name).stored_shape([shape[0], shape[1]])
                .with_context(|| format!("{name}: Inkling FP8 role or geometry"))?;
            let split_axis = match split { None => CoreSplit::Whole, Some(_) if axis == 0 => CoreSplit::Output,
                Some(_) => CoreSplit::Input };
            if split.is_some() {
                if stem == HEAD {
                    stored[0] = crate::shared::vocabulary::shard(stored[0], 128, split.unwrap())?.len();
                } else {
                    ensure!(stored[axis] % 256 == 0, "{name}: FP8 TP2 requires whole rank-local blocks");
                    stored[axis] /= 2;
                }
            }
            let layout = NativeLayout { stem: stem.to_owned(), shape: stored, fp8: true,
                rank: split.unwrap_or(0), ranks: if split.is_some() { 2 } else { 1 }, split: split_axis };
            let prefix = projection_prefix(stem, layout.rank, layout.ranks);
            self.tensors.extend([("values", stored[0]*stored[1]), ("scales", stored[0]/128*(stored[1]/128)*4)]
                .into_iter().map(|(suffix, bytes)| (format!("{prefix}.{suffix}"), bytes)));
            self.fp8.insert(stem.to_owned(), layout);
            return Ok(());
        }
        if !self.checkpoint.contains_tensor(&format!("{stem}.trellis")) {
            return self.shard(name, DType::Bf16, shape, axis, split);
        }
        let expected = BTreeMap::from([(name.clone(), shape.to_vec())]);
        let mut projections = crate::formats::trellis_core::read_projections(self.checkpoint, &expected,
            |member| member.strip_prefix(stem).is_some_and(|rest| rest.starts_with('.')),
            |candidate| (candidate == stem).then_some(CoreContract { family: "Inkling 2", widths: &[6, 8],
                split: if split.is_none() || stem == HEAD { CoreSplit::Whole } else if axis == 0 { CoreSplit::Output } else { CoreSplit::Input },
                padded_outputs: stem == HEAD, view_varying: true }))?;
        let projection = projections.remove(stem).context("Inkling core projection headers absent")?;
        let layout = if stem == HEAD && split.is_some() {
            CoreWeightLayout::output_shard(&projection, split.unwrap())?
        } else { CoreWeightLayout::new(&projection,
            if split.is_some() && axis == 1 { TrellisSide::Input } else { TrellisSide::Output }, split.unwrap_or(0),
            if split.is_some() { 2 } else { 1 })? };
        let prefix = projection_prefix(stem, layout.rank, layout.ranks);
        self.tensors.extend(layout.arena_tensors(&prefix)?);
        self.core.insert(stem.to_owned(), layout);
        Ok(())
    }

    fn plain(&mut self, name: String, dtype: DType, shape: &[usize]) -> Result<()> {
        let header = self.checkpoint.header(&name)
            .with_context(|| format!("Inkling 2 arena requires checkpoint header {name}"))?;
        ensure!(header.dtype == dtype && header.shape == shape,
            "Inkling 2 arena {name}: expected {dtype:?} {shape:?}, found {:?} {:?}", header.dtype, header.shape);
        let bytes_per_element = match dtype {
            DType::Bf16 => 2, DType::F32 => 4,
            _ => anyhow::bail!("unsupported Inkling 2 arena dtype {dtype:?}"),
        };
        let bytes = shape.iter().try_fold(bytes_per_element, |bytes: usize, dim|
            bytes.checked_mul(*dim).context("Inkling 2 arena tensor extent overflow"))?;
        ensure!(header.byte_length == bytes as u64, "Inkling 2 arena {name}: header byte extent differs");
        self.tensors.push((name, bytes));
        Ok(())
    }

    /// A header-checked tensor of which this rank keeps `share` (numerator,
    /// denominator) of the rows along `axis`: the view is the rank's half of
    /// the checkpoint rows, named by rank so the two halves never alias.
    fn shard(&mut self, name: String, dtype: DType, shape: &[usize], axis: usize, split: Option<usize>) -> Result<()> {
        let Some(rank) = split else { return self.plain(name, dtype, shape) };
        let header = self.checkpoint.header(&name)
            .with_context(|| format!("Inkling 2 arena requires checkpoint header {name}"))?;
        ensure!(header.dtype == dtype && header.shape == shape,
            "Inkling 2 arena {name}: expected {dtype:?} {shape:?}, found {:?} {:?}", header.dtype, header.shape);
        ensure!(shape[axis] % 2 == 0, "Inkling 2 arena {name}: axis {axis} of {shape:?} does not split over two ranks");
        let bytes_per_element = match dtype { DType::Bf16 => 2, DType::F32 => 4, _ => anyhow::bail!("unsupported Inkling 2 arena dtype {dtype:?}") };
        let bytes = shape.iter().try_fold(bytes_per_element, |bytes: usize, dim|
            bytes.checked_mul(*dim).context("Inkling 2 arena tensor extent overflow"))?;
        let bytes = if name == HEAD_WEIGHT {
            bytes / shape[0] * crate::shared::vocabulary::shard(shape[0], 128, rank)?.len()
        } else { bytes / 2 };
        self.tensors.push((format!("{name}#rank{rank}"), bytes));
        Ok(())
    }

    fn layer(&mut self, cfg: &Inkling2Config, layer: usize, split: Option<usize>) -> Result<()> {
        let root = format!("{TEXT}layers.{layer}");
        let (h, d, k) = (cfg.hidden, cfg.head_dim, cfg.sconv_kernel);
        let local = cfg.local[layer];
        let a = format!("{root}.attn");
        let q = cfg.heads.checked_mul(d).context("Inkling 2 query width overflow")?;
        let kv = cfg.kv_heads.checked_mul(d).context("Inkling 2 KV width overflow")?;
        let r = cfg.heads.checked_mul(cfg.d_rel).context("Inkling 2 relative width overflow")?;
        // Head-parallel operands: half the query/KV heads per split rank (output
        // rows of the projections, input columns of wo_ud).
        for (name, n, k, axis) in [("wq_du", q, h, 0), ("wr_du", r, h, 0), ("wk_dv", kv, h, 0), ("wv_dv", kv, h, 0),
            ("gate_attn", q, h, 0), ("attn_sink_proj", cfg.heads, h, 0), ("wo_ud", h, q, 1)] {
            self.projection(format!("{a}.{name}.weight"), &[n, k], axis, split)?;
        }
        self.shard(format!("{a}.q_l2_clip.scale"), DType::Bf16, &[cfg.heads, d], 0, split)?;
        self.shard(format!("{a}.rel_l2_clip.scale"), DType::Bf16, &[cfg.heads, cfg.d_rel], 0, split)?;
        // The relative bias projection is shared by every head.
        self.plain(format!("{a}.rel_logits_proj.proj"), DType::Bf16, &[cfg.d_rel, if local { cfg.window } else { cfg.rel_extent }])?;
        for name in ["k", "v"] {
            if local {
                self.shard(format!("{a}.{name}_sconv.weight"), DType::Bf16, &[kv, 1, k], 0, split)?;
            } else {
                self.shard(format!("{a}.{name}_sconv.static_w"), DType::Bf16, &[k, kv], 1, split)?;
                self.shard(format!("{a}.{name}_sconv.weight_proj"), DType::Bf16, &[cfg.kv_heads * k, h], 0, split)?;
            }
        }
        for name in ["attn_sconv", "mlp_sconv"] {
            self.plain(format!("{root}.{name}.weight"), DType::Bf16, &[h, 1, k])?;
        }
        let norms: &[&str] = if local { &["attn_norm"] } else { &["q_input_norm", "k_input_norm", "v_input_norm"] };
        for norm in norms.iter().chain(["mlp_norm"].iter()) {
            self.plain(format!("{root}.{norm}.weight"), DType::Bf16, &[h])?;
        }
        self.plain(format!("{root}.mlp.gate.weight"), DType::Bf16, &[cfg.experts, h])?;
        self.plain(format!("{root}.mlp.gate.bias"), DType::F32, &[cfg.experts])?;
        self.plain(format!("{TEXT}attn_res.mlp_queries.{layer}"), DType::F32, &[h])?;
        if layer > 0 {
            let shape: Vec<usize> = if local { vec![h] } else { vec![3, h] };
            self.plain(format!("{TEXT}attn_res.attn_queries.{}", layer - 1), DType::F32, &shape)?;
        }
        Ok(())
    }
}
