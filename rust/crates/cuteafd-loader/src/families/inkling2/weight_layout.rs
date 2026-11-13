//! Checkpoint-native target operands, packed once per physical coordinator.
//! Layer ownership is supplied by placement; this module does not choose it.
//! The original BF16 core is the stage-1 view; MC compact cores (K8/K6
//! trellis attention projections and head) join through the shared
//! `dense_trellis` inventory in stage 2.
use super::Inkling2Config;
use crate::placement::LayerMode;
use crate::plan::families::inkling2::TEXT;
use crate::{plan::checkpoint::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

/// A target core's immutable views. Expert onboarding, media and the bundled
/// drafter are independently retired scopes and must not be added here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinatorWeights {
    pub rank: usize,
    pub layers: Vec<usize>,
    pub arena: WeightArenaLayout,
}

impl CoordinatorWeights {
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
        (0..ranks).map(|rank| {
            let layers = modes.iter().enumerate().filter_map(|(layer, mode)| match mode {
                LayerMode::Whole { gpu, .. } => (usize::from(*gpu) == rank).then_some(layer),
                LayerMode::HeadSplit | LayerMode::ContextSplit => Some(layer),
            }).collect::<Vec<_>>();
            let mut builder = Builder { checkpoint, tensors: Vec::new() };
            let (h, v) = (cfg.hidden, cfg.vocab_size);
            if rank == 0 {
                builder.plain(format!("{TEXT}embed.weight"), DType::Bf16, &[v, h])?;
                builder.plain(format!("{TEXT}embed_norm.weight"), DType::Bf16, &[h])?;
            }
            if rank + 1 == ranks {
                builder.plain(format!("{TEXT}unembed.weight"), DType::Bf16, &[v, h])?;
                builder.plain(format!("{TEXT}norm.weight"), DType::Bf16, &[h])?;
                builder.plain(format!("{TEXT}attn_res.output_query"), DType::F32, &[h])?;
            }
            for &layer in &layers {
                let split = !matches!(modes[layer], LayerMode::Whole { .. });
                ensure!(!split || ranks == 2, "Inkling 2 split attention needs two coordinator ranks");
                builder.layer(cfg, layer, if split { Some(rank) } else { None })?;
            }
            Ok(Self { rank, layers, arena: WeightArenaLayout::new(format!("inkling2/target/{rank}"), builder.tensors)? })
        }).collect()
    }

    /// Raw checkpoint operands need only a bounded reused source buffer: no
    /// dequantization, row repacking, scale conversion or device staging.
    pub fn mount_load(&self, checkpoint: &Checkpoint) -> Result<crate::shared::dense_read::MountLoad> {
        use crate::shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES};
        let mut load = MountLoad::default();
        for (view_name, view) in self.arena.views() {
            let (name, share) = match view_name.split_once('#') { Some((name, _)) => (name, 2), None => (view_name.as_str(), 1) };
            let header = checkpoint.header(name).with_context(|| format!("Inkling 2 mount lacks {name}"))?;
            ensure!(header.byte_length == view.bytes() as u64 * share, "Inkling 2 mount {name}: admitted bytes differ");
            let rows = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
            let rows = rows.div_ceil(share as usize).max(1);
            ensure!(view.bytes() % rows == 0, "Inkling 2 mount {name}: invalid row extent");
            let row_bytes = view.bytes() / rows;
            load.merge(MountLoad::read(ReadBatch::new(rows, row_bytes, row_bytes, STEP_BYTES)?));
        }
        Ok(load)
    }

    /// Charge one arena backing and its one driver-allocation tail. Tensor
    /// views are aliases, not additional resident-weight charges.
    pub fn weight_components(&self, maximum_pages: u64)
        -> std::result::Result<Vec<crate::placement::planner::Component>, crate::placement::planner::Refusal> {
        use crate::placement::planner::{weight_arena_components, Lifetime, Location, Refusal};
        let rank = u8::try_from(self.rank).map_err(|_| Refusal::Overflow("Inkling 2 coordinator rank"))?;
        weight_arena_components(&self.arena, Location::coordinator(rank), Lifetime::RESIDENT,
            maximum_pages, cuteafd_core::memory_layout::MemoryKind::Weights)
    }
}

struct Builder<'a> {
    checkpoint: &'a Checkpoint,
    tensors: Vec<(String, usize)>,
}

impl Builder<'_> {
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
            bytes.checked_mul(*dim).context("Inkling 2 arena tensor extent overflow"))? / 2;
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
            self.shard(format!("{a}.{name}.weight"), DType::Bf16, &[n, k], axis, split)?;
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
