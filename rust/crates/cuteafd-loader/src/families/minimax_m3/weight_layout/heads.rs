//! Native TP2 head/neuron slices. Every range addresses unchanged checkpoint bytes.
use super::*;
use std::{collections::BTreeMap, ops::Range};

/// Source tensor rows and byte columns; destination rows are packed unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSlice {
    pub rows: Range<usize>,
    pub columns: Range<usize>,
    pub row_bytes: usize,
}
impl TensorSlice {
    pub fn bytes(&self) -> usize { self.rows.len() * self.columns.len() }
}

/// Both ranks execute every attention and FFN layer with explicit local N/K.
/// The ordinary Whole descriptor and its callers remain unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadsWeights {
    pub target: CoordinatorWeights,
    pub slices: BTreeMap<String, TensorSlice>,
}
impl HeadsWeights {
    pub fn plan(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, embedding: EmbeddingHome) -> Result<[Self; 2]> {
        ensure!(matches!(embedding, EmbeddingHome::Device(0) | EmbeddingHome::HostPinned),
            "MiniMax heads embedding belongs on entry rank or pinned host");
        ensure!(MiniMaxM3Config::from_hf(&checkpoint.config)? == *cfg,
            "MiniMax heads configuration differs from checkpoint");
        ensure!(cfg.heads % 2 == 0 && cfg.kv_heads % 2 == 0 && cfg.msa.index_heads % 2 == 0
            && cfg.shared_intermediate % 256 == 0 && cfg.dense_intermediate % 256 == 0,
            "MiniMax heads requires native aligned TP2 geometry");
        if embedding == EmbeddingHome::HostPinned {
            checkpoint.require_untied_embedding(&format!("{TEXT}embed_tokens.weight"))?;
        }
        let plan = |rank| -> Result<Self> {
            let mut builder = Builder { checkpoint, tensors: Vec::new(), projections: Vec::new() };
            builder.plain(format!("{TEXT}norm.weight"), DType::Bf16, &[cfg.hidden])?;
            if rank == 0 && embedding == EmbeddingHome::Device(0) {
                builder.plain(format!("{TEXT}embed_tokens.weight"), DType::Bf16, &[cfg.vocab_size, cfg.hidden])?;
            }
            if rank == 1 {
                builder.plain("language_model.lm_head.weight".into(), DType::Bf16, &[cfg.vocab_size, cfg.hidden])?;
            }
            for layer in 0..cfg.layers { builder.layer(cfg, layer)?; }
            let mut slices = BTreeMap::new();
            let mut tensors = Vec::new();
            for (name, _) in builder.tensors {
                let header = checkpoint.header(&name).context("MiniMax heads source header missing")?;
                let rows = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
                let row_bytes = usize::try_from(header.byte_length)? / rows;
                let mut slice = TensorSlice { rows: 0..rows, columns: 0..row_bytes, row_bytes };
                match shard(&name) {
                    Shard::Rows => { let half = rows / 2; slice.rows = rank * half..(rank + 1) * half; }
                    Shard::Columns => { let half = row_bytes / 2; slice.columns = rank * half..(rank + 1) * half; }
                    Shard::Replicated => {}
                }
                ensure!(!slice.rows.is_empty() && !slice.columns.is_empty(), "MiniMax empty heads slice {name}");
                tensors.push((name.clone(), slice.bytes()));
                slices.insert(name, slice);
            }
            for projection in &mut builder.projections {
                match shard(&format!("{}.weight", projection.stem)) {
                    Shard::Rows => projection.output /= 2,
                    Shard::Columns => projection.input /= 2,
                    Shard::Replicated => {}
                }
                ensure!(projection.input % 128 == 0 && projection.output % 32 == 0,
                    "MiniMax heads native geometry is not aligned: {}", projection.stem);
            }
            Ok(Self { target: CoordinatorWeights { rank, layers: (0..cfg.layers).collect(),
                arena: WeightArenaLayout::new(format!("minimax/target/{rank}"), tensors)?,
                projections: builder.projections }, slices })
        };
        Ok([plan(0)?, plan(1)?])
    }

    /// Identical row windows to DenseLoad: full source rows are cached for
    /// partial-column operands, without re-reading a row within one GPU load.
    pub fn mount_load(&self, checkpoint: &Checkpoint) -> Result<crate::shared::dense_read::MountLoad> {
        use crate::shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES};
        let mut load = MountLoad::default();
        for (name, slice) in &self.slices {
            let header = checkpoint.header(name).context("MiniMax heads mount source missing")?;
            let rows = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
            ensure!(rows > 0 && header.byte_length as usize == rows * slice.row_bytes
                && slice.rows.end <= rows && slice.columns.end <= slice.row_bytes
                && self.target.arena.view(name).is_some_and(|view| view.bytes() == slice.bytes()),
                "MiniMax heads source/view extent differs: {name}");
            load.merge(MountLoad::read(ReadBatch::new(slice.rows.len(), slice.row_bytes,
                slice.columns.len(), STEP_BYTES)?));
        }
        Ok(load)
    }
}

#[derive(Clone, Copy)]
enum Shard { Rows, Columns, Replicated }
fn shard(name: &str) -> Shard {
    if [".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.",
        ".self_attn.index_q_proj.", ".gate_proj.", ".up_proj."].iter().any(|stem| name.contains(stem)) {
        Shard::Rows
    } else if [".self_attn.o_proj.", ".down_proj."].iter().any(|stem| name.contains(stem)) {
        Shard::Columns
    } else { Shard::Replicated }
}
