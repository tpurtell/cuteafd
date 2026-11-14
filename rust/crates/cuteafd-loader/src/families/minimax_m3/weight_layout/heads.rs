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
        Self::plan_selected(checkpoint, cfg, embedding, None)
    }

    fn plan_selected(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, embedding: EmbeddingHome,
        attention_owners: Option<&[usize]>) -> Result<[Self; 2]> {
        if let Some(owners) = attention_owners {
            ensure!(owners.len() == cfg.layers && owners.iter().all(|&rank| rank < 2),
                "MiniMax split FFN attention owners must cover every layer");
        }
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
            // Whole attention stays on its owner; every rank retains the FFN.
            let retain = |name: &str| attention_owners.is_none_or(|owners| {
                let Some(rest) = name.strip_prefix(&format!("{TEXT}layers.")) else { return true; };
                let Some((layer, _)) = rest.split_once('.') else { return true; };
                let layer: usize = layer.parse().expect("builder emits numeric layer IDs");
                // The sole GPU0 router supplies exact routes to local TP2 peers.
                if rank == 1 && router_operand(name) { return false; }
                ffn_operand(name) || owners[layer] == rank
            });
            builder.tensors.retain(|(name, _)| retain(name));
            builder.projections.retain(|projection| retain(&projection.stem));
            let sharding = |name: &str| if attention_owners.is_none() || ffn_operand(name) {
                shard(name)
            } else { Shard::Replicated };
            let mut slices = BTreeMap::new();
            let mut tensors = Vec::new();
            for (name, _) in builder.tensors {
                let header = checkpoint.header(&name).context("MiniMax heads source header missing")?;
                let rows = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
                let row_bytes = usize::try_from(header.byte_length)? / rows;
                let mut slice = TensorSlice { rows: 0..rows, columns: 0..row_bytes, row_bytes };
                match sharding(&name) {
                    Shard::Rows => { let half = rows / 2; slice.rows = rank * half..(rank + 1) * half; }
                    Shard::Columns => { let half = row_bytes / 2; slice.columns = rank * half..(rank + 1) * half; }
                    Shard::Replicated => {}
                }
                ensure!(!slice.rows.is_empty() && !slice.columns.is_empty(), "MiniMax empty heads slice {name}");
                tensors.push((name.clone(), slice.bytes()));
                slices.insert(name, slice);
            }
            for projection in &mut builder.projections {
                match sharding(&format!("{}.weight", projection.stem)) {
                    Shard::Rows => projection.output /= 2,
                    Shard::Columns => projection.input /= 2,
                    Shard::Replicated => {}
                }
                ensure!(projection.input % 128 == 0 && projection.output % 32 == 0,
                    "MiniMax heads native geometry is not aligned: {}", projection.stem);
            }
            let layers = (0..cfg.layers).filter(|&layer|
                attention_owners.is_none_or(|owners| owners[layer] == rank)).collect();
            Ok(Self { target: CoordinatorWeights { rank, layers,
                arena: WeightArenaLayout::new(format!("minimax/target/{rank}"), tensors)?,
                projections: builder.projections }, slices })
        };
        Ok([plan(0)?, plan(1)?])
    }

    /// Identical row windows to DenseLoad: full source rows are cached for
    /// partial-column operands, without re-reading a row within one GPU load.
    pub fn mount_load(&self, checkpoint: &Checkpoint) -> Result<crate::shared::dense_read::MountLoad> {
        slice_mount_load(checkpoint, &self.target, &self.slices)
    }
}

/// Whole attention on its assigned rank, with a native neuron half of every
/// dense/shared FFN on each rank. `target.layers` describes attention/KV only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitFfnWeights {
    pub target: CoordinatorWeights,
    pub slices: BTreeMap<String, TensorSlice>,
}
impl SplitFfnWeights {
    pub fn plan(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, attention_owners: &[usize],
        embedding: EmbeddingHome) -> Result<[Self; 2]> {
        Ok(HeadsWeights::plan_selected(checkpoint, cfg, embedding, Some(attention_owners))?
            .map(|weights| Self { target: weights.target, slices: weights.slices }))
    }

    pub fn mount_load(&self, checkpoint: &Checkpoint) -> Result<crate::shared::dense_read::MountLoad> {
        slice_mount_load(checkpoint, &self.target, &self.slices)
    }
}

pub fn slice_mount_load(checkpoint: &Checkpoint, target: &CoordinatorWeights,
    slices: &BTreeMap<String, TensorSlice>) -> Result<crate::shared::dense_read::MountLoad> {
    use crate::shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES};
    let mut load = MountLoad::default();
    for (name, slice) in slices {
        let header = checkpoint.header(name).context("MiniMax sliced mount source missing")?;
        let rows = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
        ensure!(rows > 0 && header.byte_length as usize == rows * slice.row_bytes
            && slice.rows.end <= rows && slice.columns.end <= slice.row_bytes
            && target.arena.view(name).is_some_and(|view| view.bytes() == slice.bytes()),
            "MiniMax sliced source/view extent differs: {name}");
        load.merge(MountLoad::read(ReadBatch::new(slice.rows.len(), slice.row_bytes,
            slice.columns.len(), STEP_BYTES)?));
    }
    Ok(load)
}

fn router_operand(name: &str) -> bool {
    name.ends_with(".block_sparse_moe.gate.weight") || name.ends_with(".block_sparse_moe.e_score_correction_bias")
}

fn ffn_operand(name: &str) -> bool {
    name.contains(".post_attention_layernorm.") || name.contains(".block_sparse_moe.")
        || name.contains(".mlp.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_split(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config) -> Result<()> {
        use crate::placement::planner::ComponentKind;
        let owners = (0..cfg.layers).map(|layer| usize::from(layer >= cfg.layers / 2)).collect::<Vec<_>>();
        let single_before = CoordinatorWeights::plan(checkpoint, cfg, 1, &vec![0; cfg.layers])?;
        let whole = CoordinatorWeights::plan(checkpoint, cfg, 2, &owners)?;
        let split = SplitFfnWeights::plan(checkpoint, cfg, &owners, EmbeddingHome::Device(0))?;
        let heads = HeadsWeights::plan(checkpoint, cfg, EmbeddingHome::Device(0))?;
        for rank in 0..2 {
            assert_eq!(split[rank].target.layers, whole[rank].layers);
            for layer in 0..cfg.layers {
                for suffix in ["gate.weight", "e_score_correction_bias"] {
                    let name = format!("{TEXT}layers.{layer}.block_sparse_moe.{suffix}");
                    assert_eq!(split[rank].slices.contains_key(&name), rank == 0 && cfg.moe[layer]);
                }
            }
            assert_eq!(split[rank].target.arena.bytes(), split[rank].slices.values().map(TensorSlice::bytes).sum::<usize>());
            assert_eq!(split[rank].target.arena.padding_bytes(), 0);
            let components = split[rank].target.weight_components(1024)?;
            assert_eq!(components.iter().filter(|c| matches!(c.kind, ComponentKind::AllocationGranularity(_))).count(), 1);
            for (name, slice) in &split[rank].slices {
                let header = checkpoint.header(name).unwrap();
                let view = split[rank].target.arena.view(name).unwrap();
                assert_eq!(view.bytes(), slice.bytes());
                if ffn_operand(name) { assert_eq!(slice, &heads[rank].slices[name]); }
                else { assert_eq!(slice.bytes(), header.byte_length as usize); }
            }
            for layer in 0..cfg.layers {
                for norm in ["post_attention_layernorm.weight", "input_layernorm.weight"] {
                    let name = format!("{TEXT}layers.{layer}.{norm}");
                    assert_eq!(split[rank].slices.contains_key(&name), norm.starts_with("post_") || owners[layer] == rank);
                }
                for projection in &single_before[0].projections {
                    if !projection.stem.starts_with(&format!("{TEXT}layers.{layer}.")) { continue; }
                    let local = split[rank].target.projections.iter().find(|p| p.stem == projection.stem);
                    if ffn_operand(&projection.stem) {
                        let mut expected = projection.clone();
                        match shard(&format!("{}.weight", expected.stem)) {
                            Shard::Rows => expected.output /= 2,
                            Shard::Columns => expected.input /= 2,
                            Shard::Replicated => unreachable!(),
                        }
                        assert_eq!(local, Some(&expected));
                    } else { assert_eq!(local, (owners[layer] == rank).then_some(projection)); }
                }
            }
            split[rank].mount_load(checkpoint)?;
            eprintln!("MiniMax split FFN rank {rank}: payload={} arena={} attention_layers={}",
                split[rank].target.arena.payload_bytes(), split[rank].target.arena.bytes(), split[rank].target.layers.len());
        }
        let unique = single_before[0].arena.payload_bytes();
        let duplicate = split[0].slices.iter().filter(|(name, slice)| split[1].slices.get(*name) == Some(*slice))
            .map(|(_, slice)| slice.bytes()).sum::<usize>();
        assert_eq!(split.iter().map(|w| w.target.arena.payload_bytes()).sum::<usize>(), unique + duplicate);
        assert_eq!(single_before, CoordinatorWeights::plan(checkpoint, cfg, 1, &vec![0; cfg.layers])?);
        Ok(())
    }

    #[test]
    fn split_ffn_native_fixture_keeps_attention_inventory_separate() -> Result<()> {
        let (directory, _) = crate::plan::families::minimax::tests::concurrency_matrix_snapshot();
        let checkpoint = Checkpoint::open(directory.path())?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        assert_split(&checkpoint, &cfg)
    }

    #[test]
    #[ignore = "requires installed MiniMax M3.1 native checkpoint headers"]
    fn installed_split_ffn_preserves_whole_attention_and_exact_native_halves() -> Result<()> {
        let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from).unwrap_or_else(||
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface"));
        let checkpoint = Checkpoint::open(&home.join("hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"))?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        assert_split(&checkpoint, &cfg)
    }

    #[test]
    #[ignore = "requires installed MiniMax M3.1 native checkpoint headers"]
    fn installed_heads_arenas_match_native_header_payloads_and_one_tail() -> Result<()> {
        use crate::placement::planner::ComponentKind;
        let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from).unwrap_or_else(||
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface"));
        let checkpoint = Checkpoint::open(&home.join("hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"))?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let owners = (0..cfg.layers).map(|layer| usize::from(layer >= cfg.layers / 2)).collect::<Vec<_>>();
        for embedding in [EmbeddingHome::Device(0), EmbeddingHome::HostPinned] {
            let heads = HeadsWeights::plan(&checkpoint, &cfg, embedding)?;
            let whole = CoordinatorWeights::plan_with_embedding(&checkpoint, &cfg, 2, &owners, embedding)?;
            let unique = whole.iter().flat_map(|w| w.arena.views().keys())
                .map(|name| (name.clone(), checkpoint.header(name).unwrap().byte_length as usize))
                .collect::<BTreeMap<_, _>>();
            let replicated = unique.iter().filter(|(name, bytes)| heads.iter().all(|h|
                h.slices.get(*name).is_some_and(|s| s.bytes() == **bytes)))
                .map(|(_, bytes)| bytes).sum::<usize>();
            let actual = heads.iter().map(|h| h.target.arena.payload_bytes()).sum::<usize>();
            assert_eq!(actual, unique.values().sum::<usize>() + replicated);
            for h in heads {
                assert_eq!(h.target.layers.len(), 60);
                assert_eq!(h.target.arena.padding_bytes(), 0);
                assert_eq!(h.target.arena.bytes(), h.slices.values().map(TensorSlice::bytes).sum::<usize>());
                let components = h.target.weight_components(1024)?;
                assert_eq!(components.iter().filter(|c| matches!(c.kind,
                    ComponentKind::AllocationGranularity(_))).count(), 1);
                h.mount_load(&checkpoint)?;
                eprintln!("MiniMax heads rank {} embedding {embedding:?}: payload={} arena={} single tail",
                    h.target.rank, h.target.arena.payload_bytes(), h.target.arena.bytes());
            }
        }
        Ok(())
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
