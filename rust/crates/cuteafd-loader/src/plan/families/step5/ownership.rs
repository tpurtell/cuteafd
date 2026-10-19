//! Pre-allocation binding of the shared layer plan to native Step-5 owners.
use super::Config;
use crate::placement::{FfnMode, LayerExecutionPlan, LayerMode, RoutedReplication};
use anyhow::{ensure, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRank {
    pub query_heads: usize,
    pub kv_heads: usize,
    /// Whole native FFN replicas, not unqualified tensor-parallel halves.
    pub ffn: bool,
    /// Dispatches and reduces the single Spark wave. A head-split peer keeps
    /// its native shared FFN but only receives the BF16 routed result.
    pub routed_wave: bool,
}

#[derive(Debug, Clone)]
pub struct ExecutionOwnership {
    ranks: usize,
    execution: LayerExecutionPlan,
    full: Vec<bool>,
    moe: Vec<bool>,
}

impl ExecutionOwnership {
    pub fn new(cfg: &Config, ranks: usize, execution: LayerExecutionPlan) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2) && execution.modes().len() == cfg.layers,
            "Step-5 execution rank or layer count");
        ensure!((cfg.layers, cfg.heads, cfg.groups, cfg.dim, cfg.hidden, cfg.window)
            == (92, 64, 4, 192, 4096, 512)
            && cfg.full.len() == cfg.layers + cfg.mtp && cfg.moe.len() == cfg.full.len()
            && cfg.full.iter().enumerate().all(|(l, &full)| full == (l < 92 && l % 4 == 3))
            && cfg.moe[..cfg.layers].iter().enumerate().all(|(l, &moe)| moe == (3..91).contains(&l)),
            "Step-5 execution native geometry");
        ensure!(execution.spec().entry_gpu < ranks as u8 && execution.spec().head_gpu < ranks as u8,
            "Step-5 execution entry/head rank");
        ensure!(execution.spec().row_bytes == 4096 * 4 && execution.spec().rows <= 4096,
            "Step-5 residual hop geometry");
        crate::placement::validate_layer_pipeline_lanes(execution.spec().lanes,
            usize::try_from(execution.spec().lanes)?)?;
        for (layer, mode) in execution.modes().iter().enumerate() {
            match mode {
                LayerMode::Whole { gpu, ffn: FfnMode::Owner } =>
                    ensure!((*gpu as usize) < ranks, "Step-5 layer {layer} owner rank"),
                LayerMode::HeadSplit => ensure!(ranks == 2 && !cfg.full[layer],
                    "Step-5 head split requires two ranks and SWA, layer {layer}"),
                LayerMode::ContextSplit => anyhow::bail!("Step-5 context split has no native cache binding"),
                LayerMode::Whole { ffn: FfnMode::Split, .. } =>
                    anyhow::bail!("Step-5 split FFN has no qualified native half-width executor"),
            }
        }
        let owners = Self { ranks, execution, full: cfg.full.clone(), moe: cfg.moe.clone() };
        for lane in 0..owners.execution.spec().lanes as usize {
            owners.execution.validate_routed_replications(&owners.routed_replications(lane)?)?;
        }
        Ok(owners)
    }

    /// Publication flags 0..4 belong exclusively to head partials. Routed
    /// copies use 4..8, with one full native receive allocation per layer/lane.
    /// The ordinal counts actual routed copies, not physical layer parity.
    pub fn routed_replications(&self, lane: usize) -> Result<Vec<Option<RoutedReplication>>> {
        ensure!(lane < self.execution.spec().lanes as usize, "Step-5 routed lane outside admission");
        let mut ordinal = 0;
        Ok(self.execution.modes().iter().enumerate().map(|(layer, mode)| {
            if *mode != LayerMode::HeadSplit || !self.moe[layer] { return None; }
            let copy = RoutedReplication { source: 0, slot: 4 + 2 * lane + ordinal % 2,
                storage: 1 + lane * self.execution.modes().len() + layer };
            ordinal += 1;
            Some(copy)
        }).collect())
    }

    pub fn ranks(&self) -> usize { self.ranks }
    pub fn execution(&self) -> &LayerExecutionPlan { &self.execution }

    pub fn validate_config(&self, cfg: &Config) -> Result<()> {
        ensure!((cfg.layers, cfg.heads, cfg.groups, cfg.dim, cfg.hidden, cfg.window)
            == (92, 64, 4, 192, 4096, 512) && cfg.full == self.full && cfg.moe == self.moe,
            "Step-5 ownership/checkpoint geometry differs");
        Ok(())
    }

    /// Persistent transports are per owning rank, not per simultaneous wave.
    pub fn routed_wave_ranks(&self) -> Result<Vec<usize>> {
        let mut ranks = Vec::new();
        for rank in 0..self.ranks {
            let mut owns = false;
            for layer in 0..self.execution.modes().len() {
                owns |= self.layer(layer, rank)?.routed_wave;
            }
            if owns { ranks.push(rank); }
        }
        Ok(ranks)
    }

    pub fn layer(&self, layer: usize, rank: usize) -> Result<LayerRank> {
        ensure!(rank < self.ranks, "Step-5 execution rank");
        Ok(match self.execution.mode(layer)? {
            LayerMode::Whole { gpu, .. } => {
                let owns = rank == usize::from(gpu);
                LayerRank { query_heads: if owns { 64 } else { 0 },
                    kv_heads: if owns { 4 } else { 0 }, ffn: owns,
                    routed_wave: owns && self.moe[layer] }
            }
            LayerMode::HeadSplit => LayerRank { query_heads: 32, kv_heads: 2, ffn: true,
                routed_wave: rank == 0 && self.moe[layer] },
            LayerMode::ContextSplit => unreachable!("validated modes"),
        })
    }
}

#[cfg(test)]
pub(crate) fn plan(modes: &[LayerMode], entry: u8, head: u8) -> LayerExecutionPlan {
    use crate::placement::*;
    let spec = HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: 4096 * 4,
        rows: 64, lanes: 1, entry_gpu: entry, head_gpu: head,
        terminal: if matches!(modes.last(), Some(LayerMode::Whole { ffn: FfnMode::Owner, .. })) {
            TerminalHome::AttentionOwner
        } else { TerminalHome::Replicated } };
    let placement = Placement { pool_tokens: 0, pool_limits: PoolLimits(vec![]),
        attention_placement: AttentionPlacement::Layers, attention_by_kind: vec![],
        peer_row_bytes: 0, onboard_layers: 0,
        layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Spark }).collect(),
        movables: vec![], expert_ranges: vec![], residual: residual_homes(modes, &spec).unwrap(),
        hops: plan_hops(modes, &spec).unwrap(), tp2: None, items: vec![] };
    LayerExecutionPlan::new(&placement, spec).unwrap()
}
