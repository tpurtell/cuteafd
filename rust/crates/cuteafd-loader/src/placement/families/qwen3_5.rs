//! Dense hybrid Qwen admission. Every byte is future to the supplied baseline;
//! the executor supplies native packed sizes and exported workspace costs.
use crate::families::{qwen::HybridAttention, qwen3_5::Qwen35Config};
use crate::placement::*;
use anyhow::{ensure, Result};
use cuteafd_core::memory_layout::{Basis, Category};

pub const PAGE_ROWS: u64 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
/// Only the initial single-device dense executor; not a claim of TP2 support.
pub const SINGLE: ExecutorModes = ExecutorModes { family: "qwen3_5", modes: &[WHOLE],
    hops: false, attention: AttentionPlacement::Heads };

#[derive(Debug, Clone, Copy)]
pub struct LayerBytes {
    pub weights: u64,
    /// Active recurrence and convolution slots, excluding retained prefix marks.
    pub state: u64,
    pub replay: u64,
    pub prefix: u64,
}

/// Drafter-owned allocations, excluding the target embedding and vocabulary head.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrafterBytes {
    pub owned: u64,
    pub scratch: u64,
}

impl DrafterBytes {
    fn movable(self) -> Movable {
        Movable { id: MovableId::Drafter, allowed: vec![0], expert_arena: false,
            parts: vec![Bytes2 { resident: self.owned, staging: 0 },
                Bytes2 { resident: self.scratch, staging: 0 }], conditional: vec![] }
    }
}

pub struct DenseInputs<'a> {
    pub cfg: &'a Qwen35Config,
    pub gpu: GpuBudget,
    pub layers: &'a [LayerBytes],
    /// Embedding, head, native norm, workspaces, graphs and pending loaded code.
    /// Callers retain the categories and names of their actual allocations.
    pub fixed: Vec<Demand>,
    pub drafter: Option<DrafterBytes>,
    pub requested_pool: Option<u64>,
    pub max_context: u64,
    /// Checkpoint/export-native K|V record width (BF16 initially, native FP8 later).
    pub record_bytes: u64,
    pub pool_overhead: u64,
    pub pool_bounded: Vec<BoundedDemand>,
}

pub fn request(inputs: DenseInputs<'_>) -> Result<PlacementRequest> {
    ensure!(!inputs.cfg.attention.is_empty() && inputs.layers.len() == inputs.cfg.attention.len(),
        "dense Qwen layer byte inventory differs from checkpoint");
    ensure!(inputs.record_bytes > 0 && inputs.max_context > 0, "invalid dense Qwen cache geometry");
    ensure!(inputs.fixed.iter().all(|item| item.gpu == 0)
        && inputs.pool_bounded.iter().all(|item| item.gpu == 0),
        "single-device dense Qwen demand names a peer");
    let layers = inputs.cfg.attention.iter().zip(inputs.layers).map(|(&kind, bytes)| {
        ensure!(bytes.weights > 0, "dense Qwen layer has no native weight admission");
        let mut extra = Vec::new();
        let kv = match kind {
            HybridAttention::Gdn => {
                ensure!(bytes.state > 0 && bytes.replay > 0, "dense GDN lacks exact state/replay admission");
                for (category, group, value) in [(Category::Kv, "GDN state", bytes.state),
                    (Category::Workspace, "GDN replay", bytes.replay),
                    (Category::Prefix, "GDN exact marks", bytes.prefix)] {
                    extra.push(LayerExtraDemand::new(category, group, value, LayerResidency::Owner, Basis::Exact));
                }
                0
            }
            HybridAttention::Full => {
                ensure!(bytes.state == 0 && bytes.replay == 0 && bytes.prefix == 0,
                    "full GQA must not carry fabricated recurrent state");
                inputs.record_bytes.checked_mul(PAGE_ROWS).ok_or_else(|| anyhow::anyhow!("dense KV unit size overflow"))?
            }
        };
        Ok(LayerDemand { kind: if kind == HybridAttention::Gdn { AttentionClass::Gdn } else { AttentionClass::Gqa },
            weights: ModeBytes { whole: bytes.weights, ..Default::default() },
            kv_unit: KvDemand { unit_bytes_whole: kv, ..Default::default() },
            fixed_bytes: ModeBytes::default(), extra, context_indexer: false, colocate: None,
            experts: None, modes: vec![WHOLE] })
    }).collect::<Result<_>>()?;
    Ok(PlacementRequest {
        inventory: Inventory { gpus: vec![inputs.gpu], spark_ranks: 0, peer_access: false },
        attention_placement: None, context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0,
        pool: PoolPolicy::resolve(&[inputs.gpu.capacity_bytes], inputs.max_context,
            inputs.requested_pool, PAGE_ROWS, true),
        layers, pool_overhead: vec![inputs.pool_overhead], pool_bounded: inputs.pool_bounded,
        fixed: inputs.fixed, movables: inputs.drafter.map(DrafterBytes::movable).into_iter().collect(),
        expert_workspace: 0, tp2_workspace: [0; 2],
        onboard: Onboard::Auto, expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![WHOLE], ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: inputs.cfg.hidden as u64 * 2,
            rows: 0, lanes: 0, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner },
        executor: SINGLE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen::{GdnGeometry, GdnOutputGate};

    fn cfg() -> Qwen35Config {
        Qwen35Config { hidden: 4096, intermediate: 12288, vocab: 248320,
            attention: vec![HybridAttention::Gdn, HybridAttention::Full], heads: 16, kv_heads: 4, head_dim: 256,
            gdn: GdnGeometry { key_heads: 16, value_heads: 32, key_dim: 128, value_dim: 128,
                conv_kernel: 4, output_gate: GdnOutputGate::Silu }, rope_dim: 64, rope_theta: 1e7,
            rms_norm_eps: 1e-6, max_context: 262144, mtp_layers: 0, tied_embeddings: false }
    }

    fn inputs<'a>(cfg: &'a Qwen35Config, layers: &'a [LayerBytes], scratch: u64) -> DenseInputs<'a> {
        DenseInputs { cfg, layers, gpu: GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 1 << 30,
            baseline: Baseline::Measured { free_bytes: 14 << 30 } },
            fixed: vec![Demand::new(0, Category::Workspace, "native AOT scratch", scratch, Basis::Exact)],
            drafter: None, requested_pool: None, max_context: 262144, record_bytes: 4096,
            pool_overhead: 4, pool_bounded: Vec::new() }
    }

    #[test]
    fn native_scratch_and_exact_marks_are_charged_before_dense_pool() -> Result<()> {
        let cfg = cfg();
        let layers = [LayerBytes { weights: 1 << 30, state: 1 << 20, replay: 1 << 20, prefix: 2 << 20 },
            LayerBytes { weights: 1 << 30, state: 0, replay: 0, prefix: 0 }];
        let req = request(inputs(&cfg, &layers, 2 << 30))?;
        assert!(req.layers.iter().all(|layer| layer.experts.is_none()));
        assert_eq!(req.layers[0].kv_unit.unit_bytes_whole, 0);
        assert_eq!(req.layers[1].kv_unit.unit_bytes_whole, 64 * 4096);
        let base = solve(&req)?;
        assert!(base.layers.iter().all(|layer| layer.experts == ExpertHome::Dense));
        assert_eq!(base.pool_tokens, 2 << 20);
        let constrained = solve(&request(inputs(&cfg, &layers, 5 << 30))?)?;
        assert!(constrained.pool_tokens < base.pool_tokens && constrained.pool_tokens >= 262144);
        assert!(constrained.items[0].iter().any(|item| item.group == "native AOT scratch" && item.bytes == 5 << 30));
        let mut explicit = inputs(&cfg, &layers, 5 << 30);
        explicit.requested_pool = Some(2 << 20);
        assert!(solve(&request(explicit)?).is_err());
        Ok(())
    }

    #[test]
    fn drafter_owned_and_scratch_are_charged_once_before_pool_admission() -> Result<()> {
        let cfg = cfg();
        let layers = [LayerBytes { weights: 1 << 30, state: 1 << 20, replay: 1 << 20, prefix: 0 },
            LayerBytes { weights: 1 << 30, state: 0, replay: 0, prefix: 0 }];
        let base = solve(&request(inputs(&cfg, &layers, 2 << 30))?)?;
        let mut with_draft = inputs(&cfg, &layers, 2 << 30);
        with_draft.drafter = Some(DrafterBytes { owned: 3 << 30, scratch: 1 << 30 });
        let placement = solve(&request(with_draft)?)?;
        assert_eq!(placement.movables, [(MovableId::Drafter, 0)]);
        assert!(placement.pool_tokens < base.pool_tokens && placement.pool_tokens >= 262144);
        let charged: u64 = placement.items[0].iter().filter(|item| item.category == Category::Drafter)
            .map(|item| item.bytes).sum();
        assert_eq!(charged, 4 << 30);
        let mut explicit = inputs(&cfg, &layers, 2 << 30);
        explicit.drafter = Some(DrafterBytes { owned: 3 << 30, scratch: 1 << 30 });
        explicit.requested_pool = Some(base.pool_tokens);
        assert!(solve(&request(explicit)?).is_err());
        let mut overflow = inputs(&cfg, &layers, 0);
        overflow.drafter = Some(DrafterBytes { owned: u64::MAX, scratch: 1 });
        assert!(solve(&request(overflow)?).is_err());
        Ok(())
    }

    #[test]
    fn admission_rejects_missing_or_fabricated_layer_costs() {
        let cfg = cfg();
        assert!(request(inputs(&cfg, &[], 1)).is_err());
        let layers = [LayerBytes { weights: 1, state: 1, replay: 0, prefix: 0 }; 2];
        assert!(request(inputs(&cfg, &layers, 1)).is_err());
        let layers = [LayerBytes { weights: 1, state: 1, replay: 1, prefix: 0 }; 2];
        assert!(request(inputs(&cfg, &layers, 1)).is_err());
    }
}
