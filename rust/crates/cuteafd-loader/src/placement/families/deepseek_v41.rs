//! V4.1 CED ownership stays fixed; pool and local experts use shared admission.
use super::super::*;

/// Growth beyond the calibrated 48-row serving lanes. Keep verifier storage
/// independent of prefill rows and neural width; both target lanes retain it.
pub fn verify_workspace_growth(gpus: usize, rank: usize) -> u64 {
    assert!((1..=2).contains(&gpus) && rank < gpus);
    let extra_rows = cuteafd_core::V41_VERIFY_ROWS.saturating_sub(48) as u64;
    let sampler_row = 64 + 64 + 2 * 129280u64.div_ceil(32) * 4
        + 2048 * 4 + 256 * (4 + 8) + 32;
    let normalization_row = 40960 + 16 + 10240 + 10240;
    let head_row = if gpus == 1 {
        normalization_row + 129280 * 4 + 8 + sampler_row
    } else {
        10240 + 64640 * 4 + if rank == 0 { 8 } else {
            24 + normalization_row + 129280 * 4 + sampler_row
        }
    };
    let sparse_row = 10 * 64 * 514 * 4 + 120;
    2 * extra_rows * (head_row + sparse_row)
}

/// All weights, lanes, graphs and prefix state are charged in the inventory's
/// baseline. Pool units include private tails supplied by the caller as fixed.
pub fn request(
    inventory: Inventory,
    pool: PoolPolicy,
    unit_bytes: Vec<u64>,
    experts: Vec<ExpertCost>,
    workspace: u64,
    onboard: Onboard,
) -> PlacementRequest {
    let gpus = inventory.gpus.len();
    let count = experts.len();
    PlacementRequest {
        inventory,
        attention_placement: Some(AttentionPlacement::Heads),
        context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0,
        pool,
        pool_overhead: unit_bytes,
        fixed: vec![],
        movables: vec![],
        layers: experts
            .into_iter()
            .enumerate()
            .map(|(layer, cost)| LayerDemand {
                kind: AttentionClass::Csa,
                weights: ModeBytes::default(),
                kv_unit: ModeBytes::default().into(),
                fixed_bytes: ModeBytes::default(),
                extra: vec![],
                context_indexer: false,
                colocate: None,
                experts: Some(cost),
                modes: vec![LayerMode::Whole {
                    gpu: u8::from(gpus == 2 && layer >= count / 2),
                    ffn: FfnMode::Owner,
                }],
            })
            .collect(),
        expert_workspace: workspace,
        tp2_workspace: [workspace; 2],
        onboard,
        expert_gpus: gpus,
        policy: LayerPolicy::default(),
        hops: HopSpec { row_bytes: 0, rows: 0, lanes: 0, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner },
        executor: super::DEEPSEEK_V41,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ced_ownership_and_tp2_loading_peak_are_preserved() {
        let inventory = Inventory {
            gpus: vec![
                GpuBudget {
                    capacity_bytes: 8 << 30,
                    headroom_bytes: 0,
                    baseline: Baseline::Measured {
                        free_bytes: 8 << 30
                    }
                };
                2
            ],
            spark_ranks: 4,
            peer_access: true,
        };
        let pool = PoolPolicy::resolve(&[8 << 30; 2], 512, Some(512), 512, false);
        let cost = ExpertCost {
            whole: Bytes2 {
                resident: 200 << 20,
                staging: 32 << 20,
            },
            half: [Bytes2 {
                resident: 100 << 20,
                staging: 16 << 20,
            }; 2],
            tp2: true,
            spark_ok: true,
        };
        let placement = solve(&request(
            inventory,
            pool,
            vec![1024; 2],
            vec![cost; 40],
            64 << 20,
            Onboard::Layers(3),
        ))
        .unwrap();
        assert_eq!(placement.onboard_layers, 3);
        for (layer, assignment) in placement.layers.iter().enumerate() {
            assert_eq!(
                assignment.mode,
                LayerMode::Whole {
                    gpu: u8::from(layer >= 20),
                    ffn: FfnMode::Owner
                }
            );
            assert_eq!(
                assignment.experts,
                if layer < 3 {
                    ExpertHome::RtxTp2
                } else {
                    ExpertHome::Spark
                }
            );
        }
        assert_eq!(placement.tp2.unwrap().peak_bytes, [380 << 20; 2]);
    }
}
