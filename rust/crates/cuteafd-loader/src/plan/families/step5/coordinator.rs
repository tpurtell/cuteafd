//! Native source-weight contract, not an executable Step-5 coordinator.
//! Reuses the planner inventory and shared dense header/admission validator.
use super::{Config, Model};
use crate::formats::native_dense::NativeDenseLayout;
use crate::plan::{checkpoint::Checkpoint, spec::Component};
use anyhow::{ensure, Result};
use cuteafd_core::dense_attention::{
    GqaGeometry, NormWeight, RotaryArithmetic, RouterBiasOrder, SigmoidRouter, SlidingWindow,
};

#[derive(Debug, Clone)]
pub struct CoordinatorContract {
    pub weights: NativeDenseLayout,
    pub norm_weight: NormWeight,
    pub norm_eps: f32,
    pub router: SigmoidRouter,
    pub layers: Vec<LayerContract>,
}

#[derive(Debug, Clone)]
pub struct LayerContract {
    pub layer: usize,
    pub geometry: GqaGeometry,
    pub theta: f64,
    pub rotary_arithmetic: RotaryArithmetic,
    /// None means full-history sparse attention, not dense causal attention.
    pub sliding: Option<SlidingWindow>,
}

impl CoordinatorContract {
    /// Requires every selected native target header. Routed experts, vision
    /// and independently owned MTP stages are not charged as target weights.
    /// Full tensors are conservatively replicated; no unimplemented TP packing
    /// or BF16-to-FP8 conversion is credited as a memory saving.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let model = Model::open(checkpoint)?;
        let cfg: &Config = &model.cfg;
        let declarations = model
            .tensors
            .iter()
            .filter(|(_, t)| {
                !matches!(
                    t.role.component,
                    Component::RoutedExpert | Component::Vision | Component::Speculator
                )
            })
            .map(|(name, t)| (name.clone(), t.shape.clone(), t.dtype.clone()));
        let weights = NativeDenseLayout::from_headers(checkpoint, declarations)?;
        let layers = (0..cfg.layers)
            .map(|layer| {
                Ok(LayerContract {
                    layer,
                    geometry: GqaGeometry {
                        query_heads: cfg.heads,
                        kv_heads: cfg.groups,
                        qk_dim: cfg.dim,
                        value_dim: cfg.dim,
                        rotary_dim: cfg.rotary[layer],
                    }
                    .validate()?,
                    theta: cfg.theta[layer],
                    rotary_arithmetic: RotaryArithmetic::FusedImaginary,
                    sliding: (!cfg.full[layer]).then_some(SlidingWindow {
                        window: cfg.window as u64,
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            layers.len() == 92,
            "target coordinator must exclude MTP stages"
        );
        Ok(Self {
            weights,
            norm_weight: NormWeight::ZeroCentered,
            norm_eps: cfg.norm_eps,
            router: SigmoidRouter {
                topk: cfg.topk,
                factor: 3.0,
                epsilon: 1e-20,
                bias_order: RouterBiasOrder::SubtractSelected,
            },
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::families::step5::tests::snapshot;
    use cuteafd_core::DType;

    #[test]
    fn native_coordinator_preserves_source_precision_and_layer_schedule() {
        let dir = snapshot("fp8");
        let cp = Checkpoint::open(dir.path()).unwrap();
        let c = CoordinatorContract::from_checkpoint(&cp).unwrap();
        assert_eq!(c.norm_eps, 1e-5);
        assert_eq!(c.norm_weight, NormWeight::ZeroCentered);
        assert_eq!(c.router.bias_order, RouterBiasOrder::SubtractSelected);
        assert_eq!(c.layers.len(), 92);
        assert_eq!(c.layers.iter().filter(|l| l.sliding.is_some()).count(), 69);
        for layer in &c.layers {
            assert_eq!(layer.geometry.value_dim, 192);
            assert_eq!(layer.rotary_arithmetic, RotaryArithmetic::FusedImaginary);
            assert_eq!(
                layer.geometry.rotary_dim,
                if layer.layer % 4 == 3 { 64 } else { 192 }
            );
            assert_eq!(
                layer.sliding.map(|s| s.window),
                if layer.layer % 4 == 3 {
                    None
                } else {
                    Some(512)
                }
            );
        }
        assert!(c
            .weights
            .tensors
            .iter()
            .all(|t| !t.meta.name.contains("layers.92.")
                && !t.meta.name.contains("vision")
                && !t.meta.name.contains("moe.gate_proj")));
        let find = |name: &str| {
            c.weights
                .tensors
                .iter()
                .find(|t| t.meta.name == name)
                .unwrap()
        };
        assert_eq!(
            find("model.layers.3.moe.gate.weight").meta.dtype,
            DType::Bf16
        );
        assert_eq!(
            find("model.layers.3.moe.router_bias").meta.dtype,
            DType::F32
        );
        assert_eq!(
            find("model.layers.0.self_attn.q_norm.weight").meta.dtype,
            DType::F32
        );
        assert_eq!(
            find("model.layers.0.self_attn.v_proj.weight").meta.shape,
            [768, 4096]
        );
        assert_eq!(
            find("model.layers.0.self_attn.g_proj.weight").meta.shape,
            [64, 4096]
        );
        let exact: u64 = c
            .weights
            .tensors
            .iter()
            .map(|t| t.meta.byte_length.max(256))
            .sum();
        assert_eq!(c.weights.resident_bytes, exact);
        assert_eq!(exact, 27_651_962_624);
        assert_eq!(c.weights.tensors.len(), 1467);
        assert_eq!(
            c.weights.check_weight_budget(exact + 12, &[5, 7]).unwrap(),
            exact + 12
        );
        assert!(c.weights.check_weight_budget(exact + 11, &[5, 7]).is_err());
        assert!(c
            .weights
            .check_weight_budget(u64::MAX, &[u64::MAX])
            .is_err());
        // The contract does not depend on routed storage or sidecars.
        let nv = snapshot("nvfp4");
        assert_eq!(
            CoordinatorContract::from_checkpoint(&Checkpoint::open(nv.path()).unwrap())
                .unwrap()
                .weights
                .resident_bytes,
            exact
        );
    }

    #[test]
    fn headers_fail_closed_before_native_allocation() {
        let dir = snapshot("fp8");
        for mutation in 0..5 {
            let mut cp = Checkpoint::open(dir.path()).unwrap();
            let t = cp
                .tensors
                .iter_mut()
                .find(|t| t.meta.name == "model.layers.0.self_attn.q_norm.weight")
                .unwrap();
            match mutation {
                0 => t.meta.dtype = DType::Bf16,
                1 => t.meta.shape = vec![191],
                2 => t.meta.byte_length -= 1,
                3 => t.meta.byte_offset = u64::MAX,
                _ => cp
                    .tensors
                    .retain(|t| t.meta.name != "model.layers.0.self_attn.q_norm.weight"),
            }
            assert!(
                CoordinatorContract::from_checkpoint(&cp).is_err(),
                "mutation {mutation}"
            );
        }
        for bad in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!("nan"),
        ] {
            let mut cp = Checkpoint::open(dir.path()).unwrap();
            cp.config["text_config"]["rms_norm_eps"] = bad;
            assert!(CoordinatorContract::from_checkpoint(&cp).is_err());
        }
    }
}
