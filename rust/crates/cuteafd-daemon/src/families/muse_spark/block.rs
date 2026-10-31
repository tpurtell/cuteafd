//! Muse Spark's layer chain over an operation backend, in the order of the
//! official `MuseSparkTextDecoderLayer.forward` and `MuseSparkMoE.forward`.
//!
//! The chain owns the order and the residual gates' operands; a backend owns
//! the arithmetic. The CPU reference backend (`reference.rs`) checks the
//! chain against the shared Python reference; the device backend binds the
//! native exports to the same calls.
use anyhow::{ensure, Result};
use cuteafd_loader::families::muse_spark::MuseSparkConfig;

/// One routed selection per row: global expert ids and their renormalized weights.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Routes {
    pub(crate) experts: Vec<u32>,
    pub(crate) weights: Vec<f32>,
}

/// The norms of a layer, by the module that owns them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Norm {
    /// `input_layernorm` (centered).
    Input,
    /// `post_attention_layernorm` (weightless, `post_norm_eps`).
    PostAttention,
    /// `pre_feedforward_layernorm` (centered).
    PreFeedforward,
    /// `post_feedforward_layernorm` (input-scaled, `post_norm_eps`).
    PostFeedforward,
}

/// Which residual gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gate { Attention, Feedforward }

/// The arithmetic a backend supplies, on batches of BF16 rows (device planes
/// on a GPU, vectors on the CPU reference), always in model order.
pub(crate) trait Ops {
    type Rows;

    /// Embedding lookup and its weightless RMSNorm.
    fn embed(&mut self, tokens: &[u32]) -> Result<Self::Rows>;
    fn norm(&mut self, layer: usize, norm: Norm, x: &Self::Rows) -> Result<Self::Rows>;
    /// GQA attention with q/k/o per-head norms and the sigmoid output gate;
    /// RoPE on sliding layers, NoPE on full; appends each row's K/V.
    fn attention(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    fn residual_gate(&mut self, layer: usize, gate: Gate, residual: &Self::Rows, x: &Self::Rows) -> Result<Self::Rows>;
    /// FP32 router on the pre-feedforward-normalized row.
    fn route(&mut self, layer: usize, x: &Self::Rows) -> Result<Vec<Routes>>;
    /// `pre_expert_proj` then `pre_expert_norm` (centered): hidden -> expert width.
    fn pre_expert(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    /// Routed experts at the expert width, each output post-normed, weighted and
    /// summed (on Sparks: dispatch, owned-route partials, the rank reduction).
    fn experts(&mut self, layer: usize, x: &Self::Rows, routes: &[Routes]) -> Result<Self::Rows>;
    /// `post_expert_proj`: expert width -> hidden.
    fn post_expert(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    /// Final scaled RMSNorm and the softcapped head; F32 logits per row.
    fn head(&mut self, x: &Self::Rows) -> Result<Vec<Vec<f32>>>;
}

/// `MuseSparkTextDecoderLayer.forward` for `layers` layers, then the head.
pub(crate) fn forward<O: Ops>(cfg: &MuseSparkConfig, ops: &mut O, tokens: &[u32], layers: usize)
    -> Result<Vec<Vec<f32>>> {
    ensure!(layers <= cfg.layers, "Muse Spark has {} layers, asked for {layers}", cfg.layers);
    let mut hidden = ops.embed(tokens)?;
    for layer in 0..layers {
        hidden = forward_layer(ops, layer, hidden)?;
    }
    ops.head(&hidden)
}

/// One decoder layer: gated attention residual, then the gated MoE residual.
pub(crate) fn forward_layer<O: Ops>(ops: &mut O, layer: usize, hidden: O::Rows) -> Result<O::Rows> {
    let normed = ops.norm(layer, Norm::Input, &hidden)?;
    let attended = ops.attention(layer, &normed)?;
    let attended = ops.norm(layer, Norm::PostAttention, &attended)?;
    let hidden = ops.residual_gate(layer, Gate::Attention, &hidden, &attended)?;
    let normed = ops.norm(layer, Norm::PreFeedforward, &hidden)?;
    let moe = moe(ops, layer, &normed)?;
    let moe = ops.norm(layer, Norm::PostFeedforward, &moe)?;
    ops.residual_gate(layer, Gate::Feedforward, &hidden, &moe)
}

/// `MuseSparkMoE.forward`: route on the normalized row, project down and norm,
/// experts with per-expert post norms, project back up.
fn moe<O: Ops>(ops: &mut O, layer: usize, x: &O::Rows) -> Result<O::Rows> {
    let routes = ops.route(layer, x)?;
    let expert_in = ops.pre_expert(layer, x)?;
    let combined = ops.experts(layer, &expert_in, &routes)?;
    ops.post_expert(layer, &combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Trace { calls: Vec<String> }

    impl Ops for Trace {
        type Rows = String;
        fn embed(&mut self, _: &[u32]) -> Result<String> { self.calls.push("embed".into()); Ok("e".into()) }
        fn norm(&mut self, layer: usize, norm: Norm, x: &String) -> Result<String> {
            self.calls.push(format!("{layer} {norm:?}")); Ok(format!("n({x})"))
        }
        fn attention(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("{layer} attention")); Ok("a".into()) }
        fn residual_gate(&mut self, layer: usize, gate: Gate, residual: &String, x: &String) -> Result<String> {
            self.calls.push(format!("{layer} {gate:?} gate")); Ok(format!("g({residual},{x})"))
        }
        fn route(&mut self, layer: usize, _: &String) -> Result<Vec<Routes>> { self.calls.push(format!("{layer} route")); Ok(vec![]) }
        fn pre_expert(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("{layer} pre")); Ok("p".into()) }
        fn experts(&mut self, layer: usize, _: &String, _: &[Routes]) -> Result<String> { self.calls.push(format!("{layer} experts")); Ok("x".into()) }
        fn post_expert(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("{layer} post")); Ok("q".into()) }
        fn head(&mut self, _: &String) -> Result<Vec<Vec<f32>>> { self.calls.push("head".into()); Ok(vec![]) }
    }

    #[test]
    fn decoder_layer_order_follows_the_official_module() {
        let cfg = MuseSparkConfig::from_hf(&cuteafd_loader::plan::testing::muse_spark_config()).unwrap();
        let mut trace = Trace::default();
        forward(&cfg, &mut trace, &[1, 2], 2).unwrap();
        assert_eq!(&trace.calls[..13], ["embed", "0 Input", "0 attention", "0 PostAttention", "0 Attention gate",
            "0 PreFeedforward", "0 route", "0 pre", "0 experts", "0 post", "0 PostFeedforward", "0 Feedforward gate",
            "1 Input"]);
        assert_eq!(trace.calls.last().unwrap(), "head");
    }
}
