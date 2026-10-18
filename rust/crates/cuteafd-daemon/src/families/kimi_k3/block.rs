//! Kimi K3's layer chain over an operation backend, in the order of the
//! bundled `modeling_kimi_linear.py` (`KimiLinearModel.forward`,
//! `KimiDecoderLayer._forward_attn_residual`, `KimiSparseMoeBlock.forward`).
//!
//! The chain owns the attention-residual bookkeeping (which candidates each
//! mix sees, when a block is stored, when the prefix sum restarts) and the
//! latent-MoE order; a backend owns the arithmetic. The CPU reference backend
//! (`reference.rs`) checks the chain; the device backend binds the W4-W8
//! exports to the same calls, so serving is integration, not a new chain.
use anyhow::{ensure, Result};
use cuteafd_loader::families::kimi_k3::config::{KimiAttention, KimiK3Config};

/// The mix sites of a layer: before attention and before the FFN; the
/// model's output mix precedes the final norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MixSite {
    Attention(usize),
    Ffn(usize),
    Output,
}

/// One routed selection per row: expert ids and their renormalized weights.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Routes {
    pub(crate) experts: Vec<u32>,
    pub(crate) weights: Vec<f32>,
}

/// The arithmetic a backend supplies. Buffers are `Self::Rows`: a batch of
/// `rows` BF16 rows of the width each call names (device planes on a GPU,
/// vectors on the CPU reference). Every call is in model order; a backend may
/// fuse adjacent calls when it records the same rounding points.
pub(crate) trait Ops {
    type Rows;

    /// `_apply_attn_res`: softmax mix over `stored` block residuals plus `current`.
    fn residual_mix(&mut self, site: MixSite, stored: &[&Self::Rows], current: &Self::Rows) -> Result<Self::Rows>;
    /// `KimiRMSNorm` with the layer's `input_layernorm`/`post_attention_layernorm`
    /// (or the model norm for `None`).
    fn norm(&mut self, layer: Option<usize>, attention: bool, x: &Self::Rows) -> Result<Self::Rows>;
    /// KDA over `x` (normalized), advancing each row's sequence state.
    fn kda(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    /// Full causal NoPE MLA with the sigmoid output gate, appending each row's
    /// latent record to its sequence's cache.
    fn mla(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    /// Situ dense MLP (layer 0).
    fn dense_ffn(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    /// FP32 sigmoid router with selection-only bias, renormalized top-k.
    fn route(&mut self, layer: usize, x: &Self::Rows) -> Result<Vec<Routes>>;
    /// `routed_expert_down_proj`: hidden -> latent.
    fn latent_down(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    /// Routed experts in latent space, weighted and summed over the selection
    /// (on Sparks: dispatch, per-rank partials, and the complete reduction).
    fn experts(&mut self, layer: usize, latent: &Self::Rows, routes: &[Routes]) -> Result<Self::Rows>;
    /// `routed_expert_norm` over the complete sum, then `routed_expert_up_proj`.
    fn latent_up(&mut self, layer: usize, combined: &Self::Rows) -> Result<Self::Rows>;
    /// Situ shared experts on the original normalized row.
    fn shared_ffn(&mut self, layer: usize, x: &Self::Rows) -> Result<Self::Rows>;
    /// BF16 `a + b` as `torch` adds two BF16 tensors.
    fn add(&mut self, a: &Self::Rows, b: &Self::Rows) -> Result<Self::Rows>;
}

/// Run every layer and the output mix + final norm over embedded rows.
/// Returns the normalized hidden rows the head consumes.
pub(crate) fn forward<O: Ops>(cfg: &KimiK3Config, ops: &mut O, embedded: O::Rows, layers: usize) -> Result<O::Rows>
where O::Rows: Clone {
    ensure!(layers <= cfg.layers, "Kimi K3 has {} layers, asked for {layers}", cfg.layers);
    // Stored block residuals for this pass; at most `cfg.max_residual_blocks()`.
    let mut stored: Vec<O::Rows> = Vec::with_capacity(cfg.max_residual_blocks());
    let mut hidden = embedded;
    for layer in 0..layers {
        // prefix_sum = hidden_states (this layer's input).
        let prefix = hidden;
        let attention_in = if stored.is_empty() { prefix.clone() }
            else { ops.residual_mix(MixSite::Attention(layer), &refs(&stored), &prefix)? };
        // A block boundary stores the prefix sum and restarts it from the attention output.
        let restart = layer % cfg.residual_block == 0;
        if restart {
            ensure!(stored.len() < cfg.max_residual_blocks(), "residual block store overflow at layer {layer}");
            stored.push(prefix.clone());
        }
        let normed = ops.norm(Some(layer), true, &attention_in)?;
        let attended = match cfg.attention[layer] {
            KimiAttention::Kda => ops.kda(layer, &normed)?,
            KimiAttention::Mla => ops.mla(layer, &normed)?,
        };
        let prefix = if restart { attended } else { ops.add(&prefix, &attended)? };
        let ffn_in = ops.residual_mix(MixSite::Ffn(layer), &refs(&stored), &prefix)?;
        let normed = ops.norm(Some(layer), false, &ffn_in)?;
        let ffn = if cfg.dense[layer] { ops.dense_ffn(layer, &normed)? } else { moe(ops, layer, &normed)? };
        hidden = ops.add(&prefix, &ffn)?;
    }
    let output = if stored.is_empty() { hidden } else { ops.residual_mix(MixSite::Output, &refs(&stored), &hidden)? };
    ops.norm(None, false, &output)
}

/// `KimiSparseMoeBlock.forward`: route on the original row, experts in latent
/// space, norm after the complete combine, up-project, add the shared FFN.
fn moe<O: Ops>(ops: &mut O, layer: usize, x: &O::Rows) -> Result<O::Rows> {
    let routes = ops.route(layer, x)?;
    let latent = ops.latent_down(layer, x)?;
    let combined = ops.experts(layer, &latent, &routes)?;
    let routed = ops.latent_up(layer, &combined)?;
    let shared = ops.shared_ffn(layer, x)?;
    ops.add(&routed, &shared)
}

fn refs<T>(items: &[T]) -> Vec<&T> { items.iter().collect() }

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the call sequence and which candidates each mix saw.
    #[derive(Default)]
    struct Trace { calls: Vec<String> }

    impl Ops for Trace {
        type Rows = String;
        fn residual_mix(&mut self, site: MixSite, stored: &[&String], current: &String) -> Result<String> {
            let list: Vec<&str> = stored.iter().map(|s| s.as_str()).collect();
            self.calls.push(format!("mix {site:?} [{}] + {current}", list.join(",")));
            Ok(format!("mix({current})"))
        }
        fn norm(&mut self, layer: Option<usize>, attention: bool, x: &String) -> Result<String> {
            self.calls.push(format!("norm {layer:?} {attention}"));
            Ok(format!("n({x})"))
        }
        fn kda(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("kda {layer}")); Ok(format!("a{layer}")) }
        fn mla(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("mla {layer}")); Ok(format!("a{layer}")) }
        fn dense_ffn(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("dense {layer}")); Ok(format!("f{layer}")) }
        fn route(&mut self, layer: usize, _: &String) -> Result<Vec<Routes>> { self.calls.push(format!("route {layer}")); Ok(vec![]) }
        fn latent_down(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("down {layer}")); Ok("l".into()) }
        fn experts(&mut self, layer: usize, _: &String, _: &[Routes]) -> Result<String> { self.calls.push(format!("experts {layer}")); Ok("e".into()) }
        fn latent_up(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("up {layer}")); Ok("u".into()) }
        fn shared_ffn(&mut self, layer: usize, _: &String) -> Result<String> { self.calls.push(format!("shared {layer}")); Ok("s".into()) }
        fn add(&mut self, a: &String, b: &String) -> Result<String> { Ok(format!("{a}+{b}")) }
    }

    fn cfg() -> KimiK3Config {
        KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap()
    }

    #[test]
    fn residual_blocks_follow_the_reference_schedule() {
        let cfg = cfg();
        let mut trace = Trace::default();
        forward(&cfg, &mut trace, "emb".into(), 93).unwrap();
        let mixes: Vec<&String> = trace.calls.iter().filter(|c| c.starts_with("mix")).collect();
        // Layer 0: no attention mix (no stored blocks yet); the FFN mix sees the
        // embedding as the first stored block and the attention output as prefix.
        assert_eq!(mixes[0], "mix Ffn(0) [emb] + a0");
        // Layer 1 mixes before attention over [emb] and its input.
        assert!(mixes[1].starts_with("mix Attention(1) [emb] + "));
        // Layer 12 stores its input as the second block before attention; its
        // FFN mix sees two blocks and a restarted prefix (its attention output).
        let l12 = mixes.iter().find(|m| m.starts_with("mix Ffn(12)")).unwrap();
        assert!(l12.ends_with("] + a12") && l12.matches(',').count() >= 1, "{l12}");
        // The output mix sees all 8 stored blocks.
        let output = mixes.last().unwrap();
        assert!(output.starts_with("mix Output"));
        assert_eq!(trace.calls.iter().filter(|c| c.starts_with("mla")).count(), 24);
        assert_eq!(trace.calls.iter().filter(|c| c.starts_with("dense")).count(), 1);
        // Latent MoE order within a routed layer.
        let i = trace.calls.iter().position(|c| c == "route 1").unwrap();
        assert_eq!(&trace.calls[i..i + 5], ["route 1", "down 1", "experts 1", "up 1", "shared 1"]);
        assert_eq!(trace.calls.last().unwrap(), "norm None false");
    }
}
