//! A CPU `Ops` over dense BF16 weights in FP32 arithmetic with the reference's
//! rounding points: the chain's oracle for small synthetic geometries and for
//! single-layer slices of the real checkpoint. KDA's recurrence is injected
//! (its numerics belong to the shared KDA work); routed experts take
//! dequantized weights, the contract an EXL3 kernel must meet after decode.
use super::block::{MixSite, Ops, Routes};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::kimi_k3::{config::KimiK3Config, reference as contract};

/// `[rows][width]` BF16-valued rows.
pub(crate) type Rows = Vec<Vec<f32>>;

/// A row-major `[n, k]` BF16 matrix (`nn.Linear.weight`).
#[derive(Clone)]
pub(crate) struct Matrix { pub(crate) n: usize, pub(crate) k: usize, pub(crate) w: Vec<f32> }

impl Matrix {
    /// `F.linear` in BF16: FP32 accumulation, one rounding of each output.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.k);
        (0..self.n).map(|i| contract::bf16(self.w[i * self.k..(i + 1) * self.k].iter().zip(x).map(|(w, v)| w * v).sum())).collect()
    }
}

/// `KimiMLP` / `KimiBlockSparseMLP` weights (gate, up, down).
#[derive(Clone)]
pub(crate) struct Ffn { pub(crate) gate: Matrix, pub(crate) up: Matrix, pub(crate) down: Matrix }

impl Ffn {
    fn apply(&self, activation: cuteafd_core::FfnActivation, x: &[f32]) -> Vec<f32> {
        let (g, u) = (self.gate.apply(x), self.up.apply(x));
        self.down.apply(&g.iter().zip(&u).map(|(g, u)| activation.evaluate(*g, *u)).collect::<Vec<_>>())
    }
}

/// One MLA layer's weights.
#[derive(Clone)]
pub(crate) struct Mla {
    pub(crate) q_a: Matrix, pub(crate) q_a_norm: Vec<f32>, pub(crate) q_b: Matrix,
    pub(crate) kv_a: Matrix, pub(crate) kv_a_norm: Vec<f32>, pub(crate) kv_b: Matrix,
    pub(crate) g: Matrix, pub(crate) o: Matrix,
}

/// One routed layer's weights.
#[derive(Clone)]
pub(crate) struct Moe {
    pub(crate) router: Matrix, pub(crate) bias: Vec<f32>, pub(crate) down: Matrix, pub(crate) norm: Vec<f32>,
    pub(crate) up: Matrix, pub(crate) experts: Vec<Ffn>, pub(crate) shared: Ffn,
}

#[derive(Clone)]
pub(crate) struct Layer {
    pub(crate) input_norm: Vec<f32>, pub(crate) post_norm: Vec<f32>,
    pub(crate) attention_mix: (Vec<f32>, Vec<f32>), pub(crate) ffn_mix: (Vec<f32>, Vec<f32>),
    pub(crate) mla: Option<Mla>, pub(crate) dense: Option<Ffn>, pub(crate) moe: Option<Moe>,
}

/// Injected KDA: `(layer, normalized rows) -> attention output rows`, keeping
/// whatever per-sequence state the implementation needs.
pub(crate) type Kda<'a> = dyn FnMut(usize, &Rows) -> Result<Rows> + 'a;

pub(crate) struct Reference<'a> {
    pub(crate) cfg: KimiK3Config,
    pub(crate) layers: Vec<Layer>,
    pub(crate) norm: Vec<f32>,
    pub(crate) output_mix: (Vec<f32>, Vec<f32>),
    pub(crate) kda: Box<Kda<'a>>,
    /// Per MLA layer: every earlier row's (k [heads][q_head], v [heads][v]) of one sequence.
    cache: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
}

impl<'a> Reference<'a> {
    pub(crate) fn new(cfg: KimiK3Config, layers: Vec<Layer>, norm: Vec<f32>, output_mix: (Vec<f32>, Vec<f32>),
        kda: Box<Kda<'a>>) -> Self {
        let cache = vec![Vec::new(); cfg.layers];
        Self { cfg, layers, norm, output_mix, kda, cache }
    }

    fn layer(&self, layer: usize) -> Result<&Layer> { self.layers.get(layer).context("layer outside the reference weights") }
}

impl Ops for Reference<'_> {
    type Rows = Rows;

    fn residual_mix(&mut self, site: MixSite, stored: &[&Rows], current: &Rows) -> Result<Rows> {
        let (norm, proj) = match site {
            MixSite::Attention(l) => &self.layer(l)?.attention_mix,
            MixSite::Ffn(l) => &self.layer(l)?.ffn_mix,
            MixSite::Output => &self.output_mix,
        };
        Ok((0..current.len()).map(|r| {
            let candidates: Vec<&[f32]> = stored.iter().map(|s| s[r].as_slice()).chain([current[r].as_slice()]).collect();
            contract::residual_mix(&candidates, norm, proj, self.cfg.rms_norm_eps)
        }).collect())
    }

    fn norm(&mut self, layer: Option<usize>, attention: bool, x: &Rows) -> Result<Rows> {
        let weight = match layer {
            None => &self.norm,
            Some(l) => if attention { &self.layer(l)?.input_norm } else { &self.layer(l)?.post_norm },
        };
        Ok(x.iter().map(|row| contract::rms_norm(row, weight, self.cfg.rms_norm_eps)).collect())
    }

    fn kda(&mut self, layer: usize, x: &Rows) -> Result<Rows> { (self.kda)(layer, x) }

    /// One sequence, rows in order; NoPE: q = [nope | raw rope], k = [kv_b nope | shared raw k_rot].
    fn mla(&mut self, layer: usize, x: &Rows) -> Result<Rows> {
        let cfg = &self.cfg;
        let (heads, nope, rope, v_dim) = (cfg.heads, cfg.qk_nope_dim, cfg.qk_rope_dim, cfg.v_head_dim);
        let eps = contract::MLA_RANK_NORM_EPS;
        let mla = self.layers[layer].mla.clone().context("MLA weights")?;
        let scale = ((nope + rope) as f64).powf(-0.5);
        let mut out = Vec::with_capacity(x.len());
        for row in x {
            let q = mla.q_b.apply(&contract::rms_norm(&mla.q_a.apply(row), &mla.q_a_norm, eps));
            let compressed = mla.kv_a.apply(row);
            let (latent, k_rot) = compressed.split_at(cfg.kv_lora_rank);
            let kv = mla.kv_b.apply(&contract::rms_norm(latent, &mla.kv_a_norm, eps));
            let k: Vec<f32> = (0..heads).flat_map(|h| kv[h * (nope + v_dim)..h * (nope + v_dim) + nope].iter().chain(k_rot).copied()).collect();
            let v: Vec<f32> = (0..heads).flat_map(|h| kv[h * (nope + v_dim) + nope..(h + 1) * (nope + v_dim)].to_vec()).collect();
            self.cache[layer].push((k, v));
            let history = &self.cache[layer];
            let mut attended = vec![0f32; heads * v_dim];
            for h in 0..heads {
                let qh = &q[h * (nope + rope)..(h + 1) * (nope + rope)];
                let scores: Vec<f64> = history.iter().map(|(k, _)| qh.iter().zip(&k[h * (nope + rope)..(h + 1) * (nope + rope)])
                    .map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale).collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = exp.iter().sum();
                // Softmax in FP32, probabilities rounded to the query dtype (BF16), then PV.
                for (p, (_, v)) in exp.iter().zip(history) {
                    let p = contract::bf16((p / total) as f32);
                    for d in 0..v_dim { attended[h * v_dim + d] += p * v[h * v_dim + d]; }
                }
            }
            let gate = mla.g.apply(row);
            let gated: Vec<f32> = attended.iter().zip(&gate).map(|(a, g)| contract::bf16(contract::bf16(*a) * contract::bf16(1.0 / (1.0 + (-g).exp())))).collect();
            out.push(mla.o.apply(&gated));
        }
        Ok(out)
    }

    fn dense_ffn(&mut self, layer: usize, x: &Rows) -> Result<Rows> {
        let ffn = self.layer(layer)?.dense.as_ref().context("dense FFN weights")?;
        Ok(x.iter().map(|row| ffn.apply(self.cfg.activation(), row)).collect())
    }

    fn route(&mut self, layer: usize, x: &Rows) -> Result<Vec<Routes>> {
        let moe = self.layer(layer)?.moe.as_ref().context("MoE weights")?;
        Ok(x.iter().map(|row| {
            // FP32 logits from the BF16 weight and row, no output rounding.
            let logits: Vec<f32> = (0..moe.router.n).map(|e| moe.router.w[e * moe.router.k..(e + 1) * moe.router.k]
                .iter().zip(row).map(|(w, v)| w * v).sum()).collect();
            let selected = contract::route(&logits, &moe.bias, self.cfg.topk, self.cfg.routed_scale as f32);
            Routes { experts: selected.iter().map(|(e, _)| *e as u32).collect(), weights: selected.iter().map(|(_, w)| *w).collect() }
        }).collect())
    }

    fn latent_down(&mut self, layer: usize, x: &Rows) -> Result<Rows> {
        let moe = self.layer(layer)?.moe.as_ref().context("MoE weights")?;
        Ok(x.iter().map(|row| moe.down.apply(row)).collect())
    }

    /// Weighted sum in FP32 of each expert's BF16 output, rounded once (`moe_infer`).
    fn experts(&mut self, layer: usize, latent: &Rows, routes: &[Routes]) -> Result<Rows> {
        ensure!(latent.len() == routes.len(), "one selection per row");
        let moe = self.layer(layer)?.moe.as_ref().context("MoE weights")?;
        let activation = self.cfg.activation();
        latent.iter().zip(routes).map(|(row, routes)| {
            let mut sum = vec![0f32; row.len()];
            for (expert, weight) in routes.experts.iter().zip(&routes.weights) {
                let y = moe.experts.get(*expert as usize).context("expert id outside the layer")?.apply(activation, row);
                for (s, v) in sum.iter_mut().zip(y) { *s += weight * v; }
            }
            Ok(sum.into_iter().map(contract::bf16).collect())
        }).collect()
    }

    fn latent_up(&mut self, layer: usize, combined: &Rows) -> Result<Rows> {
        let moe = self.layer(layer)?.moe.as_ref().context("MoE weights")?;
        Ok(combined.iter().map(|row| moe.up.apply(&contract::rms_norm(row, &moe.norm, self.cfg.rms_norm_eps))).collect())
    }

    fn shared_ffn(&mut self, layer: usize, x: &Rows) -> Result<Rows> {
        let moe = self.layer(layer)?.moe.as_ref().context("MoE weights")?;
        Ok(x.iter().map(|row| moe.shared.apply(self.cfg.activation(), row)).collect())
    }

    fn add(&mut self, a: &Rows, b: &Rows) -> Result<Rows> {
        Ok(a.iter().zip(b).map(|(a, b)| a.iter().zip(b).map(|(x, y)| contract::bf16(x + y)).collect()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::kimi_k3::block::forward;
    use serde_json::json;

    /// A tiny Kimi geometry: 4 layers, residual block 2, MLA at 1-based 2 and 4.
    fn tiny() -> KimiK3Config {
        let mut config = cuteafd_loader::plan::testing::kimi_k3_config();
        let t = &mut config["text_config"];
        for (key, value) in [("hidden_size", 16), ("intermediate_size", 24), ("num_hidden_layers", 4), ("num_experts", 6),
            ("num_experts_per_token", 2), ("moe_intermediate_size", 8), ("routed_expert_hidden_size", 8),
            ("num_shared_experts", 1), ("attn_res_block_size", 2), ("num_attention_heads", 2), ("num_key_value_heads", 2),
            ("q_lora_rank", 8), ("kv_lora_rank", 8), ("qk_nope_head_dim", 4), ("qk_rope_head_dim", 2), ("v_head_dim", 4)] {
            t[key] = json!(value);
        }
        t["linear_attn_config"]["full_attn_layers"] = json!([2, 4]);
        t["linear_attn_config"]["kda_layers"] = json!([1, 3]);
        t["linear_attn_config"]["num_heads"] = json!(2);
        t["linear_attn_config"]["head_dim"] = json!(8);
        KimiK3Config::from_hf(&config).unwrap()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            contract::bf16(((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5)
        }
        fn matrix(&mut self, n: usize, k: usize) -> Matrix { Matrix { n, k, w: (0..n * k).map(|_| self.next()).collect() } }
        fn vector(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| contract::bf16(1.0 + self.next())).collect() }
        fn ffn(&mut self, h: usize, i: usize) -> Ffn { Ffn { gate: self.matrix(i, h), up: self.matrix(i, h), down: self.matrix(h, i) } }
    }

    fn weights(cfg: &KimiK3Config, rng: &mut Rng) -> Vec<Layer> {
        let h = cfg.hidden;
        let q_head = cfg.qk_nope_dim + cfg.qk_rope_dim;
        (0..cfg.layers).map(|l| Layer {
            input_norm: rng.vector(h), post_norm: rng.vector(h),
            attention_mix: (rng.vector(h), rng.vector(h)), ffn_mix: (rng.vector(h), rng.vector(h)),
            mla: (cfg.attention[l] == cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla).then(|| Mla {
                q_a: rng.matrix(cfg.q_lora_rank, h), q_a_norm: rng.vector(cfg.q_lora_rank),
                q_b: rng.matrix(cfg.heads * q_head, cfg.q_lora_rank),
                kv_a: rng.matrix(cfg.kv_lora_rank + cfg.qk_rope_dim, h), kv_a_norm: rng.vector(cfg.kv_lora_rank),
                kv_b: rng.matrix(cfg.heads * (cfg.qk_nope_dim + cfg.v_head_dim), cfg.kv_lora_rank),
                g: rng.matrix(cfg.heads * cfg.v_head_dim, h), o: rng.matrix(h, cfg.heads * cfg.v_head_dim) }),
            dense: cfg.dense[l].then(|| rng.ffn(h, cfg.dense_intermediate)),
            moe: (!cfg.dense[l]).then(|| Moe {
                router: rng.matrix(cfg.experts, h), bias: (0..cfg.experts).map(|_| rng.next()).collect(),
                down: rng.matrix(cfg.latent, h), norm: rng.vector(cfg.latent), up: rng.matrix(h, cfg.latent),
                experts: (0..cfg.experts).map(|_| rng.ffn(cfg.latent, cfg.moe_intermediate)).collect(),
                shared: rng.ffn(h, cfg.moe_intermediate * cfg.shared_experts) }),
        }).collect()
    }

    fn model(seed: u64) -> (KimiK3Config, Vec<Layer>, Vec<f32>, (Vec<f32>, Vec<f32>), Rows) {
        let cfg = tiny();
        let mut rng = Rng(seed);
        let layers = weights(&cfg, &mut rng);
        let (norm, mix) = (rng.vector(cfg.hidden), (rng.vector(cfg.hidden), rng.vector(cfg.hidden)));
        let rows = (0..5).map(|_| (0..cfg.hidden).map(|_| rng.next()).collect()).collect();
        (cfg, layers, norm, mix, rows)
    }

    /// A stand-in recurrence with per-sequence state (not KDA's numerics):
    /// enough to show state carries across calls in row order.
    fn running_sum() -> Box<Kda<'static>> {
        let mut state: std::collections::HashMap<usize, Vec<f32>> = Default::default();
        Box::new(move |layer, x: &Rows| Ok(x.iter().map(|row| {
            let s = state.entry(layer).or_insert_with(|| vec![0.0; row.len()]);
            for (s, v) in s.iter_mut().zip(row) { *s = contract::bf16(0.5 * *s + v); }
            s.clone()
        }).collect()))
    }

    #[test]
    fn mla_rank_norms_do_not_inherit_decoder_epsilon() {
        let (mut cfg, layers, norm, mix, mut rows) = model(19);
        for value in rows.iter_mut().flatten() { *value = contract::bf16(*value * 0.001); }
        let mut base = Reference::new(cfg.clone(), layers.clone(), norm.clone(), mix.clone(), running_sum());
        let expected = base.mla(1, &rows).unwrap();
        cfg.rms_norm_eps = 0.1;
        let mut changed = Reference::new(cfg, layers, norm, mix, running_sum());
        assert_eq!(expected, changed.mla(1, &rows).unwrap());
        assert_ne!(base.norm(Some(1), true, &rows).unwrap(), changed.norm(Some(1), true, &rows).unwrap());
    }

    #[test]
    fn prefill_and_decode_agree_and_outputs_are_finite() {
        let (cfg, layers, norm, mix, rows) = model(7);
        let mut whole = Reference::new(cfg.clone(), layers.clone(), norm.clone(), mix.clone(), running_sum());
        let all = forward(&cfg, &mut whole, rows.clone(), cfg.layers).unwrap();
        assert!(all.iter().flatten().all(|v| v.is_finite()));
        // Decode one row at a time through the same caches: every op is row-local
        // except attention state, which both paths advance in order.
        let mut stepwise = Reference::new(cfg.clone(), layers, norm, mix, running_sum());
        let steps: Rows = rows.iter().map(|row| forward(&cfg, &mut stepwise, vec![row.clone()], cfg.layers).unwrap().remove(0)).collect();
        assert_eq!(all, steps);
    }

    #[test]
    fn latent_norm_and_routing_change_the_result() {
        let (cfg, mut layers, norm, mix, rows) = model(11);
        let mut base = Reference::new(cfg.clone(), layers.clone(), norm.clone(), mix.clone(), running_sum());
        let reference = forward(&cfg, &mut base, rows.clone(), cfg.layers).unwrap();
        // Scaling the latent norm weight must move the output (the norm is live, after the combine).
        for layer in layers.iter_mut().filter_map(|l| l.moe.as_mut()) { for w in &mut layer.norm { *w = contract::bf16(*w * 2.0); } }
        let mut scaled = Reference::new(cfg.clone(), layers, norm, mix, running_sum());
        assert_ne!(forward(&cfg, &mut scaled, rows, cfg.layers).unwrap(), reference);
    }
}
