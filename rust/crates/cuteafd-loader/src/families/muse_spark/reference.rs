//! CPU contracts for Muse Spark operations, transcribed from the official
//! `transformers` `modeling_muse_spark.py` (5.22.1, the shared reference
//! `python/reference/families/muse_spark/model.py` pins its hash). They are
//! the oracles native exports are compared against: explicit BF16 rounding
//! points, FP64 accumulation where the reference reduces in FP32 (kernel
//! comparisons then bound reduction order, not semantics).
//!
//! Every `(1 + w)` of a centered or input-scaled norm is a BF16 add: the
//! module computes `1.0 + self.weight` on the BF16 parameter before `.float()`.

/// Round to BF16 (nearest even), as `tensor.to(torch.bfloat16)`.
pub fn bf16(value: f32) -> f32 {
    if value.is_nan() { return value; }
    let bits = value.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff_0000)
}

fn inv_rms(x: impl Iterator<Item = f32> + Clone, n: usize, eps: f64) -> f64 {
    let mean = x.map(|v| (v as f64).powi(2)).sum::<f64>() / n as f64;
    1.0 / (mean + eps).sqrt()
}

/// `MuseSparkRMSNorm(with_scale=False)`: `x * (mean(x^2) + eps)^-0.5` in FP32,
/// rounded to the input dtype (BF16). Embedding, q/k/o per head and the
/// post-attention norm.
pub fn rms(x: &[f32], eps: f64) -> Vec<f32> {
    let inv = inv_rms(x.iter().copied(), x.len(), eps);
    x.iter().map(|v| bf16((*v as f64 * inv) as f32)).collect()
}

/// `MuseSparkRMSNorm(with_scale=True)` (the final norm): plain `w`, not `1 + w`.
pub fn rms_scaled(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let inv = inv_rms(x.iter().copied(), x.len(), eps);
    x.iter().zip(weight).map(|(v, w)| bf16(((*v as f64 * inv) as f32) * w)).collect()
}

/// `MuseSparkTextCenteredRMSNorm`: normalize in FP32, multiply by `bf16(1 + w)`, round.
/// Input, pre-feedforward and pre-expert norms.
pub fn centered(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let inv = inv_rms(x.iter().copied(), x.len(), eps);
    x.iter().zip(weight).map(|(v, w)| bf16(((*v as f64 * inv) as f32) * bf16(1.0 + w))).collect()
}

/// `MuseSparkTextInputScaledRMSNorm`: scale by `bf16(1 + w)` first, round to
/// BF16, then normalize in FP32 and round. Post-feedforward and post-expert norms.
pub fn input_scaled(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let scaled: Vec<f32> = x.iter().zip(weight).map(|(v, w)| bf16(v * bf16(1.0 + w))).collect();
    rms(&scaled, eps)
}

/// `MuseSparkResidualGate` on BF16 rows: FP32 gate math (the module is kept
/// FP32), `alpha * residual + beta * h`, rounded to BF16.
pub fn residual_gate(residual: &[f32], hidden: &[f32], gate: &[f32], temperature: f32) -> Vec<f32> {
    residual.iter().zip(hidden).zip(gate).map(|((r, h), g)| {
        let w = g / temperature;
        let beta = 1.0 / (1.0 + (-w).exp());
        let alpha = ((1.0 / (1.0 + w.exp())) * (1.0 + beta)).max(1e-3).sqrt();
        bf16(alpha * r + beta * h)
    }).collect()
}

/// `MuseSparkTopkRouter`: FP32 logits times `output_multiplier`, sigmoid; the
/// bias only selects; weights renormalized by `sum + 1e-15`, times `scale`.
/// Output in descending biased-score order, ties to the lower id (the native
/// selector's order; `torch.topk(sorted=False)` order is arbitrary).
pub fn route(logits: &[f32], bias: &[f32], topk: usize, multiplier: f32, scale: f32) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = logits.iter().map(|x| 1.0 / (1.0 + (-(x * multiplier)).exp())).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])).then(a.cmp(b)));
    order.truncate(topk);
    let total: f32 = order.iter().map(|e| scores[*e]).sum::<f32>() + 1e-15;
    order.into_iter().map(|e| (e, scores[e] / total * scale)).collect()
}

/// The head: `softcap * tanh(logits * multiplier / softcap)` on F32 logits.
pub fn softcap(logit: f32, multiplier: f32, softcap: f32) -> f32 {
    ((logit * multiplier) / softcap).tanh() * softcap
}

/// One routed expert at the expert width: `down(silu(gate) * up)` in BF16
/// linears, then the input-scaled post-expert norm (before the routing weight).
pub fn expert(x: &[f32], gate_up: &[f32], down: &[f32], intermediate: usize, norm: &[f32], eps: f64) -> Vec<f32> {
    let hidden = x.len();
    let linear = |w: &[f32], n: usize, k: usize, v: &[f32]| -> Vec<f32> {
        (0..n).map(|i| bf16(w[i * k..(i + 1) * k].iter().zip(v).map(|(a, b)| a * b).sum())).collect()
    };
    let gu = linear(gate_up, 2 * intermediate, hidden, x);
    let act: Vec<f32> = (0..intermediate).map(|i| {
        let g = gu[i];
        bf16(bf16(g / (1.0 + (-g).exp())) * gu[intermediate + i])
    }).collect();
    input_scaled(&linear(down, hidden, intermediate, &act), norm, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_and_input_scaled_round_one_plus_weight_in_bf16() {
        // 1 + 0.00390625 (2^-8) is exact in BF16; 1 + 0.001 rounds to 1.0.
        let w = [0.001f32, 0.00390625];
        let x = [1.0f32, 1.0];
        let c = centered(&x, &w, 0.0);
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], bf16(1.00390625));
        let s = input_scaled(&[2.0, 2.0], &[0.001, 0.001], 0.0);
        assert_eq!(s, vec![1.0, 1.0]);
    }

    #[test]
    fn router_selects_on_bias_and_weights_unbiased_scores() {
        let logits = [0.0f32, 1.0, -1.0, 2.0];
        let bias = [10.0f32, 0.0, 0.0, 0.0];
        let routes = route(&logits, &bias, 2, 0.5, 1.0);
        assert_eq!(routes[0].0, 0, "bias selects expert 0");
        assert_eq!(routes[1].0, 3);
        let s0 = 0.5f32; let s3 = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((routes[0].1 - s0 / (s0 + s3 + 1e-15)).abs() < 1e-7);
        assert!((routes.iter().map(|r| r.1).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn residual_gate_and_softcap_follow_the_module() {
        let out = residual_gate(&[1.0], &[1.0], &[0.0], 0.3);
        assert_eq!(out[0], bf16((0.75f32).sqrt() + 0.5));
        assert!((softcap(1000.0, 0.1768, 20.0) - 20.0).abs() < 1e-3);
        assert!((softcap(1.0, 0.5, 20.0) - 20.0 * (0.5f32 / 20.0).tanh()).abs() < 1e-7);
    }
}
