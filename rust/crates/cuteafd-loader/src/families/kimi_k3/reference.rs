//! CPU contracts for Kimi K3 operations that have no existing shared kernel,
//! transcribed from the checkpoint's bundled `modeling_kimi_linear.py`. They
//! are the oracles kernel exports are compared against: explicit rounding
//! points, FP64 accumulation where the reference reduces in FP32 (kernel
//! comparisons then bound reduction-order differences, not semantics).

/// Round to BF16 (nearest even), as `tensor.to(torch.bfloat16)`.
pub fn bf16(value: f32) -> f32 {
    if value.is_nan() { return value; }
    let bits = value.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff_0000)
}

/// The release's zero-padded `A_log` narrowing lives with the shared KDA packer.
pub use crate::kda::narrow_a_log;

/// `KimiRMSNorm`: normalize in FP32, round to BF16, then multiply by the BF16
/// weight in BF16 (the product is rounded again).
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let mean = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64;
    let scale = (1.0 / (mean + eps).sqrt()) as f32;
    x.iter().zip(weight).map(|(v, w)| bf16(w * bf16(v * scale))).collect()
}

/// `_apply_attn_res`: a softmax mixture over stored block residuals plus the
/// current prefix sum. Scores are FP32 dots of the RMS-normalized candidate
/// with `norm.weight * proj.weight`; the values mixed are the original
/// candidates; the result is rounded to BF16. `candidates` are BF16 rows,
/// stored blocks first, prefix sum last.
pub fn residual_mix(candidates: &[&[f32]], norm: &[f32], proj: &[f32], eps: f64) -> Vec<f32> {
    let width = norm.len();
    let score_weight: Vec<f32> = norm.iter().zip(proj).map(|(n, p)| n * p).collect();
    let scores: Vec<f64> = candidates.iter().map(|v| {
        debug_assert_eq!(v.len(), width);
        let mean = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / width as f64;
        let inv = 1.0 / (mean + eps).sqrt();
        v.iter().zip(&score_weight).map(|(x, s)| *x as f64 * inv * *s as f64).sum()
    }).collect();
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = exp.iter().sum();
    (0..width).map(|i| bf16(candidates.iter().zip(&exp).map(|(v, e)| v[i] as f64 * e / total).sum::<f64>() as f32)).collect()
}

/// `KimiMoEGate`: FP32 logits, sigmoid scores, selection on `score + bias`,
/// weights from the unbiased scores renormalized by `sum + 1e-20`, times
/// `scale`. Ties in the biased score choose the lower expert id (torch.topk
/// leaves tie order unspecified; kernels must use this rule).
pub fn route(logits: &[f32], bias: &[f32], topk: usize, scale: f32) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = logits.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])).then(a.cmp(b)));
    order.truncate(topk);
    let total: f32 = order.iter().map(|e| scores[*e]).sum::<f32>() + 1e-20;
    order.into_iter().map(|e| (e, scores[e] / total * scale)).collect()
}

/// The routed half of `KimiSparseMoeBlock` after the latent down projection:
/// each selected expert's BF16 output is weighted and summed in FP32, rounded
/// to BF16, then `routed_expert_norm` (over the full sum, never per rank).
/// The caller then applies `routed_expert_up_proj` and adds the shared FFN.
pub fn latent_combine(outputs: &[(f32, &[f32])], norm: &[f32], eps: f64) -> Vec<f32> {
    let width = norm.len();
    let summed: Vec<f32> = (0..width).map(|i| bf16(outputs.iter().map(|(w, y)| w * y[i]).sum::<f32>())).collect();
    rms_norm(&summed, norm, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }

    #[test]
    fn a_log_narrows_only_exact_zero_padding() {
        let mut raw: Vec<f32> = (0..96).map(|h| (h as f32 + 1.0).ln()).collect();
        raw.extend([0.0f32; 32]);
        assert_eq!(narrow_a_log(&f32_bytes(&raw), 96).unwrap(), raw[..96].to_vec());
        for bad in [-0.0f32, f32::NAN, 1e-30] {
            let mut tail = raw.clone();
            tail[127] = bad;
            assert!(narrow_a_log(&f32_bytes(&tail), 96).is_err(), "tail {bad}");
        }
        assert!(narrow_a_log(&f32_bytes(&raw[..90]), 96).is_err());
    }

    #[test]
    fn residual_mix_is_a_convex_combination_of_original_values() {
        let a = vec![1.0f32, -2.0, 0.5, 4.0];
        let b = vec![bf16(0.25), 3.0, -1.0, bf16(0.1)];
        let norm = vec![1.0f32; 4];
        // Zero projection: uniform weights.
        let mixed = residual_mix(&[&a, &b], &norm, &[0.0; 4], 1e-5);
        for i in 0..4 { assert_eq!(mixed[i], bf16((a[i] + b[i]) / 2.0)); }
        // A single candidate returns itself (layer 0 has no stored blocks; the
        // reference skips the mix there, which this also agrees with).
        assert_eq!(residual_mix(&[&a], &norm, &[3.0; 4], 1e-5), a);
        // Scale-invariant scores: doubling a candidate does not change its weight.
        let a2: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        let proj = [0.3f32, -0.1, 0.7, 0.2];
        let w = |v: &[f32]| { let m = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / 4.0;
            v.iter().zip(&proj).map(|(x, p)| *x as f64 / (m + 1e-5).sqrt() * *p as f64).sum::<f64>() };
        assert!((w(&a) - w(&a2)).abs() < 1e-6);
    }

    #[test]
    fn routing_selects_on_bias_but_weights_without_it() {
        let logits = [0.0f32, 2.0, -1.0, 1.0];
        let bias = [5.0f32, 0.0, 0.0, 0.0];
        let routes = route(&logits, &bias, 2, 1.0);
        assert_eq!(routes.iter().map(|r| r.0).collect::<Vec<_>>(), vec![0, 1]);
        let s = |x: f32| 1.0 / (1.0 + (-x).exp());
        assert!((routes[0].1 - s(0.0) / (s(0.0) + s(2.0))).abs() < 1e-7);
        assert!((routes.iter().map(|r| r.1).sum::<f32>() - 1.0).abs() < 1e-6);
        // Ties choose the lower id.
        assert_eq!(route(&[1.0, 1.0, 1.0], &[0.0; 3], 2, 1.0).iter().map(|r| r.0).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn latent_norm_follows_the_complete_sum() {
        let (y0, y1) = (vec![1.0f32, 2.0, 3.0, 4.0], vec![-1.0f32, 0.5, 2.0, 0.0]);
        let norm = vec![1.0f32; 4];
        let whole = latent_combine(&[(0.75, &y0), (0.25, &y1)], &norm, 1e-6);
        // Normalizing each partial and summing is a different model.
        let partial: Vec<f32> = latent_combine(&[(0.75, &y0)], &norm, 1e-6).iter()
            .zip(latent_combine(&[(0.25, &y1)], &norm, 1e-6)).map(|(a, b)| a + b).collect();
        assert_ne!(whole, partial);
        let rms = (whole.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-2);
    }
}
