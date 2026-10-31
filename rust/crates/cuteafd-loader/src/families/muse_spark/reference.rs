//! CPU contracts for Muse Spark operations, transcribed from the official
//! `transformers` `modeling_muse_spark.py` (5.22.1, the shared reference
//! `python/reference/families/muse_spark/model.py` pins its hash). They are
//! the oracles native exports are compared against.
//!
//! Exactness rule (native exports against these functions; [`Tolerance`]):
//! - Discrete outcomes are exact: expert ids, their order, tie order (lower id).
//! - Elementwise rounding points are exact: every BF16 rounding is the
//!   module's (`tensor.to(bfloat16)`; a BF16 tensor op rounds its result), and
//!   every `(1 + w)` of a centered or input-scaled norm is a BF16 add (the
//!   module computes `1.0 + self.weight` on the BF16 parameter). Arithmetic is
//!   FP32, one rounding per operation, no fused multiply-add, in this order.
//! - Transcendental and reduction results (sigmoid, tanh, the inverse RMS from
//!   a sum of squares) lie within [`Tolerance`] of these functions: device
//!   transcendentals do not match libm bit for bit, nor does PyTorch's own
//!   reduction order. [`sum_squares`] fixes a canonical FP32 order that
//!   kernels should follow; outputs downstream of a transcendental or a
//!   reduction are compared in BF16 ulps.
//! - Strict row invariance: a row's result is identical regardless of the
//!   batch or prefill-unit size it runs in (gated by the engine, N=1 vs N).
//! - Parameter-only transcendentals (the residual gate's alpha/beta) are
//!   computed once at load by these functions and passed to kernels, so they
//!   are exact by construction.

/// Bounds for native outputs downstream of a transcendental or reduction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Max distance in BF16 ulps (of the contract's value) of a BF16 output.
    pub bf16_ulps: u32,
    /// Max relative distance of an FP32 output (router weights).
    pub f32_relative: f32,
}

impl Tolerance {
    /// Norm outputs (inverse RMS from a reduction): one BF16 ulp.
    pub const NORM: Self = Self { bf16_ulps: 1, f32_relative: 0.0 };
    /// Output gate and softcap (sigmoid / tanh of a BF16 operand): one BF16 ulp.
    pub const TRANSCENDENTAL: Self = Self { bf16_ulps: 1, f32_relative: 0.0 };
    /// Router weights (sigmoid then renormalization): 2 FP32 ulps relative.
    pub const ROUTER: Self = Self { bf16_ulps: 0, f32_relative: 2.0 * f32::EPSILON };

    /// Whether BF16-valued `actual` is within `bf16_ulps` of `expected`, elementwise.
    pub fn bf16_close(self, actual: &[f32], expected: &[f32]) -> bool {
        actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, e)| {
            let ulp = if *e == 0.0 { f32::from_bits(0x0080_0000) }
                else { f32::from_bits(e.abs().to_bits() & 0x7f80_0000) * 2f32.powi(-7) };
            (a - e).abs() <= ulp * self.bf16_ulps as f32
        })
    }

    /// Whether FP32 `actual` is within the relative bound of `expected`.
    pub fn f32_close(self, actual: &[f32], expected: &[f32]) -> bool {
        actual.len() == expected.len() && actual.iter().zip(expected)
            .all(|(a, e)| (a - e).abs() <= e.abs() * self.f32_relative)
    }
}

thread_local! {
    /// Test-only: run the contracts without BF16 rounding (the semantic check
    /// against the official model in FP32/FP64). Never set outside tests.
    static EXACT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with every BF16 rounding of these contracts disabled (test-only).
pub fn without_bf16_rounding<T>(f: impl FnOnce() -> T) -> T {
    EXACT.with(|exact| exact.set(true));
    let result = f();
    EXACT.with(|exact| exact.set(false));
    result
}

/// Round to BF16 (nearest even), as `tensor.to(torch.bfloat16)`.
pub fn bf16(value: f32) -> f32 {
    if value.is_nan() || EXACT.with(|exact| exact.get()) { return value; }
    let bits = value.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff_0000)
}

/// Warps of the canonical reduction (one 256-thread CTA per row).
pub const REDUCE_WARPS: usize = 8;

/// Canonical FP32 sum of squares of one row of width `n` (a multiple of 4):
/// columns form slabs of 128 (the last may be partial); slab `b` belongs to
/// warp `b % 8`; lane `l` owns columns `4l..4l+4` of each of its warp's slabs
/// (columns past `n` contribute nothing). Each thread adds its squares in
/// ascending slab then column order; each warp folds its lanes by the
/// butterfly `x += shfl_xor(x, 1, 2, 4, 8, 16)`; lane 0's warp totals add in
/// ascending warp order. Every product and sum rounds once (no FMA). A row of
/// at most 128 columns (a 64-wide head) is therefore one warp's butterfly.
pub fn sum_squares(x: &[f32]) -> f32 {
    let n = x.len();
    assert!(n % 4 == 0 && n > 0, "canonical reduction needs whole 4-column lane groups");
    let slabs = n.div_ceil(128);
    let mut warps = [[0f32; 32]; REDUCE_WARPS];
    for (warp, lanes) in warps.iter_mut().enumerate() {
        for (lane, total) in lanes.iter_mut().enumerate() {
            for slab in (warp..slabs).step_by(REDUCE_WARPS) {
                for j in 0..4 {
                    if let Some(&v) = x.get(slab * 128 + lane * 4 + j) {
                        *total += v * v;
                    }
                }
            }
        }
        for stage in 0..5 {
            let partner: [f32; 32] = std::array::from_fn(|lane| lanes[lane ^ (1 << stage)]);
            for (value, other) in lanes.iter_mut().zip(partner) { *value += other; }
        }
    }
    warps.iter().fold(0f32, |total, lanes| total + lanes[0])
}

/// `1 / sqrt(mean(x^2) + eps)` from the canonical sum, FP32, correctly rounded.
fn inv_rms(x: &[f32], eps: f64) -> f32 {
    1.0 / ((sum_squares(x) / x.len() as f32) + eps as f32).sqrt()
}

/// `MuseSparkRMSNorm(with_scale=False)`: `x * (mean(x^2) + eps)^-0.5` in FP32,
/// rounded to the input dtype (BF16). Embedding, q/k/o per head and the
/// post-attention norm.
pub fn rms(x: &[f32], eps: f64) -> Vec<f32> {
    let inv = inv_rms(x, eps);
    x.iter().map(|v| bf16(v * inv)).collect()
}

/// `MuseSparkRMSNorm(with_scale=True)` (the final norm): plain `w`, not `1 + w`.
pub fn rms_scaled(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let inv = inv_rms(x, eps);
    x.iter().zip(weight).map(|(v, w)| bf16((v * inv) * w)).collect()
}

/// `MuseSparkTextCenteredRMSNorm`: normalize in FP32, multiply by `bf16(1 + w)`, round.
/// Input, pre-feedforward and pre-expert norms.
pub fn centered(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let inv = inv_rms(x, eps);
    x.iter().zip(weight).map(|(v, w)| bf16((v * inv) * bf16(1.0 + w))).collect()
}

/// `MuseSparkTextInputScaledRMSNorm`: scale by `bf16(1 + w)` first, round to
/// BF16, then normalize in FP32 and round. Post-feedforward and post-expert norms.
pub fn input_scaled(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let scaled: Vec<f32> = x.iter().zip(weight).map(|(v, w)| bf16(v * bf16(1.0 + w))).collect();
    rms(&scaled, eps)
}

/// `MuseSparkResidualGate` coefficients of one gate parameter column (the
/// module is kept FP32): `w = g / T`, `beta = sigmoid(w)`,
/// `alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))`. They depend only on the
/// parameter, so engines compute them once at load with this function and
/// pass them to the residual kernel (no device transcendentals to match).
pub fn residual_gate_coefficients(gate: &[f32], temperature: f32) -> Vec<(f32, f32)> {
    gate.iter().map(|g| {
        let w = g / temperature;
        let beta = 1.0 / (1.0 + (-w).exp());
        let alpha = ((1.0 / (1.0 + w.exp())) * (1.0 + beta)).max(1e-3).sqrt();
        (alpha, beta)
    }).collect()
}

/// The gated residual on BF16 rows: `bf16(alpha * residual + beta * h)`, two
/// FP32 products then one FP32 add (no FMA), as the module's promoted ops.
pub fn residual_gate_apply(residual: &[f32], hidden: &[f32], coefficients: &[(f32, f32)]) -> Vec<f32> {
    residual.iter().zip(hidden).zip(coefficients).map(|((r, h), (alpha, beta))| {
        let (a, b) = (alpha * r, beta * h);
        bf16(a + b)
    }).collect()
}

/// [`residual_gate_coefficients`] then [`residual_gate_apply`].
pub fn residual_gate(residual: &[f32], hidden: &[f32], gate: &[f32], temperature: f32) -> Vec<f32> {
    residual_gate_apply(residual, hidden, &residual_gate_coefficients(gate, temperature))
}

/// `MuseSparkTopkRouter`: FP32 logits times `output_multiplier`, sigmoid; the
/// bias only selects; weights renormalized by `sum + 1e-15`, times `scale`.
/// Output in descending biased-score order, ties to the lower id (the native
/// selector's order; `torch.topk(sorted=False)` order is arbitrary). The
/// weight sum adds the selected scores in that order. Native routers gate on
/// the identical selected set and order, weights within 2 FP32 ulps (the
/// sigmoid's `exp` is not correctly rounded on either side).
pub fn route(logits: &[f32], bias: &[f32], topk: usize, multiplier: f32, scale: f32) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = logits.iter().map(|x| 1.0 / (1.0 + (-(x * multiplier)).exp())).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])).then(a.cmp(b)));
    order.truncate(topk);
    let total: f32 = order.iter().map(|e| scores[*e]).sum::<f32>() + 1e-15;
    order.into_iter().map(|e| (e, scores[e] / total * scale)).collect()
}

/// The head: the official `lm_head` is a BF16 linear, so its logit is BF16;
/// `logit` may be the F32 output of a native (K6) head, which this rounds to
/// BF16 first. Each of `* output_multiplier`, `/ softcap`, `tanh` and
/// `* softcap` is then a BF16 tensor op in the module
/// (`MuseSparkForConditionalGeneration.forward`), so each rounds to BF16; the
/// sampler receives the final BF16 value widened to F32. `tanh` is evaluated
/// in FP32 on the BF16 operand and rounded (PyTorch's BF16 tanh).
pub fn softcap(logit: f32, multiplier: f32, softcap: f32) -> f32 {
    let scaled = bf16(bf16(logit) * multiplier);
    let divided = bf16(scaled / softcap);
    bf16(bf16(divided.tanh()) * softcap)
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
        let w = [0.001f32, 0.00390625, 0.0, 0.0];
        let x = [1.0f32; 4];
        let c = centered(&x, &w, 0.0);
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], bf16(1.00390625));
        let s = input_scaled(&[2.0; 4], &[0.001; 4], 0.0);
        assert_eq!(s, vec![1.0; 4]);
    }

    #[test]
    fn tolerance_counts_bf16_ulps_of_the_contract_value() {
        let e = [1.0f32, -3.0, 0.0];
        assert!(Tolerance::NORM.bf16_close(&e, &e));
        // One BF16 ulp at 1.0 is 2^-7, at 3.0 is 2^-6.
        assert!(Tolerance::NORM.bf16_close(&[1.0078125, -3.015625, 0.0], &e));
        assert!(!Tolerance::NORM.bf16_close(&[1.015625, -3.0, 0.0], &e));
        assert!(Tolerance::ROUTER.f32_close(&[0.5 + f32::EPSILON / 2.0], &[0.5]));
        assert!(!Tolerance::ROUTER.f32_close(&[0.5 + 2.0 * f32::EPSILON], &[0.5]));
        assert!(!Tolerance::ROUTER.f32_close(&[0.5 + 1e-5], &[0.5]));
    }

    #[test]
    fn canonical_reduction_order_is_the_documented_one() {
        // 64 wide: one warp; lanes 0..16 hold four columns each, butterfly order.
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mut lanes = [0f32; 32];
        for lane in 0..16 { for j in 0..4 { lanes[lane] += x[lane * 4 + j] * x[lane * 4 + j]; } }
        for stage in 0..5 {
            let partner: [f32; 32] = std::array::from_fn(|l| lanes[l ^ (1 << stage)]);
            for (v, o) in lanes.iter_mut().zip(partner) { *v += o; }
        }
        assert_eq!(sum_squares(&x).to_bits(), lanes[0].to_bits());
        // 4096 wide: 32 slabs over 8 warps, four per warp (the route-norm kernel's map).
        let y: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.013).cos()).collect();
        let total = sum_squares(&y);
        assert!((total - y.iter().map(|v| v * v).sum::<f32>()).abs() < 1e-2);
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
        assert_eq!(softcap(1000.0, 0.1768, 20.0), 20.0);
        // Module order: each BF16 tensor op rounds (1.0*0.5 -> 0.5, /20 -> bf16(0.025), tanh, *20).
        let divided = bf16(0.5f32 / 20.0);
        assert_eq!(softcap(1.0, 0.5, 20.0), bf16(bf16(divided.tanh()) * 20.0));
        assert_ne!(divided, 0.025f32, "the divide rounds to BF16");
    }
}
