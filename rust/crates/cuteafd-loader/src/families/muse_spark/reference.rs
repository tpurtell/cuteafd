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

/// `muse_tp_combine` for one row under TP residency. `partials[route][rank]`
/// are each rank's dequantized F32 partial down-proj outputs (E wide) for the
/// row's routed experts in route order; `weights` the routing weights; `norm`
/// the layer's one `post_expert_norm` weight. Per route: sum the ranks in
/// ascending order in F32, round to BF16 (the official BF16 down linear's
/// output), apply the input-scaled norm, multiply by the routing weight (F32,
/// then BF16, as the module's BF16 tensor op), then sum the routes in route
/// order in F32 and round to BF16 (`index_add_` into BF16 `final_hidden_states`
/// accumulates in route order; each add rounds to BF16).
pub fn tp_combine(partials: &[Vec<Vec<f32>>], weights: &[f32], norm: &[f32], eps: f64) -> Vec<f32> {
    let width = norm.len();
    let mut out = vec![0f32; width];
    for (ranks, &weight) in partials.iter().zip(weights) {
        let summed: Vec<f32> = (0..width).map(|i| bf16(ranks.iter().fold(0f32, |acc, p| acc + p[i]))).collect();
        let normed = input_scaled(&summed, norm, eps);
        for i in 0..width {
            out[i] = bf16(out[i] + bf16(normed[i] * weight));
        }
    }
    out
}

/// E4M3 RNE (finite, saturating at 448; subnormal step 2^-9).
fn e4m3(v: f32) -> f32 {
    let a = v.abs().min(448.0);
    if a == 0.0 { return 0.0f32.copysign(v); }
    let exp = a.log2().floor().max(-6.0);
    let step = (exp - 3.0).exp2();
    ((a / step).round_ties_even() * step).min(448.0).copysign(v)
}

/// NVFP4 partial record (E2M1 values, one E4M3 scale per 16): the dequantized
/// values the Spark epilogue encodes and the combine decodes. Per block:
/// `scale = e4m3(amax / 6)` (RNE); an all-zero block (amax 0) takes scale code
/// 0x00 (+0.0) and values 0, dequantizing to exactly 0; a nonzero block whose
/// rounded scale is 0 takes the smallest subnormal 2^-9. Values
/// `e2m1(x / scale)` RNE onto {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating at 6
/// (ties to the even code: the lower grid value on an exact half).
pub fn nvfp4_roundtrip(x: &[f32]) -> Vec<f32> {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    x.chunks(16).flat_map(|block| {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 { return vec![0.0; block.len()]; }
        let scale = { let s = e4m3(amax / 6.0); if s == 0.0 { (-9f32).exp2() } else { s } };
        block.iter().map(|v| {
            let a = (v / scale).abs().min(6.0);
            let mut best = 0;
            for (i, g) in GRID.iter().enumerate() {
                let (d, db) = ((a - g).abs(), (a - GRID[best]).abs());
                if d < db || (d == db && i % 2 == 0 && best % 2 == 1) { best = i; }
            }
            (GRID[best] * scale).copysign(*v)
        }).collect()
    }).collect()
}

/// FP8 partial record (E4M3 values, one UE8M0 scale per 32): the dequantized
/// values. Per block: `scale` is the smallest power of two with
/// `amax / scale <= 448` (`2^ceil(log2(amax / 448))`, exponent clamped to
/// [-127, 127]); values `e4m3(x / scale)` RNE. An all-zero block (amax 0) takes
/// scale code 0 (2^-127) and values 0, dequantizing to exactly 0.
pub fn fp8_roundtrip(x: &[f32]) -> Vec<f32> {
    x.chunks(32).flat_map(|block| {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 { return vec![0.0; block.len()]; }
        let mut exp = (amax / 448.0).log2().ceil().clamp(-127.0, 127.0) as i32;
        // The smallest power of two: ceil can land one high when amax/448 is a power of two's neighbour.
        while exp > -127 && amax / (exp as f32 - 1.0).exp2() <= 448.0 { exp -= 1; }
        while amax / (exp as f32).exp2() > 448.0 { exp += 1; }
        let scale = (exp as f32).exp2();
        block.iter().map(|v| e4m3(v / scale) * scale).collect()
    }).collect()
}

/// The UE8M0 scale code of an FP8 block (exponent + 127; 0 for an all-zero block).
pub fn fp8_scale_code(block: &[f32]) -> u8 {
    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 { return 0; }
    let mut exp = (amax / 448.0).log2().ceil().clamp(-127.0, 127.0) as i32;
    while exp > -127 && amax / (exp as f32 - 1.0).exp2() <= 448.0 { exp -= 1; }
    while amax / (exp as f32).exp2() > 448.0 { exp += 1; }
    (exp + 127) as u8
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

    /// BF16 and NVFP4 partials take the same combine: identical partials give
    /// identical output, and a format's effect is only its dequantized values.
    #[test]
    fn tp_combine_rounds_the_rank_sum_once_and_is_format_neutral() {
        let width = 32;
        let norm: Vec<f32> = (0..width).map(|i| (i as f32 - 16.0) / 64.0).collect();
        let partial = |seed: u32| -> Vec<f32> {
            (0..width).map(|i| ((((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 - 500.0) / 997.0).collect()
        };
        let routes: Vec<Vec<Vec<f32>>> = (0..3).map(|r| (0..6).map(|k| partial(r * 7 + k)).collect()).collect();
        let weights = [0.5f32, 0.3, 0.2];
        let a = tp_combine(&routes, &weights, &norm, 1e-8);
        assert_eq!(a, tp_combine(&routes, &weights, &norm, 1e-8));
        // One rank holding the whole sum equals six ranks whose F32 sum is the same bits.
        let whole: Vec<Vec<Vec<f32>>> = routes.iter().map(|ranks| {
            vec![(0..width).map(|i| ranks.iter().fold(0f32, |acc, p| acc + p[i])).collect()]
        }).collect();
        assert_eq!(tp_combine(&whole, &weights, &norm, 1e-8), a);
        // The single-expert contract is the combine of one route with weight 1.
        let one = tp_combine(&routes[..1], &[1.0], &norm, 1e-8);
        let summed: Vec<f32> = (0..width).map(|i| bf16(routes[0].iter().fold(0f32, |acc, p| acc + p[i]))).collect();
        assert_eq!(one, input_scaled(&summed, &norm, 1e-8).iter().map(|v| bf16(bf16(*v) + 0.0)).collect::<Vec<_>>());
        // Round trips are idempotent and bounded by their format step.
        for (name, codec) in [("nvfp4", nvfp4_roundtrip as fn(&[f32]) -> Vec<f32>), ("fp8", fp8_roundtrip)] {
            let q = codec(&routes[0][0]);
            assert_eq!(codec(&q), q, "{name}");
            assert!(q.iter().zip(&routes[0][0]).all(|(a, b)| (a - b).abs() <= b.abs().max(1e-3)), "{name}");
            // Zero blocks dequantize to exactly zero, and identical partials combine identically.
            assert!(codec(&[0.0; 32]).iter().all(|v| v.to_bits() == 0), "{name}");
        }
        assert_eq!(fp8_scale_code(&[0.0; 32]), 0);
        // 448 fits scale 2^0 exactly; 449 needs 2^1; 1.75 (448 * 2^-8) needs 2^-8.
        assert_eq!(fp8_scale_code(&[448.0; 32]), 127);
        assert_eq!(fp8_scale_code(&[449.0; 32]), 128);
        assert_eq!(fp8_scale_code(&[1.75; 32]), 127 - 8);
        assert_eq!(fp8_roundtrip(&[449.0; 32])[0], 448.0);
    }
}
