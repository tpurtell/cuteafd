//! CPU contracts/oracles for dense coordinator operations, not GPU dispatch.
//! Reduction order here is scalar; native parallel reductions need independent
//! qualification. Rounding boundaries and selection-bias order are explicit.
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
#[error("invalid dense attention contract: {0}")]
pub struct ContractError(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputPrecision {
    F32,
    Bf16,
}

impl OutputPrecision {
    pub fn round(self, value: f32) -> f32 {
        match self {
            Self::F32 => value,
            Self::Bf16 => {
                let bits = value.to_bits();
                if bits & 0x7fff_ffff > 0x7f80_0000 {
                    return f32::from_bits((bits & 0xffff_0000) | 0x0040_0000);
                }
                f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormWeight {
    Conventional,
    ZeroCentered,
}

pub fn rms_norm(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    kind: NormWeight,
    output: OutputPrecision,
) -> Result<Vec<f32>, ContractError> {
    if x.is_empty() || x.len() != weight.len() || !eps.is_finite() || eps <= 0.0 {
        return Err(ContractError("norm dimensions or epsilon"));
    }
    let variance = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (variance + eps).sqrt().recip();
    Ok(x.iter()
        .zip(weight)
        .map(|(&v, &w)| {
            let scale = match kind {
                NormWeight::Conventional => w,
                NormWeight::ZeroCentered => w + 1.0,
            };
            output.round((v * inv) * scale)
        })
        .collect())
}

/// Each delta has already been rounded by its producer. Deltas are added in
/// the supplied order; combine parallel FFN branches before residual addition.
pub fn residual_add(
    residual: &[f32],
    deltas: &[&[f32]],
    output: OutputPrecision,
) -> Result<Vec<f32>, ContractError> {
    if residual.is_empty() || deltas.iter().any(|d| d.len() != residual.len()) {
        return Err(ContractError("residual dimensions"));
    }
    Ok(residual
        .iter()
        .enumerate()
        .map(|(i, &r)| output.round(deltas.iter().fold(r, |sum, d| sum + d[i])))
        .collect())
}

/// FP32 accumulator oracle over row-major weights. Projection input/weights
/// must already have checkpoint precision; output rounding belongs to the op.
pub fn linear(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    output: OutputPrecision,
) -> Result<Vec<f32>, ContractError> {
    if x.is_empty() || rows == 0 || rows.checked_mul(x.len()) != Some(weight.len()) {
        return Err(ContractError("linear dimensions"));
    }
    Ok(weight
        .chunks_exact(x.len())
        .map(|w| output.round(x.iter().zip(w).fold(0.0f32, |sum, (x, w)| sum + x * w)))
        .collect())
}

/// Widen independently rounded routed/shared outputs, then add in FP32.
/// The resulting contribution is reduced across TP before residual addition.
pub fn ffn_branches(routed: &[f32], shared: &[f32]) -> Result<Vec<f32>, ContractError> {
    if routed.is_empty() || routed.len() != shared.len() {
        return Err(ContractError("FFN branch dimensions"));
    }
    Ok(routed.iter().zip(shared).map(|(r, s)| r + s).collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub qk_dim: usize,
    pub value_dim: usize,
    pub rotary_dim: usize,
}

impl GqaGeometry {
    pub fn validate(self) -> Result<Self, ContractError> {
        if self.query_heads == 0
            || self.kv_heads == 0
            || self.qk_dim == 0
            || self.value_dim == 0
            || self.query_heads % self.kv_heads != 0
            || self.rotary_dim == 0
            || self.rotary_dim % 2 != 0
            || self.rotary_dim > self.qk_dim
        {
            return Err(ContractError("GQA geometry"));
        }
        self.query_heads
            .checked_mul(self.qk_dim)
            .and_then(|_| self.query_heads.checked_mul(self.value_dim))
            .and_then(|_| self.kv_heads.checked_mul(self.qk_dim))
            .and_then(|_| self.kv_heads.checked_mul(self.value_dim))
            .ok_or(ContractError("GQA extent overflow"))?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryArithmetic {
    /// Deployed Step-5: real uses separate multiplies/subtract; imaginary
    /// uses fma(real, sin, rounded(imaginary * cos)).
    FusedImaginary,
    /// Labelled model.py CPU convenience fallback, not native parity.
    UnfusedFallback,
}

impl RotaryArithmetic {
    /// NeoX half pairing over only the leading rotary span. Inputs/cache
    /// values already have producer/storage precision (BF16 for Step-5).
    pub fn evaluate(
        self,
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        output: OutputPrecision,
    ) -> Result<Vec<f32>, ContractError> {
        let half = cos.len();
        if half == 0 || sin.len() != half || half > x.len() / 2 {
            return Err(ContractError("rotary dimensions"));
        }
        let mut result = x.to_vec();
        for i in 0..half {
            result[i] = output.round(x[i] * cos[i] - x[i + half] * sin[i]);
            let imaginary_cos = x[i + half] * cos[i];
            result[i + half] = output.round(match self {
                Self::FusedImaginary => x[i].mul_add(sin[i], imaginary_cos),
                Self::UnfusedFallback => x[i] * sin[i] + imaginary_cos,
            });
        }
        Ok(result)
    }
}

pub fn sigmoid_head_gate(
    attention: &[f32],
    logits: &[f32],
    value_dim: usize,
    output: OutputPrecision,
) -> Result<Vec<f32>, ContractError> {
    if value_dim == 0
        || logits.is_empty()
        || logits.len().checked_mul(value_dim) != Some(attention.len())
    {
        return Err(ContractError("head gate dimensions"));
    }
    Ok(attention
        .chunks_exact(value_dim)
        .zip(logits)
        .flat_map(|(head, &logit)| {
            let gate = output.round(1.0 / (1.0 + (-logit).exp()));
            head.iter().map(move |&value| output.round(value * gate))
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterBiasOrder {
    /// Existing shared router: selection uses corrected scores, returned
    /// weights gather original probabilities without subtract-after-add.
    GatherOriginal,
    /// Official Step-5: retain rounding from corrected_score - selected_bias.
    SubtractSelected,
}

#[derive(Debug, Clone, Copy)]
pub struct SigmoidRouter {
    pub topk: usize,
    pub factor: f32,
    pub epsilon: f32,
    pub bias_order: RouterBiasOrder,
}

impl SigmoidRouter {
    /// Ties use ascending expert id for determinism, not a torch.topk parity
    /// claim. Callers comparing official selected ids must exclude score ties.
    pub fn evaluate(
        self,
        logits: &[f32],
        bias: &[f32],
    ) -> Result<Vec<(usize, f32)>, ContractError> {
        if self.topk == 0
            || self.topk > logits.len()
            || logits.len() != bias.len()
            || !self.factor.is_finite()
            || self.factor <= 0.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || logits.iter().chain(bias).any(|v| !v.is_finite())
        {
            return Err(ContractError("router dimensions or values"));
        }
        let probabilities: Vec<_> = logits.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let scores: Vec<_> = probabilities.iter().zip(bias).map(|(p, b)| p + b).collect();
        if scores.iter().any(|s| !s.is_finite()) {
            return Err(ContractError("router score overflow"));
        }
        let mut ids: Vec<_> = (0..logits.len()).collect();
        ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let mut selected: Vec<_> = ids[..self.topk]
            .iter()
            .map(|&id| {
                (
                    id,
                    match self.bias_order {
                        RouterBiasOrder::GatherOriginal => probabilities[id],
                        RouterBiasOrder::SubtractSelected => scores[id] - bias[id],
                    },
                )
            })
            .collect();
        let denominator = selected.iter().map(|(_, p)| p).sum::<f32>() + self.epsilon;
        for (_, weight) in &mut selected {
            *weight = (*weight / denominator) * self.factor;
        }
        Ok(selected)
    }
}

/// Address/state oracle. It plans writes; it does not own or modify KV. Verify
/// writes require separate storage until commit to preserve committed history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingWindow {
    pub window: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingWrite {
    pub source_start: u64,
    pub absolute_start: u64,
    pub count: u64,
    window: u64,
}

impl SlidingWindow {
    pub fn write(self, start: u64, count: u64) -> Result<RingWrite, ContractError> {
        if self.window == 0 {
            return Err(ContractError("zero sliding window"));
        }
        start
            .checked_add(count)
            .ok_or(ContractError("ring position overflow"))?;
        let keep = count.min(self.window);
        Ok(RingWrite {
            source_start: count - keep,
            absolute_start: start + count - keep,
            count: keep,
            window: self.window,
        })
    }

    /// Causal window at a query position, independent of ring wrap.
    pub fn visible(self, query: u64, key: u64) -> Result<bool, ContractError> {
        if self.window == 0 {
            return Err(ContractError("zero sliding window"));
        }
        Ok(key <= query && query - key < self.window)
    }
}

impl RingWrite {
    pub fn slots(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.count).map(|i| {
            (
                self.source_start + i,
                (self.absolute_start + i) % self.window,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BF: OutputPrecision = OutputPrecision::Bf16;

    #[test]
    fn linear_and_parallel_ffn_keep_projection_and_residual_boundaries() {
        let x = [BF.round(1.1), BF.round(-2.3)];
        let w = [BF.round(0.3), BF.round(0.7)];
        let f32_out = linear(&x, &w, 1, OutputPrecision::F32).unwrap();
        let bf_out = linear(&x, &w, 1, BF).unwrap();
        assert_eq!(bf_out[0], BF.round(f32_out[0]));
        assert_eq!(ffn_branches(&[256.0], &[1.0]).unwrap(), [257.0]);
        // Summing the branches first is distinguishable from folding them
        // separately into a large FP32 residual.
        let delta = ffn_branches(&[1.0], &[1.0]).unwrap();
        assert_eq!(
            residual_add(&[16_777_216.0], &[&delta], OutputPrecision::F32).unwrap(),
            [16_777_218.0]
        );
        assert!(linear(&x, &w, 2, BF).is_err());
    }

    #[test]
    fn norm_weight_and_residual_precision_are_independent() {
        let x = [2.0, -3.0];
        let w = [0.0, 0.0];
        assert_eq!(
            rms_norm(&x, &w, 1e-5, NormWeight::Conventional, BF).unwrap(),
            vec![0.0, -0.0]
        );
        let zero = rms_norm(&x, &w, 1e-5, NormWeight::ZeroCentered, OutputPrecision::F32).unwrap();
        assert_eq!(
            zero[0].to_bits(),
            (2.0f32 * (6.5f32 + 1e-5).sqrt().recip()).to_bits()
        );
        let r = residual_add(&[256.0], &[&[0.5], &[0.5]], OutputPrecision::F32).unwrap();
        assert_eq!(r, [257.0]);
        assert_ne!(r, residual_add(&[256.0], &[&[0.5], &[0.5]], BF).unwrap());
        assert!(rms_norm(&[], &[], 1e-5, NormWeight::ZeroCentered, BF).is_err());
        assert!(residual_add(&[1.0], &[&[]], BF).is_err());
    }

    #[test]
    fn rotary_keeps_tail_and_rounds_once_after_fp32_arithmetic() {
        let x = [1.0, 2.0, 3.0, 4.0, -0.0];
        let out = RotaryArithmetic::FusedImaginary
            .evaluate(&x, &[0.0, 1.0], &[1.0, 0.0], BF)
            .unwrap();
        assert_eq!(&out[..4], &[-3.0, 2.0, 1.0, 4.0]);
        assert_eq!(out[4].to_bits(), x[4].to_bits());
        let x = [BF.round(0.371), BF.round(-0.913)];
        let c = BF.round(0.731);
        let s = BF.round(0.693);
        assert_eq!(
            RotaryArithmetic::FusedImaginary
                .evaluate(&x, &[c], &[s], BF)
                .unwrap(),
            [BF.round(x[0] * c - x[1] * s), BF.round(x[0] * s + x[1] * c)]
        );
        for (value_dim, rotary_dim) in [(128, 64), (192, 64), (192, 192)] {
            GqaGeometry {
                query_heads: 64,
                kv_heads: 4,
                qk_dim: 192,
                value_dim,
                rotary_dim,
            }
            .validate()
            .unwrap();
        }
        assert!(RotaryArithmetic::FusedImaginary
            .evaluate(&x, &[1.0, 2.0], &[1.0, 2.0], BF)
            .is_err());
    }

    #[test]
    fn deployed_rotary_contract_does_not_silently_use_cpu_fallback() {
        let x = [f32::from_bits(0x3f80_0001), -1.0];
        let cos = [1.0];
        let sin = [f32::from_bits(0x3f7f_fffe)];
        let native = RotaryArithmetic::FusedImaginary
            .evaluate(&x, &cos, &sin, OutputPrecision::F32)
            .unwrap();
        let fallback = RotaryArithmetic::UnfusedFallback
            .evaluate(&x, &cos, &sin, OutputPrecision::F32)
            .unwrap();
        assert_eq!(native[0].to_bits(), fallback[0].to_bits());
        assert_eq!(native[1].to_bits(), 0xa880_0000);
        assert_eq!(fallback[1].to_bits(), 0);
        // Independent fixed BF16 vectors (FP32-built, BF16-stored caches).
        let x = [f32::from_bits(0x3ebe_0000), f32::from_bits(0xbf6a_0000)];
        let out = RotaryArithmetic::FusedImaginary
            .evaluate(
                &x,
                &[f32::from_bits(0x3f3b_0000)],
                &[f32::from_bits(0x3f31_0000)],
                BF,
            )
            .unwrap();
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0x3f67_0000, 0xbed2_0000]
        );
    }

    #[test]
    fn head_gate_rounds_sigmoid_before_multiplication() {
        let logit: f32 = 0.03125;
        let value: f32 = 1.234375;
        let gate = BF.round(1.0 / (1.0 + (-logit).exp()));
        assert_eq!(
            sigmoid_head_gate(&[value, value, 2.0, 2.0], &[logit, 0.0], 2, BF).unwrap(),
            [BF.round(value * gate), BF.round(value * gate), 1.0, 1.0]
        );
    }

    #[test]
    fn router_bias_order_is_observable_and_legacy_is_not_rewritten() {
        let router = SigmoidRouter {
            topk: 2,
            factor: 3.0,
            epsilon: 1e-20,
            bias_order: RouterBiasOrder::GatherOriginal,
        };
        let logits = [-1.0, 1.0, 0.0];
        let bias = [16_777_216.0, 0.0, -1.0];
        let legacy = router.evaluate(&logits, &bias).unwrap();
        let step = SigmoidRouter {
            bias_order: RouterBiasOrder::SubtractSelected,
            ..router
        }
        .evaluate(&logits, &bias)
        .unwrap();
        assert_eq!(legacy.iter().map(|v| v.0).collect::<Vec<_>>(), [0, 1]);
        let p = 1.0 / (1.0 + 1.0f32.exp());
        let q = 1.0 / (1.0 + (-1.0f32).exp());
        assert_eq!(
            legacy[0].1.to_bits(),
            ((p / (p + q + 1e-20)) * 3.0).to_bits()
        );
        assert_eq!(step[0].1, 0.0);
        assert_ne!(legacy[0].1, step[0].1);
        assert!(router.evaluate(&[f32::NAN, 0.0], &[0.0, 0.0]).is_err());
    }

    #[test]
    fn ring_wrap_large_append_verify_commit_and_snapshot_restore() {
        for window in [128, 256, 512] {
            let ring = SlidingWindow { window };
            let apply = |cache: &mut Vec<u64>, start, count| {
                for (source, slot) in ring.write(start, count).unwrap().slots() {
                    cache[slot as usize] = start + source;
                }
            };
            let mut chunked = vec![u64::MAX; window as usize];
            let mut stepped = chunked.clone();
            apply(&mut chunked, 0, window * 3 + 7);
            for position in 0..window * 3 + 7 {
                apply(&mut stepped, position, 1);
            }
            assert_eq!(chunked, stepped);
            let saved = stepped.clone();
            // Verify stays private; rejected proposals never mutate committed KV.
            let mut verify = saved.clone();
            apply(&mut verify, window * 3 + 7, 9);
            assert_eq!(stepped, saved);
            apply(&mut stepped, window * 3 + 7, 3);
            let mut fresh = vec![u64::MAX; window as usize];
            apply(&mut fresh, 0, window * 3 + 10);
            assert_eq!(stepped, fresh);
            stepped.copy_from_slice(&saved);
            assert_eq!(stepped, chunked);
            assert!(ring.visible(window * 3, window * 2 + 1).unwrap());
            assert!(!ring.visible(window * 3, window * 2).unwrap());
            assert!(!ring.visible(7, 8).unwrap());
        }
        assert!(SlidingWindow { window: 0 }.write(0, 1).is_err());
        assert!(SlidingWindow { window: 512 }.write(u64::MAX, 1).is_err());
        assert_eq!(BF.round(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(BF.round(f32::NAN).is_nan());
    }
}
