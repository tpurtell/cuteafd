//! FFN arithmetic is part of the checkpoint contract, not a kernel tuning knob.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FfnActivation {
    /// Existing kernels: clamp gate before SiLU, round SiLU to BF16, multiply.
    SiluBf16Product { limit: Option<f32> },
    /// Step-5: clamp SiLU(gate) from above and up symmetrically in FP32,
    /// multiply in FP32, then round once to BF16.
    SiluPostClampFp32Product { limit: f32 },
    /// Kimi K3 (`hidden_act=situ`): in FP32,
    /// `beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)`,
    /// rounded once to BF16. `linear_beta` absent leaves `up` unbounded.
    Situ { beta: f32, linear_beta: Option<f32> },
}

impl FfnActivation {
    pub fn validate(self) -> Result<Self, &'static str> {
        let bad = |value: f32| !value.is_finite() || value <= 0.0;
        match self {
            Self::SiluBf16Product { limit } if limit.is_some_and(bad) =>
                Err("FFN activation limit must be finite and positive"),
            Self::SiluPostClampFp32Product { limit } if bad(limit) =>
                Err("FFN activation limit must be finite and positive"),
            Self::Situ { beta, linear_beta } if bad(beta) || linear_beta.is_some_and(bad) =>
                Err("Situ beta values must be finite and positive"),
            _ => Ok(self),
        }
    }

    /// CPU oracle with the same explicit rounding boundaries as the exports.
    /// Inputs are projection outputs (already rounded by their producer).
    pub fn evaluate(self, gate: f32, up: f32) -> f32 {
        match self {
            Self::SiluBf16Product { limit } => {
                let (gate, up) = limit.map_or((gate, up), |limit|
                    (gate.min(limit), up.clamp(-limit, limit)));
                bf16(bf16(gate / (1.0 + (-gate).exp())) * up)
            }
            Self::SiluPostClampFp32Product { limit } => {
                bf16((gate / (1.0 + (-gate).exp())).min(limit) * up.clamp(-limit, limit))
            }
            Self::Situ { beta, linear_beta } => {
                let gate = beta * (gate / beta).tanh() / (1.0 + (-gate).exp());
                let up = linear_beta.map_or(up, |b| b * (up / b).tanh());
                bf16(gate * up)
            }
        }
    }
}

fn bf16(value: f32) -> f32 {
    let bits = u64::from(value.to_bits());
    f32::from_bits((((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_arithmetic_stays_bit_identical() {
        let legacy_bf16 = |value: f32| {
            let bits = u64::from(value.to_bits());
            f32::from_bits((((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16) as u32)
        };
        for bits in [0, 1, 0x8000_0000, 0x3f80_8000, 0x3f81_8000,
            0x7f7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffff_ffff] {
            assert_eq!(bf16(f32::from_bits(bits)).to_bits(), legacy_bf16(f32::from_bits(bits)).to_bits());
        }
        for limit in [None, Some(10.0)] {
            let contract = FfnActivation::SiluBf16Product { limit };
            for g in -256..=256 {
                for u in [-19.0f32, -7.125, -0.01, 0.0, 0.01, 3.14, 19.0] {
                    let (mut gate, mut up) = (g as f32 / 16.0, u);
                    if let Some(limit) = limit { gate = gate.min(limit); up = up.clamp(-limit, limit); }
                    let expected = bf16(bf16(gate / (1.0 + (-gate).exp())) * up);
                    assert_eq!(contract.evaluate(g as f32 / 16.0, u).to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn post_clamp_and_single_rounding_are_distinct() {
        let step5 = FfnActivation::SiluPostClampFp32Product { limit: 7.0 };
        assert_eq!(step5.evaluate(8.0, 1.0), 7.0);
        assert_eq!(step5.evaluate(8.0, 9.0), 49.0);
        assert_eq!(step5.evaluate(8.0, -9.0), -49.0);
        // At limit 7, BF16 can hide the clamp-order difference for saturated
        // gates. A lower limit isolates clamp order independently of rounding.
        assert_ne!(FfnActivation::SiluPostClampFp32Product { limit: 2.0 }.evaluate(8.0, 1.0),
            FfnActivation::SiluBf16Product { limit: Some(2.0) }.evaluate(8.0, 1.0));
        assert!((-128..128).any(|g| (-128..128).any(|u| {
            let (gate, up) = (g as f32 / 32.0, u as f32 / 32.0);
            step5.evaluate(gate, up).to_bits() != FfnActivation::SiluBf16Product { limit: None }.evaluate(gate, up).to_bits()
        })));
    }

    #[test]
    fn invalid_limits_fail_closed() {
        for limit in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(FfnActivation::SiluPostClampFp32Product { limit }.validate().is_err());
            assert!(FfnActivation::Situ { beta: limit, linear_beta: None }.validate().is_err());
            assert!(FfnActivation::Situ { beta: 4.0, linear_beta: Some(limit) }.validate().is_err());
        }
        assert!(FfnActivation::Situ { beta: 4.0, linear_beta: Some(25.0) }.validate().is_ok());
    }

    #[test]
    fn situ_matches_reference_formula() {
        let situ = FfnActivation::Situ { beta: 4.0, linear_beta: Some(25.0) };
        // Values from the pinned reference SituAndMul, evaluated in FP64 then
        // rounded once to BF16 (FP32 intermediates agree at BF16 output).
        let reference = |a: f64, b: f64| {
            let v = 4.0 * (a / 4.0).tanh() / (1.0 + (-a).exp()) * 25.0 * (b / 25.0).tanh();
            bf16(v as f32)
        };
        for a in [-40.0, -6.0, -1.0, -0.125, 0.0, 0.5, 3.0, 9.0, 80.0] {
            for b in [-300.0, -30.0, -1.5, 0.0, 0.75, 12.0, 60.0, 1000.0] {
                assert_eq!(situ.evaluate(a as f32, b as f32).to_bits(), reference(a, b).to_bits(), "a={a} b={b}");
            }
        }
        // Both arms saturate: the gate at beta, up at linear_beta.
        assert_eq!(situ.evaluate(80.0, 1000.0), bf16(100.0));
        // Distinct from plain SiLU product.
        assert_ne!(situ.evaluate(9.0, 60.0), FfnActivation::SiluBf16Product { limit: None }.evaluate(9.0, 60.0));
    }
}
