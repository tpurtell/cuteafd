pub mod deepseek;
pub mod dflash;
pub mod dspark;
pub mod glm;
pub mod kimi;
pub mod mimo;
pub mod qwen;
pub mod qwen35;

use crate::plan::format::{Encoding, QuantOperand, ScaleEncoding};

/// `Ok` when `ok`, else the reason.
pub(crate) fn require(ok: bool, reason: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(reason()) }
}

/// The operand as a contract message names it: label and logical shape.
pub(crate) fn describe(operand: &QuantOperand) -> String {
    format!("{} {:?}", operand.label(), operand.logical)
}

/// BF16 only.
pub(crate) fn bf16(operand: &QuantOperand, what: &str) -> Result<(), String> {
    require(operand.is_plain(&[Encoding::Bf16]), || format!("{what} must be BF16, found {}", describe(operand)))
}

/// BF16 or FP32.
pub(crate) fn bf16_or_f32(operand: &QuantOperand, what: &str) -> Result<(), String> {
    require(operand.is_plain(&[Encoding::Bf16, Encoding::F32]), || {
        format!("{what} must be BF16 or FP32, found {}", describe(operand))
    })
}

/// FP8 E4M3 with FP32 128x128 block scales: the HF block-FP8 layout GLM and
/// MiMo stage (`fp8_block_dequant` over FP32 grids).
pub(crate) fn fp8_f32_block128(operand: &QuantOperand) -> bool {
    operand.is_fp8_block(128, &[ScaleEncoding::F32])
}

/// BF16, or E4M3 with FP32 128x128 block scales.
pub(crate) fn bf16_or_fp8_block128(operand: &QuantOperand, what: &str) -> Result<(), String> {
    require(operand.is_plain(&[Encoding::Bf16]) || fp8_f32_block128(operand), || {
        format!("{what} must be BF16 or E4M3 with FP32 128x128 block scales, found {}", describe(operand))
    })
}

/// The `.weight` projection name at the end of `stem` (`q_proj` of
/// `model.layers.3.self_attn.q_proj`).
pub(crate) fn leaf(stem: &str) -> &str {
    stem.rsplit_once('.').map_or(stem, |(_, leaf)| leaf)
}

pub mod step5;
