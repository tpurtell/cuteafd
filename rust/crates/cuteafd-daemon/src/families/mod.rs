//! One directory per model family (long ids; short tags stay in C symbols,
//! AOT program prefixes and package directories).

pub(crate) mod deepseek_v4;
pub(crate) mod deepseek_v41;
pub(crate) mod glm5;
pub(crate) mod glm5_flash;
pub(crate) mod kimi_k3;
pub(crate) mod mimo_v2;
pub(crate) mod qwen4;
pub(crate) mod qwen3_5;
