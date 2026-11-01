//! Served context and output limits, shared by HTTP conversion and tokenized admission.
use anyhow::{ensure, Result};

pub const MAX_CONTEXT_TOKENS: u32 = 1_048_576;
/// Default `--max-output-tokens` in every family (TJ 2026-11-01, issue #951): 128K.
pub const DEFAULT_MAX_OUTPUT_TOKENS: i64 = 131_072;

#[derive(Clone, Copy, Debug)]
pub struct NativeLimits {
    context: u32,
    /// `None`: no output cap beyond the context (`--max-output-tokens` ≤ 0).
    output: Option<u32>,
}

impl Default for NativeLimits {
    fn default() -> Self {
        Self::new(MAX_CONTEXT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS).expect("default limits are valid")
    }
}

impl NativeLimits {
    /// `output` ≤ 0 leaves output bounded only by the context left after the prompt.
    pub fn new(context: u32, output: i64) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT_TOKENS).contains(&context),
            "context limit must be 1..={MAX_CONTEXT_TOKENS}"
        );
        let output = (output > 0).then(|| u32::try_from(output).unwrap_or(u32::MAX).min(context));
        Ok(Self { context, output })
    }

    pub fn context(self) -> u32 {
        self.context
    }

    /// The largest output a request may ask for (the context when uncapped).
    pub fn output(self) -> u32 {
        self.output.unwrap_or(self.context)
    }

    pub fn requested_output(self, requested: Option<u32>) -> Result<usize> {
        let requested = requested.unwrap_or(self.output());
        ensure!(requested > 0, "max_tokens must be positive");
        Ok(requested.min(self.output()) as usize)
    }

    pub fn output_for_prompt(self, prompt_tokens: usize, requested: usize) -> Result<usize> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens < self.context as usize,
            "native prompt must contain 1..{} tokens to leave room for output",
            self.context
        );
        ensure!(requested > 0, "max_tokens must be positive");
        Ok(requested
            .min(self.output() as usize)
            .min(self.context as usize - prompt_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_defaults_to_128k_and_nonpositive_means_context_minus_prompt() {
        let limits = NativeLimits::default();
        assert_eq!(limits.requested_output(None).unwrap(), 131_072);
        assert_eq!(limits.requested_output(Some(u32::MAX)).unwrap(), 131_072);
        assert_eq!(limits.output_for_prompt(1_048_575, usize::MAX).unwrap(), 1);
        assert_eq!(limits.output_for_prompt(1, usize::MAX).unwrap(), 131_072);
        for prompt in [0, 1_048_576, usize::MAX] {
            assert!(limits.output_for_prompt(prompt, 1).is_err());
        }
        for uncapped in [0, -1] {
            let limits = NativeLimits::new(1 << 20, uncapped).unwrap();
            assert_eq!(limits.output(), 1 << 20);
            assert_eq!(limits.requested_output(None).unwrap(), 1 << 20);
            assert_eq!(limits.output_for_prompt(1000, usize::MAX).unwrap(), (1 << 20) - 1000);
        }
        let small = NativeLimits::new(256, 128).unwrap();
        assert_eq!(small.requested_output(None).unwrap(), 128);
        assert_eq!(small.output_for_prompt(240, 128).unwrap(), 16);
        assert_eq!(small.output_for_prompt(240, 8).unwrap(), 8);
        assert!(small.requested_output(Some(0)).is_err());
        assert!(small.output_for_prompt(1, 0).is_err());
        // An output cap above the context is the context.
        assert_eq!(NativeLimits::new(256, 1 << 20).unwrap().output(), 256);
        for context in [0, 1_048_577] {
            assert!(NativeLimits::new(context, 1).is_err());
        }
    }
}
