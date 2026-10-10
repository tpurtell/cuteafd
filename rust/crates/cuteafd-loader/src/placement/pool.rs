//! The one KV pool target: 2M tokens on cards above 32 GiB, 1M at 32 GiB or
//! less, never below the compiled context.
use cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS;

/// Agentic context floor for automatic Spark-free layouts.
pub const AGENTIC_FLOOR_TOKENS: u64 = 262_144;
const SMALL_CARD_BYTES: u64 = 32 << 30;
const SMALL_CARD_POOL_TOKENS: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolPolicy {
    /// An explicit pool is strict: admission fails rather than shrink it.
    pub requested: Option<u64>,
    /// Automatic pools stop here.
    pub target: u64,
    /// The smallest pool admission accepts. Spark-free layouts need the
    /// compiled context and the agentic floor; automatic pools with Sparks
    /// only the agentic floor (or less context), the serving context then
    /// clamped to the pool (v2: a 31.8 GiB V4 Flash served a 905K pool).
    pub floor: u64,
    /// The largest pool the build can index; a fixed onboard fills up to it.
    pub ceiling: u64,
    pub unit_rows: u64,
}

impl PoolPolicy {
    /// `card_bytes` are the coordinator GPUs' totals; `context` the compiled
    /// serving context; `requested` 0 or `None` means automatic.
    pub fn resolve(card_bytes: &[u64], context: u64, requested: Option<u64>, unit_rows: u64,
        spark_free: bool) -> Self {
        let small = card_bytes.iter().any(|&bytes| bytes <= SMALL_CARD_BYTES);
        let target = if small { SMALL_CARD_POOL_TOKENS } else { DEFAULT_GPU_KV_TOKENS }.max(context);
        // An explicit pool is its own floor (the serving context is clamped to
        // it later); automatic pools keep the compiled context, and Spark-free
        // ones the agentic floor too.
        let floor = match requested.filter(|&n| n > 0) {
            Some(n) => n,
            None if spark_free => context.max(AGENTIC_FLOOR_TOKENS),
            None => context.min(AGENTIC_FLOOR_TOKENS),
        };
        Self { requested: requested.filter(|&n| n > 0), target, floor, ceiling: u64::MAX, unit_rows }
    }

    pub(crate) fn wanted_units(&self) -> u64 {
        self.requested.unwrap_or(self.target).div_ceil(self.unit_rows.max(1))
    }

    pub(crate) fn ceiling_units(&self) -> u64 {
        self.ceiling / self.unit_rows.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_follow_the_smallest_card_and_the_context() {
        let gib = 1u64 << 30;
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, None, 256, false).target, 2 << 20);
        assert_eq!(PoolPolicy::resolve(&[96 * gib, 32 * gib], 131_072, None, 256, false).target, 1 << 20);
        assert_eq!(PoolPolicy::resolve(&[32 * gib], 1 << 21, None, 256, false).target, 1 << 21);
        let explicit = PoolPolicy::resolve(&[96 * gib], 131_072, Some(32_768), 256, true);
        assert_eq!((explicit.requested, explicit.floor), (Some(32_768), 32_768));
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256, true).floor, AGENTIC_FLOOR_TOKENS);
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256, true).requested, None);
        // With Sparks an automatic pool may fall short of the context (which serving clamps to it).
        assert_eq!(PoolPolicy::resolve(&[32 * gib], 1 << 20, None, 256, false).floor, AGENTIC_FLOOR_TOKENS);
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, None, 256, false).floor, 131_072);
    }
}
