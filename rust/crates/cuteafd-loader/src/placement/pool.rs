//! The one KV pool target: 2M tokens on cards above 32 GiB, 1M at 32 GiB or
//! less, never below the compiled context.
use cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS;

/// Agentic context floor for automatic Spark-free layouts.
pub const AGENTIC_FLOOR_TOKENS: u64 = 262_144;
const SMALL_CARD_BYTES: u64 = 32 << 30;
const SMALL_CARD_POOL_TOKENS: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolPolicy {
    /// Fixed or operator-set context requirement; zero for automatic context.
    /// Independent of the mutable fit floor and the selected pool.
    pub compiled_context: u64,
    /// An explicit pool is strict: admission fails rather than shrink it.
    pub requested: Option<u64>,
    /// Automatic pools stop here.
    pub target: u64,
    /// Spark-free layouts fail below this; pools with Sparks may go lower
    /// (the serving context is then clamped to the pool).
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
        // Explicit pools must cover the compiled context. Only automatic,
        // Spark-free pools additionally retain the agentic floor.
        let floor = match requested.filter(|&n| n > 0) {
            Some(n) => n.max(context),
            None if spark_free => context.max(AGENTIC_FLOOR_TOKENS),
            None => context,
        };
        Self { compiled_context: context, requested: requested.filter(|&n| n > 0), target, floor,
            ceiling: u64::MAX, unit_rows }
    }

    /// Automatic checkpoint context may be clamped by serving to an explicit
    /// diagnostic pool. Fixed compiled extents must not use this mode.
    pub fn with_automatic_context(mut self, automatic: bool) -> Self {
        if automatic {
            self.compiled_context = 0;
            if let Some(requested) = self.requested {
                self.floor = requested;
            }
        }
        self
    }

    /// Pin a selected pool for re-admission without inventing an operator request.
    pub fn pin_selected(&mut self, tokens: u64) {
        self.target = tokens;
        self.floor = tokens;
        self.ceiling = tokens;
    }

    pub fn validate_explicit(&self) -> Result<(), super::PlacementError> {
        if let Some(requested) = self.requested.filter(|&n| n > 0 && n < self.compiled_context) {
            return Err(super::PlacementError::RequestedBelowContext {
                requested, floor: self.compiled_context,
            });
        }
        Ok(())
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
        assert_eq!((explicit.requested, explicit.floor), (Some(32_768), 131_072));
        assert_eq!(explicit.validate_explicit(), Err(super::super::PlacementError::RequestedBelowContext {
            requested: 32_768, floor: 131_072,
        }));
        assert!(PoolPolicy::resolve(&[96 * gib], 32_768, Some(32_768), 256, true)
            .validate_explicit().is_ok());
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256, true).floor, AGENTIC_FLOOR_TOKENS);
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256, true).requested, None);

    }
}
