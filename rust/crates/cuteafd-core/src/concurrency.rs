//! Boot-time concurrency formulas. Program capacities remain explicit admission limits.
use thiserror::Error;

pub const MAX_CONCURRENCY: u64 = 64;
/// Internal CLI default; zero remains an explicit request to disable retention.
pub const AUTO_ENTRIES: usize = usize::MAX;

pub const fn state_slots(concurrency: u64) -> u64 { concurrency }
pub const fn prefix_entries(concurrency: u64) -> u64 {
    let entries = (5 * concurrency).div_ceil(4);
    if entries < 20 { 20 } else { entries }
}
/// Zero selects derived storage; explicit diagnostic sizes must cover active requests.
pub const fn request_slots(concurrency: u64, explicit: u64) -> u64 {
    if explicit > concurrency { explicit } else { concurrency }
}
pub const fn members_per_lane(concurrency: u64, lanes: u64) -> u64 { concurrency.div_ceil(lanes) }
pub const fn queue_depth(concurrency: u64) -> u64 { concurrency }

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{family}: requested concurrency {requested}; {cause} capacity {capacity}")]
pub struct ConcurrencyError {
    pub family: String,
    pub requested: u64,
    pub capacity: u64,
    pub cause: &'static str,
}

pub fn validate(family: &str, concurrency: u64) -> Result<(), ConcurrencyError> {
    let (capacity, cause) = match family {
        "deepseek_v41" => (MAX_CONCURRENCY, "native request/descriptor program"),
        "deepseek_v4" | "glm5" | "glm5_flash" | "mimo_v2" | "qwen4" | "qwen3_5" | "kimi_k3" | "step5" | "mistral_large4" =>
            (64, "decode program"),
        _ => (MAX_CONCURRENCY, "boot-time concurrency"),
    };
    if concurrency == 0 || concurrency > capacity {
        Err(ConcurrencyError { family: family.into(), requested: concurrency, capacity, cause })
    } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn v41_two_lanes_cover_the_boot_concurrency_matrix() {
        for c in [1, 4, 16, 17, 32, 64] {
            validate("deepseek_v41", c).unwrap();
            let members = members_per_lane(c, 2);
            assert!(members <= crate::V41_MAX_LANE_MEMBERS as u64);
            assert!(members <= crate::V41_VERIFY_ROWS as u64);
            assert!(members * crate::MAX_DSPARK_PROPOSALS as u64 <= crate::V41_DRAFT_WIDE_ROWS as u64);
        }
        let error = validate("deepseek_v41", 65).unwrap_err();
        assert_eq!((error.requested, error.capacity), (65, 64));
        assert!(error.to_string().contains("deepseek_v41"));
        assert_eq!(crate::V41_DECODE_CAPACITIES, [1, 16, crate::V41_VERIFY_ROWS]);
        for draft in [crate::V41_DRAFT_LEGACY_ROWS, crate::V41_DRAFT_WIDE_ROWS] {
            assert!(!crate::V41_DECODE_CAPACITIES.contains(&draft));
        }
    }
    #[test]
    fn every_family_has_explicit_capacity_at_all_concurrencies() {
        for family in ["deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "mimo_v2", "qwen4", "qwen3_5",
            "kimi_k3", "step5", "muse_spark", "mistral_large4"] {
            for c in [1, 4, 16, 17, 32, 64, 65, 128] {
                assert_eq!(state_slots(c), c);
                assert_eq!(prefix_entries(c), 20.max((c * 5).div_ceil(4)));
                for lanes in [1, 2] {
                    assert_eq!(members_per_lane(c, lanes), c.div_ceil(lanes));
                }
                match validate(family, c) {
                    Ok(()) => assert_eq!(queue_depth(c), c),
                    Err(error) => {
                        assert!(error.capacity < c);
                        assert!(error.to_string().contains(family));
                        assert!(error.to_string().contains("capacity"));
                    }
                }
            }
        }
        assert!(validate("glm5", 0).is_err());
        assert!(validate("muse_spark", 129).is_err());
        assert_eq!(request_slots(16, 0), 16);
        assert_eq!(request_slots(64, 16), 64);
        assert_eq!(request_slots(4, 20), 20);
    }
}
