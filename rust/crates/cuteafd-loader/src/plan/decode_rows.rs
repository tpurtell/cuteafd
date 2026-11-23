//! The per-lane verify-row budget contract every family shares (#2184, #2135).
//!
//! `DECODE_ROWS` (`--decode-rows`) bounds the total target rows (anchors plus
//! drafts) verified by one decode lane in a single step. `auto` resolves to
//! [`DEFAULT_DECODE_ROWS`] (64, today's behaviour). Plan, serve and golden all
//! resolve it once at boot beside `CONCURRENCY` and `DECODE_LANES`.
use super::PlanError;

/// Default per-lane verify-row budget when `DECODE_ROWS=auto` (or unset).
pub const DEFAULT_DECODE_ROWS: u32 = 64;

/// Compiled verify-row support declared by a [`super::Family`] descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRowSupport {
    /// Largest per-lane verify-row count compiled programs support on one coordinator GPU.
    pub single_gpu_max: u32,
    /// Largest per-lane verify-row count compiled programs support on two coordinator GPUs.
    pub multi_gpu_max: u32,
    /// Native drafter block width (draft tokens per step, excluding the anchor)
    /// used to bound `CONCURRENCY * (max_draft + 1)` when `DRAFT_LENGTH` is unset.
    pub native_draft_block: u32,
    /// Missing-program explanation when a request exceeds `single_gpu_max`.
    pub missing_single_gpu: &'static str,
    /// Missing-program explanation when a request exceeds `multi_gpu_max`.
    pub missing_multi_gpu: &'static str,
}

impl DecodeRowSupport {
    /// Same compiled capacity on one and two coordinator GPUs.
    pub const fn fixed(max: u32, native_draft_block: u32, missing: &'static str) -> Self {
        Self {
            single_gpu_max: max,
            multi_gpu_max: max,
            native_draft_block,
            missing_single_gpu: missing,
            missing_multi_gpu: missing,
        }
    }

    /// Compiled capacity ceiling for `rtx_gpus` coordinator GPUs.
    pub const fn max_for_gpus(self, rtx_gpus: usize) -> u32 {
        if rtx_gpus > 1 { self.multi_gpu_max } else { self.single_gpu_max }
    }

    /// Missing-program explanation for `rtx_gpus` coordinator GPUs.
    pub const fn missing_for_gpus(self, rtx_gpus: usize) -> &'static str {
        if rtx_gpus > 1 { self.missing_multi_gpu } else { self.missing_single_gpu }
    }
}

/// Default descriptor support for drafter-only checkpoints (no target verify loop).
pub const DRAFTER_SUPPORT: DecodeRowSupport = DecodeRowSupport::fixed(
    DEFAULT_DECODE_ROWS,
    0,
    "a drafter checkpoint has no target verify loop of its own",
);

/// Boot inputs for [`resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRowsRequest {
    /// Explicit `--decode-rows` / `DECODE_ROWS` value; `None` means `auto` (64).
    pub requested: Option<u32>,
    /// Resolved boot concurrency (`CONCURRENCY`).
    pub concurrency: u64,
    /// Resolved decode lanes (`DECODE_LANES`).
    pub decode_lanes: u8,
    /// Coordinator GPU count (1 or 2).
    pub rtx_gpus: usize,
    /// Explicit `DRAFT_LENGTH` override, if set.
    pub draft_length: Option<u32>,
}

/// CLI/config setting for `DECODE_ROWS` / `--decode-rows`: `auto` or an explicit positive count.
///
/// Wrapped as a newtype rather than `Option<u32>` so clap's derive macro parses
/// `"auto"` and integers directly through `FromStr` without treating `Option`
/// as an absent flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeRowsSetting(pub Option<u32>);

impl DecodeRowsSetting {
    pub const AUTO: Self = Self(None);

    pub const fn explicit(rows: u32) -> Self {
        Self(Some(rows))
    }

    pub const fn as_option(self) -> Option<u32> {
        self.0
    }
}

impl std::str::FromStr for DecodeRowsSetting {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_setting(value).map(Self)
    }
}

impl std::fmt::Display for DecodeRowsSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(rows) => write!(f, "{rows}"),
            None => f.write_str("auto"),
        }
    }
}

/// Parse a `DECODE_ROWS` / `--decode-rows` spelling: `"auto"` or a positive integer.
pub fn parse_setting(value: &str) -> Result<Option<u32>, String> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    match trimmed.parse::<u32>() {
        Ok(rows) if rows > 0 => Ok(Some(rows)),
        _ => Err("must be 'auto' or a positive integer".into()),
    }
}

/// Resolve the per-lane verify-row budget once at boot before layout or CUDA
/// allocation. `None` (`auto`) resolves to [`DEFAULT_DECODE_ROWS`] (64).
pub fn resolve(family: &dyn super::Family, request: DecodeRowsRequest) -> Result<u32, PlanError> {
    let Some(rows) = request.requested else {
        return Ok(DEFAULT_DECODE_ROWS);
    };
    if rows == 0 {
        return Err(PlanError::InvalidOption {
            option: "--decode-rows",
            reason: "must be positive".into(),
        });
    }
    let support = family.decode_rows();
    let capacity = support.max_for_gpus(request.rtx_gpus);
    if rows > capacity {
        return Err(PlanError::UnsupportedDecodeRows {
            value: rows,
            capacity,
            family: family.id().into(),
            reason: support.missing_for_gpus(request.rtx_gpus),
        });
    }
    let min_rows = cuteafd_core::concurrency::members_per_lane(
        request.concurrency.max(1),
        u64::from(request.decode_lanes.max(1)),
    );
    if u64::from(rows) < min_rows {
        return Err(PlanError::InvalidOption {
            option: "--decode-rows",
            reason: format!(
                "{rows} is smaller than lane member count {min_rows} (concurrency {}, {} lane(s))",
                request.concurrency, request.decode_lanes
            ),
        });
    }
    let effective_draft = match request.draft_length {
        Some(length) => length.min(support.native_draft_block),
        None => support.native_draft_block,
    };
    let demand = request.concurrency.max(1).saturating_mul(u64::from(effective_draft) + 1);
    let ceiling = u64::from(DEFAULT_DECODE_ROWS).max(demand);
    if u64::from(rows) > ceiling {
        return Err(PlanError::InvalidOption {
            option: "--decode-rows",
            reason: format!(
                "{rows} exceeds boot verify demand {ceiling} (concurrency {} x (max draft {effective_draft} + 1)); raise CONCURRENCY or lower DECODE_ROWS",
                request.concurrency
            ),
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(requested: Option<u32>, concurrency: u64, decode_lanes: u8, rtx_gpus: usize, draft_length: Option<u32>) -> DecodeRowsRequest {
        DecodeRowsRequest { requested, concurrency, decode_lanes, rtx_gpus, draft_length }
    }

    #[test]
    fn parse_setting_accepts_auto_and_positive_integers() {
        assert_eq!(parse_setting("auto").unwrap(), None);
        assert_eq!(parse_setting("AUTO").unwrap(), None);
        assert_eq!(parse_setting("64").unwrap(), Some(64));
        assert_eq!(parse_setting("128").unwrap(), Some(128));
        assert_eq!(parse_setting("256").unwrap(), Some(256));
        assert!(parse_setting("0").is_err());
        assert!(parse_setting("-1").is_err());
        assert!(parse_setting("wide").is_err());
    }

    #[test]
    fn every_family_resolves_auto_to_64_and_refuses_above_program_capacity() {
        for family in super::super::family::registry() {
            assert_eq!(resolve(*family, req(None, 16, 1, 1, None)).unwrap(), DEFAULT_DECODE_ROWS, "{}", family.id());
            assert_eq!(resolve(*family, req(Some(64), 1, 1, 1, None)).unwrap(), 64, "{}", family.id());
            for gpus in [1, 2] {
                let cap = family.decode_rows().max_for_gpus(gpus);
                let over = cap + 1;
                let err = resolve(*family, req(Some(over), 64, 1, gpus, None)).unwrap_err();
                assert!(matches!(err, PlanError::UnsupportedDecodeRows { value, capacity, .. }
                    if value == over && capacity == cap), "{} gpus={gpus}: {err}", family.id());
            }
        }
    }

    #[test]
    fn v41_supports_up_to_256_bounded_by_concurrency_and_draft_length() {
        let v41 = &crate::plan::families::deepseek::DEEPSEEK_V41;
        assert_eq!(resolve(v41, req(None, 16, 2, 2, None)).unwrap(), 64);
        assert_eq!(resolve(v41, req(Some(128), 16, 1, 2, None)).unwrap(), 128);
        assert_eq!(resolve(v41, req(Some(128), 16, 2, 2, None)).unwrap(), 128);
        assert_eq!(resolve(v41, req(Some(256), 32, 1, 2, None)).unwrap(), 256);
        assert_eq!(resolve(v41, req(Some(96), 16, 1, 1, Some(5))).unwrap(), 96);
        // Exceeds concurrency * (draft + 1) = 16 * 8 = 128.
        assert!(matches!(resolve(v41, req(Some(129), 16, 1, 2, None)), Err(PlanError::InvalidOption { .. })));
        // DRAFT_LENGTH=3 caps demand at 16 * 4 = 64.
        assert!(matches!(resolve(v41, req(Some(128), 16, 1, 2, Some(3))), Err(PlanError::InvalidOption { .. })));
        // Below lane member floor: 16 requests on 1 lane need >= 16 rows.
        assert!(matches!(resolve(v41, req(Some(15), 16, 1, 2, None)), Err(PlanError::InvalidOption { .. })));
        assert_eq!(resolve(v41, req(Some(16), 16, 1, 2, None)).unwrap(), 16);
        // Above compiled 256-row program capacity.
        let err = resolve(v41, req(Some(512), 64, 1, 2, None)).unwrap_err();
        assert!(matches!(err, PlanError::UnsupportedDecodeRows { value: 512, capacity: 256, .. }), "{err}");
    }

    #[test]
    fn hy4_refuses_decode_rows_not_covered_by_split_k_exports() {
        let hy4 = &crate::plan::families::hy4_flash::HY4_FLASH;
        for gpus in [1, 2] {
            assert_eq!(resolve(hy4, req(Some(32), 16, 1, gpus, None)).unwrap(), 32);
            assert_eq!(resolve(hy4, req(Some(64), 16, 1, gpus, None)).unwrap(), 64);
            let err = resolve(hy4, req(Some(128), 32, 1, gpus, None)).unwrap_err();
            assert!(matches!(err, PlanError::UnsupportedDecodeRows { value: 128, capacity: 64, .. }), "{err}");
            assert!(err.to_string().contains("Hy4-flash split-K decode exports"), "{err}");
        }
    }

    #[test]
    fn glmf_supports_128_on_one_gpu_and_refuses_128_on_two_gpus_naming_missing_program() {
        let glmf = &crate::plan::families::glm::GLM5_FLASH;
        assert_eq!(resolve(glmf, req(Some(128), 16, 1, 1, None)).unwrap(), 128);
        let err = resolve(glmf, req(Some(128), 16, 1, 2, None)).unwrap_err();
        assert!(matches!(err, PlanError::UnsupportedDecodeRows { value: 128, capacity: 64, .. }), "{err}");
        assert!(err.to_string().contains("glmf2_"), "{err}");
        let err = resolve(glmf, req(Some(256), 32, 1, 1, None)).unwrap_err();
        assert!(matches!(err, PlanError::UnsupportedDecodeRows { value: 256, capacity: 128, .. }), "{err}");
        assert!(err.to_string().contains("glmf_"), "{err}");
    }
}
