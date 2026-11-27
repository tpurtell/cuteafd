//! Shared Hy4 family daemon primitives (`hy4` and `hy4_flash`).
//!
//! Lifts the genuinely identical compact-core native linear owner, absorbed
//! KV-B / dense weight loading helpers, router program contract validation,
//! and golden scoring helpers so `hy4` and `hy4_flash` share one implementation
//! without copy-paste.

use anyhow::{ensure, Result};
use cuteafd_ffi::programs::ProgramSpec;

pub(crate) mod golden;
pub(crate) mod native;
pub(crate) mod weights;

/// Expected low-row skinny GEMV threshold shared with GLM (`dsv4_ffn.ROUTER_SKINNY_MAX_ROWS`).
pub(crate) const ROUTER_SKINNY_MAX_ROWS: u64 = 64;

/// Validate that a loaded Hy4 / Hy4-flash router program is the shared
/// skinny-GEMV (<= 64 rows, row-invariant) / TMA-MMA (> 64 rows) FP32 scores
/// program and not a stale `Bf16PrefillKernel` (`skinny_rows=0`) export.
pub(crate) fn validate_router_spec(spec: &ProgramSpec, hidden: u64, experts: u64) -> Result<()> {
    let g = &spec.geometry;
    ensure!(
        g["hidden"].as_u64() == Some(hidden) && g["experts"].as_u64() == Some(experts),
        "{} router geometry {:?} does not match expected hidden={hidden} experts={experts}",
        spec.name,
        g
    );
    ensure!(
        g["skinny_max_rows"].as_u64() == Some(ROUTER_SKINNY_MAX_ROWS)
            && g["exact_row_cap"].as_u64() == Some(ROUTER_SKINNY_MAX_ROWS),
        "{} must publish skinny_max_rows={ROUTER_SKINNY_MAX_ROWS} and exact_row_cap={ROUTER_SKINNY_MAX_ROWS}, got {:?}",
        spec.name,
        g
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn spec(name: &str, geometry: serde_json::Value) -> ProgramSpec {
        ProgramSpec {
            name: name.into(),
            pointers: vec!["x".into(), "w".into(), "logits".into()],
            scalar_kinds: vec![0],
            scratch: HashMap::new(),
            capacity_rows: 4096,
            geometry,
        }
    }

    #[test]
    fn validate_router_spec_accepts_skinny_router_for_hy4_and_hy4_flash() {
        let hy4 = spec(
            "hy4_router",
            json!({"hidden": 6144, "experts": 256, "skinny_max_rows": 64, "exact_row_cap": 64}),
        );
        validate_router_spec(&hy4, 6144, 256).unwrap();

        let hy4f = spec(
            "hy4f_router",
            json!({"hidden": 3072, "experts": 512, "skinny_max_rows": 64, "exact_row_cap": 64}),
        );
        validate_router_spec(&hy4f, 3072, 512).unwrap();
    }

    #[test]
    fn validate_router_spec_rejects_bf16_prefill_or_mismatched_geometry() {
        let stale = spec(
            "hy4_router",
            json!({"width": 6144, "outputs": 256, "output_dtype": "torch.float32", "skinny_rows": 0, "exact_row_cap": 2147483647}),
        );
        assert!(validate_router_spec(&stale, 6144, 256).is_err());

        let wrong_cap = spec(
            "hy4_router",
            json!({"hidden": 6144, "experts": 256, "skinny_max_rows": 0, "exact_row_cap": 2147483647}),
        );
        assert!(validate_router_spec(&wrong_cap, 6144, 256).is_err());

        let wrong_shape = spec(
            "hy4f_router",
            json!({"hidden": 6144, "experts": 256, "skinny_max_rows": 64, "exact_row_cap": 64}),
        );
        assert!(validate_router_spec(&wrong_shape, 3072, 512).is_err());
    }
}
