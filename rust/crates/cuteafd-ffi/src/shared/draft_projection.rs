//! Preloaded BF16 drafter projections with one canonical per-row K reduction.
use crate::programs::{Program, ProgramSpec, Programs, Scalar};
use std::{collections::BTreeMap, ffi::c_void};
use thiserror::Error;

/// Covers context updates, block drafting and target verification independently.
pub fn batch_rows(context_rows: usize, block: usize, sequences: usize, verify_rows: usize) -> usize {
    context_rows.max(block.saturating_mul(sequences)).max(verify_rows)
}

#[derive(Debug, Error)]
pub enum ProjectionError {
    #[error("BF16 drafter projection K={k}, outputs={n} is missing or invalid: {reason}")]
    Shape { k: usize, n: usize, reason: String },
    #[error("BF16 drafter projection K={k}, outputs={n}: startup requires {requested} rows, exported max_rows={capacity}")]
    Capacity { k: usize, n: usize, requested: usize, capacity: usize },
    #[error("BF16 drafter projection invalid buffers or rows")]
    Arguments,
    #[error("BF16 drafter projection launch: {0}")]
    Launch(#[from] anyhow::Error),
}

type Result<T> = std::result::Result<T, ProjectionError>;

/// Resolves every admitted shape before capture; launch never loads programs.
pub struct DraftProjections<'a> {
    programs: BTreeMap<(usize, usize), Program<'a>>,
}

impl<'a> DraftProjections<'a> {
    pub fn load(programs: &'a Programs<'a>, shapes: &[(usize, usize)], max_rows: usize) -> Result<Self> {
        // Validate the entire admission before loading even the first CUDA program.
        for &(k, n) in shapes {
            let name = format!("draft_bf16_k{k}_n{n}");
            let spec = programs.spec(&name)
                .map_err(|e| ProjectionError::Shape { k, n, reason: e.to_string() })?;
            admit(spec, k, n, max_rows)?;
        }
        let mut loaded = BTreeMap::new();
        for &(k, n) in shapes {
            if loaded.contains_key(&(k, n)) { continue; }
            let name = format!("draft_bf16_k{k}_n{n}");
            let invalid = |reason: String| ProjectionError::Shape { k, n, reason };
            let spec = programs.spec(&name).map_err(|e| invalid(e.to_string()))?;
            validate(spec, k, n).map_err(invalid)?;
            let program = programs.program(&name, &["x", "w", "out"])
                .map_err(|e| ProjectionError::Shape { k, n, reason: e.to_string() })?;
            loaded.insert((k, n), program);
        }
        Ok(Self { programs: loaded })
    }

    /// # Safety
    /// Device buffers hold nonoverlapping BF16 [rows,k], [n,k] and [rows,n]
    /// on the supplied stream. Their owners survive all queued work.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(&self, x: *const c_void, weight: *const c_void, out: *mut c_void,
        rows: usize, k: usize, n: usize, stream: *mut c_void) -> Result<()> {
        if rows == 0 || [x, weight, out.cast_const()].iter()
            .any(|p| p.is_null() || *p as usize % 16 != 0) {
            return Err(ProjectionError::Arguments);
        }
        let program = self.programs.get(&(k, n)).ok_or_else(|| ProjectionError::Shape {
            k, n, reason: "shape was not admitted at startup".into() })?;
        require_rows(program.spec(), k, n, rows)?;
        // SAFETY: the caller retains the documented buffers and stream; shape and rows are checked.
        unsafe { program.launch(&[x.cast_mut(), weight.cast_mut(), out], &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }
}

fn admit(spec: &ProgramSpec, k: usize, n: usize, rows: usize) -> Result<()> {
    validate(spec, k, n).map_err(|reason| ProjectionError::Shape { k, n, reason })?;
    require_rows(spec, k, n, rows)
}

fn require_rows(spec: &ProgramSpec, k: usize, n: usize, rows: usize) -> Result<()> {
    let capacity = spec.capacity_rows as usize;
    if rows == 0 || rows > capacity {
        return Err(ProjectionError::Capacity { k, n, requested: rows, capacity });
    }
    Ok(())
}

fn validate(spec: &ProgramSpec, k: usize, n: usize) -> std::result::Result<(), String> {
    let g = &spec.geometry;
    if k == 0 || n == 0 || k % 8 != 0 || n % 8 != 0
        || spec.name != format!("draft_bf16_k{k}_n{n}")
        || spec.capacity_rows == 0 || spec.capacity_rows > i32::MAX as u32
        || spec.scalar_kinds != b"i" || !spec.scratch.is_empty()
        || !spec.pointers.iter().map(String::as_str).eq(["x", "w", "out"])
        || g["width"].as_u64() != Some(k as u64) || g["outputs"].as_u64() != Some(n as u64)
        || g["output_dtype"].as_str() != Some("torch.bfloat16")
        || g["reduction"].as_str() != Some("ordered_k16")
        || g["tiles_by_rows"] != serde_json::json!([[16, 1, 4, 16, 64, 6]]) {
        return Err("expected canonical K16 BF16 ABI, M16N64 small-row route and positive signed32 row capacity".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> ProgramSpec {
        ProgramSpec { name: "draft_bf16_k5120_n7168".into(), pointers: vec!["x".into(), "w".into(), "out".into()],
            scalar_kinds: b"i".to_vec(), scratch: Default::default(), capacity_rows: 2048,
            geometry: serde_json::json!({"width":5120,"outputs":7168,"output_dtype":"torch.bfloat16",
                "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}) }
    }
    #[test]
    fn startup_refuses_block_times_concurrency_before_loading() {
        let s = spec();
        let rows = batch_rows(2048, 64, 64, 64);
        assert!(matches!(admit(&s, 5120, 7168, rows),
            Err(ProjectionError::Capacity { k: 5120, n: 7168, requested: 4096, capacity: 2048 })));
        assert!(admit(&s, 5120, 7168, batch_rows(2048, 8, 64, 64)).is_ok());
        assert!(admit(&s, 5120, 7168, batch_rows(1, 7, 64, 2049)).is_err());
        assert!(admit(&s, 5120, 7168, batch_rows(1, usize::MAX, 64, 1)).is_err());
        // The manifest, not a duplicated host constant, sets the admissible bound.
        let mut smaller = s.clone();
        smaller.capacity_rows = 256;
        assert!(matches!(admit(&smaller, 5120, 7168, batch_rows(1, 7, 64, 64)),
            Err(ProjectionError::Capacity { requested: 448, capacity: 256, .. })));
        let mut larger = s;
        larger.capacity_rows = 4096;
        assert!(admit(&larger, 5120, 7168, rows).is_ok());
    }

    #[test]
    fn canonical_manifest_is_required() {
        assert!(validate(&spec(),5120,7168).is_ok());
        for field in ["width","outputs","output_dtype","reduction","tiles_by_rows"] {
            let mut s=spec(); s.geometry[field]=serde_json::Value::Null;
            assert!(validate(&s,5120,7168).is_err(), "{field}");
        }
        let mut s=spec(); s.capacity_rows=0; assert!(validate(&s,5120,7168).is_err());
        let mut s=spec(); s.pointers.swap(0,1); assert!(validate(&s,5120,7168).is_err());
    }
}
