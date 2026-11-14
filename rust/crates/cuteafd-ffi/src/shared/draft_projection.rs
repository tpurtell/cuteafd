//! Shared preloaded BF16 target/drafter projections with one canonical per-row K reduction.
use crate::{CuteafdDeviceBuffer, programs::{Program, ProgramSpec, Programs, Scalar}};

/// The materialized projection boundary; F32 preserves TP2 partial accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionOutput { Bf16, F32 }

impl ProjectionOutput {
    pub const fn bytes(self) -> usize { match self { Self::Bf16 => 2, Self::F32 => 4 } }
    pub const fn torch_dtype(self) -> &'static str {
        match self { Self::Bf16 => "torch.bfloat16", Self::F32 => "torch.float32" }
    }
    pub fn program_name(self, k: usize, n: usize) -> String {
        let suffix = match self { Self::Bf16 => "", Self::F32 => "_f32" };
        format!("draft_bf16_k{k}_n{n}{suffix}")
    }
    pub(crate) fn require_output(self, out: CuteafdDeviceBuffer, rows: usize, n: usize) -> Result<()> {
        let bytes = rows.checked_mul(n).and_then(|v| v.checked_mul(self.bytes()))
            .ok_or(ProjectionError::Arguments)?;
        if rows == 0 || out.ptr.is_null() || out.ptr as usize % 16 != 0 || out.bytes < bytes {
            return Err(ProjectionError::Arguments);
        }
        Ok(())
    }
}
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
    output: ProjectionOutput,
}

impl<'a> DraftProjections<'a> {
    pub fn load(programs: &'a Programs<'a>, shapes: &[(usize, usize)], max_rows: usize) -> Result<Self> {
        Self::load_with_output(programs, shapes, max_rows, ProjectionOutput::Bf16)
    }

    pub fn load_with_output(programs: &'a Programs<'a>, shapes: &[(usize, usize)], max_rows: usize,
        output: ProjectionOutput) -> Result<Self> {
        // Validate the entire admission before loading even the first CUDA program.
        for &(k, n) in shapes {
            let name = output.program_name(k, n);
            let spec = programs.spec(&name)
                .map_err(|e| ProjectionError::Shape { k, n, reason: e.to_string() })?;
            admit_output(spec, k, n, max_rows, output)?;
        }
        let mut loaded = BTreeMap::new();
        for &(k, n) in shapes {
            if loaded.contains_key(&(k, n)) { continue; }
            let name = output.program_name(k, n);
            let invalid = |reason: String| ProjectionError::Shape { k, n, reason };
            let spec = programs.spec(&name).map_err(|e| invalid(e.to_string()))?;
            validate_output(spec, k, n, output).map_err(invalid)?;
            let program = programs.program(&name, &["x", "w", "out"])
                .map_err(|e| ProjectionError::Shape { k, n, reason: e.to_string() })?;
            loaded.insert((k, n), program);
        }
        Ok(Self { programs: loaded, output })
    }

    pub fn output(&self) -> ProjectionOutput { self.output }

    /// Loaded programs participating in the family's packing exactness proof.
    pub fn program_names(&self) -> impl Iterator<Item = &str> {
        self.programs.values().map(|program| program.spec().name.as_str())
    }

    /// # Safety
    /// Input/weight pointers and the stream satisfy `launch`'s device/lifetime contract.
    /// The output owner survives all queued work; its extent is checked before launching.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_checked(&self, x: *const c_void, weight: *const c_void,
        out: CuteafdDeviceBuffer, rows: usize, k: usize, n: usize, stream: *mut c_void) -> Result<()> {
        self.output.require_output(out, rows, n)?;
        // SAFETY: caller retains all owners; the typed output extent is checked above.
        unsafe { self.launch(x, weight, out.ptr, rows, k, n, stream) }
    }

    /// # Safety
    /// Device buffers hold nonoverlapping BF16 [rows,k], [n,k] and output-typed [rows,n]
    /// on the supplied stream. Their owners survive all queued work. This raw-pointer
    /// entry bypasses output extent checks; new native-core callers use `launch_checked`.
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

/// Fail closed on the same ordered-K16 ABI during CPU-only placement.
pub fn validate_package_geometry(manifest: &serde_json::Value, k: usize, n: usize, rows: usize) -> Result<String> {
    validate_package_geometry_with_output(manifest, k, n, rows, ProjectionOutput::Bf16)
}

pub fn validate_package_geometry_with_output(manifest: &serde_json::Value, k: usize, n: usize,
    rows: usize, output: ProjectionOutput) -> Result<String> {
    let name = output.program_name(k, n);
    let invalid = |reason: &str| ProjectionError::Shape { k, n, reason: reason.into() };
    let entries = manifest["programs"].as_array().ok_or_else(|| invalid("program inventory"))?;
    let mut matches = entries.iter().filter(|p| p["name"].as_str() == Some(&name));
    let entry = matches.next().ok_or_else(|| invalid("missing ordered-K16 BF16 program"))?;
    if matches.next().is_some() { return Err(invalid("duplicate BF16 program")); }
    let pointers = entry["pointers"].as_array().ok_or_else(|| invalid("program pointers"))?.iter()
        .map(|p| p["name"].as_str().map(str::to_owned).ok_or_else(|| invalid("pointer name")))
        .collect::<Result<Vec<_>>>()?;
    let scalars = entry["scalars"].as_array().ok_or_else(|| invalid("program scalars"))?;
    if scalars.len() != 1 || scalars[0]["type"] != "int32" {
        return Err(invalid("expected one int32 row scalar"));
    }
    let scratch = entry["scratch_bytes_at_capacity"].as_object().ok_or_else(|| invalid("program scratch"))?;
    if !scratch.is_empty() { return Err(invalid("canonical BF16 projection has no scratch")); }
    let capacity_rows = entry["capacity_rows"].as_u64().and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| invalid("program capacity"))?;
    let spec = ProgramSpec { name: name.clone(), pointers, scalar_kinds: b"i".to_vec(),
        scratch: Default::default(), capacity_rows, geometry: entry["geometry"].clone() };
    admit_output(&spec, k, n, rows, output)?;
    Ok(name)
}

#[cfg(test)]
fn admit(spec: &ProgramSpec, k: usize, n: usize, rows: usize) -> Result<()> {
    admit_output(spec, k, n, rows, ProjectionOutput::Bf16)
}

fn admit_output(spec: &ProgramSpec, k: usize, n: usize, rows: usize, output: ProjectionOutput) -> Result<()> {
    validate_spec(spec, k, n, rows, output)
}

/// Validate a cached typed program during CPU-only placement, without reparsing its manifest.
pub fn validate_spec(spec: &ProgramSpec, k: usize, n: usize, rows: usize,
    output: ProjectionOutput) -> Result<()> {
    validate_output(spec, k, n, output).map_err(|reason| ProjectionError::Shape { k, n, reason })?;
    require_rows(spec, k, n, rows)
}

fn require_rows(spec: &ProgramSpec, k: usize, n: usize, rows: usize) -> Result<()> {
    let capacity = spec.capacity_rows as usize;
    if rows == 0 || rows > capacity {
        return Err(ProjectionError::Capacity { k, n, requested: rows, capacity });
    }
    Ok(())
}

#[cfg(test)]
fn validate(spec: &ProgramSpec, k: usize, n: usize) -> std::result::Result<(), String> {
    validate_output(spec, k, n, ProjectionOutput::Bf16)
}

fn validate_output(spec: &ProgramSpec, k: usize, n: usize, output: ProjectionOutput) -> std::result::Result<(), String> {
    let g = &spec.geometry;
    if k == 0 || n == 0 || k % 8 != 0 || n % 8 != 0
        || spec.name != output.program_name(k, n)
        || spec.capacity_rows == 0 || spec.capacity_rows > i32::MAX as u32
        || spec.scalar_kinds != b"i" || !spec.scratch.is_empty()
        || !spec.pointers.iter().map(String::as_str).eq(["x", "w", "out"])
        || g["width"].as_u64() != Some(k as u64) || g["outputs"].as_u64() != Some(n as u64)
        || g["output_dtype"].as_str() != Some(output.torch_dtype())
        || g["reduction"].as_str() != Some("ordered_k16")
        || !valid_tiles(&g["tiles_by_rows"]) {
        return Err("expected canonical K16 projection ABI, valid row tiles and positive signed32 row capacity".into());
    }
    Ok(())
}

fn valid_tiles(value: &serde_json::Value) -> bool {
    let Some(tiles) = value.as_array() else { return false; };
    let mut previous = 0;
    tiles.iter().all(|entry| {
        let Some(entry) = entry.as_array() else { return false; };
        let Some(rows) = entry.first().and_then(serde_json::Value::as_u64) else { return false; };
        let valid = (4..=6).contains(&entry.len()) && rows > previous
            && entry.iter().all(|v| v.as_u64().is_some_and(|n| n > 0));
        previous = rows;
        valid
    })
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
    fn program_names_reports_only_loaded_programs() {
        for output in [ProjectionOutput::Bf16, ProjectionOutput::F32] {
            let registry = DraftProjections { programs: BTreeMap::new(), output };
            assert_eq!(registry.program_names().collect::<Vec<&str>>(), Vec::<&str>::new());
        }
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
    fn muse_bridge_capacity_raise_preserves_all_existing_row_admissions() {
        let mut previous = spec();
        previous.name = ProjectionOutput::Bf16.program_name(8192, 2048);
        previous.geometry["width"] = serde_json::json!(8192);
        previous.geometry["outputs"] = serde_json::json!(2048);
        let mut raised = previous.clone();
        raised.capacity_rows = 4096;
        for rows in [1, 4, 8, 16, 64, 128, 512, 1024, 2048] {
            assert!(admit(&previous, 8192, 2048, rows).is_ok());
            assert!(admit(&raised, 8192, 2048, rows).is_ok());
        }
        assert!(admit(&previous, 8192, 2048, 2752).is_err());
        assert!(admit(&raised, 8192, 2048, 2752).is_ok());
        assert!(admit(&raised, 8192, 2048, 4096).is_ok());
        assert!(admit(&raised, 8192, 2048, 4097).is_err());
        assert_eq!(previous.geometry, raised.geometry);
    }

    #[test]
    fn offline_target_admission_matches_runtime_abi() {
        let entry = serde_json::json!({"name":"draft_bf16_k5120_n7168", "capacity_rows":4096,
            "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}], "scalars":[{"type":"int32"}],
            "scratch_bytes_at_capacity":{}, "geometry":spec().geometry});
        let manifest = serde_json::json!({"programs":[entry.clone()]});
        assert_eq!(validate_package_geometry(&manifest,5120,7168,4096).unwrap(), "draft_bf16_k5120_n7168");
        assert!(validate_package_geometry(&manifest,5120,7168,4097).is_err());
        assert!(validate_package_geometry(&manifest,5120,8192,1).is_err());
        assert!(validate_package_geometry(&serde_json::json!({"programs":[entry.clone(),entry]}),5120,7168,1).is_err());
        for (field, value) in [("geometry",serde_json::Value::Null),
            ("capacity_rows",serde_json::json!(0)), ("pointers",serde_json::json!([])),
            ("scalars",serde_json::json!([{"type":"float32"}])),
            ("scratch_bytes_at_capacity",serde_json::json!({"scratch":256}))] {
            let mut broken=manifest.clone(); broken["programs"][0][field]=value;
            assert!(validate_package_geometry(&broken,5120,7168,1).is_err(), "{field}");
        }
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

    #[test]
    fn typed_spec_matches_runtime_admission() {
        for output in [ProjectionOutput::Bf16, ProjectionOutput::F32] {
            let mut s = spec();
            s.name = output.program_name(5120, 7168);
            s.geometry["output_dtype"] = output.torch_dtype().into();
            assert!(validate_spec(&s, 5120, 7168, 2048, output).is_ok());
            assert!(validate_spec(&s, 5120, 7168, 0, output).is_err());
            assert!(validate_spec(&s, 5120, 7168, 2049, output).is_err());
            s.geometry["reduction"] = "unordered".into();
            assert!(validate_spec(&s, 5120, 7168, 1, output).is_err());
        }
    }

    #[test]
    fn f32_registry_refuses_bf16_sized_output() {
        let registry = DraftProjections { programs: BTreeMap::new(), output: ProjectionOutput::F32 };
        let output = CuteafdDeviceBuffer { ptr: 16usize as *mut c_void, bytes: 8 * 7168 * 2, device_id: 0, flags: 0 };
        // SAFETY: undersized output is rejected before program resolution or any pointer use.
        assert!(matches!(unsafe { registry.launch_checked(std::ptr::null(), std::ptr::null(), output,
            8, 6144, 7168, std::ptr::null_mut()) }, Err(ProjectionError::Arguments)));
        assert!(ProjectionOutput::Bf16.require_output(output, 8, 7168).is_ok());
        assert!(ProjectionOutput::F32.require_output(CuteafdDeviceBuffer { bytes: 64 * 7168 * 4,
            ..output }, 8, 7168).is_ok());
        let mut f32 = spec();
        f32.name.push_str("_f32"); f32.geometry["output_dtype"] = "torch.float32".into();
        assert!(validate_output(&f32,5120,7168,ProjectionOutput::F32).is_ok());
        assert!(validate(&f32,5120,7168).is_err());
    }

}
