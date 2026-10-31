//! Shared native trellis ABI v1. All storage is owned by the engine before
//! capture; the F32 core result is exposed for TP reduction before output H128.
use super::programs::{Program, ProgramSpec, Programs, Scalar};
use crate::CuteafdDeviceBuffer as Buffer;
use std::ffi::c_void;
use thiserror::Error;

// Native prefill writes F32 output directly; only rows below 130 use split-K partials.
pub const TRELLIS_PARTIAL_ROWS: usize = 129;

const INPUT: &[&str] = &["input", "suh", "rotated", "input_block_scales"];
const CORE: &[&str] = &["rotated_inputs", "payload", "input_block_scales", "partial", "output"];
const OUTPUT: &[&str] = &["input", "svh", "output"];

#[derive(Debug, Error)]
pub enum DenseTrellisError {
    #[error("native dense trellis invalid {0}")]
    Invalid(&'static str),
    #[error("dense_trellis_k6 has no native K{bits} row policy for [N={output}, K={input}], side={side}, codebook={codebook}")]
    MissingGeometry { input: usize, output: usize, side: usize, codebook: String, bits: usize },
    #[error("native dense trellis program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, DenseTrellisError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrellisDType { Bf16, F32 }
impl TrellisDType {
    fn bytes(self) -> usize { match self { Self::Bf16 => 2, Self::F32 => 4 } }
    fn name(self) -> &'static str { match self { Self::Bf16 => "bf16", Self::F32 => "f32" } }
}

#[derive(Debug, Clone, Copy)]
pub struct DenseTrellisGeometry {
    pub input: usize, pub output: usize, pub bits: usize,
    pub side: usize, pub rows: usize,
    pub input_dtype: TrellisDType, pub output_dtype: TrellisDType,
}

/// Private rotation id zero: each independently calibrated projection owns its
/// H128 input. The pointer table contains exactly its rotated buffer's address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseTrellisWorkspace {
    pub rotated: usize, pub pointer_table: usize, pub block_scales: usize,
    pub partial: usize, pub accumulator: usize,
}
impl DenseTrellisGeometry {
    pub fn workspace(self) -> Result<DenseTrellisWorkspace> {
        let Self { input: k, output: n, bits, side, rows, .. } = self;
        if !matches!(bits, 2..=6) || !matches!(side, 1 | 2) || k == 0 || n == 0
            || k % 128 != 0 || n % 128 != 0 || rows == 0
            || [k, n, rows].iter().any(|&v| v > i32::MAX as usize) {
            return Err(DenseTrellisError::Invalid("K2-K6/H128/side/row geometry"));
        }
        let product = |values: &[usize]| values.iter().try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or(DenseTrellisError::Invalid("workspace size overflow"));
        let splits = k.div_ceil(1024);
        let limit = (i32::MAX as usize) / 2;
        if product(&[rows.div_ceil(16), n / 128, splits])? > limit
            || product(&[rows, n / 128])? > limit || product(&[k / 16, n / 16])? > limit {
            return Err(DenseTrellisError::Invalid("native int32 task/tile indexing"));
        }
        // Also prove all external input/output and payload byte extents fit.
        product(&[k, n, bits])?;
        product(&[rows, k, self.input_dtype.bytes()])?;
        product(&[rows, n, self.output_dtype.bytes()])?;
        Ok(DenseTrellisWorkspace { rotated: product(&[rows, k, 2])?, pointer_table: 8,
            block_scales: k / 128 * 2, partial: product(&[rows.min(TRELLIS_PARTIAL_ROWS), splits, n, 4])?,
            accumulator: product(&[rows, n, 4])? })
    }
}

pub struct DenseTrellis<'a> {
    input: Program<'a>, core: Program<'a>, output: Program<'a>,
    geometry: DenseTrellisGeometry, workspace: DenseTrellisWorkspace, ctas: i32,
}

impl<'a> DenseTrellis<'a> {
    /// Names come from the package manifest, not a family-specific convention.
    /// Resolve all programs and query the runtime SM count before graph capture.
    pub fn load(programs: &'a Programs<'a>, names: [&str; 3], geometry: DenseTrellisGeometry,
        codebook: &str) -> Result<Self> {
        let workspace = geometry.workspace()?;
        validate_manifest([programs.spec(names[0])?, programs.spec(names[1])?, programs.spec(names[2])?],
            geometry, codebook, workspace)?;
        let input = programs.program(names[0], INPUT)?;
        let core = programs.program(names[1], CORE)?;
        let output = programs.program(names[2], OUTPUT)?;
        let ctas = programs.library().sm_count()?.checked_mul(4)
            .and_then(|v| i32::try_from(v).ok()).filter(|&v| v > 0)
            .ok_or(DenseTrellisError::Invalid("runtime SM count"))?;
        Ok(Self { input, core, output, geometry, workspace, ctas })
    }

    /// Charged at the caller's maximum live rows, not the package's prefill
    /// capacity. The native kernel indexes partials by the live row count.
    pub fn workspace(&self) -> DenseTrellisWorkspace { self.workspace }

    /// Full suh is applied before H128; emitted K128 scales are exactly one.
    pub fn input(&self, rows: usize, input: Buffer, suh: Buffer, rotated: Buffer,
        scales: Buffer, stream: *mut c_void) -> Result<()> {
        self.rows(rows)?;
        let k = self.geometry.input;
        buffers(&[(input, rows*k*self.geometry.input_dtype.bytes()), (suh, k*2),
            (rotated, rows*k*2), (scales, self.workspace.block_scales)])?;
        disjoint(&[input, suh, rotated, scales])?;
        // SAFETY: shape/device/extents and nonaliasing checked; the engine keeps
        // every allocation and stream live through completion and graph replay.
        unsafe { self.input.launch(&[input.ptr, suh.ptr, rotated.ptr, scales.ptr],
            &[Scalar::I32(rows as i32), Scalar::I32(self.ctas)], stream)?; }
        Ok(())
    }

    /// The caller initializes pointer_table[0] to its rotated input allocation
    /// before capture. Core output remains F32, suitable for K-split TP reduction.
    pub fn core(&self, rows: usize, pointer_table: Buffer, payload: Buffer, scales: Buffer,
        partial: Buffer, accumulator: Buffer, stream: *mut c_void) -> Result<()> {
        self.rows(rows)?;
        let g = self.geometry;
        buffers(&[(pointer_table, 8), (payload, g.input*g.output*g.bits/8),
            (scales, self.workspace.block_scales), (partial, rows.min(TRELLIS_PARTIAL_ROWS)*g.input.div_ceil(1024)*g.output*4),
            (accumulator, rows*g.output*4)])?;
        disjoint(&[pointer_table, payload, scales, partial, accumulator])?;
        // SAFETY: buffer metadata and scratch extents checked. The caller owns
        // the initialized pointer table and rotated inputs through completion.
        unsafe { self.core.launch(&[pointer_table.ptr, payload.ptr, scales.ptr, partial.ptr, accumulator.ptr],
            &[Scalar::I32(rows as i32), Scalar::I32(0), Scalar::I32(self.ctas)], stream)?; }
        Ok(())
    }

    /// Apply svh once, after any F32 TP reduction. Stored extents are retained:
    /// padded kv_a N640 must be trimmed to logical N576 by the following stage.
    pub fn output(&self, rows: usize, accumulator: Buffer, svh: Buffer, output: Buffer,
        stream: *mut c_void) -> Result<()> {
        self.rows(rows)?;
        let n = self.geometry.output;
        buffers(&[(accumulator, rows*n*4), (svh, n*2), (output, rows*n*self.geometry.output_dtype.bytes())])?;
        disjoint(&[accumulator, svh, output])?;
        // SAFETY: extents/devices and nonaliasing checked; caller retains storage
        // and stream until all queued work and captured replays have drained.
        unsafe { self.output.launch(&[accumulator.ptr, svh.ptr, output.ptr],
            &[Scalar::I32(rows as i32), Scalar::I32(self.ctas)], stream)?; }
        Ok(())
    }

    fn rows(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.geometry.rows { return Err(DenseTrellisError::Invalid("live rows")); }
        Ok(())
    }
}

fn validate_manifest(specs: [&ProgramSpec; 3], g: DenseTrellisGeometry, codebook: &str,
    w: DenseTrellisWorkspace) -> Result<()> {
    if !matches!(codebook, "mcg" | "mul1" | "e4m3" | "f16") {
        return Err(DenseTrellisError::Invalid("codebook"));
    }
    let compiled_rows = specs[1].capacity_rows as usize;
    DenseTrellisGeometry { rows: compiled_rows, ..g }.workspace()?;
    let compiled_partial = compiled_rows.checked_mul(g.input.div_ceil(1024))
        .and_then(|v|v.checked_mul(g.output)).and_then(|v|v.checked_mul(4))
        .ok_or(DenseTrellisError::Invalid("manifest scratch overflow"))?;
    for (i, spec) in specs.iter().enumerate() {
        let pointers = [INPUT, CORE, OUTPUT][i];
        if compiled_rows < g.rows || spec.capacity_rows as usize != compiled_rows
            || spec.scalar_kinds.as_slice() != [b"ii".as_slice(), b"iii", b"ii"][i]
            || spec.pointers.iter().map(String::as_str).ne(pointers.iter().copied())
            || spec.geometry["max_rows"].as_u64() != Some(compiled_rows as u64) {
            return Err(DenseTrellisError::Invalid("manifest capacity/pointers/scalars"));
        }
    }
    let input = &specs[0].geometry;
    let core = &specs[1].geometry;
    let output = &specs[2].geometry;
    if !matching_rotated_dtype(input, core)
        || input["features"].as_u64() != Some(g.input as u64) || input["input_dtype"].as_str() != Some(g.input_dtype.name())
        || output["features"].as_u64() != Some(g.output as u64) || output["output_dtype"].as_str() != Some(g.output_dtype.name())
        || !specs[0].scratch.is_empty() || !specs[2].scratch.is_empty() || specs[1].scratch.len() != 1
        || core["codebook"].as_str() != Some(codebook)
        || [("abi_version", 1), ("input_features", g.input), ("output_features", g.output),
            ("bits", g.bits), ("sharded_side", g.side), ("rotation_count", 1),
            ("split_k_blocks", 8), ("splits", g.input.div_ceil(1024)), ("ctas_per_sm", 4)]
            .iter().any(|&(key, value)| core[key].as_u64() != Some(value as u64))
        || w != g.workspace()?
        || specs[1].scratch.get("partial").copied() != Some(compiled_partial as u64) {
        return Err(DenseTrellisError::Invalid("manifest geometry/dtype/scratch"));
    }
    Ok(())
}

/// Validate the same native ABI for offline admission without loading CUDA.
/// The runtime additionally verifies the compiled library's table at load.
pub fn validate_package_geometry(manifest: &serde_json::Value, input: usize, output: usize,
    side: usize, codebook: &str, rows: usize, output_dtype: TrellisDType) -> Result<[String; 3]> {
    validate_package_geometry_bits(manifest, input, output, side, 6, codebook, rows, output_dtype)
}

/// As [`validate_package_geometry`] at an explicit trellis width (K4 MTP stages).
pub fn validate_package_geometry_bits(manifest: &serde_json::Value, input: usize, output: usize,
    side: usize, bits: usize, codebook: &str, rows: usize, output_dtype: TrellisDType) -> Result<[String; 3]> {
    let policy = DenseTrellisPolicy::from_geometry_metadata(&manifest["families"]["dense_trellis_k6"],
        input, output, side, bits, codebook)?;
    policy.validate_rows(rows)?;
    let names = [policy.native_input, policy.native, policy.output_program];
    let entries = manifest["programs"].as_array().ok_or(DenseTrellisError::Invalid("program inventory"))?;
    let specs = names.iter().map(|name| -> Result<ProgramSpec> {
        let mut matches = entries.iter().filter(|p| p["name"].as_str() == Some(name));
        let entry = matches.next().ok_or(DenseTrellisError::Invalid("missing native program"))?;
        if matches.next().is_some() { return Err(DenseTrellisError::Invalid("duplicate native program")); }
        let pointers = entry["pointers"].as_array().ok_or(DenseTrellisError::Invalid("program pointers"))?
            .iter().map(|p| p["name"].as_str().map(str::to_owned)
                .ok_or(DenseTrellisError::Invalid("program pointer name"))).collect::<Result<Vec<_>>>()?;
        let scalar_kinds = entry["scalars"].as_array().ok_or(DenseTrellisError::Invalid("program scalars"))?
            .iter().map(|s| match s["type"].as_str() {
                Some("int32") => Ok(b'i'), Some("int64") => Ok(b'l'), Some("float32") => Ok(b'f'),
                _ => Err(DenseTrellisError::Invalid("program scalar type")),
            }).collect::<Result<Vec<_>>>()?;
        let scratch = entry["scratch_bytes_at_capacity"].as_object()
            .ok_or(DenseTrellisError::Invalid("program scratch"))?.iter().map(|(name, n)|
                n.as_u64().map(|n| (name.clone(), n)).ok_or(DenseTrellisError::Invalid("program scratch extent")))
            .collect::<Result<_>>()?;
        let capacity_rows = entry["capacity_rows"].as_u64().and_then(|n| u32::try_from(n).ok())
            .ok_or(DenseTrellisError::Invalid("program capacity"))?;
        Ok(ProgramSpec { name: name.clone(), pointers, scalar_kinds, scratch, capacity_rows,
            geometry: entry["geometry"].clone() })
    }).collect::<Result<Vec<_>>>()?;
    let geometry = DenseTrellisGeometry { input, output, bits, side, rows,
        input_dtype: TrellisDType::Bf16, output_dtype };
    validate_manifest([&specs[0], &specs[1], &specs[2]], geometry, codebook, geometry.workspace()?)?;
    Ok(names)
}

fn matching_rotated_dtype(input: &serde_json::Value, core: &serde_json::Value) -> bool {
    let input_dtype = input.get("output_dtype").map_or(Some("f16"), serde_json::Value::as_str);
    let core_dtype = core.get("input_dtype").map_or(Some("f16"), serde_json::Value::as_str);
    matches!(input_dtype, Some("f16" | "bf16")) && input_dtype == core_dtype
}

fn buffers(items: &[(Buffer, usize)]) -> Result<()> {
    let device = items[0].0.device_id;
    if device < 0 || items.iter().any(|(b, bytes)| b.ptr.is_null() || b.bytes < *bytes
        || b.device_id != device || b.flags != 0 || b.ptr as usize % 16 != 0
        || (b.ptr as usize).checked_add(b.bytes).is_none()) {
        return Err(DenseTrellisError::Invalid("buffer extent/alignment/device"));
    }
    Ok(())
}
fn disjoint(items: &[Buffer]) -> Result<()> {
    for (i, a) in items.iter().enumerate() {
        for b in &items[..i] {
            if (a.ptr as usize) < b.ptr as usize + b.bytes && (b.ptr as usize) < a.ptr as usize + a.bytes {
                return Err(DenseTrellisError::Invalid("overlapping buffers"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn geometry() -> DenseTrellisGeometry {
        DenseTrellisGeometry { input: 7168, output: 6144, bits: 6, side: 1, rows: 64,
            input_dtype: TrellisDType::Bf16, output_dtype: TrellisDType::F32 }
    }
    fn specs(g: DenseTrellisGeometry) -> [ProgramSpec; 3] {
        let spec = |names: &[&str], kinds: &[u8], geometry| ProgramSpec { name: "fixture".into(),
            pointers: names.iter().map(|s| s.to_string()).collect(), scalar_kinds: kinds.to_vec(),
            scratch: HashMap::new(), capacity_rows: g.rows as u32, geometry };
        let input = spec(INPUT, b"ii", json!({"features":g.input,"input_dtype":"bf16","max_rows":g.rows}));
        let mut core = spec(CORE, b"iii", json!({"abi_version":1,"input_features":g.input,
            "output_features":g.output,"bits":g.bits,"codebook":"mcg","sharded_side":g.side,
            "rotation_count":1,"max_rows":g.rows,"split_k_blocks":8,"splits":7,"ctas_per_sm":4}));
        core.scratch.insert("partial".into(), (g.rows*g.input.div_ceil(1024)*g.output*4) as u64);
        let output = spec(OUTPUT, b"ii", json!({"features":g.output,"output_dtype":"f32","max_rows":g.rows}));
        [input, core, output]
    }
    #[test]
    fn scratch_and_manifest_contract_preserve_f32_logits() {
        let g = geometry();
        let w = g.workspace().unwrap();
        assert_eq!(w, DenseTrellisWorkspace { rotated: 917504, pointer_table: 8,
            block_scales: 112, partial: 11010048, accumulator: 1572864 });
        let mut s = specs(g);
        validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", w).unwrap();
        s[2].geometry["output_dtype"] = json!("bf16");
        assert!(validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", w).is_err());
    }
    #[test]
    fn prefill_workspace_bounds_narrow_split_k_storage() {
        for rows in [64,129,130,2048,4096] {
            let g=DenseTrellisGeometry {rows,..geometry()};
            assert_eq!(g.workspace().unwrap().partial,rows.min(TRELLIS_PARTIAL_ROWS)*7*6144*4);
        }
    }
    #[test]
    fn decode_workspace_uses_live_ceiling_not_wide_compiled_capacity() {
        let g = geometry();
        let wide = DenseTrellisGeometry { rows: 4096, ..g };
        let mut s = specs(wide);
        s[1].scratch.insert("partial".into(), (wide.rows*wide.input.div_ceil(1024)*wide.output*4) as u64);
        assert!(validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", g.workspace().unwrap()).is_ok());
        assert!(g.workspace().unwrap().partial < wide.workspace().unwrap().partial);
        s[0].capacity_rows = 64;
        assert!(validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", g.workspace().unwrap()).is_err());
    }
    #[test]
    fn stale_packages_and_bad_geometry_fail_before_launch() {
        let g = geometry();
        for key in ["abi_version", "bits", "rotation_count", "splits", "sharded_side"] {
            let mut s = specs(g);
            s[1].geometry[key] = json!(999);
            assert!(validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", g.workspace().unwrap()).is_err(), "{key}");
        }
        let mut wide_input = specs(g);
        wide_input[0].geometry["output_dtype"] = json!("bf16");
        assert!(validate_manifest([&wide_input[0], &wide_input[1], &wide_input[2]], g, "mcg", g.workspace().unwrap()).is_err());
        let mut bad = g;
        bad.output = 576;
        assert!(bad.workspace().is_err());
        bad.output = 640;
        assert!(bad.workspace().is_ok());
        bad.input = usize::MAX;
        assert!(bad.workspace().is_err());
    }
    #[test]
    fn rotated_dtype_is_validated_between_input_and_core() {
        let g = geometry();
        for dtype in ["f16", "bf16"] {
            let mut s = specs(g);
            s[0].geometry["output_dtype"] = json!(dtype);
            s[1].geometry["input_dtype"] = json!(dtype);
            validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", g.workspace().unwrap()).unwrap();
            s[1].geometry["input_dtype"] = json!(if dtype == "f16" { "bf16" } else { "f16" });
            assert!(validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", g.workspace().unwrap()).is_err());
            for invalid in [json!("f32"), json!(null), json!(17)] {
                s[0].geometry["output_dtype"] = invalid.clone();
                s[1].geometry["input_dtype"] = invalid;
                assert!(validate_manifest([&s[0], &s[1], &s[2]], g, "mcg", g.workspace().unwrap()).is_err());
            }
        }
    }
    #[test]
    fn rejects_aliases_short_buffers_and_wrong_devices() {
        let b = |at, bytes, device_id| Buffer { ptr: at as *mut c_void, bytes, device_id, flags: 0 };
        let a = b(0x1000, 128, 0);
        let c = b(0x1080, 128, 0);
        buffers(&[(a,128),(c,128)]).unwrap();
        disjoint(&[a,c]).unwrap();
        assert!(disjoint(&[a,b(0x1040,128,0)]).is_err());
        assert!(buffers(&[(a,129)]).is_err());
        assert!(buffers(&[(a,128),(b(0x1080,128,1),128)]).is_err());
        assert!(buffers(&[(b(usize::MAX - 15, 32, 0),16)]).is_err());
        assert!(buffers(&[(Buffer { flags: 1, ..a },128)]).is_err());
    }
}

/// Validated package policy, not a model-name-based routing heuristic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTrellisPolicy {
    pub shape: String, pub input: usize, pub output: usize, pub side: usize,
    /// Trellis width; the K6 tables predate the key and omit it.
    pub bits: usize,
    pub codebook: String, pub native: String, pub native_input: String, pub output_program: String,
}
impl DenseTrellisPolicy {
    pub fn parse(value: &serde_json::Value) -> Result<Self> {
        let integer = |key: &str| value[key].as_u64().and_then(|v| usize::try_from(v).ok())
            .ok_or(DenseTrellisError::Invalid("row-policy integer"));
        let text = |key: &str| value[key].as_str().filter(|v| !v.is_empty()).map(str::to_owned)
            .ok_or(DenseTrellisError::Invalid("row-policy string"));
        let policy=Self { shape:text("shape")?, input:integer("input_features")?, output:integer("output_features")?,
            side:integer("sharded_side")?,
            bits:if value.get("bits").is_none() { 6 } else { integer("bits")? }, codebook:text("codebook")?, native:text("native_program")?,
            native_input:text("native_input")?, output_program:text("output_program")? };
        policy.validate()?;
        Ok(policy)
    }
    fn validate(&self) -> Result<()> {
        if self.input==0 || self.input%128!=0 || self.output==0 || self.output%128!=0 || !matches!(self.side,1|2)
            || !matches!(self.bits,2..=6)
            || !matches!(self.codebook.as_str(),"mcg"|"mul1")
            || [&self.shape,&self.native,&self.native_input,&self.output_program].iter().any(|name|name.is_empty()) {
            return Err(DenseTrellisError::Invalid("row-policy geometry/route/scratch"));
        }
        Ok(())
    }
    pub fn from_package(programs: &Programs<'_>, shape: &str, codebook: &str) -> Result<Self> {
        Self::from_metadata(programs.family_metadata("dense_trellis_k6")?,shape,codebook)
    }
    fn from_metadata(family:&serde_json::Value,shape:&str,codebook:&str)->Result<Self> {
        Self::select_metadata(family, |p| p["shape"].as_str()==Some(shape) && p["codebook"].as_str()==Some(codebook))
    }
    /// Projection labels do not affect arithmetic. In particular, MLA TP2 g/o
    /// share the KDA TP2 geometries; select exactly one policy, never the first.
    pub fn from_package_geometry(programs:&Programs<'_>,input:usize,output:usize,side:usize,codebook:&str)->Result<Self> {
        Self::from_package_geometry_bits(programs,input,output,side,6,codebook)
    }
    pub fn from_package_geometry_bits(programs:&Programs<'_>,input:usize,output:usize,side:usize,bits:usize,
        codebook:&str)->Result<Self> {
        Self::from_geometry_metadata(programs.family_metadata("dense_trellis_k6")?,input,output,side,bits,codebook)
    }
    fn from_geometry_metadata(family:&serde_json::Value,input:usize,output:usize,side:usize,bits:usize,
        codebook:&str)->Result<Self> {
        Self::select_metadata(family, |p| p["input_features"].as_u64()==Some(input as u64)
            && p["output_features"].as_u64()==Some(output as u64)
            && p["sharded_side"].as_u64()==Some(side as u64) && p["codebook"].as_str()==Some(codebook)
            && p.get("bits").map_or(Some(6), serde_json::Value::as_u64)==Some(bits as u64))
            .map_err(|error| match error {
                DenseTrellisError::Invalid("missing row policy") => DenseTrellisError::MissingGeometry {
                    input, output, side, codebook: codebook.to_owned(), bits,
                },
                other => other,
            })
    }
    fn select_metadata(family:&serde_json::Value,select:impl Fn(&serde_json::Value)->bool)->Result<Self> {
        if family["abi_version"].as_u64()!=Some(1) { return Err(DenseTrellisError::Invalid("row-policy ABI")); }
        let entries=family["row_policies"].as_array().ok_or(DenseTrellisError::Invalid("row-policy list"))?;
        let mut matches=entries.iter().filter(|p|select(p));
        let selected=matches.next().ok_or(DenseTrellisError::Invalid("missing row policy"))?;
        if matches.next().is_some() { return Err(DenseTrellisError::Invalid("duplicate row policy")); }
        let mut policy=Self::parse(selected)?;
        for name in [&mut policy.native,&mut policy.native_input,&mut policy.output_program] {
            *name=format!("dense_trellis_k6_{name}");
        }
        Ok(policy)
    }
    pub fn validate_rows(&self, rows: usize) -> Result<()> {
        if !(1..=4096).contains(&rows) { return Err(DenseTrellisError::Invalid("row-policy live rows")); }
        Ok(())
    }
}

/// Programs are resolved before capture. Scratch is caller-owned and reusable
/// only on the serialized lane; the native core writes F32 before one egress.
pub struct DenseTrellisDispatch<'a> {
    native: DenseTrellis<'a>, policy: DenseTrellisPolicy,
}
#[derive(Debug, Clone, Copy)]
pub struct DenseTrellisBuffers {
    pub input: Buffer, pub payload: Buffer, pub suh: Buffer, pub svh: Buffer,
    pub rotated: Buffer, pub pointer_table: Buffer, pub scales: Buffer, pub partial: Buffer,
    pub accumulator: Buffer,
}
impl<'a> DenseTrellisDispatch<'a> {
    pub fn load(programs: &'a Programs<'a>, policy: DenseTrellisPolicy, rows: usize,
        output_dtype: TrellisDType) -> Result<Self> {
        policy.validate()?;
        policy.validate_rows(rows)?;
        let native=DenseTrellis::load(programs,[&policy.native_input,&policy.native,&policy.output_program],
            DenseTrellisGeometry{input:policy.input,output:policy.output,bits:policy.bits,side:policy.side,rows,
                input_dtype:TrellisDType::Bf16,output_dtype},&policy.codebook)?;
        Ok(Self {native,policy})
    }
    pub fn workspace(&self) -> DenseTrellisWorkspace { self.native.workspace() }
    pub fn policy(&self) -> &DenseTrellisPolicy { &self.policy }
    /// Emit only F32 core values. K-split callers must reduce these values before egress.
    pub fn core(&self, rows: usize, b: DenseTrellisBuffers, stream: *mut c_void) -> Result<()> {
        self.native.rows(rows)?;
        let p=&self.policy;
        buffers(&[(b.input,rows*p.input*2),(b.suh,p.input*2),(b.rotated,rows*p.input*2),
            (b.pointer_table,8),(b.payload,p.input*p.output*p.bits/8),(b.scales,p.input/128*2),
            (b.partial,rows.min(TRELLIS_PARTIAL_ROWS)*p.input.div_ceil(1024)*p.output*4),
            (b.accumulator,rows*p.output*4)])?;
        disjoint(&[b.input,b.suh,b.rotated,b.pointer_table,b.payload,b.scales,b.partial,b.accumulator])?;
        self.native.input(rows,b.input,b.suh,b.rotated,b.scales,stream)?;
        self.native.core(rows,b.pointer_table,b.payload,b.scales,b.partial,b.accumulator,stream)
    }
    pub fn output(&self, rows: usize, accumulator: Buffer, svh: Buffer, output: Buffer,
        stream: *mut c_void) -> Result<()> { self.native.output(rows,accumulator,svh,output,stream) }
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    fn policy() -> serde_json::Value {
        serde_json::json!({"shape":"test","input_features":128,"output_features":256,"sharded_side":2,
            "codebook":"mcg","native_program":"core","native_input":"input","output_program":"output"})
    }
    #[test]
    fn package_resolves_prefixed_names_and_rejects_duplicate_policy() {
        let mut family=serde_json::json!({"abi_version":1,"row_policies":[policy()]});
        let p=DenseTrellisPolicy::from_metadata(&family,"test","mcg").unwrap();
        assert_eq!(p.native,"dense_trellis_k6_core");
        assert_eq!(p.output_program,"dense_trellis_k6_output");
        assert!(DenseTrellisPolicy::from_metadata(&family,"test","mul1").is_err());
        family["row_policies"].as_array_mut().unwrap().push(policy());
        assert!(DenseTrellisPolicy::from_metadata(&family,"test","mcg").is_err());
    }
    #[test]
    fn mla_tp2_gate_output_reuse_exact_kda_geometry_and_reject_ambiguity() {
        let make=|shape:&str,k:usize,n:usize,side:usize,book:&str| {
            let mut p=policy();
            for (key,value) in [("shape",serde_json::json!(shape)),("input_features",serde_json::json!(k)),
                ("output_features",serde_json::json!(n)),("sharded_side",serde_json::json!(side)),
                ("codebook",serde_json::json!(book)),
                ("native_program",serde_json::json!(format!("trellis_{shape}_{book}")))] {p[key]=value;}
            p
        };
        let mut entries=Vec::new();
        for book in ["mcg","mul1"] {
            for (shape,k,n,side) in [("kda_qkvg_tp2",7168,6144,1),("kda_o_tp2",6144,7168,2),
                ("mla_g_owned",7168,12288,1),("mla_o_owned",12288,7168,1)] {
                entries.push(make(shape,k,n,side,book));
            }
        }
        let mut family=serde_json::json!({"abi_version":1,"row_policies":entries});
        for book in ["mcg","mul1"] {
            for (k,n,side,name) in [(7168,6144,1,"kda_qkvg_tp2"),(6144,7168,2,"kda_o_tp2")] {
                let resolved=DenseTrellisPolicy::from_geometry_metadata(&family,k,n,side,6,book).unwrap();
                assert_eq!(resolved.shape,name);
                assert_eq!(resolved.native,format!("dense_trellis_k6_trellis_{name}_{book}"));
            }
        }
        assert!(DenseTrellisPolicy::from_geometry_metadata(&family,6144,7168,1,6,"mcg").is_err());
        assert!(DenseTrellisPolicy::from_geometry_metadata(&family,7168,6144,1,6,"f16").is_err());
        family["row_policies"].as_array_mut().unwrap().push(make("mla_g_tp2",7168,6144,1,"mcg"));
        assert!(matches!(DenseTrellisPolicy::from_geometry_metadata(&family,7168,6144,1,6,"mcg"),
            Err(DenseTrellisError::Invalid("duplicate row policy"))));
        family["abi_version"]=serde_json::json!(2);
        assert!(DenseTrellisPolicy::from_geometry_metadata(&family,6144,7168,2,6,"mul1").is_err());
    }
    #[test]
    fn missing_geometry_names_native_shape_without_fallback() {
        let family=serde_json::json!({"abi_version":1,"row_policies":[policy()]});
        let error=DenseTrellisPolicy::from_geometry_metadata(&family,6144,576,1,6,"mcg").unwrap_err();
        assert!(matches!(&error, DenseTrellisError::MissingGeometry {
            input:6144,output:576,side:1,codebook,bits:6,
        } if codebook=="mcg"));
        let message=error.to_string();
        for field in ["dense_trellis_k6", "K6", "N=576", "K=6144", "side=1", "codebook=mcg"] {
            assert!(message.contains(field), "{message}");
        }
        let mut invalid=family;
        invalid["abi_version"]=serde_json::json!(2);
        assert!(matches!(DenseTrellisPolicy::from_geometry_metadata(&invalid,6144,576,1,6,"mcg"),
            Err(DenseTrellisError::Invalid("row-policy ABI"))));
    }
    #[test]
    fn native_policy_accepts_all_live_rows_and_validates_geometry() {
        let p=DenseTrellisPolicy::parse(&policy()).unwrap();
        for rows in [1,8,63,64,129,130,4096] {p.validate_rows(rows).unwrap();}
        assert!(p.validate_rows(0).is_err());assert!(p.validate_rows(4097).is_err());
        for key in ["input_features","output_features","sharded_side"] {
            let mut value=policy();value[key]=serde_json::json!(0);
            assert!(DenseTrellisPolicy::parse(&value).is_err(),"{key}");
        }
    }

}

fn reduced_egress<T>(reduce: impl FnOnce()->Result<T>,egress:impl FnOnce(T)->Result<()>) -> Result<()> {
    egress(reduce()?)
}
impl DenseTrellisDispatch<'_> {
    /// Both rank partials are retained/published as BF16. Widen before the
    /// rank-ordered F32 add, then perform exactly one H128/svh egress.
    pub fn reduce_output(&self,add:&crate::Bf16F32Add<'_>,rows:usize,rank0:Buffer,rank1:Buffer,
        reduced:Buffer,svh:Buffer,output:Buffer,stream:*mut c_void)->Result<()> {
        self.native.rows(rows)?;
        if self.policy.side!=2 { return Err(DenseTrellisError::Invalid("K-split reduction side")); }
        reduced_egress(|| {
            // SAFETY: Bf16F32Add validates disjoint extents/devices; the caller
            // publishes peer input first and retains all owners through drain.
            unsafe { add.launch(rank0,rank1,reduced,rows*self.policy.output,stream)?; }
            Ok(reduced)
        },|sum|self.output(rows,sum,svh,output,stream))
    }
}
#[cfg(test)]
mod reduction_tests {
    use super::*;
    #[test]
    fn f32_reduce_precedes_exactly_one_egress_and_failure_never_narrows() {
        let order=std::cell::RefCell::new(Vec::new());
        reduced_egress(||{order.borrow_mut().push("F32 add");Ok(())},|_|{order.borrow_mut().push("BF16 egress");Ok(())}).unwrap();
        assert_eq!(*order.borrow(),["F32 add","BF16 egress"]);
        order.borrow_mut().clear();
        assert!(reduced_egress(||->Result<()>{Err(DenseTrellisError::Invalid("failed reduction"))},|_|{
            order.borrow_mut().push("BF16 egress");Ok(())}).is_err());
        assert!(order.borrow().is_empty());
    }
}

/// Package-owned grouped launch for projections sharing one activation and
/// geometry (Kimi KDA q/k/v/g). Each member keeps its own payload, suh/svh and
/// block scales; one input, core and output launch cover all members, and
/// each member's result is bitwise-equal to its standalone native launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedPolicy { pub groups: usize, pub max_rows: usize,
    pub native: String, pub native_input: String, pub output_program: String }
impl GroupedPolicy {
    pub fn parse(value: &serde_json::Value) -> Result<Option<Self>> {
        let Some(grouped) = value.get("grouped") else { return Ok(None) };
        let integer = |key: &str| grouped[key].as_u64().and_then(|v| usize::try_from(v).ok())
            .ok_or(DenseTrellisError::Invalid("grouped integer"));
        let text = |key: &str| grouped[key].as_str().filter(|v| !v.is_empty())
            .map(|name| format!("dense_trellis_k6_{name}")).ok_or(DenseTrellisError::Invalid("grouped program"));
        let policy = Self { groups: integer("groups")?, max_rows: integer("max_rows")?,
            native: text("native_program")?, native_input: text("native_input")?, output_program: text("output_program")? };
        if policy.groups != GROUPS || policy.max_rows != 4096 {
            return Err(DenseTrellisError::Invalid("grouped groups/rows"));
        }
        Ok(Some(policy))
    }
}

impl GroupedPolicy {
    /// The grouped launch of the single policy matching this exact geometry, if any.
    pub fn from_package_geometry(programs: &Programs<'_>, input: usize, output: usize, side: usize,
        codebook: &str) -> Result<Option<Self>> {
        let family = programs.family_metadata("dense_trellis_k6")?;
        let entries = family["row_policies"].as_array().ok_or(DenseTrellisError::Invalid("row-policy list"))?;
        let mut matches = entries.iter().filter(|p| p["input_features"].as_u64() == Some(input as u64)
            && p["output_features"].as_u64() == Some(output as u64)
            && p["sharded_side"].as_u64() == Some(side as u64) && p["codebook"].as_str() == Some(codebook)
            && p.get("bits").is_none_or(|bits| bits == 6));
        let selected = matches.next().ok_or_else(|| DenseTrellisError::MissingGeometry {
            input, output, side, codebook: codebook.to_owned(), bits: 6,
        })?;
        if matches.next().is_some() { return Err(DenseTrellisError::Invalid("duplicate row policy")); }
        Self::parse(selected)
    }
}

pub const GROUPS: usize = 4;
const GROUPED_INPUT: &[&str] = &["input", "q_suh", "k_suh", "v_suh", "g_suh", "q_rotated", "k_rotated",
    "v_rotated", "g_rotated", "q_input_block_scales", "k_input_block_scales", "v_input_block_scales",
    "g_input_block_scales"];
const GROUPED_CORE: &[&str] = &["rotated_inputs", "q_payload", "k_payload", "v_payload", "g_payload",
    "q_input_block_scales", "k_input_block_scales", "v_input_block_scales", "g_input_block_scales",
    "partial", "q_output", "k_output", "v_output", "g_output"];
const GROUPED_OUTPUT: &[&str] = &["q_input", "k_input", "v_input", "g_input", "q_svh", "k_svh", "v_svh",
    "g_svh", "q_output", "k_output", "v_output", "g_output"];

/// Minimum grouped scratch at `rows` live rows. Callers place each
/// member in a fixed quarter of the lane's (larger, merged) scratch owners, so
/// the pointer table and captured graphs never depend on live rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedWorkspace {
    pub rotated: usize, pub pointer_table: usize, pub block_scales: usize,
    pub partial: usize, pub accumulator: usize,
}
impl GroupedWorkspace {
    pub fn new(input: usize, output: usize, rows: usize) -> Result<Self> {
        if !(1..=4096).contains(&rows) { return Err(DenseTrellisError::Invalid("grouped rows")); }
        let single = DenseTrellisGeometry { input, output, bits: 6, side: 1, rows,
            input_dtype: TrellisDType::Bf16, output_dtype: TrellisDType::Bf16 }.workspace()?;
        let times = |bytes: usize| bytes.checked_mul(GROUPS).ok_or(DenseTrellisError::Invalid("grouped overflow"));
        Ok(Self { rotated: times(single.rotated)?, pointer_table: GROUPS * 8,
            block_scales: times(single.block_scales)?, partial: times(single.partial)?,
            accumulator: times(single.accumulator)? })
    }
    /// Member `index`'s fixed quarter of a grouped owner.
    pub fn slice(buffer: Buffer, index: usize) -> Result<Buffer> {
        if index >= GROUPS || buffer.bytes % (GROUPS * 16) != 0 {
            return Err(DenseTrellisError::Invalid("grouped slice"));
        }
        let each = buffer.bytes / GROUPS;
        // SAFETY: offset stays inside the parent extent.
        Ok(Buffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(index * each).cast() }, bytes: each, ..buffer })
    }
    /// Little-endian pointer table naming each member's rotated quarter.
    pub fn pointer_table(rotated: Buffer) -> Result<Vec<u8>> {
        (0..GROUPS).map(|i| Self::slice(rotated, i).map(|b| (b.ptr as u64).to_le_bytes()))
            .collect::<Result<Vec<_>>>().map(|v| v.concat())
    }
}

/// One member's resident storage (payload and its own calibration vectors).
#[derive(Debug, Clone, Copy)]
pub struct GroupedMember { pub payload: Buffer, pub suh: Buffer, pub svh: Buffer, pub output: Buffer }

pub struct GroupedTrellis<'a> {
    input: Program<'a>, core: Program<'a>, output: Program<'a>,
    k: usize, n: usize, ctas: i32,
}
impl<'a> GroupedTrellis<'a> {
    /// Resolved before capture beside the member's standalone dispatch.
    pub fn load(programs: &'a Programs<'a>, base: &DenseTrellisPolicy, grouped: &GroupedPolicy) -> Result<Self> {
        let (k, n) = (base.input, base.output);
        let [input, core, output] = [&grouped.native_input, &grouped.native, &grouped.output_program]
            .map(|name| programs.spec(name));
        let (input, core, output) = (input?, core?, output?);
        let splits = k.div_ceil(1024);
        let capacity = grouped.max_rows as u32;
        if base.side != 1 || [input, core, output].iter().any(|spec| spec.capacity_rows != capacity
            || spec.geometry["max_rows"].as_u64() != Some(capacity as u64)
            || spec.geometry["groups"].as_u64() != Some(GROUPS as u64))
            || input.scalar_kinds != b"ii" || core.scalar_kinds != b"iii" || output.scalar_kinds != b"ii"
            || !input.scratch.is_empty() || !output.scratch.is_empty()
            || input.geometry["features"].as_u64() != Some(k as u64) || input.geometry["input_dtype"].as_str() != Some("bf16")
            || !matching_rotated_dtype(&input.geometry, &core.geometry)
            || output.geometry["features"].as_u64() != Some(n as u64) || output.geometry["output_dtype"].as_str() != Some("bf16")
            || core.geometry["codebook"].as_str() != Some(base.codebook.as_str())
            || [("abi_version", 1), ("input_features", k), ("output_features", n), ("bits", 6), ("sharded_side", 1),
                ("rotation_count", GROUPS), ("split_k_blocks", 8), ("splits", splits), ("ctas_per_sm", 4)]
                .iter().any(|&(key, value)| core.geometry[key].as_u64() != Some(value as u64))
            || core.scratch.get("partial").copied() != Some((GROUPS * grouped.max_rows * splits * n * 4) as u64) {
            return Err(DenseTrellisError::Invalid("grouped manifest geometry/ABI"));
        }
        let ctas = programs.library().sm_count()?.checked_mul(4)
            .and_then(|v| i32::try_from(v).ok()).filter(|&v| v > 0)
            .ok_or(DenseTrellisError::Invalid("runtime SM count"))?;
        Ok(Self { input: programs.program(&grouped.native_input, GROUPED_INPUT)?,
            core: programs.program(&grouped.native, GROUPED_CORE)?,
            output: programs.program(&grouped.output_program, GROUPED_OUTPUT)?, k, n, ctas })
    }

    pub fn geometry(&self) -> (usize, usize) { (self.k, self.n) }

    /// Input H128 (per-member suh), grouped core, then per-member svh egress.
    /// `pointer_table` must already hold [`GroupedWorkspace::pointer_table`] of
    /// `rotated`; rotated, scales and accumulator are used by fixed quarters.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, rows: usize, input: Buffer, members: [GroupedMember; GROUPS],
        rotated: Buffer, pointer_table: Buffer, scales: Buffer, partial: Buffer, accumulator: Buffer,
        stream: *mut c_void) -> Result<()> {
        let (k, n) = (self.k, self.n);
        let w = GroupedWorkspace::new(k, n, rows)?;
        let mut checked = vec![(input, rows*k*2), (rotated, w.rotated), (pointer_table, w.pointer_table),
            (scales, w.block_scales), (partial, w.partial), (accumulator, w.accumulator)];
        for m in &members {
            checked.extend([(m.payload, k*n*6/8), (m.suh, k*2), (m.svh, n*2), (m.output, rows*n*2)]);
        }
        buffers(&checked)?;
        disjoint(&checked.iter().map(|(b, _)| *b).collect::<Vec<_>>())?;
        let quarters = |buffer: Buffer, need: usize| -> Result<Vec<Buffer>> {
            let parts = (0..GROUPS).map(|i| GroupedWorkspace::slice(buffer, i)).collect::<Result<Vec<_>>>()?;
            if parts[0].bytes < need { return Err(DenseTrellisError::Invalid("grouped quarter extent")); }
            Ok(parts)
        };
        let rot = quarters(rotated, rows*k*2)?;
        let sc = quarters(scales, k/128*2)?;
        let acc = quarters(accumulator, rows*n*4)?;
        let scalars = [Scalar::I32(rows as i32), Scalar::I32(self.ctas)];
        // SAFETY: every member extent, device, alignment and nonaliasing is
        // checked above; quarters are fixed offsets inside checked owners. The
        // caller retains all storage and the table through completion/replay.
        unsafe {
            let mut p = vec![input.ptr];
            p.extend(members.iter().map(|m| m.suh.ptr));
            p.extend(rot.iter().map(|b| b.ptr));
            p.extend(sc.iter().map(|b| b.ptr));
            self.input.launch(&p, &scalars, stream)?;
            let mut p = vec![pointer_table.ptr];
            p.extend(members.iter().map(|m| m.payload.ptr));
            p.extend(sc.iter().map(|b| b.ptr));
            p.push(partial.ptr);
            p.extend(acc.iter().map(|b| b.ptr));
            self.core.launch(&p, &[Scalar::I32(rows as i32), Scalar::I32(0), Scalar::I32(self.ctas)], stream)?;
            let mut p = acc.iter().map(|b| b.ptr).collect::<Vec<_>>();
            p.extend(members.iter().map(|m| m.svh.ptr));
            p.extend(members.iter().map(|m| m.output.ptr));
            self.output.launch(&p, &scalars, stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod grouped_tests {
    use super::*;
    #[test]
    fn grouped_policy_and_workspace_use_fixed_capacity_slices() {
        let value = serde_json::json!({"grouped":{"groups":4,"max_rows":4096,"native_program":"trellis_kda_qkvg_tp2_mcg_g4",
            "native_input":"trellis_input_k7168_bf16_g4","output_program":"trellis_output_n6144_bf16_g4"}});
        let p = GroupedPolicy::parse(&value).unwrap().unwrap();
        assert_eq!(p.native, "dense_trellis_k6_trellis_kda_qkvg_tp2_mcg_g4");
        assert!(GroupedPolicy::parse(&serde_json::json!({})).unwrap().is_none());
        for (key, bad) in [("groups", 2), ("max_rows", 64)] {
            let mut v = value.clone(); v["grouped"][key] = serde_json::json!(bad);
            assert!(GroupedPolicy::parse(&v).is_err(), "{key}");
        }
        let w = GroupedWorkspace::new(7168, 6144, 63).unwrap();
        assert_eq!((w.rotated, w.pointer_table, w.block_scales, w.partial, w.accumulator),
            (4*63*7168*2, 32, 4*56*2, 4*63*7*6144*4, 4*63*6144*4));
        assert!(GroupedWorkspace::new(7168, 6144, 4097).is_err());
        assert_eq!(GroupedWorkspace::new(7168,6144,4096).unwrap().partial,4*129*7*6144*4);
        let b = Buffer { ptr: 0x10000 as *mut c_void, bytes: w.rotated, device_id: 0, flags: 0 };
        let s = GroupedWorkspace::slice(b, 3).unwrap();
        assert_eq!((s.ptr as usize, s.bytes), (0x10000 + 3*63*7168*2, 63*7168*2));
        assert!(GroupedWorkspace::slice(b, 4).is_err());
        assert!(GroupedWorkspace::slice(Buffer { bytes: 72, ..b }, 0).is_err());
        let table = GroupedWorkspace::pointer_table(b).unwrap();
        assert_eq!(table.len(), 32);
        assert_eq!(u64::from_le_bytes(table[..8].try_into().unwrap()), 0x10000);
        assert_eq!(u64::from_le_bytes(table[24..].try_into().unwrap()), (0x10000 + 3*63*7168*2) as u64);
    }
}

#[cfg(test)]
mod bits_tests {
    use super::*;
    /// One geometry at K6 (target core) and K4 (MTP) resolves by width; the K6
    /// tables written before the `bits` key keep resolving as K6.
    #[test]
    fn row_policy_selects_trellis_width_and_k6_tables_default() {
        let entry = |bits: Option<u64>, program: &str| {
            let mut p = serde_json::json!({"shape":program,"input_features":4096,"output_features":12288,
                "sharded_side":1,"codebook":"mcg","native_program":program,"native_input":"in","output_program":"out"});
            if let Some(bits) = bits { p["bits"] = serde_json::json!(bits); }
            p
        };
        let family = serde_json::json!({"abi_version":1,"row_policies":[entry(None,"q"), entry(Some(4),"mtp_q")]});
        let k6 = DenseTrellisPolicy::from_geometry_metadata(&family,4096,12288,1,6,"mcg").unwrap();
        let k4 = DenseTrellisPolicy::from_geometry_metadata(&family,4096,12288,1,4,"mcg").unwrap();
        assert_eq!((k6.bits, k6.shape.as_str(), k4.bits, k4.shape.as_str()), (6, "q", 4, "mtp_q"));
        let error = DenseTrellisPolicy::from_geometry_metadata(&family,4096,12288,1,5,"mcg").unwrap_err();
        assert!(error.to_string().contains("K5"), "{error}");
        let k4_geometry = DenseTrellisGeometry { input: 4096, output: 12288, bits: 4, side: 1, rows: 64,
            input_dtype: TrellisDType::Bf16, output_dtype: TrellisDType::Bf16 };
        // Workspace depends only on K/N/rows; payload extents scale with width.
        assert_eq!(k4_geometry.workspace().unwrap(), DenseTrellisGeometry { bits: 6, ..k4_geometry }.workspace().unwrap());
        assert!(DenseTrellisGeometry { bits: 7, ..k4_geometry }.workspace().is_err());
    }
}
