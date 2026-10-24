//! Typed bindings for the EXLR8 native dense companions beside the ABI-v1
//! linears (`native_dense`): trellis H128 input/output transforms, the exact
//! E4M3 trellis unpack and its block-FP8 W8A16 prefill GEMM (unit grid, F32
//! out), the split-gate SwiGLU, F32 -> BF16 column packing and the indexer
//! weights projection. Every contract is checked from the manifest before any
//! kernel loads; launches take caller-owned device storage.
use super::programs::{Program, ProgramSpec, Programs, Scalar};
use std::ffi::c_void;

#[derive(Debug, thiserror::Error)]
#[error("invalid native dense companion: {0}")]
pub struct GlueError(String);

type Result<T> = std::result::Result<T, GlueError>;

fn invalid(message: impl Into<String>) -> GlueError {
    GlueError(message.into())
}

/// A companion program's kind (the manifest's `op`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    TrellisInput { features: u32, rotated_bf16: bool },
    TrellisOutput { features: u32 },
    Mx8Quant { features: u32 },
    Unpack { input: u32, output: u32, side: u32 },
    Prefill { input: u32, output: u32 },
    SwigluSplit { inter: u32 },
    NarrowPack { width: u32, pitch: u32 },
    IndexWeights { width: u32, outputs: u32 },
}

impl Op {
    fn contract(self) -> (Vec<(&'static str, &'static str, String, &'static str)>, &'static [&'static str]) {
        match self {
            Op::TrellisInput { features: k, rotated_bf16 } => (vec![
                ("input", "bfloat16", format!("[rows,{k}]"), "in"),
                ("suh", "float16", format!("[{k}]"), "in"),
                ("rotated", if rotated_bf16 { "bfloat16" } else { "float16" }, format!("[rows,{k}]"), "out"),
                ("input_block_scales", "float16", format!("[{}]", k / 128), "out")], &["rows", "cta_limit"]),
            Op::TrellisOutput { features: n } => (vec![
                ("input", "float32", format!("[rows,{n}]"), "in"),
                ("svh", "float16", format!("[{n}]"), "in"),
                ("output", "bfloat16", format!("[rows,{n}]"), "out")], &["rows", "cta_limit"]),
            Op::Mx8Quant { .. } => (vec![
                ("rotated_inputs", "int64", "[4]".into(), "in"),
                ("records", "uint8", "[record_bytes]".into(), "out")], &["rows", "rotation_id", "cta_limit"]),
            Op::Unpack { input: k, output: n, .. } => (vec![
                ("payload", "uint8", format!("[{}]", u64::from(k) * u64::from(n) / 2), "in"),
                ("output", "uint8", format!("[{n},{k}]"), "out")], &[]),
            Op::Prefill { input: k, output: n } => (vec![
                ("x", "bfloat16", format!("[rows,{k}]"), "in"),
                ("w", "uint8", format!("[{n},{k}]"), "in"),
                ("scale", "float32", format!("[{},{}]", n.div_ceil(128), k / 128), "in"),
                ("out", "float32", format!("[rows,{n}]"), "out")], &["rows"]),
            Op::SwigluSplit { inter } => (vec![
                ("gate", "bfloat16", format!("[rows,{inter}]"), "in"),
                ("up", "bfloat16", format!("[rows,{inter}]"), "in"),
                ("hidden", "bfloat16", format!("[rows,{inter}]"), "out")], &["rows"]),
            Op::NarrowPack { width, pitch } => (vec![
                ("input", "float32", format!("[rows,{width}]"), "in"),
                ("output", "bfloat16", format!("[rows,{pitch}]"), "out")], &["rows", "c0"]),
            Op::IndexWeights { width, outputs } => (vec![
                ("x", "bfloat16", format!("[rows,{width}]"), "in"),
                ("w", "bfloat16", format!("[{outputs},{width}]"), "in"),
                ("out", "float32", format!("[rows,{outputs}]"), "out")], &["rows"]),
        }
    }
}

fn u32_of(g: &serde_json::Value, key: &str) -> Result<u32> {
    g[key].as_u64().and_then(|v| u32::try_from(v).ok()).filter(|&v| v > 0)
        .ok_or_else(|| invalid(format!("geometry.{key}")))
}

/// The companion kind of manifest entry `entry` (`op` and geometry), or None
/// when it is not a companion (the ABI-v1 linears validate in `native_dense`).
pub fn op_of(entry: &serde_json::Value) -> Result<Option<Op>> {
    let g = &entry["geometry"];
    Ok(Some(match entry["op"].as_str() {
        Some("dense_mxfp8_quant") => {
            let features = u32_of(g, "input_features")?;
            if ![2048, 6144, 8192, 12288, 16384].contains(&features)
                || g["input_dtype"] != "f16" || g["rotation_count"] != 4 || g["max_rows"] != 4096 {
                return Err(invalid("MX8 quant geometry"));
            }
            Op::Mx8Quant { features }
        }
        Some("trellis_input") => {
            if g["input_dtype"].as_str() != Some("bf16") { return Err(invalid("trellis input takes BF16 rows")); }
            let rotated_bf16 = match g["output_dtype"].as_str() { None | Some("f16") => false, Some("bf16") => true,
                _ => return Err(invalid("trellis input rotated dtype")) };
            Op::TrellisInput { features: u32_of(g, "features")?, rotated_bf16 }
        }
        Some("trellis_output") => {
            if g["output_dtype"].as_str() != Some("bf16") { return Err(invalid("trellis output is BF16")); }
            Op::TrellisOutput { features: u32_of(g, "features")? }
        }
        Some("trellis_unpack_e4m3") => {
            if g["bits"].as_u64() != Some(4) || g["codebook"].as_str() != Some("e4m3") || g["abi_version"].as_u64() != Some(1) {
                return Err(invalid("unpack requires K4 e4m3 ABI v1"));
            }
            Op::Unpack { input: u32_of(g, "input_features")?, output: u32_of(g, "output_features")?,
                side: u32_of(g, "sharded_side")? }
        }
        Some("trellis_fp8_prefill") => {
            if g["scales"].as_str() != Some("block") || g["output_dtype"].as_str() != Some("f32") {
                return Err(invalid("prefill GEMM is block-scaled with F32 output"));
            }
            Op::Prefill { input: u32_of(g, "input_features")?, output: u32_of(g, "output_features")? }
        }
        Some("swiglu_split") => Op::SwigluSplit { inter: u32_of(g, "inter")? },
        Some("f32_narrow_pack") => Op::NarrowPack { width: u32_of(g, "width")?, pitch: u32_of(g, "pitch")? },
        Some("bf16_projection") => {
            if g["output_dtype"].as_str() != Some("torch.float32") { return Err(invalid("weights projection is F32")); }
            Op::IndexWeights { width: u32_of(g, "width")?, outputs: u32_of(g, "outputs")? }
        }
        _ => return Ok(None),
    }))
}

/// Validate a companion entry's pointer and scalar ABI against its kind.
pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<Op> {
    let op = op_of(entry)?.ok_or_else(|| invalid(format!("{}: not a native dense companion", spec.name)))?;
    let (pointers, scalars) = op.contract();
    let listed = entry["pointers"].as_array().ok_or_else(|| invalid("pointer metadata"))?;
    if listed.len() != pointers.len() || !listed.iter().zip(&pointers).all(|(p, (name, dtype, shape, role))|
        p["name"].as_str() == Some(name) && p["dtype"].as_str() == Some(dtype)
            && p["shape"].as_str() == Some(shape.as_str()) && p["role"].as_str() == Some(role))
        || !spec.pointers.iter().map(String::as_str).eq(pointers.iter().map(|p| p.0)) {
        return Err(invalid(format!("{}: pointer name/dtype/shape/role", spec.name)));
    }
    let listed = entry["scalars"].as_array().ok_or_else(|| invalid("scalar metadata"))?;
    if listed.len() != scalars.len() || !listed.iter().zip(scalars).all(|(s, n)|
        s["name"].as_str() == Some(n) && s["type"].as_str() == Some("int32"))
        || spec.scalar_kinds != vec![b'i'; scalars.len()] || !spec.scratch.values().all(|&b| b == 0) {
        return Err(invalid(format!("{}: scalar ABI or scratch", spec.name)));
    }
    if !matches!(op, Op::Unpack { .. }) && spec.capacity_rows != 4096 {
        return Err(invalid(format!("{}: row capacity", spec.name)));
    }
    Ok(op)
}

/// A bound companion program (runtime CTA budget queried at bind).
pub struct Companion<'a> {
    program: Program<'a>,
    pub op: Op,
    cta_limit: i32,
}

impl<'a> Companion<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str, op: Op) -> Result<Self> {
        let spec = programs.spec(name).map_err(|e| invalid(e.to_string()))?;
        let names: Vec<&str> = op.contract().0.iter().map(|p| p.0).collect();
        if !spec.pointers.iter().map(String::as_str).eq(names.iter().copied()) {
            return Err(invalid(format!("{name}: bound with another kind")));
        }
        let cta_limit = programs.library().sm_count().map_err(|e| invalid(e.to_string()))?
            .checked_mul(4).and_then(|n| i32::try_from(n).ok()).filter(|&n| n > 0)
            .ok_or_else(|| invalid("runtime SM count"))?;
        let program = programs.program(name, &names).map_err(|e| invalid(e.to_string()))?;
        Ok(Self { program, op, cta_limit })
    }

    /// # Safety
    /// `pointers` are device buffers of the kind's contract (in its pointer
    /// order) on the bound device, live through stream completion; `rows` live
    /// rows (ignored by the unpack), `c0` the pack's first column.
    pub unsafe fn launch(&self, pointers: &[*mut c_void], rows: u32, c0: u32, stream: *mut c_void) -> Result<()> {
        let (contract, scalar_names) = self.op.contract();
        if pointers.len() != contract.len() || pointers.iter().any(|p| p.is_null()) {
            return Err(invalid("companion storage"));
        }
        if !matches!(self.op, Op::Unpack { .. }) && (rows == 0 || rows > 4096) {
            return Err(invalid("companion rows outside compiled capacity"));
        }
        if let Op::NarrowPack { width, pitch } = self.op {
            if c0.checked_add(width).is_none_or(|end| end > pitch) {
                return Err(invalid("pack columns outside the destination pitch"));
            }
        }
        let scalars: Vec<Scalar> = scalar_names.iter().map(|&name| Scalar::I32(match name {
            "rows" => rows as i32, "cta_limit" => self.cta_limit, "c0" => c0 as i32, _ => 0 })).collect();
        // SAFETY: the caller's contract; pointer order and scalar types are the validated ABI.
        unsafe { self.program.launch(pointers, &scalars, stream) }.map_err(|e| invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(op: &str, geometry: serde_json::Value, pointers: serde_json::Value, scalars: &[&str]) -> serde_json::Value {
        json!({"op": op, "geometry": geometry, "pointers": pointers,
            "scalars": scalars.iter().map(|n| json!({"name": n, "type": "int32"})).collect::<Vec<_>>()})
    }

    fn spec(name: &str, e: &serde_json::Value, capacity: u32) -> ProgramSpec {
        ProgramSpec { name: name.into(), pointers: e["pointers"].as_array().unwrap().iter()
            .map(|p| p["name"].as_str().unwrap().into()).collect(),
            scalar_kinds: vec![b'i'; e["scalars"].as_array().unwrap().len()], scratch: Default::default(),
            capacity_rows: capacity, geometry: e["geometry"].clone() }
    }

    #[test]
    fn native_mx8_quant_contract_rejects_geometry_and_abi_mutations() {
        let quant = entry("dense_mxfp8_quant",
            json!({"input_features":2048,"input_dtype":"f16","rotation_count":4,"max_rows":4096}),
            json!([{"name":"rotated_inputs","dtype":"int64","shape":"[4]","role":"in"},
                {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"out"}]),
            &["rows", "rotation_id", "cta_limit"]);
        for features in [2048, 6144, 8192, 12288, 16384] {
            let mut candidate = quant.clone();
            candidate["geometry"]["input_features"] = json!(features);
            assert_eq!(validate_entry(&spec("q", &candidate, 4096), &candidate).unwrap(),
                Op::Mx8Quant { features });
        }
        for (field, value) in [("input_features", json!(4096)), ("input_dtype", json!("bf16")),
            ("rotation_count", json!(1)), ("max_rows", json!(64))] {
            let mut bad = quant.clone();
            bad["geometry"][field] = value;
            assert!(validate_entry(&spec("q", &bad, 4096), &bad).is_err(), "{field}");
        }
        for (field, value) in [("name", "partial"), ("dtype", "float32"),
            ("shape", "[0]"), ("role", "scratch")] {
            let mut bad = quant.clone();
            bad["pointers"][1][field] = json!(value);
            assert!(validate_entry(&spec("q", &bad, 4096), &bad).is_err(), "{field}");
        }
        let mut bad = quant.clone();
        bad["scalars"][1]["name"] = json!("c0");
        assert!(validate_entry(&spec("q", &bad, 4096), &bad).is_err());
        assert!(validate_entry(&spec("q", &quant, 64), &quant).is_err());
        let mut bad = spec("q", &quant, 4096);
        bad.scratch.insert("records".into(), 1);
        assert!(validate_entry(&bad, &quant).is_err());
    }

    #[test]
    fn exported_companion_contracts_validate_and_mutations_fail() {
        let prefill = entry("trellis_fp8_prefill", json!({"input_features": 2048, "output_features": 16384,
            "max_rows": 4096, "scales": "block", "output_dtype": "f32", "compute_warps": 8, "num_stages": 3}),
            json!([{"name":"x","dtype":"bfloat16","shape":"[rows,2048]","role":"in"},
                {"name":"w","dtype":"uint8","shape":"[16384,2048]","role":"in"},
                {"name":"scale","dtype":"float32","shape":"[128,16]","role":"in"},
                {"name":"out","dtype":"float32","shape":"[rows,16384]","role":"out"}]), &["rows"]);
        assert_eq!(validate_entry(&spec("exlr8_dense_prefill_q_b", &prefill, 4096), &prefill).unwrap(),
            Op::Prefill { input: 2048, output: 16384 });
        let unpack = entry("trellis_unpack_e4m3", json!({"input_features": 2048, "output_features": 16384,
            "sharded_side": 1, "bits": 4, "codebook": "e4m3", "abi_version": 1}),
            json!([{"name":"payload","dtype":"uint8","shape":"[16777216]","role":"in"},
                {"name":"output","dtype":"uint8","shape":"[16384,2048]","role":"out"}]), &[]);
        assert!(validate_entry(&spec("exlr8_dense_unpack_q_b", &unpack, 4096), &unpack).is_ok());
        let pack = entry("f32_narrow_pack", json!({"width": 576, "pitch": 2624}),
            json!([{"name":"input","dtype":"float32","shape":"[rows,576]","role":"in"},
                {"name":"output","dtype":"bfloat16","shape":"[rows,2624]","role":"out"}]), &["rows", "c0"]);
        assert!(validate_entry(&spec("exlr8_dense_f32_narrow_pack_n576_p2624", &pack, 4096), &pack).is_ok());
        let input = entry("trellis_input", json!({"features": 2048, "input_dtype": "bf16", "max_rows": 4096,
            "output_dtype": "bf16"}),
            json!([{"name":"input","dtype":"bfloat16","shape":"[rows,2048]","role":"in"},
                {"name":"suh","dtype":"float16","shape":"[2048]","role":"in"},
                {"name":"rotated","dtype":"bfloat16","shape":"[rows,2048]","role":"out"},
                {"name":"input_block_scales","dtype":"float16","shape":"[16]","role":"out"}]), &["rows", "cta_limit"]);
        assert_eq!(validate_entry(&spec("exlr8_dense_trellis_input_k2048_bf16", &input, 4096), &input).unwrap(),
            Op::TrellisInput { features: 2048, rotated_bf16: true });
        // Mutations: dtype, shape, scalar name, capacity, non-unit scales mode.
        let mut bad = prefill.clone();
        bad["pointers"][2]["shape"] = json!("[128,8]");
        assert!(validate_entry(&spec("p", &bad, 4096), &bad).is_err());
        let mut bad = prefill.clone();
        bad["geometry"]["scales"] = json!("row_kmajor");
        assert!(validate_entry(&spec("p", &bad, 4096), &bad).is_err());
        let mut bad = pack.clone();
        bad["scalars"][1]["name"] = json!("col");
        assert!(validate_entry(&spec("p", &bad, 4096), &bad).is_err());
        assert!(validate_entry(&spec("p", &prefill, 64), &prefill).is_err());
        let mut bad = unpack.clone();
        bad["geometry"]["codebook"] = json!("mcg");
        assert!(validate_entry(&spec("u", &bad, 4096), &bad).is_err());
        assert!(op_of(&json!({"op": "trellis_linear"})).unwrap().is_none());
    }
}

#[cfg(test)]
mod exported_manifest {
    /// Every exported EXLR8 dense entry the activation binds validates through
    /// the program loader's routing (companion by op, ABI-v1 linear by name).
    /// Set CUTEAFD_EXLR8_DENSE_ABIS to the export's binding ABI listing.
    #[test]
    #[ignore = "requires the exported binding ABI listing"]
    fn every_exported_dense_entry_validates() {
        let path = std::env::var("CUTEAFD_EXLR8_DENSE_ABIS").expect("CUTEAFD_EXLR8_DENSE_ABIS");
        let listing: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut checked = 0;
        for entry in listing["programs"].as_array().unwrap() {
            let name = entry["name"].as_str().unwrap();
            let spec = super::ProgramSpec { name: name.into(), pointers: entry["pointers"].as_array().unwrap().iter()
                .map(|p| p["name"].as_str().unwrap().into()).collect(),
                scalar_kinds: entry["scalars"].as_array().unwrap().iter().map(|s|
                    if s["type"] == "int32" { b'i' } else { b'?' }).collect(),
                scratch: entry["scratch_bytes_at_capacity"].as_object().unwrap().iter()
                    .map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect(),
                capacity_rows: entry["capacity_rows"].as_u64().unwrap() as u32, geometry: entry["geometry"].clone() };
            if name.starts_with("glm_stage_") || name.starts_with("glm2_stage_") {
                super::super::glm_stage::validate_entry(&spec, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
            } else if super::op_of(entry).unwrap().is_some() {
                super::validate_entry(&spec, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
            } else {
                super::super::native_dense::validate_entry(&spec, entry).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
            checked += 1;
        }
        // 76 dense operands/companions, 14 full-head and 18 TP2 attention stages.
        assert_eq!(checked, 108);
    }
}
