//! Dormant ABI-v1 bindings for checkpoint-native coordinator operands.
//! Trellis rotations and output transforms remain caller-owned.
use super::programs::{Program, ProgramSpec, Programs, Scalar};
use std::ffi::c_void;

#[derive(Debug, thiserror::Error)]
#[error("invalid native dense binding: {0}")]
pub struct DenseError(String);

type Result<T> = std::result::Result<T, DenseError>;

fn invalid(message: impl Into<String>) -> DenseError {
    DenseError(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Trellis,
    TrellisA8,
    Int4G32,
    Mxfp8,
    RowFp8,
    Embedding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseGeometry {
    pub encoding: Encoding,
    pub input_features: u64,
    pub output_features: u64,
    pub side: u64,
    pub max_rows: u32,
}

fn geometry(spec: &ProgramSpec) -> Result<DenseGeometry> {
    let name = spec
        .name
        .strip_prefix("exlr8_dense_")
        .ok_or_else(|| invalid("program is not a native dense operand"))?;
    let bf16_input = name.ends_with("_bf16");
    let a8 = name.ends_with("_a8");
    let shape = name.strip_suffix("_a8").or_else(|| name.strip_suffix("_bf16")).unwrap_or(name);
    let (encoding, k, n, side) = match shape {
        "trellis_q_b" => (Encoding::Trellis, 2048, 16384, 1),
        "trellis_o" => (Encoding::Trellis, 16384, 6144, 2),
        "trellis_gate_up" => (Encoding::Trellis, 6144, 12288, 1),
        "trellis_down" => (Encoding::Trellis, 12288, 6144, 2),
        "trellis_q_b_tp2" => (Encoding::Trellis, 2048, 8192, 1),
        "trellis_o_tp2" => (Encoding::Trellis, 8192, 6144, 2),
        "trellis_gate_up_tp2" => (Encoding::Trellis, 6144, 6144, 1),
        "trellis_down_tp2" => (Encoding::Trellis, 6144, 6144, 2),
        "int4_q_b" => (Encoding::Int4G32, 2048, 16384, 1),
        "int4_o" => (Encoding::Int4G32, 16384, 6144, 2),
        "int4_q_b_tp2" => (Encoding::Int4G32, 2048, 8192, 1),
        "int4_o_tp2" => (Encoding::Int4G32, 8192, 6144, 2),
        "mxfp8_q_a" => (Encoding::Mxfp8, 6144, 2048, 1),
        "mxfp8_kv_a" => (Encoding::Mxfp8, 6144, 576, 1),
        "mxfp8_index_q" => (Encoding::Mxfp8, 2048, 4096, 1),
        "mxfp8_index_k" => (Encoding::Mxfp8, 6144, 128, 1),
        "fp8_row_lm_head" => (Encoding::RowFp8, 6144, 154880, 1),
        "fp8_row_embedding" => (Encoding::Embedding, 6144, 154880, 1),
        _ => {
            return Err(invalid(format!(
                "unsupported native dense program {}",
                spec.name
            )));
        }
    };
    if bf16_input && encoding != Encoding::Trellis {
        return Err(invalid("BF16 core suffix requires trellis encoding"));
    }
    let encoding = if a8 && encoding == Encoding::Trellis { Encoding::TrellisA8 }
        else if a8 { return Err(invalid("A8 core suffix requires trellis encoding")); } else { encoding };
    let g = &spec.geometry;
    let field = |key: &str, expected| -> Result<()> {
        if g[key].as_u64() != Some(expected) {
            return Err(invalid(format!("{}: geometry.{key}", spec.name)));
        }
        Ok(())
    };
    if !a8 { field("abi_version", 1)?; }
    field("max_rows", 4096)?;
    if spec.capacity_rows != 4096 {
        return Err(invalid(
            "native dense row capacity differs from manifest geometry",
        ));
    }
    if encoding == Encoding::Embedding {
        field("hidden", k)?;
        field("vocabulary", n)?;
        field("ctas_per_sm", 4)?;
    } else {
        field("input_features", k)?;
        field("output_features", n)?;
        field("sharded_side", side)?;
        if !a8 { field("ctas_per_sm", 4)?; }
        if matches!(encoding, Encoding::Trellis | Encoding::TrellisA8) {
            let expected_dtype = if bf16_input { "bf16" } else { "f16" };
            if g["input_dtype"].as_str().unwrap_or("f16") != expected_dtype {
                return Err(invalid("native trellis input dtype disagrees with program name"));
            }
            field("bits", 4)?;
            field("rotation_count", 4)?;
            if a8 {
                if g["encoding"].as_str() != Some("trellis") { return Err(invalid("A8 requires trellis weights")); }
            } else {
                field("split_k_blocks", 8)?;
                field("splits", k.div_ceil(1024))?;
            }
            if g["codebook"].as_str() != Some("e4m3") {
                return Err(invalid("native trellis requires checkpoint E4M3 codebook"));
            }
        } else {
            let expected = match encoding {
                Encoding::Int4G32 => "int4_g32",
                Encoding::Mxfp8 => "mxfp8",
                _ => "fp8_row",
            };
            field("splits", 1)?;
            field("split_k", k)?;
            let groups = if encoding == Encoding::RowFp8 { 2 } else { 1 };
            let mut segments = 1;
            while n / (8 * groups) * segments < 2048
                && segments < 8
                && (k / 128) % (segments * 2) == 0
            {
                segments *= 2;
            }
            field("segments", segments)?;
            if g["encoding"].as_str() != Some(expected) || g["core"].as_str() != Some("tile") {
                return Err(invalid(
                    "native scaled encoding/core disagrees with operand",
                ));
            }
        }
    }
    Ok(DenseGeometry {
        encoding,
        input_features: k,
        output_features: n,
        side,
        max_rows: 4096,
    })
}

impl DenseGeometry {
    /// Per-lane scratch; charge before allocation. Scaled ABI-v1 partial is unused.
    pub fn scratch_bytes(self, rows: u32) -> Result<u64> {
        if rows == 0 || rows > self.max_rows {
            return Err(invalid("rows outside compiled native dense capacity"));
        }
        Ok(if self.encoding == Encoding::TrellisA8 {
            u64::from(rows).div_ceil(16) * (self.input_features / 32) * 528
        } else if self.encoding == Encoding::Trellis {
            u64::from(rows) * self.input_features.div_ceil(1024) * self.output_features * 4
        } else {
            0
        })
    }

    /// Exact E4M3 prefill reduces split-K on chip; metadata retains the ABI's
    /// conservative capacity so older manifests remain independently verifiable.
    fn launch_scratch_bytes(self, rows: u32) -> Result<u64> {
        let bytes = self.scratch_bytes(rows)?;
        Ok(if self.encoding == Encoding::Trellis && rows > 64 { 0 } else { bytes })
    }

    fn pointers(self) -> &'static [&'static str] {
        match self.encoding {
            Encoding::Trellis => &[
                "rotated_inputs",
                "payload",
                "input_block_scales",
                "partial",
                "output",
            ],
            Encoding::TrellisA8 => &["rotated_inputs", "payload", "scales", "records", "output"],
            Encoding::Embedding => &["ids", "payload", "scales", "output"],
            _ => &["x", "payload", "scales", "partial", "output"],
        }
    }
}

fn validate_spec(spec: &ProgramSpec) -> Result<DenseGeometry> {
    let g = geometry(spec)?;
    let kinds: &[u8] = if matches!(g.encoding, Encoding::Trellis | Encoding::TrellisA8) {
        b"iii"
    } else {
        b"ii"
    };
    let scratch = if matches!(g.encoding, Encoding::Embedding | Encoding::TrellisA8) {
        std::collections::HashMap::new()
    } else {
        std::collections::HashMap::from([("partial".to_string(), g.scratch_bytes(g.max_rows)?)])
    };
    if !spec
        .pointers
        .iter()
        .map(String::as_str)
        .eq(g.pointers().iter().copied())
        || spec.scalar_kinds != kinds
        || spec.scratch != scratch
    {
        return Err(invalid(
            "native dense pointer order, scalar kinds or scratch",
        ));
    }
    Ok(g)
}

/// Validate the exported contract without loading kernels or allocating GPU storage.
pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<DenseGeometry> {
    let g = validate_spec(spec)?;
    let a8 = g.encoding == Encoding::TrellisA8;
    let trellis = matches!(g.encoding, Encoding::Trellis | Encoding::TrellisA8);
    let embedding = g.encoding == Encoding::Embedding;
    let scalar_names: &[&str] = if trellis {
        &["rows", "rotation_id", "cta_limit"]
    } else {
        &["rows", "cta_limit"]
    };
    let scalars = entry["scalars"]
        .as_array()
        .ok_or_else(|| invalid("missing scalar ABI"))?;
    if scalars.len() != scalar_names.len()
        || !scalars
            .iter()
            .zip(scalar_names)
            .all(|(s, n)| s["name"].as_str() == Some(*n) && s["type"].as_str() == Some("int32"))
    {
        return Err(invalid("native dense scalar names/types"));
    }
    let k = g.input_features;
    let n = g.output_features;
    let scale_type = match g.encoding {
        Encoding::Trellis | Encoding::TrellisA8 | Encoding::Int4G32 => "float16",
        Encoding::Mxfp8 => "uint8",
        _ => "float32",
    };
    let (dtypes, shapes, roles): (Vec<&str>, Vec<String>, Vec<&str>) = if embedding {
        (
            vec!["int32", "uint8", "float32", "bfloat16"],
            vec![
                "[rows]".into(),
                format!("[{},{}]", n, k),
                format!("[{n}]"),
                format!("[rows,{k}]"),
            ],
            vec!["in", "in", "in", "out"],
        )
    } else {
        let bytes = k * n
            / if trellis || g.encoding == Encoding::Int4G32 {
                2
            } else {
                1
            };
        let scales = if trellis {
            k / 128
        } else if g.encoding == Encoding::RowFp8 {
            n
        } else {
            n * k / 32
        };
        (
            vec![
                if trellis { "int64" } else { "bfloat16" },
                "uint8",
                scale_type,
                if a8 { "uint8" } else { "float32" },
                "float32",
            ],
            vec![
                if trellis {
                    "[4]".into()
                } else {
                    format!("[rows,{k}]")
                },
                format!("[{bytes}]"),
                format!("[{scales}]"),
                if a8 {
                    "[record_bytes]".into()
                } else if trellis {
                    format!("[rows,{},{n}]", k.div_ceil(1024))
                } else {
                    "[0]".into()
                },
                format!("[rows,{n}]"),
            ],
            vec!["in", "in", "in", if a8 { "in" } else { "scratch" }, "out"],
        )
    };
    let pointers = entry["pointers"]
        .as_array()
        .ok_or_else(|| invalid("missing pointer metadata"))?;
    if pointers.len() != dtypes.len()
        || !pointers.iter().enumerate().all(|(i, p)| {
            p["name"].as_str() == Some(g.pointers()[i])
                && p["dtype"].as_str() == Some(dtypes[i])
                && p["shape"].as_str() == Some(shapes[i].as_str())
                && p["role"].as_str() == Some(roles[i])
        })
    {
        return Err(invalid("native dense pointer dtype/shape/role"));
    }
    Ok(g)
}

/// Resolved at startup; no live-row compilation, allocation or policy lookup.
pub struct DenseLinear<'a> {
    program: Program<'a>,
    geometry: DenseGeometry,
    cta_limit: i32,
}

impl<'a> DenseLinear<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let g = validate_spec(programs.spec(name).map_err(|e| invalid(e.to_string()))?)?;
        if g.encoding == Encoding::Embedding {
            return Err(invalid("embedding gather is not a dense linear launch"));
        }
        let sms = programs
            .library()
            .sm_count()
            .map_err(|e| invalid(e.to_string()))?;
        let cta_limit = sms
            .checked_mul(4)
            .filter(|&n| n > 0)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| invalid("invalid runtime SM count"))?;
        let program = programs
            .program(name, g.pointers())
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            program,
            geometry: g,
            cta_limit,
        })
    }

    pub fn geometry(&self) -> DenseGeometry {
        self.geometry
    }

    /// # Safety
    /// All pointers must be device storage for the validated manifest geometry,
    /// on the bound device, and survive until the stream completes. For trellis,
    /// `input` is the I64 table of four F16 rotated inputs; otherwise BF16 rows.
    /// Caller owns every rotation/output transform and admits per-lane scratch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        input: *mut c_void,
        payload: *mut c_void,
        scales: *mut c_void,
        partial: *mut c_void,
        partial_bytes: u64,
        output: *mut c_void,
        rows: u32,
        rotation: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let required = self.geometry.launch_scratch_bytes(rows)?;
        let trellis = matches!(self.geometry.encoding, Encoding::Trellis | Encoding::TrellisA8);
        if partial_bytes < required
            || (required > 0 && partial.is_null())
            || input.is_null()
            || payload.is_null()
            || scales.is_null()
            || output.is_null()
            || (trellis && rotation >= 4)
            || (!trellis && rotation != 0)
        {
            return Err(invalid("native dense launch storage or rotation"));
        }
        let mut scalars = vec![Scalar::I32(rows as i32)];
        if trellis {
            scalars.push(Scalar::I32(rotation as i32));
        }
        scalars.push(Scalar::I32(self.cta_limit));
        // SAFETY: pointer contracts are delegated to the caller; rows, rotation,
        // scratch extent and manifest ABI are checked before the generic launcher.
        unsafe {
            self.program
                .launch(&[input, payload, scales, partial, output], &scalars, stream)
        }
        .map_err(|e| invalid(e.to_string()))
    }
}

/// Row-FP8 vocabulary gather. `ids` contains arbitrary vocabulary IDs, not row offsets.
/// This deliberately does not replace the indexed/copy/fallback BF16 gather ABI.
pub struct DenseEmbedding<'a> {
    program: Program<'a>,
    geometry: DenseGeometry,
    cta_limit: i32,
}

impl<'a> DenseEmbedding<'a> {
    pub fn bind(programs: &'a Programs<'a>, name: &str) -> Result<Self> {
        let geometry = validate_spec(programs.spec(name).map_err(|e| invalid(e.to_string()))?)?;
        if geometry.encoding != Encoding::Embedding {
            return Err(invalid("native embedding requires the row-FP8 gather ABI"));
        }
        let cta_limit = programs
            .library()
            .sm_count()
            .map_err(|e| invalid(e.to_string()))?
            .checked_mul(4)
            .filter(|&n| n > 0)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| invalid("invalid runtime SM count"))?;
        let program = programs
            .program(name, geometry.pointers())
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            program,
            geometry,
            cta_limit,
        })
    }

    pub fn geometry(&self) -> DenseGeometry {
        self.geometry
    }

    /// # Safety
    /// IDs are I32[rows], payload U8[vocabulary,hidden], scales F32[vocabulary],
    /// and output BF16[rows,hidden] on the bound device. All storage survives
    /// stream completion. Out-of-vocabulary IDs produce zero rows.
    pub unsafe fn launch(
        &self,
        ids: *mut c_void,
        payload: *mut c_void,
        scales: *mut c_void,
        output: *mut c_void,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        validate_embedding_launch(self.geometry, ids, payload, scales, output, rows)?;
        // SAFETY: the caller retains correctly sized device operands through completion.
        unsafe {
            self.program.launch(
                &[ids, payload, scales, output],
                &[Scalar::I32(rows as i32), Scalar::I32(self.cta_limit)],
                stream,
            )
        }
        .map_err(|e| invalid(e.to_string()))
    }
}

fn validate_embedding_launch(
    g: DenseGeometry,
    ids: *mut c_void,
    payload: *mut c_void,
    scales: *mut c_void,
    output: *mut c_void,
    rows: u32,
) -> Result<()> {
    if g.encoding != Encoding::Embedding
        || rows == 0
        || rows > g.max_rows
        || [ids, payload, scales, output].iter().any(|p| p.is_null())
    {
        return Err(invalid("native embedding launch storage or row capacity"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn trellis() -> (ProgramSpec, serde_json::Value) {
        let entry = json!({"scalars": [
            {"name":"rows","type":"int32"}, {"name":"rotation_id","type":"int32"},
            {"name":"cta_limit","type":"int32"}], "pointers": [
            {"name":"rotated_inputs","dtype":"int64","shape":"[4]","role":"in"},
            {"name":"payload","dtype":"uint8","shape":"[8388608]","role":"in"},
            {"name":"input_block_scales","dtype":"float16","shape":"[16]","role":"in"},
            {"name":"partial","dtype":"float32","shape":"[rows,2,8192]","role":"scratch"},
            {"name":"output","dtype":"float32","shape":"[rows,8192]","role":"out"}]});
        let spec = ProgramSpec {
            name: "exlr8_dense_trellis_q_b_tp2".into(),
            pointers: entry["pointers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["name"].as_str().unwrap().to_string())
                .collect(),
            scalar_kinds: b"iii".to_vec(),
            capacity_rows: 4096,
            scratch: HashMap::from([("partial".into(), 268435456)]),
            geometry: json!({"abi_version":1,"input_features":2048,"output_features":8192,
                "bits":4,"codebook":"e4m3","sharded_side":1,"rotation_count":4,
                "max_rows":4096,"split_k_blocks":8,"splits":2,"ctas_per_sm":4}),
        };
        (spec, entry)
    }

    #[test]
    fn native_dense_trellis_manifest_and_lane_scratch() {
        let (spec, entry) = trellis();
        let g = validate_entry(&spec, &entry).unwrap();
        assert_eq!(g.encoding, Encoding::Trellis);
        assert_eq!(g.scratch_bytes(1).unwrap(), 65536);
        assert_eq!(g.scratch_bytes(4096).unwrap(), 268435456);
        for rows in [1, 16, 64] {
            assert_eq!(g.launch_scratch_bytes(rows).unwrap(), g.scratch_bytes(rows).unwrap());
        }
        for rows in [65, 80, 127, 128, 129, 130, 4096] {
            assert_eq!(g.launch_scratch_bytes(rows).unwrap(), 0);
        }
        assert!(g.launch_scratch_bytes(0).is_err());
        assert!(g.launch_scratch_bytes(4097).is_err());
        assert!(g.scratch_bytes(0).is_err());
        assert!(g.scratch_bytes(4097).is_err());
        // Dense ABI v1 is deliberately not the routed expert ABI.
        assert!(super::super::exlr8::validate_entry(&spec, &entry).is_err());
    }

    #[test]
    fn native_dense_a8_manifest_uses_caller_owned_records() {
        let (mut spec, mut entry) = trellis();
        spec.name.push_str("_a8");
        spec.geometry = json!({"input_features":2048,"output_features":8192,
            "encoding":"trellis","sharded_side":1,"input_dtype":"f16",
            "bits":4,"codebook":"e4m3","rotation_count":4,"max_rows":4096});
        spec.scratch.clear();
        spec.pointers[2] = "scales".into();
        spec.pointers[3] = "records".into();
        entry["pointers"][2]["name"] = json!("scales");
        entry["pointers"][3] = json!({"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"in"});
        let g = validate_entry(&spec, &entry).unwrap();
        assert_eq!(g.encoding, Encoding::TrellisA8);
        assert_eq!(g.launch_scratch_bytes(4096).unwrap(), 256 * 64 * 528);
        assert_eq!(g.launch_scratch_bytes(17).unwrap(), 2 * 64 * 528);
        for (field, value) in [("input_dtype", json!("bf16")), ("encoding", json!("mxfp8")),
            ("rotation_count", json!(1)), ("bits", json!(3))] {
            let mut bad = spec.clone();
            bad.geometry[field] = value;
            assert!(validate_entry(&bad, &entry).is_err(), "{field}");
        }
        entry["pointers"][3]["role"] = json!("scratch");
        assert!(validate_entry(&spec, &entry).is_err());
    }

    #[test]
    fn native_dense_bf16_program_requires_matching_static_dtype() {
        let (mut spec, mut entry) = trellis();
        spec.name.push_str("_bf16");
        assert!(validate_entry(&spec, &entry).is_err());
        spec.geometry["input_dtype"] = json!("bf16");
        entry["geometry"]["input_dtype"] = json!("bf16");
        assert!(validate_entry(&spec, &entry).is_ok());
        spec.geometry["input_dtype"] = json!("fp8");
        assert!(validate_entry(&spec, &entry).is_err());
        spec.name = spec.name.trim_end_matches("_bf16").to_owned();
        spec.geometry["input_dtype"] = json!("bf16");
        assert!(validate_entry(&spec, &entry).is_err());
    }

    #[test]
    fn native_dense_manifest_rejects_wrong_contract_before_load() {
        let (spec, entry) = trellis();
        for (field, value) in [
            ("abi_version", json!(2)),
            ("bits", json!(3)),
            ("output_features", json!(16384)),
            ("sharded_side", json!(2)),
            ("rotation_count", json!(1)),
            ("splits", json!(1)),
            ("max_rows", json!(64)),
            ("codebook", json!("f16")),
        ] {
            let mut bad = spec.clone();
            bad.geometry[field] = value;
            assert!(validate_entry(&bad, &entry).is_err(), "{field}");
        }
        let mut bad = spec.clone();
        bad.scratch.clear();
        assert!(validate_entry(&bad, &entry).is_err());
        let mut bad = spec.clone();
        bad.pointers.swap(0, 1);
        assert!(validate_entry(&bad, &entry).is_err());
        let mut bad = spec.clone();
        bad.scalar_kinds[0] = b'l';
        assert!(validate_entry(&bad, &entry).is_err());
        let mut bad = spec.clone();
        bad.capacity_rows = 64;
        assert!(validate_entry(&bad, &entry).is_err());
        for (field, value) in [("dtype", "bfloat16"), ("shape", "[2]"), ("role", "out")] {
            let mut bad = entry.clone();
            bad["pointers"][0][field] = json!(value);
            assert!(validate_entry(&spec, &bad).is_err());
        }
        let mut bad = entry.clone();
        bad["scalars"][1]["name"] = json!("groups");
        assert!(validate_entry(&spec, &bad).is_err());
    }

    #[test]
    fn native_dense_int4_and_embedding_keep_native_types() {
        let mut spec = ProgramSpec {
            name: "exlr8_dense_int4_o_tp2".into(),
            pointers: ["x", "payload", "scales", "partial", "output"]
                .map(str::to_string)
                .to_vec(),
            scalar_kinds: b"ii".to_vec(),
            capacity_rows: 4096,
            scratch: HashMap::from([("partial".into(), 0)]),
            geometry: json!({"abi_version":1,"input_features":8192,"output_features":6144,
                "encoding":"int4_g32","sharded_side":2,"max_rows":4096,
                "splits":1,"split_k":8192,"segments":4,"ctas_per_sm":4,"core":"tile"}),
        };
        let mut entry = json!({"scalars":[{"name":"rows","type":"int32"},
            {"name":"cta_limit","type":"int32"}], "pointers":[
            {"name":"x","dtype":"bfloat16","shape":"[rows,8192]","role":"in"},
            {"name":"payload","dtype":"uint8","shape":"[25165824]","role":"in"},
            {"name":"scales","dtype":"float16","shape":"[1572864]","role":"in"},
            {"name":"partial","dtype":"float32","shape":"[0]","role":"scratch"},
            {"name":"output","dtype":"float32","shape":"[rows,6144]","role":"out"}]});
        let g = validate_entry(&spec, &entry).unwrap();
        assert_eq!(g.encoding, Encoding::Int4G32);
        assert_eq!(g.scratch_bytes(4096).unwrap(), 0);
        entry["pointers"][2]["dtype"] = json!("float32");
        assert!(validate_entry(&spec, &entry).is_err());
        spec.name = "exlr8_dense_fp8_row_embedding".into();
        spec.geometry = json!({"abi_version":1,"hidden":6144,"vocabulary":154880,
            "max_rows":4096,"ctas_per_sm":4});
        spec.scratch.clear();
        spec.pointers = ["ids", "payload", "scales", "output"]
            .map(str::to_string)
            .to_vec();
        entry["pointers"] = json!([
            {"name":"ids","dtype":"int32","shape":"[rows]","role":"in"},
            {"name":"payload","dtype":"uint8","shape":"[154880,6144]","role":"in"},
            {"name":"scales","dtype":"float32","shape":"[154880]","role":"in"},
            {"name":"output","dtype":"bfloat16","shape":"[rows,6144]","role":"out"}]);
        let geometry = validate_entry(&spec, &entry).unwrap();
        assert_eq!(geometry.encoding, Encoding::Embedding);
        let pointer = std::ptr::NonNull::<u32>::dangling().as_ptr().cast();
        for rows in [1, 4, 4096] {
            validate_embedding_launch(geometry, pointer, pointer, pointer, pointer, rows).unwrap();
        }
        for rows in [0, 4097, u32::MAX] {
            assert!(
                validate_embedding_launch(geometry, pointer, pointer, pointer, pointer, rows)
                    .is_err()
            );
        }
        for missing in 0..4 {
            let mut pointers = [pointer; 4];
            pointers[missing] = std::ptr::null_mut();
            assert!(
                validate_embedding_launch(
                    geometry,
                    pointers[0],
                    pointers[1],
                    pointers[2],
                    pointers[3],
                    1
                )
                .is_err()
            );
        }
        for (field, value) in [("dtype", "uint8"), ("shape", "[rows,1]"), ("role", "out")] {
            let mut bad = entry.clone();
            bad["pointers"][0][field] = json!(value);
            assert!(validate_entry(&spec, &bad).is_err());
        }
    }
}
