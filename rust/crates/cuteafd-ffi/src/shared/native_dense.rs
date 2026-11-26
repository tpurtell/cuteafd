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
    Mxfp8A8,
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

fn keyed_geometry(name: &str) -> Option<(Encoding, u64, u64, u64)> {
    let (encoding, shape) = if let Some(shape) = name.strip_prefix("trellis_k") {
        (Encoding::Trellis, shape)
    } else if let Some(shape) = name.strip_prefix("mxfp8_k") {
        (Encoding::Mxfp8, shape)
    } else if let Some(shape) = name.strip_prefix("fp8_row_k") {
        (Encoding::RowFp8, shape)
    } else if let Some(shape) = name.strip_prefix("fp8_row_embedding_k") {
        (Encoding::Embedding, shape)
    } else { return None; };
    let (k, rest) = shape.split_once("_n")?;
    let (n, side) = if encoding == Encoding::Embedding { (rest, "1") }
        else { rest.split_once("_s")? };
    let (k, n, side): (u64, u64, u64) = (k.parse().ok()?, n.parse().ok()?, side.parse().ok()?);
    let canonical = if encoding == Encoding::Embedding { format!("{k}_n{n}") }
        else { format!("{k}_n{n}_s{side}") };
    if canonical != shape || k == 0 || n == 0 || !matches!(side, 1 | 2)
        || (encoding == Encoding::Trellis && (k % 128 != 0 || n % 128 != 0))
        || (matches!(encoding, Encoding::Mxfp8 | Encoding::RowFp8) && (k % 128 != 0 || n % 32 != 0)) {
        return None;
    }
    if k > i32::MAX as u64 || n > i32::MAX as u64
        || k.checked_mul(n).and_then(|v| v.checked_mul(4096 * 4)).is_none()
        || (encoding == Encoding::Embedding && (k % 2 != 0 || 4096 * k.div_ceil(256) > i32::MAX as u64 / 2)) {
        return None;
    }
    Some((encoding, k, n, side))
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
        _ => keyed_geometry(shape).ok_or_else(|| invalid(format!(
            "unsupported native dense program {}", spec.name)))?,
    };
    if bf16_input && encoding != Encoding::Trellis {
        return Err(invalid("BF16 core suffix requires trellis encoding"));
    }
    let encoding = match (a8, encoding) {
        (true, Encoding::Trellis) => Encoding::TrellisA8,
        (true, Encoding::Mxfp8) => Encoding::Mxfp8A8,
        (true, _) => return Err(invalid("A8 suffix requires trellis or MXFP8 encoding")),
        (false, encoding) => encoding,
    };
    let g = &spec.geometry;
    let field = |key: &str, expected| -> Result<()> {
        if g[key].as_u64() != Some(expected) {
            return Err(invalid(format!("{}: geometry.{key}", spec.name)));
        }
        Ok(())
    };
    field("abi_version", if a8 { 2 } else { 1 })?;
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
        // MX8 uses runtime cta_limit and on-chip reduction, not A16 split-K metadata.
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
        } else if encoding == Encoding::Mxfp8A8 {
            if (k / 32) * 256 > (i32::MAX as u64) / 8 {
                return Err(invalid("native MXFP8 A8 record indexing extent"));
            }
            if g["encoding"].as_str() != Some("mxfp8") || g["input_dtype"].as_str() != Some("bf16") {
                return Err(invalid("native MXFP8 A8 requires BF16 ingress and MXFP8 weights"));
            }
            field("rotation_count", 1)?;
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
        Ok(if matches!(self.encoding, Encoding::TrellisA8 | Encoding::Mxfp8A8) {
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
            Encoding::TrellisA8 | Encoding::Mxfp8A8 => &["rotated_inputs", "payload", "scales", "records", "output"],
            Encoding::Embedding => &["ids", "payload", "scales", "output"],
            _ => &["x", "payload", "scales", "partial", "output"],
        }
    }
}

pub fn validate_spec(spec: &ProgramSpec) -> Result<DenseGeometry> {
    let g = geometry(spec)?;
    let kinds: &[u8] = if matches!(g.encoding, Encoding::Trellis | Encoding::TrellisA8 | Encoding::Mxfp8A8) {
        b"iii"
    } else {
        b"ii"
    };
    let scratch = if matches!(g.encoding, Encoding::Embedding | Encoding::TrellisA8 | Encoding::Mxfp8A8) {
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
    let a8 = matches!(g.encoding, Encoding::TrellisA8 | Encoding::Mxfp8A8);
    let trellis = matches!(g.encoding, Encoding::Trellis | Encoding::TrellisA8);
    let embedding = g.encoding == Encoding::Embedding;
    let scalar_names: &[&str] = if trellis || a8 {
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
        Encoding::Mxfp8 | Encoding::Mxfp8A8 => "uint8",
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
                if trellis || a8 { "int64" } else { "bfloat16" },
                "uint8",
                scale_type,
                if a8 { "uint8" } else { "float32" },
                "float32",
            ],
            vec![
                if trellis {
                    "[4]".into()
                } else if a8 {
                    "[1]".into()
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
    /// `input` is an I64 table of four rotated inputs. MXFP8 A8 takes an I64
    /// table of one BF16 input; other scaled encodings take BF16 rows directly.
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
        validate_linear_launch(self.geometry, [input, payload, scales, partial, output],
            partial_bytes, rows, rotation)?;
        let trellis = matches!(self.geometry.encoding, Encoding::Trellis | Encoding::TrellisA8);
        let mut scalars = vec![Scalar::I32(rows as i32)];
        if trellis || self.geometry.encoding == Encoding::Mxfp8A8 {
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

/// Checked native MXFP8 W8A8 execution. Quantize once per distinct BF16 input;
/// sibling projections may reuse its records without another activation read.
/// The caller initializes `inputs[0]` to the stable activation address before
/// enqueue/capture, orders producers and consumers on `stream`, and retains all
/// buffers and the stream through completion and graph replay.
pub struct Mxfp8A8<'a> {
    linear: DenseLinear<'a>,
    quantizer: super::native_dense_glue::Companion<'a>,
    narrow: super::device_ops::F32Bf16Narrow<'a>,
    library: &'a crate::NativeLibrary,
    device: i32,
}

impl<'a> Mxfp8A8<'a> {
    pub fn bind(programs: &'a Programs<'a>, linear: &str, quantizer: &str) -> Result<Self> {
        use super::native_dense_glue::{op_of, Companion, Op};
        let geometry = validate_spec(programs.spec(linear).map_err(|e| invalid(e.to_string()))?)?;
        if geometry.encoding != Encoding::Mxfp8A8 {
            return Err(invalid("checked MXFP8 execution requires native BF16 A8 geometry"));
        }
        let expected = Op::Mx8QuantBf16 { features: u32::try_from(geometry.input_features)
            .map_err(|_| invalid("MXFP8 input width"))? };
        let spec = programs.spec(quantizer).map_err(|e| invalid(e.to_string()))?;
        let entry = serde_json::json!({"op":"dense_mxfp8_quant", "geometry":spec.geometry});
        if op_of(&entry).map_err(|e| invalid(e.to_string()))? != Some(expected)
            || spec.scalar_kinds != b"iii" || spec.capacity_rows != 4096 || !spec.scratch.is_empty() {
            return Err(invalid("MXFP8 quantizer differs from native linear geometry"));
        }
        let library = programs.library();
        let device = library.cuda_get_device().map_err(|e| invalid(e.to_string()))?;
        let linear = DenseLinear::bind(programs, linear)?;
        let quantizer = Companion::bind(programs, quantizer, expected).map_err(|e| invalid(e.to_string()))?;
        let narrow = library.f32_bf16_narrow().map_err(|e| invalid(e.to_string()))?;
        Ok(Self { linear, quantizer, narrow, library, device })
    }

    pub fn geometry(&self) -> DenseGeometry { self.linear.geometry() }

    /// `activation` is the packed BF16 destination named by the stable table.
    /// No table transfer, workspace allocation or synchronization occurs here.
    pub fn quantize(&self, rows: u32, inputs: crate::CuteafdDeviceBuffer,
        activation: crate::CuteafdDeviceBuffer, records: crate::CuteafdDeviceBuffer,
        stream: *mut c_void) -> Result<()> {
        validate_mxfp8_quantize(self.geometry(), rows, [inputs, activation, records], self.device)?;
        self.current_device()?;
        // SAFETY: the BF16 table/activation/record extents, alignment, device and
        // writable nonaliasing were checked. The owner initializes the table and
        // retains all storage and the ordered stream through completion/replay.
        unsafe { self.quantizer.launch(&[inputs.ptr, records.ptr], rows, 0, stream) }
            .map_err(|e| invalid(e.to_string()))
    }

    /// Consume prepared records; this does not requantize the shared input.
    #[allow(clippy::too_many_arguments)]
    pub fn project(&self, rows: u32, inputs: crate::CuteafdDeviceBuffer,
        payload: crate::CuteafdDeviceBuffer, scales: crate::CuteafdDeviceBuffer,
        records: crate::CuteafdDeviceBuffer, output: crate::CuteafdDeviceBuffer,
        stream: *mut c_void) -> Result<()> {
        validate_mxfp8_project(self.geometry(), rows, [inputs, payload, scales, records, output], self.device)?;
        self.current_device()?;
        // SAFETY: native operand extents, alignment, device and writable
        // nonaliasing checked; the caller retains storage and the ordered stream.
        unsafe { self.linear.launch(inputs.ptr, payload.ptr, scales.ptr, records.ptr,
            records.bytes as u64, output.ptr, rows, 0, stream) }
    }

    /// Consume the F32 projection at the reference's BF16 rounding boundary.
    /// The separately admitted destination remains caller-owned; enqueue this on
    /// the projection stream before the serial F32 temporary is reused.
    pub fn narrow(&self, rows: u32, source: crate::CuteafdDeviceBuffer,
        destination: crate::CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        validate_mxfp8_narrow(self.geometry(), rows, [source, destination], self.device)?;
        self.current_device()?;
        let count = mxfp8_extent(&[u64::from(rows), self.geometry().output_features])?;
        // SAFETY: F32/BF16 extents, alignment, bound device and nonaliasing were
        // checked. The owner orders this after projection and retains both buffers.
        unsafe { self.narrow.launch(source.ptr, destination.ptr, count, stream) }
            .map_err(|e| invalid(e.to_string()))
    }

    fn current_device(&self) -> Result<()> {
        if self.library.cuda_get_device().map_err(|e| invalid(e.to_string()))? != self.device {
            return Err(invalid("MXFP8 launch is not on its bound device"));
        }
        Ok(())
    }
}

fn mxfp8_extent(values: &[u64]) -> Result<usize> {
    values.iter().try_fold(1u64, |a, b| a.checked_mul(*b))
        .and_then(|bytes| usize::try_from(bytes).ok()).ok_or_else(|| invalid("MXFP8 operand extent overflow"))
}

fn validate_mxfp8_narrow(g: DenseGeometry, rows: u32,
    buffers: [crate::CuteafdDeviceBuffer; 2], device: i32) -> Result<()> {
    if g.encoding != Encoding::Mxfp8A8 || buffers[0].device_id != device {
        return Err(invalid("MXFP8 encoding or bound device"));
    }
    g.scratch_bytes(rows)?;
    let count = mxfp8_extent(&[u64::from(rows), g.output_features])?;
    let source = count.checked_mul(4).ok_or_else(|| invalid("MXFP8 narrowing extent"))?;
    let destination = count.checked_mul(2).ok_or_else(|| invalid("MXFP8 narrowing extent"))?;
    super::projected_buffers::validate_buffers(&[(buffers[0], source), (buffers[1], destination)],
        &[4, 2], &[1]).map_err(invalid)
}

fn validate_mxfp8_quantize(g: DenseGeometry, rows: u32,
    buffers: [crate::CuteafdDeviceBuffer; 3], device: i32) -> Result<()> {
    if g.encoding != Encoding::Mxfp8A8 || buffers[0].device_id != device {
        return Err(invalid("MXFP8 encoding or bound device"));
    }
    let records = usize::try_from(g.scratch_bytes(rows)?).map_err(|_| invalid("MXFP8 records extent"))?;
    let activation = mxfp8_extent(&[u64::from(rows), g.input_features, 2])?;
    super::projected_buffers::validate_buffers(&[(buffers[0], 8), (buffers[1], activation),
        (buffers[2], records)], &[8, 8, 16], &[2]).map_err(invalid)
}

fn validate_mxfp8_project(g: DenseGeometry, rows: u32,
    buffers: [crate::CuteafdDeviceBuffer; 5], device: i32) -> Result<()> {
    if g.encoding != Encoding::Mxfp8A8 || buffers[0].device_id != device {
        return Err(invalid("MXFP8 encoding or bound device"));
    }
    let records = usize::try_from(g.scratch_bytes(rows)?).map_err(|_| invalid("MXFP8 records extent"))?;
    let payload = mxfp8_extent(&[g.input_features, g.output_features])?;
    let output = mxfp8_extent(&[u64::from(rows), g.output_features, 4])?;
    super::projected_buffers::validate_buffers(&[(buffers[0], 8), (buffers[1], payload),
        (buffers[2], payload / 32), (buffers[3], records), (buffers[4], output)],
        &[8, 8, 1, 16, 8], &[4]).map_err(invalid)
}

fn validate_linear_launch(g: DenseGeometry, pointers: [*mut c_void; 5],
    partial_bytes: u64, rows: u32, rotation: u32) -> Result<()> {
    let required = g.launch_scratch_bytes(rows)?;
    let trellis = matches!(g.encoding, Encoding::Trellis | Encoding::TrellisA8);
    if partial_bytes < required || (required > 0 && pointers[3].is_null())
        || [pointers[0], pointers[1], pointers[2], pointers[4]].iter().any(|p| p.is_null())
        || (trellis && rotation >= 4) || (!trellis && rotation != 0) {
        return Err(invalid("native dense launch storage or rotation"));
    }
    Ok(())
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
    fn native_dense_geometry_keys_validate_and_reject_mutations() {
        let (mut spec, entry) = trellis();
        spec.name = "exlr8_dense_trellis_k2048_n8192_s1".into();
        let legacy = validate_entry(&spec, &entry).unwrap();
        assert_eq!(legacy.input_features, 2048);
        for name in ["trellis_k02048_n8192_s1", "trellis_k2048_n8192_s0",
            "trellis_k2048_n8193_s1", "trellis_k2048_n8192_s1_junk",
            "trellis_k2048_n8192_s1_bf16_a8", "mxfp8_k128_n31_s1",
            "mxfp8_k64_n32_s1", "fp8_row_embedding_k4095_n154880",
            "trellis_k18446744073709551615_n8192_s1"] {
            spec.name = format!("exlr8_dense_{name}");
            assert!(validate_entry(&spec, &entry).is_err(), "{name}");
        }
        assert_eq!(keyed_geometry("mxfp8_k4096_n32_s1"), Some((Encoding::Mxfp8, 4096, 32, 1)));
        assert_eq!(keyed_geometry("mxfp8_k128_n8192_s1"), Some((Encoding::Mxfp8, 128, 8192, 1)));
        spec.name = "exlr8_dense_trellis_k1536_n8192_s1".into();
        assert!(validate_entry(&spec, &entry).is_err());
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
            "bits":4,"codebook":"e4m3","rotation_count":4,"max_rows":4096,"abi_version":2});
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
            ("rotation_count", json!(1)), ("bits", json!(3)),
            ("abi_version", json!(1)), ("abi_version", json!(null))] {
            let mut bad = spec.clone();
            bad.geometry[field] = value;
            assert!(validate_entry(&bad, &entry).is_err(), "{field}");
        }
        entry["pointers"][3]["role"] = json!("scratch");
        assert!(validate_entry(&spec, &entry).is_err());
    }

    #[test]
    fn native_mxfp8_a8_manifest_keeps_native_weights_and_bf16_ingress() {
        let (mut spec, mut entry) = trellis();
        spec.name = "exlr8_dense_mxfp8_k6144_n8192_s1_a8".into();
        spec.geometry = json!({"input_features":6144,"output_features":8192,
            "encoding":"mxfp8","sharded_side":1,"input_dtype":"bf16",
            "rotation_count":1,"max_rows":4096,"abi_version":2});
        spec.scratch.clear();
        spec.pointers[2] = "scales".into();
        spec.pointers[3] = "records".into();
        entry["pointers"] = json!([
            {"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
            {"name":"payload","dtype":"uint8","shape":"[50331648]","role":"in"},
            {"name":"scales","dtype":"uint8","shape":"[1572864]","role":"in"},
            {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"in"},
            {"name":"output","dtype":"float32","shape":"[rows,8192]","role":"out"}]);
        let g = validate_entry(&spec, &entry).unwrap();
        assert_eq!(g.encoding, Encoding::Mxfp8A8);
        for (rows, tiles) in [(1, 1), (16, 1), (17, 2), (4096, 256)] {
            assert_eq!(g.launch_scratch_bytes(rows).unwrap(), tiles * 192 * 528);
        }
        let pointer = std::ptr::NonNull::<u32>::dangling().as_ptr().cast();
        let records = g.scratch_bytes(17).unwrap();
        validate_linear_launch(g, [pointer; 5], records, 17, 0).unwrap();
        assert!(validate_linear_launch(g, [pointer; 5], records - 1, 17, 0).is_err());
        for rows in [0, 4097, u32::MAX] {
            assert!(validate_linear_launch(g, [pointer; 5], records, rows, 0).is_err());
        }
        for rotation in [1, 4, u32::MAX] {
            assert!(validate_linear_launch(g, [pointer; 5], records, 17, rotation).is_err());
        }
        for missing in 0..5 {
            let mut pointers = [pointer; 5];
            pointers[missing] = std::ptr::null_mut();
            assert!(validate_linear_launch(g, pointers, records, 17, 0).is_err());
        }
        for (field, value) in [("input_dtype", json!("f16")), ("encoding", json!("trellis")),
            ("rotation_count", json!(4)), ("abi_version", json!(1)), ("max_rows", json!(64))] {
            let mut bad = spec.clone();
            bad.geometry[field] = value;
            assert!(validate_entry(&bad, &entry).is_err(), "{field}");
        }
        for (pointer, field, value) in [(0, "shape", "[4]"), (2, "dtype", "float16"),
            (3, "role", "scratch")] {
            let mut bad = entry.clone();
            bad["pointers"][pointer][field] = json!(value);
            assert!(validate_entry(&spec, &bad).is_err(), "{pointer}/{field}");
        }
    }

    #[test]
    fn checked_mxfp8_operands_reject_extent_alignment_device_flags_and_aliasing() {
        use crate::CuteafdDeviceBuffer as Buffer;
        let g = DenseGeometry { encoding: Encoding::Mxfp8A8, input_features: 128,
            output_features: 32, side: 1, max_rows: 4096 };
        let buffer = |address, bytes| Buffer { ptr: address as *mut c_void, bytes, device_id: 1, flags: 0 };
        for rows in [1, 16, 17, 2752, 4096] {
            let records = g.scratch_bytes(rows).unwrap() as usize;
            let quant = [buffer(0x1000usize, 8), buffer(0x100000, rows as usize * 128 * 2),
                buffer(0x10000000, records)];
            let project = [quant[0], buffer(0x20000000, 4096), buffer(0x30000000, 128),
                quant[2], buffer(0x40000000, rows as usize * 32 * 4)];
            validate_mxfp8_quantize(g, rows, quant, 1).unwrap();
            validate_mxfp8_project(g, rows, project, 1).unwrap();
            let narrow = [project[4], buffer(0x50000000, rows as usize * 32 * 2)];
            validate_mxfp8_narrow(g, rows, narrow, 1).unwrap();
            for index in 0..narrow.len() {
                for mutation in 0..6 {
                    let mut bad = narrow;
                    match mutation {
                        0 => bad[index].bytes -= 1,
                        1 => bad[index].ptr = std::ptr::null_mut(),
                        2 => bad[index].device_id = 0,
                        3 => bad[index].flags = 1,
                        4 => bad[index].ptr = ((bad[index].ptr as usize) + 1) as *mut c_void,
                        _ => bad[index].ptr = (usize::MAX - 7) as *mut c_void,
                    }
                    assert!(validate_mxfp8_narrow(g, rows, bad, 1).is_err(), "narrow {index}/{mutation}");
                }
            }
            let mut overlapping = narrow;
            overlapping[1].ptr = overlapping[0].ptr;
            assert!(validate_mxfp8_narrow(g, rows, overlapping, 1).is_err());
            for index in 0..quant.len() {
                for mutation in 0..6 {
                    let mut bad = quant;
                    match mutation {
                        0 => bad[index].bytes -= 1,
                        1 => bad[index].ptr = std::ptr::null_mut(),
                        2 => bad[index].device_id = 0,
                        3 => bad[index].flags = 1,
                        4 => bad[index].ptr = ((bad[index].ptr as usize) + 1) as *mut c_void,
                        _ => bad[index].ptr = (usize::MAX - 7) as *mut c_void,
                    }
                    assert!(validate_mxfp8_quantize(g, rows, bad, 1).is_err(), "quant {index}/{mutation}");
                }
            }
            for index in 0..project.len() {
                for mutation in 0..6 {
                    if index == 2 && mutation == 4 { continue; } // Byte scales have no stronger alignment.
                    let mut bad = project;
                    match mutation {
                        0 => bad[index].bytes -= 1,
                        1 => bad[index].ptr = std::ptr::null_mut(),
                        2 => bad[index].device_id = 0,
                        3 => bad[index].flags = 1,
                        4 => bad[index].ptr = ((bad[index].ptr as usize) + 1) as *mut c_void,
                        _ => bad[index].ptr = (usize::MAX - 7) as *mut c_void,
                    }
                    assert!(validate_mxfp8_project(g, rows, bad, 1).is_err(), "project {index}/{mutation}");
                }
            }
            for source in 0..2 {
                let mut bad = quant;
                bad[2].ptr = bad[source].ptr;
                assert!(validate_mxfp8_quantize(g, rows, bad, 1).is_err());
            }
            for source in 0..4 {
                let mut bad = project;
                bad[4].ptr = bad[source].ptr;
                assert!(validate_mxfp8_project(g, rows, bad, 1).is_err());
            }
            for invalid in [0, 4097, u32::MAX] {
                assert!(validate_mxfp8_quantize(g, invalid, quant, 1).is_err());
                assert!(validate_mxfp8_project(g, invalid, project, 1).is_err());
                assert!(validate_mxfp8_narrow(g, invalid, narrow, 1).is_err());
            }
            assert!(validate_mxfp8_narrow(g, rows, narrow, 0).is_err());
            assert!(validate_mxfp8_quantize(g, rows, quant, 0).is_err());
            assert!(validate_mxfp8_project(g, rows, project, 0).is_err());
        }
    }

    #[test]
    fn checked_mxfp8_dispatch_separates_quantization_from_sibling_projections() -> anyhow::Result<()> {
        use crate::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let scalars = json!([{"name":"rows","type":"int32"},
            {"name":"rotation_id","type":"int32"},{"name":"cta_limit","type":"int32"}]);
        let manifest = json!({"programs":[
            {"name":"exlr8_dense_mxfp8_k128_n32_s1_a8","op":"dense_mxfp8",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":128,"output_features":32,"sharded_side":1,"encoding":"mxfp8",
                    "input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"payload","dtype":"uint8","shape":"[4096]","role":"in"},
                    {"name":"scales","dtype":"uint8","shape":"[128]","role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"in"},
                    {"name":"output","dtype":"float32","shape":"[rows,32]","role":"out"}]},
            {"name":"exlr8_dense_quant_k128_bf16_a8","op":"dense_mxfp8_quant",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":128,"input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"out"}]}
        ]});
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let path = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let names = manifest["programs"].as_array().unwrap();
        let op = Mxfp8A8::bind(&programs, names[0]["name"].as_str().unwrap(), names[1]["name"].as_str().unwrap())?;
        assert_eq!(fixture.events()?.matches('D').count(), 0);
        let mut buffers = Vec::new();
        for bytes in [8, 17 * 128 * 2, 2 * 4 * 528, 4096, 128, 17 * 32 * 4, 17 * 32 * 2] {
            buffers.push(library.alloc_device_buffer(bytes, crate::memory_ledger::Tag::workspace("fixture/mxfp8"))?);
        }
        library.copy_h2d(buffers[0], &(buffers[1].ptr as u64).to_le_bytes())?;
        // The CPU allocator marks host fallback for correct release accounting.
        // Only dispatch views simulate CUDA flags; keep the originals for free.
        let views = buffers.iter().map(|b| crate::CuteafdDeviceBuffer { flags: 0, ..*b }).collect::<Vec<_>>();
        // The no-math fixture observes dispatch only; it never emulates MXFP8.
        let ready = buffers[1].ptr as usize as u32;
        fixture.register_reply(&library, 0, ready, b'P')?;
        fixture.register_reply(&library, 1, ready, b'Q')?;
        let stream = std::ptr::null_mut();
        op.quantize(17, views[0], views[1], views[2], stream)?;
        for _ in 0..2 { op.project(17, views[0], views[3], views[4], views[2], views[5], stream)?; }
        assert_eq!(fixture.events()?.matches('Q').count(), 1);
        assert_eq!(fixture.events()?.matches('P').count(), 2);
        assert_eq!(fixture.events()?.matches('D').count(), 7);
        let values = [1.0f32, 1.00390625, -2.0078125, 0.33333334];
        let expected = [0x3f80u16, 0x3f80, 0xc000, 0x3eab];
        let source = values.iter().cycle().take(17 * 32)
            .flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        library.copy_h2d(buffers[5], &source)?;
        op.narrow(17, views[5], views[6], stream)?;
        let mut actual = vec![0u8; 17 * 32 * 2];
        // SAFETY: `stream` produced `buffers[6]` and remains live on this device.
        unsafe { library.copy_d2h(&mut actual, buffers[6], stream)?; }
        for (index, bytes) in actual.chunks_exact(2).enumerate() {
            assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), expected[index % expected.len()]);
        }
        library.cuda_set_device(1)?;
        assert!(op.quantize(17, views[0], views[1], views[2], stream).is_err());
        assert!(op.project(17, views[0], views[3], views[4], views[2], views[5], stream).is_err());
        assert!(op.narrow(17, views[5], views[6], stream).is_err());
        library.cuda_set_device(0)?;
        assert_eq!(fixture.events()?.matches('Q').count(), 1);
        assert_eq!(fixture.events()?.matches('P').count(), 2);
        for buffer in &mut buffers { library.free_device_buffer(buffer)?; }
        Ok(())
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
