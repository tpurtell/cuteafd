//! Coordinator programs: the exported b12x programs behind the generic
//! launcher in `native/shared/src/dsv4_programs.cc`. DeepSeek V4 (`dsv4f_*`,
//! `dsv4p_*`, `CUTEAFD_ENABLE_DSV4_AOT`) and GLM 5.x (`glm_*`,
//! `CUTEAFD_ENABLE_GLM_AOT`) share the table and its manifest.
//!
//! Callers name each program and pass its pointers in the documented order;
//! [`Programs::with_manifest`] checks that order against the exporter's
//! `dsv4_programs.json` so a stale image cannot silently shift arguments.
use crate::NativeLibrary;
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::path::Path;

/// Extents compiled into the coordinator programs. They describe one
/// sequence's index, not the shared token pool or its physical GPU replicas.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgramCapacities {
    pub decode_rows: Option<usize>,
    pub prefill_rows: Option<usize>,
    pub max_context: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramCapacityError {
    Invalid { field: &'static str },
    MissingContext { family: &'static str },
    ContextExceeded { family: &'static str, requested: usize, compiled: usize },
}

impl fmt::Display for ProgramCapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { field } => write!(f, "program manifest capacities.{field} must be a positive integer that fits usize"),
            Self::MissingContext { family } => write!(f, "{family}: program manifest lacks capacities.max_context; export matching coordinator programs with an explicit --max-context before loading weights"),
            Self::ContextExceeded { family, requested, compiled } => write!(f, "{family}: requested context {requested} exceeds compiled index extent {compiled}; export matching coordinator programs with CUTEAFD_DSV4_MAX_CONTEXT={requested} (--max-context {requested}), or lower --max-context to {compiled}"),
        }
    }
}

impl std::error::Error for ProgramCapacityError {}

impl ProgramCapacities {
    fn from_manifest(manifest: &serde_json::Value) -> std::result::Result<Self, ProgramCapacityError> {
        let capacities = &manifest["capacities"];
        let field = |name: &'static str| match capacities.get(name) {
            None => Ok(None),
            Some(value) => value.as_u64().and_then(|v| usize::try_from(v).ok()).filter(|&v| v > 0)
                .map(Some).ok_or(ProgramCapacityError::Invalid { field: name }),
        };
        Ok(Self { decode_rows: field("decode_rows")?, prefill_rows: field("prefill_rows")?,
            max_context: field("max_context")? })
    }

    /// Indexed families must not send a wider page table than the compiled
    /// top-k route or its scratch can hold. Check before engine allocations.
    pub fn require_context(self, family: &'static str, requested: usize)
        -> std::result::Result<(), ProgramCapacityError> {
        let compiled = self.max_context.ok_or(ProgramCapacityError::MissingContext { family })?;
        if requested > compiled {
            return Err(ProgramCapacityError::ContextExceeded { family, requested, compiled });
        }
        if requested == 0 {
            return Err(ProgramCapacityError::Invalid { field: "requested max_context" });
        }
        Ok(())
    }
}

#[repr(C)]
struct ProgramInfo {
    name: [u8; 64],
    pointers: u32,
    scalars: u32,
    scalar_kinds: [u8; 16],
}

impl Default for ProgramInfo {
    fn default() -> Self {
        Self { name: [0; 64], pointers: 0, scalars: 0, scalar_kinds: [0; 16] }
    }
}

type CountFn = unsafe extern "C" fn() -> u32;
type InfoFn = unsafe extern "C" fn(u32, *mut ProgramInfo) -> i32;
type LoadFn = unsafe extern "C" fn(u32) -> i32;
type LaunchFn = unsafe extern "C" fn(u32, *const *mut c_void, *const u64, *mut c_void) -> i32;

/// One scalar launch argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    I32(i32),
    I64(i64),
    F32(f32),
}

impl Scalar {
    fn kind(self) -> u8 {
        match self {
            Self::I32(_) => b'i',
            Self::I64(_) => b'l',
            Self::F32(_) => b'f',
        }
    }
    fn slot(self) -> u64 {
        match self {
            Self::I32(v) => v as i64 as u64,
            Self::I64(v) => v as u64,
            Self::F32(v) => u64::from(v.to_bits()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProgramSpec {
    pub name: String,
    pub pointers: Vec<String>,
    pub scalar_kinds: Vec<u8>,
    /// Scratch pointer name -> bytes at the program's capacity (manifest only).
    pub scratch: HashMap<String, u64>,
    pub capacity_rows: u32,
}

pub struct Programs<'a> {
    library: &'a NativeLibrary,
    load: LoadFn,
    launch: LaunchFn,
    programs: HashMap<String, (u32, ProgramSpec)>,
    capacities: ProgramCapacities,
}

impl NativeLibrary {
    /// The DeepSeek V4 programs this library was built with.
    pub fn programs(&self) -> Result<Programs<'_>> {
        let count = *unsafe { self.lib.get::<CountFn>(b"cuteafd_dsv4_program_count") }
            .context("native library was built without coordinator programs (CUTEAFD_ENABLE_DSV4_AOT / CUTEAFD_ENABLE_GLM_AOT)")?;
        let info = unsafe { *self.lib.get::<InfoFn>(b"cuteafd_dsv4_program_info")? };
        let load = unsafe { *self.lib.get::<LoadFn>(b"cuteafd_dsv4_program_load")? };
        let launch = unsafe { *self.lib.get::<LaunchFn>(b"cuteafd_dsv4_program_launch")? };
        let mut programs = HashMap::new();
        // SAFETY: the count and info entry points only read the static table.
        for index in 0..unsafe { count() } {
            let mut raw = ProgramInfo::default();
            let status = unsafe { info(index, &mut raw) };
            ensure!(status == 0, "DeepSeek V4 program {index} info failed with {status}");
            let name = c_string(&raw.name);
            let kinds = c_string(&raw.scalar_kinds).into_bytes();
            ensure!(kinds.len() == raw.scalars as usize, "program {name} scalar table is inconsistent");
            let spec = ProgramSpec {
                name: name.clone(),
                // Positional until a manifest names them.
                pointers: (0..raw.pointers).map(|i| format!("#{i}")).collect(),
                scalar_kinds: kinds,
                scratch: HashMap::new(),
                capacity_rows: 0,
            };
            programs.insert(name, (index, spec));
        }
        Ok(Programs { library: self, load, launch, programs, capacities: ProgramCapacities::default() })
    }
}

impl<'a> Programs<'a> {
    /// Attaches pointer names, scratch sizes and capacities from the
    /// exporter manifest, rejecting any disagreement with the native table.
    /// Images built before the rename ship the manifest as DSV4_PROGRAMS.json;
    /// a missing PROGRAMS.json falls back to that name beside it.
    pub fn with_manifest(mut self, path: &Path) -> Result<Self> {
        let fallback = path.with_file_name("DSV4_PROGRAMS.json");
        let path = if !path.exists() && path.file_name() == Some("PROGRAMS.json".as_ref()) && fallback.exists() {
            fallback.as_path()
        } else {
            path
        };
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let manifest: serde_json::Value = serde_json::from_str(&text)?;
        self.capacities = ProgramCapacities::from_manifest(&manifest)?;
        let entries = manifest["programs"].as_array().context("manifest has no programs")?;
        ensure!(entries.len() == self.programs.len(), "manifest lists {} programs, the library {}",
            entries.len(), self.programs.len());
        for entry in entries {
            let name = entry["name"].as_str().context("program without a name")?;
            let (_, spec) = self.programs.get_mut(name)
                .with_context(|| format!("manifest program {name} is not in the library"))?;
            let pointers: Vec<String> = entry["pointers"].as_array().context("pointers")?
                .iter().map(|p| p["name"].as_str().unwrap_or_default().to_string()).collect();
            ensure!(pointers.len() == spec.pointers.len(), "{name}: manifest and library disagree on pointers");
            spec.pointers = pointers;
            spec.capacity_rows = entry["capacity_rows"].as_u64().unwrap_or(0) as u32;
            if let Some(scratch) = entry["scratch_bytes_at_capacity"].as_object() {
                spec.scratch = scratch.iter().filter_map(|(k, v)| Some((k.clone(), v.as_u64()?))).collect();
            }
        }
        Ok(self)
    }

    /// Loads every program's kernels on the current device (startup, before
    /// the first request pays for it).
    pub fn load_all(&self) -> Result<()> {
        self.load_matching(|_| true).map(|_| ())
    }

    /// Loads the kernels of the programs whose names `keep` accepts on the current device, and
    /// returns how many it loaded and skipped. A skipped program still loads when it is first
    /// resolved ([`Self::program`]), so `keep` must accept every program a caller launches where
    /// a lazy load cannot happen (inside a stream capture).
    pub fn load_matching(&self, keep: impl Fn(&str) -> bool) -> Result<(usize, usize)> {
        let (mut loaded, mut skipped) = (0, 0);
        for (name, (index, _)) in &self.programs {
            if !keep(name) {
                skipped += 1;
                continue;
            }
            // SAFETY: loading reads the static table and loads a CUDA library.
            let status = unsafe { (self.load)(*index) };
            ensure!(status == 0, "loading {name} failed with CUDA status {status}");
            loaded += 1;
        }
        Ok((loaded, skipped))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.programs.keys().map(String::as_str)
    }

    pub fn spec(&self, name: &str) -> Result<&ProgramSpec> {
        Ok(&self.programs.get(name).with_context(|| format!("no DeepSeek V4 program {name}"))?.1)
    }

    pub fn capacities(&self) -> ProgramCapacities {
        self.capacities
    }

    /// Resolves a program and loads its kernels on the current device.
    pub fn program(&self, name: &str, pointer_names: &[&str]) -> Result<Program<'_>> {
        let (index, spec) = self.programs.get(name).with_context(|| format!("no DeepSeek V4 program {name}"))?;
        ensure!(pointer_names.len() == spec.pointers.len(), "{name} takes {} pointers, caller passes {}",
            spec.pointers.len(), pointer_names.len());
        if !spec.pointers[0].starts_with('#') {
            for (expected, given) in spec.pointers.iter().zip(pointer_names) {
                ensure!(expected == given, "{name}: pointer order {:?} != {:?}", spec.pointers, pointer_names);
            }
        }
        // SAFETY: loading reads the program table and loads a CUDA library on
        // the caller's current device.
        let status = unsafe { (self.load)(*index) };
        ensure!(status == 0, "loading {name} failed with CUDA status {status}");
        Ok(Program { programs: self, index: *index, spec })
    }

    pub fn library(&self) -> &'a NativeLibrary {
        self.library
    }
}

pub struct Program<'a> {
    programs: &'a Programs<'a>,
    index: u32,
    spec: &'a ProgramSpec,
}

impl Program<'_> {
    pub fn spec(&self) -> &ProgramSpec {
        self.spec
    }

    /// # Safety
    /// Every pointer must reference live device memory of the documented
    /// shape for `rows`, on the device the program was loaded on, and stay
    /// valid until the stream reaches this launch.
    pub unsafe fn launch(&self, pointers: &[*mut c_void], scalars: &[Scalar], stream: *mut c_void) -> Result<()> {
        ensure!(pointers.len() == self.spec.pointers.len(), "{}: pointer count", self.spec.name);
        ensure!(scalars.len() == self.spec.scalar_kinds.len(), "{}: scalar count", self.spec.name);
        for (scalar, kind) in scalars.iter().zip(&self.spec.scalar_kinds) {
            if scalar.kind() != *kind {
                bail!("{}: scalar kinds {:?} do not match {:?}", self.spec.name,
                    String::from_utf8_lossy(&self.spec.scalar_kinds), scalars);
            }
        }
        let slots: Vec<u64> = scalars.iter().map(|s| s.slot()).collect();
        // SAFETY: counts and kinds were checked against the native table; the
        // caller guarantees pointer validity (see the function contract).
        let status = unsafe { (self.programs.launch)(self.index, pointers.as_ptr(), slots.as_ptr(), stream) };
        ensure!(status == 0, "{} launch failed with status {status}", self.spec.name);
        Ok(())
    }
}

fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_slots() {
        assert_eq!(Scalar::I32(-1).slot(), u64::MAX);
        assert_eq!(Scalar::F32(1.0).slot(), 0x3f80_0000);
        assert_eq!(c_string(b"abc\0def"), "abc");
    }

    #[test]
    fn indexed_context_uses_the_manifest_extent_not_the_pool_size() {
        let manifest = serde_json::json!({"capacities": {
            "decode_rows": 64, "prefill_rows": 4096, "max_context": 131072
        }});
        let limits = ProgramCapacities::from_manifest(&manifest).unwrap();
        assert_eq!(limits.decode_rows, Some(64));
        assert_eq!(limits.prefill_rows, Some(4096));
        for family in ["glm5", "glm5_flash", "qwen4"] {
            limits.require_context(family, 1).unwrap();
            limits.require_context(family, 131072).unwrap();
            let error = limits.require_context(family, 131073).unwrap_err();
            assert_eq!(error, ProgramCapacityError::ContextExceeded {
                family, requested: 131073, compiled: 131072,
            });
            assert!(error.to_string().contains("CUTEAFD_DSV4_MAX_CONTEXT=131073"));
        }
    }

    #[test]
    fn missing_or_invalid_index_extent_is_unsupported_before_allocation() {
        for manifest in [serde_json::json!({}), serde_json::json!({"capacities": {"decode_rows": 64}})] {
            assert_eq!(ProgramCapacities::from_manifest(&manifest).unwrap().require_context("glm5_flash", 1),
                Err(ProgramCapacityError::MissingContext { family: "glm5_flash" }));
        }
        for value in [serde_json::json!(0), serde_json::json!(-1), serde_json::json!(null),
            serde_json::json!(1.5), serde_json::json!("131072")] {
            assert_eq!(ProgramCapacities::from_manifest(&serde_json::json!({"capacities": {"max_context": value}})),
                Err(ProgramCapacityError::Invalid { field: "max_context" }));
        }
        assert!(ProgramCapacities { max_context: Some(131072), ..Default::default() }
            .require_context("qwen4", 0).is_err());
    }
}

type HeadLaunchFn = unsafe extern "C" fn(*mut c_void, *const u16, *const u16, *mut f32, i32, *mut c_void) -> i32;

/// cuBLAS vocabulary head at the model width with pedantic FP32 accumulation
/// (the reference promotes the projection to FP32); logits that need not
/// follow it may take the tensor-op launch ([`VocabularyHead::launch_tensor_op`]).
pub struct VocabularyHead<'a> {
    library: &'a NativeLibrary,
    handle: *mut c_void,
    launch: HeadLaunchFn,
    /// `cuteafd_vocabulary_head_launch_tensor_op`, when the native library has it.
    launch_tensor_op: Option<HeadLaunchFn>,
}

/// Bytes of caller-owned cuBLAS workspace the head needs.
pub const VOCABULARY_HEAD_WORKSPACE: usize = 4 << 20;

impl NativeLibrary {
    /// # Safety
    /// `workspace` must be live device memory of at least
    /// [`VOCABULARY_HEAD_WORKSPACE`] bytes that outlives the head.
    pub unsafe fn vocabulary_head(&self, workspace: *mut c_void, width: u32, max_rows: u32) -> Result<VocabularyHead<'_>> {
        type Create = unsafe extern "C" fn(*mut c_void, u64, i32, i32, *mut *mut c_void) -> i32;
        let create = *unsafe { self.lib.get::<Create>(b"cuteafd_vocabulary_head_create") }?;
        let launch = *unsafe { self.lib.get(b"cuteafd_vocabulary_head_launch_width") }?;
        let launch_tensor_op = unsafe { self.lib.get(b"cuteafd_vocabulary_head_launch_tensor_op") }.ok().map(|f| *f);
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            create(workspace, VOCABULARY_HEAD_WORKSPACE as u64, i32::try_from(width)?, i32::try_from(max_rows)?, &mut handle)
        };
        ensure!(status == 0, "vocabulary head creation failed with {status}");
        Ok(VocabularyHead { library: self, handle, launch, launch_tensor_op })
    }
}

impl NativeLibrary {
    /// DeepSeek V4 routing from FP32 router logits (see
    /// `cuteafd_dsv4_router_select`): exactly one of `bias` (score layers) and
    /// `tid2eid` (hash layers, with `tokens`) is non-null.
    ///
    /// # Safety
    /// Every pointer is live device memory of its documented shape on the
    /// stream's device; `logits` may be rewritten in place.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dsv4_router_select(&self, logits: *mut c_void, bias: *const c_void, tid2eid: *const c_void,
        tokens: *const c_void, ids: *mut c_void, routing: *mut c_void, rows: usize, experts: usize, topk: usize,
        route_scale: f32, stream: *mut c_void) -> Result<()> {
        type Select = unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *const c_void, *mut c_void,
            *mut c_void, i32, i32, i32, f32, *mut c_void) -> i32;
        let select = *unsafe { self.lib.get::<Select>(b"cuteafd_dsv4_router_select") }?;
        let status = unsafe {
            select(logits, bias, tid2eid, tokens, ids, routing, i32::try_from(rows)?, i32::try_from(experts)?,
                i32::try_from(topk)?, route_scale, stream)
        };
        ensure!(status == 0, "DeepSeek V4 router select failed with {status}");
        Ok(())
    }

    /// [`Self::dsv4_router_select`] with `sigmoid` selecting GLM's sigmoid
    /// scores (noaux_tc) instead of sqrtsoftplus.
    ///
    /// # Safety
    /// As [`Self::dsv4_router_select`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn router_select(&self, logits: *mut c_void, bias: *const c_void, tid2eid: *const c_void,
        tokens: *const c_void, ids: *mut c_void, routing: *mut c_void, rows: usize, experts: usize, topk: usize,
        route_scale: f32, sigmoid: bool, stream: *mut c_void) -> Result<()> {
        type Select = unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *const c_void, *mut c_void,
            *mut c_void, i32, i32, i32, f32, i32, *mut c_void) -> i32;
        let select = *unsafe { self.lib.get::<Select>(b"cuteafd_router_select") }?;
        let status = unsafe {
            select(logits, bias, tid2eid, tokens, ids, routing, i32::try_from(rows)?, i32::try_from(experts)?,
                i32::try_from(topk)?, route_scale, i32::from(sigmoid), stream)
        };
        ensure!(status == 0, "DeepSeek V4 router select failed with {status}");
        Ok(())
    }
}

impl NativeLibrary {
    /// Softmax top-`topk` routing (Qwen, `cuteafd_router_select_softmax`):
    /// FP32 `logits` [rows, experts] (rounded to BF16 first with
    /// `round_bf16`), U32 `ids` and FP32 `routing` [rows, topk] = the top-k
    /// softmax probabilities renormalized over the top-k (BF16-rounded with
    /// `round_bf16`) times `route_scale`.
    ///
    /// # Safety
    /// Device pointers of those sizes, live until the stream drains.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn router_select_softmax(&self, logits: *const c_void, ids: *mut c_void, routing: *mut c_void,
        rows: usize, experts: usize, topk: usize, route_scale: f32, round_bf16: bool, stream: *mut c_void)
        -> Result<()> {
        type Select = unsafe extern "C" fn(*const c_void, *mut c_void, *mut c_void, i32, i32, i32, f32, i32,
            *mut c_void) -> i32;
        let select = *unsafe { self.lib.get::<Select>(b"cuteafd_router_select_softmax") }?;
        let status = unsafe {
            select(logits, ids, routing, i32::try_from(rows)?, i32::try_from(experts)?, i32::try_from(topk)?,
                route_scale, i32::from(round_bf16), stream)
        };
        ensure!(status == 0, "softmax router select failed with {status}");
        Ok(())
    }
}

/// dSpark drafter kernels (native/families/deepseek_v4/cuda/dsv4_dspark.cu).
impl NativeLibrary {
    /// `out` BF16 [rows, cols] = FP8 E4M3 `w` [rows, cols] times its FP32
    /// 128x128 block `scale` [ceil(rows/128), ceil(cols/128)].
    ///
    /// # Safety
    /// Every pointer is live device memory of its shape on the stream's device.
    pub unsafe fn fp8_block_dequant(&self, w: *const c_void, scale: *const c_void, out: *mut c_void, rows: usize,
        cols: usize, stream: *mut c_void) -> Result<()> {
        type Dequant = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Dequant>(b"cuteafd_fp8_block_dequant") }?;
        let status = unsafe { f(w, scale, out, i32::try_from(rows)?, i32::try_from(cols)?, stream) };
        ensure!(status == 0, "FP8 block dequantization failed with {status}");
        Ok(())
    }

    /// BF16 `w` [rows, cols] -> E4M3 `q` [rows, cols] with FP32 128x128 block
    /// `scale` [ceil(rows/128), ceil(cols/128)] (amax / 448 per block), the
    /// inverse of [`Self::fp8_block_dequant`].
    ///
    /// # Safety
    /// Every pointer is live device memory of its shape on the stream's device.
    pub unsafe fn fp8_block_quant(&self, w: *const c_void, q: *mut c_void, scale: *mut c_void, rows: usize,
        cols: usize, stream: *mut c_void) -> Result<()> {
        type Quant = unsafe extern "C" fn(*const c_void, *mut c_void, *mut c_void, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Quant>(b"cuteafd_fp8_block_quant") }?;
        let status = unsafe { f(w, q, scale, i32::try_from(rows)?, i32::try_from(cols)?, stream) };
        ensure!(status == 0, "FP8 block quantization failed with {status}");
        Ok(())
    }

    /// BF16 `w` [rows, cols] -> E4M3 `q` [rows, cols] with FP32 `scale`
    /// [rows, cols / 128], one per row and 128-wide K block (amax / 448),
    /// the layout of the MmaFp8Gemv row-scale programs. `cols % 128 == 0`.
    ///
    /// # Safety
    /// Every pointer is live device memory of its shape on the stream's device.
    pub unsafe fn fp8_row_quant(&self, w: *const c_void, q: *mut c_void, scale: *mut c_void, rows: usize,
        cols: usize, stream: *mut c_void) -> Result<()> {
        type Quant = unsafe extern "C" fn(*const c_void, *mut c_void, *mut c_void, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Quant>(b"cuteafd_fp8_row_quant") }?;
        let status = unsafe { f(w, q, scale, i32::try_from(rows)?, i32::try_from(cols)?, stream) };
        ensure!(status == 0, "FP8 row quantization failed with {status}");
        Ok(())
    }

    /// [`Self::fp8_block_quant`] / [`Self::fp8_row_quant`] under scale `rule`
    /// (0 amax / 448, 1 the smallest power of two >= it, 2 whichever of the
    /// two leaves the smaller squared error per block); `row` selects the
    /// per-row x 128-K layout.
    ///
    /// # Safety
    /// Every pointer is live device memory of its shape on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_quant_rule(&self, w: *const c_void, q: *mut c_void, scale: *mut c_void, rows: usize,
        cols: usize, row: bool, rule: i32, stream: *mut c_void) -> Result<()> {
        type Quant = unsafe extern "C" fn(*const c_void, *mut c_void, *mut c_void, i32, i32, i32, *mut c_void) -> i32;
        let name: &[u8] = if row { b"cuteafd_fp8_row_quant_rule" } else { b"cuteafd_fp8_block_quant_rule" };
        let f = *unsafe { self.lib.get::<Quant>(name) }?;
        let status = unsafe { f(w, q, scale, i32::try_from(rows)?, i32::try_from(cols)?, rule, stream) };
        ensure!(status == 0, "FP8 quantization (rule {rule}) failed with {status}");
        Ok(())
    }

    /// Mean over the four mHC copies of `rows` stream rows [rows, 4, hidden]
    /// into columns `offset..offset + hidden` of `out` [rows, stride], BF16.
    ///
    /// # Safety
    /// Both buffers are live device memory of those shapes on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dsv4_hc_mean(&self, stream: *const c_void, out: *mut c_void, rows: usize, hidden: usize,
        stride: usize, offset: usize, cuda_stream: *mut c_void) -> Result<()> {
        type HcMean = unsafe extern "C" fn(*const c_void, *mut c_void, i32, i32, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<HcMean>(b"cuteafd_dsv4_hc_mean") }?;
        let status = unsafe {
            f(stream, out, i32::try_from(rows)?, i32::try_from(hidden)?, i32::try_from(stride)?,
                i32::try_from(offset)?, cuda_stream)
        };
        ensure!(status == 0, "dSpark hc mean failed with {status}");
        Ok(())
    }

    /// `out` [rows, n] BF16 = RMSNorm(x [rows, k] BF16 @ w^T) * norm, `w` FP8
    /// E4M3 [n, k] with UE8M0 128x128 block `scale`; `work` holds rows * n FP32.
    ///
    /// # Safety
    /// Every pointer is live device memory of its shape on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dsv4_fp8_linear_rmsnorm(&self, x: *const c_void, w: *const c_void, scale: *const c_void,
        norm: *const c_void, out: *mut c_void, work: *mut c_void, rows: usize, n: usize, k: usize, eps: f32,
        cuda_stream: *mut c_void) -> Result<()> {
        type Linear = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *const c_void, *mut c_void,
            *mut c_void, i32, i32, i32, f32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Linear>(b"cuteafd_dsv4_fp8_linear_rmsnorm") }?;
        let status = unsafe {
            f(x, w, scale, norm, out, work, i32::try_from(rows)?, i32::try_from(n)?, i32::try_from(k)?, eps, cuda_stream)
        };
        ensure!(status == 0, "dSpark main projection failed with {status}");
        Ok(())
    }

    /// Workspace bytes [`Self::dsv4_markov_drafts`] needs for `sequences`.
    pub fn dsv4_markov_workspace(&self, sequences: usize) -> Result<usize> {
        type Workspace = unsafe extern "C" fn(i32) -> u64;
        let f = *unsafe { self.lib.get::<Workspace>(b"cuteafd_dsv4_markov_workspace") }?;
        Ok(usize::try_from(unsafe { f(i32::try_from(sequences)?) })?)
    }

    /// Greedy dSpark drafts [sequences, block] U32 from head `logits`
    /// [sequences * block, vocab] FP32 and the Markov head (`w1`, `w2` BF16
    /// [vocab, rank]); step 0 conditions on `first` [sequences] U32.
    ///
    /// # Safety
    /// Every pointer is live device memory of its shape on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dsv4_markov_drafts(&self, logits: *const c_void, w1: *const c_void, w2: *const c_void,
        first: *const c_void, drafts: *mut c_void, workspace: *mut c_void, sequences: usize, block: usize,
        vocab: usize, rank: usize, cuda_stream: *mut c_void) -> Result<()> {
        type Drafts = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *const c_void, *mut c_void,
            *mut c_void, i32, i32, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Drafts>(b"cuteafd_dsv4_markov_drafts") }?;
        let status = unsafe {
            f(logits, w1, w2, first, drafts, workspace, i32::try_from(sequences)?, i32::try_from(block)?,
                i32::try_from(vocab)?, i32::try_from(rank)?, cuda_stream)
        };
        ensure!(status == 0, "dSpark Markov drafts failed with {status}");
        Ok(())
    }
}

impl NativeLibrary {
    /// [`Self::vocabulary_head`] for a `vocab`-row head (GLM: 154880).
    ///
    /// # Safety
    /// As [`Self::vocabulary_head`].
    pub unsafe fn vocabulary_head_rows(&self, workspace: *mut c_void, width: u32, max_rows: u32, vocab: u32)
        -> Result<VocabularyHead<'_>> {
        type Create = unsafe extern "C" fn(*mut c_void, u64, i32, i32, i32, *mut *mut c_void) -> i32;
        let create = *unsafe { self.lib.get::<Create>(b"cuteafd_vocabulary_head_create_vocab") }?;
        let launch = *unsafe { self.lib.get(b"cuteafd_vocabulary_head_launch_width") }?;
        let launch_tensor_op = unsafe { self.lib.get(b"cuteafd_vocabulary_head_launch_tensor_op") }.ok().map(|f| *f);
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            create(workspace, VOCABULARY_HEAD_WORKSPACE as u64, i32::try_from(width)?, i32::try_from(max_rows)?,
                i32::try_from(vocab)?, &mut handle)
        };
        ensure!(status == 0, "vocabulary head creation failed with {status}");
        Ok(VocabularyHead { library: self, handle, launch, launch_tensor_op })
    }
}

impl VocabularyHead<'_> {
    /// # Safety
    /// `input` BF16 [rows, width], `weight` BF16 [vocab, width] and `logits`
    /// FP32 [rows, vocab] must be live device memory on the head's device.
    pub unsafe fn launch(&self, input: *const u16, weight: *const u16, logits: *mut f32, rows: u32,
        stream: *mut c_void) -> Result<()> {
        let status = unsafe { (self.launch)(self.handle, input, weight, logits, i32::try_from(rows)?, stream) };
        ensure!(status == 0, "vocabulary head launch failed with {status}");
        Ok(())
    }

    /// [`Self::launch`] as a BF16 tensor-core GEMM with FP32 accumulation and
    /// FP32 logits, reading the head once for every row count. For logits that
    /// need not follow the reference's FP32 promotion, such as a drafter's,
    /// whose proposals the target verifies; the caller chooses.
    ///
    /// # Safety
    /// As [`Self::launch`].
    pub unsafe fn launch_tensor_op(&self, input: *const u16, weight: *const u16, logits: *mut f32, rows: u32,
        stream: *mut c_void) -> Result<()> {
        let launch = self.launch_tensor_op.context(
            "this native library has no cuteafd_vocabulary_head_launch_tensor_op (rebuild it with this checkout)")?;
        let status = unsafe { launch(self.handle, input, weight, logits, i32::try_from(rows)?, stream) };
        ensure!(status == 0, "tensor-op vocabulary head launch of {rows} rows failed with {status}");
        Ok(())
    }
}

impl Drop for VocabularyHead<'_> {
    fn drop(&mut self) {
        type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
        // SAFETY: the handle came from the matching create entry point.
        if let Ok(destroy) = unsafe { self.library.lib.get::<Destroy>(b"cuteafd_v41_markov_destroy") } {
            unsafe { destroy(self.handle) };
        }
    }
}
