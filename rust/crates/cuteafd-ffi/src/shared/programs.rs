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
    pub geometry: serde_json::Value,
}

pub struct Programs<'a> {
    library: &'a NativeLibrary,
    load: LoadFn,
    launch: LaunchFn,
    programs: HashMap<String, (u32, ProgramSpec)>,
    card_specs: HashMap<String, [ProgramSpec; 2]>,
    capacities: ProgramCapacities,
    families: serde_json::Value,
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
                geometry: serde_json::Value::Null,
            };
            programs.insert(name, (index, spec));
        }
        Ok(Programs { library: self, load, launch, programs, card_specs: HashMap::new(), capacities: ProgramCapacities::default(), families: serde_json::Value::Null })
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
        let mut manifest: serde_json::Value = serde_json::from_str(&text)?;
        if manifest["programs"].as_array().is_some_and(|programs| programs.iter().any(|p| p.get("card_plans").is_some())) {
            cuteafd_core::coordinator_programs::select_card_plans(&mut manifest,
                self.library.sm_count()?.try_into().context("runtime SM count exceeds u32")?)?;
        }
        self.capacities = ProgramCapacities::from_manifest(&manifest)?;
        self.families = manifest.get("families").cloned().unwrap_or_default();
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
            spec.geometry = entry["geometry"].clone();
            if let Some(scratch) = entry["scratch_bytes_at_capacity"].as_object() {
                spec.scratch = scratch.iter().filter_map(|(k, v)| Some((k.clone(), v.as_u64()?))).collect();
            }
            if let Some(plans) = entry["card_plans"].as_array() {
                let mut specs = [spec.clone(), spec.clone()];
                for (index, sms) in [170, 188].into_iter().enumerate() {
                    let plan = plans.iter().find(|p| p["sm_count"].as_u64() == Some(sms))
                        .context("validated card plan disappeared")?;
                    specs[index].geometry = plan["geometry"].clone();
                    specs[index].scratch = plan["scratch_bytes_at_capacity"].as_object().context("card scratch")?
                        .iter().map(|(k, v)| Ok((k.clone(), v.as_u64().context("card scratch bytes")?)))
                        .collect::<Result<_>>()?;
                }
                self.card_specs.insert(name.to_string(), specs);
            }
            if name.starts_with("glm_stage_") || name.starts_with("glm2_stage_") {
                super::glm_stage::validate_entry(spec, entry)?;
            } else if name.starts_with("exlr8_dense_") {
                // Companions (transforms, unpack, prefill GEMM, glue) by their op; the
                // ABI-v1 linears and the embedding gather by name.
                if super::native_dense_glue::op_of(entry)?.is_some() {
                    super::native_dense_glue::validate_entry(spec, entry)?;
                } else {
                    super::native_dense::validate_entry(spec, entry)?;
                }
            } else if name.starts_with("exlr8_") {
                super::exlr8::validate_entry(spec, entry)?;
                let expected: &[(&str, &str)] = match name {
                    "exlr8_a16_grouped" | "exlr8_a8_grouped" => &[("rows", "int32"), ("cta_limit", "int32"), ("unit_limit", "int32")],
                    "exlr8_a16_serving" | "exlr8_shared_a16_serving" => &[("rows", "int32"), ("cta_limit", "int32")],
                    "exlr8_group" | "exlr8_shared_group" | "exlr8_grouped_group" => &[("rows", "int32"), ("cta_limit", "int32")],
                    _ => &[("rows", "int32"), ("groups", "int32"), ("cta_limit", "int32")],
                };
                let scalars = entry["scalars"].as_array().context("EXLR8 scalar ABI missing")?;
                ensure!(scalars.len() == expected.len() && scalars.iter().zip(expected).all(
                    |(s, (n, t))| s["name"].as_str() == Some(*n) && s["type"].as_str() == Some(*t)),
                    "EXLR8 scalar names or types disagree with ABI");
            }
        }
        Ok(self)
    }

    /// Set recipe scratch before any serving owner allocates workspaces.
    /// Retained per-card specs are narrowed independently; native ABI is unchanged.
    pub fn with_attention_plan(mut self, plan: cuteafd_core::coordinator_programs::AttentionReductionPlan,
        keep: impl Fn(&str) -> bool) -> Result<Self> {
        let select = |spec: &mut ProgramSpec| -> Result<()> {
            cuteafd_core::coordinator_programs::select_attention_scratch(&spec.geometry, &mut spec.scratch, plan)?;
            Ok(())
        };
        for (_, spec) in self.programs.values_mut() {
            if keep(&spec.name) { select(spec)?; }
        }
        for (name, specs) in &mut self.card_specs {
            if keep(name) { for spec in specs { select(spec)?; } }
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
            ensure!(status != -12001,
                "loading {name} refused: insufficient resident SMs for its fused diagnostic CTA barrier");
            ensure!(status == 0, "loading {name} failed with CUDA status {status}");
            loaded += 1;
        }
        Ok((loaded, skipped))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.programs.keys().map(String::as_str)
    }

    pub fn spec(&self, name: &str) -> Result<&ProgramSpec> {
        if let Some(specs) = self.card_specs.get(name) {
            let sms = self.library.sm_count()?;
            ensure!(sms > 0, "runtime SM count must be positive");
            return Ok(&specs[usize::from(sms >= 188)]);
        }
        Ok(&self.programs.get(name).with_context(|| format!("no DeepSeek V4 program {name}"))?.1)
    }

    pub fn capacities(&self) -> ProgramCapacities {
        self.capacities
    }

    /// Package-owned policies accompany the program table; callers validate
    /// their schema and linked program geometry before allocating or capturing.
    pub fn family_metadata(&self, family: &str) -> Result<&serde_json::Value> {
        self.families.get(family).with_context(|| format!("manifest has no family metadata for {family}"))
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
        ensure!(status != -12001,
            "loading {name} refused: insufficient resident SMs for its fused diagnostic CTA barrier");
        ensure!(status == 0, "loading {name} failed with CUDA status {status}");
        let spec = self.spec(name)?;
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
    launch_tensor_invariant: Option<HeadLaunchFn>,
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
        // SAFETY: this optional symbol has the same ABI as the other head launches.
        let launch_tensor_invariant = unsafe { self.lib.get(b"cuteafd_vocabulary_head_launch_tensor_invariant") }
            .ok().map(|f| *f);
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            create(workspace, VOCABULARY_HEAD_WORKSPACE as u64, i32::try_from(width)?, i32::try_from(max_rows)?, &mut handle)
        };
        ensure!(status == 0, "vocabulary head creation failed with {status}");
        Ok(VocabularyHead { library: self, handle, launch, launch_tensor_op, launch_tensor_invariant })
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

    /// Load-time BF16 -> E4M3 quantization of `count` scale units (the `Job`
    /// ABI of native/shared/cuda/fp8_load_quant.h), each reading device-visible
    /// (pinned or mapped host) source fragments and writing its final rows and
    /// scales; bit-for-bit the coordinator loaders' host rule.
    ///
    /// # Safety
    /// `jobs` holds `count` valid jobs in device-visible memory; every source,
    /// value and scale pointer they name stays live and unaliased until the
    /// stream drains. Values and scales are on the stream's device.
    pub unsafe fn fp8_load_quant(&self, jobs: *const c_void, count: usize, stream: *mut c_void) -> Result<()> {
        type Quant = unsafe extern "C" fn(*const c_void, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Quant>(b"cuteafd_fp8_load_quant") }?;
        let status = unsafe { f(jobs, i32::try_from(count)?, stream) };
        ensure!(status == 0, "FP8 load quantization failed with {status}");
        Ok(())
    }

    /// BF16 `src` (device-visible, e.g. pinned host) widened to F32 `dst` (exact).
    ///
    /// # Safety
    /// `src` holds `n` BF16 values and `dst` `n` F32 on the stream's device, live until it drains.
    pub unsafe fn load_widen_bf16_f32(&self, src: *const c_void, dst: *mut c_void, n: usize, stream: *mut c_void)
        -> Result<()> {
        type Widen = unsafe extern "C" fn(*const c_void, *mut c_void, i64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Widen>(b"cuteafd_load_widen_bf16_f32") }?;
        let status = unsafe { f(src, dst, i64::try_from(n)?, stream) };
        ensure!(status == 0, "BF16 load widening failed with {status}");
        Ok(())
    }

    /// `n` F32 values in `src` (device-visible) narrowed to BF16 (round to
    /// nearest even, NaN quieted) into `dst` on the current device.
    ///
    /// # Safety
    /// `src` holds `n` F32 and `dst` room for `n` BF16, live through the stream.
    pub unsafe fn load_narrow_f32_bf16(&self, src: *const c_void, dst: *mut c_void, n: usize, stream: *mut c_void)
        -> Result<()> {
        type Narrow = unsafe extern "C" fn(*const c_void, *mut c_void, i64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Narrow>(b"cuteafd_load_narrow_f32_bf16") }?;
        let status = unsafe { f(src, dst, i64::try_from(n)?, stream) };
        ensure!(status == 0, "F32 load narrowing failed with {status}");
        Ok(())
    }

    /// `n` BF16 pairs `a[i]`, `b[i]` (device-visible) promoted to F32 and
    /// multiplied once into `dst` (F32) on the current device.
    ///
    /// # Safety
    /// `a` and `b` hold `n` BF16 and `dst` room for `n` F32, live through the stream.
    pub unsafe fn load_bf16_product_f32(&self, a: *const c_void, b: *const c_void, dst: *mut c_void, n: usize,
        stream: *mut c_void) -> Result<()> {
        type Product = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, i64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Product>(b"cuteafd_load_bf16_product_f32") }?;
        let status = unsafe { f(a, b, dst, i64::try_from(n)?, stream) };
        ensure!(status == 0, "BF16 load product failed with {status}");
        Ok(())
    }

    /// Staged EXL3 trellis rows (`rows` K16 rows of `tiles` 16x16 tiles of
    /// `tile_bytes`, checkpoint order) reordered into a native shard `dst` of
    /// `k16` K16 rows at local row `k0`: `side` 0 output [N128][K16][8],
    /// 1 input [K128][N128][8][8] (`cuteafd_load_quant::trellis_tile`).
    ///
    /// # Safety
    /// `src` (device-visible) holds the staged rows and `dst` (current device)
    /// the whole shard, both live until the stream drains.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn load_trellis_tiles(&self, src: *const c_void, dst: *mut c_void, rows: usize, k0: usize,
        tiles: usize, tile_bytes: usize, side: i32, k16: usize, stream: *mut c_void) -> Result<()> {
        type Tiles = unsafe extern "C" fn(*const c_void, *mut c_void, i64, i64, i64, i64, i32, i64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Tiles>(b"cuteafd_load_trellis_tiles") }?;
        let status = unsafe { f(src, dst, i64::try_from(rows)?, i64::try_from(k0)?, i64::try_from(tiles)?,
            i64::try_from(tile_bytes)?, side, i64::try_from(k16)?, stream) };
        ensure!(status == 0, "EXL3 trellis tile reorder failed with {status}");
        Ok(())
    }

    /// MLA `kv_b` [heads * (nope + v), lat] (BF16, device-visible) absorbed per head
    /// into `uk` [heads][lat][nope] and `uv` [heads][v][lat] for heads `h0..h0 + heads`.
    ///
    /// # Safety
    /// `kv_b` holds the whole tensor; `uk`/`uv` hold `heads` heads on the stream's
    /// device; all live until the stream drains.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn load_kvb_absorb(&self, kv_b: *const c_void, uk: *mut c_void, uv: *mut c_void, h0: usize,
        heads: usize, nope: usize, v: usize, lat: usize, stream: *mut c_void) -> Result<()> {
        type Absorb = unsafe extern "C" fn(*const c_void, *mut c_void, *mut c_void, i64, i64, i64, i64, i64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Absorb>(b"cuteafd_load_kvb_absorb") }?;
        let status = unsafe { f(kv_b, uk, uv, i64::try_from(h0)?, i64::try_from(heads)?, i64::try_from(nope)?,
            i64::try_from(v)?, i64::try_from(lat)?, stream) };
        ensure!(status == 0, "kv_b load absorption failed with {status}");
        Ok(())
    }

    /// FP8 MLA `kv_b` rows of `heads` heads (E4M3 `[heads * (nope + v), lat]`, the
    /// first at kv_b row `row0`) with the whole tensor's FP32 128x128 `grid` (both
    /// device-visible) split per head into the GLM FP8 programs' `uk`
    /// `[heads][lat][nope]` + `uk_scale` `[heads][lat][nope / 64]` and `uv`
    /// `[heads][v][lat]` + `uv_scale` `[heads][v][lat / 64]`: bytes unchanged,
    /// scales the grid entry of each 64-wide tile.
    ///
    /// # Safety
    /// `kv_b` holds the staged heads' rows and `grid` the whole grid; the four outputs hold `heads`
    /// heads on the stream's device; all live until the stream drains.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn load_kvb_split_fp8(&self, kv_b: *const c_void, grid: *const c_void, uk: *mut c_void,
        uk_scale: *mut c_void, uv: *mut c_void, uv_scale: *mut c_void, row0: usize, heads: usize, nope: usize,
        v: usize, lat: usize, stream: *mut c_void) -> Result<()> {
        type Split = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, *mut c_void, *mut c_void,
            *mut c_void, i64, i64, i64, i64, i64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Split>(b"cuteafd_load_kvb_split_fp8") }?;
        let status = unsafe { f(kv_b, grid, uk, uk_scale, uv, uv_scale, i64::try_from(row0)?, i64::try_from(heads)?,
            i64::try_from(nope)?, i64::try_from(v)?, i64::try_from(lat)?, stream) };
        ensure!(status == 0, "FP8 kv_b split failed with {status}");
        Ok(())
    }

    /// The load-quantization `Job` ABI: (job bytes, segment bytes, max columns).
    pub fn fp8_load_quant_abi(&self) -> Result<(u64, u64, u64)> {
        type Abi = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64) -> i32;
        let f = *unsafe { self.lib.get::<Abi>(b"cuteafd_fp8_load_quant_abi") }?;
        let (mut job, mut segment, mut cols) = (0, 0, 0);
        // SAFETY: three writable u64 outputs.
        let status = unsafe { f(&mut job, &mut segment, &mut cols) };
        ensure!(status == 0, "FP8 load quantization ABI query failed with {status}");
        Ok((job, segment, cols))
    }

    /// FP32 128x128 block `grid` [ceil(rows/128), kb] widened to per-row scales
    /// stored K-block major, `kmajor` [kb, rows] (exact gather).
    ///
    /// # Safety
    /// Both pointers are live device memory of those shapes on the stream's device.
    pub unsafe fn fp8_grid_kmajor(&self, grid: *const c_void, kmajor: *mut c_void, rows: usize, kb: usize,
        stream: *mut c_void) -> Result<()> {
        type Widen = unsafe extern "C" fn(*const c_void, *mut c_void, i64, i64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Widen>(b"cuteafd_fp8_grid_kmajor") }?;
        let status = unsafe { f(grid, kmajor, i64::try_from(rows)?, i64::try_from(kb)?, stream) };
        ensure!(status == 0, "FP8 K-major grid widening failed with {status}");
        Ok(())
    }

    /// Per-row FP32 scales gathered from `grid` [*, grid_cols] (columns
    /// `col0..col0 + kb` of row `index[r]`, or zero for `u32::MAX`) into
    /// `row_major` [rows, kb] and/or `k_major` [kb, rows] (null to skip).
    ///
    /// # Safety
    /// Every non-null pointer is a live device buffer of its shape (`index`:
    /// `rows` u32) on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn load_scale_gather(&self, grid: *const c_void, index: *const c_void, row_major: *mut c_void,
        k_major: *mut c_void, rows: usize, kb: usize, grid_cols: usize, col0: usize, stream: *mut c_void) -> Result<()> {
        type Gather = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, *mut c_void, i64, i64, i64, i64,
            *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<Gather>(b"cuteafd_load_scale_gather") }?;
        let status = unsafe { f(grid, index, row_major, k_major, i64::try_from(rows)?, i64::try_from(kb)?,
            i64::try_from(grid_cols)?, i64::try_from(col0)?, stream) };
        ensure!(status == 0, "load-time scale gather failed with {status}");
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
        // SAFETY: this optional symbol has the same ABI as the other head launches.
        let launch_tensor_invariant = unsafe { self.lib.get(b"cuteafd_vocabulary_head_launch_tensor_invariant") }
            .ok().map(|f| *f);
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            create(workspace, VOCABULARY_HEAD_WORKSPACE as u64, i32::try_from(width)?, i32::try_from(max_rows)?,
                i32::try_from(vocab)?, &mut handle)
        };
        ensure!(status == 0, "vocabulary head creation failed with {status}");
        Ok(VocabularyHead { library: self, handle, launch, launch_tensor_op, launch_tensor_invariant })
    }
}

impl VocabularyHead<'_> {
    /// BF16 target projection with one fixed cuBLASLt tile and reduction per head.
    /// Unlike the drafter's heuristic route, row count never selects another algorithm.
    ///
    /// # Safety
    /// As [`Self::launch`]. Serialize handle/workspace use; both must outlive all
    /// captured graphs and pending launches.
    pub unsafe fn launch_tensor_invariant(&self, input: *const u16, weight: *const u16,
        logits: *mut f32, rows: u32, stream: *mut c_void) -> Result<()> {
        let launch = self.launch_tensor_invariant.context(
            "this native library has no fixed tensor vocabulary head (rebuild it with this checkout)")?;
        // SAFETY: the caller guarantees buffer, device and stream lifetimes.
        let status = unsafe { launch(self.handle, input, weight, logits, i32::try_from(rows)?, stream) };
        ensure!(status == 0, "fixed tensor vocabulary head launch of {rows} rows failed with {status}");
        Ok(())
    }

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
