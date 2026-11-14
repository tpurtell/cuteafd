//! Family-independent view of a Hugging Face checkpoint: configuration JSON,
//! the safetensors index and every shard header. Nothing here reads tensor data.
use crate::{catalog::MAX_JSON_BYTES, read_safetensors_metadata, SafetensorsTensorMetadata};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub mod identity;
pub mod mc;
pub mod view;
pub use mc::CoreView;
pub use view::CheckpointView;

/// Storage discovery is shared by inventories, catalogs and bootstrap identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat { HfIndexed, HfSingle, Exlr8 }

/// The default Hugging Face safetensors index.
pub const HF_INDEX: &str = "model.safetensors.index.json";

/// The safetensors index a checkpoint declares: `config.json`'s
/// `transformers_weights` (Mistral's `consolidated.safetensors.index.json`,
/// honored by transformers' loader), else the standard name. The declared
/// name must be a plain `*.index.json` file name inside the snapshot.
pub fn index_name(snapshot: &Path) -> Result<String> {
    let path = snapshot.join("config.json");
    if !path.is_file() { return Ok(HF_INDEX.into()); }
    let config = read_json(&path)?;
    index_name_from_config(&config)
}

fn index_name_from_config(config: &Value) -> Result<String> {
    match config.get("transformers_weights") {
        None | Some(Value::Null) => Ok(HF_INDEX.into()),
        Some(Value::String(name)) => {
            ensure!(name.ends_with(".index.json") && !name.contains(['/', '\\']) && !name.starts_with('.'),
                "config.json transformers_weights {name:?} is not a safetensors index file name");
            Ok(name.clone())
        }
        Some(other) => anyhow::bail!("config.json transformers_weights must be a file name, found {other}"),
    }
}

/// The path of the checkpoint's declared safetensors index.
pub fn index_path(snapshot: &Path) -> Result<PathBuf> {
    Ok(snapshot.join(index_name(snapshot)?))
}

impl CheckpointFormat {
    pub fn detect_view(view: &CheckpointView) -> Result<Self> {
        if !view.mode().compact() && super::exlr8::recognizes(view.snapshot())? {
            return Ok(Self::Exlr8);
        }
        if view.weight_map().is_some() { return Ok(Self::HfIndexed); }
        ensure!(view.snapshot().join("model.safetensors").is_file(),
            "checkpoint has no recognized tensor inventory");
        Ok(Self::HfSingle)
    }

    pub fn detect(snapshot: &Path) -> Result<Self> {
        if super::exlr8::recognizes(snapshot)? { return Ok(Self::Exlr8); }
        if index_path(snapshot)?.is_file() { return Ok(Self::HfIndexed); }
        ensure!(snapshot.join("model.safetensors").is_file(),
            "checkpoint has no recognized tensor inventory");
        Ok(Self::HfSingle)
    }
}

/// Arithmetic declared by the checkpoint, independent of optional load optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleNumerics { CheckpointPrecision, W4A4, W8A8, W8CheckpointActivations }

fn projection_stem(module: &str) -> &str {
    module.strip_suffix(".weight_packed").or_else(|| module.strip_suffix(".weight")).unwrap_or(module)
}

fn module_match(pattern: &str, module: &str) -> bool {
    if !pattern.contains('*') {
        return module == pattern || module.strip_prefix(pattern).is_some_and(|s| s.starts_with('.'));
    }
    let mut remainder = module;
    let parts: Vec<_> = pattern.split('*').collect();
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() { continue; }
        let Some(at) = remainder.find(part) else { return false; };
        if i == 0 && at != 0 { return false; }
        remainder = &remainder[at + part.len()..];
    }
    pattern.ends_with('*') || remainder.is_empty()
}

/// One tensor as the checkpoint stores it.
#[derive(Debug, Clone)]
pub struct CheckpointTensor {
    pub shard: String,
    pub meta: SafetensorsTensorMetadata,
}

#[derive(Debug)]
pub struct Checkpoint {
    /// Immutable selection propagated to catalogs, push records and identities.
    pub view: Option<CheckpointView>,
    pub snapshot: PathBuf,
    /// `config.json`, verbatim.
    pub config: Value,
    /// Standard external quantization metadata, without the separately validated tensor_storage map.
    pub quantize_config: Option<Value>,
    /// Authoritative inventory, including tensors whose shards are not local.
    pub weight_map: BTreeMap<String, String>,
    /// Headers selected for this role's payload, sorted by name.
    pub tensors: Vec<CheckpointTensor>,
    /// Shards named by the index that are not readable (absent or incomplete).
    pub missing_shards: Vec<String>,
    pub shard_bytes: u64,
}

pub(crate) fn validate_index_headers(shard: &str, names: &BTreeSet<&str>,
    headers: &[SafetensorsTensorMetadata]) -> Result<()> {
    let present: BTreeSet<_> = headers.iter().map(|tensor| tensor.name.as_str()).collect();
    for name in names {
        ensure!(present.contains(name), "indexed tensor {name} missing from shard {shard} header");
    }
    Ok(())
}

pub fn read_json(path: &Path) -> Result<Value> {
    Ok(crate::read_json_file(path)?)
}

/// Storage of one projection in the selected index, before optional conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionStorage {
    Native { dtype: cuteafd_core::DType, shape: Vec<usize> },
    /// Compressed-tensors packed E2M1; shape is logical, not the packed U8 extent.
    CompressedNvfp4 { shape: Vec<usize> },
    Exl3,
}

#[derive(Debug, Clone)]
pub struct SelectedProjection {
    pub storage: ProjectionStorage,
    /// Only active indexed tensors; includes native scales or EXL3 buffers.
    pub tensors: Vec<CheckpointTensor>,
    pub source_bytes: u64,
}

impl Checkpoint {
    pub fn core_view(&self) -> CoreView {
        self.view.as_ref().map_or(CoreView::Original, CheckpointView::mode)
    }

    pub fn converts_to_fp8(&self, name: &str) -> bool {
        self.view.as_ref().is_some_and(|view| view.converts_to_fp8(name))
    }

    /// Push conversion is selected per tensor, not by a family's global view mode.
    pub fn projection_view(&self, stem: &str) -> CoreView {
        if self.converts_to_fp8(&format!("{stem}.weight")) { CoreView::Fp8Compact }
        else { CoreView::Original }
    }

    /// Families declare executable roles, never a second conversion selection.
    pub fn validate_fp8_roles(&self, family: &str,
        role: impl Fn(&str) -> crate::formats::native_attention::Fp8Role) -> Result<()> {
        for name in self.view.iter().flat_map(|view| view.fp8_tensors()) {
            let declaration = role(name);
            let (role, supported) = (declaration.name, declaration.supported);
            ensure!(supported, "{name}: role {role} cannot execute block FP8 in family {family}");
            let header = self.header(name).with_context(|| format!("{name}: role {role} in family {family}: selected BF16 header missing"))?;
            ensure!(header.dtype == cuteafd_core::DType::Bf16 && header.shape.len() == 2
                && declaration.stored_shape([header.shape[0], header.shape[1]]).is_ok(),
                "{name}: role {role} in family {family}: fp8-compact requires a BF16 matrix with whole 128x128 blocks, found {:?} {:?}",
                header.dtype, header.shape);
        }
        Ok(())
    }

    /// Whether a payload or classification header is available, independent of index membership.
    pub fn has_header(&self, name: &str) -> bool {
        self.header(name).is_some()
    }

    /// Validated metadata for classification, including native operands not owned by this role.
    pub fn header(&self, name: &str) -> Option<&SafetensorsTensorMetadata> {
        [self.tensors.as_slice(), self.classification_headers()].into_iter().find_map(|headers|
            headers.binary_search_by(|tensor| tensor.meta.name.as_str().cmp(name))
                .ok().map(|at| &headers[at].meta))
    }

    fn classification_headers(&self) -> &[CheckpointTensor] {
        self.view.as_ref().map_or(&[], |view| view.classification_headers.as_slice())
    }

    /// Per-projection classification shared by runtime and push planning.
    /// Call on a role-opened checkpoint so all selected projection headers exist.
    pub fn projection(&self, stem: &str) -> Result<SelectedProjection> {
        let prefix = format!("{stem}.");
        let mut tensors = Vec::new();
        for headers in [self.tensors.as_slice(), self.classification_headers()] {
            let start = headers.partition_point(|tensor| tensor.meta.name.as_str() < prefix.as_str());
            tensors.extend(headers[start..].iter()
                .take_while(|tensor| tensor.meta.name.starts_with(&prefix))
                .filter(|tensor| !tensor.meta.name[prefix.len()..].contains('.')
                    && self.contains_tensor(&tensor.meta.name)).cloned());
        }
        let weight = tensors.iter().find(|tensor| tensor.meta.name == format!("{stem}.weight"));
        let packed = tensors.iter().any(|tensor| tensor.meta.name == format!("{stem}.weight_packed"));
        let trellis = tensors.iter().any(|tensor| tensor.meta.name == format!("{stem}.trellis"));
        ensure!(usize::from(weight.is_some()) + usize::from(packed) + usize::from(trellis) <= 1,
            "{stem}: checkpoint contains conflicting native, packed or EXL3 storage");
        let storage = if packed {
            let members = tensors.iter().map(|tensor|
                (tensor.meta.name[prefix.len()..].to_owned(), tensor)).collect();
            let operand = super::format::detect(&members)
                .map_err(|error| anyhow::anyhow!("{}: {}", error.tensor, error.reason))?;
            ensure!(operand.is_compressed_nvfp4(), "{stem}: invalid compressed NVFP4 storage");
            ProjectionStorage::CompressedNvfp4 { shape: operand.logical }
        } else if trellis { ProjectionStorage::Exl3 } else {
            let weight = weight.with_context(|| format!("{stem}: selected projection headers missing"))?;
            ProjectionStorage::Native { dtype: weight.meta.dtype.clone(), shape: weight.meta.shape.clone() }
        };
        let source_bytes = tensors.iter().try_fold(0u64, |sum, tensor| sum.checked_add(tensor.meta.byte_length)
            .context("projection source bytes overflow"))?;
        Ok(SelectedProjection { storage, tensors, source_bytes })
    }

    /// Reads the configuration, the index (or a single `model.safetensors`) and
    /// every shard header. Missing shards are recorded rather than fatal so a
    /// partially downloaded checkpoint can still be planned.
    pub fn open(snapshot: &Path) -> Result<Self> {
        Self::open_filtered(snapshot, None, |_| true)
    }

    pub fn open_view(view: &CheckpointView) -> Result<Self> {
        Self::open_view_filtered(view, None, |_| true, None)
    }

    pub fn coordinator_view(view: &CheckpointView, local_experts: bool, speculator: bool) -> Result<Self> {
        let inventory = Self::inventory_view(view)?;
        let role = super::files::ReadRole::Coordinator { local_experts, speculator };
        let names = super::files::required_tensors(&inventory, role)?;
        Self::open_view_filtered(view, Some(&role.label()), |name| names.contains(name), None)
    }

    /// Coordinator ownership uses the planner's family classification, not
    /// whichever headers happen to be readable on this host.
    pub fn coordinator(snapshot: &Path, local_experts: bool, speculator: bool) -> Result<Self> {
        super::files::open_role(snapshot, super::files::ReadRole::Coordinator { local_experts, speculator })
    }

    /// Reads the index/configuration plus declared native-attention and scale-free
    /// declared-A4 headers needed to classify storage before opening a role.
    pub fn inventory(snapshot: &Path) -> Result<Self> {
        Self::inventory_view(&CheckpointView::select(snapshot, CoreView::Original)?)
    }

    pub fn inventory_view(view: &CheckpointView) -> Result<Self> {
        let snapshot = view.snapshot();
        let mut checkpoint = Self::open_view_filtered(view, None, |_| false, None)?;
        let native = view.native_core_names().collect::<BTreeSet<_>>();
        let mut required = BTreeMap::<String, BTreeSet<String>>::new();
        for name in &native {
            let shard = checkpoint.weight_map.get(*name).context("declared native attention is not indexed")?;
            required.entry(shard.clone()).or_default().insert((*name).to_owned());
        }
        for (name, shard) in &checkpoint.weight_map {
            let Some(module) = name.strip_suffix(".weight") else { continue };
            if checkpoint.declared_numerics(module)? == ModuleNumerics::W4A4
                && ["weight_scale", "weight_scale_2", "input_scale"].iter()
                    .all(|suffix| !checkpoint.contains_tensor(&format!("{module}.{suffix}"))) {
                required.entry(shard.clone()).or_default().insert(name.clone());
            }
        }
        let mut classification = Vec::new();
        for (shard, names) in required {
            let headers = read_safetensors_metadata(&snapshot.join(&shard))?;
            validate_index_headers(&shard, &names.iter().map(String::as_str).collect(), &headers)?;
            for meta in headers {
                if names.contains(&meta.name) {
                    let tensor = CheckpointTensor { shard: shard.clone(), meta };
                    if native.contains(tensor.meta.name.as_str()) { classification.push(tensor); }
                    else { checkpoint.tensors.push(tensor); }
                }
            }
        }
        checkpoint.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        classification.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        checkpoint.view.as_mut().unwrap().classification_headers = std::sync::Arc::new(classification);
        Ok(checkpoint)
    }

    /// A runtime role declares its payload reads before any shard is opened.
    /// Declared native-attention headers remain available for classification only,
    /// outside `tensors` and role payload bytes. Every selected header must be readable.
    pub fn open_view_for_role(view: &CheckpointView, role: &str, needed: impl Fn(&str) -> bool) -> Result<Self> {
        Self::open_view_filtered(view, Some(role), needed, None)
    }

    pub fn open_for_role(snapshot: &Path, role: &str, needed: impl Fn(&str) -> bool) -> Result<Self> {
        Self::open_filtered(snapshot, Some(role), needed)
    }

    pub(crate) fn open_with_header_validation(snapshot: &Path) -> Result<(Self, Result<(), String>)> {
        Self::open_view_with_header_validation(&CheckpointView::select(snapshot, CoreView::Original)?)
    }

    pub(crate) fn open_view_with_header_validation(view: &CheckpointView) -> Result<(Self, Result<(), String>)> {
        let mut validation = Ok(());
        let checkpoint = Self::open_view_filtered(view, None, |_| true, Some(&mut validation))?;
        Ok((checkpoint, validation))
    }

    fn open_filtered(snapshot: &Path, role: Option<&str>, needed: impl Fn(&str) -> bool) -> Result<Self> {
        Self::open_filtered_validated(snapshot, role, needed, None)
    }

    fn open_filtered_validated(snapshot: &Path, role: Option<&str>, needed: impl Fn(&str) -> bool,
        validation: Option<&mut Result<(), String>>) -> Result<Self> {
        let view = CheckpointView::select(snapshot, CoreView::Original)?;
        Self::open_view_filtered(&view, role, needed, validation)
    }

    fn open_view_filtered(view: &CheckpointView, role: Option<&str>, needed: impl Fn(&str) -> bool,
        mut validation: Option<&mut Result<(), String>>) -> Result<Self> {
        let snapshot = view.snapshot();
        let config = view.config()?;
        if CheckpointFormat::detect_view(view)? == CheckpointFormat::Exlr8 {
            let mut checkpoint = super::exlr8::inventory(snapshot, config, needed)?;
            checkpoint.view = Some(view.clone());
            debug_assert!(checkpoint.tensors.windows(2).all(|pair| pair[0].meta.name <= pair[1].meta.name));
            return Ok(checkpoint);
        }
        let quantize_config = view.quantize_config();
        let index_path = view.index_path();
        let weight_map: BTreeMap<String, String> = if let Some(index) = view.weight_map() {
            index.clone()
        } else {
            let single = snapshot.join("model.safetensors");
            ensure!(
                single.is_file(),
                "{} has neither {} nor model.safetensors",
                snapshot.display(), index_path.file_name().unwrap_or_default().to_string_lossy()
            );
            read_safetensors_metadata(&single)?
                .into_iter()
                .map(|meta| (meta.name, "model.safetensors".to_owned()))
                .collect()
        };
        // Native headers classify eager family cores even when this role never
        // reads their payloads. Share the role's once-per-shard header pass.
        let native = if role.is_some() { view.native_core_names().collect::<BTreeSet<_>>() }
            else { BTreeSet::new() };
        // Group once instead of scanning the complete index for every shard.
        let mut shards = BTreeMap::<&String, BTreeSet<&str>>::new();
        for (name, shard) in &weight_map {
            ensure!(!shard.is_empty() && Path::new(shard).components().all(|component|
                matches!(component, std::path::Component::Normal(_))),
                "invalid checkpoint shard path {shard:?}");
            if needed(name) || native.contains(name.as_str()) { shards.entry(shard).or_default().insert(name); }
        }
        let mut tensors = Vec::with_capacity(weight_map.len());
        let mut classification_headers = Vec::new();
        let mut missing_shards = Vec::new();
        let mut shard_bytes = 0u64;
        for (shard, names) in shards {
            let path = snapshot.join(shard);
            let tensor = names.first().context("selected shard has no tensor")?;
            let requirement = || format!("role {} needs tensor {tensor} in shard {shard} at snapshot {}",
                role.unwrap_or("planner"), snapshot.display());
            let metadata = match path.metadata() {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => {
                    ensure!(role.is_none(), "{}: absent or unreadable", requirement());
                    missing_shards.push(shard.clone());
                    continue;
                }
            };
            let headers = match read_safetensors_metadata(&path) {
                Ok(headers) => headers,
                Err(error) => {
                    ensure!(role.is_none(), "{}: {error:#}", requirement());
                    missing_shards.push(shard.clone());
                    continue;
                }
            };
            if let Some(result) = validation.as_deref_mut().filter(|result| result.is_ok()) {
                *result = validate_index_headers(shard, &names, &headers).map_err(|error| error.to_string());
            }
            if validation.is_none() && role.is_none() {
                validate_index_headers(shard, &names, &headers)?;
            }
            if let Some(role) = role {
                let header_names: BTreeSet<_> = headers.iter().map(|meta| meta.name.as_str()).collect();
                for name in &names {
                    ensure!(header_names.contains(name),
                        "role {role} needs tensor {name} in shard {shard} at snapshot {}: header is missing",
                        snapshot.display());
                }
            }
            if names.iter().any(|name| needed(name)) { shard_bytes += metadata.len(); }
            for meta in headers {
                if names.contains(meta.name.as_str()) {
                    let selected = needed(&meta.name);
                    let tensor = CheckpointTensor { shard: shard.clone(), meta };
                    if selected { tensors.push(tensor); }
                    else { classification_headers.push(tensor); }
                }
            }
        }
        tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        classification_headers.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        let mut view = view.clone();
        view.classification_headers = std::sync::Arc::new(classification_headers);
        debug_assert!(tensors.windows(2).all(|pair| pair[0].meta.name <= pair[1].meta.name));
        Ok(Self {
            view: Some(view),
            snapshot: snapshot.to_path_buf(),
            config,
            quantize_config,
            weight_map,
            tensors,
            missing_shards,
            shard_bytes,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        // In-memory fixtures predating the HF index use their complete headers.
        let from_headers = self.weight_map.is_empty();
        self.weight_map.keys().map(String::as_str)
            .chain(self.tensors.iter().filter(move |_| from_headers).map(|t| t.meta.name.as_str()))
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        if !self.weight_map.is_empty() { return self.weight_map.contains_key(name); }
        self.has_header(name)
    }

    /// Select each module's declared arithmetic. Ignore rules take precedence;
    /// class-wide Linear rules apply only to linear weights, not arbitrary tensors.
    pub fn module_numerics(&self, module: &str) -> Result<ModuleNumerics> {
        let stem = projection_stem(module);
        ensure!(!(self.contains_tensor(&format!("{stem}.weight")) && self.contains_tensor(&format!("{stem}.weight_packed"))),
            "{stem}: checkpoint contains both native and packed storage");
        let dtype = ["weight", "weight_packed"].iter()
            .find_map(|suffix| self.header(&format!("{stem}.{suffix}"))).map(|header| header.dtype.clone());
        if ["weight_packed", "weight_global_scale", "input_global_scale"].iter()
            .any(|suffix| self.contains_tensor(&format!("{stem}.{suffix}")))
            && self.declared_numerics(stem)? == ModuleNumerics::W4A4 {
            self.validate_native_members(stem,
                &["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"], true)?;
        }
        match dtype {
            Some(dtype) => self.module_numerics_with_header(stem, dtype),
            None => self.declared_numerics(stem),
        }
    }

    /// Complete unquantized BF16 headers override an overbroad ModelOpt declaration;
    /// partial quantization is never treated as an unquantized checkpoint module.
    pub fn module_numerics_with_header(&self, module: &str, dtype: cuteafd_core::DType) -> Result<ModuleNumerics> {
        let module = projection_stem(module);
        let declared = self.declared_numerics(module)?;
        let compressed = self.contains_tensor(&format!("{module}.weight_packed"));
        let mxfp8 = self.quantization().is_some_and(|q| q.get("quant_method").and_then(Value::as_str) == Some("mxfp8"));
        if compressed && declared == ModuleNumerics::W4A4 {
            ensure!(!self.contains_tensor(&format!("{module}.weight")), "{module}: checkpoint contains both native and packed storage");
            ensure!(dtype == cuteafd_core::DType::U8, "{module}.weight_packed: compressed NVFP4 requires U8 storage");
            self.validate_native_members(module,
                &["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"], true)?;
            return Ok(declared);
        }
        if mxfp8 && declared == ModuleNumerics::W8A8 {
            ensure!(dtype == cuteafd_core::DType::F8E4M3, "{module}.weight: declared MXFP8 requires E4M3 storage");
            self.validate_native_members(module, &["weight", "weight_scale_inv"], false)?;
            return Ok(declared);
        }
        if declared != ModuleNumerics::W4A4 { return Ok(declared); }
        let scales = ["weight_scale", "weight_scale_2", "input_scale"]
            .map(|suffix| self.contains_tensor(&format!("{module}.{suffix}")));
        if dtype == cuteafd_core::DType::Bf16 && scales.iter().all(|present| !present) {
            static WARNED: std::sync::OnceLock<std::sync::Mutex<BTreeSet<(PathBuf, String)>>> = std::sync::OnceLock::new();
            let mut warned = WARNED.get_or_init(Default::default).lock()
                .map_err(|_| anyhow::anyhow!("checkpoint numerics warning registry poisoned"))?;
            if warned.insert((self.snapshot.clone(), module.to_owned())) {
                tracing::warn!(module, "quantization config/header disagreement: declared W4A4 but BF16 header has no quantization scales; preserving checkpoint precision");
            }
            return Ok(ModuleNumerics::CheckpointPrecision);
        }
        for (suffix, present) in ["weight_scale", "weight_scale_2", "input_scale"].into_iter().zip(scales) {
            ensure!(present, "{module}.{suffix}: declared W4A4 has partial or invalid NVFP4 storage ({dtype:?}); missing scale tensor");
        }
        ensure!(matches!(dtype, cuteafd_core::DType::U8 | cuteafd_core::DType::I8),
            "{module}.weight: declared W4A4 has partial or invalid NVFP4 storage ({dtype:?})");
        Ok(declared)
    }

    fn validate_native_members(&self, module: &str, suffixes: &[&str], compressed: bool) -> Result<()> {
        let mut members = BTreeMap::new();
        for suffix in suffixes {
            let name = format!("{module}.{suffix}");
            ensure!(self.contains_tensor(&name), "{name}: declared native arithmetic is missing tensor");
            if let Some(meta) = self.header(&name) {
                members.insert((*suffix).to_owned(), CheckpointTensor { shard: String::new(), meta: meta.clone() });
            }
        }
        // Inventory-only views may not have read the payload shard yet.
        if members.len() != suffixes.len() { return Ok(()); }
        let refs = members.iter().map(|(suffix, tensor)| (suffix.clone(), tensor)).collect();
        let operand = super::format::detect(&refs)
            .map_err(|error| anyhow::anyhow!("{}: {}", error.tensor, error.reason))?;
        ensure!(if compressed { operand.is_compressed_nvfp4() } else {
            operand.is_fp8_row_groups(32, &[super::format::ScaleEncoding::Ue8m0])
                && operand.logical.len() == 2 && operand.logical.iter().all(|dim| *dim > 0)
                && operand.logical[1] % 32 == 0
        }, "{module}: native storage does not match declared arithmetic");
        Ok(())
    }

    fn declared_numerics(&self, module: &str) -> Result<ModuleNumerics> {
        let module = projection_stem(module);
        // Stacked MoE projections are parameters of the MoE, not child Linear
        // modules. In particular, a router-module wildcard must not accidentally
        // match the lexical gate_proj parameter stem. Keep the original stem for
        // storage/scale validation and projection-target declarations below.
        let owner = module.rsplit_once('.').and_then(|(parent, parameter)| {
            (parent.ends_with(".moe") && matches!(parameter, "gate_proj" | "up_proj" | "down_proj"))
                .then_some(parent)
        }).unwrap_or(module);
        self.declared_numerics_for_owner(module, owner)
    }

    fn declared_numerics_for_owner(&self, module: &str, owner: &str) -> Result<ModuleNumerics> {
        let Some(root) = self.quantization() else { return Ok(ModuleNumerics::CheckpointPrecision) };
        let q = root.get("quantization").unwrap_or(root);
        if root.get("quant_method").and_then(Value::as_str) == Some("mxfp8") {
            if q.get("ignored_layers").and_then(Value::as_array).is_some_and(|rules|
                rules.iter().filter_map(Value::as_str).any(|rule| module_match(rule, owner))) {
                return Ok(ModuleNumerics::CheckpointPrecision);
            }
            if module.contains(".experts.") && q.get("moe_quant_algo").is_some() {
                ensure!(q.get("moe_quant_algo").and_then(Value::as_str) == Some("NVFP4")
                    && q.get("moe_quant_format").and_then(Value::as_str) == Some("nvfp4-pack-quantized")
                    && q.get("group_size").and_then(Value::as_u64) == Some(16),
                    "{module}: unsupported packed expert arithmetic");
                return Ok(ModuleNumerics::W4A4);
            }
            // Norms, embeddings and other non-linear parameters are not MXFP8.
            if !(module.ends_with("_proj") || module.ends_with(".w1") || module.ends_with(".w2")
                || module.ends_with(".w3") || self.contains_tensor(&format!("{module}.weight_scale_inv"))) {
                return Ok(ModuleNumerics::CheckpointPrecision);
            }
            ensure!(q.get("activation_scheme").and_then(Value::as_str) == Some("dynamic")
                && q.get("weight_block_size") == Some(&serde_json::json!([1, 32])),
                "{module}: native MXFP8 requires dynamic activations and [1,32] weight blocks");
            return Ok(ModuleNumerics::W8A8);
        }
        if root.get("quant_method").and_then(Value::as_str) != Some("modelopt")
            && root.pointer("/producer/name").and_then(Value::as_str) != Some("modelopt") {
            return Ok(ModuleNumerics::CheckpointPrecision);
        }
        if ["ignore", "exclude_modules"].iter().any(|key| q.get(key).and_then(Value::as_array).is_some_and(|rules|
            rules.iter().filter_map(Value::as_str).any(|rule| module_match(rule, owner)))) {
            return Ok(ModuleNumerics::CheckpointPrecision);
        }
        let mut selected = None;
        if let Some(groups) = q.get("config_groups").and_then(Value::as_object) {
            for group in groups.values() {
                let applies = group.get("targets").and_then(Value::as_array).is_some_and(|targets|
                    targets.iter().filter_map(Value::as_str).any(|target|
                        module_match(target, module) || (target == "Linear" &&
                            (module.contains(".experts.") || module.ends_with("_proj") ||
                                module.ends_with(".w1") || module.ends_with(".w2") || module.ends_with(".w3")))));
                if !applies { continue; }
                let w = group.pointer("/weights/num_bits").and_then(Value::as_u64);
                let a = group.pointer("/input_activations/num_bits").and_then(Value::as_u64);
                let numerics = match (w, a) {
                    (Some(4), Some(4)) => {
                        ensure!(group.pointer("/weights/group_size").and_then(Value::as_u64) == Some(16)
                            && group.pointer("/input_activations/group_size").and_then(Value::as_u64) == Some(16),
                            "{module}: W4A4 requires declared K16 scales");
                        ModuleNumerics::W4A4
                    }
                    (Some(8), Some(8)) => ModuleNumerics::W8A8,
                    (Some(8), None) => ModuleNumerics::W8CheckpointActivations,
                    _ => anyhow::bail!("{module}: unsupported declared weight/activation bits {w:?}/{a:?}"),
                };
                ensure!(selected.is_none_or(|old| old == numerics), "{module}: conflicting quantization groups");
                selected = Some(numerics);
            }
        }
        if let Some(numerics) = selected { return Ok(numerics); }
        if let Some(layers) = q.get("quantized_layers").and_then(Value::as_object) {
            let declaration = layers.iter().filter(|(name, _)| module_match(name, module))
                .max_by_key(|(name, _)| name.len()).map(|(_, value)| value);
            if let Some(declaration) = declaration {
                return match declaration.get("quant_algo").and_then(Value::as_str) {
                    Some("NVFP4") => Ok(ModuleNumerics::W4A4),
                    Some("FP8") => Ok(ModuleNumerics::W8A8),
                    other => anyhow::bail!("{module}: unsupported declared quantization {other:?}"),
                };
            }
        }
        // Older standard ModelOpt exports declare a checkpoint-wide algorithm.
        if (module.contains(".experts.") || module.ends_with("_proj"))
            && q.get("quant_algo").and_then(Value::as_str) == Some("NVFP4") {
            return Ok(ModuleNumerics::W4A4);
        }
        Ok(ModuleNumerics::CheckpointPrecision)
    }

    pub fn declared_module_numerics(&self) -> Result<BTreeMap<String, ModuleNumerics>> {
        self.tensor_names().filter(|name| name.ends_with(".weight") || name.ends_with(".weight_packed"))
            .map(projection_stem)
            .map(|module| Ok((module.to_owned(), self.module_numerics(module)?))).collect()
    }

    /// Native NVFP4 experts quantize original BF16 activations on the worker.
    /// Use the complete index, never a role's local expert/MTP header subset.
    pub fn spark_input_is_bf16(&self) -> bool {
        let layers = opt_usize_field(self.text_config(), "num_hidden_layers");
        self.tensor_names().any(|name| {
            let stem = if let Some(stem) = name.strip_suffix(".input_global_scale") {
                if !["weight_packed", "weight_scale", "weight_global_scale"].iter()
                    .all(|suffix| self.contains_tensor(&format!("{stem}.{suffix}"))) { return false; }
                if self.module_numerics(stem).ok() != Some(ModuleNumerics::W4A4) { return false; }
                stem
            } else if let Some(stem) = name.strip_suffix(".input_scale") {
                if !self.contains_tensor(&format!("{stem}.weight_scale_2")) { return false; }
                stem
            } else { return false; };
            if !stem.contains(".experts.") {
                return false;
            }
            // Legacy fixtures/undeclared checkpoints retain header detection.
            // Declared modules use the same selector as runtime and plan.
            if self.quantization().is_some_and(|root| {
                let q = root.get("quantization").unwrap_or(root);
                q.get("config_groups").is_some() || q.get("quantized_layers").is_some()
                    || q.get("quant_algo").is_some()
            }) && self.module_numerics(stem).ok() != Some(ModuleNumerics::W4A4) {
                return false;
            }
            let parts: Vec<_> = stem.split('.').collect();
            if parts.iter().any(|part| matches!(*part, "mtp" | "nextn" | "dspark")) { return false; }
            let layer = parts.iter().position(|part| *part == "layers")
                .and_then(|at| parts.get(at + 1)).and_then(|value| value.parse::<usize>().ok());
            layer.is_some_and(|layer| layers.is_none_or(|count| layer < count))
        })
    }

    /// Explicit requirements also reject tensors absent from the index itself.
    pub fn open_required(snapshot: &Path, role: &str, names: &BTreeSet<String>) -> Result<Self> {
        let inventory = Self::inventory(snapshot)?;
        for name in names {
            ensure!(inventory.contains_tensor(name),
                "role {role} needs tensor {name} at snapshot {}: absent from weight_map", snapshot.display());
        }
        Self::open_for_role(snapshot, role, |name| names.contains(name))
    }

    /// The text model configuration: `text_config` when present, else the root.
    pub fn text_config(&self) -> &Value {
        self.config
            .get("text_config")
            .filter(|value| value.is_object())
            .unwrap_or(&self.config)
    }

    /// `quantization_config` from config.json, else the external file.
    pub fn quantization(&self) -> Option<&Value> {
        self.config
            .get("quantization_config")
            .or_else(|| self.text_config().get("quantization_config"))
            .or(self.quantize_config.as_ref())
    }

    pub fn architectures(&self) -> Vec<String> {
        self.config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn model_type(&self) -> Option<&str> {
        self.config.get("model_type").and_then(Value::as_str)
    }

    /// Host placement cannot free a tensor also used by the vocabulary head.
    /// Require a separate head in addition to the config's tying declaration.
    pub fn require_untied_embedding(&self, name: &str) -> Result<()> {
        ensure!(self.text_config().get("tie_word_embeddings").and_then(Value::as_bool) != Some(true)
            && self.config.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true),
            "{name}: host embedding is ineligible: tie_word_embeddings=true");
        let embedding = self.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("host embedding tensor {name} is missing"))?;
        let family = super::family::detect(self)
            .context("host embedding requires a recognized model family")?;
        let model = family.open(self)?;
        // Use the loader's component inventory and operand validation: a trellis
        // head has no .weight, and every companion must remain disjoint too.
        let heads = super::format::group_tensor_refs(self.tensors.iter().filter(|tensor| {
            let stem = super::format::split_stem(&tensor.meta.name).0;
            family.classify(model.spec(), &format!("{stem}.weight"))
                .is_some_and(|role| role.component == super::spec::Component::LmHead)
        }));
        let mut separate_head = false;
        for members in heads.values() {
            let operand = super::format::detect(members)
                .with_context(|| format!("{name}: host embedding is ineligible: malformed LM head"))?;
            separate_head |= operand.matrix().is_some() && (members.contains_key("weight")
                || matches!(operand.encoding, super::format::Encoding::Exl3 { .. }));
        }
        ensure!(separate_head, "{name}: host embedding is ineligible: no separate LM head tensor");
        for head in heads.values().flat_map(|members| members.values()) {
            let same_file = head.shard == embedding.shard
                || std::fs::canonicalize(self.snapshot.join(&head.shard)).ok().zip(
                    std::fs::canonicalize(self.snapshot.join(&embedding.shard)).ok())
                    .is_some_and(|(a, b)| a == b);
            let overlaps = head.meta.byte_offset < embedding.meta.byte_offset.saturating_add(embedding.meta.byte_length)
                && embedding.meta.byte_offset < head.meta.byte_offset.saturating_add(head.meta.byte_length);
            ensure!(!(same_file && overlaps),
                "{name}: host embedding is ineligible: LM head {} aliases its storage", head.meta.name);
        }
        Ok(())
    }
}

/// Reads an integer config field, checking `text_config` first.
pub fn usize_field(config: &Value, key: &str) -> Result<usize> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("config field {key} is missing or not an unsigned integer"))
}

pub fn opt_usize_field(config: &Value, key: &str) -> Option<usize> {
    config.get(key).and_then(Value::as_u64).map(|value| value as usize)
}

#[cfg(test)]
mod sliced_tests {
    use super::*;
    use crate::plan::testing::{t, write_safetensors};
    use serde_json::json;

    #[test]
    fn header_presence_is_distinct_from_index_membership() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        write_safetensors(&dir.path().join("model.safetensors"), &[
            t("a.weight", "BF16", &[8, 8]), t("z.weight", "BF16", &[8, 8]),
        ]);
        let mut checkpoint = Checkpoint::open(dir.path()).unwrap();
        checkpoint.weight_map.remove("a.weight");
        checkpoint.weight_map.insert("missing.weight".into(), "absent.safetensors".into());
        for name in ["", "a", "a.weight", "a.weight.child", "missing.weight", "z.weight", "zz"] {
            assert_eq!(checkpoint.has_header(name),
                checkpoint.tensors.iter().any(|tensor| tensor.meta.name == name), "{name}");
        }
        assert!(checkpoint.has_header("a.weight"));
        assert!(!checkpoint.contains_tensor("a.weight"));
        assert!(!checkpoint.has_header("missing.weight"));
        assert!(checkpoint.contains_tensor("missing.weight"));
        checkpoint.weight_map.clear();
        assert!(checkpoint.contains_tensor("a.weight"));
        checkpoint.tensors.clear();
        assert!(!checkpoint.has_header("a.weight"));
    }

    #[test]
    fn projection_prefix_lookup_matches_full_scan() {
        for mixed in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.json"), "{}").unwrap();
            let mut tensors = crate::plan::testing::exl3("a", 128, 128, 6);
            tensors.extend(crate::plan::testing::exl3("a_sibling", 128, 128, 6));
            tensors.extend([t("a.child.weight", "BF16", &[8, 8]), t("a.weight", "BF16", &[8, 8])]);
            if mixed {
                tensors.extend([t("native.weight", "F8_E4M3", &[128, 128]),
                    t("native.weight_scale_inv", "F32", &[1, 1]),
                    t("native.child.weight", "BF16", &[8, 8]),
                    t("collision.weight", "BF16", &[8, 8]),
                    t("collision.trellis", "I16", &[8])]);
            }
            write_safetensors(&dir.path().join("model.safetensors"), &tensors);
            let mut checkpoint = Checkpoint::open(dir.path()).unwrap();
            // Retain an unindexed header to exercise the authoritative index filter.
            checkpoint.weight_map = checkpoint.tensors.iter().filter(|tensor|
                tensor.meta.name != "a.weight" && !tensor.meta.name.ends_with(".child.weight"))
                .map(|tensor| (tensor.meta.name.clone(), tensor.shard.clone())).collect();
            for stem in ["a", "a_sibling", "native", "collision", "missing"] {
                let prefix = format!("{stem}.");
                let scan: Vec<_> = checkpoint.tensors.iter().filter(|tensor|
                    tensor.meta.name.starts_with(&prefix)
                        && !tensor.meta.name[prefix.len()..].contains('.')
                        && checkpoint.contains_tensor(&tensor.meta.name)).collect();
                let weight = scan.iter().find(|tensor| tensor.meta.name == format!("{stem}.weight"));
                let trellis = scan.iter().any(|tensor| tensor.meta.name == format!("{stem}.trellis"));
                let actual = checkpoint.projection(stem);
                if (weight.is_some() && trellis) || (weight.is_none() && !trellis) {
                    assert!(actual.is_err(), "{mixed}/{stem}");
                    continue;
                }
                let actual = actual.unwrap();
                let expected = if trellis { ProjectionStorage::Exl3 } else {
                    let weight = weight.unwrap();
                    ProjectionStorage::Native { dtype: weight.meta.dtype.clone(), shape: weight.meta.shape.clone() }
                };
                assert_eq!(actual.storage, expected, "{mixed}/{stem}");
                assert_eq!(actual.source_bytes, scan.iter().map(|tensor| tensor.meta.byte_length).sum::<u64>());
                assert_eq!(actual.tensors.len(), scan.len());
                for (actual, expected) in actual.tensors.iter().zip(scan) {
                    assert_eq!(actual.shard, expected.shard);
                    assert_eq!(actual.meta.name, expected.meta.name);
                    assert_eq!(actual.meta.dtype, expected.meta.dtype);
                    assert_eq!(actual.meta.shape, expected.meta.shape);
                    assert_eq!(actual.meta.byte_offset, expected.meta.byte_offset);
                    assert_eq!(actual.meta.byte_length, expected.meta.byte_length);
                }
            }
        }
    }

    #[test]
    fn spark_input_agrees_between_coordinator_and_worker_for_every_family_and_format() {
        for prefix in ["layers.0.ffn", "model.layers.0.mlp", "model.language_model.layers.0.mlp"] {
            for family in ["deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "qwen4", "mimo_v2"] {
                for format in ["nvfp4", "mxfp4", "fp8", "exl3", "bf16"] {
                    let dir = tempfile::tempdir().unwrap();
                    std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&json!({
                        "model_type": family, "num_hidden_layers": 1
                    })).unwrap()).unwrap();
                    let mut tensors = vec![t("lm_head.weight", "BF16", &[2]),
                        t(format!("{prefix}.experts.0.w1.weight"), "U8", &[2])];
                    if format == "nvfp4" {
                        tensors.extend([t(format!("{prefix}.experts.0.w1.input_scale"), "F32", &[1]),
                            t(format!("{prefix}.experts.0.w1.weight_scale_2"), "F32", &[1])]);
                    }
                    // Dense NVFP4 and a differently quantized MTP never select the backbone wire.
                    for stem in ["model.layers.0.mlp.gate_proj", "model.layers.1.mlp.experts.0.w1",
                        "model.mtp.layers.0.mlp.experts.0.w1", "mtp.0.ffn.experts.0.w1"] {
                        tensors.extend([t(format!("{stem}.input_scale"), "F32", &[1]),
                            t(format!("{stem}.weight_scale_2"), "F32", &[1])]);
                    }
                    write_safetensors(&dir.path().join("model.safetensors"), &tensors);
                    let coordinator = Checkpoint::open_for_role(dir.path(), "coordinator", |name| name == "lm_head.weight").unwrap();
                    let worker = Checkpoint::inventory(dir.path()).unwrap();
                    assert_eq!(coordinator.tensors.len(), 1);
                    assert_eq!(coordinator.spark_input_is_bf16(), format == "nvfp4", "{family}/{format}/{prefix}");
                    assert_eq!(coordinator.spark_input_is_bf16(), worker.spark_input_is_bf16());
                }
            }
        }
    }

    #[test]
    fn declared_numerics_respect_ignore_and_mixed_groups() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&json!({
            "quantization_config": {"quant_method":"modelopt", "ignore":["*.shared_experts.*"],
                "config_groups": {
                    "a4":{"targets":["model.layers.0.mlp.experts"],
                        "weights":{"num_bits":4,"group_size":16},
                        "input_activations":{"num_bits":4,"group_size":16}},
                    "a8":{"targets":["mtp.layers.0.mlp.experts"],
                        "weights":{"num_bits":8},"input_activations":{"num_bits":8}}
                }}
        })).unwrap()).unwrap();
        write_safetensors(&dir.path().join("model.safetensors"), &[t("head.weight", "BF16", &[2])]);
        let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.experts.17.gate_proj.weight").unwrap(), ModuleNumerics::W4A4);
        assert_eq!(checkpoint.module_numerics("mtp.layers.0.mlp.experts.3.down_proj").unwrap(), ModuleNumerics::W8A8);
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.shared_experts.gate_proj").unwrap(), ModuleNumerics::CheckpointPrecision);
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.experts_extra.gate_proj").unwrap(), ModuleNumerics::CheckpointPrecision);
    }

    #[test]
    fn modelopt_hf_metadata_and_legacy_algorithm_respect_exclusions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        write_safetensors(&dir.path().join("model.safetensors"), &[t("head.weight", "BF16", &[2])]);
        std::fs::write(dir.path().join("hf_quant_config.json"), serde_json::to_vec(&json!({
            "producer": {"name":"modelopt"}, "quantization": {
                "quant_algo":"NVFP4", "exclude_modules":["*.shared_experts.*"]
            }
        })).unwrap()).unwrap();
        let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.experts.3.gate_proj").unwrap(), ModuleNumerics::W4A4);
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.shared_experts.gate_proj").unwrap(), ModuleNumerics::CheckpointPrecision);
        assert_eq!(checkpoint.module_numerics("head").unwrap(), ModuleNumerics::CheckpointPrecision);
    }

    #[test]
    fn glmf_nextn_bf16_header_preserves_precision_but_partial_nvfp4_fails() {
        // NVIDIA GLM-5.3-Flash-NVFP4 layer 45 is BF16 [2048,4096],
        // absent from both ignore lists despite the class-wide Linear declaration.
        let stem = "model.language_model.layers.45.mlp.experts.65.gate_proj";
        let mut checkpoint = Checkpoint { view: None, snapshot:PathBuf::new(), config:Value::Null,
            quantize_config:None, weight_map:BTreeMap::new(), tensors:Vec::new(),
            missing_shards:Vec::new(), shard_bytes:0 };
        checkpoint.config = json!({"text_config":{"num_hidden_layers":45,"num_nextn_predict_layers":1},
            "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4",
                "ignore":["model.language_model.layers.44.self_attn*"],
                "config_groups":{"group_0":{"targets":["Linear"],
                    "weights":{"num_bits":4,"group_size":16},
                    "input_activations":{"num_bits":4,"group_size":16}}}}});
        checkpoint.tensors = vec![CheckpointTensor { shard:"model-00002-of-00033.safetensors".into(),
            meta: crate::SafetensorsTensorMetadata { name:format!("{stem}.weight"), dtype:cuteafd_core::DType::Bf16,
                shape:vec![2048,4096], byte_offset:0, byte_length:2048*4096*2 } }];
        checkpoint.weight_map.clear();
        assert_eq!(checkpoint.module_numerics(stem).unwrap(), ModuleNumerics::CheckpointPrecision);
        checkpoint.tensors[0].meta.dtype = cuteafd_core::DType::U8;
        checkpoint.tensors[0].meta.shape = vec![2048,2048];
        assert!(checkpoint.module_numerics(stem).unwrap_err().to_string().contains("partial or invalid"));
        checkpoint.tensors[0].meta.dtype = cuteafd_core::DType::Bf16;
        checkpoint.weight_map.insert(format!("{stem}.weight_scale"), "scales.safetensors".into());
        assert!(checkpoint.module_numerics(stem).is_err());
        checkpoint.tensors[0].meta.dtype = cuteafd_core::DType::U8;
        for suffix in ["weight_scale", "weight_scale_2", "input_scale"] {
            checkpoint.weight_map.insert(format!("{stem}.{suffix}"), "scales.safetensors".into());
        }
        assert_eq!(checkpoint.module_numerics(stem).unwrap(), ModuleNumerics::W4A4);
    }

    #[test]
    fn nested_declared_numerics_control_bf16_ingress() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let stem = "model.layers.0.mlp.experts.0.gate_proj";
        write_safetensors(&dir.path().join("model.safetensors"), &[
            t(format!("{stem}.weight"), "U8", &[1]),
            t(format!("{stem}.input_scale"), "F32", &[1]),
            t(format!("{stem}.weight_scale_2"), "F32", &[1]),
        ]);
        for (bits, bf16) in [(4, true), (8, false)] {
            std::fs::write(dir.path().join("hf_quant_config.json"), serde_json::to_vec(&json!({
                "producer":{"name":"modelopt"}, "quantization":{"config_groups":{
                    "experts":{"targets":[stem],
                        "weights":{"num_bits":bits,"group_size":16},
                        "input_activations":{"num_bits":bits,"group_size":16}}
                }}
            })).unwrap()).unwrap();
            assert_eq!(Checkpoint::inventory(dir.path()).unwrap().spark_input_is_bf16(), bf16);
        }
    }

    fn mixed_mxfp8_config() -> Value {
        json!({"num_hidden_layers":1, "quantization_config":{
            "quant_method":"mxfp8", "activation_scheme":"dynamic", "weight_block_size":[1,32],
            "group_size":16, "moe_quant_algo":"NVFP4", "moe_quant_format":"nvfp4-pack-quantized",
            "ignored_layers":["lm_head", "vision_tower", "*.block_sparse_moe.gate"]}})
    }

    fn compressed_expert(stem: &str) -> Vec<crate::plan::testing::Tensor> {
        vec![t(format!("{stem}.weight_packed"), "U8", &[16,32]),
            t(format!("{stem}.weight_scale"), "F8_E4M3", &[16,4]),
            t(format!("{stem}.weight_global_scale"), "F32", &[]),
            t(format!("{stem}.input_global_scale"), "F32", &[])]
    }

    #[test]
    fn mxfp8_mixed_native_numerics_are_role_independent() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), mixed_mxfp8_config().to_string()).unwrap();
        let routed = "language_model.model.layers.0.block_sparse_moe.experts.0.w1";
        let dense = "language_model.model.layers.0.self_attn.index_q_proj";
        let mut tensors = compressed_expert(routed);
        tensors.extend([t(format!("{dense}.weight"), "F8_E4M3", &[16,64]),
            t(format!("{dense}.weight_scale_inv"), "U8", &[16,2]),
            t("language_model.model.layers.0.block_sparse_moe.gate.weight", "F32", &[16,64]),
            t("language_model.model.layers.0.self_attn.q_norm.weight", "BF16", &[64]),
            t("lm_head.weight", "BF16", &[16,64]),
            t("vision_tower.q_proj.weight", "BF16", &[16,64])]);
        write_safetensors(&dir.path().join("model.safetensors"), &tensors);
        for checkpoint in [Checkpoint::open(dir.path()).unwrap(), Checkpoint::inventory(dir.path()).unwrap(),
            Checkpoint::open_for_role(dir.path(), "coordinator", |name| name == "lm_head.weight").unwrap()] {
            for suffix in ["", ".weight_packed"] {
                assert_eq!(checkpoint.module_numerics(&format!("{routed}{suffix}")).unwrap(), ModuleNumerics::W4A4);
            }
            assert_eq!(checkpoint.module_numerics(dense).unwrap(), ModuleNumerics::W8A8);
            for module in ["lm_head", "vision_tower.q_proj", "language_model.model.layers.0.block_sparse_moe.gate",
                "language_model.model.layers.0.self_attn.q_norm"] {
                assert_eq!(checkpoint.module_numerics(module).unwrap(), ModuleNumerics::CheckpointPrecision, "{module}");
            }
            let map = checkpoint.declared_module_numerics().unwrap();
            assert_eq!(map[routed], ModuleNumerics::W4A4);
            assert_eq!(map[dense], ModuleNumerics::W8A8);
            assert!(checkpoint.spark_input_is_bf16());
        }
        let projection = Checkpoint::open(dir.path()).unwrap().projection(routed).unwrap();
        assert_eq!(projection.storage, ProjectionStorage::CompressedNvfp4 { shape:vec![16,64] });
        assert_eq!(projection.source_bytes, 16*32 + 16*4 + 2*4);
        assert_eq!(projection.tensors.len(), 4);
    }

    #[test]
    fn mxfp8_native_contracts_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let routed = "model.layers.0.mlp.experts.0.w1";
        let dense = "model.layers.0.self_attn.q_proj";
        for (key, value) in [("activation_scheme", json!("static")), ("weight_block_size", json!([1,16])),
            ("group_size", json!(32)), ("moe_quant_algo", json!("FP8")), ("moe_quant_format", json!("modelopt"))] {
            let mut config = mixed_mxfp8_config();
            config["quantization_config"][key] = value;
            std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
            write_safetensors(&dir.path().join("model.safetensors"), &compressed_expert(routed));
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            let module = if matches!(key, "activation_scheme" | "weight_block_size") { dense } else { routed };
            assert!(checkpoint.module_numerics(module).is_err(), "{key}");
        }
        std::fs::write(dir.path().join("config.json"), mixed_mxfp8_config().to_string()).unwrap();
        let native = [t(format!("{dense}.weight"), "F8_E4M3", &[16,64]),
            t(format!("{dense}.weight_scale_inv"), "U8", &[16,2])];
        for removed in 0..4 {
            let mut tensors = compressed_expert(routed);
            tensors.remove(removed);
            tensors.extend(native.clone());
            write_safetensors(&dir.path().join("model.safetensors"), &tensors);
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            assert!(checkpoint.projection(routed).is_err(), "missing member {removed}");
            assert!(checkpoint.module_numerics(routed).is_err(), "missing member {removed}");
        }
        for bad in [t(format!("{dense}.weight_scale_inv"), "F32", &[16,2]),
            t(format!("{dense}.weight_scale_inv"), "U8", &[16,4])] {
            write_safetensors(&dir.path().join("model.safetensors"), &[native[0].clone(), bad]);
            assert!(Checkpoint::open(dir.path()).unwrap().module_numerics(dense).is_err());
        }
        let mut tensors = compressed_expert(routed);
        tensors.push(t(format!("{routed}.weight"), "BF16", &[16,64]));
        write_safetensors(&dir.path().join("model.safetensors"), &tensors);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        assert!(checkpoint.projection(routed).is_err());
        assert!(checkpoint.module_numerics(routed).is_err());
    }

    #[test]
    fn compressed_nvfp4_wire_ignores_draft_and_out_of_range_experts() {
        for stem in ["model.layers.1.mlp.experts.0.w1", "model.mtp.layers.0.mlp.experts.0.w1",
            "nextn.layers.0.mlp.experts.0.w1", "dspark.layers.0.mlp.experts.0.w1"] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.json"), mixed_mxfp8_config().to_string()).unwrap();
            write_safetensors(&dir.path().join("model.safetensors"), &compressed_expert(stem));
            assert!(!Checkpoint::inventory(dir.path()).unwrap().spark_input_is_bf16(), "{stem}");
        }
    }

    #[test]
    fn real_nvidia_configs_select_declared_module_numerics() {
        // Optional local inventory gate: no tensor payload reads or GPU needed.
        let Some(root) = std::env::var_os("CUTEAFD_NVFP4_CONFIG_ROOT") else { return };
        let root = Path::new(&root);
        for (model, routed, preserved, a8) in [
            ("GLM-5.3-NVFP4", "model.layers.3.mlp.experts.0.gate_proj", "model.layers.3.mlp.shared_experts.gate_proj", Some("model.layers.0.mlp.gate_proj")),
            ("GLM-5.3-Flash-NVFP4", "model.language_model.layers.3.mlp.experts.0.gate_proj", "model.language_model.layers.3.mlp.shared_experts.gate_proj", None),
            ("DeepSeek-V4.1-Flash-NVFP4", "layers.0.ffn.experts.0.w1", "layers.0.ffn.shared_experts.w1", None),
            ("Qwen3.8-Flash-Next-NVFP4", "model.language_model.layers.0.mlp.experts.0.gate_proj", "model.language_model.layers.0.mlp.shared_expert.gate_proj", Some("mtp.layers.0.mlp.experts.0.gate_proj")),
        ] {
            let snapshots = root.join(format!("models--nvidia--{model}/snapshots"));
            let path = std::fs::read_dir(snapshots).unwrap().map(|entry| entry.unwrap().path())
                .find(|path| path.join("config.json").is_file()).unwrap();
            let checkpoint = Checkpoint::inventory(&path).unwrap();
            assert_eq!(checkpoint.module_numerics(routed).unwrap(), ModuleNumerics::W4A4, "{model}");
            assert_eq!(checkpoint.module_numerics(preserved).unwrap(), ModuleNumerics::CheckpointPrecision, "{model}");
            if let Some(module) = a8 {
                assert_eq!(checkpoint.module_numerics(module).unwrap(), ModuleNumerics::W8A8, "{model}");
            }
            if model == "GLM-5.3-Flash-NVFP4" {
                let nextn = "model.language_model.layers.45.mlp.experts.65.gate_proj";
                assert_eq!(checkpoint.module_numerics(nextn).unwrap(), ModuleNumerics::CheckpointPrecision);
                let header = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{nextn}.weight")).unwrap();
                assert_eq!(header.meta.dtype, cuteafd_core::DType::Bf16);
                assert_eq!(header.meta.shape, [2048,4096]);
            }
            assert!(checkpoint.spark_input_is_bf16(), "{model}");
            let numerics = checkpoint.declared_module_numerics().unwrap();
            assert!(numerics.values().any(|value| *value == ModuleNumerics::W4A4), "{model}");
            // Compare every indexed weight, not just representative modules:
            // legacy per-expert exports must keep exactly the same ignore and
            // quantized sets under module ownership resolution.
            for module in numerics.keys() {
                assert_eq!(checkpoint.declared_numerics(module).unwrap(),
                    checkpoint.declared_numerics_for_owner(module, module).unwrap(),
                    "changed legacy declaration for {model}/{module}");
            }
            let report = super::super::plan(&path, &super::super::PlanOptions {
                placement: super::super::ExpertPlacement::Sparks { ranks: 6 },
                layout: Some(Default::default()), ..Default::default()
            }).unwrap();
            assert!(report.config_error.is_none(), "{model}: {:?}", report.config_error);
            eprintln!("{model}: {} numerics entries unchanged; live planner accepted config", numerics.len());
        }
    }

    #[test]
    fn role_headers_and_index_presence_are_independent() {
        let dir = tempfile::tempdir().unwrap();
        let names = ["lm_head.weight", "model.layers.1.mlp.experts.7.gate_proj.weight",
            "model.mtp.layers.0.eh_proj.weight", "model.visual.weight"];
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let map: BTreeMap<_, _> = names.iter().enumerate().map(|(i, name)|
            (name.to_string(), format!("shard{i}.safetensors"))).collect();
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": map})).unwrap()).unwrap();
        for (i, name) in names.iter().enumerate() {
            write_safetensors(&dir.path().join(format!("shard{i}.safetensors")), &[t(*name, "BF16", &[2])]);
        }
        let full = Checkpoint::open(dir.path()).unwrap();
        std::fs::remove_file(dir.path().join("shard1.safetensors")).unwrap();
        std::fs::write(dir.path().join("shard3.safetensors"), b"corrupt").unwrap();
        let local = |name: &str| name == names[0] || name == names[2];
        let sliced = Checkpoint::open_for_role(dir.path(), "coordinator", local).unwrap();
        assert_eq!(full.tensor_names().collect::<Vec<_>>(), sliced.tensor_names().collect::<Vec<_>>());
        assert_eq!(full.tensors.iter().filter(|t| local(&t.meta.name)).map(|t| &t.meta).collect::<Vec<_>>(),
            sliced.tensors.iter().map(|t| &t.meta).collect::<Vec<_>>());
        assert!(sliced.contains_tensor(names[1]));
        assert!(sliced.contains_tensor(names[3]));
        let error = Checkpoint::open_for_role(dir.path(), "spark0", |name| name == names[1]).unwrap_err().to_string();
        for expected in ["spark0", names[1], "shard1.safetensors", dir.path().to_str().unwrap()] {
            assert!(error.contains(expected), "{error}");
        }
        assert!(Checkpoint::open_for_role(dir.path(), "vision", |name| name == names[3]).is_err());
        std::fs::remove_file(dir.path().join("shard0.safetensors")).unwrap();
        let error = Checkpoint::open_for_role(dir.path(), "coordinator", local).unwrap_err().to_string();
        assert!(error.contains(names[0]) && error.contains("shard0.safetensors"));
    }

    #[test]
    fn shard_paths_stay_inside_snapshot_even_for_unselected_roles() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        for shard in ["/outside.safetensors", "../outside.safetensors", "dense/../../outside.safetensors", ""] {
            std::fs::write(dir.path().join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"weight_map": {"model.visual.weight": shard}})).unwrap()).unwrap();
            for selected in [false, true] {
                let error = Checkpoint::open_for_role(dir.path(), "vision", |_| selected).unwrap_err().to_string();
                assert!(error.contains("invalid checkpoint shard path"), "{shard:?}: {error}");
            }
        }
    }

    #[test]
    fn grouped_shards_ignore_unindexed_and_wrong_shard_headers() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a.weight":"a.safetensors","z.weight":"z.safetensors"}}"#).unwrap();
        write_safetensors(&dir.path().join("a.safetensors"), &[
            t("a.weight", "BF16", &[2]), t("z.weight", "BF16", &[2]), t("extra.weight", "BF16", &[2])]);
        write_safetensors(&dir.path().join("z.safetensors"), &[t("z.weight", "BF16", &[2])]);
        let (_, validation) = Checkpoint::open_with_header_validation(dir.path()).unwrap();
        assert!(validation.is_ok());
        let checkpoint = Checkpoint::open_for_role(dir.path(), "coordinator", |_| true).unwrap();
        assert_eq!(checkpoint.tensors.iter().map(|tensor| (tensor.meta.name.as_str(), tensor.shard.as_str()))
            .collect::<Vec<_>>(), [("a.weight", "a.safetensors"), ("z.weight", "z.safetensors")]);
        assert_eq!(Checkpoint::open_for_role(dir.path(), "coordinator", |name| name == "z.weight")
            .unwrap().tensors.len(), 1);
    }

    #[test]
    fn selected_tensor_missing_from_header_is_named() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"head.weight":"head.safetensors"}}"#).unwrap();
        write_safetensors(&dir.path().join("head.safetensors"), &[t("other.weight", "BF16", &[2])]);
        let error = Checkpoint::open_for_role(dir.path(), "coordinator", |_| true).unwrap_err().to_string();
        assert!(error.contains("head.weight") && error.contains("head.safetensors"));
        let required = BTreeSet::from(["absent.weight".to_string()]);
        assert!(Checkpoint::open_required(dir.path(), "coordinator", &required).unwrap_err()
            .to_string().contains("absent from weight_map"));
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    use crate::SafetensorsTensorMetadata;
    use cuteafd_core::DType;

    fn fixture() -> Checkpoint {
        let t = |name: &str, offset| CheckpointTensor { shard: "model.safetensors".into(),
            meta: SafetensorsTensorMetadata { name: name.into(), dtype: DType::Bf16,
                shape: vec![64, 128], byte_offset: offset, byte_length: 16384 } };
        Checkpoint { view: None, snapshot: Default::default(), config: super::super::testing::glm5_config(),
            quantize_config: None, weight_map: Default::default(), missing_shards: vec![], shard_bytes: 0,
            tensors: vec![t("model.embed_tokens.weight", 0), t("lm_head.weight", 16384)] }
    }

    const GLMF_EMBEDDING: &str = "model.language_model.embed_tokens.weight";

    fn published_quantized_head() -> Checkpoint {
        let source: Value = serde_json::from_str(include_str!(
            "checkpoint/glm53f-k4-core-k6-head.json")).unwrap();
        let mut checkpoint = fixture();
        checkpoint.config = source["config"].clone();
        checkpoint.tensors = source["header"].as_object().unwrap().iter().map(|(name, header)| {
            let offsets = header["data_offsets"].as_array().unwrap();
            CheckpointTensor { shard: source["shard"].as_str().unwrap().into(),
                meta: SafetensorsTensorMetadata { name: name.clone(),
                    dtype: DType::from_safetensors(header["dtype"].as_str().unwrap()),
                    shape: header["shape"].as_array().unwrap().iter()
                        .map(|dim| dim.as_u64().unwrap() as usize).collect(),
                    byte_offset: offsets[0].as_u64().unwrap(),
                    byte_length: offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap(),
                } }
        }).collect();
        checkpoint
    }

    #[test]
    fn host_embedding_accepts_published_glmf_k6_head_headers() {
        let checkpoint = published_quantized_head();
        assert_eq!(checkpoint.text_config()["tie_word_embeddings"], false);
        assert!(!checkpoint.tensors.iter().any(|t| t.meta.name == "lm_head.weight"));
        checkpoint.require_untied_embedding(GLMF_EMBEDDING).unwrap();
    }

    #[test]
    fn host_embedding_refuses_tied_quantized_head() {
        for root in [true, false] {
            let mut checkpoint = published_quantized_head();
            if root { checkpoint.config["tie_word_embeddings"] = Value::Bool(true); }
            else { checkpoint.config["text_config"]["tie_word_embeddings"] = Value::Bool(true); }
            assert!(checkpoint.require_untied_embedding(GLMF_EMBEDDING).unwrap_err()
                .to_string().contains("tie_word_embeddings=true"));
        }
    }

    #[test]
    fn host_embedding_refuses_missing_or_incomplete_quantized_head() {
        let checkpoint = published_quantized_head();
        for member in checkpoint.tensors.iter().filter(|t| t.meta.name.starts_with("lm_head.")) {
            let mut incomplete = published_quantized_head();
            incomplete.tensors.retain(|t| t.meta.name != member.meta.name);
            assert!(incomplete.require_untied_embedding(GLMF_EMBEDDING).is_err(), "{}", member.meta.name);
        }
        let mut missing = published_quantized_head();
        missing.tensors.retain(|t| t.meta.name == GLMF_EMBEDDING);
        assert!(missing.require_untied_embedding(GLMF_EMBEDDING).unwrap_err()
            .to_string().contains("no separate LM head"));
    }

    #[test]
    fn host_embedding_refuses_every_overlapping_quantized_head_member() {
        for member in published_quantized_head().tensors.iter().filter(|t| t.meta.name.starts_with("lm_head.")) {
            let mut checkpoint = published_quantized_head();
            let offset = checkpoint.tensors.iter().find(|t| t.meta.name == GLMF_EMBEDDING).unwrap().meta.byte_offset;
            checkpoint.tensors.iter_mut().find(|t| t.meta.name == member.meta.name).unwrap().meta.byte_offset = offset;
            let error = checkpoint.require_untied_embedding(GLMF_EMBEDDING).unwrap_err().to_string();
            assert!(error.contains("aliases") && error.contains(&member.meta.name), "{error}");
        }
    }

    #[test]
    fn host_embedding_rejects_a_lone_head_companion() {
        let mut checkpoint = published_quantized_head();
        checkpoint.tensors.retain(|t| t.meta.name == GLMF_EMBEDDING || t.meta.name == "lm_head.mcg");
        assert!(checkpoint.require_untied_embedding(GLMF_EMBEDDING).is_err());
    }

    #[test]
    fn host_embedding_checks_fp8_head_scales_too() {
        let mut checkpoint = fixture();
        checkpoint.tensors[1].meta.dtype = DType::F8E4M3;
        let mut scale = checkpoint.tensors[1].clone();
        scale.meta = SafetensorsTensorMetadata { name: "lm_head.weight_scale_inv".into(),
            dtype: DType::F32, shape: vec![1, 1], byte_offset: 32768, byte_length: 4 };
        checkpoint.tensors.push(scale);
        checkpoint.require_untied_embedding("model.embed_tokens.weight").unwrap();
        checkpoint.tensors[2].meta.byte_offset = 0;
        assert!(checkpoint.require_untied_embedding("model.embed_tokens.weight").unwrap_err()
            .to_string().contains("lm_head.weight_scale_inv aliases"));
    }

    #[test]
    fn host_embedding_requires_untied_distinct_storage() {
        let mut c = fixture();
        assert!(c.require_untied_embedding("model.embed_tokens.weight").is_ok());
        c.config["tie_word_embeddings"] = Value::Bool(true);
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("tie_word_embeddings"));
        c.config = serde_json::json!({"text_config":{"tie_word_embeddings":true}});
        assert!(c.require_untied_embedding("model.embed_tokens.weight").is_err());
        c.config = fixture().config;
        c.tensors[1].meta.byte_offset = 8192;
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("aliases"));
        c.tensors.pop();
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("no separate LM head"));
    }
}
