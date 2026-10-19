//! EXL3 storage maps, read from the checkpoint itself.
//!
//! exllamav3 stores a quantized linear `M` (`K` inputs, `N` outputs) as
//! `M.trellis` (I16 `[K/16, N/16, 16 * bits]`), the unpacked input/output
//! sign-and-scale vectors `M.suh` (F16 `[K]`) and `M.svh` (F16 `[N]`), and a
//! codebook marker whose presence selects the codebook: `M.mcg` (MCG; exllamav3
//! writes I32 `[]`, other exporters I32 `[1]`), `M.mul1` (MUL1), or none
//! (3INST). Releases before `suh`/`svh` stored packed sign bitfields `M.su` /
//! `M.sv` (I16). The marker's four bytes hold the codebook multiplier
//! (MCG 0xCBAC1FED), which exllamav3 has locked and no longer reads.
//!
//! `config.json` `quantization_config` carries only `quant_method`, `version`,
//! `bits` (the checkpoint-wide average), `head_bits`, `codebook`, `out_scales`
//! and `calibration`. A storage map (`tensor_storage`) lists every module's
//! stored tensors with `quant_format`, `bits_per_weight` (the trellis's last
//! dimension / 16) and the marker multiplier: GPTQModel writes one to
//! `quantize_config.json` (`{shape, torch_dtype}` entries), exllamav3's
//! `create_quantization_config_json` to `quantization_config.json`
//! (`{dtype: "torch.*", n_bytes, shape}`). Everything in it but the multiplier
//! value is in the safetensors headers, so the map is derived from them, and a
//! map that is present must agree with the derivation. The multiplier is
//! checked where the payload is read (`V41Exl3Residency::read_into`).
use crate::SafetensorsTensorMetadata;
use cuteafd_core::DType;
use serde_json::Value;
use std::collections::BTreeMap;

/// The MCG codebook multiplier, the only one the trellis decoders implement.
pub const EXL3_MCG_MULTIPLIER: u64 = 0xcbac_1fed;

/// Tensor suffixes that belong to an EXL3 module (`bias` aside).
pub const EXL3_SUFFIXES: [&str; 7] = ["trellis", "suh", "svh", "mcg", "mul1", "su", "sv"];

/// Where a checkpoint's EXL3 storage map came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Exl3StorageSource {
    /// GPTQModel's `quantize_config.json`, checked against the headers.
    GptqModel,
    /// exllamav3's `quantization_config.json`, checked against the headers.
    Exllamav3,
    /// No storage map: derived from `config.json` and the headers alone.
    Headers,
}

impl Exl3StorageSource {
    pub fn describe(self) -> &'static str {
        match self {
            Self::GptqModel => "quantize_config.json (GPTQModel), agrees with the tensor headers",
            Self::Exllamav3 => "quantization_config.json (exllamav3), agrees with the tensor headers",
            Self::Headers => "derived from config.json and the tensor headers (standard exllamav3, no storage map)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exl3StoredTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// One EXL3 module as stored: MCG codebook, unpacked rotations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exl3Module {
    pub bits: usize,
    pub input_features: usize,
    pub output_features: usize,
    /// `trellis`, `suh`, `svh` and `mcg`, by suffix.
    pub tensors: BTreeMap<String, Exl3StoredTensor>,
}

/// EXL3 modules by name (the tensor names without their suffix).
pub type Exl3StorageMap = BTreeMap<String, Exl3Module>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Exl3StorageError {
    /// A tensor whose header no EXL3 reader accepts.
    #[error("{tensor}: {reason}")]
    Malformed { tensor: String, reason: String },
    /// A well-formed exllamav3 variant this build has no kernels for.
    #[error("{tensor}: {variant} (this build runs the MCG codebook with unpacked suh/svh)")]
    Unsupported { tensor: String, variant: String },
    /// `config.json` declares a contract this build does not run.
    #[error("config.json quantization_config.{key}={value}: {reason}")]
    Config { key: String, value: String, reason: String },
    /// A storage map entry that cannot be read.
    #[error("{file}: {module}: {reason}")]
    Map { file: String, module: String, reason: String },
    /// A storage map that disagrees with the checkpoint's headers.
    #[error("{file} disagrees with the safetensors headers at {module}: {detail}")]
    Mismatch { file: String, module: String, detail: String },
}

/// Whether `shape` is an MCG marker shape: exllamav3 writes a scalar, other
/// exporters a one-element vector. Both hold the same four bytes.
pub fn mcg_marker_shape(shape: &[usize]) -> bool {
    shape.is_empty() || shape == [1]
}

fn dtype_label(dtype: &DType) -> String {
    format!("{dtype:?}").to_ascii_lowercase()
}

/// Validates one module's members (`member(suffix)`, plus the full suffix
/// list) against the EXL3 storage contract.
pub fn exl3_module<'a>(
    stem: &str,
    suffixes: &[&str],
    member: impl Fn(&str) -> Option<&'a SafetensorsTensorMetadata>,
) -> Result<Exl3Module, Exl3StorageError> {
    let name = |suffix: &str| format!("{stem}.{suffix}");
    let malformed = |suffix: &str, reason: String| Exl3StorageError::Malformed { tensor: name(suffix), reason };
    let unsupported = |suffix: &str, variant: &str| Exl3StorageError::Unsupported {
        tensor: name(suffix),
        variant: variant.to_owned(),
    };
    if member("mul1").is_some() {
        return Err(unsupported("mul1", "EXL3 MUL1 codebook marker"));
    }
    if let Some(suffix) = ["su", "sv"].into_iter().find(|s| member(s).is_some()) {
        return Err(unsupported(suffix, "packed EXL3 sign bitfield (su/sv, written by exllamav3 before suh/svh)"));
    }
    let Some(trellis) = member("trellis") else {
        let present = suffixes.iter().copied().find(|s| EXL3_SUFFIXES.contains(s)).unwrap_or("trellis");
        return Err(malformed(present, "EXL3 companion without a trellis".into()));
    };
    let [k16, n16, packed] = trellis.shape.as_slice() else {
        return Err(malformed("trellis", format!("trellis of rank {} {:?} (expected [K/16, N/16, 16 x bits])",
            trellis.shape.len(), trellis.shape)));
    };
    if trellis.dtype != DType::I16 {
        return Err(malformed("trellis", format!("trellis dtype {} (expected i16)", dtype_label(&trellis.dtype))));
    }
    if *packed == 0 || packed % 16 != 0 || !(1..=8).contains(&(packed / 16)) {
        return Err(malformed("trellis", format!("trellis last dim {packed} is not 16 x bits for 1..=8 bits")));
    }
    let (k, n, bits) = (k16 * 16, n16 * 16, packed / 16);
    let expected = (k * n * bits / 8) as u64;
    if trellis.byte_length != expected {
        return Err(malformed("trellis", format!("{} bytes for {k} x {n} at K{bits} (expected {expected})",
            trellis.byte_length)));
    }
    let mut tensors = BTreeMap::new();
    tensors.insert("trellis".to_owned(), Exl3StoredTensor { dtype: DType::I16, shape: trellis.shape.clone() });
    for (suffix, dtype, shape, bytes) in [
        ("suh", DType::F16, vec![k], 2 * k as u64),
        ("svh", DType::F16, vec![n], 2 * n as u64),
        ("mcg", DType::I32, vec![], 4),
    ] {
        let Some(tensor) = member(suffix) else {
            return Err(if suffix == "mcg" {
                unsupported("trellis", "EXL3 3INST codebook (no .mcg or .mul1 marker)")
            } else {
                malformed(suffix, "missing EXL3 companion (a trellis needs suh, svh and mcg)".into())
            });
        };
        let shape_ok = if suffix == "mcg" { mcg_marker_shape(&tensor.shape) } else { tensor.shape == shape };
        if tensor.dtype != dtype || !shape_ok || tensor.byte_length != bytes {
            let expected = if suffix == "mcg" { "i32 [] or [1]".to_owned() } else {
                format!("{} {shape:?}", dtype_label(&dtype))
            };
            return Err(malformed(suffix, format!("{} {:?} {} bytes (expected {expected}, {bytes} bytes)",
                dtype_label(&tensor.dtype), tensor.shape, tensor.byte_length)));
        }
        tensors.insert(suffix.to_owned(), Exl3StoredTensor { dtype, shape: tensor.shape.clone() });
    }
    if let Some(extra) = suffixes.iter().find(|s| !["trellis", "suh", "svh", "mcg"].contains(s)) {
        return Err(if *extra == "bias" {
            unsupported("bias", "EXL3 projection bias")
        } else {
            malformed(extra, format!("unexpected member .{extra} beside an EXL3 trellis"))
        });
    }
    Ok(Exl3Module { bits, input_features: k, output_features: n, tensors })
}

/// The storage map of every EXL3 module in `headers`: each stem with an EXL3
/// member (`trellis`, `suh`, `svh`, `mcg`, `mul1`, `su`, `sv`) must be a
/// complete MCG module.
pub fn derive_storage_map<'a>(
    headers: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>,
) -> Result<Exl3StorageMap, Exl3StorageError> {
    let mut groups: BTreeMap<&str, BTreeMap<&str, &SafetensorsTensorMetadata>> = BTreeMap::new();
    for tensor in headers {
        if let Some((stem, suffix)) = tensor.name.rsplit_once('.') {
            groups.entry(stem).or_default().insert(suffix, tensor);
        }
    }
    let mut map = BTreeMap::new();
    for (stem, members) in groups {
        if !members.keys().any(|suffix| EXL3_SUFFIXES.contains(suffix)) {
            continue;
        }
        let suffixes: Vec<&str> = members.keys().copied().collect();
        map.insert(stem.to_owned(), exl3_module(stem, &suffixes, |suffix| members.get(suffix).copied())?);
    }
    Ok(map)
}

/// Keep compact configuration fields, streaming the per-tensor map instead of
/// retaining its million-node JSON tree. When supplied, headers remain the authority.
pub(crate) fn read_storage_metadata(path: &std::path::Path, limit: u64,
    derived: Option<&Exl3StorageMap>) -> anyhow::Result<Value> {
    use anyhow::{ensure, Context};
    use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor, Error};
    use std::io::Read;
    struct Storage<'a> { file: &'a str, derived: &'a Exl3StorageMap }
    impl<'de> DeserializeSeed<'de> for Storage<'_> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
            de.deserialize_map(self)
        }
    }
    impl<'de> Visitor<'de> for Storage<'_> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("EXL3 tensor_storage map") }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
            let mut seen = std::collections::BTreeSet::new();
            let mut entries = 0;
            while let Some(name) = map.next_key::<String>()? {
                let entry: Value = map.next_value()?;
                entries += 1;
                let declared = parse_storage_map(self.file, &serde_json::json!({&name: entry}))
                    .map_err(M::Error::custom)?;
                if let Some(value) = declared.get(&name) {
                    let Some((key, actual)) = self.derived.get_key_value(&name) else {
                        return Err(M::Error::custom(format!("{} disagrees with the safetensors headers at {name}: the map lists this module; the checkpoint has no such EXL3 tensors", self.file)));
                    };
                    if !seen.insert(key.as_str()) { return Err(M::Error::custom(format!("duplicate EXL3 module {name}"))); }
                    if value != actual {
                        return Err(M::Error::custom(format!("{} disagrees with the safetensors headers at {name}: map {}, headers {}",
                            self.file, describe_module(value), describe_module(actual))));
                    }
                }
            }
            if entries == 0 { return Err(M::Error::custom("tensor_storage missing or empty")); }
            if let Some(name) = self.derived.keys().find(|name| !seen.contains(name.as_str())) {
                return Err(M::Error::custom(format!("{} disagrees with the safetensors headers at {name}: the checkpoint stores this EXL3 module; the map does not list it", self.file)));
            }
            Ok(())
        }
    }
    struct Metadata<'a> { file: &'a str, derived: Option<&'a Exl3StorageMap> }
    impl<'de> Visitor<'de> for Metadata<'_> {
        type Value = Value;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("quantization metadata object") }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Value, M::Error> {
            let mut fields = serde_json::Map::new();
            let mut storage = false;
            while let Some(key) = map.next_key::<String>()? {
                if key == "tensor_storage" {
                    if storage { return Err(M::Error::custom("duplicate tensor_storage")); }
                    storage = true;
                    if let Some(derived) = self.derived {
                        map.next_value_seed(Storage { file: self.file, derived })?;
                        fields.insert(key, Value::Bool(true));
                    } else { map.next_value::<IgnoredAny>()?; }
                } else {
                    let value = map.next_value()?;
                    if fields.insert(key.clone(), value).is_some() {
                        return Err(M::Error::custom(format!("duplicate metadata key {key}")));
                    }
                }
            }
            if self.derived.is_some() && !storage { return Err(M::Error::custom("missing tensor_storage")); }
            Ok(Value::Object(fields))
        }
    }
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = file.metadata()?.len();
    ensure!(size <= limit, "{} exceeds metadata size limit", path.display());
    // Slice parsing avoids serde_json's per-byte Read adapter; no JSON tree is retained.
    let mut bytes = Vec::with_capacity(usize::try_from(size)?.checked_add(1).context("metadata size overflow")?);
    file.take(limit.checked_add(1).context("metadata limit overflow")?).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "{} exceeds metadata size limit", path.display());
    let mut de = serde_json::Deserializer::from_slice(&bytes);
    let label = path.file_name().and_then(|s| s.to_str()).unwrap_or("metadata");
    let result = serde::Deserializer::deserialize_map(&mut de, Metadata { file: label, derived })
        .with_context(|| format!("parsing {}", path.display()))?;
    de.end()?;
    Ok(result)
}

fn dtype_from_map(value: &str) -> Option<DType> {
    Some(match value.strip_prefix("torch.").unwrap_or(value) {
        "int16" => DType::I16,
        "int32" => DType::I32,
        "float16" | "half" => DType::F16,
        "bfloat16" => DType::Bf16,
        "float32" | "float" => DType::F32,
        _ => return None,
    })
}

fn dtype_bytes(dtype: &DType) -> u64 {
    match dtype {
        DType::I16 | DType::F16 | DType::Bf16 => 2,
        DType::I32 | DType::F32 => 4,
        _ => 0,
    }
}

/// Reads a storage map's EXL3 entries. Entries without `quant_format` are
/// unquantized modules (exllamav3 lists every module) and are skipped.
pub fn parse_storage_map(file: &str, storage: &Value) -> Result<Exl3StorageMap, Exl3StorageError> {
    let map_error = |module: &str, reason: String| Exl3StorageError::Map {
        file: file.to_owned(),
        module: module.to_owned(),
        reason,
    };
    let entries = storage.as_object().filter(|entries| !entries.is_empty())
        .ok_or_else(|| map_error("tensor_storage", "missing or empty".into()))?;
    let mut map = BTreeMap::new();
    for (module, entry) in entries {
        let Some(format) = entry.get("quant_format") else { continue };
        if format != "exl3" {
            return Err(map_error(module, format!("quant_format {format} (expected exl3)")));
        }
        if let Some(multiplier) = entry.get("mcg_multiplier") {
            if multiplier.as_u64() != Some(EXL3_MCG_MULTIPLIER) {
                return Err(Exl3StorageError::Unsupported {
                    tensor: format!("{module}.mcg"),
                    variant: format!("MCG multiplier {multiplier} (the decoders implement 0xCBAC1FED)"),
                });
            }
        }
        if entry.get("mul1_multiplier").is_some() {
            return Err(Exl3StorageError::Unsupported {
                tensor: format!("{module}.mul1"),
                variant: "EXL3 MUL1 codebook marker".into(),
            });
        }
        let stored = entry.get("stored_tensors").and_then(Value::as_object)
            .ok_or_else(|| map_error(module, "no stored_tensors".into()))?;
        let mut headers = BTreeMap::new();
        for (name, tensor) in stored {
            let suffix = name.strip_prefix(module.as_str()).and_then(|rest| rest.strip_prefix('.'))
                .ok_or_else(|| map_error(module, format!("stored tensor {name} belongs to another module")))?;
            let dtype_text = tensor.get("torch_dtype").or_else(|| tensor.get("dtype")).and_then(Value::as_str)
                .ok_or_else(|| map_error(module, format!("{name} has no dtype")))?;
            let dtype = dtype_from_map(dtype_text)
                .ok_or_else(|| map_error(module, format!("{name} has unknown dtype {dtype_text}")))?;
            let shape = tensor.get("shape").and_then(Value::as_array)
                .and_then(|dims| dims.iter().map(|d| d.as_u64().map(|d| d as usize)).collect::<Option<Vec<_>>>())
                .ok_or_else(|| map_error(module, format!("{name} has no integer shape")))?;
            let bytes = dtype_bytes(&dtype) * shape.iter().product::<usize>() as u64;
            if let Some(declared) = tensor.get("n_bytes") {
                if declared.as_u64() != Some(bytes) {
                    return Err(map_error(module, format!("{name} n_bytes {declared} for {dtype_text} {shape:?}")));
                }
            }
            headers.insert(suffix.to_owned(), SafetensorsTensorMetadata {
                name: name.clone(),
                dtype,
                shape,
                byte_offset: 0,
                byte_length: bytes,
            });
        }
        let suffixes: Vec<&str> = headers.keys().map(String::as_str).collect();
        let parsed = exl3_module(module, &suffixes, |suffix| headers.get(suffix))?;
        let declared = entry.get("bits_per_weight").and_then(Value::as_u64);
        if declared != Some(parsed.bits as u64) {
            return Err(map_error(module, format!("bits_per_weight {} but its trellis is K{}",
                entry.get("bits_per_weight").unwrap_or(&Value::Null), parsed.bits)));
        }
        map.insert(module.clone(), parsed);
    }
    Ok(map)
}

fn describe_module(module: &Exl3Module) -> String {
    let tensors: Vec<String> = module.tensors.iter()
        .map(|(suffix, t)| format!("{suffix} {} {:?}", dtype_label(&t.dtype), t.shape))
        .collect();
    format!("K{} [{}]", module.bits, tensors.join(", "))
}

/// Checks a declared storage map against the one derived from the headers:
/// the same modules, each with the same bits and stored tensors.
pub fn verify_storage_map(
    file: &str,
    declared: &Exl3StorageMap,
    derived: &Exl3StorageMap,
) -> Result<(), Exl3StorageError> {
    let mismatch = |module: &str, detail: String| Exl3StorageError::Mismatch {
        file: file.to_owned(),
        module: module.to_owned(),
        detail,
    };
    if let Some(module) = derived.keys().find(|module| !declared.contains_key(*module)) {
        return Err(mismatch(module, "the checkpoint stores this EXL3 module; the map does not list it".into()));
    }
    for (module, entry) in declared {
        let Some(actual) = derived.get(module) else {
            return Err(mismatch(module, "the map lists this module; the checkpoint has no such EXL3 tensors".into()));
        };
        if entry != actual {
            return Err(mismatch(module, format!("map {}, headers {}", describe_module(entry), describe_module(actual))));
        }
    }
    Ok(())
}

/// Checks the `config.json` `quantization_config` of an EXL3 checkpoint
/// without a storage map: the keys exllamav3 (and GPTQModel) define must name
/// the layout this build runs. Unknown keys are publisher annotations
/// (exllamav3 ignores them too); the per-tensor contract is the headers'.
pub fn check_quantization_config(quant: &Value) -> Result<(), Exl3StorageError> {
    let config = |key: &str, value: &Value, reason: &str| Exl3StorageError::Config {
        key: key.to_owned(),
        value: value.to_string(),
        reason: reason.to_owned(),
    };
    let object = quant.as_object().ok_or_else(|| config("", quant, "not an object"))?;
    let method = object.get("quant_method").unwrap_or(&Value::Null);
    if method != "exl3" {
        return Err(config("quant_method", method, "not an EXL3 checkpoint"));
    }
    let bits = object.get("bits").unwrap_or(&Value::Null);
    if !bits.as_f64().is_some_and(|bits| bits.is_finite() && (1.0..=8.0).contains(&bits)) {
        return Err(config("bits", bits, "EXL3 bits must be a number in 1..=8"));
    }
    for (key, value) in object {
        let ok = match key.as_str() {
            "method" | "format" | "checkpoint_format" => value == "exl3",
            "codebook" => value == "mcg",
            // Whether the quantizer folded output scales into svh; the
            // kernels apply svh per channel either way.
            "out_scales" => matches!(value.as_str(), Some("auto" | "never")),
            "group_size" => value == -1,
            "desc_act" | "lm_head" => value == false,
            "pack_dtype" => value == "int32",
            "head_bits" => value.as_f64().is_some_and(|bits| bits > 0.0),
            "tensor_storage" => false,
            _ => true,
        };
        if !ok {
            let reason = match key.as_str() {
                "codebook" => "this build runs the MCG codebook only",
                "tensor_storage" => "an inline storage map belongs in quantize_config.json",
                _ => "unsupported EXL3 storage contract",
            };
            return Err(config(key, value, reason));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(name: &str, dtype: DType, shape: &[usize]) -> SafetensorsTensorMetadata {
        let bytes = dtype_bytes(&dtype) * shape.iter().product::<usize>() as u64;
        SafetensorsTensorMetadata { name: name.into(), dtype, shape: shape.to_vec(), byte_offset: 0, byte_length: bytes }
    }

    /// One exllamav3 projection `[n, k]` at `bits` with the given marker shape.
    fn projection(stem: &str, n: usize, k: usize, bits: usize, mcg: &[usize]) -> Vec<SafetensorsTensorMetadata> {
        vec![
            header(&format!("{stem}.trellis"), DType::I16, &[k / 16, n / 16, 16 * bits]),
            header(&format!("{stem}.suh"), DType::F16, &[k]),
            header(&format!("{stem}.svh"), DType::F16, &[n]),
            header(&format!("{stem}.mcg"), DType::I32, mcg),
        ]
    }

    #[test]
    fn streaming_metadata_checks_each_module_and_keeps_only_compact_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("quantize_config.json");
        let derived = derive_storage_map(&projection("m.up_proj", 256, 512, 4, &[1])).unwrap();
        let document = serde_json::json!({"quant_method": "exl3", "bits": 4,
            "tensor_storage": {"m.up_proj": gptqmodel_entry("m.up_proj", 4, 512, 256, serde_json::json!([1]))}});
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let compact = read_storage_metadata(&path, 1 << 20, None).unwrap();
        assert_eq!(compact, serde_json::json!({"quant_method": "exl3", "bits": 4}));
        let checked = read_storage_metadata(&path, 1 << 20, Some(&derived)).unwrap();
        assert_eq!(checked["tensor_storage"], true);
        assert!(read_storage_metadata(&path, 4, None).is_err());
        for bad in [
            serde_json::json!({"tensor_storage": {}}),
            serde_json::json!({"quant_method": "exl3"}),
            serde_json::json!({"tensor_storage": {"m.up_proj": gptqmodel_entry("m.up_proj", 3, 512, 256, serde_json::json!([1]))}}),
            serde_json::json!({"tensor_storage": {"m.extra": gptqmodel_entry("m.extra", 4, 512, 256, serde_json::json!([1]))}}),
        ] {
            std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
            assert!(read_storage_metadata(&path, 1 << 20, Some(&derived)).is_err());
        }
        for bad in ["{\"bits\":4,\"bits\":3}", "{\"tensor_storage\":{},\"tensor_storage\":{}}", "{} trailing"] {
            std::fs::write(&path, bad).unwrap();
            assert!(read_storage_metadata(&path, 1 << 20, None).is_err());
        }
    }

    #[test]
    fn scalar_and_vector_markers_derive_the_same_module() {
        let scalar = derive_storage_map(&projection("e.gate_proj", 256, 512, 4, &[])).unwrap();
        let vector = derive_storage_map(&projection("e.gate_proj", 256, 512, 4, &[1])).unwrap();
        let (a, b) = (&scalar["e.gate_proj"], &vector["e.gate_proj"]);
        assert_eq!((a.bits, a.input_features, a.output_features), (4, 512, 256));
        assert_eq!((b.bits, b.input_features, b.output_features), (4, 512, 256));
        assert_eq!(b.tensors["mcg"].shape, [1]);
        // Native tensors are not EXL3 modules.
        let mut headers = projection("e.up_proj", 256, 512, 3, &[1]);
        headers.push(header("e.norm.weight", DType::Bf16, &[512]));
        assert_eq!(derive_storage_map(&headers).unwrap().keys().collect::<Vec<_>>(), ["e.up_proj"]);
    }

    #[test]
    fn unsupported_variants_name_the_tensor() {
        let mut mul1 = projection("p", 32, 64, 3, &[]);
        mul1[3].name = "p.mul1".into();
        let mut packed = projection("p", 32, 64, 3, &[]);
        packed[1] = header("p.su", DType::I16, &[4]);
        let three_inst: Vec<_> = projection("p", 32, 64, 3, &[]).into_iter().take(3).collect();
        let mut wide = projection("p", 32, 64, 3, &[]);
        wide[3] = header("p.mcg", DType::I32, &[2]);
        let mut bias = projection("p", 32, 64, 3, &[]);
        bias.push(header("p.bias", DType::F16, &[32]));
        let orphan = vec![header("p.svh", DType::F16, &[32])];
        for (headers, tensor, text) in [
            (mul1, "p.mul1", "MUL1 codebook"),
            (packed, "p.su", "packed EXL3 sign bitfield"),
            (three_inst, "p.trellis", "3INST codebook"),
            (wide, "p.mcg", "expected i32 [] or [1]"),
            (bias, "p.bias", "projection bias"),
            (orphan, "p.svh", "without a trellis"),
        ] {
            let error = derive_storage_map(&headers).unwrap_err();
            let (Exl3StorageError::Malformed { tensor: named, .. } | Exl3StorageError::Unsupported { tensor: named, .. }) =
                &error else { panic!("{error}") };
            assert_eq!(named, tensor, "{error}");
            assert!(error.to_string().contains(text), "{error}");
        }
    }

    fn gptqmodel_entry(stem: &str, bits: usize, k: usize, n: usize, mcg: Value) -> Value {
        serde_json::json!({"quant_format": "exl3", "bits_per_weight": bits, "mcg_multiplier": 3417055213u64,
            "stored_tensors": {
                format!("{stem}.trellis"): {"torch_dtype": "int16", "shape": [k / 16, n / 16, 16 * bits]},
                format!("{stem}.suh"): {"torch_dtype": "float16", "shape": [k]},
                format!("{stem}.svh"): {"torch_dtype": "float16", "shape": [n]},
                format!("{stem}.mcg"): {"torch_dtype": "int32", "shape": mcg}}})
    }

    #[test]
    fn declared_maps_agree_with_headers_or_name_the_module() {
        let derived = derive_storage_map(&projection("m.up_proj", 256, 512, 4, &[1])).unwrap();
        // exllamav3's own spelling, with n_bytes and an unquantized module.
        let exllamav3 = serde_json::json!({
            "m.up_proj": {"quant_format": "exl3", "bits_per_weight": 4, "mcg_multiplier": 3417055213u64,
                "stored_tensors": {
                    "m.up_proj.trellis": {"dtype": "torch.int16", "n_bytes": 65536, "shape": [32, 16, 64]},
                    "m.up_proj.suh": {"dtype": "torch.float16", "n_bytes": 1024, "shape": [512]},
                    "m.up_proj.svh": {"dtype": "torch.float16", "n_bytes": 512, "shape": [256]},
                    "m.up_proj.mcg": {"dtype": "torch.int32", "n_bytes": 4, "shape": [1]}}},
            "m.norm": {"stored_tensors": {"m.norm.weight": {"dtype": "torch.bfloat16", "n_bytes": 1024, "shape": [512]}}},
        });
        let declared = parse_storage_map("quantization_config.json", &exllamav3).unwrap();
        verify_storage_map("quantization_config.json", &declared, &derived).unwrap();
        // GPTQModel's spelling, but the map says the marker is a scalar.
        let gptq = serde_json::json!({"m.up_proj": gptqmodel_entry("m.up_proj", 4, 512, 256, serde_json::json!([]))});
        let declared = parse_storage_map("quantize_config.json", &gptq).unwrap();
        let error = verify_storage_map("quantize_config.json", &declared, &derived).unwrap_err();
        assert!(matches!(&error, Exl3StorageError::Mismatch { module, .. } if module == "m.up_proj"), "{error}");
        assert!(error.to_string().contains("mcg i32 []") && error.to_string().contains("mcg i32 [1]"), "{error}");
        // A different tier, a module the map lacks, a module the checkpoint lacks.
        let k3 = serde_json::json!({"m.up_proj": gptqmodel_entry("m.up_proj", 3, 512, 256, serde_json::json!([1]))});
        let error = parse_storage_map("quantize_config.json", &k3).and_then(|declared| {
            verify_storage_map("quantize_config.json", &declared, &derived)
        }).unwrap_err();
        assert!(error.to_string().contains("map K3") && error.to_string().contains("headers K4"), "{error}");
        let mut both = projection("m.up_proj", 256, 512, 4, &[1]);
        both.extend(projection("m.down_proj", 512, 256, 4, &[1]));
        let error = verify_storage_map("x", &derive_storage_map(&projection("m.up_proj", 256, 512, 4, &[1])).unwrap(),
            &derive_storage_map(&both).unwrap()).unwrap_err();
        assert!(matches!(&error, Exl3StorageError::Mismatch { module, detail, .. }
            if module == "m.down_proj" && detail.contains("does not list")), "{error}");
        let error = verify_storage_map("x", &derive_storage_map(&both).unwrap(), &derived).unwrap_err();
        assert!(error.to_string().contains("no such EXL3 tensors"), "{error}");
        // Map entries that disagree with themselves.
        let mut lying = gptqmodel_entry("m.up_proj", 3, 512, 256, serde_json::json!([1]));
        lying["bits_per_weight"] = serde_json::json!(4);
        let error = parse_storage_map("q", &serde_json::json!({"m.up_proj": lying})).unwrap_err();
        assert!(error.to_string().contains("bits_per_weight 4 but its trellis is K3"), "{error}");
        let mut multiplier = gptqmodel_entry("m.up_proj", 4, 512, 256, serde_json::json!([1]));
        multiplier["mcg_multiplier"] = serde_json::json!(1);
        let error = parse_storage_map("q", &serde_json::json!({"m.up_proj": multiplier})).unwrap_err();
        assert!(error.to_string().contains("MCG multiplier 1"), "{error}");
    }

    #[test]
    fn quantization_config_keys_name_the_unsupported_value() {
        let standard = serde_json::json!({"quant_method": "exl3", "version": "0.0.43", "bits": 4, "head_bits": 16,
            "codebook": "mcg", "calibration": {"rows": 250, "cols": 2048}, "out_scales": "auto",
            "scope": "routed_experts_only"});
        check_quantization_config(&standard).unwrap();
        for (key, value, text) in [
            ("codebook", serde_json::json!("mul1"), "MCG codebook only"),
            ("codebook", serde_json::json!("3inst"), "MCG codebook only"),
            ("out_scales", serde_json::json!("always"), "storage contract"),
            ("bits", serde_json::json!(12), "1..=8"),
            ("quant_method", serde_json::json!("gptq"), "not an EXL3"),
            ("group_size", serde_json::json!(128), "storage contract"),
        ] {
            let mut config = standard.clone();
            config[key] = value;
            let error = check_quantization_config(&config).unwrap_err();
            assert!(matches!(&error, Exl3StorageError::Config { key: named, .. } if named == key), "{error}");
            assert!(error.to_string().contains(text), "{error}");
        }
    }
}
