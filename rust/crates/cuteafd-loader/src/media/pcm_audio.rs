//! Header-derived plan for a bundled BF16 raw-waveform pyramid (Muse Spark).
use super::audio_tower::{AudioTensorRead, AudioTowerError};
use crate::{read_safetensors_metadata, SafetensorsTensorMetadata};
use cuteafd_core::DType;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek, SeekFrom},
    path::{Component, Path},
};

type Result<T> = std::result::Result<T, AudioTowerError>;
const PREFIX: &str = "model.audio_tower.";
pub const WIDTHS: [usize; 4] = [256, 512, 1024, 2048];
pub const PATCHES: [usize; 4] = [16, 6, 5, 4];
pub const SAMPLES_PER_TOKEN: usize = 1920;
pub const OUTPUT_WIDTH: usize = 8192;
pub const CHUNK_TOKENS: usize = 16;
fn unsupported(s: impl Into<String>) -> AudioTowerError {
    AudioTowerError::Unsupported(s.into())
}
fn align(n: u64) -> u64 {
    n.div_ceil(256) * 256
}

#[derive(Clone, Debug)]
pub struct PcmAudioPlan {
    weight_bytes: u64,
    reads: BTreeMap<String, AudioTensorRead>,
    pub audio_token: u32,
}
/// Matches pcm_audio_runtime.cu scratch(), including six raw BF16 history rows
/// per depthwise convolution. Streaming capacity is fixed, not clip-proportional.
pub fn scratch_bytes(samples: usize) -> Result<u64> {
    if !(1..=7_200_000).contains(&samples) {
        return Err(unsupported("PCM must contain 1..7200000 samples"));
    }
    let tokens = samples.div_ceil(SAMPLES_PER_TOKEN).min(CHUNK_TOKENS) as u64;
    let mut rows = tokens * SAMPLES_PER_TOKEN as u64;
    let mut largest = tokens * OUTPUT_WIDTH as u64;
    for (h, p) in WIDTHS.into_iter().zip(PATCHES) {
        rows /= p as u64;
        largest = largest.max(rows * h as u64);
    }
    Ok(align(largest * 2) * 3
        + align(largest * 8) * 2
        + WIDTHS
            .into_iter()
            .map(|h| align(h as u64 * 6 * 2) * 2)
            .sum::<u64>())
}
impl PcmAudioPlan {
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn reads(&self) -> &BTreeMap<String, AudioTensorRead> {
        &self.reads
    }
    pub fn required_bytes(&self, samples: usize) -> Result<u64> {
        self.weight_bytes
            .checked_add(scratch_bytes(samples)?)
            .ok_or_else(|| unsupported("PCM admission overflow"))
    }
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let root: serde_json::Value = crate::read_json_file(&snapshot.join("config.json"))?;
        let processor: serde_json::Value =
            crate::read_json_file(&snapshot.join("processor_config.json"))?;
        validate_config(&root, &processor)?;
        let mut files = BTreeSet::new();
        let index = snapshot.join("model.safetensors.index.json");
        if index.exists() {
            let index: serde_json::Value = crate::read_json_file(&index)?;
            for (key, file) in index["weight_map"]
                .as_object()
                .ok_or_else(|| unsupported("missing weight_map"))?
            {
                if !key.starts_with(PREFIX) {
                    continue;
                }
                let file = file
                    .as_str()
                    .ok_or_else(|| unsupported("invalid PCM shard"))?;
                let path = Path::new(file);
                if path.is_absolute()
                    || path
                        .components()
                        .any(|p| !matches!(p, Component::Normal(_)))
                {
                    return Err(unsupported(format!("invalid PCM shard path {file}")));
                }
                files.insert(snapshot.join(file));
            }
        } else {
            files.insert(snapshot.join("model.safetensors"));
        }
        let mut catalog = BTreeMap::new();
        for path in files {
            for tensor in
                read_safetensors_metadata(&path).map_err(|e| unsupported(e.to_string()))?
            {
                if tensor.name.starts_with(PREFIX)
                    && catalog
                        .insert(tensor.name.clone(), (path.clone(), tensor))
                        .is_some()
                {
                    return Err(unsupported("duplicate PCM tensor"));
                }
            }
        }
        let mut result = Self {
            weight_bytes: 0,
            reads: BTreeMap::new(),
            audio_token: root["audio_token_id"].as_u64().unwrap() as u32,
        };
        let shapes = tensor_shapes();
        if catalog.len() != shapes.len() {
            return Err(unsupported(format!(
                "expected {} PCM tensors, found {}",
                shapes.len(),
                catalog.len()
            )));
        }
        for (name, shape) in shapes {
            let full = format!("{PREFIX}{name}");
            let (path, metadata) = catalog
                .get(&full)
                .ok_or_else(|| unsupported(format!("missing {full}")))?;
            validate_tensor(metadata, &shape)?;
            let destination = align(result.weight_bytes);
            result.weight_bytes = destination
                .checked_add(metadata.byte_length)
                .ok_or_else(|| unsupported("PCM weights overflow"))?;
            result.reads.insert(
                name,
                AudioTensorRead {
                    metadata: metadata.clone(),
                    path: path.clone(),
                    destination,
                    resident_dtype: DType::Bf16,
                },
            );
        }
        result.weight_bytes = align(result.weight_bytes);
        Ok(result)
    }
    /// The returned bytes are exactly the admitted arena, without a transformed
    /// second copy. Call only after checking the caller's admission budget.
    pub fn read_weights(&self, admitted: u64) -> Result<Vec<u8>> {
        if admitted < self.weight_bytes {
            return Err(AudioTowerError::Admission {
                required: self.weight_bytes,
                admitted,
            });
        }
        let mut bytes = vec![
            0;
            usize::try_from(self.weight_bytes)
                .map_err(|_| unsupported("PCM arena too large"))?
        ];
        for read in self.reads.values() {
            let mut file = std::fs::File::open(&read.path)?;
            file.seek(SeekFrom::Start(read.metadata.byte_offset))?;
            let start = read.destination as usize;
            file.read_exact(&mut bytes[start..start + read.metadata.byte_length as usize])?;
        }
        Ok(bytes)
    }
}
fn validate_tensor(m: &SafetensorsTensorMetadata, shape: &[usize]) -> Result<()> {
    let bytes = shape
        .iter()
        .try_fold(2u64, |n, &d| n.checked_mul(d as u64))
        .ok_or_else(|| unsupported("PCM shape overflow"))?;
    if m.dtype != DType::Bf16 || m.shape != shape || m.byte_length != bytes {
        return Err(unsupported(format!(
            "{}: {:?} {:?}, need BF16 {shape:?}",
            m.name, m.dtype, m.shape
        )));
    }
    Ok(())
}
fn validate_config(root: &serde_json::Value, processor: &serde_json::Value) -> Result<()> {
    let a = &root["audio_config"];
    let p = &processor["feature_extractor"];
    if root["model_type"] != "muse_spark"
        || a["model_type"] != "muse_spark_audio"
        || a["hidden_sizes"] != serde_json::json!(WIDTHS)
        || a["patch_sizes"] != serde_json::json!(PATCHES)
        || a["depths"] != serde_json::json!([2, 2, 2, 2])
        || a["num_hidden_layers"] != 4
        || a["conv_kernel_size"] != 7
        || a["mlp_ratio"].as_f64() != Some(4.0)
        || a["rms_norm_eps"].as_f64() != Some(1e-6)
        || a["hidden_act"] != "gelu"
        || a["out_hidden_size"] != OUTPUT_WIDTH
        || root["text_config"]["hidden_size"] != OUTPUT_WIDTH
        || root["audio_token_id"]
            .as_u64()
            .is_none_or(|v| v > u32::MAX as u64)
        || p["sampling_rate"] != 24000
        || p["samples_per_token"] != SAMPLES_PER_TOKEN
        || p["feature_size"] != 1
        || p["padding_value"].as_f64() != Some(0.0)
        || p["padding_side"] != "right"
    {
        return Err(unsupported(
            "bundled Muse PCM geometry/processor mismatch; add matching PCM exporter",
        ));
    }
    Ok(())
}
pub fn tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    let mut previous = 1;
    for (i, (h, p)) in WIDTHS.into_iter().zip(PATCHES).enumerate() {
        let prefix = format!("layers.{i}");
        shapes.insert(
            format!("{prefix}.patch_embedder.proj.weight"),
            vec![h, previous * p],
        );
        shapes.insert(format!("{prefix}.patch_embedder.proj.bias"), vec![h]);
        shapes.insert(format!("{prefix}.patch_embedder.norm.weight"), vec![h]);
        for b in 0..2 {
            let prefix = format!("{prefix}.conv_blocks.{b}");
            for (name, shape) in [
                ("conv.weight", vec![h, 1, 7]),
                ("conv.bias", vec![h]),
                ("norm.weight", vec![h]),
                ("mlp.fc1.weight", vec![h * 4, h]),
                ("mlp.fc1.bias", vec![h * 4]),
                ("mlp.fc2.weight", vec![h, h * 4]),
                ("mlp.fc2.bias", vec![h]),
            ] {
                shapes.insert(format!("{prefix}.{name}"), shape);
            }
        }
        previous = h;
    }
    for (name, shape) in [
        ("norm.weight", vec![previous]),
        ("proj.weight", vec![OUTPUT_WIDTH, previous]),
        ("proj.bias", vec![OUTPUT_WIDTH]),
        ("post_proj_norm.weight", vec![OUTPUT_WIDTH]),
    ] {
        shapes.insert(name.into(), shape);
    }
    shapes
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_seventy_two_tensors_and_bounded_workspace() {
        assert_eq!(tensor_shapes().len(), 72);
        assert!(scratch_bytes(1).unwrap() > 0);
        assert_eq!(
            scratch_bytes(1920 * 16).unwrap(),
            scratch_bytes(7_200_000).unwrap()
        );
        assert!(scratch_bytes(0).is_err());
        assert!(scratch_bytes(7_200_001).is_err());
    }
    #[test]
    fn header_only_snapshot_admission_matches_native_ledger() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let config = serde_json::json!({"model_type":"muse_spark", "audio_token_id":200038,
            "text_config":{"hidden_size":8192}, "audio_config":{
                "model_type":"muse_spark_audio", "hidden_sizes":WIDTHS, "patch_sizes":PATCHES,
                "depths":[2,2,2,2], "num_hidden_layers":4, "conv_kernel_size":7,
                "mlp_ratio":4, "rms_norm_eps":1e-6, "hidden_act":"gelu", "out_hidden_size":8192}});
        let processor = serde_json::json!({"feature_extractor":{"sampling_rate":24000,
            "samples_per_token":1920, "feature_size":1, "padding_value":0, "padding_side":"right"}});
        std::fs::write(root.join("config.json"), config.to_string()).unwrap();
        std::fs::write(root.join("processor_config.json"), processor.to_string()).unwrap();
        let mut header = serde_json::Map::new();
        let mut cursor = 0u64;
        for (name, shape) in tensor_shapes() {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            header.insert(format!("{PREFIX}{name}"), serde_json::json!({
                "dtype":"BF16", "shape":shape, "data_offsets":[cursor,cursor+bytes]}));
            cursor += bytes;
        }
        let mut encoded = serde_json::to_vec(&header).unwrap();
        encoded.resize(encoded.len().div_ceil(8)*8, b' ');
        let mut file = std::fs::File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&encoded).unwrap();
        // Sparse payload proves planning reads headers, not the 225 MiB weights.
        file.set_len(8 + encoded.len() as u64 + cursor).unwrap();
        let plan = PcmAudioPlan::from_snapshot(root).unwrap();
        assert_eq!(plan.reads().len(), 72);
        assert_eq!(plan.audio_token, 200038);
        assert_eq!(plan.weight_bytes(), 235_680_768);
        assert_eq!(scratch_bytes(7_200_000).unwrap(), 10_905_600);
        assert_eq!(plan.required_bytes(7_200_000).unwrap(), 246_586_368);
        assert!(matches!(plan.read_weights(plan.weight_bytes()-1), Err(AudioTowerError::Admission { .. })));
    }
    #[test]
    fn reject_dtype_and_truncated_payload() {
        let mut m = SafetensorsTensorMetadata {
            name: "model.audio_tower.proj.weight".into(),
            dtype: DType::Bf16,
            shape: vec![8192, 2048],
            byte_offset: 0,
            byte_length: 8192 * 2048 * 2,
        };
        assert!(validate_tensor(&m, &[8192, 2048]).is_ok());
        m.dtype = DType::F32;
        assert!(validate_tensor(&m, &[8192, 2048]).is_err());
        m.dtype = DType::Bf16;
        m.byte_length -= 2;
        assert!(validate_tensor(&m, &[8192, 2048]).is_err());
    }
}
