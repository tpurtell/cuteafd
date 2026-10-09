//! Immutable HF dataset fetches, checksum validation and sealed compact references.
use crate::reference::{CompactPosition, Reference, Top, Window};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

pub const REPOSITORY: &str = "wrldsuksgo2mars/cuteafd-fidelity";
pub const REVISION: &str = "01a0948a62a478a0a9355dd5b57fe4a49bc6cca0";
pub const CONFIG: &str = "deepseek_v41-v2_20261005";
pub const FLASH_REVISION: &str = "a8af5c3fecfa31bfbf4d0558362d9ab79451842f";
pub const FLASH_CONFIG: &str = "mimo_v2-v2_20261007_flash_mopd_qkvfixed";
pub const FLASH_MANIFEST_SHA256: &str = "ef94e9772f8fe227c6b8c12fe741906276920b2dae4518bc4aa02eed6f104758";
pub const MIMO_PRO_REVISION: &str = "52e79096f3bab9f70e821d4599512901c6b84f3e";
pub const MIMO_PRO_CONFIG: &str = "mimo_v2-v2_20261007_pro_mopd_qkvfixed";
pub const MIMO_PRO_MANIFEST_SHA256: &str = "5f43ae32419525e257c56325a64c0faf774a5a44c8742d0131f216f07b39ac2d";
pub const MIMO_PRO_MEDIA_REVISION: &str = MIMO_PRO_REVISION;
pub const MIMO_PRO_MEDIA_CONFIG: &str = "mimo_v2-v2_20261007_pro_mopd_qkvfixed_media";
pub const MIMO_PRO_MEDIA_MANIFEST_SHA256: &str = "c12a621f96ce6b812a00921b98b0f0b064aabf12aa890ff87e6948c53d00ea5e";
pub const RETIRED_FLASH_REVISION: &str = "5db25a78dc2708df991df43b36636562e522a3b0";
pub const RETIRED_FLASH_CONFIG: &str = "mimo_v2-v2_20261005_flash_mopd";
pub const GLMF_REVISION: &str = "a7e7d1b4d82329acebe54ca88dc71d47d0d2056d";
pub const GLMF_CONFIG: &str = "glm5_flash-v2_20261005_bf16root";
pub const QWEN_REVISION: &str = "3e0ccef6cff461baf39ba838627edd93cb687be4";
pub const QWEN_CONFIG: &str = "qwen4-v2_20261005_fp8";
pub const V4FLASH_REVISION: &str = "de92b14a7dabecd4d5dd4f7c799920842a5c3834";
pub const V4FLASH_CONFIG: &str = "deepseek_v4-v2_20261005_v4flash";
pub const GLM_REVISION: &str = "581cdf3b5ee5a63dd5a7726b9d33ea16cf105b0d";
pub const GLM_CONFIG: &str = "glm5-v2_20261006_exl3k4_bf16root";
pub const V4PRO_REVISION: &str = "5528730649150b8d8753e052f001f203b36ca12f";
pub const V4PRO_CONFIG: &str = "deepseek_v4-v2_20261005_v4pro";

/// Sealed bench subset v1: agentic A at three context lengths, human code C,
/// structured/tool D, plain E and the frozen context anchor. Never resample.
pub const QUICK_WINDOWS: [&str; 8] = ["legacy", "a00", "a04", "a08", "a20", "c00", "d00", "e00"];

pub fn quick_windows(reference: &Reference) -> Result<Vec<Window>> {
    reference.validate()?;
    let selected: Vec<_> = reference.windows.iter().filter(|w| QUICK_WINDOWS.contains(&w.id.as_str())).cloned().collect();
    ensure!(selected.len() == 8, "dataset lacks the sealed 8-window quick subset v1");
    ensure!(selected.iter().all(|w| w.positions.len() == 512), "quick subset row coverage differs");
    ensure!(selected.iter().any(|w| w.block == "A") && selected.iter().any(|w| w.block == "E")
        && ["ctx", "gen"].iter().all(|role| selected.iter().any(|w| w.positions.iter().any(|p| w.roles[p.pos] == *role))), "invalid quick subset");
    Ok(selected)
}

pub const STANDARD_TIER: &str = "standard-v2";
pub const STANDARD_SPLIT_SHA256: &str = "2af36f78359d9ce7f37d0d2d4740c57ca11f0a77734bc13e774d37e416eee65b";
pub const STANDARD_FALLBACK: &str = "decode only; prefill not admitted";
// Pro's sealed split has a 1.1218 gen-row ratio; each half has its own verdict and bootstrap.
const STANDARD_GEN_ROW_RATIO_MAX: f64 = 1.15;
// Window ids, positions and blocks are model-independent; buckets/gen rows depend on tokenization/generations.
const STANDARD_BUCKET_DIFF_MAX: i32 = 3;

/// Sealed after balancing all seven published text sets, not sampled at runtime.
/// Each half has A/B/C/D/E counts 13/6/5/3/5. The singleton legacy anchor
/// belongs to prefill; decode has the comparable context-only C windows.
pub const STANDARD_DECODE: [&str; 32] = [
    "a00", "a01", "a04", "a05", "a08", "a09", "a10", "a12", "a17", "a19", "a21", "a24", "a25",
    "b01", "b02", "b06", "b07", "b08", "b10", "c01", "c04", "c06", "c08", "c09",
    "d01", "d02", "d04", "e00", "e02", "e05", "e06", "e07",
];
pub const STANDARD_PREFILL: [&str; 32] = [
    "a02", "a03", "a06", "a07", "a11", "a13", "a14", "a15", "a16", "a18", "a20", "a22", "a23",
    "b00", "b03", "b04", "b05", "b09", "b11", "c00", "c02", "c03", "c05", "c07",
    "d00", "d03", "d05", "e01", "e03", "e04", "e08", "legacy",
];

pub fn standard_balance(reference: &Reference) -> Value {
    let mut balance = json!({});
    for (path, ids) in [("decode", STANDARD_DECODE), ("prefill", STANDARD_PREFILL)] {
        let mut buckets = std::collections::BTreeMap::new();
        let mut generated = 0usize;
        for w in reference.windows.iter().filter(|w| ids.contains(&w.id.as_str())) {
            *buckets.entry(w.bucket.as_str()).or_insert(0usize) += 1;
            generated += w.positions.iter().filter(|p| w.roles[p.pos] == "gen").count();
        }
        balance[path] = json!({"context_buckets":buckets,"generated_positions":generated});
    }
    balance
}

pub fn prefill_admitted(settings: &[crate::report::Setting]) -> bool {
    settings.iter().any(|s| s.name == "full-prefill-logits"
        && s.value.as_deref().is_some_and(|v| matches!(v, "true" | "on" | "1")))
}

/// Fail closed if a future publication changes the sealed split's composition.
pub fn standard_windows(reference: &Reference, path: &str, admitted: bool) -> Result<Vec<Window>> {
    use std::collections::{BTreeMap, BTreeSet};
    reference.validate()?;
    let ids: BTreeSet<_> = STANDARD_DECODE.iter().chain(&STANDARD_PREFILL).copied().collect();
    ensure!(reference.windows.len() == 64 && ids.len() == 64
        && reference.windows.iter().all(|w| ids.contains(w.id.as_str()) && w.positions.len() == 512),
        "dataset differs from sealed standard-v2 windows");
    let halves: Vec<Vec<_>> = [STANDARD_DECODE, STANDARD_PREFILL].iter().map(|ids|
        ids.iter().map(|id| reference.windows.iter().find(|w| w.id == *id).unwrap()).collect()).collect();
    let mut buckets = Vec::new();
    let mut generated = Vec::new();
    for half in &halves {
        let mut blocks = BTreeMap::new();
        let mut counts = BTreeMap::new();
        let mut roles = BTreeMap::new();
        for w in half {
            *blocks.entry(w.block.as_str()).or_insert(0) += 1;
            *counts.entry(w.bucket.as_str()).or_insert(0i32) += 1;
            for p in &w.positions { *roles.entry(w.roles[p.pos].as_str()).or_insert(0usize) += 1; }
        }
        ensure!(blocks == BTreeMap::from([("A",13), ("B",6), ("C",5), ("D",3), ("E",5)]),
            "standard-v2 block stratification differs");
        ensure!(roles.get("ctx").copied().unwrap_or(0) > 0 && roles.get("gen").copied().unwrap_or(0) > 0,
            "standard-v2 requires context and generated rows in each half");
        generated.push(roles["gen"]);
        buckets.push(counts);
    }
    ensure!(buckets[0].keys().chain(buckets[1].keys()).all(|k|
        (buckets[0].get(k).copied().unwrap_or(0) - buckets[1].get(k).copied().unwrap_or(0)).abs() <= STANDARD_BUCKET_DIFF_MAX),
        "standard-v2 context buckets are not balanced");
    ensure!(*generated.iter().max().unwrap() as f64 <= STANDARD_GEN_ROW_RATIO_MAX * *generated.iter().min().unwrap() as f64,
        "standard-v2 generated row counts differ by more than 15%");
    ensure!(matches!(path, "decode" | "prefill") && (admitted || path == "decode"),
        "standard-v2 prefill scoring not admitted");
    if !admitted { return Ok(reference.windows.clone()); }
    Ok(halves[usize::from(path == "prefill")].iter().map(|w| (*w).clone()).collect())
}

pub fn ensure_valid_publication(repo: &str, commit: &str, config: &str) -> Result<()> {
    ensure!(!(repo == REPOSITORY && config == RETIRED_FLASH_CONFIG),
        "superseded MiMo Flash fidelity reference at {commit}, scores not valid (QKV scale bug); use {FLASH_CONFIG} at {FLASH_REVISION}");
    Ok(())
}

pub fn unavailable(model: &str) -> Option<String> {
    let Some((commit, config)) = default_publication(model) else {
        return Some(format!("no verified published fidelity config for {model}"));
    };
    ensure_valid_publication(REPOSITORY, commit, config).err().map(|e| e.to_string())
}

fn base_model(model: &str) -> Option<&'static str> {
    // Served IDs may retain the HF namespace or its cache-directory spelling.
    let name = model.rsplit('/').next()?.rsplit("--").next()?;
    for base in ["Qwen3.8-Flash-Next", "GLM-5.3-Flash", "DeepSeek-V4.1-Flash", "DeepSeek-V4-Flash-0731", "DeepSeek-V4-Pro-0813", "GLM-5.3"] {
        if name == base || name.strip_prefix(base).is_some_and(|suffix|
            suffix.starts_with('-') && !suffix.starts_with("-BF16") && !suffix.starts_with("-FP8")
                && !(base == "GLM-5.3" && suffix.starts_with("-Flash")) && !suffix.to_ascii_lowercase().contains("speculator")
                && !suffix.to_ascii_lowercase().contains("dflash")) {
            return Some(base);
        }
    }
    None
}

pub fn same_base_checkpoint(reference: &str, served: &str) -> bool {
    default_publication(reference).is_some_and(|p| default_publication(served) == Some(p))
}

pub fn default_publication(model: &str) -> Option<(&'static str, &'static str)> {
    if let Some(p) = crate::fidelity_match::official(model) { return Some((p.revision, p.config)); }
    match base_model(model) {
        Some("Qwen3.8-Flash-Next") => return Some((QWEN_REVISION, QWEN_CONFIG)),
        Some("GLM-5.3-Flash") => return Some((GLMF_REVISION, GLMF_CONFIG)),
        Some("DeepSeek-V4-Flash-0731") => return Some((V4FLASH_REVISION, V4FLASH_CONFIG)),
        Some("DeepSeek-V4-Pro-0813") => return Some((V4PRO_REVISION, V4PRO_CONFIG)),
        Some("GLM-5.3") => return Some((GLM_REVISION, GLM_CONFIG)),
        Some("DeepSeek-V4.1-Flash") => return Some((REVISION, CONFIG)),
        _ => {},
    }
    match model {
        "XiaomiMiMo/MiMo-V2.6-Flash-MOPD" => Some((FLASH_REVISION, FLASH_CONFIG)),
        "XiaomiMiMo/MiMo-V2.6-Pro-MOPD" => Some((MIMO_PRO_REVISION, MIMO_PRO_CONFIG)),
        _ => None,
    }
}

fn component(text: &str) -> bool {
    !text.is_empty() && text != "." && text != ".."
        && text.bytes().all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
}
fn revision(text: &str) -> bool {
    text.len() == 40 && text.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn safe_path(root: &Path, path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    ensure!(!path.as_os_str().is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "unsafe dataset path");
    Ok(root.join(path))
}
fn checked(root: &Path, path: &str, hash: &str) -> Result<Vec<u8>> {
    let bytes = std::fs::read(safe_path(root, path)?)?;
    ensure!(digest(&bytes) == hash, "dataset checksum differs: {path}");
    Ok(bytes)
}

fn fetch(agent: &ureq::Agent, root: &Path, repo: &str, commit: &str, path: &str,
    hash: Option<&str>) -> Result<Vec<u8>> {
    use std::io::Read;
    let local = safe_path(root, path)?;
    if local.exists() {
        let bytes = std::fs::read(&local)?;
        if let Some(hash) = hash { ensure!(digest(&bytes) == hash, "cached dataset checksum differs: {path}"); }
        return Ok(bytes);
    }
    let url = format!("https://huggingface.co/datasets/{repo}/resolve/{commit}/{path}");
    let mut bytes = Vec::new();
    agent.get(&url).call()?.into_reader().take(32 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 32 * 1024 * 1024, "oversized dataset file");
    if let Some(hash) = hash { ensure!(digest(&bytes) == hash, "download checksum differs: {path}"); }
    std::fs::create_dir_all(local.parent().context("dataset parent")?)?;
    // No partially downloaded file can become a valid cache hit.
    let mut file = tempfile::NamedTempFile::new_in(local.parent().unwrap())?;
    std::io::Write::write_all(&mut file, &bytes)?;
    file.persist_noclobber(&local).map_err(|e| e.error)?;
    Ok(bytes)
}

pub fn download(agent: &ureq::Agent, cache: &Path, repo: &str, commit: &str, config: &str)
    -> Result<(Reference, String, Value)> {
    ensure_valid_publication(repo, commit, config)?;
    ensure!(revision(commit), "dataset revision must be an immutable lowercase 40-hex HF commit");
    let parts: Vec<_> = repo.split('/').collect();
    ensure!(parts.len() == 2 && parts.iter().all(|p| component(p)) && component(config), "unsafe dataset identity");
    let root = cache.join(parts.join("--")).join(commit);
    let index: Value = serde_json::from_slice(&fetch(agent, &root, repo, commit, "configs.json", None)?)?;
    ensure!(index["schema"] == "cuteafd.fidelity.configs/1", "unknown dataset index schema");
    let entries = index["configs"].as_array().context("dataset configs")?;
    let matches: Vec<_> = entries.iter().filter(|e| e["name"] == config).collect();
    ensure!(matches.len() == 1, "missing or duplicate dataset config");
    let path = matches[0]["path"].as_str().context("config path")?;
    ensure!(path == format!("{config}/manifest.json"), "unexpected config manifest path");
    let bytes = fetch(agent, &root, repo, commit, path, Some(matches[0]["sha256"].as_str().context("manifest checksum")?))?;
    if repo == REPOSITORY {
        let expected = match (commit, config) {
            (FLASH_REVISION, FLASH_CONFIG) => Some(FLASH_MANIFEST_SHA256),
            (MIMO_PRO_REVISION, MIMO_PRO_CONFIG) => Some(MIMO_PRO_MANIFEST_SHA256),
            (MIMO_PRO_MEDIA_REVISION, MIMO_PRO_MEDIA_CONFIG) => Some(MIMO_PRO_MEDIA_MANIFEST_SHA256),
            (GLM_REVISION, GLM_CONFIG) => Some("cc973cec8df82119ccd53367d014c73b271808a561f81d3629b06093de3b00e0"),
            (V4PRO_REVISION, V4PRO_CONFIG) => Some("7b4fd7b51670ff03fbfab6a21b72d999163ddee9bd3eec34c89f1d6c2f83488f"),
            _ => None,
        };
        if let Some(expected) = expected { ensure!(digest(&bytes) == expected, "pinned manifest checksum differs"); }
    }
    let manifest: Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == config, "dataset config identity differs");
    if repo == REPOSITORY {
        if let Some(p) = crate::fidelity_match::PUBLICATIONS.iter().find(|p| p.revision == commit && p.config == config) {
            ensure!(manifest["root_checkpoint"]["id"] == p.root,
                "pinned publication logit root differs from highest official root {}", p.root);
        }
    }
    let base = root.join(config);
    for (path, hash) in [("windows.json", "windows_sha256"), ("qualification.json", "qualification_sha256")] {
        fetch(agent, &root, repo, commit, &format!("{config}/{path}"), Some(manifest[hash].as_str().context("dataset checksum")?))?;
    }
    for entry in manifest["files"].as_array().context("dataset files")? {
        let path = entry["path"].as_str().context("tensor path")?;
        safe_path(&base, path)?;
        fetch(agent, &root, repo, commit, &format!("{config}/{path}"), Some(entry["sha256"].as_str().context("tensor checksum")?))?;
    }
    let reference = load(&base, &manifest)?;
    Ok((reference, digest(&bytes), json!({"repository":repo,"revision":commit,"config":config})))
}

fn tensor<'a>(bytes: &'a [u8], name: &str, dtype: &str, shape: &[usize]) -> Result<&'a [u8]> {
    ensure!(bytes.len() >= 8, "truncated safetensors prefix");
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    ensure!(n <= 1 << 20 && n as usize <= bytes.len() - 8, "invalid safetensors header size");
    let start = 8 + n as usize;
    let header: Value = serde_json::from_slice(&bytes[8..start])?;
    let t = &header[name];
    ensure!(t["dtype"] == dtype && t["shape"] == json!(shape), "dataset tensor shape/dtype differs: {name}");
    let offsets = t["data_offsets"].as_array().context("tensor offsets")?;
    ensure!(offsets.len() == 2, "invalid tensor offsets");
    let a = offsets[0].as_u64().context("tensor start")?;
    let b = offsets[1].as_u64().context("tensor end")?;
    let width = match dtype { "U32" | "F32" => 4, "F16" => 2, "U8" => 1, _ => unreachable!() };
    let size = shape.iter().try_fold(width, |n: usize, &x| n.checked_mul(x)).context("tensor size overflow")?;
    ensure!(b.checked_sub(a) == Some(size as u64) && b <= (bytes.len() - start) as u64, "invalid tensor extent");
    Ok(&bytes[start + a as usize..start + b as usize])
}
fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}
fn floats(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks_exact(4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect()
}

pub fn load(root: &Path, manifest: &Value) -> Result<Reference> {
    ensure!(manifest["schema"] == "cuteafd.fidelity.dataset/1" && manifest["top_k"] == 1024
        && manifest["kind"] == "reference-top1024-plus-tail", "unsupported dataset format");
    let panel: Value = serde_json::from_slice(&checked(root, "windows.json", manifest["windows_sha256"].as_str().context("windows hash")?)?)?;
    let qualification: Value = serde_json::from_slice(&checked(root, "qualification.json", manifest["qualification_sha256"].as_str().context("qualification hash")?)?)?;
    ensure!(panel["set_sha256"] == manifest["set_sha256"] && qualification["set_sha256"] == manifest["set_sha256"]
        && qualification["qualifies"] == true, "unqualified or mismatched dataset set");
    for shape in ["decode", "prefill"] {
        ensure!(qualification["shapes"][shape]["qualifies"] == true
            && qualification["shapes"][shape]["pass_verdict"] == qualification["shapes"][shape]["original_pass"],
            "dataset lacks paired qualification for {shape}");
        for metric in ["delta_difference", "bound_difference"] {
            let difference = qualification["shapes"][shape][metric].as_f64().context("qualification difference")?;
            ensure!(difference.is_finite() && difference.abs() <= 1e-4,
                "dataset compact qualification exceeds 1e-4 nat for {shape}/{metric}");
        }
    }
    let mut reference: Reference = serde_json::from_value(json!({"name":manifest["config"],
        "models":[manifest["checkpoint"]],"vocab":manifest["vocab"],"expect":manifest["expect"],
        "schema":"cuteafd.fidelity.reference/2","checkpoint":manifest["checkpoint"],
        "set_sha256":manifest["set_sha256"],"quick_windows":panel["quick_windows"]}))?;
    let files = manifest["files"].as_array().context("tensor files")?;
    let windows = panel["windows"].as_array().context("panel windows")?;
    ensure!(files.len() == windows.len(), "dataset window coverage differs");
    for raw in windows {
        let matches: Vec<_> = files.iter().filter(|e| e["window"] == raw["id"]).collect();
        ensure!(matches.len() == 1, "missing or duplicate tensor window");
        let entry = matches[0];
        let bytes = checked(root, entry["path"].as_str().context("tensor path")?, entry["sha256"].as_str().context("tensor hash")?)?;
        ensure!(entry["bytes"].as_u64() == Some(bytes.len() as u64), "tensor byte length differs");
        let mut value = raw.clone(); value["positions"] = json!([]); value["top_k"] = json!(1024);
        let mut window: Window = serde_json::from_value(value)?;
        let n = window.tokens.len().checked_sub(window.score_from).context("invalid score start")?;
        ensure!(n == 512 && entry["shape"] == json!([n,1024]), "dataset scored row count differs");
        let positions = u32s(tensor(&bytes, "positions", "U32", &[n])?);
        let next = u32s(tensor(&bytes, "next_token_ids", "U32", &[n])?);
        let roles = tensor(&bytes, "roles", "U8", &[n])?;
        let ids = u32s(tensor(&bytes, "top_ids", "U32", &[n,1024])?);
        let lps: Vec<_> = tensor(&bytes, "top_log_probs", "F16", &[n,1024])?.chunks_exact(2)
            .map(|b| crate::fidelity_rows::half(u16::from_le_bytes(b.try_into().unwrap()))).collect();
        let tails = floats(tensor(&bytes, "tail_log_mass", "F32", &[n])?);
        let next_lps = floats(tensor(&bytes, "next_token_log_prob", "F32", &[n])?);
        ensure!(window.roles.len() == window.tokens.len(), "invalid role coverage");
        for i in 0..n {
            let pos = positions[i] as usize;
            ensure!(pos == window.score_from + i && next[i] == window.tokens[pos]
                && roles[i] <= 1 && (roles[i] == 1) == (window.roles[pos] == "gen"), "tensor token/position/role differs");
            let mut mass = lps[i*1024..(i+1)*1024].to_vec(); mass.push(tails[i]);
            crate::fidelity_rows::normalize(&mut mass)?;
            let top = ids[i*1024..(i+1)*1024].iter().zip(&mass).map(|(&id,&lp)| Top {id,lp}).collect();
            window.positions.push(CompactPosition {pos,next:next[i],next_lp:next_lps[i],top,tail_lp:mass[1024]});
        }
        reference.windows.push(window);
    }
    reference.validate()?;
    Ok(reference)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publications_and_retired_mimo_fail_closed() {
        assert_eq!(default_publication("wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"), Some((GLM_REVISION, GLM_CONFIG)));
        assert_eq!(default_publication("wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1"), Some((V4PRO_REVISION, V4PRO_CONFIG)));
        assert_eq!(default_publication("XiaomiMiMo/MiMo-V2.6-Flash-MOPD"), Some((FLASH_REVISION, FLASH_CONFIG)));
        assert_eq!(default_publication("XiaomiMiMo/MiMo-V2.6-Pro-MOPD"), Some((MIMO_PRO_REVISION, MIMO_PRO_CONFIG)));
        assert!(unavailable("XiaomiMiMo/MiMo-V2.6-Pro-MOPD").is_none());
        assert!(unavailable("XiaomiMiMo/MiMo-V2.6-Flash-MOPD").is_none());
        assert!(ensure_valid_publication(REPOSITORY, FLASH_REVISION, FLASH_CONFIG).is_ok());
        let retired = ensure_valid_publication(REPOSITORY, RETIRED_FLASH_REVISION, RETIRED_FLASH_CONFIG).unwrap_err();
        assert!(retired.to_string().contains(FLASH_CONFIG));
        // Later dataset commits can still contain the superseded config.
        assert!(ensure_valid_publication(REPOSITORY, FLASH_REVISION, RETIRED_FLASH_CONFIG).is_err());
        assert!(unavailable("XiaomiMiMo/MiMo-V2.6-Pro-RL").is_some());
        assert!(unavailable("zai-org/GLM-5.3-Flashlight").is_some());
        for quant in ["deepseek-ai/DeepSeek-V4.1-Flash", "nvidia/DeepSeek-V4.1-Flash-NVFP4",
            "nvidia--DeepSeek-V4.1-Flash-NVFP4", "wrldsuksgo2mars/DeepSeek-V4.1-Flash-EXL3-K3.25-v1"] {
            assert_eq!(default_publication(quant), Some((REVISION, CONFIG)), "{quant}");
        }
        for other in ["deepseek-ai/DeepSeek-V4.1-Flashlight", "deepseek-ai/DeepSeek-V4.1",
            "other/DeepSeek-V4.1-Flash-speculator", "other/DeepSeek-V4.1-Flash-DFlash2"] {
            assert!(unavailable(other).is_some(), "{other}");
        }
    }

    #[test]
    fn prefill_admission_uses_resolved_launch_setting_only() {
        use crate::report::Setting;
        assert!(!prefill_admitted(&[]));
        for value in ["false", "off", "0", "true", "on", "1"] {
            let setting = Setting { name:"full-prefill-logits".into(), value:Some(value.into()),
                default:Some("false".into()), source:"cli".into() };
            assert_eq!(prefill_admitted(&[setting]), matches!(value,"true"|"on"|"1"));
        }
    }

    #[test]
    fn sealed_quick_subset_is_eight_balanced_windows_and_fails_closed() {
        use crate::reference::{CompactPosition, Top};
        let mut reference: Reference = serde_json::from_value(json!({"name":"synthetic", "models":[], "vocab":1,
            "schema":"cuteafd.fidelity.reference/2", "checkpoint":"test", "set_sha256":"sealed",
            "expect":{"top1_min":0.9,"kl_max":0.06}})).unwrap();
        for id in QUICK_WINDOWS {
            let role = if id == "legacy" || id == "c00" { "ctx" } else { "gen" };
            reference.windows.push(Window { id: id.into(), block: if id.starts_with('a') { "A" } else { "E" }.into(),
                bucket:"0-2K".into(), tokens:vec![0;513], roles:vec![role.into();513], score_from:1, top_k:1, media:vec![],
                positions:(1..513).map(|pos| CompactPosition {pos,next:0,next_lp:0.0,
                    top:vec![Top{id:0,lp:0.0}],tail_lp:f64::NEG_INFINITY}).collect() });
        }
        let selected = quick_windows(&reference).unwrap();
        assert_eq!(selected.len(),8); assert_eq!(selected.iter().map(|w|w.positions.len()).sum::<usize>(),4096);
        reference.windows.pop(); assert!(quick_windows(&reference).is_err());
    }

    #[test]
    fn standard_split_is_sealed_stratified_and_order_independent() {
        use std::collections::{BTreeMap, BTreeSet};
        let mut reference: Reference = serde_json::from_value(json!({"name":"synthetic", "models":[], "vocab":1,
            "schema":"cuteafd.fidelity.reference/2", "checkpoint":"test", "set_sha256":"sealed",
            "expect":{"top1_min":0.9,"kl_max":0.06}})).unwrap();
        // Real sealed V4.1 metadata tests buckets and role counts without external files/network.
        let panel: Value = serde_json::from_str(include_str!("../../../../set/deepseek_v41/v2_20261005/windows.json")).unwrap();
        for raw in panel["windows"].as_array().unwrap() {
            let mut raw = raw.clone(); raw["tokens"] = json!(vec![0u32; raw["roles"].as_array().unwrap().len()]);
            raw["positions"] = json!([]); raw["top_k"] = json!(1);
            let mut w: Window = serde_json::from_value(raw).unwrap();
            w.positions = (w.score_from..w.tokens.len()).map(|pos| CompactPosition {
                pos, next:0, next_lp:0.0, top:vec![Top{id:0,lp:0.0}], tail_lp:f64::NEG_INFINITY }).collect();
            reference.windows.push(w);
        }
        let seal = STANDARD_DECODE.iter().chain(&STANDARD_PREFILL).copied().collect::<Vec<_>>().join("\n");
        assert_eq!(digest(seal.as_bytes()),STANDARD_SPLIT_SHA256);
        let decode = standard_windows(&reference,"decode",true).unwrap();
        let prefill = standard_windows(&reference,"prefill",true).unwrap();
        assert_eq!(decode.iter().map(|w| w.id.as_str()).collect::<Vec<_>>(), STANDARD_DECODE);
        assert_eq!(prefill.iter().map(|w| w.id.as_str()).collect::<Vec<_>>(), STANDARD_PREFILL);
        assert_eq!((decode.len(),prefill.len()),(32,32));
        let ids: BTreeSet<_> = decode.iter().chain(&prefill).map(|w| &w.id).collect();
        assert_eq!(ids.len(),64);
        for half in [&decode,&prefill] {
            let mut blocks = BTreeMap::new();
            for w in half { *blocks.entry(w.block.as_str()).or_insert(0) += 1; }
            assert_eq!(blocks,BTreeMap::from([("A",13),("B",6),("C",5),("D",3),("E",5)]));
        }
        let gen = |half: &[Window]| half.iter().flat_map(|w| w.positions.iter().map(|p| &w.roles[p.pos])).filter(|r| *r == "gen").count();
        assert_eq!((gen(&decode),gen(&prefill)),(8998,8658));
        assert!(prefill.iter().any(|w| w.id == "legacy"));
        reference.windows.reverse();
        assert_eq!(standard_windows(&reference,"decode",true).unwrap().iter().map(|w| w.id.as_str()).collect::<Vec<_>>(), STANDARD_DECODE);
        assert_eq!(standard_windows(&reference,"decode",false).unwrap().len(),64);
        assert!(standard_windows(&reference,"prefill",false).is_err());
        let saved = reference.clone();
        reference.windows[0].block = "A".into();
        assert!(standard_windows(&reference,"decode",true).is_err());
        reference = saved.clone();
        for w in &mut reference.windows {
            if STANDARD_DECODE.contains(&w.id.as_str()) { w.roles.fill("ctx".into()); }
        }
        assert!(standard_windows(&reference,"decode",true).is_err());
        reference = saved.clone();
        for w in &mut reference.windows {
            if STANDARD_DECODE.contains(&w.id.as_str()) && w.block != "C" { w.roles.fill("gen".into()); }
        }
        assert!(standard_windows(&reference,"decode",true).unwrap_err().to_string().contains("more than 15%"));
        reference = saved;
        for w in &mut reference.windows {
            if STANDARD_DECODE.contains(&w.id.as_str()) { w.bucket = "2-8K".into(); }
        }
        assert!(standard_windows(&reference,"decode",true).is_err());
    }

    fn published_window_metadata() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/published-text-windows.json")).unwrap()
    }

    #[test]
    fn published_text_tiers_preserve_sealed_windows_and_generated_balance() {
        let publications = [
            (REVISION, CONFIG, 8998, 8658), (FLASH_REVISION, FLASH_CONFIG, 7169, 6838),
            (GLMF_REVISION, GLMF_CONFIG, 7500, 7604), (QWEN_REVISION, QWEN_CONFIG, 8351, 7902),
            (V4FLASH_REVISION, V4FLASH_CONFIG, 9059, 8342), (GLM_REVISION, GLM_CONFIG, 7497, 7962),
            (V4PRO_REVISION, V4PRO_CONFIG, 9406, 8648), (MIMO_PRO_REVISION, MIMO_PRO_CONFIG, 8786, 7832),
        ];
        let metadata = published_window_metadata();
        assert_eq!(metadata.as_array().unwrap().len(), publications.len());
        for (commit, config, decode_gen, prefill_gen) in publications {
            let panel = metadata.as_array().unwrap().iter().find(|p| p["config"] == config).unwrap();
            assert_eq!(panel["revision"], commit);
            let mut reference: Reference = serde_json::from_value(json!({"name":config, "models":[], "vocab":1,
                "schema":"cuteafd.fidelity.reference/2", "checkpoint":"test", "set_sha256":"sealed",
                "expect":{"top1_min":0.9,"kl_max":0.06}})).unwrap();
            // Published id/block/bucket/scored-row/gen-row metadata, without tokens or logits.
            for row in panel["windows"].as_array().unwrap() {
                let n = row[3].as_u64().unwrap() as usize;
                let gen = row[4].as_u64().unwrap() as usize;
                assert_eq!(n, 512);
                let mut roles = vec!["ctx".into(); n + 1];
                roles[n + 1 - gen..].fill("gen".into());
                reference.windows.push(Window { id:row[0].as_str().unwrap().into(), block:row[1].as_str().unwrap().into(),
                    bucket:row[2].as_str().unwrap().into(), tokens:vec![0; n + 1], roles, score_from:1, top_k:1, media:vec![],
                    positions:(1..=n).map(|pos| CompactPosition {pos,next:0,next_lp:0.0,
                        top:vec![Top{id:0,lp:0.0}],tail_lp:f64::NEG_INFINITY}).collect() });
            }
            assert_eq!(reference.selected_windows(true).unwrap().len(), 64);
            assert_eq!(quick_windows(&reference).unwrap().len(), 8);
            let gen = |windows: &[Window]| windows.iter().flat_map(|w| w.positions.iter().map(|p| &w.roles[p.pos]))
                .filter(|role| *role == "gen").count();
            let decode = standard_windows(&reference,"decode",true).unwrap_or_else(|e| panic!("{config}: {e}"));
            let prefill = standard_windows(&reference,"prefill",true).unwrap_or_else(|e| panic!("{config}: {e}"));
            assert_eq!(decode.iter().map(|w| w.id.as_str()).collect::<Vec<_>>(), STANDARD_DECODE);
            assert_eq!(prefill.iter().map(|w| w.id.as_str()).collect::<Vec<_>>(), STANDARD_PREFILL);
            assert_eq!((gen(&decode), gen(&prefill)), (decode_gen, prefill_gen), "{config}");
            let balance = standard_balance(&reference);
            assert_eq!(balance["decode"]["generated_positions"], decode_gen);
            assert_eq!(balance["prefill"]["generated_positions"], prefill_gen);
            let bucket_difference = ["0-2K", "2-8K", "8-16K"].iter().map(|bucket|
                (balance["decode"]["context_buckets"][bucket].as_i64().unwrap()
                    - balance["prefill"]["context_buckets"][bucket].as_i64().unwrap()).abs()).max().unwrap();
            assert!(bucket_difference <= i64::from(STANDARD_BUCKET_DIFF_MAX), "{config}");
            let ratio = decode_gen.max(prefill_gen) as f64 / decode_gen.min(prefill_gen) as f64;
            assert!(ratio <= STANDARD_GEN_ROW_RATIO_MAX, "{config}: {ratio}");
            if config == MIMO_PRO_CONFIG {
                assert_eq!(decode_gen + prefill_gen, 16618);
                assert!((ratio - 1.1218).abs() < 0.0001);
                assert_eq!(balance["decode"]["context_buckets"], json!({"0-2K":13,"2-8K":10,"8-16K":9}));
                assert_eq!(balance["prefill"]["context_buckets"], json!({"0-2K":16,"2-8K":7,"8-16K":9}));
                assert_eq!(standard_windows(&reference,"decode",false).unwrap().len(),64);
            } else {
                assert!(ratio <= 1.1, "{config}: {ratio}");
                assert!(bucket_difference <= 1, "{config}: {bucket_difference}");
            }
        }
    }

    #[test]
    fn pro_manifest_pins_reject_even_self_consistent_cached_indexes() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("wrldsuksgo2mars--cuteafd-fidelity").join(MIMO_PRO_REVISION);
        for config in [MIMO_PRO_CONFIG, MIMO_PRO_MEDIA_CONFIG] {
            std::fs::create_dir_all(root.join(config)).unwrap();
            let manifest = b"{}";
            std::fs::write(root.join(config).join("manifest.json"), manifest).unwrap();
            let index = json!({"schema":"cuteafd.fidelity.configs/1","configs":[{
                "name":config,"path":format!("{config}/manifest.json"),"sha256":digest(manifest)}]});
            std::fs::write(root.join("configs.json"), serde_json::to_vec(&index).unwrap()).unwrap();
            let error = download(&ureq::Agent::new(),temporary.path(),REPOSITORY,MIMO_PRO_REVISION,config).unwrap_err();
            assert!(error.to_string().contains("pinned manifest checksum differs"), "{config}: {error}");
        }
    }

    #[test]
    #[ignore = "requires published immutable HF commit and network access"]
    fn published_dataset_fetch_roundtrip() {
        let commit = std::env::var("CUTEAFD_FIDELITY_HF_REVISION").unwrap();
        let config = std::env::var("CUTEAFD_FIDELITY_HF_CONFIG").unwrap_or_else(|_| CONFIG.into());
        let expected_hash = std::env::var("CUTEAFD_FIDELITY_HF_MANIFEST_SHA256")
            .unwrap_or_else(|_| "6f2b22f3ed4882765c759b565baab960f7e1563a7fe2be2bbd05540ac4f2f70c".into());
        let cache = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_HF_CACHE").unwrap());
        let (reference, hash, identity) = download(&ureq::AgentBuilder::new()
            .timeout_read(std::time::Duration::from_secs(120)).build(), &cache, REPOSITORY, &commit, &config).unwrap();
        assert_eq!(reference.windows.len(), 64);
        assert_eq!(reference.windows.iter().map(|w| w.positions.len()).sum::<usize>(), 32768);
        assert_eq!(hash, expected_hash);
        assert_eq!(quick_windows(&reference).unwrap().len(),8);
        assert_eq!(standard_windows(&reference,"decode",true).unwrap().len(),32);
        assert_eq!(standard_windows(&reference,"prefill",true).unwrap().len(),32);
        if commit == FLASH_REVISION && config == FLASH_CONFIG {
            assert_eq!((reference.expect.top1_min, reference.expect.kl_max), (0.93, 0.04));
            let tripwires = reference.expect.tripwires.as_ref().unwrap();
            assert_eq!((tripwires.confident_top1_min, tripwires.top3_min), (0.96, 0.97));
            assert_eq!(quick_windows(&reference).unwrap().len(), 8);
        }
        if commit == MIMO_PRO_REVISION && config == MIMO_PRO_CONFIG {
            assert_eq!(reference.checkpoint, "XiaomiMiMo/MiMo-V2.6-Pro-MOPD");
            let metadata = published_window_metadata();
            let panel = metadata.as_array().unwrap().iter().find(|p| p["config"] == config).unwrap();
            for w in &reference.windows {
                let row = panel["windows"].as_array().unwrap().iter().find(|row| row[0] == w.id).unwrap();
                assert_eq!((w.block.as_str(), w.bucket.as_str()), (row[1].as_str().unwrap(), row[2].as_str().unwrap()));
                assert_eq!(w.positions.len() as u64, row[3].as_u64().unwrap());
                assert_eq!(w.positions.iter().filter(|p| w.roles[p.pos] == "gen").count() as u64, row[4].as_u64().unwrap());
            }
        }
        assert_eq!(identity, json!({"repository": REPOSITORY, "revision": commit, "config": config}));
        let again = download(&ureq::Agent::new(), &cache, REPOSITORY, &commit, &config).unwrap();
        assert_eq!(hash, again.1);
        assert_eq!(serde_json::to_value(reference).unwrap(), serde_json::to_value(again.0).unwrap());
    }

    #[test]
    #[ignore = "requires published immutable HF commit and network access"]
    fn published_pro_media_fetch_roundtrip() {
        let cache = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_HF_CACHE").unwrap());
        let (reference, hash, identity) = download(&ureq::AgentBuilder::new()
            .timeout_read(std::time::Duration::from_secs(120)).build(), &cache, REPOSITORY,
            MIMO_PRO_MEDIA_REVISION, MIMO_PRO_MEDIA_CONFIG).unwrap();
        assert_eq!(hash, MIMO_PRO_MEDIA_MANIFEST_SHA256);
        assert_eq!(identity, json!({"repository":REPOSITORY,"revision":MIMO_PRO_MEDIA_REVISION,"config":MIMO_PRO_MEDIA_CONFIG}));
        assert_eq!(reference.checkpoint, "XiaomiMiMo/MiMo-V2.6-Pro-MOPD");
        assert_eq!(reference.selected_windows(true).unwrap().len(), 8);
        assert_eq!(reference.windows.iter().map(|w| w.positions.len()).sum::<usize>(), 4096);
        assert!(reference.windows.iter().all(|w| w.media.len() == 1 && w.media[0].kind == "image"));
    }

    #[test]
    #[ignore = "requires locally prepared sealed dataset"]
    fn prepared_dataset_and_saved_pair() {
        use crate::fidelity::{compare, Run};
        let task = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_TASK_ROOT").unwrap());
        let root = task.join("hf-dataset").join(CONFIG);
        let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let reference = load(&root, &manifest).unwrap();
        assert_eq!(reference.windows.len(), 64);
        assert_eq!(reference.windows.iter().map(|w| w.positions.len()).sum::<usize>(), 32768);
        // Exercise the production compact scorer against a real saved full-row
        // window and the independent Python top1024 validation, not just parsing.
        let original = task.join("calibration/baseline-20261005T092958Z");
        let saved: crate::fidelity::Run = serde_json::from_slice(&std::fs::read(original.join("full-decode.json")).unwrap()).unwrap();
        let expected: crate::fidelity::Run = serde_json::from_slice(&std::fs::read(task.join("top1024-validation/baseline-decode.json")).unwrap()).unwrap();
        for window in &reference.windows {
            let records: std::collections::BTreeMap<_,_> = saved.score.records.iter().filter(|p| p.window == window.id).map(|p| (p.position,p)).collect();
            for p in &window.positions {
                let old = records[&p.pos];
                assert_eq!(old.reference_argmax, p.top[0].id);
                assert_eq!(old.confident, p.top[0].lp.exp() >= 0.5);
                // Old compact NLL came from unsealed logits; the dataset uses
                // the sealed F16 full row (measured max difference 0.01347 nat).
                assert!((old.ref_nll + p.next_lp).abs() < 0.016, "{}:{} NLL {} vs {}", window.id, p.pos, old.ref_nll, -p.next_lp);
            }
        }
        let window = &reference.windows[0];
        let mut scored = crate::reference::Fidelity::from_records(saved.score.records.into_iter().filter(|p| p.window == window.id).collect());
        crate::fidelity_rows::score_compact(reference.vocab, window, &original.join("dump-decode/window-000"), &mut scored).unwrap();
        let expected: Vec<_> = expected.score.records.iter().filter(|p| p.window == window.id).collect();
        assert_eq!(scored.records.len(), expected.len());
        for (got, want) in scored.records.iter().zip(expected) {
            assert!((got.kl - want.kl).abs() < 1e-10);
            assert!((got.nll - want.nll).abs() < 1e-5);
        }
        let mut report = json!({});
        for shape in ["decode", "prefill"] {
            let mut runs = Vec::new();
            for arm in ["baseline", "candidate"] {
                let path = task.join("top1024-validation").join(format!("{arm}-{shape}.json"));
                let mut run: Run = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                for window in &reference.windows {
                    let positions: std::collections::BTreeMap<_,_> = window.positions.iter().map(|p| (p.pos,p)).collect();
                    for record in run.score.records.iter_mut().filter(|p| p.window == window.id) {
                        let p = positions[&record.position];
                        record.ref_nll = -p.next_lp;
                        record.reference_argmax = p.top[0].id;
                        record.confident = p.top[0].lp.exp() >= 0.5;
                    }
                }
                run.score = crate::reference::Fidelity::from_records(run.score.records);
                run.kl_kind = "qualified-top1024-plus-tail".into();
                run.dataset = Some(json!({"repository":REPOSITORY,"config":CONFIG,"revision":"0".repeat(40)}));
                std::fs::write(path, serde_json::to_vec(&run).unwrap()).unwrap();
                runs.push(run);
            }
            let comparison = compare(&runs[1], &runs[0], 0.005, 0.005, 5000, 20260829).unwrap();
            assert!(comparison.pass && comparison.absolute_pass && comparison.tripwires.is_empty());
            report[shape] = serde_json::to_value(comparison).unwrap();
        }
        std::fs::write(task.join("top1024-validation/rust-verdicts.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }

    #[test]
    #[ignore = "requires a locally prepared family dataset and saved paired rows"]
    fn prepared_family_dataset_and_saved_pair() {
        use crate::fidelity::{compare, Run};
        let root = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_CONFIG_DIR").unwrap());
        let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        if std::env::var_os("CUTEAFD_FIDELITY_EXPECT_DRAFT").is_some() {
            assert!(load(&root, &manifest).unwrap_err().to_string().contains("unqualified"));
            return;
        }
        let reference = load(&root, &manifest).unwrap();
        let panel: Value = serde_json::from_slice(&std::fs::read(root.join("windows.json")).unwrap()).unwrap();
        let windows = panel["windows"].as_array().unwrap();
        let scored_positions: usize = windows.iter().map(|w|
            w["tokens"].as_array().unwrap().len() - w["score_from"].as_u64().unwrap() as usize).sum();
        assert_eq!(reference.windows.len(), windows.len());
        assert_eq!(reference.windows.iter().map(|w| w.positions.len()).sum::<usize>(), scored_positions);
        let validation = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_VALIDATION_DIR").unwrap());
        let arms = PathBuf::from(std::env::var("CUTEAFD_FIDELITY_ARMS_DIR").unwrap());
        let expected: Value = serde_json::from_slice(&std::fs::read(validation.join("report.json")).unwrap()).unwrap();
        for shape in ["decode", "prefill"] {
            let mut runs = Vec::new();
            for arm in 0..2 {
                let path = validation.join(format!("baseline-{arm}-compact-{shape}.json"));
                let mut run: Run = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                let window = &reference.windows[0];
                let mut actual = crate::reference::Fidelity::from_records(run.score.records.iter()
                    .filter(|p| p.window == window.id).cloned().collect());
                let dump = arms.join(format!("baseline-{arm}/dump-{shape}/window-000"));
                crate::fidelity_rows::score_compact(reference.vocab, window, &dump, &mut actual).unwrap();
                let rows: Vec<_> = run.score.records.iter().filter(|p| p.window == window.id).collect();
                assert_eq!(actual.records.len(), rows.len());
                for (got, want) in actual.records.iter().zip(rows) {
                    assert!((got.kl - want.kl).abs() < 1e-10);
                    assert!((got.nll - want.nll).abs() < 1e-10);
                }
                run.kl_kind = "qualified-top1024-plus-tail".into();
                run.dataset = Some(json!({"repository":REPOSITORY,"config":manifest["config"],"revision":"0".repeat(40)}));
                runs.push(run);
            }
            let got = compare(&runs[1], &runs[0], 0.005, 0.005, 5000, 20260829).unwrap();
            assert_eq!(got.pass, expected["shapes"][shape]["pass_verdict"].as_bool().unwrap());
            assert!(got.pass && got.absolute_pass && got.tripwires.is_empty());
            for (value, field) in [(got.kl_delta, "kl_delta"), (got.kl_upper95, "kl_upper95"),
                (got.top1_upper95, "top1_upper95")] {
                assert!((value - expected["shapes"][shape][field].as_f64().unwrap()).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn cache_checksums_and_unqualified_panels_fail_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let commit = "a".repeat(40);
        let root = temporary.path().join("owner--repo").join(&commit);
        std::fs::create_dir_all(root.join("config")).unwrap();
        let manifest = b"{}";
        std::fs::write(root.join("config/manifest.json"), manifest).unwrap();
        let index = json!({"schema":"cuteafd.fidelity.configs/1","configs":[{
            "name":"config","path":"config/manifest.json","sha256":"bad checksum"}]});
        std::fs::write(root.join("configs.json"), serde_json::to_vec(&index).unwrap()).unwrap();
        let error = download(&ureq::Agent::new(),temporary.path(),"owner/repo",&commit,"config").unwrap_err();
        assert!(error.to_string().contains("cached dataset checksum differs"));
        assert!(download(&ureq::Agent::new(),temporary.path(),"owner/repo","main","config").is_err());
        let panel = json!({"set_sha256":"set","windows":[]});
        let qualification = json!({"set_sha256":"set","qualifies":false});
        let p = serde_json::to_vec(&panel).unwrap();
        let q = serde_json::to_vec(&qualification).unwrap();
        std::fs::write(root.join("windows.json"), &p).unwrap();
        std::fs::write(root.join("qualification.json"), &q).unwrap();
        let manifest = json!({"schema":"cuteafd.fidelity.dataset/1","top_k":1024,
            "kind":"reference-top1024-plus-tail","set_sha256":"set",
            "windows_sha256":digest(&p),"qualification_sha256":digest(&q)});
        assert!(load(&root,&manifest).unwrap_err().to_string().contains("unqualified"));
        std::fs::write(root.join("windows.json"), b"corrupt").unwrap();
        assert!(load(&root,&manifest).unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn identities_and_tensor_bounds_fail_closed() {
        assert!(revision("0123456789abcdef0123456789abcdef01234567"));
        assert!(!revision("main")); assert!(!revision(&"g".repeat(40)));
        assert!(!component("..")); assert!(!component("a/b"));
        assert!(safe_path(Path::new("/cache"), "../secret").is_err());
        assert!(safe_path(Path::new("/cache"), "/secret").is_err());
        let header = serde_json::to_vec(&json!({"x":{"dtype":"U32","shape":[1],"data_offsets":[0,4]}})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec(); bytes.extend(header); bytes.extend(42u32.to_le_bytes());
        assert_eq!(u32s(tensor(&bytes,"x","U32",&[1]).unwrap()), vec![42]);
        assert!(tensor(&bytes,"x","F32",&[1]).is_err());
        assert!(tensor(&bytes,"x","U32",&[2]).is_err());
        bytes.pop(); assert!(tensor(&bytes,"x","U32",&[1]).is_err());
        assert!(tensor(&[],"x","U32",&[1]).is_err());
    }
}
