//! CPU-only G1 runner for scripts/qualify/media/preprocess-goldens.py.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::media::{EncoderId, ImageFamily, ImageProcessor, ProcessorConfig};
use sha2::{Digest, Sha256};
use std::path::Path;
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: media-preprocess MANIFEST.json")?;
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let root = Path::new(&path).parent().context("manifest directory")?;
    let mut failures = 0;
    let cases = manifest["cases"].as_array().context("cases")?;
    for case in cases {
        let family = match case["family"].as_str().context("family")? {
            "mimo" => ImageFamily::Mimo,
            "qwen" => ImageFamily::Qwen,
            "glm_flash" => ImageFamily::GlmFlash,
            _ => anyhow::bail!("unknown processor"),
        };
        let config = if family == ImageFamily::GlmFlash && manifest.get("glm_processor").is_some() {
            let source = &manifest["glm_processor"];
            let snapshot = Path::new(source["snapshot"].as_str().context("GLM snapshot")?);
            ensure!(digest(&std::fs::read(snapshot.join("processor_config.json"))?)
                == source["processor_sha256"].as_str().context("GLM processor hash")?,
                "GLM processor changed since fixture generation");
            ProcessorConfig::from_snapshot(snapshot, family)?
        } else {
            ProcessorConfig::for_family(family)
        }.with_detail(case["low"].as_bool().unwrap_or(false));
        let bytes = std::fs::read(root.join(case["file"].as_str().context("file")?))?;
        let image = config.prepare(&bytes, EncoderId([0; 32]))?;
        let rgb = digest(&image.rgb8);
        let patches = image.patches(&config)?;
        let patch_bytes: Vec<u8> = patches.iter().flat_map(|v| v.to_le_bytes()).collect();
        let patch = digest(&patch_bytes);
        let grid = serde_json::to_value(image.grid)?;
        if rgb != case["rgb_sha256"].as_str().context("RGB digest")?
            || patch != case["patch_sha256"].as_str().context("patch digest")?
            || grid != case["grid"]
        {
            failures += 1;
            eprintln!(
                "FAIL {} {} low={}: grid={grid} rgb={rgb} patches={patch}",
                case["family"], case["file"], case["low"]
            );
        }
    }
    println!(
        "G1: {} cases, {} byte-exact, {failures} failures",
        cases.len(),
        cases.len() - failures
    );
    ensure!(
        failures == 0,
        "preprocessing differs from pinned PIL reference"
    );
    Ok(())
}
