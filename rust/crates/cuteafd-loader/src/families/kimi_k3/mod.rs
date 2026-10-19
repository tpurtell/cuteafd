//! Kimi K3 (`kimi_k3` wrapping `text_config.model_type=kimi_linear`) checkpoint readers.
pub mod config;
pub mod kda;
pub mod mla;
pub mod reference;
pub mod w8;

/// Validate sealed role artifacts at runtime, never during checkpoint planning.
pub fn validate_expert_inventory(path: &std::path::Path, root: &std::path::Path) -> anyhow::Result<()> {
    use crate::plan::checkpoint::read_json;
    use anyhow::{ensure, Context};
    let programs = read_json(path)?;
    ensure!(programs["schema"] == 1 && programs["programs"].is_array(), "invalid PROGRAMS.json inventory");
    let package = read_json(&root.join("manifest.json"))?;
    ensure!(package["schema"] == "cuteafd.exl3-package.v1" && package["geometry"] == "kimik3",
        "Kimi K3 EXL3 package schema/geometry mismatch");
    let revision = package["sparkinfer_revision"].as_str().filter(|s| s.len() == 40)
        .context("Kimi K3 package lacks source revision")?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json")))?;
    ensure!(lock["revision"] == revision, "Kimi K3 package/source revision mismatch");
    let (layouts, output) = match package["role"].as_str() {
        Some("spark") => ((0..6).map(|r| (format!("tp6-rank{r}"), 512)).collect::<Vec<_>>(), "bf16"),
        Some("coordinator") => {
            ensure!(programs["sparkinfer_revision"] == revision, "Kimi K3 program/package revision mismatch");
            let inventory = programs["programs"].as_array().unwrap();
            for name in ["kimik3_router_scores", "kimik3_expert_input_quant", "kimik3_latent_down",
                "kimik3_latent_up", "kimik3_rmsnorm_h3584"] {
                let entries = inventory.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
                ensure!(entries.len() == 1 && entries[0]["object_sha256"].as_str().is_some_and(|s| s.len() == 64),
                    "Kimi K3 program inventory requires one sealed {name}");
                let geometry = &entries[0]["geometry"];
                match name {
                    "kimik3_router_scores" => ensure!(geometry["hidden"] == 7168 && geometry["experts"] == 896
                        && geometry["top_k"] == 16, "Kimi router geometry mismatch"),
                    "kimik3_latent_down" => ensure!(geometry["width"] == 7168 && geometry["outputs"] == 3584,
                        "Kimi latent down geometry mismatch"),
                    "kimik3_latent_up" => ensure!(geometry["width"] == 3584 && geometry["outputs"] == 7168,
                        "Kimi latent up geometry mismatch"),
                    _ => ensure!(geometry["hidden"] == 3584, "Kimi latent input geometry mismatch"),
                }
            }
            (vec![("rtx-tp1".into(), 3072), ("rtx-tp2".into(), 1536)], "fp32")
        }
        _ => anyhow::bail!("Kimi K3 EXL3 package must declare coordinator or spark role"),
    };
    let variants = package["variants"].as_array().context("Kimi K3 package lacks variants")?;
    for (layout, width) in layouts {
        for capacity in cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::KIMI_K3).capacities(4096)? {
            let directory = format!("{layout}/m{capacity}");
            let entries = variants.iter().filter(|v| v["directory"] == directory).collect::<Vec<_>>();
            ensure!(entries.len() == 1, "Kimi K3 package requires one {directory}");
            let meta = read_json(&root.join(&directory).join("v41_exl3.json"))?;
            for entry in [entries[0], &meta] {
                ensure!(entry["intermediate"] == width && entry["experts"] == 896 && entry["top_k"] == 16
                    && entry["capacity"] == capacity && entry["bits"] == serde_json::json!([2, 3])
                    && entry["output_dtype"] == output, "Kimi K3 package geometry mismatch: {directory}");
            }
            ensure!(meta["sparkinfer_revision"] == revision && meta["hidden"] == 3584
                && meta["activation"] == "situ" && meta["descriptor_bits"] == 10
                && meta["descriptor_rows"].as_u64().is_none_or(|n| n == 3)
                && meta["native_info_version"].as_u64().is_none_or(|n| n == 2)
                && meta["decode_schedule"].is_null(), "Kimi K3 EXL3 native contract mismatch: {directory}");
            for artifact in ["libcuteafd_exl3.so", "routes/libv41_exl3_routes.so", "trellis_lut.bin"] {
                ensure!(root.join(&directory).join(artifact).is_file(),
                    "Kimi K3 package missing {directory}/{artifact}");
            }
        }
    }
    Ok(())
}
