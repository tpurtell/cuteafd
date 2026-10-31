//! The two candidate DFlash2 drafters' own configs (pinned revisions) against
//! the shared DFlash2 runtime: RadixArk fits the existing 128-wide-head
//! kernels; NVIDIA's 64-wide heads bind the native `_d64` drafter attention.
use crate::shared::speculation::dflash::DflashConfig;

fn read(name: &str) -> anyhow::Result<DflashConfig> {
    let dir = tempfile::tempdir()?;
    std::fs::copy(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/families/muse_spark/fixtures/drafters").join(name), dir.path().join("config.json"))?;
    DflashConfig::read(dir.path())
}

#[test]
fn radixark_drafter_fits_the_shared_dflash2_kernels() {
    // RadixArk/Muse-Spark-Open-DFlash2 @ 46c90f41.
    let cfg = read("radixark.json").unwrap();
    assert_eq!((cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.layers, cfg.block), (8192, 64, 8, 128, 5, 8));
    assert_eq!(cfg.taps, [1, 16, 30, 44, 59]);
    assert_eq!((cfg.vocab, cfg.window, cfg.mask_token), (202048, 2048, 200002));
    assert_eq!(cfg.theta, 500000.0);
}

#[test]
fn nvidia_drafter_names_the_missing_head_dim_64_kernels() {
    // nvidia/Muse-Spark-Open-816B-A42B-DFlash2 @ 22e2669e: flat rope_theta, head_dim 64.
    let error = read("nvidia.json").unwrap_err().to_string();
    assert!(error.contains("128-wide heads"), "{error}");
}
