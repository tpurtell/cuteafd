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
fn nvidia_drafter_binds_the_head_dim_64_kernels() {
    // nvidia/Muse-Spark-Open-816B-A42B-DFlash2 @ 22e2669e: flat rope_theta, head_dim 64.
    let cfg = read("nvidia.json").unwrap();
    assert_eq!((cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.layers, cfg.block), (8192, 128, 16, 64, 5, 8));
    assert_eq!(cfg.taps, [3, 19, 31, 47, 59]);
    assert_eq!((cfg.vocab, cfg.window, cfg.mask_token), (202048, 2048, 201817));
    assert_eq!(cfg.theta, 500000.0);
    // 128 / 16 heads x block 8 = 64 queries per kv head: the native kernels' bound.
    assert_eq!(cfg.heads / cfg.kv_heads * cfg.block, 64);
}

/// Both candidates are charged on the head GPU from their own configs and
/// pass the target-head check (softcap 20, the target's output multiplier).
#[test]
fn both_drafters_are_charged_against_the_target_head() {
    let target = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
        &cuteafd_loader::plan::testing::muse_spark_config()).unwrap();
    for (name, head_dim) in [("radixark.json", 128), ("nvidia.json", 64)] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/muse_spark/fixtures/drafters").join(name);
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let drafter = cuteafd_loader::placement::families::muse_spark::drafter_inventory(&target, &config, 16, 16, 202_112,
            188, cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8).unwrap_or_else(|error| panic!("{name}: {error:#}"));
        assert_eq!((drafter.head_dim, drafter.block, drafter.sequences), (head_dim, 8, 16), "{name}");
        assert!(drafter.owned > 0 && drafter.scratch > 0, "{name}");
    }
}
