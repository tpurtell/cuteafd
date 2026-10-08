//! Resolve checkpoint-full context before constructing API or engine limits.
use anyhow::{ensure, Context, Result};
use std::path::Path;

pub(crate) fn checkpoint_context(snapshot: &Path, manifest: &Path, family: &str, requested: usize) -> Result<usize> {
    let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    let checkpoint: usize = cuteafd_loader::serving_capacity::checkpoint_context_limit(&config)?
        .with_context(|| format!("{family} checkpoint lacks max_position_embeddings"))?.try_into()?;
    let compiled = if family == "mimo_v2" { None } else {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
        Some(usize::try_from(value["capacities"]["max_context"].as_u64()
            .filter(|&n| n > 0).context("program manifest lacks positive capacities.max_context")?)?)
    };
    let supported = compiled.map_or(checkpoint, |n| n.min(checkpoint));
    if requested == 0 {
        if supported < checkpoint {
            tracing::warn!(family, checkpoint_context = checkpoint, compiled_context = supported,
                "checkpoint context exceeds compiled support; serving the compiled context extent");
        }
        Ok(supported)
    } else {
        ensure!(requested <= supported,
            "{family}: requested context {requested} exceeds checkpoint/compiled support {supported} (checkpoint {checkpoint}); lower MAX_CONTEXT_TOKENS or export wider programs");
        Ok(requested)
    }
}

pub(crate) fn pool_context(family: &str, context: usize, automatic: bool, pool: usize, unit: usize) -> Result<usize> {
    if context <= pool { return Ok(context); }
    ensure!(automatic, "{family}: explicit context {context} > admitted pool supports {pool}; set POOL_TOKENS/MAX_CONTEXT_TOKENS to change");
    let supported = pool.saturating_sub(unit.max(64));
    ensure!(supported > 0, "{family}: admitted pool {pool} cannot hold one request plus its safety margin");
    tracing::warn!(family, checkpoint_context = context, admitted_pool = pool, max_context = supported,
        "checkpoint context exceeds admitted pool; serving reduced max_context; set POOL_TOKENS/MAX_CONTEXT_TOKENS to change");
    Ok(supported)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_zero_resolves_checkpoint_context_and_explicit_wins() {
        for (family, model_type, tokens) in [("glm5", "glm_moe_dsa", 202752),
            ("glm5_flash", "glm_moe_dsa", 131072), ("qwen4", "qwen3_next", 262144),
            ("deepseek_v4", "deepseek_v4", 1048576), ("mimo_v2", "mimo_v2", 1048576)] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.json"), serde_json::json!({
                "model_type": model_type, "max_position_embeddings": tokens }).to_string()).unwrap();
            let manifest = dir.path().join("PROGRAMS.json");
            std::fs::write(&manifest, r#"{"capacities":{"max_context":2097152}}"#).unwrap();
            assert_eq!(checkpoint_context(dir.path(), &manifest, family, 0).unwrap(), tokens);
            assert_eq!(checkpoint_context(dir.path(), &manifest, family, 32768).unwrap(), 32768);
            assert!(checkpoint_context(dir.path(), &manifest, family, tokens + 1).is_err());
        }
    }

    #[test]
    fn defaults_clamp_to_compiled_support_and_pool_with_margin_but_explicit_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"text_config":{"max_position_embeddings":1048576}}"#).unwrap();
        let manifest = dir.path().join("PROGRAMS.json");
        std::fs::write(&manifest, r#"{"capacities":{"max_context":131072}}"#).unwrap();
        assert_eq!(checkpoint_context(dir.path(), &manifest, "qwen4", 0).unwrap(), 131072);
        assert!(checkpoint_context(dir.path(), &manifest, "qwen4", 131073).is_err());
        assert_eq!(pool_context("mimo_v2", 1048576, true, 962560, 64).unwrap(), 962496);
        assert!(pool_context("mimo_v2", 1048576, false, 962560, 64).is_err());
        assert_eq!(pool_context("glm5", 32768, false, 65536, 256).unwrap(), 32768);
        assert!(pool_context("glm5", 131072, true, 64, 256).is_err());
    }
}
