//! Offline model-card ancestry and explicit official precision upgrades.
use crate::fidelity_dataset as dataset;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct Publication {
    pub root: &'static str,
    pub siblings: &'static [&'static str],
    pub revision: &'static str,
    pub config: &'static str,
}

// root_checkpoint.id in the pinned manifests is the logit root; checkpoint is
// only the text-generation source. No third-party or suffix-derived aliases.
pub const PUBLICATIONS: &[Publication] = &[
    Publication { root: "zai-org/GLM-5.3-BF16", siblings: &["zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16"], revision: dataset::GLM_REVISION, config: dataset::GLM_CONFIG },
    Publication { root: "zai-org/GLM-5.3-Flash-BF16", siblings: &["zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash-BF16"], revision: dataset::GLMF_REVISION, config: dataset::GLMF_CONFIG },
    Publication { root: "Qwen/Qwen3.8-Flash-Next", siblings: &["Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next-FP8"], revision: dataset::QWEN_REVISION, config: dataset::QWEN_CONFIG },
    Publication { root: "deepseek-ai/DeepSeek-V4.1-Flash", siblings: &["deepseek-ai/DeepSeek-V4.1-Flash"], revision: dataset::REVISION, config: dataset::CONFIG },
    Publication { root: "deepseek-ai/DeepSeek-V4-Flash-0731", siblings: &["deepseek-ai/DeepSeek-V4-Flash-0731"], revision: dataset::V4FLASH_REVISION, config: dataset::V4FLASH_CONFIG },
    Publication { root: "deepseek-ai/DeepSeek-V4-Pro-0813", siblings: &["deepseek-ai/DeepSeek-V4-Pro-0813"], revision: dataset::V4PRO_REVISION, config: dataset::V4PRO_CONFIG },
    Publication { root: "XiaomiMiMo/MiMo-V2.6-Flash-MOPD", siblings: &["XiaomiMiMo/MiMo-V2.6-Flash-MOPD"], revision: dataset::FLASH_REVISION, config: dataset::FLASH_CONFIG },
    Publication { root: "XiaomiMiMo/MiMo-V2.6-Pro-MOPD", siblings: &["XiaomiMiMo/MiMo-V2.6-Pro-MOPD"], revision: dataset::MIMO_PRO_REVISION, config: dataset::MIMO_PRO_CONFIG },
];

pub fn official(model: &str) -> Option<&'static Publication> {
    PUBLICATIONS.iter().find(|p| p.siblings.contains(&model))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    pub reference_match: String,
    pub reference_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_checkpoint: Option<String>,
    pub resolved_chain: Vec<Vec<String>>,
    pub revision: String,
    pub config: String,
}
impl Resolution {
    fn new(how: &str, p: &Publication, chains: Vec<Vec<String>>) -> Self {
        Self { reference_match: how.into(), reference_root: p.root.into(), text_checkpoint: None,
            resolved_chain: chains, revision: p.revision.into(), config: p.config.into() }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Parents { One(String), Many(Vec<String>) }
#[derive(Deserialize)]
struct Card {
    base_model: Option<Parents>,
    base_model_relation: Option<String>,
}
fn card(snapshot: &Path) -> Result<Option<Card>> {
    let Ok(file) = std::fs::File::open(snapshot.join("README.md")) else { return Ok(None); };
    let mut text = String::new();
    if file.take(1024 * 1024 + 1).read_to_string(&mut text).is_err() { return Ok(None); }
    ensure!(text.len() <= 1024 * 1024, "model card exceeds 1 MiB");
    let mut lines = text.lines();
    if lines.next() != Some("---") { return Ok(None); }
    let mut yaml = String::new();
    for line in lines {
        if line == "---" {
            return serde_yaml::from_str(&yaml).context("malformed model-card YAML").map(Some);
        }
        yaml.push_str(line); yaml.push('\n');
    }
    bail!("malformed model-card YAML: missing closing front-matter delimiter")
}

fn repo_parts(model: &str) -> Option<(&str, &str)> {
    let (org, name) = model.split_once('/')?;
    let valid = |s: &str| !s.is_empty() && s != "." && s != ".."
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b));
    (valid(org) && valid(name)).then_some((org, name))
}

/// Never fetch cards. Prefer the local main ref; an unambiguous lone snapshot
/// also works for revision-only caches. Ambiguous unpinned revisions fail closed.
pub fn local_snapshot(hub: &Path, model: &str) -> Option<PathBuf> {
    let (org, name) = repo_parts(model)?;
    let repo = hub.join(format!("models--{org}--{name}"));
    if let Ok(revision) = std::fs::read_to_string(repo.join("refs/main")) {
        let revision = revision.trim();
        if !revision.is_empty() && revision.bytes().all(|b| b.is_ascii_hexdigit()) {
            let path = repo.join("snapshots").join(revision);
            if path.is_dir() { return Some(path); }
        }
    }
    let mut paths: Vec<_> = std::fs::read_dir(repo.join("snapshots")).ok()?
        .filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    (paths.len() == 1).then(|| paths.remove(0))
}

fn served_snapshot(hub: &Path, model: &str, snapshot: Option<&Path>) -> Option<PathBuf> {
    let Some(snapshot) = snapshot else { return local_snapshot(hub, model); };
    if snapshot.is_dir() { return Some(snapshot.into()); }
    // A remote/container path may differ from the client's mount. Only remap
    // the exact served revision, never substitute the cache's current main.
    if snapshot.parent().and_then(Path::file_name).is_some_and(|name| name == "snapshots") {
        if let Some((org, name)) = repo_parts(model) {
            if let Some(revision) = snapshot.file_name() {
                let local = hub.join(format!("models--{org}--{name}")).join("snapshots").join(revision);
                if local.is_dir() { return Some(local); }
            }
        }
    }
    Some(snapshot.into())
}

fn check_kind(snapshot: &Path) -> Result<()> {
    use cuteafd_loader::plan::{checkpoint::{read_json, Checkpoint}, family};
    let path = snapshot.join("config.json");
    if !path.exists() { return Ok(()); }
    let checkpoint = Checkpoint { snapshot: snapshot.into(), config: read_json(&path)?,
        quantize_config: None, weight_map: Default::default(), tensors: Vec::new(), missing_shards: Vec::new(), shard_bytes: 0 };
    ensure!(!family::registry().iter().any(|f| matches!(f.id(), "dflash2" | "dspark") && f.detect(&checkpoint)),
        "drafter checkpoint");
    Ok(())
}

fn walk(model: &str, snapshot: Option<&Path>, hub: &Path, depth: usize,
    visited: &mut BTreeSet<String>) -> Result<(&'static Publication, Vec<Vec<String>>)> {
    ensure!(depth <= 4, "base_model depth cap of 4 exceeded at {model}");
    ensure!(visited.insert(model.into()), "base_model cycle at {model}");
    let local = served_snapshot(hub, model, snapshot);
    let card = if let Some(path) = &local { check_kind(path)?; card(path)? } else { None };
    if let Some(relation) = card.as_ref().and_then(|c| c.base_model_relation.as_deref()) {
        ensure!(!matches!(relation, "adapter" | "finetune"), "{relation} relation for {model}");
        ensure!(matches!(relation, "quantized" | "merge"), "unsupported base_model relation {relation} for {model}");
    }
    if let Some(p) = official(model).filter(|_| card.as_ref().is_none_or(|c| c.base_model.is_none())) {
        visited.remove(model);
        let mut chain = vec![model.into()];
        if model != p.root { chain.push(p.root.into()); }
        return Ok((p, vec![chain]));
    }
    let Some(parents) = card.and_then(|c| c.base_model) else {
        bail!("base_model {model} has no pinned publication or local parent card");
    };
    let parents = match parents { Parents::One(p) => vec![p], Parents::Many(p) => p };
    ensure!(!parents.is_empty(), "empty base_model list for {model}");
    let mut publication: Option<&Publication> = None;
    let mut chains = Vec::new();
    for parent in parents {
        ensure!(repo_parts(&parent).is_some(), "invalid base_model repo id {parent}");
        let (p, ancestry) = walk(&parent, None, hub, depth + 1, visited)?;
        ensure!(publication.is_none_or(|previous| previous.root == p.root), "disagreeing base_model parents for {model}");
        publication = Some(p);
        for mut chain in ancestry { chain.insert(0, model.into()); chains.push(chain); }
    }
    let publication = publication.unwrap();
    if let Some((revision, config)) = dataset::default_publication(model) {
        ensure!(revision == publication.revision && config == publication.config,
            "base_model and name match disagree for {model}: {} vs {config}", publication.config);
    }
    visited.remove(model);
    Ok((publication, chains))
}

pub fn resolve_in(model: &str, snapshot: Option<&Path>, hub: &Path) -> Result<Resolution> {
    let local = served_snapshot(hub, model, snapshot);
    if let Some(path) = &local {
        ensure!(path.is_dir(), "base_model unknown for {model}: served snapshot {} is not available locally", path.display());
        check_kind(path)?;
    }
    let card = local.as_deref().map(card).transpose()?.flatten();
    // A malformed/present ancestry must not silently fall back to a name.
    if card.as_ref().is_some_and(|c| c.base_model.is_some() || c.base_model_relation.is_some()) {
        let (p, chains) = walk(model, local.as_deref(), hub, 0, &mut BTreeSet::new())?;
        return Ok(Resolution::new("base_model", p, chains));
    }
    let publication = official(model).or_else(|| {
        let (revision, config) = dataset::default_publication(model)?;
        PUBLICATIONS.iter().find(|p| p.revision == revision && p.config == config)
    }).with_context(|| format!("no verified published fidelity config for {model}; missing base_model card"))?;
    let chain = if model == publication.root { vec![model.into()] } else { vec![model.into(), publication.root.into()] };
    Ok(Resolution::new("name", publication, vec![chain]))
}

pub fn resolve(model: &str, snapshot: Option<&Path>) -> Result<Resolution> {
    let hub = std::env::var_os("HF_HUB_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(||
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface")).join("hub")
    });
    resolve_in(model, snapshot, &hub)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn snapshot(hub: &Path, model: &str, yaml: Option<&str>, config: serde_json::Value) -> PathBuf {
        let repo = hub.join(format!("models--{}", model.replace('/', "--")));
        let path = repo.join("snapshots/abcdef");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), "abcdef").unwrap();
        std::fs::write(path.join("config.json"), config.to_string()).unwrap();
        if let Some(yaml) = yaml {
            std::fs::write(path.join("README.md"), format!("---\n{yaml}\n---\n# Model\n")).unwrap();
        }
        path
    }
    fn add(hub: &Path, model: &str, yaml: &str) -> PathBuf {
        snapshot(hub, model, Some(yaml), json!({"architectures":["GlmMoeDsaForCausalLM"]}))
    }

    #[test]
    fn card_rows_and_official_precision_upgrades() {
        let tmp = tempfile::tempdir().unwrap(); let hub = tmp.path();
        let rows = [
            ("nvidia/DeepSeek-V4.1-Flash-NVFP4", "base_model: [deepseek-ai/DeepSeek-V4.1-Flash]", dataset::CONFIG),
            ("wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1", "base_model: deepseek-ai/DeepSeek-V4.1-Flash", dataset::CONFIG),
            ("diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-test", "base_model: deepseek-ai/DeepSeek-V4.1-Flash\nbase_model_relation: quantized", dataset::CONFIG),
            ("wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1", "base_model: zai-org/GLM-5.3-Flash-BF16", dataset::GLMF_CONFIG),
            ("wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1", "base_model: [wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-v1, nvidia/Qwen3.8-Flash-Next-NVFP4]\nbase_model_relation: merge", dataset::QWEN_CONFIG),
        ];
        add(hub, "wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-v1", "base_model: Qwen/Qwen3.8-Flash-Next-FP8");
        add(hub, "nvidia/Qwen3.8-Flash-Next-NVFP4", "base_model: Qwen/Qwen3.8-Flash-Next");
        for (model, yaml, expected) in rows {
            let path = add(hub, model, yaml);
            let r = resolve_in(model, Some(&path), hub).unwrap();
            assert_eq!(r.reference_match, "base_model"); assert_eq!(r.config, expected);
            assert_eq!(r.resolved_chain[0][0], model);
            assert!(r.resolved_chain.iter().all(|c| c.last() == Some(&r.reference_root)));
            if model.contains("PLE") {
                assert_eq!(r.resolved_chain.len(), 2);
                assert!(r.resolved_chain[0].contains(&"Qwen/Qwen3.8-Flash-Next-FP8".into()));
            }
        }
        for p in PUBLICATIONS {
            for sibling in p.siblings {
                let r = resolve_in(sibling, None, hub).unwrap();
                assert_eq!(r.reference_root, p.root); assert_eq!(r.config, p.config);
                let quant = format!("test/quant-{}", sibling.replace('/', "-"));
                let path = add(hub, &quant, &format!("base_model: {sibling}"));
                let r = resolve_in(&quant, Some(&path), hub).unwrap();
                assert_eq!(r.reference_root, p.root); assert_eq!(r.resolved_chain[0][1], *sibling);
            }
        }
        for model in ["zai-org/GLM-5.3-BF16-Instruct", "someone/GLM-5.3-BF16", "Qwen/Qwen3.8-Flash-Next-FP8-Dynamic"] {
            assert!(resolve_in(model, None, hub).is_err(), "{model}");
        }
        let path = add(hub, "someone/GLM-5.3-BF16", "base_model: zai-org/GLM-5.3");
        assert_eq!(resolve_in("someone/GLM-5.3-BF16", Some(&path), hub).unwrap().reference_root, "zai-org/GLM-5.3-BF16");
    }

    #[test]
    fn drafters_relations_and_conflicting_names_fail_closed() {
        let tmp = tempfile::tempdir().unwrap(); let hub = tmp.path();
        for (model, config) in [
            ("incoai/GLM-5.3-Flash-DFlash2", json!({"architectures":["DFlash2DraftModel"]})),
            ("RedHatAI/GLM-5.3-Flash-speculator.dspark-preview", json!({"architectures":["DSparkDraftModel"]})),
            ("test/innocent-name", json!({"speculators_model_type":"dspark"})),
        ] {
            let path = snapshot(hub, model, Some("base_model: [zai-org/GLM-5.3-Flash]"), config);
            assert!(resolve_in(model, Some(&path), hub).unwrap_err().to_string().contains("drafter checkpoint"));
        }
        let path = add(hub, "nvidia/GLM-5.3-NVFP4", "base_model: zai-org/GLM-5.3");
        let remote = Path::new("/container/hub/models--nvidia--GLM-5.3-NVFP4/snapshots/abcdef");
        assert_eq!(resolve_in("nvidia/GLM-5.3-NVFP4", Some(remote), hub).unwrap().reference_match, "base_model");
        let missing_revision = remote.with_file_name("123456");
        assert!(resolve_in("nvidia/GLM-5.3-NVFP4", Some(&missing_revision), hub)
            .unwrap_err().to_string().contains("base_model unknown"));
        assert!(path.exists());
        for relation in ["finetune", "adapter", "unknown"] {
            let path = add(hub, "test/quant", &format!("base_model: zai-org/GLM-5.3\nbase_model_relation: {relation}"));
            assert!(resolve_in("test/quant", Some(&path), hub).unwrap_err().to_string().contains(relation));
        }
        let path = add(hub, "nvidia/GLM-5.3-NVFP4", "base_model: Qwen/Qwen3.8-Flash-Next");
        assert!(resolve_in("nvidia/GLM-5.3-NVFP4", Some(&path), hub).unwrap_err().to_string().contains("disagree"));
        let path = add(hub, "zai-org/GLM-5.3", "base_model: Qwen/Qwen3.8-Flash-Next");
        assert!(resolve_in("zai-org/GLM-5.3", Some(&path), hub).unwrap_err().to_string().contains("disagree"));
    }

    #[test]
    fn cycles_depth_disagreeing_parents_and_bad_cards() {
        let tmp = tempfile::tempdir().unwrap(); let hub = tmp.path();
        add(hub, "test/a", "base_model: test/b"); add(hub, "test/b", "base_model: test/a");
        assert!(resolve_in("test/a", None, hub).unwrap_err().to_string().contains("cycle"));
        for n in 0..4 { add(hub, &format!("test/d{n}"), &format!("base_model: test/d{}", n+1)); }
        add(hub, "test/d4", "base_model: zai-org/GLM-5.3");
        assert!(resolve_in("test/d0", None, hub).unwrap_err().to_string().contains("depth cap"));
        assert!(resolve_in("test/d1", None, hub).is_ok());
        let path = add(hub, "test/merge", "base_model: [zai-org/GLM-5.3, Qwen/Qwen3.8-Flash-Next]");
        assert!(resolve_in("test/merge", Some(&path), hub).unwrap_err().to_string().contains("disagreeing"));
        let path = snapshot(hub, "nvidia/GLM-5.3-NVFP4", None, json!({}));
        assert_eq!(resolve_in("nvidia/GLM-5.3-NVFP4", Some(&path), hub).unwrap().reference_match, "name");
        for yaml in ["base_model: [broken", "base_model: 12", "base_model: {oops: map}", "base_model: []", "base_model: test/unknown", "base_model: ../../escape"] {
            let path = add(hub, "nvidia/GLM-5.3-NVFP4", yaml);
            assert!(resolve_in("nvidia/GLM-5.3-NVFP4", Some(&path), hub).is_err(), "{yaml}");
        }
        let path = add(hub, "nvidia/GLM-5.3-NVFP4", "license: mit");
        std::fs::write(path.join("README.md"), "---\nbase_model: zai-org/GLM-5.3\n").unwrap();
        assert!(resolve_in("nvidia/GLM-5.3-NVFP4", Some(&path), hub).is_err());
    }

    #[test]
    fn provenance_survives_run_and_panel_serialization() {
        let p = &PUBLICATIONS[0];
        let r = Resolution::new("base_model", p, vec![vec!["test/quant".into(), p.siblings[0].into(), p.root.into()]]);
        let mut run: crate::fidelity::Run = serde_json::from_value(json!({
            "schema":"cuteafd.fidelity.run/2", "arm":"test", "checkpoint":"test/quant", "set_sha256":"x",
            "reference_sha256":"x", "tier":"quick", "path_shape":"decode-shaped", "kl_kind":"compact",
            "verify_rows":null, "engine":"test", "settings":{}, "seconds":0,
            "score":crate::reference::Fidelity::from_records(Vec::new()), "floor_top1":0, "floor_kl":0
        })).unwrap();
        run.reference_selection = Some(r);
        let saved = serde_json::to_value(&run).unwrap();
        assert_eq!(saved["reference_match"], "base_model");
        assert_eq!(saved["resolved_chain"][0][1], "zai-org/GLM-5.3");
        let read: crate::fidelity::Run = serde_json::from_value(saved).unwrap();
        let panel = crate::panels::fidelity::record(&read, None);
        assert_eq!(panel["reference_root"], p.root);
        assert_eq!(panel["reference_match"], "base_model");
    }

    #[test]
    #[ignore = "CPU live cache audit: requires this host's release inventory and cached manifests"]
    fn live_release_models_and_precision_sanity() {
        let hub = Path::new("/mnt/sparknest/hf-home/hub");
        let inventory: serde_json::Value = serde_json::from_slice(&std::fs::read(
            "/home/tj/.cache/cuteafd/builds/release-v2-rc2/kit/models.json").unwrap()).unwrap();
        for (label, entry) in inventory.as_object().unwrap() {
            let model = entry["settings"]["MODEL_ID"].as_str().unwrap();
            let snapshot = entry["settings"]["MODEL_REVISION"].as_str().map(|revision|
                hub.join(format!("models--{}", model.replace('/', "--"))).join("snapshots").join(revision))
                .or_else(|| local_snapshot(hub, model));
            let r = resolve_in(model, snapshot.as_deref(), hub).unwrap();
            println!("{label}: {model} -> {} -> {} ({}) {:?}", r.reference_root, r.config, r.reference_match, r.resolved_chain);
            if let Some((revision, config)) = dataset::default_publication(model) {
                assert_eq!((&r.revision, &r.config), (&revision.to_owned(), &config.to_owned()));
            }
        }
        for model in ["nvidia/DeepSeek-V4.1-Flash-NVFP4", "wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1"] {
            assert_eq!(resolve_in(model, None, hub).unwrap().config, dataset::CONFIG);
        }
        for p in PUBLICATIONS {
            let hf = hub.join("datasets--wrldsuksgo2mars--cuteafd-fidelity/snapshots").join(p.revision).join(p.config).join("manifest.json");
            let local = Path::new("/home/tj/.cache/cuteafd/fidelity/wrldsuksgo2mars--cuteafd-fidelity").join(p.revision).join(p.config).join("manifest.json");
            let bytes = std::fs::read(&hf).or_else(|_| std::fs::read(&local)).unwrap();
            let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(manifest["root_checkpoint"]["id"], p.root, "{} logit root", p.config);
            println!("AUDIT {}: text={}, logits={}", p.config, manifest["checkpoint"], p.root);
            let Some(root) = local_snapshot(hub, p.root) else { continue; };
            let root: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
            let root_text = root.get("text_config").unwrap_or(&root);
            assert_eq!(root_text["vocab_size"], manifest["vocab"]);
            for sibling in p.siblings {
                let Some(path) = local_snapshot(hub, sibling) else { continue; };
                let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
                let text = config.get("text_config").unwrap_or(&config);
                assert_eq!(config["architectures"], root["architectures"]);
                for key in ["hidden_size", "num_hidden_layers", "vocab_size"] { assert_eq!(text[key], root_text[key], "{sibling}/{key}"); }
            }
        }
    }
}
