//! Published placement-input provenance and checksum validation.
use serde_json::Value;
use std::fs;

/// Live V4 planner inputs are compiled by compare-sm-exports.py, not fixtures.
/// A missing inventory or a changed fork pin must fail the requested live gate.
pub fn v4_placement_inputs() -> anyhow::Result<std::path::PathBuf> {
    let root = std::env::var_os("CUTEAFD_V4_PLACEMENT_INPUTS").map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home)
            .join(".cache/cuteafd/builds/v4-placement-inputs/current")))
        .ok_or_else(|| anyhow::anyhow!("set HOME or CUTEAFD_V4_PLACEMENT_INPUTS"))?;
    validate_v4_placement_inputs(&root)?;
    Ok(root)
}

/// Hash each listed repo file (sorted, deduplicated) by its repo-relative path.
fn hashed_files<'a>(root: &std::path::Path, paths: impl IntoIterator<Item = &'a str>)
    -> anyhow::Result<serde_json::Map<String, Value>> {
    use anyhow::Context;
    use sha2::{Digest, Sha256};
    let mut files = serde_json::Map::new();
    for path in paths.into_iter().collect::<std::collections::BTreeSet<_>>() {
        let bytes = fs::read(root.join(path)).with_context(|| format!("hashing placement source {path}"))?;
        files.insert(path.into(), Value::String(format!("{:x}", Sha256::digest(bytes))));
    }
    Ok(files)
}

fn placement_sources(root: &std::path::Path) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&fs::read(root.join("scripts/build/placement-sources.json"))?)?)
}

/// Compiled tier: every file whose bytes can change a compiled program, as
/// scripts/build/placement-sources.json lists them (a default-tier script test
/// checks the list covers the exporters' full import closure), plus the list
/// itself and the derived job specs. Keep identical to
/// export-placement-inputs.py::compiled_hashes.
fn compiled_source_hashes(root: &std::path::Path, kind: &str) -> anyhow::Result<Value> {
    let sources = placement_sources(root)?;
    let listed = sources["compiled"][kind].as_array()
        .ok_or_else(|| anyhow::anyhow!("placement-sources.json has no compiled.{kind} list"))?;
    let mut paths = listed.iter().map(|path| path.as_str()
        .ok_or_else(|| anyhow::anyhow!("non-string compiled source"))).collect::<anyhow::Result<Vec<_>>>()?;
    paths.extend(["scripts/build/placement-sources.json", "scripts/build/placement-jobs.json"]);
    Ok(Value::Object(hashed_files(root, paths)?))
}

/// Selection tier: inputs that select what installed planner tests plan and
/// admit (live example configs and their revisions, measured Spark memory)
/// without changing compiled bytes. Keep identical to
/// export-placement-inputs.py::selection_hashes.
fn selection_source_hashes(root: &std::path::Path, kind: &str) -> anyhow::Result<Value> {
    use sha2::{Digest, Sha256};
    let sources = placement_sources(root)?;
    let spec = &sources["selection"][kind];
    let listed = spec["files"].as_array()
        .ok_or_else(|| anyhow::anyhow!("placement-sources.json has no selection.{kind} files"))?;
    let mut paths = listed.iter().map(|path| path.as_str().map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("non-string selection source"))).collect::<anyhow::Result<Vec<_>>>()?;
    let mut live = Vec::new();
    if spec["live_configs"].as_bool().ok_or_else(|| anyhow::anyhow!("selection.{kind}.live_configs missing"))? {
        let matrix: Value = serde_json::from_slice(&fs::read(root.join("benchmarks/current-matrix.json"))?)?;
        let retired = matrix["retired"].as_array().ok_or_else(|| anyhow::anyhow!("missing retired config list"))?
            .iter().filter_map(|entry| entry["config"].as_str()).collect::<std::collections::BTreeSet<_>>();
        for entry in fs::read_dir(root.join("examples/configs"))? {
            let path = entry?.path();
            if path.extension().is_some_and(|extension| extension == "config") {
                let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
                if !retired.contains(relative.as_str()) {
                    live.push(relative);
                }
            }
        }
        live.sort();
        paths.extend(live.iter().cloned());
    }
    let mut files = hashed_files(root, paths.iter().map(String::as_str))?;
    if spec["live_configs"].as_bool() == Some(true) {
        files.insert("installed_live_configs".into(),
            Value::String(format!("{:x}", Sha256::digest(live.join("\n").as_bytes()))));
    }
    Ok(Value::Object(files))
}

/// Both tiers must equal the checkout's: a stale compiled tier means stale
/// programs; a stale selection tier means the publication was taken for a
/// different installed inventory or measured Spark memory.
fn validate_sources(metadata: &Value, root: &std::path::Path, kind: &str, stale: &str) -> anyhow::Result<()> {
    let compiled = compiled_source_hashes(root, kind)?;
    anyhow::ensure!(metadata["engine_sources_sha256"] == compiled,
        "{stale}: export source fingerprints differ ({})", differing(&metadata["engine_sources_sha256"], &compiled));
    let selection = selection_source_hashes(root, kind)?;
    anyhow::ensure!(metadata["selection_sources_sha256"] == selection,
        "{stale}: selection source fingerprints differ ({}); re-attest with export-placement-inputs.py --reattest-from",
        differing(&metadata["selection_sources_sha256"], &selection));
    Ok(())
}

fn differing(recorded: &Value, expected: &Value) -> String {
    let (Some(recorded), Some(expected)) = (recorded.as_object(), expected.as_object()) else {
        return "fingerprint missing".into();
    };
    let keys = recorded.keys().chain(expected.keys()).collect::<std::collections::BTreeSet<_>>();
    let changed = keys.into_iter().filter(|key| recorded.get(*key) != expected.get(*key)).collect::<Vec<_>>();
    let shown = changed.iter().take(6).map(|key| key.as_str()).collect::<Vec<_>>().join(", ");
    if changed.len() > 6 { format!("{shown} and {} more", changed.len() - 6) } else { shown }
}

/// A verified publication; reads recheck the bytes actually returned to the caller.
#[derive(Debug)]
pub struct PublishedPlacementInputs {
    root: std::path::PathBuf,
    files: std::collections::BTreeMap<String, String>,
    fork_revision: String,
    fork_tree_sha256: String,
}
impl PublishedPlacementInputs {
    pub fn checksums(&self) -> &std::collections::BTreeMap<String, String> { &self.files }
    pub fn fork_revision(&self) -> &str { &self.fork_revision }
    pub fn fork_tree_sha256(&self) -> &str { &self.fork_tree_sha256 }

    pub fn read(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        use anyhow::Context;
        use sha2::{Digest, Sha256};
        let expected = self.files.get(name).context("file absent from placement provenance")?;
        let bytes = fs::read(self.root.join(name)).with_context(|| format!("reading placement input {name}"))?;
        anyhow::ensure!(*expected == format!("{:x}", Sha256::digest(&bytes)),
            "placement input checksum mismatch: {name}");
        Ok(bytes)
    }
}

/// Installed planners consume a published CPU export, never an ambient WIP slot.
/// Explicit manifest overrides receive the same pin/source/checksum validation.
pub fn validate_placement_inputs(manifest: &std::path::Path) -> anyhow::Result<PublishedPlacementInputs> {
    use anyhow::Context;
    use sha2::{Digest, Sha256};
    let regenerate = "regenerate with scripts/build/cuteafd-dev.sh cpu -- python3 scripts/build/export-placement-inputs.py --output <NVMe>/placement-inputs/<sha> --staging <NVMe>/placement-inputs/staging-<sha>";
    (|| -> anyhow::Result<PublishedPlacementInputs> {
        anyhow::ensure!(manifest.is_file(), "required PROGRAMS.json missing: {}", manifest.display());
        let share = manifest.parent().context("PROGRAMS.json parent")?;
        anyhow::ensure!(share.file_name().is_some_and(|name| name == "share"),
            "placement manifest must use share/PROGRAMS.json");
        let root = share.parent().context("placement publication root")?;
        let metadata: Value = serde_json::from_slice(&fs::read(root.join("INPUTS.json"))
            .with_context(|| format!("missing placement provenance {}", root.join("INPUTS.json").display()))?)?;
        let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../../third_party/sparkinfer.lock.json")))?;
        anyhow::ensure!(metadata["schema"] == 1 && metadata["engine_commit"].as_str().is_some_and(|sha| sha.len() == 40),
            "invalid placement input provenance");
        anyhow::ensure!(metadata["sparkinfer_revision"] == lock["revision"]
            && metadata["source_tree_sha256"] == lock["source_tree_sha256"], "stale placement inputs: fork pin mismatch");
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        validate_sources(&metadata, &source, "all", "stale placement inputs")?;
        let exports = metadata["exports"].as_array().context("missing placement worker provenance")?;
        anyhow::ensure!(!exports.is_empty() && exports.iter().all(|export|
            export["sparkinfer_revision"] == lock["revision"]
            && export["source_tree_sha256"] == lock["source_tree_sha256"]
            && export["cuda_initialized"] == false), "placement worker provenance mismatch");
        let files = metadata["files"].as_object().context("missing placement file inventory")?;
        let name = "share/PROGRAMS.json";
        anyhow::ensure!(files.contains_key(name), "PROGRAMS.json missing from placement provenance");
        let mut checksums = std::collections::BTreeMap::new();
        for (path, expected) in files {
            anyhow::ensure!(!path.is_empty() && std::path::Path::new(path).components().all(|part|
                matches!(part, std::path::Component::Normal(_))), "invalid placement input path {path}");
            let actual = format!("{:x}", Sha256::digest(fs::read(root.join(path))
                .with_context(|| format!("missing placement input {path}"))?));
            anyhow::ensure!(expected.as_str() == Some(&actual), "placement input checksum mismatch: {path}");
            checksums.insert(path.clone(), actual);
        }
        Ok(PublishedPlacementInputs { root: root.into(), files: checksums,
            fork_revision: metadata["sparkinfer_revision"].as_str().context("fork revision")?.into(),
            fork_tree_sha256: metadata["source_tree_sha256"].as_str().context("fork tree hash")?.into() })
    })().with_context(|| format!("placement inputs at {}; {regenerate}", manifest.display()))
}

fn validate_v4_placement_sources(metadata: &Value, root: &std::path::Path) -> anyhow::Result<()> {
    validate_sources(metadata, root, "v4", "stale V4 placement inputs")
        .map_err(|error| anyhow::anyhow!("{error:#}; regenerate inputs"))
}

fn validate_v4_placement_inputs(root: &std::path::Path) -> anyhow::Result<()> {
    use anyhow::Context;
    use sha2::{Digest, Sha256};
    let metadata: Value = serde_json::from_slice(&fs::read(root.join("INPUTS.json"))
        .with_context(|| format!("missing live V4 inputs at {}; regenerate with scripts/build/compare-sm-exports.py --v4-placement-inputs", root.display()))?)?;
    let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json")))?;
    anyhow::ensure!(metadata["schema"] == 1 && metadata["engine_commit"].as_str().is_some_and(|s| s.len() == 40),
        "invalid V4 input provenance");
    anyhow::ensure!(metadata["sparkinfer_revision"] == lock["revision"]
        && metadata["source_tree_sha256"] == lock["source_tree_sha256"],
        "stale V4 placement inputs: exported fork {} differs from tree {}; regenerate inputs",
        metadata["sparkinfer_revision"], lock["revision"]);
    validate_v4_placement_sources(&metadata, &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."))?;
    let exports = metadata["exports"].as_array().ok_or_else(|| anyhow::anyhow!("missing V4 worker provenance"))?;
    anyhow::ensure!(exports.len() == 3 && exports.iter().all(|export|
        export["sparkinfer_revision"] == lock["revision"]
        && export["source_tree_sha256"] == lock["source_tree_sha256"]
        && export["cuda_initialized"] == false), "V4 worker provenance mismatch");
    let files = metadata["files"].as_object().ok_or_else(|| anyhow::anyhow!("V4 input file inventory missing"))?;
    for context in [131072, 1048576] {
        for model in ["flash", "pro"] {
            for rtx in [1, 2] {
                let manifest = format!("{model}-{rtx}-{context}/PROGRAMS.json");
                anyhow::ensure!(files.contains_key(&manifest), "missing required V4 manifest {manifest}");
                let document: Value = serde_json::from_slice(&fs::read(root.join(&manifest))
                    .with_context(|| format!("missing V4 manifest {manifest}"))?)?;
                validate_v4_native_experts(&manifest, &document)?;
                if model == "pro" {
                    for tp in [1, 2] {
                        for rows in cuteafd_core::EXL3_CAPACITIES {
                            let path = format!("{model}-{rtx}-{context}/exl3/exl3-dsv4p-k23/rtx-tp{tp}/m{rows}/v41_exl3.json");
                            anyhow::ensure!(files.contains_key(&path), "missing required V4 package {path}");
                        }
                    }
                }
            }
        }
    }
    for (path, expected) in files {
        anyhow::ensure!(std::path::Path::new(path).components().all(|p| matches!(p, std::path::Component::Normal(_))),
            "invalid V4 input path {path}");
        let actual = format!("{:x}", Sha256::digest(fs::read(root.join(path))
            .with_context(|| format!("missing V4 placement input {path}"))?));
        anyhow::ensure!(expected.as_str() == Some(&actual), "V4 placement input checksum mismatch: {path}");
    }
    Ok(())
}

/// Spark worlds charge the native expert-wave segment programs from the case
/// manifest, so a V4 publication whose export geometry omits `native_experts`
/// leaves every live Spark plan refusing at admission.
fn validate_v4_native_experts(manifest: &str, document: &Value) -> anyhow::Result<()> {
    let programs = document["programs"].as_array()
        .ok_or_else(|| anyhow::anyhow!("V4 manifest {manifest} has no program list"))?;
    for family in ["dsv4f", "dsv4p"] {
        let prefix = format!("native_experts_{family}_");
        anyhow::ensure!(programs.iter().any(|program| program["name"].as_str()
            .is_some_and(|name| name.starts_with(&prefix))),
            "V4 manifest {manifest} is missing {prefix}* native expert programs; republish with the native_experts geometry");
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_test_publication(root: &std::path::Path, files: &[(&str, &[u8])]) -> std::path::PathBuf {
    use sha2::{Digest, Sha256};
    let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json"))).unwrap();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut checksums = serde_json::Map::new();
    for (name, bytes) in files {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
        checksums.insert((*name).into(), Value::String(format!("{:x}", Sha256::digest(bytes))));
    }
    let metadata = serde_json::json!({"schema": 1, "engine_commit": "a".repeat(40),
        "sparkinfer_revision": lock["revision"], "source_tree_sha256": lock["source_tree_sha256"],
        "engine_sources_sha256": compiled_source_hashes(&source, "all").unwrap(),
        "selection_sources_sha256": selection_source_hashes(&source, "all").unwrap(),
        "exports": [{"sparkinfer_revision": lock["revision"],
            "source_tree_sha256": lock["source_tree_sha256"], "cuda_initialized": false}], "files": checksums});
    fs::write(root.join("INPUTS.json"), metadata.to_string()).unwrap();
    root.join("share/PROGRAMS.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A synthetic checkout: a placement-sources.json listing two compiled files
    /// and the selection inputs, the job specs, live and retired configs.
    fn sources_fixture(root: &std::path::Path) {
        for relative in ["python/tools/aot", "scripts/build", "scripts/lib", "examples/configs", "benchmarks",
            "rust/crates/cuteafd-loader/src/serving_capacity"] {
            fs::create_dir_all(root.join(relative)).unwrap();
        }
        fs::write(root.join("scripts/build/placement-sources.json"), json!({"schema": 1,
            "compiled": {"all": ["python/tools/aot/export.py", "scripts/build/layout.json"],
                         "v4": ["python/tools/aot/export.py"]},
            "selection": {"all": {"files": ["rust/crates/cuteafd-loader/src/serving_capacity/spark_memory.json",
                                            "scripts/lib/wip_aot_scopes.py"], "live_configs": true},
                          "v4": {"files": [], "live_configs": false}}}).to_string()).unwrap();
        for relative in ["python/tools/aot/export.py", "scripts/build/layout.json", "scripts/build/placement-jobs.json",
            "rust/crates/cuteafd-loader/src/serving_capacity/spark_memory.json", "scripts/lib/wip_aot_scopes.py",
            "examples/configs/live.config", "examples/configs/retired.config", "python/tools/aot/unlisted.py"] {
            fs::write(root.join(relative), "original").unwrap();
        }
        fs::write(root.join("benchmarks/current-matrix.json"),
            json!({"retired": [{"config": "examples/configs/retired.config"}], "cells": [{"report": "old"}]}).to_string()).unwrap();
    }

    fn fingerprint(root: &std::path::Path, kind: &str) -> Value {
        json!({"engine_sources_sha256": compiled_source_hashes(root, kind).unwrap(),
               "selection_sources_sha256": selection_source_hashes(root, kind).unwrap()})
    }

    #[test]
    fn compiled_tier_tracks_listed_sources_and_job_specs_only() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        sources_fixture(root);
        for kind in ["all", "v4"] {
            let metadata = fingerprint(root, kind);
            validate_sources(&metadata, root, kind, "stale").unwrap();
            for (path, refused) in [("python/tools/aot/export.py", true), ("scripts/build/placement-jobs.json", true),
                ("scripts/build/layout.json", kind == "all"), ("python/tools/aot/unlisted.py", false)] {
                fs::write(root.join(path), "changed").unwrap();
                let error = validate_sources(&metadata, root, kind, "stale").err().map(|error| error.to_string());
                assert_eq!(error.is_some(), refused, "{kind} {path}: {error:?}");
                if let Some(error) = error {
                    assert!(error.contains("export source fingerprints differ") && error.contains(path), "{error}");
                }
                fs::write(root.join(path), "original").unwrap();
            }
            // Listing an added file changes the tier: the list itself is hashed.
            let mut sources: Value = serde_json::from_slice(&fs::read(root.join("scripts/build/placement-sources.json")).unwrap()).unwrap();
            let original = sources.clone();
            sources["compiled"][kind].as_array_mut().unwrap().push(json!("python/tools/aot/unlisted.py"));
            fs::write(root.join("scripts/build/placement-sources.json"), sources.to_string()).unwrap();
            assert!(validate_sources(&metadata, root, kind, "stale").is_err());
            fs::write(root.join("scripts/build/placement-sources.json"), original.to_string()).unwrap();
        }
    }

    #[test]
    fn selection_tier_tracks_measured_memory_live_configs_and_revisions_but_not_retired_or_cards() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        sources_fixture(root);
        let metadata = fingerprint(root, "all");
        let before = &metadata["selection_sources_sha256"];
        assert!(before.get("examples/configs/retired.config").is_none());
        let refused = |path: &str, bytes: &str| {
            let original = fs::read(root.join(path)).unwrap();
            fs::write(root.join(path), bytes).unwrap();
            let error = validate_sources(&metadata, root, "all", "stale").err().map(|error| error.to_string());
            fs::write(root.join(path), original).unwrap();
            error
        };
        for path in ["rust/crates/cuteafd-loader/src/serving_capacity/spark_memory.json",
            "examples/configs/live.config", "scripts/lib/wip_aot_scopes.py"] {
            let error = refused(path, "MODEL_REVISION=changed").expect(path);
            assert!(error.contains("selection source fingerprints differ") && error.contains(path)
                && error.contains("--reattest-from"), "{error}");
        }
        assert!(refused("examples/configs/retired.config", "retired edit").is_none());
        assert!(refused("benchmarks/current-matrix.json",
            &json!({"retired": [{"config": "examples/configs/retired.config"}], "cells": [{"report": "new"}]}).to_string()).is_none());
        let error = refused("benchmarks/current-matrix.json", &json!({"retired": [], "cells": []}).to_string()).unwrap();
        assert!(error.contains("installed_live_configs") && error.contains("examples/configs/retired.config"), "{error}");
        // The V4 inventory selects no configs, so they never stale it.
        let v4 = fingerprint(root, "v4");
        fs::write(root.join("examples/configs/live.config"), "changed").unwrap();
        validate_sources(&v4, root, "v4", "stale").unwrap();
        // A publication predating the selection tier is stale, not silently accepted.
        let mut old = metadata.clone();
        old.as_object_mut().unwrap().remove("selection_sources_sha256");
        fs::write(root.join("examples/configs/live.config"), "original").unwrap();
        assert!(validate_sources(&old, root, "all", "stale").unwrap_err().to_string().contains("fingerprint missing"));
    }

    #[test]
    fn checked_in_placement_sources_hash_on_this_checkout() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        for kind in ["all", "v4"] {
            let compiled = compiled_source_hashes(&root, kind).unwrap();
            assert!(compiled.get("scripts/build/placement-jobs.json").is_some());
            assert!(compiled.get("rust/crates/cuteafd-core/src/dspark_verify.rs").is_some());
            assert!(compiled.as_object().unwrap().keys().all(|path| !path.starts_with("examples/")));
            selection_source_hashes(&root, kind).unwrap();
        }
    }

    #[test]
    fn published_placement_inputs_fail_closed_for_missing_stale_and_modified_files() {
        use sha2::{Digest, Sha256};
        let directory = tempfile::tempdir().unwrap();
        let share = directory.path().join("share");
        fs::create_dir(&share).unwrap();
        let manifest = share.join("PROGRAMS.json");
        let metadata_path = directory.path().join("INPUTS.json");
        assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("PROGRAMS.json missing"));
        fs::write(&manifest, r#"{"programs":[]}"#).unwrap();
        assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("missing placement provenance"));
        let lock: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../../third_party/sparkinfer.lock.json"))).unwrap();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let valid = json!({"schema": 1, "engine_commit": "a".repeat(40),
            "sparkinfer_revision": lock["revision"], "source_tree_sha256": lock["source_tree_sha256"],
            "engine_sources_sha256": compiled_source_hashes(&root, "all").unwrap(),
            "selection_sources_sha256": selection_source_hashes(&root, "all").unwrap(),
            "exports": [{"sparkinfer_revision": lock["revision"],
                "source_tree_sha256": lock["source_tree_sha256"], "cuda_initialized": false}],
            "files": {"share/PROGRAMS.json": format!("{:x}", Sha256::digest(fs::read(&manifest).unwrap()))}});
        fs::write(&metadata_path, valid.to_string()).unwrap();
        validate_placement_inputs(&manifest).unwrap();
        for (field, value, expected) in [
            ("sparkinfer_revision", json!("wrong"), "fork pin mismatch"),
            ("engine_sources_sha256", json!({}), "source fingerprints differ"),
            ("selection_sources_sha256", json!({}), "selection source fingerprints differ"),
            ("exports", json!([]), "worker provenance mismatch"),
            ("files", json!({}), "PROGRAMS.json missing from placement provenance"),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            fs::write(&metadata_path, invalid.to_string()).unwrap();
            let error = format!("{:#}", validate_placement_inputs(&manifest).unwrap_err());
            assert!(error.contains(expected) && error.contains("export-placement-inputs.py"), "{error}");
        }
        fs::write(&metadata_path, valid.to_string()).unwrap();
        fs::write(&manifest, "modified").unwrap();
        assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("checksum mismatch"));
    }

    #[test]
    fn published_reads_recheck_returned_bytes_and_refuse_unlisted_paths() {
        let directory = tempfile::tempdir().unwrap();
        let bytes = b"published programs";
        let manifest = write_test_publication(directory.path(), &[("share/PROGRAMS.json", bytes.as_slice())]);
        let publication = validate_placement_inputs(&manifest).unwrap();
        assert_eq!(publication.read("share/PROGRAMS.json").unwrap(), bytes);
        assert_eq!(publication.fork_revision().len(), 40);
        assert_eq!(publication.fork_tree_sha256().len(), 64);
        for name in ["", "unlisted.json", "../PROGRAMS.json", "/PROGRAMS.json"] {
            assert!(publication.read(name).unwrap_err().to_string().contains("absent from placement provenance"));
        }
        fs::write(&manifest, "changed after validation").unwrap();
        assert!(publication.read("share/PROGRAMS.json").unwrap_err().to_string().contains("checksum mismatch"));
    }

    #[test]
    fn published_inputs_reject_escaping_and_empty_inventory_paths() {
        let directory = tempfile::tempdir().unwrap();
        let manifest = write_test_publication(directory.path(), &[("share/PROGRAMS.json", b"programs")]);
        let metadata = directory.path().join("INPUTS.json");
        let valid: Value = serde_json::from_slice(&fs::read(&metadata).unwrap()).unwrap();
        for name in ["", "../escape", "/absolute"] {
            let mut invalid = valid.clone();
            invalid["files"][name] = json!("a".repeat(64));
            fs::write(&metadata, invalid.to_string()).unwrap();
            assert!(format!("{:#}", validate_placement_inputs(&manifest).unwrap_err()).contains("invalid placement input path"));
        }
    }

    #[test]
    fn live_v4_manifests_require_native_expert_segment_programs() {
        let complete = json!({"programs": [
            {"name": "native_experts_dsv4f_w512_segments"},
            {"name": "native_experts_dsv4p_w1024_segments"},
            {"name": "dsv4f_mhc_pre"}]});
        validate_v4_native_experts("flash-1-131072/PROGRAMS.json", &complete).unwrap();
        for missing in ["native_experts_dsv4f_w512_segments", "native_experts_dsv4p_w1024_segments"] {
            let mut programs = complete["programs"].as_array().unwrap().clone();
            programs.retain(|program| program["name"] != missing);
            let document = json!({"programs": programs});
            let error = validate_v4_native_experts("flash-1-131072/PROGRAMS.json", &document)
                .unwrap_err().to_string();
            assert!(error.contains("is missing native_experts_") && error.contains("republish"), "{error}");
        }
        assert!(validate_v4_native_experts("flash-1-131072/PROGRAMS.json", &json!({})).is_err());
    }

    #[test]
    fn live_v4_inputs_refuse_missing_and_stale_provenance() {
        let root = tempfile::tempdir().unwrap();
        assert!(validate_v4_placement_inputs(root.path()).unwrap_err().to_string().contains("missing live V4 inputs"));
        fs::write(root.path().join("INPUTS.json"), json!({"schema": 1, "engine_commit": "a".repeat(40),
            "sparkinfer_revision": "wrong", "source_tree_sha256": "wrong", "files": {}}).to_string()).unwrap();
        assert!(validate_v4_placement_inputs(root.path()).unwrap_err().to_string().contains("stale V4 placement inputs"));
    }
}
