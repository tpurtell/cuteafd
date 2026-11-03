//! GLM 5.x (glm_moe_dsa) family: configuration for the generic engine.
pub mod config;
pub mod core;
pub mod names;
pub mod draft_representation;
pub mod native;
pub use config::{GlmDsaConfig, GlmIndexer, GlmVariant};

/// Stage ABIs follow the checkpoint's dense core, not the build's union of exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensePrograms { Fp8, Exlr8, Trellis }

impl DensePrograms {
    pub fn stage(self, name: &str, split: bool) -> bool {
        let stage = name.strip_prefix("glm_stage_")
            .or_else(|| if split { name.strip_prefix("glm2_stage_") } else { None });
        let Some(stage) = stage else { return false; };
        match self {
            Self::Fp8 => false,
            Self::Trellis => true,
            // EXLR8 uses its own SwiGLU and FP8 context absorb. Keep its existing
            // non-context BF16 companions, but never load new K6-only stages.
            Self::Exlr8 => stage.rsplit_once("_m").is_some_and(|(kind, rows)|
                rows.parse::<usize>().is_ok_and(|rows| rows > 0)
                    && matches!(kind, "rank_pack" | "query_rope" | "index_post"
                        | "absorb" | "absorb_bf16" | "values" | "values_bf16"
                        | "query_rope_context" | "absorb_context")),
        }
    }

    pub fn program_set(self, manifest: &serde_json::Value, split: bool) -> crate::placement::ProgramSet {
        let mut families = vec!["glm"];
        if split { families.push("glm2"); }
        if self == Self::Exlr8 { families.push("exlr8"); }
        if self != Self::Fp8 {
            families.push("glm_stage");
            if split { families.push("glm2_stage"); }
        }
        let mut set = crate::placement::ProgramSet::from_manifest(manifest, &families);
        let keep = |name: &str| match cuteafd_core::coordinator_programs::program_namespace(name) {
            Some("glm_stage" | "glm2_stage") => self.stage(name, split),
            _ => true,
        };
        set.names.retain(|name| keep(name));
        set.scratch.retain(|(name, _)| keep(name));
        set
    }
}

#[cfg(test)]
mod exported_manifest {
    use super::DensePrograms;
    use cuteafd_ffi::{glm_stage, native_dense, native_dense_glue, programs::ProgramSpec};
    use serde_json::Value;
    use std::collections::BTreeSet;

    fn expected_names(manifest: &Value) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for entry in manifest["programs"].as_array().expect("manifest programs") {
            let name = entry["name"].as_str().expect("manifest program name");
            if entry["family"] == "exlr8_dense" || DensePrograms::Exlr8.stage(name, true) {
                assert!(names.insert(name.to_owned()), "duplicate manifest entry {name}");
            }
        }
        assert!(names.iter().any(|name| name.starts_with("exlr8_dense_")), "missing dense catalog");
        for prefix in ["glm_stage_", "glm2_stage_"] {
            assert!(names.iter().any(|name| name.starts_with(prefix)), "missing stage catalog {prefix}");
        }
        names
    }

    fn validate_listing(manifest: &Value, listing: &Value) -> Result<usize, String> {
        let expected = expected_names(manifest);
        let mut checked = BTreeSet::new();
        for entry in listing["programs"].as_array().ok_or("binding ABI programs")? {
            let name = entry["name"].as_str().ok_or("binding ABI name")?;
            if !checked.insert(name.to_owned()) { return Err(format!("duplicate ABI entry {name}")); }
            if !expected.contains(name) { return Err(format!("unexpected ABI entry {name}")); }
            let spec = ProgramSpec { name: name.into(), pointers: entry["pointers"].as_array().ok_or("pointers")?.iter()
                .map(|p| p["name"].as_str().unwrap().into()).collect(),
                scalar_kinds: entry["scalars"].as_array().ok_or("scalars")?.iter().map(|s|
                    if s["type"] == "int32" { b'i' } else { b'?' }).collect(),
                scratch: entry["scratch_bytes_at_capacity"].as_object().ok_or("scratch")?.iter()
                    .map(|(k, v)| (k.clone(), v.as_u64().unwrap())).collect(),
                capacity_rows: entry["capacity_rows"].as_u64().ok_or("capacity")? as u32,
                geometry: entry["geometry"].clone() };
            let result = if DensePrograms::Exlr8.stage(name, true) {
                glm_stage::validate_entry(&spec, entry).map(|_| ()).map_err(|e| e.to_string())
            } else if native_dense_glue::op_of(entry).map_err(|e| e.to_string())?.is_some() {
                native_dense_glue::validate_entry(&spec, entry).map(|_| ()).map_err(|e| e.to_string())
            } else {
                native_dense::validate_entry(&spec, entry).map(|_| ()).map_err(|e| e.to_string())
            };
            result.map_err(|e| format!("{name}: {e}"))?;
        }
        if checked != expected {
            return Err(format!("ABI coverage: missing {:?}, unexpected {:?}",
                expected.difference(&checked).collect::<Vec<_>>(), checked.difference(&expected).collect::<Vec<_>>()));
        }
        Ok(checked.len())
    }

    #[test]
    fn expected_native_abis_follow_manifest_and_actual_stage_policy() {
        let manifest = serde_json::json!({"programs": [
            {"name": "exlr8_dense_geometry", "family": "exlr8_dense"},
            {"name": "glm_stage_rank_pack_m64", "family": "glm_stage"},
            {"name": "glm2_stage_values_m4096", "family": "glm2_stage"},
            {"name": "glm_stage_swiglu_i2048", "family": "glm_stage"},
            {"name": "glm_stage_absorb_bf16_context_m64", "family": "glm_stage"},
            {"name": "glm2_stage_absorb_bf16_context_m64", "family": "glm2_stage"}
        ]});
        let expected = expected_names(&manifest);
        assert_eq!(expected, BTreeSet::from(["exlr8_dense_geometry".into(),
            "glm_stage_rank_pack_m64".into(), "glm2_stage_values_m4096".into()]));
        assert!(validate_listing(&manifest, &serde_json::json!({"programs": []}))
            .unwrap_err().contains("missing"));
    }

    /// The independent complete manifest supplies expected names, while the ABI
    /// listing supplies contracts. No fixed count or copied stage policy.
    #[test]
    #[ignore = "requires complete export manifest and selected binding ABI listing"]
    fn every_exported_dense_entry_validates() {
        let read = |key| -> Value {
            serde_json::from_slice(&std::fs::read(std::env::var(key).expect(key)).unwrap()).unwrap()
        };
        let manifest = read("CUTEAFD_EXLR8_DENSE_MANIFEST");
        let listing = read("CUTEAFD_EXLR8_DENSE_ABIS");
        let checked = validate_listing(&manifest, &listing).unwrap();
        eprintln!("validated {checked} selected dense/stage ABIs against complete manifest");
        let mut missing = listing.clone();
        missing["programs"].as_array_mut().unwrap().pop();
        assert!(validate_listing(&manifest, &missing).unwrap_err().contains("missing"));
        let mut duplicate = listing.clone();
        let first = duplicate["programs"][0].clone();
        duplicate["programs"].as_array_mut().unwrap().push(first);
        assert!(validate_listing(&manifest, &duplicate).unwrap_err().contains("duplicate"));
        let mut unexpected = listing.clone();
        unexpected["programs"][0]["name"] = Value::String("unselected_abi".into());
        assert!(validate_listing(&manifest, &unexpected).unwrap_err().contains("unexpected"));
    }
}
