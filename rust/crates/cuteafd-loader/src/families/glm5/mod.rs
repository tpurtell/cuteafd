//! GLM 5.x (glm_moe_dsa) family: configuration for the generic engine.
pub mod config;
pub mod core;
pub mod names;
pub mod draft_representation;
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
