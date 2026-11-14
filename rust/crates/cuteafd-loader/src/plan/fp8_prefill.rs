//! Shared activation policy. Selecting a route never converts a weight or owns storage.
use serde::Serialize;
use std::{collections::BTreeSet, fmt, str::FromStr};

pub const ENV: &str = "CUTEAFD_FP8_PREFILL";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Policy {
    #[default]
    Auto,
    Off,
    Groups(BTreeSet<String>),
}
impl FromStr for Policy {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            _ => {
                let mut groups = BTreeSet::new();
                for group in value.split(',') {
                    if !matches!(group, "mla" | "index" | "attention" | "ffn" | "kda-in" | "kda-o" | "experts") {
                        return Err(format!("FP8_PREFILL must be auto, off or a projection group list; invalid group {group:?}"));
                    }
                    if !groups.insert(group.to_owned()) {
                        return Err(format!("FP8_PREFILL repeats group {group:?}"));
                    }
                }
                Ok(Self::Groups(groups))
            }
        }
    }
}
impl fmt::Display for Policy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"), Self::Off => f.write_str("off"),
            Self::Groups(groups) => f.write_str(&groups.iter().map(String::as_str).collect::<Vec<_>>().join(",")),
        }
    }
}
impl Policy {
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(ENV) {
            Ok(value) => value.parse(),
            Err(std::env::VarError::NotPresent) => Ok(Self::Auto),
            Err(error) => Err(format!("{ENV}: {error}")),
        }
    }
    pub fn requests(&self, group: Group) -> bool {
        match self {
            Self::Auto => group.auto,
            Self::Off => false,
            Self::Groups(groups) => groups.contains(group.name),
        }
    }
    /// `None` retains the package's format-aware default. In particular MXFP4
    /// arithmetic is not an FP8-weight projection and must not be overridden.
    pub fn experts(&self, fp8_weights: bool) -> Option<bool> {
        if !fp8_weights { return None; }
        match self {
            Self::Auto => None,
            Self::Off => Some(false),
            Self::Groups(groups) => groups.contains("experts").then_some(true),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Group {
    pub name: &'static str,
    pub auto: bool,
}
impl Group {
    pub const fn auto(name: &'static str) -> Self { Self { name, auto: true } }
    pub const fn opt_in(name: &'static str) -> Self { Self { name, auto: false } }
}
pub const GLMF_GROUPS: &[Group] = &[
    Group::auto("mla"), Group::auto("ffn"), Group::opt_in("kda-in"), Group::opt_in("kda-o"), Group::auto("experts"),
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode { W8a8, W8a16, Unchanged }

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Resolution {
    pub group: String,
    pub mode: Mode,
    /// Existing admitted program workspace; this record is never another charge.
    pub scratch_owner: Option<String>,
    pub hint: Option<String>,
}

/// Both planner and runtime supply resident-format and compiled-route facts.
/// A missing route cannot make an otherwise executable checkpoint unsupported.
pub fn resolve(policy: &Policy, group: Group, resident_fp8: bool,
    w8a8_available: bool, scratch_owner: Option<&str>) -> Resolution {
    let requested = policy.requests(group);
    let mode = if !resident_fp8 { Mode::Unchanged }
        else if requested && w8a8_available { Mode::W8a8 } else { Mode::W8a16 };
    Resolution {
        group: group.name.into(),
        scratch_owner: (mode == Mode::W8a8).then(|| scratch_owner.map(str::to_owned)).flatten(),
        hint: (resident_fp8 && requested && !w8a8_available).then(||
            format!("{}: no W8A8 prefill program for the admitted projection; using W8A16", group.name)),
        mode,
    }
}

/// Report the effective Flash groups using its admitted native stage inventory and
/// the same fused-program scratch metadata that admission consumes.
pub(crate) fn flash_report(checkpoint: &super::Checkpoint, cfg: &crate::families::glm5_flash::GlmNextConfig,
    policy: &Policy, groups: &[Group], manifest: Option<&serde_json::Value>, ranks: usize,
    representation: crate::families::glm5_flash::resident::GlmfRepresentation) -> anyhow::Result<Vec<Resolution>> {
    use crate::families::glm5_flash::{core::CoreInventory, core_admission::{fp8_projection_name, fp8_projection_prefix}, GlmNextAttention};
    let core = CoreInventory::from_checkpoint(cfg, checkpoint)?;
    let has = |name: &str| manifest.and_then(|m| m["programs"].as_array())
        .is_some_and(|programs| programs.iter().any(|program| program["name"] == name));
    let group_of = |name: &str| -> Option<&str> {
        let rest = name.strip_prefix("model.language_model.layers.")?;
        let (layer, leaf) = rest.split_once('.')?;
        let layer = layer.parse::<usize>().ok()?;
        if layer >= cfg.layers { return None; }
        if leaf.starts_with("mlp.shared_experts.") || cfg.dense[layer] && leaf.starts_with("mlp.") { return Some("ffn"); }
        if !leaf.starts_with("self_attn.") { return None; }
        Some(if cfg.attention[layer] == GlmNextAttention::Kda {
            if leaf.starts_with("self_attn.o_proj.") { "kda-o" } else { "kda-in" }
        } else { "mla" })
    };
    let mut result = Vec::new();
    for &group in groups.iter().filter(|group| group.name != "experts") {
        let resident = checkpoint.tensors.iter().any(|tensor| {
            tensor.meta.name.ends_with(".weight") && group_of(&tensor.meta.name) == Some(group.name)
                && (tensor.meta.dtype == cuteafd_core::DType::F8E4M3 || checkpoint.converts_to_fp8(&tensor.meta.name))
        }) || representation.kda_fp8 && matches!(group.name, "kda-in" | "kda-o");
        let mut available = true;
        let mut native = false;
        for layer in 0..cfg.layers {
            for rank in 0..ranks {
                for (name, layout) in core.layer_native(cfg, layer, ranks, None)?[rank].iter()
                    .filter(|(_, layout)| layout.fp8) {
                    let full = format!("model.language_model.layers.{layer}.{name}.weight");
                    if group_of(&full) != Some(group.name) { continue; }
                    native = true;
                    available &= has(&fp8_projection_name(fp8_projection_prefix(layout), layout.shape, 4096));
                }
            }
        }
        if !native {
            let prefix = if ranks == 2 { "glmf2" } else { "glmf" };
            available = match group.name {
                "mla" => has(&format!("{prefix}_mla_producer_m4096")) && has(&format!("{prefix}_o_m4096")),
                "ffn" => [cfg.moe_intermediate, cfg.dense_intermediate].into_iter()
                    .all(|inter| has(&format!("{prefix}_ffn_i{}_m4096", inter / ranks))),
                "kda-in" | "kda-o" => has(&format!("{prefix}_kda_w8_m4096")),
                _ => false,
            };
        }
        result.push(resolve(policy, group, resident, available,
            Some(if native { "selected native core: prefill stage scratch" } else { "steps: prefill program scratch" })));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grammar_is_shared_and_strict() {
        for value in ["auto", "off", "mla,ffn", "experts", "kda-in,kda-o"] {
            let policy: Policy = value.parse().unwrap();
            assert_eq!(policy.to_string().parse::<Policy>().unwrap(), policy);
        }
        for value in ["", "all", "none", "auto,mla", "ffn,ffn", "mla,", " ml a"] {
            assert!(value.parse::<Policy>().is_err(), "{value}");
        }
    }
    #[test]
    fn defaults_preserve_flash_and_non_fp8_weights() {
        for group in GLMF_GROUPS {
            assert_eq!(resolve(&Policy::Auto, *group, false, true, Some("scratch")).mode, Mode::Unchanged);
            assert_eq!(resolve(&Policy::Auto, *group, true, true, Some("scratch")).mode,
                if group.auto { Mode::W8a8 } else { Mode::W8a16 });
        }
    }
    #[test]
    fn fallback_has_no_extra_owner_or_refusal() {
        let resolution = resolve(&Policy::Auto, Group::auto("mla"), true, false, Some("scratch"));
        assert_eq!(resolution.mode, Mode::W8a16);
        assert!(resolution.scratch_owner.is_none());
        assert!(resolution.hint.is_some());
    }
    #[test]
    fn expert_scope_preserves_format_defaults() {
        for policy in [Policy::Auto, Policy::Off, "experts".parse().unwrap(), "mla".parse().unwrap()] {
            assert_eq!(policy.experts(false), None);
        }
        assert_eq!(Policy::Auto.experts(true), None);
        assert_eq!(Policy::Off.experts(true), Some(false));
        assert_eq!("experts".parse::<Policy>().unwrap().experts(true), Some(true));
        assert_eq!("mla".parse::<Policy>().unwrap().experts(true), None);
    }
}
