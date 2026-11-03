//! Check a built program table against its manifest and requested checkpoint family.
use anyhow::{Context, Result, ensure};
use clap::Args;
use cuteafd_ffi::{NativeLibrary, exlr8::{KernelFamily, KernelKind}};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub(crate) struct ProgramsCheckArgs {
    #[arg(long)]
    pub(crate) native_lib: PathBuf,
    #[arg(long)]
    pub(crate) manifest: PathBuf,
    #[arg(long, value_parser = ["coordinator", "spark"], requires = "exlr8_family")]
    pub(crate) exlr8_role: Option<String>,
    /// Explicit checkpoint family scopes, not inferred from the built inventory.
    #[arg(long, requires = "exlr8_role")]
    pub(crate) exlr8_family: Option<String>,
}

pub(crate) fn run(args: ProgramsCheckArgs) -> Result<()> {
    // SAFETY: loading a native library runs its initializers; the build output is ours.
    let library = unsafe { NativeLibrary::load(&args.native_lib)? };
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let names: BTreeSet<&str> = programs.names().collect();
    let exlr8 = expert_programs(names.iter().copied());
    if let Some(role) = &args.exlr8_role {
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
        validate_exlr8(&manifest, &names, role, args.exlr8_family.as_deref()
            .context("EXLR8 requires explicit checkpoint family scope")?, |name| Ok(programs.spec(name)?.clone()))?;
    }
    println!("programs-check: {} programs ({} EXLR8: {}) validated against {}", names.len(), exlr8.len(),
        exlr8.iter().copied().collect::<Vec<_>>().join(","), args.manifest.display());
    Ok(())
}

fn validate_exlr8(manifest: &serde_json::Value, names: &BTreeSet<&str>, role: &str,
    requested: &str, spec: impl Fn(&str) -> Result<cuteafd_ffi::programs::ProgramSpec>) -> Result<()> {
    use KernelKind::*;
    let scopes: Vec<_> = requested.split([';', ',']).collect();
    ensure!(!scopes.is_empty() && scopes.iter().all(|scope| matches!(*scope, "glm" | "glmf")),
        "EXLR8 requires explicit checkpoint family scope glm and/or glmf");
    ensure!(matches!(role, "coordinator" | "spark"), "unsupported EXLR8 role {role}");
    for scope in scopes {
        let (family, key) = match scope {
            "glm" => (KernelFamily::Glm, "exlr8"),
            "glmf" => (KernelFamily::GlmFlash, "exlr8_glmf"),
            _ => unreachable!("validated family scope"),
        };
        let geometry = family.geometry();
        let exported = &manifest["families"][key];
        ensure!(exported["role"].as_str() == Some(role),
            "EXLR8 {scope} checkpoint requires {role} programs (hidden {}, experts {}), not {}",
            geometry.hidden, geometry.experts, exported["role"]);
        let recorded: BTreeSet<&str> = exported["programs"].as_array().context("manifest records no EXLR8 programs")?
            .iter().map(|p| p.as_str().context("EXLR8 program name")).collect::<Result<_>>()?;
        let present: BTreeSet<&str> = expert_programs(names.iter().copied()).into_iter()
            .filter(|name| name.starts_with(&format!("{key}_"))
                && (family != KernelFamily::Glm || !name.starts_with("exlr8_glmf_"))).collect();
        ensure!(!recorded.is_empty() && recorded == present,
            "EXLR8 {scope} library programs differ from its recorded {role} set");
        let mut required = vec![Routed, Group, GroupedDecode, GroupedA16, GroupedGroup];
        if role == "coordinator" { required.extend([Shared, SharedGroup]); }
        for kind in required {
            let name = family.program(kind);
            ensure!(recorded.contains(name), "EXLR8 {scope} checkpoint requires program {name}");
            let spec = spec(name)?;
            let entry = manifest["programs"].as_array().context("manifest programs")?.iter()
                .find(|entry| entry["name"] == name).context("required EXLR8 program entry")?;
            cuteafd_ffi::exlr8::validate_entry(&spec, entry)?;
        }
    }
    Ok(())
}

fn expert_programs<'a>(names: impl Iterator<Item = &'a str>) -> BTreeSet<&'a str> {
    names.filter(|n| n.starts_with("exlr8_") && !n.starts_with("exlr8_dense_")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_dense_programs_are_not_expert_programs() {
        let names = ["exlr8_a16_serving", "exlr8_shared_group", "exlr8_dense_trellis_o_tp2",
            "exlr8_dense_fp8_row_lm_head", "glm_norm"];
        let experts: Vec<_> = expert_programs(names.into_iter()).into_iter().collect();
        assert_eq!(experts, ["exlr8_a16_serving", "exlr8_shared_group"]);
    }

    #[test]
    fn glm_expert_inventory_cannot_seal_a_flash_checkpoint() {
        let manifest = serde_json::json!({"families":{"exlr8":{"role":"coordinator",
            "hidden":6144,"experts":256,"programs":["exlr8_a16_serving"]}},
            "programs":[{"name":"exlr8_a16_serving"}]});
        let names = BTreeSet::from(["exlr8_a16_serving"]);
        let error = validate_exlr8(&manifest, &names, "coordinator", "glmf", |_| unreachable!()).unwrap_err().to_string();
        assert!(error.contains("glmf checkpoint requires coordinator programs (hidden 4096, experts 288)"), "{error}");
        assert!(validate_exlr8(&manifest, &names, "coordinator", "", |_| unreachable!()).unwrap_err().to_string()
            .contains("explicit checkpoint family scope"));
    }
}
