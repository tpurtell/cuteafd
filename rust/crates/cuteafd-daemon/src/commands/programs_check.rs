//! `cuteafd programs-check`: load a built library's program table with its
//! PROGRAMS.json through the engine's fail-closed loader, as a slot seal or
//! image inventory step. Nothing is launched and no weights are read.
use anyhow::{Context, Result, ensure};
use clap::Args;
use cuteafd_ffi::NativeLibrary;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub(crate) struct ProgramsCheckArgs {
    /// The native library whose program table is checked.
    #[arg(long)]
    pub(crate) native_lib: PathBuf,
    /// The exporter manifest installed beside it.
    #[arg(long)]
    pub(crate) manifest: PathBuf,
    /// Require the EXLR8 programs the manifest records for this role
    /// (`families.exlr8`), each present in the library and validated.
    #[arg(long, value_parser = ["coordinator", "spark"])]
    pub(crate) exlr8_role: Option<String>,
}

pub(crate) fn run(args: ProgramsCheckArgs) -> Result<()> {
    // SAFETY: loading a native library runs its initializers; the build output is ours.
    let library = unsafe { NativeLibrary::load(&args.native_lib)? };
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let names: BTreeSet<&str> = programs.names().collect();
    let exlr8 = expert_programs(names.iter().copied());
    if let Some(role) = &args.exlr8_role {
        let text = std::fs::read_to_string(&args.manifest)?;
        let manifest: serde_json::Value = serde_json::from_str(&text)?;
        let family = &manifest["families"]["exlr8"];
        ensure!(family["role"].as_str() == Some(role.as_str()),
            "{}: EXLR8 programs were exported for role {}, not {role}", args.manifest.display(), family["role"]);
        let recorded: BTreeSet<&str> = family["programs"].as_array().context("manifest records no EXLR8 programs")?
            .iter().map(|p| p.as_str().context("EXLR8 program name")).collect::<Result<_>>()?;
        ensure!(!recorded.is_empty() && recorded == exlr8,
            "EXLR8 programs in the library {exlr8:?} differ from the manifest's {role} set {recorded:?}");
    }
    println!("programs-check: {} programs ({} EXLR8: {}) validated against {}", names.len(), exlr8.len(),
        exlr8.iter().copied().collect::<Vec<_>>().join(","), args.manifest.display());
    Ok(())
}

/// The routed/shared EXLR8 expert programs among `names`: `exlr8_*` except the
/// checkpoint-native dense operands (`exlr8_dense_*`), which validate under
/// their own ABI and are not recorded in `families.exlr8`.
fn expert_programs<'a>(names: impl Iterator<Item = &'a str>) -> BTreeSet<&'a str> {
    names.filter(|n| n.starts_with("exlr8_") && !n.starts_with("exlr8_dense_")).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_dense_programs_are_not_expert_programs() {
        let names = ["exlr8_a16_serving", "exlr8_shared_group", "exlr8_dense_trellis_o_tp2",
            "exlr8_dense_fp8_row_lm_head", "glm_norm"];
        let experts: Vec<_> = super::expert_programs(names.into_iter()).into_iter().collect();
        assert_eq!(experts, ["exlr8_a16_serving", "exlr8_shared_group"]);
    }
}
