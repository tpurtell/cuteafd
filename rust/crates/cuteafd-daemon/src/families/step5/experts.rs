//! The routed-expert storage a Step-5 checkpoint ships, read from its own
//! headers: native FP8/NVFP4 (the official releases, with BF16 tails) or EXL3
//! (the K4 + Core-K6 publication, K6 tails). One view selects the Spark/RTX
//! package, the wire each layer's request uses and the package's exports.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::formats::fp8_experts::ExpertFormat;
use cuteafd_loader::OfficialV41Catalog;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum StepExperts {
    /// Native FP8 block-128 or ModelOpt NVFP4 (BF16 tails 88-90).
    Native(ExpertFormat),
    /// Standard EXL3 decoder tiers and codebook (K4 experts, K6 tails).
    Exl3(Vec<usize>, cuteafd_loader::Exl3Codebook),
}

impl StepExperts {
    pub(super) fn from_catalog(catalog: &OfficialV41Catalog) -> Result<Self> {
        if let Some(manifest) = catalog.exl3() {
            return Ok(Self::Exl3(manifest.decoder_tiers().to_vec(), manifest.codebook()));
        }
        let format = catalog.fp8().context("Step-5 routed experts need FP8, NVFP4 or EXL3 headers")?.format();
        ensure!(matches!(format, ExpertFormat::Fp8Block128 | ExpertFormat::Nvfp4),
            "Step-5 BF16 is a fidelity reference, not a serving format");
        Ok(Self::Native(format))
    }

    pub(super) fn is_exl3(&self) -> bool { matches!(self, Self::Exl3(..)) }

    /// The installed package's manifest beside the native library.
    pub(super) fn package_manifest(&self, native_lib: &Path) -> Result<PathBuf> {
        let lib = native_lib.parent().context("native library directory")?;
        Ok(match self {
            Self::Native(format) => lib.join("fp8").join(format.package_name("step5")).join("manifest.json"),
            Self::Exl3(tiers, codebook) => lib.join("exl3")
                .join(crate::shared::experts::exl3::package_name_for("step5", tiers, *codebook)).join("manifest.json"),
        })
    }

    /// The code-allowance key of the loaded expert package.
    pub(super) fn package_name(&self) -> String {
        match self {
            Self::Native(format) => format.package_name("step5"),
            Self::Exl3(tiers, codebook) => crate::shared::experts::exl3::package_name_for("step5", tiers, *codebook),
        }
    }

    /// Native FP8 mixes K32 layers with BF16 tails; EXL3 uses K32 throughout.
    pub(super) fn ingress(&self) -> cuteafd_core::SparkIngress {
        use cuteafd_core::SparkIngress;
        match self {
            Self::Exl3(..) => SparkIngress::Fp8K32,
            Self::Native(format) if format.bf16_input() => SparkIngress::Bf16,
            Self::Native(_) => SparkIngress::Native,
        }
    }

    /// The wire one routed layer's request uses.
    pub(super) fn layer_bf16_input(&self, catalog: &OfficialV41Catalog, layer: usize) -> Result<bool> {
        Ok(match self {
            Self::Native(_) => catalog.fp8().context("Step-5 native expert headers")?.layer_format(layer)?.bf16_input(),
            Self::Exl3(..) => false,
        })
    }

    /// Capacities the installed package exports, identical across layouts.
    pub(super) fn exports(&self, manifest: &serde_json::Value) -> Result<Vec<u32>> {
        match self {
            Self::Native(_) => super::serve::package_exports(manifest),
            Self::Exl3(_, codebook) => {
                codebook.check_aot_manifest(manifest)?;
                let mut capacities = manifest["variants"].as_array().context("EXL3 package variants missing")?
                    .iter().map(|v| v["capacity"].as_u64().context("EXL3 variant capacity")
                        .and_then(|c| Ok(u32::try_from(c)?)))
                    .collect::<Result<Vec<_>>>()?;
                ensure!(manifest["geometry"] == "step5", "EXL3 package geometry is not step5");
                capacities.sort_unstable();
                capacities.dedup();
                ensure!(!capacities.is_empty(), "EXL3 Step-5 package has no capacities");
                Ok(capacities)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exl3_package_resolves_family_tiers_and_fp8_wire() {
        let exl3 = StepExperts::Exl3(vec![4, 6], cuteafd_loader::Exl3Codebook::Mcg);
        assert_eq!(exl3.package_name(), "exl3-step5-k46");
        assert_eq!(exl3.package_manifest(Path::new("/opt/cuteafd/lib/libcuteafd.so")).unwrap(),
            Path::new("/opt/cuteafd/lib/exl3/exl3-step5-k46/manifest.json"));
        let mul1 = StepExperts::Exl3(vec![4, 6], cuteafd_loader::Exl3Codebook::Mul1);
        assert_eq!(mul1.package_name(), "exl3-step5-k46-mul1");
        assert_eq!(mul1.package_manifest(Path::new("/opt/cuteafd/lib/libcuteafd.so")).unwrap(),
            Path::new("/opt/cuteafd/lib/exl3/exl3-step5-k46-mul1/manifest.json"));
        assert_eq!(exl3.ingress(), cuteafd_core::SparkIngress::Fp8K32);
        let manifest = serde_json::json!({"geometry": "step5", "variants": [
            {"capacity": 80}, {"capacity": 1}, {"capacity": 80}, {"capacity": 4096}]});
        assert_eq!(exl3.exports(&manifest).unwrap(), [1, 80, 4096]);
        assert!(mul1.exports(&manifest).is_err());
        let mut mul1_manifest = manifest;
        mul1_manifest["codebook"] = serde_json::json!("mul1");
        assert_eq!(mul1.exports(&mul1_manifest).unwrap(), [1, 80, 4096]);
        assert!(exl3.exports(&serde_json::json!({"geometry": "glmf", "variants": [{"capacity": 1}]})).is_err());
        let nvfp4 = StepExperts::Native(ExpertFormat::Nvfp4);
        assert_eq!(nvfp4.ingress(), cuteafd_core::SparkIngress::Bf16);
        assert_eq!(nvfp4.package_name(), "fp8-step5-nvfp4a4");
    }

    #[test]
    fn native_fp8_endpoint_accepts_k32_layers_and_bf16_tails() {
        let fp8 = StepExperts::Native(ExpertFormat::Fp8Block128);
        assert_eq!(fp8.ingress(), cuteafd_core::SparkIngress::Native);
    }
}
