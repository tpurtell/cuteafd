//! CPU-only EXL3 semantic image, addressed after the device prefix in a push
//! rank's logical arena. It is never exposed to kernels or copied to the GPU.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{
    push_load::builder::{Builder, Destination},
    OfficialV41Catalog, V41Exl3Residency,
};

pub(crate) struct ValidationLayout {
    entries: Vec<(String, cuteafd_loader::Exl3Codebook)>,
    base: u64,
}
impl ValidationLayout {
    pub fn exl3(
        residencies: &[V41Exl3Residency],
        base: u64,
        admitted_host_bytes: usize,
    ) -> Result<Self> {
        let entries = residencies
            .iter()
            .flat_map(|r| r.loads.iter().map(move |load| (r.codebook, load)))
            .filter_map(|(codebook, load)| {
                load.tensor
                    .strip_suffix(".trellis")
                    .map(|prefix| (format!("{prefix}.{}", codebook.marker()), codebook))
            })
            .collect::<Vec<_>>();
        let bytes = entries
            .len()
            .checked_mul(4)
            .context("EXL3 validation size overflow")?;
        ensure!(
            bytes <= admitted_host_bytes && base.checked_add(bytes as u64).is_some(),
            "EXL3 validation image exceeds admission"
        );
        Ok(Self { entries, base })
    }
    pub fn bytes(&self) -> usize {
        self.entries.len() * 4
    }
    pub fn base(&self) -> u64 {
        self.base
    }
    pub fn append(
        &self,
        builder: &mut Builder,
        catalog: &OfficialV41Catalog,
        rank: u32,
    ) -> Result<()> {
        for (index, (name, _)) in self.entries.iter().enumerate() {
            let source = Builder::v41_source(catalog, name)?;
            ensure!(source.bytes == 4, "EXL3 multiplier is not a scalar");
            builder.contiguous(
                &source,
                0,
                4,
                Destination {
                    rank,
                    offset: self.base + index as u64 * 4,
                    bytes: 4,
                },
            )?;
        }
        Ok(())
    }
    pub fn validate(&self, image: &[u8]) -> Result<()> {
        ensure!(
            image.len() == self.bytes(),
            "EXL3 validation image length mismatch"
        );
        for ((name, codebook), bytes) in self.entries.iter().zip(image.chunks_exact(4)) {
            ensure!(
                u32::from_le_bytes(bytes.try_into().unwrap()) == codebook.multiplier(),
                "unsupported EXL3 {} multiplier for {name}",
                codebook.marker().to_ascii_uppercase()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validation_image_is_exact_and_semantic_not_a_checksum() -> Result<()> {
        let layout = ValidationLayout {
            entries: vec![("projection.mcg".into(), cuteafd_loader::Exl3Codebook::Mcg)],
            base: 4096,
        };
        assert!(layout.validate(&[0; 4]).is_err());
        assert!(layout.validate(&[]).is_err());
        assert!(layout.validate(&0x83dcd12du32.to_le_bytes()).is_err());
        layout.validate(&0xcbac1fedu32.to_le_bytes())?;
        assert_eq!(layout.base(), 4096);

        let mul1_layout = ValidationLayout {
            entries: vec![("projection.mul1".into(), cuteafd_loader::Exl3Codebook::Mul1)],
            base: 8192,
        };
        assert!(mul1_layout.validate(&0xcbac1fedu32.to_le_bytes()).is_err());
        mul1_layout.validate(&0x83dcd12du32.to_le_bytes())?;
        Ok(())
    }
}
