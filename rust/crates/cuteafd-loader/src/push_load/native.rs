//! Native expert images: raw planes occupy their final resident regions.
//! Only non-monotonic byte permutations remain for the post-Done pass.
use anyhow::{ensure, Context, Result};

#[derive(Debug, Clone)]
pub struct NativePlanes {
    pub hidden: usize,
    pub intermediate: usize,
    pub kernel_intermediate: usize,
    pub experts: usize,
    pub nvfp4: bool,
    pub scale_codes: bool,
    pub planes: [(usize, usize); 4],
    pub vectors: [(usize, usize); 4],
    pub bytes: usize,
}
impl NativePlanes {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        kernel_intermediate: usize,
        experts: usize,
        nvfp4: bool,
    ) -> Result<Self> {
        Self::with_scale_codes(hidden, intermediate, kernel_intermediate, experts, nvfp4, false)
    }

    /// Select the package's immutable scale representation before allocation.
    pub fn with_scale_codes(hidden: usize, intermediate: usize, kernel_intermediate: usize,
        experts: usize, nvfp4: bool, scale_codes: bool) -> Result<Self> {
        ensure!(!nvfp4 || !scale_codes, "NVFP4 cannot use UE8M0 scale codes");
        ensure!(
            hidden > 0
                && hidden % 256 == 0
                && intermediate > 0
                && intermediate % 32 == 0
                && kernel_intermediate >= intermediate
                && kernel_intermediate % (if nvfp4 { 64 } else { 32 }) == 0
                && (nvfp4 || Some(kernel_intermediate)
                    == cuteafd_core::mxfp4_layouts::stored_width(hidden, intermediate))
                && kernel_intermediate <= 8192
                && experts > 0,
            "invalid native push geometry"
        );
        let sizes = if scale_codes {
            crate::formats::scale_codes::expert_planes(hidden, intermediate, kernel_intermediate)
                .context("invalid coded expert geometry")?.map(|n| n as u64)
        } else {
            crate::placement::families::deepseek_v41::resident_planes(
                hidden as u64, kernel_intermediate as u64, nvfp4)?
        };
        let mut cursor = 0usize;
        let mut reserve = |size: usize| -> Result<(usize, usize)> {
            cursor = cursor
                .checked_add(255)
                .context("native plane alignment overflow")?
                & !255;
            let start = cursor;
            cursor = cursor
                .checked_add(size)
                .context("native plane extent overflow")?;
            Ok((start, size))
        };
        let mut planes = [(0, 0); 4];
        for (plane, size) in planes.iter_mut().zip(sizes) {
            *plane = reserve(
                usize::try_from(size)?
                    .checked_mul(experts)
                    .context("native expert plane overflow")?,
            )?;
        }
        let mut vectors = [(0, 0); 4];
        if nvfp4 {
            for vector in &mut vectors {
                *vector = reserve(experts.checked_mul(4).context("native scalar overflow")?)?;
            }
        }
        Ok(Self {
            hidden,
            intermediate,
            kernel_intermediate,
            experts,
            nvfp4,
            scale_codes,
            planes,
            vectors,
            bytes: cursor,
        })
    }
    pub fn expert_plane(&self, plane: usize, expert: usize) -> Result<(usize, usize)> {
        ensure!(expert < self.experts, "native expert outside plane");
        let &(offset, total) = self.planes.get(plane).context("native plane absent")?;
        let stride = total / self.experts;
        Ok((offset + expert * stride, stride))
    }
    /// One scratch allocation is reused for every plane on this rank. Weights
    /// already have their final layout for NVFP4; only its scale planes move.
    pub fn scratch_bytes(&self) -> usize {
        self.planes
            .iter()
            .enumerate()
            .filter(|(i, _)| if self.nvfp4 { i % 2 == 1 } else { !self.scale_codes || i % 2 == 0 })
            .map(|(_, (_, bytes))| bytes / self.experts)
            .max()
            .unwrap_or(0)
    }
    pub fn generated_ranges(&self) -> Vec<std::ops::Range<usize>> {
        if !self.scale_codes { return Vec::new(); }
        [1, 3].map(|i| self.planes[i].0..self.planes[i].0 + self.planes[i].1).to_vec()
    }
    pub fn scalar_bytes(&self) -> usize {
        if self.nvfp4 {
            self.experts * 24
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_is_the_actual_rank_plane_not_the_whole_layer() -> Result<()> {
        for (width, scratch) in [(2304, 11_796_480), (576, 3_276_800), (384, 1_966_080)] {
            let layout = NativePlanes::new(5120, width, width.div_ceil(128) * 128, 256, false)?;
            assert_eq!(layout.scratch_bytes(), scratch);
            assert!(layout.bytes > scratch * 256);
        }
        let nv = NativePlanes::new(5120, 576, 640, 256, true)?;
        assert_eq!(nv.scratch_bytes(), 409_600);
        assert_eq!(nv.scalar_bytes(), 6144);
        assert!(NativePlanes::new(5120, 576, 576, 256, false).is_err());
        Ok(())
    }
}
