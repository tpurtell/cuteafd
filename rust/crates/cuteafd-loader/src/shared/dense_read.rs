//! Loading-only row windows shared by native dense admission and the mount reader.
use anyhow::{ensure, Context, Result};

pub use cuteafd_ffi::SYNC_H2D_CHUNK_BYTES as STEP_BYTES;
pub const PINNED_FLOOR_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadBatch {
    pub rows: usize,
    pub packed_bytes: usize,
    pub cached_bytes: usize,
}

impl ReadBatch {
    pub fn new(rows: usize, row_bytes: usize, width: usize, limit: usize) -> Result<Self> {
        ensure!(rows > 0 && width > 0 && width <= row_bytes && limit > 0, "invalid dense read window");
        let rows = rows.min((limit / width).max(1));
        Ok(Self { rows,
            packed_bytes: rows.checked_mul(width).context("dense packed read overflow")?,
            cached_bytes: if width == row_bytes { 0 } else {
                rows.checked_mul(row_bytes).context("dense row cache overflow")?
            },
        })
    }

    pub fn pinned_bytes(self) -> usize { self.packed_bytes.max(PINNED_FLOOR_BYTES) }
}

/// Reused host owners of one mount runner; ranks and projections merge by
/// maximum because each operation drains before the next reuses its storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MountLoad {
    pub pinned_source: u64,
    pub full_row_cache: u64,
    pub pinned_tables: u64,
}

impl MountLoad {
    pub fn merge(&mut self, other: Self) {
        self.pinned_source = self.pinned_source.max(other.pinned_source);
        self.full_row_cache = self.full_row_cache.max(other.full_row_cache);
        self.pinned_tables = self.pinned_tables.max(other.pinned_tables);
    }

    pub fn read(batch: ReadBatch) -> Self {
        Self { pinned_source: batch.pinned_bytes() as u64,
            full_row_cache: batch.cached_bytes as u64, pinned_tables: 0 }
    }

    /// Header-derived upper estimate for row-window readers with at most
    /// `column_parts` equal column shares. Actual records validate before use.
    /// The caller supplies only the selected dense sources, never experts.
    pub fn from_headers<'a>(headers: impl IntoIterator<Item = &'a crate::SafetensorsTensorMetadata>,
        column_parts: usize) -> Result<Self> {
        ensure!(column_parts > 0, "dense mount needs a positive column partition count");
        let mut out = Self::default();
        for header in headers {
            let bytes = usize::try_from(header.byte_length)?;
            if bytes == 0 { continue; }
            let rows = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
            ensure!(rows > 0 && bytes % rows == 0, "{}: invalid dense source rows", header.name);
            let row_bytes = bytes / rows;
            // An indivisible source row may exceed the preferred read window.
            let packed = bytes.min(STEP_BYTES.max(row_bytes));
            let cached = bytes.min(STEP_BYTES.checked_mul(column_parts)
                .context("dense mount row bound overflow")?.max(row_bytes));
            out.merge(Self { pinned_source: packed.max(PINNED_FLOOR_BYTES) as u64,
                full_row_cache: cached as u64, pinned_tables: 0 });
            if header.dtype == cuteafd_core::DType::Bf16 && header.shape.len() == 2 && header.shape[1] >= 128 {
                // Quant jobs can concatenate source matrices in one step. The
                // narrowest selected row bounds their aggregate job count.
                let width = (row_bytes / column_parts).max(1);
                let quant_rows = (STEP_BYTES / width).max(1);
                out.pinned_source = out.pinned_source.max(STEP_BYTES as u64);
                out.pinned_tables = out.pinned_tables.max(
                    cuteafd_ffi::programs::fp8_load_table_bytes(quant_rows, 1)? as u64);
            }
        }
        Ok(out)
    }

    pub fn validate(self, actual: Self) -> Result<()> {
        ensure!(actual.pinned_source <= self.pinned_source && actual.full_row_cache <= self.full_row_cache
            && actual.pinned_tables <= self.pinned_tables,
            "dense mount staging exceeds admitted Load owners: actual {actual:?}, admitted {self:?}");
        Ok(())
    }

    pub fn allocations(self, prefix: &str) -> Vec<crate::placement::planner::Allocation> {
        use crate::placement::planner as p;
        use cuteafd_core::memory_layout::Category;
        let load = p::Lifetime { first: p::Phase::Load, last: p::Phase::Load };
        [("source", self.pinned_source, p::Location::PINNED),
         ("rows", self.full_row_cache, p::Location::HOST),
         ("tables", self.pinned_tables, p::Location::PINNED)].into_iter()
            .filter(|(_, bytes, _)| *bytes > 0).map(|(name, bytes, location)|
                p::Allocation::buffer(format!("{prefix}.{name}"), location, bytes, load,
                    Category::Staging, "header-bounded reused dense mount reader storage")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mount_load_estimates_selected_headers_and_keeps_owners_load_only() -> Result<()> {
        use crate::placement::planner as p;
        let header = crate::SafetensorsTensorMetadata { name: "o_proj.weight".into(),
            dtype: cuteafd_core::DType::Bf16, shape: vec![4096, 16384],
            byte_offset: 0, byte_length: 4096 * 16384 * 2 };
        let estimate = MountLoad::from_headers([&header], 2)?;
        assert_eq!(estimate, MountLoad { pinned_source: STEP_BYTES as u64,
            full_row_cache: (2 * STEP_BYTES) as u64, pinned_tables: 1 << 20 });
        for width in [16384, 32768] {
            estimate.validate(MountLoad::read(ReadBatch::new(4096, 32768, width, STEP_BYTES)?))?;
        }
        let mut combined = estimate;
        combined.merge(estimate);
        assert_eq!(combined, estimate);
        let owners = estimate.allocations("test.mount");
        assert_eq!(owners.len(), 3);
        assert!(owners.iter().all(|owner| matches!(owner.location, p::Location::HOST | p::Location::PINNED)
            && owner.lifetime == p::Lifetime { first: p::Phase::Load, last: p::Phase::Load }));
        for field in 0..3 {
            let mut actual = estimate;
            match field { 0 => actual.pinned_source += 1, 1 => actual.full_row_cache += 1,
                _ => actual.pinned_tables += 1 }
            assert!(estimate.validate(actual).is_err());
        }
        assert!(MountLoad::from_headers([&header], 0).is_err());
        Ok(())
    }

    #[test]
    fn partial_columns_charge_full_rows_for_the_packed_batch() {
        let batch = ReadBatch::new(16384, 8192, 4096, STEP_BYTES).unwrap();
        assert_eq!(batch.rows, 8192);
        assert_eq!(batch.packed_bytes, STEP_BYTES);
        assert_eq!(batch.cached_bytes, 2 * STEP_BYTES);
        assert_eq!(ReadBatch::new(16384, 8192, 8192, STEP_BYTES).unwrap().cached_bytes, 0);
        assert!(ReadBatch::new(1, 1, 0, STEP_BYTES).is_err());
        assert!(ReadBatch::new(2, usize::MAX, 1, usize::MAX).is_err());
    }
}
