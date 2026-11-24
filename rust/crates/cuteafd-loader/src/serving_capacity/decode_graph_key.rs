//! Shared exact-row decode graph table-geometry normalization.
//!
//! Exact-row lazy decode graph families (GLM Flash, GLM 5.3 / Mistral Large 4,
//! and future exact-row families) share two key invariants:
//! 1. Decode page-table strides round up to powers of two from [`MIN_PAGE_STRIDE`]
//!    (64 pages = 4,096 tokens), capped by the device pool and the max-context
//!    table width, so all short sequences share one table shape.
//! 2. Segments that never read a page/pool table (non-MLA layers in GLM Flash;
//!    shared-indexer attention, combine/`o_proj`, FFN/MoE, and head in GLM 5.3/ML4)
//!    and short steps that do not run long-context pool top-k zero out unused
//!    table dimensions in their graph key so short and long steps share graphs.

/// Narrowest decode page-table stride in 64-token pages (4,096 tokens): sequences
/// up to that length share one table shape and reuse the same decode graphs.
pub const MIN_PAGE_STRIDE: usize = 64;

/// Power-of-two decode page-table stride from [`MIN_PAGE_STRIDE`], capped by
/// the device pool (`pool_pages`) and a table row's columns (`table_pages`).
pub fn decode_page_stride(pages: usize, pool_pages: usize, table_pages: usize) -> usize {
    pages.max(1).next_power_of_two().max(MIN_PAGE_STRIDE).min(pool_pages).min(table_pages)
}

/// Power-of-two decode `(page_stride, pool_stride)` for two-level (page + pool)
/// tables, sharing [`decode_page_stride`] for the page table. Only a long step
/// keys the pool stride.
pub fn decode_strides(pages: usize, pool_pages: usize, pool: (usize, usize), table: (usize, usize)) -> (usize, usize) {
    (
        decode_page_stride(pages, pool.0, table.0),
        pool_pages.max(1).next_power_of_two().min(pool.1).min(table.1),
    )
}

/// Normalized decode table geometry baked into exact-row graph keys.
///
/// - Two-level families (GLM Flash) call [`Self::keyed`] (a short step keys
///   neither `pool_width` nor `pool_stride`) and [`Self::for_table`] for non-MLA
///   segments.
/// - Single-level page-table families (GLM 5.3 / Mistral Large 4) call
///   [`Self::paged`] so only segments that actually read the page table key
///   `(table_width, table_stride)` while all other segments normalize to
///   [`Self::unused`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize)]
pub struct GraphGeometry {
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
    pub long: bool,
}

impl GraphGeometry {
    /// Geometry for segments that never read a page or pool table.
    pub const fn unused() -> Self {
        Self { pool_width: 0, page_stride: 0, pool_stride: 0, long: false }
    }

    /// Two-level (page + pool) table geometry: short steps (`!long`) never run
    /// pool top-k and therefore key neither `pool_width` nor `pool_stride`.
    pub fn keyed(pool_width: usize, page_stride: usize, pool_stride: usize, long: bool) -> Self {
        if long {
            Self { pool_width, page_stride, pool_stride, long }
        } else {
            Self { pool_width: 0, page_stride, pool_stride: 0, long }
        }
    }

    /// Single-level page-table geometry (`table_width`, `table_stride`), or
    /// [`Self::unused`] when this segment does not read the page table.
    pub fn paged(table_width: usize, table_stride: usize, reads_page_table: bool) -> Self {
        Self { pool_width: table_width, page_stride: table_stride, pool_stride: 0, long: false }
            .for_table(reads_page_table)
    }

    /// Keep table geometry only on segments that read the page/pool table.
    pub fn for_table(self, reads_table: bool) -> Self {
        if reads_table { self } else { Self::unused() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_page_stride_floors_at_min_and_caps_by_pool_and_table() {
        assert_eq!(decode_page_stride(0, 2048, 4096), MIN_PAGE_STRIDE);
        assert_eq!(decode_page_stride(1, 2048, 4096), MIN_PAGE_STRIDE);
        assert_eq!(decode_page_stride(17, 2048, 4096), MIN_PAGE_STRIDE);
        assert_eq!(decode_page_stride(64, 2048, 4096), 64);
        assert_eq!(decode_page_stride(65, 2048, 4096), 128);
        assert_eq!(decode_page_stride(510, 2048, 4096), 512);
        assert_eq!(decode_page_stride(513, 2048, 4096), 1024);
        // Pool cap and table cap each clamp the stride (including below MIN_PAGE_STRIDE).
        assert_eq!(decode_page_stride(4000, 2048, 4096), 2048);
        assert_eq!(decode_page_stride(4000, 8192, 3000), 3000);
        assert_eq!(decode_page_stride(5, 32, 4096), 32);
        assert_eq!(decode_page_stride(5, 2048, 16), 16);
    }

    #[test]
    fn decode_strides_shares_page_floor_and_caps_pool_independently() {
        assert_eq!(decode_strides(5, 2, (2048, 512), (2048, 512)), (MIN_PAGE_STRIDE, 2));
        assert_eq!(decode_strides(65, 17, (2048, 512), (2048, 512)), (128, 32));
        assert_eq!(decode_strides(4000, 1000, (2048, 512), (4096, 1024)), (2048, 512));
        assert_eq!(decode_strides(5, 2, (32, 8), (2048, 512)), (32, 2));
        assert_eq!(decode_strides(5, 2, (2048, 512), (16, 4)), (16, 2));
    }

    #[test]
    fn graph_geometry_short_step_keys_neither_pool_width_nor_pool_stride() {
        let a = GraphGeometry::keyed(16, MIN_PAGE_STRIDE, 32, false);
        let b = GraphGeometry::keyed(1024, MIN_PAGE_STRIDE, 512, false);
        assert_eq!(a, b);
        assert_eq!(a, GraphGeometry { pool_width: 0, page_stride: MIN_PAGE_STRIDE, pool_stride: 0, long: false });

        let long_a = GraphGeometry::keyed(16, 512, 32, true);
        let long_b = GraphGeometry::keyed(32, 512, 32, true);
        assert_ne!(long_a, long_b);
        assert_eq!((long_a.pool_width, long_a.page_stride, long_a.pool_stride, long_a.long), (16, 512, 32, true));

        assert_eq!(long_a.for_table(false), GraphGeometry::unused());
        assert_eq!(long_a.for_table(true), long_a);
    }

    #[test]
    fn graph_geometry_paged_normalizes_non_table_segments_to_unused() {
        let short_unused = GraphGeometry::paged(64, 64, false);
        let long_unused = GraphGeometry::paged(1024, 1024, false);
        assert_eq!(short_unused, GraphGeometry::unused());
        assert_eq!(short_unused, long_unused);

        let short_used = GraphGeometry::paged(64, 64, true);
        let long_used = GraphGeometry::paged(1024, 1024, true);
        assert_ne!(short_used, long_used);
        assert_eq!((long_used.pool_width, long_used.page_stride), (1024, 1024));
    }
}
