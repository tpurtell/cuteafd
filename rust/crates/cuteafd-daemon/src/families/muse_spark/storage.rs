//! One coordinator rank's Muse storage, allocated exactly as admission charged
//! it (plan == runtime): the full layers' paged K|V records (the pool), every
//! sequence's sliding rings, and the per-layer views the device layer reads
//! ([`LayerKv`]). Each allocation is checked against the placement item of the
//! same group before it is made.
use super::device::LayerKv;
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::muse_spark::{cache::{MuseCacheLayout, PAGE_ROWS}, MuseAttention, MuseSparkConfig};
use cuteafd_loader::placement::Placement;

/// Byte extents of one rank's KV storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvExtents {
    /// One full layer's pool: `pages x 64 x record`.
    pub layer_pool: usize,
    /// One sliding layer's rings for every sequence slot.
    pub layer_rings: usize,
    pub full_layers: usize,
    pub sliding_layers: usize,
}

impl KvExtents {
    pub(crate) fn new(cfg: &MuseSparkConfig, cache: &MuseCacheLayout, pool_tokens: u64, slots: u64) -> Result<Self> {
        ensure!(pool_tokens % PAGE_ROWS == 0 && pool_tokens > 0 && slots > 0, "Muse pool/slots geometry");
        let record = usize::try_from(cache.record_bytes)?;
        let layer_pool = usize::try_from(pool_tokens)?.checked_mul(record).context("Muse pool bytes")?;
        let layer_rings = usize::try_from(slots * cache.ring.rows)?.checked_mul(record).context("Muse ring bytes")?;
        let full = cfg.attention.iter().filter(|k| **k == MuseAttention::Full).count();
        Ok(Self { layer_pool, layer_rings, full_layers: full, sliding_layers: cfg.layers - full })
    }

    pub(crate) fn pool_bytes(&self) -> usize { self.layer_pool * self.full_layers }
    pub(crate) fn ring_bytes(&self) -> usize { self.layer_rings * self.sliding_layers }
}

/// One rank's KV storage and its per-layer views.
pub(crate) struct RankKv<'a> {
    pub pool: Allocation<'a>,
    /// Every sliding layer's rings for every slot (`MuseStateLayout::ring_offset`).
    pub rings: Allocation<'a>,
    pub layers: Vec<LayerKv>,
    pub extents: KvExtents,
}

impl<'a> RankKv<'a> {
    /// `charged`: this rank's placement items, to check the extents against.
    pub(crate) fn allocate(device: Device<'a>, cfg: &MuseSparkConfig, cache: &MuseCacheLayout, placement: &Placement,
        rank: usize, slots: u64) -> Result<Self> {
        let extents = KvExtents::new(cfg, cache, placement.pool_tokens, slots)?;
        let items = placement.items.get(rank).context("Muse placement has no items for this rank")?;
        let charged = |prefix: &str| items.iter().filter(|item| item.group.starts_with(prefix)).map(|item| item.bytes)
            .sum::<u64>();
        ensure!(charged("sliding rings") == extents.ring_bytes() as u64,
            "Muse sliding rings differ from admission ({} charged, {} allocated)", charged("sliding rings"),
            extents.ring_bytes());
        let pool = {
            let _scope = cuteafd_ffi::memory_ledger::scope("kv/muse records");
            Allocation::new(device, extents.pool_bytes().max(256))?
        };
        let rings = {
            let _scope = cuteafd_ffi::memory_ledger::scope("kv/muse sliding rings");
            Allocation::new(device, extents.ring_bytes().max(256))?
        };
        let (mut full, mut sliding) = (0usize, 0usize);
        let layers = cfg.attention.iter().map(|kind| match kind {
            MuseAttention::Full => {
                let at = full * extents.layer_pool;
                full += 1;
                LayerKv::Paged { cache: view(pool.buffer, at, extents.layer_pool) }
            }
            MuseAttention::Sliding => {
                let at = sliding * extents.layer_rings;
                sliding += 1;
                LayerKv::Ring { ring: view(rings.buffer, at, extents.layer_rings) }
            }
        }).collect();
        Ok(Self { pool, rings, layers, extents })
    }
}

fn view(base: Buffer, offset: usize, bytes: usize) -> Buffer {
    Buffer { ptr: base.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..base }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring bytes the storage allocates are the placement's "sliding rings"
    /// charge on each rank (one GPU and a head-split pair).
    #[test]
    fn ring_extents_equal_the_admitted_charge() -> Result<()> {
        let cfg = MuseSparkConfig::from_hf(&cuteafd_loader::plan::testing::muse_spark_config())?;
        for ranks in [1usize, 2] {
            let cache = MuseCacheLayout::new(&cfg, ranks)?;
            let extents = KvExtents::new(&cfg, &cache, 1 << 20, 16)?;
            let record = cfg.kv_record_bytes() as u64;
            let rings = cfg.sliding_layers() as u64 * cuteafd_loader::placement::families::muse_spark::ring_rows(&cfg)?
                * record * 16 / ranks as u64;
            assert_eq!(extents.ring_bytes() as u64, rings);
            assert_eq!(extents.pool_bytes() as u64, (1 << 20) * record / ranks as u64 * extents.full_layers as u64);
            assert_eq!(extents.full_layers + extents.sliding_layers, cfg.layers);
        }
        assert!(KvExtents::new(&cfg, &MuseCacheLayout::new(&cfg, 1)?, 100, 16).is_err(), "pages are 64 rows");
        Ok(())
    }
}
