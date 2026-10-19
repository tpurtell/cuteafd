//! Family caller for W7's native NoPE MLA chain. W9 supplies the owning
//! allocations, launch metadata and TP2 partial-sum exchange.
use anyhow::{ensure, Context, Result};
use crate::shared::{memory::device::{Allocation, Device, Stream}, placement::layers::AttentionShare, prefix::view};
use cuteafd_core::prefix::ReuseRule;
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer, KimiMla, KimiMlaBatch, KimiMlaMode, KimiMlaWeights, MLA_PAGE_ROWS, MLA_RECORD_BYTES};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::{FfnMode, LayerMode, Placement}};
use std::{ffi::c_void, rc::Rc};

/// MLA pages alone could resume at a page boundary, but KDA recurrent state
/// belongs to the captured frontier. The whole family must restore exactly.
pub(crate) const PREFIX_REUSE: ReuseRule = ReuseRule::EXACT;

pub(crate) struct Mla<'a> {
    native: KimiMla<'a>,
    layers: Vec<KimiAttention>,
}

impl<'a> Mla<'a> {
    pub(crate) fn load(cfg: &KimiK3Config, programs: &'a Programs<'a>, heads: usize,
                       capacity: usize, mode: KimiMlaMode) -> Result<Self> {
        ensure!(cfg.hidden == 7168 && cfg.heads == 96 && cfg.q_lora_rank == 1536
            && cfg.kv_lora_rank == 512 && cfg.qk_nope_dim == 128 && cfg.qk_rope_dim == 64
            && cfg.v_head_dim == 128, "unsupported Kimi native MLA geometry");
        ensure!(cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi MLA cache accounting mismatch");
        Ok(Self { native: KimiMla::load(programs, heads, capacity, mode)?, layers: cfg.attention.clone() })
    }

    pub(crate) fn scratch_bytes(&self) -> [usize; 3] { self.native.scratch_bytes() }
    pub(crate) fn max_context(&self) -> usize { self.native.max_context() }
    pub(crate) fn splits(&self) -> usize { self.native.splits() }

    /// Output is head-local at 48 heads and complete at 96 heads. This call
    /// never applies the TP2 meet twice or substitutes an ungated projection.
    pub(crate) fn forward(&self, layer: usize, weights: KimiMlaWeights,
                          batch: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        ensure!(self.layers.get(layer) == Some(&KimiAttention::Mla), "layer {layer} is not Kimi MLA");
        self.native.launch(weights, batch, stream)?;
        Ok(())
    }
}

/// Per-layer cache residency comes from accepted attention placement, never
/// alternating-layer assumptions. Head split intentionally keeps both copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheLayout {
    layers: Vec<Option<AttentionShare>>,
    pages: usize,
    layer_bytes: usize,
    bytes: usize,
}

impl CacheLayout {
    pub(crate) fn new(cfg: &KimiK3Config, placement: &Placement, rank: usize) -> Result<Self> {
        ensure!(rank < 2 && placement.layers.len() == cfg.layers && cfg.attention.len() == cfg.layers,
            "Kimi MLA cache placement inventory/rank");
        ensure!(cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi MLA cache record geometry");
        ensure!(cfg.attention.iter().zip(&placement.layers).all(|(kind, assignment)| match kind {
            KimiAttention::Mla => matches!(assignment.mode, LayerMode::HeadSplit
                | LayerMode::Whole { gpu: 0 | 1, ffn: FfnMode::Split }),
            KimiAttention::Kda => assignment.mode == LayerMode::HeadSplit,
        }), "Kimi MLA cache placement has no executable attention mode");
        let tokens = usize::try_from(placement.pool_tokens)?;
        ensure!(tokens > 0 && tokens % MLA_PAGE_ROWS == 0, "Kimi MLA cache pool is not whole pages");
        let pages = tokens / MLA_PAGE_ROWS;
        ensure!(pages <= i32::MAX as usize, "Kimi MLA page count exceeds native ABI");
        let layer_bytes = tokens.checked_mul(MLA_RECORD_BYTES).context("Kimi MLA cache extent overflow")?;
        let layers = cfg.attention.iter().zip(&placement.layers).map(|(kind, assignment)| -> Result<_> {
            if *kind != KimiAttention::Mla { return Ok(None); }
            let share = AttentionShare::bind(assignment.mode, 2)?[rank];
            Ok(share.present().then_some(share))
        }).collect::<Result<Vec<_>>>()?;
        let bytes = layer_bytes.checked_mul(layers.iter().flatten().count()).context("Kimi MLA cache total overflow")?;
        Ok(Self { layers, pages, layer_bytes, bytes })
    }

    pub(crate) fn bytes(&self) -> usize { self.bytes }

    pub(super) fn page_range(&self, layer: usize, page: usize, rows: usize) -> Result<(usize, usize)> {
        ensure!(self.layers.get(layer).is_some_and(Option::is_some), "Kimi MLA cache absent on nonowner or non-MLA layer");
        ensure!(page < self.pages && (1..=MLA_PAGE_ROWS).contains(&rows),
            "Kimi MLA prefix page/rows outside admitted pool");
        Ok((page * MLA_PAGE_ROWS * MLA_RECORD_BYTES, rows * MLA_RECORD_BYTES))
    }

    pub(super) fn allocate<'a>(&self, device: Device<'a>, readers: Vec<Rc<Stream<'a>>>) -> Result<CacheBank<'a>> {
        ensure!(!readers.is_empty() && readers.len() <= 2 && readers.iter().all(|stream|
            stream.device.id == device.id && std::ptr::eq(stream.device.library, device.library)),
            "Kimi MLA cache readers differ from admitted device/lanes");
        ensure!(readers.iter().enumerate().all(|(i, stream)| readers[..i].iter()
            .all(|previous| !Rc::ptr_eq(previous, stream))), "Kimi MLA lanes share a stream owner");
        let stream = Stream::new(device)?;
        let _scope = cuteafd_ffi::memory_ledger::scope("kv/kimi MLA pages");
        let layers = self.layers.iter().map(|share| -> Result<_> {
            if share.is_none() { return Ok(None); }
            let allocation = Rc::new(Allocation::new(device, self.layer_bytes)?);
            ensure!(allocation.buffer.bytes == self.layer_bytes && allocation.buffer.device_id == device.id,
                "Kimi MLA cache allocation differs from admitted extent/device");
            Ok(Some(allocation))
        }).collect::<Result<Vec<_>>>()?;
        Ok(CacheBank { stream, readers, layers, layout: self.clone() })
    }
}

/// Allocation owners are retained by prefix copies and graph owners as needed;
/// all compute readers and the cache-copy stream drain before bank destruction.
pub(crate) struct CacheBank<'a> {
    pub stream: Stream<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    layers: Vec<Option<Rc<Allocation<'a>>>>,
    layout: CacheLayout,
}

impl Drop for CacheBank<'_> {
    fn drop(&mut self) {
        for stream in &self.readers { stream.drain_or_abort("Kimi MLA lane reader"); }
        self.stream.drain_or_abort("Kimi MLA prefix copies");
    }
}

impl<'a> CacheBank<'a> {
    pub(crate) fn layer(&self, layer: usize) -> Result<CuteafdDeviceBuffer> {
        self.layers.get(layer).and_then(Option::as_ref).map(|owner| owner.buffer)
            .context("Kimi MLA cache absent on nonowner or non-MLA layer")
    }

    pub(crate) fn page(&self, layer: usize, page: usize, rows: usize) -> Result<CuteafdDeviceBuffer> {
        let (offset, bytes) = self.layout.page_range(layer, page, rows)?;
        view(self.layer(layer)?, offset, bytes)
    }

    pub(crate) fn pages(&self) -> usize { self.layout.pages }

    pub(crate) fn owners(&self) -> Vec<Rc<Allocation<'a>>> { self.layers.iter().flatten().cloned().collect() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn cache_destruction_drains_all_lane_readers_before_page_storage() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let readers = (0..2).map(|_| Rc::new(Stream { device, raw: std::ptr::null_mut() })).collect::<Vec<_>>();
        let bytes = MLA_PAGE_ROWS * MLA_RECORD_BYTES;
        let bank = CacheBank { stream: Stream { device, raw: std::ptr::null_mut() }, readers: readers.clone(),
            layers: vec![None, Some(Rc::new(Allocation::new(device, bytes)?))],
            layout: CacheLayout { layers: vec![None, Some(AttentionShare::Whole)], pages: 1, layer_bytes: bytes, bytes } };
        assert!(bank.layer(0).is_err());
        assert!(bank.layer(2).is_err());
        assert_eq!(bank.pages(), 1);
        assert_eq!(bank.owners().len(), 1);
        let page = bank.page(1, 0, MLA_PAGE_ROWS)?;
        assert_eq!(page.ptr, bank.layer(1)?.ptr);
        assert_eq!(page.bytes, bytes);
        assert!(bank.page(1, 1, 1).is_err());
        drop(readers);
        drop(bank);
        let events = fixture.events()?;
        let first_free = events.find('d').context("Kimi MLA fixture released no storage")?;
        assert_eq!(events[..first_free].matches('S').count(), 6,
            "both compute readers and copy stream drain before the first page free: {events}");
        assert_eq!(events.matches('d').count(), 1);
        Ok(())
    }

    #[test]
    fn native_cache_bytes_and_recurrent_prefix_policy() {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        assert_eq!(cfg.mla_record_values() * 2, MLA_RECORD_BYTES);
        assert_eq!(cfg.mla_layers() * MLA_RECORD_BYTES, 27 * 1024);
        assert!(!PREFIX_REUSE.partial());
        assert_eq!(PREFIX_REUSE.skipped(65, 65), 65);
        assert_eq!(PREFIX_REUSE.skipped(64, 65), 0);
        assert_eq!(PREFIX_REUSE.skipped(65, 128), 0);
    }
}
