//! Family caller for W7's native NoPE MLA chain. W9 supplies the owning
//! allocations, launch metadata and TP2 partial-sum exchange.
use cuteafd_core::MlaCacheFormat;
use anyhow::{ensure, Context, Result};
use crate::shared::{memory::{device::{Allocation, Device, Stream}, Tag}, placement::layers::AttentionShare, prefix::view};
use cuteafd_core::prefix::ReuseRule;
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer, KimiMla, KimiMlaBatch, KimiMlaMode, KimiMlaWeights, MLA_PAGE_ROWS, MLA_RECORD_BYTES};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::{FfnMode, LayerMode, Placement}};
use std::{ffi::c_void, rc::Rc};

/// MLA pages alone could resume at a page boundary, but KDA recurrent state
/// belongs to the captured frontier. The whole family must restore exactly.
pub(crate) const PREFIX_REUSE: ReuseRule = ReuseRule::EXACT;

pub(crate) struct Mla<'a> {
    native: MlaNative<'a>,
    layers: Vec<KimiAttention>,
}

enum MlaNative<'a> {
    Dense(KimiMla<'a>),
    Sparse { projection: cuteafd_ffi::KimiMlaProjection<'a>, attention: cuteafd_ffi::KimiSparseMla<'a> },
}

impl<'a> Mla<'a> {
    pub(crate) fn load(cfg: &KimiK3Config, programs: &'a Programs<'a>, heads: usize,
                       capacity: usize, mode: KimiMlaMode, cache_format: MlaCacheFormat) -> Result<Self> {
        ensure!(cfg.hidden == 7168 && cfg.heads == 96 && cfg.q_lora_rank == 1536
            && cfg.kv_lora_rank == 512 && cfg.qk_nope_dim == 128 && cfg.qk_rope_dim == 64
            && cfg.v_head_dim == 128, "unsupported Kimi native MLA geometry");
        ensure!(cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi MLA cache accounting mismatch");
        let native = if cfg.dsa.is_some() {
            MlaNative::Sparse { projection: cuteafd_ffi::KimiMlaProjection::load_k31(programs, heads, capacity, cache_format)?,
                attention: cuteafd_ffi::KimiSparseMla::load(programs, heads, capacity, cache_format, None)? }
        } else { MlaNative::Dense(KimiMla::load(programs, heads, capacity, mode, cache_format)?) };
        Ok(Self { native, layers: cfg.attention.clone() })
    }

    pub(crate) fn scratch_bytes(&self) -> [usize; 3] {
        match &self.native {
            MlaNative::Dense(native) => native.scratch_bytes(),
            MlaNative::Sparse { projection, attention } => {
                let [producer, output] = projection.scratch_bytes();
                [producer, attention.scratch_bytes(), output]
            }
        }
    }
    pub(crate) fn max_context(&self) -> usize {
        match &self.native { MlaNative::Dense(native) => native.max_context(),
            MlaNative::Sparse { .. } => cuteafd_loader::families::kimi_k3::COMPILED_CONTEXT }
    }
    pub(crate) fn splits(&self) -> usize {
        match &self.native { MlaNative::Dense(native) => native.splits(), MlaNative::Sparse { .. } => 16 }
    }

    pub(crate) fn produce(&self, layer: usize, weights: KimiMlaWeights,
        batch: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        ensure!(self.layers.get(layer) == Some(&KimiAttention::Mla), "layer {layer} is not Kimi MLA");
        match &self.native {
            MlaNative::Dense(native) => native.produce(weights, batch, stream)?,
            MlaNative::Sparse { projection, .. } => projection.produce(weights, batch, stream)?,
        }
        Ok(())
    }

    pub(crate) fn output(&self, layer: usize, weights: KimiMlaWeights,
        batch: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        ensure!(self.layers.get(layer) == Some(&KimiAttention::Mla), "layer {layer} is not Kimi MLA");
        match &self.native {
            MlaNative::Dense(native) => native.output(weights, batch, stream)?,
            MlaNative::Sparse { projection, .. } => projection.output(weights, batch, stream)?,
        }
        Ok(())
    }

    pub(crate) fn attention(&self,layer:usize,batch:KimiMlaBatch,stream:*mut c_void)->Result<()> {
        ensure!(self.layers.get(layer)==Some(&KimiAttention::Mla),"layer {layer} is not Kimi MLA");
        match &self.native {
            MlaNative::Dense(native) => native.attention(batch, stream)?,
            MlaNative::Sparse { .. } => anyhow::bail!("Kimi K3.1 requires causal index selection before MLA"),
        }
        Ok(())
    }

    pub(crate) fn selected(&self, layer: usize, batch: cuteafd_ffi::KimiSparseBatch,
        stream: *mut c_void) -> Result<()> {
        ensure!(self.layers.get(layer) == Some(&KimiAttention::Mla), "layer {layer} is not Kimi MLA");
        match &self.native {
            MlaNative::Sparse { attention, .. } => attention.launch(batch, stream)?,
            MlaNative::Dense(_) => anyhow::bail!("Kimi K3 dense MLA cannot consume DSA token slots"),
        }
        Ok(())
    }

    /// Output is head-local at 48 heads and complete at 96 heads. This call
    /// never applies the TP2 meet twice or substitutes an ungated projection.
    pub(crate) fn forward(&self, layer: usize, weights: KimiMlaWeights,
                          batch: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        ensure!(self.layers.get(layer) == Some(&KimiAttention::Mla), "layer {layer} is not Kimi MLA");
        match &self.native {
            MlaNative::Dense(native) => native.launch(weights, batch, stream)?,
            MlaNative::Sparse { .. } => anyhow::bail!("Kimi K3.1 requires the shared index/selected-MLA chain"),
        }
        Ok(())
    }
}

/// Per-layer cache residency comes from accepted attention placement, never
/// alternating-layer assumptions. Head split intentionally keeps both copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheLayout {
    format: MlaCacheFormat,
    layers: Vec<Option<AttentionShare>>,
    pages: usize,
    rank: usize,
    parity: bool,
    layer_bytes: usize,
    index_layer_bytes: usize,
    bytes: usize,
}

impl CacheLayout {
    pub(crate) fn new(cfg: &KimiK3Config, placement: &Placement, rank: usize, format: MlaCacheFormat) -> Result<Self> {
        Self::for_ranks(cfg, placement, rank, format, 2)
    }

    pub(crate) fn for_ranks(cfg: &KimiK3Config, placement: &Placement, rank: usize,
        format: MlaCacheFormat, ranks: usize) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2) && rank < ranks
            && placement.layers.len() == cfg.layers && cfg.attention.len() == cfg.layers,
            "Kimi MLA cache placement inventory/rank");
        ensure!(cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi MLA cache record geometry");
        let parity = placement.layers.iter().any(|layer| layer.mode == LayerMode::ContextSplit);
        ensure!(cfg.attention.iter().zip(&placement.layers).all(|(kind, assignment)| {
            if ranks == 1 {
                assignment.mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }
            } else {
                match kind {
                    KimiAttention::Mla if parity => assignment.mode == LayerMode::ContextSplit,
                    KimiAttention::Mla => matches!(assignment.mode, LayerMode::HeadSplit
                        | LayerMode::Whole { gpu: 0 | 1, ffn: FfnMode::Split }),
                    KimiAttention::Kda => assignment.mode == LayerMode::HeadSplit,
                }
            }
        }), "Kimi MLA cache placement has no executable attention mode");
        let tokens = usize::try_from(placement.pool_tokens)?;
        ensure!(tokens > 0 && tokens % MLA_PAGE_ROWS == 0, "Kimi MLA cache pool is not whole pages");
        let pages = tokens / MLA_PAGE_ROWS;
        ensure!(pages <= i32::MAX as usize, "Kimi MLA page count exceeds native ABI");
        ensure!(!parity || pages % 2 == 0, "Kimi context pool needs equal whole-page shards");
        let local_tokens = if parity { tokens / 2 } else { tokens };
        let layer_bytes = local_tokens.checked_mul(format.record_bytes()).context("Kimi MLA cache extent overflow")?;
        let layers = cfg.attention.iter().zip(&placement.layers).map(|(kind, assignment)| -> Result<_> {
            if *kind != KimiAttention::Mla { return Ok(None); }
            let share = AttentionShare::bind(assignment.mode, ranks)?[rank];
            Ok(share.present().then_some(share))
        }).collect::<Result<Vec<_>>>()?;
        // Index records follow the same logical-page parity as the latent cache:
        // each rank scores its own half and the shared merge joins the runs.
        let index_layer_bytes = local_tokens.checked_mul(cfg.dsa.map_or(0, |index| index.head_dim + 4))
            .context("Kimi index cache extent overflow")?;
        let bytes = layer_bytes.checked_add(index_layer_bytes).and_then(|per_layer|
            per_layer.checked_mul(layers.iter().flatten().count())).context("Kimi MLA cache total overflow")?;
        Ok(Self { format, layers, pages, rank, parity, layer_bytes, index_layer_bytes, bytes })
    }

    pub(crate) fn bytes(&self) -> usize { self.bytes }

    fn backing_bytes(bytes: usize) -> u64 {
        use cuteafd_loader::placement::planner::{SM120_ALLOCATION_GRANULARITY_BYTES as G,
            SM120_SHARED_REQUEST_BYTES as SMALL};
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        if bytes <= SMALL { bytes } else { bytes.div_ceil(G).saturating_mul(G) }
    }

    pub(crate) fn mla_reserved_bytes(&self) -> u64 {
        Self::backing_bytes(self.layer_bytes).saturating_mul(self.layers.iter().flatten().count() as u64)
    }

    pub(crate) fn index_reserved_bytes(&self) -> u64 {
        if self.index_layer_bytes == 0 { return 0; }
        Self::backing_bytes(self.index_layer_bytes).saturating_mul(self.layers.iter().flatten().count() as u64)
    }

    /// Physical residency of the cache: every present layer is one cudaMalloc
    /// of `layer_bytes` (plus one `index_layer_bytes` backing on DSA layers),
    /// rounded up to the SM120 allocation granularity (a request at or below
    /// the shared-block threshold costs its own bytes).
    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.mla_reserved_bytes().saturating_add(self.index_reserved_bytes())
    }

    pub(super) fn page_range(&self, layer: usize, page: usize, rows: usize) -> Result<(usize, usize)> {
        ensure!(self.layers.get(layer).is_some_and(Option::is_some), "Kimi MLA cache absent on nonowner or non-MLA layer");
        ensure!(page < self.pages && (1..=MLA_PAGE_ROWS).contains(&rows),
            "Kimi MLA prefix page/rows outside admitted pool");
        ensure!(!self.parity || page % 2 == self.rank, "Kimi context page belongs to the other rank");
        let local_page = if self.parity { page >> 1 } else { page };
        Ok((local_page * MLA_PAGE_ROWS * self.format.record_bytes(), rows * self.format.record_bytes()))
    }

    pub(super) fn allocate<'a>(&self, device: Device<'a>, readers: Vec<Rc<Stream<'a>>>) -> Result<CacheBank<'a>> {
        ensure!(!readers.is_empty() && readers.len() <= 2 && readers.iter().all(|stream|
            stream.device.id == device.id && std::ptr::eq(stream.device.library, device.library)),
            "Kimi MLA cache readers differ from admitted device/lanes");
        ensure!(readers.iter().enumerate().all(|(i, stream)| readers[..i].iter()
            .all(|previous| !Rc::ptr_eq(previous, stream))), "Kimi MLA lanes share a stream owner");
        let stream = Stream::new(device)?;
        let layers = self.layers.iter().map(|share| -> Result<_> {
            if share.is_none() { return Ok(None); }
            let allocation = Rc::new(Allocation::new(device, self.layer_bytes, Tag::kv("kimi/mla-pages"))?);
            ensure!(allocation.buffer.bytes == self.layer_bytes && allocation.buffer.device_id == device.id,
                "Kimi MLA cache allocation differs from admitted extent/device");
            Ok(Some(allocation))
        }).collect::<Result<Vec<_>>>()?;
        let indices = self.layers.iter().map(|share| -> Result<_> {
            if share.is_none() || self.index_layer_bytes == 0 { return Ok(None); }
            Ok(Some(Rc::new(Allocation::new(device, self.index_layer_bytes, Tag::kv("kimi/index-pages"))?)))
        }).collect::<Result<Vec<_>>>()?;
        Ok(CacheBank { stream, readers, layers, indices, layout: self.clone() })
    }
}

/// Allocation owners are retained by prefix copies and graph owners as needed;
/// all compute readers and the cache-copy stream drain before bank destruction.
pub(crate) struct CacheBank<'a> {
    pub stream: Stream<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    layers: Vec<Option<Rc<Allocation<'a>>>>,
    indices: Vec<Option<Rc<Allocation<'a>>>>,
    layout: CacheLayout,
}

impl Drop for CacheBank<'_> {
    fn drop(&mut self) {
        for stream in &self.readers { stream.drain_or_abort("Kimi MLA lane reader"); }
        self.stream.drain_or_abort("Kimi MLA prefix copies");
    }
}

impl<'a> CacheBank<'a> {
    pub(super) fn check_reader(&self, lane: usize, stream: &Rc<Stream<'a>>) -> Result<()> {
        ensure!(self.readers.get(lane).is_some_and(|reader| Rc::ptr_eq(reader, stream))
            && self.stream.device.id == stream.device.id
            && std::ptr::eq(self.stream.device.library, stream.device.library),
            "Kimi MLA bank differs from admitted lane stream");
        Ok(())
    }

    pub(crate) fn layer(&self, layer: usize) -> Result<CuteafdDeviceBuffer> {
        self.layers.get(layer).and_then(Option::as_ref).map(|owner| owner.buffer)
            .context("Kimi MLA cache absent on nonowner or non-MLA layer")
    }

    pub(crate) fn page(&self, layer: usize, page: usize, rows: usize) -> Result<CuteafdDeviceBuffer> {
        let (offset, bytes) = self.layout.page_range(layer, page, rows)?;
        view(self.layer(layer)?, offset, bytes)
    }

    pub(crate) fn index(&self, layer: usize) -> Result<CuteafdDeviceBuffer> {
        self.indices.get(layer).and_then(Option::as_ref).map(|owner| owner.buffer)
            .context("Kimi index cache absent on nonowner or non-DSA layer")
    }

    pub(crate) fn page_segments(&self, layer: usize, page: usize, rows: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(page < self.layout.pages && (1..=MLA_PAGE_ROWS).contains(&rows), "Kimi index prefix page/rows outside admission");
        let mut segments = Vec::new();
        if self.owns_page(page as u32) { segments.push(self.page(layer, page, rows)?); }
        if self.layout.index_layer_bytes != 0 && self.owns_page(page as u32) {
            let cache = self.index(layer)?;
            let local_page = if self.layout.parity { page >> 1 } else { page };
            // Keys and scales follow the same parity owner as the MLA record.
            segments.push(view(cache, local_page * 8448, rows * 128)?);
            segments.push(view(cache, local_page * 8448 + 8192, rows * 4)?);
        }
        Ok(segments)
    }

    pub(crate) fn index_page_bytes(&self) -> usize { if self.layout.index_layer_bytes == 0 { 0 } else { 8448 } }

    pub(crate) fn format(&self) -> MlaCacheFormat { self.layout.format }

    pub(crate) fn pages(&self) -> usize { self.layout.pages }

    pub(crate) fn local_pages(&self) -> usize {
        if self.layout.parity { self.layout.pages / 2 } else { self.layout.pages }
    }

    pub(crate) fn parity(&self) -> bool { self.layout.parity }

    pub(crate) fn owns_page(&self, page: u32) -> bool {
        !self.layout.parity || page as usize % 2 == self.layout.rank
    }

    pub(crate) fn owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.layers.iter().chain(&self.indices).flatten().cloned().collect()
    }
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
            layers: vec![None, Some(Rc::new(Allocation::new(device, bytes, crate::shared::memory::TEST)?))],
            indices: vec![None, None],
            layout: CacheLayout { format: MlaCacheFormat::Bf16, layers: vec![None, Some(AttentionShare::Whole)], pages: 1, rank: 0, parity: false, layer_bytes: bytes, index_layer_bytes: 0, bytes } };
        assert!(bank.layer(0).is_err());
        assert!(bank.layer(2).is_err());
        assert_eq!(bank.pages(), 1);
        assert_eq!(bank.owners().len(), 1);
        let page = bank.page(1, 0, MLA_PAGE_ROWS)?;
        assert_eq!(page.ptr, bank.layer(1)?.ptr);
        assert_eq!(page.bytes, bytes);
        assert!(bank.page(1, 1, 1).is_err());
        for (lane, stream) in readers.iter().enumerate() {
            bank.check_reader(lane, stream)?;
            assert!(bank.check_reader(1 - lane, stream).is_err());
        }
        assert!(bank.check_reader(2, &readers[0]).is_err());
        let impostor = Rc::new(Stream { device, raw: std::ptr::null_mut() });
        assert!(bank.check_reader(0, &impostor).is_err());
        drop(impostor);
        let bank = Rc::new(bank);
        let registered = Rc::downgrade(&bank);
        let retained = bank.clone();
        drop(bank);
        assert!(registered.upgrade().is_some());
        let before = fixture.events()?;
        assert!(!before.contains('d'));
        drop(readers);
        drop(retained);
        assert!(registered.upgrade().is_none());
        let events = fixture.events()?;
        let first_free = events.find('d').context("Kimi MLA fixture released no storage")?;
        assert_eq!(events[before.len()..first_free].matches('S').count(), 6,
            "both compute readers and copy stream drain before the first page free: {events}");
        assert_eq!(events.matches('d').count(), 1);
        Ok(())
    }

    #[test]
    fn context_pages_have_one_physical_owner_and_compact_offsets() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut placement = super::super::admission::admit(super::super::admission::tests::inputs(&cfg))?.placement;
        for (kind, layer) in cfg.attention.iter().zip(&mut placement.layers) {
            if *kind == KimiAttention::Mla { layer.mode = LayerMode::ContextSplit; }
        }
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            let shards = [CacheLayout::new(&cfg, &placement, 0, format)?,
                CacheLayout::new(&cfg, &placement, 1, format)?];
            assert!(shards.iter().all(|shard| shard.parity));
            assert_eq!(shards.iter().map(CacheLayout::bytes).sum::<usize>(),
                placement.pool_tokens as usize * format.record_bytes() * cfg.mla_layers());
            let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Mla).unwrap();
            for page in [0, 1, 2, 3, shards[0].pages - 2, shards[0].pages - 1] {
                let owner = page % 2;
                let (offset, bytes) = shards[owner].page_range(layer, page, 17)?;
                assert_eq!(offset, (page >> 1) * MLA_PAGE_ROWS * format.record_bytes());
                assert_eq!(bytes, 17 * format.record_bytes());
                assert!(offset + bytes <= shards[owner].layer_bytes);
                assert!(shards[1 - owner].page_range(layer, page, 17).is_err());
            }
            let mut mixed = placement.clone();
            mixed.layers[layer].mode = LayerMode::HeadSplit;
            assert!(CacheLayout::new(&cfg, &mixed, 0, format).is_err());
            let mut odd = placement.clone();
            odd.pool_tokens = 3 * MLA_PAGE_ROWS as u64;
            assert!(CacheLayout::new(&cfg, &odd, 0, format).is_err());
        }
        Ok(())
    }

    #[test]
    fn k31_context_index_cache_follows_page_parity() -> Result<()> {
        use cuteafd_loader::families::kimi_k3::config::KimiDsaConfig;
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut placement = super::super::admission::admit(super::super::admission::tests::inputs(&cfg))?.placement;
        cfg.dsa = Some(KimiDsaConfig { heads: 32, head_dim: 128, topk: 2048 });
        for (kind, layer) in cfg.attention.iter().zip(&mut placement.layers) {
            if *kind == KimiAttention::Mla { layer.mode = LayerMode::ContextSplit; }
        }
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            for rank in 0..2 {
                let layout = CacheLayout::new(&cfg, &placement, rank, format)?;
                // Index records follow the same logical-page parity as the
                // latent cache: each rank owns and is charged for half.
                assert_eq!(layout.index_layer_bytes, placement.pool_tokens as usize / 2 / 64 * 8448);
                assert_eq!(layout.layer_bytes, placement.pool_tokens as usize / 2 * format.record_bytes());
                assert_eq!(layout.bytes(), (layout.layer_bytes + layout.index_layer_bytes) * cfg.mla_layers());
                assert_eq!(layout.index_reserved_bytes(),
                    CacheLayout::backing_bytes(layout.index_layer_bytes) * cfg.mla_layers() as u64);
                assert!(layout.reserved_bytes() >= layout.bytes() as u64);
            }
        }
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
