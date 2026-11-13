//! Cold golden cache storage. Physical page IDs come from the shared page pool.
use super::attention;
use crate::shared::memory::{device::{Allocation, Stream}, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_engine::prefix::RefPagePool;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::minimax_m3::{MiniMaxM3Config,
    execution::{MsaCache, MsaRope, UnitMetadata}};
use std::{collections::BTreeMap, ops::Range, rc::Rc};

/// One cold sequence, with all pages reserved before the first producer launch.
/// This does not publish prefix state or reconstruct any cached model state.
pub(crate) struct Cache<'a> {
    stream: Rc<Stream<'a>>,
    pub layout: MsaCache,
    pub rope: MsaRope,
    pool: RefPagePool,
    pages: Vec<u32>,
    planes: BTreeMap<String, Allocation<'a>>,
    cos_sin: Allocation<'a>,
    tokens: usize,
}
impl Drop for Cache<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("MiniMax cold cache retirement");
        self.pool.release(&self.pages);
    }
}
impl<'a> Cache<'a> {
    pub fn new(stream: Rc<Stream<'a>>, cfg: &MiniMaxM3Config, tokens: usize) -> Result<Self> {
        let positions = u32::try_from(tokens)?;
        let rope = MsaRope::new(cfg, positions)?;
        let count = positions.div_ceil(64);
        let layout = MsaCache::new(u32::try_from(cfg.layers)?, count, 1, positions)?;
        let mut pool = RefPagePool::new(count as usize, 64);
        let pages = pool.alloc_for(0..count as usize)?;
        let device = stream.device;
        let planes = layout.buffers()?.into_iter().map(|(name, bytes)| {
            Ok((name, Allocation::new(device, usize::try_from(bytes)?, Tag::kv("minimax/cache"))?))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        let cos_sin = Allocation::new(device, tokens.checked_mul(256).context("MiniMax rotary extent")?,
            Tag::workspace("minimax/rope"))?;
        let owner = Self { stream, layout, rope, pool, pages, planes, cos_sin, tokens };
        // Tables are immutable after construction; per-unit lengths are separate.
        let mut table = vec![0; owner.layout.table_width as usize * 4];
        for (entry, page) in table.chunks_exact_mut(4).zip(&owner.pages) {
            entry.copy_from_slice(&(*page as i32).to_le_bytes());
        }
        device.run(|| {
            device.library.copy_h2d(owner.plane("page_table")?, &table)?;
            device.library.copy_h2d(owner.plane("lengths")?, &0i32.to_le_bytes())?;
            device.library.copy_h2d(owner.cos_sin.buffer, &rotary_bytes(tokens, cfg.rope_theta as f32))
        })?;
        Ok(owner)
    }
    fn plane(&self, name: &str) -> Result<Buffer> {
        self.planes.get(name).map(|owner| owner.buffer)
            .with_context(|| format!("MiniMax cache lacks {name}"))
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.planes { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        self.cos_sin.inventory(inventory, &format!("{prefix}.rope.cos_sin"))
    }
    pub fn metadata(&self, capacity: u32) -> Result<Metadata<'a>> {
        let layout = UnitMetadata::new(capacity)?;
        let planes = layout.buffers().into_iter().map(|(name, bytes)| {
            Ok((name.into(), Allocation::new(self.stream.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/metadata"))?))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Metadata { stream: Rc::clone(&self.stream), capacity: capacity as usize,
            span: None, planes })
    }
    pub fn prepare(&self, metadata: &mut Metadata<'a>, span: Range<usize>) -> Result<()> {
        ensure!(span.start < span.end && span.end <= self.tokens && span.len() <= metadata.capacity
            && Rc::ptr_eq(&self.stream, &metadata.stream), "MiniMax cold metadata extent/stream");
        // A slot is reused only after its previous consumer has completed.
        self.stream.drain()?;
        metadata.span = None;
        let positions = span.clone().flat_map(|row| (row as i64).to_le_bytes()).collect::<Vec<_>>();
        let slots = span.clone().flat_map(|row| {
            (i64::from(self.pages[row / 64]) * 64 + (row % 64) as i64).to_le_bytes()
        }).collect::<Vec<_>>();
        let causal = span.clone().flat_map(|row| ((row + 1) as i32).to_le_bytes()).collect::<Vec<_>>();
        let scales = [1.0f32; 4].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let offsets = [0i32, span.len() as i32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        self.stream.device.run(|| {
            for (name, bytes) in [("positions", positions), ("slots", slots),
                ("batch_ids", vec![0; span.len() * 4]), ("causal_lengths", causal),
                ("cache_lengths", (span.end as i32).to_le_bytes().to_vec()),
                ("query_offsets", offsets), ("key_descale", scales.clone()), ("value_descale", scales)] {
                self.stream.device.library.copy_h2d(metadata.plane(name)?, &bytes)?;
            }
            Ok(())
        })?;
        metadata.span = Some(span);
        Ok(())
    }
    /// The shared executor must order each layer's preceding unit before this
    /// one, so all earlier cache rows referenced by this request are initialized.
    pub fn request(&self, metadata: &Metadata<'a>, layer: usize) -> Result<attention::Request> {
        let span = metadata.span.as_ref().context("MiniMax metadata is not prepared")?;
        ensure!(layer < self.layout.layers as usize && Rc::ptr_eq(&self.stream, &metadata.stream),
            "MiniMax cache request owner/layer");
        Ok(attention::Request { rows: span.len(), context: span.end,
            positions_capacity: self.tokens, table_width: self.layout.table_width as usize,
            table_stride: self.layout.table_width as usize, pool_pages: self.layout.pool_pages as usize,
            positions: metadata.plane("positions")?, slots: metadata.plane("slots")?,
            batch_ids: metadata.plane("batch_ids")?, cos_sin: self.cos_sin.buffer,
            pages: self.plane("page_table")?, causal_lengths: metadata.plane("causal_lengths")?,
            cache_lengths: metadata.plane("cache_lengths")?, query_offsets: metadata.plane("query_offsets")?,
            key_descale: metadata.plane("key_descale")?, value_descale: metadata.plane("value_descale")?,
            key_cache: self.plane(&format!("layer.{layer}.keys"))?,
            value_cache: self.plane(&format!("layer.{layer}.values"))?,
            index_cache: self.plane(&format!("layer.{layer}.index"))? })
    }
}

pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const PAGE_PLANES: [(&str, usize); 3] =
    [("keys", 32768), ("values", 32768), ("index", 8448)];

/// Request state contains references and a logical frontier, never device storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
}

impl crate::shared::qwen::scheduler::Placement for Placement {
    fn len(&self) -> usize { self.len }
    fn set_len(&mut self, len: usize) { self.len = len; }
    fn units(&self) -> &[u32] { &self.units }
    fn slot(&self) -> i32 { self.slot as i32 }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] { &[] }
}

/// One admitted rank's cache planes. Logical page IDs are shared between ranks;
/// each rank stores only the layers it actually executes (no KV replication).
pub(crate) struct PageBank<'a> {
    pub stream: Rc<Stream<'a>>,
    readers: Vec<Rc<Stream<'a>>>,
    pub layout: MsaCache,
    pub rope: MsaRope,
    pub layers: Range<usize>,
    pub model_layers: usize,
    planes: BTreeMap<String, Rc<Allocation<'a>>>,
    cos_sin: Rc<Allocation<'a>>,
}

impl Drop for PageBank<'_> {
    fn drop(&mut self) {
        for reader in &self.readers { reader.drain_or_abort("MiniMax page bank retirement"); }
    }
}

impl<'a> PageBank<'a> {
    /// Every stream touching pages must be retained before Ready. Cache/rope
    /// extents are passed from admission, never selected by the runtime.
    pub fn new(readers: Vec<Rc<Stream<'a>>>, cfg: &MiniMaxM3Config,
        layout: MsaCache, rope: MsaRope, layers: Range<usize>) -> Result<Self> {
        let stream = Rc::clone(readers.first().context("MiniMax cache has no reader")?);
        let device = stream.device;
        ensure!(!layers.is_empty() && layers.end <= cfg.layers && layers.len() == layout.layers as usize
            && readers.iter().all(|reader| reader.device.id == device.id
                && std::ptr::eq(reader.device.library, device.library)),
            "MiniMax page bank layers/readers differ from admission");
        ensure!(MsaCache::new(layout.layers, layout.pool_pages, layout.requests, rope.positions)? == layout
            && MsaRope::new(cfg, rope.positions)? == rope,
            "MiniMax page bank geometry differs from admission");
        let mut planes = BTreeMap::new();
        for (local, bytes) in layout.buffers()? {
            let name = if let Some(suffix) = local.strip_prefix("layer.") {
                let (layer, plane) = suffix.split_once('.').context("MiniMax cache plane name")?;
                format!("layer.{}.{}", layers.start + layer.parse::<usize>()?, plane)
            } else { local };
            let tag = if name.starts_with("layer.") { Tag::kv("minimax/kv") }
                else { Tag::workspace("minimax/cache") };
            planes.insert(name, Rc::new(Allocation::new(device, usize::try_from(bytes)?, tag)?));
        }
        let cos_sin = Rc::new(Allocation::new(device, usize::try_from(rope.buffers()[0].1)?,
            Tag::workspace("minimax/rope"))?);
        let bank = Self { stream, readers, layout, rope, layers, model_layers: cfg.layers, planes, cos_sin };
        device.run(|| {
            for allocation in bank.planes.values() {
                // SAFETY: new admitted backing and retained stream; no request
                // can read pages before initialization has completed.
                unsafe { device.library.cuda_zero_bytes_async(allocation.buffer,
                    allocation.buffer.bytes, bank.stream.raw)?; }
            }
            Ok(())
        })?;
        bank.stream.drain()?;
        device.run(|| device.library.copy_h2d(bank.cos_sin.buffer,
            &rotary_bytes(rope.positions as usize, cfg.rope_theta as f32)))?;
        Ok(bank)
    }

    pub fn readers(&self) -> &[Rc<Stream<'a>>] { &self.readers }
    pub fn owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.planes.values().cloned().chain([Rc::clone(&self.cos_sin)]).collect()
    }
    pub fn plane(&self, name: &str) -> Result<Buffer> {
        self.planes.get(name).map(|owner| owner.buffer)
            .with_context(|| format!("MiniMax page bank lacks {name}"))
    }
    pub fn page(&self, layer: usize, plane: &str, page: u32) -> Result<Buffer> {
        ensure!(self.layers.contains(&layer) && page < self.layout.pool_pages,
            "MiniMax page outside resident layer/pool");
        let stride = PAGE_PLANES.iter().find(|&&(name, _)| name == plane)
            .context("MiniMax unknown page plane")?.1;
        view(self.plane(&format!("layer.{layer}.{plane}"))?, page as usize * stride, stride)
    }
    pub fn validate(&self, placement: &Placement, len: usize) -> Result<()> {
        ensure!(placement.slot < self.layout.requests as usize && len <= self.rope.positions as usize
            && !placement.units.is_empty() && placement.units.len() <= self.layout.table_width as usize
            && len.div_ceil(PAGE_ROWS) <= placement.units.len(), "MiniMax request slot/context/pages");
        let mut unique = std::collections::BTreeSet::new();
        ensure!(placement.units.iter().all(|&page| page < self.layout.pool_pages && unique.insert(page)),
            "MiniMax request has duplicate or out-of-pool page refs");
        Ok(())
    }
    pub fn reserve_tables(&self, stream: *mut std::ffi::c_void) -> Result<()> {
        self.stream.device.run(|| {
            for slot in 0..self.layout.requests as usize {
                let width = self.layout.table_width as usize;
                for buffer in [view(self.plane("page_table")?, slot * width * 4, width * 4)?,
                    view(self.plane("lengths")?, slot * 4, 4)?] {
                    // SAFETY: retained fixed table backing and live copy stream,
                    // reserved before requests or graph capture can begin.
                    unsafe { self.stream.device.library.reserve_h2d_ordered(buffer, stream)?; }
                }
            }
            Ok(())
        })
    }
    /// The caller fences every reader onto `stream` before updating the table.
    pub fn install(&self, placement: &Placement, stream: *mut std::ffi::c_void) -> Result<()> {
        self.validate(placement, placement.len)?;
        let width = self.layout.table_width as usize;
        let mut table = vec![0; width * 4];
        for (entry, &page) in table.chunks_exact_mut(4).zip(&placement.units) {
            entry.copy_from_slice(&(page as i32).to_le_bytes());
        }
        let pages = view(self.plane("page_table")?, placement.slot * width * 4, width * 4)?;
        let length = view(self.plane("lengths")?, placement.slot * 4, 4)?;
        self.stream.device.run(|| {
            // SAFETY: fixed retained tables, all readers joined by the caller;
            // outside capture, and ordered staging retains the host bytes.
            unsafe {
                self.stream.device.library.copy_h2d_ordered(pages, &table, stream)?;
                self.stream.device.library.copy_h2d_ordered(length, &(placement.len as i32).to_le_bytes(), stream)
            }
        })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, gpu: u8) -> Result<()> {
        for (name, allocation) in &self.planes {
            let id = if let Some(suffix) = name.strip_prefix("layer.") {
                format!("minimax.kv.{suffix}")
            } else { format!("minimax.cache.{gpu}.{name}") };
            allocation.inventory(inventory, &id)?;
        }
        self.cos_sin.inventory(inventory, &format!("minimax.rope.{gpu}.cos_sin"))
    }
}

pub(crate) fn view(buffer: Buffer, offset: usize, bytes: usize) -> Result<Buffer> {
    ensure!(bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "MiniMax buffer view exceeds admitted owner");
    Ok(Buffer { ptr: buffer.ptr.wrapping_byte_add(offset), bytes, ..buffer })
}

pub(crate) struct Metadata<'a> {
    stream: Rc<Stream<'a>>,
    capacity: usize,
    span: Option<Range<usize>>,
    planes: BTreeMap<String, Allocation<'a>>,
}
impl Drop for Metadata<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax request metadata retirement"); }
}
impl Metadata<'_> {
    fn plane(&self, name: &str) -> Result<Buffer> {
        self.planes.get(name).map(|owner| owner.buffer)
            .with_context(|| format!("MiniMax metadata lacks {name}"))
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.planes { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::Device;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_core::memory_layout::Category;
    use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};

    fn config() -> MiniMaxM3Config {
        MiniMaxM3Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json")).unwrap()).unwrap()
    }
    fn read(library: &cuteafd_ffi::NativeLibrary, buffer: Buffer, bytes: usize) -> Result<Vec<u8>> {
        let mut out = vec![0; bytes];
        library.copy_d2h(&mut out, buffer)?;
        Ok(out)
    }
    #[test]
    fn cold_metadata_crosses_pages_without_aliasing_siblings_and_retires_cleanly() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 32 << 20)?;
        let device = Device { library: &library, id: 1 };
        let stream = Rc::new(Stream::new(device)?);
        let mut cfg = config(); cfg.layers = 2;
        let cache = Cache::new(Rc::clone(&stream), &cfg, 129)?;
        assert_eq!(cache.pool.used(), 3);
        let mut first = cache.metadata(65)?;
        let mut second = cache.metadata(65)?;
        assert!(cache.request(&first, 0).is_err());
        cache.prepare(&mut first, 0..65)?;
        cache.prepare(&mut second, 65..129)?;
        let a = cache.request(&first, 0)?;
        let b = cache.request(&second, 1)?;
        assert_eq!((a.rows, a.context, b.rows, b.context), (65, 65, 64, 129));
        assert_ne!(a.positions.ptr, b.positions.ptr);
        assert_ne!(a.key_cache.ptr, b.key_cache.ptr);
        assert_eq!(a.pages.ptr, b.pages.ptr);
        device.run(|| {
            assert_eq!(read(&library, a.positions, 65 * 8)?,
                (0..65i64).flat_map(i64::to_le_bytes).collect::<Vec<_>>());
            assert_eq!(read(&library, b.slots, 64 * 8)?,
                (65..129i64).flat_map(i64::to_le_bytes).collect::<Vec<_>>());
            assert_eq!(read(&library, b.causal_lengths, 64 * 4)?,
                (66..130i32).flat_map(i32::to_le_bytes).collect::<Vec<_>>());
            assert_eq!(read(&library, b.query_offsets, 8)?,
                [0i32, 64].into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>());
            assert_eq!(read(&library, b.cache_lengths, 4)?, 129i32.to_le_bytes());
            assert_eq!(read(&library, b.key_descale, 16)?,
                [1f32; 4].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>());
            let rope = read(&library, a.cos_sin, 256)?;
            assert_eq!(&rope[..128], [1f32; 32].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>());
            assert_eq!(&rope[128..], &[0; 128]);
            Ok(())
        })?;
        assert!(cache.prepare(&mut second, 65..131).is_err());
        assert!(cache.prepare(&mut second, 0..66).is_err());
        assert!(cache.request(&second, 2).is_err());
        let ids = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&ids)?, "minimax", Lifetime::TARGET,
            Category::Kv, "cold cache");
        cache.inventory(&mut inventory, "cache")?;
        assert_eq!(inventory.receipts.len(), cache.layout.buffers()?.len() + 1);
        let mut metadata = Inventory::new(CoordinatorDevices::new(&ids)?, "minimax", Lifetime::TARGET,
            Category::Workspace, "request metadata");
        first.inventory(&mut metadata, "unit.0")?;
        assert_eq!(metadata.receipts.len(), UnitMetadata::new(65)?.buffers().len());
        drop(first); drop(second); drop(cache); drop(stream);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 32 << 20);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        assert_eq!(library.cuda_get_device()?, 0);
        Ok(())
    }
    #[test]
    fn rope_is_fp32_cos_half_then_sin_half_for_each_position() {
        let bytes = rotary_bytes(3, 10_000_000.0);
        let values = bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
        assert_eq!(values.len(), 3 * 64);
        for position in 0..3 {
            for lane in 0..32 {
                let phase = position as f32 / 10_000_000.0f32.powf(lane as f32 / 32.0);
                assert!((values[position * 64 + lane] - phase.cos()).abs() < 2e-7);
                assert!((values[position * 64 + 32 + lane] - phase.sin()).abs() < 2e-7);
            }
        }
    }
}

fn rotary_bytes(positions: usize, theta: f32) -> Vec<u8> {
    let frequencies = std::array::from_fn::<_, 32, _>(|lane| 1.0 / theta.powf((2 * lane) as f32 / 64.0));
    let mut table = Vec::with_capacity(positions * 256);
    for position in 0..positions {
        for sine in [false, true] {
            for frequency in frequencies {
                let phase = position as f32 * frequency;
                table.extend_from_slice(&if sine { phase.sin() } else { phase.cos() }.to_le_bytes());
            }
        }
    }
    table
}
