//! Cold golden cache storage. Physical page IDs come from the shared page pool.
use super::attention;
use crate::shared::memory::{device::{Allocation, Stream}, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_engine::prefix::RefPagePool;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::{families::minimax_m3::{MiniMaxM3Config,
    execution::{MsaCache, MsaRope, UnitMetadata}}, placement::families::minimax_m3::Owner};
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
        Metadata::new(Rc::clone(&self.stream), UnitMetadata::new(capacity)?)
    }
    pub fn prepare(&self, metadata: &mut Metadata<'a>, span: Range<usize>) -> Result<()> {
        ensure!(metadata.mode == MetadataMode::Cold && span.start < span.end
            && span.end <= self.tokens && span.len() <= metadata.capacity
            && Rc::ptr_eq(&self.stream, &metadata.stream), "MiniMax cold metadata extent/stream");
        // A slot is reused only after its previous consumer has completed.
        self.stream.drain()?;
        metadata.span = None;
        let positions = span.clone().flat_map(|row| (row as i64).to_le_bytes()).collect::<Vec<_>>();
        let slots = span.clone().flat_map(|row| {
            (i64::from(self.pages[row / 64]) * 64 + (row % 64) as i64).to_le_bytes()
        }).collect::<Vec<_>>();
        let causal = span.clone().flat_map(|row| ((row + 1) as i32).to_le_bytes()).collect::<Vec<_>>();
        let offsets = [0i32, span.len() as i32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        self.stream.device.run(|| {
            for (name, bytes) in [("positions", positions), ("slots", slots),
                ("batch_ids", vec![0; span.len() * 4]), ("causal_lengths", causal),
                ("cache_lengths", (span.end as i32).to_le_bytes().to_vec()),
                ("query_offsets", offsets)] {
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
        device.run(|| {
            // SAFETY: admitted table and decode stream retained by bank.
            unsafe { device.library.reserve_h2d_ordered(bank.plane("decode_table")?, bank.stream.raw) }
        })?;
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
    fn validate_metadata(&self, metadata: &Metadata<'a>) -> Result<()> {
        ensure!(self.readers.iter().any(|reader| Rc::ptr_eq(reader, &metadata.stream)),
            "MiniMax metadata is not on a retained page-bank reader");
        Ok(())
    }
    /// Leaves the canonical slot table and placement frontier unchanged. The
    /// shared scheduler commits them only after the unit's arithmetic drains.
    pub fn prepare_segments(&self, metadata: &mut Metadata<'a>, appends: &[Append<'_>]) -> Result<()> {
        self.prepare_metadata(metadata, appends, false)
    }
    pub fn prepare_decode(&self, metadata: &mut Metadata<'a>, placements: &[&Placement]) -> Result<()> {
        let appends = placements.iter().enumerate().map(|(first_row, &placement)|
            Append { first_row, placement, rows: 1 }).collect::<Vec<_>>();
        self.prepare_metadata(metadata, &appends, true)
    }
    fn prepare_metadata(&self, metadata: &mut Metadata<'a>, appends: &[Append<'_>], decode: bool) -> Result<()> {
        self.validate_metadata(metadata)?;
        ensure!(metadata.mode == if decode { MetadataMode::Decode } else { MetadataMode::Prefill },
            "MiniMax metadata mode differs from execution");
        ensure!(!appends.is_empty() && appends.len() <= metadata.layout.requests as usize
            && appends.len() <= self.layout.requests as usize
            && (!decode || Rc::ptr_eq(&metadata.stream, &self.stream)),
            "MiniMax metadata request count or decode stream differs from admission");
        let mut planes = metadata.layout.buffers().into_iter()
            .filter(|(name, _)| metadata.mode.uploads(name))
            .map(|(name, bytes)| Ok((name, vec![0; usize::try_from(bytes)?])))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let width = self.layout.table_width as usize;
        let mut table = if decode { vec![0; self.plane("decode_table")?.bytes] } else { Vec::new() };
        let mut segments = Vec::with_capacity(appends.len());
        let mut next = 0usize;
        let mut cursor = 0u32;
        let mut request_slots = std::collections::BTreeSet::new();
        let mut append_slots = std::collections::BTreeSet::new();
        for (index, append) in appends.iter().enumerate() {
            ensure!(append.rows > 0 && append.first_row == next && (!decode || append.rows == 1),
                "MiniMax segments must partition contiguous live arithmetic rows");
            next = next.checked_add(append.rows).context("MiniMax live rows overflow")?;
            ensure!(next <= metadata.capacity && request_slots.insert(append.placement.slot),
                "MiniMax metadata capacity or duplicate request slot");
            let end = append.placement.len.checked_add(append.rows).context("MiniMax append frontier overflow")?;
            self.validate(append.placement, end)?;
            let row = if decode { index } else { metadata.layout.segment_row(cursor)? as usize };
            ensure!(row.checked_add(append.rows).is_some_and(|end| end <= metadata.layout.metadata_rows() as usize),
                "MiniMax aligned segment exceeds metadata owner");
            for local in 0..append.rows {
                let position = append.placement.len + local;
                let slot = usize::try_from(append.placement.units[position / PAGE_ROWS])? * PAGE_ROWS
                    + position % PAGE_ROWS;
                ensure!(append_slots.insert(slot), "MiniMax appended rows alias physical cache slots");
                put(&mut planes, "positions", (row + local) * 8, &(position as i64).to_le_bytes())?;
                put(&mut planes, "slots", (row + local) * 8, &(slot as i64).to_le_bytes())?;
                put(&mut planes, "causal_lengths", (row + local) * 4, &((position + 1) as i32).to_le_bytes())?;
            }
            let block = metadata.layout.segment_offset(index as u32)? as usize;
            let length = if decode { index * 4 } else { block };
            put(&mut planes, "cache_lengths", length, &(end as i32).to_le_bytes())?;
            if !decode {
                put(&mut planes, "query_offsets", block + 4, &(append.rows as i32).to_le_bytes())?;
            }
            if decode {
                for (entry, page) in table[index * width * 4..(index + 1) * width * 4]
                    .chunks_exact_mut(4).zip(&append.placement.units) {
                    entry.copy_from_slice(&(*page as i32).to_le_bytes());
                }
            }
            segments.push(PreparedSegment { first_row: append.first_row, metadata_row: row,
                slot: append.placement.slot, rows: append.rows, end });
            cursor = u32::try_from(row + append.rows)?;
        }
        metadata.stream.drain()?;
        if decode { for reader in &self.readers { reader.drain()?; } }
        metadata.span = None;
        metadata.segments.clear();
        metadata.bank_table = None;
        if decode {
            self.stream.device.run(|| {
                // SAFETY: all readers drained; full-table staging reserved before Ready.
                unsafe { self.stream.device.library.copy_h2d_ordered(self.plane("decode_table")?, &table, self.stream.raw) }
            })?;
        }
        metadata.upload(&planes)?;
        metadata.segments = segments;
        metadata.decode = decode;
        metadata.bank_table = Some(self.plane("page_table")?.ptr as usize);
        Ok(())
    }
    fn request(&self, metadata: &Metadata<'a>, layer: usize, segment: usize) -> Result<attention::Request> {
        self.validate_metadata(metadata)?;
        ensure!(self.layers.contains(&layer) && metadata.bank_table == Some(self.plane("page_table")?.ptr as usize),
            "MiniMax metadata is unprepared or outside resident bank/layers");
        let prepared = metadata.segments.get(segment).context("MiniMax metadata segment missing")?;
        let row = prepared.metadata_row;
        let block = metadata.layout.segment_offset(segment as u32)? as usize;
        let plane = |name, offset, bytes| view(metadata.plane(name)?, offset, bytes);
        let width = self.layout.table_width as usize;
        let decode = metadata.decode;
        Ok(attention::Request {
            rows: if decode { metadata.segments.len() } else { prepared.rows },
            context: if decode { metadata.segments.iter().map(|segment| segment.end).max().unwrap() }
                else { prepared.end },
            positions_capacity: self.rope.positions as usize, table_width: width,
            table_stride: width, pool_pages: self.layout.pool_pages as usize,
            positions: if decode { metadata.plane("positions")? } else { plane("positions", row * 8, prepared.rows * 8)? },
            slots: if decode { metadata.plane("slots")? } else { plane("slots", row * 8, prepared.rows * 8)? },
            batch_ids: if decode { metadata.plane("batch_ids")? } else { plane("batch_ids", row * 4, prepared.rows * 4)? },
            causal_lengths: if decode { metadata.plane("causal_lengths")? }
                else { plane("causal_lengths", row * 4, prepared.rows * 4)? },
            cache_lengths: if decode { metadata.plane("cache_lengths")? } else { plane("cache_lengths", block, 4)? },
            query_offsets: if decode { metadata.plane("query_offsets")? } else { plane("query_offsets", block, 8)? },
            key_descale: if decode { metadata.plane("key_descale")? } else { plane("key_descale", block, 16)? },
            value_descale: if decode { metadata.plane("value_descale")? } else { plane("value_descale", block, 16)? },
            cos_sin: self.cos_sin.buffer,
            pages: if decode { self.plane("decode_table")? }
                else { view(self.plane("page_table")?, prepared.slot * width * 4, width * 4)? },
            key_cache: self.plane(&format!("layer.{layer}.keys"))?,
            value_cache: self.plane(&format!("layer.{layer}.values"))?,
            index_cache: self.plane(&format!("layer.{layer}.index"))?,
        })
    }
    pub fn segments(&self, metadata: &Metadata<'a>, layer: usize) -> Result<Vec<attention::Segment>> {
        ensure!(!metadata.decode && !metadata.segments.is_empty(), "MiniMax segments are not prepared");
        metadata.segments.iter().enumerate().map(|(index, segment)|
            Ok(attention::Segment { first_row: segment.first_row, request: self.request(metadata, layer, index)? })).collect()
    }
    pub fn decode_request(&self, metadata: &Metadata<'a>, layer: usize) -> Result<attention::DecodeRequest> {
        ensure!(metadata.decode && !metadata.segments.is_empty(), "MiniMax decode is not prepared");
        Ok(attention::DecodeRequest { batch: metadata.segments.len(), request: self.request(metadata, layer, 0)? })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, gpu: u8) -> Result<()> {
        for (name, allocation) in &self.planes {
            let id = if let Some(suffix) = name.strip_prefix("layer.") {
                let (layer, plane) = suffix.split_once('.').context("MiniMax KV inventory name")?;
                let plane = PAGE_PLANES.iter().find(|&&(name, _)| name == plane)
                    .context("MiniMax KV inventory plane")?.0;
                Owner::Kv { layer: layer.parse()?, plane }.inventory_name().unwrap()
            } else { format!("{}.{name}", Owner::Cache { gpu: gpu.into() }.inventory_name().unwrap()) };
            allocation.inventory(inventory, &id)?;
        }
        self.cos_sin.inventory(inventory,
            &format!("{}.cos_sin", Owner::Rope { gpu: gpu.into() }.inventory_name().unwrap()))
    }
}

pub(crate) fn view(buffer: Buffer, offset: usize, bytes: usize) -> Result<Buffer> {
    ensure!(bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "MiniMax buffer view exceeds admitted owner");
    Ok(Buffer { ptr: buffer.ptr.wrapping_byte_add(offset), bytes, ..buffer })
}

/// Arithmetic rows are dense; segment metadata alone is padded for ABI alignment.
pub(crate) struct Append<'p> {
    pub first_row: usize,
    pub placement: &'p Placement,
    pub rows: usize,
}

struct PreparedSegment {
    first_row: usize,
    metadata_row: usize,
    slot: usize,
    rows: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetadataMode { Cold, Prefill, Decode }
impl MetadataMode {
    fn uploads(self, name: &str) -> bool {
        match self {
            Self::Cold => false,
            Self::Prefill => matches!(name, "positions" | "slots" | "batch_ids"
                | "causal_lengths" | "cache_lengths" | "query_offsets"),
            Self::Decode => matches!(name, "positions" | "slots" | "causal_lengths" | "cache_lengths"),
        }
    }
}

pub(crate) struct Metadata<'a> {
    stream: Rc<Stream<'a>>,
    pub layout: UnitMetadata,
    mode: MetadataMode,
    capacity: usize,
    span: Option<Range<usize>>,
    segments: Vec<PreparedSegment>,
    bank_table: Option<usize>,
    decode: bool,
    planes: BTreeMap<String, Allocation<'a>>,
}
impl Drop for Metadata<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax request metadata retirement"); }
}
impl<'a> Metadata<'a> {
    /// Cold golden uses synchronous uploads, with no retained ordered staging.
    pub fn new(stream: Rc<Stream<'a>>, layout: UnitMetadata) -> Result<Self> {
        Self::for_mode(stream, layout, MetadataMode::Cold)
    }
    pub fn new_prefill(stream: Rc<Stream<'a>>, layout: UnitMetadata) -> Result<Self> {
        Self::for_mode(stream, layout, MetadataMode::Prefill)
    }
    pub fn new_decode(stream: Rc<Stream<'a>>, layout: UnitMetadata) -> Result<Self> {
        ensure!(layout.rows <= 64 && layout.requests == layout.rows,
            "MiniMax decode metadata must match admitted concurrency");
        Self::for_mode(stream, layout, MetadataMode::Decode)
    }
    fn for_mode(stream: Rc<Stream<'a>>, layout: UnitMetadata, mode: MetadataMode) -> Result<Self> {
        ensure!(UnitMetadata::for_requests(layout.rows, layout.requests)? == layout
            && (mode != MetadataMode::Cold || layout.requests == 1),
            "MiniMax metadata differs from admission");
        let planes = layout.buffers().into_iter().map(|(name, bytes)| {
            Ok((name.into(), Allocation::new(stream.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/metadata"))?))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        let metadata = Self { stream, layout, mode, capacity: layout.rows as usize, span: None,
            segments: Vec::new(), bank_table: None, decode: false, planes };
        metadata.stream.device.run(|| {
            let library = metadata.stream.device.library;
            for name in ["key_descale", "value_descale"] {
                let dst = metadata.plane(name)?;
                let bytes = (0..dst.bytes / 4).flat_map(|_| 1.0f32.to_le_bytes()).collect::<Vec<_>>();
                library.copy_h2d(dst, &bytes)?;
            }
            if mode == MetadataMode::Decode {
                for name in ["batch_ids", "query_offsets"] {
                    let dst = metadata.plane(name)?;
                    let bytes = (0..dst.bytes / 4).flat_map(|i| (i as i32).to_le_bytes()).collect::<Vec<_>>();
                    library.copy_h2d(dst, &bytes)?;
                }
            }
            for (name, owner) in &metadata.planes {
                if mode.uploads(name) {
                    // SAFETY: private fixed destination/stream, reserved before Ready.
                    unsafe { library.reserve_h2d_ordered(owner.buffer, metadata.stream.raw)?; }
                }
            }
            Ok(())
        })?;
        Ok(metadata)
    }
    fn upload(&self, planes: &BTreeMap<&str, Vec<u8>>) -> Result<()> {
        self.stream.device.run(|| {
            for (name, bytes) in planes {
                ensure!(self.mode.uploads(name), "MiniMax attempted upload to constant metadata {name}");
                // SAFETY: validated full planes retain stable backing and staging;
                // the helper orders reuse after the previous transfer on this stream.
                unsafe { self.stream.device.library.copy_h2d_ordered(self.plane(name)?, bytes, self.stream.raw)?; }
            }
            Ok(())
        })
    }
    fn plane(&self, name: &str) -> Result<Buffer> {
        self.planes.get(name).map(|owner| owner.buffer)
            .with_context(|| format!("MiniMax metadata lacks {name}"))
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.planes { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}

fn put(planes: &mut BTreeMap<&str, Vec<u8>>, name: &str, offset: usize, bytes: &[u8]) -> Result<()> {
    let plane = planes.get_mut(name).context("MiniMax metadata plane missing")?;
    let end = offset.checked_add(bytes.len()).context("MiniMax metadata write overflow")?;
    plane.get_mut(offset..end).context("MiniMax metadata write exceeds admitted plane")?.copy_from_slice(bytes);
    Ok(())
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
    fn packed_segments_keep_live_rows_dense_and_decode_slots_canonical() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 32 << 20)?;
        let device = Device { library: &library, id: 1 };
        let stream = Rc::new(Stream::new(device)?);
        let mut cfg = config(); cfg.layers = 2;
        let bank = PageBank::new(vec![Rc::clone(&stream)], &cfg,
            MsaCache::new(2, 16, 4, 257)?, MsaRope::new(&cfg, 257)?, 0..2)?;
        bank.reserve_tables(stream.raw)?;
        let placements = [Placement { units: vec![7, 2, 9], slot: 3, len: 63 },
            Placement { units: vec![4, 11, 1], slot: 0, len: 128 },
            Placement { units: vec![8, 6, 13], slot: 2, len: 1 }];
        for placement in &placements { bank.install(placement, stream.raw)?; }
        stream.drain()?;
        let canonical = device.run(|| read(&library, bank.plane("page_table")?, bank.plane("page_table")?.bytes))?;
        let lengths = device.run(|| read(&library, bank.plane("lengths")?, bank.plane("lengths")?.bytes))?;
        let original = placements.clone();
        let mut metadata = Metadata::new_prefill(Rc::clone(&stream), UnitMetadata::for_requests(16, 4)?)?;
        let mut decode_metadata = Metadata::new_decode(Rc::clone(&stream), UnitMetadata::for_requests(4, 4)?)?;
        for owner in [&metadata, &decode_metadata] {
            for (name, allocation) in &owner.planes {
                assert_eq!(device.run(|| library.ordered_h2d_staging_buffer(allocation.buffer, stream.raw))?
                    .is_some(), owner.mode.uploads(name), "{name} {:?}", owner.mode);
            }
        }
        let requests = [Append { first_row: 0, placement: &placements[0], rows: 3 },
            Append { first_row: 3, placement: &placements[1], rows: 5 },
            Append { first_row: 8, placement: &placements[2], rows: 1 }];
        bank.prepare_segments(&mut metadata, &requests)?;
        let segments = bank.segments(&metadata, 1)?;
        assert_eq!(segments.iter().map(|s| (s.first_row, s.request.rows, s.request.context))
            .collect::<Vec<_>>(), vec![(0, 3, 66), (3, 5, 133), (8, 1, 2)]);
        assert!(bank.decode_request(&metadata, 0).is_err());
        for (index, segment) in segments.iter().enumerate() {
            let r = segment.request;
            for buffer in [r.positions, r.slots, r.batch_ids, r.causal_lengths,
                r.cache_lengths, r.query_offsets, r.key_descale, r.value_descale, r.pages] {
                assert_eq!(buffer.ptr as usize % 16, 0);
            }
            assert_eq!(r.pages.ptr, bank.plane("page_table")?.ptr
                .wrapping_byte_add(placements[index].slot * bank.layout.table_width as usize * 4));
            stream.drain()?;
            device.run(|| {
                assert_eq!(read(&library, r.positions, r.rows * 8)?,
                    (placements[index].len..placements[index].len + r.rows)
                        .flat_map(|value| (value as i64).to_le_bytes()).collect::<Vec<_>>());
                assert_eq!(read(&library, r.query_offsets, 8)?,
                    [0, r.rows as i32].into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>());
                assert_eq!(read(&library, r.batch_ids, r.rows * 4)?, vec![0; r.rows * 4]);
                assert_eq!(read(&library, r.cache_lengths, 4)?, (r.context as i32).to_le_bytes());
                Ok(())
            })?;
        }
        assert!(bank.prepare_decode(&mut metadata, &[&placements[0]]).is_err());
        assert!(bank.prepare_segments(&mut decode_metadata, &requests).is_err());
        drop(metadata);
        let mut metadata = decode_metadata;
        let stable = metadata.planes.values().map(|owner| owner.buffer.ptr as usize).collect::<Vec<_>>();
        let staging = device.run(|| library.ordered_h2d_staging_buffer(bank.plane("decode_table")?, stream.raw))?
            .context("decode-table staging not reserved")?;
        assert_eq!(staging.bytes, bank.plane("decode_table")?.bytes);
        for active in [vec![&placements[1], &placements[0], &placements[2]], vec![&placements[2]]] {
            bank.prepare_decode(&mut metadata, &active)?;
            let decode = bank.decode_request(&metadata, 0)?;
            assert!(bank.segments(&metadata, 0).is_err());
            assert_eq!((decode.batch, decode.request.rows, decode.request.context),
                (active.len(), active.len(), active.iter().map(|p| p.len + 1).max().unwrap()));
            stream.drain()?;
            device.run(|| {
                let r = decode.request;
                assert_eq!(read(&library, r.positions, active.len() * 8)?,
                    active.iter().flat_map(|p| (p.len as i64).to_le_bytes()).collect::<Vec<_>>());
                assert_eq!(read(&library, r.batch_ids, active.len() * 4)?,
                    (0..active.len()).flat_map(|i| (i as i32).to_le_bytes()).collect::<Vec<_>>());
                assert_eq!(read(&library, r.cache_lengths, active.len() * 4)?,
                    active.iter().flat_map(|p| ((p.len + 1) as i32).to_le_bytes()).collect::<Vec<_>>());
                assert_eq!(read(&library, r.query_offsets, (active.len() + 1) * 4)?,
                    (0..=active.len()).flat_map(|i| (i as i32).to_le_bytes()).collect::<Vec<_>>());
                let width = bank.layout.table_width as usize;
                let compact = read(&library, r.pages, r.pages.bytes)?;
                for (row, p) in active.iter().enumerate() {
                    let expected = p.units.iter().copied().chain(std::iter::repeat(0)).take(width)
                        .flat_map(|v| (v as i32).to_le_bytes()).collect::<Vec<_>>();
                    assert_eq!(&compact[row * width * 4..(row + 1) * width * 4], expected);
                }
                assert!(compact[active.len() * width * 4..].iter().all(|&v| v == 0));
                assert_eq!(read(&library, bank.plane("page_table")?, canonical.len())?, canonical);
                assert_eq!(read(&library, bank.plane("lengths")?, lengths.len())?, lengths);
                let next = library.ordered_h2d_staging_buffer(bank.plane("decode_table")?, stream.raw)?.unwrap();
                assert_eq!((next.ptr, next.bytes), (staging.ptr, staging.bytes));
                Ok(())
            })?;
            assert_eq!(stable, metadata.planes.values().map(|owner| owner.buffer.ptr as usize).collect::<Vec<_>>());
        }
        assert_eq!(placements, original);
        assert!(bank.prepare_decode(&mut metadata, &[&placements[0], &placements[0]]).is_err());
        let alias = Placement { slot: 1, ..placements[0].clone() };
        assert!(bank.prepare_decode(&mut metadata, &[&placements[0], &alias]).is_err());
        assert!(bank.prepare_segments(&mut metadata,
            &[Append { first_row: 1, placement: &placements[0], rows: 1 }]).is_err());
        assert!(bank.segments(&metadata, 2).is_err());
        drop(metadata); drop(bank); drop(stream);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 32 << 20);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn metadata_modes_retain_only_mutable_ordered_staging_at_admitted_concurrency() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let device = Device { library: &library, id: 1 };
        let stream = Rc::new(Stream::new(device)?);
        for concurrency in [1, 16, 64] {
            let layout = UnitMetadata::for_requests(concurrency, concurrency)?;
            for owner in [Metadata::new_decode(Rc::clone(&stream), layout)?,
                Metadata::new_prefill(Rc::clone(&stream), layout)?,
                Metadata::new(Rc::clone(&stream), UnitMetadata::new(concurrency)?)?] {
                let mut pinned = 0;
                for (name, allocation) in &owner.planes {
                    let staging = device.run(|| library.ordered_h2d_staging_buffer(allocation.buffer, stream.raw))?;
                    assert_eq!(staging.is_some(), owner.mode.uploads(name));
                    if let Some(staging) = staging {
                        assert_eq!(staging.bytes, allocation.buffer.bytes);
                        pinned += staging.bytes;
                    }
                    if matches!(name.as_str(), "key_descale" | "value_descale") {
                        assert_eq!(device.run(|| read(&library, allocation.buffer, allocation.buffer.bytes))?,
                            (0..allocation.buffer.bytes / 4).flat_map(|_| 1.0f32.to_le_bytes()).collect::<Vec<_>>());
                    }
                    if owner.mode == MetadataMode::Decode && matches!(name.as_str(), "batch_ids" | "query_offsets") {
                        assert_eq!(device.run(|| read(&library, allocation.buffer, allocation.buffer.bytes))?,
                            (0..allocation.buffer.bytes / 4).flat_map(|i| (i as i32).to_le_bytes()).collect::<Vec<_>>());
                    }
                }
                let buffers = owner.layout.buffers().into_iter().collect::<BTreeMap<_, _>>();
                let mutable = buffers["positions"] + buffers["slots"] + buffers["causal_lengths"]
                    + buffers["cache_lengths"];
                let expected = match owner.mode {
                    MetadataMode::Cold => 0,
                    MetadataMode::Decode => mutable,
                    MetadataMode::Prefill => mutable + buffers["batch_ids"] + buffers["query_offsets"],
                };
                assert_eq!(pinned as u64, expected);
            }
        }
        assert!(Metadata::new_decode(Rc::clone(&stream), UnitMetadata::new(16)?).is_err());
        assert!(Metadata::new_decode(Rc::clone(&stream), UnitMetadata::for_requests(65, 65)?).is_err());
        drop(stream);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
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
