//! Cold golden cache storage. Physical page IDs come from the shared page pool.
use super::attention;
use crate::shared::memory::{device::{Allocation, Stream}, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_engine::prefix::RefPagePool;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::{families::minimax_m3::{MiniMaxM3Config,
    execution::{MsaCache, MsaRope, UnitMetadata}}, placement::families::minimax_m3::{Owner, RequestCapacity}};
use std::{collections::{BTreeMap, BTreeSet}, ops::Range, rc::Rc};

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

/// Streams are provided by startup and retained by every metadata/cache reader.
pub(crate) struct RequestStreams<'a> {
    pub decode: Rc<Stream<'a>>,
    pub prefill: Vec<Rc<Stream<'a>>>,
}

pub(crate) struct RequestRank<'a> {
    pub decode: Metadata<'a>,
    pub prefill: Vec<Metadata<'a>>,
    pub bank: Rc<PageBank<'a>>,
}

/// Entry-rank token ids never alias prefill or routing scratch. The retained
/// stream orders each upload after the preceding decode consumer.
pub(crate) struct DecodeTokens<'a> {
    stream: Rc<Stream<'a>>,
    ids: Allocation<'a>,
}
impl Drop for DecodeTokens<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax decode token retirement"); }
}
impl<'a> DecodeTokens<'a> {
    fn new(stream: Rc<Stream<'a>>, rows: u32) -> Result<Self> {
        let ids = Allocation::new(stream.device, rows as usize * 4, Tag::workspace("minimax/decode-tokens"))?;
        stream.device.run(|| {
            // SAFETY: private admitted destination and retained decode stream;
            // reservation precedes all requests and staging sealing.
            unsafe { stream.device.library.reserve_h2d_ordered(ids.buffer, stream.raw) }
        })?;
        Ok(Self { stream, ids })
    }
    pub fn upload(&self, tokens: &[u32]) -> Result<Buffer> {
        ensure!(!tokens.is_empty() && tokens.len() <= self.ids.buffer.bytes / 4,
            "MiniMax decode tokens exceed admitted concurrency");
        let bytes = tokens.iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<_>>();
        self.stream.device.run(|| {
            // SAFETY: the dedicated full-capacity destination and reserved
            // staging survive every consumer on this same retained stream.
            unsafe { self.stream.device.library.copy_h2d_ordered(self.ids.buffer, &bytes, self.stream.raw) }
        })?;
        Ok(self.ids.buffer)
    }
    fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        let name = Owner::DecodeTokenIds { gpu: 0 }.inventory_name().unwrap();
        self.ids.inventory(inventory, &name)?;
        upload_inventory(inventory, &format!("{name}.upload"), self.stream.device.library,
            self.ids.buffer, self.stream.raw)
    }
}

/// Retained request storage only, not arithmetic, transport or a serving loop.
/// Prefix-copy staging is inventoried by MiniMaxPrefix after it is attached.
pub(crate) struct RequestState<'a> {
    pub capacity: RequestCapacity,
    pub tokens: DecodeTokens<'a>,
    pub ranks: Vec<RequestRank<'a>>,
}

impl<'a> RequestState<'a> {
    pub fn new(cfg: &MiniMaxM3Config, capacity: RequestCapacity,
        admitted: &cuteafd_loader::placement::families::minimax_m3::AdmittedMiniMax,
        streams: Vec<RequestStreams<'a>>) -> Result<Self> {
        use cuteafd_loader::placement::{FfnMode, LayerMode};
        let capacity = checked_capacity(capacity)?;
        let solved = &admitted.solved;
        ensure!(matches!(streams.len(), 1 | 2) && solved.decisions.lanes == 1
            && solved.decisions.prefill.rows == u64::from(capacity.prefill.rows)
            && solved.decisions.prefill.waves > 0,
            "MiniMax request storage differs from admitted lane/unit capacity");
        let waves = usize::try_from(solved.decisions.prefill.waves)?;
        let context = u32::try_from(solved.request.context.maximum)?;
        ensure!(context > 0 && solved.pool_page_rows == PAGE_ROWS as u64
            && solved.pool_tokens > 0 && solved.pool_tokens % PAGE_ROWS as u64 == 0
            && solved.served_context() > 0 && solved.served_context() <= u64::from(context)
            && solved.served_context() <= solved.pool_tokens
            && admitted.placement.pool_tokens == solved.pool_tokens,
            "MiniMax request storage differs from admitted pool/context");
        let pages = u32::try_from(solved.pool_tokens / PAGE_ROWS as u64)?;
        let modes = &solved.decisions.attention;
        ensure!(modes.len() == cfg.layers && admitted.placement.layers.len() == cfg.layers
            && admitted.placement.layers.iter().map(|layer| layer.mode).eq(modes.iter().copied()),
            "MiniMax request storage differs from admitted attention modes");
        let mut layers = vec![Vec::new(); streams.len()];
        for (layer, mode) in modes.iter().enumerate() {
            let LayerMode::Whole { gpu, ffn: FfnMode::Owner | FfnMode::Split } = mode else {
                anyhow::bail!("MiniMax request storage requires whole-layer owners");
            };
            layers.get_mut(usize::from(*gpu)).context("MiniMax attention rank is not admitted")?.push(layer);
        }
        let globals = capacity.cache(context)?;
        let library = streams[0].decode.device.library;
        let mut devices = BTreeSet::new();
        for (rank, readers) in streams.iter().enumerate() {
            for (name, bytes) in globals.buffers()?.into_iter().filter(|(name, _)| !name.starts_with("layer.")) {
                let id = format!("{}.{name}", Owner::Cache { gpu: rank }.id());
                let charged = solved.facts.allocations.iter().filter(|owner| owner.id == id).collect::<Vec<_>>();
                ensure!(charged.len() == 1 && charged[0].view.is_none() && charged[0].bytes == bytes
                    && charged[0].location == cuteafd_loader::placement::planner::Location::coordinator(rank as u8)
                    && charged[0].lifetime == cuteafd_loader::placement::planner::Lifetime::TARGET,
                    "MiniMax request capacity differs from admitted cache owner {id}");
            }
            let owned = &layers[rank];
            ensure!(!owned.is_empty() && owned.windows(2).all(|pair| pair[1] == pair[0] + 1),
                "MiniMax request storage requires a nonempty contiguous whole-layer range per rank");
            let device = readers.decode.device;
            ensure!(devices.insert(device.id) && std::ptr::eq(device.library, library)
                && readers.prefill.len() == waves
                && readers.prefill.iter().all(|stream| stream.device.id == device.id
                    && std::ptr::eq(stream.device.library, library)),
                "MiniMax request streams differ from admitted physical ranks/waves");
        }
        let token_id = Owner::DecodeTokenIds { gpu: 0 }.id();
        for (id, location) in [(token_id.clone(), cuteafd_loader::placement::planner::Location::coordinator(0)),
            (format!("{token_id}.upload"), cuteafd_loader::placement::planner::Location::PINNED)] {
            let owners = solved.facts.allocations.iter().filter(|owner| owner.id == id).collect::<Vec<_>>();
            ensure!(owners.len() == 1 && owners[0].view.is_none()
                && owners[0].bytes == u64::from(capacity.decode.rows) * 4 && owners[0].location == location
                && owners[0].lifetime == cuteafd_loader::placement::planner::Lifetime::TARGET,
                "MiniMax decode token owner differs from admission: {id}");
        }
        // Validate every rank before the first allocation: a later malformed
        // reader or split mode cannot leave a partially constructed request set.
        let tokens = DecodeTokens::new(Rc::clone(&streams[0].decode), capacity.decode.rows)?;
        let rope = MsaRope::new(cfg, context)?;
        let mut ranks = Vec::with_capacity(streams.len());
        for (readers, owned) in streams.into_iter().zip(layers) {
            let range = owned[0]..owned[owned.len() - 1] + 1;
            let layout = MsaCache::new(u32::try_from(range.len())?, pages, capacity.decode.requests, context)?;
            let all = std::iter::once(Rc::clone(&readers.decode))
                .chain(readers.prefill.iter().cloned()).collect();
            let bank = Rc::new(PageBank::new(all, cfg, capacity, layout, rope, range)?);
            let decode = Metadata::new_decode(readers.decode, capacity)?;
            let prefill = readers.prefill.into_iter().map(|stream| Metadata::new_prefill(stream, capacity))
                .collect::<Result<Vec<_>>>()?;
            ranks.push(RequestRank { decode, prefill, bank });
        }
        Ok(Self { capacity, tokens, ranks })
    }

    pub fn banks(&self) -> Vec<Rc<PageBank<'a>>> {
        self.ranks.iter().map(|rank| Rc::clone(&rank.bank)).collect()
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        use cuteafd_loader::placement::planner::{Lifetime, Location};
        ensure!(inventory.lifetime() == Lifetime::TARGET,
            "MiniMax request inventory lifetime differs from admission");
        for (gpu, rank) in self.ranks.iter().enumerate() {
            ensure!(inventory.location(rank.bank.stream.device.id)? == Location::coordinator(gpu as u8),
                "MiniMax request inventory rank mapping differs from construction");
        }
        ensure!(Rc::ptr_eq(&self.tokens.stream, &self.ranks[0].decode.stream),
            "MiniMax token input differs from entry-rank decode stream");
        self.tokens.inventory(inventory)?;
        for (gpu, rank) in self.ranks.iter().enumerate() {
            rank.bank.inventory(inventory, u8::try_from(gpu)?)?;
            rank.decode.inventory(inventory,
                &format!("{}.metadata", Owner::Decode { gpu }.inventory_name().unwrap()))?;
            for (slot, metadata) in rank.prefill.iter().enumerate() {
                metadata.inventory(inventory,
                    &format!("{}.metadata", Owner::Unit { gpu, slot: slot as u64 }.inventory_name().unwrap()))?;
            }
        }
        Ok(())
    }
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
        capacity: RequestCapacity, layout: MsaCache, rope: MsaRope, layers: Range<usize>) -> Result<Self> {
        let globals = checked_capacity(capacity)?.cache(rope.positions)?;
        ensure!(layout.requests == globals.requests && layout.table_width == globals.table_width,
            "MiniMax cache globals differ from admitted request capacity");
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
                for (_, buffer) in self.table_destinations(slot)? {
                    // SAFETY: retained fixed table backing and live copy stream,
                    // reserved before requests or graph capture can begin.
                    unsafe { self.stream.device.library.reserve_h2d_ordered(buffer, stream)?; }
                }
            }
            Ok(())
        })
    }
    fn table_destinations(&self, slot: usize) -> Result<[(&'static str, Buffer); 2]> {
        ensure!(slot < self.layout.requests as usize, "MiniMax table slot outside admission");
        let width = self.layout.table_width as usize;
        Ok([("table", view(self.plane("page_table")?, slot * width * 4, width * 4)?),
            ("length", view(self.plane("lengths")?, slot * 4, 4)?)])
    }
    /// Prefix-copy staging is keyed by interior destination and copy stream.
    /// Observe existing owners without reserving a second copy.
    pub fn prefix_upload_inventory(&self, inventory: &mut Inventory<'_>, gpu: u8,
        stream: *mut std::ffi::c_void) -> Result<()> {
        for slot in 0..self.layout.requests as usize {
            for (plane, destination) in self.table_destinations(slot)? {
                let name = Owner::PrefixH2d { gpu: gpu.into(), slot: slot as u64, plane }
                    .inventory_name().unwrap();
                upload_inventory(inventory, &name, self.stream.device.library, destination, stream)?;
            }
        }
        Ok(())
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
            .filter(|(name, _)| metadata.uploads.contains(name))
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
            &format!("{}.cos_sin", Owner::Rope { gpu: gpu.into() }.inventory_name().unwrap()))?;
        upload_inventory(inventory,
            &format!("{}.upload", Owner::DecodeTable { gpu: gpu.into() }.inventory_name().unwrap()),
            self.stream.device.library, self.plane("decode_table")?, self.stream.raw)
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

pub(crate) struct Metadata<'a> {
    stream: Rc<Stream<'a>>,
    pub layout: UnitMetadata,
    mode: MetadataMode,
    uploads: BTreeSet<&'static str>,
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
        Self::for_mode(stream, layout, MetadataMode::Cold, Vec::new())
    }
    pub fn new_prefill(stream: Rc<Stream<'a>>, capacity: RequestCapacity) -> Result<Self> {
        let capacity = checked_capacity(capacity)?;
        Self::for_mode(stream, capacity.metadata(true), MetadataMode::Prefill, capacity.metadata_uploads(true))
    }
    pub fn new_decode(stream: Rc<Stream<'a>>, capacity: RequestCapacity) -> Result<Self> {
        let capacity = checked_capacity(capacity)?;
        Self::for_mode(stream, capacity.metadata(false), MetadataMode::Decode, capacity.metadata_uploads(false))
    }
    fn for_mode(stream: Rc<Stream<'a>>, layout: UnitMetadata, mode: MetadataMode,
        uploads: Vec<(&'static str, u64)>) -> Result<Self> {
        ensure!(UnitMetadata::for_requests(layout.rows, layout.requests)? == layout
            && (mode != MetadataMode::Cold || layout.requests == 1),
            "MiniMax metadata differs from admission");
        let planes = layout.buffers().into_iter().map(|(name, bytes)| {
            Ok((name.into(), Allocation::new(stream.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/metadata"))?))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        let metadata = Self { stream, layout, mode, capacity: layout.rows as usize, span: None,
            uploads: uploads.iter().map(|&(name, _)| name).collect(),
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
            for (name, bytes) in uploads {
                let buffer = metadata.plane(name)?;
                ensure!(buffer.bytes as u64 == bytes, "MiniMax upload differs from admitted metadata");
                // SAFETY: private fixed destination/stream, reserved before Ready.
                unsafe { library.reserve_h2d_ordered(buffer, metadata.stream.raw)?; }
            }
            Ok(())
        })?;
        Ok(metadata)
    }
    fn upload(&self, planes: &BTreeMap<&str, Vec<u8>>) -> Result<()> {
        self.stream.device.run(|| {
            for (name, bytes) in planes {
                ensure!(self.uploads.contains(name), "MiniMax attempted upload to constant metadata {name}");
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
        for (name, owner) in &self.planes {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
            if self.uploads.contains(name.as_str()) {
                upload_inventory(inventory, &format!("{prefix}.{name}.upload"),
                    self.stream.device.library, owner.buffer, self.stream.raw)?;
            }
        }
        Ok(())
    }
}

fn checked_capacity(capacity: RequestCapacity) -> Result<RequestCapacity> {
    ensure!(RequestCapacity::new(u64::from(capacity.decode.rows), u64::from(capacity.prefill.rows))? == capacity,
        "MiniMax request capacity differs from admitted descriptor");
    Ok(capacity)
}

fn upload_inventory(inventory: &mut Inventory<'_>, name: &str, library: &cuteafd_ffi::NativeLibrary,
    destination: Buffer, stream: *mut std::ffi::c_void) -> Result<()> {
    let staging = library.ordered_h2d_staging_buffer(destination, stream)?
        .with_context(|| format!("MiniMax admitted ordered upload missing: {name}"))?;
    ensure!(staging.bytes == destination.bytes
        && staging.flags & cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_PINNED != 0,
        "MiniMax ordered upload extent/type differs from admission: {name}");
    inventory.pinned(name, staging)
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
        let capacity = RequestCapacity::new(4, 16)?;
        let bank = PageBank::new(vec![Rc::clone(&stream)], &cfg, capacity,
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
        let mut metadata = Metadata::new_prefill(Rc::clone(&stream), capacity)?;
        let mut decode_metadata = Metadata::new_decode(Rc::clone(&stream), capacity)?;
        for owner in [&metadata, &decode_metadata] {
            for (name, allocation) in &owner.planes {
                assert_eq!(device.run(|| library.ordered_h2d_staging_buffer(allocation.buffer, stream.raw))?
                    .is_some(), owner.uploads.contains(name.as_str()), "{name} {:?}", owner.mode);
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
            let capacity = RequestCapacity::new(u64::from(concurrency), u64::from(concurrency))?;
            for owner in [Metadata::new_decode(Rc::clone(&stream), capacity)?,
                Metadata::new_prefill(Rc::clone(&stream), capacity)?,
                Metadata::new(Rc::clone(&stream), UnitMetadata::new(concurrency)?)?] {
                let mut pinned = 0;
                for (name, allocation) in &owner.planes {
                    let staging = device.run(|| library.ordered_h2d_staging_buffer(allocation.buffer, stream.raw))?;
                    assert_eq!(staging.is_some(), owner.uploads.contains(name.as_str()));
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
        let mut capacity = RequestCapacity::new(16, 32)?;
        capacity.decode = UnitMetadata::new(16)?;
        assert!(Metadata::new_decode(Rc::clone(&stream), capacity).is_err());
        capacity.decode = UnitMetadata::for_requests(65, 65)?;
        assert!(Metadata::new_decode(Rc::clone(&stream), capacity).is_err());
        capacity = RequestCapacity::new(64, 32)?;
        capacity.prefill = UnitMetadata::for_requests(32, 16)?;
        assert!(Metadata::new_prefill(Rc::clone(&stream), capacity).is_err());
        drop(stream);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    fn request_admission(cfg: &MiniMaxM3Config, ranks: usize, capacity: RequestCapacity)
        -> Result<cuteafd_loader::placement::families::minimax_m3::AdmittedMiniMax> {
        use cuteafd_loader::placement::{AttentionPlacement, ExpertHome, FfnMode, LayerAssignment,
            LayerMode, Placement as ModelPlacement, PoolLimits};
        use cuteafd_loader::placement::planner::*;
        let modes = cuteafd_loader::placement::families::minimax_m3::layer_owners(cfg.layers, ranks)
            .into_iter().map(|gpu| LayerMode::Whole { gpu, ffn: FfnMode::Owner }).collect::<Vec<_>>();
        let shape = PrefillShape { rows: capacity.prefill.rows.into(), waves: 3 };
        let pages = u64::from(capacity.decode.requests).max(8);
        let pool_tokens = pages * PAGE_ROWS as u64;
        let mut allocations: Vec<_> = (0..ranks).map(|rank| -> Result<Vec<_>> {
            Ok(capacity.cache(257)?.buffers()?.into_iter().filter(|(name, _)| !name.starts_with("layer."))
                .map(|(name, bytes)| cuteafd_loader::placement::planner::Allocation::buffer(
                    format!("{}.{name}", Owner::Cache { gpu: rank }.id()), Location::coordinator(rank as u8),
                    bytes, Lifetime::TARGET, Category::Kv, "admitted cache globals")).collect())
        }).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect();
        let id = Owner::DecodeTokenIds { gpu: 0 }.id();
        for (id, location, category) in [(id.clone(), Location::coordinator(0), Category::Workspace),
            (format!("{id}.upload"), Location::PINNED, Category::Staging)] {
            allocations.push(cuteafd_loader::placement::planner::Allocation::buffer(id, location,
                u64::from(capacity.decode.rows) * 4, Lifetime::TARGET, category, "decode token ids"));
        }
        let solved = ResolvedPlan { encoder_diagnostics: vec![],
            request: cuteafd_loader::placement::families::minimax_m3::request(Context::automatic(257),
                Some(pool_tokens), cuteafd_loader::placement::Onboard::Layers(0), shape,
                Operation::Serving, None, Default::default()),
            decisions: Decisions { attention: modes.clone(), onboard_layers: vec![vec![]; ranks],
                expert_homes: Default::default(), embedding: None, draft_home: None, head_home: None,
                intake_home: None, encoder: None, prefill: shape, lanes: 1, spark_reply_formats: None },
            pool_tokens, pool_page_rows: PAGE_ROWS as u64, budgets: vec![], phases: vec![],
            facts: Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![], allocations,
                deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
                driver_page_tables: vec![], allocation_granularity: vec![] } };
        Ok(cuteafd_loader::placement::families::minimax_m3::AdmittedMiniMax {
            placement: ModelPlacement { pool_tokens, pool_limits: PoolLimits(vec![]),
                attention_placement: AttentionPlacement::Layers, attention_by_kind: vec![],
                peer_row_bytes: 0, onboard_layers: 0, max_pool_tokens_no_onboard: None,
                layers: modes.into_iter().map(|mode| LayerAssignment { mode, experts: ExpertHome::Dense }).collect(),
                movables: vec![], expert_ranges: vec![], residual: vec![], hops: vec![], tp2: None, items: vec![] },
            solved, onboard: 0, expert_backings: vec![], push_host_envelope: None, draft_home: None, encoder: None })
    }

    fn request_streams<'a>(library: &'a cuteafd_ffi::NativeLibrary, ids: &[i32]) -> Result<Vec<RequestStreams<'a>>> {
        ids.iter().map(|&id| {
            let device = Device { library, id };
            Ok(RequestStreams { decode: Rc::new(Stream::new(device)?),
                prefill: (0..3).map(|_| Stream::new(device).map(Rc::new)).collect::<Result<Vec<_>>>()? })
        }).collect()
    }

    #[test]
    fn split_ffn_preserves_whole_attention_cache_owners_and_heads_still_refuses() -> Result<()> {
        use cuteafd_loader::placement::{FfnMode, LayerMode};
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.extent_only(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        fixture.strict_device_free(&library, true)?;
        for id in 0..2 { fixture.configure_memory(&library, id, 1 << 30)?; }
        let mut cfg = config(); cfg.layers = 4;
        let ids = [1, 0];
        let capacity = RequestCapacity::new(16, 128)?;
        let owner = request_admission(&cfg, 2, capacity)?;
        let mut split = owner.clone();
        for mode in &mut split.solved.decisions.attention {
            let LayerMode::Whole { ffn, .. } = mode else { unreachable!() };
            *ffn = FfnMode::Split;
        }
        for (layer, mode) in split.placement.layers.iter_mut().zip(&split.solved.decisions.attention) {
            layer.mode = *mode;
        }
        let observe = |admitted: &cuteafd_loader::placement::families::minimax_m3::AdmittedMiniMax| -> Result<_> {
            let state = RequestState::new(&cfg, capacity, admitted, request_streams(&library, &ids)?)?;
            let ranges = state.ranks.iter().map(|rank| rank.bank.layers.clone()).collect::<Vec<_>>();
            let mut actual = Inventory::new(CoordinatorDevices::new(&ids)?, "minimax", Lifetime::TARGET,
                Category::Workspace, "retained request bundle");
            state.inventory(&mut actual)?;
            let owners = actual.receipts.iter().map(|receipt| {
                let owner = &receipt.allocation;
                (owner.id.clone(), owner.bytes, owner.location, owner.lifetime, owner.category)
            }).collect::<Vec<_>>();
            Ok((ranges, owners))
        };
        assert_eq!(observe(&owner)?, observe(&split)?);
        let mut heads = split;
        heads.solved.decisions.attention[0] = LayerMode::HeadSplit;
        heads.placement.layers[0].mode = LayerMode::HeadSplit;
        let error = RequestState::new(&cfg, capacity, &heads, request_streams(&library, &ids)?)
            .err().context("heads cache must refuse")?;
        assert!(error.to_string().contains("requires whole-layer owners"));
        assert_eq!(fixture.mapped_bytes(&library)?, 0);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn request_state_retains_admitted_whole_layer_banks_and_every_metadata_slot() -> Result<()> {
        use cuteafd_loader::placement::planner::Location;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        for id in 0..2 { fixture.configure_memory(&library, id, 64 << 20)?; }
        let mut cfg = config(); cfg.layers = 2;
        for ids in [vec![1], vec![1, 0]] {
            for concurrency in [1, 16, 64] {
                let capacity = RequestCapacity::new(concurrency, 128)?;
                let admitted = request_admission(&cfg, ids.len(), capacity)?;
                let mut state = RequestState::new(&cfg, capacity, &admitted, request_streams(&library, &ids)?)?;
                assert_eq!(state.capacity, capacity);
                assert_eq!(state.ranks.len(), ids.len());
                let banks = state.banks();
                for (gpu, rank) in state.ranks.iter_mut().enumerate() {
                    assert!(Rc::ptr_eq(&banks[gpu], &rank.bank));
                    assert_eq!(rank.bank.layout.pool_pages as u64 * PAGE_ROWS as u64, admitted.solved.pool_tokens);
                    assert_eq!(rank.prefill.len(), 3);
                    assert!(Rc::ptr_eq(&rank.decode.stream, &rank.bank.readers()[0]));
                    for (slot, metadata) in rank.prefill.iter().enumerate() {
                        assert_eq!(metadata.layout, capacity.prefill);
                        assert!(Rc::ptr_eq(&metadata.stream, &rank.bank.readers()[slot + 1]));
                    }
                    let placement = Placement { units: vec![0], slot: 0, len: 0 };
                    rank.bank.prepare_decode(&mut rank.decode, &[&placement])?;
                    assert_eq!(rank.bank.decode_request(&rank.decode, rank.bank.layers.start)?.batch, 1);
                    for metadata in &mut rank.prefill {
                        rank.bank.prepare_segments(metadata, &[Append { first_row: 0, placement: &placement, rows: 1 }])?;
                        assert_eq!(rank.bank.segments(metadata, rank.bank.layers.start)?.len(), 1);
                    }
                }
                let inventory = || -> Result<Inventory<'_>> {
                    let mut actual = Inventory::new(CoordinatorDevices::new(&ids)?, "minimax", Lifetime::TARGET,
                        Category::Workspace, "retained request bundle");
                    state.inventory(&mut actual)?;
                    Ok(actual)
                };
                let first = inventory()?;
                let second = inventory()?;
                let observed = |inventory: &Inventory<'_>| inventory.receipts.iter().map(|receipt| {
                    let owner = &receipt.allocation;
                    (owner.id.clone(), owner.bytes, owner.location, owner.lifetime, receipt.address)
                }).collect::<Vec<_>>();
                assert_eq!(observed(&first), observed(&second), "inventory is a stable observation");
                let actual = first.receipts.iter().map(|receipt| (receipt.allocation.id.clone(),
                    (receipt.allocation.bytes, receipt.allocation.location, receipt.allocation.lifetime)))
                    .collect::<BTreeMap<_, _>>();
                assert_eq!(actual.len(), first.receipts.len());
                let token_id = Owner::DecodeTokenIds { gpu: 0 }.id();
                assert_eq!(actual.get(&token_id), Some(&(concurrency * 4, Location::coordinator(0), Lifetime::TARGET)));
                assert_eq!(actual.get(&format!("{token_id}.upload")), Some(&(concurrency * 4, Location::PINNED, Lifetime::TARGET)));
                assert!(!actual.contains_key(&Owner::DecodeTokenIds { gpu: 1 }.id()));
                let staging = Device { library: &library, id: ids[0] }.run(|| library.ordered_h2d_staging_buffer(
                    state.tokens.ids.buffer, state.tokens.stream.raw))?.unwrap();
                for count in [1, concurrency as usize, 1] {
                    let tokens = (0..count).map(|index| index as u32 + 103).collect::<Vec<_>>();
                    let buffer = state.tokens.upload(&tokens)?;
                    state.tokens.stream.drain()?;
                    let bytes = Device { library: &library, id: ids[0] }.run(|| read(&library, buffer, count * 4))?;
                    assert_eq!(bytes, tokens.iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<_>>());
                    let next = Device { library: &library, id: ids[0] }.run(|| library.ordered_h2d_staging_buffer(
                        buffer, state.tokens.stream.raw))?.unwrap();
                    assert_eq!((next.ptr, next.bytes), (staging.ptr, staging.bytes));
                }
                assert!(state.tokens.upload(&[]).is_err());
                assert!(state.tokens.upload(&vec![1; concurrency as usize + 1]).is_err());
                let mut expected_owners = 2;
                for (gpu, rank) in state.ranks.iter().enumerate() {
                    let location = Location::coordinator(gpu as u8);
                    let mut expected = Vec::new();
                    for layer in rank.bank.layers.clone() {
                        for (plane, stride) in PAGE_PLANES {
                            expected.push((Owner::Kv { layer, plane }.id(), stride as u64
                                * u64::from(rank.bank.layout.pool_pages), location));
                        }
                    }
                    for (name, bytes) in capacity.cache(257)?.buffers()?.into_iter()
                        .filter(|(name, _)| !name.starts_with("layer.")) {
                        expected.push((format!("{}.{name}", Owner::Cache { gpu }.id()), bytes, location));
                    }
                    expected.push((format!("{}.upload", Owner::DecodeTable { gpu }.id()),
                        u64::from(capacity.decode.requests) * u64::from(capacity.cache(257)?.table_width) * 4, Location::PINNED));
                    for (name, bytes) in MsaRope::new(&cfg, 257)?.buffers() {
                        expected.push((format!("{}.{name}", Owner::Rope { gpu }.id()), bytes, location));
                    }
                    for mode in 0..4 {
                        let prefill = mode > 0;
                        let owner = if prefill { Owner::Unit { gpu, slot: mode - 1 } } else { Owner::Decode { gpu } };
                        for (name, bytes) in capacity.metadata(prefill).buffers() {
                            expected.push((format!("{}.metadata.{name}", owner.id()), bytes, location));
                        }
                        for (name, bytes) in capacity.metadata_uploads(prefill) {
                            expected.push((format!("{}.metadata.{name}.upload", owner.id()), bytes, Location::PINNED));
                        }
                    }
                    expected_owners += expected.len();
                    for (id, bytes, location) in expected {
                        assert_eq!(actual.get(&id), Some(&(bytes, location, Lifetime::TARGET)), "{id}");
                    }
                }
                assert_eq!(actual.len(), expected_owners, "no extra physical request owners");
                drop(banks); drop(state);
                for &id in &ids {
                    assert_eq!(Device { library: &library, id }.run(|| library.cuda_physical_memory_info())?.0, 64 << 20);
                }
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn request_state_rejects_non_whole_or_mismatched_admission_before_allocating() -> Result<()> {
        use cuteafd_loader::placement::{FfnMode, LayerMode};
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let mut cfg = config(); cfg.layers = 2;
        let capacity = RequestCapacity::new(16, 128)?;
        for case in 0..10 {
            let mut admitted = request_admission(&cfg, 2, capacity)?;
            let mut streams = request_streams(&library, &[1, 0])?;
            match case {
                0 => admitted.solved.decisions.attention[0] = LayerMode::HeadSplit,
                1 => admitted.solved.decisions.attention[0] = LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
                2 => admitted.solved.decisions.prefill.waves = 2,
                3 => admitted.solved.pool_tokens += 1,
                4 => admitted.placement.pool_tokens += PAGE_ROWS as u64,
                5 => streams[1].decode = Rc::clone(&streams[0].decode),
                6 => admitted.solved.decisions.lanes = 2,
                7 => admitted.solved.facts.allocations[0].bytes += 4,
                8 => admitted.solved.facts.allocations.clear(),
                9 => admitted.solved.decisions.prefill.rows = 64,
                _ => unreachable!(),
            }
            let before = fixture.events()?;
            assert!(RequestState::new(&cfg, capacity, &admitted, streams).is_err(), "case {case}");
            let after = fixture.events()?;
            assert!(!after[before.len()..].contains('D'), "case {case} allocated before validating");
        }
        let admitted = request_admission(&cfg, 1, capacity)?;
        assert!(RequestState::new(&cfg, capacity, &admitted, vec![]).is_err());
        Ok(())
    }

    #[test]
    fn decode_token_owners_reject_bad_admission_before_allocating() -> Result<()> {
        use cuteafd_loader::placement::planner::{Location, View};
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let mut cfg = config(); cfg.layers = 2;
        let capacity = RequestCapacity::new(16, 128)?;
        let token_id = Owner::DecodeTokenIds { gpu: 0 }.id();
        for id in [token_id.clone(), format!("{token_id}.upload")] {
            for case in 0..6 {
                let mut admitted = request_admission(&cfg, 2, capacity)?;
                let owners = &mut admitted.solved.facts.allocations;
                let index = owners.iter().position(|owner| owner.id == id).unwrap();
                match case {
                    0 => { owners.remove(index); }
                    1 => owners[index].bytes += 4,
                    2 => owners[index].location = Location::coordinator(1),
                    3 => owners.push(owners[index].clone()),
                    4 => owners[index].lifetime = Lifetime::RESIDENT,
                    5 => owners[index].view = Some(View { backing: "unexpected".into(), offset: 0 }),
                    _ => unreachable!(),
                }
                let streams = request_streams(&library, &[1, 0])?;
                let before = fixture.events()?;
                assert!(RequestState::new(&cfg, capacity, &admitted, streams).is_err(), "{id} case {case}");
                let after = fixture.events()?;
                assert!(!after[before.len()..].contains('D'), "{id} case {case} allocated before validating");
            }
        }
        Ok(())
    }

    #[test]
    fn retained_request_owners_equal_shared_descriptor_on_one_and_two_ranks() -> Result<()> {
        use super::super::prefix::MiniMaxPrefix;
        use cuteafd_loader::placement::planner::Location;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        for device in 0..2 { fixture.configure_memory(&library, device, 64 << 20)?; }
        let mut cfg = config(); cfg.layers = 2;
        let context = 257;
        let pool_pages = 8;
        for ranks in [1, 2] {
            let ids = (0..ranks).collect::<Vec<i32>>();
            for concurrency in [1, 16, 64] {
                for unit_rows in [32, 128] {
                    let capacity = RequestCapacity::new(concurrency, unit_rows)?;
                    let mut banks = Vec::new();
                    let mut metadata = Vec::new();
                    let mut expected = BTreeMap::new();
                    let mut add = |id: String, bytes, location| {
                        assert!(expected.insert(id, (bytes, location, Lifetime::TARGET)).is_none());
                    };
                    for (rank, &id) in ids.iter().enumerate() {
                        let location = Location::coordinator(rank as u8);
                        let device = Device { library: &library, id };
                        let readers = (0..4).map(|_| Stream::new(device).map(Rc::new)).collect::<Result<Vec<_>>>()?;
                        let first = rank * cfg.layers / ranks as usize;
                        let end = (rank + 1) * cfg.layers / ranks as usize;
                        let layout = MsaCache::new((end - first) as u32, pool_pages,
                            capacity.decode.requests, context)?;
                        let bank = Rc::new(PageBank::new(readers.clone(), &cfg, capacity, layout,
                            MsaRope::new(&cfg, context)?, first..end)?);
                        for layer in first..end {
                            for (plane, stride) in PAGE_PLANES {
                                add(Owner::Kv { layer, plane }.id(), u64::from(pool_pages) * stride as u64, location);
                            }
                        }
                        for (name, bytes) in capacity.cache(context)?.buffers()?.into_iter()
                            .filter(|(name, _)| !name.starts_with("layer.")) {
                            add(format!("{}.{name}", Owner::Cache { gpu: rank }.id()), bytes, location);
                        }
                        add(format!("{}.upload", Owner::DecodeTable { gpu: rank }.id()),
                            u64::from(capacity.decode.requests) * u64::from(layout.table_width) * 4, Location::PINNED);
                        for (name, bytes) in bank.rope.buffers() {
                            add(format!("{}.{name}", Owner::Rope { gpu: rank }.id()), bytes, location);
                        }
                        for slot in 0..concurrency {
                            for (plane, bytes) in [("table", u64::from(layout.table_width) * 4), ("length", 4)] {
                                add(Owner::PrefixH2d { gpu: rank, slot, plane }.id(), bytes, Location::PINNED);
                            }
                        }
                        for (stream_index, stream) in readers.iter().enumerate() {
                            let prefill = stream_index != 0;
                            let owner = if prefill { Owner::Unit { gpu: rank, slot: (stream_index - 1) as u64 } }
                                else { Owner::Decode { gpu: rank } };
                            let prefix = format!("{}.metadata", owner.id());
                            for (name, bytes) in capacity.metadata(prefill).buffers() {
                                add(format!("{prefix}.{name}"), bytes, location);
                            }
                            for (name, bytes) in capacity.metadata_uploads(prefill) {
                                add(format!("{prefix}.{name}.upload"), bytes, Location::PINNED);
                            }
                            let metadata_owner = if prefill { Metadata::new_prefill(Rc::clone(stream), capacity)? }
                                else { Metadata::new_decode(Rc::clone(stream), capacity)? };
                            metadata.push((metadata_owner, format!("{}.metadata", owner.inventory_name().unwrap())));
                        }
                        banks.push(bank);
                    }
                    let scratch = Rc::new(Allocation::new(banks[0].stream.device, 256,
                        Tag::workspace("minimax/unit"))?);
                    let prefix = MiniMaxPrefix::new(banks.clone(), scratch)?;
                    let collect = || -> Result<Inventory<'_>> {
                        let mut inventory = Inventory::new(CoordinatorDevices::new(&ids)?, "minimax",
                            Lifetime::TARGET, Category::Workspace, "retained request owners");
                        for (rank, bank) in banks.iter().enumerate() { bank.inventory(&mut inventory, rank as u8)?; }
                        for (owner, name) in &metadata { owner.inventory(&mut inventory, name)?; }
                        prefix.upload_inventory(&mut inventory)?;
                        Ok(inventory)
                    };
                    let inventory = collect()?;
                    let actual = inventory.receipts.iter().map(|receipt| {
                        let owner = &receipt.allocation;
                        assert!(owner.view.is_none() && receipt.address != 0);
                        (owner.id.clone(), (owner.bytes, owner.location, owner.lifetime))
                    }).collect::<BTreeMap<_, _>>();
                    assert_eq!(actual.len(), inventory.receipts.len(), "duplicate physical owner ID");
                    assert_eq!(actual, expected, "{ranks} ranks C{concurrency} unit{unit_rows}");
                    let second = collect()?;
                    assert_eq!(inventory.receipts.iter().map(|r| (&r.allocation.id, r.address)).collect::<Vec<_>>(),
                        second.receipts.iter().map(|r| (&r.allocation.id, r.address)).collect::<Vec<_>>(),
                        "inventory must only observe stable retained backing");
                    drop(prefix); drop(metadata); drop(banks);
                    for &id in &ids {
                        assert_eq!(Device { library: &library, id }.run(|| library.cuda_physical_memory_info())?.0, 64 << 20);
                    }
                }
            }
        }
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
