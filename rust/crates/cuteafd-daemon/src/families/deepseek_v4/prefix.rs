//! DeepSeek V4 (Flash / Pro) as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! Paged state, one 256-token unit per page index: on each compressed layer the unit's C4 page
//! (64 FP8 584-byte records, 37,440 B) and its index page (64 x 128 FP8 keys and their FP32
//! scales, 8,448 B), or its C128 page (2 records, 1,728 B): 3.9 KB per token on Flash, 5.6 KB on
//! Pro. Units are refcounted: full units are shared, nobody writes them again (every sequence
//! appends past its own length; a group's record is written once, when its last row is
//! processed), and a partial tail unit is copied whole (its rows past the snapshot are the
//! owner's, and the new sequence writes them before anything reads them).
//!
//! Mark: what is positional and per sequence. Every layer's window ring (the dSpark stages'
//! too, so a restored sequence drafts warm) keeps the last 128 positions: 128 x 584 B per layer
//! = 3.4 MB (Flash, 43 + 3 layers) / 4.8 MB (Pro, 61 + 3); the compressed layers' FP32 rolling
//! state (rows addressed by absolute position mod 16 for C4, mod 256 for C128) is copied whole:
//! 24.3 / 37.4 MB. A restore copies the window rows into the new sequence's own ring (same ring
//! geometry, so the same ring offsets) and the state into its own state slot. Restores are exact
//! frontiers only (`ReuseRule::EXACT`): no aligned position makes the C4 overlap state
//! unnecessary, so a snapshot exists at the length it was captured and nowhere else, and
//! `capture_reach` is 0. The commit point is the placement's length: verify steps write their
//! rejected rows at positions past it and set the length back.
//!
//! Every copy is enqueued on the engine stream, in order with the forward passes, and `drain`
//! synchronizes it. Under a head split each GPU keeps identical caches and compressor state
//! (the split replicates them): units, window rows and state are copied on each GPU's stream,
//! into each GPU's own mark arena (the same slot on both).
//!
//! Under `context` (#58) a unit's C4 pages live once, unit id `i` on GPU `i & 1` at local page
//! `i >> 1` ([`PageOwners::Parity`]); its C128 pages and every mark stay replicated. A tail copy
//! runs the C4 pages on the owner and the C128 pages on both GPUs; the host tier stores each
//! unit's C4 pages from its owner and its C128 pages and marks from GPU0, then copies those to
//! GPU1 after a host restore ([`PrefixFamily::page_replicas`], [`PrefixFamily::mark_replicas`]).
use super::engine::Engine;
use super::metadata::{compressed_page_bytes, INDEX_PAGE_BYTES, MAIN_PAGE_BYTES, SOURCE_PAGE_TOKENS, WINDOW};
use super::pool::{Placement, PoolAllocator};
use crate::shared::memory::DeviceAllocation;
use crate::shared::prefix::view;
use crate::shared::spark_intake::SparkLink;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PageOwners, PrefixFamily, ReuseRule,
    TailCopy};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::DeviceRange;

/// An FP8 record's payload (448 E4M3 + 64 BF16 RoPE dims) and its scales (7 UE8M0 + pad).
/// A page holds every row's payload, then every row's scales.
const PAYLOAD_BYTES: usize = 576;
const SCALE_BYTES: usize = 8;
/// A layer's window rows in a mark: 128 payloads, then 128 scales.
const WINDOW_MARK_BYTES: usize = WINDOW * (PAYLOAD_BYTES + SCALE_BYTES);
/// C4 index page: 64 rows of 128 FP8 keys, then 64 FP32 scales.
const INDEX_ROWS: usize = 64;
const INDEX_KEY_BYTES: usize = 128;

/// One layer's device buffers.
struct LayerBuffers {
    /// Logical backbone layer, followed by GPU0-only dSpark stages.
    layer: usize,
    /// The head-split rank whose GPU holds them (0 without a split).
    rank: usize,
    /// Window pool (every sequence's ring).
    main: CuteafdDeviceBuffer,
    /// Compression ratio (0: window only) and the compressed pool.
    compressed: Option<(usize, CuteafdDeviceBuffer)>,
    /// C4 index pool.
    index: Option<CuteafdDeviceBuffer>,
    /// Compressor state arrays and each sequence's stride in them.
    states: Vec<(CuteafdDeviceBuffer, usize)>,
}

pub(crate) struct Dsv4Prefix<'e, 'a> {
    engine: &'e Engine<'a>,
    layers: Vec<LayerBuffers>,
    /// One mark's bytes on each rank's GPU, and their sum.
    rank_mark_bytes: Vec<usize>,
    mark_bytes: usize,
    /// Per rank: its marks' arena.
    arenas: Vec<DeviceAllocation<'a>>,
    slots: usize,
}

fn layer_buffers(engine: &Engine<'_>) -> Result<Vec<LayerBuffers>> {
    let sequences = engine.shape.sequences;
    (0..engine.ranks()).flat_map(|rank| engine.caches_on(rank).iter().map(move |cache| (rank, cache))).map(|(rank, cache)| {
        let compressed = match (&cache.compressed, &cache.index) {
            (Some(c), Some(_)) => Some((4, c.buffer)),
            (Some(c), None) => Some((128, c.buffer)),
            _ => None,
        };
        let states = cache.states.iter().map(|s| {
            ensure!(s.buffer.bytes % sequences == 0, "compressor state of {} B for {sequences} sequences", s.buffer.bytes);
            Ok((s.buffer, s.buffer.bytes / sequences))
        }).collect::<Result<_>>()?;
        Ok(LayerBuffers { layer: cache.layer, rank, main: cache.main.buffer, compressed, index: cache.index.as_ref().map(|i| i.buffer), states })
    }).collect()
}

impl<'e, 'a> Dsv4Prefix<'e, 'a> {
    /// The family over `engine`'s buffers with a device arena of `slots(mark_bytes)` marks.
    pub fn new(engine: &'e Engine<'a>, slots: impl FnOnce(usize) -> usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix");
        let layers = layer_buffers(engine)?;
        let shape = engine.shape;
        for layer in &layers {
            ensure!(layer.main.bytes >= shape.window_pages() * MAIN_PAGE_BYTES, "window pool smaller than the rings");
            if let Some((ratio, buffer)) = layer.compressed {
                let units = if ratio == 4 { engine.c4_local_units() } else { shape.units };
                ensure!(buffer.bytes >= units * compressed_page_bytes(ratio), "C{ratio} pool smaller than the units");
            }
        }
        let rank_mark_bytes: Vec<usize> = (0..engine.ranks()).map(|rank| layers.iter().filter(|l| l.rank == rank)
            .map(Self::layer_mark_bytes).sum()).collect();
        Self::validate_layers(&layers)?;
        let mark_bytes = rank_mark_bytes.iter().sum();
        let slots = slots(mark_bytes);
        let arenas = if slots > 0 {
            rank_mark_bytes.iter().enumerate().map(|(rank, &bytes)| {
                engine.on(rank, || DeviceAllocation::new(engine.library, (slots * bytes).max(256)))
            }).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok(Self { engine, layers, rank_mark_bytes, mark_bytes, arenas, slots })
    }

    /// Geometry agrees only within replicas of the same logical layer; an owned
    /// layer may exist solely on GPU1 and need not mirror a GPU0 prefix.
    fn validate_layers(layers: &[LayerBuffers]) -> Result<()> {
        let mut seen = std::collections::BTreeMap::new();
        for layer in layers {
            let geometry = (layer.compressed.map(|(r, _)| r), layer.index.is_some(),
                layer.states.iter().map(|&(_, stride)| stride).collect::<Vec<_>>());
            let replicas = seen.entry(layer.layer).or_insert_with(|| (geometry.clone(), Vec::new()));
            ensure!(replicas.0 == geometry, "logical layer {} has different replica geometry", layer.layer);
            ensure!(!replicas.1.contains(&layer.rank), "logical layer {} repeats rank {}", layer.layer, layer.rank);
            replicas.1.push(layer.rank);
        }
        Ok(())
    }

    fn canonical_layers(&self) -> impl Iterator<Item = &LayerBuffers> {
        canonical_layers(&self.layers)
    }

    /// Marks in the rank arena, paired by logical identity rather than prefix position.
    fn layer_marks(&self, slot: MarkSlot) -> Vec<(&LayerBuffers, DeviceRange)> {
        let mut offsets: Vec<_> = self.rank_mark_bytes.iter().map(|bytes| slot.0 as usize * bytes).collect();
        self.layers.iter().map(|layer| {
            let bytes = Self::layer_mark_bytes(layer);
            let range = DeviceRange { addr: self.arenas[layer.rank].buffer.ptr as u64 + offsets[layer.rank] as u64, bytes };
            offsets[layer.rank] += bytes;
            (layer, range)
        }).collect()
    }

    /// One layer's share of a mark: its window rows and compressor state.
    fn layer_mark_bytes(layer: &LayerBuffers) -> usize {
        WINDOW_MARK_BYTES + layer.states.iter().map(|&(_, stride)| stride).sum::<usize>()
    }

    /// Bytes of one mark for `engine` (window rows and compressor state of every layer, on every GPU).
    pub fn mark_bytes_of(engine: &Engine<'_>) -> Result<usize> {
        Ok(layer_buffers(engine)?.iter().map(Self::layer_mark_bytes).sum())
    }

    pub fn mark_bytes(&self) -> usize {
        self.mark_bytes
    }

    pub fn page_bytes(&self) -> usize {
        // One rank's layers: under heads every GPU holds every page, under context a unit's C4
        // pages live on one GPU and its C128 pages are stored from GPU0.
        self.canonical_layers().map(|l| match l.compressed {
            Some((4, _)) => compressed_page_bytes(4) + INDEX_PAGE_BYTES,
            Some((ratio, _)) => compressed_page_bytes(ratio),
            None => 0,
        }).sum()
    }

    pub fn slots(&self) -> usize {
        if self.arenas.is_empty() { 0 } else { self.slots }
    }

    /// A device buffer of the engine (the host tier's copy-engine template).
    pub fn template(&self) -> CuteafdDeviceBuffer {
        self.layers[0].main
    }

    /// A copy between two views on rank `rank`'s GPU.
    fn copy(&self, rank: usize, dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer) -> Result<()> {
        debug_assert_eq!(dst.bytes, src.bytes);
        // SAFETY: both views lie inside live allocations of that GPU (checked by `view`); the copy
        // is ordered on its stream with every forward pass that reads or writes them.
        self.engine.on(rank, || unsafe {
            self.engine.library.copy_d2d_async(dst, src, src.bytes, self.engine.stream_of(rank))
        })
    }

    /// Whether C4 pages are partitioned by unit parity (context attention).
    fn parity(&self) -> bool {
        self.engine.context_kv()
    }

    /// The device ranges of one unit on every GPU that holds them, per compressed layer: the C4
    /// page and its index page, or the C128 page, as (rank, range, C4). Under `context` a C4 page
    /// is listed once, on its owner at its local page; a C128 page on both GPUs.
    fn unit_ranges(&self, unit: u32) -> Result<Vec<(usize, CuteafdDeviceBuffer, bool)>> {
        let (owner, local) = (unit as usize & 1, unit as usize >> 1);
        let mut out = Vec::new();
        for layer in &self.layers {
            let Some((ratio, buffer)) = layer.compressed else { continue };
            let c4 = ratio == 4;
            let page = match (c4 && self.parity(), layer.rank == owner) {
                (true, false) => continue,
                (true, true) => local,
                (false, _) => unit as usize,
            };
            let bytes = compressed_page_bytes(ratio);
            out.push((layer.rank, view(buffer, page * bytes, bytes)?, c4));
            if let Some(index) = layer.index {
                out.push((layer.rank, view(index, page * INDEX_PAGE_BYTES, INDEX_PAGE_BYTES)?, c4));
            }
        }
        Ok(out)
    }

    /// Every allocation the host tier copies (both GPUs' compressed pools and mark arenas), for
    /// the copy engine's routing by owning device.
    pub fn buffers(&self) -> Vec<CuteafdDeviceBuffer> {
        self.layers.iter().flat_map(|l| l.compressed.map(|(_, b)| b).into_iter().chain(l.index))
            .chain(self.arenas.iter().map(|a| a.buffer)).collect()
    }

    /// Copy `placement`'s positional state at `len` to or from mark `slot`: per layer the window
    /// rows of positions `[len - min(len, 128), len)` (mark row `k` is position
    /// `len - min(len, 128) + k`), then every compressor state array's slice of its state slot.
    fn move_mark(&self, slot: MarkSlot, placement: &Placement, len: usize, capture: bool) -> Result<()> {
        ensure!(!self.arenas.is_empty(), "no mark arena");
        let shape = self.engine.shape;
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        ensure!(placement.state < shape.sequences, "state slot {} of {}", placement.state, shape.sequences);
        let first = len - len.min(WINDOW);
        let mut offsets: Vec<usize> = self.rank_mark_bytes.iter().map(|&bytes| slot.0 as usize * bytes).collect();
        for layer in &self.layers {
            let (rank, arena) = (layer.rank, self.arenas[layer.rank].buffer);
            let offset = &mut offsets[rank];
            let copy = |ring: CuteafdDeviceBuffer, mark: CuteafdDeviceBuffer| {
                if capture { self.copy(rank, mark, ring) } else { self.copy(rank, ring, mark) }
            };
            // Positions [first, len) in runs that stay inside one window page.
            let mut position = first;
            while position < len {
                let row = position % SOURCE_PAGE_TOKENS;
                let count = (len - position).min(SOURCE_PAGE_TOKENS - row);
                let page = placement.window_slot(&shape, position) as usize / SOURCE_PAGE_TOKENS;
                let (base, k) = (page * MAIN_PAGE_BYTES, position - first);
                copy(view(layer.main, base + row * PAYLOAD_BYTES, count * PAYLOAD_BYTES)?,
                    view(arena, *offset + k * PAYLOAD_BYTES, count * PAYLOAD_BYTES)?)?;
                copy(view(layer.main, base + SOURCE_PAGE_TOKENS * PAYLOAD_BYTES + row * SCALE_BYTES, count * SCALE_BYTES)?,
                    view(arena, *offset + WINDOW * PAYLOAD_BYTES + k * SCALE_BYTES, count * SCALE_BYTES)?)?;
                position += count;
            }
            *offset += WINDOW_MARK_BYTES;
            for &(state, stride) in &layer.states {
                copy(view(state, placement.state * stride, stride)?, view(arena, *offset, stride)?)?;
                *offset += stride;
            }
        }
        Ok(())
    }
}

impl PrefixFamily for Dsv4Prefix<'_, '_> {
    type Placement = Placement;

    fn layout(&self) -> FamilyLayout {
        FamilyLayout {
            page_rows: SOURCE_PAGE_TOKENS,
            pages: self.engine.shape.units,
            page_bytes: self.page_bytes(),
            mark_bytes: self.canonical_layers().map(Self::layer_mark_bytes).sum(),
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena,
            page_owners: if self.parity() { PageOwners::Parity } else { PageOwners::Uniform },
        }
    }

    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] {
        &placement.units
    }

    fn commit_point(&self, placement: &Placement) -> usize {
        placement.len
    }

    fn capture(&self, slot: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        if len != placement.len || len == 0 {
            return Err(format!("capture at {len}: the placement holds {} rows", placement.len).into());
        }
        Ok(self.move_mark(slot, placement, len, true)?)
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        let Some(slot) = mark else {
            return Err("DeepSeek V4 restores exact snapshots with their window and compressor state".into());
        };
        if len == 0 || len.div_ceil(SOURCE_PAGE_TOKENS) > placement.units.len() {
            return Err(format!("restore of {len} rows into {} units", placement.units.len()).into());
        }
        self.move_mark(slot, placement, len, false)?;
        placement.len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        // Whole pages: the records of groups the copied rows complete, and nothing anyone reads
        // past them. Under context both units have the tail's parity: C4 pages copy on the owner,
        // C128 pages on both GPUs.
        if self.parity() != copy.owner.is_some() || copy.owner.is_some_and(|o| u32::from(o) != copy.from & 1
            || u32::from(o) != copy.to & 1) {
            return Err(format!("tail copy {copy:?} crosses GPUs").into());
        }
        for ((rank, from, _), (to_rank, to, _)) in self.unit_ranges(copy.from)?.into_iter().zip(self.unit_ranges(copy.to)?) {
            if rank != to_rank { return Err(format!("tail copy {copy:?} pairs pages of two GPUs").into()); }
            self.copy(rank, to, from)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        let mut first_error = None;
        for rank in 0..self.engine.ranks() {
            // SAFETY: the engine owns these streams; attempt every rank even if one fails.
            if let Err(error) = unsafe { self.engine.library.cuda_stream_synchronize(self.engine.stream_of(rank)) } {
                if first_error.is_none() { first_error = Some(error); }
            }
        }
        first_error.map_or(Ok(()), |error| Err(error.into()))
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        let mut out = Vec::new();
        for layer in snapshot_page_layers(&self.layers, self.parity().then_some(page as usize & 1)) {
            let (ratio, buffer) = layer.compressed.expect("snapshot pages are compressed");
            let local = if ratio == 4 && self.parity() { page as usize >> 1 } else { page as usize };
            let bytes = compressed_page_bytes(ratio);
            let Ok(buffer) = view(buffer, local * bytes, bytes) else { return Vec::new() };
            out.push(DeviceRange { addr: buffer.ptr as u64, bytes });
            if let Some(index) = layer.index {
                let Ok(buffer) = view(index, local * INDEX_PAGE_BYTES, INDEX_PAGE_BYTES) else { return Vec::new() };
                out.push(DeviceRange { addr: buffer.ptr as u64, bytes: INDEX_PAGE_BYTES });
            }
        }
        out
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        let marks = self.layer_marks(slot);
        self.canonical_layers().map(|layer| marks.iter().find(|(l, _)| l.layer == layer.layer && l.rank == layer.rank)
            .expect("canonical layer has a mark").1).collect()
    }

    fn page_device(&self, page: u32) -> Option<u8> {
        self.parity().then_some((page & 1) as u8)
    }

    fn page_replicas(&self, page: u32) -> Vec<(DeviceRange, DeviceRange)> {
        let mut replicas = Vec::new();
        for layer in self.canonical_layers() {
            let Some((ratio, buffer)) = layer.compressed else { continue };
            if ratio == 4 && self.parity() { continue; }
            let Some(peer) = self.layers.iter().find(|l| l.layer == layer.layer && l.rank != layer.rank) else { continue };
            let from = DeviceRange { addr: buffer.ptr as u64 + page as u64 * compressed_page_bytes(ratio) as u64,
                bytes: compressed_page_bytes(ratio) };
            let to = DeviceRange { addr: peer.compressed.expect("validated replica").1.ptr as u64
                + page as u64 * from.bytes as u64, bytes: from.bytes };
            replicas.push((from, to));
            if let (Some(a), Some(b)) = (layer.index, peer.index) {
                replicas.push((DeviceRange { addr: a.ptr as u64 + page as u64 * INDEX_PAGE_BYTES as u64, bytes: INDEX_PAGE_BYTES },
                    DeviceRange { addr: b.ptr as u64 + page as u64 * INDEX_PAGE_BYTES as u64, bytes: INDEX_PAGE_BYTES }));
            }
        }
        replicas
    }

    fn mark_replicas(&self, slot: MarkSlot) -> Vec<(DeviceRange, DeviceRange)> {
        let marks = self.layer_marks(slot);
        self.canonical_layers().flat_map(|layer| {
            let from = marks.iter().find(|(l, _)| l.layer == layer.layer && l.rank == layer.rank).expect("canonical mark").1;
            marks.iter().filter(move |(l, _)| l.layer == layer.layer && l.rank != layer.rank).map(move |(_, to)| (from, *to))
        }).collect()
    }

    fn copy_replicas(&self, copies: &[(DeviceRange, DeviceRange)]) -> Result<(), BoxError> {
        if copies.is_empty() { return Ok(()); }
        let buffers = self.buffers();
        let buffer = |range: DeviceRange| -> Result<CuteafdDeviceBuffer> {
            let allocation = buffers.iter().find(|b| range.addr >= b.ptr as u64
                && range.addr.checked_add(range.bytes as u64).is_some_and(|end| end <= b.ptr as u64 + b.bytes as u64))
                .context("replica range is not registered")?;
            view(*allocation, (range.addr - allocation.ptr as u64) as usize, range.bytes)
        };
        for &(from, to) in copies {
            let (from, to) = (buffer(from)?, buffer(to)?);
            if from.bytes != to.bytes { return Err("replica ranges differ in length".into()); }
            let rank = self.arenas.iter().position(|a| a.buffer.device_id == to.device_id)
                .context("replica destination device is absent")?;
            self.copy(rank, to, from)?;
        }
        Ok(())
    }

}

fn canonical_layers(layers: &[LayerBuffers]) -> impl Iterator<Item = &LayerBuffers> {
    let mut canonical = std::collections::BTreeMap::new();
    for layer in layers {
        let entry = canonical.entry(layer.layer).or_insert(layer);
        if layer.rank < entry.rank { *entry = layer; }
    }
    canonical.into_values()
}

/// Canonical logical order is identical for both parities; only the C4 source
/// rank changes under context placement. Owned layers keep their actual rank.
fn snapshot_page_layers(layers: &[LayerBuffers], parity: Option<usize>) -> Vec<&LayerBuffers> {
    canonical_layers(layers).filter_map(|layer| {
        let (ratio, _) = layer.compressed?;
        match parity {
            Some(rank) if ratio == 4 => layers.iter().find(|l| l.layer == layer.layer && l.rank == rank),
            _ => Some(layer),
        }
    }).collect()
}

/// FNV-1a over 32-bit words: a digest of streams and logits fast enough for gigabytes.
#[derive(Clone, Copy)]
struct Digest(u64);

impl Default for Digest {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Digest {
    fn word(&mut self, word: u32) {
        self.0 = (self.0 ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn bytes(&mut self, bytes: &[u8]) {
        let words = bytes.chunks_exact(4);
        let tail = words.remainder().iter().fold(0u32, |w, &b| (w << 8) | u32::from(b));
        words.for_each(|w| self.word(u32::from_le_bytes(w.try_into().expect("4 bytes"))));
        self.word(tail);
        self.word(bytes.len() as u32);
    }
}

/// One continued prefill: every layer's output digest (single-lane chunks), every row's logits
/// digest and argmax, and the last row.
pub(crate) struct SuffixRun {
    pub layers: Vec<u64>,
    pub logits: u64,
    pub argmax: Vec<usize>,
    pub last: Vec<f32>,
}

/// What a resume check runs the engine with.
pub(crate) struct Runner<'r, 'e, 'a> {
    pub engine: &'r Engine<'a>,
    pub transports: &'r mut [SparkLink<'e>],
    pub runtime: &'r tokio::runtime::Runtime,
    /// Prefill without per-layer downloads, so chunks of 512 rows or more run as two lanes (the
    /// serving path); layers are then not digested.
    pub lanes: bool,
}

impl Runner<'_, '_, '_> {
    /// Prefill `tokens` into `placement` in chunks of `chunk` rows, digesting each layer's streams
    /// (unless `lanes`) and (with `logits`) every row's logits.
    fn prefill(&mut self, placement: &mut Placement, tokens: &[u32], chunk: usize, logits: bool) -> Result<SuffixRun> {
        let vocab = self.engine.cfg.vocab_size;
        let mut layers: Vec<Digest> = Vec::new();
        let mut logit_digest = Digest::default();
        let (mut argmax, mut last) = (Vec::new(), Vec::new());
        for part in tokens.chunks(chunk) {
            let mut on_layer = |layer: usize, rows: &[u8]| -> Result<()> {
                if layers.len() <= layer {
                    layers.resize_with(layer + 1, Default::default);
                }
                layers[layer].bytes(rows);
                Ok(())
            };
            let rows = if logits { part.len() } else { 0 };
            let digest: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
                if self.lanes { None } else { Some(&mut on_layer) };
            let out = self.engine.prefill(placement, part, self.transports, self.runtime, rows, digest)?;
            if logits {
                for values in out.chunks_exact(vocab) {
                    values.iter().for_each(|v| logit_digest.word(v.to_bits()));
                    argmax.push(values.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i));
                }
                last = out[out.len() - vocab..].to_vec();
            }
        }
        Ok(SuffixRun { layers: layers.iter().map(|d| d.0).collect(), logits: logit_digest.0, argmax, last })
    }

    fn decode(&mut self, placement: &mut Placement, token: u32) -> Result<Vec<f32>> {
        self.engine.decode(&mut [(placement, token)], self.transports, self.runtime)
    }
}

fn download(engine: &Engine<'_>, range: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
    // SAFETY: the engine owns this stream; draining it retires every write to `range`.
    for rank in 0..engine.ranks() {
        unsafe { engine.library.cuda_stream_synchronize(engine.stream_of(rank))? };
    }
    let mut bytes = vec![0u8; range.bytes];
    engine.library.copy_d2h(&mut bytes, range)?;
    Ok(bytes)
}

/// Every paged byte of `placement`'s first `len` rows: per unit and compressed layer (of every
/// GPU that holds it), the records (payload, scales) of the groups completed below `len` and, for
/// C4, their index keys and scales.
fn paged_rows(family: &Dsv4Prefix<'_, '_>, placement: &Placement, len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for (u, &unit) in placement.units.iter().enumerate().take(len.div_ceil(SOURCE_PAGE_TOKENS)) {
        let mut ranges = family.unit_ranges(unit)?.into_iter();
        let owner = unit as usize & 1;
        for layer in &family.layers {
            let Some((ratio, _)) = layer.compressed else { continue };
            if ratio == 4 && family.parity() && layer.rank != owner { continue; }
            let page_rows = SOURCE_PAGE_TOKENS / ratio;
            let rows = (len / ratio).saturating_sub(u * page_rows).min(page_rows);
            let page = download(family.engine, ranges.next().context("unit range")?.1)?;
            out.extend_from_slice(&page[..rows * PAYLOAD_BYTES]);
            out.extend_from_slice(&page[page_rows * PAYLOAD_BYTES..][..rows * SCALE_BYTES]);
            if layer.index.is_some() {
                let page = download(family.engine, ranges.next().context("index range")?.1)?;
                out.extend_from_slice(&page[..rows * INDEX_KEY_BYTES]);
                out.extend_from_slice(&page[INDEX_ROWS * INDEX_KEY_BYTES..][..rows * 4]);
            }
        }
    }
    Ok(out)
}

/// The bytes mark `slot` holds for a snapshot at `len` on every GPU, in rank order: per layer its
/// `min(len, 128)` window rows (rows past them are left over from earlier marks) and its compressor
/// state. A head split's GPU1 marks are compared too (they must equal GPU0's backbone prefix).
fn mark(family: &Dsv4Prefix<'_, '_>, slot: u32, len: usize) -> Result<Vec<u8>> {
    let rows = len.min(WINDOW);
    let mut out = Vec::new();
    for (rank, (arena, &mark_bytes)) in family.arenas.iter().zip(&family.rank_mark_bytes).enumerate() {
        let bytes = download(family.engine, view(arena.buffer, slot as usize * mark_bytes, mark_bytes)?)?;
        let mut offset = 0;
        for layer in family.layers.iter().filter(|l| l.rank == rank) {
            out.extend_from_slice(&bytes[offset..][..rows * PAYLOAD_BYTES]);
            out.extend_from_slice(&bytes[offset + WINDOW * PAYLOAD_BYTES..][..rows * SCALE_BYTES]);
            offset += WINDOW_MARK_BYTES;
            let states: usize = layer.states.iter().map(|&(_, stride)| stride).sum();
            out.extend_from_slice(&bytes[offset..][..states]);
            offset += states;
        }
    }
    Ok(out)
}

/// Exercise production canonical segments and device-routed copy streams. The
/// destination has independent units and a separate mark: no shared page or
/// device-to-device snapshot copy can accidentally satisfy the host proof.
fn host_round_trip(family: &Dsv4Prefix<'_, '_>, source: &Placement, destination: &mut Placement, len: usize) -> Result<()> {
    use cuteafd_hostcache::copy::{CopyEngine, Stream};
    use cuteafd_hostcache::pool::{HostRange, PinnedMemory};
    let err = |e: BoxError| anyhow::anyhow!("{e}");
    let count = len.div_ceil(SOURCE_PAGE_TOKENS);
    ensure!(source.units.len() >= count && destination.units.len() >= count, "host proof has too few units");
    let mut pairs = Vec::new();
    for (&from, &to) in source.units.iter().zip(&destination.units).take(count) {
        let (a, b) = (family.page_segments(from), family.page_segments(to));
        ensure!(a.iter().map(|r| r.bytes).sum::<usize>() == family.page_bytes()
            && b.iter().map(|r| r.bytes).sum::<usize>() == family.page_bytes()
            && a.len() == b.len() && a.iter().zip(&b).all(|(a, b)| a.bytes == b.bytes), "host page geometries differ");
        pairs.extend(a.into_iter().zip(b));
    }
    let (a, b) = (family.mark_segments(MarkSlot(0)), family.mark_segments(MarkSlot(1)));
    ensure!(a.len() == b.len() && a.iter().zip(&b).all(|(a, b)| a.bytes == b.bytes), "host mark geometries differ");
    pairs.extend(a.into_iter().zip(b));
    let bytes = pairs.iter().try_fold(0usize, |sum, (a, _)| sum.checked_add(a.bytes)).context("host snapshot size overflow")?;
    family.drain().map_err(err)?;
    // The family and engine remain borrowed until both copy streams drain.
    let mut copy = crate::shared::prefix::CudaCopyEngine::registered(family.engine.library, &family.buffers())?;
    let host = copy.allocate_chunk(bytes)?;
    let mut offset = 0;
    let extents: Vec<_> = pairs.iter().map(|&(from, to)| {
        let range = HostRange { chunk: host.id, offset, bytes: from.bytes };
        offset += from.bytes;
        (from, range, to)
    }).collect();
    let result = (|| -> Result<()> {
        copy.d2h_many(Stream::Store, &extents.iter().map(|&(from, host, _)| (from, host)).collect::<Vec<_>>())?;
        copy.synchronize(Stream::Store)?;
        copy.h2d_many(Stream::Restore, &extents.iter().map(|&(_, host, to)| (host, to)).collect::<Vec<_>>())?;
        copy.synchronize(Stream::Restore)?;
        for &page in destination.units.iter().take(count) {
            family.copy_replicas(&family.page_replicas(page)).map_err(err)?;
        }
        family.copy_replicas(&family.mark_replicas(MarkSlot(1))).map_err(err)?;
        family.restore(Some(MarkSlot(1)), destination, len).map_err(err)?;
        family.drain().map_err(err)?;
        Ok(())
    })();
    // Attempt every barrier even after issue failure. Quarantine prevents the
    // engine's allocations being freed if queued copies cannot be retired.
    let store = copy.synchronize(Stream::Store);
    let restore = copy.synchronize(Stream::Restore);
    let device = family.drain().map_err(err);
    if let Err(error) = store.and(restore).and(device) {
        family.engine.library.quarantine_module_after_failed_drain();
        return Err(error.context("host exact-cache proof failed to drain"));
    }
    copy.release_chunk(host)?;
    result
}

/// One `--resume-at` case.
pub(crate) struct ResumeCase {
    pub at: usize,
    pub n: usize,
    pub chunk: usize,
}

/// dsv4-golden --resume-at: for each case prefill `tokens[..P]` into sequence A, capture its
/// canonical snapshot to pinned host memory, promote it into sequence B's independent
/// units and state slot, compare same-chunk cold-prefill bytes, and continue both with the same chunks and `decode` greedy
/// single-row steps, and compare every layer's rows, every logit, every paged row, the final
/// positional state of both and the mark round trip byte for byte (a restore must be exact). A
/// straight prefill without the boundary at P is reported too (informational: chunking changes
/// may round differently). With `cold`, B is prefilled from scratch on its own units instead (no
/// restore): the floor of what the kernels themselves vary. Each case runs `repeat` times on
/// fresh sequences. Returns whether every case passed.
pub(crate) fn resume_check(runner: &mut Runner<'_, '_, '_>, tokens: &[u32], cases: &[ResumeCase], decode: usize,
    cold: bool, repeat: usize) -> Result<bool> {
    let engine = runner.engine;
    ensure!(engine.shape.sequences >= 3, "--resume-at needs --max-sequences 3 or more");
    let family = Dsv4Prefix::new(engine, |_| 3)?;
    let mut allocator = PoolAllocator::for_engine(engine)?;
    let mut passed = true;
    for case in cases {
        match resume_case(runner, &family, &mut allocator, tokens, case, decode, cold, repeat) {
            Ok(ok) => passed &= ok,
            Err(error) => {
                println!("resume at {} of {} (chunks of {}): FAILED: {error:#}", case.at, case.n, case.chunk);
                passed = false;
            }
        }
    }
    Ok(passed)
}

#[allow(clippy::too_many_arguments)]
fn resume_case(runner: &mut Runner<'_, '_, '_>, family: &Dsv4Prefix<'_, '_>, allocator: &mut PoolAllocator,
    tokens: &[u32], case: &ResumeCase, decode: usize, cold: bool, repeat: usize) -> Result<bool> {
    let engine = runner.engine;
    let &ResumeCase { at, n, chunk } = case;
    ensure!(at > 0 && at < n && n <= tokens.len(), "--resume-at {at} must lie inside the {n} prefilled tokens");
    ensure!(n + decode < engine.max_context, "{n} + {decode} tokens exceed the {}-token context", engine.max_context);
    let chunk = chunk.clamp(1, if runner.lanes { engine.prefill_capacity() } else { engine.prefill_rows });
    let embed = &tokens[..n];
    let err = |e: BoxError| anyhow::anyhow!("{e}");
    let repeat = repeat.max(1);
    let (mut identical, mut marks_identical, mut last) = (0, 0, None);
    for _ in 0..repeat {
        // A: prefill [0, P), capture, continue in place.
        let mut a = allocator.admit(n + decode)?;
        runner.prefill(&mut a, &embed[..at], chunk, false)?;
        family.drain().map_err(err)?;
        let started = std::time::Instant::now();
        family.capture(MarkSlot(0), &a, at).map_err(err)?;
        // B: host-demote the canonical snapshot and promote into independent
        // units, a separate mark and its own ring/state slot.
        let mut b = if cold {
            let mut b = allocator.admit(n + decode)?;
            runner.prefill(&mut b, &embed[..at], chunk, false)?;
            family.capture(MarkSlot(1), &b, at).map_err(err)?;
            family.restore(Some(MarkSlot(1)), &mut b, at).map_err(err)?;
            b
        } else {
            let mut b = allocator.admit(n + decode)?;
            host_round_trip(family, &a, &mut b, at)?;
            b
        };
        family.drain().map_err(err)?;
        let restore_ms = started.elapsed().as_secs_f64() * 1e3;
        // The restored state reads back exactly as the captured one, and every paged row agrees.
        family.capture(MarkSlot(1), &b, at).map_err(err)?;
        let mark_equal = mark(family, 0, at)? == mark(family, 1, at)?;
        let state_at = paged_rows(family, &a, at)? == paged_rows(family, &b, at)?;
        // And both equal a cold prefill of the same prefix on its own units: every GPU's owned C4
        // rows, its C128 rows, window rows and carry (exact restores, AGENTS.md).
        let cold_equal = {
            let mut c = allocator.admit(at + 1)?;
            let cold = runner.prefill(&mut c, &embed[..at], chunk, false).and_then(|_| {
                family.drain().map_err(err)?;
                family.capture(MarkSlot(2), &c, at).map_err(err)?;
                Ok(paged_rows(family, &c, at)? == paged_rows(family, &b, at)? && mark(family, 2, at)? == mark(family, 1, at)?)
            });
            allocator.release(c);
            cold?
        };
        let straight = runner.prefill(&mut a, &embed[at..], chunk, true)?;
        let restored = runner.prefill(&mut b, &embed[at..], chunk, true)?;
        let first_layer = straight.layers.iter().zip(&restored.layers).position(|(x, y)| x != y);
        let logits_equal = straight.logits == restored.logits && straight.argmax.len() == n - at;
        let max_diff = straight.last.iter().zip(&restored.last).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        // Greedy single-row decode steps from both (decode graphs, the restored ring and state).
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i as u32);
        let (mut next_a, mut next_b) = (argmax(&straight.last), argmax(&restored.last));
        let mut decode_equal = next_a == next_b;
        for _ in 0..decode {
            let la = runner.decode(&mut a, next_a)?;
            let lb = runner.decode(&mut b, next_b)?;
            decode_equal &= la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits());
            (next_a, next_b) = (argmax(&la), argmax(&lb));
        }
        let len = a.len;
        let paged_equal = paged_rows(family, &a, len)? == paged_rows(family, &b, len)?;
        // Both sequences' positional state at the end (window rows, compressor state).
        family.capture(MarkSlot(1), &a, len).map_err(err)?;
        family.capture(MarkSlot(2), &b, len).map_err(err)?;
        let state_equal = mark(family, 1, len)? == mark(family, 2, len)?;
        println!("resume at {at} of {n} (chunks of {chunk}{}, {decode} decode steps): paged rows at {at} {} | \
            host demote/promote {} | restored == same-chunk cold prefill on every GPU {} | layers {} | \
            logits {} (last row max |diff| {max_diff:.3e}) | decode {} | paged rows 0..{len} {} | window+compressor \
            state {} | mark round trip {} ({} B), capture+restore {restore_ms:.1} ms",
            if runner.lanes { ", lanes" } else { "" }, if state_at { "identical" } else { "DIFFER" },
            if cold { "not exercised (cold floor)" } else { "exercised" }, if cold_equal { "identical" } else { "DIFFERS" },
            if runner.lanes { "not digested".to_string() }
            else { first_layer.map_or("identical".to_string(), |l| format!("differ from layer {l}")) },
            if logits_equal { "identical" } else { "DIFFER" }, if decode_equal { "identical" } else { "DIFFERS" },
            if paged_equal { "identical" } else { "DIFFER" }, if state_equal { "identical" } else { "DIFFERS" },
            if mark_equal { "identical" } else { "DIFFERS" }, family.mark_bytes());
        allocator.release(b);
        allocator.release(a);
        identical += usize::from(first_layer.is_none() && logits_equal && decode_equal && paged_equal && state_equal
            && mark_equal);
        marks_identical += usize::from(mark_equal && state_at && cold_equal);
        last = Some(restored);
    }
    let restored = last.context("no attempt")?;
    // C: one prefill with no boundary at P (chunking changes may round differently; informational).
    let mut c = allocator.admit(n)?;
    let whole = runner.prefill(&mut c, embed, chunk, true);
    allocator.release(c);
    let whole = whole?;
    let x = &whole.argmax[at..];
    let agree = x.iter().zip(&restored.argmax).filter(|(p, q)| p == q).count();
    let last_equal = whole.last.iter().zip(&restored.last).all(|(p, q)| p.to_bits() == q.to_bits());
    println!("vs one straight prefill without the boundary at {at}: suffix top-1 agreement {agree}/{}, last row logits {}",
        x.len(), if last_equal { "identical" } else { "differ" });
    println!("resume at {at} of {n} (chunks of {chunk}): {identical}/{repeat} attempts byte-identical, mark round trip \
        and paged rows at {at} identical in {marks_identical}/{repeat}{}",
        if cold { " [cold floor: B prefilled, not restored]" } else { "" });
    Ok(marks_identical == repeat && identical == repeat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_marks_use_logical_layers_not_a_mirrored_rank_prefix() {
        let buffer = CuteafdDeviceBuffer::default();
        let layer = |layer, rank, ratio| LayerBuffers { layer, rank, main: buffer,
            compressed: (ratio > 0).then_some((ratio, buffer)), index: (ratio == 4).then_some(buffer),
            states: vec![(buffer, ratio * 64)] };
        // C4 owners alternate here only to stress serialization. C128 is replicated;
        // the last window-only dSpark stage is on GPU0.
        let mut layers = vec![layer(0, 0, 4), layer(2, 0, 128), layer(3, 0, 0), layer(1, 1, 4), layer(2, 1, 128)];
        Dsv4Prefix::validate_layers(&layers).unwrap();
        assert_eq!(canonical_layers(&layers).map(|l| (l.layer, l.rank)).collect::<Vec<_>>(), [(0, 0), (1, 1), (2, 0), (3, 0)]);
        layers.swap(0, 2);
        Dsv4Prefix::validate_layers(&layers).unwrap();
        assert_eq!(canonical_layers(&layers).map(|l| l.layer).collect::<Vec<_>>(), [0, 1, 2, 3]);
        layers.push(layer(2, 1, 128));
        assert!(Dsv4Prefix::validate_layers(&layers).is_err());
        layers.pop();
        layers[4].states[0].1 += 1;
        assert!(Dsv4Prefix::validate_layers(&layers).is_err());
    }

    #[test]
    fn context_page_order_is_logical_and_identical_across_parities() {
        let buffer = CuteafdDeviceBuffer::default();
        let layer = |layer, rank, ratio| LayerBuffers { layer, rank, main: buffer,
            compressed: Some((ratio, buffer)), index: (ratio == 4).then_some(buffer), states: Vec::new() };
        // Rank-major iteration would interleave C128 before C4 on odd pages.
        let layers = vec![layer(0, 0, 4), layer(1, 0, 128), layer(2, 0, 4),
            layer(0, 1, 4), layer(1, 1, 128), layer(2, 1, 4)];
        for rank in [0, 1] {
            assert_eq!(snapshot_page_layers(&layers, Some(rank)).iter().map(|l| (l.layer, l.rank)).collect::<Vec<_>>(),
                [(0, rank), (1, 0), (2, rank)]);
        }
        let owned = vec![layer(2, 1, 4), layer(0, 0, 4), layer(1, 0, 128), layer(1, 1, 128)];
        assert_eq!(snapshot_page_layers(&owned, None).iter().map(|l| (l.layer, l.rank)).collect::<Vec<_>>(),
            [(0, 0), (1, 0), (2, 1)]);
    }

    #[test]
    fn mark_rows_and_digest() {
        assert_eq!(WINDOW_MARK_BYTES, 74_752);
        // Flash: 46 window rings (43 layers + 3 stages) of 128 x 584 B = 3.4 MB.
        assert_eq!(46 * WINDOW_MARK_BYTES, 3_438_592);
        let (mut a, mut b) = (Digest::default(), Digest::default());
        a.bytes(&[1, 2, 3, 4, 5]);
        b.bytes(&[1, 2, 3, 4, 6]);
        assert_ne!(a.0, b.0);
        let mut c = Digest::default();
        c.bytes(&[1, 2, 3, 4, 5]);
        assert_eq!(a.0, c.0);
    }
}
