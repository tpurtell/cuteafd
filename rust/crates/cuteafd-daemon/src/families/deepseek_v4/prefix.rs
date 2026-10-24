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
    /// Per rank: the prefix copy stream, fenced against that rank's engine stream.
    copies: crate::shared::prefix::PrefixCopies<'a>,
}

/// Device-to-device copies `(rank, destination, source)` queued as one batch.
type Copies = Vec<(usize, CuteafdDeviceBuffer, CuteafdDeviceBuffer)>;

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
        Ok(LayerBuffers { rank, main: cache.main.buffer, compressed, index: cache.index.as_ref().map(|i| i.buffer), states })
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
        Self::mirror_prefix(&layers)?;
        let mark_bytes = rank_mark_bytes.iter().sum();
        let slots = slots(mark_bytes);
        let arenas = if slots > 0 {
            rank_mark_bytes.iter().enumerate().map(|(rank, &bytes)| {
                engine.on(rank, || DeviceAllocation::new(engine.library, (slots * bytes).max(256)))
            }).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let copies = crate::shared::prefix::PrefixCopies::new(engine.library, (0..engine.ranks())
            .map(|rank| Ok(vec![crate::shared::prefix::Reader {
                device: engine.on(rank, || engine.library.cuda_get_device())?, stream: engine.stream_of(rank) }]))
            .collect::<Result<_>>()?)?;
        Ok(Self { engine, layers, rank_mark_bytes, mark_bytes, arenas, slots, copies })
    }

    /// A head split's GPU1 mark must be byte-for-byte the prefix of GPU0's: GPU1's layers are GPU0's
    /// first layers in the same order with the same mark geometry (GPU0 then adds the dSpark
    /// stages). The host tier stores GPU0's mark and restores GPU1's from that prefix.
    fn mirror_prefix(layers: &[LayerBuffers]) -> Result<()> {
        let geometry = |rank: usize| layers.iter().filter(move |l| l.rank == rank)
            .map(|l| (l.compressed.map(|(r, _)| r), l.index.is_some(), l.states.iter().map(|&(_, s)| s).collect::<Vec<_>>()));
        let (gpu0, gpu1): (Vec<_>, Vec<_>) = (geometry(0).collect(), geometry(1).collect());
        ensure!(gpu1.is_empty() || gpu0.starts_with(&gpu1), "GPU1's mark layers are not GPU0's prefix");
        Ok(())
    }

    /// One layer's share of a mark: its window rows and compressor state.
    fn layer_mark_bytes(layer: &LayerBuffers) -> usize {
        WINDOW_MARK_BYTES + layer.states.iter().map(|&(_, stride)| stride).sum::<usize>()
    }

    /// Bytes of one mark for `engine` (window rows and compressor state of every layer, on every GPU).
    pub fn mark_bytes_of(engine: &Engine<'_>) -> Result<usize> {
        Ok(layer_buffers(engine)?.iter().map(Self::layer_mark_bytes).sum())
    }

    /// Snapshot geometry before allocating any retained marks or host tables.
    pub fn layout_of(engine: &Engine<'_>) -> Result<FamilyLayout> {
        let layers = layer_buffers(engine)?;
        Self::mirror_prefix(&layers)?;
        let rank_marks: Vec<_> = (0..engine.ranks()).map(|rank| layers.iter().filter(|l| l.rank == rank)
            .map(Self::layer_mark_bytes).sum()).collect();
        let page_bytes = layers.iter().filter(|l| l.rank == 0).map(Self::layer_page_bytes).sum();
        Ok(snapshot_layout(engine.shape.units, page_bytes, &rank_marks, engine.context_kv()))
    }

    pub fn mark_bytes(&self) -> usize {
        self.mark_bytes
    }

    pub fn page_bytes(&self) -> usize {
        // One rank's layers: under heads every GPU holds every page, under context a unit's C4
        // pages live on one GPU and its C128 pages are stored from GPU0.
        self.layers.iter().filter(|l| l.rank == 0).map(Self::layer_page_bytes).sum()
    }

    fn layer_page_bytes(layer: &LayerBuffers) -> usize {
        match layer.compressed {
            Some((4, _)) => compressed_page_bytes(4) + INDEX_PAGE_BYTES,
            Some((ratio, _)) => compressed_page_bytes(ratio),
            None => 0,
        }
    }

    pub fn slots(&self) -> usize {
        if self.arenas.is_empty() { 0 } else { self.slots }
    }

    /// A device buffer of the engine (the host tier's copy-engine template).
    pub fn template(&self) -> CuteafdDeviceBuffer {
        self.layers[0].main
    }

    /// Queues `copies` on their ranks' prefix copy streams.
    fn submit(&self, copies: &Copies) -> Result<()> {
        // SAFETY: every view lies inside a live engine allocation of its rank (checked by `view`,
        // or a registered replica range); the cache drains the copy streams before it hands any
        // of that storage to someone else.
        unsafe { self.copies.copy_d2d(copies) }
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
        let mut copies = Copies::new();
        for layer in &self.layers {
            let (rank, arena) = (layer.rank, self.arenas[layer.rank].buffer);
            let offset = &mut offsets[rank];
            let mut copy = |ring: CuteafdDeviceBuffer, mark: CuteafdDeviceBuffer| -> Result<()> {
                copies.push(if capture { (rank, mark, ring) } else { (rank, ring, mark) });
                Ok(())
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
        self.submit(&copies)
    }
}

fn snapshot_layout(pages: usize, page_bytes: usize, rank_mark_bytes: &[usize], parity: bool) -> FamilyLayout {
    FamilyLayout {
        page_rows: SOURCE_PAGE_TOKENS,
        pages,
        page_bytes,
        // Host slabs hold the lead mark only; device arenas still reserve every
        // rank's part and use their original per-rank strides.
        mark_bytes: rank_mark_bytes[0],
        draft_bytes: 0,
        rule: ReuseRule::EXACT,
        mark_store: MarkStore::Arena,
        page_owners: if parity { PageOwners::Parity } else { PageOwners::Uniform },
    }
}

impl PrefixFamily for Dsv4Prefix<'_, '_> {
    type Placement = Placement;

    fn layout(&self) -> FamilyLayout {
        snapshot_layout(self.engine.shape.units, self.page_bytes(), &self.rank_mark_bytes, self.parity())
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
        let mut copies = Copies::new();
        for ((rank, from, _), (to_rank, to, _)) in self.unit_ranges(copy.from)?.into_iter().zip(self.unit_ranges(copy.to)?) {
            if rank != to_rank { return Err(format!("tail copy {copy:?} pairs pages of two GPUs").into()); }
            copies.push((rank, to, from));
        }
        Ok(self.submit(&copies)?)
    }

    fn drain(&self) -> Result<(), BoxError> {
        Ok(self.copies.drain()?)
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        // Each byte once: C4 pages from their owner (every rank's under heads is rank 0's copy),
        // C128 pages from GPU0.
        self.unit_ranges(page).unwrap_or_default().into_iter()
            .filter(|&(rank, _, c4)| rank == 0 || (c4 && self.parity()))
            .map(|(_, b, _)| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        // GPU0's mark: a head split's GPU1 holds an identical copy ([`Self::mark_replicas`]).
        self.arenas.iter().zip(&self.rank_mark_bytes).take(1).map(|(arena, &bytes)| DeviceRange {
            addr: arena.buffer.ptr as u64 + (slot.0 as usize * bytes) as u64,
            bytes,
        }).collect()
    }

    fn page_device(&self, page: u32) -> Option<u8> {
        self.parity().then_some((page & 1) as u8)
    }

    fn page_replicas(&self, page: u32) -> Vec<(DeviceRange, DeviceRange)> {
        if !self.parity() { return Vec::new(); }
        let ranges = self.unit_ranges(page).unwrap_or_default();
        let replicated = |rank: usize| ranges.iter().filter(move |&&(r, _, c4)| r == rank && !c4)
            .map(|(_, b, _)| DeviceRange { addr: b.ptr as u64, bytes: b.bytes });
        replicated(0).zip(replicated(1)).collect()
    }

    fn mark_replicas(&self, slot: MarkSlot) -> Vec<(DeviceRange, DeviceRange)> {
        if !self.parity() || self.arenas.len() != 2 { return Vec::new(); }
        // GPU1's mark is GPU0's backbone layers in the same order ([`Self::mirror_prefix`]); GPU0's
        // continues with the dSpark stages' rings, which GPU1 does not hold.
        let bytes = self.rank_mark_bytes[1];
        let at = |rank: usize| self.arenas[rank].buffer.ptr as u64 + (slot.0 as usize * self.rank_mark_bytes[rank]) as u64;
        vec![(DeviceRange { addr: at(0), bytes }, DeviceRange { addr: at(1), bytes })]
    }

    fn copy_replicas(&self, copies: &[(DeviceRange, DeviceRange)]) -> Result<(), BoxError> {
        if copies.is_empty() { return Ok(()); }
        if !self.parity() { return Err("a heads split has no replicated snapshot bytes".into()); }
        let template = self.template();
        let mut batch = Copies::new();
        for &(from, to) in copies {
            let buffer = |r: DeviceRange| CuteafdDeviceBuffer { ptr: r.addr as *mut std::ffi::c_void, bytes: r.bytes, ..template };
            if from.bytes != to.bytes { return Err("replica ranges differ in length".into()); }
            // A peer read queued on GPU1's copy stream, ordered before its next forward pass.
            batch.push((1, buffer(to), buffer(from)));
        }
        Ok(self.submit(&batch)?)
    }
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
        self.engine.decode(&mut [(placement, token)], self.transports.first_mut(), self.runtime)
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

/// One `--resume-at` case.
pub(crate) struct ResumeCase {
    pub at: usize,
    pub n: usize,
    pub chunk: usize,
}

/// dsv4-golden --resume-at: for each case prefill `tokens[..P]` into sequence A, capture its
/// snapshot (shared units, the copied tail unit, the window/compressor mark), restore it into
/// sequence B (its own state slot), continue both with the same chunks and `decode` greedy
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
        // B: a second sequence restored from the snapshot (shared full units, its own tail unit,
        // ring and state slot).
        let mut b = if cold {
            let mut b = allocator.admit(n + decode)?;
            runner.prefill(&mut b, &embed[..at], chunk, false)?;
            family.capture(MarkSlot(1), &b, at).map_err(err)?;
            family.restore(Some(MarkSlot(1)), &mut b, at).map_err(err)?;
            b
        } else {
            let (mut b, copy) = allocator.fork(&a, at, n + decode)?;
            if let Some(copy) = copy {
                family.copy_rows(copy).map_err(err)?;
            }
            family.restore(Some(MarkSlot(0)), &mut b, at).map_err(err)?;
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
            restored == cold prefill of the prefix on every GPU {} | layers {} | \
            logits {} (last row max |diff| {max_diff:.3e}) | decode {} | paged rows 0..{len} {} | window+compressor \
            state {} | mark round trip {} ({} B), capture+restore {restore_ms:.1} ms",
            if runner.lanes { ", lanes" } else { "" }, if state_at { "identical" } else { "DIFFER" },
            if cold_equal { "identical" } else { "DIFFERS" },
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
    fn retained_stage_window_rows_match_cold_prefill_at_page_and_ring_boundaries() {
        use super::super::pool::PoolShape;
        for chunk in [128usize, 256, 1024] {
            let shape = PoolShape::new(2, chunk, 16);
            let source = Placement::new(0, Vec::new());
            let destination = Placement::new(1, Vec::new());
            for end in [63usize, 64, 65, 127, 128, 129, 255, 256, 257, 4097] {
                let first = end.saturating_sub(WINDOW);
                let mut cold = vec![[0u8; 8]; shape.window_pages() * SOURCE_PAGE_TOKENS];
                for position in first..end {
                    cold[source.window_slot(&shape, position) as usize] = (position as u64 + 1).to_le_bytes();
                }
                let mark = (first..end).map(|p| cold[source.window_slot(&shape, p) as usize]).collect::<Vec<_>>();
                let mut restored = vec![[0xa5; 8]; cold.len()];
                for (offset, position) in (first..end).enumerate() {
                    restored[destination.window_slot(&shape, position) as usize] = mark[offset];
                }
                for position in first..end {
                    assert_eq!(restored[destination.window_slot(&shape, position) as usize],
                        cold[source.window_slot(&shape, position) as usize]);
                }
                let next = (end as u64 + 1).to_le_bytes();
                restored[destination.window_slot(&shape, end) as usize] = next;
                cold[source.window_slot(&shape, end) as usize] = next;
                for position in (end + 1).saturating_sub(WINDOW)..=end {
                    assert_eq!(restored[destination.window_slot(&shape, position) as usize],
                        cold[source.window_slot(&shape, position) as usize]);
                }
            }
        }
    }

    /// GPU1's mark is the prefix of GPU0's (the host tier restores GPU1 from it): the same
    /// backbone layers in the same order; GPU0 adds the dSpark stages' window rings after them.
    #[test]
    fn gpu1_marks_are_the_prefix_of_gpu0s() {
        let buffer = CuteafdDeviceBuffer::default();
        let layer = |rank: usize, ratio: usize| LayerBuffers { rank, main: buffer,
            compressed: (ratio > 0).then_some((ratio, buffer)), index: (ratio == 4).then_some(buffer),
            states: match ratio { 4 => vec![(buffer, 65_536), (buffer, 65_536), (buffer, 16_384), (buffer, 16_384)],
                128 => vec![(buffer, 524_288), (buffer, 524_288)], _ => Vec::new() } };
        let backbone = [0, 0, 4, 128, 4];
        let mut layers: Vec<_> = backbone.iter().map(|&r| layer(0, r)).chain([layer(0, 0), layer(0, 0)]).collect();
        layers.extend(backbone.iter().map(|&r| layer(1, r)));
        assert!(Dsv4Prefix::mirror_prefix(&layers).is_ok());
        let bytes = |rank| layers.iter().filter(|l| l.rank == rank).map(Dsv4Prefix::layer_mark_bytes).sum::<usize>();
        assert_eq!(bytes(0) - bytes(1), 2 * WINDOW_MARK_BYTES, "GPU0 adds the two stage rings");
        // A GPU1 layer order that differs from GPU0's is refused.
        layers.swap(7, 9);
        assert!(Dsv4Prefix::mirror_prefix(&layers).is_err());
    }

    #[test]
    fn context_host_quota_matches_planner_and_stores_one_mark_with_peer_replica() {
        use cuteafd_loader::plan::{self, layout::LayoutOptions, PlanOptions};
        use cuteafd_loader::placement::AttentionPlacement;
        use cuteafd_core::prefix::HostCacheGeometry;
        let buffer = CuteafdDeviceBuffer::default();
        for stages in [0, 3] {
            let dir = tempfile::tempdir().unwrap();
            plan::testing::write_v4_snapshot_with_dspark(dir.path(), stages);
            let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(dir.path(), stages).unwrap();
            let layer = |rank: usize, ratio: usize| LayerBuffers { rank, main: buffer,
                compressed: (ratio > 0).then_some((ratio, buffer)), index: (ratio == 4).then_some(buffer),
                states: match ratio {
                    4 => vec![(buffer, 65_536), (buffer, 65_536), (buffer, 16_384), (buffer, 16_384)],
                    128 => vec![(buffer, 524_288), (buffer, 524_288)], _ => Vec::new(),
                } };
            let mut layers: Vec<_> = cfg.compress_ratios.iter().map(|&ratio| layer(0, ratio)).collect();
            layers.extend((0..stages).map(|_| layer(0, 0)));
            layers.extend(cfg.compress_ratios.iter().map(|&ratio| layer(1, ratio)));
            Dsv4Prefix::mirror_prefix(&layers).unwrap();
            let marks: Vec<_> = (0..2).map(|rank| layers.iter().filter(|l| l.rank == rank)
                .map(Dsv4Prefix::layer_mark_bytes).sum()).collect();
            let page_bytes = cfg.compress_ratios.iter().map(|&ratio| match ratio {
                4 => compressed_page_bytes(4) + INDEX_PAGE_BYTES,
                128 => compressed_page_bytes(128), _ => 0,
            }).sum();
            let runtime = snapshot_layout(32, page_bytes, &marks, true);
            assert_eq!(runtime.host_layout().tail, marks[0]);
            assert_eq!(marks[0] - marks[1], stages * WINDOW_MARK_BYTES);
            let host = HostCacheGeometry { unit_rows: runtime.page_rows as u64,
                page_bytes: runtime.page_bytes as u64, mark_bytes: runtime.mark_bytes as u64, draft_bytes: 0 };
            for concurrency in [16, 128] {
                let options = LayoutOptions { rtx_bytes: vec![96 << 30; 2], context_tokens: 4096,
                    pool_tokens: Some(8192), concurrency, native_mtp_layers: stages,
                    host_memory: Some((128 << 30, 128 << 30)), ..Default::default() };
                let report = plan::plan(dir.path(), &PlanOptions { layout: Some(options),
                    attention_placement: Some(AttentionPlacement::Context), ..Default::default() }).unwrap();
                let expected = host.resolve(None, 4096, concurrency, 128 << 30, host.chunk_bytes()).unwrap();
                assert_eq!(report.host_budget_bytes, expected, "stages={stages}, C={concurrency}");
                let note = format!("host prefix: {} B per mark;", marks[0]);
                assert!(report.memory_layout.as_ref().unwrap().notes.iter().any(|n| n.starts_with(&note)),
                    "{}", plan::render(&report));
                let required = crate::shared::prefix::budget::retained_bytes(runtime,
                    concurrency as usize, 4096, host.chunk_bytes()).unwrap();
                assert_eq!(required.min(32 << 30) / host.chunk_bytes() * host.chunk_bytes(), expected);
            }
        }
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
