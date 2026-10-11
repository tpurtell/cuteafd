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
use super::engine::Engine;
use super::metadata::{compressed_page_bytes, INDEX_PAGE_BYTES, MAIN_PAGE_BYTES, SOURCE_PAGE_TOKENS, WINDOW};
use super::pool::{Placement, PoolAllocator};
use crate::shared::memory::DeviceAllocation;
use crate::shared::prefix::view;
use crate::shared::spark_intake::SparkLink;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy};
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
                ensure!(buffer.bytes >= shape.units * compressed_page_bytes(ratio), "C{ratio} pool smaller than the units");
            }
        }
        let rank_mark_bytes: Vec<usize> = (0..engine.ranks()).map(|rank| layers.iter().filter(|l| l.rank == rank)
            .map(Self::layer_mark_bytes).sum()).collect();
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
        self.layers.iter().map(|l| match l.compressed {
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

    /// The device ranges of one unit, per compressed layer: the C4 page and its index page, or
    /// the C128 page.
    fn unit_ranges(&self, unit: u32) -> Result<Vec<(usize, CuteafdDeviceBuffer)>> {
        let unit = unit as usize;
        let mut out = Vec::new();
        for layer in &self.layers {
            let Some((ratio, buffer)) = layer.compressed else { continue };
            let bytes = compressed_page_bytes(ratio);
            out.push((layer.rank, view(buffer, unit * bytes, bytes)?));
            if let Some(index) = layer.index {
                out.push((layer.rank, view(index, unit * INDEX_PAGE_BYTES, INDEX_PAGE_BYTES)?));
            }
        }
        Ok(out)
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
            mark_bytes: self.mark_bytes,
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena,
            page_owners: Default::default(),
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
        // past them.
        for ((rank, from), (_, to)) in self.unit_ranges(copy.from)?.into_iter().zip(self.unit_ranges(copy.to)?) {
            self.copy(rank, to, from)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        for rank in 0..self.engine.ranks() {
            // SAFETY: the engine owns these streams.
            unsafe { self.engine.library.cuda_stream_synchronize(self.engine.stream_of(rank))? };
        }
        Ok(())
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.unit_ranges(page).unwrap_or_default().into_iter()
            .map(|(_, b)| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.arenas.iter().zip(&self.rank_mark_bytes).map(|(arena, &bytes)| DeviceRange {
            addr: arena.buffer.ptr as u64 + (slot.0 as usize * bytes) as u64,
            bytes,
        }).collect()
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

/// Every paged byte of `placement`'s first `len` rows: per unit and compressed layer, the
/// records (payload, scales) of the groups completed below `len` and, for C4, their index keys
/// and scales.
fn paged_rows(family: &Dsv4Prefix<'_, '_>, placement: &Placement, len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for (u, &unit) in placement.units.iter().enumerate().take(len.div_ceil(SOURCE_PAGE_TOKENS)) {
        let mut ranges = family.unit_ranges(unit)?.into_iter();
        for layer in &family.layers {
            let Some((ratio, _)) = layer.compressed else { continue };
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

/// The bytes mark `slot` holds for a snapshot at `len`: per layer its `min(len, 128)` window rows
/// (rows past them are left over from earlier marks) and its compressor state.
fn mark(family: &Dsv4Prefix<'_, '_>, slot: u32, len: usize) -> Result<Vec<u8>> {
    let range = family.mark_segments(MarkSlot(slot))[0];
    let bytes = download(family.engine, CuteafdDeviceBuffer { ptr: range.addr as *mut std::ffi::c_void,
        bytes: range.bytes, ..family.template() })?;
    let rows = len.min(WINDOW);
    let (mut out, mut offset) = (Vec::with_capacity(bytes.len()), 0);
    // Rank 0's arena (a head split's rank 1 keeps an identical copy of its layers' rows).
    for layer in family.layers.iter().filter(|l| l.rank == 0) {
        out.extend_from_slice(&bytes[offset..][..rows * PAYLOAD_BYTES]);
        out.extend_from_slice(&bytes[offset + WINDOW * PAYLOAD_BYTES..][..rows * SCALE_BYTES]);
        offset += WINDOW_MARK_BYTES;
        let states: usize = layer.states.iter().map(|&(_, stride)| stride).sum();
        out.extend_from_slice(&bytes[offset..][..states]);
        offset += states;
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
    let mut allocator = PoolAllocator::new(engine.shape);
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
        println!("resume at {at} of {n} (chunks of {chunk}{}, {decode} decode steps): paged rows at {at} {} | layers {} | \
            logits {} (last row max |diff| {max_diff:.3e}) | decode {} | paged rows 0..{len} {} | window+compressor \
            state {} | mark round trip {} ({} B), capture+restore {restore_ms:.1} ms",
            if runner.lanes { ", lanes" } else { "" }, if state_at { "identical" } else { "DIFFER" },
            if runner.lanes { "not digested".to_string() }
            else { first_layer.map_or("identical".to_string(), |l| format!("differ from layer {l}")) },
            if logits_equal { "identical" } else { "DIFFER" }, if decode_equal { "identical" } else { "DIFFERS" },
            if paged_equal { "identical" } else { "DIFFER" }, if state_equal { "identical" } else { "DIFFERS" },
            if mark_equal { "identical" } else { "DIFFERS" }, family.mark_bytes());
        allocator.release(b);
        allocator.release(a);
        identical += usize::from(first_layer.is_none() && logits_equal && decode_equal && paged_equal && state_equal
            && mark_equal);
        marks_identical += usize::from(mark_equal && state_at);
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
