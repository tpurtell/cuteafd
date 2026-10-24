//! Qwen 3.8 Flash Next as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! Paged state, one 256-row allocation unit per page index: on each of the 12 full-attention
//! layers the unit's four 64-row record pages of BF16 K/V (2,048 B per row) and raw index keys
//! (256 B per row), and the pool page of the same index (64 pooled block keys of 4 tokens,
//! 256 B each) = 7.27 MB per unit (28,416 B per token); with `--mtp` the MTP layer's pages ride
//! along (2,368 B per token more). Refcounted units strictly below the earliest writable row
//! are shared. The remaining retained units are copied, including the last full unit when
//! MTP still needs to consume hidden(P-1) at a page-aligned target frontier.
//!
//! Mark: the GDN layers' recurrent state is per sequence and overwritten by every step, so a
//! snapshot copies the sequence's whole state slot: every GDN layer's FP32 state `[48, 128,
//! 128]` and conv window (the last three q/k/v inputs), 36 layers, and the PLE conv state =
//! 110.3 MiB. The arena is sized by decoding lanes, with the host tier holding the rest. A
//! restore copies the mark back into the new sequence's own slot and maps its pool pages. The
//! PLE n-gram context is a pure function of the token ids: the caller recomputes it
//! (`engine::history_of`). The capture point must be where the state is: `state_len` (a
//! speculative verify leaves the state behind the placement until its kept rows are committed),
//! so `capture_reach` is 0.
//!
//! Restores are exact frontiers only (`ReuseRule::EXACT`): recurrent state exists at the points
//! it was captured and nowhere else. Drafting marks retain canonical MTP KV through P-2
//! and hidden(P-1) in the mark. Shared lookup rejects predictor-incomplete marks only for
//! drafting requests. The next prompt token or sampled token closes the restored pending pair.
//!
//! Each logical mark preserves global-layer ordering across owner-local arenas. Copies enqueue
//! on the owning execution stream; drain retires work on every owner.
use super::engine::{history_of, Allocator, Qwen4Engine, Qwen4Placement, BLOCK, INDEX_DIM, PAGE_ROWS,
    UNIT_ROWS};
use crate::shared::memory::device::{Allocation, Device};
use std::rc::Rc;
use crate::shared::prefix::view;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy, Mark, RestoreContext, RestorePlan};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::DeviceRange;

/// Raw index-key bytes per row (BF16, 128 wide).
const KEY_BYTES: usize = INDEX_DIM * 2;
/// Pooled block key bytes (BF16, 128 wide), and a pool page of 64 of them.
const BLOCK_KEY_BYTES: usize = INDEX_DIM * 2;
const POOL_PAGE_BYTES: usize = PAGE_ROWS * BLOCK_KEY_BYTES;

use crate::shared::qwen::MarkLayout;

pub(crate) struct Qwen4Prefix<'e, 'a> {
    engine: &'e Qwen4Engine<'a>,
    /// Per paged layer: records, raw index keys, pooled block keys.
    paged: Vec<[CuteafdDeviceBuffer; 3]>,
    mark_bytes: usize,
    mark_layout: MarkLayout,
    arenas: Vec<Rc<Allocation<'a>>>,
    slots: usize,
    /// Per execution owner: the prefix copy stream, fenced against that owner's stream; and
    /// the owner index of each device.
    copies: crate::shared::prefix::PrefixCopies<'a>,
    owners: Vec<i32>,
}

/// Device copies `(owner, destination, source)` queued as one batch.
type Copies = Vec<(usize, CuteafdDeviceBuffer, CuteafdDeviceBuffer)>;

impl<'e, 'a> Qwen4Prefix<'e, 'a> {
    /// Resolve the real owner geometry without allocating the mark arenas.
    pub(super) fn geometry(engine: &Qwen4Engine<'_>) -> Result<FamilyLayout> {
        let paged = engine.paged_buffers();
        ensure!(!paged.is_empty(), "Qwen 3.8 Flash Next without a full-attention layer");
        let regions = engine.snapshot_regions(0);
        let mark = MarkLayout::new(&regions.iter().map(|r| (r.device_id, r.bytes)).collect::<Vec<_>>())?;
        ensure!(mark.bytes > 0, "Qwen 3.8 Flash Next without GDN state");
        Ok(FamilyLayout { page_rows: UNIT_ROWS, pages: engine.pool_pages,
            page_bytes: paged.len() * (UNIT_ROWS * (engine.kv_record_bytes + KEY_BYTES) + POOL_PAGE_BYTES),
            mark_bytes: mark.bytes, draft_bytes: 0, rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena, page_owners: Default::default() })
    }

    /// The family over `engine`'s buffers with a device arena of `slots(mark_bytes)` marks.
    pub fn new(engine: &'e Qwen4Engine<'a>, slots: impl FnOnce(usize) -> usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix");
        let paged = engine.paged_buffers();
        ensure!(!paged.is_empty(), "Qwen 3.8 Flash Next without a full-attention layer");
        let state_regions = engine.snapshot_regions(0);
        for buffer in paged.iter().flatten().chain(&state_regions) {
            engine.state_stream(buffer.device_id)?;
        }
        for [records, keys, pools] in &paged {
            ensure!(records.bytes >= engine.pages * PAGE_ROWS * engine.kv_record_bytes
                && keys.bytes >= engine.pages * PAGE_ROWS * KEY_BYTES
                && pools.bytes >= engine.pool_pages * POOL_PAGE_BYTES, "attention cache buffers smaller than the units");
        }
        let mark_layout = MarkLayout::new(&state_regions.iter().map(|r| (r.device_id, r.bytes)).collect::<Vec<_>>())?;
        let mark_bytes = mark_layout.bytes;
        ensure!(mark_bytes > 0, "Qwen 3.8 Flash Next without GDN state");
        let slots = slots(mark_bytes);
        let arenas = if slots == 0 { Vec::new() } else {
            mark_layout.owners.iter().map(|&(id, bytes)| {
                let bytes = slots.checked_mul(bytes).context("Qwen mark arena overflow")?;
                Allocation::new(Device { library: engine.library, id }, bytes).map(Rc::new)
            }).collect::<Result<Vec<_>>>()?
        };
        let streams = engine.owner_streams();
        let copies = crate::shared::prefix::PrefixCopies::new(engine.library, streams.iter()
            .map(|&(device, stream)| vec![crate::shared::prefix::Reader { device, stream }]).collect())?;
        let owners = streams.into_iter().map(|(device, _)| device).collect();
        Ok(Self { engine, paged, mark_bytes, mark_layout, arenas, slots, copies, owners })
    }

    pub fn mark_bytes(&self) -> usize {
        self.mark_bytes
    }

    pub fn page_bytes(&self) -> usize {
        self.paged.len() * (UNIT_ROWS * (self.engine.kv_record_bytes + KEY_BYTES) + POOL_PAGE_BYTES)
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Adds a copy to `copies` on its device's owner.
    fn copy(&self, copies: &mut Copies, dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(dst.bytes == src.bytes && dst.device_id == src.device_id, "Qwen snapshot region mismatch");
        let owner = self.owners.iter().position(|&device| device == dst.device_id)
            .context("Qwen snapshot device without an execution owner")?;
        copies.push((owner, dst, src));
        Ok(())
    }

    /// Queues `copies` on their owners' prefix copy streams.
    fn submit(&self, copies: &Copies) -> Result<()> {
        // SAFETY: views belong to retained allocations of their owner's device (checked by
        // `view`); the cache drains the copy streams before it hands that storage to anyone.
        unsafe { self.copies.copy_d2d(copies) }
    }

    /// The device ranges of one unit, per paged layer: records, raw index keys, pooled keys.
    fn unit_ranges(&self, unit: u32) -> Result<Vec<CuteafdDeviceBuffer>> {
        let unit = unit as usize;
        let mut out = Vec::with_capacity(3 * self.paged.len());
        for &[records, keys, pools] in &self.paged {
            out.push(view(records, unit * UNIT_ROWS * self.engine.kv_record_bytes, UNIT_ROWS * self.engine.kv_record_bytes)?);
            out.push(view(keys, unit * UNIT_ROWS * KEY_BYTES, UNIT_ROWS * KEY_BYTES)?);
            out.push(view(pools, unit * POOL_PAGE_BYTES, POOL_PAGE_BYTES)?);
        }
        Ok(out)
    }

    pub fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        let mut owners = self.engine.snapshot_owners();
        owners.extend(self.arenas.iter().cloned());
        owners
    }

    fn mark_buffers(&self, slot: MarkSlot) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        self.mark_layout.regions.iter().map(|&(owner, offset, bytes)| {
            let stride = self.mark_layout.owners[owner].1;
            let arena = self.arenas.get(owner).context("no mark arena")?;
            view(arena.buffer, slot.0 as usize * stride + offset, bytes)
        }).collect()
    }

    /// Copy global state-region order without inter-device copies or approximation.
    fn move_mark(&self, slot: MarkSlot, state: i32, capture: bool) -> Result<()> {
        ensure!(state >= 0 && (state as usize) < self.engine.slots, "state slot {state} of {}", self.engine.slots);
        let regions = self.engine.snapshot_regions(state as usize);
        let marks = self.mark_buffers(slot)?;
        ensure!(regions.len() == marks.len(), "Qwen snapshot region count changed");
        let mut copies = Copies::new();
        for (region, mark) in regions.into_iter().zip(marks) {
            if capture { self.copy(&mut copies, mark, region)?; } else { self.copy(&mut copies, region, mark)?; }
        }
        self.submit(&copies)
    }

}

#[derive(Debug, Clone, Default)]
pub(crate) struct Frontier {
    pub target: usize,
    pub predictors: Vec<usize>,
    pub pending: bool,
}

impl Frontier {
    pub fn supports_predictors(&self, frontier: usize) -> bool {
        // Row P-1 consumes token(P), which is outside the retained prefix.
        // Only the unconsumed hidden row is valid for arbitrary continuations.
        self.target == frontier && frontier > 0 && self.pending
            && self.predictors.as_slice() == [frontier - 1]
    }
}

impl PrefixFamily<Frontier> for Qwen4Prefix<'_, '_> {
    type Placement = Qwen4Placement;

    fn write_frontier(&self, len: usize) -> usize {
        if self.engine.weights.mtp.is_some() { len.saturating_sub(1) } else { len }
    }

    fn record_format(&self) -> &'static str {
        match self.engine.kv_format {
            cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16 => "qwen4_bf16_v1",
            cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 => "qwen4_e4m3_token_head_f32_v1",
        }
    }

    fn layout(&self) -> FamilyLayout {
        FamilyLayout {
            page_rows: UNIT_ROWS,
            pages: self.engine.pool_pages,
            page_bytes: self.page_bytes(),
            mark_bytes: self.mark_bytes,
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena,
            page_owners: Default::default(),
        }
    }

    fn pages<'p>(&self, placement: &'p Qwen4Placement) -> &'p [u32] {
        &placement.units
    }

    fn commit_point(&self, placement: &Qwen4Placement) -> usize {
        placement.state_len.min(placement.len)
    }

    fn capture_meta(&self, placement: &Qwen4Placement, len: usize) -> Result<Frontier, BoxError> {
        if len != placement.len || len != placement.state_len {
            return Err("Qwen snapshot target frontier mismatch".into());
        }
        Ok(Frontier { target: len, predictors: self.engine.weights.mtp.as_ref()
            .map(|_| vec![placement.mtp_frontier.get()]).unwrap_or_default(),
            pending: placement.mtp_snapshot_pending.get() && placement.mtp_frontier.get() == len.saturating_sub(1) })
    }

    fn restore_with(&self, mark: Option<&Mark>, saved: Option<&Frontier>, placement: &mut Qwen4Placement,
        plan: &RestorePlan, _context: &RestoreContext<'_>) -> Result<(), BoxError> {
        let saved = saved.ok_or("Qwen snapshot predictor metadata absent")?;
        if saved.target != plan.resume() || saved.predictors.iter().any(|&p| p > saved.target) {
            return Err("Qwen snapshot frontier metadata mismatch".into());
        }
        let Some(Mark::Slot(slot)) = mark else { return Err("Qwen snapshot mark absent".into()); };
        self.restore(Some(*slot), placement, plan.resume())?;
        placement.mtp_frontier.set(saved.predictors.first().copied().unwrap_or(0));
        placement.mtp_snapshot_pending.set(saved.pending);
        Ok(())
    }

    fn capture(&self, slot: MarkSlot, placement: &Qwen4Placement, len: usize) -> Result<(), BoxError> {
        if len != placement.len || len != placement.state_len {
            return Err(format!("capture at {len}: the placement holds {} rows, its state {}", placement.len,
                placement.state_len).into());
        }
        Ok(self.move_mark(slot, placement.slot, true)?)
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Qwen4Placement, len: usize) -> Result<(), BoxError> {
        let Some(slot) = mark else {
            return Err("Qwen 3.8 Flash Next restores exact snapshots with their GDN state".into());
        };
        if len == 0 || len.div_ceil(UNIT_ROWS) > placement.units.len() {
            return Err(format!("restore of {len} rows into {} units", placement.units.len()).into());
        }
        self.engine.map_pools(placement)?;
        self.move_mark(slot, placement.slot, false)?;
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let blocks = copy.rows / BLOCK;
        let mut copies = Copies::new();
        for &[records, keys, pools] in &self.paged {
            for (buffer, row) in [(records, self.engine.kv_record_bytes), (keys, KEY_BYTES)] {
                self.copy(&mut copies, view(buffer, copy.to as usize * UNIT_ROWS * row, copy.rows * row)?,
                    view(buffer, copy.from as usize * UNIT_ROWS * row, copy.rows * row)?)?;
            }
            // The complete blocks' pooled keys.
            if blocks > 0 {
                let (from, to) = (copy.from as usize * POOL_PAGE_BYTES, copy.to as usize * POOL_PAGE_BYTES);
                self.copy(&mut copies, view(pools, to, blocks * BLOCK_KEY_BYTES)?,
                    view(pools, from, blocks * BLOCK_KEY_BYTES)?)?;
            }
        }
        Ok(self.submit(&copies)?)
    }

    fn drain(&self) -> Result<(), BoxError> {
        Ok(self.copies.drain()?)
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.unit_ranges(page).unwrap_or_default().into_iter()
            .map(|b| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.mark_buffers(slot).unwrap_or_default().into_iter()
            .map(|buffer| DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes }).collect()
    }
}

/// One continued prefill: every layer's output digest, every row's logits digest and argmax,
/// and the last row.
pub(crate) struct SuffixRun {
    pub layers: Vec<u64>,
    pub logits: u64,
    pub argmax: Vec<usize>,
    pub last: Vec<f32>,
}

/// Prefill `tokens` into `placement` in chunks of `chunk` rows, digesting each layer's streams
/// and (with `logits`) every row's logits.
pub(crate) fn prefill_digest(engine: &Qwen4Engine<'_>, placement: &mut Qwen4Placement, tokens: &[u32], chunk: usize,
    logits: bool) -> Result<SuffixRun> {
    use std::hash::{Hash, Hasher};
    let vocab = engine.cfg.vocab_size;
    let mut hashers: Vec<std::collections::hash_map::DefaultHasher> = Vec::new();
    let mut logit_hash = std::collections::hash_map::DefaultHasher::new();
    let (mut argmax, mut last) = (Vec::new(), Vec::new());
    for part in tokens.chunks(chunk) {
        let mut on_layer = |layer: usize, rows: &[u8]| -> Result<()> {
            if hashers.len() <= layer {
                hashers.resize_with(layer + 1, Default::default);
            }
            rows.hash(&mut hashers[layer]);
            Ok(())
        };
        let out = engine.prefill_forced(placement, part, Some(&mut on_layer), None, if logits { part.len() } else { 1 })?
            .context("the resume check needs every layer")?;
        if logits {
            for values in out.chunks_exact(vocab) {
                values.iter().for_each(|v| v.to_bits().hash(&mut logit_hash));
                argmax.push(values.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i));
            }
            last = out[out.len() - vocab..].to_vec();
        }
    }
    Ok(SuffixRun { layers: hashers.iter().map(Hasher::finish).collect(), logits: logit_hash.finish(), argmax, last })
}

fn download(engine: &Qwen4Engine<'_>, range: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
    let stream = engine.state_stream(range.device_id)?;
    Device { library: engine.library, id: range.device_id }.run(|| {
        // SAFETY: the owner stream retires every write to this live range.
        unsafe { engine.library.cuda_stream_synchronize(stream)? };
        let mut bytes = vec![0u8; range.bytes];
        engine.library.copy_d2h(&mut bytes, range)?;
        Ok(bytes)
    })
}

/// Every byte of state slot `slot` (GDN conv and recurrent state per layer, PLE conv state).
fn slot_state(engine: &Qwen4Engine<'_>, slot: i32) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for region in engine.slot_regions(usize::try_from(slot)?) {
        out.extend(download(engine, region)?);
    }
    Ok(out)
}

/// Every paged byte of `placement`'s first `len` rows (records and raw keys of each row, pooled
/// keys of each complete block), layer by layer, in position order. Without the MTP layer's
/// pages: a restored sequence's MTP history is the writer's (drafts only).
fn paged_rows(family: &Qwen4Prefix<'_, '_>, placement: &Qwen4Placement, len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let blocks = len / BLOCK;
    let layers = family.paged.len() - usize::from(family.engine.weights.mtp.is_some());
    for (u, &unit) in placement.units.iter().enumerate().take(len.div_ceil(UNIT_ROWS)) {
        let rows = (len - u * UNIT_ROWS).min(UNIT_ROWS);
        let unit_blocks = (blocks.saturating_sub(u * PAGE_ROWS)).min(PAGE_ROWS);
        for ranges in family.unit_ranges(unit)?.chunks_exact(3).take(layers) {
            out.extend_from_slice(&download(family.engine, ranges[0])?[..rows * family.engine.kv_record_bytes]);
            out.extend_from_slice(&download(family.engine, ranges[1])?[..rows * KEY_BYTES]);
            out.extend_from_slice(&download(family.engine, ranges[2])?[..unit_blocks * BLOCK_KEY_BYTES]);
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

/// qwen4-golden --resume-at P: prefill `tokens[..P]` into sequence A, capture its snapshot
/// (shared units, the copied tail unit, the state mark), restore it into sequence B (its own
/// state slot, pool-page mapping and recomputed n-gram history), continue both with the same
/// chunks and `decode` greedy single-row steps, and compare every layer's rows, every logit,
/// every paged row, the state slot and the mark round trip byte for byte (a restore must be
/// exact). A straight prefill without the boundary at P is reported too (informational:
/// chunking changes may round differently). With `cold`, B is prefilled from scratch on its own
/// units instead (no restore): the floor of what the kernels themselves vary. Each case runs
/// `repeat` times on fresh sequences. Returns whether every case passed.
pub(crate) fn resume_check(engine: &Qwen4Engine<'_>, tokens: &[u32], cases: &[ResumeCase], decode: usize, cold: bool,
    repeat: usize) -> Result<bool> {
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--resume-at needs every layer");
    let family = Qwen4Prefix::new(engine, |_| 2)?;
    let mut allocator = Allocator::new(engine.pages, engine.slots, &engine.cfg);
    let mut passed = true;
    for case in cases {
        if engine.weights.mtp.is_some() && case.at >= 3 {
            drafting_resume_check(engine, &family, &mut allocator, tokens, case.at)?;
        }
        match resume_case(engine, &family, &mut allocator, tokens, case, decode, cold, repeat) {
            Ok(ok) => passed &= ok,
            Err(error) => {
                println!("resume at {} of {} (chunks of {}): FAILED: {error:#}", case.at, case.n, case.chunk);
                passed = false;
            }
        }
    }
    Ok(passed)
}

/// Native byte oracle: a two-row accepted turn must execute its first pending
/// pair and retain only its final hidden row, just like a cold canonical prefix.
fn drafting_resume_check(engine: &Qwen4Engine<'_>, family: &Qwen4Prefix<'_, '_>, allocator: &mut Allocator,
    tokens: &[u32], at: usize) -> Result<()> {
    use super::speculate::{self, MtpSeq};
    ensure!(at >= 3 && at < tokens.len(), "MTP resume needs at least three target rows");
    let err = |error: BoxError| anyhow::anyhow!("{error}");
    let mut writer = allocator.admit(at + 1)?;
    let mut cold = allocator.admit(at + 1)?;
    let mut writer_seq = MtpSeq::default();
    let mut cold_seq = MtpSeq::default();
    let writer_slot = writer.slot;
    for (placement, seq) in [(&mut writer, &mut writer_seq), (&mut cold, &mut cold_seq)] {
        for start in (0..at - 2).step_by(64) {
            let end = (start + 64).min(at - 2);
            engine.prefill(placement, &tokens[start..end])?;
            speculate::prefill_chunk(engine, placement, start, &tokens[start..end], Some(tokens[end]), seq)?;
        }
        engine.verify(&mut [(placement, &tokens[at - 2..at])], None)?;
        if placement.slot == writer_slot {
            // Production accept stashes two rows, then executes all but the
            // last before the turn snapshot is published.
            speculate::accept(engine, &mut [speculate::Verified { placement, seq,
                start: at - 2, history: history_of(&engine.cfg, &tokens[..at - 2]),
                rows: &tokens[at - 2..at], first_row: 0, kept: Some((2, tokens[at])),
                finished: true, neural: true }], false)?;
        } else {
            // Cold oracle writes canonical row P-2 directly from the target,
            // and preserves hidden(P-1), without normalization or accept.
            engine.mtp_stash(true, &[(placement.slot, 1, 1, 0)])?;
            let rows = vec![super::engine::MtpRow { position: at - 2, token: tokens[at - 1], source: 0 }];
            engine.mtp_step(true, &[super::engine::MtpGroup { placement, rows }], super::engine::MtpSource::Target,
                &[], super::engine::MtpTokens::Host(&tokens[at - 1..at]),
                super::engine::MtpOut::Download { logits: false })?;
            placement.mtp_frontier.set(at - 1);
            placement.mtp_snapshot_pending.set(true);
        }
    }
    let saved = family.capture_meta(&writer, at).map_err(err)?;
    ensure!(saved.supports_predictors(at), "drafting writer did not publish complete MTP state");
    family.capture(MarkSlot(0), &writer, at).map_err(err)?;
    // Compare only canonical live rows, not padding/proposal bytes beyond P-1.
    let canonical = |placement: &Qwen4Placement| -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for region in engine.snapshot_regions(placement.slot as usize) { out.extend(download(engine, region)?); }
        for (index, &unit) in placement.units.iter().enumerate().take(at.div_ceil(UNIT_ROWS)) {
            let rows = (at - 1).saturating_sub(index * UNIT_ROWS).min(UNIT_ROWS);
            let blocks = ((at - 1) / BLOCK).saturating_sub(index * PAGE_ROWS).min(PAGE_ROWS);
            let ranges = family.unit_ranges(unit)?;
            let mtp = &ranges[ranges.len() - 3..];
            for (range, len) in [(mtp[0], rows * engine.kv_record_bytes), (mtp[1], rows * KEY_BYTES),
                (mtp[2], blocks * BLOCK_KEY_BYTES)] {
                out.extend_from_slice(&download(engine, range)?[..len]);
            }
        }
        Ok(out)
    };
    let expected = canonical(&cold)?;
    ensure!(canonical(&writer)? == expected, "two-row turn normalization differs from cold MTP state");
    allocator.release(cold);
    let (mut reader, copy) = allocator.fork(&writer, &tokens[..at], at + 1)?;
    if let Some(copy) = copy { family.copy_rows(copy).map_err(err)?; }
    let plan = RestorePlan { snapshot_end: at, target_end: at, lag: cuteafd_engine::prefix::LaggedState {
        source_end: at, replay_start: at }, fidelity: cuteafd_engine::prefix::RestoreFidelity::Exact };
    family.restore_with(Some(&Mark::Slot(MarkSlot(0))), Some(&saved), &mut reader, &plan,
        &RestoreContext { native_tokens: tokens, media: &[] }).map_err(err)?;
    ensure!(canonical(&reader)? == expected, "drafting reader differs byte-exactly from cold MTP state");
    let mut resumed = MtpSeq::default();
    speculate::restore_pending(engine, &reader, at, None, &mut resumed)?;
    ensure!(resumed.pending == vec![(at - 1, 0)] && resumed.open, "restored MTP stash bookkeeping differs");
    allocator.release(reader);
    allocator.release(writer);
    // Exercise the production shared lookup, not just the metadata helper:
    // a no-spec writer has target bytes but no canonical predictor history.
    let mut cache = cuteafd_engine::prefix::PrefixCache::<crate::shared::prefix::CudaCopyEngine<'_>, Frontier>::new(
        family.layout(), cuteafd_engine::prefix::PrefixConfig { entries: 2, mark_slots: 2,
            keep_logits: false, min_tokens: 1 }, None)?;
    let place = |units| Qwen4Placement::new(units, 0, history_of(&engine.cfg, &[]));
    let mut target_only = cache.admit_cold(family, at, at + 1, place)?.placement;
    for part in tokens[..at].chunks(64) { engine.prefill(&mut target_only, part)?; }
    ensure!(cache.capture(family, cuteafd_engine::prefix::SnapshotKind::Turn, &tokens[..at],
        &target_only, cuteafd_engine::prefix::After::default())?, "Qwen target-only mark not retained");
    cache.release(family, &target_only.units)?;
    let compatible = |saved: &Frontier, frontier| saved.supports_predictors(frontier);
    let mut drafting = cache.admit_native_compatible(family, &tokens[..at + 1], &tokens[..at + 1], &[],
        at + 1, false, Some(&compatible), place)?;
    ensure!(drafting.resume == 0, "Qwen drafting reader restored target-only MTP pages");
    let mut draft_state = MtpSeq::default();
    for start in (0..at - 2).step_by(64) {
        let end = (start + 64).min(at - 2);
        engine.prefill(&mut drafting.placement, &tokens[start..end])?;
        speculate::prefill_chunk(engine, &drafting.placement, start, &tokens[start..end], Some(tokens[end]), &mut draft_state)?;
    }
    engine.verify(&mut [(&mut drafting.placement, &tokens[at - 2..at])], None)?;
    speculate::accept(engine, &mut [speculate::Verified { placement: &mut drafting.placement, seq: &mut draft_state,
        start: at - 2, history: history_of(&engine.cfg, &tokens[..at - 2]), rows: &tokens[at - 2..at],
        first_row: 0, kept: Some((2, tokens[at])), finished: true, neural: true }], false)?;
    ensure!(canonical(&drafting.placement)? == expected, "Qwen no-spec writer/drafting miss differs from cold MTP state");
    ensure!(cache.capture(family, cuteafd_engine::prefix::SnapshotKind::Turn, &tokens[..at],
        &drafting.placement, cuteafd_engine::prefix::After::default())?, "Qwen drafting turn not retained");
    cache.release(family, &drafting.placement.units)?;
    let restored = cache.admit_native_compatible(family, &tokens[..at + 1], &tokens[..at + 1], &[],
        at + 1, false, Some(&compatible), place)?;
    ensure!(restored.resume == at, "Qwen drafting reader missed its predictor-complete turn");
    ensure!(canonical(&restored.placement)? == expected, "Qwen shared drafting cache hit differs from cold MTP state");
    let pending_row = |placement: &Qwen4Placement| -> Result<Vec<u8>> {
        let ranges = family.unit_ranges(placement.units[(at - 1) / UNIT_ROWS])?;
        let records = ranges[ranges.len() - 3];
        let offset = (at - 1) % UNIT_ROWS * engine.kv_record_bytes;
        download(engine, view(records, offset, engine.kv_record_bytes)?)
    };
    let retained_pending = pending_row(&restored.placement)?;
    let mut hit_seq = MtpSeq::default();
    speculate::restore_pending(engine, &restored.placement, at, Some(tokens[at]), &mut hit_seq)?;
    ensure!(restored.placement.mtp_frontier.get() == at && hit_seq.pending.is_empty(),
        "Qwen continuation did not consume its restored final hidden row");
    let peer_place = |units| Qwen4Placement::new(units, 1, history_of(&engine.cfg, &[]));
    let mut different = tokens[..at + 1].to_vec();
    different[at] = (different[at] + 1) % engine.cfg.vocab_size as u32;
    let peer = cache.admit_native_compatible(family, &different, &different, &[],
        at + 1, false, Some(&compatible), peer_place)?;
    ensure!(peer.resume == at && pending_row(&peer.placement)? == retained_pending,
        "Qwen first continuation mutated retained predictor row");
    let first_pending = pending_row(&restored.placement)?;
    let mut peer_seq = MtpSeq::default();
    speculate::restore_pending(engine, &peer.placement, at, Some(different[at]), &mut peer_seq)?;
    let check_place = |units| Qwen4Placement::new(units, 2, history_of(&engine.cfg, &[]));
    let check = cache.admit_native_compatible(family, &tokens[..at + 1], &tokens[..at + 1], &[],
        at + 1, false, Some(&compatible), check_place)?;
    ensure!(check.resume == at && pending_row(&check.placement)? == retained_pending,
        "Qwen interleaved continuation mutated retained predictor row");
    ensure!(canonical(&check.placement)? == expected && pending_row(&restored.placement)? == first_pending,
        "Qwen continuations do not own independent canonical predictor state");
    for placement in [&restored.placement, &peer.placement, &check.placement] {
        cache.release(family, &placement.units)?;
    }
    cache.clear(family)?;
    println!("Qwen MTP drafting turn restore at={at} stashed_rows=2 canonical_kv=true hidden_stash=true byte_exact=true");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resume_case(engine: &Qwen4Engine<'_>, family: &Qwen4Prefix<'_, '_>, allocator: &mut Allocator, tokens: &[u32],
    case: &ResumeCase, decode: usize, cold: bool, repeat: usize) -> Result<bool> {
    let &ResumeCase { at, n, chunk } = case;
    ensure!(at > 0 && at < n && n <= tokens.len(), "--resume-at {at} must lie inside the {n} prefilled tokens");
    let chunk = chunk.clamp(1, engine.prefill_rows);
    let embed = &tokens[..n];
    let err = |e: BoxError| anyhow::anyhow!("{e}");
    let repeat = repeat.max(1);
    let (mut identical, mut marks_identical, mut last) = (0, 0, None);
    for _ in 0..repeat {
        // A: prefill [0, P), capture, continue in place.
        let mut a = allocator.admit(n + decode)?;
        prefill_digest(engine, &mut a, &embed[..at], chunk, false)?;
        family.drain().map_err(err)?;
        let started = std::time::Instant::now();
        family.capture(MarkSlot(0), &a, at).map_err(err)?;
        // B: a second sequence restored from the snapshot (shared full units, its own tail and slot).
        let mut b = if cold {
            let mut b = allocator.admit(n + decode)?;
            prefill_digest(engine, &mut b, &embed[..at], chunk, false)?;
            family.capture(MarkSlot(1), &b, at).map_err(err)?;
            family.restore(Some(MarkSlot(1)), &mut b, at).map_err(err)?;
            b
        } else {
            let (mut b, copy) = allocator.fork(&a, &embed[..at], n + decode)?;
            if let Some(copy) = copy {
                family.copy_rows(copy).map_err(err)?;
            }
            family.restore(Some(MarkSlot(0)), &mut b, at).map_err(err)?;
            b
        };
        family.drain().map_err(err)?;
        let restore_ms = started.elapsed().as_secs_f64() * 1e3;
        ensure!(b.history == history_of(&engine.cfg, &embed[..at]) && b.history == a.history,
            "the recomputed n-gram history differs from the prefilled one");
        // The restored state reads back exactly as the captured one.
        family.capture(MarkSlot(1), &b, at).map_err(err)?;
        let mark = |slot: u32| -> Result<Vec<u8>> {
            let mut bytes = Vec::new();
            for region in family.mark_buffers(MarkSlot(slot))? {
                bytes.extend(download(engine, region)?);
            }
            Ok(bytes)
        };
        let mark_equal = mark(0)? == mark(1)?;
        // The state at P (every paged row and the state slot): what a restore must reproduce.
        let state_at = paged_rows(family, &a, at)? == paged_rows(family, &b, at)?
            && slot_state(engine, a.slot)? == slot_state(engine, b.slot)?;
        let straight = prefill_digest(engine, &mut a, &embed[at..], chunk, true)?;
        let restored = prefill_digest(engine, &mut b, &embed[at..], chunk, true)?;
        let first_layer = straight.layers.iter().zip(&restored.layers).position(|(x, y)| x != y);
        let logits_equal = straight.logits == restored.logits;
        let max_diff = straight.last.iter().zip(&restored.last).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        // Greedy single-row decode steps from both (decode graphs, the restored pool mapping).
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i as u32);
        let (mut next_a, mut next_b) = (argmax(&straight.last), argmax(&restored.last));
        let mut decode_equal = next_a == next_b;
        for _ in 0..decode {
            let la = engine.verify(&mut [(&mut a, &[next_a][..])], None)?.context("decode logits")?;
            let lb = engine.verify(&mut [(&mut b, &[next_b][..])], None)?.context("decode logits")?;
            decode_equal &= la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits());
            (next_a, next_b) = (argmax(&la), argmax(&lb));
        }
        let len = a.len;
        let paged_equal = paged_rows(family, &a, len)? == paged_rows(family, &b, len)?;
        let state_equal = slot_state(engine, a.slot)? == slot_state(engine, b.slot)?;
        println!("resume at {at} of {n} (chunks of {chunk}, {decode} decode steps): state at {at} {} | layers {} | \
            logits {} (last row max |diff| {max_diff:.3e}) | decode {} | paged rows 0..{len} {} | GDN/PLE state {} | \
            mark round trip {} ({} B), capture+restore {restore_ms:.1} ms", if state_at { "identical" } else { "DIFFERS" },
            first_layer.map_or("identical".to_string(), |l| format!("differ from layer {l}")),
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
    let whole = prefill_digest(engine, &mut c, embed, chunk, true)?;
    allocator.release(c);
    let x = &whole.argmax[at..];
    let agree = x.iter().zip(&restored.argmax).filter(|(p, q)| p == q).count();
    let last_equal = whole.last.iter().zip(&restored.last).all(|(p, q)| p.to_bits() == q.to_bits());
    println!("vs one straight prefill without the boundary at {at}: suffix top-1 agreement {agree}/{}, last row logits {}",
        x.len(), if last_equal { "identical" } else { "differ" });
    println!("resume at {at} of {n} (chunks of {chunk}): {identical}/{repeat} attempts byte-identical, mark round trip \
        and state at {at} identical in {marks_identical}/{repeat}{}", if cold { " [cold floor: B prefilled, not restored]" } else { "" });
    Ok(marks_identical == repeat && identical == repeat)
}

#[cfg(test)]
mod tests {
    use super::super::engine::Qwen4Placement;
    use cuteafd_loader::families::qwen4::NgramHistory;

    #[test]
    fn logical_marks_keep_global_order_with_compact_owner_arenas() {
        let layout = super::MarkLayout::new(&[(1, 8), (0, 16), (1, 32), (0, 4)]).unwrap();
        assert_eq!(layout.owners, [(1, 40), (0, 20)]);
        assert_eq!(layout.regions, [(0, 0, 8), (1, 0, 16), (0, 8, 32), (1, 16, 4)]);
        assert_eq!(layout.bytes, 60);
        let single = super::MarkLayout::new(&[(0, 8), (0, 16)]).unwrap();
        assert_eq!(single.owners, [(0, 24)]);
        assert_eq!(single.regions, [(0, 0, 8), (0, 8, 16)]);
        assert!(super::MarkLayout::new(&[(0, usize::MAX), (1, 1)]).is_err());
        assert!(super::MarkLayout::new(&[(0, usize::MAX), (0, 1)]).is_err());
    }

    #[test]
    fn units_expand_to_record_and_pool_pages() {
        let p = Qwen4Placement::new(vec![2, 0], 1, NgramHistory(vec![]));
        assert_eq!(p.pages, vec![8, 9, 10, 11, 0, 1, 2, 3]);
        assert_eq!(p.pool_pages, vec![2, 0]);
        // Position 300: the second unit (index 0), its record page 0 (page 0), row 44; block 74
        // (row 10 of pool page 0) completes at 299.
        assert_eq!(p.record(300).unwrap(), 44);
        assert_eq!(p.pool_slot(299).unwrap(), 10);
        assert_eq!(p.pool_slot(300).unwrap(), -1);
        assert_eq!(p.pool_slot(255).unwrap(), 2 * 64 + 63);
    }
}

#[cfg(test)]
mod frontier_tests {
    use super::Frontier;

    #[test]
    fn target_only_marks_do_not_restore_predictors_but_canonical_stashes_do() {
        for target in [1usize, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1492] {
            let mut mark = Frontier { target, predictors: vec![0], pending: false };
            assert!(!mark.supports_predictors(target));
            mark.predictors[0] = target - 1;
            assert!(!mark.supports_predictors(target));
            mark.pending = true;
            assert!(mark.supports_predictors(target));
            assert!(!mark.supports_predictors(target + 1));
            mark.predictors[0] = target;
            mark.pending = false;
            assert!(!mark.supports_predictors(target), "known-next-token state is not prefix-independent");
        }
    }
}
