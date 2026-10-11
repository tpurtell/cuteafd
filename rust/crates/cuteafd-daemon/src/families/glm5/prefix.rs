//! GLM 5.3 (glm_moe_dsa) as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! The pages are the whole state: every layer's FP8 latent records (656 B per row) and every
//! full-indexer layer's DSA index keys (64 x 128 E4M3 keys then 64 FP32 scales per 64-row page)
//! share page ids, 64 rows per page (78 x 41,984 + 21 x 8,448 B = 3.45 MB per page, 54 KB per
//! token). There is no recurrent or windowed state, so a snapshot has no mark: a restore
//! shares the full pages, copies the partial tail page and sets the length. Full pages are
//! shared by reference; nobody writes them again (every sequence appends past its own length).
//!
//! Exact frontiers by default (the deepest retained snapshot whose tokens prefix the request).
//! With `--prefix-partial on` a request that shares only part of a snapshot resumes at its last
//! common page boundary (`ReuseRule::paged`): the shared rows are the snapshot's own bytes, so
//! this is exact too. The DFlash2 drafter is not captured: a restored sequence drafts cold
//! with `valid_from` at the restore point. Under a head split both GPUs hold identical copies
//! of the latent and index pages (the replicated MLA projection and indexer write them): tail
//! copies run on each GPU's stream, and the host tier is off.
use super::engine::{GlmEngine, GlmPlacement, INDEX_PAGE_BYTES, PAGE_ROWS, RECORD_BYTES};
use crate::shared::memory::DeviceAllocation;
use crate::shared::prefix::view;
use anyhow::Result;
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::DeviceRange;

/// Index-key page: 64 rows x 128 E4M3, then 64 FP32 scales.
const INDEX_KEY_BYTES: usize = 128;
const INDEX_SCALES: usize = PAGE_ROWS * INDEX_KEY_BYTES;

pub(crate) struct GlmPrefix<'e, 'a> {
    engine: &'e GlmEngine<'a>,
    /// Per rank and layer: latent records and, on full-indexer layers, index keys.
    paged: Vec<(usize, CuteafdDeviceBuffer, Option<CuteafdDeviceBuffer>)>,
    partial: bool,
    /// The host tier's stand-in tail (one byte host restores overwrite and nothing reads).
    scratch: DeviceAllocation<'a>,
}

impl<'e, 'a> GlmPrefix<'e, 'a> {
    pub fn new(engine: &'e GlmEngine<'a>, partial: bool) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix");
        let paged = (0..engine.ranks())
            .flat_map(|rank| engine.paged_buffers_on(rank).into_iter().map(move |(records, index)| (rank, records, index)))
            .collect();
        Ok(Self { engine, paged, partial, scratch: DeviceAllocation::new(engine.library, 256)? })
    }

    pub fn page_bytes(&self) -> usize {
        self.paged.iter().map(|(_, _, index)| PAGE_ROWS * RECORD_BYTES + index.map_or(0, |_| INDEX_PAGE_BYTES)).sum()
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
}

impl PrefixFamily for GlmPrefix<'_, '_> {
    type Placement = GlmPlacement;

    fn layout(&self) -> FamilyLayout {
        FamilyLayout {
            page_rows: PAGE_ROWS,
            pages: self.engine.pages,
            page_bytes: self.page_bytes(),
            mark_bytes: 0,
            draft_bytes: 0,
            rule: if self.partial { ReuseRule::paged(PAGE_ROWS) } else { ReuseRule::EXACT },
            mark_store: MarkStore::Arena,
            page_owners: Default::default(),
        }
    }

    fn pages<'p>(&self, placement: &'p GlmPlacement) -> &'p [u32] {
        &placement.pages
    }

    fn commit_point(&self, placement: &GlmPlacement) -> usize {
        placement.len
    }

    fn capture(&self, _slot: MarkSlot, _placement: &GlmPlacement, _len: usize) -> Result<(), BoxError> {
        Err("GLM 5.3 snapshots have no mark".into())
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut GlmPlacement, len: usize) -> Result<(), BoxError> {
        if mark.is_some() || len == 0 || len.div_ceil(PAGE_ROWS) > placement.pages.len() {
            return Err(format!("restore of {len} rows into {} pages", placement.pages.len()).into());
        }
        placement.len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let (from, to, rows) = (copy.from as usize, copy.to as usize, copy.rows);
        let record_page = PAGE_ROWS * RECORD_BYTES;
        for &(rank, records, index) in &self.paged {
            self.copy(rank, view(records, to * record_page, rows * RECORD_BYTES)?,
                view(records, from * record_page, rows * RECORD_BYTES)?)?;
            if let Some(index) = index {
                let (from, to) = (from * INDEX_PAGE_BYTES, to * INDEX_PAGE_BYTES);
                self.copy(rank, view(index, to, rows * INDEX_KEY_BYTES)?, view(index, from, rows * INDEX_KEY_BYTES)?)?;
                self.copy(rank, view(index, to + INDEX_SCALES, rows * 4)?, view(index, from + INDEX_SCALES, rows * 4)?)?;
            }
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
        let page = page as usize;
        let record_page = PAGE_ROWS * RECORD_BYTES;
        let mut out = Vec::new();
        for &(_, records, index) in self.paged.iter().filter(|(rank, _, _)| *rank == 0) {
            out.push(DeviceRange { addr: records.ptr as u64 + (page * record_page) as u64, bytes: record_page });
            if let Some(index) = index {
                out.push(DeviceRange { addr: index.ptr as u64 + (page * INDEX_PAGE_BYTES) as u64, bytes: INDEX_PAGE_BYTES });
            }
        }
        out
    }

    fn mark_segments(&self, _slot: MarkSlot) -> Vec<DeviceRange> {
        Vec::new()
    }

    fn host_tail(&self) -> Vec<DeviceRange> {
        vec![DeviceRange { addr: self.scratch.buffer.ptr as u64, bytes: 1 }]
    }
}

/// One continued prefill: every layer's output digest, every row's logits digest and argmax,
/// and the last row.
struct SuffixRun {
    layers: Vec<u64>,
    logits: u64,
    argmax: Vec<usize>,
    last: Vec<f32>,
}

type Experts<'t, 'l> = Option<(&'t mut crate::shared::spark_intake::SparkLink<'l>, &'t tokio::runtime::Runtime)>;

/// Prefill `embed` into `placement` in chunks of `chunk` rows, digesting each layer's rows and
/// (with `logits`) every row's logits.
fn prefill_digest(engine: &GlmEngine<'_>, placement: &mut GlmPlacement, tokens: &[u32], chunk: usize, logits: bool,
    experts: &mut Experts<'_, '_>) -> Result<SuffixRun> {
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
        let transport = experts.as_mut().map(|(t, r)| (&mut **t, *r));
        let out = engine.prefill_rows_logits(placement, part, transport, Some(&mut on_layer),
            if logits { part.len() } else { 1 })?
            .ok_or_else(|| anyhow::anyhow!("the resume check needs every layer"))?;
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

fn download(engine: &GlmEngine<'_>, addr: u64, bytes: usize, template: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
    // SAFETY: the engine owns this stream; draining it retires every write to the range.
    for rank in 0..engine.ranks() {
        unsafe { engine.library.cuda_stream_synchronize(engine.stream_of(rank))? };
    }
    let mut out = vec![0u8; bytes];
    engine.library.copy_d2h(&mut out, CuteafdDeviceBuffer { ptr: addr as *mut std::ffi::c_void, bytes, ..template })?;
    Ok(out)
}

/// Every paged byte of `placement`'s first `len` rows (latent records; index keys and scales on
/// full-indexer layers), layer by layer, in position order.
fn paged_rows(family: &GlmPrefix<'_, '_>, placement: &GlmPlacement, len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let record_page = PAGE_ROWS * RECORD_BYTES;
    for (p, &page) in placement.pages.iter().enumerate().take(len.div_ceil(PAGE_ROWS)) {
        let (page, rows) = (page as usize, (len - p * PAGE_ROWS).min(PAGE_ROWS));
        for &(_, records, index) in &family.paged {
            out.extend_from_slice(&download(family.engine, records.ptr as u64 + (page * record_page) as u64,
                rows * RECORD_BYTES, records)?);
            if let Some(index) = index {
                let bytes = download(family.engine, index.ptr as u64 + (page * INDEX_PAGE_BYTES) as u64,
                    INDEX_PAGE_BYTES, index)?;
                out.extend_from_slice(&bytes[..rows * INDEX_KEY_BYTES]);
                out.extend_from_slice(&bytes[INDEX_SCALES..INDEX_SCALES + rows * 4]);
            }
        }
    }
    Ok(out)
}

/// glm-golden --resume-at P: prefill `tokens[..P]` into sequence A, capture its snapshot (shared
/// pages, the copied tail page), restore it into sequence B, continue both with the same chunks
/// and `decode` greedy single-row steps, and compare every layer's rows, every logit and every
/// paged row byte for byte (a restore must be exact). A straight prefill without the boundary
/// at P is reported too (informational: chunking changes may round differently).
/// With `cold`, B is prefilled from scratch on its own pages instead (no restore): the floor of
/// what the kernels themselves vary. The check runs `repeat` times on fresh sequences (every
/// attempt must be identical: the DSA top-k is deterministic, ties going to the lower index).
#[allow(clippy::too_many_arguments)]
pub(crate) fn resume_check(engine: &GlmEngine<'_>, tokens: &[u32],
    at: usize, n: usize, chunk: usize, decode: usize, cold: bool, repeat: usize, mut experts: Experts<'_, '_>)
    -> Result<()> {
    use super::engine::PageAllocator;
    use anyhow::{ensure, Context};
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--resume-at needs every layer");
    ensure!(engine.full_prefill_logits, "--resume-at needs every prefill row's logits");
    ensure!(experts.is_some() || engine.skip_routed(), "--resume-at needs Spark peers or --skip-routed-experts");
    ensure!(at > 0 && at < n && n <= tokens.len(), "--resume-at {at} must lie inside the {n} prefilled tokens");
    let chunk = chunk.clamp(1, engine.prefill_rows);
    let embed = &tokens[..n];
    let family = GlmPrefix::new(engine, false)?;
    let mut allocator = PageAllocator::new(engine.pages);
    let err = |e: BoxError| anyhow::anyhow!("{e}");
    let (mut identical, mut states_identical, mut last) = (0, 0, None);
    for _ in 0..repeat.max(1) {
    // A: prefill [0, P), capture (pages only), continue in place.
    let mut a = allocator.admit(n + decode)?;
    prefill_digest(engine, &mut a, &embed[..at], chunk, false, &mut experts)?;
    family.drain().map_err(err)?;
    let started = std::time::Instant::now();
    // B: a second sequence restored from the snapshot (shared full pages, its own tail page).
    let mut b = if cold {
        let mut b = allocator.admit(n + decode)?;
        prefill_digest(engine, &mut b, &embed[..at], chunk, false, &mut experts)?;
        b
    } else {
        let (mut b, copy) = allocator.fork(&a, at, n + decode)?;
        if let Some(copy) = copy {
            family.copy_rows(copy).map_err(err)?;
        }
        family.restore(None, &mut b, at).map_err(err)?;
        b
    };
    family.drain().map_err(err)?;
    let restore_ms = started.elapsed().as_secs_f64() * 1e3;
    // The state at P (every paged row): what a restore must reproduce.
    let state_at = paged_rows(&family, &a, at)? == paged_rows(&family, &b, at)?;
    let straight = prefill_digest(engine, &mut a, &embed[at..], chunk, true, &mut experts)?;
    let restored = prefill_digest(engine, &mut b, &embed[at..], chunk, true, &mut experts)?;
    let first_layer = straight.layers.iter().zip(&restored.layers).position(|(x, y)| x != y);
    let logits_equal = straight.logits == restored.logits;
    let max_diff = straight.last.iter().zip(&restored.last).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i as u32);
    let (mut next_a, mut next_b) = (argmax(&straight.last), argmax(&restored.last));
    let mut decode_equal = next_a == next_b;
    for _ in 0..decode {
        let transport = experts.as_mut().map(|(t, r)| (&mut **t, *r));
        let la = engine.verify(&mut [(&mut a, 1)], &[next_a], transport, None)?.context("decode logits")?;
        let transport = experts.as_mut().map(|(t, r)| (&mut **t, *r));
        let lb = engine.verify(&mut [(&mut b, 1)], &[next_b], transport, None)?.context("decode logits")?;
        decode_equal &= la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits());
        (next_a, next_b) = (argmax(&la), argmax(&lb));
    }
    let len = a.len;
    let paged_equal = paged_rows(&family, &a, len)? == paged_rows(&family, &b, len)?;
    println!("resume at {at} of {n} (chunks of {chunk}, {decode} decode steps): state at {at} {} | layers {} | logits \
        {} (last row max |diff| {max_diff:.3e}) | decode {} | paged rows 0..{len} {} | fork+restore {restore_ms:.1} ms",
        if state_at { "identical" } else { "DIFFERS" },
        first_layer.map_or("identical".to_string(), |l| format!("differ from layer {l}")),
        if logits_equal { "identical" } else { "DIFFER" }, if decode_equal { "identical" } else { "DIFFERS" },
        if paged_equal { "identical" } else { "DIFFER" });
    allocator.release(b);
    allocator.release(a);
    identical += usize::from(first_layer.is_none() && logits_equal && decode_equal && paged_equal);
    states_identical += usize::from(state_at);
    last = Some(restored);
    }
    let restored = last.context("no attempt")?;
    let repeat = repeat.max(1);
    // C: one prefill with no boundary at P (chunking changes may round differently; informational).
    let mut c = allocator.admit(n)?;
    let whole = prefill_digest(engine, &mut c, embed, chunk, true, &mut experts)?;
    let x = &whole.argmax[at..];
    let agree = x.iter().zip(&restored.argmax).filter(|(p, q)| p == q).count();
    let last_equal = whole.last.iter().zip(&restored.last).all(|(p, q)| p.to_bits() == q.to_bits());
    println!("vs one straight prefill without the boundary at {at}: suffix top-1 agreement {agree}/{}, last row logits {}",
        x.len(), if last_equal { "identical" } else { "differ" });
    allocator.release(c);
    println!("resume at {at} of {n} (chunks of {chunk}): {identical}/{repeat} attempts byte-identical, state at {at} \
        identical in {states_identical}/{repeat}{}", if cold { " [cold floor: B prefilled, not restored]" } else { "" });
    ensure!(states_identical == repeat, "the restored state differs from the captured one");
    ensure!(identical == repeat, "the restored sequence's continuation differs from the straight one");
    Ok(())
}
