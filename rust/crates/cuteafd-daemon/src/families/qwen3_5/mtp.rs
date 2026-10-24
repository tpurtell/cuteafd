//! Checkpoint-native dense MTP math; target embedding, head and scratch stay shared.
use super::*;
use super::super::weights::{DenseFormat, DenseLayer};
use cuteafd_engine::prefix::{DraftBinding, DraftPageAttachment, DraftPageId, DraftPagePool};
use cuteafd_hostcache::copy::DeviceRange;
use cuteafd_loader::families::qwen4::Qwen4KvCache;
use std::path::Path;

/// A page retains 64 native KV rows and just the frontier's last target hidden.
/// Earlier pages' hidden sidecars are stale, so attachment order selects the tail.
#[derive(Clone, Copy)]
struct PageLayout { record_bytes: usize, hidden_bytes: usize }

impl PageLayout {
    fn bytes(self) -> usize { 64 * self.record_bytes + self.hidden_bytes }

    fn tail(frontier: usize, pages: usize) -> Result<usize> {
        ensure!(frontier > 0 && pages == frontier.div_ceil(64), "dense MTP sidecar frontier/page count differs");
        Ok((frontier - 1) / 64)
    }
}

struct Mtp<'a> {
    library: &'a NativeLibrary,
    stream: *mut std::ffi::c_void,
    layer: DenseLayer<'a>,
    fc: Allocation<'a>,
    norm_hidden: Allocation<'a>,
    norm_embedding: Allocation<'a>,
    norm: Allocation<'a>,
    concat: Allocation<'a>,
    hidden: Allocation<'a>,
    logits: Allocation<'a>,
    kv: Allocation<'a>,
    sidecars: Allocation<'a>,
    layout: PageLayout,
    programs: DensePrograms,
    cfg: Qwen35Config,
    homes: LayerStateMap,
    pool: DraftPagePool,
    binding: DraftBinding,
    bindings: Vec<Option<DraftBinding>>,
}

impl Mtp<'_> {
    fn select_slot(&mut self, slot: usize) -> Result<()> {
        self.binding = self.bindings.get(slot).copied().flatten().context("dense MTP slot is not bound")?;
        Ok(())
    }

    fn discard_slot(&mut self, slot: usize) -> Result<()> {
        ensure!(slot < self.bindings.len(), "dense MTP slot outside admission");
        // SAFETY: no binding is recycled while a producer or snapshot copy still reads it.
        unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
        if let Some(binding) = self.bindings[slot] { self.pool.release(binding)?; }
        self.bindings[slot] = None;
        Ok(())
    }

    fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.discard_slot(slot)?;
        self.binding = self.pool.bind(slot, 0)?;
        self.bindings[slot] = Some(self.binding);
        Ok(())
    }
}

fn format(reader: &WeightReader<'_>) -> Result<DenseFormat> {
    match reader.tensor("mtp.layers.0.mlp.gate_proj.weight")?.meta.dtype {
        cuteafd_core::DType::Bf16 => Ok(DenseFormat::Bf16),
        cuteafd_core::DType::F8E4M3 => Ok(DenseFormat::Fp8),
        ref dtype => anyhow::bail!("unsupported dense MTP projection format {dtype:?}"),
    }
}

fn fixed_bytes(reader: &WeightReader<'_>, cfg: &Qwen35Config, format: DenseFormat) -> Result<u64> {
    ensure!(cfg.mtp_layers == 1, "dense MTP requires one checkpoint-native layer");
    let mut bytes = reader.dense_layer_bytes("mtp.layers.0", HybridAttention::Full, format)?;
    for (name, shape) in [("fc", vec![cfg.hidden, 2 * cfg.hidden]), ("norm", vec![cfg.hidden]),
        ("pre_fc_norm_hidden", vec![cfg.hidden]), ("pre_fc_norm_embedding", vec![cfg.hidden])] {
        let tensor = &reader.tensor(&format!("mtp.{name}.weight"))?.meta;
        ensure!(tensor.dtype == cuteafd_core::DType::Bf16 && tensor.shape == shape,
            "mtp.{name}: expected native BF16 {shape:?}");
        bytes = bytes.checked_add(tensor.byte_length.max(256)).context("dense MTP inventory overflow")?;
    }
    // Preserve target hidden rows and keep draft logits separate from target prefill logits.
    Ok(bytes + ((HEAD_ROWS * 2 + ROWS) * cfg.hidden * 2 + HEAD_ROWS * cfg.vocab * 4) as u64)
}

impl<'a> Mtp<'a> {
    fn load(engine: &DenseEngine<'a>, reader: &WeightReader<'a>, format: DenseFormat,
        programs: DensePrograms, expected: u64) -> Result<Self> {
        let device = engine.stream.device;
        let mut cfg = engine.cfg.clone();
        cfg.attention = vec![HybridAttention::Full];
        let homes = LayerStateMap::new(&cfg.attention, &[0], 1)?;
        let mut pool = DraftPagePool::new(engine.pool_tokens, engine.slot_count(), cfg.max_context, 0x514d545000000001)?;
        let binding = pool.bind(0, 0)?;
        let mut bindings = vec![None; engine.slot_count()];
        bindings[0] = Some(binding);
        let layout = PageLayout { record_bytes: Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim),
            hidden_bytes: cfg.hidden * 2 };
        let out = Self {
            library: device.library, stream: engine.stream.raw,
            layer: reader.dense_layer("mtp.layers.0", HybridAttention::Full, format)?,
            fc: reader.bf16("mtp.fc.weight")?, norm_hidden: reader.bf16("mtp.pre_fc_norm_hidden.weight")?,
            norm_embedding: reader.bf16("mtp.pre_fc_norm_embedding.weight")?, norm: reader.bf16("mtp.norm.weight")?,
            concat: Allocation::new(device, HEAD_ROWS * 2 * cfg.hidden * 2)?,
            hidden: Allocation::new(device, ROWS * cfg.hidden * 2)?,
            logits: Allocation::new(device, HEAD_ROWS * cfg.vocab * 4)?,
            kv: Allocation::new(device, engine.pool_tokens * layout.record_bytes)?,
            sidecars: Allocation::new(device, engine.pool_tokens.div_ceil(64) * layout.hidden_bytes)?,
            layout, programs, cfg, homes, pool, binding, bindings,
        };
        let actual = out.layer.bytes() + [&out.fc, &out.norm_hidden, &out.norm_embedding, &out.norm, &out.concat,
            &out.hidden, &out.logits]
            .iter().map(|allocation| allocation.buffer.bytes).sum::<usize>();
        ensure!(actual as u64 == expected, "dense MTP admitted fixed bytes differ from allocations");
        Ok(out)
    }

    fn norm(&self, engine: &DenseEngine<'_>, source: CuteafdDeviceBuffer, weight: CuteafdDeviceBuffer,
        destination: CuteafdDeviceBuffer, rows: usize) -> Result<()> {
        ensure!(source.bytes >= rows * self.cfg.hidden * 2 && destination.bytes >= rows * self.cfg.hidden * 2,
            "dense MTP norm plane too small");
        let program = engine.programs.program(&format!("{}_norm", self.programs.prefix), &["x", "weight", "out"])?;
        // SAFETY: checked source/destination rows and owned BF16 weight survive the shared stream drain.
        unsafe { program.launch(&[source.ptr, weight.ptr, destination.ptr], &[Scalar::I32(i32::try_from(rows)?)], engine.stream.raw) }
    }

    /// Teacher-forced canonical pairs use the target position, never position+1.
    /// Hidden rows are already target-final-normalized, as specified by vLLM.
    fn step(&mut self, engine: &mut DenseEngine<'_>, hidden: CuteafdDeviceBuffer,
        tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        self.step_device(engine, hidden, tokens, start)?.to_host(self.library)
    }

    fn step_device(&mut self, engine: &mut DenseEngine<'_>, hidden: CuteafdDeviceBuffer,
        tokens: &[u32], start: usize) -> Result<crate::shared::token_io::DeviceLogits> {
        let rows = tokens.len();
        ensure!((1..=HEAD_ROWS).contains(&rows), "dense MTP rows exceed shared target head");
        ensure!(self.pool.frontier(self.binding)?.1 == start, "dense MTP canonical frontier differs");
        engine.embedding.check(tokens)?;
        let reservation = self.pool.reserve(self.binding, start, rows)?;
        let result = (|| -> Result<crate::shared::token_io::DeviceLogits> {
            let kv = self.pool.destinations(reservation)?;
            let pages = self.pool.page_table(reservation)?;
            self.pairs_device(engine, hidden, tokens, start, kv, pages)
        })();
        // Recycled destinations are reused only by later work on this one stream.
        self.pool.commit(reservation, if result.is_ok() { rows } else { 0 })?;
        result
    }

    fn pairs_device(&self, engine: &mut DenseEngine<'_>, hidden: CuteafdDeviceBuffer, tokens: &[u32],
        start: usize, kv: Vec<i64>, pages: Vec<i32>) -> Result<crate::shared::token_io::DeviceLogits> {
        let rows = tokens.len();
        ensure!((1..=HEAD_ROWS).contains(&rows), "dense MTP pairs exceed shared target head");
        engine.embedding.check(tokens)?;
        let positions = (start..start + rows).map(i64::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
        let table = rows::Rows { tokens: vec![], physical: vec![], groups: vec![], slots: vec![0; rows],
            first: vec![0; rows], positions, kv, pages, stride: 0 };
        engine.row_tables(&table)?;
        let p = engine.planes;
        engine.embedding.embed(tokens, engine.ids.buffer, 1, p.residual, engine.stream.raw)?;
        self.norm(engine, p.residual, self.norm_embedding.buffer, p.norm, rows)?;
        self.norm(engine, hidden, self.norm_hidden.buffer, p.delta, rows)?;
        let pitch = self.cfg.hidden * 2;
        let second = CuteafdDeviceBuffer { ptr: self.concat.buffer.ptr.cast::<u8>().wrapping_add(pitch).cast(),
            bytes: self.concat.buffer.bytes - pitch, ..self.concat.buffer };
        // SAFETY: concat owns [64,2H] BF16; pitched, disjoint halves retain [embedding,hidden] order.
        unsafe {
            engine.stream.device.library.copy_d2d_2d_async(self.concat.buffer, 2 * pitch, p.norm, pitch,
                pitch, rows, engine.stream.raw)?;
            engine.stream.device.library.copy_d2d_2d_async(second, 2 * pitch, p.delta, pitch,
                pitch, rows, engine.stream.raw)?;
            engine.stream.device.library.linear_bf16(self.concat.buffer.ptr, self.fc.buffer.ptr,
                p.residual.ptr, rows, 2 * self.cfg.hidden, self.cfg.hidden, engine.stream.raw)?;
        }
        let blocks = Blocks { programs: engine.programs, admitted: &self.programs, cfg: &self.cfg,
            homes: &self.homes, bank: &engine.bank, stream: engine.stream.raw };
        blocks.run(0, &self.layer, p, Some(self.kv.buffer), &Step {
            rows, capacity: HEAD_ROWS, table_stride: 0, splits: 1, speculative: false })?;
        self.norm(engine, p.residual, self.norm.buffer, p.norm, rows)?;
        engine.target_head(p.norm, self.logits.buffer, rows)
    }

    fn sidecar(&self, page: usize) -> Result<CuteafdDeviceBuffer> {
        crate::shared::prefix::view(self.sidecars.buffer, page * self.layout.hidden_bytes, self.layout.hidden_bytes)
    }

    fn tail_hidden(&self, frontier: usize) -> Result<CuteafdDeviceBuffer> {
        let pages = self.pool.committed_page_table(self.binding)?;
        let tail = PageLayout::tail(frontier, pages.len())?;
        self.sidecar(pages[tail] as usize)
    }

    /// A target frontier P reserves P rows, but KV(P-1) waits for token(P).
    fn pending(&mut self, hidden: CuteafdDeviceBuffer, frontier: usize) -> Result<()> {
        ensure!(frontier > 0 && self.pool.frontier(self.binding)?.1 == frontier - 1,
            "dense MTP pending row differs from target frontier");
        let row = self.pool.reserve(self.binding, frontier - 1, 1)?;
        self.pool.commit(row, 1)?;
        let destination = self.tail_hidden(frontier)?;
        // SAFETY: the admitted sidecar and target hidden are read and rewritten only on this stream.
        unsafe { self.library.copy_d2d_async(destination, hidden, self.layout.hidden_bytes, self.stream) }
    }

    fn resolve_pending(&self, engine: &mut DenseEngine<'_>, token: u32, frontier: usize) -> Result<Vec<f32>> {
        self.resolve_pending_device(engine, token, frontier)?.to_host(self.library)
    }

    fn resolve_pending_device(&self, engine: &mut DenseEngine<'_>, token: u32, frontier: usize)
        -> Result<crate::shared::token_io::DeviceLogits> {
        ensure!(self.pool.frontier(self.binding)?.1 == frontier, "dense MTP pending frontier differs");
        let pages = self.pool.committed_page_table(self.binding)?;
        let position = frontier.checked_sub(1).context("dense MTP has no pending row")?;
        let destination = pages[position / 64] as i64 * 64 + (position % 64) as i64;
        self.pairs_device(engine, self.tail_hidden(frontier)?, &[token], position, vec![destination], pages)
    }

    fn draft_chain(&mut self, engine: &mut DenseEngine<'_>, anchor: u32, frontier: usize,
        depth: usize, select: impl FnMut(crate::shared::token_io::DeviceLogits, usize) -> Result<u32>)
        -> Result<Vec<u32>> {
        ensure!((1..=3).contains(&depth), "dense MTP chain depth outside 1..=3");
        let logits = self.resolve_pending_device(engine, anchor, frontier)?;
        self.chain_from_logits(engine, logits, frontier, depth, select)
    }

    fn chain_from_logits(&mut self, engine: &mut DenseEngine<'_>,
        logits: crate::shared::token_io::DeviceLogits, frontier: usize, depth: usize,
        mut select: impl FnMut(crate::shared::token_io::DeviceLogits, usize) -> Result<u32>) -> Result<Vec<u32>> {
        ensure!((1..=3).contains(&depth), "dense MTP chain depth outside 1..=3");
        let reservation = if depth > 1 { Some(self.pool.reserve(self.binding, frontier, depth - 1)?) } else { None };
        let result = (|| -> Result<Vec<u32>> {
            let mut tokens = vec![select(logits, 0)?];
            if let Some(reservation) = reservation {
                let destinations = self.pool.destinations(reservation)?;
                let pages = self.pool.page_table(reservation)?;
                for step in 1..depth {
                    // pairs_device reuses norm for embeddings; preserve normalized MTP feedback first.
                    // SAFETY: the owned hidden row and norm plane remain live on the same producer stream.
                    unsafe { self.library.copy_d2d_async(self.hidden.buffer, engine.planes.norm,
                        self.layout.hidden_bytes, self.stream)?; }
                    let logits = self.pairs_device(engine, self.hidden.buffer, &[tokens[step - 1]],
                        frontier + step - 1, vec![destinations[step - 1]], pages.clone())?;
                    tokens.push(select(logits, step)?);
                }
            }
            Ok(tokens)
        })();
        // Recursive feedback is not canonical target hidden, even when its token is accepted.
        // Recycle the provisional suffix; update() recomputes it later on this same stream.
        if let Some(reservation) = reservation { self.pool.commit(reservation, 0)?; }
        result
    }

    fn copy_pages(&self, copies: &[cuteafd_engine::prefix::TailCopy]) -> Result<()> {
        for copy in copies {
            let bytes = 64 * self.layout.record_bytes;
            let source = crate::shared::prefix::view(self.kv.buffer, copy.from as usize * bytes, copy.rows * self.layout.record_bytes)?;
            let destination = crate::shared::prefix::view(self.kv.buffer, copy.to as usize * bytes, source.bytes)?;
            // SAFETY: pool retains both complete owners through the final stream drain.
            unsafe {
                self.library.copy_d2d_async(destination, source, source.bytes, self.stream)?;
                self.library.copy_d2d_async(self.sidecar(copy.to as usize)?, self.sidecar(copy.from as usize)?,
                    self.layout.hidden_bytes, self.stream)?;
            }
        }
        // SAFETY: source/destination references stay retained until all copies retire.
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
    }
}

/// Native cache ownership uses the same aggregate page credits and typed
/// attachments as block drafting; hidden sidecars travel with their KV page.
pub(crate) struct MtpCache<'a>(std::cell::RefCell<Mtp<'a>>);

pub(crate) struct Admission {
    format: DenseFormat,
    programs: DensePrograms,
    fixed: u64,
    pub(crate) bytes: DrafterBytes,
}

impl Admission {
    pub(crate) fn new(reader: &WeightReader<'_>, cfg: &Qwen35Config, programs: &Programs<'_>) -> Result<Self> {
        let format = format(reader)?;
        let programs = DensePrograms::new(cfg, programs, Qwen4KvCache::Bf16, format)?;
        let fixed = fixed_bytes(reader, cfg, format)?;
        let layout = PageLayout { record_bytes: Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim),
            hidden_bytes: cfg.hidden * 2 };
        let bytes = DrafterBytes::native_mtp(reader.checkpoint, cfg)?;
        ensure!(bytes.owned == fixed + 12 && bytes.pool_unit == layout.bytes() as u64,
            "native MTP planner and runtime inventory differ");
        Ok(Self { format, programs, fixed, bytes })
    }
}

impl<'a> MtpCache<'a> {
    pub(crate) fn load(engine: &DenseEngine<'a>, reader: &WeightReader<'a>, admission: Admission) -> Result<Self> {
        ensure!(engine.planes.scratch.bytes >= admission.programs.scratch_bytes,
            "MTP scratch exceeds shared target admission");
        Ok(Self(std::cell::RefCell::new(Mtp::load(engine, reader, admission.format,
            admission.programs, admission.fixed)?)))
    }

    /// Retain canonical INPUT rows only; the sampled bonus has no target hidden yet.
    pub(crate) fn update(&self, engine: &mut DenseEngine<'a>, slot: usize, start: usize,
        tokens: &[u32], target_hidden: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= HEAD_ROWS, "dense MTP update rows outside admission");
        let mut mtp = self.0.borrow_mut();
        mtp.select_slot(slot)?;
        ensure!(mtp.pool.frontier(mtp.binding)?.1 == start, "dense MTP update differs from target frontier");
        let bytes = tokens.len() * mtp.layout.hidden_bytes;
        ensure!(target_hidden.bytes >= bytes, "dense MTP target hidden extent differs");
        // SAFETY: copy on the target producer stream before MTP reuses shared residual/norm planes.
        if mtp.hidden.buffer.ptr != target_hidden.ptr {
            unsafe { mtp.library.copy_d2d_async(mtp.hidden.buffer, target_hidden, bytes, mtp.stream)?; }
        }
        mtp.norm(engine, mtp.hidden.buffer, engine.weights.norm.buffer, mtp.hidden.buffer, tokens.len())?;
        if start > 0 { mtp.resolve_pending_device(engine, tokens[0], start)?; }
        if tokens.len() > 1 {
            let hidden = mtp.hidden.buffer;
            mtp.step_device(engine, hidden, &tokens[1..], start)?;
        }
        let tail = crate::shared::prefix::view(mtp.hidden.buffer, (tokens.len() - 1) * mtp.layout.hidden_bytes,
            mtp.layout.hidden_bytes)?;
        mtp.pending(tail, start + tokens.len())
    }

    pub(crate) fn capture_hidden(&self, source: CuteafdDeviceBuffer, rows: usize) -> Result<()> {
        let mtp = self.0.borrow();
        let bytes = rows * mtp.layout.hidden_bytes;
        ensure!(rows > 0 && bytes <= mtp.hidden.buffer.bytes && bytes <= source.bytes,
            "dense MTP hidden capture exceeds admission");
        // SAFETY: source is a live target producer plane; destination remains owned through stream drain.
        unsafe { mtp.library.copy_d2d_async(mtp.hidden.buffer, source, bytes, mtp.stream) }
    }

    pub(crate) fn update_captured(&self, engine: &mut DenseEngine<'a>, slot: usize, start: usize,
        tokens: &[u32]) -> Result<()> {
        for first in (0..tokens.len()).step_by(HEAD_ROWS) {
            let end = (first + HEAD_ROWS).min(tokens.len());
            let hidden = {
                let mtp = self.0.borrow();
                crate::shared::prefix::view(mtp.hidden.buffer, first * mtp.layout.hidden_bytes,
                    (end - first) * mtp.layout.hidden_bytes)?
            };
            self.update(engine, slot, start + first, &tokens[first..end], hidden)?;
        }
        Ok(())
    }

    pub(crate) fn draft_chain(&self, engine: &mut DenseEngine<'a>, slot: usize, anchor: u32,
        frontier: usize, depth: usize,
        select: impl FnMut(crate::shared::token_io::DeviceLogits, usize) -> Result<u32>) -> Result<Vec<u32>> {
        let mut mtp = self.0.borrow_mut();
        mtp.select_slot(slot)?;
        mtp.draft_chain(engine, anchor, frontier, depth, select)
    }

    pub(crate) fn draft_device(&self, engine: &mut DenseEngine<'a>, slot: usize, anchor: u32,
        frontier: usize) -> Result<crate::shared::token_io::DeviceLogits> {
        let mut mtp = self.0.borrow_mut();
        mtp.select_slot(slot)?;
        mtp.resolve_pending_device(engine, anchor, frontier)
    }
}


impl MtpCache<'_> {
    pub(crate) fn reset_context(&self, slot: usize) -> Result<()> { self.0.borrow_mut().reset_slot(slot) }
    pub(crate) fn discard_context(&self, slot: usize) -> Result<()> { self.0.borrow_mut().discard_slot(slot) }
    pub(crate) fn admit_pages(&self, slot: usize, end: usize) -> Result<()> {
        let mut mtp = self.0.borrow_mut();
        mtp.select_slot(slot)?;
        let binding = mtp.binding;
        Ok(mtp.pool.admit(binding, end, 2)?)
    }
    pub(crate) fn capture_pages(&self, slot: usize, frontier: usize) -> Result<DraftPageAttachment> {
        let mut mtp = self.0.borrow_mut();
        mtp.select_slot(slot)?;
        ensure!(mtp.pool.frontier(mtp.binding)?.1 == frontier && frontier > 0,
            "dense MTP snapshot requires exact pending frontier");
        // SAFETY: producers retire before their page membership is published.
        unsafe { mtp.library.cuda_stream_synchronize(mtp.stream)?; }
        let binding = mtp.binding;
        let (snapshot, copies) = mtp.pool.capture_writable(binding, frontier - 1)?;
        if let Err(error) = mtp.copy_pages(&copies) {
            // SAFETY: failed copies keep destination ownership until they retire.
            if unsafe { mtp.library.cuda_stream_synchronize(mtp.stream) }.is_ok() {
                mtp.pool.release_snapshot(snapshot)?;
            }
            return Err(error);
        }
        Ok(mtp.pool.attachment(snapshot)?)
    }
    pub(crate) fn pages_valid(&self, saved: &DraftPageAttachment) -> bool {
        self.0.borrow().pool.check_attachment(saved).is_ok()
    }
    pub(crate) fn pages_revocable(&self, saved: &DraftPageAttachment) -> bool {
        let mtp = self.0.borrow();
        mtp.pool.check_attachment(saved).is_ok() && mtp.pool.snapshot_revocable(saved.snapshot)
    }
    pub(crate) fn retain_pages(&self, saved: &DraftPageAttachment) -> Result<()> {
        let mut mtp = self.0.borrow_mut();
        mtp.pool.check_attachment(saved)?;
        Ok(mtp.pool.retain_snapshot(saved.snapshot)?)
    }
    pub(crate) fn release_pages(&self, saved: &DraftPageAttachment) -> Result<()> {
        let mut mtp = self.0.borrow_mut();
        // SAFETY: host-copy readers are retired by the caller before releasing their attachment.
        unsafe { mtp.library.cuda_stream_synchronize(mtp.stream)?; }
        mtp.pool.check_attachment(saved)?;
        Ok(mtp.pool.release_snapshot(saved.snapshot)?)
    }
    pub(crate) fn promote_pages(&self, saved: &DraftPageAttachment) -> Result<DraftPageAttachment> {
        Ok(self.0.borrow_mut().pool.promote(saved)?)
    }
    pub(crate) fn restore_pages(&self, slot: usize, saved: &DraftPageAttachment, frontier: usize) -> Result<()> {
        ensure!(frontier > 0 && saved.origin == 0 && saved.frontier == frontier,
            "dense MTP restore differs from target frontier");
        let mut mtp = self.0.borrow_mut();
        ensure!(slot < mtp.bindings.len(), "dense MTP restore outside admitted slots");
        mtp.pool.check_attachment(saved)?;
        // SAFETY: old binding producers retire before its physical pages can be recycled.
        unsafe { mtp.library.cuda_stream_synchronize(mtp.stream)?; }
        mtp.pool.pin_snapshot(saved.snapshot)?;
        if let Err(error) = mtp.discard_slot(slot) {
            mtp.pool.unpin_snapshot(saved.snapshot)?;
            return Err(error);
        }
        let (binding, copies) = match mtp.pool.restore_writable(slot, saved.snapshot, frontier, frontier - 1) {
            Ok(restored) => restored,
            Err(error) => { mtp.pool.unpin_snapshot(saved.snapshot)?; return Err(error.into()); }
        };
        mtp.bindings[slot] = Some(binding);
        mtp.binding = binding;
        let result = mtp.copy_pages(&copies);
        // SAFETY: a failed drain quarantines the source pin and destination binding.
        unsafe { mtp.library.cuda_stream_synchronize(mtp.stream)?; }
        mtp.pool.unpin_snapshot(saved.snapshot)?;
        result
    }
    pub(crate) fn page_bytes(&self) -> usize { self.0.borrow().layout.bytes() }
    pub(crate) fn page_segments(&self, page: DraftPageId) -> Result<Vec<DeviceRange>> {
        let mtp = self.0.borrow();
        ensure!((page.index as usize) < mtp.pool.capacity(), "dense MTP page outside admitted pool");
        let bytes = 64 * mtp.layout.record_bytes;
        Ok(vec![crate::shared::prefix::cuda_copy::range(crate::shared::prefix::view(mtp.kv.buffer,
            page.index as usize * bytes, bytes)?),
            crate::shared::prefix::cuda_copy::range(mtp.sidecar(page.index as usize)?)])
    }
}

impl crate::shared::prefix::cuda_copy::SnapshotOwner for MtpCache<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<CuteafdDeviceBuffer>> {
        let mtp = self.0.borrow();
        Ok(vec![mtp.kv.buffer, mtp.sidecars.buffer])
    }
}

impl Drop for Mtp<'_> {
    fn drop(&mut self) {
        // SAFETY: the target stream outlives this owner, including partially enqueued error paths.
        if let Err(error) = unsafe { self.library.cuda_stream_synchronize(self.stream) } {
            tracing::error!(%error, "retiring dense MTP allocations");
        }
    }
}

fn bf16(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16)).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> Result<f64> {
    ensure!(a.len() == b.len() && !a.is_empty() && a.iter().chain(b).all(|v| v.is_finite()),
        "dense MTP nonfinite or mismatched comparison rows");
    let (dot, aa, bb) = a.iter().zip(b).fold((0., 0., 0.), |(dot, aa, bb), (&a, &b)| {
        let (a, b) = (f64::from(a), f64::from(b));
        (dot + a * b, aa + a * a, bb + b * b)
    });
    ensure!(aa > 0. && bb > 0., "dense MTP zero comparison norm");
    Ok(dot / (aa * bb).sqrt())
}

pub(crate) fn oracle<'a>(args: &super::super::GoldenArgs, library: &'a NativeLibrary,
    programs: &'a Programs<'a>, checkpoint: &'a Checkpoint, cfg: &Qwen35Config,
    target: &Path, reference: &Path, tokens: &[u32]) -> Result<()> {
    use sha2::{Digest, Sha256};
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(reference.join("meta.json"))?)?;
    let rows = tokens.len().checked_sub(1).context("dense MTP empty target")?;
    let width = cfg.hidden * 2;
    let source = std::fs::read(target.join(format!("layer{:02}.bin", cfg.attention.len() - 1)))?;
    let expected = std::fs::read(reference.join("mtp_streams.bin"))?;
    ensure!(rows > 1 && source.len() == tokens.len() * width && expected.len() == rows * width,
        "dense MTP reference hidden rows differ");
    ensure!(meta["family"] == "qwen3_5" && meta["rows"].as_u64() == Some(rows as u64)
        && meta["hidden"].as_u64() == Some(cfg.hidden as u64) && meta["vocab"].as_u64() == Some(cfg.vocab as u64)
        && meta["target_hidden"] == "post-final-norm" && meta["feedback"] == "post-mtp.norm"
        && meta["concat"] == serde_json::json!(["embedding", "hidden"]) && meta["position"] == "target"
        && meta["vllm_revision"] == "0510bd66795683d41f0dcdbcb374bfbd6e91595f"
        && meta["vllm_mtp_blob"] == "2acac6b5b7c5979fbca477dabcd0a0aa5bfb0ff7"
        && meta["source_pin"]["revision"] == "62d7ebd7de4938e072b7aaeb881593b79dc56835"
        && meta["source_pin"]["modeling_sha256"] == "7788020ca357c6c8fb6e750fd847dd80ac77625902213b47f4949c8085344a58"
        && meta["target_hidden_sha256"] == format!("{:x}", Sha256::digest(&source))
        && meta["tokens_sha256"] == format!("{:x}", Sha256::digest(std::fs::read(target.join("tokens.bin"))?)),
        "dense MTP independent reference contract/provenance differs");
    let selected = meta["logit_rows"].as_array().context("dense MTP logit rows missing")?.iter()
        .map(|row| usize::try_from(row.as_u64().context("dense MTP invalid logit row")?).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!selected.is_empty() && selected.iter().all(|&r| r < rows) && selected.windows(2).all(|r| r[0] < r[1]),
        "dense MTP logit selection invalid");
    let payload = std::fs::read(reference.join("mtp_logits.bin"))?;
    ensure!(payload.len() == selected.len() * cfg.vocab * 4, "dense MTP reference logits shape differs");
    let ref_logits = payload.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    let payload = std::fs::read(reference.join("mtp_argmax.bin"))?;
    ensure!(payload.len() == rows * 4, "dense MTP reference argmax shape differs");
    let ref_best = payload.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    let reader = WeightReader { library, checkpoint };
    let format = format(&reader)?;
    let admitted = DensePrograms::new(cfg, programs, Qwen4KvCache::Bf16, format)?;
    let fixed = fixed_bytes(&reader, cfg, format)?;
    // Target planes/scratch/head stay shared; each pooled page adds just one hidden sidecar.
    let layout = PageLayout { record_bytes: Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim), hidden_bytes: width };
    let demand = DrafterBytes { owned: fixed + (HEAD_ROWS * width) as u64, scratch: 0,
        pool_unit: layout.bytes() as u64 };
    let mut engine = DenseEngine::with_prefix(library, programs, checkpoint, cfg.clone(), args.pool_tokens,
        Some(demand), 1, None, args.kv_cache, false)?;
    ensure!(engine.planes.scratch.bytes >= admitted.scratch_bytes, "MTP scratch exceeds shared target admission");
    let input = Allocation::new(engine.stream.device, HEAD_ROWS * width)?;
    let mut mtp = Mtp::load(&engine, &reader, format, admitted, fixed)?;
    mtp.pool.admit(mtp.binding, rows, HEAD_ROWS)?;
    let (mut hidden_cos, mut logit_cos, mut agree, mut nll) = (Vec::new(), Vec::new(), 0usize, 0.);
    for first in (0..rows).step_by(HEAD_ROWS) {
        let end = (first + HEAD_ROWS).min(rows);
        library.copy_h2d(input.buffer, &source[first * width..end * width])?;
        mtp.norm(&engine, input.buffer, engine.weights.norm.buffer, input.buffer, end - first)?;
        let logits = mtp.step(&mut engine, input.buffer, &tokens[first + 1..end + 1], first)?;
        let byte_count = (end - first) * width;
        let mut bytes = vec![0; byte_count];
        library.copy_d2h(&mut bytes, CuteafdDeviceBuffer { bytes: byte_count, ..engine.planes.norm })?;
        let ours = bf16(&bytes);
        let theirs = bf16(&expected[first * width..end * width]);
        for (a, b) in ours.chunks_exact(cfg.hidden).zip(theirs.chunks_exact(cfg.hidden)) { hidden_cos.push(cosine(a, b)?); }
        for (offset, row) in logits.chunks_exact(cfg.vocab).enumerate() {
            let position = first + offset;
            ensure!(row.iter().all(|v| v.is_finite()), "dense MTP nonfinite logits");
            let best = row.iter().enumerate().max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i))).unwrap().0;
            agree += usize::from(best as u32 == ref_best[position]);
            if position + 2 < tokens.len() {
                let top = f64::from(row[best]);
                nll += top + row.iter().map(|&v| (f64::from(v) - top).exp()).sum::<f64>().ln()
                    - f64::from(row[tokens[position + 2] as usize]);
            }
            if let Ok(index) = selected.binary_search(&position) {
                logit_cos.push(cosine(row, &ref_logits[index * cfg.vocab..(index + 1) * cfg.vocab])?);
            }
        }
    }
    ensure!(meta["chain_depth"] == 3 && meta["chain_layer"] == 0
        && meta["chain_feedback"] == "post-mtp.norm" && meta["chain_rows"] == meta["logit_rows"],
        "dense MTP recursive reference contract differs");
    let chain_logits = std::fs::read(reference.join("mtp_chain_logits.bin"))?;
    let chain_hidden = std::fs::read(reference.join("mtp_chain_streams.bin"))?;
    let chain_best = std::fs::read(reference.join("mtp_chain_argmax.bin"))?;
    let chains = selected.len() * 3;
    ensure!(chain_logits.len() == chains * cfg.vocab * 4 && chain_hidden.len() == chains * width
        && chain_best.len() == chains * 4, "dense MTP recursive reference shape differs");
    let chain_logits = chain_logits.chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    let chain_best = chain_best.chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    mtp.reset_slot(0)?;
    mtp.pool.admit(mtp.binding, rows + 2, 3)?;
    let (mut chain_hidden_min, mut chain_logit_min, mut chain_agree) = (1.0f64, 1.0f64, 0usize);
    for position in 0..rows {
        library.copy_h2d(input.buffer, &source[position * width..(position + 1) * width])?;
        mtp.norm(&engine, input.buffer, engine.weights.norm.buffer, input.buffer, 1)?;
        let logits = mtp.step_device(&mut engine, input.buffer, &tokens[position + 1..position + 2], position)?;
        if let Ok(index) = selected.binary_search(&position) {
            let feedback = CuteafdDeviceBuffer { bytes: width, ..engine.planes.norm };
            mtp.chain_from_logits(&mut engine, logits, position + 1, 3, |logits, step| {
                let scores = logits.to_host(library)?;
                let row = index * 3 + step;
                chain_logit_min = chain_logit_min.min(cosine(&scores,
                    &chain_logits[row * cfg.vocab..(row + 1) * cfg.vocab])?);
                let mut bytes = vec![0; width];
                library.copy_d2h(&mut bytes, feedback)?;
                chain_hidden_min = chain_hidden_min.min(cosine(&bf16(&bytes),
                    &bf16(&chain_hidden[row * width..(row + 1) * width]))?);
                let best = scores.iter().enumerate()
                    .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i))).unwrap().0 as u32;
                chain_agree += usize::from(best == chain_best[row]);
                Ok(best)
            })?;
            ensure!(mtp.pool.frontier(mtp.binding)?.1 == position + 1,
                "dense MTP recursive proposal changed canonical frontier");
        }
    }
    println!("dense MTP chain depths 1/2/3: {chains} rows, hidden cosine worst {chain_hidden_min:.8}, logits cosine worst {chain_logit_min:.8}, greedy agreement {chain_agree}/{chains}");
    ensure!(chain_hidden_min >= 0.99 && chain_logit_min >= 0.99 && chain_agree == chains,
        "dense MTP independent recursive oracle failed");
    let hidden_min = hidden_cos.into_iter().fold(1., f64::min);
    let logit_min = logit_cos.into_iter().fold(1., f64::min);
    let nll = nll / (rows - 1) as f64;
    let reference_nll = meta["mean_nll"].as_f64().context("dense MTP reference NLL missing")?;
    println!("dense MTP oracle: {rows} rows, hidden cosine worst {hidden_min:.8}, logits cosine worst {logit_min:.8}, greedy agreement {agree}/{rows}, mean NLL {nll:.8}, reference {reference_nll:.8}");
    ensure!(hidden_min >= 0.99 && logit_min >= 0.99 && (nll - reference_nll).abs() <= args.nll_tolerance,
        "dense MTP independent oracle failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn argmax(logits: &[f32]) -> u32 {
        logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
    }

    #[test]
    #[ignore = "requires dense checkpoint, matching native library and an allocated GPU"]
    fn pending_row_native_continuation_is_exact_at_page_boundaries() -> Result<()> {
        use cuteafd_engine::prefix::{PrefixFamily, MarkSlot, TailCopy};
        use std::cell::RefCell;
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_DENSE_TEST_SNAPSHOT")?);
        let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_DENSE_TEST_MANIFEST")?);
        // SAFETY: library outlives every allocation, stream and prefix-copy owner.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let cfg = Qwen35Config::read(&snapshot)?;
        let checkpoint = Checkpoint::open(&snapshot)?;
        let reader = WeightReader { library: &library, checkpoint: &checkpoint };
        let format = format(&reader)?;
        let admitted = DensePrograms::new(&cfg, &programs, Qwen4KvCache::Bf16, format)?;
        let fixed = fixed_bytes(&reader, &cfg, format)?;
        let layout = PageLayout { record_bytes: Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim),
            hidden_bytes: cfg.hidden * 2 };
        let engine = RefCell::new(DenseEngine::with_prefix(&library, &programs, &checkpoint, cfg.clone(),
            Some(262144), Some(DrafterBytes { owned: fixed + (HEAD_ROWS * layout.hidden_bytes) as u64,
                scratch: 0, pool_unit: layout.bytes() as u64 }), 2,
            Some(crate::shared::prefix::MarkRule { lanes: 2, entries: 2, budget_bytes: 1 << 30, host: false }),
            std::env::var("CUTEAFD_DENSE_TEST_KV_CACHE").unwrap_or_else(|_| "fp8".into()).parse()?, false)?);
        let mut mtp = Mtp::load(&engine.borrow(), &reader, format, admitted, fixed)?;
        let input = Allocation::new(engine.borrow().stream.device, HEAD_ROWS * layout.hidden_bytes)?;
        let selector = RefCell::new(crate::shared::token_io::TokenSelector::new(&library,
            crate::shared::token_io::SelectPlacement::Device, cfg.vocab, 1)?);
        let family = prefix::DensePrefix::new(&engine)?;
        let boxed = |error: cuteafd_engine::prefix::BoxError| anyhow::anyhow!("dense MTP boundary: {error}");
        let download = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes];
            library.copy_d2h(&mut bytes, buffer)?;
            Ok(bytes)
        };
        for frontier in [64usize, 65, 128, 129] {
            let tokens = (0..frontier).map(|i| 100 + (i % 113) as u32).collect::<Vec<_>>();
            let mut placement = prefix::Placement { units: vec![0, 1, 2], slot: 0, len: 0, state_len: 0 };
            family.bind(&mut placement).map_err(boxed)?;
            mtp.reset_slot(0)?;
            mtp.pool.admit(mtp.binding, frontier + 8, HEAD_ROWS)?;
            let mut anchor = 0;
            for first in (0..frontier).step_by(HEAD_ROWS) {
                let end = (first + HEAD_ROWS).min(frontier);
                let mut target = engine.borrow_mut();
                let logits = target.prefill_tapped(0, &tokens[first..end], 1, |layer, hidden| {
                    if layer + 1 == cfg.attention.len() {
                        // SAFETY: capture before target scratch reuse; both owners survive drain.
                        unsafe { library.copy_d2d_async(input.buffer, hidden,
                            (end - first) * layout.hidden_bytes, mtp.stream)?; }
                    }
                    Ok(())
                })?.to_host(&library)?;
                anchor = argmax(&logits);
                mtp.norm(&target, input.buffer, target.weights.norm.buffer, input.buffer, end - first)?;
                let known = if end == frontier { end - first - 1 } else { end - first };
                if known > 0 { mtp.step(&mut target, input.buffer, &tokens[first + 1..first + known + 1], first)?; }
                if end == frontier {
                    mtp.pending(crate::shared::prefix::view(input.buffer, (end - first - 1) * layout.hidden_bytes,
                        layout.hidden_bytes)?, frontier)?;
                }
            }
            placement.len = frontier;
            placement.state_len = frontier;
            family.capture(MarkSlot(0), &placement, frontier).map_err(boxed)?;
            for page in 0..frontier.div_ceil(64) {
                family.copy_rows(TailCopy { from: page as u32, to: page as u32 + 3,
                    rows: (frontier - page * 64).min(64), owner: None }).map_err(boxed)?;
            }
            family.drain().map_err(boxed)?;
            let pages = mtp.pool.committed_page_table(mtp.binding)?;
            let poison = vec![0xc1u8; layout.hidden_bytes];
            for &page in &pages[..pages.len() - 1] { library.copy_h2d(mtp.sidecar(page as usize)?, &poison)?; }
            let initialized = |mtp: &Mtp<'_>| -> Result<Vec<u8>> {
                let pages = mtp.pool.committed_page_table(mtp.binding)?;
                let mut bytes = Vec::new();
                for (i, page) in pages.iter().enumerate() {
                    let rows = (frontier - 1).saturating_sub(i * 64).min(64);
                    if rows > 0 { bytes.extend(download(crate::shared::prefix::view(mtp.kv.buffer,
                        *page as usize * 64 * layout.record_bytes, rows * layout.record_bytes)?)?); }
                }
                bytes.extend(download(mtp.tail_hidden(frontier)?)?);
                Ok(bytes)
            };
            let saved = initialized(&mtp)?;
            let cache = MtpCache(std::cell::RefCell::new(mtp));
            let attachment = cache.capture_pages(0, frontier)?;
            assert!(cache.pages_valid(&attachment));
            assert!(!cache.pages_revocable(&attachment));
            cache.retain_pages(&attachment)?;
            assert!(cache.pages_revocable(&attachment));
            assert_eq!(cache.page_bytes(), layout.bytes());
            let continue_from = |mtp: &mut Mtp<'_>, slot: usize| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
                let mut target = engine.borrow_mut();
                let device = mtp.resolve_pending_device(&mut target, anchor, frontier)?;
                let batch = crate::shared::token_io::SelectBatch { rows: vec![crate::shared::token_io::RowSelect {
                    sampling: cuteafd_core::TargetSamplingParams::greedy(), position: frontier as u64,
                    mask: None, logprob: false }] };
                let selected = selector.borrow_mut().select(&device, &batch)?.into_iter().next()
                    .context("dense MTP selection returned no row")??;
                let first = device.to_host(&library)?;
                assert_eq!(selected.token, argmax(&first), "native MTP device/host prediction at {frontier}");
                let logits = target.prefill_tapped(slot, &[anchor], 1, |layer, hidden| {
                    if layer + 1 == cfg.attention.len() {
                        // SAFETY: capture target hidden on its producer stream before MTP reuses planes.
                        unsafe { library.copy_d2d_async(input.buffer, hidden, layout.hidden_bytes, mtp.stream)?; }
                    }
                    Ok(())
                })?.to_host(&library)?;
                mtp.norm(&target, input.buffer, target.weights.norm.buffer, input.buffer, 1)?;
                mtp.pending(input.buffer, frontier + 1)?;
                let second = mtp.resolve_pending(&mut target, argmax(&logits), frontier + 1)?;
                Ok((first, logits, second))
            };
            let expected = continue_from(&mut cache.0.borrow_mut(), 0)?;
            family.discard(&mut placement).map_err(boxed)?;
            cache.discard_context(0)?;
            placement.units = vec![3, 4, 5];
            placement.slot = 1;
            family.restore(Some(MarkSlot(0)), &mut placement, frontier).map_err(boxed)?;
            // Host promotion restores both independent segments, including the frontier hidden.
            let host = attachment.pages.iter().map(|&page| cache.page_segments(page)?.into_iter().map(|range| {
                download(CuteafdDeviceBuffer { ptr: range.addr as *mut _, bytes: range.bytes,
                    device_id: 0, ..Default::default() })
            }).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
            cache.release_pages(&attachment)?;
            assert!(!cache.pages_valid(&attachment));
            let promoted = cache.promote_pages(&attachment)?;
            for (&page, segments) in promoted.pages.iter().zip(&host) {
                for (range, bytes) in cache.page_segments(page)?.into_iter().zip(segments) {
                    library.copy_h2d(CuteafdDeviceBuffer { ptr: range.addr as *mut _, bytes: range.bytes,
                        device_id: 0, ..Default::default() }, bytes)?;
                }
            }
            cache.restore_pages(1, &promoted, frontier)?;
            cache.admit_pages(1, frontier + 8)?;
            let mut restored = cache.0.borrow_mut();
            assert_eq!(initialized(&restored)?, saved, "initialized KV/hidden at {frontier}");
            let pending_page = restored.pool.committed_page_table(restored.binding)?[(frontier - 1) / 64] as usize;
            library.copy_h2d(crate::shared::prefix::view(restored.kv.buffer, (pending_page * 64 + (frontier - 1) % 64)
                * layout.record_bytes, layout.record_bytes)?, &vec![0xa5; layout.record_bytes])?;
            let actual = continue_from(&mut restored, 1)?;
            drop(restored);
            for (name, expected, actual) in [("draft", &expected.0, &actual.0),
                ("target", &expected.1, &actual.1), ("next draft", &expected.2, &actual.2)] {
                assert!(expected.iter().map(|v| v.to_bits()).eq(actual.iter().map(|v| v.to_bits())),
                    "{name} continuation logits at {frontier}");
                assert_eq!(argmax(expected), argmax(actual), "{name} continuation token at {frontier}");
            }
            family.discard(&mut placement).map_err(boxed)?;
            cache.release_pages(&promoted)?;
            cache.discard_context(1)?;
            cache.reset_context(0)?;
            mtp = cache.0.into_inner();
            println!("dense MTP pending frontier {frontier}: cross-slot host-promoted KV/hidden, GPU selection and target/next-draft logits/tokens byte-exact");
        }
        Ok(())
    }

    #[test]
    fn sidecar_budget_is_one_bf16_hidden_row_per_native_page() {
        let layout = PageLayout { record_bytes: Qwen4KvCache::Bf16.record_bytes(4, 256), hidden_bytes: 5120 * 2 };
        assert_eq!(layout.bytes(), 64 * 4096 + 10240);
        assert_eq!(layout.bytes() * 4096, 1_115_684_864);
    }

    #[test]
    fn pending_tail_sidecar_follows_attachment_order_across_page_boundaries() -> Result<()> {
        for frontier in [64usize, 65, 128, 129] {
            let mut pool = DraftPagePool::new(1024, 2, 262144, 17)?;
            let binding = pool.bind(0, 0)?;
            let reservation = pool.reserve(binding, 0, frontier)?;
            pool.commit(reservation, frontier)?;
            let source = pool.committed_page_table(binding)?;
            let tail = PageLayout::tail(frontier, source.len())?;
            let expected = vec![0x3f80u16; 5120];
            // Only the last sidecar is meaningful, independent of numeric page IDs.
            let mut sidecars = source.iter().map(|&page| (page as u32, vec![0x7fc1u16; 5120]))
                .collect::<HashMap<_, _>>();
            sidecars.insert(source[tail] as u32, expected.clone());
            let (snapshot, copies) = pool.capture_writable(binding, frontier - 1)?;
            assert_eq!(copies.len(), 1);
            assert_eq!(copies[0].from, source[tail] as u32);
            for copy in copies { sidecars.insert(copy.to, sidecars[&copy.from].clone()); }
            let saved = pool.attachment(snapshot)?;
            pool.release(binding)?;
            let (restored, copies) = pool.restore_writable(1, snapshot, frontier, frontier - 1)?;
            assert_eq!(copies.len(), 1);
            for copy in copies { sidecars.insert(copy.to, sidecars[&copy.from].clone()); }
            let pages = pool.committed_page_table(restored)?;
            assert_ne!(pages[tail] as u32, saved.pages[tail].index);
            assert_eq!(sidecars[&(pages[PageLayout::tail(frontier, pages.len())?] as u32)], expected);
            for &page in &pages[..tail] { assert!(sidecars[&(page as u32)].iter().all(|&v| v == 0x7fc1)); }
            pool.release(restored)?;
            pool.release_snapshot(snapshot)?;
            assert_eq!(pool.free(), pool.capacity());
        }
        assert!(PageLayout::tail(0, 0).is_err());
        assert!(PageLayout::tail(65, 1).is_err());
        Ok(())
    }
}
