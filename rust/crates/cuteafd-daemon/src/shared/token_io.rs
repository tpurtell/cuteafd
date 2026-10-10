//! Token input and output on the coordinator GPU for every generic-engine
//! family: the token embedding table (a resident BF16 copy of
//! `embed_tokens`, gathered by device token ids) and next-token selection
//! (greedy argmax and the GPU sampler over device logits; only ids and
//! statuses come back to the host).
//!
//! Both have a host twin selected by flag: `--embedding-placement host` keeps
//! one pinned mapped table, gathered by stream-ordered kernels, and
//! `--token-select host` downloads the logits rows and selects with
//! [`TargetSamplingParams::select_token`], the reference the device path
//! must reproduce (token-identical for greedy rows).
use crate::shared::memory::{DeviceAllocation, HostAllocation, ResidentWeight};
use crate::shared::sampler::{TargetSamplingRowRequest, TargetSamplingWave, SAMPLING_MAX_RETAINED};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{TargetSamplingError, TargetSamplingParams};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdV41SamplerRow, NativeLibrary};
use cuteafd_loader::SafetensorsTensorMetadata;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub(crate) use crate::shared::memory::EmbedPlacement;

#[path = "token_io/scores.rs"]
pub(crate) mod scores;
pub(crate) use scores::ScoreRows;

/// Where next tokens are selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum SelectPlacement {
    /// Greedy argmax and the GPU sampler; ids come back, logits stay on the GPU.
    Device,
    /// Logits rows come back and the host selects (the reference path).
    Host,
}

/// The token I/O flags every generic-engine command takes.
#[derive(Debug, Clone, Copy, clap::Args)]
pub(crate) struct TokenIoArgs {
    /// Where the token embedding table lives: `gpu` keeps a BF16 copy on the
    /// coordinator GPU; `host` keeps one pinned mapped copy (untied tables only).
    #[arg(long = "embedding-placement", alias = "embed-placement", value_enum, default_value_t = EmbedPlacement::Gpu)]
    pub embed_placement: EmbedPlacement,
    /// Where next tokens are selected: `device` (only ids come back) or
    /// `host` (logits rows come back; the reference path).
    #[arg(long, value_enum, default_value_t = SelectPlacement::Device)]
    pub token_select: SelectPlacement,
}

impl Default for TokenIoArgs {
    fn default() -> Self {
        Self { embed_placement: EmbedPlacement::Gpu, token_select: SelectPlacement::Device }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum TokenIoError {
    #[error("embedding tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
    #[error("token {token} is outside the {vocab}-row embedding table")]
    TokenOutOfRange { token: u32, vocab: usize },
    #[error("{rows} rows exceed the token selector's {capacity}")]
    Capacity { rows: usize, capacity: usize },
}

/// The BF16 `[vocab, hidden]` embedding tensor in a checkpoint shard.
#[derive(Debug, Clone)]
pub(crate) struct EmbedSource {
    pub path: PathBuf,
    pub offset: u64,
    pub vocab: usize,
    pub hidden: usize,
    pub snapshot: Option<PathBuf>,
    pub tensor_name: String,
}

impl EmbedSource {
    /// `meta` describes the tensor inside `shard` (a file of `snapshot`).
    pub fn new(snapshot: &Path, shard: &str, meta: &SafetensorsTensorMetadata, hidden: usize) -> Result<Self> {
        let error = |detail: String| TokenIoError::Tensor { name: meta.name.clone(), detail };
        ensure!(meta.dtype == cuteafd_core::DType::Bf16, error(format!("dtype {:?}, not BF16", meta.dtype)));
        ensure!(meta.shape.len() == 2 && meta.shape[1] == hidden,
            error(format!("shape {:?}, not [vocab, {hidden}]", meta.shape)));
        let vocab = meta.shape[0];
        ensure!(meta.byte_length == (vocab * hidden * 2) as u64, error(format!("{} bytes", meta.byte_length)));
        Ok(Self { path: snapshot.join(shard), offset: meta.byte_offset, vocab, hidden,
            snapshot: Some(snapshot.to_owned()), tensor_name: meta.name.clone() })
    }

    fn bytes(&self) -> usize {
        self.vocab * self.hidden * 2
    }
}

/// The token embedding table: one device-local or mapped pinned copy.
pub(crate) struct TokenEmbedding<'a> {
    library: &'a NativeLibrary,
    source: EmbedSource,
    /// The shard, kept open: an open through the sparknest mount costs more
    /// than the row reads.
    file: std::fs::File,
    table: ResidentWeight<'a>,
}

impl<'a> TokenEmbedding<'a> {
    /// Opens the table. With [`EmbedPlacement::Gpu`] the table is allocated
    /// first (so it is admitted with the weights) and filled on a background
    /// thread while `during` (the caller's weight load) runs on this one.
    pub fn load<T>(library: &'a NativeLibrary, source: EmbedSource, placement: EmbedPlacement,
        during: impl FnOnce() -> Result<T>) -> Result<(Self, T)> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("embedding");
        let file = std::fs::File::open(&source.path).with_context(|| format!("opening {}", source.path.display()))?;
        if placement == EmbedPlacement::Host {
            if let Some(snapshot) = &source.snapshot {
                cuteafd_loader::plan::checkpoint::Checkpoint::open(snapshot)?
                    .require_untied_embedding(&source.tensor_name)?;
            }
        }
        let mut table = ResidentWeight::new(library, source.bytes(), placement == EmbedPlacement::Host)?;
        let device = library.cuda_get_device()?;
        let (address, started) = (table.buffer().ptr as usize, std::time::Instant::now());
        let result = std::thread::scope(|scope| -> Result<T> {
            let filler = match &mut table {
                ResidentWeight::Host { storage, .. } => {
                    let bytes = storage.bytes_mut();
                    let (file, offset) = (&file, source.offset);
                    scope.spawn(move || file.read_exact_at(bytes, offset).map_err(anyhow::Error::from))
                }
                ResidentWeight::Device(_) => scope.spawn(|| fill_table(library, device, &file, &source, address)),
            };
            let value = during();
            let filled = filler.join().map_err(|_| anyhow::anyhow!("embedding table loader panicked"))?;
            filled.context("loading the embedding table")?;
            value
        })?;
        tracing::info!(vocab = source.vocab, hidden = source.hidden, gib = source.bytes() as f64 / (1u64 << 30) as f64,
            ?placement, device_bytes = table.device_bytes(), pinned_bytes = if table.is_host() { source.bytes() } else { 0 },
            elapsed_ms = started.elapsed().as_millis() as u64, "token embedding placement (single residency)");
        Ok((Self { library, source, file, table }, result))
    }

    pub fn placement(&self) -> EmbedPlacement {
        if self.table.is_host() { EmbedPlacement::Host } else { EmbedPlacement::Gpu }
    }

    pub fn device_gather(&self) -> bool { true }

    pub fn vocab(&self) -> usize {
        self.source.vocab
    }

    pub fn hidden(&self) -> usize {
        self.source.hidden
    }

    pub fn check(&self, tokens: &[u32]) -> Result<(), TokenIoError> {
        match tokens.iter().find(|&&t| t as usize >= self.source.vocab) {
            Some(&token) => Err(TokenIoError::TokenOutOfRange { token, vocab: self.source.vocab }),
            None => Ok(()),
        }
    }

    /// The rows of `tokens` read from the shard (BF16 `[tokens, hidden]`):
    /// the bytes every placement reproduces.
    pub fn host_rows(&self, tokens: &[u32]) -> Result<Vec<u8>> {
        self.check(tokens)?;
        let row = self.source.hidden * 2;
        let mut out = vec![0u8; tokens.len() * row];
        for (slot, &token) in out.chunks_exact_mut(row).zip(tokens) {
            match &self.table {
                ResidentWeight::Host { storage, .. } => slot.copy_from_slice(&storage.bytes()[token as usize * row..(token as usize + 1) * row]),
                ResidentWeight::Device(_) => self.file.read_exact_at(slot, self.source.offset + u64::from(token) * row as u64)?,
            }
        }
        Ok(out)
    }

    /// [`Self::host_rows`] with each row repeated `copies` times.
    pub fn host_rows_repeated(&self, tokens: &[u32], copies: usize) -> Result<Vec<u8>> {
        let rows = self.host_rows(tokens)?;
        if copies == 1 {
            return Ok(rows);
        }
        let row = self.source.hidden * 2;
        let mut out = Vec::with_capacity(rows.len() * copies);
        for r in rows.chunks_exact(row) {
            for _ in 0..copies {
                out.extend_from_slice(r);
            }
        }
        Ok(out)
    }

    /// Writes `copies` copies of each token's row into `out` (BF16
    /// `[tokens * copies, hidden]`) on `stream`. With the GPU table the ids go
    /// up into `ids` (at least `tokens.len()` U32) and the rows are gathered
    /// on the device from either the local or mapped pinned table.
    pub fn embed(&self, tokens: &[u32], ids: CuteafdDeviceBuffer, copies: usize, out: CuteafdDeviceBuffer,
        stream: *mut c_void) -> Result<()> {
        ensure!(!tokens.is_empty() && out.bytes >= tokens.len() * copies * self.source.hidden * 2,
            "embedding of {} rows x {copies} exceeds its output", tokens.len());
        self.check(tokens)?;
        ensure!(ids.bytes >= tokens.len() * 4, "token id buffer of {} bytes for {} ids", ids.bytes, tokens.len());
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.library.copy_h2d(CuteafdDeviceBuffer { bytes: bytes.len(), ..ids }, &bytes)?;
        // SAFETY: `ids` holds the ids (the copy above completed), `out` holds the rows.
        unsafe { self.gather(ids.ptr, std::ptr::null(), tokens.len(), copies, std::ptr::null(), out.ptr, stream) }
    }

    /// Gathers the rows of `rows` device-resident ids (`ids[index[r]]` with a
    /// non-null `index`; an id `>= vocab` takes `fallback`, one BF16 row, or
    /// zeros) into `out` on `stream`.
    ///
    /// # Safety
    /// `ids` (and `index`) hold the ids (indices) once `stream` reaches the
    /// gather; `out` is a live `[rows * copies, hidden]` BF16 buffer and
    /// `fallback` is null or a live row, all on this table's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gather(&self, ids: *const c_void, index: *const c_void, rows: usize, copies: usize,
        fallback: *const c_void, out: *mut c_void, stream: *mut c_void) -> Result<()> {
        let table = self.table.buffer();
        // SAFETY: the caller's contract; the table is a live [vocab, hidden] BF16 buffer.
        unsafe { self.library.cuda_embed_gather_bf16_async(table.ptr, self.source.vocab, self.source.hidden,
            ids, index, rows, copies, fallback, out, stream) }
    }

    /// [`Self::gather`] for either placement, without a device-id download.
    /// `index` and `fallback` pair the device pointer with its host reference.
    ///
    /// # Safety
    /// As for [`Self::gather`]; `ids` is a device buffer holding every id the
    /// rows read.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn embed_device_ids(&self, ids: CuteafdDeviceBuffer, index: Option<(*const c_void, &[u32])>,
        rows: usize, copies: usize, fallback: Option<(*const c_void, &[u8])>, out: CuteafdDeviceBuffer,
        stream: *mut c_void) -> Result<()> {
        let index_ptr = index.map_or(std::ptr::null(), |(device, _)| device);
        let fallback = fallback.map_or(std::ptr::null(), |(device, _)| device);
        // SAFETY: both placements expose a stable device-visible table; caller owns the stream and outputs.
        unsafe { self.gather(ids.ptr, index_ptr, rows, copies, fallback, out.ptr, stream) }
    }

    /// Compares every byte of the resident table with the shard (the
    /// embedding gate); returns the bytes compared.
    pub fn verify_resident(&self) -> Result<usize> {
        let table = self.table.buffer();
        let chunk = 64usize << 20;
        let (mut host, mut device) = (vec![0u8; chunk], vec![0u8; chunk]);
        let total = self.source.bytes();
        let mut at = 0;
        while at < total {
            let n = chunk.min(total - at);
            self.file.read_exact_at(&mut host[..n], self.source.offset + at as u64)?;
            // SAFETY: `at + n` lies inside the table.
            let ptr = unsafe { table.ptr.cast::<u8>().add(at) }.cast();
            self.library.copy_d2h(&mut device[..n], CuteafdDeviceBuffer { ptr, bytes: n, ..table })?;
            ensure!(host[..n] == device[..n], "resident embedding table differs from the shard at byte {at}");
            at += n;
        }
        Ok(total)
    }
}

/// Reads the shard's table into the device allocation at `address` through
/// pinned 64 MiB chunks (runs on the loader thread).
fn fill_table(library: &NativeLibrary, device: i32, file: &std::fs::File, source: &EmbedSource, address: usize)
    -> Result<()> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("embedding");
    library.cuda_set_device(device)?;
    let chunk = 64usize << 20;
    let mut staging = HostAllocation::new(library, chunk)?;
    let total = source.bytes();
    let mut at = 0;
    while at < total {
        let n = chunk.min(total - at);
        file.read_exact_at(&mut staging.bytes_mut()[..n], source.offset + at as u64)?;
        let dst = CuteafdDeviceBuffer { ptr: (address + at) as *mut c_void, bytes: n, device_id: device,
            ..CuteafdDeviceBuffer::default() };
        library.copy_host_buffer_h2d(dst, staging.buffer, n)?;
        at += n;
    }
    Ok(())
}

/// FP32 logits rows on the device: `rows` rows of `vocab`, `stride` floats
/// apart, valid until the producing engine's next step.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeviceLogits {
    pub ptr: *const c_void,
    pub rows: usize,
    pub vocab: usize,
    pub stride: usize,
    /// The producing engine's stream (selection is ordered after it).
    pub stream: *mut c_void,
    /// The engine's own greedy selection of these rows, when it ran one
    /// (inside its decode graph): U32 ids and U32 statuses, `rows` each.
    pub greedy: Option<(*const c_void, *const c_void)>,
}

impl DeviceLogits {
    /// Row `row` downloaded (synchronizes the stream).
    pub fn row_host(&self, library: &NativeLibrary, row: usize) -> Result<Vec<f32>> {
        ensure!(row < self.rows, "logits row {row} of {}", self.rows);
        self.rows_host(library, row, 1)
    }

    /// Every row downloaded (synchronizes the stream).
    pub fn to_host(&self, library: &NativeLibrary) -> Result<Vec<f32>> {
        self.rows_host(library, 0, self.rows)
    }

    fn rows_host(&self, library: &NativeLibrary, first: usize, n: usize) -> Result<Vec<f32>> {
        // SAFETY: the producing engine's stream; the logits are complete once it drains.
        unsafe { library.cuda_stream_synchronize(self.stream)? };
        let mut out = vec![0f32; n * self.vocab];
        for (i, dst) in out.chunks_exact_mut(self.vocab).enumerate() {
            // SAFETY: row first + i lies inside the logits rows.
            let ptr = unsafe { self.ptr.cast::<f32>().add((first + i) * self.stride) }.cast_mut().cast();
            // SAFETY: f32 has no invalid bit patterns; the slice is the row's bytes.
            let bytes = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), self.vocab * 4) };
            library.copy_d2h(bytes, CuteafdDeviceBuffer { ptr, bytes: self.vocab * 4, device_id: library.cuda_get_device()?,
                ..CuteafdDeviceBuffer::default() })?;
        }
        Ok(out)
    }
}

/// One row to select: the request's sampling, the row's draw position, its
/// grammar mask (`None`: every token allowed) and whether its log-probability
/// is wanted.
#[derive(Debug, Clone)]
pub(crate) struct RowSelect {
    pub sampling: TargetSamplingParams,
    pub position: u64,
    pub mask: Option<Vec<u32>>,
    pub logprob: bool,
}

/// A selected token and, when asked for, its log-probability under the raw logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Selected {
    pub token: u32,
    pub logprob: Option<f32>,
}

/// The rows of one selection, in logits-row order.
#[derive(Debug, Default)]
pub(crate) struct SelectBatch {
    pub rows: Vec<RowSelect>,
}

impl SelectBatch {
    /// One sequence's verify rows `input` (the next token, then drafts): row
    /// `j` draws at `first_position + j`, masked by the grammar along the
    /// drafts (the authoritative state already holds `input[0]`). Drafts must
    /// be grammar-legal ([`crate::shared::constraints::State::truncate_proposal`]).
    pub fn push_sequence(&mut self, sampling: TargetSamplingParams,
        constraint: Option<&crate::shared::constraints::State<'_>>, input: &[u32], first_position: u64)
        -> Result<()> {
        let masks = match constraint {
            Some(state) => state.prepare_verification_masks(input)?,
            None => vec![None; input.len()],
        };
        for (j, mask) in masks.into_iter().enumerate() {
            self.rows.push(RowSelect { sampling, position: first_position + j as u64, mask, logprob: false });
        }
        Ok(())
    }

    /// [`Self::push_sequence`] that fails one sequence, never the batch: on a
    /// grammar error its rows go unmasked (the step still runs for its
    /// siblings) and the error comes back for the caller to send to that
    /// request instead of committing its rows.
    pub fn push_sequence_isolated(&mut self, sampling: TargetSamplingParams,
        constraint: Option<&crate::shared::constraints::State<'_>>, input: &[u32], first_position: u64)
        -> Option<String> {
        let rows = self.rows.len();
        match self.push_sequence(sampling, constraint, input, first_position) {
            Ok(()) => None,
            Err(error) => {
                tracing::warn!("request grammar failed: {error:#}");
                self.rows.truncate(rows);
                self.push_sequence(sampling, None, input, first_position).expect("unmasked rows cannot fail");
                Some(format!("request grammar: {error:#}"))
            }
        }
    }

    /// A single row (a prefill's last row) under the grammar's current mask.
    pub fn push_next(&mut self, sampling: TargetSamplingParams,
        constraint: Option<&mut crate::shared::constraints::State<'_>>, position: u64) -> Result<()> {
        let mask = match constraint {
            Some(state) => state.mask()?.map(<[u32]>::to_vec),
            None => None,
        };
        self.rows.push(RowSelect { sampling, position, mask, logprob: false });
        Ok(())
    }

}

pub(crate) type RowResult = std::result::Result<Selected, TargetSamplingError>;

/// Selects next tokens from device logits (see the module docs).
pub(crate) struct TokenSelector<'a> {
    library: &'a NativeLibrary,
    placement: SelectPlacement,
    vocab: usize,
    capacity: usize,
    /// Greedy kernel outputs: ids, statuses, log-probabilities (U32/U32/F32
    /// per row), and their pinned landing.
    out: DeviceAllocation<'a>,
    landing: HostAllocation<'a>,
    /// The GPU sampler, allocated by the first sampled or masked selection.
    wave: Option<TargetSamplingWave<'a>>,
    /// Selections served by the greedy kernel, the sampler, the host.
    pub counts: [u64; 3],
}

/// What the device does with one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Greedy,
    /// K1 + K2 (temperature / min-p).
    Fast,
    /// K1..K5 (top-k <= 256, top-p).
    Ordered,
    /// Downloaded and selected on the host (top-k beyond the device's retained list).
    Host,
}

fn route(params: TargetSamplingParams) -> Route {
    if params.is_greedy() {
        return Route::Greedy;
    }
    match params.top_k() {
        Some(k) if k > SAMPLING_MAX_RETAINED as usize => Route::Host,
        Some(_) => Route::Ordered,
        None if params.top_p() < 1.0 => Route::Ordered,
        None => Route::Fast,
    }
}

/// The sampler's parameter block for a row (V4.1's resolution of the
/// disabled encodings: `top_k None -> 0`, `min_p 0 -> ln = -inf`).
fn sampler_row(params: TargetSamplingParams, position: u64, row: u32, masked: bool, route: Route)
    -> CuteafdV41SamplerRow {
    let mut flags = if masked { 0 } else { cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_NO_MASK };
    let mask_row = if masked { row } else { cuteafd_ffi::CUTEAFD_V41_SAMPLER_NO_MASK_ROW };
    if route == Route::Host {
        // A device-neutral unconstrained greedy row; the host re-selects it.
        return CuteafdV41SamplerRow { position, output_row: row,
            flags: cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_GREEDY | cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE
                | cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, ..CuteafdV41SamplerRow::default() };
    }
    if route == Route::Greedy {
        flags |= cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_GREEDY | cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_STRICT_FINITE;
    }
    CuteafdV41SamplerRow {
        seed: params.seed(),
        position,
        temperature: params.temperature(),
        top_p: params.top_p(),
        min_p: params.min_p(),
        top_k: params.top_k().map_or(0, |k| k as u32),
        mask_row,
        flags,
        output_row: row,
        ln_min_p: if params.min_p() > 0.0 { params.min_p().ln() } else { f32::NEG_INFINITY },
        ..CuteafdV41SamplerRow::default()
    }
}

/// `log_softmax(logits)[token]`.
fn host_logprob(logits: &[f32], token: u32) -> f32 {
    let max = logits.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v - max).exp()).sum();
    logits[token as usize] - max - sum.ln() as f32
}

impl<'a> TokenSelector<'a> {
    /// A selector for up to `capacity` rows of `vocab` logits.
    pub fn new(library: &'a NativeLibrary, placement: SelectPlacement, vocab: usize, capacity: usize) -> Result<Self> {
        ensure!(capacity > 0 && vocab > 0, "token selector of {capacity} rows x {vocab}");
        Ok(Self { library, placement, vocab, capacity, out: DeviceAllocation::new(library, capacity * 12)?,
            landing: HostAllocation::new(library, capacity * 12)?, wave: None, counts: [0; 3] })
    }

    /// Allocates the GPU sampler now (eager start-up); otherwise the first sampled or masked
    /// selection does.
    pub fn reserve_sampler(&mut self) -> Result<()> {
        if self.wave.is_none() && self.placement != SelectPlacement::Host {
            self.wave = Some(TargetSamplingWave::new(self.library, self.capacity.min(128), self.vocab)?);
        }
        Ok(())
    }

    /// Selects `batch.rows[i]` from logits row `i`.
    pub fn select(&mut self, logits: &DeviceLogits, batch: &SelectBatch) -> Result<Vec<RowResult>> {
        let rows = &batch.rows;
        let n = rows.len();
        ensure!(n <= logits.rows && logits.vocab == self.vocab && logits.stride >= self.vocab,
            "selection of {n} rows x {} from {} logits rows x {}", self.vocab, logits.rows, logits.vocab);
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.capacity {
            return Err(TokenIoError::Capacity { rows: n, capacity: self.capacity }.into());
        }
        if self.placement == SelectPlacement::Host {
            self.counts[2] += n as u64;
            let host = logits.to_host(self.library)?;
            return Ok(rows.iter().enumerate()
                .map(|(i, r)| Self::host_select(&host[i * self.vocab..][..self.vocab], r)).collect());
        }
        let routes: Vec<Route> = rows.iter().map(|r| route(r.sampling)).collect();
        if routes.iter().all(|&r| r == Route::Greedy) && rows.iter().all(|r| r.mask.is_none()) {
            return self.greedy(logits, rows);
        }
        self.sampled(logits, rows, &routes)
    }

    fn host_select(logits: &[f32], row: &RowSelect) -> RowResult {
        let token = row.sampling.select_token(logits, row.mask.as_deref(), row.position)? as u32;
        Ok(Selected { token, logprob: row.logprob.then(|| host_logprob(logits, token)) })
    }

    /// Downloads row `i` and selects it on the host.
    fn host_row(&mut self, logits: &DeviceLogits, i: usize, row: &RowSelect) -> Result<RowResult> {
        self.counts[2] += 1;
        Ok(Self::host_select(&logits.row_host(self.library, i)?, row))
    }

    fn greedy(&mut self, logits: &DeviceLogits, rows: &[RowSelect]) -> Result<Vec<RowResult>> {
        let n = rows.len();
        let logprob = rows.iter().any(|r| r.logprob);
        let at = |offset: usize| -> *mut c_void {
            // SAFETY: offsets stay inside the 12-byte-per-row output buffer.
            unsafe { self.out.buffer.ptr.cast::<u8>().add(offset) }.cast()
        };
        let (ids, status, lp) = (at(0), at(4 * self.capacity), at(8 * self.capacity));
        let landing = |offset: usize, bytes: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: offsets stay inside the 12-byte-per-row landing buffer.
            ptr: unsafe { self.landing.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..self.landing.buffer
        };
        let device = |ptr: *const c_void, bytes: usize| CuteafdDeviceBuffer { ptr: ptr.cast_mut(), bytes, ..self.out.buffer };
        // SAFETY: the logits rows, the engine's greedy outputs and ours are
        // live device buffers of these shapes; the stream orders every copy.
        unsafe {
            match logits.greedy.filter(|_| !logprob) {
                Some((engine_ids, engine_status)) => {
                    self.library.copy_d2h_host_buffer_async(landing(0, 4 * n), device(engine_ids, 4 * n), 4 * n,
                        logits.stream)?;
                    self.library.copy_d2h_host_buffer_async(landing(4 * self.capacity, 4 * n),
                        device(engine_status, 4 * n), 4 * n, logits.stream)?;
                }
                None => {
                    self.library.cuda_logits_greedy_f32_async(logits.ptr, n, self.vocab, logits.stride, ids,
                        if logprob { lp } else { std::ptr::null_mut() }, status, logits.stream)?;
                    self.library.copy_d2h_host_buffer_async(self.landing.buffer,
                        CuteafdDeviceBuffer { bytes: 12 * self.capacity, ..self.out.buffer }, 12 * self.capacity,
                        logits.stream)?;
                }
            }
            self.library.cuda_stream_synchronize(logits.stream)?;
        }
        self.counts[0] += n as u64;
        let landing = self.landing.bytes().to_vec();
        let word = |offset: usize| u32::from_le_bytes(landing[offset..offset + 4].try_into().unwrap());
        let mut out = Vec::with_capacity(n);
        for (i, row) in rows.iter().enumerate() {
            if word(4 * self.capacity + 4 * i) != 0 {
                // A non-finite logit: the host reports exactly what it would have.
                out.push(self.host_row(logits, i, row)?);
                continue;
            }
            let logprob = row.logprob.then(|| f32::from_bits(word(8 * self.capacity + 4 * i)));
            out.push(Ok(Selected { token: word(4 * i), logprob }));
        }
        Ok(out)
    }

    fn sampled(&mut self, logits: &DeviceLogits, rows: &[RowSelect], routes: &[Route]) -> Result<Vec<RowResult>> {
        let n = rows.len();
        ensure!(logits.stride == self.vocab, "the GPU sampler reads contiguous logits rows");
        if self.wave.is_none() {
            self.wave = Some(TargetSamplingWave::new(self.library, self.capacity.min(128), self.vocab)?);
        }
        let words = self.vocab.div_ceil(32);
        let masked = rows.iter().any(|r| r.mask.is_some());
        let mut arena = if masked { vec![0u32; n * words] } else { Vec::new() };
        let mut requests = Vec::with_capacity(n);
        for (i, (row, &route)) in rows.iter().zip(routes).enumerate() {
            let mask = row.mask.as_ref().filter(|_| route != Route::Host);
            if let Some(mask) = mask {
                ensure!(mask.len() == words, "grammar mask of {} words for a {}-token vocabulary", mask.len(), self.vocab);
                arena[i * words..(i + 1) * words].copy_from_slice(mask);
            }
            requests.push(TargetSamplingRowRequest {
                row: sampler_row(row.sampling, row.position, i as u32, mask.is_some(), route),
                greedy: matches!(route, Route::Greedy | Route::Host),
            });
        }
        let ordered = routes.contains(&Route::Ordered);
        let buffer = CuteafdDeviceBuffer { ptr: logits.ptr.cast_mut(), bytes: n * self.vocab * 4,
            ..self.out.buffer };
        let wave = self.wave.as_mut().context("sampler")?;
        wave.upload(&requests, masked.then_some(arena.as_slice()), words, logits.stream)?;
        wave.launch(buffer, n, ordered, logits.stream)?;
        // SAFETY: the sampler's copies are queued on this stream.
        unsafe { self.library.cuda_stream_synchronize(logits.stream)? };
        let sampled = wave.output(buffer, n)?;
        self.counts[1] += n as u64;
        let mut out = Vec::with_capacity(n);
        for (i, row) in rows.iter().enumerate() {
            let status = sampled.status[i];
            let device = status == cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK && routes[i] != Route::Host;
            if !device || row.logprob {
                // Host routes, device statuses (the host reports the same
                // error, or selects what the device could not) and logprobs.
                out.push(self.host_row(logits, i, row)?);
                continue;
            }
            out.push(Ok(Selected { token: sampled.ids[i], logprob: None }));
        }
        Ok(out)
    }
}

/// The token I/O gate on a live engine (every family's golden command runs
/// it with `--token-check N`):
/// - the resident embedding table equals the shard byte for byte, and device
///   gathers of a token sample (with repeats) equal the shard's rows;
/// - device greedy selection equals the host's on every step of an `steps`
///   token greedy decode from `first` (`step(token)` runs one decode step);
/// - sampled selection on the first step's logits: the device draws against
///   the host sampler's at the same (seed, position), as exact matches and a
///   chi-square over the tokens either drew.
pub(crate) fn gate(library: &NativeLibrary, embedding: &TokenEmbedding<'_>, first: u32, steps: usize,
    mut step: impl FnMut(u32) -> Result<DeviceLogits>) -> Result<()> {
    let vocab = embedding.vocab();
    if embedding.device_gather() {
        let started = std::time::Instant::now();
        let bytes = embedding.verify_resident()?;
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut sample: Vec<u32> = (0..61).map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % vocab as u64) as u32
        }).collect();
        sample.extend([0, vocab as u32 - 1, sample[3], sample[3]]);
        let row = embedding.hidden() * 2;
        let (ids, out) = (DeviceAllocation::new(library, sample.len() * 4)?,
            DeviceAllocation::new(library, sample.len() * 4 * row)?);
        let stream = library.cuda_stream_create()?;
        let checked = (|| -> Result<()> {
            embedding.embed(&sample, ids.buffer, 4, out.buffer, stream)?;
            // SAFETY: our stream.
            unsafe { library.cuda_stream_synchronize(stream)? };
            let mut device = vec![0u8; sample.len() * 4 * row];
            library.copy_d2h(&mut device, out.buffer)?;
            ensure!(device == embedding.host_rows_repeated(&sample, 4)?, "device-gathered rows differ from the shard");
            Ok(())
        })();
        // SAFETY: drained above (or failed before queueing more).
        unsafe { library.cuda_stream_synchronize(stream)?; library.cuda_stream_destroy(stream)? };
        checked?;
        println!("token gate: embedding table {:.2} GiB identical to the shard; {} gathered rows x4 identical ({:.1} s)",
            bytes as f64 / (1u64 << 30) as f64, sample.len(), started.elapsed().as_secs_f64());
    }
    let mut selector: Option<TokenSelector<'_>> = None;
    let greedy = |logits: &DeviceLogits| -> SelectBatch {
        SelectBatch { rows: (0..logits.rows).map(|i| RowSelect { sampling: TargetSamplingParams::greedy(),
            position: i as u64, mask: None, logprob: false }).collect() }
    };
    let (mut token, mut tokens, mut engine_greedy, mut first_logits) = (first, Vec::new(), 0usize, None);
    for _ in 0..steps {
        let logits = step(token)?;
        ensure!(logits.rows >= 1, "gate step without logits rows");
        if selector.is_none() {
            selector = Some(TokenSelector::new(library, SelectPlacement::Device, logits.vocab, 64)?);
        }
        let device = selector.as_mut().context("selector")?;
        let host = logits.to_host(library)?;
        let last = logits.rows - 1;
        let row = &host[last * logits.vocab..][..logits.vocab];
        let expected = TargetSamplingParams::greedy().select_token(row, None, 0)? as u32;
        // The engine's own (in-graph) selection, then the selector's kernel.
        engine_greedy += usize::from(logits.greedy.is_some());
        let batch = greedy(&logits);
        let ours = device.select(&logits, &batch)?;
        let plain = device.select(&DeviceLogits { greedy: None, ..logits }, &batch)?;
        for (i, (a, b)) in ours.iter().zip(&plain).enumerate() {
            let want = TargetSamplingParams::greedy().select_token(&host[i * logits.vocab..][..logits.vocab], None, 0)?
                as u32;
            let (a, b) = (a.as_ref().map(|s| s.token).ok(), b.as_ref().map(|s| s.token).ok());
            ensure!(a == Some(want) && b == Some(want), "greedy row {i}: engine {a:?}, kernel {b:?}, host {want}");
        }
        if first_logits.is_none() {
            first_logits = Some(row.to_vec());
        }
        token = expected;
        tokens.push(token);
    }
    println!("token gate: {steps} greedy steps, device selection identical to the host ({engine_greedy} from the \
        engine's decode graph); tokens {:?}", &tokens[..tokens.len().min(24)]);
    let (Some(row), Some(mut selector)) = (first_logits, selector) else { return Ok(()) };
    // The fast path (temperature only) and the ordered one (top-k, top-p), hot enough to spread.
    for params in [TargetSamplingParams::new(2.0, 1.0, None, 0.0, 1234)?,
        TargetSamplingParams::new(1.5, 0.95, Some(64), 0.0, 99)?] {
        sample_check(library, &mut selector, &row, params)?;
    }
    Ok(())
}

/// Sampled selection of one logits row at 4096 positions on the device and
/// on the host.
fn sample_check(library: &NativeLibrary, selector: &mut TokenSelector<'_>, row: &[f32], params: TargetSamplingParams)
    -> Result<()> {
    let vocab = row.len();
    let rows = 64usize;
    let buffer = DeviceAllocation::new(library, rows * vocab * 4)?;
    let bytes: Vec<u8> = (0..rows).flat_map(|_| row.iter().flat_map(|v| v.to_le_bytes())).collect();
    library.copy_h2d(buffer.buffer, &bytes)?;
    let stream = library.cuda_stream_create()?;
    let logits = DeviceLogits { ptr: buffer.buffer.ptr, rows, vocab, stride: vocab, stream, greedy: None };
    let (mut matched, mut draws) = (0usize, 0usize);
    let mut counts: std::collections::HashMap<u32, (f64, f64)> = std::collections::HashMap::new();
    let result = (|| -> Result<()> {
        for batch in 0..64u64 {
            let positions: Vec<u64> = (0..rows as u64).map(|i| batch * rows as u64 + i).collect();
            let select = SelectBatch { rows: positions.iter().map(|&position| RowSelect { sampling: params, position,
                mask: None, logprob: false }).collect() };
            for (sel, &position) in selector.select(&logits, &select)?.into_iter().zip(&positions) {
                let device = sel?.token;
                let host = params.select_token(row, None, position)? as u32;
                matched += usize::from(device == host);
                draws += 1;
                counts.entry(device).or_default().0 += 1.0;
                counts.entry(host).or_default().1 += 1.0;
            }
        }
        Ok(())
    })();
    // SAFETY: our stream; the selections synchronized it.
    unsafe { library.cuda_stream_destroy(stream)? };
    result?;
    // Two-sample chi-square over equal-mass bins of softmax(logits / T),
    // tokens ordered by logit: exact matches depend on how flat the row is
    // (a different summation order moves the cumulative boundaries), the
    // distribution must not.
    let (mut chi2, dof) = (0f64, SAMPLE_BINS - 1);
    for (d, h) in mass_bins(row, params.temperature(), &counts) {
        if d + h > 0.0 {
            chi2 += (d - h).powi(2) / (d + h);
        }
    }
    let critical = chi_square_critical(dof);
    println!("token gate: sampled (T {}, top-p {}, top-k {:?}, seed {}) {draws} draws: {matched} identical to the \
        host sampler ({:.2}%), {} distinct tokens, chi-square {chi2:.2} over {dof} dof (p=0.001 bound {critical:.1})",
        params.temperature(), params.top_p(), params.top_k(), params.seed(), 100.0 * matched as f64 / draws as f64,
        counts.len());
    ensure!(chi2 <= critical, "device sampling departs from the host sampler's distribution");
    Ok(())
}

/// Equal-mass bins of the sampling distribution for [`sample_check`].
const SAMPLE_BINS: usize = 32;

/// Device and host draw counts per bin: tokens by descending logit, cut into
/// [`SAMPLE_BINS`] runs of equal softmax(logits / T) mass.
fn mass_bins(row: &[f32], temperature: f32, counts: &std::collections::HashMap<u32, (f64, f64)>) -> Vec<(f64, f64)> {
    let mut order: Vec<usize> = (0..row.len()).filter(|&i| row[i].is_finite()).collect();
    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    let top = order.first().map_or(0.0, |&i| f64::from(row[i]));
    let weight = |i: usize| ((f64::from(row[i]) - top) / f64::from(temperature.max(1e-6))).exp();
    let total: f64 = order.iter().map(|&i| weight(i)).sum();
    let mut bins = vec![(0f64, 0f64); SAMPLE_BINS];
    let mut mass = 0f64;
    for &i in &order {
        let bin = ((mass / total * SAMPLE_BINS as f64) as usize).min(SAMPLE_BINS - 1);
        mass += weight(i);
        if let Some(&(d, h)) = counts.get(&(i as u32)) {
            bins[bin].0 += d;
            bins[bin].1 += h;
        }
    }
    bins
}

/// Wilson-Hilferty upper 0.1% point of chi-square with `dof` degrees of freedom.
fn chi_square_critical(dof: usize) -> f64 {
    let k = dof as f64;
    let z = 3.0902;
    k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_follow_the_device_sampler_limits() {
        let p = |t: f32, top_p: f32, k: Option<usize>| TargetSamplingParams::new(t, top_p, k, 0.0, 7).unwrap();
        assert_eq!(route(TargetSamplingParams::greedy()), Route::Greedy);
        assert_eq!(route(p(0.7, 1.0, Some(1))), Route::Greedy);
        assert_eq!(route(p(0.7, 1.0, None)), Route::Fast);
        assert_eq!(route(p(0.7, 0.9, None)), Route::Ordered);
        assert_eq!(route(p(0.7, 1.0, Some(256))), Route::Ordered);
        assert_eq!(route(p(0.7, 1.0, Some(257))), Route::Host);
    }

    #[test]
    fn sampler_rows_resolve_disabled_encodings() {
        let params = TargetSamplingParams::new(0.8, 1.0, None, 0.0, 42).unwrap();
        let row = sampler_row(params, 9, 3, false, Route::Fast);
        assert_eq!((row.top_k, row.output_row, row.position, row.seed), (0, 3, 9, 42));
        assert_eq!(row.mask_row, cuteafd_ffi::CUTEAFD_V41_SAMPLER_NO_MASK_ROW);
        assert_ne!(row.flags & cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 0);
        assert_eq!(row.ln_min_p, f32::NEG_INFINITY);
        let masked = sampler_row(TargetSamplingParams::greedy(), 1, 2, true, Route::Greedy);
        assert_eq!(masked.mask_row, 2);
        assert_eq!(masked.flags & cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_NO_MASK, 0);
        assert_ne!(masked.flags & cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_GREEDY, 0);
        let host = sampler_row(params, 1, 0, true, Route::Host);
        assert_eq!(host.temperature, 0.0);
        assert_eq!(host.mask_row, cuteafd_ffi::CUTEAFD_V41_SAMPLER_NO_MASK_ROW);
    }

    #[test]
    fn chi_square_bound_matches_tables() {
        assert!((chi_square_critical(31) - 61.10).abs() < 0.3);
        assert!((chi_square_critical(7) - 24.32).abs() < 0.3);
    }

    #[test]
    fn mass_bins_split_the_distribution_evenly() {
        let flat = vec![0.5f32; 1024];
        let counts = (0..1024u32).map(|t| (t, (1.0, 2.0))).collect();
        let bins = mass_bins(&flat, 1.0, &counts);
        assert!(bins.iter().all(|&b| b == (32.0, 64.0)), "{bins:?}");
        // A dominant token fills the first bins alone; the tail lands in the last.
        let mut peaked = vec![-20.0f32; 1024];
        peaked[7] = 10.0;
        let bins = mass_bins(&peaked, 1.0, &counts);
        assert_eq!(bins[0], (1.0, 2.0));
        assert_eq!(bins[SAMPLE_BINS - 1], (1023.0, 2046.0));
    }

    #[test]
    fn host_logprob_is_log_softmax() {
        let logits = [1.0f32, 2.0, 3.0, f32::NEG_INFINITY];
        let z: f64 = [1.0f64, 2.0, 3.0].iter().map(|v| (v - 3.0f64).exp()).sum();
        assert!((host_logprob(&logits, 2) - (-(z.ln()) as f32)).abs() < 1e-6);
    }
}
