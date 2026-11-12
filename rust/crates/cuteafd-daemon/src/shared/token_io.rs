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
use crate::shared::peer_split::RankDevice;
use anyhow::{ensure, Context, Result};
use cuteafd_core::{TargetSamplingError, TargetSamplingParams};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdV41SamplerRow, NativeLibrary};
use cuteafd_loader::SafetensorsTensorMetadata;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use crate::shared::memory::Tag;
use cuteafd_core::memory_layout::ArenaTag;

pub(crate) use crate::shared::memory::EmbedPlacement;

#[path = "token_io/native.rs"]
#[allow(dead_code)] // Dormant until native dense switch-on; never selected by the official path.
pub(crate) mod native;
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

    pub(crate) fn allocation_request(&self, id: &str, home: cuteafd_loader::placement::planner::EmbeddingHome)
        -> Result<cuteafd_loader::placement::planner::Allocation> {
        let bytes = self.vocab.checked_mul(self.hidden).and_then(|n| n.checked_mul(2))
            .context("embedding allocation extent overflow")?;
        Self::request(id, home, bytes)
    }

    pub(crate) fn placeholder_request(id: &str, rank: u8)
        -> Result<cuteafd_loader::placement::planner::Allocation> {
        Self::request(id, cuteafd_loader::placement::planner::EmbeddingHome::Device(rank),
            cuteafd_loader::shared::token_table::native_embedding_owner_bytes(0))
    }

    fn request(id: &str, home: cuteafd_loader::placement::planner::EmbeddingHome, bytes: usize)
        -> Result<cuteafd_loader::placement::planner::Allocation> {
        use cuteafd_loader::placement::planner::{Allocation, EmbeddingHome, Lifetime, Location};
        ensure!(bytes > 0, "empty embedding allocation");
        let location = match home {
            EmbeddingHome::Device(rank) => Location::coordinator(rank),
            EmbeddingHome::HostPinned => Location::PINNED,
        };
        Ok(Allocation::buffer(id, location, bytes as u64, Lifetime::RESIDENT,
            cuteafd_core::memory_layout::Category::Embedding, "token embedding"))
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
    /// An EXLR8 checkpoint's row-FP8 table (the rank-0 vocabulary owner, shared
    /// with its head): gathers decode `bf16(e4m3 * scale)` per row and
    /// `table` is an unused 256-byte placeholder.
    row_fp8: Option<std::rc::Rc<native::NativeVocabulary<'a>>>,
}

impl<'a> TokenEmbedding<'a> {
    pub(crate) fn allocation_receipts(&self, devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>, prefix: &str)
        -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::Lifetime;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, prefix,
            Lifetime::RESIDENT, Category::Weights, "embedding owners");
        match &self.table {
            ResidentWeight::Device(owner) => owner.inventory(&mut inventory, "table")?,
            ResidentWeight::Host { storage, .. } => inventory.pinned("table", storage.buffer)?,
        }
        if let Some(vocabulary) = &self.row_fp8 {
            vocabulary.embedding.inventory(&mut inventory, "native")?;
        }
        Ok(inventory.receipts)
    }

    pub(crate) fn table_receipt(&self, devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>,
        prefix: &str, name: &str) -> Result<cuteafd_loader::placement::planner::Receipt> {
        use cuteafd_loader::placement::planner::Lifetime;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, prefix,
            Lifetime::RESIDENT, cuteafd_core::memory_layout::Category::Embedding, "token embedding");
        if let ResidentWeight::Device(owner) = &self.table {
            ensure!(owner.backing.as_ref().and_then(|allocation| allocation.weight_backing()).is_none(),
                "arena-backed embedding requires allocation_receipts, not table_receipt");
        }
        match &self.table {
            ResidentWeight::Device(owner) => inventory.device(name, owner.buffer)?,
            ResidentWeight::Host { storage, .. } => inventory.pinned(name, storage.buffer)?,
        }
        inventory.receipts.pop().context("embedding table receipt")
    }

    pub(crate) fn allocation_aliases(&self, census: &mut crate::shared::memory::receipts::Census,
        devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>, prefix: &str) -> Result<()> {
        if let ResidentWeight::Host { alias, .. } = &self.table {
            census.mapped(format!("{prefix}.table.mapped"), format!("{prefix}.table"), *alias, devices)?;
        }
        Ok(())
    }

    pub(crate) fn shares_vocabulary(&self, vocabulary: &std::rc::Rc<native::NativeVocabulary<'_>>) -> bool {
        self.row_fp8.as_ref().is_some_and(|owner| std::rc::Rc::ptr_eq(owner, vocabulary))
    }

    /// Opens the table. With [`EmbedPlacement::Gpu`] the table is allocated
    /// first (so it is admitted with the weights) and filled on a background
    /// thread while `during` runs. Under dense recording, its operand is
    /// recorded on the caller's thread instead, and filled before publication.
    pub fn load<T>(library: &'a NativeLibrary, source: EmbedSource, placement: EmbedPlacement,
        during: impl FnOnce() -> Result<T>) -> Result<(Self, T)> {
        Self::load_with(library, source, placement, during, |_, bytes| {
            crate::shared::memory::device::Allocation::new(crate::shared::memory::device::Device {
                library, id: library.cuda_get_device()?,
            }, bytes, Tag::embedding("shared/embedding"))
        })
    }

    pub fn load_with<T>(library: &'a NativeLibrary, source: EmbedSource, placement: EmbedPlacement,
        during: impl FnOnce() -> Result<T>,
        mut allocate: impl FnMut(&str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>,
    ) -> Result<(Self, T)> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("embedding");
        let file = std::fs::File::open(&source.path).with_context(|| format!("opening {}", source.path.display()))?;
        if placement == EmbedPlacement::Host {
            if let Some(snapshot) = &source.snapshot {
                cuteafd_loader::plan::checkpoint::Checkpoint::open(snapshot)?
                    .require_untied_embedding(&source.tensor_name)?;
            }
        }
        let home = if placement == EmbedPlacement::Host {
            cuteafd_loader::placement::planner::EmbeddingHome::HostPinned
        } else { cuteafd_loader::placement::planner::EmbeddingHome::Device(0) };
        let request = source.allocation_request("embedding.table", home)?;
        let mut table = if placement == EmbedPlacement::Host {
            ResidentWeight::new(library, usize::try_from(request.bytes)?, true, ArenaTag::embedding("shared/embedding"))?
        } else {
            let owner = allocate("embedding.table", usize::try_from(request.bytes)?)?;
            ensure!(std::ptr::eq(owner.device.library, library), "embedding allocation library mismatch");
            ResidentWeight::Device(DeviceAllocation::from_allocation(owner))
        };
        let device = library.cuda_get_device()?;
        ensure!(table.buffer().bytes as u64 == request.bytes
            && table.is_host() == (placement == EmbedPlacement::Host)
            && (table.is_host() || table.buffer().device_id == device), "embedding allocation differs from request");
        let (address, started) = (table.buffer().ptr as usize, std::time::Instant::now());
        // The recorder is thread-local: record the GPU destination before the
        // caller's weights, on this thread, without starting a mount reader.
        let record = placement == EmbedPlacement::Gpu
            && crate::shared::experts::rtx::dense::is_recording();
        let result = if record {
            fill_table(library, device, &source, address)?;
            during()
        } else { std::thread::scope(|scope| -> Result<T> {
            let filler = match &mut table {
                ResidentWeight::Host { storage, .. } => {
                    let bytes = storage.bytes_mut();
                    let (file, offset) = (&file, source.offset);
                    scope.spawn(move || file.read_exact_at(bytes, offset).map_err(anyhow::Error::from))
                }
                ResidentWeight::Device(_) => scope.spawn(|| fill_table(library, device, &source, address)),
            };
            let value = during();
            let filled = filler.join().map_err(|_| anyhow::anyhow!("embedding table loader panicked"))?;
            filled.context("loading the embedding table")?;
            value
        }) }?;
        tracing::info!(vocab = source.vocab, hidden = source.hidden, gib = source.bytes() as f64 / (1u64 << 30) as f64,
            ?placement, device_bytes = table.device_bytes(), pinned_bytes = if table.is_host() { source.bytes() } else { 0 },
            elapsed_ms = started.elapsed().as_millis() as u64, "token embedding placement (single residency)");
        Ok((Self { library, source, file, table, row_fp8: None }, result))
    }

    /// The row-FP8 table of an EXLR8 checkpoint's vocabulary owner (`source`
    /// names the payload `embed_tokens.fp8`; its scales follow the owner).
    pub fn row_fp8(library: &'a NativeLibrary, source: EmbedSource,
        owner: std::rc::Rc<native::NativeVocabulary<'a>>) -> Result<Self> {
        Self::row_fp8_with(library, source, owner, |_, bytes| {
            crate::shared::memory::device::Allocation::new(crate::shared::memory::device::Device {
                library, id: library.cuda_get_device()?,
            }, bytes, Tag::embedding("shared/embedding"))
        })
    }

    /// Vocabulary planes are loaded through `NativeVocabulary::load_prepared_with`;
    /// this owner injects only its distinct embedding placeholder.
    pub fn row_fp8_with(library: &'a NativeLibrary, source: EmbedSource,
        owner: std::rc::Rc<native::NativeVocabulary<'a>>,
        mut allocate: impl FnMut(&str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>) -> Result<Self> {
        let file = std::fs::File::open(&source.path).with_context(|| format!("opening {}", source.path.display()))?;
        ensure!(owner.embedding.payload().bytes == source.vocab * source.hidden
            && owner.embedding.scales().bytes == source.vocab * 4
            && owner.embedding.payload().device_id == owner.embedding.scales().device_id,
            "row-FP8 embedding payload/scales differ from [{}, {}] on one device", source.vocab, source.hidden);
        let request = EmbedSource::placeholder_request("embedding.table", 0)?;
        let allocation = allocate("embedding.table", usize::try_from(request.bytes)?)?;
        ensure!(std::ptr::eq(allocation.device.library, library), "embedding allocation library mismatch");
        let table = ResidentWeight::Device(DeviceAllocation::from_allocation(allocation));
        ensure!(table.buffer().bytes as u64 == request.bytes
            && table.buffer().device_id == library.cuda_get_device()? && !table.buffer().ptr.is_null(),
            "native embedding placeholder differs from request");
        tracing::info!(vocab = source.vocab, hidden = source.hidden, "token embedding: checkpoint row-FP8 (single residency)");
        Ok(Self { library, source, file, table, row_fp8: Some(owner) })
    }

    pub fn placement(&self) -> EmbedPlacement {
        if self.table.is_host() { EmbedPlacement::Host } else { EmbedPlacement::Gpu }
    }

    /// CUDA owner of the resident table or mapped alias.
    pub(crate) fn owner(&self) -> crate::shared::memory::device::Device<'a> {
        let id = self.row_fp8.as_ref().map_or_else(|| self.table.buffer().device_id,
            |owner| owner.embedding.residency().device);
        crate::shared::memory::device::Device { library: self.library, id }
    }

    /// Device bytes of this table only; the vocabulary head charges its own table.
    pub(crate) fn device_bytes(&self) -> usize {
        self.row_fp8.as_ref().map_or_else(|| self.table.device_bytes(),
            |owner| owner.embedding.residency().bytes)
    }

    /// All owned device allocations, including the native table placeholder.
    pub(crate) fn device_owner_bytes(&self) -> usize {
        self.row_fp8.as_ref().map_or_else(|| self.table.device_bytes(),
            |owner| cuteafd_loader::shared::token_table::native_embedding_owner_bytes(
                owner.embedding.residency().bytes))
    }

    pub(crate) fn validate_residency(&self, device: i32, bytes: u64) -> Result<()> {
        validate_embedding_residency(self.owner().id, self.device_bytes(), device, bytes)
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
        if let Some(owner) = &self.row_fp8 {
            for (slot, &token) in out.chunks_exact_mut(row).zip(tokens) {
                owner.embedding.host_row(self.library, token as usize, self.source.hidden, slot)?;
            }
            return Ok(out);
        }
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
        // SAFETY: the supplied stream owns ids and its gather; the helper keeps
        // pinned input live and orders replacement after previous consumers.
        unsafe { self.library.copy_h2d_ordered(ids, &bytes, stream)?; }
        // SAFETY: `ids` becomes initialized before the gather on this stream; out holds the rows.
        unsafe { self.gather(ids.ptr, std::ptr::null(), tokens.len(), copies, std::ptr::null(), out.ptr, None,
            RankDevice { device: ids.device_id, stream }) }
    }

    /// Gathers the rows of `rows` device-resident ids (`ids[index[r]]` with a
    /// non-null `index`; an id `>= vocab` takes `fallback`, one BF16 row, or
    /// zeros) into `out` on `stream`.
    ///
    /// # Safety
    /// `producer.device` owns its live stream. `ids` (and `index`) hold the ids
    /// (indices) once that stream reaches the gather; `out` is a live
    /// `[rows * copies, hidden]` BF16 buffer and `fallback` is null or a live
    /// row, all on the producer's device. The table may live on a peer with
    /// peer access enabled from this device;
    /// its upload must finish before gathering and its lifetime spans the work.
    /// Row-FP8 with multiple copies requires disjoint caller scratch of at least
    /// `[rows, hidden]` BF16 on this device, live until the same stream drains.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gather(&self, ids: *const c_void, index: *const c_void, rows: usize, copies: usize,
        fallback: *const c_void, out: *mut c_void, scratch: Option<CuteafdDeviceBuffer>,
        producer: RankDevice) -> Result<()> {
        let RankDevice { device, stream } = producer;
        if let Some(owner) = &self.row_fp8 {
            ensure!(index.is_null() && fallback.is_null(),
                "the row-FP8 embedding gathers plain ids (no index or fallback)");
            return gather_row_fp8_copies(rows, self.source.hidden, copies, out, scratch, device, |op| {
                match op {
                    // SAFETY: the caller retains ids, output and optional scratch through this stream.
                    EmbeddingOp::Gather(output) => unsafe { owner.embed(ids.cast_mut(), output, rows, stream) },
                    // SAFETY: checked disjoint pitched views, ordered after the gather on this stream.
                    EmbeddingOp::Copy { dst, src, width, dst_pitch } => unsafe {
                        self.library.copy_device_rows_async(dst, src, width, rows, dst_pitch, width, stream)
                    },
                }
            });
        }
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
        unsafe { self.gather(ids.ptr, index_ptr, rows, copies, fallback, out.ptr, None,
            RankDevice { device: ids.device_id, stream }) }
    }

    /// Compares every byte of the resident table with the shard (the
    /// embedding gate); returns the bytes compared.
    pub fn verify_resident(&self) -> Result<usize> {
        ensure!(self.row_fp8.is_none(), "the row-FP8 table is verified by its owner's checksummed load");
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

enum EmbeddingOp {
    Gather(*mut c_void),
    Copy { dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer, width: usize, dst_pitch: usize },
}

fn gather_row_fp8_copies(rows: usize, hidden: usize, copies: usize, out: *mut c_void,
    scratch: Option<CuteafdDeviceBuffer>, device: i32, mut launch: impl FnMut(EmbeddingOp) -> Result<()>) -> Result<()> {
    ensure!(rows > 0 && hidden > 0 && copies > 0 && !out.is_null(), "empty row-FP8 embedding gather");
    let width = hidden.checked_mul(2).context("embedding row overflow")?;
    let bytes = rows.checked_mul(width).context("embedding scratch overflow")?;
    let pitch = copies.checked_mul(width).context("embedding copies overflow")?;
    let output_bytes = rows.checked_mul(pitch).context("embedding output overflow")?;
    let output_end = (out as usize).checked_add(output_bytes).context("embedding output address overflow")?;
    if copies == 1 { return launch(EmbeddingOp::Gather(out)); }
    let src = scratch.context("row-FP8 embedding copies require caller scratch")?;
    ensure!(!src.ptr.is_null() && src.bytes >= bytes && src.device_id == device && src.flags == 0,
        "row-FP8 embedding scratch differs from required device/extent");
    let scratch_end = (src.ptr as usize).checked_add(bytes).context("embedding scratch address overflow")?;
    ensure!(output_end <= src.ptr as usize || scratch_end <= out as usize,
        "row-FP8 embedding scratch overlaps output");
    launch(EmbeddingOp::Gather(src.ptr))?;
    for copy in 0..copies {
        let offset = copy * width;
        let dst = CuteafdDeviceBuffer { ptr: out.cast::<u8>().wrapping_add(offset).cast(),
            bytes: output_bytes - offset, device_id: device, flags: 0 };
        launch(EmbeddingOp::Copy { dst, src, width, dst_pitch: pitch })?;
    }
    Ok(())
}

pub(super) fn validate_embedding_residency(actual_device: i32, actual_bytes: usize,
    device: i32, bytes: u64) -> Result<()> {
    ensure!(actual_device == device && actual_bytes as u64 == bytes,
        "embedding differs from admitted owner/bytes: actual GPU {actual_device}, {actual_bytes} bytes; admitted GPU {device}, {bytes} bytes");
    Ok(())
}

/// E4M3 byte to F32 (finite codes; NaN for 0x7f/0xff).
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((b >> 3) & 0xf, b & 7);
    if e == 15 && m == 7 { return f32::NAN; }
    sign * if e == 0 { f32::from(m) / 8.0 * 2f32.powi(-6) } else { (1.0 + f32::from(m) / 8.0) * 2f32.powi(i32::from(e) - 7) }
}

/// F32 to BF16 bits, round to nearest even (NaN quieted).
fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() { return ((bits >> 16) | 0x40) as u16; }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// One unchanged operand for both pushed and bounded mounted GPU loading.
fn fill_table(library: &NativeLibrary, device: i32, source: &EmbedSource, address: usize) -> Result<()> {
    use crate::shared::experts::rtx::dense::{DenseLoad, Operand, Rect};
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("embedding");
    let bytes = source.bytes();
    let row_bytes = source.hidden * 2;
    DenseLoad::mount(library).operand(Operand { device,
        buffer: CuteafdDeviceBuffer { ptr: address as *mut c_void, bytes, device_id: device,
            ..CuteafdDeviceBuffer::default() },
        lands: vec![(Rect { name: source.tensor_name.clone(), path: source.path.clone(), base: source.offset,
            row_bytes, rows: 0..source.vocab, cols: 0..row_bytes }, 0)], bytes })
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
        if self.stride == self.vocab {
            return self.download(library, &[(first, n)]);
        }
        self.download(library, &(first..first + n).map(|row| (row, 1)).collect::<Vec<_>>())
    }

    /// Rows `[row, row + count)` of each span, concatenated. Every copy queues on
    /// the producing stream behind the logits, so one drain completes them all.
    pub(crate) fn download(&self, library: &NativeLibrary, spans: &[(usize, usize)]) -> Result<Vec<f32>> {
        let total: usize = spans.iter().map(|&(_, count)| count).sum();
        ensure!(spans.iter().all(|&(row, count)| row + count <= self.rows && (count == 1 || self.stride == self.vocab)),
            "logits download outside {} rows", self.rows);
        let mut out = vec![0f32; total * self.vocab];
        // SAFETY: DeviceLogits retains its producing stream through this download.
        let device_id = unsafe { library.cuda_stream_device(self.stream)? };
        let _device = crate::shared::memory::device::Device { library, id: device_id }.enter()?;
        let mut rest = out.as_mut_slice();
        let queued = spans.iter().try_for_each(|&(row, count)| {
            let (dst, tail) = std::mem::take(&mut rest).split_at_mut(count * self.vocab);
            rest = tail;
            // SAFETY: row `row` lies inside the logits rows.
            let ptr = unsafe { self.ptr.cast::<f32>().add(row * self.stride) }.cast_mut().cast();
            // SAFETY: f32 has no invalid bit patterns; the slice is `dst`'s bytes.
            let bytes = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), dst.len() * 4) };
            // SAFETY: `out` outlives the drain below; the source rows are live logits of this stream.
            unsafe { library.copy_d2h_async(bytes, CuteafdDeviceBuffer { ptr, bytes: bytes.len(), device_id,
                ..CuteafdDeviceBuffer::default() }, self.stream) }
        });
        // SAFETY: the producing engine's stream; the queued copies complete before `out`
        // is read or, after a failed enqueue, dropped.
        let drained = unsafe { library.cuda_stream_synchronize(self.stream) };
        queued.and(drained)?;
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
    /// Per CUDA device: an ordering event recorded behind a selection's
    /// landing copies, so the host waits for those and not for whatever is
    /// queued on the logits stream after them.
    /// The step's sampling wait: sleeps, does not spin the engine thread.
    landed: crate::shared::memory::BlockingWait<'a>,
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

fn check_selector_device(output: i32, stream: i32) -> Result<()> {
    ensure!(output == stream, "token selector output operand on device {output}, logits stream on device {stream}");
    Ok(())
}

impl<'a> TokenSelector<'a> {
    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        inventory.device(&format!("{name}.greedy"), self.out.buffer)?;
        inventory.pinned(&format!("{name}.landing"), self.landing.buffer)?;
        if let Some(wave) = &self.wave { wave.allocation_receipts(inventory, &format!("{name}.sampler"))?; }
        Ok(())
    }

    /// A selector for up to `capacity` rows of `vocab` logits.
    pub fn new(library: &'a NativeLibrary, placement: SelectPlacement, vocab: usize, capacity: usize) -> Result<Self> {
        ensure!(capacity > 0 && vocab > 0, "token selector of {capacity} rows x {vocab}");
        Ok(Self { library, placement, vocab, capacity, out: DeviceAllocation::new(library, capacity * 12, Tag::workspace("shared/token-out"))?,
            landing: HostAllocation::new(library, capacity * 12, Tag::staging("shared/token-out"))?, wave: None, landed: crate::shared::memory::BlockingWait::new(library), counts: [0; 3] })
    }

    /// Waits until `stream` passed the work queued on it so far (the
    /// selection's landing copies), through this device's ordering event,
    /// then checks that no Spark exchange released this step over a failed
    /// wire.
    ///
    /// # Safety
    /// `stream` is a live stream of the current device.
    unsafe fn wait_landed(&mut self, stream: *mut c_void) -> Result<()> {
        // SAFETY: forwarded from this function's contract.
        unsafe { self.landed.wait(stream)? };
        crate::shared::spark_waves::healthy()
    }

    /// Allocates the GPU sampler now (eager start-up); otherwise the first sampled or masked
    /// selection does.
    pub fn reserve_sampler(&mut self) -> Result<()> {
        let _device = crate::shared::memory::device::Device { library: self.library, id: self.out.buffer.device_id }.enter()?;
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
        // SAFETY: selection consumes live rows on the producing engine's stream.
        let owner = unsafe { self.library.cuda_stream_device(logits.stream)? };
        let _device = crate::shared::memory::device::Device { library: self.library, id: owner }.enter()?;
        self.bind_device(owner)?;
        if self.placement == SelectPlacement::Host {
            self.counts[2] += n as u64;
            let host = logits.to_host(self.library)?;
            crate::shared::spark_waves::healthy()?;
            return Ok(rows.iter().enumerate()
                .map(|(i, r)| Self::host_select(&host[i * self.vocab..][..self.vocab], r)).collect());
        }
        let routes: Vec<Route> = rows.iter().map(|r| route(r.sampling)).collect();
        if routes.iter().all(|&r| r == Route::Greedy) && rows.iter().all(|r| r.mask.is_none()) {
            return self.greedy(logits, rows);
        }
        self.sampled(logits, rows, &routes)
    }

    /// A selector can follow a different head, but must never reuse the old
    /// head's output/sampler allocations on the new stream. Selections drain
    /// their landing copies before returning, so the old arena is retired.
    fn bind_device(&mut self, owner: i32) -> Result<()> {
        let previous = self.out.buffer.device_id;
        if previous != owner {
            let out = DeviceAllocation::new(self.library, self.capacity * 12, Tag::workspace("shared/token-out"))?;
            let old = std::mem::replace(&mut self.out, out);
            let wave = self.wave.take();
            crate::shared::memory::device::Device { library: self.library, id: previous }.run(|| {
                drop(wave);
                drop(old);
                Ok(())
            })?;
        }
        check_selector_device(self.out.buffer.device_id, owner)
    }

    fn host_select(logits: &[f32], row: &RowSelect) -> RowResult {
        let token = row.sampling.select_token(logits, row.mask.as_deref(), row.position)? as u32;
        Ok(Selected { token, logprob: row.logprob.then(|| host_logprob(logits, token)) })
    }

    /// Downloads every row in `pending` with one drain and selects each on the host.
    fn host_rows(&mut self, logits: &DeviceLogits, rows: &[RowSelect], pending: &[usize],
        out: &mut [Option<RowResult>]) -> Result<()> {
        if pending.is_empty() { return Ok(()); }
        self.counts[2] += pending.len() as u64;
        let spans: Vec<_> = pending.iter().map(|&i| (i, 1)).collect();
        let host = logits.download(self.library, &spans)?;
        for (k, &i) in pending.iter().enumerate() {
            out[i] = Some(Self::host_select(&host[k * self.vocab..][..self.vocab], &rows[i]));
        }
        Ok(())
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
            self.wait_landed(logits.stream)?;
        }
        self.counts[0] += n as u64;
        let landing = self.landing.bytes();
        let word = |offset: usize| u32::from_le_bytes(landing[offset..offset + 4].try_into().unwrap());
        let mut out = Vec::with_capacity(n);
        let mut pending = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if word(4 * self.capacity + 4 * i) != 0 {
                // A non-finite logit: the host reports exactly what it would have.
                pending.push(i);
                out.push(None);
                continue;
            }
            let logprob = row.logprob.then(|| f32::from_bits(word(8 * self.capacity + 4 * i)));
            out.push(Some(Ok(Selected { token: word(4 * i), logprob })));
        }
        self.host_rows(logits, rows, &pending, &mut out)?;
        Ok(out.into_iter().map(|row| row.expect("every row selected")).collect())
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
        // SAFETY: the sampler's copies are queued on this stream, the
        // current device's (the sampler just launched on it).
        unsafe { self.wait_landed(logits.stream)? };
        let sampled = self.wave.as_mut().context("sampler")?.output(buffer, n)?;
        self.counts[1] += n as u64;
        let mut out = Vec::with_capacity(n);
        let mut pending = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let status = sampled.status[i];
            let device = status == cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK && routes[i] != Route::Host;
            if !device || row.logprob {
                // Host routes, device statuses (the host reports the same
                // error, or selects what the device could not) and logprobs.
                pending.push(i);
                out.push(None);
                continue;
            }
            out.push(Some(Ok(Selected { token: sampled.ids[i], logprob: None })));
        }
        self.host_rows(logits, rows, &pending, &mut out)?;
        Ok(out.into_iter().map(|row| row.expect("every row selected")).collect())
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
        let (ids, out) = (DeviceAllocation::new(library, sample.len() * 4, Tag::workspace("shared/embedding-check"))?,
            DeviceAllocation::new(library, sample.len() * 4 * row, Tag::workspace("shared/embedding-check"))?);
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
    let buffer = DeviceAllocation::new(library, rows * vocab * 4, Tag::workspace("shared/sampler-check"))?;
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
    fn row_fp8_copies_gather_once_then_expand_on_device() -> Result<()> {
        use std::cell::RefCell;
        for rows in [1, 3, 64] {
            for copies in [1, 4] {
                let hidden = 7;
                let width = hidden * 2;
                let output = RefCell::new(vec![0u8; rows * copies * width]);
                let scratch = RefCell::new(vec![0u8; rows * width]);
                let out = output.borrow_mut().as_mut_ptr().cast();
                let src = CuteafdDeviceBuffer { ptr: scratch.borrow_mut().as_mut_ptr().cast(),
                    bytes: rows * width, device_id: 2, flags: 0 };
                let mut events = Vec::new();
                gather_row_fp8_copies(rows, hidden, copies, out, (copies > 1).then_some(src), 2, |op| {
                    match op {
                        EmbeddingOp::Gather(ptr) => {
                            assert!(events.is_empty());
                            events.push("gather");
                            let mut table = if ptr == out { output.borrow_mut() } else {
                                assert_eq!(ptr, src.ptr); scratch.borrow_mut()
                            };
                            for (row, bytes) in table.chunks_exact_mut(width).enumerate() {
                                bytes.fill((row + 1) as u8);
                            }
                        }
                        EmbeddingOp::Copy { dst, src: actual_src, width: actual_width, dst_pitch } => {
                            assert_eq!(events.first(), Some(&"gather"));
                            assert_eq!(actual_src.ptr, src.ptr);
                            assert_eq!(actual_width, width);
                            assert_eq!(dst_pitch, copies * width);
                            let offset = dst.ptr as usize - out as usize;
                            assert_eq!(dst.bytes, rows * dst_pitch - offset);
                            for row in 0..rows {
                                let at = offset + row * dst_pitch;
                                output.borrow_mut()[at..at + width]
                                    .copy_from_slice(&scratch.borrow()[row * width..(row + 1) * width]);
                            }
                            events.push("copy");
                        }
                    }
                    Ok(())
                })?;
                assert_eq!(events.iter().filter(|&&event| event == "gather").count(), 1);
                assert_eq!(events.len(), if copies == 1 { 1 } else { 1 + copies });
                for (row, bytes) in output.borrow().chunks_exact(copies * width).enumerate() {
                    assert!(bytes.iter().all(|&byte| byte == (row + 1) as u8));
                }
                if copies > 1 {
                    for bad in [None, Some(CuteafdDeviceBuffer { bytes: src.bytes - 1, ..src }),
                        Some(CuteafdDeviceBuffer { device_id: 0, ..src }),
                        Some(CuteafdDeviceBuffer { ptr: out, ..src })] {
                        let mut launched = false;
                        assert!(gather_row_fp8_copies(rows, hidden, copies, out, bad, 2,
                            |_| { launched = true; Ok(()) }).is_err());
                        assert!(!launched);
                    }
                }
            }
        }
        let mut launches = 0;
        let mut output = [0u8; 32];
        let mut scratch = [0u8; 8];
        let src = CuteafdDeviceBuffer { ptr: scratch.as_mut_ptr().cast(), bytes: 8,
            device_id: 0, flags: 0 };
        assert!(gather_row_fp8_copies(1, 4, 4, output.as_mut_ptr().cast(), Some(src), 0, |_| {
            launches += 1; anyhow::bail!("injected gather failure")
        }).is_err());
        assert_eq!(launches, 1);
        Ok(())
    }

    #[test]
    fn row_fp8_peer_table_copies_use_the_producing_stream_device() -> Result<()> {
        // Every family's row-FP8 gather goes through this shared entry point.
        let source = include_str!("token_io.rs").split("#[cfg(test)]").next().unwrap();
        let gather = source.split("pub unsafe fn gather(").nth(1).unwrap()
            .split("let table = self.table.buffer();").next().unwrap();
        assert!(gather.contains("let RankDevice { device, stream } = producer;"));
        assert!(!gather.contains("cuda_stream_device"), "stream queries invalidate graph capture");
        assert!(!gather.contains("cuda_get_device"));
        assert!(!gather.contains("embedding.residency().device"));
        for (table_device, stream_device) in [(0, 1), (1, 0)] {
            let mut output = [0u8; 64];
            let mut scratch = [0u8; 16];
            let src = CuteafdDeviceBuffer { ptr: scratch.as_mut_ptr().cast(), bytes: scratch.len(),
                device_id: stream_device, flags: 0 };
            let mut copies = 0;
            gather_row_fp8_copies(2, 4, 4, output.as_mut_ptr().cast(), Some(src), stream_device, |op| {
                match op {
                    EmbeddingOp::Gather(ptr) => assert_eq!(ptr, src.ptr),
                    EmbeddingOp::Copy { dst, src, .. } => {
                        assert_eq!(dst.device_id, stream_device);
                        assert_eq!(src.device_id, stream_device);
                        assert_ne!(dst.device_id, table_device);
                        copies += 1;
                    }
                }
                Ok(())
            })?;
            assert_eq!(copies, 4);
            assert!(gather_row_fp8_copies(2, 4, 4, output.as_mut_ptr().cast(),
                Some(CuteafdDeviceBuffer { device_id: table_device, ..src }), stream_device,
                |_| panic!("peer-device scratch must fail before launch")).is_err());
        }
        Ok(())
    }

    #[test]
    fn selection_and_download_follow_the_producing_stream_device() {
        let source = include_str!("token_io.rs").split("#[cfg(test)]").next().unwrap();
        let download = source.split("fn download(").nth(1).unwrap().split("/// One row to select").next().unwrap();
        assert!(download.find("cuda_stream_device(self.stream)").unwrap() < download.find("copy_d2h_async(").unwrap());
        assert!(!download.contains("cuda_get_device()"));
        let select = source.split("pub fn select(").nth(1).unwrap().split("fn bind_device(").next().unwrap();
        assert!(select.find("cuda_stream_device(logits.stream)").unwrap() < select.find("self.greedy(").unwrap());
        assert!(select.find("self.bind_device(owner)?").unwrap() < select.find("self.sampled(").unwrap());
        let memory = include_str!("memory.rs");
        let wait = memory.split("pub(crate) unsafe fn wait(").nth(1).unwrap().split("impl Drop for BlockingWait").next().unwrap();
        assert!(wait.find("cuda_stream_device(stream)").unwrap() < wait.find("cuda_event_create_blocking()").unwrap());
        assert!(!wait.contains("cuda_get_device()"));
    }

    #[test]
    fn selector_device_check_names_the_rejected_operand() {
        for device in [0, 1] { check_selector_device(device, device).unwrap(); }
        let error = check_selector_device(0, 1).unwrap_err().to_string();
        assert!(error.contains("output operand on device 0") && error.contains("logits stream on device 1"), "{error}");
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn embedding_recording_defers_payload_io_and_replays_mount_bytes() -> Result<()> {
        use crate::shared::experts::rtx::dense;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        std::fs::write(fixture.directory().join("events"), [])?;
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let path = dir.path().join("embedding.bin");
        // An empty shard can be opened, but any eager payload read must fail.
        std::fs::write(&path, [])?;
        let source = EmbedSource { path: path.clone(), offset: 7, vocab: 5, hidden: 3,
            snapshot: None, tensor_name: "model.embed_tokens.weight".into() };
        let payload = (0..source.bytes()).map(|i| (i * 29 + 11) as u8).collect::<Vec<_>>();
        let before = fixture.events()?.matches('H').count();
        let ((pushed, value), records) = dense::recording(|| {
            TokenEmbedding::load(&library, source.clone(), EmbedPlacement::Gpu, || {
                assert_eq!(fixture.events()?.matches('H').count(), before);
                Ok(17)
            })
        })?;
        assert_eq!(value, 17);
        assert_eq!(records.summary(), (payload.len(), 1, 0, 0, 0, 0));
        assert_eq!(records.operands[0].buffer.ptr, pushed.table.buffer().ptr);
        let rect = &records.operands[0].lands[0].0;
        assert_eq!((rect.base, rect.rows.clone(), rect.cols.clone()), (7, 0..5, 0..6));
        assert_eq!(fixture.events()?.matches('H').count(), before);
        std::fs::write(&path, [vec![0xcc; 7], payload.clone()].concat())?;
        dense::fill_records_by_mount(&library, &records, |_| std::ptr::null_mut())?;
        let (mounted, ()) = TokenEmbedding::load(&library, source.clone(), EmbedPlacement::Gpu, || Ok(()))?;
        assert_eq!(pushed.verify_resident()?, payload.len());
        assert_eq!(mounted.verify_resident()?, payload.len());
        let read = |table: &TokenEmbedding<'_>| -> Result<Vec<u8>> {
            let mut bytes = vec![0; payload.len()];
            library.copy_d2h(&mut bytes, table.table.buffer())?;
            Ok(bytes)
        };
        assert_eq!(read(&pushed)?, payload);
        assert_eq!(read(&mounted)?, payload);
        let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(
            crate::shared::memory::device::Device { library: &library, id: 0 },
            cuteafd_loader::serving_capacity::WeightArenaLayout::new("test/embedding",
                [("embedding.table".into(), payload.len())])?, crate::shared::memory::TEST_ARENA)?;
        let (packed, ()) = TokenEmbedding::load_with(&library, source.clone(), EmbedPlacement::Gpu,
            || Ok(()), |name, bytes| arena.view(name, bytes))?;
        let ((recorded, ()), packed_records) = dense::recording(||
            TokenEmbedding::load_with(&library, source.clone(), EmbedPlacement::Gpu,
                || Ok(()), |name, bytes| arena.view(name, bytes)))?;
        assert_eq!(packed_records.operands[0].buffer.ptr, recorded.table.buffer().ptr);
        dense::fill_records_by_mount(&library, &packed_records, |_| std::ptr::null_mut())?;
        drop(arena);
        assert_eq!(read(&packed)?, payload);
        assert_eq!(read(&recorded)?, payload);
        let physical = [0];
        let receipts = packed.allocation_receipts(
            cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?, "embedding")?;
        assert_eq!(receipts.len(), 2);
        assert!(receipts.iter().all(|receipt| receipt.allocation.lifetime
            == cuteafd_loader::placement::planner::Lifetime::RESIDENT));
        assert_eq!(receipts[0].allocation.id, "test/embedding");
        assert_eq!(receipts[1].allocation.view.as_ref().context("embedding view")?.backing, "test/embedding");
        let error = packed.table_receipt(
            cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?, "embedding", "table")
            .err().context("singular receipt must reject arena views")?;
        assert!(error.to_string().contains("allocation_receipts"));
        // Mapped-host embeddings remain immediate and never enter the GPU group.
        let ((host, ()), host_records) = dense::recording(||
            TokenEmbedding::load(&library, source.clone(), EmbedPlacement::Host, || Ok(())))?;
        assert_eq!(host_records.summary(), (0, 0, 0, 0, 0, 0));
        assert_eq!(host.device_bytes(), 0);
        assert_eq!(host.host_rows(&[4, 0])?, [payload[24..30].to_vec(), payload[..6].to_vec()].concat());
        // Flash combines a caller-thread embedding recorder with an explicit
        // model sink on the entry GPU. Neither source may be read eagerly.
        fixture.configure_memory(&library, 1, 1 << 30)?;
        let devices = [0, 1].map(|id| crate::shared::memory::device::Device { library: &library, id });
        std::fs::write(&path, [])?;
        let load = dense::DenseLoad::push(&library);
        let ((peer_embedding, model), mut combined) = dense::recording(|| devices[1].run(||
            TokenEmbedding::load(&library, source.clone(), EmbedPlacement::Gpu, || devices[0].run(|| {
                let model = DeviceAllocation::new(&library, source.bytes(), crate::shared::memory::TEST)?;
                load.operand(dense::Operand { device: 0, buffer: model.buffer, bytes: source.bytes(),
                    lands: vec![(dense::Rect { name: "model.weight".into(), path: path.clone(), base: 7,
                        row_bytes: source.bytes(), rows: 0..1, cols: 0..source.bytes() }, 0)] })?;
                Ok(model)
            }))))?;
        assert_eq!(combined.operands.len(), 1);
        combined.extend(load.sink.as_ref().context("model sink")?.take_all());
        assert_eq!(combined.summary(), (2 * payload.len(), 2, 0, 0, 0, 0));
        assert_eq!(combined.operands.iter().map(|operand| operand.device).collect::<Vec<_>>(), [1, 0]);
        assert_eq!(peer_embedding.owner().id, 1);
        assert_eq!(model.buffer.device_id, 0);
        std::fs::write(&path, [vec![0xcc; 7], payload.clone()].concat())?;
        dense::fill_records_by_mount(&library, &combined, |_| std::ptr::null_mut())?;
        devices[1].run(|| -> Result<()> { assert_eq!(read(&peer_embedding)?, payload); Ok(()) })?;
        devices[0].run(|| -> Result<()> {
            let mut actual = vec![0; payload.len()];
            library.copy_d2h(&mut actual, model.buffer)?;
            assert_eq!(actual, payload);
            Ok(())
        })?;
        Ok(())
    }

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

#[cfg(test)]
mod row_fp8_rule_tests {
    #[test]
    fn e4m3_decode_and_bf16_rounding() {
        assert_eq!(super::e4m3_to_f32(0x38), 1.0);
        assert_eq!(super::e4m3_to_f32(0xb8), -1.0);
        assert_eq!(super::e4m3_to_f32(0x7e), 448.0);
        assert_eq!(super::e4m3_to_f32(0x01), 2f32.powi(-9));
        assert!(super::e4m3_to_f32(0x7f).is_nan());
        assert_eq!(super::bf16_rne(1.0), 0x3f80);
        // Ties to even: 1 + 2^-8 sits halfway between 0x3f80 and 0x3f81.
        assert_eq!(super::bf16_rne(1.0 + 2f32.powi(-8)), 0x3f80);
        assert_eq!(super::bf16_rne(1.0 + 3.0 * 2f32.powi(-8)), 0x3f82);
    }
}
