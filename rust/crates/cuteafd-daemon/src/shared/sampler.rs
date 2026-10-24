//! The GPU target sampler (`native/shared/cuda/sampling_gpu.cu`) as a reusable
//! wave: per-row parameter blocks, the packed grammar-mask arena and every
//! scratch arena the K1..K5 kernels read, sized for one vocabulary. First
//! written for DeepSeek V4.1's target head; every family's device token
//! selection launches through it.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdV41SamplerRow, NativeLibrary};
use std::ffi::c_void;

/// Chunk-6 radix histogram arena, allocated once now so a later chunk needs no
/// new allocation. 2048 buckets per row (design §11.1).
pub(crate) const SAMPLING_HISTOGRAM_BUCKETS: usize = 2048;

/// The largest mask arena the wave can address (`mask_row` is a `u32`).
const SAMPLING_MAX_MASK_ROWS: usize = 128;

/// Per-row stride of the chunk-3a rank-ordered retained-id arena, in ids
/// (design §4.5/§11.1: "who allocates them and how the §11.1 top-k region is
/// carved into the per-row `rank_order_capacity` stride (plus the
/// k-beyond-capacity fallback) is a chunk-4 decision").
///
/// **Chunk-4a decision.** K5's inclusive-prefix table holds one f32 weight per
/// rank in `__shared__` storage, so it cannot serve a retained list wider than
/// `kBlock` (256) at all: it reports `INTERNAL` for `top_k >= 257` regardless of
/// the arena. The arena is therefore sized to exactly that supported limit
/// (`capacity 80 x 256 x 4 B = 80 KiB`, plus 160 KiB u64 staging), and every row
/// with `top_k >= SAMPLING_MAX_RETAINED` is routed to the CPU sampler
/// **before** the launch rather than materializing a list K5 would refuse. The
/// alternative — sizing the arena at the full vocabulary — would cost 41 MiB at
/// capacity 80 to serve a configuration the kernel still rejects.
pub(crate) const SAMPLING_MAX_RETAINED: u32 = 256;

/// Per-row work for one target-sampler launch.
///
/// `greedy` is the *host* resolution of `temperature < 1e-5 || top_k == 1`.
/// The kernel re-derives it anyway, so this is only what tells the host which
/// rows still need the CPU path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TargetSamplingRowRequest {
    pub(crate) row: CuteafdV41SamplerRow,
    pub(crate) greedy: bool,
}

/// Device-selected rows for one target wave, in batch-row order.
///
/// `scores` is the raw maximum logit and is only meaningful where the row was
/// greedy; `status_detail` has already had the device's `0xFFFFFFFF`
/// "no detail" sentinel normalized to 0 (approved design deviation).
pub(crate) struct SampledTargetRows {
    pub(crate) ids: Vec<u32>,
    pub(crate) scores: Vec<f32>,
    pub(crate) status: Vec<u32>,
    pub(crate) status_detail: Vec<u32>,
    /// The head's vocabulary-projection buffer these ids were selected from.
    /// Retained so a caller can download a specific row without re-running the
    /// pass; it is a view, not an owner.
    pub(crate) logits: CuteafdDeviceBuffer,
}

impl SampledTargetRows {
    pub(crate) fn rows(&self) -> usize {
        self.ids.len()
    }
}

/// The device's "no detail" sentinel normalized to 0, matching the ABI note in
/// `sampling_gpu.h` (approved design deviation).
pub(crate) fn normalize_status_detail(value: u32) -> u32 {
    if value == cuteafd_ffi::CUTEAFD_V41_SAMPLER_NO_DETAIL {
        0
    } else {
        value
    }
}

/// Reusable per-projection device storage for the v4.1 GPU target-sampler.
///
/// Allocated once with the [`TargetHeadWave`] and freed with it: the zero-
/// allocation contract of design §11.1. Everything is stream-ordered on the
/// head stream, so no new host synchronization is introduced — the sampler's
/// small D2H is enqueued before the drain the scheduler already performs.
pub(crate) struct TargetSamplingWave<'a> {
    /// The native library that owns the stream and every buffer below.
    pub(crate) library: &'a NativeLibrary,
    /// The kernel entry points. Production is [`SamplerStages::native`]; the
    /// daemon tests substitute a recording double so the launch *sequence*
    /// (which stages, for which rows, with which parameter blocks) can be
    /// pinned without a GPU.
    pub(crate) stages: SamplerStages,
    pub(crate) capacity: usize,
    /// Logits per row (the mask, bitmap and histogram arenas are sized from it).
    pub(crate) vocab: usize,
    pub(crate) mask_words: usize,
    /// 64-byte per-row parameter blocks. `Pinned` holds the staging copy.
    pub(crate) param_device: DeviceAllocation<'a>,
    pub(crate) param_pinned: HostAllocation<'a>,
    /// Packed `capacity * mask_words` mask arena plus its staging copy.
    pub(crate) mask_device: DeviceAllocation<'a>,
    pub(crate) mask_pinned: HostAllocation<'a>,
    /// Per-row K1 reductions.
    pub(crate) scratch: DeviceAllocation<'a>,
    /// Top-k membership bitmap arena (chunk 3, allocated now).
    pub(crate) bitmap: DeviceAllocation<'a>,
    /// Chunk-6 radix histogram arena (allocated now).
    pub(crate) histogram: DeviceAllocation<'a>,
    pub(crate) ids_device: DeviceAllocation<'a>,
    pub(crate) status_device: DeviceAllocation<'a>,
    pub(crate) detail_device: DeviceAllocation<'a>,
    pub(crate) scores_device: DeviceAllocation<'a>,
    pub(crate) total_device: DeviceAllocation<'a>,
    pub(crate) nucleus_device: DeviceAllocation<'a>,
    /// Chunk-3a rank-ordered retained ids, `capacity * rank_order_capacity` u32,
    /// and its matching u64 staging (`rank_order_scratch`). Both are written
    /// only by K3/K4 and read only by K5 (design §4.5).
    pub(crate) rank_order_ids: DeviceAllocation<'a>,
    pub(crate) rank_order_scratch: DeviceAllocation<'a>,
    /// K3/K4's per-row `out_retained_count` (and `out_pivot_passes`, diagnostic).
    /// K5 consumes `rank_retained_count` by block row.
    pub(crate) retained_count_device: DeviceAllocation<'a>,
    pub(crate) pivot_passes_device: DeviceAllocation<'a>,
    pub(crate) ids_pinned: HostAllocation<'a>,
    pub(crate) status_pinned: HostAllocation<'a>,
    pub(crate) detail_pinned: HostAllocation<'a>,
    pub(crate) scores_pinned: HostAllocation<'a>,
    /// Pinned staging for the K3/K4 retained counts K5 needs on device.
    pub(crate) retained_count_staging: HostAllocation<'a>,
    /// The parameter blocks the last upload staged, so the launch can pass them
    /// by value to the FFI wrapper without re-borrowing the pinned buffer.
    pub(crate) params: Vec<CuteafdV41SamplerRow>,
    /// Rows of the arenas that the last upload actually filled.
    pub(crate) param_rows: usize,
    pub(crate) mask_rows: usize,
    /// Which stages the last launch enqueued (the routing record).
    pub(crate) stage_runs: SamplerStageRuns,
    /// The per-row status K1/K5 published (raw, before `SampledTargetRows`
    /// normalization) and the retained counts K3/K4 wrote. Retained for the
    /// tests that pin the routing contract; production reads the same values
    /// through [`SampledTargetRows`].
    pub(crate) last_status: Vec<u32>,
    pub(crate) last_retained: Vec<u32>,
}

/// The three device-sampler stages one sampled launch enqueues, as a seam.
///
/// Production is [`NativeSamplerStages`], which forwards to the C ABI wrappers
/// exactly as the pre-chunk-4a `launch()` did. The trait exists so the launch
/// *sequence* and the per-row routing can be unit-tested without a GPU: a
/// recording implementation records which stages ran, for which rows, with
/// which parameter blocks, and can return ids a CPU sampler produced. That is
/// what lets a daemon test assert "the ordered path ran for exactly the ordered
/// rows" without pretending a kernel ran.
///
/// It is deliberately narrow: one method per C entry point, in the same order
/// the production stream enqueues them, and no host synchronization. Memory
/// staging is not abstracted: it touches no kernel arithmetic, and the
/// recording double has no device memory to stage into.
pub(crate) trait SamplerStageLauncher {
    /// K1 + K2 on `stream`: `cuteafd_cuda_v41_target_sample_async`.
    #[allow(clippy::too_many_arguments)]
    fn launch_prepare(
        &self,
        library: &NativeLibrary,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        masks: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        out_status_detail: CuteafdDeviceBuffer,
        out_scores: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    /// K3 + K4 on `stream`: `cuteafd_cuda_v41_topk_select_async`.
    #[allow(clippy::too_many_arguments)]
    fn launch_topk_select(
        &self,
        library: &NativeLibrary,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        masks: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: CuteafdDeviceBuffer,
        rank_order_scratch: CuteafdDeviceBuffer,
        rank_order_capacity: usize,
        out_retained_count: CuteafdDeviceBuffer,
        out_pivot_passes: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;

    /// K5 on `stream`: `cuteafd_cuda_v41_nucleus_async`.
    #[allow(clippy::too_many_arguments)]
    fn launch_nucleus(
        &self,
        library: &NativeLibrary,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        masks: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: CuteafdDeviceBuffer,
        rank_order_capacity: usize,
        rank_retained_count: CuteafdDeviceBuffer,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<()>;
}

/// The three entry points over a real [`NativeLibrary`]. Zero-sized: the
/// library arrives per call, so a wave can hold this by shared reference
/// regardless of the wave's lifetime.
pub(crate) struct NativeSamplerStages;

impl SamplerStageLauncher for NativeSamplerStages {
    fn launch_prepare(
        &self,
        library: &NativeLibrary,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        masks: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        out_status_detail: CuteafdDeviceBuffer,
        out_scores: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            library.cuda_v41_target_sample_async(
                logits, rows, vocab, logits_stride, params, params_device, masks,
                mask_words_per_row, out_indices, out_status, out_status_detail,
                out_scores, None, None, scratch, stream,
            )
        }
    }
    fn launch_topk_select(
        &self,
        library: &NativeLibrary,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        masks: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: CuteafdDeviceBuffer,
        rank_order_scratch: CuteafdDeviceBuffer,
        rank_order_capacity: usize,
        out_retained_count: CuteafdDeviceBuffer,
        out_pivot_passes: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            library.cuda_v41_topk_select_async(
                logits, rows, vocab, logits_stride, params, params_device, masks,
                mask_words_per_row, Some(rank_order_ids), Some(rank_order_scratch),
                rank_order_capacity, out_retained_count, out_pivot_passes, scratch, stream,
            )
        }
    }
    fn launch_nucleus(
        &self,
        library: &NativeLibrary,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        vocab: usize,
        logits_stride: usize,
        params: &[CuteafdV41SamplerRow],
        params_device: CuteafdDeviceBuffer,
        masks: Option<CuteafdDeviceBuffer>,
        mask_words_per_row: usize,
        rank_order_ids: CuteafdDeviceBuffer,
        rank_order_capacity: usize,
        rank_retained_count: CuteafdDeviceBuffer,
        out_indices: CuteafdDeviceBuffer,
        out_status: CuteafdDeviceBuffer,
        scratch: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            library.cuda_v41_nucleus_async(
                logits, rows, vocab, logits_stride, params, params_device, masks,
                mask_words_per_row,
                if rank_order_capacity == 0 { None } else { Some(rank_order_ids) },
                rank_order_capacity, rank_retained_count, out_indices, out_status, None,
                None, scratch, stream,
            )
        }
    }
}

/// The borrowed pair a [`TargetSamplingWave`] launches through.
#[derive(Clone, Copy)]
pub(crate) struct SamplerStages {
    launcher: &'static dyn SamplerStageLauncher,
}

impl SamplerStages {
    /// The production entry points.
    pub(crate) const fn native() -> Self {
        Self { launcher: &NativeSamplerStages }
    }
    /// Bind a substitute launcher (the recording double in the daemon tests).
    #[cfg(test)]
    pub(crate) const fn with(launcher: &'static dyn SamplerStageLauncher) -> Self {
        Self { launcher }
    }
    /// The launcher behind the seam, for the recording test's assertions.
    #[cfg(test)]
    pub(crate) fn launcher_id(&self) -> usize {
        self.launcher as *const dyn SamplerStageLauncher as *const () as usize
    }
    fn launcher(&self) -> &dyn SamplerStageLauncher {
        self.launcher
    }
}

/// Which stages one sampled launch enqueued, and the rows it enqueued them for.
///
/// Chunk 4a is the first caller that needs to reason about *per-row* kernel
/// eligibility: the single `cuteafd_cuda_v41_target_sample[_async]` entry point
/// already dispatches K1 (every row) and K2 (fast-path rows only), but K5's
/// ordered path is two extra entry points that must be enqueued whenever **any**
/// row is ordered. This is the observable record of that decision, used by the
/// route table test and by the reporting/timing instrumentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SamplerStageRuns {
    /// K1 + K2 ran (always; K1 is the mask/finiteness/status stage every row
    /// needs, including a row that is only there to be counted).
    pub(crate) prepare: bool,
    /// K3 + K4 ran, with this `rank_order_capacity`.
    pub(crate) topk_select: Option<usize>,
    /// K5 ran.
    pub(crate) nucleus: bool,
}

/// The `(mask buffer, mask_words_per_row)` pair for one sampled launch.
///
/// The two must travel together: when no row needs a mask, `upload()` stages
/// nothing and leaves `mask_rows == 0`, and both the FFI validator
/// (`mask_words_per_row must be 0 when no mask buffer is supplied`) and the
/// kernel (`MASK_WIDTH`) reject a non-zero width with a null arena. The wave's
/// own `mask_words` (4040) is always non-zero, so passing it unconditionally
/// made every all-unconstrained round — a traced all-greedy round, or a
/// constrained member whose grammar needs no mask at any row — fail its lane.
pub(crate) fn mask_launch_arguments(mask_rows: usize, mask_device: CuteafdDeviceBuffer, mask_words: usize,
) -> (Option<CuteafdDeviceBuffer>, usize) {
    if mask_rows == 0 {
        (None, 0)
    } else {
        (Some(mask_device), mask_words)
    }
}

impl<'a> TargetSamplingWave<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, capacity: usize, vocab: usize) -> Result<Self> {
        Self::allocate(library, SamplerStages::native(), capacity, vocab)
    }

    fn mask_arena_bytes(capacity: usize, vocab: usize) -> usize {
        capacity * vocab.div_ceil(32) * 4
    }
    /// Per-row bytes of the chunk-3a rank-order arena: `rank_order_capacity` u32
    /// ids plus the same count of u64 staging entries, which K3/K4 require to be
    /// supplied together (the FFI validator is all-or-nothing).
    pub(crate) fn rank_order_bytes(capacity: usize) -> usize {
        capacity
            * SAMPLING_MAX_RETAINED as usize
            * (std::mem::size_of::<u32>() + std::mem::size_of::<u64>())
    }
    /// Device bytes the sampler adds to the head wave budget (design §11.1).
    pub(crate) fn device_bytes(capacity: usize, vocab: usize) -> usize {
        cuteafd_core::target_sampler_device_bytes(capacity, vocab)
    }
    /// Allocate every sampler arena against `library` and bind `stages`.
    ///
    /// Split out of [`Self::new`] so a test can pair real device buffers with a
    /// recording launcher; the allocation sizes are the ones
    /// [`Self::device_bytes`] charges to the head wave.
    pub(crate) fn allocate(library: &'a NativeLibrary, stages: SamplerStages, capacity: usize, vocab: usize)
        -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("sampler");
        ensure!(
            (1..=SAMPLING_MAX_MASK_ROWS).contains(&capacity),
            "sampling capacity must be 1..{SAMPLING_MAX_MASK_ROWS}"
        );
        let retained = SAMPLING_MAX_RETAINED as usize;
        Ok(Self {
            library,
            stages,
            capacity,
            vocab,
            mask_words: vocab.div_ceil(32),
            param_device: DeviceAllocation::new(
                library,
                capacity * cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES,
            )?,
            param_pinned: HostAllocation::new(
                library,
                capacity * cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES,
            )?,
            mask_device: DeviceAllocation::new(library, Self::mask_arena_bytes(capacity, vocab))?,
            mask_pinned: HostAllocation::new(library, Self::mask_arena_bytes(capacity, vocab))?,
            scratch: DeviceAllocation::new(
                library,
                capacity * cuteafd_ffi::CUTEAFD_V41_SAMPLER_SCRATCH_BYTES,
            )?,
            bitmap: DeviceAllocation::new(library, Self::mask_arena_bytes(capacity, vocab))?,
            histogram: DeviceAllocation::new(
                library,
                capacity * SAMPLING_HISTOGRAM_BUCKETS * 4,
            )?,
            ids_device: DeviceAllocation::new(library, capacity * 4)?,
            status_device: DeviceAllocation::new(library, capacity * 4)?,
            detail_device: DeviceAllocation::new(library, capacity * 4)?,
            scores_device: DeviceAllocation::new(library, capacity * 4)?,
            total_device: DeviceAllocation::new(library, capacity * 4)?,
            nucleus_device: DeviceAllocation::new(library, capacity * 4)?,
            rank_order_ids: DeviceAllocation::new(library, capacity * retained * 4)?,
            rank_order_scratch: DeviceAllocation::new(library, capacity * retained * 8)?,
            retained_count_device: DeviceAllocation::new(library, capacity * 4)?,
            pivot_passes_device: DeviceAllocation::new(library, capacity * 4)?,
            ids_pinned: HostAllocation::new(library, capacity * 4)?,
            status_pinned: HostAllocation::new(library, capacity * 4)?,
            detail_pinned: HostAllocation::new(library, capacity * 4)?,
            scores_pinned: HostAllocation::new(library, capacity * 4)?,
            retained_count_staging: HostAllocation::new(library, capacity * 4)?,
            params: Vec::new(),
            param_rows: 0,
            mask_rows: 0,
            stage_runs: SamplerStageRuns { prepare: false, topk_select: None, nucleus: false },
            last_status: Vec::new(),
            last_retained: Vec::new(),
        })
    }

    /// Fill the pinned parameter staging from the per-row requests.
    ///
    /// `mask_staging` is the host's packed `rows * mask_words` arena; a row
    /// whose `flags` has `NO_MASK` set is unconstrained and its arena slice is
    /// not read. Only the rows that are actually masked are copied to the
    /// device.
    ///
    /// The §5.3 remainder rule is applied here, immediately before the upload:
    /// every bit `>= vocab` of a masked row's final word is cleared, so a
    /// whole-word reader cannot mistake an out-of-range bit for a real token.
    pub(crate) fn upload(
        &mut self,
        requests: &[TargetSamplingRowRequest],
        mask_staging: Option<&[u32]>,
        mask_words: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let rows = requests.len();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "sampling rows exceed the wave capacity"
        );
        ensure!(
            mask_words == self.mask_words,
            "sampling mask width differs from the arena"
        );
        if let Some(mask_staging) = mask_staging {
            ensure!(
                mask_staging.len() == rows * self.mask_words,
                "sampling mask staging extent differs"
            );
        }
        {
            let staging = self.param_pinned.bytes_mut();
            for (slot, request) in requests.iter().enumerate() {
                ensure!(
                    request.row.output_row as usize == slot,
                    "sampling output_row must equal the batch slot"
                );
                let offset = slot * cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES;
                staging[offset..offset + cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES]
                    .copy_from_slice(bytemuck_bytes(&request.row));
            }
        }
        let mut mask_rows = 0;
        if let Some(mask_staging) = mask_staging {
            let staging = self.mask_pinned.bytes_mut();
            for (slot, request) in requests.iter().enumerate() {
                if request.row.flags & cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_NO_MASK != 0 {
                    continue;
                }
                let target_row = request.row.mask_row as usize;
                ensure!(
                    target_row < rows,
                    "sampling mask_row is outside the uploaded mask rows"
                );
                let source = slot * self.mask_words;
                let target = target_row * self.mask_words;
                // Copy straight into the pinned row, then clear its tail bits in place.
                let pinned = &mut staging[target * 4..(target + self.mask_words) * 4];
                pinned.copy_from_slice(bytemuck_slice(&mask_staging[source..source + self.mask_words]));
                let remainder = self.vocab % 32;
                if remainder != 0 {
                    let last = &mut pinned[((self.vocab - 1) / 32) * 4..][..4];
                    let word = u32::from_ne_bytes(last.try_into().expect("four mask bytes"))
                        & ((1u32 << remainder) - 1);
                    last.copy_from_slice(&word.to_ne_bytes());
                }
                mask_rows = mask_rows.max(target_row + 1);
            }
        }
        let param_bytes = rows * cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES;
        unsafe {
            self.library.copy_host_buffer_h2d_async(
                self.param_device.buffer,
                self.param_pinned.buffer,
                param_bytes,
                stream,
            )?;
            if mask_rows > 0 {
                self.library.copy_host_buffer_h2d_async(
                    self.mask_device.buffer,
                    self.mask_pinned.buffer,
                    mask_rows * self.mask_words * 4,
                    stream,
                )?;
            }
        }
        self.params = requests.iter().map(|request| request.row).collect();
        self.param_rows = rows;
        self.mask_rows = mask_rows;
        Ok(())
    }

    /// Upload the parameter blocks and (only) the masked rows' arena, then
    /// enqueue the per-row kernel sequence on the head stream.
    ///
    /// **The sequence is the whole per-row routing contract.** One entry point
    /// serves many row classes, so a caller cannot "call the kernel for the
    /// ordered rows": K1/K2 are one entry point over every row and K3/K4/K5 are
    /// two more, each of which is a per-row no-op for the rows its contract does
    /// not cover (`sampling_gpu.cu`: `k2_applicable`, the K3/K4 eligibility
    /// block, and K5's ordered-row class). The rules are:
    ///
    /// * K1 + K2 always run. K1 masks, checks finiteness and publishes the
    ///   per-row status every other stage consumes; K2 draws for exactly the
    ///   fast-path rows (`top_k == 0 && top_p >= 1.0`, `k2_applicable`), so a
    ///   `temperature 0.7 + min_p 0.05` row is complete after this entry point.
    /// * K3 + K4 + K5 run exactly when **at least one** row is ordered
    ///   (`top_k in 1..=256 && top_k < survivor_count`, or `top_k == 0` with
    ///   `top_p < 1.0` for K5's case 3). K3/K4 are no-ops for every other row
    ///   (`top_k == 0` is the disabled encoding and never enters K3) and K5
    ///   leaves a fast-path row untouched, so enqueuing them for a batch that
    ///   contains one ordered row costs the other rows a per-row eligibility
    ///   test, not a wrong token. Skipping them would leave the ordered row's
    ///   `out_indices` at the caller's sentinel.
    /// * `rank_order_capacity` is fixed at
    ///   [`SAMPLING_MAX_RETAINED`] (K5's `kBlock` limit), so the
    ///   validator's `capacity >= max top_k` holds for every servable row; a row
    ///   above the limit is routed to the CPU sampler by the caller **before**
    ///   this call rather than enqueued for a kernel that would report
    ///   `INTERNAL`.
    ///
    /// The small D2H copies stay exactly where chunk 1 put them: enqueued on the
    /// same stream before the caller's existing drain, so this adds no host
    /// synchronization (design §11.2). `rank_retained_count` is downloaded too
    /// because K5 consumes it as a device input while K3/K4 write it.
    pub(crate) fn launch(&mut self, logits: CuteafdDeviceBuffer, rows: usize, ordered_rows: bool,
        stream: *mut c_void,
    ) -> Result<SamplerStageRuns> {
        ensure!(
            rows == self.param_rows && logits.bytes == rows * self.vocab * 4,
            "sampling launch shape differs from the uploaded rows"
        );
        let (masks, mask_words_per_row) =
            mask_launch_arguments(self.mask_rows, self.mask_device.buffer, self.mask_words);
        // K1 reads the parameter block on device, so launch the buffer
        // `upload()` filled; the host copy is only the validator's input.
        let params = self.params.clone();
        self.stages.launcher().launch_prepare(
            self.library,
            logits,
            rows,
            self.vocab,
            self.vocab,
            &params,
            self.param_device.buffer,
            masks,
            mask_words_per_row,
            self.ids_device.buffer,
            self.status_device.buffer,
            self.detail_device.buffer,
            self.scores_device.buffer,
            self.scratch.buffer,
            stream,
        )?;
        let mut runs = SamplerStageRuns { prepare: true, topk_select: None, nucleus: false };
        if ordered_rows {
            let capacity = SAMPLING_MAX_RETAINED as usize;
            self.stages.launcher().launch_topk_select(
                self.library,
                logits,
                rows,
                self.vocab,
                self.vocab,
                &params,
                self.param_device.buffer,
                masks,
                mask_words_per_row,
                self.rank_order_ids.buffer,
                self.rank_order_scratch.buffer,
                capacity,
                self.retained_count_device.buffer,
                self.pivot_passes_device.buffer,
                self.scratch.buffer,
                stream,
            )?;
            runs.topk_select = Some(capacity);
            self.stages.launcher().launch_nucleus(
                self.library,
                logits,
                rows,
                self.vocab,
                self.vocab,
                &params,
                self.param_device.buffer,
                masks,
                mask_words_per_row,
                self.rank_order_ids.buffer,
                capacity,
                self.retained_count_device.buffer,
                self.ids_device.buffer,
                // K5's own per-row status channel: a K5-class row that cannot
                // produce a defined token writes INTERNAL here and still returns
                // OK from the entry point, so the caller must read this buffer
                // (design §20 item 1). Passing K1's buffer means the last writer
                // wins for a row both stages report on, which is what the caller
                // observes.
                self.status_device.buffer,
                self.scratch.buffer,
                stream,
            )?;
            runs.nucleus = true;
            unsafe {
                self.library.copy_d2h_host_buffer_async(
                    self.retained_count_staging.buffer,
                    self.retained_count_device.buffer,
                    rows * 4,
                    stream,
                )?;
            }
        }
        unsafe {
            self.library.copy_d2h_host_buffer_async(
                self.ids_pinned.buffer,
                self.ids_device.buffer,
                rows * 4,
                stream,
            )?;
            self.library.copy_d2h_host_buffer_async(
                self.status_pinned.buffer,
                self.status_device.buffer,
                rows * 4,
                stream,
            )?;
            self.library.copy_d2h_host_buffer_async(
                self.detail_pinned.buffer,
                self.detail_device.buffer,
                rows * 4,
                stream,
            )?;
            self.library.copy_d2h_host_buffer_async(
                self.scores_pinned.buffer,
                self.scores_device.buffer,
                rows * 4,
                stream,
            )?;
        }
        self.params = params;
        self.stage_runs = runs;
        Ok(runs)
    }

    pub(crate) fn output(&mut self, logits: CuteafdDeviceBuffer, rows: usize) -> Result<SampledTargetRows> {
        ensure!(
            rows == self.param_rows,
            "sampling output rows differ from the uploaded rows"
        );
        let words = |bytes: &[u8]| -> Vec<u32> {
            bytes[..rows * 4]
                .chunks_exact(4)
                .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
                .collect()
        };
        let status = words(self.status_pinned.bytes());
        let detail: Vec<u32> =
            words(self.detail_pinned.bytes()).into_iter().map(normalize_status_detail).collect();
        self.last_status = status.clone();
        self.last_retained = if self.stage_runs.topk_select.is_some() {
            words(self.retained_count_staging.bytes())
        } else {
            Vec::new()
        };
        Ok(SampledTargetRows {
            ids: words(self.ids_pinned.bytes()),
            scores: self.scores_pinned.bytes()[..rows * 4]
                .chunks_exact(4)
                .map(|word| f32::from_ne_bytes(word.try_into().unwrap()))
                .collect(),
            status,
            status_detail: detail,
            logits,
        })
    }

    /// Which stages the last launch enqueued (the per-row routing record).
    /// Test-only observability; production reads the effects.
    #[cfg(test)]
    pub(crate) fn last_stages(&self) -> SamplerStageRuns {
        self.stage_runs
    }
    /// K3/K4's per-row retained counts from the last launch, empty when the
    /// ordered path did not run. Test-only observability.
    #[cfg(test)]
    pub(crate) fn last_retained_counts(&self) -> &[u32] {
        &self.last_retained
    }
    /// K1/K5's per-row status from the last launch, normalized as
    /// [`SampledTargetRows::status`] is. Test-only observability.
    #[cfg(test)]
    pub(crate) fn last_status_codes(&self) -> &[u32] {
        &self.last_status
    }
}

/// Raw bytes of a plain-old-data value, for the pinned staging copy.
pub(crate) fn bytemuck_bytes<T: Copy>(value: &T) -> &[u8] {
    // Safety: `T` is `Copy` and `CuteafdV41SamplerRow` is `repr(C)` with no
    // padding holes that Rust would leave uninitialized (every field is written
    // by `Default`), so reading its object representation is defined.
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

pub(crate) fn bytemuck_slice<T: Copy>(values: &[T]) -> &[u8] {
    // Safety: as above; `T` is a plain integer type.
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

#[cfg(test)]
mod allocation_accounting_tests {
    use super::*;

    #[test]
    fn shared_device_bytes_match_every_sampler_arena() {
        for rows in [1, 7, 64, 128] {
            for vocab in [1usize, 31, 32, 33, 128896] {
                let expected = rows * (cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES
                    + cuteafd_ffi::CUTEAFD_V41_SAMPLER_SCRATCH_BYTES
                    + SAMPLING_HISTOGRAM_BUCKETS * std::mem::size_of::<u32>()
                    + 8 * std::mem::size_of::<u32>())
                    + 2 * TargetSamplingWave::mask_arena_bytes(rows, vocab)
                    + TargetSamplingWave::rank_order_bytes(rows);
                assert_eq!(TargetSamplingWave::device_bytes(rows, vocab), expected);
            }
        }
    }
}
