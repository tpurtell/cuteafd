//! The GLM 5.3 Flash vocabulary head: one resident representation, shared by
//! the target's logits and the DFlash drafter.
//!
//! `Bf16`: the checkpoint's `lm_head.weight`. `Fp8` (`--fp8-head`): E4M3 with
//! FP32 scales per output row and 128-wide K block (`[vocab, hidden/128]`),
//! quantized at load; no BF16 copy is kept, and every row count (decode,
//! verify, prefill tails, `--nll` full prefill, draft blocks) runs the
//! `glmf_head_fp8` program in [`FP8_HEAD_ROWS`]-row spans.
//! `Packed` (`--fp8-head packed`): the shared exact 12-bit copy ([`crate::shared::packed_head`]),
//! packed from the landed BF16 head once every operand has landed ([`GlmfHead::pack`]), which
//! releases the BF16 after the pack verifies bit for bit. Target logits (every row count) read it
//! exactly through the tensor-core GEMV in each lead step workspace's head workspace (the BF16
//! head's cuBLAS workspace, unused by this head); the drafter, which borrows the target's head,
//! reads its coarse plane (E4M3-like traffic and numerics, as V4.1's dSpark drafter does) with
//! scratch of its own ([`GlmfHead::bind_draft`]), and takes the FP8 head's selector prior.
//!
//! The target's logits through the BF16 head take the shared route
//! ([`crate::shared::target_head`], `--target-head`): BF16 tensor cores at every row count, or the
//! pedantic FP32 GEMM as a diagnostic. The FP8 and packed heads run their own GEMVs either way
//! (an explicit `tensor` with them is refused, `exact` is inert).
use crate::shared::memory::DeviceAllocation;
use crate::shared::packed_head::PackedHead;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar};
use cuteafd_ffi::NativeLibrary;
pub(crate) use cuteafd_loader::families::glm5_flash::resident::GlmfHeadFormat;
use std::ffi::c_void;
pub(crate) use crate::shared::target_head::TargetHead;

/// Rows one `glmf_head_fp8` launch takes (its GEMV's M tile).
pub(crate) const FP8_HEAD_ROWS: usize = 16;

/// The target head's mode: an explicit `--target-head`, else the shared setting
/// (`CUTEAFD_TARGET_HEAD`, tensor by default). Over the FP8 head it is inert (its own program).
pub(crate) fn resolve(requested: Option<TargetHead>) -> TargetHead {
    requested.unwrap_or_else(TargetHead::configured)
}

/// `(first row, rows)` spans of at most [`FP8_HEAD_ROWS`] covering `0..rows`.
pub(crate) fn fp8_head_spans(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows).step_by(FP8_HEAD_ROWS).map(move |first| (first, FP8_HEAD_ROWS.min(rows - first)))
}

pub(crate) enum GlmfHead<'a> {
    K6(crate::shared::dense_trellis::CoreWeights<'a>),
    Bf16(DeviceAllocation<'a>),
    /// Checkpoint row-FP8; the embedding and drafter retain this same owner.
    RowFp8(std::rc::Rc<crate::shared::token_io::native::NativeVocabulary<'a>>),
    BlockFp8 { values: DeviceAllocation<'a>, scales: DeviceAllocation<'a> },
    Fp8 { values: DeviceAllocation<'a>, scales: DeviceAllocation<'a> },
    /// The exact 12-bit copy of the BF16 head and, once a drafter binds it, the drafter's
    /// coarse-plane GEMV scratch (a workspace, not part of the head's resident bytes).
    Packed { head: PackedHead<'a>, draft_scratch: Option<DeviceAllocation<'a>> },
}

impl<'a> GlmfHead<'a> {
    pub(super) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        match self {
            Self::K6(owner) => {
                owner.payload.inventory(inventory, "head.trellis")?;
                owner.suh.inventory(inventory, "head.suh")?;
                owner.svh.inventory(inventory, "head.svh")?;
            }
            Self::Bf16(owner) => owner.inventory(inventory, "head")?,
            Self::Fp8 { values, scales } | Self::BlockFp8 { values, scales } => {
                values.inventory(inventory, "head.values")?;
                scales.inventory(inventory, "head.scales")?;
            }
            // Vocabulary is shared with TokenEmbedding, which visits both retained planes.
            Self::RowFp8(owner) => owner.head.inventory(inventory, "head")?,
            // The drafter's coarse-plane scratch is a workspace, not head weights.
            Self::Packed { head, .. } => head.inventory(inventory, "head")?,
        }
        Ok(())
    }

    /// The head's resident weight allocations.
    pub fn allocations(&self) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        match self {
            Self::K6(weight) => vec![weight.payload.buffer, weight.suh.buffer, weight.svh.buffer],
            Self::Bf16(weight) => vec![weight.buffer],
            Self::RowFp8(owner) => vec![owner.head.payload(), owner.head.scales()],
            Self::Fp8 { values, scales } | Self::BlockFp8 { values, scales } => vec![values.buffer, scales.buffer],
            Self::Packed { head, .. } => vec![head.buffer()],
        }
    }

    pub fn bytes(&self) -> usize {
        self.allocations().iter().map(|a| a.bytes).sum()
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::K6(_) => "K6",
            Self::RowFp8(_) => "row-FP8",
            Self::BlockFp8 { .. } => "block-FP8",
            head => head.format().map_or("?", GlmfHeadFormat::name),
        }
    }

    /// The `--fp8-head` form this head is (none for a native checkpoint's own head).
    pub fn format(&self) -> Option<GlmfHeadFormat> {
        match self {
            Self::Bf16(_) => Some(GlmfHeadFormat::Bf16),
            Self::Fp8 { .. } => Some(GlmfHeadFormat::Fp8),
            Self::Packed { .. } => Some(GlmfHeadFormat::Packed),
            Self::K6(_) | Self::RowFp8(_) | Self::BlockFp8 { .. } => None,
        }
    }

    /// The checkpoint BF16 head, when that is the resident representation.
    pub fn bf16(&self) -> Option<&DeviceAllocation<'a>> {
        match self {
            Self::Bf16(weight) => Some(weight),
            Self::Fp8 { .. } | Self::BlockFp8 { .. } | Self::K6(_) | Self::RowFp8(_) | Self::Packed { .. } => None,
        }
    }

    /// `--fp8-head packed`: packs this landed BF16 `[vocab, hidden]` head on the current device
    /// into its exact 12-bit copy, which verifies itself bit for bit, and releases the BF16 (as
    /// V4.1's packed vocabulary does). Run once every operand has landed (the mount, or the
    /// committed push group). A pack that overflows its exception table or fails verification
    /// fails the load: admission charged the packed bytes, not the BF16 ones. A failed drain
    /// retains the BF16 source (the pack's reads may not have retired) and quarantines the
    /// destination and native module until process teardown.
    /// `arena`: the target weight arena whose `packed12` view receives the copy (None: an
    /// allocation of its own, as in tests).
    pub fn pack(self, library: &'a NativeLibrary, vocab: usize, hidden: usize,
        arena: Option<&crate::shared::memory::weight_arena::DeviceWeightArena<'a>>) -> Result<Self> {
        let Self::Bf16(weight) = self else {
            anyhow::bail!("--fp8-head packed packs the BF16 head; the resident head is {}", self.name());
        };
        ensure!(weight.buffer.bytes == vocab * hidden * 2, "packed head geometry differs from BF16 [{vocab}, {hidden}]");
        // Target logits run the GEMV in the step workspaces' head workspace. This shape query also
        // fixes the native GEMV's SM count on this device before any graph capture.
        let target = library.packed_head_workspace(cuteafd_core::packed_head::PASS_ROWS as usize, hidden, vocab)?;
        ensure!(target <= cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE,
            "the exact 12-bit head's GEMV needs {target} bytes of scratch, more than the step workspaces' {}-byte \
             head workspace; serve --fp8-head true or false", cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE);
        let stream = library.cuda_stream_create()?;
        let device = crate::shared::memory::device::Device { library, id: library.cuda_get_device()? };
        let packed = match PackedHead::pack_with(library, weight.buffer.ptr, vocab, hidden, stream, |name, bytes|
            crate::shared::memory::weight_arena::weight_allocation(device, arena, name, bytes,
                cuteafd_core::memory_layout::ArenaTag::weights("glmf/target"))) {
            Err(error) if library.is_quarantined_after_failed_drain() => {
                // The pack could not prove its reads of the source retired.
                std::mem::forget(weight);
                return Err(error);
            }
            packed => packed,
        };
        // SAFETY: the pack drained the stream, which has no other users.
        unsafe { library.cuda_stream_destroy(stream)?; }
        drop(weight);
        let head = packed.context("--fp8-head packed: the exact 12-bit head did not verify against the BF16 head; \
            serve --fp8-head false (BF16) or true (FP8)")?;
        let (exceptions, row_exceptions) = head.exceptions();
        tracing::info!(vocab, hidden, bytes = head.bytes(), exceptions, row_exceptions, scratch_bytes = target,
            "GLM 5.3 Flash exact 12-bit head packed and verified; BF16 head released");
        Ok(Self::Packed { head, draft_scratch: None })
    }

    /// Allocates the drafter's coarse-plane GEMV scratch for draft steps of up to `rows` rows on
    /// the current device (a packed head only; admission charges one GEMV pass's scratch, which
    /// serves every row count).
    pub fn bind_draft(&mut self, library: &'a NativeLibrary, rows: usize) -> Result<()> {
        if let Self::Packed { head, draft_scratch } = self {
            ensure!(draft_scratch.is_none(), "packed head drafter scratch bound twice");
            *draft_scratch = Some(crate::shared::packed_head::scratch(library, rows, &[(head.k, head.n)],
                crate::shared::memory::Tag::drafter_workspace("glmf/draft-head-projection"))?);
        }
        Ok(())
    }

    /// FP32 logits `[rows, vocab]` of BF16 rows `x [rows, hidden]` through the packed head: the
    /// exact BF16 values (target logits) in `scratch`, or with `coarse` (the drafter) the coarse
    /// plane in the drafter's bound scratch. Errors on any other head.
    ///
    /// # Safety
    /// `x` and `logits` hold `rows` rows of `hidden` BF16 and `vocab` FP32 values; `scratch` (exact
    /// only) is live device memory; all stay live until `stream` reaches the launch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_packed(&self, library: &NativeLibrary, x: *const c_void, logits: *mut f32, rows: usize,
        coarse: bool, scratch: Option<cuteafd_ffi::CuteafdDeviceBuffer>, stream: *mut c_void) -> Result<()> {
        let Self::Packed { head, draft_scratch } = self else {
            anyhow::bail!("the GLMF head is {}; the packed head GEMV has no weight", self.name());
        };
        let scratch = if coarse { draft_scratch.as_ref().map(|s| s.buffer).context("packed head drafter scratch not bound")? }
            else { scratch.context("packed head target scratch missing")? };
        // SAFETY: the caller's contract; the planes are this head's live regions.
        unsafe { head.apply(library, x, logits.cast(), true, rows, coarse, scratch, stream) }
    }

    /// FP32 logits `[rows, vocab]` of BF16 rows `x [rows, hidden]` through the
    /// FP8 head. Row-FP8 uses 16-row GEMV spans; block-FP8 launches once,
    /// selecting GEMV through 16 rows and W8A16 GEMM above. Errors on a BF16 head.
    ///
    /// # Safety
    /// `x` and `logits` must hold `rows` rows of `hidden` BF16 and `vocab` FP32
    /// values, live until `stream` reaches these launches.
    pub unsafe fn launch_fp8(&self, programs: &Programs<'_>, x: *const c_void, logits: *mut f32, rows: usize,
        hidden: usize, vocab: usize, stream: *mut c_void) -> Result<()> {
        let (values, scales, block) = match self {
            Self::Fp8 { values, scales } => (values, scales, false),
            Self::BlockFp8 { values, scales } => (values, scales, true),
            _ => anyhow::bail!("the GLMF head has no FP8 operand"),
        };
        ensure!(values.buffer.bytes == vocab * hidden
            && scales.buffer.bytes >= if block { vocab/128*(hidden/128)*4 } else { vocab*(hidden/128)*4 },
            "FP8 head operands do not match [{vocab}, {hidden}]");
        ensure!(rows <= i32::MAX as usize, "FP8 head row count {rows} exceeds its scalar ABI");
        let program = programs.program(if block { "glmf_head_block_fp8" } else { "glmf_head_fp8" }, &["x", "w_fp8", "scale", "logits"])?;
        if block {
            if rows > 0 {
                // SAFETY: the caller's input/output extents and this head's live operands.
                unsafe { program.launch(&[x.cast_mut(), values.buffer.ptr, scales.buffer.ptr, logits.cast()],
                    &[Scalar::I32(rows as i32)], stream)?; }
            }
            return Ok(());
        }
        for (first, n) in fp8_head_spans(rows) {
            let x = x.cast::<u8>().wrapping_add(first * hidden * 2).cast_mut().cast();
            let out = logits.cast::<u8>().wrapping_add(first * vocab * 4).cast();
            // SAFETY: rows first..first + n of the caller's input and logits (its
            // contract) and this head's live operands, ordered on `stream`.
            unsafe { program.launch(&[x, values.buffer.ptr, scales.buffer.ptr, out], &[Scalar::I32(n as i32)],
                stream)? };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An explicit mode always wins; unset follows the shared setting (tensor by default).
    #[test]
    fn the_target_head_resolves_explicit_modes_and_otherwise_the_shared_setting() {
        assert_eq!(resolve(Some(TargetHead::Exact)), TargetHead::Exact);
        assert_eq!(resolve(Some(TargetHead::Tensor)), TargetHead::Tensor);
        assert_eq!(resolve(None), TargetHead::configured());
    }

    #[test]
    fn spans_cover_every_row_count_without_a_fallback() {
        for rows in 1..=4096 {
            let mut next = 0;
            for (first, n) in fp8_head_spans(rows) {
                assert_eq!(first, next);
                assert!((1..=FP8_HEAD_ROWS).contains(&n));
                next += n;
            }
            assert_eq!(next, rows);
        }
        assert_eq!(fp8_head_spans(0).count(), 0);
    }

    /// GPU check of `--fp8-head packed` on a real GLM 5.3 Flash head: [`GlmfHead::pack`] packs and
    /// verifies `lm_head.weight` and releases the BF16; the planner's bytes and scratch equal the
    /// native ones; the exact GEMV (target logits) and the coarse GEMV (drafter) of 70 rows (two
    /// passes) match FP64 CPU sums of the BF16 and coarse-plane weights on sampled tokens; a graph
    /// replay of the exact GEMV reproduces the direct launch bit for bit.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT (GLM 5.3 Flash with a BF16 lm_head) and a CUDA GPU"]
    fn packed_head_packs_the_checkpoint_head_and_projects_it_exactly() -> Result<()> {
        use cuteafd_core::packed_head::{choose_base, decode_coarse, encode, workspace_bytes, PackedHeadLayout};
        // SAFETY: the test's native library, loaded once.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let tensor = crate::shared::checkpoint_read::read_tensor(&checkpoint, "lm_head.weight")?;
        let [vocab, hidden] = tensor.info.shape[..] else { anyhow::bail!("lm_head.weight is not a matrix") };
        ensure!(tensor.info.dtype == cuteafd_core::DType::Bf16, "lm_head.weight is not BF16");
        let weights: Vec<u16> = tensor.bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        let source = DeviceAllocation::new(&library, tensor.bytes.len(), crate::shared::memory::TEST)?;
        library.copy_h2d(source.buffer, &tensor.bytes)?;
        let mut head = GlmfHead::Bf16(source).pack(&library, vocab, hidden, None)?;
        assert_eq!(head.format(), Some(GlmfHeadFormat::Packed));
        assert_eq!(head.bytes() as u64, GlmfHeadFormat::Packed.resident_bytes(vocab as u64, hidden as u64));
        assert_eq!(head.bytes(), PackedHeadLayout::new(vocab, hidden)?.total_bytes());
        let sms = library.sm_count()? as u64;
        for rows in [1, 16, 64, 65, 128, 4096] {
            assert_eq!(library.packed_head_workspace(rows, hidden, vocab)? as u64,
                workspace_bytes(rows as u64, hidden as u64, vocab as u64, sms), "{rows} rows on {sms} SMs");
        }
        let rows = 70;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let x: Vec<u16> = (0..rows * hidden).map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).to_bits() >> 16) as u16
        }).collect();
        let bf16 = |bits: u16| f32::from_bits(u32::from(bits) << 16) as f64;
        let input = DeviceAllocation::new(&library, rows * hidden * 2, crate::shared::memory::TEST)?;
        library.copy_h2d(input.buffer, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        let scratch = crate::shared::packed_head::scratch(&library, rows, &[(hidden, vocab)], crate::shared::memory::TEST)?;
        head.bind_draft(&library, rows)?;
        let stream = library.cuda_stream_create()?;
        let project = |coarse: bool, logits: &DeviceAllocation<'_>| -> Result<Vec<f32>> {
            // SAFETY: the rows, logits and scratch are live allocations of these shapes; the stream drains.
            unsafe {
                head.launch_packed(&library, input.buffer.ptr, logits.buffer.ptr.cast(), rows, coarse,
                    Some(scratch.buffer), stream)?;
                library.cuda_stream_synchronize(stream)?;
            }
            let mut bytes = vec![0u8; rows * vocab * 4];
            library.copy_d2h_quiescent(&mut bytes, logits.buffer)?;
            Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
        };
        let logits = DeviceAllocation::new(&library, rows * vocab * 4, crate::shared::memory::TEST)?;
        for coarse in [false, true] {
            let out = project(coarse, &logits)?;
            for token in (0..vocab).step_by(997).chain([vocab - 1]) {
                let row = &weights[token * hidden..(token + 1) * hidden];
                let (e0, _) = choose_base(row);
                let w = |i: usize| if coarse { bf16(decode_coarse(encode(row[i], e0).0, e0)) } else { bf16(row[i]) };
                for r in 0..rows {
                    let (mut sum, mut magnitude) = (0f64, 0f64);
                    for i in 0..hidden {
                        let product = w(i) * bf16(x[r * hidden + i]);
                        sum += product;
                        magnitude += product.abs();
                    }
                    let gpu = f64::from(out[r * vocab + token]);
                    ensure!((gpu - sum).abs() <= 1e-4 * magnitude + 1e-6,
                        "coarse {coarse}: row {r} token {token}: GPU {gpu} versus {sum} (|sum| {magnitude})");
                }
            }
        }
        // Decode graphs capture the target head: a replay of the captured exact GEMV is the launch.
        let direct = project(false, &logits)?;
        let replayed = DeviceAllocation::new(&library, rows * vocab * 4, crate::shared::memory::TEST)?;
        // SAFETY: capture records launches over live buffers that outlive the executable's replay.
        let exec = unsafe {
            library.cuda_graph_begin_capture(stream)?;
            let launched = head.launch_packed(&library, input.buffer.ptr, replayed.buffer.ptr.cast(), rows, false,
                Some(scratch.buffer), stream);
            let exec = library.cuda_graph_end_capture_for_test(stream)?;
            launched?;
            library.cuda_graph_launch(exec, stream)?;
            library.cuda_stream_synchronize(stream)?;
            exec
        };
        let mut bytes = vec![0u8; rows * vocab * 4];
        library.copy_d2h_quiescent(&mut bytes, replayed.buffer)?;
        let graph: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        ensure!(graph.iter().zip(&direct).all(|(a, b)| a.to_bits() == b.to_bits()), "graph replay differs from the launch");
        // SAFETY: the replay drained; nothing else uses the executable or the stream.
        unsafe {
            library.cuda_graph_exec_destroy(exec)?;
            library.cuda_stream_destroy(stream)?;
        }
        eprintln!("PASS packed [{vocab}, {hidden}] head: {} bytes, exact and coarse GEMVs, graph replay", head.bytes());
        Ok(())
    }
}
