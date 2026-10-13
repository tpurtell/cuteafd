//! Official dSpark confidence is a raw FP32 projection, without sigmoid.
use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

type ConfidenceFn =
    unsafe extern "C" fn(*const u16, *const u16, *const u16, *mut f32, i32, *mut c_void) -> i32;
pub struct V41DsparkConfidence<'a> {
    _library: &'a NativeLibrary,
    launch: ConfidenceFn,
}
impl NativeLibrary {
    pub fn v41_dspark_confidence(&self) -> Result<V41DsparkConfidence<'_>> {
        let launch = unsafe {
            *self
                .lib
                .get::<ConfidenceFn>(b"cuteafd_v41_dspark_confidence")?
        };
        Ok(V41DsparkConfidence {
            _library: self,
            launch,
        })
    }
}
impl V41DsparkConfidence<'_> {
    /// # Safety
    /// Initialized BF16 inputs and FP32 output must belong to the stream's device
    /// and remain live through completion; output must not overlap inputs.
    pub unsafe fn launch(
        &self,
        hidden: CuteafdDeviceBuffer,
        markov: CuteafdDeviceBuffer,
        weight: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid dSpark confidence rows");
        for (buffer, bytes) in [
            (hidden, rows * 10240),
            (markov, rows * 512),
            (weight, 10752),
            (output, rows * 4),
        ] {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes,
                "dSpark confidence buffer is null or undersized"
            );
        }
        let status = unsafe {
            (self.launch)(
                hidden.ptr.cast(),
                markov.ptr.cast(),
                weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native dSpark confidence CUDA status {status}");
        Ok(())
    }
}

type MarkovCreate = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> i32;
type MarkovDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type MarkovLaunch =
    unsafe extern "C" fn(*mut c_void, *const u16, *const u16, *mut f32, i32, *mut c_void) -> i32;
pub struct V41VocabularyProjection<'a> {
    _library: &'a NativeLibrary,
    handle: *mut c_void,
    launch: MarkovLaunch,
    /// `cuteafd_vocabulary_head_launch_tensor_op` on this handle, when the native library has it.
    launch_tensor_op: Option<MarkovLaunch>,
    destroy: MarkovDestroy,
    width: usize,
    max_rows: usize,
    vocab_rows: usize,
}
impl NativeLibrary {
    /// # Safety
    /// Workspace must remain allocated on the current device until this handle
    /// and all graphs captured with it are destroyed; serialize handle use.
    pub unsafe fn v41_dspark_markov(
        &self,
        workspace: CuteafdDeviceBuffer,
    ) -> Result<V41VocabularyProjection<'_>> {
        unsafe { self.v41_head_projection(workspace, false) }
    }
    /// # Safety
    /// Same workspace/device/graph lifetime contract as v41_dspark_markov.
    pub unsafe fn v41_vocabulary_head(
        &self,
        workspace: CuteafdDeviceBuffer,
    ) -> Result<V41VocabularyProjection<'_>> {
        unsafe { self.v41_head_projection(workspace, true) }
    }
    /// Create a projection for one contiguous set of vocabulary rows.
    ///
    /// # Safety
    /// Same workspace/device/graph lifetime contract as v41_vocabulary_head.
    /// The caller retains the shard's global token offset separately.
    pub unsafe fn v41_vocabulary_shard(
        &self,
        workspace: CuteafdDeviceBuffer,
        vocab_rows: usize,
    ) -> Result<V41VocabularyProjection<'_>> {
        ensure!((1..=129280).contains(&vocab_rows), "invalid vocabulary shard rows");
        type Create = unsafe extern "C" fn(*mut c_void, u64, i32, *mut *mut c_void) -> i32;
        let create = unsafe { *self.lib.get::<Create>(b"cuteafd_v41_vocabulary_shard_create")? };
        let launch = unsafe { *self.lib.get::<MarkovLaunch>(b"cuteafd_v41_vocabulary_head_launch")? };
        let destroy = unsafe { *self.lib.get::<MarkovDestroy>(b"cuteafd_v41_markov_destroy")? };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { create(workspace.ptr, workspace.bytes as u64, vocab_rows as i32, &mut handle) };
        ensure!(status == 0 && !handle.is_null(), "vocabulary shard initialization status {status}");
        let launch_tensor_op = unsafe { self.lib.get::<MarkovLaunch>(b"cuteafd_vocabulary_head_launch_tensor_op") }
            .ok().map(|f| *f);
        Ok(V41VocabularyProjection { _library: self, handle, launch, launch_tensor_op, destroy,
            width: 5120, max_rows: 128, vocab_rows })
    }
    /// Merge local greedy candidates after the peer candidates arrive.
    ///
    /// # Safety
    /// All six buffers belong to the stream's device and remain live through
    /// completion. Inputs are complete and outputs are disjoint from each other
    /// and the inputs. Invalid candidates publish UINT32_MAX/NaN.
    pub unsafe fn v41_vocabulary_merge_greedy(
        &self, candidates: [(CuteafdDeviceBuffer, CuteafdDeviceBuffer); 2],
        output: (CuteafdDeviceBuffer, CuteafdDeviceBuffer), rows: usize,
        split: usize, stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=128).contains(&rows) && (1..129280).contains(&split),
            "invalid vocabulary greedy merge shape");
        for buffer in [candidates[0].0, candidates[0].1, candidates[1].0,
            candidates[1].1, output.0, output.1] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= rows * 4
                && buffer.device_id == output.0.device_id, "invalid vocabulary merge buffer");
        }
        type Merge = unsafe extern "C" fn(*const u32, *const f32, *const u32,
            *const f32, *mut u32, *mut f32, i32, i32, *mut c_void) -> i32;
        let launch = unsafe { *self.lib.get::<Merge>(b"cuteafd_v41_vocabulary_merge_greedy")? };
        let status = unsafe { launch(candidates[0].0.ptr.cast(), candidates[0].1.ptr.cast(),
            candidates[1].0.ptr.cast(), candidates[1].1.ptr.cast(), output.0.ptr.cast(),
            output.1.ptr.cast(), rows as i32, split as i32, stream) };
        ensure!(status == 0, "vocabulary greedy merge status {status}");
        Ok(())
    }
    unsafe fn v41_head_projection(
        &self,
        workspace: CuteafdDeviceBuffer,
        full: bool,
    ) -> Result<V41VocabularyProjection<'_>> {
        let create_name: &[u8] = if full {
            b"cuteafd_v41_vocabulary_head_create"
        } else {
            b"cuteafd_v41_markov_create"
        };
        let launch_name: &[u8] = if full {
            b"cuteafd_v41_vocabulary_head_launch"
        } else {
            b"cuteafd_v41_markov_launch"
        };
        let create = unsafe { *self.lib.get::<MarkovCreate>(create_name)? };
        let launch = unsafe { *self.lib.get::<MarkovLaunch>(launch_name)? };
        let destroy = unsafe {
            *self
                .lib
                .get::<MarkovDestroy>(b"cuteafd_v41_markov_destroy")?
        };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { create(workspace.ptr, workspace.bytes as u64, &mut handle) };
        ensure!(
            status == 0 && !handle.is_null(),
            "Markov initialization status {status}"
        );
        let launch_tensor_op = unsafe { self.lib.get::<MarkovLaunch>(b"cuteafd_vocabulary_head_launch_tensor_op") }
            .ok().map(|f| *f);
        Ok(V41VocabularyProjection {
            _library: self,
            handle,
            launch,
            launch_tensor_op,
            destroy,
            width: if full { 5120 } else { 256 },
            max_rows: if full { 128 } else { 16 },
            vocab_rows: 129280,
        })
    }
}
impl V41VocabularyProjection<'_> {
    pub const WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
    /// # Safety
    /// Initialized inputs and disjoint output must remain valid on the handle's
    /// device through stream completion, and must not overlap its workspace.
    pub unsafe fn launch(
        &self,
        embedding: CuteafdDeviceBuffer,
        weight: CuteafdDeviceBuffer,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe { self.launch_with(embedding, weight, logits, rows, stream, false) }
    }
    /// [`Self::launch`] as a BF16 tensor-core GEMM with FP32 accumulation and FP32 logits
    /// (`cuteafd_vocabulary_head_launch_tensor_op` on the same handle and workspace), for the
    /// model-width heads (`v41_vocabulary_head`, `v41_vocabulary_shard`).
    ///
    /// # Safety
    /// As [`Self::launch`].
    pub unsafe fn launch_tensor_op(
        &self,
        embedding: CuteafdDeviceBuffer,
        weight: CuteafdDeviceBuffer,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe { self.launch_with(embedding, weight, logits, rows, stream, true) }
    }
    unsafe fn launch_with(
        &self,
        embedding: CuteafdDeviceBuffer,
        weight: CuteafdDeviceBuffer,
        logits: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
        tensor_op: bool,
    ) -> Result<()> {
        ensure!(
            (1..=self.max_rows).contains(&rows),
            "invalid vocabulary projection rows"
        );
        let launch = if tensor_op {
            ensure!(self.width == 5120, "the tensor-op launch takes the model-width vocabulary head");
            self.launch_tensor_op.context(
                "this native library has no cuteafd_vocabulary_head_launch_tensor_op (rebuild it with this checkout)")?
        } else {
            self.launch
        };
        for (buffer, bytes) in [
            (embedding, rows * self.width * 2),
            (weight, self.vocab_rows * self.width * 2),
            (logits, rows * self.vocab_rows * 4),
        ] {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes,
                "invalid Markov buffer"
            );
        }
        let status = unsafe {
            launch(
                self.handle,
                embedding.ptr.cast(),
                weight.ptr.cast(),
                logits.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "Markov launch status {status}");
        Ok(())
    }
}
impl Drop for V41VocabularyProjection<'_> {
    fn drop(&mut self) {
        let status = unsafe { (self.destroy)(self.handle) };
        if status != 0 {
            eprintln!("Markov handle destruction status {status}");
        }
    }
}

type DraftStepFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const u64,
    *const f32,
    *mut f32,
    *mut u32,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41DraftStep<'a> {
    _library: &'a NativeLibrary,
    launch: DraftStepFn,
}
impl NativeLibrary {
    pub fn v41_draft_step(&self) -> Result<V41DraftStep<'_>> {
        let launch = unsafe { *self.lib.get::<DraftStepFn>(b"cuteafd_v41_draft_step_rng")? };
        Ok(V41DraftStep {
            _library: self,
            launch,
        })
    }
}
impl V41DraftStep<'_> {
    /// # Safety
    /// All buffers must live on the stream device through completion. Shared+bias
    /// must be finite and temperatures finite/nonnegative. RNG [rows,2] contains
    /// seeds and base Philox subsequences with room for (position+1)*256 draws.
    /// Output spans must not overlap each other or any input.
    pub unsafe fn launch(
        &self,
        shared: CuteafdDeviceBuffer,
        bias: CuteafdDeviceBuffer,
        rng: CuteafdDeviceBuffer,
        temperatures: CuteafdDeviceBuffer,
        adjusted: CuteafdDeviceBuffer,
        tokens: CuteafdDeviceBuffer,
        rows: usize,
        position: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=16).contains(&rows), "invalid draft sampling rows");
        ensure!(position < 7, "invalid draft position");
        for (buffer, bytes) in [
            (shared, rows * 129280 * 4),
            (bias, rows * 129280 * 4),
            (rng, rows * 16),
            (temperatures, rows * 4),
            (adjusted, rows * 129280 * 4),
            (tokens, rows * 4),
        ] {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes,
                "invalid draft sampling buffer"
            );
        }
        let status = unsafe {
            (self.launch)(
                shared.ptr.cast(),
                bias.ptr.cast(),
                rng.ptr.cast(),
                temperatures.ptr.cast(),
                adjusted.ptr.cast(),
                tokens.ptr.cast(),
                rows as i32,
                position as i32,
                stream,
            )
        };
        ensure!(status == 0, "draft sampling CUDA status {status}");
        Ok(())
    }
}
