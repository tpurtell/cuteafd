//! The bundled mtp_dspark drafter's recurrent Markov head
//! (`native/families/mistral_large4/cuda/ml4_dspark.cu`). Every pointer is device
//! memory of the documented shape on the stream's device; the stream orders them.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

fn check(status: i32, what: &str) -> Result<()> {
    ensure!(status == 0, "Mistral Large 4 dSpark {what} failed with {status}");
    Ok(())
}

fn i(value: usize) -> Result<i32> {
    Ok(i32::try_from(value)?)
}

impl NativeLibrary {
    /// Workspace bytes [`Self::ml4_dspark_markov`] needs.
    pub fn ml4_dspark_markov_workspace(&self, sequences: usize, block: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32) -> u64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_ml4_dspark_markov_workspace") }?;
        Ok(usize::try_from(unsafe { f(i(sequences)?, i(block)?) })?)
    }

    /// `norms` [vocab] F32: the (slackened) row norms of the Markov output projection
    /// (BF16 [vocab, 512]), once at load.
    ///
    /// # Safety
    /// `output` and `norms` are live device buffers of those shapes on `stream`'s device.
    pub unsafe fn ml4_dspark_markov_norms(&self, output: P, norms: M, vocab: usize, rank: usize, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_ml4_dspark_markov_norms") }?;
        check(unsafe { f(output, norms, i(vocab)?, i(rank)?, stream) }, "Markov norms")
    }

    /// Greedy drafts [sequences, block] U32 from the soft-capped base `logits`
    /// [sequences * block, vocab] FP32 and the final-norm rows `hidden`
    /// [sequences * block, hidden_width] BF16 through the RNN Markov head
    /// (`embedding`/`output` BF16 [vocab, 512], `joint_w` BF16 [1536, 1024 +
    /// hidden_width], `joint_b` BF16 [1536], `norms` from
    /// [`Self::ml4_dspark_markov_norms`]); step 0 conditions on `anchors` [sequences]
    /// U32. At most 32 sequences.
    ///
    /// # Safety
    /// Every buffer is live device memory of its documented shape on `stream`'s
    /// device; `workspace` holds [`Self::ml4_dspark_markov_workspace`] bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ml4_dspark_markov(&self, logits: P, hidden: P, embedding: P, joint_w: P, joint_b: P, output: P,
        norms: P, anchors: P, drafts: M, workspace: M, sequences: usize, block: usize, vocab: usize, rank: usize,
        hidden_width: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, M, M, i32, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_ml4_dspark_markov") }?;
        check(unsafe {
            f(logits, hidden, embedding, joint_w, joint_b, output, norms, anchors, drafts, workspace, i(sequences)?,
                i(block)?, i(vocab)?, i(rank)?, i(hidden_width)?, stream)
        }, "Markov drafts")
    }
}
