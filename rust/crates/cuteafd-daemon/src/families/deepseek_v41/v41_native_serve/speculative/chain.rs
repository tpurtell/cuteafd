//! Static draft-chain dispatch for the shared request lifecycle.
use super::*;
use crate::families::deepseek_v41::v41_experts::dspark::DistributedDsparkChain;

pub(crate) trait DraftChain<'a> {
    fn execution_device(&self) -> Option<crate::shared::memory::device::Device<'a>>;
    fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()>;
    fn stage_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()>;
    /// # Safety
    /// Windows, bindings and staged inputs obey the concrete chain's contract;
    /// backing storage remains live until completion or chain destruction.
    unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>>;
    /// Drop a pending replay so its requests can be released.
    fn cancel_replay(&mut self);
    /// Record a CUDA/replay failure as terminal for this chain.
    fn note_failure(&mut self, error: &anyhow::Error);
    /// Draft width of the next (or pending) proposal.
    fn width(&self) -> usize;
    /// Select the width (5 or 7, within the loaded maximum) before staging.
    fn set_width(&mut self, width: usize) -> Result<()>;
    /// A terminal draft (dead expert wire or a sticky CUDA/replay failure) is
    /// not a reusable target-only fallback: the lane must fail the round
    /// instead of verifying anchors forever. Mirrors GLM Flash's `require_live`
    /// and MiMo's `is_terminal`.
    fn require_live(&self) -> Result<()>;
}
/// The synchronous draft operations the single-lane `propose` path needs. Only
/// the local chain implements them; the distributed chain drafts through
/// `poll_propose` instead.
pub(crate) trait ProposeChain<'a>: DraftChain<'a> {
    fn has_graph(&self, count: usize) -> bool;
    /// Whether this count may still capture at its current width (false once
    /// the startup inventory is sealed without it).
    fn capture_allowed(&self, count: usize) -> bool;
    /// # Safety
    /// Same contract as [`DraftChain::begin_replay`]; runs the proposal without
    /// a captured executable.
    unsafe fn run(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    /// Stage the seed tokens and upload them to the device.
    fn set_tokens(&mut self, tokens: &[i32]) -> Result<()>;
    /// Reserve a fresh sampling attempt (synchronizes and uploads).
    fn prepare_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()>;
    /// # Safety
    /// Same contract as [`DraftChain::begin_replay`].
    unsafe fn capture(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    /// # Safety
    /// Same contract as [`DraftChain::begin_replay`].
    unsafe fn replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()>;
    fn copy_d2h(&self, dst: &mut [u8], buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<()>;
    fn draft_output(&self) -> Result<[cuteafd_ffi::CuteafdDeviceBuffer; 3]>;
}
impl<'w, 'a> ProposeChain<'a> for DsparkChain<'w, 'a> {
    fn has_graph(&self, count: usize) -> bool { self.has_graph(count) }
    fn capture_allowed(&self, count: usize) -> bool {
        self.capture_allowed(count)
    }
    unsafe fn run(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        // SAFETY: forwards the caller's stable window/binding contract.
        unsafe { self.execute(windows, bindings).map(drop) }
    }
    fn set_tokens(&mut self, tokens: &[i32]) -> Result<()> { self.set_tokens(tokens) }
    fn prepare_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()> {
        self.prepare_sampling(rngs, temperatures)
    }
    unsafe fn capture(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        // SAFETY: forwards the caller's stable window/binding contract.
        unsafe { self.capture(windows, bindings) }
    }
    unsafe fn replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        // SAFETY: forwards the caller's stable window/binding contract.
        unsafe { self.replay(windows, bindings).map(drop) }
    }
    fn copy_d2h(&self, dst: &mut [u8], buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<()> { self.copy_d2h(dst, buffer) }
    fn draft_output(&self) -> Result<[cuteafd_ffi::CuteafdDeviceBuffer; 3]> { self.draft_output() }
}

macro_rules! chain {
    ($ty:ident, $device:expr, $live:expr) => {
        impl<'w, 'a> DraftChain<'a> for $ty<'w, 'a> {
            fn execution_device(&self) -> Option<crate::shared::memory::device::Device<'a>> { ($device)(self) }
            fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
                self.stage_tokens(tokens)
            }
            fn stage_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()> {
                self.stage_sampling(rngs, temperatures)
            }
            unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
                bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
                // SAFETY: forwards the caller's window, binding and lifetime contract.
                unsafe { self.begin_replay(windows, bindings) }
            }
            fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
                self.poll_replay()
            }
            fn cancel_replay(&mut self) { self.cancel_replay() }
            fn note_failure(&mut self, error: &anyhow::Error) { self.note_failure(error) }
            fn width(&self) -> usize { self.width() }
            fn set_width(&mut self, width: usize) -> Result<()> {
                self.set_width(width)
            }
            fn require_live(&self) -> Result<()> { ($live)(self) }
        }
    };
}
chain!(DsparkChain, |_: &DsparkChain| None, |chain: &DsparkChain| chain.require_live());
chain!(DistributedDsparkChain, |chain: &DistributedDsparkChain<'w, 'a>| Some(chain.device()),
    |chain: &DistributedDsparkChain<'w, 'a>| { chain.require_live()?; super::require_peer_live() });
