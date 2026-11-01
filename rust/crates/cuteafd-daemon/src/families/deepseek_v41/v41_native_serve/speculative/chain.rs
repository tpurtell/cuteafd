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
    /// Draft width of the next (or pending) proposal.
    fn width(&self) -> usize;
    /// Select the width (5 or 7, within the loaded maximum) before staging.
    fn set_width(&mut self, width: usize) -> Result<()>;
    /// A terminal draft (dead expert wire) is not a reusable target-only
    /// fallback: the lane must fail the round instead of verifying anchors
    /// forever. Mirrors GLM Flash's `require_live` and MiMo's `is_terminal`.
    fn require_live(&self) -> Result<()>;
}
macro_rules! chain {
    ($ty:ident, $device:expr, $live:expr) => {
        impl<'w, 'a> DraftChain<'a> for $ty<'w, 'a> {
            fn execution_device(&self) -> Option<crate::shared::memory::device::Device<'a>> { ($device)(self) }
            fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> { self.stage_tokens(tokens) }
            fn stage_sampling(&mut self, rngs: &mut [&mut cuteafd_core::DsparkRng], temperatures: &[f32]) -> Result<()> {
                self.stage_sampling(rngs, temperatures)
            }
            unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
                bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
                unsafe { self.begin_replay(windows, bindings) }
            }
            fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> { self.poll_replay() }
            fn width(&self) -> usize { self.width() }
            fn set_width(&mut self, width: usize) -> Result<()> { self.set_width(width) }
            fn require_live(&self) -> Result<()> { ($live)(self) }
        }
    };
}
chain!(DsparkChain, |_: &DsparkChain| None, |_: &DsparkChain| Ok(()));
chain!(DistributedDsparkChain, |chain: &DistributedDsparkChain<'w, 'a>| Some(chain.device()),
    |_: &DistributedDsparkChain<'w, 'a>| super::require_peer_live());
