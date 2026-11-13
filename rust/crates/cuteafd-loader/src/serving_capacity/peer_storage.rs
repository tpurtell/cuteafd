//! Physical peer-exchange backings, shared with the native allocation path.
use anyhow::{ensure, Context, Result};

/// The private peer-wait diagnostic (`CUTEAFD_DIAG_PEER_WAIT=1`): per-issue push/wait
/// progress records and both ranks' frontier dumps on a timed-out wait. The launcher
/// forwards every `CUTEAFD_DIAG_*` variable to the coordinator (`wip-cards --arm-env`).
pub const PEER_WAIT_DIAGNOSTIC: &str = "CUTEAFD_DIAG_PEER_WAIT";

/// Whether the peer-wait diagnostic is on in this process.
pub fn peer_wait_diagnostics() -> bool {
    std::env::var(PEER_WAIT_DIAGNOSTIC).is_ok_and(|value| value == "1")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerStorageLayout {
    pub receive_bytes: usize,
    pub control_bytes: usize,
    pub progress_bytes: usize,
}

impl PeerStorageLayout {
    pub fn new(slots: usize, receive_bytes: usize, diagnostics: bool) -> Result<Self> {
        ensure!(slots > 0 && receive_bytes > 0, "empty peer storage shape");
        let flags = slots.checked_add(1).context("peer control count overflow")?;
        Ok(Self {
            receive_bytes: receive_bytes.max(256),
            control_bytes: flags.checked_mul(16).context("peer control size overflow")?.max(256),
            progress_bytes: if diagnostics {
                flags.checked_mul(std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>())
                    .context("peer progress size overflow")?.max(256)
            } else { 0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn charges_flags_once_and_diagnostics_only_when_enabled() {
        let plain = PeerStorageLayout::new(8, 3, false).unwrap();
        assert_eq!((plain.receive_bytes, plain.control_bytes, plain.progress_bytes), (256, 256, 0));
        let diagnostic = PeerStorageLayout::new(32, 4096, true).unwrap();
        assert_eq!(diagnostic.control_bytes, 33 * 16);
        assert_eq!(diagnostic.progress_bytes, (33 * std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256));
        assert!(PeerStorageLayout::new(0, 1, false).is_err());
        assert!(PeerStorageLayout::new(1, 0, false).is_err());
        assert!(PeerStorageLayout::new(usize::MAX, 1, true).is_err());
    }
}
