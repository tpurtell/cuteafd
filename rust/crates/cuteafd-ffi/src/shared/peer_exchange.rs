//! Two-GPU peer exchange and the P2P probe (`native/shared/cuda/peer_exchange.cu`).
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

/// A strided byte plane for [`NativeLibrary::peer_push_planes`]
/// (`cuteafd_peer_plane_t`): `rows` rows of `row_bytes`, pitches in bytes,
/// everything 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PeerPlane {
    pub destination: *mut c_void,
    pub source: *const c_void,
    pub rows: u64,
    pub row_bytes: u64,
    pub destination_pitch: u64,
    pub source_pitch: u64,
}

/// Device progress record, layout-compatible with `cuteafd_peer_progress_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PeerProgress {
    pub push_issue: u64,
    pub published_issue: u64,
    pub wait_issue: u64,
    pub published: u32,
    pub observed: u32,
    pub expected: u32,
    pub push_phase: u32,
    pub wait_phase: u32,
    pub reserved: u32,
}

/// Graph-stable identity and persistent exchange pointers for opt-in diagnostics.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PeerDiagnostic {
    pub local: *mut PeerProgress,
    pub peer: *const PeerProgress,
    pub local_control: *const u32,
    pub peer_control: *const u32,
    pub issue: u64,
    pub host_stream: u64,
    pub rank: u32,
    pub slot: u32,
    pub slots: u32,
}

#[cfg(test)]
mod diagnostic_abi_tests {
    #[test]
    fn diagnostic_layout_matches_native_header() {
        use super::{PeerDiagnostic, PeerProgress};
        assert_eq!(std::mem::size_of::<PeerProgress>(), 48);
        assert_eq!(std::mem::offset_of!(PeerProgress, published), 24);
        assert_eq!(std::mem::size_of::<PeerDiagnostic>(), 64);
        assert_eq!(std::mem::offset_of!(PeerDiagnostic, issue), 32);
        assert_eq!(std::mem::offset_of!(PeerDiagnostic, rank), 48);
    }
}

/// `CUTEAFD_PEER_MAX_PLANES`.
pub const PEER_MAX_PLANES: usize = 4;

/// One `cuteafd_p2p_probe` measurement (see `cuteafd_peer_exchange.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pTest {
    CopyEngine,
    SmPull,
    SmPush,
    HostBounce,
    CopyEnginePingPongEvents,
    SmPushPingPongEvents,
    FlagPingPong,
    FlagPingPongGraph,
    FlagExchange,
    FlagExchangeGraph,
    CopyEnginePingPongGraph,
}

impl P2pTest {
    pub const ALL: [P2pTest; 11] = [
        Self::CopyEngine,
        Self::SmPull,
        Self::SmPush,
        Self::HostBounce,
        Self::CopyEnginePingPongEvents,
        Self::SmPushPingPongEvents,
        Self::FlagPingPong,
        Self::FlagPingPongGraph,
        Self::FlagExchange,
        Self::FlagExchangeGraph,
        Self::CopyEnginePingPongGraph,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::CopyEngine => "copy engine, one-way",
            Self::SmPull => "SM pull, one-way",
            Self::SmPush => "SM push, one-way",
            Self::HostBounce => "pinned-host bounce, one-way (host-timed)",
            Self::CopyEnginePingPongEvents => "copy engine + events, hop",
            Self::SmPushPingPongEvents => "SM push + events, hop",
            Self::FlagPingPong => "SM push + flag, hop",
            Self::FlagPingPongGraph => "SM push + flag, hop (graph)",
            Self::FlagExchange => "SM push + flag, two-way exchange",
            Self::FlagExchangeGraph => "SM push + flag, two-way exchange (graph)",
            Self::CopyEnginePingPongGraph => "copy engine + events, hop (graph)",
        }
    }
}

/// One rank's reply frames for [`NativeLibrary::pull_frames`]
/// (`cuteafd_pull_frames_t`): written by the host in mapped pinned memory,
/// read by the kernel when it runs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PullFrames {
    pub count: u32,
    pub reserved: u32,
    pub frame: [PullEntry; PULL_MAX_FRAMES],
}

/// One frame of a [`PullFrames`]: `bytes` from the device-visible `source`
/// to `offset` bytes into the rank's plane (all 4-byte multiples).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PullEntry {
    pub source: u64,
    pub offset: u64,
    pub bytes: u64,
}

/// Frames per rank and ranks per [`NativeLibrary::pull_frames`] launch.
pub const PULL_MAX_FRAMES: usize = 4;
pub const PULL_MAX_RANKS: usize = 8;

impl NativeLibrary {
    /// Preload diagnostic variants before an exchange can enqueue any wait.
    pub fn peer_diagnostic_initialize(&self) -> Result<()> {
        type F = unsafe extern "C" fn() -> i32;
        // SAFETY: preload only, no pointers or queued GPU work.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_peer_diagnostic_initialize")?() };
        ensure!(status == 0, "loading peer diagnostics failed with CUDA error {status}");
        Ok(())
    }

    /// # Safety
    /// As `peer_push_signal`; diagnostic controls/progress on both ranks must
    /// remain allocated and peer-accessible until every eager/captured use drains.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn peer_push_signal_diagnostic(&self, destination: *mut c_void, source: *const c_void,
        bytes: usize, flag: *mut u32, send_state: *mut u32, blocks: u32, stream: *mut c_void,
        diagnostic: &PeerDiagnostic) -> Result<()> {
        type F = unsafe extern "C" fn(*mut c_void, *const c_void, u64, *mut u32, *mut u32, u32,
            *mut c_void, *const PeerDiagnostic) -> i32;
        // SAFETY: caller retains all device owners; native copies the descriptor into kernel arguments.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_peer_push_signal_diagnostic")?(
            destination, source, bytes as u64, flag, send_state, blocks, stream, diagnostic) };
        ensure!(status == 0, "diagnostic peer push failed with CUDA error {status}");
        Ok(())
    }

    /// # Safety
    /// As `peer_push_planes` and `peer_push_signal_diagnostic`.
    pub unsafe fn peer_push_planes_diagnostic(&self, planes: &[PeerPlane], flag: *mut u32,
        send_state: *mut u32, blocks: u32, stream: *mut c_void, diagnostic: &PeerDiagnostic) -> Result<()> {
        type F = unsafe extern "C" fn(*const PeerPlane, u32, *mut u32, *mut u32, u32,
            *mut c_void, *const PeerDiagnostic) -> i32;
        ensure!((1..=PEER_MAX_PLANES).contains(&planes.len()), "{} planes in one push", planes.len());
        // SAFETY: caller retains plane/device owners; native copies all host descriptors.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_peer_push_planes_diagnostic")?(
            planes.as_ptr(), planes.len() as u32, flag, send_state, blocks, stream, diagnostic) };
        ensure!(status == 0, "diagnostic peer planes failed with CUDA error {status}");
        Ok(())
    }

    /// # Safety
    /// As `peer_wait_abortable` (null aborted: `peer_wait`), with persistent,
    /// peer-accessible controls/progress on both ranks through graph drainage.
    pub unsafe fn peer_wait_diagnostic(&self, flag: *const u32, recv_state: *mut u32,
        aborted: *const u32, stream: *mut c_void, diagnostic: &PeerDiagnostic) -> Result<()> {
        type F = unsafe extern "C" fn(*const u32, *mut u32, *const u32, *mut c_void, *const PeerDiagnostic) -> i32;
        // SAFETY: caller retains words and diagnostic owners; descriptor is copied by native.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_peer_wait_diagnostic")?(
            flag, recv_state, aborted, stream, diagnostic) };
        ensure!(status == 0, "diagnostic peer wait failed with CUDA error {status}");
        Ok(())
    }

    /// Optional component-fixture ABI, absent from production libraries.
    /// Streams are only compared by the shim, never dereferenced here.
    #[doc(hidden)]
    pub fn terminal_fault_fixture_configure(
        &self,
        mode: u32,
        lead: *mut c_void,
        peer: *mut c_void,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(u32, *mut c_void, *mut c_void);
        // SAFETY: fixture-only entry point records opaque stream identities.
        unsafe {
            self.lib.get::<F>(b"cuteafd_terminal_test_configure")?(mode, lead, peer);
        }
        Ok(())
    }

    /// Stop fixture observation and return actual device/pinned free calls.
    #[doc(hidden)]
    pub fn terminal_fault_fixture_finish(&self) -> Result<[u64; 2]> {
        type F = unsafe extern "C" fn(u32) -> u64;
        // SAFETY: counter-only test shim, retained by this library.
        let f = unsafe { self.lib.get::<F>(b"cuteafd_terminal_test_finish")? };
        Ok(unsafe { [f(0), f(1)] })
    }

    /// Optional private fixture: registered store/restore stream identities.
    #[doc(hidden)]
    pub fn terminal_fault_fixture_copy_streams(&self, streams: [*mut c_void; 4]) -> Result<()> {
        type F = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
        // SAFETY: the fixture records opaque identities; the owning cache lives.
        unsafe { self.lib.get::<F>(b"cuteafd_terminal_test_copy_streams")?(streams[0], streams[1], streams[2], streams[3]); }
        Ok(())
    }

    /// Optional fixture counter: releases after every owned stream completed.
    #[doc(hidden)]
    pub fn terminal_fault_fixture_late_frees(&self) -> Result<[u64; 2]> {
        type F = unsafe extern "C" fn(u32) -> u64;
        // SAFETY: counter-only symbol is retained by this library.
        let f = unsafe { self.lib.get::<F>(b"cuteafd_terminal_test_late_frees")? };
        Ok(unsafe { [f(0), f(1)] })
    }

    /// Clear fixture injection; this is not engine recovery or reuse.
    #[doc(hidden)]
    pub fn terminal_fault_fixture_clear(&self) -> Result<()> {
        type F = unsafe extern "C" fn();
        // SAFETY: clears an atomic in the fixture-only shim.
        unsafe {
            self.lib.get::<F>(b"cuteafd_terminal_test_clear_fault")?();
        }
        Ok(())
    }

    /// Check the optional terminal-abort ABI without creating CUDA state.
    pub fn peer_abort_available(&self) -> Result<()> {
        // SAFETY: lookup only; no entry point or CUDA call is invoked.
        unsafe {
            self.lib
                .get::<unsafe extern "C" fn() -> i32>(b"cuteafd_peer_abort_initialize")?;
            self.lib
                .get::<unsafe extern "C" fn(*const u32, *mut u32, *const u32, *mut c_void) -> i32>(
                    b"cuteafd_peer_wait_abortable",
                )?;
            self.lib
                .get::<unsafe extern "C" fn(*mut u32, *mut c_void) -> i32>(
                    b"cuteafd_peer_abort_publish",
                )?;
        }
        Ok(())
    }

    /// Preload the optional abort kernels before any spinning wait is queued.
    pub fn peer_abort_initialize(&self) -> Result<()> {
        type F = unsafe extern "C" fn() -> i32;
        // SAFETY: the entry point loads kernels and dereferences no arguments.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_peer_abort_initialize")?() };
        ensure!(
            status == 0,
            "loading terminal peer-abort kernels failed with CUDA error {status}"
        );
        Ok(())
    }

    /// # Safety
    /// All three words are persistent device memory on `stream`'s device.
    /// `aborted` is initially zero, bound unchanged in graphs, and never reset;
    /// after publishing it the owner must reject reuse and drain before release.
    pub unsafe fn peer_wait_abortable(
        &self,
        flag: *const u32,
        recv_state: *mut u32,
        aborted: *const u32,
        stream: *mut c_void,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(*const u32, *mut u32, *const u32, *mut c_void) -> i32;
        // SAFETY: the caller owns the persistent words and stream described above.
        let status = unsafe {
            self.lib.get::<F>(b"cuteafd_peer_wait_abortable")?(flag, recv_state, aborted, stream)
        };
        ensure!(
            status == 0,
            "abortable peer wait failed with CUDA error {status}"
        );
        Ok(())
    }

    /// # Safety
    /// `aborted` is persistent local device memory. `stream` is an independent
    /// stream on that device, with no dependency on a blocked compute stream.
    pub unsafe fn peer_abort_publish(&self, aborted: *mut u32, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*mut u32, *mut c_void) -> i32;
        // SAFETY: the caller supplies the independent stream and live word.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_peer_abort_publish")?(aborted, stream) };
        ensure!(
            status == 0,
            "publishing terminal peer abort failed with CUDA error {status}"
        );
        Ok(())
    }

    /// Microseconds per operation of `test` moving `bytes` between devices
    /// `a` and `b` (hops are half a round trip). `ingress` bits 0/1 stream
    /// host->device copies into `a`/`b` meanwhile. Allocates and synchronizes:
    /// a diagnostic, never on a serving path.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_probe(
        &self,
        a: i32,
        b: i32,
        bytes: usize,
        test: P2pTest,
        ingress: u32,
        iterations: u32,
        blocks: u32,
    ) -> Result<f64> {
        type F = unsafe extern "C" fn(i32, i32, u64, i32, u32, u32, u32, *mut f64) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_p2p_probe") }?;
        let mut us = 0.0;
        // SAFETY: the probe owns every allocation it touches; `us` outlives the call.
        let status = unsafe {
            f(
                a,
                b,
                bytes as u64,
                test as i32,
                ingress,
                iterations,
                blocks,
                &mut us,
            )
        };
        ensure!(
            status == 0,
            "P2P probe {:?} of {bytes} bytes failed with CUDA error {status}",
            test
        );
        Ok(us)
    }

    /// Loads every runtime-registered native kernel on the current device.
    /// CuTe cudaLibrary programs still require their explicit AOT load path.
    /// Successful loads are cached per library/device; failed loads are retryable.
    pub fn tp2_native_initialize(&self) -> Result<()> {
        let device = self.cuda_get_device()?;
        let mut loaded = self.native_modules_loaded.lock().unwrap();
        if loaded.contains(&device) { return Ok(()); }
        type F = unsafe extern "C" fn() -> i32;
        // SAFETY: initialization takes no pointers and launches no device work.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_tp2_native_initialize")?() };
        ensure!(status == 0, "loading TP2 native kernels failed with CUDA error {status}");
        loaded.insert(device);
        Ok(())
    }

    /// Does not load modules or queue work. Admission must initialize this
    /// library on each physical device before any exchange can start.
    pub fn native_modules_preloaded(&self) -> Result<bool> {
        Ok(self.native_modules_loaded.lock().unwrap().contains(&self.cuda_get_device()?))
    }

    /// Loads the exchange kernels on the current device (before any wait is
    /// queued there; see `cuteafd_peer_exchange_initialize`).
    pub fn peer_exchange_initialize(&self) -> Result<()> {
        type F = unsafe extern "C" fn() -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_exchange_initialize") }?;
        let status = unsafe { f() };
        ensure!(
            status == 0,
            "loading the peer exchange kernels failed with CUDA error {status}"
        );
        Ok(())
    }

    /// `out = bf16(a + b)` over `count` BF16 elements on `stream`.
    ///
    /// # Safety
    /// `a`, `b` and `out` are live device buffers of `count` elements on the
    /// stream's device, `out` disjoint from both, producers ordered before.
    pub unsafe fn peer_add_bf16(
        &self,
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        count: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            *mut c_void,
            u64,
            *mut c_void,
        ) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_add_bf16_async") }?;
        let status = unsafe { f(a, b, out, count as u64, stream) };
        ensure!(
            status == 0,
            "BF16 add of {count} elements failed with CUDA error {status}"
        );
        Ok(())
    }

    /// Pushes `bytes` from local `source` to peer `destination` and publishes
    /// the next sequence to the peer's `flag` (see `cuteafd_peer_push_signal`).
    ///
    /// # Safety
    /// `stream` belongs to the current (source) device with peer access to the
    /// destination's device; `destination` and `flag` are live peer memory,
    /// `source` and `send_state` (u32 [2]) live local memory, 16-byte aligned
    /// with `bytes % 16 == 0`; the source stays unchanged and the destination
    /// unread by others until the peer's matching wait.
    pub unsafe fn peer_push_signal(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        flag: *mut u32,
        send_state: *mut u32,
        blocks: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(
            *mut c_void,
            *const c_void,
            u64,
            *mut u32,
            *mut u32,
            u32,
            *mut c_void,
        ) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_push_signal") }?;
        let status = unsafe {
            f(
                destination,
                source,
                bytes as u64,
                flag,
                send_state,
                blocks,
                stream,
            )
        };
        ensure!(
            status == 0,
            "peer push of {bytes} bytes failed with CUDA error {status}"
        );
        Ok(())
    }

    /// Copies up to four byte regions (`(destination, source, bytes)`, any
    /// alignment) with SM loads and stores on `stream`, then a system-scope
    /// fence (`cuteafd_sm_copy_regions`).
    ///
    /// # Safety
    /// Every pointer is a device pointer (or a mapped host alias) valid for
    /// its `bytes` on the stream's device until the copy completes; the
    /// regions do not overlap.
    pub unsafe fn sm_copy_regions(&self, regions: &[(*mut c_void, *const c_void, u64)], stream: *mut c_void)
        -> Result<()> {
        type F = unsafe extern "C" fn(*const *mut c_void, *const *const c_void, *const u64, u32, *mut c_void) -> i32;
        ensure!((1..=4).contains(&regions.len()), "{} regions in one SM copy", regions.len());
        let destinations: Vec<_> = regions.iter().map(|r| r.0).collect();
        let sources: Vec<_> = regions.iter().map(|r| r.1).collect();
        let bytes: Vec<_> = regions.iter().map(|r| r.2).collect();
        // SAFETY: the caller vouches for the regions; native copies the descriptors.
        let status = unsafe { self.lib.get::<F>(b"cuteafd_sm_copy_regions")?(destinations.as_ptr(), sources.as_ptr(),
            bytes.as_ptr(), regions.len() as u32, stream) };
        ensure!(status == 0, "SM region copy failed with CUDA error {status}");
        Ok(())
    }

    /// Pulls each rank's published reply frames ([`PullFrames`], in mapped
    /// pinned memory, read when the kernel runs) into its device plane on
    /// `stream` (see `cuteafd_pull_frames`). One launch for up to
    /// [`PULL_MAX_RANKS`] ranks.
    ///
    /// # Safety
    /// Every `replies[i]` is the device-visible address of a live mapped
    /// [`PullFrames`] whose frames (when its count is non-zero by the time the
    /// kernel runs) are live mapped pinned memory, unchanged until the kernel
    /// completes; every plane is a live device buffer of its capacity on the
    /// stream's device, unread until later work on the stream.
    pub unsafe fn pull_frames(&self, replies: &[*const PullFrames], planes: &[(*mut c_void, u64)],
        stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const *const PullFrames, *const *mut c_void, *const u64, u32, *mut c_void) -> i32;
        ensure!((1..=PULL_MAX_RANKS).contains(&replies.len()) && replies.len() == planes.len(),
            "{} replies into {} planes in one pull", replies.len(), planes.len());
        let destinations: Vec<_> = planes.iter().map(|plane| plane.0).collect();
        let capacity: Vec<_> = planes.iter().map(|plane| plane.1).collect();
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_pull_frames") }?;
        // SAFETY: the caller vouches for the descriptors and planes; native copies the arrays.
        let status = unsafe { f(replies.as_ptr(), destinations.as_ptr(), capacity.as_ptr(), replies.len() as u32, stream) };
        ensure!(status == 0, "pulling {} ranks' reply frames failed with CUDA error {status}", replies.len());
        Ok(())
    }

    /// Copies up to [`PEER_MAX_PLANES`] strided planes (local or peer) in one
    /// launch, then publishes the next sequence to `flag` exactly as
    /// [`NativeLibrary::peer_push_signal`] does, so the two share a link
    /// (see `cuteafd_peer_push_planes`). `link` `None` copies only.
    ///
    /// # Safety
    /// As [`NativeLibrary::peer_push_signal`] for every plane: `stream` is on
    /// the current device, which has peer access to any peer pointer; every
    /// source stays unchanged and every destination unread until the matching
    /// wait (or, without a link, until later work on this stream).
    pub unsafe fn peer_push_planes(
        &self,
        planes: &[PeerPlane],
        link: Option<(*mut u32, *mut u32)>,
        blocks: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(*const PeerPlane, u32, *mut u32, *mut u32, u32, *mut c_void) -> i32;
        ensure!((1..=PEER_MAX_PLANES).contains(&planes.len()), "{} planes in one push", planes.len());
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_push_planes") }?;
        let (flag, send_state) = link.unwrap_or((std::ptr::null_mut(), std::ptr::null_mut()));
        let status = unsafe { f(planes.as_ptr(), planes.len() as u32, flag, send_state, blocks, stream) };
        ensure!(status == 0, "peer push of {} planes failed with CUDA error {status}", planes.len());
        Ok(())
    }

    /// Publishes a host mailbox on `stream`: writes `words` to `descriptor`,
    /// then the next sequence to `flag` (see `cuteafd_host_signal`).
    ///
    /// # Safety
    /// `flag` and `descriptor` are live pinned, device-mapped host memory,
    /// `send_state` (u32 [1]) live memory of the stream's device; the host
    /// side reads the mailbox only after it sees the sequence.
    pub unsafe fn host_signal(&self, flag: *mut u32, send_state: *mut u32, descriptor: *mut u32, words: [u32; 4],
        stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*mut u32, *mut u32, *mut u32, *const u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_host_signal") }?;
        let status = unsafe { f(flag, send_state, descriptor, words.as_ptr(), stream) };
        ensure!(status == 0, "host mailbox signal failed with CUDA error {status}");
        Ok(())
    }

    /// Spins on `stream` until `flag` reaches the next sequence.
    ///
    /// # Safety
    /// `flag` and `recv_state` (u32 [1]) are live memory of the stream's
    /// device; a peer push for every wait is (or will be) enqueued, or the
    /// stream never drains.
    pub unsafe fn peer_wait(
        &self,
        flag: *const u32,
        recv_state: *mut u32,
        stream: *mut c_void,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(*const u32, *mut u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_wait") }?;
        let status = unsafe { f(flag, recv_state, stream) };
        ensure!(status == 0, "peer wait failed with CUDA error {status}");
        Ok(())
    }
}
