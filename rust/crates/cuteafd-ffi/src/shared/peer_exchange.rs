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

impl NativeLibrary {
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

    /// Write-mode Spark completions on `stream` (see `cuteafd_spark_wait_written`).
    ///
    /// # Safety
    /// `flags` (`ranks` u64 words `stride_words` apart) is device memory the
    /// NICs write; `state` and `error` are live memory mapped on the stream's
    /// device; one wave was (or will be) posted per wait.
    pub unsafe fn spark_wait_written(&self, flags: *const u64, ranks: u32, stride_words: u32, state: *mut u32,
        error: *mut u32, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const u64, u32, u32, *mut u32, *mut u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_spark_wait_written") }?;
        let status = unsafe { f(flags, ranks, stride_words, state, error, stream) };
        ensure!(status == 0, "Spark written-completion wait failed with CUDA error {status}");
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
