//! How a TP2 layer's two routed halves meet.
//!
//! Fused combine numerics (PLAN section 2): each rank forms
//! `p_r = dtype(FP32(routed_r) + FP32(shared_r))`, the FFN exchange carries
//! `p_r` to the other GPU, and both GPUs compute `BF16(FP32(p_0) + FP32(p_1))`
//! in rank order, so both hold the same bits. Spark reduction instead sums
//! routed planes first and adds shared after; this is a numerics change with
//! its own fidelity gate per format.
use super::{PartialDtype, PartialLayout};
use crate::shared::memory::device::{Allocation, Device, Event, PeerTransfer, Stream};
use anyhow::{ensure, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, RtxPartialDtype, RtxTp2Combine, V41Tp2ExpertReducer,
};
use std::ffi::c_void;

/// The dtype of the FFN exchange payload under [`Combine::FusedAllReduce`].
pub(crate) type ExchangeDtype = RtxPartialDtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Combine {
    /// HeadSplit / `Whole{ffn: Split}` layers: each rank adds its routed
    /// partial to its shared/dense half, then the existing FFN slot exchange
    /// sums the two in rank order on both GPUs.
    FusedAllReduce { exchange: ExchangeDtype },
    /// `Whole{ffn: Owner}` layers (V4.1): inputs and routes broadcast, routed
    /// then shared reduced in rank order onto the owner, added in BF16.
    OwnerReduce { owner: u8 },
}

impl Combine {
    /// `CUTEAFD_TP2_EXCHANGE=bf16|f32` overrides the fused exchange dtype.
    pub(crate) fn fused_from_env(default: ExchangeDtype) -> Result<Self> {
        let exchange = match std::env::var("CUTEAFD_TP2_EXCHANGE").ok().as_deref() {
            None | Some("") => default,
            Some("bf16") => ExchangeDtype::Bf16,
            Some("f32") => ExchangeDtype::F32,
            Some(other) => anyhow::bail!("CUTEAFD_TP2_EXCHANGE must be bf16 or f32, got {other:?}"),
        };
        Ok(Self::FusedAllReduce { exchange })
    }
}

/// The fused all-reduce's two kernels on one rank.
pub(crate) struct FusedCombine<'a> {
    kernels: RtxTp2Combine<'a>,
    pub exchange: ExchangeDtype,
}

impl<'a> FusedCombine<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, exchange: ExchangeDtype) -> Result<Self> {
        Ok(Self {
            kernels: library.rtx_tp2_combine()?,
            exchange,
        })
    }

    /// Bytes of one rank's exchange payload for `rows` rows of `hidden`.
    pub(crate) fn payload_bytes(&self, rows: usize, hidden: usize) -> usize {
        rows * hidden * self.exchange.element_bytes()
    }

    /// This rank's payload: its routed partial (FP32, or null for a layer
    /// whose routed experts are not RTX-resident) plus its shared half.
    /// # Safety
    /// As [`RtxTp2Combine::partial`], on the current device.
    pub(crate) unsafe fn partial(
        &self,
        routed: *const f32,
        shared: *const u16,
        out: *mut c_void,
        rows: usize,
        hidden: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe {
            self.kernels
                .partial(routed, shared, out, rows * hidden, self.exchange, stream)
        }
    }

    /// Both payloads summed in rank order into BF16 `out`.
    /// # Safety
    /// As [`RtxTp2Combine::sum`], on the current device.
    pub(crate) unsafe fn sum(
        &self,
        rank0: *const c_void,
        rank1: *const c_void,
        out: *mut u16,
        rows: usize,
        hidden: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe {
            self.kernels
                .sum(rank0, rank1, out, rows * hidden, self.exchange, stream)
        }
    }
}

/// One-way FP32 token-partial reduction into an existing owner's BF16 plane.
/// Bind once per lane and retain through graph retirement. This borrows the
/// ordinary hidden receive slot; shared FFN planes never enter this reduction.
#[allow(dead_code)] // Bound by families as their owner-only front ends land.
pub(crate) struct TokenOwnerReduce<'a> {
    exchange_streams: [std::rc::Rc<Stream<'a>>; 2],
    received: [*mut c_void; 2],
    capacity: usize,
    streams: [std::rc::Rc<Stream<'a>>; 2],
    produced: [Event<'a>; 2],
    completed: [Event<'a>; 2],
    kernels: RtxTp2Combine<'a>,
    slot: usize,
    rows: usize,
    width: usize,
}

#[allow(dead_code)]
impl<'a> TokenOwnerReduce<'a> {
    pub(crate) fn new(exchange: &crate::shared::peer_split::PeerExchange<'a>,
        exchange_streams: [std::rc::Rc<Stream<'a>>; 2],
        streams: [std::rc::Rc<Stream<'a>>; 2], slot: usize, rows: usize, width: usize) -> Result<Self> {
        exchange.require_live()?;
        let library = streams[0].device.library;
        ensure!(std::ptr::eq(library, streams[1].device.library)
            && streams.iter().enumerate().all(|(rank, stream)| stream.device.id == exchange.device(rank)),
            "token owner reducer lane devices differ from exchange");
        ensure!(exchange_streams.iter().enumerate().all(|(rank, stream)|
            std::ptr::eq(library, stream.device.library) && stream.device.id == exchange.device(rank)
                && stream.raw == exchange.stream(rank)), "token owner reducer exchange streams differ");
        ensure!(slot != crate::shared::peer_split::DIRECT, "token owner reducer needs a receive slot");
        let received = [exchange.recv(0, slot)?, exchange.recv(1, slot)?];
        let bytes = token_extent(rows, width)?.1;
        ensure!(bytes <= exchange.slot_bytes(), "token owner reducer exceeds admitted receive slot");
        for rank in 0..2 {
            token_range(CuteafdDeviceBuffer { ptr: received[rank], bytes: exchange.slot_bytes(),
                device_id: exchange.device(rank), flags: 0 }, bytes, exchange.device(rank), 16)?;
        }
        let devices = streams.each_ref().map(|stream| stream.device);
        Ok(Self { exchange_streams, received, capacity: exchange.slot_bytes(), streams,
            produced: [Event::new(devices[0])?, Event::new(devices[1])?],
            completed: [Event::new(devices[0])?, Event::new(devices[1])?],
            kernels: library.rtx_tp2_combine()?, slot, rows, width })
    }

    /// # Safety
    /// All planes remain live through the enclosing submission/graph lifetime.
    /// Each FP32 partial is final on this binding's exact lane stream. Subsequent
    /// consumers and all other users of the borrowed slot follow those streams;
    /// the binding outlives graph replay and is dropped only after terminal abort
    /// on failure. No caller may enqueue a competing use between these operations.
    pub(crate) unsafe fn reduce(&mut self, scope: &super::pair::SubmissionScope<'_>,
        exchange: &crate::shared::peer_split::PeerExchange<'a>, owner: usize,
        rows: usize, partials: [CuteafdDeviceBuffer; 2], output: CuteafdDeviceBuffer,
        dtype: PartialDtype, layout: PartialLayout) -> Result<()> {
        ensure!(rows <= self.rows, "token owner reducer exceeds admitted rows");
        ensure!(exchange.slot_bytes() == self.capacity, "token owner reducer exchange capacity changed");
        for rank in 0..2 {
            ensure!(exchange.device(rank) == self.exchange_streams[rank].device.id
                && exchange.stream(rank) == self.exchange_streams[rank].raw
                && exchange.recv(rank, self.slot)? == self.received[rank],
                "token owner reducer exchange binding changed");
        }
        let width = self.width;
        token_owner_with(scope, &mut TokenOwnerBinding { reducer: self, exchange },
            owner, rows, width, partials, output, dtype, layout)
    }
}

impl Drop for TokenOwnerReduce<'_> {
    fn drop(&mut self) {
        // Binding teardown only: graph nodes may still name these events.
        for stream in self.streams.iter().chain(&self.exchange_streams) {
            stream.drain_or_abort("token owner reducer binding");
        }
    }
}

struct TokenOwnerBinding<'b, 'a> {
    reducer: &'b mut TokenOwnerReduce<'a>,
    exchange: &'b crate::shared::peer_split::PeerExchange<'a>,
}

trait TokenOwnerProtocol {
    fn require_live(&self) -> Result<()>;
    fn device(&self, rank: usize) -> i32;
    fn capacity(&self) -> usize;
    fn received(&self, owner: usize) -> Result<*mut c_void>;
    fn producers(&mut self) -> Result<()>;
    fn publish(&self, remote: usize, source: *const c_void, bytes: usize) -> Result<()>;
    fn land(&self, owner: usize) -> Result<()>;
    fn sum(&self, owner: usize, ranks: [*const c_void; 2], output: *mut u16, count: usize) -> Result<()>;
    fn consumers(&mut self, owner: usize) -> Result<()>;
}

impl TokenOwnerProtocol for TokenOwnerBinding<'_, '_> {
    fn require_live(&self) -> Result<()> { self.exchange.require_live() }
    fn device(&self, rank: usize) -> i32 { self.exchange.device(rank) }
    fn capacity(&self) -> usize { self.exchange.slot_bytes() }
    fn received(&self, owner: usize) -> Result<*mut c_void> { self.exchange.recv(owner, self.reducer.slot) }
    fn producers(&mut self) -> Result<()> {
        for rank in 0..2 { self.reducer.produced[rank].record(&self.reducer.streams[rank])?; }
        // Both exact lane streams include their previous slot consumers. Join
        // both before either exchange can publish into that slot again.
        for at in 0..2 {
            for event in &self.reducer.produced {
                self.exchange.on(at, || {
                    // SAFETY: retained event was recorded on the exact producer.
                    unsafe { self.reducer.streams[at].device.library.cuda_stream_wait_event(self.exchange.stream(at), event.raw) }
                })?;
            }
        }
        Ok(())
    }
    fn publish(&self, remote: usize, source: *const c_void, bytes: usize) -> Result<()> {
        self.exchange.push(remote, self.reducer.slot, source, bytes)
    }
    fn land(&self, owner: usize) -> Result<()> { self.exchange.wait(owner, self.reducer.slot) }
    fn sum(&self, owner: usize, ranks: [*const c_void; 2], output: *mut u16, count: usize) -> Result<()> {
        self.exchange.on(owner, || {
            // SAFETY: validated disjoint token planes; producer joins and peer
            // landing precede the native fixed-rank FP32 sum on this stream.
            unsafe { self.reducer.kernels.sum(ranks[0], ranks[1], output, count,
                RtxPartialDtype::F32, self.exchange.stream(owner)) }
        })
    }
    fn consumers(&mut self, owner: usize) -> Result<()> {
        let event = &self.reducer.completed[owner];
        self.exchange.on(owner, || {
            // SAFETY: this event belongs to the chosen owner and survives replay.
            unsafe { self.reducer.streams[owner].device.library.cuda_event_record(event.raw, self.exchange.stream(owner)) }
        })?;
        for stream in &self.reducer.streams {
            stream.device.run(|| {
                // SAFETY: both exact consumers follow this reduction. The remote
                // join also orders the next sender after the owner's last read.
                unsafe { stream.device.library.cuda_stream_wait_event(stream.raw, event.raw) }
            })?;
        }
        Ok(())
    }
}

fn token_extent(rows: usize, width: usize) -> Result<(usize, usize)> {
    let count = rows.checked_mul(width).filter(|&count| rows > 0 && width > 0 && count > 0)
        .ok_or_else(|| anyhow::anyhow!("invalid token owner reduction extent"))?;
    let bytes = count.checked_mul(4).ok_or_else(|| anyhow::anyhow!("token partial extent overflows"))?;
    ensure!(count <= 4096 * 16384 && bytes % 16 == 0, "token reduction extent violates native ABI");
    Ok((count, bytes))
}

fn token_range(buffer: CuteafdDeviceBuffer, bytes: usize, device: i32, alignment: usize) -> Result<(usize, usize)> {
    ensure!(!buffer.ptr.is_null() && buffer.bytes >= bytes && buffer.device_id == device,
        "token owner reduction plane extent/device mismatch");
    let start = buffer.ptr as usize;
    ensure!(start % alignment == 0, "token owner reduction plane alignment");
    Ok((start, start.checked_add(bytes).ok_or_else(|| anyhow::anyhow!("token plane address overflows"))?))
}

fn token_disjoint(a: (usize, usize), b: (usize, usize)) -> Result<()> {
    ensure!(a.1 <= b.0 || b.1 <= a.0, "token owner reduction planes overlap");
    Ok(())
}

fn token_owner_with(_scope: &super::pair::SubmissionScope<'_>, protocol: &mut impl TokenOwnerProtocol,
    owner: usize, rows: usize, width: usize, partials: [CuteafdDeviceBuffer; 2],
    output: CuteafdDeviceBuffer, dtype: PartialDtype, layout: PartialLayout) -> Result<()> {
    protocol.require_live()?;
    ensure!(owner < 2 && dtype == PartialDtype::F32 && layout == PartialLayout::Tokens,
        "token owner reduction requires FP32 token sums and a valid owner");
    ensure!(protocol.device(0) != protocol.device(1), "token owner reduction needs distinct devices");
    let (count, bytes) = token_extent(rows, width)?;
    ensure!(bytes <= protocol.capacity(), "token owner reduction exceeds receive slot");
    let local = token_range(partials[owner], bytes, protocol.device(owner), 16)?;
    token_range(partials[1 - owner], bytes, protocol.device(1 - owner), 16)?;
    let out = token_range(output, count * 2, protocol.device(owner), 2)?;
    let received = protocol.received(owner)?;
    let staging = token_range(CuteafdDeviceBuffer { ptr: received, bytes: protocol.capacity(),
        device_id: protocol.device(owner), flags: 0 }, bytes, protocol.device(owner), 16)?;
    token_disjoint(local, out)?;
    token_disjoint(local, staging)?;
    token_disjoint(out, staging)?;
    protocol.producers()?;
    protocol.publish(1 - owner, partials[1 - owner].ptr, bytes)?;
    protocol.land(owner)?;
    let mut ranks = partials.map(|buffer| buffer.ptr.cast_const());
    ranks[1 - owner] = received.cast_const();
    protocol.sum(owner, ranks, output.ptr.cast(), count)?;
    protocol.consumers(owner)
}

/// Owner-side routed reduction. Shared contributions remain a separate reduction;
/// the caller adds the two BF16 results only after both complete.
pub(crate) struct OwnerReduce<'a> {
    transfer: PeerTransfer<'a>,
    local_ready: Event<'a>,
    staging: Allocation<'a>,
    output: Allocation<'a>,
    reducer: V41Tp2ExpertReducer<'a>,
    capacity: u32,
    local_rank: u8,
    geometry: cuteafd_core::ExpertGeometry,
}

impl<'a> OwnerReduce<'a> {
    pub(crate) fn workspace_bytes(
        capacity: u32,
        geometry: cuteafd_core::ExpertGeometry,
    ) -> Result<usize> {
        V41Tp2ExpertReducer::require_owner_geometry(geometry)?;
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid owner reduction capacity"
        );
        Ok(usize::try_from(cuteafd_loader::placement::families::deepseek_v41::owner_workspace(
            capacity as u64, geometry.hidden as u64, geometry.topk as u64)?)?)
    }

    pub(crate) fn new(
        devices: [Device<'a>; 2],
        policy: Combine,
        capacity: u32,
        geometry: cuteafd_core::ExpertGeometry,
    ) -> Result<Self> {
        Self::workspace_bytes(capacity, geometry)?;
        let Combine::OwnerReduce { owner } = policy else {
            anyhow::bail!("owner reducer requires explicit OwnerReduce policy")
        };
        ensure!(
            owner < 2
                && devices[0].id != devices[1].id
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid owner reduction devices"
        );
        let local = devices[owner as usize];
        Ok(Self {
            transfer: PeerTransfer::new(devices[1 - owner as usize], local)?,
            local_ready: Event::new(local)?,
            staging: Allocation::new(
                local,
                capacity as usize * geometry.hidden as usize * geometry.topk as usize * 4,
            )?,
            output: Allocation::new(local, capacity as usize * geometry.hidden as usize * 2)?,
            reducer: local.run(|| local.library.v41_tp2_expert_reducer())?,
            capacity,
            local_rank: owner,
            geometry,
        })
    }

    /// # Safety
    /// Both rank outputs are immutable, live and ordered on their producer
    /// streams through completion or cancellation drainage. Deferred callers
    /// retain them until the enclosing stage chain completes.
    pub(crate) async unsafe fn reduce(
        &mut self,
        local: &Allocation<'a>,
        remote: &Allocation<'a>,
        local_producer: &Stream<'a>,
        remote_producer: &Stream<'a>,
        rows: u32,
        dtype: PartialDtype,
        layout: PartialLayout,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            (1..=self.capacity).contains(&rows),
            "owner reduction exceeds capacity"
        );
        let columns = match layout {
            PartialLayout::Tokens => 1,
            PartialLayout::Routes { topk } => {
                ensure!(
                    topk == self.geometry.topk as usize,
                    "owner route planes differ from admitted geometry"
                );
                topk
            }
        };
        ensure!(
            dtype != PartialDtype::Bf16 || matches!(layout, PartialLayout::Routes { .. }),
            "BF16 owner reduction requires route planes"
        );
        let bytes = rows as usize
            * self.geometry.hidden as usize
            * columns
            * if dtype == PartialDtype::F32 { 4 } else { 2 };
        ensure!(
            local.device.id == self.output.device.id
                && local.buffer.bytes >= bytes
                && local_producer.device.id == local.device.id
                && std::ptr::eq(local.device.library, self.output.device.library)
                && std::ptr::eq(local_producer.device.library, self.output.device.library),
            "owner reduction local output/producer mismatch"
        );
        let started = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG)
            .then(std::time::Instant::now);
        if !crate::shared::memory::chain::deferred() {
            remote_producer.wait().await?;
        }
        let producer_us = started.map(|start| start.elapsed().as_micros() as u64);
        self.local_ready.record(local_producer)?;
        let ready = &self.local_ready;
        let output = &self.output;
        let reducer = &self.reducer;
        let local_rank = self.local_rank;
        // SAFETY: the caller retains rank outputs; transfer retains the callback
        // and drains copy/reduction on any partial enqueue or cancellation.
        unsafe {
            self.transfer
                .copy_then(
                    remote,
                    &mut self.staging,
                    remote_producer,
                    bytes,
                    |peer, stream| {
                        output
                            .device
                            .library
                            .cuda_stream_wait_event(stream, ready.raw)?;
                        let (rank0, rank1) = if local_rank == 0 {
                            (local.buffer, peer)
                        } else {
                            (peer, local.buffer)
                        };
                        if dtype == PartialDtype::Bf16 {
                            reducer.reduce_bf16_routes(rank0, rank1, output.buffer, rows, stream)
                        } else {
                            reducer.reduce(
                                rank0,
                                rank1,
                                output.buffer,
                                rows,
                                layout == PartialLayout::Tokens,
                                stream,
                            )
                        }
                    },
                )
                .await?;
        }
        if let Some(started) = started {
            tracing::debug!(target: "cuteafd::timing", rows, token_sums=layout == PartialLayout::Tokens,
                ?layout, producer_us=producer_us.unwrap(),
                copy_reduce_us=started.elapsed().as_micros() as u64-producer_us.unwrap(),
                "TP2 routed completion");
        }
        let mut output = self.output.buffer;
        output.bytes = rows as usize * self.geometry.hidden as usize * 2;
        Ok(output)
    }
}

#[cfg(test)]
mod token_owner_tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use crate::shared::experts::rtx::pair::{submit_scoped, TerminalStepError};

    struct Protocol {
        owner: usize,
        capacity: usize,
        calls: RefCell<Vec<&'static str>>,
        fail: Option<usize>,
        retired: Cell<bool>,
        operands: [Vec<f32>; 2],
        result: RefCell<Vec<u16>>,
    }

    impl Protocol {
        fn new(owner: usize) -> Self {
            Self { owner, capacity: 64, calls: RefCell::new(Vec::new()), fail: None,
                retired: Cell::new(false), operands: [vec![1.00390625, -2.0, 3.0, -0.5], vec![0.00390625, 0.5, -1.0, 0.25]],
                result: RefCell::new(Vec::new()) }
        }
        fn call(&self, name: &'static str) -> Result<()> {
            let index = self.calls.borrow().len();
            self.calls.borrow_mut().push(name);
            ensure!(self.fail != Some(index), "injected token reduction failure");
            Ok(())
        }
        fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
            std::array::from_fn(|rank| CuteafdDeviceBuffer { ptr: (0x1000 + rank * 0x1000) as *mut c_void,
                bytes: self.capacity, device_id: self.device(rank), flags: 0 })
        }
        fn output(&self) -> CuteafdDeviceBuffer {
            CuteafdDeviceBuffer { ptr: 0x4000usize as *mut c_void, bytes: self.capacity,
                device_id: self.device(self.owner), flags: 0 }
        }
    }

    impl TokenOwnerProtocol for Protocol {
        fn require_live(&self) -> Result<()> { ensure!(!self.retired.get(), "retired reducer"); Ok(()) }
        fn device(&self, rank: usize) -> i32 { [3, 7][rank] }
        fn capacity(&self) -> usize { self.capacity }
        fn received(&self, owner: usize) -> Result<*mut c_void> {
            assert_eq!(owner, self.owner);
            Ok(0x3000usize as *mut c_void)
        }
        fn producers(&mut self) -> Result<()> { self.call("join both exact lane producers") }
        fn publish(&self, remote: usize, source: *const c_void, bytes: usize) -> Result<()> {
            assert_eq!(remote, 1 - self.owner);
            assert_eq!(source, self.inputs()[remote].ptr.cast_const());
            assert!(bytes <= self.capacity);
            self.call("remote-only push")
        }
        fn land(&self, owner: usize) -> Result<()> {
            assert_eq!(owner, self.owner);
            self.call("owner wait")
        }
        fn sum(&self, owner: usize, ranks: [*const c_void; 2], output: *mut u16, count: usize) -> Result<()> {
            assert_eq!(owner, self.owner);
            let mut expected = self.inputs().map(|buffer| buffer.ptr.cast_const());
            expected[1 - owner] = 0x3000usize as *const c_void;
            assert_eq!(ranks, expected, "native operands must remain rank0 then rank1");
            assert_eq!(output, self.output().ptr.cast());
            self.call("rank-ordered FP32 sum")?;
            self.result.replace((0..count).map(|i| bf16(self.operands[0][i] + self.operands[1][i])).collect());
            Ok(())
        }
        fn consumers(&mut self, owner: usize) -> Result<()> {
            assert_eq!(owner, self.owner);
            self.call("join both exact lane consumers")
        }
    }

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn reduce(protocol: &mut Protocol) -> Result<()> {
        let inputs = protocol.inputs();
        let output = protocol.output();
        let owner = protocol.owner;
        submit_scoped(|| Ok(()), || Ok(()), |scope|
            token_owner_with(scope, protocol, owner, 1, 4, inputs, output,
                PartialDtype::F32, PartialLayout::Tokens))
    }

    #[test]
    fn token_sum_is_one_way_fixed_rank_and_rounds_only_after_adding() -> Result<()> {
        for owner in 0..2 {
            let mut protocol = Protocol::new(owner);
            reduce(&mut protocol)?;
            assert_eq!(*protocol.calls.borrow(), ["join both exact lane producers", "remote-only push",
                "owner wait", "rank-ordered FP32 sum", "join both exact lane consumers"]);
            assert_eq!(*protocol.result.borrow(), [bf16(1.0078125), bf16(-1.5), bf16(2.0), bf16(-0.25)]);
            let rounded_first = f32::from_bits((bf16(1.00390625) as u32) << 16) + 0.00390625;
            assert_ne!(bf16(1.0078125), bf16(rounded_first), "fixture distinguishes premature BF16 narrowing");
        }
        assert_eq!(token_extent(4096, 3584)?.1, 4096 * 7168 * 2,
            "Kimi FP32 latent payload borrows the existing BF16 hidden slot exactly");
        Ok(())
    }

    #[test]
    fn token_reduction_rejects_invalid_planes_before_any_enqueue() {
        for case in 0..19 {
            let mut protocol = Protocol::new(0);
            let mut inputs = protocol.inputs();
            let mut output = protocol.output();
            let (mut owner, mut rows, mut width) = (0, 1, 4);
            let (mut dtype, mut layout) = (PartialDtype::F32, PartialLayout::Tokens);
            match case {
                0 => owner = 2,
                1 => rows = 0,
                2 => width = 0,
                3 => width = usize::MAX,
                4 => dtype = PartialDtype::Bf16,
                5 => layout = PartialLayout::Routes { topk: 16 },
                6 => inputs[0].bytes = 15,
                7 => inputs[1].device_id = 3,
                8 => inputs[1].ptr = std::ptr::null_mut(),
                9 => output.ptr = inputs[0].ptr,
                10 => output.ptr = 0x3000usize as *mut c_void,
                11 => output.bytes = 3,
                12 => protocol.capacity = 15,
                13 => inputs[0].ptr = (usize::MAX - 15) as *mut c_void,
                14 => inputs[1].ptr = 0x2004usize as *mut c_void,
                15 => width = 2,
                16 => { width = 4096 * 16384 + 4; protocol.capacity = usize::MAX; },
                17 => inputs[0].ptr = 0x1004usize as *mut c_void,
                _ => output.ptr = 0x4001usize as *mut c_void,
            }
            let result = submit_scoped(|| Ok(()), || Ok(()), |scope|
                token_owner_with(scope, &mut protocol, owner, rows, width, inputs, output, dtype, layout));
            assert!(result.is_err(), "case {case}");
            assert!(protocol.calls.borrow().is_empty(), "case {case} queued work");
        }
    }

    #[test]
    fn token_reduction_slot_reuse_is_ordered_across_owners_and_requests() -> Result<()> {
        use crate::shared::peer_split::order::{check_submission_faults, Schedule};
        let mut schedule = Schedule::default();
        for _request in 0..3 {
            for owner in [0, 1, 1, 0] {
                let mut protocol = Protocol::new(owner);
                for _ in 0..3 { reduce(&mut protocol)?; }
                // This checker covers balanced native publication generations,
                // not CUDA event edges (covered by the four-stream model below).
                schedule.push(1 - owner, "token-partial", 0, "FP32 token push");
                schedule.wait(owner, "token-partial", 0, "owner landing before sum/consumer joins");
            }
            schedule.end_request();
        }
        check_submission_faults(&schedule).unwrap();
        Ok(())
    }

    #[test]
    fn token_reduction_event_joins_prevent_early_slot_and_sender_reuse() {
        #[derive(Clone, Copy)]
        enum Op { Produce(usize), WaitProduced(usize, usize), Push(usize), Land(usize),
            Sum(usize), Complete(usize), WaitComplete(usize) }
        #[derive(Clone, Default)]
        struct State {
            at: [usize; 4], produced: [[bool; 2]; 4], complete: [bool; 4],
            published: [bool; 4], landed: [bool; 4], consumed: [bool; 4],
        }
        fn visit(queues: &[Vec<Op>; 4], state: State, seen: &mut std::collections::HashSet<[usize; 4]>) {
            if !seen.insert(state.at) { return; }
            let mut progressed = false;
            for stream in 0..4 {
                let Some(op) = queues[stream].get(state.at[stream]) else { continue };
                let mut next = state.clone();
                match *op {
                    Op::Produce(step) => {
                        assert!(step == 0 || next.consumed[step - 1], "sender reused before owner sum");
                        next.produced[step][stream] = true;
                    }
                    Op::WaitProduced(step, rank) => if !next.produced[step][rank] { continue },
                    Op::Push(step) => {
                        assert!(step == 0 || next.consumed[step - 1], "receive slot overwritten before owner sum");
                        next.published[step] = true;
                    }
                    Op::Land(step) => {
                        if !next.published[step] { continue }
                        next.landed[step] = true;
                    }
                    Op::Sum(step) => {
                        assert!(next.landed[step] && next.produced[step] == [true; 2]);
                        next.consumed[step] = true;
                    }
                    Op::Complete(step) => { assert!(next.consumed[step]); next.complete[step] = true; }
                    Op::WaitComplete(step) => if !next.complete[step] { continue },
                }
                next.at[stream] += 1;
                progressed = true;
                visit(queues, next, seen);
            }
            assert!(progressed || (0..4).all(|stream| state.at[stream] == queues[stream].len()),
                "event graph deadlocked");
        }
        // Re-recorded CUDA events bind waits to the generation current at host
        // enqueue. Enumerate every legal device interleaving, including across
        // request boundaries, for both repeated and alternating owners.
        for owners in [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]] {
            let mut queues: [Vec<Op>; 4] = std::array::from_fn(|_| Vec::new());
            for (step, owner) in owners.into_iter().enumerate() {
                for rank in 0..2 {
                    queues[rank].extend([Op::Produce(step), Op::WaitComplete(step)]);
                    queues[rank + 2].extend([Op::WaitProduced(step, 0), Op::WaitProduced(step, 1)]);
                }
                queues[3 - owner].push(Op::Push(step));
                queues[2 + owner].extend([Op::Land(step), Op::Sum(step), Op::Complete(step)]);
            }
            visit(&queues, State::default(), &mut std::collections::HashSet::new());
        }
    }

    #[test]
    fn token_reduction_enqueue_failures_and_cancellation_use_submission_teardown() {
        for owner in 0..2 {
            for failure in 0..5 {
                for cancel in [false, true] {
                    let protocol = RefCell::new(Protocol::new(owner));
                    protocol.borrow_mut().fail = Some(failure);
                    let drained = Cell::new(0);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| submit_scoped(
                        || protocol.borrow().require_live(),
                        || { protocol.borrow().retired.set(true); drained.set(drained.get() + 1); Ok(()) },
                        |scope| {
                            let mut protocol = protocol.borrow_mut();
                            let inputs = protocol.inputs();
                            let output = protocol.output();
                            let result = token_owner_with(scope, &mut *protocol, owner, 1, 4,
                                inputs, output, PartialDtype::F32, PartialLayout::Tokens);
                            if cancel { panic!("injected cancellation after enqueue prefix"); }
                            result
                        })));
                    if cancel { assert!(result.is_err()); }
                    else { assert!(result.unwrap().unwrap_err().downcast_ref::<TerminalStepError>().is_some()); }
                    assert_eq!(drained.get(), 1);
                    assert!(protocol.borrow().retired.get());
                    assert!(submit_scoped(|| protocol.borrow().require_live(),
                        || panic!("retired submission drained twice"), |_| Ok(())).is_err());
                }
            }
        }
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;
    use cuteafd_core::ExpertGeometry;

    #[test]
    fn owner_workspace_matches_native_abi_and_rejects_unsupported_geometry() -> Result<()> {
        let geometry = ExpertGeometry::DEEPSEEK_V41;
        assert_eq!(
            OwnerReduce::workspace_bytes(16, geometry)?,
            16 * 5120 * (6 * 4 + 2)
        );
        for unsupported in [
            ExpertGeometry::DEEPSEEK_V4_FLASH,
            ExpertGeometry::GLM5,
            ExpertGeometry {
                topk: 3,
                ..geometry
            },
        ] {
            assert!(OwnerReduce::workspace_bytes(16, unsupported).is_err());
        }
        assert!(OwnerReduce::workspace_bytes(0, geometry).is_err());
        assert!(OwnerReduce::workspace_bytes(4097, geometry).is_err());
        Ok(())
    }
}
