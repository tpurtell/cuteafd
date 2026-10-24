//! Ordinary six-rank Spark exchange in latent space. Shared staging, transport,
//! intake and ordered FP32 plane reduction are reused without a device proxy.
use super::device::LocalLane;
use crate::shared::{
    experts::staging::RouteStaging,
    memory::{HostAllocation, device::{Event, Stream}},
    spark_intake::SparkLink,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertGeometry;
use cuteafd_ffi::KimiExpertInput;
use cuteafd_loader::placement::{Demand, Placement};
use cuteafd_transport::{expert::{SparkWorld, SparkExpertWave}, ExpertProtocolV2Request,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
    TcpTransportConfig};
use cuteafd_transport::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use std::{net::SocketAddr, rc::Rc, sync::Arc, time::Duration};

const RANKS: usize = 6;
const LATENT: usize = 3584;
const TOPK: usize = 16;

/// The intake owns six full-capacity BF16 planes on the dispatch device.
/// Pinned route storage is host memory; all device planes precede admission.
/// `rows` is the lane's whole in-flight span: `waves` units, one return slot each.
pub(crate) fn demand(rows: usize, waves: usize, lane: usize) -> Result<Demand> {
    demand_with(rows, waves, None, lane)
}

pub(crate) fn demand_with(rows: usize, waves: usize, unit: Option<usize>, lane: usize) -> Result<Demand> {
    ensure!((1..=cuteafd_loader::families::kimi_k3::allocation::LANE_ROWS_MAX).contains(&rows), "Kimi Spark row capacity");
    let shape = cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(rows, waves, unit)?;
    Ok(cuteafd_loader::families::kimi_k3::allocation::spark_demand(shape, lane))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ticket { layer: usize, rows: usize, sequence: u64 }

/// A dispatched wave and the latent output rows its reduce writes.
struct Outstanding { ticket: Ticket, wave: SparkExpertWave, output: cuteafd_ffi::CuteafdDeviceBuffer }

/// Engine-retained wave owner; tickets never own the transport or copy sources.
/// The main engine thread owns the transport. Drop completes copies before
/// releasing pinned storage; terminal QP ownership retains unproved sources.
pub(crate) struct RoutedLane<'a> {
    stream: Rc<Stream<'a>>,
    link: Option<SparkLink<'a>>,
    runtime: tokio::runtime::Runtime,
    library_owner: Arc<cuteafd_ffi::NativeLibrary>,
    terminal: bool,
    staging: HostAllocation<'a>,
    ready: Event<'a>,
    capacity: usize,
    /// Most outstanding waves; each holds one intake return slot of `unit` rows.
    waves: usize,
    unit: usize,
    sequence: std::cell::Cell<u64>,
    /// Dispatched waves in dispatch order, received oldest first.
    outstanding: std::collections::VecDeque<Outstanding>,
}

impl Drop for RoutedLane<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Kimi routed lane copies and consumers");
        let link = self.link.take();
        crate::shared::decode_graph::fatal_drain(
            self.stream.device.run(|| { drop(link); Ok(()) }), "Kimi routed lane device-scoped release");
    }
}

impl<'a> RoutedLane<'a> {
    pub(super) fn connect(library_owner: &'a Arc<cuteafd_ffi::NativeLibrary>,
        local: &LocalLane<'a>, placement: &Placement, world: SparkWorld,
        peers: Vec<SocketAddr>, config: TcpTransportConfig, waves: usize, unit: usize) -> Result<Self> {
        ensure!(std::ptr::eq(library_owner.as_ref(), local.stream().device.library),
            "Kimi Spark library owner differs from admitted device");
        ensure!(cuteafd_core::expert_geometry() == ExpertGeometry::KIMI_K3,
            "Kimi Spark helpers require the process latent expert geometry before native loading");
        ensure!(peers.len() == RANKS, "Kimi routed exchange requires six real Spark ranks");
        let charge = demand_with(local.rows(), waves, Some(unit), local.lane_index())?;
        let charges = placement.items[1].iter().filter(|item| item.group == charge.group).collect::<Vec<_>>();
        ensure!(charges.len() == 1 && charges[0].category == charge.category
            && charges[0].basis == charge.basis && charges[0].bytes == charge.bytes,
            "Kimi Spark intake differs from accepted admission");
        let stream = local.stream().clone();
        // Each wave is one unit's rows: staging and transport serve a unit.
        let layout = RouteStaging::new(unit, TOPK, KimiExpertInput::ROW_BYTES)?;
        let library = library_owner.as_ref();
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        let (mut link, staging, ready) = stream.device.run(|| {
            library.v41_compact_reducer()?.require_plane_reduction(RANKS as u32)?;
            let mut link = SparkLink::new_waves(library, world, &peers, u32::try_from(unit)?, config, LATENT * 2,
                waves, u32::try_from(unit)?)?;
            link.enable_terminal_ownership(Arc::clone(library_owner))?;
            Ok((link, HostAllocation::new(library, layout.bytes)?, Event::new(stream.device)?))
        })?;
        ensure!(link.world_size() == RANKS, "Kimi Spark transport world differs from admission");
        if let Some(layer) = placement.layers.iter().position(|assignment|
            assignment.experts == cuteafd_loader::placement::ExpertHome::Spark) {
            stream.device.run(|| link.bootstrap(&bootstrap_request(layer, unit)?))?;
            ensure!(link.transport.installed_grouping()?.len() == RANKS,
                "Kimi bootstrap requires installed grouping from all six ranks");
        }
        Ok(Self { stream, link: Some(link), runtime, library_owner: Arc::clone(library_owner),
            terminal: false, staging, ready, capacity: unit, waves, unit, sequence: std::cell::Cell::new(0),
            outstanding: std::collections::VecDeque::new() })
    }

    pub(crate) fn installed_grouping(&self)
        -> Result<Vec<cuteafd_transport::worker_selection::InstalledExpertGrouping>> {
        self.link.as_ref().context("Kimi Spark link retired")?.transport.installed_grouping()
    }

    /// Checked QP quiescence precedes copy drainage and owner release.
    pub(crate) fn quiesce_links(&mut self) -> Result<()> {
        self.stream.device.run(|| {
            if let Some(link) = &mut self.link { link.terminal_quiesce()?; }
            Ok(())
        })?;
        self.terminal = true;
        Ok(())
    }

    pub(crate) fn drain_copies(&self) -> Result<()> {
        self.stream.device.run(|| match &self.link {
            Some(link) => link.intake.terminal_drain_copies(),
            None => Ok(()),
        })
    }

    pub(crate) fn release_payloads(&mut self) -> Result<()> {
        ensure!(self.terminal, "Kimi Spark sources are not terminal");
        self.drain_copies()?;
        self.stream.device.run(|| {
            if let Some(link) = &mut self.link { link.terminal_release()?; }
            drop(self.link.take());
            Ok(())
        })?;
        self.outstanding.clear();
        Ok(())
    }

    fn check(&self, local: &LocalLane<'_>, rows: usize) -> Result<()> {
        ensure!(self.link.is_some() && !self.terminal && Rc::ptr_eq(&self.stream, local.stream())
            && std::ptr::eq(self.library_owner.as_ref(), self.stream.device.library)
            && (1..=self.capacity).contains(&rows),
            "Kimi Spark wave differs from retained lane stream or capacity");
        Ok(())
    }

    /// Route copies precede the shared FFN on this stream. Fence only the copy
    /// event so the host can parse ordinary protocol rows while the FFN runs.
    pub(crate) fn begin(&mut self, local: &mut LocalLane<'a>, layer: usize, rows: usize,
        kind: ExpertV2SourceKind, routes: cuteafd_ffi::KimiRoutes,
        wire: cuteafd_ffi::CuteafdDeviceBuffer,
        probe: Option<&mut crate::shared::draft::probe::RoundProbe<'a>>) -> Result<Ticket> {
        self.check(local, rows)?;
        ensure!(self.outstanding.len() < self.waves, "Kimi main link already has {} outstanding waves", self.waves);
        ensure!(rows <= self.unit, "Kimi wave of {rows} rows exceeds its {}-row return slot", self.unit);
        let sequence = self.sequence.get().checked_add(1).context("Kimi Spark request sequence overflow")?;
        let layout = RouteStaging::new(rows, TOPK, KimiExpertInput::ROW_BYTES)?;
        self.stream.device.run(|| {
            // SAFETY: LocalLane and this engine-retained staging owner survive
            // partial enqueue failures through whole-submission teardown.
            unsafe { layout.enqueue(self.stream.device.library, self.staging.buffer,
                [routes.ids, routes.weights, wire], self.stream.raw) }
        })?;
        self.ready.record(&self.stream)?;
        local.ffn(layer, rows)?;
        self.stream.device.run(|| {
            // SAFETY: ordered copy fence, not a compute drain; this event
            // covers only copies before the shared FFN;
            // pinned staging remains exclusively owned until this fence.
            unsafe { self.stream.device.library.cuda_event_synchronize(self.ready.raw) }
        })?;
        if let Some(probe) = probe { probe.staged(layer, &self.staging.bytes()[..layout.route_bytes]); }
        let request = request(layer, rows, sequence, kind, &layout, self.staging.bytes())?;
        let ticket = Ticket { layer, rows, sequence };
        self.sequence.set(sequence);
        let output = local.buffer("expert combined")?;
        let link = self.link.as_mut().context("Kimi Spark lane retired")?;
        let wave = self.stream.device.run(|| link.dispatch(&request))?;
        self.outstanding.push_back(Outstanding { ticket, wave, output });
        Ok(ticket)
    }

    pub(crate) fn has_link(&self) -> bool { self.link.is_some() }

    /// Runs `body` with this lane's link driven by a transport worker (a
    /// prefill call): [`Self::begin_wave`] and [`Self::finish_wave`] then send
    /// and land through it, never waiting for the GPU on this thread. The
    /// link is lent to the worker for the call and restored after it (also
    /// on error, so terminal teardown still finds it).
    pub(crate) fn with_waves_mut<T>(&mut self,
        body: impl FnOnce(&mut Self, &mut crate::shared::spark_waves::Waves<'_, 'a>) -> Result<T>) -> Result<T> {
        ensure!(self.outstanding.is_empty() && !self.terminal, "Kimi prefill transport needs an idle lane");
        let mut link = self.link.take().context("Kimi Spark lane retired")?;
        let bytes = RouteStaging::new(self.unit, TOPK, KimiExpertInput::ROW_BYTES)?.bytes;
        let device = self.stream.device;
        let result = device.run(|| link.with_waves(bytes, |waves| body(self, waves)));
        self.link = Some(link);
        result
    }

    /// [`Self::begin`] for a prefill wave on the link's transport worker: the
    /// route copies queue into the wave slot's staging ahead of the shared
    /// FFN; the worker builds and posts once they complete.
    pub(crate) fn begin_wave(&self, waves: &mut crate::shared::spark_waves::Waves<'_, '_>, local: &mut LocalLane<'a>,
        layer: usize, rows: usize, kind: ExpertV2SourceKind, routes: cuteafd_ffi::KimiRoutes,
        wire: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<crate::shared::spark_waves::WaveTicket> {
        ensure!(Rc::ptr_eq(&self.stream, local.stream()) && (1..=self.unit).contains(&rows),
            "Kimi Spark wave differs from retained lane stream or unit");
        ensure!((1..93).contains(&layer), "Kimi Spark request geometry");
        let layout = RouteStaging::new(rows, TOPK, KimiExpertInput::ROW_BYTES)?;
        let sequence = self.sequence.get().checked_add(1).context("Kimi Spark request sequence overflow")?;
        let ticket = self.stream.device.run(|| {
            // SAFETY: LocalLane retains these operands through whole-submission
            // teardown; the copies read them in stream order.
            unsafe { waves.send(rows, layout, [routes.ids, routes.weights, wire], self.stream.raw,
                Box::new(move |routes, wire| request_parts(layer, rows, sequence, kind, routes, wire))) }
        })?;
        self.sequence.set(sequence);
        local.ffn(layer, rows)?;
        Ok(ticket)
    }

    /// [`Self::finish`] for a prefill wave: the lane stream waits for the
    /// wave's planes and reduces them into the latent output rows.
    pub(crate) fn finish_wave(&self, waves: &mut crate::shared::spark_waves::Waves<'_, '_>, local: &LocalLane<'a>,
        ticket: crate::shared::spark_waves::WaveTicket) -> Result<()> {
        let output = local.buffer("expert combined")?;
        let (slot, rows) = (ticket.slot(), ticket.rows());
        self.stream.device.run(|| {
            waves.land(ticket, self.stream.raw)?;
            // SAFETY: `land` orders all six live planes before this exact lane
            // stream; the output is a disjoint admitted latent allocation.
            unsafe { waves.reduce(slot, std::ptr::null(), output.ptr.cast(), rows, self.stream.raw) }
        })
    }

    /// All six latent partials are reduced once with FP32 accumulation and one
    /// BF16 rounding. No hidden-space shared FFN participates in this reduction.
    pub(crate) fn finish(&mut self, local: &LocalLane<'a>, ticket: Ticket) -> Result<()> {
        self.check(local, ticket.rows)?;
        ensure!(self.outstanding.front().is_some_and(|front| front.ticket == ticket),
            "Kimi Spark completion differs from the oldest dispatched wave");
        let Outstanding { wave, output, .. } = self.outstanding.pop_front().expect("front wave present");
        let landing = local.buffer("expert combined")?;
        ensure!(output.ptr == landing.ptr && output.bytes == landing.bytes,
            "Kimi Spark wave lands into other rows than it was sent for");
        let link = self.link.as_mut().context("Kimi Spark lane retired")?;
        self.stream.device.run(|| {
            self.runtime.block_on(async {
                tokio::time::timeout(Duration::from_secs(120), link.receive(wave, ticket.rows, self.stream.raw))
                    .await.context("Kimi main-link receive timed out")?
            })?;
            // SAFETY: receive orders all six live intake planes before this exact
            // lane stream; output is a disjoint admitted latent allocation.
            unsafe { self.stream.device.library.v41_compact_reducer()?.reduce_planes(
                link.intake.pointers(), RANKS as u32, std::ptr::null(),
                output.ptr.cast(), ticket.rows as u32, self.stream.raw)?; }
            link.intake.consumed(self.stream.raw)
        })?;
        Ok(())
    }
}

/// Control bootstrap sizes rings for the widest real request; these rows
/// are never dispatched to the expert kernels.
fn bootstrap_request(layer: usize, rows: usize) -> Result<ExpertProtocolV2Request> {
    ensure!((1..93).contains(&layer) && (1..=4096).contains(&rows), "Kimi bootstrap geometry");
    let layout = RouteStaging::new(rows, TOPK, KimiExpertInput::ROW_BYTES)?;
    let staged = vec![0; layout.bytes];
    request(layer, rows, 1, ExpertV2SourceKind::Prefill, &layout, &staged)
}

fn request(layer: usize, rows: usize, sequence: u64, kind: ExpertV2SourceKind,
    layout: &RouteStaging, bytes: &[u8]) -> Result<ExpertProtocolV2Request> {
    request_parts(layer, rows, sequence, kind, layout.routes(bytes)?, layout.wire(bytes)?.to_vec().into())
}

fn request_parts(layer: usize, rows: usize, sequence: u64, kind: ExpertV2SourceKind,
    routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, wire: bytes::Bytes) -> Result<ExpertProtocolV2Request> {
    ensure!((1..93).contains(&layer) && (1..=4096).contains(&rows) && sequence > 0,
        "Kimi Spark request geometry or sequence");
    ensure!(routes.len() == rows * TOPK && routes.iter().all(|route|
        route.expert_id < 896 && route.gate_weight.is_finite() && route.gate_weight >= 0.0),
        "Kimi Spark route outside E896/K16 or invalid gate weight");
    let mut request = ExpertProtocolV2Request::new_bytes(sequence, 17, layer as u32, LATENT as u32,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: row as u64,
            source_kind: kind, source_request_id: sequence, token_position: row as u64,
            route_offset: (row * TOPK) as u32, route_count: TOPK as u32 }).collect(),
        routes, wire)?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_owner_keeps_main_link_and_exact_library_until_terminal_release() {
        let source = include_str!("spark.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("link: Option<SparkLink<'a>>"));
        assert!(source.contains("library_owner: Arc<cuteafd_ffi::NativeLibrary>"));
        assert!(source.contains("std::ptr::eq(library_owner.as_ref(), local.stream().device.library)"));
        assert!(source.contains("Arc::clone(library_owner)"));
        assert!(!source.contains("SparkLane"));
        assert!(!source.contains("submit_lane"));
        assert!(source.contains("link.dispatch(&request)"));
        assert!(source.contains("link.receive(wave, ticket.rows, self.stream.raw)"));
        assert!(source.find("link.enable_terminal_ownership").unwrap() < source.find("link.bootstrap").unwrap());
        assert!(source.find("link.bootstrap").unwrap() < source.find("link.transport.installed_grouping()").unwrap());
        let release = source.split("pub(crate) fn release_payloads").nth(1).unwrap()
            .split("fn check").next().unwrap();
        assert!(release.find("self.drain_copies()?").unwrap() < release.find("link.terminal_release()?").unwrap());
        assert!(release.find("link.terminal_release()?").unwrap() < release.find("self.link.take()").unwrap());
        assert!(release.contains("self.stream.device.run(||"));
        assert!(source.contains("link.terminal_quiesce()?"));
        assert!(!source.contains("not failure-safe"));
    }

    #[test]
    fn widest_bootstrap_matches_real_request_storage_without_consuming_a_wave() -> Result<()> {
        for rows in [1, 64, 4096] {
            let request = bootstrap_request(6, rows)?;
            assert_eq!(request.header.layer_id, 6);
            assert_eq!(request.header.row_count as usize, rows);
            assert_eq!(request.rows.len(), rows);
            assert_eq!(request.routes.len(), rows * TOPK);
            assert_eq!(request.hidden_payload.len(), rows * KimiExpertInput::ROW_BYTES);
            assert!(request.routes.iter().all(|route| route.expert_id == 0 && route.gate_weight == 0.));
        }
        assert!(bootstrap_request(0, 4096).is_err());
        assert!(bootstrap_request(1, 0).is_err());
        assert!(bootstrap_request(1, 4097).is_err());
        Ok(())
    }

    #[test]
    fn request_is_latent_fp8_k16_with_compact_six_rank_returns() -> Result<()> {
        for rows in [1, 16, 4096] {
            let layout = RouteStaging::new(rows, TOPK, KimiExpertInput::ROW_BYTES)?;
            let mut bytes = vec![0; layout.bytes];
            for route in 0..rows * TOPK {
                bytes[route * 4..][..4].copy_from_slice(&((route % 896) as u32).to_le_bytes());
                bytes[layout.route_bytes + route * 4..][..4].copy_from_slice(&0.0625f32.to_le_bytes());
            }
            bytes[2 * layout.route_bytes..].fill(0x71);
            let request = request(92, rows, 7, ExpertV2SourceKind::MtpVerify, &layout, &bytes)?;
            assert_eq!(request.header.hidden_dim, 3584);
            assert_eq!(request.header.hidden_row_stride_bytes as usize, KimiExpertInput::ROW_BYTES);
            assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Fp8E4m3Ue8m0K32);
            assert_eq!(request.header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
                EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16);
            assert_eq!(request.routes.len(), rows * TOPK);
            assert!(request.routes.iter().all(|route| route.gate_weight.to_bits() == 0.0625f32.to_bits()));
            assert_eq!(request.hidden_payload.len(), rows * KimiExpertInput::ROW_BYTES);
            assert!(request.hidden_payload.iter().all(|byte| *byte == 0x71));
            assert_eq!(request.rows.last().unwrap().route_offset, ((rows - 1) * TOPK) as u32);
            if rows % 2 == 0 && rows >= 128 {
                assert_eq!(demand(rows, 2, 1)?.gpu, 1);
                assert_eq!(demand(rows, 2, 1)?.bytes, (6 * rows * 3584 * 2) as u64);
            }
            assert_eq!(demand(rows.max(64).next_multiple_of(64), 1, 1)?.bytes,
                (6 * rows.max(64).next_multiple_of(64) * 3584 * 2) as u64);
        }
        Ok(())
    }

    #[test]
    fn malformed_routes_and_dense_layer_refuse_before_transport() -> Result<()> {
        let layout = RouteStaging::new(1, TOPK, KimiExpertInput::ROW_BYTES)?;
        let mut bytes = vec![0; layout.bytes];
        assert!(request(0, 1, 1, ExpertV2SourceKind::Decode, &layout, &bytes).is_err());
        assert!(request(93, 1, 1, ExpertV2SourceKind::Decode, &layout, &bytes).is_err());
        for weight in [f32::NAN, f32::INFINITY, -0.5] {
            bytes[layout.route_bytes..][..4].copy_from_slice(&weight.to_le_bytes());
            assert!(request(1, 1, 1, ExpertV2SourceKind::Decode, &layout, &bytes).is_err());
        }
        bytes[layout.route_bytes..][..4].fill(0);
        bytes[..4].copy_from_slice(&896u32.to_le_bytes());
        assert!(request(1, 1, 1, ExpertV2SourceKind::Decode, &layout, &bytes).is_err());
        assert!(demand(0, 1, 0).is_err());
        assert!(demand(8193, 1, 0).is_err());
        assert!(demand(8192, 2, 0).is_ok(), "two 4096-row units on one lane");
        assert!(demand(4096, 3, 0).is_ok() && demand(128, 3, 0).is_err());
        Ok(())
    }
}
