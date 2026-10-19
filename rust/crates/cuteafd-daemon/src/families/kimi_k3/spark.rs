//! Ordinary six-rank Spark exchange in latent space. Shared staging, transport,
//! intake and ordered FP32 plane reduction are reused without a device proxy.
use super::device::LocalLane;
use crate::shared::{
    experts::staging::RouteStaging,
    memory::{HostAllocation, device::{Event, Stream}},
    spark_intake::SparkLane,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, memory_layout::{Basis, Category}};
use cuteafd_ffi::KimiExpertInput;
use cuteafd_loader::placement::{Demand, Placement};
use cuteafd_transport::{expert::SparkWorld, ExpertProtocolV2Request,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
    TcpTransportConfig};
use cuteafd_transport::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use std::{net::SocketAddr, rc::Rc, sync::Arc, time::Duration};

const RANKS: usize = 6;
const LATENT: usize = 3584;
const TOPK: usize = 16;

/// The intake owns six full-capacity BF16 planes on the dispatch device.
/// Pinned route storage is host memory; all device planes precede admission.
pub(crate) fn demand(rows: usize, lane: usize) -> Result<Demand> {
    ensure!((1..=4096).contains(&rows), "Kimi Spark row capacity");
    Ok(Demand::new(0, Category::Workspace, format!("Kimi Spark lane {lane} latent intake"),
        (RANKS * rows * LATENT * 2) as u64, Basis::Exact))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ticket { layer: usize, rows: usize, sequence: u64 }

/// Engine-retained wave owner; tickets never own the transport or copy sources.
/// Drop completes copies before releasing pinned storage, and the ordinary lane
/// joins its transport thread before the intake's planes are released.
pub(crate) struct RoutedLane<'a> {
    stream: Rc<Stream<'a>>,
    link: Option<SparkLane<'a>>,
    library_owner: Arc<cuteafd_ffi::NativeLibrary>,
    terminal: bool,
    staging: HostAllocation<'a>,
    ready: Event<'a>,
    capacity: usize,
    sequence: u64,
    pending: Option<Ticket>,
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
        peers: Vec<SocketAddr>, config: TcpTransportConfig) -> Result<Self> {
        ensure!(std::ptr::eq(library_owner.as_ref(), local.stream().device.library),
            "Kimi Spark library owner differs from admitted device");
        ensure!(cuteafd_core::expert_geometry() == ExpertGeometry::KIMI_K3,
            "Kimi Spark helpers require the process latent expert geometry before native loading");
        ensure!(peers.len() == RANKS && local.has_embedding(),
            "Kimi routed exchange requires six real Spark ranks and the admitted entry rank");
        let charge = demand(local.rows(), local.lane_index())?;
        let charges = placement.items[0].iter().filter(|item| item.group == charge.group).collect::<Vec<_>>();
        ensure!(charges.len() == 1 && charges[0].category == charge.category
            && charges[0].basis == charge.basis && charges[0].bytes == charge.bytes,
            "Kimi Spark intake differs from accepted admission");
        let stream = local.stream().clone();
        let layout = RouteStaging::new(local.rows(), TOPK, KimiExpertInput::ROW_BYTES)?;
        let library = library_owner.as_ref();
        let (link, staging, ready) = stream.device.run(|| {
            library.v41_compact_reducer()?.require_plane_reduction(RANKS as u32)?;
            Ok((SparkLane::new(library, world, peers, local.rows() as u32, config, LATENT * 2)?,
                HostAllocation::new(library, layout.bytes)?, Event::new(stream.device)?))
        })?;
        ensure!(link.world_size() == RANKS, "Kimi Spark transport world differs from admission");
        Ok(Self { stream, link: Some(link), library_owner: Arc::clone(library_owner),
            terminal: false, staging, ready, capacity: local.rows(), sequence: 0, pending: None })
    }

    /// #511: diagnostic-only successful-drain boundary, not failure-safe.
    /// Keep the whole lane and its exact library owner until copies are drained.
    pub(crate) fn quiesce_links(&mut self) -> Result<()> {
        ensure!(self.pending.is_none(), "Kimi interim teardown cannot quiesce a failed wave (#511)");
        self.terminal = true;
        tracing::warn!("Kimi diagnostic teardown: not failure-safe; #511 checked quiescence pending");
        Ok(())
    }

    pub(crate) fn drain_copies(&self) -> Result<()> {
        self.stream.device.run(|| match &self.link {
            Some(link) => link.intake.terminal_drain_copies(),
            None => Ok(()),
        })
    }

    pub(crate) fn release_payloads(&mut self) -> Result<()> {
        ensure!(self.terminal && self.pending.is_none(), "Kimi Spark sources are not terminal");
        self.drain_copies()?;
        // #511: successful cold diagnostics only. Drop joins before intake release.
        let link = self.link.take();
        self.stream.device.run(|| { drop(link); Ok(()) })
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
        kind: ExpertV2SourceKind) -> Result<Ticket> {
        self.check(local, rows)?;
        ensure!(self.pending.is_none(), "Kimi routed lane already has an outstanding wave");
        let sequence = self.sequence.checked_add(1).context("Kimi Spark request sequence overflow")?;
        let (routes, wire) = local.route(layer, rows)?;
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
            // SAFETY: this event covers only copies before the shared FFN;
            // pinned staging remains exclusively owned until this fence.
            unsafe { self.stream.device.library.cuda_event_synchronize(self.ready.raw) }
        })?;
        let request = request(layer, rows, sequence, kind, &layout, self.staging.bytes())?;
        let ticket = Ticket { layer, rows, sequence };
        // Retain the terminal wave identity before handing work to the lane.
        self.pending = Some(ticket);
        self.sequence = sequence;
        let link = self.link.as_mut().context("Kimi Spark lane retired")?;
        self.stream.device.run(|| link.submit(rows, Box::new(move || Ok(request))))?;
        Ok(ticket)
    }

    /// All six latent partials are reduced once with FP32 accumulation and one
    /// BF16 rounding. No hidden-space shared FFN participates in this reduction.
    pub(crate) fn finish(&mut self, local: &LocalLane<'a>, ticket: Ticket) -> Result<()> {
        self.check(local, ticket.rows)?;
        ensure!(self.pending == Some(ticket), "Kimi Spark completion differs from retained wave");
        let output = local.buffer("expert combined")?;
        let link = self.link.as_mut().context("Kimi Spark lane retired")?;
        self.stream.device.run(|| {
            link.wait(ticket.rows, Duration::from_secs(120), self.stream.raw)?;
            // SAFETY: wait orders all six live intake planes before this exact
            // lane stream; output is a disjoint admitted latent allocation.
            unsafe { self.stream.device.library.v41_compact_reducer()?.reduce_planes(
                link.intake.pointers(), RANKS as u32, std::ptr::null(),
                output.ptr.cast(), ticket.rows as u32, self.stream.raw)?; }
            link.intake.consumed(self.stream.raw)
        })?;
        self.pending = None;
        Ok(())
    }
}

fn request(layer: usize, rows: usize, sequence: u64, kind: ExpertV2SourceKind,
    layout: &RouteStaging, bytes: &[u8]) -> Result<ExpertProtocolV2Request> {
    ensure!((1..93).contains(&layer) && (1..=4096).contains(&rows) && sequence > 0,
        "Kimi Spark request geometry or sequence");
    let routes = layout.routes(bytes)?;
    ensure!(routes.len() == rows * TOPK && routes.iter().all(|route|
        route.expert_id < 896 && route.gate_weight.is_finite() && route.gate_weight >= 0.0),
        "Kimi Spark route outside E896/K16 or invalid gate weight");
    let mut request = ExpertProtocolV2Request::new(sequence, 17, layer as u32, LATENT as u32,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: row as u64,
            source_kind: kind, source_request_id: sequence, token_position: row as u64,
            route_offset: (row * TOPK) as u32, route_count: TOPK as u32 }).collect(),
        routes, layout.wire(bytes)?.to_vec())?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_owner_keeps_shared_lane_and_exact_library_until_terminal_release() {
        let source = include_str!("spark.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("link: Option<SparkLane<'a>>"));
        assert!(source.contains("library_owner: Arc<cuteafd_ffi::NativeLibrary>"));
        assert!(source.contains("std::ptr::eq(library_owner.as_ref(), local.stream().device.library)"));
        assert!(source.contains("Arc::clone(library_owner)"));
        assert!(!source.contains("let SparkLane {"));
        let release = source.split("pub(crate) fn release_payloads").nth(1).unwrap()
            .split("fn check").next().unwrap();
        assert!(release.find("self.drain_copies()?").unwrap() < release.find("self.link.take()").unwrap());
        assert!(release.contains("self.stream.device.run(|| { drop(link); Ok(()) })"));
        assert!(source.contains("not failure-safe"));
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
            assert_eq!(demand(rows, 1)?.bytes, (6 * rows * 3584 * 2) as u64);
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
        assert!(demand(0, 0).is_err());
        assert!(demand(4097, 0).is_err());
        Ok(())
    }
}
