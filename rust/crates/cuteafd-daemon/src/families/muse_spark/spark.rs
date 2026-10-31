//! The Muse Spark expert path ([`SparkExperts`]) over the shared Spark link:
//! one persistent endpoint per Spark (decode, verify and prefill share its QP;
//! `placement::families::muse_spark::spark_endpoint`), the shared exchange
//! (`SparkLink` / `spark_waves`) and its intake, which lands each rank's
//! partial reply in that rank's own plane of the wave's return slot.
//!
//! - `send`: the lane's routes and FP8 K32 wire rows (`RankDevice::expert_wire`)
//!   copy into the slot's staging on the lane stream; the exchange worker builds
//!   the request (`wave_request`, with the step's explicit per-row regions) and
//!   posts it once they land. The host never waits on the GPU.
//! - `land`: the lane stream waits for the slot's planes; the wave's
//!   `rank_regions` tables and row maps are uploaded into the fixed device
//!   tables on the same stream (addresses fixed, contents per wave), so the
//!   combine reads every rank's regions in place: no gather.
//! - `consumed`: after `RankDevice::combine`, the slot's planes are released.
use super::experts::{wave_request, CombineTables, LandedWave, ReplyPlanView, SparkExperts, Wave, WaveSegment, WaveStep,
    MAX_RANKS, TOPK, WIDTH, WIRE_ROW_BYTES};
use crate::shared::{experts::staging::RouteStaging, memory::device::{Allocation, Stream}, spark_intake::SparkLink,
    spark_waves::WaveTicket};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::muse_spark::replies::{Cutoff, ReplyPlan};
use cuteafd_transport::{expert::SparkWorld, ExpertV2SourceKind, TcpTransportConfig};
use std::{collections::VecDeque, net::SocketAddr, rc::Rc, sync::Arc};

/// A sent wave and its regions (the plan the request carried).
struct InFlight { wave: Wave, ticket: Option<WaveTicket>, slot: usize, plan: ReplyPlan }


pub(crate) struct MuseSparkPath<'a> {
    stream: Rc<Stream<'a>>,
    link: Option<SparkLink<'a>>,
    library_owner: Arc<cuteafd_ffi::NativeLibrary>,
    cutoff: Cutoff,
    /// The fixed combine tables (one per return slot, so a slot's image is
    /// never rewritten while an earlier wave's combine may still read it).
    tables: Allocation<'a>,
    layout: CombineTables,
    unit: usize,
    waves: usize,
    sequence: u64,
    in_flight: VecDeque<InFlight>,
    terminal: bool,
}

impl Drop for MuseSparkPath<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Muse Spark path copies and combines");
        let link = self.link.take();
        crate::shared::decode_graph::fatal_drain(
            self.stream.device.run(|| { drop(link); Ok(()) }), "Muse Spark path device-scoped release");
    }
}

/// The widest wave the path dispatches: `rows` rows of one request with every
/// row in the wide region (it sizes the slot staging; the response rings come
/// from the sealed endpoint). Its routes and payload are zero; it is never sent.
pub(crate) fn bootstrap_request(layer: usize, rows: usize, cutoff: Cutoff)
    -> Result<cuteafd_transport::ExpertProtocolV2Request> {
    let routes = (0..rows * TOPK).map(|i| cuteafd_transport::ExpertProtocolV2RouteEntry {
        row_index: (i / TOPK) as u32, expert_id: 0, gate_weight: 0.0 }).collect();
    wave_request(layer, 1, ExpertV2SourceKind::Prefill, &[WaveSegment { request: 1, rows }], cutoff, routes,
        bytes::Bytes::from(vec![0; rows * WIRE_ROW_BYTES]))
}

impl<'a> MuseSparkPath<'a> {
    /// Connects the Spark link (`waves` return slots of `unit` rows, each rank's
    /// plane `intake_row_bytes` per row, the worst admitted reply mix) and
    /// bootstraps it at the widest wave. `intake_row_bytes` is the planner's
    /// (`replies::intake_row_bytes`): the same charge admission accepted.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn connect(library_owner: &'a Arc<cuteafd_ffi::NativeLibrary>, stream: Rc<Stream<'a>>,
        world: SparkWorld, peers: Vec<SocketAddr>, config: TcpTransportConfig,
        cutoff: Cutoff, unit: usize, waves: usize, intake_row_bytes: usize, first_spark_layer: usize) -> Result<Self> {
        ensure!(std::ptr::eq(library_owner.as_ref(), stream.device.library),
            "Muse Spark library owner differs from the lane device");
        ensure!(cuteafd_core::expert_geometry() == cuteafd_core::ExpertGeometry::MUSE_SPARK,
            "Muse Spark path requires the process expert geometry before native loading");
        ensure!((2..=MAX_RANKS).contains(&peers.len()), "Muse Spark serves 2..={MAX_RANKS} Spark ranks");
        ensure!(intake_row_bytes % 16 == 0 && unit > 0, "Muse partial planes must keep 16-byte record bases");
        let library = library_owner.as_ref();
        let layout = CombineTables { capacity: unit };
        let tables = {
            let _scope = cuteafd_ffi::memory_ledger::scope("workspace/muse combine tables");
            Allocation::new(stream.device, layout.bytes() * waves)?
        };
        let mut link = stream.device.run(|| {
            let mut link = SparkLink::new_waves(library, world, &peers, u32::try_from(unit)?, config,
                intake_row_bytes, waves, u32::try_from(unit)?)?;
            link.enable_terminal_ownership(Arc::clone(library_owner))?;
            Ok(link)
        })?;
        ensure!(link.world_size() == peers.len(), "Muse Spark transport world differs from its peers");
        stream.device.run(|| link.bootstrap(&bootstrap_request(first_spark_layer, unit, cutoff)?))?;
        ensure!(link.installed_grouping()?.len() == peers.len(), "Muse bootstrap requires every rank's grouping");
        // Reserve each slot's ordered-upload staging before readiness: no upload
        // allocates host memory under queued work.
        for slot in 0..waves {
            let table = Self::slot_tables(&tables, layout, slot);
            // SAFETY: a live admitted allocation on the lane's device and stream.
            stream.device.run(|| unsafe { library.reserve_h2d_ordered(table, stream.raw) })?;
        }
        Ok(Self { stream, link: Some(link), library_owner: Arc::clone(library_owner), cutoff, tables, layout,
            unit, waves, sequence: 0, in_flight: VecDeque::new(), terminal: false })
    }

    fn slot_tables(tables: &Allocation<'_>, layout: CombineTables, slot: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
        cuteafd_ffi::CuteafdDeviceBuffer { ptr: tables.buffer.ptr.cast::<u8>().wrapping_add(slot * layout.bytes()).cast(),
            bytes: layout.bytes(), ..tables.buffer }
    }

    fn link(&mut self) -> Result<&mut SparkLink<'a>> {
        ensure!(!self.terminal, "Muse Spark path is terminal");
        self.link.as_mut().context("Muse Spark link retired")
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

    pub(crate) fn release_payloads(&mut self) -> Result<()> {
        ensure!(self.terminal, "Muse Spark sources are not terminal");
        self.stream.device.run(|| {
            if let Some(link) = &self.link { link.terminal_drain_copies()?; }
            if let Some(link) = &mut self.link { link.terminal_release()?; }
            drop(self.link.take());
            Ok(())
        })?;
        self.in_flight.clear();
        Ok(())
    }
}

impl SparkExperts for MuseSparkPath<'_> {
    fn send(&mut self, layer: usize, step: &WaveStep, sources: [cuteafd_ffi::CuteafdDeviceBuffer; 3]) -> Result<Wave> {
        let rows = step.rows();
        ensure!(!step.segments.is_empty() && step.segments.iter().all(|s| s.rows > 0) && (1..=self.unit).contains(&rows),
            "Muse Spark wave of {rows} rows outside its {}-row unit", self.unit);
        ensure!(self.in_flight.len() < self.waves, "Muse Spark path has {} waves in flight", self.waves);
        let sequence = self.sequence.checked_add(1).context("Muse Spark request sequence overflow")?;
        let (segments, kind, cutoff) = (step.segments.clone(), step.kind, self.cutoff);
        let plan = ReplyPlan::new(cutoff, &segments.iter().map(|s| s.rows).collect::<Vec<_>>());
        let layout = RouteStaging::new(rows, TOPK, WIRE_ROW_BYTES)?;
        let stream = self.stream.raw;
        let device = self.stream.device;
        let link = self.link()?;
        let ticket = device.run(|| {
            // SAFETY: the lane retains these planes through teardown; the routes and
            // wire were written on this stream before this call (`RankDevice::layer_front`).
            unsafe { link.exchange()?.send(rows, layout, sources, stream, Box::new(move |routes, wire|
                wave_request(layer, sequence, kind, &segments, cutoff, routes, wire))) }
        })?;
        self.sequence = sequence;
        let wave = Wave { layer, rows, sequence };
        let slot = ticket.slot();
        self.in_flight.push_back(InFlight { wave, ticket: Some(ticket), slot, plan });
        Ok(wave)
    }

    fn land(&mut self, wave: Wave) -> Result<LandedWave> {
        let front = self.in_flight.front_mut().context("Muse Spark land without a wave in flight")?;
        ensure!(front.wave == wave, "Muse Spark lands {wave:?}, oldest is {:?}", front.wave);
        let ticket = front.ticket.take().context("Muse Spark wave already landed")?;
        let (slot, plan) = (front.slot, front.plan.clone());
        let (layout, stream, device) = (self.layout, self.stream.raw, self.stream.device);
        let tables = Self::slot_tables(&self.tables, layout, slot);
        let library = self.library_owner.clone();
        let link = self.link()?;
        let ranks = device.run(|| {
            let exchange = link.exchange()?;
            exchange.land(ticket, stream)?;
            let ranks = exchange.ranks();
            let planes: Vec<u64> = exchange.planes(slot)[..ranks].iter().map(|p| *p as u64).collect();
            // The wire layout (`PartialReplyPlan::payload`): the BF16 region, then the
            // wide region at the next region-aligned offset.
            let wide_at = (plan.bf16_rows.len() * TOPK * WIDTH * 2)
                .next_multiple_of(cuteafd_transport::EXPERT_PROTOCOL_V2_PARTIAL_REGION_ALIGN);
            let image = layout.image(&planes, &ReplyPlanView { bf16_rows: &plan.bf16_rows,
                wide_rows: &plan.wide_rows, wide_at })?;
            // SAFETY: the slot's fixed tables on this stream's device; the ordered
            // upload waits for its own previous copy (not compute) and the combine
            // that reads it follows on this stream.
            unsafe { library.copy_h2d_ordered(tables, &image, stream)?; }
            Ok(ranks)
        })?;
        Ok(LandedWave { tables, layout, ranks, plan })
    }

    fn consumed(&mut self, wave: Wave) -> Result<()> {
        let front = self.in_flight.front().context("Muse Spark consume without a landed wave")?;
        ensure!(front.wave == wave && front.ticket.is_none(), "Muse Spark consumes {wave:?} out of order");
        let slot = front.slot;
        let (stream, device) = (self.stream.raw, self.stream.device);
        let link = self.link()?;
        device.run(|| link.exchange()?.consumed(slot, stream))?;
        self.in_flight.pop_front();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bootstrap request is the widest wave: one wide segment of the unit,
    /// zero routes and payload, the explicit region of every row set.
    #[test]
    fn bootstrap_is_the_widest_all_wide_wave() -> Result<()> {
        let cutoff = Cutoff::new(64, 8)?;
        for rows in [65, 2752, 4096] {
            let request = bootstrap_request(3, rows, cutoff)?;
            assert_eq!(request.header.row_count as usize, rows);
            assert_eq!(request.hidden_payload.len(), rows * WIRE_ROW_BYTES);
            let plan = request.partial.as_ref().unwrap();
            assert!(plan.regions.iter().all(|&r| r == 1));
        }
        // At or below the cutoff the one segment replies BF16 (still a valid widest wave).
        assert!(bootstrap_request(3, 64, cutoff)?.partial.unwrap().regions.iter().all(|&r| r == 0));
        assert!(bootstrap_request(62, 64, cutoff).is_err());
        Ok(())
    }

    /// The path holds the link and its exact library until terminal release,
    /// lands before uploading the tables, and releases a slot only after the
    /// combine (the engine calls `consumed` behind it).
    #[test]
    fn path_orders_land_tables_and_consume() {
        let source = include_str!("spark.rs").split("#[cfg(test)]").next().unwrap();
        let land = source.split("fn land(").nth(1).unwrap().split("fn consumed(").next().unwrap();
        assert!(land.find("exchange.land(ticket, stream)").unwrap() < land.find("copy_h2d_ordered").unwrap());
        assert!(!land.contains("consumed("), "the combine reads the planes after land");
        assert!(source.contains("link.enable_terminal_ownership"));
        assert!(source.find("link.enable_terminal_ownership").unwrap() < source.find("link.bootstrap").unwrap());
        assert!(source.contains("reserve_h2d_ordered"), "upload staging reserved before readiness");
        assert!(!source.contains("block_on") && !source.contains("cuda_stream_synchronize"));
        let device = include_str!("device.rs");
        let front = device.split("pub(crate) fn layer_front(").nth(1).unwrap().split("pub(crate) fn layer_back(").next().unwrap();
        let back = device.split("pub(crate) fn layer_back(").nth(1).unwrap();
        // Front: frontend, wire, send (both layouts); back: land, combine, consume, then the backend.
        for (front_call, wire) in [("self.moe_frontend(layer, rows)", "self.expert_wire(rows)"),
            ("self.moe_frontend_split(layer, rows", "self.expert_wire(rows)")] {
            let f = front.find(front_call).unwrap();
            assert!(f < f + front[f..].find(wire).unwrap(), "{front_call} before the wire");
        }
        assert!(front.matches("experts.send(layer, &step.wave").count() == 2);
        let order = ["experts.land(wave)", "self.combine(layer", "experts.consumed(wave)", "self.moe_backend"];
        let at: Vec<usize> = order.iter().map(|needle| back.find(needle).unwrap()).collect();
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "layer back order {order:?}");
    }
}
