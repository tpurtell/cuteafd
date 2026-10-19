//! Retained coordinator submission owner. Startup supplies admitted operands and
//! ordinary Spark links; this uses the shared executor and terminal sequence.
use super::{device::{LocalLane, SequenceRows, TokenIo}, layers::Execution, spark::RoutedLane};
use crate::shared::{experts::rtx::pair::{submit_scoped, teardown, Teardown},
    peer_split::{hop::HopLink, PeerExchange, TerminalState}};
use anyhow::{ensure, Result};
use cuteafd_loader::placement::LayerExecutionPlan;
use std::{cell::RefCell, rc::Rc};
use crate::shared::memory::device::Stream;

struct Lanes<'a> {
    local: Vec<[LocalLane<'a>; 2]>,
    routed: Vec<RoutedLane<'a>>,
    io: Vec<[TokenIo<'a>; 2]>,
}

pub(crate) struct Engine<'a> {
    execution: Execution<'a>,
    lanes: RefCell<Lanes<'a>>,
    hops: HopLink<'a>,
    peer: PeerExchange<'a>,
    // PeerExchange and HopLink borrow raw handles; keep their exact streams
    // after both links and their terminal abort storage have been released.
    exchange_streams: [Rc<Stream<'a>>; 2],
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.shutdown(), "Kimi engine teardown");
    }
}

fn check_exchange_streams(peer: &PeerExchange<'_>, streams: &[Rc<Stream<'_>>; 2]) -> Result<()> {
    ensure!(std::ptr::eq(streams[0].device.library, streams[1].device.library),
        "Kimi exchange stream native libraries differ");
    check_exchange_handles(std::array::from_fn(|rank| (peer.device(rank), peer.stream(rank))),
        streams.each_ref().map(|stream| (stream.device.id, stream.raw)))
}

fn check_exchange_handles(expected: [(i32, *mut std::ffi::c_void); 2],
    actual: [(i32, *mut std::ffi::c_void); 2]) -> Result<()> {
    ensure!(expected[0].0 != expected[1].0 && expected == actual
        && actual.iter().all(|(_, stream)| !stream.is_null()),
        "Kimi exchange stream owners differ from exact peer/hop handles");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchange_owners_require_exact_rank_device_and_stream() {
        let handles = [(0, 1usize as *mut _), (1, 2usize as *mut _)];
        assert!(check_exchange_handles(handles, handles).is_ok());
        assert!(check_exchange_handles(handles, [handles[1], handles[0]]).is_err());
        assert!(check_exchange_handles(handles, [(0, 3usize as *mut _), handles[1]]).is_err());
        assert!(check_exchange_handles(handles, [(2, handles[0].1), handles[1]]).is_err());
        let null = [(0, std::ptr::null_mut()), handles[1]];
        assert!(check_exchange_handles(null, null).is_err());
        let same_device = [handles[0], (0, handles[1].1)];
        assert!(check_exchange_handles(same_device, same_device).is_err());
    }
}

impl<'a> Engine<'a> {
    /// Complete startup from the measured shared admission and already charged
    /// exchange owners. Worker reconciliation precedes ordinary lane connection.
    pub(crate) fn load_admitted(exchange_streams: [Rc<Stream<'a>>; 2],
        mut admission: super::admission::Admission<'a>,
        checkpoint: &cuteafd_loader::plan::Checkpoint,
        programs: [std::rc::Rc<super::device::NativePrograms<'a>>; 2],
        peer: PeerExchange<'a>, hops: HopLink<'a>, peers: Vec<std::net::SocketAddr>,
        config: cuteafd_transport::TcpTransportConfig, compiled_rows: u32) -> Result<Self> {
        use crate::shared::placement::{spark_rings, SparkAdmission};
        use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
        use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission};
        ensure!(cuteafd_core::expert_geometry() == ExpertGeometry::KIMI_K3 && peers.len() == 6,
            "Kimi startup needs process latent geometry and six real expert ranks");
        let plan = admission.execution_plan()?;
        hops.validate_binding(&plan, &peer)?;
        check_exchange_streams(&peer, &exchange_streams)?;
        let devices = programs.each_ref().map(|programs| programs.device);
        ensure!((0..2).all(|rank| devices[rank].id == peer.device(rank))
            && std::ptr::eq(devices[0].library, devices[1].library),
            "Kimi startup programs differ from exchange devices/library");
        let rows = u32::try_from(plan.spec().rows)?;
        let endpoints = vec![SparkEndpoint { capacity: rows, ingress: SparkIngress::Fp8K32 };
            plan.spec().lanes as usize];
        let model = ModelAdmission { capacity: rows, compiled_capacity: compiled_rows,
            endpoints, rings: spark_rings(config.max_frame_bytes)? };
        model.validate(ExpertGeometry::KIMI_K3)?;
        let catalog = cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?;
        let pending = SparkAdmission::plan(&checkpoint.snapshot, ExpertGeometry::KIMI_K3,
            Some(&catalog), &admission.working_set(), 6, None, model)?.start(&peers)?;
        let weights = admission.load_weights(devices, checkpoint)?;
        let lanes = usize::try_from(plan.spec().lanes)?;
        let mut allocated = (0..lanes).map(|lane| Ok([
            admission.allocate_lane(devices[0], 0, lane)?,
            admission.allocate_lane(devices[1], 1, lane)?,
        ])).collect::<Result<Vec<_>>>()?;
        let kda = [admission.allocate_kda(devices[0], 0)?, admission.allocate_kda(devices[1], 1)?];
        let mla = [admission.allocate_mla(devices[0], 0)?, admission.allocate_mla(devices[1], 1)?];
        let mut local = Vec::with_capacity(lanes);
        let mut io = Vec::with_capacity(lanes);
        for (lane, [zero, one]) in allocated.drain(..).enumerate() {
            local.push([admission.bind_local(0, lane, zero, programs[0].clone())?,
                admission.bind_local(1, lane, one, programs[1].clone())?]);
            io.push([admission.allocate_io(devices[0], 0, lane)?,
                admission.allocate_io(devices[1], 1, lane)?]);
        }
        // Weak admission registries must not outlive these strong startup owners
        // until every persistent local binding has acquired the same operands.
        drop((weights, kda, mla));
        pending.wait()?;
        let world = SparkWorld::canonical(6)?;
        let routed = local.iter().map(|pair| admission.connect_spark(&pair[0],
            world, peers.clone(), config.clone())).collect::<Result<Vec<_>>>()?;
        Self::bind(exchange_streams, plan, peer, hops, local, routed, io)
    }

    /// Bind only exact owners already allocated through Admission. No device
    /// allocations or program loading occur after this constructor returns.
    pub(crate) fn bind(exchange_streams: [Rc<Stream<'a>>; 2],
        plan: LayerExecutionPlan, peer: PeerExchange<'a>, hops: HopLink<'a>,
        local: Vec<[LocalLane<'a>; 2]>, routed: Vec<RoutedLane<'a>>,
        io: Vec<[TokenIo<'a>; 2]>) -> Result<Self> {
        ensure!(!local.is_empty() && local.len() == routed.len() && local.len() == io.len(),
            "Kimi engine local, Spark and IO lane counts differ");
        check_exchange_streams(&peer, &exchange_streams)?;
        for (lane, pair) in local.iter().enumerate() {
            for rank in 0..2 {
                io[lane][rank].check_lane(&pair[rank], rank)?;
                ensure!(std::ptr::eq(pair[rank].stream().device.library,
                    exchange_streams[rank].device.library), "Kimi local and exchange native libraries differ");
            }
        }
        let streams = local.iter().map(|pair| pair.each_ref().map(|local| local.stream().clone())).collect();
        let execution = Execution::new(plan, &peer, &hops, streams)?;
        Ok(Self { execution, lanes: RefCell::new(Lanes { local, routed, io }), hops, peer, exchange_streams })
    }

    fn shutdown(&self) -> Result<()> {
        if self.peer.terminal_state() == TerminalState::Drained { return Ok(()); }
        let lanes = self.lanes.try_borrow_mut()?;
        let lanes = RefCell::new(lanes);
        let result = teardown(&mut Teardown {
            abort: || self.peer.publish_abort(),
            ranks: || {
                self.peer.drain_compute()?;
                for pair in &lanes.borrow().local {
                    for local in pair {
                        let stream = local.stream();
                        stream.device.run(|| {
                            // SAFETY: peer aborts are visible; every lane operand
                            // and recorded event remains owned through drainage.
                            unsafe { stream.device.library.cuda_stream_synchronize(stream.raw) }
                        })?;
                    }
                }
                Ok(())
            },
            links: || {
                // #511: this single call site is drop-and-join until checked
                // ordinary-lane QP quiescence and retention are implemented.
                for routed in &mut lanes.borrow_mut().routed { routed.quiesce_links()?; }
                Ok(())
            },
            copies: || {
                for routed in &lanes.borrow().routed { routed.drain_copies()?; }
                Ok(())
            },
            graphs: || {
                for routed in &lanes.borrow().routed { routed.release_payloads()?; }
                Ok(())
            },
            finish: |drained| self.peer.finish_terminal(drained),
        });
        crate::shared::decode_graph::fatal_drain(result, "Kimi submission shutdown");
        Ok(())
    }

    /// Diagnostic embedding-to-logits path. No graph/replay or speculative
    /// transaction qualification is implied by this uncaptured forward.
    pub(crate) fn forward(&self, tokens: &[Vec<u32>], sequences: &[Vec<SequenceRows<'_>>],
        pool: &cuteafd_engine::prefix::RefPagePool,
        mut consume: impl FnMut(usize, usize, usize, &[f32]) -> Result<()>) -> Result<()> {
        submit_scoped(|| self.peer.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            ensure!(!tokens.is_empty() && tokens.len() == sequences.len() && tokens.len() <= lanes.local.len(),
                "Kimi forward token/sequence lane count");
            let rows = tokens.iter().map(Vec::len).collect::<Vec<_>>();
            for (lane, ids) in tokens.iter().enumerate() {
                ensure!(!ids.is_empty(), "Kimi forward has an empty lane");
                for local in &mut lanes.local[lane] {
                    ensure!(sequences[lane].iter().map(|sequence| sequence.rows).sum::<usize>() == ids.len(),
                        "Kimi forward sequence rows differ from tokens");
                    local.stage_attention(&sequences[lane], pool)?;
                }
                lanes.io[lane][0].embed(ids, lanes.local[lane][0].buffer("delta")?)?;
            }
            let Lanes { local, routed, io } = &mut *lanes;
            let normalized = self.execution.run(scope, &self.peer, &self.hops,
                &mut local[..tokens.len()], &mut routed[..tokens.len()], &rows, false)?;
            for (lane, normalized) in normalized.into_iter().enumerate() {
                let stream = local[lane][1].stream();
                for first in (0..rows[lane]).step_by(64) {
                    let count = (rows[lane] - first).min(64);
                    let input = crate::shared::prefix::view(normalized, first * 7168 * 2, count * 7168 * 2)?;
                    let logits = io[lane][1].project(input, count)?;
                    let values = stream.device.run(|| {
                        // SAFETY: diagnostic download is outside capture; the
                        // exact lane's head finishes before its logits are reused.
                        unsafe { stream.device.library.cuda_stream_synchronize(stream.raw)?; }
                        let mut bytes = vec![0; logits.bytes];
                        stream.device.library.copy_d2h(&mut bytes, logits)?;
                        Ok(bytes.chunks_exact(4).map(|bytes|
                            f32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>())
                    })?;
                    consume(lane, first, count, &values)?;
                }
            }
            Ok(())
        })
    }
}
