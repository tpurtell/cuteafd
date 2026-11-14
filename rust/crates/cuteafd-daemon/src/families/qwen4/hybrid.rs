//! Bind only the routed prefix admitted before model, pool or expert allocation.
use super::{admission::ExpertAdmission, engine, EngineArgs, Opened};
use crate::shared::{experts::{layer::ExpertLayer, rtx::{self, exl3::Exl3Tp2,
    fp8moe::Fp8MoeTp2, push::{CoordinatorExperts, CoordinatorPush, PushPolicy, WholeOwner},
    RtxExpertLayer, RtxSelectionPolicy}}, memory::device::Device};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{placement::families::qwen4::ExpertCandidate, OfficialV41Catalog};
use std::{cell::RefCell, path::Path};

pub(super) fn backend_budget(candidate: &ExpertCandidate, rank: usize) -> Result<usize> {
    let weights = candidate.arenas.iter().filter(|(owner, _)| *owner == rank)
        .map(|(_, arena)| arena.bytes()).sum::<usize>();
    let workspace = candidate.workspace.iter().filter(|allocation| allocation.view.is_none()
        && allocation.location == cuteafd_loader::placement::planner::Location::coordinator(rank as u8)
        && allocation.id.starts_with("qwen.experts.rank"))
        .try_fold(0usize, |sum, allocation| sum.checked_add(usize::try_from(allocation.bytes)?)
            .context("Qwen expert workspace overflow"))?;
    weights.checked_add(workspace).context("Qwen backend budget overflow")
}

pub(super) fn push_plan(catalog: &OfficialV41Catalog, layers: &[usize], gpus: usize,
    _rows: usize, _args: &EngineArgs, candidate: Option<&ExpertCandidate>) -> Result<Option<CoordinatorExperts>> {
    if layers.is_empty() { return Ok(None); }
    let budgets = (0..gpus).map(|rank| candidate.map_or(Ok(usize::MAX),
        |candidate| backend_budget(candidate, rank))).collect::<Result<Vec<_>>>()?;
    Ok(Some(if gpus == 2 { CoordinatorExperts::Tp2 { layers: layers.to_vec(), budgets } }
        else { CoordinatorExperts::Whole { owners: vec![WholeOwner { rank: 0,
            selections: layers.iter().map(|&layer| ExpertLayer::BackboneFull { layer }).collect(),
            budget: budgets[0] }] } }))
}

fn whole_package(args: &EngineArgs, catalog: &OfficialV41Catalog) -> Result<std::path::PathBuf> {
    let root = args.native_lib.parent().unwrap_or(Path::new("."));
    if let Some(exl3) = catalog.exl3() {
        Ok(root.join("exl3").join(crate::shared::experts::exl3::package_name("qwen4", exl3.decoder_tiers())).join("rtx-tp1"))
    } else {
        Ok(args.fp8_package.clone().unwrap_or_else(|| crate::shared::experts::fp8::package_directory(
            &args.native_lib, 1, catalog.fp8().expect("validated catalog").format())))
    }
}

impl Opened {
    pub(super) fn load_hybrid<'a>(&'a self, args: &EngineArgs, devices: &[Device<'a>],
        streams: &[*mut std::ffi::c_void], admitted: &ExpertAdmission,
        host: &cuteafd_loader::placement::families::qwen4::QwenHost) -> Result<engine::HybridExperts<'a>> {
        let catalog = self.experts.as_ref().context("Qwen routed expert catalog missing")?;
        let candidate = admitted.candidate.as_ref().context("Qwen expert prefix was not admitted")?;
        let layers = candidate.layers.clone();
        admitted.validate_layers(&layers)?;
        let rows = args.prefill_wave_rows().max(engine::DECODE_ROWS);
        let gpus = devices.len();
        ensure!((1..=2).contains(&gpus) && streams.len() == gpus, "Qwen hybrid device/stream owners differ");
        let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
        let policy = PushPolicy::for_snapshot(source)?;
        let pushed = if let Some(experts) = push_plan(catalog, &layers, gpus, rows, args, Some(candidate))? {
            match policy.plan(|| {
                let envelope = host.push.iter().find(|(name, _)| *name == "qwen.host.push.routed")
                    .context("Qwen routed push lacks pre-allocation envelope")?.1.clone();
                CoordinatorPush::prepare_alone_admitted(source, Some(catalog), gpus, Some(experts), envelope)
            })? {
                Some(push) => {
                    let (pushed, ()) = push.receive_with::<rtx::push::NoDense, _>(devices.to_vec(), Some(catalog), 0, None, || ());
                    policy.committed(pushed).context("Qwen admitted routed expert push")?
                }
                None => None,
            }
        } else { None };
        let local = if layers.is_empty() { None } else if gpus == 1 {
            let package = whole_package(args, catalog)?;
            let budget = backend_budget(candidate, 0)?;
            let backend: Box<dyn RtxExpertLayer + 'a> = if catalog.exl3().is_some() {
                let weights = match pushed {
                    Some(mut ranks) => Exl3Tp2::pushed_whole_weights(devices[0], catalog, &package,
                        layers.clone(), rows, budget, RtxSelectionPolicy::CanonicalDecode,
                        ranks.pop().context("Qwen whole pushed rank missing")?.exl3()?)?,
                    None => Exl3Tp2::load_whole_weights(devices[0], catalog, &package, layers.clone(),
                        rows, budget, RtxSelectionPolicy::CanonicalDecode)?,
                };
                Box::new(weights.execution()?)
            } else {
                let tensors = catalog.fp8().context("Qwen native whole headers")?;
                Box::new(match pushed {
                    Some(mut ranks) => rtx::whole::Whole::from_pushed(devices[0], tensors, &package,
                        layers.clone(), rows, budget, ranks.pop().context("Qwen whole native pushed rank missing")?.fp8()?)?,
                    None => rtx::whole::Whole::load(devices[0], tensors, &package, layers.clone(), rows, budget)?,
                })
            };
            Some(engine::LocalRouted::Whole(rtx::pair::Whole::new(&self.library, devices[0].id, backend,
                self.cfg.hidden, rows)?))
        } else {
            let pair_devices = [devices[0], devices[1]];
            let budgets = [backend_budget(candidate, 0)?, backend_budget(candidate, 1)?];
            let pair: [Box<dyn RtxExpertLayer + 'a>; 2] = if catalog.exl3().is_some() {
                let packages = cuteafd_loader::placement::families::qwen4::dual_packages(catalog, &args.native_lib)?;
                let owners = match pushed {
                    Some(ranks) => {
                        let mut ranks = ranks.into_iter();
                        let payloads = [ranks.next().context("Qwen TP2 rank zero missing")?.exl3()?,
                            ranks.next().context("Qwen TP2 rank one missing")?.exl3()?];
                        ensure!(ranks.next().is_none(), "Qwen extra TP2 rank");
                        Exl3Tp2::pushed_pair_weights(pair_devices, catalog, cuteafd_core::ExpertGeometry::QWEN4,
                            packages.each_ref().map(|path| path.as_path()), layers.clone(), rows, budgets,
                            RtxSelectionPolicy::CanonicalDecode, payloads)?
                    }
                    None => Exl3Tp2::load_pair_weights(pair_devices, catalog, cuteafd_core::ExpertGeometry::QWEN4,
                        packages.each_ref().map(|path| path.as_path()), layers.clone(), rows, budgets,
                        RtxSelectionPolicy::CanonicalDecode)?,
                };
                [Box::new(owners[0].execution()?), Box::new(owners[1].execution()?)]
            } else {
                let tensors = catalog.fp8().context("Qwen native TP2 headers")?;
                let directory = args.fp8_package.clone().unwrap_or_else(|| crate::shared::experts::fp8::package_directory(
                    &args.native_lib, 2, tensors.format()));
                match pushed {
                    Some(ranks) => {
                        let mut ranks = ranks.into_iter();
                        let payloads = [ranks.next().context("Qwen native rank zero missing")?.fp8()?,
                            ranks.next().context("Qwen native rank one missing")?.fp8()?];
                        ensure!(ranks.next().is_none(), "Qwen extra native rank");
                        Fp8MoeTp2::pushed_pair(pair_devices, tensors, &directory, layers.clone(), rows, budgets, payloads)?
                    }
                    None => Fp8MoeTp2::load_pair(pair_devices, tensors, &directory, layers.clone(), rows, budgets)?,
                }.map(|rank| Box::new(rank) as Box<dyn RtxExpertLayer>)
            };
            let ranks = [0, 1].map(|rank| crate::shared::peer_split::RankDevice {
                device: devices[rank].id, stream: streams[rank] });
            Some(engine::LocalRouted::Tp2(RefCell::new(rtx::owner_tp2::OwnerTp2::new(&self.library,
                ranks, pair, self.cfg.hidden, self.cfg.topk, rows, engine::DECODE_ROWS,
                catalog.exl3().is_some(), "qwen4")?)))
        };
        let mtp = if admitted.draft_budget > 0 {
            Some(devices[0].run(|| self.local_mtp_experts(args, self.cfg.layers, streams[0], admitted.draft_budget))?)
        } else { None };
        let remote = admitted.placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts == cuteafd_loader::placement::ExpertHome::Spark).then_some(layer)).collect::<Vec<_>>();
        let mut transport = Vec::new();
        if !remote.is_empty() {
            let peers = args.peers.as_deref().context("Qwen remote layer without Spark peers")?.split(',')
                .map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let working = cuteafd_loader::placement::SparkWorkingSet { rtx_layers: layers.clone(), spark_layers: remote,
                spark_ranks: peers.len(), prefill_lanes: 1 };
            let view = cuteafd_loader::plan::checkpoint::CheckpointView::select(source,
                cuteafd_loader::plan::checkpoint::CoreView::Original)?;
            crate::shared::placement::SparkAdmission::plan_view(&view, cuteafd_core::ExpertGeometry::QWEN4,
                Some(catalog), &working, peers.len(), super::admission::spark_model(
                    cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(catalog)?,
                    args.prefill_wave_rows(), args.resolved_prefill()?.waves,
                    self.checkpoint.spark_input_is_bf16())?, None)?.select(&peers)?;
            for device in devices {
                let link = device.run(|| self.spark_link(args, &peers))?;
                transport.push(RefCell::new(link));
            }
        }
        Ok(engine::HybridExperts { local, local_layers: layers, transport, mtp, terminal: None })
    }

    fn spark_link<'a>(&'a self, args: &EngineArgs, peers: &[std::net::SocketAddr])
        -> Result<crate::shared::spark_intake::SparkLink<'a>> {
        let shape = args.prefill_shape()?;
        let mut link = crate::shared::spark_intake::SparkLink::new_waves(&self.library,
            cuteafd_transport::expert::SparkWorld::canonical(peers.len())?, peers,
            u32::try_from(args.prefill_wave_rows())?, cuteafd_transport::TcpTransportConfig {
                timing: cuteafd_transport::protocol_v2_timing_from_env(), timeout: std::time::Duration::from_secs(120),
                max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2, shape.map_or(1, |shape| shape.waves as usize),
            u32::try_from(rows_for_link(args, shape))?)?;
        link.enable_terminal_ownership(self.library.clone())?;
        link.bootstrap(&engine::spark_session_shape(&self.cfg, args.prefill_wave_rows(),
            self.checkpoint.spark_input_is_bf16())?)?;
        Ok(link)
    }
}

fn rows_for_link(args: &EngineArgs, shape: Option<cuteafd_loader::serving_capacity::PrefillShape>) -> usize {
    shape.map_or(args.prefill_rows, |shape| shape.unit_rows as usize).max(engine::DECODE_ROWS)
}
