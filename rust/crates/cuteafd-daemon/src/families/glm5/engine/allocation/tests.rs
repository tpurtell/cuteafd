use super::*;
use crate::shared::token_io::{EmbedPlacement, EmbedSource};
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::placement::{
    families::glm5::{
        buffers::TargetBuffers,
        exchanges::{ContextOwners, ExchangeBuffers},
    },
    planner::{self, *},
};
use cuteafd_loader::{
    families::glm5::{
        core::{CoreRankLayout, CoreStageLayout},
        native::ScratchLayout,
    },
    formats::trellis_core::ProjectionScratchLayout,
    placement::families::glm5::hooks::HookBuffers,
    plan::exlr8::dense::scratch::LinearScratchLayout,
};

fn native_layout(cfg: &GlmDsaConfig, rows: usize) -> Result<ScratchLayout> {
    ScratchLayout::new(
        cfg,
        cfg.heads / 2,
        cfg.dense_intermediate / 2,
        rows,
        LinearScratchLayout {
            rotated: rows * 128 * 2,
            table: 256,
            block_scales: 256,
            partial: 1024,
            records: (rows > 64).then_some(640),
            acc: rows * 256 * 4,
        },
    )
}

fn core_layout(cfg: &GlmDsaConfig, rows: usize) -> Result<CoreRankLayout> {
    Ok(CoreRankLayout {
        body: vec![],
        stages: Some([
            CoreStageLayout::new(cfg, DECODE_ROWS, 2, false)?,
            CoreStageLayout::new(cfg, rows, 2, false)?,
        ]),
        decode: Some(ProjectionScratchLayout::new(128, 256, DECODE_ROWS)?),
        prefill: Some(ProjectionScratchLayout::new(128, 256, rows)?),
        target_head: None,
        draft_head: None,
        retained_rows: 0,
        receive_suffix: 0,
    })
}

fn plan(
    cfg: &GlmDsaConfig,
    partitioned: bool,
    owners: Option<&[u8]>,
    pages: u64,
    shape: PrefillShape,
    full: bool,
    native: bool,
    core: bool,
    shared: Option<&cuteafd_loader::plan::exlr8::ExecutionWorkspace>,
) -> Result<ResolvedPlan> {
    let mut contract = Contract::fixed(
        128,
        128,
        2,
        Facts {phase_order: cuteafd_loader::placement::planner::PhaseOrder::DRAFT_THEN_TARGET,
            allocations: vec![],
            deferred: vec![],
            sequential: vec![],
            allowances: vec![],
        },
    );
    contract.variables = vec![Variable::Prefill(vec![shape]), Variable::Lanes(vec![1])];
    for rank in 0..2u8 {
        contract.components.extend(
            TargetBuffers {
                device: rank,
                lead: rank == 0,
                page_tokens: 128,
                maximum_pages: 128,
                context: 4096,
                partitioned,
                owned_layers: (0..cfg.layers)
                    .map(|layer| owners.is_none_or(|owners| owners[layer] == rank))
                    .collect(),
                split_heads: owners.is_none(),
                router: false,
                native_head: false,
                decode: Some(Default::default()),
                prefill: Some((shape, Default::default())),
                full_prefill_logits: full,
                payload_row_bytes: 0,
                core_rows: false,
            }
            .components(cfg)?,
        );
        contract.components.extend(
            HookBuffers {
                device: rank,
                maximum_pages: 128,
                native: native
                    .then(|| {
                        Ok::<_, anyhow::Error>([
                            native_layout(cfg, DECODE_ROWS)?,
                            native_layout(cfg, shape.rows as usize)?,
                        ])
                    })
                    .transpose()?,
                core: core
                    .then(|| core_layout(cfg, shape.rows as usize))
                    .transpose()?,
                shared_workspace: shared.cloned(),
            }
            .components()?,
        );
    }
    let facts = contract.facts(&[0, 0], pages / 2)?;
    let budgets: Vec<_> = [
        Location::coordinator(0),
        Location::coordinator(1),
        Location::PINNED,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, location)| Budget {
        id: format!("fixture.{i}"),
        locations: vec![location],
        physical_bytes: 1 << 40,
        logical_bytes: 1 << 40,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "CPU fake DSO only".into(),
            assumed: false,
        },
        headroom: 0,
    })
    .collect();
    let phases = planner::evaluate(&facts, &budgets)?;
    Ok(ResolvedPlan {
        request: Request {
            context: 4096,
            pool: Pin::Exact(pages * 64),
            onboard: Pin::Auto,
            prefill: Pin::Exact(shape),
            operation: Operation::Serving,
        },
        decisions: contract.decisions(&[0, 0])?,
        pool_tokens: pages * 64,
        budgets,
        facts,
        phases,
    })
}

fn exchange_plan(cfg: &GlmDsaConfig, buffers: &ExchangeBuffers) -> Result<ResolvedPlan> {
    let mut contract = Contract::fixed(
        64,
        buffers.maximum_pages,
        2,
        Facts {phase_order: cuteafd_loader::placement::planner::PhaseOrder::DRAFT_THEN_TARGET,
            allocations: vec![],
            deferred: vec![],
            sequential: vec![],
            allowances: vec![],
        },
    );
    contract.variables = vec![
        Variable::Prefill(vec![buffers
            .context
            .map_or(PrefillShape { rows: 64, waves: 1 }, |c| c.prefill)]),
        Variable::Lanes(vec![1]),
    ];
    contract.components = buffers.components(cfg)?;
    let facts = contract.facts(&[0, 0], 1)?;
    let budgets: Vec<_> = [
        Location::coordinator(0),
        Location::coordinator(1),
        Location::PINNED,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, location)| Budget {
        id: format!("fixture.exchange.{index}"),
        locations: vec![location],
        physical_bytes: 1 << 40,
        logical_bytes: 1 << 40,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "CPU fake DSO only".into(),
            assumed: false,
        },
        headroom: 0,
    })
    .collect();
    let phases = planner::evaluate(&facts, &budgets)?;
    Ok(ResolvedPlan {
        request: Request {
            context: 64,
            pool: Pin::Exact(64),
            onboard: Pin::Auto,
            prefill: Pin::Auto,
            operation: Operation::Serving,
        },
        decisions: contract.decisions(&[0, 0])?,
        pool_tokens: 64,
        budgets,
        facts,
        phases,
    })
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn actual_exchange_owners_match_independent_contract() -> Result<()> {
    let fixture = Fixture::build()?;
    let cfg = GlmDsaConfig::from_hf(&cuteafd_loader::plan::testing::glm5_config())?;
    let directory = tempfile::tempdir_in(fixture.directory())?;
    let path = directory.path().join("embedding.bf16");
    std::fs::write(&path, vec![0; cfg.vocab_size * cfg.hidden * 2])?;
    let diagnostics = std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1");
    for (context, waves, reverse, heterogeneous, abort) in [
        (false, 1, false, false, true),
        (false, 1, true, true, false),
        (true, 3, false, false, true),
        (true, 1, true, true, true),
    ] {
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        fixture.immediate_copies(&library)?;
        fixture.glm_target_programs(&library)?;
        for device in 0..2 {
            fixture.configure_memory(&library, device, 8 << 30)?;
        }
        let programs = library.programs()?;
        let devices = if reverse { [1, 0] } else { [0, 1] };
        library.cuda_set_device(devices[0])?;
        let weights = GlmWeights {
            layers: GlmLayer::allocation_fixture(&library, &cfg, 0, None)?,
            norm: DeviceAllocation::new(&library, 256)?,
            head: super::super::super::weights::GlmHead::Bf16(DeviceAllocation::new(
                &library, 256,
            )?),
            rope_alpha: None,
        };
        let (embedding, ()) = TokenEmbedding::load(
            &library,
            EmbedSource {
                path: path.clone(),
                offset: 0,
                vocab: cfg.vocab_size,
                hidden: cfg.hidden,
                snapshot: None,
                tensor_name: "fixture.embedding".into(),
            },
            EmbedPlacement::Gpu,
            || Ok(()),
        )?;
        let stream = std::ptr::null_mut();
        let scratch = GlmContext {
            topk_scratch: 17,
            partial_scratch: 1024,
        };
        let mut engine = GlmEngine::new(
            &library,
            &programs,
            cfg.clone(),
            weights,
            stream,
            4096,
            128,
            2,
            embedding,
            crate::shared::prefill_depth::PrefillDepth::new(waves, 128)?,
            context.then_some(scratch),
        )?;
        let layers = crate::shared::peer_split::on_device(&library, devices[1], || {
            GlmLayer::allocation_fixture(&library, &cfg, 1, None)
        })?;
        engine.attach_peer(devices[1], stream, layers)?;
        let sizes = if heterogeneous {
            vec![16, 256, 512]
        } else {
            vec![128 * cfg.hidden * 2; 4 * waves]
        };
        if heterogeneous {
            let ranks = devices.map(|device| RankDevice { device, stream });
            let decode_bytes = decode_peer_bytes(DECODE_ROWS, cfg.hidden, engine.tp2_exchange);
            engine.exchange = Some(if abort {
                PeerExchange::new_abortable_sized(&library, ranks, &sizes, decode_bytes)?
            } else {
                // The shared non-abortable constructor is also an independent physical arm.
                PeerExchange::new(&library, ranks, 3, 512, decode_bytes)?
            });
        }
        let receive_bytes = if !abort { vec![512; 3] } else { sizes.clone() };
        let extent = if reverse { 65 } else { 4096 };
        let buffers = ExchangeBuffers {
            physical_devices: devices,
            maximum_pages: 128,
            receive_bytes: receive_bytes
                .into_iter()
                .map(|bytes| bytes as u64)
                .collect(),
            terminal_abort: abort,
            diagnostics,
            context: context.then_some(ContextOwners {
                extent,
                prefill: PrefillShape {
                    rows: 128,
                    waves: waves as u64,
                },
                scratch,
            }),
        };
        let expected = exchange_plan(&cfg, &buffers)?;
        if context {
            assert!(planner::reconcile(
                &expected,
                Phase::PostTargetKv,
                &engine.exchange_receipts()?
            )
            .is_err());
            // Real constructor and production manifest selector, without numerical binding or launches.
            let manifest = serde_json::json!({"programs": [
                {"name": "glm_sparse_mla_partial_h0_s1_m64"},
                {"name": "glm_sparse_mla_partial_h32_s1_m64"},
            ]});
            engine.attach_context(extent as usize, &manifest)?;
        }
        let actual = engine.exchange_receipts()?;
        for rank in 0..2 {
            let prefix = format!("glm.exchange.rank.{rank}.");
            assert!(actual.iter().filter(|r| r.allocation.id.starts_with(&prefix)
                && r.allocation.location != Location::PINNED)
                .all(|r| r.allocation.location == Location::coordinator(rank as u8)));
        }
        planner::reconcile(&expected, Phase::PostTargetKv, &actual)?;
        assert_eq!(
            actual
                .iter()
                .filter(|r| r.allocation.id.ends_with("peer.abort"))
                .count(),
            if abort { 2 } else { 0 }
        );
        assert_eq!(
            actual
                .iter()
                .filter(|r| r.allocation.id.ends_with("peer.progress"))
                .count(),
            if diagnostics { 2 } else { 0 }
        );
        if let Some(split) = &engine.context {
            for (rank, bytes) in split.bytes().into_iter().enumerate() {
                let prefix = format!("glm.exchange.rank.{rank}.context.");
                assert_eq!(
                    actual
                        .iter()
                        .filter(|r| r.allocation.id.starts_with(&prefix)
                            && r.allocation.id != format!("{prefix}exchange")
                            && r.allocation.location != Location::PINNED)
                        .map(|r| r.allocation.bytes as usize)
                        .sum::<usize>(),
                    bytes
                );
            }
        }
        for defect in 0..7 {
            let mut broken = actual.clone();
            match defect {
                0 => {
                    broken.remove(0);
                }
                1 => {
                    let mut extra = broken[0].clone();
                    extra.allocation.id.push_str(".unexpected");
                    broken.push(extra);
                }
                2 => broken[0].allocation.bytes += 1,
                3 => {
                    broken[0].allocation.bytes += 1;
                    broken[1].allocation.bytes -= 1;
                }
                4 => broken[0].allocation.location = Location::coordinator(1),
                5 => broken[0].address = 0,
                _ => broken[1].address = broken[0].address,
            }
            assert!(
                planner::reconcile(&expected, Phase::PostTargetKv, &broken).is_err(),
                "defect {defect}"
            );
        }
        if context {
            let mut broken = actual.clone();
            broken
                .iter_mut()
                .find(|r| r.allocation.id.ends_with("context.upload.gather.0"))
                .unwrap()
                .allocation
                .bytes += 1;
            assert!(planner::reconcile(&expected, Phase::PostTargetKv, &broken).is_err());
            let mut broken = actual.clone();
            broken.retain(|r| !r.allocation.id.ends_with("context.sink"));
            assert!(planner::reconcile(&expected, Phase::PostTargetKv, &broken).is_err());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn actual_target_buffers_and_aliases_match_independent_contract() -> Result<()> {
    let fixture = Fixture::build()?;
    let cfg = GlmDsaConfig::from_hf(&cuteafd_loader::plan::testing::glm5_config())?;
    let directory = tempfile::tempdir_in(fixture.directory())?;
    let path = directory.path().join("embedding.bf16");
    std::fs::write(&path, vec![0; cfg.vocab_size * cfg.hidden * 2])?;
    for ((partitioned, owned, waves, full, pages, native, core), devices) in [
        (false, false, 1, false, 2, false, false),
        (true, false, 3, true, 64, false, false),
        (false, true, 3, false, 66, false, false),
        (false, false, 3, false, 2, true, true),
    ].into_iter().flat_map(|case| [[0, 1], [1, 0]].map(|devices| (case, devices))) {
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        fixture.immediate_copies(&library)?;
        fixture.glm_target_programs(&library)?;
        for device in 0..2 {
            fixture.configure_memory(&library, device, 8 << 30)?;
        }
        let programs = library.programs()?;
        let shared_manifest = crate::shared::experts::exlr8::owner_manifest(cuteafd_ffi::exlr8::KernelFamily::Glm);
        let shared_layout = core.then(|| cuteafd_loader::plan::exlr8::shared_workspace_layout(
            &shared_manifest, cuteafd_ffi::exlr8::KernelFamily::Glm)).transpose()?;
        let shared_path = directory.path().join("shared-PROGRAMS.json");
        std::fs::write(&shared_path, serde_json::to_vec(&shared_manifest)?)?;
        let shared_programs;
        // Only metadata and real allocation constructors run; no numerical program launches.
        assert!(programs.spec("glm_index_producer_m64").is_ok());
        library.cuda_set_device(devices[0])?;
        let owners = owned.then_some([0, 1]);
        let owners = owners.as_ref().map(|a| a.as_slice());
        let shape = PrefillShape { rows: 128, waves };
        let weights = GlmWeights {
            layers: GlmLayer::allocation_fixture(&library, &cfg, 0, owners)?,
            norm: DeviceAllocation::new(&library, 256)?,
            head: super::super::super::weights::GlmHead::Bf16(DeviceAllocation::new(
                &library, 256,
            )?),
            rope_alpha: None,
        };
        let (embedding, ()) = TokenEmbedding::load(
            &library,
            EmbedSource {
                path: path.clone(),
                offset: 0,
                vocab: cfg.vocab_size,
                hidden: cfg.hidden,
                snapshot: None,
                tensor_name: "fixture.embedding".into(),
            },
            EmbedPlacement::Gpu,
            || Ok(()),
        )?;
        let [lead_stream, peer_stream] = devices.map(|id| crate::shared::memory::device::Stream::new(
            crate::shared::memory::device::Device { library: &library, id }));
        let (lead_stream, peer_stream) = (lead_stream?, peer_stream?);
        let stream = lead_stream.raw;
        let mut engine = GlmEngine::new(
            &library,
            &programs,
            cfg.clone(),
            weights,
            stream,
            4096,
            128,
            pages as usize,
            embedding,
            crate::shared::prefill_depth::PrefillDepth::new(waves as usize, 128)?,
            partitioned.then_some(GlmContext::default()),
        )?;
        engine.full_prefill_logits = full;
        let layers = crate::shared::peer_split::on_device(&library, devices[1], || {
            GlmLayer::allocation_fixture(&library, &cfg, 1, owners)
        })?;
        engine.attach_peer(devices[1], peer_stream.raw, layers)?;
        if native {
            // Only an ownership marker is loaded; no numerical native programs run.
            engine.weights.layers[0]
                .native
                .insert("fixture".into(), DeviceAllocation::new(&library, 256)?);
            engine.native_scratch = (0..2)
                .map(|_| {
                    Ok([
                        native_layout(&cfg, DECODE_ROWS)?,
                        native_layout(&cfg, shape.rows as usize)?,
                    ])
                })
                .collect::<Result<_>>()?;
        }
        let mut expected = plan(&cfg, partitioned, owners, pages, shape, full, native, core, shared_layout.as_ref())?;
        assert!(
            planner::reconcile(&expected, Phase::Warmup, &engine.target_buffer_receipts()?)
                .is_err()
        );
        *engine.decode_workspace.borrow_mut() = Some(engine.workspace(DECODE_ROWS, true)?);
        let _ = engine.peer_workspaces(true, 1)?;
        engine.ensure_prefill_slots()?;
        fixture.exlr8_programs(&library, false)?;
        shared_programs = library.programs()?.with_manifest(&shared_path)?;
        if core {
            use crate::shared::experts::exlr8::{NativeExperts, NativePlan, shared_residency};
            use cuteafd_loader::plan::exlr8::residency::WeightLayout;
            let family = cuteafd_ffi::exlr8::KernelFamily::Glm;
            for (rank, &physical) in devices.iter().enumerate() {
                let layouts = (cfg.first_moe_layer..cfg.layers)
                    .map(|layer| shared_residency(family, layer, 2, rank)).collect::<Vec<_>>();
                expected.facts.allocations.extend(WeightLayout::from_residencies(
                    &format!("glm.experts.rank.{rank}.shared"), &layouts.iter().collect::<Vec<_>>(),
                    WeightArenaTopology::PerLayer, Location::coordinator(rank as u8), Lifetime::RESIDENT)?.allocations);
                engine.native_experts.push(crate::shared::peer_split::on_device(&library, physical, || {
                    NativeExperts::load(&library, NativePlan::allocation_fixture(family, layouts,
                        shared_layout.as_ref().unwrap().bytes))
                })?);
            }
            expected.phases = planner::evaluate(&expected.facts, &expected.budgets)?;
            // Allocate through the production CoreScratch constructor; program binding is not
            // emulated. The retained workspaces expose the actual projection owners and views.
            for rank in 0..2 {
                let device = crate::shared::memory::device::Device {
                    library: &library,
                    id: devices[rank],
                };
                let layout = core_layout(&cfg, shape.rows as usize)?;
                let allocate = |index: usize| {
                    let projection = if index == 0 {
                        layout.decode.as_ref()
                    } else {
                        layout.prefill.as_ref()
                    }
                    .unwrap();
                    super::super::super::core_runtime::CoreScratch::new(
                        device,
                        &crate::shared::dense_trellis::ProjectionLayout {
                            extents: projection.extents.clone(),
                            rows: projection.rows,
                        },
                        &layout.stages.as_ref().unwrap()[index],
                    )
                    .map(Rc::new)
                };
                let decode = allocate(0)?;
                let prefill = allocate(1)?;
                let shared = shared_layout.as_ref().unwrap();
                let shared_execution = || device.run(|| {
                    let mut execution = crate::shared::experts::exlr8::Exlr8Execution::new_shared(
                        &library, &shared_programs, cuteafd_ffi::exlr8::KernelFamily::Glm,
                        DECODE_ROWS, shared.bytes)?;
                    execution.seal_graph_storage()?;
                    Ok(Rc::new(execution))
                });
                let shared_decode = shared_execution()?;
                let shared_prefill = shared_execution()?;
                let (decode_slot, prefill_slots) = if rank == 0 {
                    (&engine.decode_workspace, &engine.prefill_slots)
                } else {
                    let peer = engine.peer()?;
                    (&peer.decode_workspace, &peer.prefill_slots)
                };
                decode_slot.borrow_mut().as_mut().unwrap().core = Some(decode);
                decode_slot.borrow_mut().as_mut().unwrap().native_shared = Some(shared_decode);
                for slot in prefill_slots.borrow_mut().iter_mut() {
                    slot.core = Some(prefill.clone());
                    slot.native_shared = Some(shared_prefill.clone());
                }
            }
        }
        if core {
            engine.native_experts.swap(0, 1);
            assert!(engine.expert_receipts().is_err(), "shared experts must follow engine rank order");
            engine.native_experts.swap(0, 1);
            let peer = engine.native_experts.pop().unwrap();
            assert!(engine.expert_receipts().is_err(), "shared experts must cover all engine ranks");
            engine.native_experts.push(peer);
            engine.cfg.first_moe_layer += 1;
            assert!(engine.expert_receipts().is_err(), "shared experts must match target layer inventory");
            engine.cfg.first_moe_layer -= 1;
        }
        // Emulate the actual warmup metadata writers without launching a numerical kernel.
        for rank in 0..2 {
            let decode = if rank == 0 {
                &engine.decode_workspace
            } else {
                &engine.peer()?.decode_workspace
            };
            let decode = decode.borrow();
            let w = decode.as_ref().context("CPU decode workspace")?;
            engine.on(rank, || {
                let tables = StepTables {
                    segments: Vec::new(),
                    decode: true,
                    positions: vec![0],
                    slots: vec![0],
                    page_table: vec![0],
                    table_width: 1,
                    table_stride: 1,
                    cache_lengths: vec![1],
                    lengths: vec![1],
                    exchange_rows: 1,
                };
                engine.put_tables(w, &tables)?;
                if rank == 0 {
                    engine.put(&w.ids, &[0u32])?;
                }
                Ok(())
            })?;
        }
        let receipts = || -> Result<Vec<Receipt>> {
            let mut actual = engine.target_buffer_receipts()?;
            actual.extend(engine.target_hook_receipts()?);
            actual.extend(engine.expert_receipts()?);
            Ok(actual)
        };
        let ordered_owners = || -> Result<BTreeMap<String, (Allocation, u64)>> {
            Ok(engine.target_buffer_receipts()?.into_iter()
                .filter(|receipt| receipt.allocation.location == Location::PINNED)
                .map(|receipt| (receipt.allocation.id.clone(), (receipt.allocation, receipt.address)))
                .collect())
        };
        let ordered_before = ordered_owners()?;
        for rank in 0..2 {
            for kind in ["decode", "prefill"] {
                let prefix = format!("glm.target.gpu.{rank}.{kind}.upload.");
                assert!(ordered_before.keys().any(|id| id.starts_with(&prefix)),
                    "missing ordered {kind} stores before sync-only release on rank {rank}");
            }
        }
        // Production releases load-only synchronous staging, not the reserved ordered stores.
        assert!(library.release_sync_h2d_staging()? > 0);
        assert_eq!(ordered_owners()?, ordered_before,
            "sync-only staging release must preserve every ordered owner and address");
        let actual = receipts()?;
        for rank in 0..2 {
            let prefix = format!("glm.target.gpu.{rank}.");
            assert!(actual.iter().filter(|r| r.allocation.id.starts_with(&prefix)
                && r.allocation.location != Location::PINNED)
                .all(|r| r.allocation.location == Location::coordinator(rank as u8)));
        }
        planner::reconcile(&expected, Phase::Warmup, &actual)?;
        // Independent serving-only admission: the prefix's registered target pools
        // remain aliases and must not appear as new serving backings.
        {
            use crate::shared::{prefix::CudaCopyEngine, token_io::{SelectPlacement, TokenSelector}};
            use cuteafd_engine::prefix::{PrefixCache, PrefixConfig, PrefixFamily};
            use planner::serving::{HostPrefixOwners, SelectorOwners, ServingOwners};
            let physical = engine.census_physical_devices()?;
            let mapping = CoordinatorDevices::new(&physical)?;
            let tag = format!("glm-serving/{devices:?}/{partitioned}/{owned}/{waves}/{core}");
            let _owner = cuteafd_ffi::memory_ledger::owner(&tag);
            for placement in [SelectPlacement::Host, SelectPlacement::Device] {
                let family = engine.on(0, || super::super::super::prefix::GlmPrefix::new(&engine, false, 3))?;
                let mut selector = engine.on(0, || TokenSelector::new(&library, placement, cfg.vocab_size, DECODE_ROWS))?;
                let copy = CudaCopyEngine::registered(&library, &family.buffers())?;
                let mut cache = PrefixCache::new(family.layout(), PrefixConfig { entries: 1, mark_slots: 0,
                    keep_logits: false, min_tokens: 1 }, Some((cuteafd_hostcache::config::Config {
                        bytes: 2 * (1 << 20) + 17, chunk_bytes: 1 << 20, min_tokens: 1, ..Default::default()
                    }, copy)))?;
                let serving = ServingOwners { selector: SelectorOwners { location: Location::coordinator(0),
                    capacity: DECODE_ROWS, vocab: cfg.vocab_size, device_sampling: placement == SelectPlacement::Device },
                    prefix: HostPrefixOwners {quota_bytes: 2 * (1 << 20) + 17, chunk_bytes: 1 << 20} };
                let mut subset = expected.clone();
                subset.facts.allocations = serving.allocations()?;
                subset.facts.allocations.extend(cuteafd_loader::placement::families::glm5::buffers::PrefixBuffers {
                    device: 0, draft_windows: None, mark_slots: 3 }.allocations(ServingOwners::LIFETIME)?);
                subset.phases = planner::evaluate(&subset.facts, &subset.budgets)?;
                let early = super::super::super::serve::serving_allocation_receipts(&family, &selector, &mut cache, mapping)?;
                assert_eq!(planner::reconcile(&subset, Phase::Ready, &early).is_ok(), placement == SelectPlacement::Host);
                selector.reserve_sampler()?;
                let receipts = super::super::super::serve::serving_allocation_receipts(&family, &selector, &mut cache, mapping)?;
                planner::reconcile(&subset, Phase::Ready, &receipts)?;
                let live = cuteafd_ffi::memory_ledger::allocations().into_iter()
                    .filter(|r| r.key.owner == Some(tag.as_str()))
                    .map(|r| ((r.key.space, r.key.device, r.address), r.bytes)).collect::<BTreeMap<_, _>>();
                let owners = receipts.iter().map(|r| {
                    let (space, device) = match r.allocation.location {
                        Location {device: planner::Device::Coordinator(rank), domain: Domain::Device} =>
                            (cuteafd_ffi::memory_ledger::Space::Device, mapping.physical_device(rank).unwrap()),
                        Location::PINNED => (cuteafd_ffi::memory_ledger::Space::Pinned, -1),
                        other => panic!("unexpected serving location {other:?}"),
                    };
                    ((space, device, r.address as usize), r.allocation.bytes as usize)
                }).collect::<BTreeMap<_, _>>();
                assert_eq!(owners.len(), receipts.len(), "serving backings must be unique");
                assert_eq!(owners, live, "prefix target aliases must not charge twice");
                for defect in 0..5 {
                    let mut wrong = receipts.clone();
                    match defect {
                        0 => {wrong.remove(0);},
                        1 => {let mut extra = wrong[0].clone(); extra.allocation.id.push_str(".extra"); wrong.push(extra);},
                        2 => wrong[0].allocation.bytes += 1,
                        3 => {wrong[0].allocation.bytes += 1; wrong[1].allocation.bytes -= 1;},
                        4 => wrong[0].address = wrong[1].address,
                        _ => unreachable!(),
                    }
                    assert!(planner::reconcile(&subset, Phase::Ready, &wrong).is_err(), "serving defect {defect}");
                }
                drop(cache); drop(selector); drop(family);
                assert!(cuteafd_ffi::memory_ledger::allocations().iter().all(|r| r.key.owner != Some(tag.as_str())));
            }
        }
        if core {
            let slots = engine.prefill_slots.borrow();
            let original = slots[0].core.as_ref().unwrap().clone();
            drop(slots);
            let device = crate::shared::memory::device::Device {
                library: &library,
                id: devices[0],
            };
            let layout = core_layout(&cfg, shape.rows as usize)?;
            let projection = layout.prefill.as_ref().unwrap();
            let replacement = Rc::new(super::super::super::core_runtime::CoreScratch::new(
                device,
                &crate::shared::dense_trellis::ProjectionLayout {
                    extents: projection.extents.clone(),
                    rows: projection.rows,
                },
                &layout.stages.as_ref().unwrap()[1],
            )?);
            engine.prefill_slots.borrow_mut()[1].core = Some(replacement);
            assert!(engine.target_hook_receipts().is_err());
            engine.prefill_slots.borrow_mut()[1].core = Some(original);
            planner::reconcile(&expected, Phase::Warmup, &receipts()?)?;
            let original = engine.prefill_slots.borrow()[1].native_shared.clone();
            let replacement = device.run(|| crate::shared::experts::exlr8::Exlr8Execution::new_shared(
                &library, &shared_programs, cuteafd_ffi::exlr8::KernelFamily::Glm,
                DECODE_ROWS, shared_layout.as_ref().unwrap().bytes).map(Rc::new))?;
            engine.prefill_slots.borrow_mut()[1].native_shared = Some(replacement);
            assert!(engine.target_hook_receipts().is_err(), "shared hook slots must retain identical owners");
            engine.prefill_slots.borrow_mut()[1].native_shared = original;
            planner::reconcile(&expected, Phase::Warmup, &receipts()?)?;
        }
        let mut wrong = actual.clone();
        let index = wrong
            .iter()
            .position(|r| r.allocation.id.ends_with("prefill.rows.h"))
            .unwrap();
        wrong[index].allocation.bytes -= 1;
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        wrong = actual.clone();
        wrong.remove(index);
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        wrong = actual.clone();
        let mut extra = actual[index].clone();
        extra.allocation.id.push_str(".unexpected");
        wrong.push(extra);
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        wrong = actual.clone();
        let view = wrong
            .iter_mut()
            .find(|r| r.allocation.id.ends_with("prefill.slot.0.h"))
            .unwrap();
        view.address += 1;
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        if waves > 1 {
            let slots = engine.prefill_slots.borrow();
            let original = slots[0].h.clone();
            drop(slots);
            let saved = engine.prefill_slots.borrow()[1].h.clone();
            engine.prefill_slots.borrow_mut()[1].h = original;
            let aliased = receipts()?;
            assert!(planner::reconcile(&expected, Phase::Warmup, &aliased).is_err());
            engine.prefill_slots.borrow_mut()[1].h = saved;
        }
        if core {
            use crate::shared::peer_split::TerminalState;
            engine.require_live()?;
            engine.retire_tp2()?;
            assert_eq!(engine.terminal.get(), TerminalState::Drained);
            assert_eq!(engine.exchange.as_ref().unwrap().terminal_state(), TerminalState::Drained);
            assert!(engine.require_live().is_err());
            engine.retire_tp2()?;
        }
    }
    Ok(())
}
