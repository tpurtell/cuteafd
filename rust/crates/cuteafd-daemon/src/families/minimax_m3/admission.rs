//! Fixed cold-golden owners, admitted before target/cache/exchange allocation.
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_loader::{families::minimax_m3::{MiniMaxM3Config, execution::*, weight_layout::CoordinatorWeights},
    placement::{planner::*, PreparedSparkLink}, plan::Checkpoint};

fn sync_h2d_bytes(checkpoint: &Checkpoint, weights: &CoordinatorWeights, rows: u32, tokens: u32) -> Result<u64> {
    use cuteafd_loader::shared::dense_read::{ReadBatch, STEP_BYTES};
    // One library-wide backing survives DenseLoad::release. The mount source
    // is different storage and is live simultaneously with this copy buffer.
    let mut bytes = (u64::from(tokens) * 256).max(u64::from(rows) * 8).max(16);
    for (name, view) in weights.arena.views() {
        let header = checkpoint.header(name).with_context(|| format!("MiniMax upload lacks {name}"))?;
        let count = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
        ensure!(count > 0 && view.bytes() % count == 0, "MiniMax upload row extent");
        let width = view.bytes() / count;
        bytes = bytes.max(ReadBatch::new(count, width, width, STEP_BYTES)?.packed_bytes as u64);
    }
    Ok(bytes)
}

pub(crate) fn reconcile_sync_h2d(library: &cuteafd_ffi::NativeLibrary, facts: &Facts,
    inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
    let planned = facts.allocations.iter().find(|owner| owner.id == "sync-h2d")
        .context("MiniMax synchronous upload owner was not admitted")?;
    let buffer = library.sync_h2d_staging_buffer()?.context("MiniMax synchronous upload backing missing")?;
    ensure!(planned.location == Location::PINNED && planned.view.is_none()
        && !buffer.ptr.is_null() && buffer.bytes > 0 && buffer.bytes as u64 <= planned.bytes
        && buffer.flags & cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_PINNED != 0,
        "MiniMax synchronous upload backing {} exceeds/violates admitted owner {}",
        buffer.bytes, planned.bytes);
    let mut allocation = planned.clone();
    allocation.bytes = buffer.bytes as u64;
    inventory.extend(vec![Receipt { allocation, address: buffer.ptr as u64 }])?;
    tracing::info!(admitted_bytes = planned.bytes, resident_bytes = buffer.bytes,
        "MiniMax synchronous upload owner reconciled");
    Ok(())
}

pub(crate) fn contract(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, weights: &CoordinatorWeights,
    manifest: &serde_json::Value, rows: u32, tokens: u32, waves: usize,
    link: &PreparedSparkLink, arch: &str) -> Result<Contract> {
    let pages = u64::from(tokens.div_ceil(64));
    let empty = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![],
        allocations: vec![], deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
        driver_page_tables: vec![], allocation_granularity: vec![] };
    let mut contract = Contract::fixed(64, pages, 1, empty);
    // Fixed domains follow placement::families::kimi::KimiRecipe::prepare: even a
    // cold one-shape contract must declare prefill and lane decisions.
    contract.variables = vec![Variable::Prefill(vec![PrefillShape {
        rows: u64::from(rows), waves: u64::try_from(waves)? }]), Variable::Lanes(vec![1])];
    let c = &mut contract.components;
    c.extend(weight_arena_components(&weights.arena, Location::coordinator(0), Lifetime::RESIDENT,
        pages, MemoryKind::Weights)?);
    c.extend(MsaCache::new(cfg.layers as u32, pages as u32, 1, tokens)?.components("cache", 0, pages)?);
    c.extend(MsaRope::new(cfg, tokens)?.components("cache.rope", 0, pages)?);
    c.extend(HeadLayout::new(cfg, rows)?.components("head", 0, pages)?);
    let (msa, _) = MsaPrograms::resolve(cfg, manifest, 1, rows, tokens)?;
    let (producers, _) = ProducerPrograms::resolve(manifest, rows)?;
    ForwardPrograms::resolve(manifest, rows)?;
    weights.resolve_dense_programs(manifest, rows)?;
    for slot in 0..waves {
        let prefix = format!("unit.{slot}");
        c.extend(UnitLayout::new(cfg, rows)?.components(&prefix, 0, Lifetime::TARGET, pages)?);
        c.extend(UnitMetadata::new(rows)?.components(&format!("metadata.{slot}"), 0, pages)?);
        c.extend(weights.dense_input(rows)?.components(&format!("{prefix}.dense"), 0, Lifetime::TARGET, pages)?);
        c.extend(weights.dense_workspace(rows)?.components(&format!("{prefix}.dense"), 0, Lifetime::TARGET, pages)?);
        c.extend(msa.workspace_components(&format!("{prefix}.attention"), 0, true, Lifetime::TARGET, pages)?);
        c.extend(producers.workspace_components(&format!("{prefix}.attention"), 0, Lifetime::TARGET, pages)?);
    }
    let mut raw = vec![Allocation::buffer("residual", Location::coordinator(0), u64::from(tokens) * 6144 * 4,
        Lifetime::TARGET, Category::Workspace, "cold scoring residual"),
        Allocation::buffer("sync-h2d", Location::PINNED, sync_h2d_bytes(checkpoint, weights, rows, tokens)?,
            Lifetime::RESIDENT, Category::Staging, "library-wide synchronous H2D backing")];
    // project downloads bytes and decodes an FP32 vector before releasing them.
    for name in ["head.download", "head.host-logits"] {
        raw.push(Allocation::buffer(name, Location::HOST,
            u64::from(rows) * cfg.vocab_size as u64 * 4, Lifetime::TARGET,
            Category::Workspace, "cold scoring head download"));
    }
    let mount = weights.mount_load(checkpoint)?;
    for (name, location, bytes) in [("source", Location::PINNED, mount.pinned_source),
        ("rows", Location::HOST, mount.full_row_cache), ("tables", Location::PINNED, mount.pinned_tables)] {
        if bytes > 0 { raw.push(Allocation::buffer(format!("mount.{name}"), location, bytes,
            Lifetime { first: Phase::Load, last: Phase::Load }, Category::Staging, "target mount")); }
    }
    let buffers: Vec<_> = raw.into_iter().map(|owner| {
        let bytes = owner.bytes;
        Component::allocation(owner, ByteCurve::fixed(bytes, pages), false, pages)
    }).collect();
    c.extend(buffers);
    c.extend(link.components(pages)?);
    let head = crate::shared::target_head::TargetHead::configured().blas_head(true);
    c.extend(serving::blas_components(arch, &[serving::BlasOwners {
        name: "minimax/head".into(), process: "coordinator".into(), thread: "golden".into(),
        location: Location::coordinator(0), gemm: head.gemm(), lt: head.lt(), lifetime: Lifetime::TARGET,
        activation: Activation::always() }], pages, PhaseOrder::TARGET_THEN_DRAFT)?);
    let stream = Allocation::buffer("driver.stream-slabs", Location::coordinator(0), cuda_stream_slab_bytes(1)?,
        Lifetime::RESIDENT, Category::Runtime, "one ordered golden stream");
    c.push(Component::allocation(stream.clone(), ByteCurve::fixed(stream.bytes, pages), false, pages));
    fixed_request_tails(c, pages, &[])?;
    contract.add_driver_page_tables();
    Ok(contract)
}

pub(crate) fn admit(contract: &Contract, free: u64, total: u64, host: (u64, u64)) -> Result<Facts> {
    let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
    let host = cuteafd_loader::placement::families::qwen4::host::HostMemory { total: host.0, available: host.1 };
    let budgets = vec![Budget { id: "rtx0".into(), locations: vec![Location::coordinator(0)],
        physical_bytes: total, logical_bytes: policy.gpu_cap_bytes(total),
        baseline: Baseline::Sampled { free, phase: Phase::Load, source: "MiniMax pre-weight CUDA sample after modules".into() },
        headroom: cuteafd_loader::placement::reserve::gpu("minimax_m3") }, host.budget("MiniMax golden")?];
    let selected = vec![0; contract.variables.len()];
    let facts = contract.facts(&selected, contract.maximum_pages)?;
    let phases = evaluate(&facts, &budgets)?;
    tracing::info!(record = %serde_json::json!({"facts":facts,"budgets":budgets,"phases":phases}),
        "MiniMax cold golden admission");
    Ok(facts)
}

/// Serving admission (#1856 H1): the same fact gatherer and shared recipe as
/// `cuteafd plan --layout`, with one measured CUDA sample per coordinator GPU
/// in place of the planned forecast. Startup (stream D) calls this before any
/// target, cache, lane or link allocation and allocates exactly the admitted
/// owners; serve itself stays refused until the minimum serving contract
/// qualifies (`commands/family.rs`).
pub(crate) mod serve_admission {
    use anyhow::{Context, Result};
    use cuteafd_loader::families::minimax_m3::admission::{gather, Coordinator, FactsRequest};
    use cuteafd_loader::placement::families::minimax_m3::{self as shared, AdmittedMiniMax};

    /// One GPU's pre-weight sample: (physical total, logical cap, free, code not yet loaded).
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Sample { pub total: u64, pub cap: u64, pub free: u64, pub pending_code: u64 }

    pub(crate) fn admit(mut facts: FactsRequest<'_>, samples: &[Sample],
        onboard: cuteafd_loader::placement::Onboard) -> Result<AdmittedMiniMax> {
        facts.coordinator = Coordinator::Measured {
            samples: samples.iter().map(|s| (s.total, s.cap, s.free)).collect(),
            loaded_code: samples.iter().map(|s| s.pending_code).collect() };
        let inputs = gather(&facts).map_err(|refused| anyhow::anyhow!("{}; {}", refused.what, refused.how))?;
        let request = shared::request(inputs.context, facts.pool_tokens, onboard, facts.prefill,
            cuteafd_loader::placement::planner::Operation::Serving, inputs.resident_weight_limit, facts.replies);
        let admitted = inputs.admit(request).context("MiniMax serving admission")?;
        cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "minimax_m3");
        Ok(admitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::receipts::Inventory;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::serving_capacity::WeightArenaLayout;

    #[test]
    fn cold_shape_domains_resolve_before_fixed_owner_facts() -> Result<()> {
        let facts = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![],
            allocations: vec![], deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
            driver_page_tables: vec![], allocation_granularity: vec![] };
        let mut contract = Contract::fixed(64, 256, 1, facts);
        contract.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 1024, waves: 3 }]),
            Variable::Lanes(vec![1])];
        let decision = contract.decisions(&[0, 0])?;
        assert_eq!(decision.prefill, PrefillShape { rows: 1024, waves: 3 });
        assert_eq!(decision.lanes, 1);
        assert!(contract.facts(&[0, 0], 256)?.allocations.is_empty());
        assert!(contract.facts(&[], 256).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires installed vendor headers and CUTEAFD_PROGRAM_MANIFEST; CPU only"]
    fn installed_minimax_cold_contract_admits_before_allocation() -> Result<()> {
        use crate::shared::{memory::device::{Allocation as DeviceAllocation, Device}, placement::spark_rings};
        use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress, SparkReplyFormat};
        use cuteafd_loader::placement::{SparkEndpointOwners, SparkIntakeCandidates};
        use cuteafd_transport::TcpTransportConfig;
        let snapshot = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
        let checkpoint = Checkpoint::open(snapshot)?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let weights = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &vec![0; cfg.layers])?.remove(0);
        let path = std::env::var("CUTEAFD_PROGRAM_MANIFEST").context("installed MiniMax CPU test needs the sealed manifest")?;
        let manifest = serde_json::from_slice(&std::fs::read(path)?)?;
        let rows = 1024;
        let tokens = 16383;
        let waves = 3;
        let endpoint = SparkEndpoint { ring_depth: None, capacity: rows,
            ingress: SparkIngress::Bf16, partial_reply_bytes: None };
        let link = PreparedSparkLink::endpoint("spark", 0,
            SparkEndpointOwners { ranks: 4, geometry: ExpertGeometry::MINIMAX_M3,
                endpoint, rings: spark_rings(TcpTransportConfig::default().max_frame_bytes)? },
            waves, true, true, SparkIntakeCandidates::Probed, SparkReplyFormat::Bf16, Lifetime::TARGET)?;
        let contract = contract(&checkpoint, &cfg, &weights, &manifest, rows, tokens, waves, &link, "sm_120")?;
        assert_cold_finalization_move(&contract)?;
        let decision = contract.decisions(&vec![0; contract.variables.len()])?;
        assert_eq!(decision.prefill, PrefillShape { rows: u64::from(rows), waves: waves as u64 });
        assert_eq!(decision.lanes, 1);
        struct Prepared(Contract);
        impl Recipe for Prepared {
            fn prepare(&self, _: &Request) -> std::result::Result<Contract, Refusal> { Ok(self.0.clone()) }
        }
        let request = Request { context: cuteafd_loader::placement::planner::Context::pinned(u64::from(tokens)),
            pool: Pin::Exact(u64::from(tokens.div_ceil(64)) * 64), onboard: Pin::Exact(0),
            prefill: Pin::Exact(decision.prefill), operation: Operation::Scoring,
            resident_weight_limit: None, spark_reply_formats: Default::default() };
        let budgets = vec![Budget { id: "rtx0".into(), locations: vec![Location::coordinator(0)],
            physical_bytes: 96 << 30, logical_bytes: 96 << 30,
            baseline: Baseline::Sampled { free: 94 << 30, phase: Phase::Load, source: "CPU fixture".into() },
            headroom: cuteafd_loader::placement::reserve::gpu("minimax_m3") },
            cuteafd_loader::placement::families::qwen4::host::HostMemory {
                total: 192 << 30, available: 160 << 30 }.budget("MiniMax CPU fixture")?];
        let resolved = solve(request, budgets, &Prepared(contract.clone()))?;
        assert_eq!(resolved.decisions.prefill, decision.prefill);
        let admitted = admit(&contract, 94 << 30, 96 << 30, (192 << 30, 160 << 30))?;
        assert_eq!(admitted.allocations, resolved.facts.allocations);
        assert_eq!(admitted.resident_weights.len(), 1);
        assert!(admitted.allocations.iter().any(|owner| owner.id == weights.arena.id()));
        for id in ["residual", "sync-h2d", "head.download", "head.host-logits", "unit.0.residual"] {
            assert!(admitted.allocations.iter().any(|owner| owner.id == id), "missing admitted {id}");
        }
        let view = cuteafd_loader::plan::checkpoint::CheckpointView::select(snapshot,
            cuteafd_loader::plan::checkpoint::CoreView::Original)?;
        let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
        let exports = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?.exports();
        let executions = cuteafd_loader::plan::spark_executions::admit_family("minimax_m3", &exports,
            None, u64::from(rows), waves as u64)?;
        let model = cuteafd_transport::worker_selection::ModelAdmission::for_clients(
            ExpertGeometry::MINIMAX_M3, endpoint, spark_rings(TcpTransportConfig::default().max_frame_bytes)?,
            1, executions)?;
        crate::shared::placement::SparkAdmission::plan_view(&view, ExpertGeometry::MINIMAX_M3,
            Some(&catalog), &cuteafd_loader::placement::SparkWorkingSet {
                rtx_layers: vec![], spark_layers: (3..cfg.layers).collect(), spark_ranks: 4, prefill_lanes: 1 },
            4, model, None)?;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 96usize << 30)?;
        // Reach the same allocation boundary as the GPU path without reading
        // checkpoint payloads or allocating the full target arena in host RAM.
        let metadata = admitted.allocations.iter().find(|owner| owner.id == "metadata.0.positions")
            .context("admitted metadata owner missing")?;
        let allocated = DeviceAllocation::new(Device { library: &library, id: 0 },
            usize::try_from(metadata.bytes)?, cuteafd_core::memory_layout::Tag::workspace("minimax/metadata"))?;
        assert_eq!(allocated.buffer.bytes as u64, metadata.bytes);
        assert!(!allocated.buffer.ptr.is_null());
        Ok(())
    }

    fn assert_cold_finalization_move(after: &Contract) -> Result<()> {
        let maximum = after.maximum_pages;
        let mut before = Vec::new();
        for component in &after.components {
            match &component.kind {
                ComponentKind::AllocationGranularity(tail) => {
                    // Weight arenas retain their existing explicit tail.
                    if tail.backings.iter().any(|backing| after.components.iter().any(|component|
                        matches!(&component.kind, ComponentKind::ResidentWeight(owner)
                            if owner.id == backing.backing))) {
                        before.push(component.clone());
                    }
                }
                ComponentKind::DriverPageTables { model: DriverPageTableModel::Sm120Portable, .. } => {},
                _ => {
                    let mut helper = vec![component.clone()];
                    if let ComponentKind::Allocation { template, .. } = &component.kind {
                        // Before extraction only descriptor helpers and raw
                        // buffers finalized. Links/stream slabs did not.
                        if !template.id.starts_with("spark.") && template.id != "driver.stream-slabs"
                            && !after.components.iter().any(|component| matches!(&component.kind,
                                ComponentKind::ResidentWeight(owner) if owner.id == template.id)) {
                            fixed_request_tails(&mut helper, maximum, &[])?;
                        }
                    }
                    before.extend(helper);
                }
            }
        }
        let mut legacy = after.clone();
        legacy.components = before;
        legacy.add_driver_page_tables();
        let added = after.components.iter().filter_map(|component| match &component.kind {
            ComponentKind::AllocationGranularity(tail) if tail.id.starts_with("spark.")
                || tail.id == "driver.stream-slabs.tail" => Some((tail.id.clone(), tail.bytes().unwrap())),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(added, (0..4).map(|rank| (format!("spark.intake.rank.{rank}.storage.tail"),
            (2 << 20) - 4096)).collect::<Vec<_>>());
        let serving_links = cuteafd_loader::placement::families::minimax_m3::spark_link(0, 4,
            PrefillShape { rows: 1024, waves: 3 }, Default::default())?;
        let mut serving = serving_links[0].components(maximum)?;
        fixed_request_tails(&mut serving, maximum, &[])?;
        for rank in 0..4 {
            let suffix = format!(".intake.rank.{rank}.storage");
            let storage = |components: &[Component]| components.iter().find_map(|component|
                match &component.kind {
                    ComponentKind::Allocation { template, .. } if template.id.ends_with(&suffix) =>
                        Some((template.bytes, template.location, template.lifetime)),
                    _ => None,
                }).expect("intake storage");
            assert_eq!(storage(&after.components), storage(&serving));
            assert_eq!(storage(&serving).0, 37_752_832);
            let tail_id = format!("minimax.spark{suffix}.tail");
            let tail = serving.iter().find_map(|component| match &component.kind {
                ComponentKind::AllocationGranularity(tail) if tail.id == tail_id => Some(tail.bytes().unwrap()),
                _ => None,
            }).expect("serving intake tail");
            assert_eq!(tail, added[rank].1);
        }
        // Removing exactly the four newly charged intake tails recovers every
        // legacy structural field, not just the total number of bytes.
        let mut corrected = after.components.iter().filter(|component| !matches!(&component.kind,
            ComponentKind::AllocationGranularity(tail) if added.iter().any(|(id, _)| id == &tail.id)))
            .map(|component| format!("{component:?}")).collect::<Vec<_>>();
        let mut original = legacy.components.iter().map(|component| format!("{component:?}")).collect::<Vec<_>>();
        corrected.sort(); original.sort();
        assert_eq!(corrected, original, "cold finalization changes only the listed transport tails");
        Ok(())
    }

    #[test]
    fn sync_upload_owner_takes_maximum_not_sum_or_mount_floor() -> Result<()> {
        let dir = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_snapshot(dir.path(), &serde_json::json!({"model_type":"fixture"}),
            &[("weight".into(), "BF16", vec![64, 128])], None);
        let checkpoint = Checkpoint::open(dir.path())?;
        let weights = CoordinatorWeights { rank: 0, layers: vec![], projections: vec![],
            arena: WeightArenaLayout::new("minimax/target/0", [("weight".into(), 64 * 128 * 2)])? };
        assert_eq!(sync_h2d_bytes(&checkpoint, &weights, 256, 2)?, 16384);
        assert_eq!(sync_h2d_bytes(&checkpoint, &weights, 4096, 2)?, 32768);
        assert_eq!(sync_h2d_bytes(&checkpoint, &weights, 4096, 1000)?, 256000);
        assert!(weights.mount_load(&checkpoint)?.pinned_source > 16384);
        let source = include_str!("admission.rs").split("#[cfg(test)]").next().unwrap();
        assert_eq!(source.matches("Allocation::buffer(\"sync-h2d\"").count(), 1);
        Ok(())
    }

    #[test]
    fn sync_upload_reconciliation_observes_backing_and_rejects_unadmitted_growth() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let devices = [0];
        let mut facts = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![],
            allocations: vec![Allocation::buffer("sync-h2d", Location::PINNED, 16, Lifetime::RESIDENT,
                Category::Staging, "library-wide synchronous H2D backing")], deferred: vec![],
            sequential: vec![], reservations: vec![], allowances: vec![], driver_page_tables: vec![],
            allocation_granularity: vec![] };
        let inventory = || Inventory::new(CoordinatorDevices::new(&devices).unwrap(), "minimax",
            Lifetime::TARGET, Category::Workspace, "fixture");
        assert!(reconcile_sync_h2d(&library, &facts, &mut inventory()).is_err());
        let destination = crate::shared::memory::device::Allocation::new(
            crate::shared::memory::device::Device { library: &library, id: 0 }, 32,
            cuteafd_core::memory_layout::Tag::workspace("fixture"))?;
        library.copy_h2d(destination.buffer, &[0; 16])?;
        let mut startup = inventory();
        reconcile_sync_h2d(&library, &facts, &mut startup)?;
        assert_eq!(startup.receipts.len(), 1);
        assert_eq!(startup.receipts[0].allocation.id, "minimax.sync-h2d");
        assert_eq!(startup.receipts[0].allocation.bytes, 16);
        let mut completion = inventory();
        reconcile_sync_h2d(&library, &facts, &mut completion)?;
        assert_eq!(startup.receipts[0].address, completion.receipts[0].address);
        library.copy_h2d(destination.buffer, &[0; 32])?;
        assert!(reconcile_sync_h2d(&library, &facts, &mut inventory()).is_err());
        facts.allocations[0].bytes = 32;
        reconcile_sync_h2d(&library, &facts, &mut inventory())?;
        assert_eq!(library.release_sync_h2d_staging()?, 32);
        Ok(())
    }
}
