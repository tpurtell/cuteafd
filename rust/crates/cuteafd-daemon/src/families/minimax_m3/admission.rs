//! Fixed cold-golden owners, admitted before target/cache/exchange allocation.
use anyhow::Result;
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_loader::{families::minimax_m3::{MiniMaxM3Config, execution::*, weight_layout::CoordinatorWeights},
    placement::{planner::*, PreparedSparkLink}, plan::Checkpoint};

pub(crate) fn contract(checkpoint: &Checkpoint, cfg: &MiniMaxM3Config, weights: &CoordinatorWeights,
    manifest: &serde_json::Value, rows: u32, tokens: u32, waves: usize,
    link: &PreparedSparkLink, arch: &str) -> Result<Contract> {
    let pages = u64::from(tokens.div_ceil(64));
    let empty = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT, resident_weights: vec![],
        allocations: vec![], deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
        driver_page_tables: vec![], allocation_granularity: vec![] };
    let mut contract = Contract::fixed(64, pages, 1, empty);
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
        Lifetime::TARGET, Category::Workspace, "cold scoring residual")];
    let mount = weights.mount_load(checkpoint)?;
    for (name, location, bytes) in [("source", Location::PINNED, mount.pinned_source),
        ("rows", Location::HOST, mount.full_row_cache), ("tables", Location::PINNED, mount.pinned_tables)] {
        if bytes > 0 { raw.push(Allocation::buffer(format!("mount.{name}"), location, bytes,
            Lifetime { first: Phase::Load, last: Phase::Load }, Category::Staging, "target mount")); }
    }
    let mut buffers = raw.into_iter().map(|owner| {
        let bytes = owner.bytes;
        Component::allocation(owner, ByteCurve::fixed(bytes, pages), false, pages)
    }).collect();
    fixed_request_tails(&mut buffers, pages, &[])?;
    c.extend(buffers);
    c.extend(link.components(pages)?);
    c.extend(serving::blas_components(arch, &[serving::BlasOwners {
        name: "minimax/head".into(), process: "coordinator".into(), thread: "golden".into(),
        location: Location::coordinator(0), gemm: true, lt: true, lifetime: Lifetime::TARGET,
        activation: Activation::always() }], pages, PhaseOrder::TARGET_THEN_DRAFT)?);
    let stream = Allocation::buffer("driver.stream-slabs", Location::coordinator(0), cuda_stream_slab_bytes(1)?,
        Lifetime::RESIDENT, Category::Runtime, "one ordered golden stream");
    c.push(Component::allocation(stream.clone(), ByteCurve::fixed(stream.bytes, pages), false, pages));
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
    let facts = contract.facts(&[], contract.maximum_pages)?;
    let phases = evaluate(&facts, &budgets)?;
    tracing::info!(record = %serde_json::json!({"facts":facts,"budgets":budgets,"phases":phases}),
        "MiniMax cold golden admission");
    Ok(facts)
}
