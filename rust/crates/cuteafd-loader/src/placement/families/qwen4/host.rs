//! Qwen's coordinator host owners, stated once for the offline plan and the
//! live engine. The recipe charges every owner against one host budget
//! sampled before loading, so the shared `charge_host` pass is not needed.
use crate::families::qwen4::{mtp_upload_shape, upload_shape, Qwen4Config, VERIFY_ROWS};
use crate::placement::planner::{
    Activation, Allocation, Allowance, Baseline as PlanBaseline, Budget, ByteCurve, Component,
    ComponentKind, Lifetime, Location, Phase, Refusal,
};
use crate::push_load::PushHostEnvelope;
use crate::serving_capacity::qwen_graphs::{QWEN_UNIT_PAGES, QWEN_UNIT_ROWS};
use cuteafd_core::memory_layout::{host_logits_bytes, Category};

/// The host and pinned domains share one physical RAM budget.
pub const HOST_BUDGET: &str = "host";

/// One coordinator host sample, taken before the process allocates host
/// owners (`host_memory_before_process` live, the injected or sampled
/// inventory offline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    pub total: u64,
    pub available: u64,
}

impl HostMemory {
    /// The same ten-percent (at least 8 GiB) reserve `charge_host` kept.
    pub fn reserve(self) -> u64 { (8u64 << 30).max(self.total / 10) }

    pub fn budget(self, source: &str) -> Result<Budget, Refusal> {
        if self.total == 0 || self.available > self.total {
            return Err(Refusal::InvalidFact(format!("{source}: invalid host memory sample")));
        }
        Ok(Budget { id: HOST_BUDGET.into(), locations: vec![Location::HOST, Location::PINNED],
            physical_bytes: self.total, logical_bytes: self.total,
            baseline: PlanBaseline::Sampled { free: self.available, phase: Phase::Load,
                source: format!("{source}: host/cgroup available before process host owners") },
            headroom: self.reserve() })
    }
}

/// The mapped PLE table's host staging and bounded row cache, at the shapes
/// `MappedTableDevice::new` allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleHost {
    pub staging: u64,
    pub worker: u64,
    pub row_cache: u64,
}

impl PleHost {
    /// `unit_rows` and `waves` are the executor's prefill unit and depth; the
    /// decode stage stages `VERIFY_ROWS` tokens.
    pub fn new(cfg: &Qwen4Config, unit_rows: u64, waves: u64, row_bytes: u64) -> Result<Self, Refusal> {
        let invalid = |what: &str| Refusal::InvalidFact(format!("Qwen PLE host: {what}"));
        let rows = usize::try_from(unit_rows.max(64)).map_err(|_| invalid("unit rows"))?;
        let waves = usize::try_from(waves.max(1)).map_err(|_| invalid("waves"))?;
        let row_bytes = usize::try_from(row_bytes).map_err(|_| invalid("row bytes"))?;
        let per_token = cfg.ple_rows();
        let prefill = rows.checked_mul(per_token).ok_or(Refusal::Overflow("Qwen PLE prefill rows"))?;
        let decode = VERIFY_ROWS.checked_mul(per_token).ok_or(Refusal::Overflow("Qwen PLE decode rows"))?;
        let staging = crate::formats::mapped_table::MappedStaging::new(prefill, waves, decode, row_bytes)
            .map_err(|error| invalid(&error.to_string()))?;
        let shares = crate::formats::mapped_table::token_table::TokenTable::shares(rows, waves,
            VERIFY_ROWS, per_token);
        let row_cache = crate::formats::mapped_table::token_table::TokenTable::cache_bytes(row_bytes, shares);
        Ok(Self { staging: (staging.total_bytes - staging.workspace_bytes) as u64,
            worker: staging.workspace_bytes as u64, row_cache: row_cache as u64 })
    }
}

/// A retained ordered-upload arena as a conservative page-linear bound: the
/// page and pool tables grow with pages; per-plane 16-byte alignment and the
/// one-plane wrap slack are a fixed term. Estimated, not exact (the actual
/// arena is piecewise in pages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadBound {
    pub at_first: u64,
    pub per_page: u64,
}

impl UploadBound {
    /// `shape(pages, pools)` is the executor's own upload geometry.
    pub fn of(shape: impl Fn(usize, usize) -> anyhow::Result<crate::shared::token_table::UploadShape>)
        -> Result<Self, Refusal> {
        let at = |units: usize| -> Result<u64, Refusal> {
            let pages = units * QWEN_UNIT_PAGES;
            let shape = shape(pages, units).map_err(|error|
                Refusal::InvalidFact(format!("Qwen upload geometry: {error:#}")))?;
            Ok(shape.capacity_bytes().map_err(|error|
                Refusal::InvalidFact(format!("Qwen upload capacity: {error:#}")))? as u64)
        };
        // capacity(u) = aligned planes + one largest aligned plane of wrap
        // slack. Raw planes are affine in units and alignment adds at most 15
        // per range, so capacity is a max of affine pieces plus bounded
        // rounding. Its steepest secant from u = 1 along a doubling ladder
        // (the largest-plane switch happens once a page table outgrows the
        // fixed planes) plus one byte per unit, over capacity(1) plus 16 bytes
        // per range, covers every extent.
        let one = at(1)?;
        let mut per_page = 0u64;
        let mut units = 2usize;
        while units <= 1 << 20 {
            let rise = at(units)?.checked_sub(one).ok_or_else(||
                Refusal::InvalidFact("Qwen upload arena shrinks with pages".into()))?;
            per_page = per_page.max(rise.div_ceil(units as u64 - 1));
            units *= 2;
        }
        let per_page = per_page.checked_add(1).ok_or(Refusal::Overflow("Qwen upload slope"))?;
        let ranges = shape(QWEN_UNIT_PAGES, 1).map_err(|error|
            Refusal::InvalidFact(format!("Qwen upload geometry: {error:#}")))?.ranges as u64;
        let at_first = ranges.checked_mul(16).and_then(|n| n.checked_add(one))
            .ok_or(Refusal::Overflow("Qwen upload slack"))?;
        Ok(Self { at_first, per_page })
    }

    pub fn at(self, units: u64) -> Option<u64> {
        units.checked_sub(1)?.checked_mul(self.per_page)?.checked_add(self.at_first)
    }
}

/// Everything the coordinator host holds for one Qwen route.
#[derive(Debug, Clone)]
pub struct QwenHost {
    pub memory: HostMemory,
    pub ple: Option<PleHost>,
    /// One prefill and one decode arena per attention owner.
    pub uploads: Vec<[UploadBound; 2]>,
    /// The separate packed-MTP prefill arena (owner 0), when Spark waves run MTP.
    pub mtp_upload: Option<UploadBound>,
    /// Pinned logits landing, all owners (FP32).
    pub pinned_logits: u64,
    /// Owner 0's separate MTP prefill workspace, when unit prefill runs MTP.
    pub mtp_pinned_logits: u64,
    /// Host FP32 scoring logits (`host_logits_bytes`).
    pub scoring_logits: u64,
    /// Host prefix retention quota and its chunk.
    pub prefix_quota: u64,
    pub prefix_chunk: u64,
    /// A media embedding cache whose tower was placed outside this solve (the
    /// live engine's already loaded tower); offline towers charge theirs as
    /// part of the encoder choice.
    pub loaded_media_cache: u64,
    pub host_embedding: u64,
    pub table_reactor: u64,
    pub preloaded_tables: u64,
    /// Spark rings, egress request slots and host-mode intake (single owner).
    pub spark: [u64; 3],
    /// Push group receive envelopes admitted before the push starts.
    pub push: Vec<(&'static str, PushHostEnvelope)>,
}

/// Shapes shared by every upload owner of one route.
#[derive(Debug, Clone, Copy)]
pub struct UploadInputs {
    pub prefill_rows: usize,
    pub waves: usize,
    pub spark: bool,
    pub mtp: bool,
    /// PLE ids staged per step only when the table is not mapped.
    pub ple_ids: bool,
}

impl UploadInputs {
    /// Prefill and decode arenas of one owner, and the separate MTP prefill.
    pub fn bounds(self, cfg: &Qwen4Config) -> Result<([UploadBound; 2], Option<UploadBound>), Refusal> {
        let units = if self.spark && self.waves > 1 { self.waves } else { 0 };
        let prefill = UploadBound::of(|pages, pools| upload_shape(self.prefill_rows, false, pages, pools,
            cfg.ple_rows(), cfg.layers, units, self.spark, self.mtp, self.ple_ids))?;
        let decode = UploadBound::of(|pages, pools| upload_shape(VERIFY_ROWS, true, pages, pools,
            cfg.ple_rows(), cfg.layers, 0, self.spark, self.mtp, self.ple_ids))?;
        let mtp = (units > 0 && self.mtp).then(|| UploadBound::of(|pages, pools|
            mtp_upload_shape(self.prefill_rows, pages, pools, units))).transpose()?;
        Ok(([prefill, decode], mtp))
    }
}

/// Pinned logits landing on each owner: decode and prefill arenas coexist.
pub fn pinned_logits_bytes(owners: usize, vocab: u64)
    -> Result<u64, Refusal> {
    let rows = 64u64.checked_add(1)
        .and_then(|n| n.checked_mul(owners as u64)).ok_or(Refusal::Overflow("Qwen pinned logits rows"))?;
    host_logits_bytes(rows, rows, vocab).ok_or(Refusal::Overflow("Qwen pinned logits"))
}

/// The separate MTP prefill workspace lands one row.
pub fn mtp_pinned_logits_bytes(prefill_shape: bool, mtp: bool, vocab: u64) -> Result<u64, Refusal> {
    if !prefill_shape || !mtp { return Ok(0); }
    host_logits_bytes(1, 1, vocab).ok_or(Refusal::Overflow("Qwen MTP pinned logits"))
}

/// Host FP32 scoring logits: one bounded decode/head wave.
pub fn scoring_logits_bytes(span_rows: u64, vocab: u64) -> Result<u64, Refusal> {
    let rows = 64;
    host_logits_bytes(rows, span_rows, vocab).ok_or(Refusal::Overflow("Qwen scoring logits"))
}

fn fixed_owner(id: String, location: Location, bytes: u64, lifetime: Lifetime, category: Category,
    group: &str, maximum: u64) -> Component {
    Component::allocation(Allocation::buffer(id, location, bytes, lifetime, category, group),
        ByteCurve::fixed(bytes, maximum), false, maximum)
}

fn estimate(source: String, location: Location, curve: ByteCurve, lifetime: Lifetime) -> Component {
    Component { activation: Activation::always(), bytes: curve,
        kind: ComponentKind::Allowance(Allowance { location, bytes: 0, lifetime, source, assumed: true }) }
}

impl QwenHost {
    pub fn budget(&self) -> Result<Budget, Refusal> { self.memory.budget("qwen4") }

    /// Host components over `maximum_pages` pool units (`QWEN_UNIT_ROWS` rows each).
    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        let max = maximum_pages;
        let post_ready = Lifetime { first: Phase::Ready, last: Phase::Growth };
        let mut out = Vec::new();
        if let Some(ple) = self.ple {
            // One ledger group, as runtime admits it: pinned row slots plus the
            // workers' pageable descriptors and scratch.
            out.push(fixed_owner("qwen.host.ple.staging".into(), Location::PINNED, ple.staging,
                Lifetime::RESIDENT, Category::Staging, "staging/mapped-table", max));
            out.push(fixed_owner("qwen.host.ple.worker".into(), Location::HOST, ple.worker,
                Lifetime::RESIDENT, Category::Staging, "staging/mapped-table", max));
            out.push(fixed_owner("qwen.host.ple.row_cache".into(), Location::HOST, ple.row_cache,
                Lifetime::RESIDENT, Category::Workspace, "mapped-table/row-cache", max));
        }
        for (owner, bounds) in self.uploads.iter().enumerate() {
            for (kind, bound) in ["prefill", "decode"].into_iter().zip(bounds) {
                out.push(estimate(format!("estimated: qwen4 owner{owner} {kind} retained uploads (page-linear bound)"),
                    Location::PINNED, ByteCurve::linear(bound.at_first, bound.per_page, max), Lifetime::TARGET));
            }
        }
        if let Some(bound) = self.mtp_upload {
            out.push(estimate("estimated: qwen4 owner0 MTP prefill retained uploads (page-linear bound)".into(),
                Location::PINNED, ByteCurve::linear(bound.at_first, bound.per_page, max), Lifetime::TARGET));
        }
        out.push(fixed_owner("qwen.host.logits.pinned".into(), Location::PINNED, self.pinned_logits,
            Lifetime::TARGET, Category::Workspace, "workspace/logits-host", max));
        out.push(fixed_owner("qwen4/owner-0/mtp-prefill/pinned-logits".into(), Location::PINNED,
            self.mtp_pinned_logits, Lifetime::TARGET, Category::Workspace, "workspace/logits-host", max));
        out.push(fixed_owner("qwen.host.logits.scoring".into(), Location::HOST, self.scoring_logits,
            post_ready, Category::Workspace, "scoring/full-logits", max));
        if self.prefix_quota > 0 {
            out.extend(crate::placement::planner::serving::HostPrefixOwners {
                quota_bytes: self.prefix_quota, chunk_bytes: self.prefix_chunk.min(self.prefix_quota) }
                .allocations("qwen.host.prefix", Lifetime { first: Phase::Warmup, last: Phase::Growth })?
                .into_iter().map(|owner| { let bytes = owner.bytes;
                    Component::allocation(owner, ByteCurve::fixed(bytes, max), false, max) }));
        }
        out.push(fixed_owner("qwen.host.media".into(), Location::HOST, self.loaded_media_cache,
            Lifetime::TARGET, Category::Embedding, "media/embedding-cache", max));
        out.push(fixed_owner("qwen.host.embedding".into(), Location::PINNED, self.host_embedding,
            Lifetime::RESIDENT, Category::Embedding, "embedding", max));
        out.push(fixed_owner("qwen.host.table_reactor".into(), Location::HOST, self.table_reactor,
            Lifetime::RESIDENT, Category::Workspace, "workspace/table-read-reactor", max));
        out.push(fixed_owner("qwen.host.tables.preloaded".into(), Location::HOST, self.preloaded_tables,
            Lifetime::RESIDENT, Category::Tables, "mapped-tables/preloaded", max));
        for (kind, bytes, group) in [("rings", self.spark[0], "transport/rdma-rings"),
            ("egress", self.spark[1], "transport/egress"), ("intake", self.spark[2], "spark-intake")] {
            out.push(fixed_owner(format!("qwen.host.spark.{kind}"), Location::PINNED, bytes,
                Lifetime::RESIDENT, Category::Transport, group, max));
        }
        for (prefix, envelope) in &self.push {
            out.extend(envelope.allocations(prefix)?.into_iter().map(|owner| { let bytes = owner.bytes;
                Component::allocation(owner, ByteCurve::fixed(bytes, max), false, max) }));
        }
        out.retain(|component| component.bytes.0.iter().any(|piece|
            piece.bytes_at_first > 0 || piece.bytes_per_page > 0));
        Ok(out)
    }
}

/// The host ledger when the device solve refused: the same owners, uploads at
/// the minimum page extent, so a refusal still names its host demand (and the
/// media cache of a requested tower).
pub fn unsolved_ledger(host: &QwenHost, media_cache: u64, pages: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout, Refusal> {
    use cuteafd_core::memory_layout::{Basis, HostLayout, Item, ItemLifetime};
    use std::collections::BTreeMap;
    let pages = pages.max(1);
    let mut groups = BTreeMap::<(String, Category, bool), u64>::new();
    let mut uploads = 0u64;
    for component in host.components(pages)? {
        let bytes = component.bytes.at(pages)?;
        match &component.kind {
            ComponentKind::Allocation { template, .. } if template.category != Category::Prefix => {
                let deferred = template.lifetime.first == Phase::Ready;
                let total = groups.entry((template.group.clone(), template.category, deferred)).or_insert(0);
                *total = total.checked_add(bytes).ok_or(Refusal::Overflow("Qwen host ledger"))?;
            }
            ComponentKind::Allowance(_) => uploads = uploads.checked_add(bytes)
                .ok_or(Refusal::Overflow("Qwen host uploads"))?,
            _ => {}
        }
    }
    let mut items: Vec<_> = groups.into_iter().map(|((group, category, deferred), bytes)| {
        let item = Item::new(category, group, "", bytes, Basis::Formula);
        if deferred { item.with_lifetime(ItemLifetime::PostReadyReserve) } else { item }
    }).collect();
    if uploads > 0 {
        items.push(Item::new(Category::Staging, "staging/tables-and-sync", "retained uploads (page-linear bound)",
            uploads, Basis::Estimated));
    }
    if media_cache > 0 {
        items.push(Item::new(Category::Embedding, "media/embedding-cache", "", media_cache, Basis::Formula));
    }
    HostLayout::new(host.memory.total, host.memory.available, host.memory.reserve(), host.prefix_quota, items)
        .ok_or(Refusal::Overflow("Qwen host ledger"))
}

/// Header-only host prefix geometry: one owner's pool unit and retained mark,
/// the aggregate a dual executor's concatenated snapshot also stores.
pub fn prefix_geometry(cfg: &Qwen4Config, layers: usize, mtp: bool,
    kv: crate::families::qwen4::Qwen4KvCache) -> Result<cuteafd_core::prefix::HostCacheGeometry, Refusal> {
    let cache = crate::serving_capacity::qwen_cache_geometry(cfg, layers, mtp, kv)
        .map_err(|error| Refusal::InvalidFact(format!("Qwen host prefix geometry: {error:#}")))?;
    let rank = cache.ranks.first().ok_or_else(|| Refusal::InvalidFact("Qwen host cache has no owner".into()))?;
    Ok(cuteafd_core::prefix::HostCacheGeometry { unit_rows: cache.logical_unit_rows,
        page_bytes: rank.persistent_unit_bytes, mark_bytes: rank.retained_mark_bytes, draft_bytes: 0 })
}

/// The automatic or explicit host prefix quota and its chunk.
pub fn prefix_quota(geometry: cuteafd_core::prefix::HostCacheGeometry, explicit: Option<u64>,
    context: u64, concurrency: u64, host_total: u64) -> Result<(u64, u64), Refusal> {
    let chunk = geometry.chunk_bytes();
    let bytes = geometry.resolve(explicit, context, concurrency, host_total, chunk)
        .ok_or_else(|| Refusal::InvalidFact(format!(
            "invalid Qwen host prefix geometry {geometry:?}, context {context}, concurrency {concurrency}")))?;
    if bytes == 0 {
        return Err(Refusal::InvalidFact(
            "host prefix quota resolves to zero; use a positive quota or explicitly disable host retention".into()));
    }
    Ok((bytes, chunk.min(bytes)))
}

/// The admitted host owners as the report's host ledger: Ready-live owners
/// by group, post-ready scoring as deferred, the prefix quota as the budget.
pub fn ledger(host: &QwenHost, solved: &crate::placement::planner::ResolvedPlan)
    -> Result<cuteafd_core::memory_layout::HostLayout, Refusal> {
    use cuteafd_core::memory_layout::{Basis, HostLayout, Item, ItemLifetime};
    use std::collections::BTreeMap;
    let mut groups = BTreeMap::<(String, Category, bool), u64>::new();
    let ready = |lifetime: Lifetime| lifetime.contains(Phase::Ready, solved.facts.phase_order);
    for allocation in solved.facts.allocations.iter().filter(|a| a.view.is_none()
        && matches!(a.location, Location::HOST | Location::PINNED) && ready(a.lifetime)
        && a.category != Category::Prefix) {
        let deferred = allocation.lifetime.first == Phase::Ready;
        let total = groups.entry((allocation.group.clone(), allocation.category, deferred)).or_insert(0);
        *total = total.checked_add(allocation.bytes).ok_or(Refusal::Overflow("Qwen host ledger"))?;
    }
    let mut items: Vec<_> = groups.into_iter().map(|((group, category, deferred), bytes)| {
        let item = Item::new(category, group, "", bytes, Basis::Formula);
        if deferred { item.with_lifetime(ItemLifetime::PostReadyReserve) } else { item }
    }).collect();
    let uploads = solved.facts.allowances.iter().filter(|a| matches!(a.location, Location::HOST | Location::PINNED)
        && ready(a.lifetime)).try_fold(0u64, |n, a| n.checked_add(a.bytes))
        .ok_or(Refusal::Overflow("Qwen host uploads"))?;
    if uploads > 0 {
        items.push(Item::new(Category::Staging, "staging/tables-and-sync", "retained uploads (page-linear bound)",
            uploads, Basis::Estimated));
    }
    HostLayout::new(host.memory.total, host.memory.available, host.memory.reserve(), host.prefix_quota, items)
        .ok_or(Refusal::Overflow("Qwen host ledger"))
}

impl QwenHost {
    /// A host with ample memory and only `owners`' retained uploads, for
    /// device-only admission fixtures (tests and diagnostics).
    pub fn ample(cfg: &Qwen4Config, owners: usize, prefill_rows: usize) -> Result<Self, Refusal> {
        let (bounds, _) = UploadInputs { prefill_rows: prefill_rows.max(64), waves: 1, spark: false,
            mtp: false, ple_ids: cfg.ple_layers.is_empty() }.bounds(cfg)?;
        Ok(Self { memory: HostMemory { total: 1 << 40, available: 1 << 40 }, ple: None,
            uploads: vec![bounds; owners], mtp_upload: None, pinned_logits: 0, mtp_pinned_logits: 0, scoring_logits: 0,
            prefix_quota: 0, prefix_chunk: 0, loaded_media_cache: 0, host_embedding: 0, table_reactor: 0,
            preloaded_tables: 0, spark: [0; 3], push: Vec::new() })
    }
}

/// Pool units of `pool_tokens` (the host curves' page domain).
pub fn units_of(pool_tokens: u64) -> u64 { pool_tokens.div_ceil(QWEN_UNIT_ROWS as u64) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_prefill_logits_are_one_separate_row_only_when_the_workspace_exists() {
        let vocab = 248_320;
        for owners in [1, 2] {
            {
                for shape in [false, true] {
                    for mtp in [false, true] {
                        let target = pinned_logits_bytes(owners, vocab).unwrap();
                        let extra = mtp_pinned_logits_bytes(shape, mtp, vocab).unwrap();
                        assert_eq!(target, owners as u64 * 65 * vocab * 4);
                        assert_eq!(extra, if shape && mtp { vocab * 4 } else { 0 });
                        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
                        let mut host = QwenHost::ample(&cfg, owners, 8192).unwrap();
                        host.pinned_logits = target;
                        host.mtp_pinned_logits = extra;
                        let components = host.components(1).unwrap();
                        let rows: Vec<_> = components.iter().filter(|c| matches!(&c.kind,
                            ComponentKind::Allocation { template, .. }
                                if template.id == "qwen4/owner-0/mtp-prefill/pinned-logits")).collect();
                        assert_eq!(rows.len(), usize::from(shape && mtp));
                        assert_eq!(rows.iter().map(|c| c.bytes.at(1).unwrap()).sum::<u64>(), extra);
                    }
                }
            }
        }
        assert_eq!(pinned_logits_bytes(1, vocab).unwrap(), 64_563_200);
        assert_eq!(mtp_pinned_logits_bytes(true, true, vocab).unwrap(), 993_280);
    }

    #[test]
    fn upload_bound_covers_every_pool_extent_it_admits() {
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        cfg.ple_layers = vec![1];
        for spark in [false, true] {
            for mtp in [false, true] {
                for ple_ids in [false, true] {
                    let inputs = UploadInputs { prefill_rows: 4096, waves: if spark { 3 } else { 1 }, spark, mtp, ple_ids };
                    let ([prefill, decode], extra) = inputs.bounds(&cfg).unwrap();
                    let units = if spark && inputs.waves > 1 { inputs.waves } else { 0 };
                    for pools in (1..=200).chain([255, 256, 257, 511, 513, 1023, 1025, 2047, 2049, 4095, 4097, 4100,
                        8191, 8193, 16383, 16385, 65537, 262145]) {
                        let pages = pools * QWEN_UNIT_PAGES;
                        let exact = |shape: crate::shared::token_table::UploadShape| shape.capacity_bytes().unwrap() as u64;
                        let p = exact(upload_shape(4096, false, pages, pools, cfg.ple_rows(), cfg.layers, units, spark, mtp, ple_ids).unwrap());
                        let d = exact(upload_shape(VERIFY_ROWS, true, pages, pools, cfg.ple_rows(), cfg.layers, 0, spark, mtp, ple_ids).unwrap());
                        assert!(prefill.at(pools as u64).unwrap() >= p, "prefill {pools}: {} < {p}", prefill.at(pools as u64).unwrap());
                        assert!(decode.at(pools as u64).unwrap() >= d, "decode {pools}: {} < {d}", decode.at(pools as u64).unwrap());
                        // Conservative but not loose: within 1% plus the fixed term.
                        assert!(decode.at(pools as u64).unwrap() - d <= decode.at_first + d / 100,
                            "decode {pools}: bound {} exact {d}", decode.at(pools as u64).unwrap());
                        assert!(prefill.at(pools as u64).unwrap() - p <= prefill.at_first + p / 100,
                            "prefill {pools}: bound {} exact {p}", prefill.at(pools as u64).unwrap());
                        if let Some(bound) = extra {
                            let m = exact(mtp_upload_shape(4096, pages, pools, units).unwrap());
                            assert!(bound.at(pools as u64).unwrap() >= m);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn host_budget_refuses_an_impossible_sample_and_keeps_the_charge_host_reserve() {
        assert!(HostMemory { total: 0, available: 0 }.budget("t").is_err());
        assert!(HostMemory { total: 1, available: 2 }.budget("t").is_err());
        let budget = HostMemory { total: 512 << 30, available: 400 << 30 }.budget("t").unwrap();
        assert_eq!(budget.headroom, (512u64 << 30) / 10);
        assert_eq!(budget.available().unwrap(), (400u64 << 30) - (512u64 << 30) / 10);
        assert_eq!(HostMemory { total: 32 << 30, available: 1 }.reserve(), 8 << 30);
    }

    #[test]
    fn ple_host_matches_the_mapped_table_allocation() {
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        cfg.ple_layers = vec![1];
        let host = PleHost::new(&cfg, 2752, 3, 160).unwrap();
        let staging = crate::formats::mapped_table::MappedStaging::new(2752 * cfg.ple_rows(), 3,
            64 * cfg.ple_rows(), 160).unwrap();
        assert_eq!(host.staging + host.worker, staging.total_bytes as u64);
        assert_eq!(host.worker, staging.workspace_bytes as u64);
        let shares = crate::formats::mapped_table::token_table::TokenTable::shares(2752, 3, 64, cfg.ple_rows());
        assert_eq!(host.row_cache,
            crate::formats::mapped_table::token_table::TokenTable::cache_bytes(160, shares) as u64);
    }
}
