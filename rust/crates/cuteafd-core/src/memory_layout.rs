//! Per-device memory layout: what the placement planner predicts each
//! coordinator GPU and Spark rank holds, by category, weight group and
//! resident format. The same categories label the runtime allocation ledger
//! (`cuteafd_ffi::memory_ledger` scopes), so a prediction can be checked
//! against what an engine actually allocated (`compare`).
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const GIB: f64 = (1u64 << 30) as f64;

/// What a block of memory is for. Ledger scopes map onto these by their first
/// path segment (`weights/fp8-pack` -> `Weights`, see [`Category::of_scope`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    /// Coordinator-resident model weights (attention, dense/shared MLP, head, norms, router).
    Weights,
    /// Token embedding table.
    Embedding,
    /// Routed experts resident on this device (Spark ranks, or RTX-local layers).
    Experts,
    /// Drafter weights and its own workspaces (DFlash, dSpark, MTP).
    Drafter,
    /// Paged KV records, index keys, recurrent state, RoPE tables.
    Kv,
    /// Prefix-cache marks and snapshot arenas on the device.
    Prefix,
    /// Step workspaces (decode, prefill lanes, verify), sampler, intake planes.
    Workspace,
    /// Peer-split exchange slots, RDMA rings and other transport buffers.
    Transport,
    /// Pinned upload staging retained after loading.
    Staging,
    /// Host-mapped tables (engram, PLE) staged on the device.
    Tables,
    /// CUDA context, loaded modules, cuBLAS handles, graph executables:
    /// what the device reports in use beyond tracked allocations.
    Runtime,
    /// Memory the host OS and other processes hold (Spark unified memory).
    Reserved,
}

impl Category {
    pub fn label(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::Embedding => "embedding",
            Self::Experts => "experts",
            Self::Drafter => "drafter",
            Self::Kv => "kv",
            Self::Prefix => "prefix",
            Self::Workspace => "workspace",
            Self::Transport => "transport",
            Self::Staging => "staging",
            Self::Tables => "tables",
            Self::Runtime => "runtime",
            Self::Reserved => "reserved",
        }
    }

    /// The planner category of a runtime ledger scope.
    pub fn of_scope(scope: &str) -> Self {
        let root = scope.split('/').next().unwrap_or(scope);
        match root {
            "weights" => Self::Weights,
            "embedding" => Self::Embedding,
            "experts" | "local-experts" => Self::Experts,
            "drafter" => Self::Drafter,
            "kv" => Self::Kv,
            "prefix" => Self::Prefix,
            "workspace" | "sampler" | "spark-intake" | "probe" => Self::Workspace,
            "transport" | "peer-split" => Self::Transport,
            "staging" => Self::Staging,
            "mapped-table" | "ple" => Self::Tables,
            "v41" => Self::of_v41_stage(scope),
            _ => Self::Runtime,
        }
    }

    fn of_v41_stage(scope: &str) -> Self {
        let stage = scope.trim_start_matches("v41/");
        if stage.starts_with("weights") || stage.contains("weights") && !stage.contains("draft") {
            if stage.contains("Engram") { Self::Tables } else { Self::Weights }
        } else if stage.contains("draft") || stage.contains("drafter") {
            Self::Drafter
        } else if stage.starts_with("kv") || stage.contains("KV cache") {
            Self::Kv
        } else if stage.contains("prefix") || stage.contains("snapshot") {
            Self::Prefix
        } else if stage.contains("expert") {
            Self::Experts
        } else if stage.contains("transport") || stage.contains("TP2") {
            Self::Transport
        } else if stage.contains("engram") {
            Self::Tables
        } else if stage.contains("workspace") || stage.contains("lane") || stage.contains("vision") {
            Self::Workspace
        } else {
            Self::Runtime
        }
    }
}

/// How an item's bytes were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    /// From checkpoint headers and the family's resident representation.
    Exact,
    /// From a geometry formula (KV units, workspace row costs).
    Formula,
    /// A per-family constant calibrated against the allocation ledger.
    Calibrated,
    /// An unqualified allowance; must be measured before driving admission.
    Estimated,
}

/// Whether admission reserves bytes already resident when serving becomes ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemLifetime {
    ReadyResident,
    /// Lazy allocations or safety headroom; charged to admission, not ready residency.
    PostReadyReserve,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub category: Category,
    /// Weight group (`attention`, `shared_expert`, ...) or a short name.
    pub group: String,
    /// Resident format (`bf16`, `fp8-block128`, `mxfp4`, ...); empty when not a weight.
    pub format: String,
    pub bytes: u64,
    pub basis: Basis,
    pub lifetime: ItemLifetime,
}

/// A worker-process census, separate from future arena and execution owners.
#[derive(Debug, Clone, Copy)]
pub struct SparkWorkerProcessResident {
    pub arch: &'static str,
    pub anonymous_pss_bytes: u64,
    pub file_pss_bytes: u64,
    pub context_phase_bytes: u64,
    pub context_file_pss_growth_bytes: u64,
    pub driver: &'static str,
    pub source: &'static str,
}

impl SparkWorkerProcessResident {
    pub const fn bytes(self) -> u64 {
        self.anonymous_pss_bytes + self.file_pss_bytes + self.context_phase_bytes
            - self.context_file_pss_growth_bytes
    }
}

/// Spark plans target GB10/SM121, independently of the coordinator/host arch.
/// No x86 worker-process census has been established by this measurement.
pub const SPARK_WORKER_PROCESS_RESIDENT: &[SparkWorkerProcessResident] = &[
    SparkWorkerProcessResident { arch: "sm_121", anonymous_pss_bytes: 1_199_439_872,
        file_pss_bytes: 221_833_216, context_phase_bytes: 230_191_104,
        context_file_pss_growth_bytes: 54_857_728, driver: "580.178.04",
        source: "moa actual Kimi TP6 worker 2026-11-07, serve_owners_before_endpoint_rings; context MemAvailable delta less overlapping file PSS growth; #1454" },
];
pub const SPARK_WORKER_PROCESS_RESIDENT_BYTES: u64 = SPARK_WORKER_PROCESS_RESIDENT[0].bytes();
pub const SPARK_WORKER_PROCESS_RESIDENT_GROUP: &str = "Spark worker process resident";
pub const SPARK_WORKER_PROCESS_RESIDENT_SOURCE: &str = SPARK_WORKER_PROCESS_RESIDENT[0].source;
pub const SPARK_ARENA_DRIVER_OVERHEAD_GROUP: &str = "driver page tables";
pub const SPARK_ARENA_DRIVER_OVERHEAD_SOURCE: &str = "moa ARM/SM121 driver 580.178.04 actual-worker 40.508/75.059/103.652-GiB arenas, 2026-11-07 #1454: zero-intercept VmallocUsed fit 10.052338 MiB/GiB and SecPageTables 1.949590 MiB/GiB, calibrated to 40+8 B/page; nvidia_pte_t 16 B/page and nv_dma pages[] 8 B/page source-known, remaining vmalloc 16 B/page unattributed; no fitted intercept or maximum envelope";

/// Kernel-field calibration: 40 B/page vmalloc metadata plus 8 B/page secondary
/// tables. The whole-host reclaim delta is noisy and is not a fixed allowance.
pub fn spark_arena_driver_overhead_bytes(resident: u64) -> u64 {
    (u128::from(resident).div_ceil(4096) * 48) as u64
}

pub fn calibrated_kimi_spark(selected: Option<(crate::ExpertGeometry, usize, u32)>) -> bool {
    matches!(selected, Some((geometry, 6, compiled))
        if geometry == crate::ExpertGeometry::KIMI_K3 && compiled <= 1024)
}

/// Coordinator ownership inside SPARK_MEMORY_CAP. SPARK_RESERVE replaces the
/// process reserve, never arena metadata or actual staging/workspace owners.
pub fn spark_runtime_reserve_bytes(resident: u64) -> Option<u64> {
    crate::memory_policy::MemoryPolicy::current().spark_reserve
        .unwrap_or(SPARK_WORKER_PROCESS_RESIDENT_BYTES)
        .checked_add(spark_arena_driver_overhead_bytes(resident))
}

/// Future allocations only: the worker process/context already reduced the
/// live MemAvailable sample. An explicit reserve remains future headroom.
pub fn spark_worker_future_bytes(selected: Option<(crate::ExpertGeometry, usize, u32)>, resident: u64) -> Option<u64> {
    crate::memory_policy::MemoryPolicy::current().spark_reserve.unwrap_or(0)
        .checked_add(spark_arena_driver_overhead_bytes(resident))?
        .checked_add(if calibrated_kimi_spark(selected) { 4096 } else { 0 })
}

pub fn spark_runtime_bytes(selected: Option<(crate::ExpertGeometry, usize, u32)>, resident: u64) -> Option<u64> {
    spark_runtime_reserve_bytes(resident)?.checked_add(if calibrated_kimi_spark(selected) { 4096 } else { 0 })
}

pub fn spark_runtime_reserve_group() -> &'static str {
    if crate::memory_policy::MemoryPolicy::current().spark_reserve.is_some() { "SPARK_RESERVE" }
    else { SPARK_WORKER_PROCESS_RESIDENT_GROUP }
}

pub fn spark_worker_process_resident_item() -> Item {
    Item::new(Category::Runtime, SPARK_WORKER_PROCESS_RESIDENT_GROUP, "",
        SPARK_WORKER_PROCESS_RESIDENT_BYTES, Basis::Calibrated).with_lifetime(ItemLifetime::ReadyResident)
}

/// Rebuild shared reserves from payload bytes, never from a prior reserve.
/// Exact I/O, workspace and upload owners are intentionally not removed.
pub fn charge_spark_runtime(device: &mut DeviceLayout) -> Option<()> {
    if device.kind != DeviceKind::Spark { return Some(()); }
    let resident = device.items.iter().filter(|item| item.category == Category::Experts)
        .try_fold(0u64, |sum, item| sum.checked_add(item.bytes))?;
    device.items.retain(|item| !matches!(item.group.as_str(), "Spark runtime residual" | "SPARK_RESERVE")
        && item.group != SPARK_WORKER_PROCESS_RESIDENT_GROUP && item.group != SPARK_ARENA_DRIVER_OVERHEAD_GROUP);
    if resident == 0 { return Some(()); }
    if let Some(bytes) = crate::memory_policy::MemoryPolicy::current().spark_reserve {
        device.items.push(Item::new(Category::Runtime, "SPARK_RESERVE", "", bytes, Basis::Formula));
    } else {
        device.items.push(spark_worker_process_resident_item());
    }
    device.items.push(Item::new(Category::Workspace, SPARK_ARENA_DRIVER_OVERHEAD_GROUP, "",
        spark_arena_driver_overhead_bytes(resident), Basis::Calibrated));
    Some(())
}

impl Item {
    pub fn new(category: Category, group: impl Into<String>, format: impl Into<String>, bytes: u64, basis: Basis) -> Self {
        Self { category, group: group.into(), format: format.into(), bytes, basis,
            lifetime: ItemLifetime::ReadyResident }
    }

    pub fn with_lifetime(self, lifetime: ItemLifetime) -> Self { Self { lifetime, ..self } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Rtx,
    Spark,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceLayout {
    pub kind: DeviceKind,
    pub index: u32,
    pub capacity_bytes: u64,
    /// CUDA allocation ceiling for a Spark worker, before host-inclusive ledger
    /// capacity adjustments. Zero on RTX devices, which do not launch workers.
    pub worker_budget_bytes: u64,
    pub items: Vec<Item>,
    /// KV tokens this device holds records for (its share of the pool).
    pub kv_tokens: u64,
}

impl DeviceLayout {
    pub fn name(&self) -> String {
        match self.kind {
            DeviceKind::Rtx => format!("rtx{}", self.index),
            DeviceKind::Spark => format!("spark{}", self.index),
        }
    }

    pub fn used_bytes(&self) -> u64 {
        self.items.iter().map(|i| i.bytes).sum()
    }

    pub fn free_bytes(&self) -> i64 {
        self.capacity_bytes as i64 - self.used_bytes() as i64
    }

    pub fn by_category(&self) -> BTreeMap<Category, u64> {
        let mut out = BTreeMap::new();
        for item in &self.items {
            *out.entry(item.category).or_default() += item.bytes;
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryLayout {
    pub devices: Vec<DeviceLayout>,
    /// Logical KV pool tokens shared by every request.
    pub pool_tokens: u64,
    /// Waste and inefficiencies the planner can name (padding, duplicates,
    /// replicated operands), with their bytes, for the inventory.
    pub waste: Vec<Waste>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Waste {
    pub device: String,
    pub what: String,
    pub bytes: u64,
}

/// Coordinator host admission. Resident items are already reflected in the
/// OS/cgroup's available bytes; only deferred items are charged against it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostLayout {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub reserve_bytes: u64,
    pub host_budget_bytes: u64,
    pub resident_bytes: u64,
    pub deferred_bytes: u64,
    pub headroom_bytes: u64,
    pub shortfall_bytes: u64,
    pub items: Vec<Item>,
}

impl HostLayout {
    pub fn new(total: u64, available: u64, reserve: u64, cache: u64, items: Vec<Item>) -> Option<Self> {
        let sum = |lifetime| items.iter().filter(|i| i.lifetime == lifetime)
            .try_fold(0u64, |n, i| n.checked_add(i.bytes));
        let resident = sum(ItemLifetime::ReadyResident)?;
        let deferred = sum(ItemLifetime::PostReadyReserve)?;
        let required = cache.checked_add(deferred)?.checked_add(reserve)?;
        let all = required.checked_add(resident)?;
        Some(Self { total_bytes: total, available_bytes: available, reserve_bytes: reserve,
            host_budget_bytes: cache, resident_bytes: resident, deferred_bytes: deferred,
            headroom_bytes: available.saturating_sub(required).min(total.saturating_sub(all)),
            shortfall_bytes: required.saturating_sub(available).max(all.saturating_sub(total)), items })
    }
}

/// A named, not-yet-allocated pinned table. The family owns the byte formula;
/// offline and live host admission share its category and future lifetime.
pub fn host_table_item(name: impl Into<String>, bytes: u64) -> Item {
    Item::new(Category::Staging, name, "pinned", bytes, Basis::Formula)
        .with_lifetime(ItemLifetime::PostReadyReserve)
}

/// Host gather plus one lane download can coexist. Single-lane scoring returns
/// the download directly. Every size is FP32, regardless of checkpoint format.
pub fn host_logits_bytes(rows: u64, lane_rows: u64, vocab: u64) -> Option<u64> {
    let live_rows = if rows > lane_rows { rows.checked_add(lane_rows)? } else { rows };
    live_rows.checked_mul(vocab)?.checked_mul(4)
}

impl MemoryLayout {
    pub fn render(&self) -> String {
        let mut out = String::new();
        let categories: Vec<Category> = {
            let mut c: Vec<Category> = self.devices.iter().flat_map(|d| d.items.iter().map(|i| i.category)).collect();
            c.sort();
            c.dedup();
            c
        };
        out.push_str(&format!("{:<12}", "GiB"));
        for device in &self.devices {
            out.push_str(&format!("{:>9}", device.name()));
        }
        out.push('\n');
        for category in &categories {
            out.push_str(&format!("{:<12}", category.label()));
            for device in &self.devices {
                let bytes = device.by_category().get(category).copied().unwrap_or(0);
                out.push_str(&format!("{:>9.2}", bytes as f64 / GIB));
            }
            out.push('\n');
        }
        for (label, f) in [("total", 0), ("capacity", 1), ("free", 2)] {
            out.push_str(&format!("{label:<12}"));
            for device in &self.devices {
                let v = match f {
                    0 => device.used_bytes() as f64,
                    1 => device.capacity_bytes as f64,
                    _ => device.free_bytes() as f64,
                };
                out.push_str(&format!("{:>9.2}", v / GIB));
            }
            out.push('\n');
        }
        out.push_str(&format!("KV pool     {} tokens\n", self.pool_tokens));
        for device in &self.devices {
            let weights: Vec<&Item> = device.items.iter()
                .filter(|i| matches!(i.category, Category::Weights | Category::Embedding | Category::Drafter | Category::Experts)
                    && !i.format.is_empty())
                .collect();
            if weights.is_empty() {
                continue;
            }
            out.push_str(&format!("{} weights:", device.name()));
            for item in weights {
                out.push_str(&format!(" {}/{} {:.2}", item.group, item.format, item.bytes as f64 / GIB));
            }
            out.push('\n');
        }
        for waste in &self.waste {
            out.push_str(&format!("waste  {:<8} {:>7.2} GiB  {}\n", waste.device, waste.bytes as f64 / GIB, waste.what));
        }
        for note in &self.notes {
            out.push_str(&format!("note   {note}\n"));
        }
        out
    }
}

/// Measured bytes by category for one device (from a ledger report).
pub type Measured = BTreeMap<Category, u64>;

/// Predicted vs measured, per category: (predicted, measured, relative error).
pub fn compare(predicted: &DeviceLayout, measured: &Measured) -> Vec<(Category, u64, u64, f64)> {
    let predicted = predicted.by_category();
    let mut keys: Vec<Category> = predicted.keys().chain(measured.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    keys.into_iter().map(|c| {
        let p = predicted.get(&c).copied().unwrap_or(0);
        let m = measured.get(&c).copied().unwrap_or(0);
        let err = if m == 0 { if p == 0 { 0.0 } else { 1.0 } } else { (p as f64 - m as f64) / m as f64 };
        (c, p, m, err)
    }).collect()
}

/// Sizes the logical KV pool: the largest whole number of `unit_tokens` units
/// every KV-owning device can hold in its free bytes, capped at `target`.
/// `per_token` gives each device's KV bytes per logical token (0: holds none).
pub fn size_pool(free: &[i64], per_token: &[u64], unit_tokens: u64, target: u64) -> u64 {
    let mut tokens = target;
    for (&free, &cost) in free.iter().zip(per_token) {
        if cost == 0 {
            continue;
        }
        let fit = if free <= 0 { 0 } else { free as u64 / cost };
        tokens = tokens.min(fit);
    }
    tokens / unit_tokens.max(1) * unit_tokens.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_admission_charges_future_owners_but_not_resident_bytes_twice() {
        let resident = Item::new(Category::Transport, "transport/rdma-rings", "", 20, Basis::Exact);
        let future = Item::new(Category::Workspace, "scoring/full-logits", "", 10, Basis::Formula)
            .with_lifetime(ItemLifetime::PostReadyReserve);
        let host = HostLayout::new(100, 60, 10, 30, vec![resident.clone(), future.clone()]).unwrap();
        assert_eq!((host.resident_bytes, host.deferred_bytes, host.headroom_bytes, host.shortfall_bytes),
            (20, 10, 10, 0));
        let tight = HostLayout::new(100, 40, 10, 30, vec![resident.clone(), future.clone()]).unwrap();
        assert_eq!(tight.shortfall_bytes, 10);
        let total = HostLayout::new(60, 60, 10, 30, vec![resident, future]).unwrap();
        assert_eq!(total.shortfall_bytes, 10);
        let encoded = serde_json::to_value(&host).unwrap();
        assert_eq!(serde_json::from_value::<HostLayout>(encoded).unwrap(), host);
    }

    #[test]
    fn named_host_table_keeps_its_future_owner_and_serialized_lifetime() {
        let item = host_table_item("synthetic/ordered-table", 1024);
        assert_eq!(item.category, Category::Staging);
        assert_eq!(item.group, "synthetic/ordered-table");
        assert_eq!(item.bytes, 1024);
        assert_eq!(item.basis, Basis::Formula);
        assert_eq!(item.lifetime, ItemLifetime::PostReadyReserve);
        let encoded = serde_json::to_value(&item).unwrap();
        assert_eq!(encoded["group"], "synthetic/ordered-table");
        assert_eq!(encoded["lifetime"], "post_ready_reserve");
        assert_eq!(serde_json::from_value::<Item>(encoded).unwrap(), item);
    }

    #[test]
    fn host_sizes_are_checked_and_logits_include_only_the_live_lane_peak() {
        assert_eq!(host_logits_bytes(4096, 4096, 154880), Some(2_537_553_920));
        assert_eq!(host_logits_bytes(8192, 4096, 154880), Some(7_612_661_760));
        assert_eq!(host_logits_bytes(64, 4096, 154880), Some(39_649_280));
        assert!(host_logits_bytes(u64::MAX, 1, 1).is_none());
        assert!(host_logits_bytes(1, 1, u64::MAX).is_none());
        assert!(HostLayout::new(u64::MAX, u64::MAX, 1, u64::MAX, vec![]).is_none());
        let item = Item::new(Category::Workspace, "resident", "", u64::MAX, Basis::Exact);
        assert!(HostLayout::new(u64::MAX, u64::MAX, 0, 0, vec![item.clone(), item]).is_none());
    }

    #[test]
    fn spark_arena_measured_terms_and_reserve_override_preserve_payload_owners() {
        assert_eq!(spark_arena_driver_overhead_bytes(1 << 30), 12 << 20);
        assert_eq!(spark_arena_driver_overhead_bytes(117_692_170_240), 1_379_205_120);
        assert_eq!(spark_arena_driver_overhead_bytes(0), 0);
        assert_eq!(spark_arena_driver_overhead_bytes(1), 48);
        assert_eq!(spark_arena_driver_overhead_bytes(u64::MAX), 216_172_782_113_783_808);
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 << 30,
            worker_budget_bytes: 110 << 30, kv_tokens: 0, items: vec![
                Item::new(Category::Experts, "arena", "", 64 << 30, Basis::Exact),
                Item::new(Category::Transport, "ring", "", 128 << 20, Basis::Exact),
                Item::new(Category::Staging, "upload", "", 4096, Basis::Exact)] };
        charge_spark_runtime(&mut device).unwrap();
        let first = device.clone();
        charge_spark_runtime(&mut device).unwrap();
        assert_eq!(device, first);
        crate::memory_policy::MemoryPolicy { spark_reserve: Some(7), ..Default::default() }.scoped(|| {
            charge_spark_runtime(&mut device).unwrap();
            assert_eq!(spark_runtime_reserve_bytes(64 << 30), Some(7 + spark_arena_driver_overhead_bytes(64 << 30)));
            assert_eq!(device.items.len(), 5);
            assert_eq!(device.items.iter().find(|item| item.group == "ring").unwrap().bytes, 128 << 20);
            assert_eq!(device.items.iter().find(|item| item.group == "upload").unwrap().bytes, 4096);
            assert_eq!(device.items.iter().find(|item| item.group == "SPARK_RESERVE").unwrap().bytes, 7);
        });
    }

    #[test]
    fn spark_process_census_is_named_ready_resident_and_not_worker_future_headroom() {
        let item = spark_worker_process_resident_item();
        assert_eq!(item.bytes, 1_596_606_464);
        assert_eq!(SPARK_WORKER_PROCESS_RESIDENT[0].arch, "sm_121");
        assert_eq!(SPARK_WORKER_PROCESS_RESIDENT[0].bytes(), item.bytes);
        assert_eq!(item.basis, Basis::Calibrated);
        let selected = Some((crate::ExpertGeometry::KIMI_K3, 6, 1024));
        let resident = 64 << 30;
        assert_eq!(spark_runtime_bytes(selected, resident).unwrap()
            - spark_worker_future_bytes(selected, resident).unwrap(), item.bytes);
        for reserve in [0, 7] {
            crate::memory_policy::MemoryPolicy { spark_reserve: Some(reserve), ..Default::default() }.scoped(|| {
                assert_eq!(spark_worker_future_bytes(selected, resident),
                    Some(spark_arena_driver_overhead_bytes(resident) + reserve + 4096));
                assert_eq!(spark_runtime_bytes(selected, resident), spark_worker_future_bytes(selected, resident));
                let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 << 30,
                    worker_budget_bytes: 110 << 30, kv_tokens: 0, items: vec![
                        Item::new(Category::Experts, "arena", "", resident, Basis::Exact)] };
                charge_spark_runtime(&mut device).unwrap();
                let driver = device.items.iter().find(|item| item.group == SPARK_ARENA_DRIVER_OVERHEAD_GROUP).unwrap();
                assert_eq!(driver.bytes, spark_arena_driver_overhead_bytes(resident));
                assert_eq!(driver.category, Category::Workspace);
                assert!(!device.items.iter().any(|item| item.group == SPARK_WORKER_PROCESS_RESIDENT_GROUP));
            });
        }
        assert_eq!(item.lifetime, ItemLifetime::ReadyResident);
        assert!(SPARK_WORKER_PROCESS_RESIDENT_SOURCE.contains("2026-11-07"));
        let encoded = serde_json::to_value(&item).unwrap();
        assert_eq!(encoded["lifetime"], "ready_resident");
        assert_eq!(serde_json::from_value::<Item>(encoded).unwrap(), item);
    }

    #[test]
    fn item_lifetime_is_explicit_and_survives_serialization() {
        let resident = Item::new(Category::Runtime, "context", "", 100, Basis::Calibrated);
        assert_eq!(resident.lifetime, ItemLifetime::ReadyResident);
        for lifetime in [ItemLifetime::ReadyResident, ItemLifetime::PostReadyReserve] {
            let item = resident.clone().with_lifetime(lifetime);
            let encoded = serde_json::to_value(&item).unwrap();
            assert_eq!(encoded["lifetime"], serde_json::to_value(lifetime).unwrap());
            assert_eq!(serde_json::from_value::<Item>(encoded).unwrap(), item);
        }
    }

    #[test]
    fn scopes_map_onto_categories() {
        assert_eq!(Category::of_scope("weights/dense-nvfp4"), Category::Weights);
        assert_eq!(Category::of_scope("drafter/workspace"), Category::Drafter);
        assert_eq!(Category::of_scope("transport/rdma-rings"), Category::Transport);
        assert_eq!(Category::of_scope("peer-split"), Category::Transport);
        assert_eq!(Category::of_scope("staging/sync-h2d"), Category::Staging);
        assert_eq!(Category::of_scope("other"), Category::Runtime);
        assert_eq!(Category::of_scope("v41/kv"), Category::Kv);
        assert_eq!(Category::of_scope("v41/draft weights"), Category::Drafter);
        assert_eq!(Category::of_scope("v41/allocated KV cache"), Category::Kv);
    }

    #[test]
    fn spark_worker_budget_is_required_and_independent_of_host_capacity() {
        let device = DeviceLayout { kind: DeviceKind::Spark, index: 0,
            capacity_bytes: 121 << 30, worker_budget_bytes: 108 << 30,
            items: vec![], kv_tokens: 0 };
        let encoded = serde_json::to_value(&device).unwrap();
        assert_eq!(encoded["worker_budget_bytes"], 108u64 << 30);
        assert_eq!(serde_json::from_value::<DeviceLayout>(encoded.clone()).unwrap(), device);
        let mut missing = encoded;
        missing.as_object_mut().unwrap().remove("worker_budget_bytes");
        assert!(serde_json::from_value::<DeviceLayout>(missing).is_err());
    }

    #[test]
    fn pool_takes_the_tightest_device_in_whole_units() {
        // 10 GiB free at 1 MiB/token and 4 GiB free at 512 KiB/token.
        let free = [10i64 << 30, 4 << 30];
        let cost = [1 << 20, 1 << 19];
        assert_eq!(size_pool(&free, &cost, 64, 1 << 30), 8192);
        assert_eq!(size_pool(&free, &cost, 64, 1000), 960);
        assert_eq!(size_pool(&[-5, 1 << 30], &[1, 0], 64, 1 << 20), 0);
    }
}
