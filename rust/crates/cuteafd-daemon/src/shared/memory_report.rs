//! Memory reports from the allocation ledger (`cuteafd_ffi::memory_ledger`):
//! per device, the bytes the runtime reports in use, the bytes this process
//! allocated by category, the untracked rest (CUDA context, modules, cuBLAS,
//! graph executables), weights by checkpoint tensor stem and resident format,
//! and any tensor resident in two formats. One JSON line per report under the
//! `cuteafd::memory` target; `scripts/bench/memory-audit.py` tabulates them.

use cuteafd_ffi::memory_ledger::{self, Space};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::time::Duration;

fn meminfo() -> Value {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else { return Value::Null };
    let mut out = Map::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(key), Some(value)) = (parts.next(), parts.next()) else { continue };
        let key = key.trim_end_matches(':');
        if matches!(key, "MemTotal" | "MemAvailable" | "Cached" | "AnonPages" | "Shmem" | "Mlocked") {
            if let Ok(kib) = value.parse::<u64>() {
                out.insert(key.to_owned(), json!(kib * 1024));
            }
        }
    }
    Value::Object(out)
}

/// Builds one report. Queries each device on the calling thread and restores
/// nothing: call it from a thread that does not launch work, or restore the
/// device afterwards (see [`log`]).
pub(crate) fn report(stage: &str) -> Value {
    let snapshot = memory_ledger::snapshot();
    let mut per_device = Vec::new();
    // Only devices this process allocated on: querying another would create a context there.
    for device in snapshot.devices() {
        let tracked = snapshot.total(Space::Device, device) + snapshot.total(Space::Managed, device);
        let scopes: BTreeMap<&str, usize> = snapshot.by_scope(Space::Device, device);
        let mut entry = json!({
            "device": device,
            "tracked": tracked,
            "peak": snapshot.peak.get(&(Space::Device, device)).copied().unwrap_or(0),
            "scopes": scopes,
        });
        if let Some((free, total)) = memory_ledger::device_memory(device) {
            let used = total - free;
            entry["used"] = json!(used);
            entry["total"] = json!(total);
            entry["untracked"] = json!(used as i64 - tracked as i64);
        }
        per_device.push(entry);
    }
    let weights: Vec<Value> = snapshot.rows.iter().filter(|r| r.key.tensor.is_some())
        .map(|r| json!([r.key.space.label(), r.key.device, r.key.scope, r.key.tensor, r.key.format, r.bytes,
            r.allocations]))
        .collect();
    let dual: Vec<Value> = snapshot.dual_formats().into_iter()
        .map(|(device, tensor, formats)| json!({"device": device, "tensor": tensor, "formats": formats}))
        .collect();
    json!({
        "stage": stage,
        "devices": per_device,
        "pinned": {
            "tracked": snapshot.total(Space::Pinned, -1),
            "peak": snapshot.peak.get(&(Space::Pinned, -1)).copied().unwrap_or(0),
            "scopes": snapshot.by_scope(Space::Pinned, -1),
        },
        "weights": weights,
        "dual_formats": dual,
        "host": meminfo(),
    })
}

/// Logs a report from a helper thread (the caller's device selection is untouched).
pub(crate) fn log(stage: &str) {
    let stage = stage.to_owned();
    let value = std::thread::spawn(move || report(&stage)).join();
    if let Ok(value) = value {
        tracing::info!(target: "cuteafd::memory", report = %value, "memory ledger");
    }
}

/// Logs a report now and again whenever device use or the ledger moves by at
/// least 64 MiB (first graph captures, lazily sized workspaces), checking every
/// `period`. Runs for the life of the process.
pub(crate) fn monitor(stage: &'static str, period: Duration) {
    if std::env::var_os("CUTEAFD_MEMORY_MONITOR").is_some_and(|v| v == "0") {
        return;
    }
    let _ = std::thread::Builder::new().name("memory-report".into()).spawn(move || {
        let mut last: Option<Vec<i64>> = None;
        let mut sequence = 0u64;
        loop {
            let value = report(stage);
            if value["devices"].as_array().is_none_or(Vec::is_empty) && value["pinned"]["tracked"] == 0 {
                std::thread::sleep(period);
                continue;
            }
            let signature: Vec<i64> = value["devices"].as_array().into_iter().flatten()
                .flat_map(|d| [d["used"].as_i64().unwrap_or(0), d["tracked"].as_i64().unwrap_or(0)])
                .chain([value["pinned"]["tracked"].as_i64().unwrap_or(0)])
                .collect();
            let moved = last.as_ref().is_none_or(|previous| previous.iter().zip(&signature)
                .any(|(a, b)| (a - b).abs() >= 64 << 20));
            if moved {
                tracing::info!(target: "cuteafd::memory", sequence, report = %value, "memory ledger");
                sequence += 1;
                last = Some(signature);
            }
            std::thread::sleep(period);
        }
    });
}

/// Frees the pinned upload staging the weight loaders grew (see
/// `NativeLibrary::release_sync_h2d_staging`) once a family's weights are resident.
pub(crate) fn release_load_staging(library: &cuteafd_ffi::NativeLibrary) {
    match library.release_sync_h2d_staging() {
        Ok(0) => {}
        Ok(bytes) => tracing::info!(bytes, "released load-time pinned upload staging"),
        Err(error) => tracing::warn!(%error, "could not release load-time pinned upload staging"),
    }
}

/// GB10 allocations reclaim page cache; raw CUDA free reports only MemFree.
/// Use this only after selecting a unified-memory device, and fail closed.
pub(crate) fn unified_available_bytes() -> anyhow::Result<usize> {
    use anyhow::Context;
    let text = std::fs::read_to_string("/proc/meminfo")
        .context("reading unified-memory admission availability")?;
    unified_available_from_meminfo(&text)
}

fn unified_available_from_meminfo(text: &str) -> anyhow::Result<usize> {
    use anyhow::Context;
    let value = text.lines().find_map(|line| line.strip_prefix("MemAvailable:"))
        .context("unified-memory admission requires MemAvailable")?;
    let mut fields = value.split_whitespace();
    let kib: usize = fields.next().context("missing MemAvailable value")?
        .parse().context("invalid MemAvailable value")?;
    anyhow::ensure!(fields.next() == Some("kB") && fields.next().is_none(),
        "invalid MemAvailable units");
    kib.checked_mul(1024).context("MemAvailable byte overflow")
}

#[cfg(test)]
mod unified_memory_tests {
    use super::unified_available_from_meminfo;

    #[test]
    fn admission_includes_reclaimable_cache_not_just_memfree() {
        let text = "MemFree: 1024 kB\nMemAvailable: 47185920 kB\nCached: 41943040 kB\n";
        assert_eq!(unified_available_from_meminfo(text).unwrap(), 45usize << 30);
        assert_eq!(unified_available_from_meminfo("MemAvailable: 0 kB\n").unwrap(), 0);
    }

    #[test]
    fn admission_fails_closed_on_missing_malformed_or_overflowed_values() {
        for text in ["MemFree: 1024 kB\n", "MemAvailable: nope kB\n",
            "MemAvailable: 42 MB\n", "MemAvailable: 42\n", "MemAvailable: 42 kB extra\n"] {
            assert!(unified_available_from_meminfo(text).is_err(), "{text}");
        }
        assert!(unified_available_from_meminfo(&format!("MemAvailable: {} kB\n", usize::MAX)).is_err());
    }
}

/// The kernel page cache (`Cached` in /proc/meminfo), bytes.
pub(crate) fn cached_bytes() -> Option<u64> {
    meminfo()["Cached"].as_u64()
}

/// One GPU holding KV records: its records per logical token and the bytes
/// that must stay free there after the pool (workspaces, graphs, drafter,
/// headroom) — the planner's per-device costs.
pub(crate) struct KvDevice {
    pub device: i32,
    pub bytes_per_token: u64,
    pub reserve_bytes: u64,
}

/// The largest pool (whole `unit_rows` units, at most `target` tokens) every
/// device can hold in its free memory now, after its reserve. Restores the
/// calling thread's device.
/// Fixed pools are checked (never silently shrunk) with the same workspace,
/// graph, state and drafter reserves as auto pools, before cache allocation.
/// `requested` is None for automatic admission.
pub(crate) fn admitted_pool_tokens(library: &cuteafd_ffi::NativeLibrary, devices: &[KvDevice], unit_rows: u64,
    target: u64, requested: Option<u64>) -> anyhow::Result<u64> {
    use cuteafd_core::serving_capacity::{DeviceMemory, GpuMemoryBudget};
    anyhow::ensure!(unit_rows > 0, "KV admission needs positive allocation units");
    let wanted = requested.unwrap_or(unit_rows).div_ceil(unit_rows)
        .checked_mul(unit_rows).ok_or_else(|| anyhow::anyhow!("KV admission token overflow"))?;
    let current = library.cuda_get_device()?;
    let mut free = Vec::with_capacity(devices.len());
    for device in devices {
        library.cuda_set_device(device.device)?;
        let sample = library.cuda_memory_info();
        library.cuda_set_device(current)?;
        let (available, total) = sample?;
        let required = device.bytes_per_token.checked_mul(wanted)
            .and_then(|bytes| bytes.checked_add(device.reserve_bytes))
            .ok_or_else(|| anyhow::anyhow!("KV admission byte overflow"))?;
        GpuMemoryBudget(total as u64).admit(DeviceMemory { device: u32::try_from(device.device)?,
            total_bytes: total as u64, baseline_free_bytes: available as u64 }, required)?;
        tracing::info!(device = device.device, fixed_reserve_bytes = device.reserve_bytes,
            bytes_per_token = device.bytes_per_token, requested_pool_tokens = ?requested,
            "KV admission including workspaces, state, graphs and draft reserve");
        free.push(available as i64 - device.reserve_bytes as i64);
    }
    let per_token: Vec<u64> = devices.iter().map(|d| d.bytes_per_token).collect();
    let tokens = requested.map_or_else(|| cuteafd_core::memory_layout::size_pool(&free, &per_token, unit_rows, target),
        |_| wanted);
    tracing::info!(tokens, target, ?free, ?per_token, "KV pool admitted after fixed costs");
    if tokens < unit_rows {
        return Err(NoKvRoom { free_after_reserve: free }.into());
    }
    Ok(tokens)
}

/// An automatic KV pool that has no room after the fixed costs: free memory less each GPU's
/// reserve, per GPU.
#[derive(Debug)]
pub(crate) struct NoKvRoom {
    pub free_after_reserve: Vec<i64>,
}

impl std::fmt::Display for NoKvRoom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no room for a KV pool after fixed costs (free after reserve {:?} bytes)", self.free_after_reserve)
    }
}

impl std::error::Error for NoKvRoom {}

/// Whether a KV admission was refused for want of memory: an automatic pool with no room after the
/// fixed costs, or a GPU that cannot hold a fixed pool beside them. Bad samples, overflows and other
/// errors are not shortfalls.
pub(crate) fn kv_shortfall(error: &anyhow::Error) -> bool {
    use cuteafd_core::serving_capacity::CapacityError;
    error.downcast_ref::<NoKvRoom>().is_some()
        || matches!(error.downcast_ref::<CapacityError>(), Some(CapacityError::GpuBudgetExceeded { .. }))
}

/// What a measured admission keeps free beside the KV pool's records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MeasuredReserve {
    /// Runtime growth the start-up cannot see (`--headroom-gib`).
    pub headroom: u64,
    /// Graph executables captured after start-up: the graph budget, else the planner's allowance.
    pub graphs: u64,
    /// Allocated with the pool or after it: recurrent state, replay records, prefix marks.
    pub later: u64,
}

/// The KV pool of an engine that allocated everything else first (drafter, transports and
/// intake, step workspaces, selector): the GPU's free memory now, less `reserve`, over its
/// records per token, in whole `unit_rows` units. No calibrated workspace, drafter or runtime
/// allowance: those are allocated, so free memory already shows them. `requested` is a fixed
/// pool, checked instead of sized.
pub(crate) fn measured_pool_tokens(library: &cuteafd_ffi::NativeLibrary, device: i32, bytes_per_token: u64,
    unit_rows: u64, reserve: MeasuredReserve, requested: Option<u64>) -> anyhow::Result<usize> {
    let reserve_bytes = reserve.headroom.checked_add(reserve.graphs).and_then(|bytes| bytes.checked_add(reserve.later))
        .ok_or_else(|| anyhow::anyhow!("KV admission reserve overflows"))?;
    tracing::info!(device, headroom_bytes = reserve.headroom, graph_bytes = reserve.graphs,
        later_bytes = reserve.later, "KV admission from measured free memory after start-up allocations");
    let tokens = admitted_pool_tokens(library, &[KvDevice { device, bytes_per_token, reserve_bytes }], unit_rows,
        cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS, requested)?;
    Ok(usize::try_from(tokens)?)
}

/// Bytes of a checkpoint directory's safetensors shards (a drafter's resident
/// size when it keeps its checkpoint representation).
pub(crate) fn safetensors_bytes(directory: &std::path::Path) -> u64 {
    std::fs::read_dir(directory).map(|entries| entries.flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|e| std::fs::metadata(e.path()).ok().map(|m| m.len())).sum()).unwrap_or(0)
}

fn admission_cache_ranks(geometry: &cuteafd_loader::serving_capacity::FamilyCacheGeometry,
    glmf_split: bool) -> Vec<cuteafd_loader::serving_capacity::RankCacheGeometry> {
    if !glmf_split { return geometry.ranks.clone(); }
    geometry.ranks.iter().flat_map(|rank| {
        let mut half = rank.clone();
        half.active_state_per_sequence_bytes /= 2;
        half.retained_mark_bytes /= 2;
        half.speculative_replay_bytes /= 2;
        [half.clone(), half]
    }).collect()
}

/// The planner's automatic KV pool for an engine about to allocate its cache:
/// each GPU's free memory now, minus what the planner says is still to come
/// there (step workspaces, peer exchange, drafter, recurrent state, prefix
/// marks, graph executables, headroom), over the family's records per token.
/// `devices` lists the KV-owning GPUs, lead first; `drafter` is an external
/// drafter checkpoint the lead GPU will load; `mark_slots` the slots of the
/// prefix mark arena the engine will allocate on every GPU (0: none).
#[allow(clippy::too_many_arguments)]
pub(crate) fn planned_pool_tokens(library: &cuteafd_ffi::NativeLibrary, snapshot: &std::path::Path, devices: &[i32],
    drafter: Option<&std::path::Path>, prefill_rows: usize, slots: usize, mark_slots: u64,
    requested: Option<u64>, future_expert_bytes: u64) -> anyhow::Result<usize> {
    planned_pool_tokens_with_extra(library, snapshot, devices, drafter, prefill_rows, slots, mark_slots, requested,
        future_expert_bytes, 0, Default::default())
}

/// As `planned_pool_tokens`, also reserving a family's optional per-GPU
/// buffers (e.g. GLM Flash split KDA partials) before admitting the pool,
/// with GLM Flash's DSA index cache layout `glmf_index`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn planned_pool_tokens_with_extra(library: &cuteafd_ffi::NativeLibrary, snapshot: &std::path::Path,
    devices: &[i32], drafter: Option<&std::path::Path>, prefill_rows: usize, slots: usize, mark_slots: u64,
    requested: Option<u64>, future_expert_bytes: u64, extra_reserve_bytes: u64,
    glmf_index: cuteafd_loader::serving_capacity::GlmfIndexCache) -> anyhow::Result<usize> {
    let reserves = vec![RankReserve { extra_bytes: extra_reserve_bytes, workspace_bytes: None }; devices.len()];
    planned_pool_tokens_with_reserves(library, snapshot, devices, drafter, prefill_rows, slots, mark_slots,
        requested, future_expert_bytes, &reserves, glmf_index, 4)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RankReserve {
    pub extra_bytes: u64,
    /// Exact workspace union replaces, rather than adds to, the planner allowance.
    pub workspace_bytes: Option<u64>,
}

pub(crate) fn lead_reserves(ranks: usize, lead_bytes: u64) -> Vec<RankReserve> {
    (0..ranks).map(|rank| RankReserve {
        extra_bytes: if rank == 0 { lead_bytes } else { 0 }, workspace_bytes: None,
    }).collect()
}

/// As `planned_pool_tokens_with_extra`, with each GPU's own reserve (`reserves`, lead first) and
/// `kda_state_bytes` per GLM Flash KDA recurrent-state element (4 FP32, 2 BF16 with `--kda-state`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn planned_pool_tokens_with_reserves(library: &cuteafd_ffi::NativeLibrary, snapshot: &std::path::Path,
    devices: &[i32], drafter: Option<&std::path::Path>, prefill_rows: usize, slots: usize, mark_slots: u64,
    requested: Option<u64>, future_expert_bytes: u64, reserves: &[RankReserve],
    glmf_index: cuteafd_loader::serving_capacity::GlmfIndexCache, kda_state_bytes: u64)
    -> anyhow::Result<usize> {
    use anyhow::Context;
    anyhow::ensure!(reserves.len() == devices.len(), "reserve must cover every admitted GPU");
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(snapshot)?;
    let family = cuteafd_loader::plan::family::detect(&checkpoint).context("no family for this checkpoint")?;
    let model = family.open(&checkpoint).map_err(|e| anyhow::anyhow!("{}", e.0))?;
    // The model contract currently describes GLM Flash's one-GPU cache.
    // Its implemented split replicates MLA and divides KDA heads per GPU.
    let glmf = family.id() == "glm5_flash";
    let cache_ranks = if glmf { 1 } else { devices.len() };
    let geometry = model.cache_geometry(cuteafd_loader::serving_capacity::CacheOptions {
        coordinator_ranks: cache_ranks, glmf_index, kda_state_bytes, ..Default::default() })?
        .with_context(|| format!("{} has no cache geometry for {} GPUs", family.id(), devices.len()))?;
    let reserve = Reserve {
        costs: cuteafd_loader::plan::layout::family_costs(family.id()),
        headroom: cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes,
        draft: drafter.map_or(0, |d| safetensors_bytes(d) + (1300 << 20)),
        prefill_rows, slots, mark_slots, future_experts: future_expert_bytes,
    };
    let kv = kv_devices(&geometry, glmf, devices, &reserve, reserves)?;
    let tokens = admitted_pool_tokens(library, &kv, geometry.logical_unit_rows.max(1),
        cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS, requested)?;
    Ok(usize::try_from(tokens)?)
}

/// What a KV-owning GPU keeps beside the pool, besides its cache geometry's state.
struct Reserve {
    costs: cuteafd_loader::plan::layout::FamilyCosts,
    headroom: u64,
    /// External drafter bytes on the lead GPU.
    draft: u64,
    prefill_rows: usize,
    /// Sequences with recurrent state.
    slots: usize,
    /// Prefix mark arena slots on every GPU.
    mark_slots: u64,
    future_experts: u64,
}

/// Per-GPU KV admission: records per token, and every byte the GPU must still hold after the
/// pool: workspaces (the planner's allowance, or each GPU's exact union in `reserves`), peer
/// exchange, the drafter, the recurrent state of `slots` sequences with its speculative replay
/// records and the prefix mark arena (both exactly as the runtime allocates them, budget or not),
/// graph executables, headroom and each GPU's extra reserve.
fn kv_devices(geometry: &cuteafd_loader::serving_capacity::FamilyCacheGeometry, glmf: bool, devices: &[i32],
    reserve: &Reserve, reserves: &[RankReserve]) -> anyhow::Result<Vec<KvDevice>> {
    let costs = &reserve.costs;
    let unit = geometry.logical_unit_rows.max(1);
    let split = devices.len() == 2;
    let ranks = admission_cache_ranks(geometry, glmf && split);
    anyhow::ensure!(ranks.len() == devices.len(), "cache geometry must cover every admitted GPU");
    anyhow::ensure!(reserves.len() == devices.len(), "reserve must cover every admitted GPU");
    Ok(devices.iter().zip(&ranks).enumerate().map(|(index, (&device, rank))| {
        let role = if !split { 0 } else if index == 0 { 1 } else { 2 };
        let workspace = reserves[index].workspace_bytes.unwrap_or(
            costs.workspace_bytes[role] * reserve.prefill_rows.max(1) as u64 / 4096);
        let state = rank.fixed_state_bytes + rank.active_state_per_sequence_bytes * reserve.slots as u64
            + rank.speculative_replay_bytes;
        let marks = rank.retained_mark_bytes * reserve.mark_slots;
        tracing::info!(device, state_bytes = state, mark_slots = reserve.mark_slots, mark_bytes = marks,
            "KV admission reserves recurrent state, replay records and prefix marks");
        KvDevice {
            device,
            bytes_per_token: (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes).div_ceil(unit),
            reserve_bytes: workspace + if split { costs.exchange_bytes } else { 0 }
                + if index == 0 { reserve.draft } else { 0 }
                + state + marks + costs.graph_bytes[role] + reserve.headroom + reserves[index].extra_bytes
                + if index == 0 { reserve.future_experts } else { 0 },
        }
    }).collect())
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use cuteafd_loader::serving_capacity::{FamilyCacheGeometry, KvPlacement, RankCacheGeometry};

    #[test]
    fn only_memory_refusals_are_kv_shortfalls() {
        use cuteafd_core::serving_capacity::CapacityError;
        let no_room = anyhow::Error::from(NoKvRoom { free_after_reserve: vec![-1_024, 4_096] });
        assert!(kv_shortfall(&no_room));
        assert_eq!(no_room.to_string(), "no room for a KV pool after fixed costs (free after reserve [-1024, 4096] bytes)");
        assert!(kv_shortfall(&no_room.context("planned admission")));
        let fixed = anyhow::Error::from(CapacityError::GpuBudgetExceeded { device: 0, required: 3, budget: 2, shortfall: 1 });
        assert!(kv_shortfall(&fixed));
        for other in [anyhow::Error::from(CapacityError::Invalid("invalid GPU budget or physical memory sample")),
            anyhow::Error::from(CapacityError::Overflow("GPU allocation budget")), anyhow::anyhow!("CUDA error 2")] {
            assert!(!kv_shortfall(&other), "{other}");
        }
    }

    #[test]
    fn full_logits_reserve_is_lead_only() {
        for ranks in [1, 2] {
            let reserves = lead_reserves(ranks, 12345);
            assert_eq!(reserves[0].extra_bytes, 12345);
            assert!(reserves.iter().skip(1).all(|r| r.extra_bytes == 0));
        }
    }

    #[test]
    fn glmf_split_admits_replicated_mla_and_half_kda_on_every_gpu() {
        let rank = RankCacheGeometry { persistent_unit_bytes: 1024, pool_metadata_unit_bytes: 4,
            active_state_per_sequence_bytes: 2048, retained_mark_bytes: 2048,
            speculative_replay_bytes: 512, fixed_state_bytes: 768, ..Default::default() };
        let geometry = FamilyCacheGeometry { logical_unit_rows: 256, placement: KvPlacement::SingleDevice,
            ranks: vec![rank.clone()] };
        assert_eq!(admission_cache_ranks(&geometry, false), vec![rank]);
        let split = admission_cache_ranks(&geometry, true);
        assert_eq!(split.len(), 2);
        for rank in split {
            assert_eq!((rank.persistent_unit_bytes, rank.pool_metadata_unit_bytes, rank.fixed_state_bytes),
                (1024, 4, 768));
            assert_eq!((rank.active_state_per_sequence_bytes, rank.retained_mark_bytes, rank.speculative_replay_bytes),
                (1024, 1024, 256));
        }
    }

    /// The admission keeps room for exactly what the runtime allocates beside the pool: the
    /// recurrent state of every slot, the replay records (with or without a GPU budget) and
    /// the mark arena's slots, halved per GPU under a head split.
    #[test]
    fn reserve_holds_the_replay_records_and_the_runtime_mark_arena() {
        let rank = RankCacheGeometry { persistent_unit_bytes: 1024, pool_metadata_unit_bytes: 4,
            active_state_per_sequence_bytes: 2048, retained_mark_bytes: 2048,
            speculative_replay_bytes: 512, fixed_state_bytes: 768, ..Default::default() };
        let geometry = FamilyCacheGeometry { logical_unit_rows: 256, placement: KvPlacement::SingleDevice,
            ranks: vec![rank] };
        let costs = cuteafd_loader::plan::layout::family_costs("glm5_flash");
        let reserve = |mark_slots| Reserve { costs, headroom: 7, draft: 11, prefill_rows: 4096, slots: 16,
            mark_slots, future_experts: 17 };
        let extra = |gpus| vec![RankReserve { extra_bytes: 13, workspace_bytes: None }; gpus];
        let fixed = |role: usize| costs.workspace_bytes[role] + costs.graph_bytes[role] + 7 + 13;
        let one = kv_devices(&geometry, true, &[0], &reserve(34), &extra(1)).unwrap();
        assert_eq!(one[0].bytes_per_token, 5);
        assert_eq!(one[0].reserve_bytes, fixed(0) + 11 + 17 + 768 + 16 * 2048 + 512 + 34 * 2048);
        let no_arena = kv_devices(&geometry, true, &[0], &reserve(0), &extra(1)).unwrap();
        assert_eq!(one[0].reserve_bytes - no_arena[0].reserve_bytes, 34 * 2048);
        let split = kv_devices(&geometry, true, &[0, 1], &reserve(34), &extra(2)).unwrap();
        assert_eq!(split[0].reserve_bytes,
            fixed(1) + costs.exchange_bytes + 11 + 17 + 768 + 16 * 1024 + 256 + 34 * 1024);
        assert_eq!(split[1].reserve_bytes, fixed(2) + costs.exchange_bytes + 768 + 16 * 1024 + 256 + 34 * 1024);
        // An exact workspace union (all-row prefill logits) replaces the planner's allowance on its GPU.
        let exact = [RankReserve { extra_bytes: 13, workspace_bytes: Some(1_000) }];
        let scoring = kv_devices(&geometry, true, &[0], &reserve(34), &exact).unwrap();
        assert_eq!(one[0].reserve_bytes - scoring[0].reserve_bytes, costs.workspace_bytes[0] - 1_000);
        assert!(kv_devices(&geometry, true, &[0, 1], &reserve(34), &extra(1)).is_err());
    }
}
