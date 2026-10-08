//! Startup-only source-pool sizing. No allocation policy runs in the token loop.
use crate::families::deepseek_v41::v41_backbone_cache::BackboneCache;
use anyhow::{ensure, Context, Result};
use std::{fmt, str::FromStr};

pub(crate) mod distributed;

// Aggregate source capacity is independent of the retained snapshot count.
const DEFAULT_POOL_TOKENS: usize = 2 * 1_048_576;
// Extra pages cover retained partial tails and active copy-on-write frontiers.
const MAX_GROUPS: usize = 131_072;
const GROUP_BYTES: usize = 5 * 256 * (68 + cuteafd_ffi::V41Kv::COMPRESSED_ROW_BYTES);
// Request scratch and graph/runtime allocations. Snapshot arenas are already live.
// Kept outside the eagerly allocated cache; this is not a CUDA process quota.
pub(super) const RUNTIME_HEADROOM: usize = 2 * 1024 * 1024 * 1024;
// Post-ready untracked growth in official-native C1/C16 and tools,
// plus 25%, rounded up to 64 MiB. Separate from transient runtime scratch.
pub(super) const SINGLE_GRAPH_RESERVE: usize = 3264 * 1024 * 1024;
pub(super) const DUAL_GPU0_GRAPH_RESERVE: usize = 1472 * 1024 * 1024;
pub(super) const DUAL_GPU1_GRAPH_RESERVE: usize = 1472 * 1024 * 1024;

pub(super) fn graph_reserve_bytes(total: usize, dual_gpu: Option<usize>) -> usize {
    // Plat2 already charges its measured fixed-shape graph envelope.
    if total <= 32usize << 30 { return 0; }
    match dual_gpu {
        None => SINGLE_GRAPH_RESERVE,
        Some(0) => DUAL_GPU0_GRAPH_RESERVE,
        Some(1) => DUAL_GPU1_GRAPH_RESERVE,
        Some(_) => unreachable!("V4.1 has at most two RTX owners"),
    }
}
// CUDA allocation granularity and modules first used after this startup sample.
const SMALL_CARD_SAMPLE_MARGIN: usize = 32 * 1024 * 1024;

pub(super) fn embedding_placement(explicit: Option<crate::shared::memory::EmbedPlacement>, total: usize)
    -> crate::shared::memory::EmbedPlacement {
    explicit.unwrap_or(if total <= 32usize << 30 {
        crate::shared::memory::EmbedPlacement::Host
    } else { crate::shared::memory::EmbedPlacement::Gpu })
}

pub(super) fn measured_pool_memory(lib: &cuteafd_ffi::NativeLibrary) -> Result<(usize, usize)> {
    let (free, total) = lib.cuda_memory_info()?;
    if total > 32usize << 30 { return Ok((free, total)); }
    let device = lib.cuda_get_device()?;
    let live = cuteafd_ffi::memory_ledger::snapshot().total(cuteafd_ffi::memory_ledger::Space::Device, device);
    let occupied = total - free;
    // Empirical small-card reserve, not a per-executable model: official V4.1 Flash,
    // 31.8 GiB/C16 grew 1,107,296,256 bytes with six shapes, and 838,860,800
    // with eight shapes plus bounded index selection, ready -> ten-batch plateau.
    // The 2 GiB envelope also covers untracked driver/allocator growth;
    // other shape sets still require their own warmed margin gate.
    let graph_budget = std::env::var("CUTEAFD_V41_GRAPH_BUDGET_MIB").ok()
        .map(|value| value.parse::<usize>()).transpose()?.unwrap_or(2048)
        .checked_mul(1 << 20).context("V4.1 graph budget overflow")?;
    let admitted_free = free.checked_sub(SMALL_CARD_SAMPLE_MARGIN)
        .and_then(|bytes| bytes.checked_sub(graph_budget))
        .context("32 GB V4.1 startup leaves no room for the measured-pool and graph margins")?;
    tracing::info!(occupied_bytes=occupied, tracked_owner_bytes=live,
        measured_context_module_allocator_bytes=occupied.saturating_sub(live),
        sample_margin_bytes=SMALL_CARD_SAMPLE_MARGIN, graph_budget_bytes=graph_budget, admitted_free_bytes=admitted_free,
        "V4.1 measured CUDA occupancy charged before pool sizing");
    Ok((admitted_free, total))
}

/// Resolve the planner's token budget after fixed owners are live (or charged
/// to a synthetic free-memory sample for deferred TP2 experts). The byte plan
/// is then passed to the existing allocator, which verifies its own formula.
pub(super) fn planned_pool_size(args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)]) -> Result<Option<ByteSize>> {
    let Some(requested) = args.pool_tokens else { return Ok(args.kv_pool_size); };
    let automatic = requested == 0;
    let target = if automatic {
        let default = if memory.iter().any(|&(_, total)| total <= 32usize << 30) { 1_048_576 } else { DEFAULT_POOL_TOKENS as u64 };
        default.max((args.max_context_tokens as u64).div_ceil(512) * 512 + args.concurrency as u64 * 512)
    } else { requested };
    let config: serde_json::Value = serde_json::from_reader(std::fs::File::open(args.snapshot.join("config.json"))?)?;
    let available: Vec<u64> = memory.iter().enumerate().map(|(gpu, &(free, total))| {
        ensure!(total > 0 && free <= total, "invalid GPU {gpu} memory information");
        let policy_ceiling = args.memory_reservation.map(|r| r.bytes(total)).transpose()?.unwrap_or(total);
        let small_card = total <= 32usize << 30;
        let ceiling = if small_card {
            policy_ceiling.min(cuteafd_core::serving_capacity::admission_ceiling(total as u64,
                97, cuteafd_core::serving_capacity::small_card_headroom_bytes(total as u64))? as usize)
        } else if automatic { policy_ceiling.min((total as u128 * 97 / 100) as usize) }
            else { policy_ceiling };
        // The small-card ceiling already includes its absolute runtime slack.
        let runtime = if small_card { 0 } else if automatic { 3usize << 30 }
            else if memory.len() == 1 { RUNTIME_HEADROOM } else { distributed::RUNTIME_HEADROOM };
        let runtime = runtime + graph_reserve_bytes(total, (memory.len() == 2).then_some(gpu));
        Ok(ceiling.checked_sub(total - free).and_then(|n| n.checked_sub(runtime))
            .with_context(|| format!("GPU {gpu} planner leaves no room after fixed owners and runtime reserve"))? as u64)
    }).collect::<Result<_>>()?;
    let groups = cuteafd_loader::serving_capacity::deepseek_v41_pool_groups(&config,
        args.concurrency as u64, args.prefix_cache_entries as u64, &available, target, automatic, args.tp2_attention)?;
    if automatic {
        ensure!(groups >= (args.max_context_tokens as u64).div_ceil(512)
            + 2 * (args.concurrency as u64 + args.prefix_cache_entries as u64),
            "automatic KV pool cannot fit one full-context request and private tails");
    }
    let bytes = usize::try_from(groups)?.checked_mul(GROUP_BYTES).context("planner KV byte overflow")?;
    tracing::info!(requested_pool_tokens=requested, admitted_pool_tokens=groups.saturating_sub(
        args.concurrency as u64 + 2 * args.prefix_cache_entries as u64) * 512,
        groups, global_bytes=bytes, ?available, "V4.1 planner source-pool admission");
    Ok(Some(ByteSize(bytes)))
}

pub(super) fn startup_phase(lib: &cuteafd_ffi::NativeLibrary, phase: &str, device_total: usize) -> Result<()> {
    if device_total <= 32usize << 30 {
        let (free, total) = lib.cuda_memory_info()?;
        let device = lib.cuda_get_device()?;
        let ledger = cuteafd_ffi::memory_ledger::snapshot();
        let space = cuteafd_ffi::memory_ledger::Space::Device;
        tracing::info!(phase, occupied_bytes=total-free, tracked_live_bytes=ledger.total(space, device),
            tracked_peak_bytes=ledger.peak.get(&(space, device)).copied().unwrap_or(0),
            "V4.1 small-card startup phase high-water ledger");
    }
    Ok(())
}

/// Small-card startup admission runs before any weight payload or device owner.
/// Loading-only owners are charged at their sequential phase, not retained in the KV ledger.
pub(super) fn admit_small_card_startup(lib: &cuteafd_ffi::NativeLibrary,
    catalog: &cuteafd_loader::OfficialV41Catalog, args: &crate::cli::NativeServeArgs) -> Result<()> {
    use super::*;
    let (_, total) = lib.cuda_memory_info()?;
    if total > 32usize << 30 { return Ok(()); }
    ensure!(args.rtx_expert_layers == LocalLayers::Count(0),
        "32 GB V4.1 startup admission requires --rtx-expert-layers 0; use a larger coordinator for local experts");
    let capacity = args.prefill_batch_tokens.max(256);
    let rows = capacity as usize;
    let head_rows = if args.dspark_draft_limit > 5 { 64 } else { 48 };
    let mut owners: Vec<(&str, usize)> = vec![
        ("backbone weights/load bound", BackboneLaneWeights::device_bytes(lib, catalog)?),
        ("cache producer weights", CacheProducerWeights::device_bytes(lib, catalog)?),
        ("index weights", IndexLaneWeights::device_bytes(lib, catalog)?),
        ("embedding", if embedding_placement(args.embedding_placement, total) == crate::shared::memory::EmbedPlacement::Gpu {
            NativeRtxTensors::plan(catalog, &["embed.weight".into()])? } else { 0 }),
        ("vocabulary resident", VocabularyHead::resident_bytes(catalog)?),
        ("head weights", TargetHeadWeights::device_bytes(catalog)?),
        ("engram weights", EngramLayerWeights::device_bytes(lib, catalog, 0)?
            + EngramLayerWeights::device_bytes(lib, catalog, 1)?),
        ("vision", if local_vision_owner(cuteafd_api::openai::vision_input_enabled(), args.vision_peers.is_some()) {
            crate::families::deepseek_v41::v41_vision::VisionRuntime::device_bytes(catalog, 9216)?
        } else { 0 }),
    ];
    let weight_bytes = owners.iter().take(4).try_fold(0usize, |n, (_, b)|
        n.checked_add(*b).context("startup weights overflow"))?;
    let engram_bytes = owners.iter().find(|(name, _)| *name == "engram weights").unwrap().1;
    let vocabulary_peak = VocabularyHead::plan(catalog)?;
    let lane = [TargetEmbeddingWave::device_bytes(rows)?,
        BackboneLane::workspace_bytes(lib, capacity)?.into_iter().sum(),
        IndexLane::workspace_bytes(lib, capacity)?.into_iter().sum(),
        BackboneExecution::workspace_bytes(lib, capacity)?, EngramDeviceRows::device_bytes(rows)?,
        EngramGate::device_bytes(lib, rows)? * 2, TargetHeadWave::device_bytes(head_rows)?,
        crate::families::deepseek_v41::v41_target_pass::TargetTapWave::device_bytes(rows)?]
        .into_iter().try_fold(0usize, |n, b| n.checked_add(b).context("target lane admission overflow"))?;
    let mut loading_peaks = vec![("vocabulary packing", weight_bytes.checked_add(engram_bytes)
        .and_then(|n| n.checked_add(lane))
        .and_then(|n| n.checked_add(vocabulary_peak)).context("vocabulary phase overflow")?)];
    owners.push(("target/prefill lanes", lane.checked_mul(2).context("lane admission overflow")?));
    if crate::families::deepseek_v41::v41_tensors::fp8_head() == crate::families::deepseek_v41::v41_tensors::Fp8Head::All {
        owners.push(("target vocabulary FP8 scratch", lib.fp8_w8a16_workspace(head_rows, 5120, 129280)?
            .max(256).checked_mul(2).context("target FP8 scratch overflow")?));
    }
    owners.push(("target/prefill transports", NativeTp4Wave::device_bytes_for(capacity, args.peers.len())?
        .checked_mul(2).context("transport admission overflow")?));
    if args.dspark {
        let tiers = catalog.exl3().map(|m| m.decoder_tiers()).unwrap_or(&[]);
        let directory = crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(&args.native_lib, tiers, "dspark");
        let (weights, runtime, staging) = crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::serving_bytes(
            lib, catalog, capacity, args.concurrency, if args.dspark_draft_limit > 5 { 7 } else { 5 }, Some(&directory))?;
        // Expert staging drains before auxiliary tensors and serving waves are allocated.
        let before_draft = owners.iter().filter(|(name, _)| *name != "vision")
            .try_fold(0usize, |n, (_, b)| n.checked_add(*b).context("pre-draft owners overflow"))?;
        loading_peaks.push(("dSpark loading", before_draft.checked_add(weights)
            .and_then(|n| n.checked_add(staging)).context("dSpark phase overflow")?));
        owners.extend([("dSpark weights", weights), ("dSpark lanes/windows", runtime)]);
        if args.dspark_draft_limit > 5 {
            let sparse = cuteafd_ffi::V41SparseAttention::split_scratch_bytes(64, 10)? / 64 * 64
                + cuteafd_ffi::V41SparseAttention::batch_descriptor_bytes(64)?;
            owners.push(("K7 sparse replacement peak", sparse * 2));
        }
    }
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    let snapshot_slots = if args.prefix_cache_entries == 0 { 0 } else { 2 * args.prefix_cache_entries as usize + 2 };
    let snapshots = snapshot_slots * (crate::families::deepseek_v41::v41_backbone_cache::BackbonePrefix::device_bytes().div_ceil(256) * 256
        + if args.dspark { 3 * 128 * 528 } else { 0 });
    owners.push(("snapshot arenas", snapshots));
    let fixed = owners.iter().try_fold(0usize, |n, (_, b)| n.checked_add(*b).context("startup fixed-owner overflow"))?;
    let peak = startup_peak(fixed, &loading_peaks)?;
    // Geometry queries initialize native modules; sample their real CUDA cost
    // now, before reading weights, rather than using a card/driver constant.
    let (free, total) = measured_pool_memory(lib)?;
    ensure!(peak <= free, "32 GB V4.1 startup refused before weights: peak {peak} bytes, free {free}, phases {loading_peaks:?}, owners {owners:?}; use --prefill-batch-tokens 256 or a larger coordinator");
    let after = free.checked_sub(fixed).with_context(|| format!("32 GB V4.1 fixed startup owners need {fixed} bytes, free {free}; set --prefill-batch-tokens 256, --rtx-expert-layers 0 or use a larger coordinator; owners {owners:?}"))?;
    let exact = planned_pool_size(args, &[(after, total)])?;
    let pool = PoolPlan::new(args.concurrency as usize, args.max_context_tokens as usize,
        args.prefix_cache_entries as usize, snapshots, exact, args.memory_reservation, after, total)
        .with_context(|| format!("32 GB V4.1 startup refused before weights: fixed-owner bound {fixed}, owners {owners:?}; use --prefill-batch-tokens 256 or increase coordinator memory"))?;
    tracing::info!(fixed_owner_bound_bytes=fixed, startup_peak_bound_bytes=peak, loading_phases=?loading_peaks, owners=?owners, admitted_cache_bytes=pool.cache_bytes,
        "V4.1 complete small-card startup admission before weight loads");
    Ok(())
}

// Remote vision reserves its tower on the Spark, never in the RTX startup bound.
fn local_vision_owner(enabled: bool, remote: bool) -> bool {
    enabled && !remote
}

fn startup_peak(fixed: usize, loading_phases: &[(&str, usize)]) -> Result<usize> {
    Ok(loading_phases.iter().map(|(_, bytes)| *bytes).max().unwrap_or(0).max(fixed))
}

#[cfg(test)]
mod startup_tests {
    #[test]
    fn startup_vision_bound_only_charges_enabled_rtx_fallback() {
        assert!(super::local_vision_owner(true, false));
        assert!(!super::local_vision_owner(true, true));
        assert!(!super::local_vision_owner(false, false));
        assert!(!super::local_vision_owner(false, true));
    }
    #[test]
    fn embedding_profile_preserves_explicit_overrides_and_pro_default() {
        use crate::shared::memory::EmbedPlacement::{Gpu, Host};
        for total in [31usize << 30, 32usize << 30, 96usize << 30] {
            assert_eq!(super::embedding_placement(Some(Gpu), total), Gpu);
            assert_eq!(super::embedding_placement(Some(Host), total), Host);
            assert_eq!(super::embedding_placement(None, total),
                if total <= 32usize << 30 { Host } else { Gpu });
        }
    }
    #[test]
    fn sequential_loading_peaks_do_not_shrink_the_serving_pool() {
        let fixed = 26usize << 30;
        let phases = [("head packing", 20usize << 30), ("draft staging", 27usize << 30)];
        assert_eq!(super::startup_peak(fixed, &phases).unwrap(), 27usize << 30);
        assert_eq!(super::startup_peak(fixed, &[]).unwrap(), fixed);
        // Retained residency, not sum of loading phases, is subtracted before KV.
        assert_eq!((32usize << 30) - fixed, 6usize << 30);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteSize(pub usize);
#[derive(Clone, Copy, Debug)]
pub(crate) enum HostBudget { Auto, Bytes(u64) }
impl FromStr for HostBudget {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value.trim() {
            "auto" => Ok(Self::Auto),
            "0" => Ok(Self::Bytes(0)),
            value => Ok(Self::Bytes(value.parse::<ByteSize>()?.0 as u64)),
        }
    }
}
impl HostBudget {
    pub fn explicit_bytes(self) -> u64 {
        match self { Self::Auto => 0, Self::Bytes(bytes) => bytes }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Reservation {
    Bytes(ByteSize),
    Percent(u64), // millionths of one percent
}
fn decimal(value: &str) -> Result<u64> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    ensure!(
        !whole.is_empty()
            && whole.bytes().all(|c| c.is_ascii_digit())
            && fraction.len() <= 6
            && fraction.bytes().all(|c| c.is_ascii_digit()),
        "expected a positive decimal with at most six fractional digits"
    );
    let whole: u64 = whole.parse()?;
    let fractional: u64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse()?
    };
    let scaled = whole
        .checked_mul(1_000_000)
        .and_then(|v| v.checked_add(fractional * 10u64.pow(6 - fraction.len() as u32)))
        .context("memory size overflow")?;
    ensure!(scaled > 0, "memory size must be positive");
    Ok(scaled)
}
impl FromStr for ByteSize {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        let end = value
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(value.len());
        let amount = decimal(&value[..end])?;
        let unit = value[end..].trim().to_ascii_lowercase();
        let multiplier: u128 = match unit.as_str() {
            "" | "b" => 1,
            "mb" => 1_000_000,
            "gb" => 1_000_000_000,
            "mib" => 1_048_576,
            "gib" => 1_073_741_824,
            _ => anyhow::bail!("memory size unit must be B, MB, GB, MiB or GiB"),
        };
        let bytes = usize::try_from(u128::from(amount) * multiplier / 1_000_000)?;
        ensure!(bytes > 0, "memory size rounds to zero bytes");
        Ok(Self(bytes))
    }
}
impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}
impl FromStr for Reservation {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if let Some(percent) = value.trim().strip_suffix('%') {
            let percent = decimal(percent.trim())?;
            ensure!(
                percent <= 100_000_000,
                "memory percentage must be in (0,100]"
            );
            Ok(Self::Percent(percent))
        } else {
            Ok(Self::Bytes(value.parse()?))
        }
    }
}
impl Reservation {
    fn bytes(self, total: usize) -> Result<usize> {
        let bytes = match self {
            Self::Bytes(size) => size.0,
            Self::Percent(percent) => (total as u128 * u128::from(percent) / 100_000_000) as usize,
        };
        ensure!(
            bytes > 0 && bytes <= total,
            "memory reservation exceeds device total or rounds to zero"
        );
        Ok(bytes)
    }
}

#[derive(Debug)]
pub(super) struct PoolPlan {
    pub pages: [usize; 4],
    pub global_bytes: usize,
    pub cache_bytes: usize,
    pub occupied_before: usize,
    pub reservation_bytes: usize,
    pub runtime_headroom_bytes: usize,
}
impl PoolPlan {
    fn from_groups(
        slots: usize,
        groups: usize,
        occupied_before: usize,
        reservation_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_GROUPS).contains(&groups),
            "source pool exceeds physical page capacity"
        );
        let pages = [groups, groups, groups, groups * 2];
        Ok(Self {
            pages,
            global_bytes: groups * GROUP_BYTES,
            cache_bytes: BackboneCache::device_bytes(slots, pages)?,
            occupied_before,
            reservation_bytes,
            runtime_headroom_bytes: RUNTIME_HEADROOM,
        })
    }
    pub fn new(
        slots: usize,
        context: usize,
        retained_turns: usize,
        _snapshot_bytes: usize,
        exact: Option<ByteSize>,
        reservation: Option<Reservation>,
        free: usize,
        total: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=1_048_576).contains(&context),
            "invalid pool context limit"
        );
        ensure!(
            free <= total && total > 0,
            "invalid device memory information"
        );
        ensure!((1..=16).contains(&slots), "invalid concurrency limit");
        let occupied = total - free;
        let ceiling = reservation
            .map(|r| r.bytes(total))
            .transpose()?
            .unwrap_or(total);
        let small_card = total <= 32usize << 30;
        let ceiling = if small_card { ceiling.min(cuteafd_core::serving_capacity::admission_ceiling(
            total as u64, 97, cuteafd_core::serving_capacity::small_card_headroom_bytes(total as u64))? as usize)
        } else { ceiling };
        // The absolute runtime floor is already charged in the ceiling.
        let runtime_headroom = if small_card { 0 } else { RUNTIME_HEADROOM + graph_reserve_bytes(total, None) };
        let available = ceiling.checked_sub(occupied).and_then(|v| v.checked_sub(runtime_headroom))
            .context("memory reservation leaves no space after existing allocations and runtime headroom")?;
        ensure!(retained_turns <= 128, "invalid retained-turn limit");
        // At most two retained frontiers per turn, plus one active tail per slot.
        let spare_groups = slots + 2 * retained_turns;
        let per_context = context.div_ceil(512);
        // Explicit budgets may trade aggregate context capacity for memory.
        // Provision at least one page per active owner plus private tail space.
        let minimum = if exact.is_none() && reservation.is_none() { per_context + 2 * (slots + retained_turns) } else { slots + spare_groups };
        let groups = if let Some(exact) = exact {
            ensure!(
                exact.0 / GROUP_BYTES <= MAX_GROUPS,
                "exact KV pool exceeds physical page capacity"
            );
            exact.0 / GROUP_BYTES
        } else if reservation.is_some() || small_card {
            // Tables saturate at the maximum logical per-request context. Find
            // the largest whole page group whose entire cache fits the budget.
            let (mut low, mut high) = (0, MAX_GROUPS);
            while low < high {
                let mid = (low + high).div_ceil(2);
                if Self::from_groups(slots, mid, occupied, ceiling)?.cache_bytes <= available {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            if reservation.is_some() { low } else { low.min((1_048_576usize / 512 + spare_groups).max(minimum)) }
        } else {
            // Snapshot arenas are already live and charged to fixed occupancy.
            // Explicit KV/total reservations retain their own sizing policy.
            (DEFAULT_POOL_TOKENS / 512 + spare_groups)
                .max(minimum)
        };
        ensure!(groups >= minimum,
            "KV pool needs at least {} global bytes for {slots} active owners plus copy-on-write headroom; increase the memory budget",
            minimum * GROUP_BYTES);
        let mut plan = Self::from_groups(slots, groups, occupied, ceiling)?;
        plan.runtime_headroom_bytes = runtime_headroom;
        ensure!(plan.cache_bytes <= available,
            "KV cache needs {} bytes but reservation leaves {available} after existing allocations and {} bytes of runtime headroom",
            plan.cache_bytes, runtime_headroom);
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planner_is_opt_in_and_explicit_pool_sizes_remain_strict() -> Result<()> {
        use clap::{Args, FromArgMatches};
        let parse = |extra: &[&str]| -> Result<crate::cli::NativeServeArgs> {
            let mut values = vec!["fixture", "--snapshot", "/missing", "--native-lib", "/missing",
                "--peers", "127.0.0.1:9000,127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003"];
            values.extend(extra);
            let matches = crate::cli::NativeServeArgs::augment_args(clap::Command::new("fixture"))
                .try_get_matches_from(values)?;
            Ok(crate::cli::NativeServeArgs::from_arg_matches(&matches)?)
        };
        let unchanged = parse(&[])?;
        assert!(unchanged.pool_tokens.is_none());
        assert!(planned_pool_size(&unchanged, &[])?.is_none());
        let exact = parse(&["--kv-pool-size", "1GiB"])?;
        assert_eq!(planned_pool_size(&exact, &[])?.unwrap().0, 1 << 30);
        assert!(parse(&["--kv-pool-size", "1GiB", "--pool-tokens", "0"]).is_err());
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("config.json"), include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json")))?;
        let mut auto = parse(&["--pool-tokens", "0"])?;
        auto.snapshot = directory.path().to_owned();
        let full = planned_pool_size(&auto, &[(96 << 30, 96 << 30)])?.unwrap();
        assert_eq!(full.0 / GROUP_BYTES, 2 * 2048 + 16 + 40);
        let small = planned_pool_size(&auto, &[(16 << 30, 32 << 30)])?.unwrap();
        assert_eq!(small.0 / GROUP_BYTES, 2048 + 2 * (16 + 20));
        auto.max_context_tokens = 32768;
        let small = planned_pool_size(&auto, &[(16 << 30, 32 << 30)])?.unwrap();
        assert_eq!(small.0 / GROUP_BYTES, 2048 + 16 + 40);
        auto.max_context_tokens = 1_048_576;
        let limited = planned_pool_size(&auto, &[((15usize << 29) + SINGLE_GRAPH_RESERVE, 96 << 30)])?.unwrap();
        assert!(limited.0 < full.0);
        auto.pool_tokens = Some(14 * 1_048_576);
        assert!(planned_pool_size(&auto, &[((15usize << 29) + SINGLE_GRAPH_RESERVE, 96 << 30)]).is_err());
        Ok(())
    }
    #[test]
    fn planner_cache_formula_matches_allocator_and_replica_owners() -> Result<()> {
        use crate::families::deepseek_v41::v41_backbone_cache::CachePlacement;
        let config: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json")))?;
        for slots in [1, 16] {
            for groups in [2 * slots + 40, 4152, 32768] {
                let pages = [groups, groups, groups, groups * 2];
                let single = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes(&config, 1,
                    slots as u64, groups as u64, false)?;
                assert_eq!(single, vec![BackboneCache::device_bytes(slots, pages)? as u64]);
                for replicated in [false, true] {
                    let actual = if replicated { BackboneCache::replicated_device_bytes(CachePlacement::encoder_decoder(), slots, pages)? }
                        else { BackboneCache::distributed_device_bytes(CachePlacement::encoder_decoder(), slots, pages)? };
                    let predicted = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes(&config, 2,
                        slots as u64, groups as u64, replicated)?;
                    assert_eq!(predicted, actual.map(|b| b as u64).to_vec());
                }
            }
        }
        Ok(())
    }
    #[test]
    fn sizes_and_reservations_validate_units_and_overflow() {
        for (value, bytes) in [
            ("37GB", 37_000_000_000),
            ("1.5GiB", 1_610_612_736),
            ("1MB", 1_000_000),
            ("1MiB", 1_048_576),
            ("123", 123),
        ] {
            assert_eq!(value.parse::<ByteSize>().unwrap().0, bytes);
        }
        for value in [
            "0",
            "-1GB",
            "NaN",
            "1TB",
            "10%",
            "1.0000001GB",
            "18446744073709551616GB",
        ] {
            assert!(value.parse::<ByteSize>().is_err(), "{value}");
        }
        assert_eq!(
            "87.5%"
                .parse::<Reservation>()
                .unwrap()
                .bytes(8_000_000_000)
                .unwrap(),
            7_000_000_000
        );
        for value in ["0%", "100.1%", "inf%"] {
            assert!(value.parse::<Reservation>().is_err());
        }
    }
    #[test]
    fn default_pool_covers_two_million_tokens_and_snapshot_tails() {
        let p = PoolPlan::new(16, 1_048_576, 24, 0, None, None, 96 << 30, 96 << 30).unwrap();
        assert_eq!(p.pages, [4160, 4160, 4160, 8320]);
        assert_eq!(p.global_bytes, 4160 * GROUP_BYTES);
        assert!(p.cache_bytes > p.global_bytes);
        let small = PoolPlan::new(16, 32768, 24, 0, None, None, 8 << 30, 96 << 30).unwrap();
        assert_eq!(small.pages, [4160, 4160, 4160, 8320]);
        assert!(PoolPlan::new(16, 1_048_576, 24, 0, None, None, 32 << 30, 96 << 30).is_ok());
        assert!(PoolPlan::new(16, 1_048_576, 24, 0, None, None, 6 << 30, 96 << 30).is_err());
    }
    #[test]
    fn small_card_without_reservation_sizes_from_free_memory() {
        let total = 32usize << 30;
        let free = 5usize << 30;
        let plan = PoolPlan::new(16, 1_048_576, 24, 0, None, None, free, total).unwrap();
        let floor = cuteafd_core::serving_capacity::SMALL_CARD_HEADROOM_BYTES as usize;
        assert!(plan.cache_bytes + floor <= free);
        assert_eq!(plan.pages[0], 2048 + 2 * (16 + 24));
        let roomy = PoolPlan::new(16, 32768, 24, 0, None, None, 20 << 30, total).unwrap();
        assert_eq!(roomy.pages[0], 2048 + 16 + 2 * 24);
        let reserved = PoolPlan::new(16, 1_048_576, 24, 0, None, Some("97%".parse().unwrap()), free, total).unwrap();
        assert!(free - reserved.cache_bytes >= floor);
        let explicit = PoolPlan::new(16, 1_048_576, 24, 0, None,
            Some(Reservation::Bytes(ByteSize(total))), free, total).unwrap();
        assert!(free - explicit.cache_bytes >= floor);
        let overflow = explicit.global_bytes + GROUP_BYTES;
        assert!(PoolPlan::new(16, 1_048_576, 24, 0, Some(ByteSize(overflow)),
            Some(Reservation::Bytes(ByteSize(total))), free, total).is_err());
        assert!(plan.pages[0] >= 2048 + 64);
        assert!(PoolPlan::new(16, 1_048_576, 24, 0, Some(ByteSize(16 << 30)), None, free, total).is_err());
    }

    #[test]
    fn snapshot_arenas_are_fixed_occupancy_without_changing_pool_tokens() {
        let tail = crate::families::deepseek_v41::v41_backbone_cache::BackbonePrefix::device_bytes().div_ceil(256) * 256;
        let bytes = 50 * (tail + 3 * cuteafd_ffi::V41DsparkCache::SLOT_BYTES);
        let free = 56 << 30;
        let total = 96 << 30;
        let old = PoolPlan::new(16, 1_048_576, 24, 0, None, None, free, total).unwrap();
        let pooled = PoolPlan::new(16, 1_048_576, 24, bytes, None, None, free-bytes, total).unwrap();
        assert_eq!(old.pages, pooled.pages);
        let explicit = PoolPlan::new(16, 1_048_576, 24, bytes,
            Some(ByteSize(old.global_bytes)), None, free-bytes, total).unwrap();
        assert_eq!(explicit.pages, old.pages);
        let short = PoolPlan::new(16, 128, 24, bytes, None, None, free-bytes, total).unwrap();
        assert!(short.pages[0] >= 2*16 + 2*24);
    }
    #[test]
    fn exact_and_total_budgets_round_down_without_undercutting_admission() {
        let total = 96 << 30;
        let free = 56 << 30;
        let default = PoolPlan::new(16, 1_048_576, 24, 0, None, None, free, total).unwrap();
        let exact = PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            Some(ByteSize(default.global_bytes + 99)),
            None,
            free,
            total,
        )
        .unwrap();
        assert_eq!(exact.pages, default.pages);
        let small =
            PoolPlan::new(2, 1_048_576, 24, 0, Some(ByteSize(1 << 30)), None, free, total).unwrap();
        assert!(small.global_bytes <= 1 << 30);
        let c2 = PoolPlan::new(2, 1_048_576, 24, 0, None, None, free, total).unwrap();
        assert_eq!(c2.pages[0], 4096 + 50);
        let reservation = Some("80GiB".parse().unwrap());
        let p = PoolPlan::new(16, 1_048_576, 24, 0, None, reservation, free, total).unwrap();
        assert!(p.cache_bytes + p.occupied_before + p.runtime_headroom_bytes <= 80 << 30);
        let next =
            PoolPlan::from_groups(16, p.pages[0] + 1, p.occupied_before, p.reservation_bytes)
                .unwrap();
        assert!(next.cache_bytes + p.occupied_before + p.runtime_headroom_bytes > 80 << 30);
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            Some(ByteSize(1 << 20)),
            None,
            free,
            total
        )
        .is_err());
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            Some(ByteSize(default.global_bytes)),
            Some("41GiB".parse().unwrap()),
            free,
            total
        )
        .is_err());
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            None,
            Some("101GiB".parse().unwrap()),
            free,
            total
        )
        .is_err());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalLayers { Auto, Count(usize) }
impl FromStr for LocalLayers {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if value == "auto" { return Ok(Self::Auto); }
        let count: usize = value.parse().context("RTX expert layers must be auto or 0..40")?;
        ensure!(count <= 40, "RTX expert layers must be auto or 0..40");
        Ok(Self::Count(count))
    }
}
#[derive(Debug)]
pub(super) struct LocalLayerPlan {
    pub layers: usize,
    pub resident_bytes: usize,
    pub workspace_bytes: usize,
    pub peak_bytes: usize,
    pub graph_reserve_bytes: usize,
}
impl LocalLayerPlan {
    /// Free memory is sampled after mandatory weights, KV and both lanes exist.
    /// Reserve both local workspaces and bounded load staging conservatively.
    pub fn new(requested: LocalLayers, budgets: &[crate::families::deepseek_v41::v41_experts::ExpertLoadBudget],
        workspace_bytes: usize, free: usize, total: usize, ceiling: usize) -> Result<Self> {
        ensure!(free <= total && ceiling <= total && budgets.len() <= 40, "invalid local memory inventory");
        let graph_reserve_bytes = graph_reserve_bytes(total, None);
        let available = ceiling.saturating_sub(total - free).saturating_sub(RUNTIME_HEADROOM)
            .saturating_sub(graph_reserve_bytes);
        let target = match requested { LocalLayers::Auto => budgets.len(), LocalLayers::Count(n) => n };
        ensure!(target <= budgets.len(), "requested RTX layers exceed available layer plans");
        let mut plan = Self { layers: 0, resident_bytes: 0, workspace_bytes: 0, peak_bytes: 0, graph_reserve_bytes };
        let mut staging = 0;
        for budget in budgets.iter().take(target) {
            let resident = plan.resident_bytes.checked_add(budget.resident_bytes).context("local weight size overflow")?;
            staging = staging.max(budget.device_staging_bytes);
            let peak = resident.checked_add(workspace_bytes).and_then(|b| b.checked_add(staging))
                .context("local layer peak size overflow")?;
            if peak > available { break; }
            plan = Self { layers: plan.layers + 1, resident_bytes: resident, workspace_bytes, peak_bytes: peak, graph_reserve_bytes };
        }
        if let LocalLayers::Count(n) = requested {
            ensure!(plan.layers == n, "requested {n} RTX expert layers but only {} fit after KV, workspaces, staging, runtime headroom and graph reserve", plan.layers);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod local_tests {
    use super::*;
    fn budgets() -> Vec<crate::families::deepseek_v41::v41_experts::ExpertLoadBudget> {
        vec![crate::families::deepseek_v41::v41_experts::ExpertLoadBudget { resident_bytes: 7 << 30,
            device_staging_bytes: 20 << 20, pinned_host_bytes: 0, read_scratch_bytes: 0 }; 40]
    }
    #[test]
    fn lazy_graph_reserve_is_charged_before_expert_admission() {
        let b = budgets();
        let free = (35usize << 30) + (600 << 20) + (20 << 20) + RUNTIME_HEADROOM;
        let plan = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20,
            free, 96 << 30, 96 << 30).unwrap();
        assert_eq!(plan.layers, 4);
        assert_eq!(plan.graph_reserve_bytes, SINGLE_GRAPH_RESERVE);
        assert!(LocalLayerPlan::new(LocalLayers::Count(5), &b, 600 << 20,
            free, 96 << 30, 96 << 30).is_err());
        for role in [None, Some(0), Some(1)] {
            assert_eq!(graph_reserve_bytes(32 << 30, role), 0);
        }
        assert_eq!(graph_reserve_bytes(96 << 30, Some(0)), DUAL_GPU0_GRAPH_RESERVE);
        assert_eq!(graph_reserve_bytes(96 << 30, Some(1)), DUAL_GPU1_GRAPH_RESERVE);
    }

    #[test]
    fn dual_graph_reserves_cover_measured_tools_growth() {
        // parity-v2c RTX2 s1, qualification plus separate stress, 2026-10-08.
        let growth = [1_184_529_440usize, 1_220_203_808];
        let quantum = 64usize << 20;
        for gpu in 0..2 {
            let measured_reserve = (growth[gpu] * 5).div_ceil(4).div_ceil(quantum) * quantum;
            assert_eq!(graph_reserve_bytes(96 << 30, Some(gpu)), measured_reserve);
        }
    }

    #[test]
    fn local_prefix_respects_workspace_staging_and_ceiling() {
        let b = budgets();
        let p = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20, (38 << 30) + SINGLE_GRAPH_RESERVE, 96 << 30, 96 << 30).unwrap();
        assert_eq!(p.layers, 5);
        assert_eq!(p.resident_bytes, 35 << 30);
        let limited = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20, (38 << 30) + SINGLE_GRAPH_RESERVE, 96 << 30, 90 << 30).unwrap();
        assert_eq!(limited.layers, 4);
        assert!(LocalLayerPlan::new(LocalLayers::Count(5), &b, 600 << 20, (38 << 30) + SINGLE_GRAPH_RESERVE, 96 << 30, 90 << 30).is_err());
        let exact = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20,
            p.peak_bytes + RUNTIME_HEADROOM + SINGLE_GRAPH_RESERVE, 96 << 30, 96 << 30).unwrap();
        assert_eq!(exact.layers, 5);
        let short = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20,
            p.peak_bytes + RUNTIME_HEADROOM + SINGLE_GRAPH_RESERVE - 1, 96 << 30, 96 << 30).unwrap();
        assert_eq!(short.layers, 4);
        let zero = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20, 1 << 30, 96 << 30, 96 << 30).unwrap();
        assert_eq!((zero.layers, zero.peak_bytes), (0, 0));
        assert!("41".parse::<LocalLayers>().is_err());
        assert!("-1".parse::<LocalLayers>().is_err());
        assert_eq!("auto".parse::<LocalLayers>().unwrap(), LocalLayers::Auto);
    }
}
