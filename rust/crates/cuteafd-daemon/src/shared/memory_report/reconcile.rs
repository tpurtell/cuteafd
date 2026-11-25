//! Signed observer-excluded use versus admission-owner reconciliation. Samples are
//! diagnostic: never turn a missing sample or a positive delta into refusal.
use super::*;
use std::sync::{Mutex, OnceLock};

fn records() -> &'static Mutex<Value> {
    static RECORDS: OnceLock<Mutex<Value>> = OnceLock::new();
    RECORDS.get_or_init(|| Mutex::new(json!({"schema_version": 1, "ready": null, "first_request": null})))
}
fn workers() -> &'static Mutex<BTreeMap<u32, Value>> {
    static WORKERS: OnceLock<Mutex<BTreeMap<u32, Value>>> = OnceLock::new();
    WORKERS.get_or_init(Default::default)
}

fn spark_owners(report: &cuteafd_core::ModelAdmissionResponse)
    -> cuteafd_core::memory_policy::DeviceBreakdown {
    // These values passed SparkAssignment::validate. Device/pinned detail rows
    // describe these same owners: only extra host backings are additive on UMA.
    let workspace = report.runtime.workspace_bytes
        .saturating_add(report.runtime.pinned_workspace_bytes)
        .saturating_add(report.mapped_ring_bytes)
        .saturating_add(report.host_exchange_bytes)
        .saturating_add(report.row_indices_bytes)
        .saturating_add(report.runtime.detail.allocations.iter()
            .filter(|owner| owner.space == "host").map(|owner| owner.bytes).sum::<u64>());
    cuteafd_core::memory_policy::DeviceBreakdown { device: format!("spark{}", report.rank),
        cap_bytes: report.runtime.sample.total_bytes, weights_bytes: report.resident_bytes,
        workspace_bytes: workspace, reserve: vec![], reserve_overridden: false,
        kv_bytes: 0, other_bytes: 0 }
}

pub(crate) fn spark_ready(report: &cuteafd_core::ModelAdmissionResponse) {
    workers().lock().unwrap_or_else(|e| e.into_inner()).insert(report.rank,
        json!({"admission_sample": report.runtime.sample, "owners": spark_owners(report),
            "owner_ledger": report,
            "physical_sample": report.runtime.detail.memory_reconciliation}));
}

pub(crate) fn snapshot() -> Value {
    records().lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn row(device: &str, measured: Option<i128>, owners: &cuteafd_core::memory_policy::DeviceBreakdown,
    graph_arena: Option<u64>, provenance: Value, reason: Option<&str>) -> Value {
    row_with_arena(device, measured, owners, graph_arena.map(|arena| (arena, arena)),
        provenance, reason)
}

/// `graph_arena` is `(current level, high water)`. Reconciliation uses the
/// current level: `charged`'s high-water keeps covering transient capture peaks
/// for admission, but charging one forever would misreport a peak that already
/// reversed. The level is refreshed from the stage's settled sample and bounded
/// by the high-water, so it can only release peak credit. Both are published.
fn row_with_arena(device: &str, measured: Option<i128>,
    owners: &cuteafd_core::memory_policy::DeviceBreakdown,
    graph_arena: Option<(u64, u64)>, provenance: Value, reason: Option<&str>) -> Value {
    let padding: u64 = owners.reserve.iter().filter(|owner| owner.name == "release padding")
        .map(|owner| owner.bytes).sum();
    let charged = owners.weights_bytes.saturating_add(owners.workspace_bytes)
        .saturating_add(owners.kv_bytes).saturating_add(owners.other_bytes)
        .saturating_add(owners.reserve_bytes()).saturating_sub(padding);
    let graph_reserve: u64 = owners.reserve.iter().filter(|owner| owner.name == "graph reserve")
        .map(|owner| owner.bytes).sum();
    let delta_vs_admitted = measured.map(|used| used - i128::from(charged));
    // No graph owner needs no replacement. A missing live controller sample is
    // unknown, not zero; over-reserve arena growth produces negative slack.
    let graph_arena = graph_arena.or((graph_reserve == 0).then_some((0, 0)));
    let graph_level = graph_arena.map(|(level, _)| level);
    let graph_high_water = graph_arena.map(|(_, high_water)| high_water);
    let slack = graph_level.map(|arena| i128::from(graph_reserve) - i128::from(arena));
    let delta_vs_measured = delta_vs_admitted.zip(slack).map(|(delta, slack)| delta + slack);
    json!({"device": device, "measured_bytes": measured, "charged_bytes": charged,
        "padding_bytes": padding, "delta_vs_admitted": delta_vs_admitted,
        "delta_vs_measured": delta_vs_measured, "graph_reserve_slack": slack,
        "graph_reserve_bytes": graph_reserve, "graph_arena_bytes": graph_level,
        "graph_arena_high_water_bytes": graph_high_water,
        "beyond_half_padding": delta_vs_measured.map(|delta| delta.abs() * 2 > i128::from(padding)),
        "drift_flag_eligible": true, "drift_flag_unavailable_reason": null,
        "graph_measurement_unavailable_reason": graph_arena.is_none().then_some("no measured graph arena high-water for admitted graph reserve"),
        "measurement_method": "raw CUDA physical total minus free; device-wide, not process attribution",
        "charge_basis": "admitted named owners excluding release padding", "owners": owners,
        "unavailable_reason": reason, "provenance": provenance})
}

fn observed_row(device: &str, sample: Option<(usize, usize, usize)>,
    owners: &cuteafd_core::memory_policy::DeviceBreakdown, graph_arena: Option<(u64, u64)>,
    provenance: Value, reason: Option<&str>) -> Value {
    let raw = sample.map(|(free, total, _)| total.saturating_sub(free) as u64);
    let observer = sample.map(|(_, _, observer)| observer as u64);
    let measured = raw.zip(observer).and_then(|(raw, observer)| raw.checked_sub(observer));
    let mut record = row_with_arena(device, measured.map(i128::from), owners, graph_arena,
        provenance, reason);
    record["raw_used_bytes"] = json!(raw);
    record["observer_bytes"] = json!(observer);
    record["observer_provenance"] = json!({
        "source": "CUTEAFD_OBSERVER_CONTEXT_FILE; measured sampler context per logical CUDA device",
        "method": "same observer_context::exclude helper as cuda_allocatable_memory_info; excluded exactly once",
        "present": observer.map(|bytes| bytes > 0)});
    record["measurement_method"] = "active-context CUDA physical total minus free, excluding independently measured observer context".into();
    record
}

fn spark_row(id: &str, stage: &str, worker: Option<&Value>,
    fallback: &cuteafd_core::memory_policy::DeviceBreakdown, provenance: &Value) -> Value {
    let admission = worker.and_then(|value| value.get("admission_sample"));
    let baseline = worker.and_then(|value|
        value["physical_sample"]["baseline_allocatable_free_bytes"].as_u64());
    let ready_free = admission.and_then(|sample| sample["free_bytes"].as_u64());
    let measured = (stage == "ready").then(|| baseline.zip(ready_free)
        .map(|(baseline, ready)| i128::from(baseline) - i128::from(ready))).flatten();
    let absolute = (stage == "ready").then(|| admission.and_then(|sample|
        sample["total_bytes"].as_u64().zip(sample["free_bytes"].as_u64())
            .and_then(|(total, free)| total.checked_sub(free)))).flatten();
    let raw = (stage == "ready").then(|| worker.and_then(|value|
        value["physical_sample"]["cuda_used_bytes"].as_u64())).flatten();
    let reason = if stage != "ready" { Some("post-request Spark control-plane telemetry unavailable; see #1631; worker logs its own samples") }
        else if baseline.is_none() { Some("worker omitted pre-weight allocatable baseline; no absolute-occupancy fallback") }
        else if ready_free.is_none() { Some("worker omitted allocatable admission sample") } else { None };
    let mut owners: Option<cuteafd_core::memory_policy::DeviceBreakdown> = worker
        .and_then(|value| value.get("owners"))
        .and_then(|value| serde_json::from_value(value.clone()).ok());
    let future_rings = worker.filter(|value|
        value["physical_sample"]["endpoint_rings_allocated"] == false)
        .and_then(|value| value["owner_ledger"]["mapped_ring_bytes"].as_u64()).unwrap_or(0);
    if let Some(owners) = owners.as_mut() {
        owners.workspace_bytes = owners.workspace_bytes.saturating_sub(future_rings);
    }
    let mut record = row(id, measured, owners.as_ref().unwrap_or(fallback), None,
        json!({"coordinator": provenance,
            "worker": worker.and_then(|value| value.get("physical_sample")),
            "owner_ledger": worker.and_then(|value| value.get("owner_ledger"))}), reason);
    record["measurement_method"] = "launch-relative pre-weight allocatable free minus retained-owner allocatable free; GB10 MemAvailable/cgroup-bounded".into();
    record["absolute_allocatable_used_bytes"] = json!(absolute);
    record["baseline_allocatable_free_bytes"] = json!(baseline);
    record["admitted_future_ring_bytes"] = json!(future_rings);
    record["raw_used_bytes"] = json!(raw);
    record["raw_measurement_method"] = "global GB10 CUDA occupancy including page cache; diagnostic only, never a drift-flag input; see #1568".into();
    record["drift_measurement_basis"] = "launch-relative allocatable consumption is diagnostic, not drift-flagged until #1568 resolves ownership and system-witness residuals".into();
    record["drift_flag_eligible"] = false.into();
    record["drift_flag_unavailable_reason"] = "Spark launch-relative ownership/system-witness residuals unresolved; see #1568".into();
    record["beyond_half_padding"] = Value::Null;
    record["charge_basis"] = "post-baseline resident/workspace and UMA host owners; no media baseline, no detail aliases, no not-yet-allocated endpoint rings".into();
    if owners.is_none() {
        for field in ["charged_bytes", "delta_vs_admitted", "delta_vs_measured", "beyond_half_padding"] {
            record[field] = Value::Null;
        }
        record["owner_unavailable_reason"] = "no reconciled worker owner ledger".into();
    }
    // Retained for provenance only, never labelled as a post-request sample.
    record["admission_sample"] = admission.cloned().unwrap_or(Value::Null);
    record
}

/// The per-rank device inventory admission registered, or `None` when the
/// family publishes no `memory_reconciliation_provenance`. A family without
/// it keeps both `/v1/stats` rows null instead of
/// reporting an empty or partial reconciliation.
fn admitted_devices(inputs: &Value) -> Option<&Vec<Value>> {
    inputs.get("memory_reconciliation_provenance")?.get("devices")?.as_array()
}

/// Ready runs on a helper thread; first-request runs only once, after completed
/// request retirement. Neither path creates a primary context or synchronizes a
/// partial multi-rank submission.
///
/// Both stages are per-process samples, so each has exactly one call site: the
/// `Once` lives there, not here, and a second site would re-sample a settled
/// device. `ready` belongs to the admission module; `first_request` is the
/// shared Qwen scheduler's post-retire hook (GLM 5, which drives its own
/// scheduler loop, guards its own call).
pub(crate) fn capture(stage: &str) {
    let inputs = memory_ledger::fault::inputs();
    if admitted_devices(&inputs).is_none() { return; }
    let worker_samples = workers().lock().unwrap_or_else(|e| e.into_inner()).clone();
    let graph_samples = crate::shared::decode_graph::device::snapshots();
    let Some(value) = capture_with(stage, &inputs, &worker_samples,
        &memory_ledger::fault::device_memory_observer,
        &|device, sample| sample.and_then(|(free, total, _)| {
            crate::shared::decode_graph::device::refreshed_arena(device, free as u64, total as u64)
        }).or_else(|| graph_samples.iter().find(|sample| sample.device == device)
            .map(|sample| (sample.level.unwrap_or(sample.charged), sample.charged))))
        else { return };
    tracing::info!(target: "cuteafd::memory", record = %value, "memory admission reconciliation");
    records().lock().unwrap_or_else(|e| e.into_inner())[stage] = value;
}

fn capture_with(stage: &str, inputs: &Value, worker_samples: &BTreeMap<u32, Value>,
    sample_device: &impl Fn(i32) -> anyhow::Result<Option<(usize, usize, usize)>>,
    graph_charge: &impl Fn(i32, Option<(usize, usize, usize)>) -> Option<(u64, u64)>) -> Option<Value> {
    let devices = admitted_devices(inputs)?;
    let provenance = &inputs["memory_reconciliation_provenance"];
    let mut rows = Vec::new();
    for (key, value) in inputs.as_object().into_iter().flatten().filter(|(key, _)| key.starts_with("device.")) {
        let Ok(owners) = serde_json::from_value::<cuteafd_core::memory_policy::DeviceBreakdown>(value.clone()) else { continue };
        let id = key.trim_start_matches("device.");
        if let Some(index) = id.strip_prefix("rtx").and_then(|n| n.parse::<usize>().ok()) {
            let physical = devices.get(index).and_then(|value| value["device"].as_i64());
            let sample = physical.and_then(|device| i32::try_from(device).ok())
                .map(sample_device).transpose();
            let sample_error = sample.as_ref().err().map(|error| format!("observer-aware CUDA sample unavailable: {error:#}"));
            let sample = sample.ok().flatten().flatten();
            let mut identity = provenance.clone();
            identity["device_sample"] = devices.get(index).cloned().unwrap_or(Value::Null);
            let graph_arena = physical.and_then(|device| i32::try_from(device).ok())
                .and_then(|device| graph_charge(device, sample));
            rows.push(observed_row(id, sample, &owners, graph_arena, identity,
                sample_error.as_deref().or_else(|| sample.is_none().then_some("no active CUDA primary context sample"))));
        } else if let Some(rank) = id.strip_prefix("spark").and_then(|n| n.trim_start_matches('.').parse::<u32>().ok()) {
            let mut record = spark_row(id, stage, worker_samples.get(&rank), &owners, provenance);
            record["rank"] = rank.into();
            rows.push(record);
        }
    }
    Some(json!({"stage": stage, "devices": rows}))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// No registered device identity means there is no physical sample to take.
    #[test]
    fn a_family_that_publishes_no_provenance_has_no_sample() {
        assert!(admitted_devices(&json!({"device.rtx0": {"device": "rtx0"}})).is_none());
        assert!(admitted_devices(&json!({"memory_reconciliation_provenance": {"family": "step5"}})).is_none());
        assert_eq!(admitted_devices(&json!({"memory_reconciliation_provenance": {"devices": [{"device": 0}]}}))
            .map(|devices| devices.len()), Some(1));
    }

    #[test]
    fn muse_ready_and_first_request_resolve_physical_device_samples() {
        let device = cuteafd_core::RuntimeSample { device: 3, total_bytes: 1000,
            free_bytes: 250, context_bytes: 100, module_bytes: 0, sms: 96,
            arch: "sm_120".into(), driver: "test".into() };
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "rtx0".into(),
            cap_bytes: 1000, weights_bytes: 600, workspace_bytes: 100, kv_bytes: 0,
            other_bytes: 0, reserve_overridden: false,
            reserve: vec![cuteafd_core::memory_policy::ReserveComponent {
                name: "graph reserve".into(), bytes: 100 }] };
        let mut inputs = json!({"memory_reconciliation_provenance": {
            "family": "muse_spark", "devices": [device]}, "device.rtx0": owners});
        for stage in ["ready", "first_request"] {
            let result = capture_with(stage, &inputs, &BTreeMap::new(), &|physical| {
                assert_eq!(physical, 3);
                Ok(Some((150, 1000, 100)))
            }, &|physical, sample| { assert_eq!(physical, 3); assert!(sample.is_some()); Some((50, 50)) }).unwrap();
            assert_eq!(result["stage"], stage);
            let row = &result["devices"][0];
            assert_eq!(row["measured_bytes"], 750);
            assert_eq!(row["graph_arena_high_water_bytes"], 50);
            assert_eq!(row["graph_reserve_slack"], 50);
            assert_eq!(row["delta_vs_measured"], 0);
            assert_eq!(row["beyond_half_padding"], false);
            assert!(row["unavailable_reason"].is_null());
        }
        // The old bare integer inventory must not fabricate a physical sample.
        inputs["memory_reconciliation_provenance"]["devices"] = json!([3]);
        let missing = capture_with("ready", &inputs, &BTreeMap::new(),
            &|_| panic!("malformed identity cannot sample CUDA"), &|_, _| None).unwrap();
        assert!(missing["devices"][0]["measured_bytes"].is_null());
        assert!(missing["devices"][0]["delta_vs_measured"].is_null());
    }

    #[test]
    fn spark_launch_relative_measurement_cancels_preexisting_occupancy_and_preserves_nulls() {
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "spark0".into(),
            cap_bytes: 100_000, weights_bytes: 600, workspace_bytes: 150, kv_bytes: 0,
            other_bytes: 0, reserve_overridden: false, reserve: vec![] };
        // 80,000 bytes of unrelated occupancy predate this launch. Only 700
        // newly consumed bytes match retained owners; 50 ring bytes come later.
        let mut worker = json!({"owners": owners, "owner_ledger": {"mapped_ring_bytes": 50},
            "admission_sample": {"total_bytes": 100_000, "free_bytes": 19_300},
            "physical_sample": {"cuda_used_bytes": 95_000,
                "baseline_allocatable_free_bytes": 20_000, "endpoint_rings_allocated": false}});
        let sample = spark_row("spark0", "ready", Some(&worker), &owners, &json!({}));
        assert_eq!(sample["raw_used_bytes"], 95_000);
        assert_eq!(sample["absolute_allocatable_used_bytes"], 80_700);
        assert_eq!(sample["measured_bytes"], 700);
        assert_eq!(sample["charged_bytes"], 700);
        assert_eq!(sample["admitted_future_ring_bytes"], 50);
        assert_eq!(sample["delta_vs_measured"], 0);
        assert!(sample["beyond_half_padding"].is_null());
        assert_eq!(sample["drift_flag_eligible"], false);
        assert!(sample["drift_flag_unavailable_reason"].as_str().unwrap().contains("#1568"));
        worker["admission_sample"]["free_bytes"] = json!(18_000);
        let unresolved = spark_row("spark0", "ready", Some(&worker), &owners, &json!({}));
        assert_eq!(unresolved["delta_vs_measured"], 1_300);
        assert!(unresolved["beyond_half_padding"].is_null());
        assert_eq!(unresolved["drift_flag_eligible"], false);
        worker["admission_sample"]["free_bytes"] = json!(20_100);
        let reclaimed = spark_row("spark0", "ready", Some(&worker), &owners, &json!({}));
        assert_eq!(reclaimed["measured_bytes"], -100);
        worker["physical_sample"]["baseline_allocatable_free_bytes"] = Value::Null;
        let missing = spark_row("spark0", "ready", Some(&worker), &owners, &json!({}));
        assert!(missing["measured_bytes"].is_null());
        assert!(missing["delta_vs_measured"].is_null());
        assert!(missing["beyond_half_padding"].is_null());
        assert!(missing["unavailable_reason"].as_str().unwrap().contains("pre-weight"));
        let post = spark_row("spark0", "first_request", Some(&worker), &owners, &json!({}));
        assert!(post["raw_used_bytes"].is_null());
        assert!(post["absolute_allocatable_used_bytes"].is_null());
        assert!(post["delta_vs_measured"].is_null());
        assert!(post["unavailable_reason"].as_str().unwrap().contains("#1631"));
        let no_ledger = spark_row("spark0", "ready", None, &owners, &json!({}));
        assert!(no_ledger["charged_bytes"].is_null());
    }

    #[test]
    fn observer_context_is_excluded_once_from_both_deltas_and_raw_sample_is_retained() {
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "rtx0".into(), cap_bytes: 900,
            weights_bytes: 600, workspace_bytes: 100, kv_bytes: 0, other_bytes: 0,
            reserve_overridden: false, reserve: vec![cuteafd_core::memory_policy::ReserveComponent {
                name: "graph reserve".into(), bytes: 100 }] };
        let sample = observed_row("rtx0", Some((150, 1000, 100)), &owners, Some((50, 50)), json!({}), None);
        assert_eq!(sample["raw_used_bytes"], 850);
        assert_eq!(sample["observer_bytes"], 100);
        assert_eq!(sample["measured_bytes"], 750);
        assert_eq!(sample["delta_vs_admitted"], -50);
        assert_eq!(sample["delta_vs_measured"], 0);
        assert_eq!(sample["observer_provenance"]["present"], true);
        let production = observed_row("rtx0", Some((250, 1000, 0)), &owners, Some((50, 50)), json!({}), None);
        assert_eq!(production["measured_bytes"], sample["measured_bytes"]);
        assert_eq!(production["delta_vs_measured"], 0);
        let missing = observed_row("rtx0", None, &owners, Some((50, 50)), json!({}), Some("no sample"));
        assert!(missing["measured_bytes"].is_null());
        assert!(missing["observer_bytes"].is_null());
        assert!(missing["delta_vs_measured"].is_null());
    }

    #[test]
    fn graph_slack_uses_the_current_level_and_still_publishes_the_high_water() {
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "rtx0".into(), cap_bytes: 900,
            weights_bytes: 600, workspace_bytes: 100, kv_bytes: 0, other_bytes: 0,
            reserve_overridden: false, reserve: vec![cuteafd_core::memory_policy::ReserveComponent {
                name: "graph reserve".into(), bytes: 100 }] };
        let sample = row_with_arena("rtx0", Some(100), &owners, Some((50, 80)), json!({}), None);
        assert_eq!(sample["graph_arena_bytes"], 50);
        assert_eq!(sample["graph_arena_high_water_bytes"], 80);
        assert_eq!(sample["graph_reserve_slack"], 50);
        assert_eq!(sample["delta_vs_admitted"], -700);
        assert_eq!(sample["delta_vs_measured"], -650);
        // A controller that never sampled reports the high-water as both.
        let fallback = row("rtx0", Some(100), &owners, Some(80), json!({}), None);
        assert_eq!(fallback["graph_arena_bytes"], 80);
        assert_eq!(fallback["graph_reserve_slack"], 20);
    }

    #[test]
    fn spark_reconciled_owners_include_uma_backings_without_detail_aliases() {
        use cuteafd_core::{ModelAdmissionResponse, RuntimeSample, SparkRuntimeInventory};
        use cuteafd_core::expert_geometry::{SparkAllocationTerm, SparkRuntimeDetail};
        let term = |space: &str, bytes| SparkAllocationTerm { space: space.into(), scope: "test".into(),
            tensor: None, format: None, bytes, allocations: 1, method: "fixture".into() };
        let report = ModelAdmissionResponse { rank: 2, resident_bytes: 100, slices: vec![],
            runtime: SparkRuntimeInventory { sample: RuntimeSample { device: 0,
                total_bytes: 1000, free_bytes: 500, context_bytes: 123, module_bytes: 0,
                sms: 48, arch: "sm_121".into(), driver: "test".into() },
                workspace_bytes: 20, pinned_workspace_bytes: 3,
                detail: SparkRuntimeDetail { allocations: vec![term("device", 120), term("pinned", 3),
                    term("host", 7)], ..Default::default() } },
            mapped_ring_bytes: 11, host_exchange_bytes: 13, row_indices_bytes: 17 };
        let owners = spark_owners(&report);
        assert_eq!(owners.device, "spark2");
        assert_eq!(owners.weights_bytes, 100);
        assert_eq!(owners.workspace_bytes, 71);
        assert!(owners.reserve.is_empty()); // The raw/global residual is not a named owner.
        let sample = row("spark2", Some(170), &owners, None, json!({}), None);
        assert_eq!(sample["charged_bytes"], 171);
        assert_eq!(sample["delta_vs_measured"], -1);
    }

    #[test]
    fn mimo_admission_publishes_one_canonical_graph_owner_per_device() {
        use cuteafd_core::memory_layout::{Basis, Item};
        use cuteafd_core::memory_policy::{DeviceBreakdown, DeviceKind};
        use cuteafd_loader::families::mimo_v2::{decode_graph::graph_layout, MimoV2Config};
        use cuteafd_loader::placement::{families::mimo_v2, reserve};
        let admission = include_str!("../../families/mimo_v2/admission.rs");
        assert_eq!(admission.matches("additional.extend(graph_layout.reservations(rank)?)").count(), 1);
        for config in [cuteafd_loader::plan::testing::mimo_flash_config(),
            cuteafd_loader::plan::testing::mimo_pro_config()] {
            let cfg = MimoV2Config::from_hf(&config).unwrap();
            for ranks in [1, 2] {
                let layout = graph_layout(&cfg, ranks, true, true).unwrap();
                for rank in 0..ranks {
                    let reservations = layout.reservations(rank).unwrap();
                    assert_eq!(reservations.len(), 1);
                    assert_eq!(reservations[0].name, "graph reserve");
                    let items: Vec<_> = reservations.iter().map(|owner| Item::new(
                        mimo_v2::reservation_category(&owner.name), mimo_v2::reservation_group(&owner.name),
                        "", owner.bytes, Basis::Formula)).collect();
                    let device = format!("rtx{rank}");
                    let owners = reserve::breakdown(&device, DeviceKind::Gpu, 96 << 30, 0, &items);
                    // log_plan publishes this serialized breakdown; capture reads the same schema.
                    let published = json!({format!("device.{device}"): owners});
                    let owners: DeviceBreakdown = serde_json::from_value(published[format!("device.{device}")].clone()).unwrap();
                    assert_eq!(owners.reserve.iter().filter(|owner| owner.name == "graph reserve").count(), 1);
                    assert!(!owners.reserve.iter().any(|owner| owner.name != "graph reserve"
                        && owner.name.to_ascii_lowercase().contains("graph")));
                    let sample = row(&device, None, &owners, Some(0), json!({}), None);
                    assert_eq!(sample["graph_reserve_bytes"], layout.reserves[rank]);
                    assert!(sample["graph_reserve_bytes"].as_u64().unwrap() > 0);
                }
            }
        }
    }


    #[test]
    fn v41_admission_owner_reaches_canonical_reconciliation_once_per_rank() {
        use cuteafd_core::memory_layout::{Basis, Category, Item, ItemLifetime};
        use cuteafd_core::memory_policy::DeviceKind;
        for ranks in [1, 2] {
            for rank in 0..ranks {
                for lanes in [1u8, 2] {
                    let owner = cuteafd_loader::placement::families::deepseek_v41::admission::graph_allowance(ranks, rank, lanes).unwrap();
                    let items = [Item { category: Category::Runtime, group: owner.source.clone(),
                        format: String::new(), bytes: owner.bytes, basis: Basis::Calibrated,
                        lifetime: ItemLifetime::PostReadyReserve }];
                    let id = format!("rtx{rank}");
                    let owners = cuteafd_loader::placement::reserve::breakdown(&id,
                        DeviceKind::Gpu, 96 << 30, 0, &items);
                    let graphs: Vec<_> = owners.reserve.iter().filter(|owner| owner.name == "graph reserve").collect();
                    assert_eq!(graphs.len(), 1);
                    assert!(graphs[0].bytes > 0);
                    assert_eq!(graphs[0].bytes, owner.bytes);
                    let measured = cuteafd_loader::placement::graph_reserve::v41_working_set(ranks, rank, lanes).unwrap();
                    let soft = cuteafd_loader::placement::graph_reserve::resolve_working_set(None, None, None, rank, measured).unwrap().base_bytes;
                    let sample = row(&id, Some(i128::from(soft)), &owners, Some(soft), json!({}), None);
                    assert_eq!(sample["graph_reserve_bytes"], owner.bytes);
                    assert_eq!(sample["graph_reserve_slack"], owner.bytes - soft);
                    assert_eq!(sample["delta_vs_measured"], 0);
                }
            }
        }
    }

    #[test]
    fn graph_slack_includes_bounded_extra_charged_before_kv() {
        let policy = cuteafd_loader::placement::graph_reserve::resolve(None, 0, false, 100, 20).unwrap();
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "rtx0".into(), cap_bytes: 1000,
            weights_bytes: 0, workspace_bytes: 0, kv_bytes: 0, other_bytes: 0, reserve_overridden: false,
            reserve: vec![cuteafd_core::memory_policy::ReserveComponent { name: "graph reserve".into(), bytes: policy.bytes }] };
        let sample = row("rtx0", Some(100), &owners, Some(100), json!({}), None);
        assert_eq!(sample["graph_reserve_bytes"], 240);
        assert_eq!(sample["graph_reserve_slack"], 140);
        assert_eq!(sample["delta_vs_measured"], 0);
    }
    #[test]
    fn signed_delta_keeps_release_padding_separate_and_preserves_missing_samples() {
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "rtx0".into(), cap_bytes: 1000,
            weights_bytes: 60, workspace_bytes: 10, kv_bytes: 20, other_bytes: 0, reserve_overridden: false,
            reserve: vec![cuteafd_core::memory_policy::ReserveComponent { name: "release padding".into(), bytes: 10 }] };
        let measured = row("rtx0", Some(85), &owners, None, json!({}), None);
        assert_eq!(measured["charged_bytes"], 90);
        assert_eq!(measured["padding_bytes"], 10);
        assert_eq!(measured["delta_vs_admitted"], -5);
        assert_eq!(measured["delta_vs_measured"], -5);
        assert_eq!(measured["graph_reserve_slack"], 0);
        assert_eq!(measured["beyond_half_padding"], false);
        let missing = row("spark0", None, &owners, None, json!({}), Some("no telemetry"));
        assert!(missing["delta_vs_admitted"].is_null());
        assert!(missing["delta_vs_measured"].is_null());
        assert!(missing["beyond_half_padding"].is_null());
        let mut owners = owners;
        owners.reserve.push(cuteafd_core::memory_policy::ReserveComponent {
            name: "graph reserve".into(), bytes: 100 });
        let unused = row("rtx0", Some(110), &owners, Some(20), json!({}), None);
        assert_eq!(unused["delta_vs_admitted"], -80);
        assert_eq!(unused["graph_reserve_slack"], 80);
        assert_eq!(unused["delta_vs_measured"], 0);
        assert_eq!(unused["beyond_half_padding"], false);
        let drift = row("rtx0", Some(116), &owners, Some(20), json!({}), None);
        assert_eq!(drift["delta_vs_measured"], 6);
        assert_eq!(drift["beyond_half_padding"], true);
        let unknown = row("rtx0", Some(110), &owners, None, json!({}), None);
        assert_eq!(unknown["delta_vs_admitted"], -80);
        assert!(unknown["delta_vs_measured"].is_null());
        assert!(unknown["graph_reserve_slack"].is_null());
        assert!(unknown["beyond_half_padding"].is_null());
        let exceeded = row("rtx0", Some(210), &owners, Some(120), json!({}), None);
        assert_eq!(exceeded["graph_reserve_slack"], -20);
        assert_eq!(exceeded["delta_vs_measured"], 0);
    }
}
