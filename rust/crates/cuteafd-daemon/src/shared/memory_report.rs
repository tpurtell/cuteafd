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

pub(crate) fn register_inputs(matches: &clap::ArgMatches, policy: &cuteafd_core::memory_policy::MemoryPolicy, weight_budget: Option<f64>) {
    let mut inputs = serde_json::to_value(policy).unwrap_or(Value::Null).as_object().cloned().unwrap_or_default();
    inputs.insert("coordinator_weight_budget_gib".into(), json!(weight_budget.unwrap_or(80.0)));
    // Allowlist only numeric/enum reservation inputs; never arbitrary arguments,
    // paths, credentials, prompts, or an environment dump.
    if let Some((_, args)) = matches.subcommand() {
        for key in ["max_context_tokens", "max_sequences", "prefill_rows", "pool_tokens", "max_prefill_tokens",
            "kv_page_tokens", "prefix_cache_bytes", "speculator", "speculation_depth"] {
            if let Ok(Some(values)) = args.try_get_raw(key) {
                inputs.insert(key.into(), json!(values.map(|v| v.to_string_lossy().into_owned()).collect::<Vec<_>>()));
            }
        }
    }
    memory_ledger::fault::set_inputs(Value::Object(inputs));
}

/// Reuse the launcher's existing instance/run mounts; never write into source cwd.
pub(crate) fn dump_location(matches: &clap::ArgMatches) -> Option<(String, std::path::PathBuf)> {
    dump_location_with(matches, std::env::var_os("CUTEAFD_PROFILE_DIR").map(std::path::PathBuf::from))
}
fn dump_location_with(matches: &clap::ArgMatches, profile: Option<std::path::PathBuf>) -> Option<(String, std::path::PathBuf)> {
    let (command, args) = matches.subcommand()?;
    let role = if command == "expertd-native" {
        format!("rank{}", args.try_get_one::<u32>("rank").ok().flatten().copied().unwrap_or(0))
    } else if command.starts_with("serve") { "coordinator".into() }
    else { return None };
    let directory = profile
        .or_else(|| args.try_get_one::<std::path::PathBuf>("usage_dir").ok().flatten().map(|dir| dir.join("memory")))?;
    Some((role, directory))
}

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

/// Query active primary contexts only, restoring the exact prior driver context.
pub(crate) fn report(stage: &str) -> Value {
    report_snapshot(stage, &memory_ledger::snapshot(), &memory_ledger::fault::device_memory)
}

/// Bytes by memory kind, plus the zero untagged counter required by the census.
fn kinds(snapshot: &memory_ledger::Snapshot, space: Space, device: i32) -> Value {
    let (kinds, untagged) = snapshot.by_kind(space, device);
    json!({"kinds": kinds.into_iter().map(|(kind, bytes)| (kind.label(), bytes)).collect::<BTreeMap<_, _>>(),
        "untagged": untagged})
}

fn report_snapshot(stage: &str, snapshot: &memory_ledger::Snapshot,
    device_memory: &impl Fn(i32) -> Option<(usize, usize)>) -> Value {
    let mut per_device = Vec::new();
    // Only devices this process allocated on: querying another would create a context there.
    for device in snapshot.devices() {
        let tracked = snapshot.total(Space::Device, device) + snapshot.total(Space::Managed, device);
        let scopes: BTreeMap<&str, usize> = snapshot.by_scope(Space::Device, device);
        // Live allocations behind every scope, so a scope's bytes can be read
        // beside the number of physical requests that carry rounding.
        let scope_allocations: BTreeMap<&str, usize> = snapshot.rows.iter()
            .filter(|row| row.key.space == Space::Device && row.key.device == device)
            .fold(BTreeMap::new(), |mut out, row| {
                *out.entry(row.key.scope).or_default() += row.allocations;
                out
            });
        let mut entry = json!({
            "device": device,
            "cuda_memory_scope": "raw cudaMemGetInfo; GB10 under-reports reclaimable page cache; host MemAvailable is allocatable headroom",
            "tracked": tracked,
            "peak": snapshot.peak.get(&(Space::Device, device)).copied().unwrap_or(0),
            "scopes": scopes,
            "scope_allocations": scope_allocations,
            "by_kind": kinds(snapshot, Space::Device, device),
            "managed_by_kind": kinds(snapshot, Space::Managed, device),
        });
        if let Some(g) = snapshot.granularity.get(&(Space::Device, device)) {
            // SM120 direct-device rounding already inside the measured bytes used.
            entry["granularity"] = json!({"tail": g.tail, "large_requests": g.large_requests,
                "small_bytes": g.small_bytes});
        }
        if let Some((free, total)) = device_memory(device) {
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
            "by_kind": kinds(snapshot, Space::Pinned, -1),
        },
        // Native malloc backing registered for RDMA (not CUDA-pinned).
        "host_registered": {
            "tracked": snapshot.total(Space::Host, -1),
            "by_kind": kinds(snapshot, Space::Host, -1),
        },
        "weights": weights,
        "dual_formats": dual,
        "host": meminfo(),
    })
}

/// Logs a report from a helper thread (the caller's device selection is untouched).
pub(crate) fn log(stage: &str) {
    if stage == "ready" {
        memory_ledger::fault::ready();
        let summary = memory_ledger::fault::compact_summary(stage);
        tracing::info!(target: "cuteafd::memory", "{summary}");
    }
    let stage = stage.to_owned();
    let value = std::thread::spawn(move || report(&stage)).join();
    if let Ok(mut value) = value {
        // Bounded diagnostic boundary: the process-wide stream census and the
        // current device's CUDA pool retention. Both are queries: no
        // allocation, no synchronization, no device switch.
        let (live, peak) = cuteafd_ffi::stream_census();
        value["streams"] = json!({"live": live, "peak": peak,
            "scope": "process-wide cudaStreamCreate/Destroy census; a context's driver slabs follow its live streams at 2 MiB per four (#1511)"});
        if let Some(pool) = memory_ledger::current_cuda_memory_snapshot() {
            value["cuda_pool"] = pool;
        }
        tracing::info!(target: "cuteafd::memory", report = %value, "memory ledger");
    }
}

/// One final error-boundary report for every command/family. Native failures
/// retain the pre-unwind sample; other errors report the remaining live owners.
pub(crate) fn finish<T>(result: anyhow::Result<T>) -> anyhow::Result<T> {
    if let Err(error) = &result {
        let retained = error.chain().find_map(|cause| cause.downcast_ref::<memory_ledger::FailureSnapshot>());
        if let Some(report) = retained.and_then(|failure| failure.report.as_ref()) {
            tracing::error!(target: "cuteafd::memory", path = %report.path.display(), complete = report.complete,
                error = %error, "retained memory failure dump\n{}", report.summary);
            return result;
        }
        let mut value = match retained {
            Some(failure) => report_snapshot("failure", &failure.ledger,
                &|device| failure.device_memory.get(&device).copied()),
            None => report("failure"),
        };
        value["error"] = json!(format!("{error:#}"));
        value["pre_unwind"] = json!(retained.is_some());
        tracing::error!(target: "cuteafd::memory", report = %value, error = %error, "memory ledger");
    }
    result
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
pub(crate) fn release_load_staging(library: &cuteafd_ffi::NativeLibrary) -> anyhow::Result<()> {
    let bytes = library.release_sync_h2d_staging()?;
    if bytes > 0 {
        tracing::info!(bytes, "released load-time pinned upload staging");
    }
    Ok(())
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

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn command_error_emits_one_final_failure_report() {
        use std::sync::{Arc, Mutex};
        #[derive(Clone)]
        struct Output(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Output {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let output = Output(Arc::new(Mutex::new(Vec::new())));
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt().without_time().with_ansi(false)
            .with_writer(move || writer.clone()).finish();
        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(finish(Ok(7)).unwrap(), 7);
            let failure = memory_ledger::FailureSnapshot::new(
                anyhow::anyhow!("startup fixture failure"), memory_ledger::Snapshot::default(), BTreeMap::new());
            let wrapped = anyhow::Error::new(crate::shared::experts::rtx::push::PushStartupError::Planning(
                anyhow::Error::new(failure))).context("family initialization");
            assert!(finish::<()>(Err(wrapped)).is_err());
        });
        let text = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
        assert_eq!(text.matches("memory ledger").count(), 1, "{text}");
        assert!(text.contains(r#""stage":"failure""#), "{text}");
        assert!(text.contains("startup fixture failure"), "{text}");
        assert!(text.contains(r#""pre_unwind":true"#), "{text}");
    }

    #[test]
    fn readiness_forwarding_keeps_snapshot_and_emits_both_ready_outputs() {
        use std::sync::{Arc, Mutex};
        #[derive(Clone)] struct Output(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Output {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let output = Output(Arc::new(Mutex::new(Vec::new())));
        let writer = output.clone();
        let subscriber = tracing_subscriber::fmt().without_time().with_ansi(false)
            .with_writer(move || writer.clone()).finish();
        tracing::subscriber::with_default(subscriber, || {
            log("ready");
            let (send, receive) = std::sync::mpsc::channel();
            let error = anyhow::Error::new(memory_ledger::FailureSnapshot::new(
                anyhow::anyhow!("original typed failure"), memory_ledger::Snapshot::default(), BTreeMap::new()));
            crate::shared::serve::forward_initialization(Err(error), true, "fixture startup", |error|
                send.send(error).map_err(|error| error.0)).unwrap();
            let error = receive.recv().unwrap();
            assert!(error.chain().any(|cause| cause.is::<memory_ledger::FailureSnapshot>()));
            assert!(finish::<()>(Err(error)).is_err());
        });
        let text = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
        assert!(text.contains("memory summary stage=ready"), "{text}");
        assert!(text.contains(r#""stage":"ready""#), "{text}");
        assert!(text.contains(r#""pre_unwind":true"#), "{text}");
        assert!(text.contains("original typed failure"), "{text}");
    }

    #[test]
    fn dump_location_requires_serving_and_an_existing_output_contract() {
        use clap::{Arg, Command};
        use std::path::PathBuf;
        let command = || Command::new("fixture")
            .subcommand(Command::new("serve-step5").arg(Arg::new("usage_dir").long("usage-dir").value_parser(clap::value_parser!(PathBuf))))
            .subcommand(Command::new("expertd-native").arg(Arg::new("rank").long("rank").value_parser(clap::value_parser!(u32))))
            .subcommand(Command::new("plan"));
        let serve = command().try_get_matches_from(["fixture", "serve-step5", "--usage-dir", "/instance/usage"]).unwrap();
        assert_eq!(dump_location_with(&serve, None), Some(("coordinator".into(), "/instance/usage/memory".into())));
        let worker = command().try_get_matches_from(["fixture", "expertd-native", "--rank", "3"]).unwrap();
        assert_eq!(dump_location_with(&worker, Some("/run/output".into())), Some(("rank3".into(), "/run/output".into())));
        assert!(dump_location_with(&worker, None).is_none());
        let plan = command().try_get_matches_from(["fixture", "plan"]).unwrap();
        assert!(dump_location_with(&plan, Some("/run/output".into())).is_none());
    }

    #[test]
    fn no_kv_room_names_every_gpu() {
        let no_room = anyhow::Error::from(NoKvRoom { free_after_reserve: vec![-1_024, 4_096] });
        assert_eq!(no_room.to_string(), "no room for a KV pool after fixed costs (free after reserve [-1024, 4096] bytes)");
    }
}
