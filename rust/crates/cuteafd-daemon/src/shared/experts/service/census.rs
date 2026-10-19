//! Independent same-process maps and system headroom; neither alone is CUDA attribution.
use anyhow::{ensure, Context, Result};
use cuteafd_core::expert_geometry::{spark_census_deltas, CensusDeviceKind, CensusDeviceWitness, SparkAllocationTerm, SparkProcessSample, SparkRuntimeDetail};
use cuteafd_ffi::{memory_ledger::{Snapshot, Space}, NativeLibrary};
use std::time::Instant;

pub(super) struct Census {
    start: Instant,
    samples: Vec<SparkProcessSample>,
    enabled: bool,
    device_kind: Option<CensusDeviceKind>,
}
impl Census {
    pub(super) fn new() -> Self {
        Self { start: Instant::now(), samples: vec![], device_kind: None,
            enabled: std::env::var("CUTEAFD_GRAPH_CENSUS").is_ok_and(|v| v == "1") }
    }
    pub(super) fn enabled(&self) -> bool { self.enabled }
    pub(super) fn identify_device(&mut self, library: &NativeLibrary) -> Result<()> {
        if !self.enabled { return Ok(()); }
        self.device_kind = Some(match library.cuda_device_info(library.cuda_get_device()?)?.integrated {
            0 => CensusDeviceKind::Discrete,
            1 => CensusDeviceKind::Integrated,
            _ => anyhow::bail!("unsupported CUDA integrated-device property"),
        });
        Ok(())
    }

    pub(super) fn sample(&mut self, phase: &str, library: Option<&NativeLibrary>) -> Result<()> {
        if !self.enabled { return Ok(()); }
        let text = std::fs::read_to_string("/proc/self/smaps")
            .context("reading process-attributable Spark census")?;
        let maps = parse_smaps(&text)?;
        let status = std::fs::read_to_string("/proc/self/status")?;
        let peak_rss = optional_kib_field(&status, "VmHWM:")?;
        let cgroup_memory = cuteafd_ffi::cgroup_memory_census()?;
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        let available = kib_field(&meminfo, "MemAvailable:")?;
        let system_headroom = cuteafd_ffi::cgroup_memory_headroom(usize::try_from(available)?)? as u64;
        let witness = self.device_kind.map(|kind| match kind {
            CensusDeviceKind::Integrated => CensusDeviceWitness::SystemAvailable,
            CensusDeviceKind::Discrete => CensusDeviceWitness::CudaMappingPss,
        });
        let ledger = cuteafd_ffi::memory_ledger::snapshot();
        let device = library.map(|l| l.cuda_get_device()).transpose()?.unwrap_or(0);
        if phase == "steady_after_first_expert_request" {
            self.samples.retain(|sample| sample.phase != phase);
        }
        self.samples.push(SparkProcessSample { phase: phase.into(), pid: std::process::id(),
            monotonic_ns: self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            rss_bytes: maps.rss, peak_rss_bytes: peak_rss, cgroup_memory,
            pss_bytes: maps.pss, private_bytes: maps.private,
            device_kind: self.device_kind, device_witness: witness,
            cuda_mapping_pss_bytes: maps.cuda_pss, anonymous_pss_bytes: maps.anon_pss,
            file_pss_bytes: maps.file_pss,
            system_available_bytes: available, system_headroom_bytes: system_headroom,
            system_total_bytes: kib_field(&meminfo, "MemTotal:")?,
            allocatable_bytes: library.map(|l| l.cuda_allocatable_memory_info()
                .map(|(free, _)| free as u64)).transpose()?,
            tracked_device_bytes: ledger.total(Space::Device, device) as u64,
            tracked_pinned_bytes: ledger.total(Space::Pinned, -1) as u64,
            raw_cuda_free_bytes: library.and_then(|l| l.cuda_physical_memory_info().ok()).map(|(free, _)| free as u64),
            method: "same_pid_status VmHWM and raw memory-cgroup current/peak/max (lifetime peaks, not isolated attribution); same_pid_smaps (CUDA device mapping PSS, anonymous PSS, file PSS); MemAvailable residency witness; cgroup-bounded headroom is an admission constraint only on unified devices; raw CUDA diagnostic only; device witness explicitly selected by integrated/discrete property; host-only maps/system agreement; quiet-host qualification external".into() });
        tracing::info!(target: "cuteafd::spark_memory", record = %serde_json::to_string(self.samples.last().unwrap())?,
            "Spark process census");
        Ok(())
    }
    pub(super) fn detail(&self, before: &Snapshot, after: &Snapshot, device: i32) -> Result<SparkRuntimeDetail> {
        let mut allocations = Vec::new();
        for row in &after.rows {
            if !((row.key.space == Space::Device && row.key.device == device)
                || row.key.space == Space::Pinned) { continue; }
            let old = before.rows.iter().find(|r| r.key == row.key);
            let bytes = row.bytes.saturating_sub(old.map_or(0, |r| r.bytes));
            if bytes == 0 { continue; }
            allocations.push(SparkAllocationTerm { space: row.key.space.label().into(),
                scope: row.key.scope.into(), tensor: row.key.tensor.map(str::to_owned),
                format: row.key.format.map(str::to_owned), bytes: bytes as u64,
                allocations: row.allocations.saturating_sub(old.map_or(0, |r| r.allocations)) as u64,
                method: "native allocation ledger owner delta; payload bytes, excludes driver granularity".into() });
        }
        let verdict = witness_verdict(&self.samples);
        let planned_context_bytes = cuteafd_loader::placement::ArchContext::for_device(
            "sm_121", self.samples.first().map_or(0, |s| s.system_total_bytes)).context_bytes;
        tracing::info!(target: "cuteafd::spark_memory", %verdict, planned_context_bytes,
            "Spark dual-witness census qualification (quiet-host evidence required externally)");
        Ok(SparkRuntimeDetail { load_transient: vec![], allocations, process_samples: self.samples.clone(),
            runtime_sample_method: format!("{verdict}; PostLoad system/cache remainder is measured_residual, not CUDA context/modules calibration"),
            // A process cannot certify that the entire host was quiet. The offline
            // qualified proof constructor also requires that external evidence.
            context_modules_qualified: false })
    }
}

fn optional_kib_field(text: &str, key: &str) -> Result<Option<u64>> {
    if text.lines().any(|line| line.starts_with(key)) { kib_field(text, key).map(Some) }
    else { Ok(None) }
}

fn kib_field(text: &str, key: &str) -> Result<u64> {
    let mut values = text.lines().filter_map(|line| line.strip_prefix(key));
    let line = values.next().with_context(|| format!("missing {key}"))?;
    ensure!(values.next().is_none(), "duplicate {key}");
    let mut tokens = line.split_whitespace();
    let number = tokens.next().context("missing memory number")?;
    ensure!(!number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()), "invalid memory number");
    let kib: u64 = number.parse()?;
    ensure!(tokens.next() == Some("kB") && tokens.next().is_none(), "unexpected memory unit");
    kib.checked_mul(1024).context("memory extent overflow")
}

#[derive(Default, Debug, PartialEq, Eq)]
struct Maps { rss: u64, pss: u64, private: u64, cuda_pss: u64, anon_pss: u64, file_pss: u64 }
fn parse_smaps(text: &str) -> Result<Maps> {
    let mut maps = Maps::default();
    let mut block = String::new();
    let mut header = String::new();
    let mut add = |header: &str, block: &str| -> Result<()> {
        if header.is_empty() { return Ok(()); }
        let rss = kib_field(block, "Rss:")?;
        let pss = kib_field(block, "Pss:")?;
        let private = kib_field(block, "Private_Clean:")?.checked_add(kib_field(block, "Private_Dirty:")?)
            .context("private memory overflow")?;
        let path = header.split_whitespace().skip(5).collect::<Vec<_>>().join(" ");
        let target = if path.starts_with("/dev/nvidia") || path.starts_with("/dev/nvmap") {
            &mut maps.cuda_pss
        } else if path.starts_with('/') { &mut maps.file_pss } else { &mut maps.anon_pss };
        *target = target.checked_add(pss).context("mapping PSS overflow")?;
        maps.rss = maps.rss.checked_add(rss).context("RSS overflow")?;
        maps.pss = maps.pss.checked_add(pss).context("PSS overflow")?;
        maps.private = maps.private.checked_add(private).context("private memory overflow")?;
        Ok(())
    };
    for line in text.lines() {
        let range = line.split_whitespace().next().unwrap_or("");
        let is_header = range.split_once('-').is_some_and(|(a,b)| !a.is_empty() && !b.is_empty()
            && a.bytes().chain(b.bytes()).all(|c| c.is_ascii_hexdigit()));
        if is_header {
            add(&header, &block)?;
            header = line.into(); block.clear();
        } else { block.push_str(line); block.push('\n'); }
    }
    add(&header, &block)?;
    ensure!(!header.is_empty(), "smaps has no mappings");
    Ok(maps)
}

fn witness_verdict(samples: &[SparkProcessSample]) -> String {
    if !samples.iter().any(|sample| sample.phase == "steady_after_first_expert_request") {
        return "pending: endpoint rings and first completed expert request not sampled; unqualified".into();
    }
    match spark_census_deltas(samples) {
        Ok(delta) => format!("witnesses consistent: {:?}; context_bytes={}; module_bytes={}; external quiet-host qualification required",
            delta.witness, delta.context, delta.modules),
        Err(error) => format!("unqualified: {error}; stop"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lifetime_peak_rss_is_optional_but_malformed_samples_fail() {
        assert_eq!(optional_kib_field("Name: worker\nVmHWM: 123 kB\n", "VmHWM:").unwrap(), Some(123 * 1024));
        assert_eq!(optional_kib_field("Name: kernel\n", "VmHWM:").unwrap(), None);
        for text in ["VmHWM: 1 kB\nVmHWM: 2 kB", "VmHWM: nope kB", "VmHWM: 4 MB",
            "VmHWM: -1 kB", "VmHWM: 18446744073709551615 kB"] {
            assert!(optional_kib_field(text, "VmHWM:").is_err(), "{text}");
        }
    }

    fn samples() -> Vec<SparkProcessSample> {
        let phases = ["before_context", "after_context_before_media", "before_expert_modules",
            "after_expert_modules_before_weights", "after_retained_arenas", "serve_owners_before_endpoint_rings",
            "steady_after_first_expert_request"];
        phases.into_iter().enumerate().map(|(i, phase)| {
            let mib = 1 << 20;
            let cuda = [0, 32, 32, 96, 1120, 1120, 1120][i] * mib;
            SparkProcessSample { phase: phase.into(), pid: 42, monotonic_ns: i as u64 + 1,
                rss_bytes: cuda, peak_rss_bytes: Some(cuda), cgroup_memory: vec![], pss_bytes: cuda, private_bytes: 0,
                tracked_device_bytes: if i >= 4 { 1024 * mib } else { 0 }, tracked_pinned_bytes: 0,
                device_kind: Some(CensusDeviceKind::Discrete), device_witness: Some(CensusDeviceWitness::CudaMappingPss),
                cuda_mapping_pss_bytes: cuda, anonymous_pss_bytes: 0, file_pss_bytes: 0,
                system_headroom_bytes: 8192 * mib - cuda,
                system_available_bytes: 8192 * mib - cuda, system_total_bytes: 16384 * mib,
                allocatable_bytes: None, raw_cuda_free_bytes: None, method: "test maps/system".into() }
        }).collect()
    }
    #[test]
    fn witnesses_require_visible_cuda_in_both_isolated_phases() {
        assert!(witness_verdict(&samples()).starts_with("witnesses consistent"));
        for bad in 0..10 {
            let mut s = samples();
            match bad {
                0 => s[1].pid = 43,
                1 => s[2].monotonic_ns = s[1].monotonic_ns,
                2 => s[1].tracked_device_bytes = 1,
                3 => s[3].tracked_pinned_bytes = 1,
                4 => s[6].cuda_mapping_pss_bytes = s[3].cuda_mapping_pss_bytes,
                5 => s[1].cuda_mapping_pss_bytes = 0,
                6 => s[3].system_available_bytes -= 32 << 20,
                7 => s[1].system_total_bytes = 0,
                8 => s.push(s[0].clone()),
                _ => s[4].tracked_device_bytes = 0,
            }
            assert!(witness_verdict(&s).starts_with("unqualified:"), "case {bad}");
        }
    }
    #[test]
    fn integrated_uses_system_control_and_host_only_maps() {
        let mut data = samples();
        for s in &mut data {
            s.device_kind = Some(CensusDeviceKind::Integrated);
            s.device_witness = Some(CensusDeviceWitness::SystemAvailable);
            s.cuda_mapping_pss_bytes = 0;
        }
        let delta = spark_census_deltas(&data).unwrap();
        assert_eq!(delta.context, 32 << 20);
        assert_eq!(delta.modules, 64 << 20);
        assert_eq!(delta.witness, CensusDeviceWitness::SystemAvailable);
        // An already resident code module can have a genuine zero delta.
        data[3].system_available_bytes = data[2].system_available_bytes;
        data[3].system_headroom_bytes = data[2].system_headroom_bytes;
        for i in 4..data.len() {
            data[i].system_available_bytes += 64 << 20;
            data[i].system_headroom_bytes += 64 << 20;
        }
        assert_eq!(spark_census_deltas(&data).unwrap().modules, 0);
        for bad in 0..8 {
            let mut d = data.clone();
            match bad {
                0 => d[0].device_witness = Some(CensusDeviceWitness::CudaMappingPss),
                1 => d[1].device_kind = Some(CensusDeviceKind::Discrete),
                2 => d[0].device_kind = None,
                3 => d[6].system_available_bytes += 100 << 20,
                4 => d[6].system_available_bytes += 100 << 20,
                5 => d[5].anonymous_pss_bytes += 32 << 20,
                6 => d[5].tracked_device_bytes += 1,
                _ => d[0].system_headroom_bytes = d[0].system_available_bytes + 1,
            }
            assert!(spark_census_deltas(&d).is_err(), "case {bad}");
        }
    }
    #[test]
    fn isolated_system_noise_is_symmetric_and_bounded() {
        let mut data = samples();
        for sample in &mut data {
            sample.device_kind = Some(CensusDeviceKind::Integrated);
            sample.device_witness = Some(CensusDeviceWitness::SystemAvailable);
            sample.cuda_mapping_pss_bytes = 0;
        }
        for phase in [3, 5] {
            for noise in [1, 16 << 20, (16 << 20) + 1] {
                let mut noisy = data.clone();
                noisy[phase].system_available_bytes = noisy[phase - 1].system_available_bytes + noise;
                noisy[phase].system_headroom_bytes = noisy[phase - 1].system_headroom_bytes + noise;
                if phase == 3 {
                    let shift = noisy[phase].system_available_bytes - data[phase].system_available_bytes;
                    for later in &mut noisy[4..] {
                        later.system_available_bytes += shift;
                        later.system_headroom_bytes += shift;
                    }
                }
                assert_eq!(spark_census_deltas(&noisy).is_ok(), noise <= 16 << 20,
                    "phase {phase}, negative delta {noise}");
            }
            for cgroup_only in [false, true] {
                let mut noisy = data.clone();
                if cgroup_only {
                    noisy[phase].system_headroom_bytes = noisy[phase].system_available_bytes + 1;
                } else {
                    noisy[phase].system_available_bytes = noisy[phase - 1].system_available_bytes + (16 << 20) + 1;
                }
                assert!(spark_census_deltas(&noisy).is_err());
            }
        }
        // Retained payload control never uses the isolated-phase noise allowance.
        data[6].system_available_bytes = data[3].system_available_bytes + 1;
        data[6].system_headroom_bytes = data[3].system_headroom_bytes + 1;
        assert!(spark_census_deltas(&data).is_err());
    }
    #[test]
    fn unified_steady_control_ignores_reclaimed_cache_and_requires_real_sample() {
        let mut data = samples();
        for sample in &mut data {
            sample.device_kind = Some(CensusDeviceKind::Integrated);
            sample.device_witness = Some(CensusDeviceWitness::SystemAvailable);
            sample.raw_cuda_free_bytes = Some(1);
            sample.cuda_mapping_pss_bytes = 0;
            sample.system_headroom_bytes = 1; // Cache charge is not residency.
        }
        assert!(spark_census_deltas(&data).is_ok());
        // Endpoint rings are retained payload and must appear in both controls.
        data[6].tracked_pinned_bytes += 64 << 20;
        data[6].system_available_bytes -= 64 << 20;
        assert!(spark_census_deltas(&data).is_ok());
        data[6].system_available_bytes -= 64 << 20;
        assert!(spark_census_deltas(&data).is_err());
        data.pop();
        assert!(spark_census_deltas(&data).is_err());
        let source = include_str!("local.rs");
        let gate = source.find("if processed && executed && pending_steady_log").unwrap();
        let tail = &source[gate..];
        assert!(tail.find("cuda_device_synchronize()?").unwrap()
            < tail.find("census.sample(\"steady_after_first_expert_request\"").unwrap());
        assert!(source.find("if skip_compute").unwrap() < source.find("executed = request.rows() > 0").unwrap());
    }
    #[test]
    fn process_snapshot_requires_explicit_fields_and_units() {
        let block = "Rss: 12 kB\nPss: 8 kB\nPrivate_Clean: 2 kB\nPrivate_Dirty: 3 kB\n";
        let maps = parse_smaps(&format!("1000-2000 rw-s 00000000 00:00 0 /dev/nvidia0\n{block}")).unwrap();
        assert_eq!((maps.rss, maps.pss, maps.private, maps.cuda_pss), (12288,8192,5120,8192));
        assert!(kib_field("Rss: 12 MB", "Rss:").is_err());
        assert!(kib_field("Rss: 18446744073709551615 kB", "Rss:").is_err());
        assert!(kib_field("Rss: 1 kB\nRss: 2 kB", "Rss:").is_err());
        assert!(parse_smaps("Rss: 1 kB").is_err());
    }
    #[test]
    fn mapping_domains_are_disjoint_and_anonymous_is_not_cuda() {
        let fields = "Rss: 10 kB\nPss: 8 kB\nPrivate_Clean: 0 kB\nPrivate_Dirty: 2 kB\n";
        let maps = parse_smaps(&format!("1000-2000 rw-p 0 00:00 0\n{fields}2000-3000 r-xp 0 00:00 1 /lib/code.so\n{fields}3000-4000 rw-s 0 00:00 0 /dev/nvmap\n{fields}")).unwrap();
        assert_eq!(maps.pss, maps.cuda_pss + maps.anon_pss + maps.file_pss);
        assert_eq!((maps.cuda_pss, maps.anon_pss, maps.file_pss), (8192,8192,8192));
    }
    #[test]
    fn retained_scope_details_are_not_context_qualification() {
        let census = Census::new();
        let detail = census.detail(&Snapshot::default(), &Snapshot::default(), 0).unwrap();
        assert!(!detail.context_modules_qualified);
        assert!(detail.runtime_sample_method.contains("measured_residual"));
    }
    #[test]
    fn census_phases_isolate_context_and_retained_modules() {
        let source = include_str!("local.rs");
        let phases = ["census.sample(\"before_native_library\"", "let library = unsafe { NativeLibrary::load",
            "census.sample(\"before_context\"", "library.cuda_set_device(library.cuda_get_device()?)?",
            "library.cuda_device_synchronize()?",
            "census.sample(\"after_context_before_media\"", "crate::shared::vision::worker::start_encoders(",
            "census.sample(\"before_expert_modules\"", "super::modules::WarmModules::load",
            "census.sample(\"after_expert_modules_before_weights\"", "load_weights(&library",
            "census.sample(\"before_retained_arenas\"", "weights.execution(&library",
            "census.sample(\"after_retained_arenas\"", "census.sample(\"serve_owners_before_endpoint_rings\""];
        let mut next = 0;
        for phase in phases {
            next += source[next..].find(phase).unwrap_or_else(|| panic!("missing ordered {phase}")) + phase.len();
        }
    }
}
