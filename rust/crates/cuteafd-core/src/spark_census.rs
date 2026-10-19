//! Device-kind-specific census witnesses shared by diagnostics and proof construction.
use super::SparkProcessSample;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CensusDeviceKind { Integrated, Discrete }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CensusDeviceWitness { SystemAvailable, CudaMappingPss }

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("unqualified Spark census: {0}")]
pub struct SparkCensusError(&'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkCensusDeltas {
    pub context: u64,
    pub modules: u64,
    pub witness: CensusDeviceWitness,
}

/// External quiet-host evidence is still required before publishing calibration.
pub fn spark_census_deltas(samples: &[SparkProcessSample]) -> Result<SparkCensusDeltas, SparkCensusError> {
    use CensusDeviceKind::{Discrete, Integrated};
    use CensusDeviceWitness::{CudaMappingPss, SystemAvailable};
    let error = |message| SparkCensusError(message);
    let phase = |name: &str| {
        let mut found = samples.iter().filter(|s| s.phase == name);
        let first = found.next().ok_or_else(|| error("missing isolated phase"))?;
        if found.next().is_some() { return Err(error("duplicate isolated phase")); }
        Ok(first)
    };
    let cb = phase("before_context")?;
    let ca = phase("after_context_before_media")?;
    let mb = phase("before_expert_modules")?;
    let ma = phase("after_expert_modules_before_weights")?;
    let host = phase("after_retained_arenas")?;
    let ready = phase("serve_owners_before_endpoint_rings")?;
    let steady = phase("steady_after_first_expert_request")?;
    let ordered = [cb, ca, mb, ma, host, ready, steady];
    let witness = match (cb.device_kind, cb.device_witness) {
        (Some(Integrated), Some(SystemAvailable)) => SystemAvailable,
        (Some(Discrete), Some(CudaMappingPss)) => CudaMappingPss,
        _ => return Err(error("unsupported device kind/witness combination")),
    };
    if cb.pid == 0 || cb.system_total_bytes == 0
        || ordered.iter().any(|s| s.pid != cb.pid || s.system_total_bytes != cb.system_total_bytes
            || s.system_available_bytes > s.system_total_bytes || s.system_headroom_bytes > s.system_available_bytes
            || s.device_kind != cb.device_kind || s.device_witness != cb.device_witness)
        || ordered.windows(2).any(|p| p[0].monotonic_ns >= p[1].monotonic_ns) {
        return Err(error("phase/process/system/device mismatch"));
    }
    let system = |before: &SparkProcessSample, after: &SparkProcessSample, isolated: bool| {
        let growth = |before: u64, after: u64| {
            match before.checked_sub(after) {
                Some(bytes) => Ok(bytes),
                // MemAvailable and cgroup samples can reclaim a small amount
                // during a payload-free phase; use the symmetric phase noise bar.
                None if isolated && after - before <= 16 << 20 => Ok(0),
                None => Err(error("system headroom increased beyond permitted phase noise")),
            }
        };
        let available = growth(before.system_available_bytes, after.system_available_bytes)?;
        // Unified CUDA allocations reclaim file cache and bypass memcg. Cgroup
        // headroom constrains host bytes, not device admission or residency.
        if witness == SystemAvailable { return Ok(available); }
        let bounded = growth(before.system_headroom_bytes, after.system_headroom_bytes)?;
        if available.abs_diff(bounded) > available / 20 {
            return Err(error("MemAvailable/cgroup phase deltas disagree by more than 5%"));
        }
        Ok(bounded)
    };
    let isolated = |before: &SparkProcessSample, after: &SparkProcessSample| {
        if before.tracked_device_bytes != after.tracked_device_bytes
            || before.tracked_pinned_bytes != after.tracked_pinned_bytes {
            return Err(error("isolated context/module phase allocated payload storage"));
        }
        let sys = system(before, after, true)?;
        if witness == SystemAvailable { return Ok(sys); }
        let maps = after.cuda_mapping_pss_bytes.checked_sub(before.cuda_mapping_pss_bytes)
            .ok_or_else(|| error("CUDA mapping PSS decreased"))?;
        if maps == 0 || maps.abs_diff(sys) > 16 << 20 {
            return Err(error("discrete CUDA mapping/system phase disagrees or is invisible"));
        }
        Ok(maps)
    };
    let context = isolated(cb, ca)?;
    if context == 0 { return Err(error("context phase has no measured growth")); }
    let modules = isolated(mb, ma)?;
    if host.tracked_device_bytes <= ma.tracked_device_bytes {
        return Err(error("missing retained device control"));
    }
    if host.tracked_device_bytes != ready.tracked_device_bytes
        || host.tracked_pinned_bytes != ready.tracked_pinned_bytes {
        return Err(error("host-only phase changed device/pinned owners"));
    }
    let host_maps = |s: &SparkProcessSample| s.anonymous_pss_bytes.checked_add(s.file_pss_bytes)
        .ok_or_else(|| error("host PSS overflow"));
    let maps = host_maps(ready)?.checked_sub(host_maps(host)?)
        .ok_or_else(|| error("host-only mapping PSS decreased"))?;
    if maps.abs_diff(system(host, ready, true)?) > 16 << 20 {
        return Err(error("host-only process-map/system phase disagrees by more than 16 MiB"));
    }
    // Include endpoint rings and lazy first-use owners without weakening the
    // earlier isolated host-only control.
    if steady.tracked_device_bytes < ready.tracked_device_bytes
        || steady.tracked_pinned_bytes < ready.tracked_pinned_bytes {
        return Err(error("steady retained owners decreased"));
    }
    let device = steady.tracked_device_bytes.checked_sub(ma.tracked_device_bytes)
        .ok_or_else(|| error("steady device ledger decreased"))?;
    let pinned = steady.tracked_pinned_bytes.checked_sub(ma.tracked_pinned_bytes)
        .ok_or_else(|| error("steady pinned ledger decreased"))?;
    let (visible, expected) = match witness {
        SystemAvailable => (system(ma, steady, false)?, device.checked_add(pinned)
            .ok_or_else(|| error("steady control overflow"))?),
        CudaMappingPss => (steady.cuda_mapping_pss_bytes.checked_sub(ma.cuda_mapping_pss_bytes)
            .ok_or_else(|| error("steady CUDA mapping PSS decreased"))?, device),
    };
    if expected == 0 || visible.abs_diff(expected) > expected / 20 {
        return Err(error("steady device/system witness disagrees by more than 5%"));
    }
    Ok(SparkCensusDeltas { context, modules, witness })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replay extracted process-sample arrays without starting CUDA or serving.
    #[test]
    #[ignore = "requires CUTEAFD_SPARK_CENSUS_REPLAY pointing to recorded rank samples"]
    fn replay_recorded_rank_census() {
        let path = std::env::var("CUTEAFD_SPARK_CENSUS_REPLAY").unwrap();
        let ranks: std::collections::BTreeMap<String, Vec<SparkProcessSample>> =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(!ranks.is_empty());
        for (rank, samples) in ranks {
            let deltas = spark_census_deltas(&samples)
                .unwrap_or_else(|error| panic!("{rank}: {error}"));
            println!("{rank}: {deltas:?}");
        }
    }
}
