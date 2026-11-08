//! Runner-declared external CUDA context: not an engine owner or usable capacity.
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

fn parse(text: &str) -> Result<Option<BTreeMap<i32, usize>>> {
    let value: serde_json::Value = serde_json::from_str(text).context("invalid observer context JSON")?;
    if value == serde_json::json!({"pending": true}) { return Ok(None); }
    let rows = value.as_object().context("observer context must be a device-to-bytes object")?;
    anyhow::ensure!(!rows.is_empty(), "observer context has no GPUs");
    let mut devices = BTreeMap::new();
    for (key, value) in rows {
        let device: i32 = key.parse().context("invalid observer CUDA ordinal")?;
        anyhow::ensure!(device >= 0 && key == &device.to_string(), "invalid observer CUDA ordinal {key}");
        let bytes = usize::try_from(value.as_u64().context("observer bytes must be an unsigned integer")?)?;
        anyhow::ensure!(bytes > 0, "observer bytes must be measured and positive");
        devices.insert(device, bytes);
    }
    Ok(Some(devices))
}

fn read(timeout: Duration, mut contents: impl FnMut() -> Result<String>) -> Result<BTreeMap<i32, usize>> {
    let start = Instant::now();
    loop {
        if let Some(devices) = parse(&contents()?)? { return Ok(devices); }
        anyhow::ensure!(start.elapsed() < timeout, "timed out waiting for measured observer context");
        std::thread::sleep(Duration::from_millis(50));
    }
}

pub(crate) fn bytes(device: i32) -> Result<usize> {
    static DEVICES: OnceLock<std::result::Result<Option<BTreeMap<i32, usize>>, String>> = OnceLock::new();
    let devices = DEVICES.get_or_init(|| {
        (|| -> Result<_> {
            let Some(path) = std::env::var_os("CUTEAFD_OBSERVER_CONTEXT_FILE") else { return Ok(None); };
            let devices = read(Duration::from_secs(30), || std::fs::read_to_string(&path)
                .with_context(|| format!("reading observer context {}", std::path::Path::new(&path).display())))?;
            for (&device, &bytes) in &devices {
                tracing::info!(device, observer_context_bytes = bytes, "excluded external observer context");
            }
            Ok(Some(devices))
        })().map_err(|error| format!("{error:#}"))
    }).as_ref().map_err(|error| anyhow::anyhow!("{error}"))?;
    match devices {
        None => Ok(0),
        Some(devices) => devices.get(&device).copied()
            .with_context(|| format!("observer context missing CUDA ordinal {device}")),
    }
}

pub(crate) fn exclude(free: usize, total: usize, observer: usize) -> Result<(usize, usize)> {
    if observer == 0 { return Ok((free, total)); }
    let total = total.checked_sub(observer).context("observer context exceeds GPU capacity")?;
    anyhow::ensure!(free <= total, "observer context exceeds measured GPU usage");
    // Used = total - free excludes the observer, while total also reserves its
    // physical bytes. A smaller explicit cap can use existing physical headroom.
    Ok((free, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_observer_preserves_context_and_reserves_physical_capacity() {
        let (total, free, tracked, observer) = (10_000, 7_000, 2_000, 550);
        let production_context = total - free - tracked;
        let (observed_free, observed_total) = exclude(free - observer, total, observer).unwrap();
        assert_eq!(observed_total - observed_free - tracked, production_context);
        assert_eq!(observed_total, total - observer);
        assert_eq!(observed_free, free - observer);
        assert_eq!(exclude(free, total, 0).unwrap(), (free, total));
        let cap = cuteafd_core::serving_capacity::GpuMemoryBudget(8_000);
        let sample = |total, free| cuteafd_core::serving_capacity::DeviceMemory {
            device: 0, total_bytes: total as u64, baseline_free_bytes: free as u64,
        };
        assert_eq!(cap.apply(sample(observed_total, observed_free)).unwrap(), cap.apply(sample(total, free)).unwrap());
    }

    #[test]
    fn observer_declaration_is_measured_per_logical_gpu_and_fails_closed() {
        assert!(parse(r#"{"pending":true}"#).unwrap().is_none());
        assert_eq!(parse(r#"{"0":123,"1":456}"#).unwrap().unwrap(), BTreeMap::from([(0, 123), (1, 456)]));
        for bad in ["{}", "null", r#"{"0":0}"#, r#"{"0":-1}"#, r#"{"0":1.5}"#, r#"{"-1":5}"#, r#"{"00":5}"#] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        assert!(exclude(10, 100, 101).is_err());
        assert!(exclude(99, 100, 2).is_err());
    }

    #[test]
    fn pending_observer_times_out_instead_of_guessing() {
        let error = read(Duration::ZERO, || Ok(r#"{"pending":true}"#.into())).unwrap_err();
        assert!(error.to_string().contains("timed out"));
        assert_eq!(read(Duration::ZERO, || Ok(r#"{"0":123}"#.into())).unwrap(), BTreeMap::from([(0, 123)]));
    }
}
