use serde::Serialize;

/// The `/v1/stats` media object; embedding bytes include admitted in-flight reservations.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MediaStats {
    pub encodes: u64,
    pub submit_waits: u64,
    pub submit_timeouts: u64,
    pub encode_timeouts: u64,
    pub encode_ms_p50: f64,
    pub encode_ms_p99: f64,
    pub cache_hits: u64,
    pub cache_bytes: usize,
    pub memo_hits: u64,
    pub prefix_skipped_images: u64,
    pub media_key_collisions: u64,
    pub pending: usize,
}

/// A bounded rolling sample window, independent of request volume.
#[derive(Default)]
pub(crate) struct Latencies {
    samples: Vec<f64>,
    next: usize,
}
impl Latencies {
    pub fn record(&mut self, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        if self.samples.len() < 1024 {
            self.samples.push(ms);
        } else {
            self.samples[self.next] = ms;
        }
        self.next = (self.next + 1) % 1024;
    }
    pub fn percentiles(&self) -> (f64, f64) {
        let mut sorted = self.samples.clone();
        sorted.sort_by(f64::total_cmp);
        let p = |percent: usize| {
            if sorted.is_empty() {
                0.0
            } else {
                sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)]
            }
        };
        (p(50), p(99))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_window_is_bounded_and_ignores_nonfinite_or_negative_samples() {
        let mut latencies = Latencies::default();
        assert_eq!(latencies.percentiles(), (0.0, 0.0));
        for ms in [f64::NAN, f64::INFINITY, -1.0] {
            latencies.record(ms);
        }
        assert!(latencies.samples.is_empty());
        for _ in 0..1024 {
            latencies.record(1.0);
        }
        for _ in 0..1024 {
            latencies.record(2.0);
        }
        assert_eq!(latencies.samples.len(), 1024);
        assert_eq!(latencies.percentiles(), (2.0, 2.0));
        assert!(serde_json::to_value(MediaStats::default()).is_ok());
    }
}
