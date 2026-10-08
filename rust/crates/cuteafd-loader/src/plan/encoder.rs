//! Pure cold-component placement. Budgets already exclude experts and fixed LM state.
use super::MediaMode;
use serde::Serialize;

const GIB: u64 = 1 << 30;

/// SM121 tower free-memory delta was 2,670,125,056 bytes for a 1,587,976,256-byte
/// ledger. Reserve 1.25 GiB for CUDA context/modules (measured overhead + margin).
pub const V41_SPARK_CUDA_OVERHEAD_BYTES: u64 = 5 * GIB / 4;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EncoderKind {
    Off,
    Rtx { gpu: usize },
    Spark { rank: usize },
    SparkIdle { host: String },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EncoderPlacement {
    pub kind: EncoderKind,
    pub weights: u64,
    pub scratch: u64,
    /// Additional Spark copies; `kind` is the first replica.
    pub replicas: Vec<usize>,
    pub reason: String,
    pub shortfall: u64,
}
impl EncoderPlacement {
    pub fn admitted_bytes(&self) -> u64 {
        if self.kind == EncoderKind::Off { 0 } else { self.weights.saturating_add(self.scratch) }
    }
    pub fn spark_ranks(&self) -> Vec<usize> {
        match self.kind {
            EncoderKind::Spark { rank } => std::iter::once(rank).chain(self.replicas.iter().copied()).collect(),
            _ => Vec::new(),
        }
    }
}
#[derive(Clone, Debug)]
pub struct EncoderGpuBudget {
    pub free_bytes: u64,
    /// Reservation for the requested KV target, not the opportunistic final pool.
    pub kv_target_bytes: u64,
}
#[derive(Clone, Debug)]
pub struct EncoderSparkBudget {
    pub rank: usize,
    pub host: String,
    pub idle: bool,
    pub expert_bytes: u64,
    pub free_bytes: u64,
}
#[derive(Clone, Debug)]
pub struct EncoderHardware {
    pub v41: bool,
    pub gpus: Vec<EncoderGpuBudget>,
    pub sparks: Vec<EncoderSparkBudget>,
}

/// D1 is intentionally one policy line; retain Spark default until live gates pass.
pub fn default_encoder_placement(hardware: &EncoderHardware, weights: u64, scratch: u64) -> EncoderPlacement {
    encoder_placement(MediaMode::Auto, hardware, weights, scratch, 1)
}

pub fn encoder_placement(mode: MediaMode, hardware: &EncoderHardware, weights: u64, scratch: u64, copies: usize) -> EncoderPlacement {
    let required = weights.saturating_add(scratch);
    let off = |reason: &str, shortfall| EncoderPlacement {
        kind: EncoderKind::Off, weights: 0, scratch: 0, replicas: vec![], reason: reason.into(), shortfall,
    };
    if mode == MediaMode::Off || weights == 0 { return off("disabled or no checkpoint tower", 0); }
    let spark_overhead = if hardware.v41 { V41_SPARK_CUDA_OVERHEAD_BYTES } else { 0 };
    let spark_required = required.saturating_add(spark_overhead);
    let make = |kind, replicas, reason: &str| {
        let overhead = if matches!(&kind, EncoderKind::Spark { .. } | EncoderKind::SparkIdle { .. }) { spark_overhead } else { 0 };
        EncoderPlacement {
            kind, weights, scratch: scratch.saturating_add(overhead), replicas, reason: reason.into(), shortfall: 0,
        }
    };
    let mut sparks: Vec<_> = hardware.sparks.iter()
        .filter(|s| s.free_bytes >= spark_required.saturating_add(GIB))
        .collect();
    sparks.sort_by_key(|s| (!s.idle, s.expert_bytes, std::cmp::Reverse(s.free_bytes), s.rank));
    if matches!(mode, MediaMode::Auto | MediaMode::Spark(_)) {
        if let MediaMode::Spark(Some(rank)) = mode {
            if let Some(index) = sparks.iter().position(|s| s.rank == rank) { sparks.swap(0, index); }
            else { sparks.clear(); }
        }
        if sparks.len() >= copies.max(1) {
            let first = sparks[0];
            if first.idle && copies <= 1 {
                return make(EncoderKind::SparkIdle { host: first.host.clone() }, vec![], "idle Spark: no expert interference");
            }
            return make(EncoderKind::Spark { rank: first.rank }, sparks.iter().skip(1).take(copies.saturating_sub(1)).map(|s| s.rank).collect(),
                "Spark: lightest expert slice, then most free memory; tower + scratch + 1 GiB admitted");
        }
        if matches!(mode, MediaMode::Spark(_)) {
            let room = hardware.sparks.iter().map(|s| s.free_bytes).max().unwrap_or(0);
            return off("requested Spark encoder/replicas unavailable", spark_required.saturating_add(GIB).saturating_sub(room));
        }
    }
    let preferred = match mode {
        MediaMode::Rtx(Some(gpu)) => gpu,
        _ if hardware.v41 => 0,
        _ => hardware.gpus.len().saturating_sub(1),
    };
    if let Some(gpu) = hardware.gpus.get(preferred) {
        if gpu.free_bytes >= required.saturating_add(gpu.kv_target_bytes) {
            return make(EncoderKind::Rtx { gpu: preferred }, vec![], "RTX: KV target preserved after tower + scratch admission");
        }
        return off("tower would reduce the requested KV target", required.saturating_add(gpu.kv_target_bytes).saturating_sub(gpu.free_bytes));
    }
    off("requested RTX device absent", required)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hw(gpus: usize, sparks: usize, gib: u64) -> EncoderHardware {
        EncoderHardware { v41: false, gpus: (0..gpus).map(|_| EncoderGpuBudget { free_bytes: gib * GIB, kv_target_bytes: 8 * GIB }).collect(),
            sparks: (0..sparks).map(|rank| EncoderSparkBudget { rank, host: format!("spark{rank}"), idle: false, expert_bytes: (10 - rank as u64) * GIB, free_bytes: 20 * GIB }).collect() }
    }
    #[test]
    fn g9_hardware_matrix_and_fallback() {
        for (gpus, gib) in [(1, 96), (2, 96), (1, 32)] {
            assert_eq!(default_encoder_placement(&hw(gpus, 2, gib), 2 * GIB, GIB).kind, EncoderKind::Spark { rank: 1 });
            assert_eq!(default_encoder_placement(&hw(gpus, 0, gib), 2 * GIB, GIB).kind, EncoderKind::Rtx { gpu: gpus - 1 });
        }
        let mut h = hw(2, 2, 96);
        for s in &mut h.sparks { s.free_bytes = GIB; }
        assert_eq!(default_encoder_placement(&h, 2 * GIB, GIB).kind, EncoderKind::Rtx { gpu: 1 });
        h.gpus[1].free_bytes = 9 * GIB;
        let off = default_encoder_placement(&h, 2 * GIB, GIB);
        assert_eq!(off.kind, EncoderKind::Off);
        assert_eq!(off.shortfall, 2 * GIB);
        h.v41 = true;
        assert_eq!(default_encoder_placement(&h, 2 * GIB, GIB).kind, EncoderKind::Rtx { gpu: 0 });
        h.sparks[0].free_bytes = 20 * GIB;
        assert_eq!(default_encoder_placement(&h, 2 * GIB, GIB).kind, EncoderKind::Spark { rank: 0 });
    }
    #[test]
    fn v41_spark_charges_measured_cuda_overhead_before_guard() {
        let mut h = hw(1, 1, 96);
        h.v41 = true;
        let ledger = 1_587_976_256;
        let admitted = ledger + V41_SPARK_CUDA_OVERHEAD_BYTES;
        h.sparks[0].free_bytes = admitted + GIB - 1;
        assert_eq!(default_encoder_placement(&h, ledger, 0).kind, EncoderKind::Rtx { gpu: 0 });
        assert_eq!(encoder_placement(MediaMode::Spark(None), &h, ledger, 0, 1).shortfall, 1);
        h.sparks[0].free_bytes += 1;
        let spark = default_encoder_placement(&h, ledger, 0);
        assert_eq!(spark.kind, EncoderKind::Spark { rank: 0 });
        assert_eq!(spark.admitted_bytes(), admitted);
        assert!(admitted > 2_670_125_056);
    }
    #[test]
    fn independent_audio_placement_sees_vision_reservation_and_preserves_kv() {
        let mut h = hw(2, 2, 96);
        for spark in &mut h.sparks { spark.expert_bytes = 80 * GIB; }
        let vision = encoder_placement(MediaMode::Auto, &h, 2 * GIB, GIB, 1);
        assert_eq!(vision.kind, EncoderKind::Spark { rank: 0 });
        h.sparks[0].free_bytes -= vision.admitted_bytes();
        let audio = encoder_placement(MediaMode::Auto, &h, 3 * GIB, 2 * GIB, 1);
        assert_eq!(audio.kind, EncoderKind::Spark { rank: 1 });
        assert_eq!(encoder_placement(MediaMode::Off, &h, 3 * GIB, 2 * GIB, 1).admitted_bytes(), 0);
        for spark in &mut h.sparks { spark.free_bytes = 6 * GIB - 1; }
        assert_eq!(encoder_placement(MediaMode::Auto, &h, 3 * GIB, 2 * GIB, 1).kind, EncoderKind::Rtx { gpu: 1 });
        h.gpus[1].free_bytes = 13 * GIB - 1;
        assert_eq!(encoder_placement(MediaMode::Auto, &h, 3 * GIB, 2 * GIB, 1).shortfall, 1);
        assert_eq!(encoder_placement(MediaMode::Spark(None), &h, 3 * GIB, 2 * GIB, 1).shortfall, 1);
    }
    #[test]
    fn idle_explicit_replicas_and_off() {
        let mut h = hw(2, 4, 96);
        h.sparks[0].idle = true;
        assert_eq!(default_encoder_placement(&h, GIB, GIB).kind, EncoderKind::SparkIdle { host: "spark0".into() });
        assert_eq!(encoder_placement(MediaMode::Spark(Some(2)), &h, GIB, GIB, 1).kind, EncoderKind::Spark { rank: 2 });
        let p = encoder_placement(MediaMode::Auto, &h, GIB, GIB, 4);
        assert_eq!(p.spark_ranks(), vec![0, 3, 2, 1]);
        assert_eq!(encoder_placement(MediaMode::Off, &h, GIB, GIB, 1).admitted_bytes(), 0);
        assert_eq!(encoder_placement(MediaMode::Rtx(Some(5)), &h, GIB, GIB, 1).kind, EncoderKind::Off);
    }
}
