//! The one Spark execution admission API used by plan, serve and golden.
//!
//! A family descriptor supplies its decode lanes and admitted decode rows; the
//! installed Spark backend supplies the export ladder; the prefill shape
//! supplies the selected unit and wave count. The prefill owner is sized once
//! at the ladder's largest unit (the reserve divided by one), never multiplied
//! by the wave count; waves only buffer per-unit I/O.
use cuteafd_core::{ExecutionAdmissionError, SparkExecutions};

/// The prefill owner's extents: scratch at `max_unit_rows` once, `waves`
/// units of `unit_rows` buffered in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillUnits {
    pub max_unit_rows: u32,
    pub unit_rows: u32,
    pub waves: u8,
}

impl PrefillUnits {
    /// The family's prefill owner at the resolved unit and wave count.
    pub fn for_family(family: &dyn crate::plan::Family, unit_rows: u64, waves: u64) -> Self {
        match family.spark_prefill() {
            SparkPrefill::Ladder => Self::ladder(family, unit_rows, waves),
            SparkPrefill::Streams(streams) => { let unit = u32::try_from(unit_rows).unwrap_or(u32::MAX);
                Self { max_unit_rows: unit, unit_rows: unit, waves: streams } }
        }
    }

    /// A shape on the shared ladder: the largest unit is the reserve divided
    /// by the fewest waves that split it (8192 / 2 today), bounded by the
    /// family's executor unit limit.
    pub fn ladder(family: &dyn crate::plan::Family, unit_rows: u64, waves: u64) -> Self {
        let limit = family.prefill_limits().max_unit_rows;
        let maximum = crate::serving_capacity::prefill_waves::unit_rows(1).min(limit).max(unit_rows);
        Self { max_unit_rows: u32::try_from(maximum).unwrap_or(u32::MAX),
            unit_rows: u32::try_from(unit_rows).unwrap_or(u32::MAX),
            waves: u8::try_from(waves.max(1)).unwrap_or(u8::MAX) }
    }
}

/// How a family's executor feeds the prefill owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkPrefill {
    /// The shared ladder: scratch at the reserve's largest unit, k units in flight.
    Ladder,
    /// A legacy executor with fixed parallel streams of its own live chunk
    /// (V4.1 CED): scratch at that chunk, `streams` units in flight.
    Streams(u8),
}

/// Rows each of the family's decode compute owners admits (one owner per
/// decode lane the family can run, `Family::decode_lanes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkDecode {
    pub rows: u32,
}

/// The installed Spark expert backend, which fixes the export ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkBackend {
    /// EXL3 trellis packages (capacities per the geometry's row policy).
    Exl3(cuteafd_core::Exl3RowPolicy),
    /// FP8/NVFP4/MXFP4 grouped packages (`fp8-*`).
    Fp8,
    /// Native V4.1 packages: main export plus m1/m80.
    Native,
    /// Native EXLR8: one 64-row decode arena and one 4096-row prefill arena.
    Exlr8,
}

const FP8_EXPORTS: [u32; 6] = [1, 16, 80, 256, 1024, 4096];
/// Native mains are the serving AOT rows; m1/m80 are retained beside them.
const NATIVE_EXPORTS: [u32; 5] = [1, 80, 256, 1024, 4096];

impl SparkBackend {
    /// The backend a Spark worker loads for `catalog`, in the worker's own
    /// selection order (EXLR8 format, then EXL3, then FP8-family, then native).
    pub fn for_catalog(catalog: &crate::OfficialV41Catalog) -> anyhow::Result<Self> {
        let geometry = catalog.routed_experts().geometry()?;
        Ok(if catalog.format() == crate::plan::checkpoint::CheckpointFormat::Exlr8 { Self::Exlr8 }
            else if catalog.exl3().is_some() { Self::Exl3(cuteafd_core::Exl3RowPolicy::for_geometry(geometry)) }
            else if catalog.fp8().is_some() { Self::Fp8 } else { Self::Native })
    }

    /// Compiled exports this backend can load for live capacities up to 4096.
    pub fn exports(self) -> Vec<u32> {
        match self {
            Self::Exl3(policy) => policy.capacities(4096).expect("4096 is a valid EXL3 extent"),
            Self::Fp8 => FP8_EXPORTS.to_vec(),
            Self::Native => NATIVE_EXPORTS.to_vec(),
            Self::Exlr8 => vec![cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS as u32, 4096],
        }
    }

    /// Pinned route staging retained per queued unit of `rows` (ownership-tagged
    /// native/EXL3 routes); FP8 and EXLR8 read retained ring frames directly.
    pub fn route_staging_bytes(self, rows: u32, topk: u32) -> u64 {
        match self {
            Self::Exl3(_) | Self::Native => u64::from(rows) * u64::from(topk) * 8,
            Self::Fp8 | Self::Exlr8 => 0,
        }
    }
}

/// Admit the typed owners for `family` over the package's `exports` (an
/// installed manifest's capacities, or [`SparkBackend::exports`]). `decode_rows`
/// overrides the descriptor's decode extent only where the family's installed
/// coordinator package fixes it (e.g. a V4 manifest's `decode_rows`).
pub fn admit(family: &dyn crate::plan::Family, exports: &[u32], decode_rows: Option<u32>,
    prefill: PrefillUnits) -> Result<SparkExecutions, ExecutionAdmissionError> {
    let lanes = family.spark_decode().map_or(0, |_| family.decode_lanes().max_lanes());
    let rows = decode_rows.or(family.spark_decode().map(|decode| decode.rows)).unwrap_or(0);
    SparkExecutions::admit(&vec![rows; usize::from(lanes)], prefill.max_unit_rows,
        prefill.unit_rows, prefill.waves, exports)
}

/// The registered descriptor for a family id (daemons know theirs by id).
pub fn descriptor(family: &str) -> anyhow::Result<&'static dyn crate::plan::Family> {
    crate::plan::family::registry().iter().copied().find(|entry| entry.id() == family)
        .ok_or_else(|| anyhow::anyhow!("Spark execution admission requires a family descriptor for {family}"))
}

/// Admit `family` at its resolved prefill unit and wave count; the descriptor
/// decides the prefill owner's shape ([`PrefillUnits::for_family`]). Plan,
/// serve and golden all call this.
pub fn admit_family(family: &str, exports: &[u32], decode_rows: Option<u32>,
    unit_rows: u64, waves: u64) -> anyhow::Result<SparkExecutions> {
    let descriptor = descriptor(family)?;
    Ok(admit(descriptor, exports, decode_rows, PrefillUnits::for_family(descriptor, unit_rows, waves))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{DecodeExecution, PrefillExecution};

    #[test]
    fn every_descriptor_admits_one_prefill_owner_at_the_ladder_maximum() {
        for family in crate::plan::family::registry().iter().filter(|family| family.spark_decode().is_some()) {
            for waves in 1..=crate::serving_capacity::prefill_waves::MAX_WAVES {
                let unit = crate::serving_capacity::prefill_waves::unit_rows(waves);
                for backend in [SparkBackend::Fp8, SparkBackend::Native, SparkBackend::Exlr8,
                    SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest),
                    SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::GlmFlashK64)] {
                    let plan = admit(*family, &backend.exports(), None, PrefillUnits::ladder(*family, unit, waves)).unwrap();
                    assert_eq!(plan.prefill, PrefillExecution { capacity: 4096, compiled_capacity: 4096,
                        unit_rows: unit as u32, buffered_units: waves as u8 }, "{} {backend:?}", family.id());
                    assert_eq!(plan.decode.len(), usize::from(family.decode_lanes().max_lanes()));
                }
            }
        }
    }

    #[test]
    fn decode_compiled_capacity_follows_the_backend_ladder() {
        let glm = &crate::plan::families::glm::GLM5;
        let k64 = SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::GlmFlashK64);
        for (backend, rows, compiled) in [(SparkBackend::Fp8, 64, 80), (SparkBackend::Native, 64, 80),
            (SparkBackend::Exlr8, 64, 64), (k64, 16, 80), (k64, 80, 80),
            (SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest), 16, 16),
            (SparkBackend::Fp8, 128, 256), (SparkBackend::Native, 128, 256)] {
            let plan = admit(glm, &backend.exports(), Some(rows), PrefillUnits::ladder(glm, 2752, 3)).unwrap();
            assert_eq!(plan.decode, [DecodeExecution { capacity: rows, compiled_capacity: compiled }], "{backend:?}");
        }
        // A legacy executor's own maximum unit selects its covering native main.
        let v41 = admit(&crate::plan::families::deepseek::DEEPSEEK_V41, &SparkBackend::Native.exports(), None,
            PrefillUnits { max_unit_rows: 256, unit_rows: 256, waves: 2 }).unwrap();
        assert_eq!(v41.decode, [DecodeExecution { capacity: 64, compiled_capacity: 80 }; 2]);
        assert_eq!(v41.prefill, PrefillExecution { capacity: 256, compiled_capacity: 256, unit_rows: 256, buffered_units: 2 });
    }

    #[test]
    fn uncovered_decode_capacity_and_over_capacity_pins_are_typed_refusals() {
        let glm = &crate::plan::families::glm::GLM5;
        let ladder = |unit, waves| PrefillUnits::ladder(glm, unit, waves);
        // A decode extent beyond the package's largest export is refused.
        assert_eq!(admit(glm, &[1, 80, 4096], Some(4097), ladder(4096, 1)).unwrap_err(),
            ExecutionAdmissionError::Uncovered { owner: "decode", rows: 4097, exports: vec![1, 80, 4096] });
        // EXLR8 covers a wider decode extent only with its 4096-row arena.
        assert_eq!(admit(glm, &SparkBackend::Exlr8.exports(), Some(80), ladder(4096, 1)).unwrap().decode[0].compiled_capacity, 4096);
        // A package without a covering prefill export is refused, not stretched.
        assert!(matches!(admit(glm, &[1, 80, 2048], None, ladder(2048, 4)),
            Err(ExecutionAdmissionError::Uncovered { owner: "prefill", rows: 4096, .. })));
        // An over-capacity unit pin needs an export beyond the ladder.
        assert!(matches!(admit(glm, &SparkBackend::Fp8.exports(), None, ladder(4160, 2)),
            Err(ExecutionAdmissionError::Uncovered { owner: "prefill", rows: 4160, .. })));
        assert!(matches!(admit(glm, &SparkBackend::Fp8.exports(), None, ladder(2048, 5)),
            Err(ExecutionAdmissionError::BufferedUnits { units: 5, .. })));
        assert!(admit_family("unknown", &SparkBackend::Fp8.exports(), None, 4096, 1).is_err());
        assert!(matches!(admit(&crate::plan::families::dflash::DFLASH2, &SparkBackend::Fp8.exports(), None, ladder(4096, 1)),
            Err(ExecutionAdmissionError::DecodeLanes { lanes: 0, .. })));
    }
}
