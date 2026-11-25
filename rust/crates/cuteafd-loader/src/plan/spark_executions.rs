//! The one Spark execution admission API used by plan, serve and golden.
//!
//! A family descriptor supplies its decode lanes and admitted decode rows; the
//! installed Spark backend supplies the export ladder; the prefill shape
//! supplies the selected unit and wave count, fixed at startup. The prefill
//! owner covers that admitted unit, never an unused larger ladder shape or
//! the sum of buffered units; waves only buffer per-unit I/O.
use cuteafd_core::{ExecutionAdmissionError, SparkExecutions};

/// Fixed prefill shape: one owner at the covering export for `unit_rows`,
/// with `waves` independent units buffered in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillUnits {
    pub unit_rows: u32,
    pub waves: u8,
}

impl PrefillUnits {
    /// A resolved shape on the shared ladder. The static executor cannot
    /// repartition into a larger unit after startup, so no larger owner is needed.
    pub fn ladder(unit_rows: u64, waves: u64) -> Self {
        Self { unit_rows: u32::try_from(unit_rows).unwrap_or(u32::MAX),
            waves: u8::try_from(waves.max(1)).unwrap_or(u8::MAX) }
    }
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
    /// V4.1 target native packages use the verifier cover.
    V41,
    /// Native EXLR8: one 64-row decode arena and one 4096-row prefill arena.
    Exlr8,
}

const FP8_EXPORTS: [u32; 6] = [1, 16, 80, 256, 1024, 4096];
/// Native mains are the serving AOT rows; m1/m80 are retained beside them.
const NATIVE_EXPORTS: [u32; 5] = [1, 80, 256, 1024, 4096];

impl SparkBackend {
    /// The backend a Spark worker loads for `catalog`, in the worker's own
    /// selection order (EXLR8 format, then FP8-family, then EXL3, then native).
    pub fn for_catalog(catalog: &crate::OfficialV41Catalog) -> anyhow::Result<Self> {
        let geometry = catalog.routed_experts().geometry()?;
        Ok(Self::for_formats(geometry,
            catalog.format() == crate::plan::checkpoint::CheckpointFormat::Exlr8,
            catalog.fp8().is_some(), catalog.exl3().is_some()))
    }

    fn for_formats(geometry: cuteafd_core::ExpertGeometry, exlr8: bool, fp8: bool, exl3: bool) -> Self {
        if exlr8 { Self::Exlr8 }
        else if fp8 { Self::Fp8 }
        else if exl3 { Self::Exl3(cuteafd_core::Exl3RowPolicy::for_geometry(geometry)) }
        else if geometry.same_shape(&cuteafd_core::ExpertGeometry::DEEPSEEK_V41) { Self::V41 }
        else { Self::Native }
    }

    /// Native MXFP4 backends share resident-header and execution-owner censuses.
    pub(crate) fn is_native_census(self) -> bool {
        matches!(self, Self::Native | Self::V41)
    }

    /// Compiled exports this backend can load for live capacities up to 4096.
    pub fn exports(self) -> Vec<u32> {
        match self {
            Self::Exl3(policy) => policy.capacities(4096).expect("4096 is a valid EXL3 extent"),
            Self::Fp8 => FP8_EXPORTS.to_vec(),
            Self::Native => NATIVE_EXPORTS.to_vec(),
            Self::V41 => cuteafd_core::V41_TARGET_CAPACITIES.map(|n| n as u32).to_vec(),
            Self::Exlr8 => vec![cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS as u32, 4096],
        }
    }

}

/// Admit the typed owners for `family` over the package's `exports` (an
/// installed manifest's capacities, or [`SparkBackend::exports`]). `decode_rows`
/// overrides the descriptor's decode extent only where the family's installed
/// coordinator package fixes it (e.g. a V4 manifest's `decode_rows`). `lanes`
/// is the resolved decode-lane count ([`super::decode_lanes::resolve`]): one
/// Spark decode execution per lane, at most what the family can run.
pub fn admit(family: &dyn crate::plan::Family, exports: &[u32], decode_rows: Option<u32>,
    prefill: PrefillUnits, lanes: u8) -> Result<SparkExecutions, ExecutionAdmissionError> {
    let lanes = family.spark_decode().map_or(0, |_| lanes.clamp(1, family.decode_lanes().max_lanes()));
    let rows = decode_rows.or(family.spark_decode().map(|decode| decode.rows)).unwrap_or(0);
    SparkExecutions::admit(&vec![rows; usize::from(lanes)], prefill.unit_rows,
        prefill.unit_rows, prefill.waves, exports)
}

/// The registered descriptor for a family id (daemons know theirs by id).
pub fn descriptor(family: &str) -> anyhow::Result<&'static dyn crate::plan::Family> {
    crate::plan::family::registry().iter().copied().find(|entry| entry.id() == family)
        .ok_or_else(|| anyhow::anyhow!("Spark execution admission requires a family descriptor for {family}"))
}

/// Admit `family` at its resolved prefill unit and wave count on the shared
/// ladder, with its default decode lanes. Plan, serve and golden all call this.
pub fn admit_family(family: &str, exports: &[u32], decode_rows: Option<u32>,
    unit_rows: u64, waves: u64) -> anyhow::Result<SparkExecutions> {
    let lanes = descriptor(family)?.decode_lanes().default_lanes(1);
    admit_family_lanes(family, exports, decode_rows, unit_rows, waves, lanes)
}

/// [`admit_family`] at a resolved decode-lane count (a family that runs one
/// or two lanes admits one Spark decode execution per resolved lane).
pub fn admit_family_lanes(family: &str, exports: &[u32], decode_rows: Option<u32>,
    unit_rows: u64, waves: u64, lanes: u8) -> anyhow::Result<SparkExecutions> {
    let descriptor = descriptor(family)?;
    Ok(admit(descriptor, exports, decode_rows, PrefillUnits::ladder(unit_rows, waves), lanes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{DecodeExecution, PrefillExecution};

    #[test]
    fn native_census_covers_both_native_capacity_ladders() {
        for backend in [SparkBackend::Native, SparkBackend::V41] {
            assert!(backend.is_native_census());
        }
        for backend in [SparkBackend::Fp8, SparkBackend::Exlr8,
            SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest)] {
            assert!(!backend.is_native_census());
        }
    }

    #[test]
    fn artifact_format_precedes_v41_native_geometry() {
        let geometry = cuteafd_core::ExpertGeometry::DEEPSEEK_V41;
        assert_eq!(SparkBackend::for_formats(geometry, false, true, false), SparkBackend::Fp8);
        assert_eq!(SparkBackend::for_formats(geometry, false, false, true),
            SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::for_geometry(geometry)));
        assert_eq!(SparkBackend::for_formats(geometry, true, true, true), SparkBackend::Exlr8);
        assert_eq!(SparkBackend::for_formats(geometry, false, false, false), SparkBackend::V41);
    }

    #[test]
    fn every_descriptor_admits_one_prefill_owner_at_the_selected_unit_export() {
        for family in crate::plan::family::registry().iter().filter(|family| family.spark_decode().is_some()) {
            for waves in 1..=crate::serving_capacity::prefill_waves::MAX_WAVES {
                let unit = crate::serving_capacity::prefill_waves::unit_rows(waves);
                for backend in [SparkBackend::Fp8, SparkBackend::Native, SparkBackend::Exlr8,
                    SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest),
                    SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::GlmFlashK64)] {
                    let lanes = family.decode_lanes().max_lanes();
                    let plan = admit(*family, &backend.exports(), None, PrefillUnits::ladder(unit, waves), lanes).unwrap();
                    assert_eq!(plan.prefill, PrefillExecution { capacity: unit as u32,
                        compiled_capacity: cuteafd_core::covering_export(&backend.exports(), unit as u32).unwrap(),
                        unit_rows: unit as u32, buffered_units: waves as u8 }, "{} {backend:?}", family.id());
                    assert_eq!(plan.decode.len(), usize::from(family.decode_lanes().max_lanes()));
                }
            }
        }
    }

    #[test]
    fn static_prefill_owner_preserves_default_exports_and_shrinks_narrow_pins() {
        for backend in [SparkBackend::Fp8, SparkBackend::Native,
            SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest)] {
            for (unit, waves) in [(4096, 2), (2752, 3), (256, 3)] {
                let compiled = match backend {
                    SparkBackend::Exl3(_) => unit as u32,
                    _ if unit == 256 => 256,
                    _ => 4096,
                };
                let plan = admit_family("glm5", &backend.exports(), None, unit, waves).unwrap();
                assert_eq!(plan.prefill, PrefillExecution { capacity: unit as u32,
                    compiled_capacity: compiled, unit_rows: unit as u32, buffered_units: waves as u8 });
                let owner = plan.owners().last().unwrap();
                assert_eq!((owner.capacity, owner.compiled, owner.staging_rows, owner.buffered_units),
                    (unit as u32, compiled, unit as u32, waves as u8));
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
            let plan = admit(glm, &backend.exports(), Some(rows), PrefillUnits::ladder(2752, 3), 1).unwrap();
            assert_eq!(plan.decode, [DecodeExecution { capacity: rows, compiled_capacity: compiled }], "{backend:?}");
        }
        // V4.1's unit selects its covering native main.
        let v41 = admit(&crate::plan::families::deepseek::DEEPSEEK_V41, &SparkBackend::V41.exports(), None,
            PrefillUnits { unit_rows: 256, waves: 2 }, 2).unwrap();
        assert_eq!(v41.decode, [DecodeExecution { capacity: 64, compiled_capacity: 64 }; 2]);
        // One resolved lane admits one decode execution; a one-lane family never two.
        let one = admit(&crate::plan::families::deepseek::DEEPSEEK_V41, &SparkBackend::V41.exports(), None,
            PrefillUnits { unit_rows: 256, waves: 2 }, 1).unwrap();
        assert_eq!(one.decode, [DecodeExecution { capacity: 64, compiled_capacity: 64 }]);
        assert_eq!(one.prefill, v41.prefill);
        assert_eq!(admit(glm, &SparkBackend::Fp8.exports(), None, PrefillUnits::ladder(2752, 3), 2).unwrap().decode.len(), 1);
        assert_eq!(v41.prefill, PrefillExecution { capacity: 256, compiled_capacity: 256, unit_rows: 256, buffered_units: 2 });
    }

    #[test]
    fn uncovered_decode_capacity_and_over_capacity_pins_are_typed_refusals() {
        let glm = &crate::plan::families::glm::GLM5;
        let ladder = PrefillUnits::ladder;
        // A decode extent beyond the package's largest export is refused.
        assert_eq!(admit(glm, &[1, 80, 4096], Some(4097), ladder(4096, 1), 1).unwrap_err(),
            ExecutionAdmissionError::Uncovered { owner: "decode", rows: 4097, exports: vec![1, 80, 4096] });
        // EXLR8 covers a wider decode extent only with its 4096-row arena.
        assert_eq!(admit(glm, &SparkBackend::Exlr8.exports(), Some(80), ladder(4096, 1), 1).unwrap().decode[0].compiled_capacity, 4096);
        // Only the selected unit needs a covering export, not unused larger shapes.
        assert_eq!(admit(glm, &[1, 80, 2048], None, ladder(2048, 4), 1).unwrap().prefill.compiled_capacity, 2048);
        assert!(matches!(admit(glm, &[1, 80, 2048], None, ladder(2752, 3), 1),
            Err(ExecutionAdmissionError::Uncovered { owner: "prefill", rows: 2752, .. })));
        // An over-capacity unit pin needs an export beyond the ladder.
        assert!(matches!(admit(glm, &SparkBackend::Fp8.exports(), None, ladder(4160, 2), 1),
            Err(ExecutionAdmissionError::Uncovered { owner: "prefill", rows: 4160, .. })));
        assert!(matches!(admit(glm, &SparkBackend::Fp8.exports(), None, ladder(2048, 5), 1),
            Err(ExecutionAdmissionError::BufferedUnits { units: 5, .. })));
        assert!(admit_family("unknown", &SparkBackend::Fp8.exports(), None, 4096, 1).is_err());
        assert!(matches!(admit(&crate::plan::families::dflash::DFLASH2, &SparkBackend::Fp8.exports(), None, ladder(4096, 1), 1),
            Err(ExecutionAdmissionError::DecodeLanes { lanes: 0, .. })));
    }
}
