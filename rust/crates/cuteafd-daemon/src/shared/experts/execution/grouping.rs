//! Startup-only package reads; queries mirror the worker's installed selectors.
use crate::shared::experts::exl3::execution::{Exl3RowPolicy, Exl3Schedule};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
#[error("expert grouping package {package}: {reason}")]
pub(crate) struct ExpertGroupingError {
    package: PathBuf,
    reason: String,
}

#[derive(Debug, Deserialize)]
struct NativeVariant {
    capacity_rows: usize,
    group_rows: usize,
    implementation: Option<String>,
    route_mode: Option<String>,
    compact_live_rows: Option<usize>,
    compact_group_rows: Option<usize>,
}
impl NativeVariant {
    fn validate(&self) -> Result<(), &'static str> {
        if self.capacity_rows == 0 || !matches!(self.group_rows, 1 | 8 | 16 | 32 | 64 | 128) {
            return Err("invalid native capacity/group_rows");
        }
        match (self.implementation.as_deref(), self.route_mode.as_deref()) {
            (Some("slices"), None)
                if self.compact_live_rows.is_none() && self.compact_group_rows.is_none() =>
            {
                Ok(())
            }
            (Some("hybrid"), None)
                if self
                    .compact_live_rows
                    .is_some_and(|n| n > 0 && n <= self.capacity_rows)
                    && self.compact_group_rows == Some(1) =>
            {
                Ok(())
            }
            (None, Some("grouped" | "direct"))
                if self.compact_live_rows.is_none()
                    && self.compact_group_rows.is_none()
                    && (self.route_mode.as_deref() != Some("direct") || self.group_rows == 1) =>
            {
                Ok(())
            }
            _ => Err("unknown or inconsistent native grouping implementation"),
        }
    }
    fn group_rows(&self, rows: usize) -> usize {
        if self.compact_live_rows.is_some_and(|cutoff| rows <= cutoff) {
            self.compact_group_rows.expect("validated compact grouping")
        } else {
            self.group_rows
        }
    }
}

#[derive(Debug)]
enum Policy {
    Native,
    Exl3(Exl3RowPolicy),
}

/// An immutable record installed with the workers, not a hot-path file reader.
#[derive(Debug)]
pub(crate) struct InstalledExpertGrouping {
    package: PathBuf,
    capacity: usize,
    policy: Policy,
    variants: Vec<NativeVariant>,
}
impl InstalledExpertGrouping {
    fn error(package: &Path, reason: impl ToString) -> ExpertGroupingError {
        ExpertGroupingError {
            package: package.to_owned(),
            reason: reason.to_string(),
        }
    }
    fn read<T: serde::de::DeserializeOwned>(package: &Path) -> Result<T, ExpertGroupingError> {
        let bytes = std::fs::read(package).map_err(|e| Self::error(package, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Self::error(package, e))
    }

    /// Exact native role directory, with the same configured capacity as expertd.
    pub(crate) fn native(directory: &Path, capacity: usize) -> Result<Self, ExpertGroupingError> {
        #[derive(Deserialize)]
        struct Manifest {
            schema: u32,
            role: String,
            variants: Vec<NativeVariant>,
        }
        let package = directory.join("v41_experts.json");
        let manifest: Manifest = Self::read(&package)?;
        if manifest.schema != 1
            || !matches!(
                manifest.role.as_str(),
                "spark" | "spark_tp2" | "spark_tp3" | "spark_tp6"
            )
            || !(1..=4096).contains(&capacity)
        {
            return Err(Self::error(
                &package,
                "unknown native schema/worker role/capacity",
            ));
        }
        for variant in &manifest.variants {
            variant.validate().map_err(|e| Self::error(&package, e))?;
        }
        let selected = |c| -> Result<NativeVariant, ExpertGroupingError> {
            let mut variants = manifest.variants.iter().filter(|v| v.capacity_rows == c);
            let variant = variants
                .next()
                .ok_or_else(|| Self::error(&package, format!("missing native capacity {c}")))?;
            if variants.next().is_some() {
                return Err(Self::error(
                    &package,
                    format!("duplicate native capacity {c}"),
                ));
            }
            Ok(NativeVariant {
                capacity_rows: variant.capacity_rows,
                group_rows: variant.group_rows,
                implementation: variant.implementation.clone(),
                route_mode: variant.route_mode.clone(),
                compact_live_rows: variant.compact_live_rows,
                compact_group_rows: variant.compact_group_rows,
            })
        };
        let mut variants = vec![selected(capacity)?];
        if capacity > 1 {
            variants.push(selected(1)?);
        }
        if capacity > 80 {
            variants.push(selected(80)?);
        }
        variants.sort_by_key(|v| v.capacity_rows);
        Ok(Self {
            package,
            capacity,
            policy: Policy::Native,
            variants,
        })
    }

    /// Exact EXL3 worker role root (containing scheduled m<capacity> directories).
    pub(crate) fn exl3(
        directory: &Path,
        capacity: usize,
        policy: Exl3RowPolicy,
        schedule: Exl3Schedule,
    ) -> Result<Self, ExpertGroupingError> {
        #[derive(Deserialize)]
        struct Manifest {
            schema: String,
            capacity: usize,
            #[serde(default = "default_route_block")]
            route_block: usize,
            direct: bool,
            decode_schedule: Option<String>,
        }
        fn default_route_block() -> usize {
            8
        }
        schedule
            .validate(policy)
            .map_err(|e| Self::error(directory, e))?;
        let capacities = policy
            .capacities(capacity)
            .map_err(|e| Self::error(directory, e))?;
        let mut variants = Vec::with_capacity(capacities.len());
        for c in capacities {
            let package = schedule.directory(directory, c).join("v41_exl3.json");
            let meta: Manifest = Self::read(&package)?;
            let scheduled = schedule == Exl3Schedule::Gb10 && matches!(c, 1 | 80);
            if meta.schema != "cuteafd.v41-exl3-aot.v1"
                || meta.capacity != c as usize
                || !matches!(meta.route_block, 8 | 16 | 32 | 64)
                || (meta.direct && meta.route_block != 8)
                || meta.decode_schedule.as_deref()
                    != if scheduled {
                        Some(schedule.name())
                    } else {
                        None
                    }
            {
                return Err(Self::error(
                    &package,
                    "unknown or inconsistent EXL3 grouping metadata",
                ));
            }
            variants.push(NativeVariant {
                capacity_rows: c as usize,
                group_rows: meta.route_block,
                implementation: None,
                route_mode: None,
                compact_live_rows: None,
                compact_group_rows: None,
            });
        }
        Ok(Self {
            package: directory.to_owned(),
            capacity,
            policy: Policy::Exl3(policy),
            variants,
        })
    }

    pub(crate) fn group_rows(&self, rows: usize) -> Result<usize, ExpertGroupingError> {
        if rows == 0 {
            return Ok(self
                .variants
                .last()
                .expect("validated installed capacities")
                .group_rows);
        }
        if rows > self.capacity {
            return Err(Self::error(
                &self.package,
                format!("rows {rows} exceed installed capacity {}", self.capacity),
            ));
        }
        let required = match self.policy {
            Policy::Native => rows,
            Policy::Exl3(policy) => policy.required_capacity(rows),
        };
        let variant = self
            .variants
            .iter()
            .find(|v| v.capacity_rows >= required)
            .ok_or_else(|| {
                Self::error(
                    &self.package,
                    format!("no installed capacity for rows {rows}"),
                )
            })?;
        Ok(variant.group_rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn native_fixture(variants: serde_json::Value) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("v41_experts.json"),
            json!({"schema":1,"role":"spark","variants":variants}).to_string(),
        )
        .unwrap();
        root
    }
    #[test]
    fn native_m1_reads_once() {
        let root =
            native_fixture(json!([{"capacity_rows":1,"group_rows":16,"implementation":"slices"}]));
        let installed = InstalledExpertGrouping::native(root.path(), 1).unwrap();
        std::fs::remove_file(root.path().join("v41_experts.json")).unwrap();
        assert_eq!(installed.group_rows(1).unwrap(), 16);
        assert_eq!(installed.group_rows(0).unwrap(), 16);
        assert!(installed.group_rows(2).is_err());
    }
    #[test]
    fn native_m80_uses_worker_small_selector_not_nearest_m16() {
        let root = native_fixture(json!([
            {"capacity_rows":1,"group_rows":1,"route_mode":"direct"},
            {"capacity_rows":16,"group_rows":16,"implementation":"slices"},
            {"capacity_rows":80,"group_rows":32,"route_mode":"grouped"},
            {"capacity_rows":256,"group_rows":64,"route_mode":"grouped"}]));
        let installed = InstalledExpertGrouping::native(root.path(), 256).unwrap();
        assert_eq!(installed.group_rows(0).unwrap(), 64);
        for (rows, group) in [(1, 1), (2, 32), (16, 32), (80, 32), (81, 64), (256, 64)] {
            assert_eq!(installed.group_rows(rows).unwrap(), group);
        }
    }
    #[test]
    fn native_hybrid_both_sides_of_compact_threshold() {
        let root = native_fixture(json!([
            {"capacity_rows":1,"group_rows":16,"implementation":"hybrid","compact_live_rows":1,"compact_group_rows":1},
            {"capacity_rows":80,"group_rows":16,"implementation":"hybrid","compact_live_rows":8,"compact_group_rows":1}]));
        let installed = InstalledExpertGrouping::native(root.path(), 80).unwrap();
        for (rows, group) in [(0, 16), (1, 1), (2, 1), (8, 1), (9, 16), (80, 16)] {
            assert_eq!(installed.group_rows(rows).unwrap(), group);
        }
    }
    #[test]
    fn unknown_missing_and_incomplete_native_metadata_name_package() {
        for variant in [
            json!({"capacity_rows":1}),
            json!({"capacity_rows":1,"group_rows":3,"implementation":"slices"}),
            json!({"capacity_rows":1,"group_rows":16,"implementation":"unknown"}),
            json!({"capacity_rows":1,"group_rows":16,"implementation":"hybrid","compact_live_rows":1}),
        ] {
            let root = native_fixture(json!([variant]));
            let error = InstalledExpertGrouping::native(root.path(), 1).unwrap_err();
            assert!(error
                .to_string()
                .contains(&root.path().display().to_string()));
        }
        let root = tempfile::tempdir().unwrap();
        assert!(InstalledExpertGrouping::native(root.path(), 1)
            .unwrap_err()
            .to_string()
            .contains("v41_experts.json"));
    }
    fn exl3_fixture(root: &Path, capacity: u32, block: Option<usize>, schedule: Exl3Schedule) {
        let directory = schedule.directory(root, capacity);
        std::fs::create_dir_all(&directory).unwrap();
        let mut meta =
            json!({"schema":"cuteafd.v41-exl3-aot.v1","capacity":capacity,"direct":false});
        if let Some(block) = block {
            meta["route_block"] = json!(block);
        }
        if schedule == Exl3Schedule::Gb10 && matches!(capacity, 1 | 80) {
            meta["decode_schedule"] = json!("gb10");
        }
        std::fs::write(directory.join("v41_exl3.json"), meta.to_string()).unwrap();
    }
    #[test]
    fn exl3_m8_legacy_default_and_explicit_group() {
        let root = tempfile::tempdir().unwrap();
        exl3_fixture(root.path(), 1, None, Exl3Schedule::Default);
        exl3_fixture(root.path(), 16, Some(16), Exl3Schedule::Default);
        exl3_fixture(root.path(), 80, Some(32), Exl3Schedule::Default);
        let installed = InstalledExpertGrouping::exl3(
            root.path(),
            16,
            Exl3RowPolicy::Nearest,
            Exl3Schedule::Default,
        )
        .unwrap();
        assert_eq!(installed.group_rows(1).unwrap(), 8);
        assert_eq!(installed.group_rows(0).unwrap(), 32);
        assert_eq!(installed.group_rows(2).unwrap(), 16);
        assert_eq!(installed.group_rows(16).unwrap(), 16);
    }
    #[test]
    fn exl3_worker_policy_and_schedule_select_actual_package() {
        let root = tempfile::tempdir().unwrap();
        for (c, block) in [(1, 8), (16, 16), (80, 64)] {
            exl3_fixture(root.path(), c, Some(block), Exl3Schedule::Gb10);
        }
        let installed = InstalledExpertGrouping::exl3(
            root.path(),
            16,
            Exl3RowPolicy::GlmFlashK64,
            Exl3Schedule::Gb10,
        )
        .unwrap();
        assert_eq!(installed.group_rows(1).unwrap(), 8);
        assert_eq!(installed.group_rows(0).unwrap(), 64);
        for rows in 2..=16 {
            assert_eq!(installed.group_rows(rows).unwrap(), 64);
        }
        exl3_fixture(root.path(), 80, Some(3), Exl3Schedule::Gb10);
        assert!(InstalledExpertGrouping::exl3(
            root.path(),
            16,
            Exl3RowPolicy::GlmFlashK64,
            Exl3Schedule::Gb10
        )
        .is_err());
    }
}
