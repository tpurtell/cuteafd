//! Checked byte and reservation inputs shared by native startup admission.
use anyhow::{Context, Result, ensure};
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteSize(pub usize);
#[derive(Clone, Copy, Debug)]
pub(crate) enum Reservation {
    Bytes(ByteSize),
    Percent(u64), // millionths of one percent
}
fn decimal(value: &str) -> Result<u64> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    ensure!(
        !whole.is_empty()
            && whole.bytes().all(|c| c.is_ascii_digit())
            && fraction.len() <= 6
            && fraction.bytes().all(|c| c.is_ascii_digit()),
        "expected a positive decimal with at most six fractional digits"
    );
    let whole: u64 = whole.parse()?;
    let fractional: u64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse()?
    };
    let scaled = whole
        .checked_mul(1_000_000)
        .and_then(|v| v.checked_add(fractional * 10u64.pow(6 - fraction.len() as u32)))
        .context("memory size overflow")?;
    ensure!(scaled > 0, "memory size must be positive");
    Ok(scaled)
}
impl FromStr for ByteSize {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        let end = value
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(value.len());
        let amount = decimal(&value[..end])?;
        let unit = value[end..].trim().to_ascii_lowercase();
        let multiplier: u128 = match unit.as_str() {
            "" | "b" => 1,
            "mb" => 1_000_000,
            "gb" => 1_000_000_000,
            "mib" => 1_048_576,
            "gib" => 1_073_741_824,
            _ => anyhow::bail!("memory size unit must be B, MB, GB, MiB or GiB"),
        };
        let bytes = usize::try_from(u128::from(amount) * multiplier / 1_000_000)?;
        ensure!(bytes > 0, "memory size rounds to zero bytes");
        Ok(Self(bytes))
    }
}
impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}
impl FromStr for Reservation {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if let Some(percent) = value.trim().strip_suffix('%') {
            let percent = decimal(percent.trim())?;
            ensure!(
                percent <= 100_000_000,
                "memory percentage must be in (0,100]"
            );
            Ok(Self::Percent(percent))
        } else {
            Ok(Self::Bytes(value.parse()?))
        }
    }
}
impl Reservation {
    pub(crate) fn bytes(self, total: usize) -> Result<usize> {
        let bytes = match self {
            Self::Bytes(size) => size.0,
            Self::Percent(percent) => (total as u128 * u128::from(percent) / 100_000_000) as usize,
        };
        ensure!(
            bytes > 0 && bytes <= total,
            "memory reservation exceeds device total or rounds to zero"
        );
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalLayers {
    Auto,
    ExpertsFirst,
    Count(usize),
}
impl LocalLayers {
    pub(crate) fn onboard(self) -> cuteafd_loader::placement::Onboard {
        use cuteafd_loader::placement::{Onboard, EXPERTS_FIRST_POOL_FLOOR};
        match self {
            Self::Auto => Onboard::Auto,
            Self::ExpertsFirst => Onboard::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR },
            Self::Count(n) => Onboard::Layers(n),
        }
    }
}
impl FromStr for LocalLayers {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => return Ok(Self::Auto),
            "max" => return Ok(Self::ExpertsFirst),
            _ => {}
        }
        let count: usize = value
            .parse()
            .context("RTX expert layers must be auto, max or 0..40")?;
        ensure!(count <= 40, "RTX expert layers must be auto, max or 0..40");
        Ok(Self::Count(count))
    }
}

/// Admit the complete future host working set once, before any allocation.
/// Families supply table byte formulas summed across every local GPU, together
/// with the full prefix-cache quota (zero when disabled) and deferred logits.
/// Separate checks for tables and cache against the same live sample are not
/// reservations: they can each fit while their combined working set does not.
pub(crate) fn admit_host_memory(cache: u64, tables: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    admit_host_memory_with_resident_formulas(cache, tables, &[], deferred_logits, headroom)
}

/// Already allocated native payload and pageable workspace have one formula
/// owner. Replace their pinned attribution, rather than charging them twice.
pub(crate) fn admit_host_memory_with_resident_formulas(cache: u64, tables: &[(&str, u64)],
    formulas: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    let memory = cuteafd_loader::formats::mapped_preload::host_memory()?;
    let snapshot = cuteafd_ffi::memory_ledger::snapshot();
    let resident = host_resident_items(snapshot.by_scope(cuteafd_ffi::memory_ledger::Space::Pinned, -1), formulas);
    let ledger = host_admission(memory, resident, cache, tables, deferred_logits, headroom)?;
    tracing::info!(host_budget_bytes = cache, host_ledger = %serde_json::to_string(&ledger)?,
        "coordinator host memory admitted");
    Ok(ledger)
}

fn host_resident_items(pinned: std::collections::BTreeMap<&'static str, usize>, formulas: &[(&str, u64)])
    -> Vec<cuteafd_core::memory_layout::Item> {
    use cuteafd_core::memory_layout::{Basis, Category, Item};
    let mut items: Vec<_> = pinned.into_iter()
        .filter(|(scope, _)| !formulas.iter().any(|&(name, _)| name == *scope))
        .map(|(scope, bytes)| Item::new(Category::of_scope(scope), scope, "pinned", bytes as u64, Basis::Exact))
        .collect();
    items.extend(formulas.iter().map(|&(name, bytes)| Item::new(Category::Staging, name,
        "pinned payload + pageable sort tuples", bytes, Basis::Formula)));
    items
}

fn host_admission((total, available): (u64, u64),
    mut resident: Vec<cuteafd_core::memory_layout::Item>, cache: u64,
    tables: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    use cuteafd_core::memory_layout::{Basis, Category, HostLayout, Item, ItemLifetime, host_table_item};
    // Resident bytes are already missing from live availability. Only these
    // future consumers and the reserve are charged against it again.
    resident.extend(tables.iter().map(|&(name, bytes)| host_table_item(name, bytes)));
    resident.push(Item::new(Category::Workspace, "scoring/full-logits", "FP32", deferred_logits, Basis::Formula)
        .with_lifetime(ItemLifetime::PostReadyReserve));
    let ledger = HostLayout::new(total, available, headroom.max(total / 10), cache, resident)
        .context("host admission byte count overflow")?;
    ensure!(ledger.shortfall_bytes == 0,
        "coordinator host memory cannot admit host cache {} plus deferred consumers {} and reserve {}: available {}, total {}, resident {}, shortfall {} bytes",
        cache, ledger.deferred_bytes, ledger.reserve_bytes, available, total,
        ledger.resident_bytes, ledger.shortfall_bytes);
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::{Basis, Category, Item, ItemLifetime};

    #[test]
    fn mapped_table_host_workspace_remains_deferred_at_every_depth() -> Result<()> {
        for depth in 1..=4 {
            for row_bytes in [160, 320] {
                let shape = cuteafd_loader::formats::mapped_table::MappedStaging::new(
                    2752 * 16, depth, 64 * 16, row_bytes)?;
                let payload = shape.slot_bytes().sum::<usize>() as u64;
                let workspace = shape.workspace_bytes as u64;
                let items = host_resident_items(std::collections::BTreeMap::from([
                    ("mapped-table", payload as usize)]), &[]);
                let demands = [("staging/mapped-table-worker", workspace)];
                let ledger = host_admission((2 << 30, 1 << 30), items.clone(), 0, &demands, 0, 0)?;
                assert_eq!((ledger.resident_bytes, ledger.deferred_bytes), (payload, workspace));
                assert_eq!(ledger.resident_bytes + ledger.deferred_bytes, shape.total_bytes as u64);
                let worker = ledger.items.iter().find(|item| item.group == demands[0].0).unwrap();
                assert_eq!((worker.bytes, worker.lifetime), (workspace, ItemLifetime::PostReadyReserve));
                let available = (2u64 << 30) / 10 + workspace;
                assert!(host_admission((2 << 30, available), items.clone(), 0, &demands, 0, 0).is_ok());
                assert!(host_admission((2 << 30, available - 1), items, 0, &demands, 0, 0).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn engram_host_formula_replaces_payload_attribution_without_double_charge() -> Result<()> {
        for k in 1..=4 {
            let rows = 8192usize.div_ceil(k.max(2)).div_ceil(64) * 64;
            let shape = cuteafd_loader::EngramStagingShape::new(rows, k)?;
            let formulas = [("v41/engram-prefill", shape.host_resident_bytes() as u64),
                ("v41/engram-decode", shape.decode_host_resident_bytes() as u64)];
            let pinned = std::collections::BTreeMap::from([
                ("v41/engram-prefill", shape.host_slots() * rows * 6337),
                ("v41/engram-decode", shape.decode_slots() * shape.decode_rows() * 6337),
                ("unrelated/pinned", 7)]);
            let items = host_resident_items(pinned, &formulas);
            assert_eq!(items.len(), 3);
            for &(name, bytes) in &formulas {
                let entry = items.iter().find(|item| item.group == name).unwrap();
                assert_eq!((entry.bytes, entry.basis, entry.lifetime),
                    (bytes, Basis::Formula, ItemLifetime::ReadyResident));
            }
            let workspace = [("v41/engram-prefill-worker", shape.host_workspace_bytes() as u64),
                ("v41/engram-decode-worker", shape.decode_host_workspace_bytes() as u64)];
            let workspace_bytes = workspace.iter().map(|f| f.1).sum::<u64>();
            let ledger = host_admission((2 << 30, 1 << 30), items.clone(), 0, &workspace, 0, 0)?;
            assert_eq!(ledger.resident_bytes, formulas.iter().map(|f| f.1).sum::<u64>() + 7);
            assert_eq!(ledger.deferred_bytes, workspace_bytes);
            assert_eq!(ledger.resident_bytes + ledger.deferred_bytes,
                (shape.host_bytes() + shape.decode_host_bytes()) as u64 + 7);
            // Worker vectors/rings do not exist at readiness: they must consume
            // future availability, not be mislabeled already-resident bytes.
            assert!(host_admission((2 << 30, workspace_bytes - 1), items, 0, &workspace, 0, 0).is_err());
        }
        Ok(())
    }
    #[test]
    fn named_pinned_tables_are_admitted_without_a_prefix_cache() -> Result<()> {
        let memory = (1000, 800);
        let resident = vec![Item::new(Category::Weights, "resident/weight", "pinned", 200, Basis::Exact)];
        let ledger = host_admission(memory, resident, 0,
            &[("synthetic/ordered-table", 300), ("synthetic/other-gpu", 200)], 100, 0)?;
        assert_eq!((ledger.host_budget_bytes, ledger.resident_bytes, ledger.deferred_bytes), (0, 200, 600));
        assert_eq!((ledger.reserve_bytes, ledger.headroom_bytes, ledger.shortfall_bytes), (100, 100, 0));
        for (name, bytes) in [("synthetic/ordered-table", 300), ("synthetic/other-gpu", 200)] {
            let item = ledger.items.iter().find(|item| item.group == name).unwrap();
            assert_eq!((item.category, item.bytes, item.lifetime),
                (Category::Staging, bytes, ItemLifetime::PostReadyReserve));
        }
        let encoded = serde_json::to_value(&ledger)?;
        assert_eq!(serde_json::from_value::<cuteafd_core::memory_layout::HostLayout>(encoded)?, ledger);
        Ok(())
    }

    #[test]
    fn host_admission_charges_future_tables_and_prefix_once_but_not_resident_twice() -> Result<()> {
        let memory = (1000, 700);
        let resident = vec![Item::new(Category::Staging, "resident/table", "pinned", 300, Basis::Exact)];
        let ledger = host_admission(memory, resident, 100, &[("synthetic/table", 200)], 100, 150)?;
        assert_eq!((ledger.resident_bytes, ledger.deferred_bytes, ledger.headroom_bytes), (300, 300, 150));
        assert_eq!(ledger.reserve_bytes, 150);
        assert_eq!(host_admission(memory, vec![], 100, &[], 100, 0)?.deferred_bytes, 100);
        Ok(())
    }

    #[test]
    fn tables_and_prefix_must_be_admitted_in_one_complete_working_set() -> Result<()> {
        let memory = (1000, 600);
        let tables = [("synthetic/table", 300)];
        assert!(host_admission(memory, vec![], 300, &[], 0, 0).is_ok());
        assert!(host_admission(memory, vec![], 0, &tables, 0, 0).is_ok());
        let error = host_admission(memory, vec![], 300, &tables, 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 100 bytes"), "{error}");
        Ok(())
    }

    #[test]
    fn host_admission_refuses_available_total_and_arithmetic_shortfalls() {
        let memory = (1000, 300);
        let error = host_admission(memory, vec![], 0, &[("synthetic/table", 201)], 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 1 bytes"), "{error}");
        let resident = vec![Item::new(Category::Staging, "resident/table", "pinned", 800, Basis::Exact)];
        let error = host_admission((1000, 1000), resident, 0,
            &[("synthetic/table", 101)], 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 1 bytes"), "{error}");
        let memory = (u64::MAX, u64::MAX);
        for (cache, tables, logits, headroom) in [
            (0, vec![("synthetic/a", u64::MAX), ("synthetic/b", 1)], 0, 0),
            (1, vec![("synthetic/a", u64::MAX)], 0, 0),
            (0, vec![("synthetic/a", u64::MAX)], 1, 0),
            (0, vec![], 1, u64::MAX),
        ] {
            let error = host_admission(memory, vec![], cache, &tables, logits, headroom).unwrap_err();
            assert!(error.to_string().contains("overflow"), "{error}");
        }
    }
    #[test]
    fn local_layers_max_uses_shared_experts_first_policy() {
        use cuteafd_loader::placement::{Onboard, EXPERTS_FIRST_POOL_FLOOR};
        assert_eq!("max".parse::<LocalLayers>().unwrap().onboard(),
            Onboard::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR });
        assert_eq!("auto".parse::<LocalLayers>().unwrap().onboard(), Onboard::Auto);
        assert_eq!("40".parse::<LocalLayers>().unwrap().onboard(), Onboard::Layers(40));
        for value in ["41", "-1", "all", "50%"] {
            assert!(value.parse::<LocalLayers>().is_err(), "{value}");
        }
    }

    #[test]
    fn sizes_and_reservations_validate_units_and_overflow() {
        for (value, bytes) in [
            ("37GB", 37_000_000_000),
            ("1.5GiB", 1_610_612_736),
            ("1MB", 1_000_000),
            ("1MiB", 1_048_576),
            ("123", 123),
        ] {
            assert_eq!(value.parse::<ByteSize>().unwrap().0, bytes);
        }
        for value in [
            "0",
            "-1GB",
            "NaN",
            "1TB",
            "10%",
            "1.0000001GB",
            "18446744073709551616GB",
        ] {
            assert!(value.parse::<ByteSize>().is_err(), "{value}");
        }
        assert_eq!(
            "87.5%"
                .parse::<Reservation>()
                .unwrap()
                .bytes(8_000_000_000)
                .unwrap(),
            7_000_000_000
        );
        for value in ["0%", "100.1%", "inf%"] {
            assert!(value.parse::<Reservation>().is_err());
        }
    }
}
