//! Linux residence and exact native evidence for the shared startup probe.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::memory_ledger::{AllocationEvent, AllocationRecord, Space};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;

#[derive(Debug)]
struct Mapping {
    begin: u64,
    end: u64,
    file: bool,
    rss: u64,
    anonymous: u64,
}

fn kib(line: &str) -> Result<u64> {
    let mut fields = line.split_whitespace();
    let bytes = fields.next().context("missing smaps value")?.parse::<u64>()?
        .checked_mul(1024).context("smaps bytes overflow")?;
    ensure!(fields.next() == Some("kB") && fields.next().is_none(), "invalid smaps units");
    Ok(bytes)
}

fn mappings(text: &str) -> Result<Vec<Mapping>> {
    let mut out: Vec<Mapping> = Vec::new();
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("Rss:") {
            out.last_mut().context("smaps RSS before mapping")?.rss = kib(value)?;
        } else if let Some(value) = line.strip_prefix("Anonymous:") {
            out.last_mut().context("smaps Anonymous before mapping")?.anonymous = kib(value)?;
        } else {
            let mut fields = line.split_whitespace();
            let Some(range) = fields.next().filter(|field| field.contains('-')) else { continue };
            let (begin, end) = range.split_once('-').context("invalid smaps address")?;
            let (begin, end) = (u64::from_str_radix(begin, 16)?, u64::from_str_radix(end, 16)?);
            let inode = fields.nth(3).context("missing smaps inode")?.parse::<u64>()?;
            ensure!(begin < end && out.last().is_none_or(|prior| prior.end <= begin), "invalid smaps mapping order");
            out.push(Mapping { begin, end, file: inode != 0, rss: 0, anonymous: 0 });
        }
    }
    ensure!(!out.is_empty(), "empty smaps");
    for mapping in &out {
        ensure!(mapping.anonymous <= mapping.rss && mapping.rss <= mapping.end - mapping.begin,
            "inconsistent smaps residence");
    }
    Ok(out)
}

fn intervals(records: &[AllocationRecord], page: u64) -> Result<(u64, Vec<(u64, u64)>)> {
    ensure!(page.is_power_of_two(), "invalid Linux page size");
    let mut requested = 0u64;
    let mut ranges = Vec::new();
    for record in records.iter().filter(|r| r.key.space == Space::Pinned) {
        ensure!(record.address != 0 && record.bytes != 0, "empty native pinned allocation");
        requested = requested.checked_add(record.bytes as u64).context("pinned extent overflow")?;
        let end = (record.address as u64).checked_add(record.bytes as u64).context("pinned address overflow")?;
        let end = end.checked_add(page - 1).context("pinned page rounding overflow")? & !(page - 1);
        ranges.push((record.address as u64 & !(page - 1), end));
    }
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (begin, end) in ranges {
        if let Some(last) = merged.last_mut().filter(|last| begin <= last.1) {
            last.1 = last.1.max(end);
        } else { merged.push((begin, end)); }
    }
    Ok((requested, merged))
}

fn pagemap_presence(entries: &[u8], first: u64, page: u64,
    missing: &mut impl FnMut(u64, u64)) -> u64 {
    let mut present = 0;
    for (index, entry) in entries.chunks_exact(8).enumerate() {
        let entry = u64::from_ne_bytes(entry.try_into().unwrap());
        if entry & (1 << 63) != 0 { present += page; }
        else { missing((first + index as u64) * page, entry); }
    }
    present
}

fn pagemap_record(address: u64, entry: u64) -> Value {
    json!({"address": address, "entry_hex": format!("{entry:#018x}"),
        "present": entry & (1 << 63) != 0, "swapped": entry & (1 << 62) != 0,
        "exclusive": entry & (1 << 56) != 0})
}

/// Unprivileged pagemap metadata only; never resolve PFNs or touch observed pages.
fn present_bytes(file: &File, begin: u64, end: u64, page: u64,
    missing: &mut impl FnMut(u64, u64)) -> Result<u64> {
    let mut present = 0;
    let mut first = begin / page;
    let last = end / page;
    let mut buffer = [0u8; 8192];
    while first < last {
        let count = (last - first).min((buffer.len() / 8) as u64) as usize;
        file.read_exact_at(&mut buffer[..count * 8], first.checked_mul(8).context("pagemap offset overflow")?)?;
        present += pagemap_presence(&buffer[..count * 8], first, page, missing);
        first += count as u64;
    }
    Ok(present)
}

fn host_accounting(maps: &[Mapping], ranges: &[(u64, u64)], page: u64,
    mut presence: impl FnMut(u64, u64) -> Result<u64>) -> Result<Value> {
    let mut rss = 0u64;
    let mut file_rss = 0u64;
    let mut pinned_ordinary = 0u64;
    let mut covered = 0u64;
    let mut present = 0u64;
    let mut findings = Vec::new();
    let mut complete = true;
    let mut ordinary_complete = true;
    for &(begin, end) in ranges {
        covered += end - begin;
        let mut cursor = begin;
        for map in maps.iter().filter(|m| m.begin < end && begin < m.end) {
            let lo = begin.max(map.begin);
            let hi = end.min(map.end);
            if lo != cursor {
                findings.push(format!("pinned range {cursor:#x}..{lo:#x} missing from smaps"));
                complete = false;
                ordinary_complete = false;
            }
            cursor = hi;
            // RSS partitioning is smaps evidence, independent of pagemap presence.
            let ordinary = if map.file { map.anonymous } else { map.rss };
            if ordinary != 0 {
                if lo == map.begin && hi == map.end {
                    pinned_ordinary += ordinary;
                } else if ordinary == map.rss && map.rss == map.end - map.begin {
                    pinned_ordinary += hi - lo;
                } else {
                    findings.push(format!("partial pinned range in mixed/nonresident smaps mapping at {lo:#x}..{hi:#x}"));
                    complete = false;
                    ordinary_complete = false;
                }
            }
            match presence(lo, hi) {
                Ok(bytes) => {
                    ensure!(bytes <= hi - lo && bytes % page == 0, "invalid pagemap presence");
                    present += bytes;
                    if bytes != hi - lo {
                        findings.push(format!("CUDA-pinned pages not fully present at {lo:#x}..{hi:#x}: {bytes} bytes"));
                        complete = false;
                    }
                }
                Err(error) => {
                    findings.push(format!("pagemap unavailable for {lo:#x}..{hi:#x}: {error}"));
                    complete = false;
                }
            }
        }
        if cursor != end {
            findings.push(format!("pinned range {cursor:#x}..{end:#x} missing from smaps"));
            complete = false;
            ordinary_complete = false;
        }
    }
    for map in maps {
        rss = rss.checked_add(map.rss).context("smaps RSS overflow")?;
        if map.file { file_rss += map.rss - map.anonymous; }
    }
    let heap = rss.checked_sub(file_rss).and_then(|v| v.checked_sub(pinned_ordinary))
        .context("pinned/file residence exceeds RSS")?;
    Ok(json!({
        "smaps_rss_bytes": rss, "file_mapping_resident_bytes": file_rss,
        "pinned_page_covered_bytes": covered, "pinned_present_bytes": present,
        "pinned_resident_in_ordinary_rss_bytes": pinned_ordinary,
        "ordinary_owner": "host-heap (unattributed)",
        "ordinary_rss_bytes": ordinary_complete.then_some(heap),
        "ordinary_rss_complete": ordinary_complete,
        "rss_reconstruction_residual_bytes": ordinary_complete.then_some(rss as i64 - (heap + file_rss + pinned_ordinary) as i64),
        "rss_tolerance_bytes": page, "complete": complete, "findings": findings,
        "scope": "phase process residence; exact pinned extents charged separately; diagnostic capacities not additive"
    }))
}

fn log_missing_pagemap(stage: &str, records: &mut Vec<Value>) {
    if records.is_empty() { return; }
    tracing::info!(target: "cuteafd::placement", probe = %json!({"schema": 2,
        "stage": stage, "kind": "pinned-missing-pages", "records": records}), "startup memory probe");
    records.clear();
}

pub(super) fn host(records: &[AllocationRecord], stage: &str) -> Result<Value> {
    let page = crate::shared::host_heap::page_size();
    let (requested, ranges) = intervals(records, page)?;
    let maps = mappings(&std::fs::read_to_string("/proc/self/smaps")?)?;
    let pagemap = File::open("/proc/self/pagemap");
    let mut missing = Vec::with_capacity(64);
    let mut missing_count = 0u64;
    let mut value = host_accounting(&maps, &ranges, page, |begin, end| {
        let file = pagemap.as_ref().map_err(|error| anyhow::anyhow!("{error}"))?;
        present_bytes(file, begin, end, page, &mut |address, entry| {
            missing_count += 1;
            missing.push(pagemap_record(address, entry));
            if missing.len() == 64 { log_missing_pagemap(stage, &mut missing); }
        })
    })?;
    log_missing_pagemap(stage, &mut missing);
    value["missing_pagemap_pages"] = json!(missing_count);
    value["pinned_requested_bytes"] = json!(requested);
    value["native_malloc_diagnostic_bytes"] = json!(records.iter().filter(|r| r.key.space == Space::Host)
        .map(|r| r.bytes as u64).sum::<u64>());
    value["statm_rss_bytes"] = match std::fs::read_to_string("/proc/self/statm") {
        Ok(text) => text.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok())
            .and_then(|pages| pages.checked_mul(page)).map_or(Value::Null, |v| json!(v)),
        Err(_) => Value::Null,
    };
    value["statm_scope"] = json!("separate potentially stale telemetry; not the smaps reconciliation baseline");
    value["vm_hwm_bytes"] = json!(super::host_high_water_bytes(&std::fs::read_to_string("/proc/self/status")?)?);
    Ok(value)
}

pub(super) fn record(record: &AllocationRecord) -> Value {
    json!({"address": record.address, "bytes": record.bytes, "space": record.key.space.label(),
        "device": record.key.device, "scope": record.key.scope, "owner": record.key.owner,
        "kind": record.key.tag.kind.label(), "component": record.key.tag.owner,
        "tensor": record.key.tensor, "format": record.key.format, "endpoint": record.endpoint})
}

pub(super) fn event(event: &AllocationEvent) -> Value {
    let (kind, allocation) = match event {
        AllocationEvent::Allocated(r) => ("allocated", r),
        AllocationEvent::Freed(r) => ("freed", r),
        AllocationEvent::Enriched(r) => ("enriched", r),
        AllocationEvent::UnobservedRelease(r) => ("unobserved-release", r),
    };
    json!({"kind": kind, "allocation": record(allocation)})
}

pub(super) fn census(census: &crate::shared::memory::receipts::Census,
    records: &[AllocationRecord], devices: &[i32]) -> Result<Value> {
    use cuteafd_loader::placement::planner::{Device, Domain};
    let mut owners = BTreeMap::new();
    for receipt in census.receipts.iter().filter(|r| r.allocation.view.is_none()) {
        let a = &receipt.allocation;
        let device = match a.location.device {
            Device::Coordinator(rank) => *devices.get(rank as usize).context("census logical rank missing")?,
            Device::Host => -1,
            Device::Spark(_) => anyhow::bail!("remote owner in coordinator census"),
        };
        let space = match a.location.domain {
            Domain::Device => Space::Device, Domain::Pinned => Space::Pinned,
            Domain::Managed => Space::Managed, Domain::Host => Space::Host,
        };
        ensure!(owners.insert((space, device, receipt.address, a.bytes), a.id.as_str()).is_none(),
            "duplicate canonical census backing");
    }
    let mut missing = Vec::new();
    let mut observed = std::collections::BTreeSet::new();
    for native in records.iter().filter(|r| r.key.space != Space::Host) {
        let key = (native.key.space, native.key.device, native.address as u64, native.bytes as u64);
        observed.insert(key);
        if !owners.contains_key(&key) { missing.push(record(native)); }
    }
    let unmatched: Vec<_> = owners.iter().filter(|(key, _)| !observed.contains(key)).map(|(key, id)|
        json!({"owner": id, "space": key.0.label(), "device": key.1, "address": key.2, "bytes": key.3})).collect();
    Ok(json!({"native_without_named_owner_count": missing.len(), "named_without_native_record_count": unmatched.len(),
        "detail_scope": "compare chunked native-live and retained-owners records by domain/device/address/extent",
        "native_identity_tolerance_bytes": 0,
        "exact_native_coverage": missing.is_empty() && unmatched.is_empty(),
        "unclassified": census.unclassified.iter().map(|u| json!({"owner": u.owner, "reason": u.reason})).collect::<Vec<_>>(),
        "diagnostic_heap_capacity_bytes": census.heap_diagnostics.iter().map(|r| r.allocation.bytes).sum::<u64>(),
        "diagnostic_heap_scope": "already inside phase RSS; not additive"}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smaps_keeps_file_cow_anonymous_and_statm_separate() -> Result<()> {
        let maps = mappings("1000-3000 rw-p 0000 00:00 0 [heap]\nRss: 8 kB\nAnonymous: 8 kB\n4000-6000 rw-p 0000 08:01 7 /weights\nRss: 8 kB\nAnonymous: 4 kB\n")?;
        let v = host_accounting(&maps, &[(0x1000, 0x2000)], 4096, |b, e| Ok(e-b))?;
        assert_eq!(v["smaps_rss_bytes"], 16384);
        assert_eq!(v["file_mapping_resident_bytes"], 4096);
        assert_eq!(v["pinned_resident_in_ordinary_rss_bytes"], 4096);
        assert_eq!(v["ordinary_rss_bytes"], 8192);
        assert_eq!(v["rss_reconstruction_residual_bytes"], 0);
        Ok(())
    }

    #[test]
    fn missing_pinned_pages_and_unavailable_pagemap_keep_measured_rss() -> Result<()> {
        let maps = mappings("1000-3000 rw-p 0000 00:00 0\nRss: 8 kB\nAnonymous: 8 kB\n4000-6000 rw-p 0000 08:01 7 /cuda\nRss: 8 kB\nAnonymous: 0 kB\n7000-9000 rw-p 0000 00:00 0\nRss: 8 kB\nAnonymous: 8 kB\n")?;
        for value in [host_accounting(&maps, &[(0x1000, 0x3000), (0x4000, 0x6000)], 4096, |_, _| Ok(4096))?,
            host_accounting(&maps, &[(0x1000, 0x3000), (0x4000, 0x6000)], 4096, |_, _| anyhow::bail!("denied"))?] {
            assert_eq!(value["complete"], false);
            assert_eq!(value["ordinary_rss_complete"], true);
            assert_eq!(value["ordinary_rss_bytes"], 8192);
            assert_eq!(value["pinned_resident_in_ordinary_rss_bytes"], 8192);
            assert_eq!(value["file_mapping_resident_bytes"], 8192);
            assert_eq!(value["rss_reconstruction_residual_bytes"], 0);
            assert!(!value["findings"].as_array().unwrap().is_empty());
        }
        Ok(())
    }

    #[test]
    fn missing_pagemap_entries_preserve_address_raw_bits_and_presence() {
        let words = [1u64 << 63, (1u64 << 62) | (1 << 56) | 7, 0];
        let bytes = words.into_iter().flat_map(u64::to_ne_bytes).collect::<Vec<_>>();
        let mut missing = Vec::new();
        assert_eq!(pagemap_presence(&bytes, 3, 4096, &mut |a, e| missing.push((a, e))), 4096);
        assert_eq!(missing, [(0x4000, words[1]), (0x5000, words[2])]);
        let record = pagemap_record(missing[0].0, missing[0].1);
        assert_eq!(record["address"], 0x4000);
        assert_eq!(record["entry_hex"], "0x4100000000000007");
        assert_eq!(record["present"], false);
        assert_eq!(record["swapped"], true);
        assert_eq!(record["exclusive"], true);
    }

    #[test]
    fn partial_mixed_mapping_cannot_guess_pinned_rss() -> Result<()> {
        let maps = mappings("1000-3000 rw-p 0000 08:01 7 /file\nRss: 8 kB\nAnonymous: 4 kB\n")?;
        let value = host_accounting(&maps, &[(0x1000, 0x2000)], 4096, |b, e| Ok(e-b))?;
        assert_eq!(value["complete"], false);
        assert!(value["ordinary_rss_bytes"].is_null());
        Ok(())
    }

    #[test]
    fn exact_native_coverage_rejects_equal_total_with_wrong_identity() -> Result<()> {
        use crate::shared::memory::receipts::Census;
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{Allocation, Lifetime, Location, Receipt};
        let native = AllocationRecord { address: 0x123000, bytes: 512, endpoint: None,
            key: cuteafd_ffi::memory_ledger::Key { tag: crate::shared::memory::TEST, space: Space::Device, device: 7, scope: "test",
                owner: None, tensor: None, format: None } };
        let mut named = Census::default();
        named.receipts.push(Receipt { address: native.address as u64,
            allocation: Allocation::buffer("test", Location::coordinator(1), 512,
                Lifetime::TARGET, Category::Workspace, "test") });
        assert_eq!(census(&named, &[native], &[3, 7])?["exact_native_coverage"], true);
        named.receipts[0].address += 1;
        let value = census(&named, &[native], &[3, 7])?;
        assert_eq!(value["exact_native_coverage"], false);
        assert_eq!(value["native_without_named_owner_count"], 1);
        assert_eq!(value["named_without_native_record_count"], 1);
        Ok(())
    }

    #[test]
    fn step5_tp2_target_mtp_embedding_census_normalizes_duplicate_arena_root() -> Result<()> {
        use crate::shared::{memory::{device::Device, receipts::{Census, Inventory},
            weight_arena::DeviceWeightArena}, token_io::{EmbedPlacement, EmbedSource, TokenEmbedding}};
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::{placement::{Baseline, GpuBudget, Inventory as PlacementInventory,
            planner::{Context as ServingContext, CoordinatorDevices, Lifetime}},
            plan::{checkpoint::Checkpoint, families::step5::admission::{self, HostInputs, MarkRule, RuntimeInputs}},
            serving_capacity::WeightArenaLayout};
        let root = tempfile::tempdir()?;
        let snapshot = root.path().join("snapshot");
        cuteafd_loader::plan::testing::write_step5_exl3_snapshot(&snapshot);
        let checkpoint = Checkpoint::open(&snapshot)?;
        let inputs = RuntimeInputs { graph_layout: None, decode_graphs: false,
            bf16_target_head: cuteafd_loader::placement::planner::serving::BlasHead::TensorLt,
            host: HostInputs::default(), resident_weight_limit: None,
            spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16,
            kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8,
            inventory: PlacementInventory { gpus: vec![GpuBudget { capacity_bytes: 96 << 30,
                reserve_bytes: 0, baseline: Baseline::Measured { free_bytes: 96 << 30 } }; 2],
                spark_ranks: 6, peer_access: true },
            rows: 64, waves: 1, unit: 64, sequences: 16,
            context: 262144, serving_context: ServingContext::pinned(262144), requested_pool: Some(262144),
            marks: MarkRule { lanes: 16, entries: 0, budget_bytes: 0, host: true },
            peer_diagnostics: false, pending_code: vec![0; 2], onboard: None,
        };
        let admitted = admission::admit_mtp(&checkpoint, &inputs, 3)?;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let ids = [0, 1];
        let devices = CoordinatorDevices::new(&ids)?;
        use cuteafd_core::memory_layout::ArenaTag;
        let arena = |rank, id, tensors, tag| DeviceWeightArena::new(Device { library: &library, id: rank },
            WeightArenaLayout::new(id, tensors)?, tag);
        let target0 = arena(0, "step5/target/0", vec![("target".into(), 257), ("embedding.table".into(), 64)],
            ArenaTag::weights("step5/target"))?;
        let target1 = arena(1, "step5/target/1", vec![("target".into(), 258)], ArenaTag::weights("step5/target"))?;
        let mtp = arena(0, "step5/mtp/0", vec![("mtp".into(), 259)], ArenaTag::drafter_weights("step5/mtp"))?;
        let table = root.path().join("embedding.bin");
        std::fs::write(&table, [0; 64])?;
        let (embedding, ()) = TokenEmbedding::load_with(&library, EmbedSource { path: table,
            offset: 0, vocab: 4, hidden: 8, snapshot: None, tensor_name: "model.embed_tokens.weight".into() },
            EmbedPlacement::Gpu, || Ok(()), |name, bytes| target0.view(name, bytes))?;
        let mut weights = Inventory::new(devices, "step5.target", Lifetime::RESIDENT, Category::Weights, "step5 weights");
        target0.view("target", 257)?.inventory(&mut weights, "target0")?;
        target1.view("target", 258)?.inventory(&mut weights, "target1")?;
        mtp.view("mtp", 259)?.inventory(&mut weights, "mtp")?;
        let mut named = Census::default();
        named.owners(weights.receipts);
        named.owners(embedding.allocation_receipts(devices, "step5.embedding")?);
        let roots = named.receipts.iter().filter(|receipt| receipt.allocation.view.is_none())
            .map(|receipt| (receipt.address as usize, receipt.allocation.bytes as usize)).collect::<std::collections::BTreeSet<_>>();
        let records = cuteafd_ffi::memory_ledger::allocations().into_iter()
            .filter(|record| roots.contains(&(record.address, record.bytes))).collect::<Vec<_>>();
        assert_eq!(records.len(), 3);
        assert_eq!(census(&named, &records, &ids).unwrap_err().to_string(), "duplicate canonical census backing");
        named.normalize(admitted.solved.facts.phase_order)?;
        assert_eq!(named.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).count(), 3);
        assert_eq!(named.receipts.iter().filter(|receipt| receipt.allocation.view.is_some()).count(), 4);
        let embedding = named.receipts.iter().find(|receipt| receipt.allocation.id == "step5.embedding.table").unwrap();
        let view = embedding.allocation.view.as_ref().unwrap();
        assert_eq!((view.backing.as_str(), view.offset), ("step5/target/0", 512));
        assert_eq!(census(&named, &records, &ids)?["exact_native_coverage"], true);
        Ok(())
    }

    #[test]
    fn pinned_page_union_preserves_requested_extents_without_double_charge() -> Result<()> {
        let make = |address, bytes| AllocationRecord { address, bytes, endpoint: None,
            key: cuteafd_ffi::memory_ledger::Key { tag: crate::shared::memory::TEST, space: Space::Pinned, device: -1, scope: "test",
                owner: None, tensor: None, format: None } };
        let (requested, covered) = intervals(&[make(0x1001, 7), make(0x1ff0, 32)], 4096)?;
        assert_eq!(requested, 39);
        assert_eq!(covered, [(0x1000, 0x3000)]);
        Ok(())
    }
}
