//! Metadata-only EXLR8 execution arenas, shared by admission and the runtime.
use super::{residency::{execution_workspace_layout, ExpertSet}, PrefillActivations};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{exlr8::{KernelFamily, KernelKind}, programs::ProgramSpec};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionVariant { SharedDecode, DecodeCompact, DecodeGrouped, Prefill }

impl ExecutionVariant {
    pub fn program(self, family: KernelFamily, activations: PrefillActivations) -> &'static str {
        family.program(match self {
            Self::SharedDecode => KernelKind::Shared,
            Self::DecodeCompact => KernelKind::Routed,
            Self::DecodeGrouped => KernelKind::GroupedDecode,
            Self::Prefill => match activations {
                PrefillActivations::A16 => KernelKind::GroupedA16,
                PrefillActivations::A8 => KernelKind::GroupedA8,
            },
        })
    }

    pub fn group(self, family: KernelFamily) -> &'static str {
        family.program(match self {
            Self::SharedDecode => KernelKind::SharedGroup,
            Self::DecodeCompact | Self::DecodeGrouped => KernelKind::Group,
            Self::Prefill => KernelKind::GroupedGroup,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionWorkspace {
    pub ranges: [(usize, usize); 11],
    pub bytes: usize,
    pub rows: usize,
    pub top_k: usize,
    pub max_groups: usize,
    pub hidden: usize,
    pub units: usize,
    pub unit_limit: usize,
    pub intermediate_bytes: usize,
    pub compact: bool,
    pub unit_tables: bool,
}

impl ExecutionWorkspace {
    /// Physical arena and direct kernel bindings; no view is an allocation.
    pub fn allocations(&self, prefix: &str, location: crate::placement::planner::Location,
        lifetime: crate::placement::planner::Lifetime) -> Result<Vec<crate::placement::planner::Allocation>> {
        use crate::placement::planner::{Allocation, Domain, View};
        use cuteafd_core::memory_layout::Category;
        ensure!(!prefix.is_empty() && location.domain == Domain::Device && self.bytes > 0,
            "invalid EXLR8 workspace owner identity or extent");
        let names = ["input", "ids", "route_weights", "output", "group_expert", "route_slots",
            "route_group_lane", "group_count", "fc1", "intermediate_fc2", "rotated"];
        let allocation = |name: &str, bytes: usize| -> Result<Allocation> {
            let mut out = Allocation::buffer(format!("{prefix}.{name}"), location, u64::try_from(bytes)?,
                lifetime, Category::Workspace, "EXLR8 expert workspace");
            out.alignment = 256;
            Ok(out)
        };
        let mut out = vec![allocation("arena", self.bytes)?];
        let mut end = 0usize;
        for (name, &(offset, bytes)) in names.iter().zip(&self.ranges) {
            ensure!(bytes > 0 && offset % 256 == 0 && offset >= end,
                "EXLR8 workspace bindings must be positive, aligned and disjoint");
            end = offset.checked_add(bytes).context("EXLR8 workspace binding overflow")?;
            ensure!(end <= self.bytes, "EXLR8 workspace binding exceeds arena");
            let mut view = allocation(name, bytes)?;
            view.view = Some(View { backing: format!("{prefix}.arena"), offset: u64::try_from(offset)? });
            out.push(view);
        }
        ensure!(end == self.bytes, "EXLR8 workspace has an unaccounted arena tail");
        let (offset, combined) = self.ranges[9];
        ensure!(self.intermediate_bytes > 0 && self.intermediate_bytes < combined
            && self.intermediate_bytes % 256 == 0, "invalid EXLR8 intermediate/FC2 split");
        for (name, offset, bytes) in [("intermediate", offset, self.intermediate_bytes),
            ("fc2", offset.checked_add(self.intermediate_bytes).context("EXLR8 FC2 offset overflow")?,
                combined - self.intermediate_bytes)] {
            let mut view = allocation(name, bytes)?;
            view.view = Some(View { backing: format!("{prefix}.arena"), offset: u64::try_from(offset)? });
            out.push(view);
        }
        Ok(out)
    }
}

/// The only owned-unit scratch sizing rule. Compact FC1 uses owned units,
/// compact intermediate retains the global stride; grouped FC1/intermediate
/// both use owned units. FC2 always spans the complete hidden domain.
pub fn execution_workspace(spec: &ProgramSpec, rows: usize, set: ExpertSet,
    variant: ExecutionVariant, unit_limit: usize, activations: PrefillActivations)
    -> Result<ExecutionWorkspace> {
    cuteafd_ffi::exlr8::validate(spec)?;
    ensure!((set == ExpertSet::Shared) == (variant == ExecutionVariant::SharedDecode),
        "EXLR8 shared decode requires the shared expert set");
    let (family, _) = KernelFamily::of(spec)?;
    ensure!(spec.name == variant.program(family, activations) && spec.geometry["abi_version"] == 3,
        "EXLR8 execution requires fused rotation ABI v3");
    ensure!(rows > 0 && rows <= spec.capacity_rows as usize, "EXLR8 rows outside admitted K1 capacity");
    ensure!(variant != ExecutionVariant::Prefill || rows > cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS,
        "EXLR8 prefill instance is only admitted above the K1 row limit");
    let dimension = |name| -> Result<usize> {
        usize::try_from(spec.geometry[name].as_u64().context("missing K1 geometry")?)
            .context("K1 geometry exceeds address space")
    };
    let hidden = dimension("hidden")?;
    let top_k = dimension("top_k")?;
    let max_groups = dimension("max_groups")?;
    let units = dimension("units")?;
    let compact = set == ExpertSet::Routed;
    let unit_tables = matches!(variant, ExecutionVariant::DecodeGrouped | ExecutionVariant::Prefill);
    ensure!(!compact || (1..=units).contains(&unit_limit), "invalid compact unit limit");
    let scratch = |name: &str| -> Result<usize> {
        let bytes = usize::try_from(*spec.scratch.get(name).context("missing K1 scratch")?)?;
        let owned = match variant {
            ExecutionVariant::DecodeCompact => name == "fc1",
            ExecutionVariant::DecodeGrouped | ExecutionVariant::Prefill => name != "fc2",
            ExecutionVariant::SharedDecode => false,
        };
        if compact && owned {
            ensure!(bytes % units == 0, "compact scratch must divide by global units");
            (bytes / units).checked_mul(unit_limit).context("compact scratch overflow")
        } else { Ok(bytes) }
    };
    let intermediate_bytes = scratch("intermediate")?;
    let (ranges, bytes) = execution_workspace_layout(rows, spec.capacity_rows as usize,
        cuteafd_ffi::exlr8::group_rows(spec)?,
        hidden, top_k, max_groups, scratch("fc1")?, intermediate_bytes, scratch("fc2")?, set)?;
    Ok(ExecutionWorkspace { ranges, bytes, rows, top_k, max_groups, hidden, units,
        unit_limit: if compact { unit_limit } else { units }, intermediate_bytes, compact, unit_tables })
}

pub(super) fn program_spec(programs: &Value, name: &str) -> Result<ProgramSpec> {
    let entries = programs["programs"].as_array().context("PROGRAMS.json programs must be an array")?;
    let matches = entries.iter().filter(|entry| entry["name"] == name).collect::<Vec<_>>();
    ensure!(matches.len() == 1, "PROGRAMS.json requires one {name}");
    let entry = matches[0];
    let pointers = entry["pointers"].as_array().context("EXLR8 pointers missing")?.iter()
        .map(|p| p["name"].as_str().map(String::from).context("EXLR8 pointer name invalid")).collect::<Result<_>>()?;
    let scalars = entry["scalars"].as_array().context("EXLR8 scalars missing")?.iter()
        .map(|s| match s["type"].as_str() {
            Some("int32") => Ok(b'i'), Some("float32") => Ok(b'f'),
            _ => anyhow::bail!("EXLR8 scalar dtype invalid"),
        }).collect::<Result<_>>()?;
    let scratch = entry["scratch_bytes_at_capacity"].as_object().context("EXLR8 scratch missing")?.iter()
        .map(|(name, bytes)| Ok((name.clone(), bytes.as_u64().context("EXLR8 scratch invalid")?)))
        .collect::<Result<_>>()?;
    Ok(ProgramSpec { name: name.into(), pointers, scalar_kinds: scalars, scratch,
        capacity_rows: u32::try_from(entry["capacity_rows"].as_u64().context("EXLR8 capacity missing")?)?,
        geometry: entry["geometry"].clone() })
}

/// Header-only shared execution layout, including direct graph-visible bindings.
pub fn shared_workspace_layout(programs: &Value, family: KernelFamily) -> Result<ExecutionWorkspace> {
    let variant = ExecutionVariant::SharedDecode;
    let group_name = variant.group(family);
    let group = program_spec(programs, group_name)?;
    let entry = programs["programs"].as_array().context("EXLR8 programs missing")?.iter()
        .find(|entry| entry["name"] == group_name).context("EXLR8 grouping entry missing")?;
    cuteafd_ffi::exlr8::validate_entry(&group, entry)?;
    ensure!(group.geometry["hidden"].as_u64() == Some(family.geometry().hidden as u64),
        "EXLR8 shared grouping hidden differs from compute family");
    let spec = program_spec(programs, variant.program(family, PrefillActivations::A16))?;
    let rows = cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS;
    let layout = execution_workspace(&spec, rows, ExpertSet::Shared, variant,
        family.geometry().units, PrefillActivations::A16)?;
    ensure!(group.geometry["top_k"] == spec.geometry["top_k"]
        && group.geometry["max_groups"] == spec.geometry["max_groups"]
        && group.capacity_rows == spec.capacity_rows
        && cuteafd_ffi::exlr8::group_rows(&group)? == cuteafd_ffi::exlr8::group_rows(&spec)?
        && group.geometry["input_row_bytes"].as_u64() == Some((layout.ranges[0].1 / rows) as u64),
        "EXLR8 shared grouping and compute geometry differ");
    Ok(layout)
}

/// Admit the shared expert from the same validated geometry and arena as runtime.
pub fn shared_workspace(programs: &Value, family: KernelFamily) -> Result<u64> {
    Ok(u64::try_from(shared_workspace_layout(programs, family)?.bytes)?)
}

/// The stream-serialized TP2 backend owns both decode variants, one prefill
/// execution and a stable BF16 output. Pair's per-lane payloads are separate.
pub fn routed_workspace(programs: &Value, family: KernelFamily, rows: u64, unit_limit: usize,
    activations: PrefillActivations) -> Result<u64> {
    ensure!((1..=4096).contains(&rows), "EXLR8 routed lane geometry");
    let rows = usize::try_from(rows)?;
    let decode_rows = cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS;
    let mut total = 0usize;
    let mut hidden = None;
    for (variant, rows, count) in [(ExecutionVariant::DecodeCompact, decode_rows, 1),
        (ExecutionVariant::DecodeGrouped, decode_rows, 1),
        (ExecutionVariant::Prefill, rows, if rows > decode_rows { 1 } else { 0 })] {
        if count == 0 { continue; }
        let spec = program_spec(programs, variant.program(family, activations))?;
        let workspace = execution_workspace(&spec, rows, ExpertSet::Routed, variant, unit_limit, activations)?;
        ensure!(hidden.is_none_or(|h| h == workspace.hidden), "EXLR8 routed workspace hidden mismatch");
        hidden = Some(workspace.hidden);
        total = total.checked_add(workspace.bytes.checked_mul(usize::try_from(count)?)
            .context("EXLR8 routed lane workspace overflow")?).context("EXLR8 routed workspace overflow")?;
    }
    let output = rows.max(decode_rows).checked_mul(hidden.context("EXLR8 missing decode workspace")?)
        .and_then(|bytes| bytes.checked_mul(2)).context("EXLR8 routed output overflow")?;
    Ok(u64::try_from(total.checked_add(output).context("EXLR8 routed workspace overflow")?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(variant: ExecutionVariant, activations: PrefillActivations) -> Value {
        let prefill = variant == ExecutionVariant::Prefill;
        let grouped = matches!(variant, ExecutionVariant::DecodeGrouped | ExecutionVariant::Prefill);
        let (rows, groups) = if prefill { (4096u64, 4096u64) } else { (64, 512) };
        let mut pointers = vec!["x", "pointers", "bits", "group_expert", "route_slots", "route_group_lane",
            "route_weights", "fc1", "intermediate", "fc2", "signs_out", "output", "group_count"];
        if grouped { pointers.extend(["unit_count", "unit_map"]); }
        let mut value = json!({"name":variant.program(KernelFamily::Glm, activations), "capacity_rows":rows,
            "pointers":pointers.iter().map(|name| json!({"name":name})).collect::<Vec<_>>(),
            "scalars":(0..if grouped {4} else {2}).map(|_| json!({"type":"int32"})).collect::<Vec<_>>(),
            "geometry":{"abi_version":3,"hidden":6144,"units":16,"top_k":8,
                "max_groups":groups,"max_rows":rows,"group_rows":16,"split_k_blocks":8,
                "fc1_splits":6,"ctas_per_sm":4},
            "scratch_bytes_at_capacity":{"fc1":groups*16*2*6*16*128*4,
                "intermediate":groups*16*16*128*2,"fc2":groups*16*6144*4}});
        if variant == ExecutionVariant::DecodeCompact {
            value["geometry"].as_object_mut().unwrap().remove("group_rows");
            value["geometry"]["serving_variant"] = json!("k1_k23_coalesced");
        }
        if variant == ExecutionVariant::DecodeGrouped { value["geometry"]["fc2_partials"] = json!("f32"); }
        if activations == PrefillActivations::A8 && prefill { value["geometry"]["activations"] = json!("a8"); }
        value
    }

    fn shared_entries(family: KernelFamily) -> Value {
        let h = family.geometry().hidden as u64;
        let mut core = entry(ExecutionVariant::SharedDecode, PrefillActivations::A16);
        core["name"] = json!(family.program(KernelKind::Shared));
        core["geometry"]["hidden"] = json!(h);
        core["geometry"]["top_k"] = json!(1);
        core["geometry"]["max_groups"] = json!(4);
        core["geometry"]["fc1_splits"] = json!(h.div_ceil(1024));
        if family == KernelFamily::GlmFlash { core["geometry"]["swiglu_limit"] = json!(10.0); }
        core["scratch_bytes_at_capacity"] = json!({"fc1":4*16*2*h.div_ceil(1024)*16*128*4,
            "intermediate":4*16*16*128*2,"fc2":4*16*h*4});
        let descriptors = [("ids", "int32", "[rows,1]".into(), "in"),
            ("x", "bfloat16", format!("[rows,{h}]"), "in"),
            ("signs_in", "int8", format!("[{h}]"), "in"),
            ("group_expert", "int32", "[4]".into(), "out"),
            ("route_slots", "int32", "[4,16]".into(), "out"),
            ("route_group_lane", "int32", "[rows,1]".into(), "out"),
            ("group_count", "int32", "[1]".into(), "out"),
            ("activation", "float16", format!("[rows,{h}]"), "out")];
        let group = json!({"name":family.program(KernelKind::SharedGroup),"capacity_rows":64,
            "geometry":{"abi_version":2,"experts":1,"top_k":1,"max_groups":4,"max_rows":64,
                "hidden":h,"input_format":"bf16","input_row_bytes":h*2,"ctas_per_sm":4,"ingress_warps":8},
            "pointers":descriptors.iter().map(|(name,dtype,shape,role)|
                json!({"name":name,"dtype":dtype,"shape":shape,"role":role})).collect::<Vec<_>>(),
            "scalars":[{"type":"int32"},{"type":"int32"}],"scratch_bytes_at_capacity":{}});
        json!({"programs":[core,group]})
    }

    #[test]
    fn named_execution_owners_keep_full_scratch_and_direct_split_bindings() {
        use crate::placement::planner::{Domain, Lifetime, Location};
        for family in [KernelFamily::Glm, KernelFamily::GlmFlash] {
            for variant in [ExecutionVariant::SharedDecode, ExecutionVariant::DecodeCompact,
                ExecutionVariant::DecodeGrouped, ExecutionVariant::Prefill] {
                for activations in [PrefillActivations::A16, PrefillActivations::A8] {
                    let mut value = if variant == ExecutionVariant::SharedDecode {
                        shared_entries(family)["programs"][0].clone()
                    } else { entry(variant, activations) };
                    let h = family.geometry().hidden as u64;
                    let g = value["geometry"]["max_groups"].as_u64().unwrap();
                    value["name"] = json!(variant.program(family, activations));
                    value["geometry"]["hidden"] = json!(h);
                    value["geometry"]["fc1_splits"] = json!(h.div_ceil(1024));
                    if family == KernelFamily::GlmFlash { value["geometry"]["swiglu_limit"] = json!(10.0); }
                    value["scratch_bytes_at_capacity"] = json!({"fc1":g*16*2*h.div_ceil(1024)*16*128*4,
                        "intermediate":g*16*16*128*2,"fc2":g*16*h*4});
                    let manifest = json!({"programs":[value]});
                    let spec = program_spec(&manifest, variant.program(family, activations)).unwrap();
                    let set = if variant == ExecutionVariant::SharedDecode { ExpertSet::Shared } else { ExpertSet::Routed };
                    let rows = if variant == ExecutionVariant::Prefill { 65 } else { 1 };
                    let layout = execution_workspace(&spec, rows, set, variant, 3, activations).unwrap();
                    let records = layout.allocations("test.execution", Location::coordinator(0), Lifetime::RESIDENT).unwrap();
                    assert_eq!(records.len(), 14);
                    let owner = &records[0];
                    assert!(owner.view.is_none());
                    assert_eq!(owner.bytes, layout.bytes as u64);
                    assert_eq!(records.iter().filter(|r| r.view.is_none()).count(), 1);
                    for binding in &records[1..] {
                        let view = binding.view.as_ref().unwrap();
                        assert_eq!(view.backing, owner.id);
                        assert_eq!(view.offset % 256, 0);
                        assert!(view.offset + binding.bytes <= owner.bytes);
                    }
                    let combined = records.iter().find(|r| r.id.ends_with(".intermediate_fc2")).unwrap();
                    let intermediate = records.iter().find(|r| r.id.ends_with(".intermediate")).unwrap();
                    let fc2 = records.iter().find(|r| r.id.ends_with(".fc2")).unwrap();
                    assert_eq!(intermediate.bytes, layout.intermediate_bytes as u64);
                    assert_eq!(intermediate.bytes + fc2.bytes, combined.bytes);
                    assert_eq!(intermediate.view.as_ref().unwrap().offset, combined.view.as_ref().unwrap().offset);
                    assert_eq!(fc2.view.as_ref().unwrap().offset,
                        combined.view.as_ref().unwrap().offset + intermediate.bytes);
                    assert!(layout.allocations("", Location::coordinator(0), Lifetime::RESIDENT).is_err());
                    assert!(layout.allocations("test", Location { domain: Domain::Pinned,
                        ..Location::coordinator(0) }, Lifetime::RESIDENT).is_err());
                    for negative in 0..6 {
                        let mut invalid = execution_workspace(&spec, rows, set, variant, 3, activations).unwrap();
                        match negative {
                            0 => invalid.ranges[1].0 += 1,
                            1 => invalid.ranges[1].0 = invalid.ranges[0].0,
                            2 => invalid.ranges[1].1 = 0,
                            3 => invalid.ranges[10].1 = usize::MAX,
                            4 => invalid.bytes += 256,
                            5 => invalid.intermediate_bytes = invalid.ranges[9].1,
                            _ => unreachable!(),
                        }
                        assert!(invalid.allocations("test", Location::coordinator(0), Lifetime::RESIDENT).is_err(),
                            "{family:?} {variant:?} {negative}");
                    }
                }
            }
        }
    }

    #[test]
    fn shared_arenas_use_the_runtime_layout_for_both_families() {
        for family in [KernelFamily::Glm, KernelFamily::GlmFlash] {
            let manifest = shared_entries(family);
            let h = family.geometry().hidden;
            let layout = shared_workspace_layout(&manifest, family).unwrap();
            let bytes = shared_workspace(&manifest, family).unwrap();
            let (ranges, expected) = execution_workspace_layout(64,64,16,h,1,4,
                4*16*2*h.div_ceil(1024)*16*128*4,4*16*16*128*2,4*16*h*4,
                ExpertSet::Shared).unwrap();
            assert_eq!(bytes, expected as u64);
            assert_eq!(layout.bytes, expected);
            assert_eq!(layout.ranges, ranges);
            assert_eq!(layout.allocations("test.shared", crate::placement::planner::Location::coordinator(0),
                crate::placement::planner::Lifetime::RESIDENT).unwrap().len(), 14);
            for missing in 0..2 {
                let mut invalid = manifest.clone();
                invalid["programs"].as_array_mut().unwrap().remove(missing);
                assert!(shared_workspace(&invalid, family).is_err());
            }
            for (field,value) in [("abi_version",1),("hidden",128),("top_k",8),("units",0),("max_groups",512)] {
                let mut invalid = manifest.clone();
                invalid["programs"][0]["geometry"][field] = json!(value);
                assert!(shared_workspace(&invalid, family).is_err(), "{field}");
            }
            for bytes in [0,u64::MAX] {
                let mut invalid = manifest.clone();
                invalid["programs"][0]["scratch_bytes_at_capacity"]["fc2"] = json!(bytes);
                assert!(shared_workspace(&invalid, family).is_err());
            }
            let mut invalid = manifest.clone();
            invalid["programs"][1]["pointers"][1]["dtype"] = json!("float16");
            assert!(shared_workspace(&invalid, family).is_err());
            invalid = manifest.clone();
            invalid["programs"].as_array_mut().unwrap().push(manifest["programs"][1].clone());
            assert!(shared_workspace(&invalid, family).is_err());
            let other = if family == KernelFamily::Glm { KernelFamily::GlmFlash } else { KernelFamily::Glm };
            assert!(shared_workspace(&manifest, other).is_err());
        }
        assert!(shared_workspace(&json!({"programs":[]}), KernelFamily::Glm).is_err());
    }

    #[test]
    fn adaptive_units_size_row_planes_without_shrinking_exported_scratch() {
        for activations in [PrefillActivations::A16, PrefillActivations::A8] {
            let programs = json!({"programs":[entry(ExecutionVariant::Prefill, activations)]});
            let spec = program_spec(&programs, ExecutionVariant::Prefill.program(KernelFamily::Glm, activations)).unwrap();
            let full = execution_workspace(&spec, 4096, ExpertSet::Routed,
                ExecutionVariant::Prefill, 8, activations).unwrap();
            for rows in [2048, 2752] {
                let unit = execution_workspace(&spec, rows, ExpertSet::Routed,
                    ExecutionVariant::Prefill, 8, activations).unwrap();
                assert_eq!(unit.rows, rows);
                assert_eq!(unit.ranges[0].1, rows * (6144 + 6144 / 32));
                assert_eq!(unit.ranges[1].1, rows * 8 * 4);
                assert_eq!(unit.ranges[3].1, rows * 6144 * 2);
                for index in [4, 5, 8, 9] {
                    assert_eq!(unit.ranges[index].1, full.ranges[index].1);
                }
                assert_eq!(unit.max_groups, full.max_groups);
                assert!(unit.bytes < full.bytes);
            }
        }
    }

    #[test]
    fn flash_arenas_derive_every_byte_from_the_validated_program_geometry() {
        for activations in [PrefillActivations::A16, PrefillActivations::A8] {
            let family = KernelFamily::GlmFlash;
            let entries = [ExecutionVariant::DecodeCompact, ExecutionVariant::DecodeGrouped, ExecutionVariant::Prefill]
                .map(|variant| {
                    let mut e = entry(variant, activations);
                    e["name"] = json!(variant.program(family, activations));
                    let g = if variant == ExecutionVariant::Prefill { 800u64 } else { 512 };
                    let r = if variant == ExecutionVariant::Prefill { 64u64 } else { 16 };
                    e["geometry"]["hidden"] = json!(4096);
                    e["geometry"]["fc1_splits"] = json!(4);
                    e["geometry"]["max_groups"] = json!(g);
                    e["geometry"]["swiglu_limit"] = json!(10.0);
                    if variant != ExecutionVariant::DecodeCompact { e["geometry"]["group_rows"] = json!(r); }
                    e["scratch_bytes_at_capacity"] = json!({"fc1":g*16*2*4*r*128*4,
                        "intermediate":g*16*r*128*2,"fc2":g*r*4096*4});
                    e
                });
            let programs = json!({"programs":entries});
            for rows in [1usize, 64, 65, 2752, 4096] {
                let mut expected = rows.max(64) * 4096 * 2;
                for variant in [ExecutionVariant::DecodeCompact, ExecutionVariant::DecodeGrouped, ExecutionVariant::Prefill] {
                    if variant == ExecutionVariant::Prefill && rows <= 64 { continue; }
                    let spec = program_spec(&programs, variant.program(family, activations)).unwrap();
                    let live = if variant == ExecutionVariant::Prefill { rows } else { 64 };
                    let arena = execution_workspace(&spec, live, ExpertSet::Routed, variant, 8, activations).unwrap();
                    assert_eq!(arena.hidden, 4096);
                    assert_eq!(arena.ranges[0].1, live * (4096 + 4096 / 32));
                    expected += arena.bytes;
                }
                assert_eq!(routed_workspace(&programs, family, rows as u64, 8, activations).unwrap(), expected as u64);
                assert!(routed_workspace(&programs, KernelFamily::Glm, rows as u64, 8, activations).is_err());
            }
        }
    }

    #[test]
    fn routed_backend_charges_both_decode_arenas_prefill_and_stable_output() {
        for activations in [PrefillActivations::A16, PrefillActivations::A8] {
            let programs = json!({"programs":[entry(ExecutionVariant::DecodeCompact, activations),
                entry(ExecutionVariant::DecodeGrouped, activations), entry(ExecutionVariant::Prefill, activations)]});
            for rows in [1u64, 64, 65, 1024, 2048, 2752, 4096] {
                let mut expected = rows.max(64) * 6144 * 2;
                for variant in [ExecutionVariant::DecodeCompact, ExecutionVariant::DecodeGrouped, ExecutionVariant::Prefill] {
                    if variant == ExecutionVariant::Prefill && rows <= 64 { continue; }
                    let spec = program_spec(&programs, variant.program(KernelFamily::Glm, activations)).unwrap();
                    let live = if variant == ExecutionVariant::Prefill { rows as usize } else { 64 };
                    let owned = execution_workspace(&spec, live, ExpertSet::Routed, variant, 8, activations).unwrap();
                    expected += owned.bytes as u64;
                    assert_eq!(owned.compact, true);
                    assert_eq!(owned.intermediate_bytes, spec.scratch["intermediate"] as usize
                        / if variant == ExecutionVariant::DecodeCompact { 1 } else { 2 });
                    let full = execution_workspace(&spec, live, ExpertSet::Routed, variant, 16, activations).unwrap();
                    assert!(full.bytes > owned.bytes);
                }
                assert_eq!(routed_workspace(&programs, KernelFamily::Glm, rows, 8, activations).unwrap(), expected);
            }
            for units in [0, 17, usize::MAX] { assert!(routed_workspace(&programs, KernelFamily::Glm, 4096, units, activations).is_err()); }
            assert!(routed_workspace(&programs, KernelFamily::Glm, 4097, 8, activations).is_err());
            let mut invalid = programs.clone();
            invalid["programs"][1]["scratch_bytes_at_capacity"]["fc2"] = json!(1);
            assert!(routed_workspace(&invalid, KernelFamily::Glm, 4096, 8, activations).is_err());
            invalid = programs.clone();
            invalid["programs"].as_array_mut().unwrap().remove(1);
            assert!(routed_workspace(&invalid, KernelFamily::Glm, 4096, 8, activations).is_err());
        }
    }
}
