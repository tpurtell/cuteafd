//! Conservative installed-package bounds until manifest-backed planning (#776).
use super::*;
use crate::formats::fp8_experts::ExpertFormat;
use crate::placement::SparkWorkingSet;

/// Installed-package scratch census (#778, issuecomment-6401148082): every
/// shipped FP8-family package's scratch at each export capacity. A typed owner
/// is charged its compiled export's maximum over the family/format packages.
/// MiMo/Step rows include their BF16-input siblings.
const CENSUS_CAPACITIES: [u32; 8] = cuteafd_core::EXL3_CAPACITIES;

fn group_scratch(family: &str, format: ExpertFormat, compiled: u32) -> anyhow::Result<u64> {
    use std::sync::OnceLock;
    static CENSUS: OnceLock<Vec<(String, String, Vec<u64>)>> = OnceLock::new();
    let census = CENSUS.get_or_init(|| {
        let rows: Vec<serde_json::Value> = serde_json::from_str(include_str!("spark_slot_census.json"))
            .expect("bundled Spark slot census is valid JSON");
        rows.into_iter().map(|row| (row["family"].as_str().unwrap_or_default().to_owned(),
            row["format"].as_str().unwrap_or_default().to_owned(),
            row["scratch"].as_array().map(|values| values.iter().filter_map(serde_json::Value::as_u64).collect())
                .unwrap_or_default())).collect()
    });
    let format_name = match format {
        ExpertFormat::Fp8Block128 => "fp8", ExpertFormat::Mxfp4 => "mxfp4",
        ExpertFormat::Nvfp4 => "nvfp4", ExpertFormat::Bf16 => "bf16", _ => "",
    };
    let index = CENSUS_CAPACITIES.iter().position(|&rows| rows == compiled)
        .ok_or_else(|| anyhow::anyhow!("no Spark scratch census at export m{compiled}"))?;
    census.iter().filter(|(f, k, _)| f == family && k == format_name)
        .filter_map(|(.., scratch)| scratch.get(index).copied()).max()
        .ok_or_else(|| anyhow::anyhow!("no calibrated Spark slot scratch for {family}/{format:?}; add the package census or manifest-backed sizing (#776)"))
}

/// Stream-ordered device input/route/output planes of one owner at `rows`.
fn group_io_bytes(family: &str, format: ExpertFormat, geometry: cuteafd_core::ExpertGeometry, rows: u32) -> u64 {
    let hidden = u64::from(geometry.hidden);
    // BF16-input siblings are installed for block-FP8 MiMo/Step; NVFP4 and
    // BF16 weights consume BF16 directly. Other packages use FP8 K32 ingress.
    let bf16 = matches!(format, ExpertFormat::Nvfp4 | ExpertFormat::Bf16)
        || (matches!(family, "mimo_v2" | "step5") && format == ExpertFormat::Fp8Block128);
    let input = if bf16 { 2 * hidden } else { hidden + hidden / 32 };
    u64::from(rows) * (input + 8 * u64::from(geometry.topk) + 2 * hidden)
}

/// The worker's retained FP8-family groups for this rank: unequal rotated
/// widths and mixed weight formats each retain separate packages; repeated
/// layers of one group do not.
fn native_groups(input: &super::super::PlanInput, family: &str, working: &SparkWorkingSet, rank: usize)
    -> anyhow::Result<Option<(cuteafd_core::ExpertGeometry, Vec<ExpertFormat>)>> {
    if !working.selected_ranks().contains(&rank) { return Ok(None); }
    let catalog = input.expert_catalog()?;
    let Some(tensors) = catalog.fp8() else { return Ok(None); };
    let shape = tensors.shape();
    let geometry = cuteafd_core::ExpertGeometry { hidden: u32::try_from(shape.hidden)?,
        experts: u32::try_from(shape.experts)?, topk: u32::try_from(shape.topk)?,
        intermediate: u32::try_from(shape.intermediate)?, layers: u32::try_from(shape.layers)? };
    let mut groups = Vec::new();
    let mut formats = Vec::new();
    for &layer in &working.spark_layers {
        if !tensors.has_layer(layer) { continue; }
        let view = tensors.for_layer_metadata(layer)?;
        let format = view.format();
        let package_family = match family {
            "glm5" => "glm", "glm5_flash" => "glmf",
            "mimo_v2" => if format == ExpertFormat::Mxfp4 {
                if shape.hidden == 6144 { "mimop" } else { "mimof" }
            } else { "mimo" },
            other => other,
        };
        let format_name = match format {
            ExpertFormat::Nvfp4 => "nvfp4", ExpertFormat::Bf16 => "bf16", _ => "fp8",
        };
        let recipe = super::super::experts::spark_recipe(&format!("{package_family}:{format_name}"),
            shape.intermediate, working.spark_ranks)?;
        let width = view.rank_width(working.spark_ranks, rank, recipe.slicing)?;
        let key = (format, view.activation(layer), width);
        if !groups.contains(&key) {
            groups.push(key);
            formats.push(format);
        }
    }
    anyhow::ensure!(!formats.is_empty(), "native Spark owner has no selected expert groups");
    Ok(Some((geometry, formats)))
}

/// One FP8-family owner: groups share one stream and one scratch arena sized
/// for the largest group package, while each group keeps its own I/O planes.
fn fp8_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry, formats: &[ExpertFormat],
    extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    let mut scratch = 0;
    let mut io = 0u64;
    for &format in formats {
        scratch = scratch.max(group_scratch(family, format, extent.compiled)?);
        io += group_io_bytes(family, format, geometry, extent.capacity);
    }
    Ok(cuteafd_core::OwnerBytes { scratch, io, pinned_staging: 0 })
}

/// EXL3 owner bytes. Prefill-scale exports charge the calibrated retained
/// arena maxima plus private capacity state (#762 census totals, which include
/// 4096 rows of K32 wire/routes; those rows are re-charged as the owner's I/O
/// at its own capacity). Decode exports (m80 and below) charge the m1/m16/m80
/// retained set (installed SM121 exports, 2026-10-24: largest GLM tp1
/// 50,445,548 B). Ownership-tagged routes stage per queued unit.
pub(crate) fn exl3_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry, extent: cuteafd_core::OwnerExtent)
    -> anyhow::Result<cuteafd_core::OwnerBytes> {
    const DECODE_SET: u64 = 51 * (1 << 20);
    let hidden = u64::from(geometry.hidden);
    let topk = u64::from(geometry.topk);
    let row = hidden + hidden / 32 + 8 * topk;
    let scratch = if extent.compiled <= 80 { DECODE_SET } else {
        let total = match family {
            "glm5_flash" => 607 * (1 << 20),
            "deepseek_v4" | "deepseek_v41" => 810 * (1 << 20),
            "kimi_k3" => crate::families::kimi_k3::SPARK_SLOT_BYTES,
            other => spark_slot_bytes(other, 4096),
        };
        anyhow::ensure!(total > 0, "no calibrated Spark EXL3 scratch for {family}; add the package census (#776)");
        total.saturating_sub(4096 * row)
    };
    Ok(cuteafd_core::OwnerBytes { scratch, io: u64::from(extent.capacity) * row,
        pinned_staging: u64::from(extent.staging_rows) * topk * 8 })
}

/// Legacy calibrated slot bound (stream, scratch, inputs and staging at
/// the compiled 4096-row export) for backends without a census.
fn legacy_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry, extent: cuteafd_core::OwnerExtent,
    staged: bool) -> cuteafd_core::OwnerBytes {
    let at_4096 = spark_slot_bytes(family, 4096);
    cuteafd_core::OwnerBytes { scratch: at_4096 * u64::from(extent.compiled.max(80)) / 4096, io: 0,
        pinned_staging: if staged { u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 } else { 0 } }
}

/// One owner's bytes on this rank's installed backend.
pub(crate) fn owner_bytes(input: &super::super::PlanInput, working: &SparkWorkingSet, family: &str,
    rank: usize) -> anyhow::Result<impl Fn(cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes>> {
    use crate::plan::spark_executions::SparkBackend;
    let catalog = input.expert_catalog()?;
    let geometry = catalog.routed_experts().geometry()?;
    let backend = SparkBackend::for_catalog(catalog)?;
    let groups = native_groups(input, family, working, rank)?;
    let family = family.to_owned();
    Ok(move |extent| match (&groups, backend) {
        (Some((geometry, formats)), SparkBackend::Fp8) => fp8_owner_bytes(&family, *geometry, formats, extent),
        (_, SparkBackend::Exl3(_)) => exl3_owner_bytes(&family, geometry, extent),
        (_, SparkBackend::Native) => Ok(legacy_owner_bytes(&family, geometry, extent, true)),
        (_, SparkBackend::Exlr8 | SparkBackend::Fp8) => Ok(legacy_owner_bytes(&family, geometry, extent, false)),
    })
}

/// Charge one selected Spark rank's typed execution owners: one prefill
/// scratch, every decode owner's scratch, and per-unit pinned staging times
/// the buffered units, through the same owner sum the worker uses.
pub(crate) fn charge_executions(device: &mut DeviceLayout, input: &super::super::PlanInput,
    working: &SparkWorkingSet, family: &str, executions: &cuteafd_core::SparkExecutions) -> anyhow::Result<()> {
    device.items.retain(|item| item.group != "execution owners" && item.group != "execution staging");
    if !working.selected_ranks().contains(&(device.index as usize)) { return Ok(()); }
    charge_owners(device, executions, owner_bytes(input, working, family, device.index as usize)?)
}

/// Record the owner sum, crediting only bytes the legacy first-slot residual
/// already reserves on this device (at most the one prefill owner it modelled).
pub(crate) fn charge_owners(device: &mut DeviceLayout, executions: &cuteafd_core::SparkExecutions,
    bytes: impl Fn(cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes>) -> anyhow::Result<()> {
    device.items.retain(|item| item.group != "execution owners" && item.group != "execution staging");
    let total = executions.workspace_bytes(&bytes)?
        .ok_or_else(|| anyhow::anyhow!("Spark execution owner workspace overflow"))?;
    let prefill = executions.owners().last().map(&bytes).transpose()?.unwrap_or_default();
    let residual = device.items.iter().filter(|item| item.group == "expert waves")
        .map(|item| item.bytes).sum::<u64>();
    let credit = residual.min(prefill.scratch + prefill.io);
    let shape = executions.prefill;
    device.items.push(Item::new(Category::Workspace, "execution owners",
        format!("{} decode + 1 prefill (m{} scratch once; {} x {}-row units); {credit} bytes already reserved",
            executions.decode.len(), shape.compiled_capacity, shape.buffered_units, shape.unit_rows),
        total.device_bytes - credit, Basis::Calibrated));
    if total.pinned_bytes > 0 {
        device.items.push(Item::new(Category::Workspace, "execution staging",
            format!("pinned route staging, {} units", shape.buffered_units), total.pinned_bytes, Basis::Formula));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(capacity: u32, compiled: u32) -> cuteafd_core::OwnerExtent {
        cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill, capacity, compiled,
            staging_rows: capacity, buffered_units: 1 }
    }

    #[test]
    fn planner_charges_every_shipped_native_package_census_at_its_export() {
        let census: serde_json::Value = serde_json::from_str(include_str!("spark_slot_census.json")).unwrap();
        for row in census.as_array().unwrap() {
            let package = row["package"].as_str().unwrap();
            let family = row["family"].as_str().unwrap();
            let format = match row["format"].as_str().unwrap() {
                "fp8" => ExpertFormat::Fp8Block128, "mxfp4" => ExpertFormat::Mxfp4,
                "nvfp4" => ExpertFormat::Nvfp4, "bf16" => ExpertFormat::Bf16, _ => unreachable!(),
            };
            let hidden = row["hidden"].as_u64().unwrap();
            let topk = row["topk"].as_u64().unwrap();
            let input = if row["bf16"].as_bool().unwrap() { 2 * hidden } else { hidden + hidden / 32 };
            let geometry = cuteafd_core::ExpertGeometry { hidden: hidden as u32, topk: topk as u32,
                experts: 1, intermediate: 1, layers: 1 };
            for (index, compiled) in CENSUS_CAPACITIES.into_iter().enumerate() {
                let rows = u64::from(compiled);
                let exact = row["scratch"][index].as_u64().unwrap() + rows * (input + 8 * topk + 2 * hidden);
                let owner = fp8_owner_bytes(family, geometry, &[format], extent(compiled, compiled)).unwrap();
                assert!(owner.scratch + owner.io >= exact, "{package} m{compiled}: planner < census {exact}");
                assert_eq!(owner.pinned_staging, 0, "FP8 reads retained ring frames");
            }
        }
    }

    #[test]
    fn decode_owner_scratch_is_its_own_export_not_the_prefill_maximum() {
        let geometry = cuteafd_core::ExpertGeometry::GLM5;
        let decode = fp8_owner_bytes("glm5", geometry, &[ExpertFormat::Nvfp4], extent(64, 80)).unwrap();
        let prefill = fp8_owner_bytes("glm5", geometry, &[ExpertFormat::Nvfp4], extent(4096, 4096)).unwrap();
        assert_eq!((decode.scratch, prefill.scratch), (142_576_640, 667_162_624));
        // Live rows below the export never shrink compiled scratch, only I/O.
        let partial = fp8_owner_bytes("glm5", geometry, &[ExpertFormat::Nvfp4], extent(2752, 4096)).unwrap();
        assert_eq!(partial.scratch, prefill.scratch);
        assert_eq!(prefill.io - partial.io, (4096 - 2752) * (4 * 6144 + 8 * 8));
        assert!(group_scratch("glm5", ExpertFormat::Nvfp4, 64).is_err());
    }

    #[test]
    fn rotated_widths_sum_worker_owners_but_repeated_layers_do_not() -> anyhow::Result<()> {
        use crate::plan::testing;
        let root = tempfile::tempdir()?;
        let mut cfg = testing::mimo_pro_config();
        cfg["num_hidden_layers"] = serde_json::json!(7);
        cfg["hybrid_layer_pattern"] = serde_json::json!([0, 1, 1, 1, 1, 1, 1]);
        cfg["moe_layer_freq"] = serde_json::json!([0, 1, 1, 1, 1, 1, 1]);
        let mut tensors = testing::mimo_pro_tensors();
        let layer = tensors.iter().filter(|(name, ..)| name.starts_with("model.layers.1."))
            .cloned().collect::<Vec<_>>();
        for index in 2..7 {
            tensors.extend(layer.iter().map(|(name, dtype, shape)|
                (name.replacen("model.layers.1.", &format!("model.layers.{index}."), 1), *dtype, shape.clone())));
        }
        testing::write_snapshot(root.path(), &cfg, &tensors, None);
        let input = crate::plan::PlanInput::open(root.path())?;
        let mut working = SparkWorkingSet { spark_layers: vec![1], spark_ranks: 6,
            prefill_lanes: 2, ..Default::default() };
        let prefill = extent(4096, 4096);
        let one = owner_bytes(&input, &working, "mimo_v2", 0)?(prefill)?;
        working.spark_layers = (1..7).collect();
        let two = owner_bytes(&input, &working, "mimo_v2", 0)?(prefill)?;
        // w352 and w320 retain distinct groups (not six layers): one shared
        // scratch at the larger group's package, each group's own I/O planes.
        assert_eq!((two.scratch, two.io), (one.scratch, 2 * one.io));
        assert!(two.scratch >= 472_126_464 && two.io >= 2 * 76_546_048);
        Ok(())
    }

    #[test]
    fn exl3_bounds_include_private_capacities_and_low_row_alias_owners() {
        use serde_json::json;
        // v3.0.1 SM121 TP2 maxima. rotation_up aliases rotation_gate at
        // m4096, but smaller capacities retain a separate owner of that name.
        for (family, hidden, topk, arenas, expected) in [
            ("glm5_flash", 4096, 8, vec![("rotation_gate", 268_435_456), ("rotation_up", 5_242_880),
                ("fc1", 134_217_728), ("activated", 67_108_864), ("output", 33_554_432),
                ("packed_route_indices", 203_648), ("block_expert_ids", 3_184), ("packed_route_count", 16),
                ("expert_offsets", 1_156), ("expert_counts", 1_152),
                ("fc1_scratch", 12_582_912), ("fc2_scratch", 12_582_912)], 635_693_380),
            ("deepseek_v4", 7168, 6, vec![("rotation_gate", 352_321_536), ("rotation_up", 6_881_280),
                ("fc1", 150_994_944), ("activated", 75_497_472), ("output", 58_720_256),
                ("packed_route_indices", 195_072), ("block_expert_ids", 3_048), ("packed_route_count", 16),
                ("expert_offsets", 1_540), ("expert_counts", 1_536),
                ("fc1_scratch", 12_582_912), ("fc2_scratch", 12_582_912)], 847_569_468),
            ("deepseek_v41", 5120, 6, vec![("rotation_gate", 251_658_240), ("rotation_up", 251_658_240),
                ("fc1", 113_246_208), ("activated", 56_623_104), ("output", 41_943_040),
                ("packed_route_indices", 109_056), ("block_expert_ids", 13_632), ("packed_route_count", 16),
                ("expert_offsets", 1_540), ("expert_counts", 1_536),
                ("fc1_scratch", 3_145_728), ("fc2_scratch", 3_145_728)], 848_604_052),
        ] {
            let manifests = cuteafd_core::EXL3_CAPACITIES.into_iter().map(|rows| {
                let mut buffers = serde_json::Map::new();
                for &(name, bytes) in &arenas {
                    buffers.insert(name.into(), json!({"allocation": name, "bytes": bytes,
                        "dtype": "census", "zero_on_create": false}));
                }
                buffers.insert("workspace".into(), json!({"allocation": "workspace", "bytes": 776,
                    "dtype": "i32", "zero_on_create": true}));
                json!({"capacity": rows, "hidden": hidden, "trellis_lut": {"bytes": 4096}, "buffers": buffers})
            }).collect::<Vec<_>>();
            let retained = crate::serving_capacity::exl3_workspace_bytes(&manifests, true).unwrap();
            let wire_routes = 4096 * (hidden + hidden / 32 + 8 * topk);
            assert_eq!(retained + wire_routes, expected, "{family}");
            let geometry = cuteafd_core::ExpertGeometry { hidden: hidden as u32, topk: topk as u32,
                experts: 1, intermediate: 1, layers: 1 };
            let owner = exl3_owner_bytes(family, geometry, extent(4096, 4096)).unwrap();
            assert!(owner.scratch + owner.io >= expected, "{family}");
        }
    }

    #[test]
    fn exl3_decode_owner_covers_every_installed_narrow_export_set() {
        // Largest retained m1/m16/m80 set among installed SM121 exports.
        let geometry = cuteafd_core::ExpertGeometry::GLM5;
        let decode = exl3_owner_bytes("glm5_flash", geometry, extent(64, 80)).unwrap();
        assert!(decode.scratch >= 50_445_548);
        assert_eq!(decode.pinned_staging, 64 * 8 * 8);
    }

    #[test]
    fn exl3_catalog_charges_one_prefill_owner_and_two_v41_decode_owners() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        crate::plan::testing::write_v41_exl3_snapshot(root.path(), 3);
        let input = crate::plan::PlanInput::open(root.path())?;
        let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            prefill_lanes: 2, ..Default::default() };
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 * GIB,
            items: vec![], kv_tokens: 0 };
        let exports = crate::plan::spark_executions::SparkBackend::for_catalog(input.expert_catalog()?)?.exports();
        let decode = 2 * (51 * (1 << 20) + 64 * (5120 + 5120 / 32 + 6 * 8));
        let charge = |device: &mut DeviceLayout, family: &str, unit: u64, waves: u64| -> anyhow::Result<(u64, u64)> {
            let executions = crate::plan::spark_executions::admit_family(family, &exports, None, unit, waves)?;
            charge_executions(device, &input, &working, "deepseek_v41", &executions)?;
            let find = |group| device.items.iter().find(|item| item.group == group).unwrap().bytes;
            Ok((find("execution owners"), find("execution staging")))
        };
        // V4.1's CED executor: two streams of its live chunk on one owner.
        for live in [256, 1024, 4096] {
            let (owners, staging) = charge(&mut device, "deepseek_v41", live, 1)?;
            let prefill = if live <= 80 { 51 * (1 << 20) } else { 810 * (1 << 20) - 4096 * (5120 + 5120 / 32 + 6 * 8) };
            assert_eq!(owners, prefill + live * (5120 + 5120 / 32 + 6 * 8) + decode);
            assert_eq!(staging, 2 * 64 * 6 * 8 + 2 * live * 6 * 8);
        }
        // A ladder executor on the same package: k units divide the reserve.
        let mut charged = Vec::new();
        for (unit, waves) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4)] {
            let (owners, staging) = charge(&mut device, "deepseek_v4", unit, waves)?;
            assert_eq!(owners, 810 * (1 << 20) + decode / 2);
            assert_eq!(staging, 64 * 6 * 8 + unit * waves * 6 * 8);
            charged.push(owners);
        }
        assert!(charged.windows(2).all(|pair| pair[0] == pair[1]), "{charged:?}");
        Ok(())
    }

    #[test]
    fn owners_credit_only_the_existing_prefill_residual() {
        let executions = cuteafd_core::SparkExecutions::admit(&[64], 4096, 2048, 4, &cuteafd_core::EXL3_CAPACITIES).unwrap();
        let bytes = |extent: cuteafd_core::OwnerExtent| Ok(cuteafd_core::OwnerBytes {
            scratch: if extent.compiled == 4096 { 200 } else { 50 }, io: 0, pinned_staging: 1 });
        for residual in [0, 100, 1000] {
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: GIB,
                kv_tokens: 0, items: vec![Item::new(Category::Workspace, "expert waves", "", residual, Basis::Calibrated)] };
            charge_owners(&mut device, &executions, bytes).unwrap();
            let charged = device.items.iter().find(|item| item.group == "execution owners").unwrap().bytes;
            assert_eq!(charged, 250 - residual.min(200));
            assert_eq!(device.items.iter().find(|item| item.group == "execution staging").unwrap().bytes, 1 + 4);
        }
    }
}
