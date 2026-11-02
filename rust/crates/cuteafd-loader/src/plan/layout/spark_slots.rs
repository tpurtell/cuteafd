//! Conservative installed-package bounds until manifest-backed planning (#776).
use super::*;
use anyhow::Context;
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
    -> anyhow::Result<Option<(cuteafd_core::ExpertGeometry, Vec<(ExpertFormat, u32)>)>> {
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
            formats.push((format, width.try_into()?));
        }
    }
    anyhow::ensure!(!formats.is_empty(), "native Spark owner has no selected expert groups");
    Ok(Some((geometry, formats)))
}

/// One FP8-family owner: groups share one stream and one scratch arena sized
/// for the largest group package, while each group keeps its own I/O planes.
fn fp8_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry, formats: &[(ExpertFormat, u32)],
    extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    let mut scratch = 0;
    let mut io = 0u64;
    let mut address_staging = 0;
    for &(format, width) in formats {
        scratch = scratch.max(group_scratch(family, format, extent.compiled)?);
        io += group_io_bytes(family, format, geometry, extent.capacity);
        let minimum = match format { ExpertFormat::Fp8Block128 => 769, ExpertFormat::Mxfp4 => 641, _ => u32::MAX };
        if let Some(rows) = extent.segment_rows().filter(|_| extent.staging_rows >= minimum) {
            // Every installed form is retained; W8A16 is the largest stream arena.
            let bytes = cuteafd_core::spark_executions::segment_scratch_bytes(geometry.hidden, geometry.experts,
                geometry.topk, width.checked_mul(2).context("segment activation width overflow")?, rows,
                cuteafd_core::spark_executions::SEGMENT_MAX_ROWS).context("segment scratch overflow")?;
            scratch = scratch.max(bytes);
            io += u64::from(rows) * (16 + u64::from(geometry.topk) * 8);
            let sibling = format == ExpertFormat::Fp8Block128 && matches!(family, "mimo_v2" | "step5");
            let old_input = u64::from(extent.capacity) * if sibling { u64::from(geometry.hidden) * 2 }
                else { u64::from(geometry.hidden) * 33 / 32 };
            io += cuteafd_core::spark_executions::segment_input_bytes(extent.capacity, rows,
                geometry.hidden, sibling).context("segmented input extent overflow")? - old_input;
            address_staging += u64::from(extent.staging_rows) * 16;
        }
    }
    Ok(cuteafd_core::OwnerBytes { scratch, io,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 * formats.len() as u64 + address_staging })
}

/// EXL3 owner bytes. Prefill-scale exports charge the calibrated retained
/// arena maxima plus private capacity state (#762 census totals, which include
/// 4096 rows of K32 wire/routes; those rows are re-charged as the owner's I/O
/// at its own capacity). Decode exports (m80 and below) charge the m1/m16/m80
/// retained set (installed SM121 exports, 2026-10-24: largest GLM tp1
/// 50,445,548 B). Packed routes stage per queued unit.
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

/// Retained allocation witnesses of the widest installed ML4 slice in this
/// world. Capacity-private state and low-row alias owners match the worker.
pub(crate) fn ml4_owner_bytes(world: usize, extent: cuteafd_core::OwnerExtent)
    -> anyhow::Result<cuteafd_core::OwnerBytes> {
    use std::sync::OnceLock;
    static CENSUS: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    let census = CENSUS.get_or_init(|| serde_json::from_str(include_str!("spark_slot_census.json"))
        .expect("bundled Spark slot census is valid JSON"));
    let row = census.iter().find(|row| row["family"] == "mistral_large4" && row["format"] == "exl3"
        && row["world"].as_u64() == Some(world as u64))
        .with_context(|| format!("no Mistral Large 4 TP{world} EXL3 allocation census"))?;
    let geometry = cuteafd_core::ExpertGeometry::MISTRAL_LARGE4;
    let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(geometry).capacities(extent.capacity as usize)?;
    anyhow::ensure!(capacities.last() == Some(&extent.compiled), "ML4 owner export and retained capacities disagree");
    let installed = row["manifests"].as_array().context("ML4 census has no capacity witnesses")?;
    let manifests = capacities.into_iter().map(|capacity| installed.iter()
        .find(|manifest| manifest["capacity"].as_u64() == Some(u64::from(capacity)))
        .cloned().with_context(|| format!("Mistral Large 4 TP{world} census has no m{capacity}")))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let retained = crate::serving_capacity::exl3_workspace_bytes(&manifests, true)?;
    let wire_routes = u64::from(geometry.hidden) * 33 / 32 + u64::from(geometry.topk) * 8;
    Ok(cuteafd_core::OwnerBytes { scratch: retained, io: u64::from(extent.capacity) * wire_routes,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 })
}

/// Native width states coexist, while their stream and live I/O are shared.
fn mxfp4_owner_bytes(geometry: cuteafd_core::ExpertGeometry,
    artifacts: &[cuteafd_core::tp_units::NativeArtifact], coded: bool,
    extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    anyhow::ensure!(!artifacts.is_empty() && artifacts.len() <= 2,
        "native owner needs one or two retained width states");
    let mut scratch = 0u64;
    let mut capacities = vec![extent.compiled];
    if extent.compiled > 1 { capacities.push(1); }
    if extent.compiled > 80 { capacities.push(80); }
    for artifact in artifacts {
        for &capacity in &capacities {
            let bytes = cuteafd_core::mxfp4_layouts::native_scratch_bytes(geometry.hidden as usize,
                geometry.experts as usize, geometry.topk as usize, artifact.logical,
                artifact.role, capacity, coded).context("invalid native owner scratch geometry")?;
            scratch = scratch.checked_add(bytes).context("native width scratch sum overflow")?;
        }
    }
    Ok(cuteafd_core::OwnerBytes { scratch,
        io: u64::from(extent.capacity) * (u64::from(geometry.hidden) * 97 / 32
            + u64::from(geometry.topk) * 8),
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 })
}

fn mxfp4_rank_artifacts(catalog: &crate::OfficialV41Catalog, working: &SparkWorkingSet,
    rank: usize) -> anyhow::Result<Vec<cuteafd_core::tp_units::NativeArtifact>> {
    let shape = catalog.routed_experts();
    let artifacts = cuteafd_core::tp_units::native_spark_artifacts(shape.hidden,
        shape.intermediate, working.spark_ranks)?;
    let mut retained = Vec::new();
    for &layer in &working.spark_layers {
        if layer < shape.first_layer { continue; }
        let unit = cuteafd_core::mxfp4_layouts::ownership_unit(shape.intermediate, working.spark_ranks);
        let shard = cuteafd_core::tp_units::LayerShard::new(shape.intermediate, unit,
            layer, working.spark_ranks, rank)?;
        let artifact = artifacts.iter().find(|artifact| artifact.logical == shard.width)
            .context("native layer has no matching artifact")?;
        if !retained.contains(artifact) { retained.push(artifact.clone()); }
    }
    Ok(retained)
}

/// Legacy calibrated slot bound (stream, scratch, inputs and staging at
/// the compiled 4096-row export) for backends without a census.
fn legacy_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry,
    extent: cuteafd_core::OwnerExtent) -> cuteafd_core::OwnerBytes {
    let at_4096 = spark_slot_bytes(family, 4096);
    cuteafd_core::OwnerBytes { scratch: at_4096 * u64::from(extent.compiled.max(80)) / 4096, io: 0,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 }
}

/// One owner's bytes on this rank's installed backend.
pub(crate) fn owner_bytes(input: &super::super::PlanInput, working: &SparkWorkingSet, family: &str,
    rank: usize) -> anyhow::Result<impl Fn(cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes>> {
    use crate::plan::spark_executions::SparkBackend;
    let catalog = input.expert_catalog()?;
    let geometry = catalog.routed_experts().geometry()?;
    let backend = SparkBackend::for_catalog(catalog)?;
    let groups = native_groups(input, family, working, rank)?;
    let world = working.spark_ranks;
    let artifacts = if matches!(backend, SparkBackend::Native) && catalog.nvfp4().is_none() {
        Some(mxfp4_rank_artifacts(catalog, working, rank)?)
    } else { None };
    let coded = catalog.mxfp4_scale_codes();
    let family = family.to_owned();
    Ok(move |extent| {
        if let Some(artifacts) = &artifacts {
            return mxfp4_owner_bytes(geometry, artifacts, coded, extent);
        }
        match (&groups, backend) {
            (Some((geometry, formats)), SparkBackend::Fp8) => fp8_owner_bytes(&family, *geometry, formats, extent),
            (_, SparkBackend::Exl3(_)) if family == "mistral_large4" => ml4_owner_bytes(world, extent),
            (_, SparkBackend::Exl3(_)) => exl3_owner_bytes(&family, geometry, extent),
            (_, SparkBackend::Native | SparkBackend::Exlr8 | SparkBackend::Fp8) =>
                Ok(legacy_owner_bytes(&family, geometry, extent)),
        }
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
    fn native_width_states_sum_scratch_but_share_io_and_route_staging() -> anyhow::Result<()> {
        use cuteafd_core::tp_units::native_spark_artifacts;
        for (geometry, coded) in [(cuteafd_core::ExpertGeometry::DEEPSEEK_V41, true),
            (cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH, false),
            (cuteafd_core::ExpertGeometry::DEEPSEEK_V4_PRO, false)] {
            for world in 2..=8 {
                let artifacts = native_spark_artifacts(geometry.hidden as usize,
                    geometry.intermediate as usize, world)?;
                for compiled in [1, 16, 80, 256, 1024, 4096] {
                    let live = compiled.min(80);
                    let shape = extent(live, compiled);
                    let total = mxfp4_owner_bytes(geometry, &artifacts, coded, shape)?;
                    let singles = artifacts.iter().map(|artifact|
                        mxfp4_owner_bytes(geometry, std::slice::from_ref(artifact), coded, shape))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    assert_eq!(total.scratch, singles.iter().map(|owner| owner.scratch).sum::<u64>());
                    assert_eq!(total.io, singles[0].io);
                    assert_eq!(total.pinned_staging, singles[0].pinned_staging);
                    let capacities = [Some(compiled), (compiled > 1).then_some(1), (compiled > 80).then_some(80)];
                    let expected = artifacts.iter().map(|a| capacities.into_iter().flatten().map(|rows|
                        cuteafd_core::mxfp4_layouts::native_scratch_bytes(geometry.hidden as usize,
                            geometry.experts as usize, geometry.topk as usize, a.logical, a.role, rows, coded).unwrap())
                        .sum::<u64>()).sum::<u64>();
                    assert_eq!(total.scratch, expected);
                    assert_eq!(total.io, u64::from(live) * (u64::from(geometry.hidden) * 97 / 32 + u64::from(geometry.topk) * 8));
                    assert_eq!(mxfp4_owner_bytes(geometry, &artifacts, coded,
                        cuteafd_core::OwnerExtent { buffered_units: 3, ..shape })?, total);
                }
            }
        }
        assert!(mxfp4_owner_bytes(cuteafd_core::ExpertGeometry::DEEPSEEK_V41, &[], true, extent(80, 80)).is_err());
        Ok(())
    }

    #[test]
    fn planner_charges_every_shipped_native_package_census_at_its_export() {
        let census: serde_json::Value = serde_json::from_str(include_str!("spark_slot_census.json")).unwrap();
        for row in census.as_array().unwrap().iter().filter(|row| row["format"] != "exl3") {
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
                let owner = fp8_owner_bytes(family, geometry, &[(format, 128)], extent(compiled, compiled)).unwrap();
                assert!(owner.scratch + owner.io >= exact, "{package} m{compiled}: planner < census {exact}");
                assert_eq!(owner.pinned_staging, rows * topk * 8, "each unit retains packed route planes");
            }
        }
    }

    #[test]
    fn ml4_census_charges_each_worlds_retained_capacity_set() -> anyhow::Result<()> {
        let census: Vec<serde_json::Value> = serde_json::from_str(include_str!("spark_slot_census.json"))?;
        let rows: Vec<_> = census.iter().filter(|row| row["family"] == "mistral_large4"
            && row["format"] == "exl3").collect();
        assert_eq!(rows.iter().map(|row| row["world"].as_u64().unwrap()).collect::<Vec<_>>(), [5, 6, 7, 8]);
        for (row, (decode, prefill, maximum)) in rows.into_iter().zip([
            (18_810_348, 358_759_748, 559_042_636),
            (18_318_828, 341_851_460, 533_876_812),
            (18_073_068, 333_397_316, 521_293_900),
            (17_827_308, 324_943_172, 508_710_988),
        ]) {
            let world = row["world"].as_u64().unwrap() as usize;
            let width = (48usize.div_ceil(world) * 128) as u64;
            let dynamic: &[u32] = match world {
                5 => &[256, 2752, 4096],
                6 => &[256, 2048, 2752],
                7 => &[256, 1024, 2752],
                8 => &[256, 2048, 2752, 4096],
                _ => unreachable!(),
            };
            assert_eq!(row["dynamic_capacities"], serde_json::json!(dynamic));
            assert_eq!(ml4_owner_bytes(world, extent(64, 80))?.scratch, decode);
            assert_eq!(ml4_owner_bytes(world, extent(2752, 2752))?.scratch, prefill);
            let largest = ml4_owner_bytes(world, extent(4096, 4096))?;
            assert_eq!(largest.scratch + largest.io, maximum, "TP{world} full retained bound");
            let manifests = row["manifests"].as_array().unwrap();
            assert_eq!(manifests.len(), CENSUS_CAPACITIES.len());
            for (index, capacity) in CENSUS_CAPACITIES.into_iter().enumerate() {
                assert_eq!(manifests[index]["capacity"], capacity);
                assert_eq!(manifests[index]["intermediate"], width);
                let retained_set = cuteafd_core::Exl3RowPolicy::Nearest.capacities(capacity as usize)?;
                let compiled = *retained_set.last().unwrap();
                let selected = manifests.iter().filter(|manifest| retained_set.contains(
                    &(manifest["capacity"].as_u64().unwrap() as u32))).cloned().collect::<Vec<_>>();
                let retained = crate::serving_capacity::exl3_workspace_bytes(&selected, true)?;
                let without_wire = crate::serving_capacity::exl3_workspace_bytes(&selected, false)?;
                assert_eq!(retained - without_wire,
                    retained_set.iter().map(|&rows| u64::from(rows)).sum::<u64>() * 6144 * 2);
                let owner = ml4_owner_bytes(world, extent(capacity, compiled))?;
                assert_eq!(owner.scratch, retained, "TP{world} m{compiled}");
                assert_eq!(owner.io, u64::from(capacity) * (6144 + 192 + 32));
                assert_eq!(owner.pinned_staging, u64::from(capacity) * 32);
                assert_eq!(owner, ml4_owner_bytes(world, cuteafd_core::OwnerExtent {
                    buffered_units: 3, ..extent(capacity, compiled) })?, "retained arenas are not per wave");
            }
            let decode = ml4_owner_bytes(world, extent(64, 80))?;
            for capacity in [2, 16, 32, 64, 80] {
                let owner = ml4_owner_bytes(world, extent(capacity, 80))?;
                assert_eq!(owner.scratch, decode.scratch, "canonical m80 floor at TP{world}/{capacity}");
                assert_eq!(owner.io, u64::from(capacity) * (6144 + 192 + 32));
            }
            assert!(decode.scratch < ml4_owner_bytes(world, extent(2752, 2752))?.scratch);
        }
        assert!(ml4_owner_bytes(4, extent(64, 80)).is_err());
        assert!(ml4_owner_bytes(9, extent(64, 80)).is_err());
        assert!(ml4_owner_bytes(5, extent(64, 256)).is_err());
        Ok(())
    }

    #[test]
    fn ml4_catalog_uses_world_census_for_every_selected_rank() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        crate::plan::testing::write_mistral_large4_snapshot(root.path(), None, true);
        let input = crate::plan::PlanInput::open(root.path())?;
        for world in 5..=8 {
            let working = SparkWorkingSet { spark_layers: vec![1], spark_ranks: world,
                prefill_lanes: 1, ..Default::default() };
            for rank in 0..world {
                for (capacity, compiled) in [(64, 80), (2752, 2752)] {
                    let expected = ml4_owner_bytes(world, extent(capacity, compiled))?;
                    assert_eq!(owner_bytes(&input, &working, "mistral_large4", rank)?(
                        extent(capacity, compiled))?, expected, "TP{world} rank{rank}");
                }
                let executions = crate::plan::spark_executions::admit_family("mistral_large4",
                    &cuteafd_core::EXL3_CAPACITIES, None, 2752, 3)?;
                let total = executions.workspace_bytes(|extent| ml4_owner_bytes(world, extent))?.unwrap();
                let prefill = ml4_owner_bytes(world, executions.owners().last().unwrap())?;
                for residual in [0, prefill.scratch + prefill.io + 1] {
                    let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32,
                        capacity_bytes: 121 * GIB, kv_tokens: 0,
                        items: vec![Item::new(Category::Workspace, "expert waves", "", residual, Basis::Calibrated)] };
                    // Recharging replaces the owner entries, never doubles them.
                    for _ in 0..2 {
                        charge_executions(&mut device, &input, &working, "mistral_large4", &executions)?;
                        assert_eq!(device.items.iter().filter(|item| item.group == "execution owners")
                            .map(|item| item.bytes).collect::<Vec<_>>(),
                            [total.device_bytes - residual.min(prefill.scratch + prefill.io)]);
                        assert_eq!(device.items.iter().filter(|item| item.group == "execution staging")
                            .map(|item| item.bytes).collect::<Vec<_>>(), [total.pinned_bytes]);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn decode_owner_scratch_is_its_own_export_not_the_prefill_maximum() {
        let geometry = cuteafd_core::ExpertGeometry::GLM5;
        let decode = fp8_owner_bytes("glm5", geometry, &[(ExpertFormat::Nvfp4, 1024)], extent(64, 80)).unwrap();
        let prefill = fp8_owner_bytes("glm5", geometry, &[(ExpertFormat::Nvfp4, 1024)], extent(4096, 4096)).unwrap();
        assert_eq!((decode.scratch, prefill.scratch), (142_576_640, 667_162_624));
        // Live rows below the export never shrink compiled scratch, only I/O.
        let partial = fp8_owner_bytes("glm5", geometry, &[(ExpertFormat::Nvfp4, 1024)], extent(2752, 4096)).unwrap();
        assert_eq!(partial.scratch, prefill.scratch);
        assert_eq!(prefill.io - partial.io, (4096 - 2752) * (4 * 6144 + 8 * 8));
        assert!(group_scratch("glm5", ExpertFormat::Nvfp4, 64).is_err());
    }

    #[test]
    fn static_prefill_owner_charges_the_selected_export_and_live_io() -> anyhow::Result<()> {
        use crate::plan::spark_executions::{admit_family, SparkBackend};
        let geometry = cuteafd_core::ExpertGeometry::STEP5;
        let formats = [(ExpertFormat::Fp8Block128, 384), (ExpertFormat::Bf16, 384)];
        let maximum = fp8_owner_bytes("step5", geometry, &formats, extent(4096, 4096))?;
        let mut scratch = Vec::new();
        for (unit, waves, compiled) in [(4096, 1, 4096), (2752, 1, 4096), (256, 3, 256)] {
            let executions = admit_family("step5", &SparkBackend::Fp8.exports(), None, unit, waves)?;
            let prefill = executions.owners().last().unwrap();
            let actual = fp8_owner_bytes("step5", geometry, &formats, prefill)?;
            assert_eq!(actual.scratch, group_scratch("step5", ExpertFormat::Fp8Block128, compiled)?
                .max(group_scratch("step5", ExpertFormat::Bf16, compiled)?));
            assert_eq!(actual.io, maximum.io / 4096 * unit);
            scratch.push(actual.scratch);
        }
        assert_eq!(scratch[0], scratch[1], "default shapes still cover the 4096 export");
        assert!(scratch[2] < scratch[0], "narrow pin must not retain maximum-unit scratch");
        Ok(())
    }

    #[test]
    fn segmented_prefill_charges_admitted_rows_and_each_actual_rank_width() -> anyhow::Result<()> {
        let geometry = cuteafd_core::ExpertGeometry::STEP5;
        let singleton = extent(2752, 4096);
        let grouped = cuteafd_core::OwnerExtent { buffered_units: 3, ..singleton };
        for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4] {
            for width in [320, 384, 1024] {
                let family = if format == ExpertFormat::Mxfp4 { "mimo_v2" } else { "step5" };
                let formats = [(format, width)];
                let one = fp8_owner_bytes(family, geometry, &formats, singleton)?;
                let three = fp8_owner_bytes(family, geometry, &formats, grouped)?;
                let exact = cuteafd_core::spark_executions::segment_scratch_bytes(
                    geometry.hidden, geometry.experts, geometry.topk, width * 2, 8256, 16384).unwrap();
                assert_eq!(three.scratch, one.scratch.max(exact));
                let sibling = format == ExpertFormat::Fp8Block128;
                let old_input = u64::from(singleton.capacity) * if sibling { u64::from(geometry.hidden) * 2 }
                    else { u64::from(geometry.hidden) * 33 / 32 };
                let input_delta = cuteafd_core::spark_executions::segment_input_bytes(singleton.capacity,
                    8256, geometry.hidden, sibling).unwrap() - old_input;
                assert_eq!(three.io - one.io, 8256 * (16 + 8 * u64::from(geometry.topk)) + input_delta);
                assert_eq!(three.pinned_staging - one.pinned_staging, 2752 * 16);
                // Negative control: the compiled bound is not the allocation.
                let overcharged = cuteafd_core::spark_executions::segment_scratch_bytes(
                    geometry.hidden, geometry.experts, geometry.topk, width * 2, 16384, 16384).unwrap();
                assert!(exact < overcharged);
                let decode = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Decode(0), ..grouped };
                assert_eq!(fp8_owner_bytes(family, geometry, &formats, decode)?, one);
            }
        }
        for format in [ExpertFormat::Bf16, ExpertFormat::Nvfp4] {
            let formats = [(format, 384)];
            assert_eq!(fp8_owner_bytes("step5", geometry, &formats, singleton)?,
                fp8_owner_bytes("step5", geometry, &formats, grouped)?);
        }
        // Step's BF16 group retains its existing scratch, but no row tables.
        let fp8 = fp8_owner_bytes("step5", geometry, &[(ExpertFormat::Fp8Block128, 384)], grouped)?;
        let bf16 = fp8_owner_bytes("step5", geometry, &[(ExpertFormat::Bf16, 384)], grouped)?;
        let both = fp8_owner_bytes("step5", geometry,
            &[(ExpertFormat::Fp8Block128, 384), (ExpertFormat::Bf16, 384)], grouped)?;
        assert_eq!(both.scratch, fp8.scratch.max(bf16.scratch));
        assert_eq!(both.io, fp8.io + bf16.io);
        assert_eq!(both.pinned_staging, fp8.pinned_staging + bf16.pinned_staging);
        Ok(())
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
    fn native_v41_catalog_charges_one_prefill_owner_and_two_decode_owners() -> anyhow::Result<()> {
        for nvfp4 in [false, true] {
            let root = tempfile::tempdir()?;
            if nvfp4 { crate::plan::testing::write_v41_nvfp4_snapshot(root.path()); }
            else { crate::plan::testing::write_v41_snapshot(root.path()); }
            let input = crate::plan::PlanInput::open(root.path())?;
            if !nvfp4 {
                let catalog = input.expert_catalog()?;
                for world in 2..=8 {
                    for rank in 0..world {
                        let mut selected = SparkWorkingSet { spark_layers: vec![3], spark_ranks: world,
                            prefill_lanes: 2, ..Default::default() };
                        assert_eq!(mxfp4_rank_artifacts(catalog, &selected, rank)?.len(), 1);
                        selected.spark_layers = (3..3 + world).collect();
                        let retained = mxfp4_rank_artifacts(catalog, &selected, rank)?;
                        assert_eq!(retained.len(), if 72 % world == 0 { 1 } else { 2 });
                        selected.spark_layers.extend(3 + world..3 + 2 * world);
                        assert_eq!(mxfp4_rank_artifacts(catalog, &selected, rank)?, retained);
                    }
                }
            }
            let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
                prefill_lanes: 2, ..Default::default() };
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 * GIB,
                items: vec![], kv_tokens: 0 };
            let exports = crate::plan::spark_executions::SparkBackend::for_catalog(input.expert_catalog()?)?.exports();
            for (unit, waves) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (256, 3)] {
                let executions = crate::plan::spark_executions::admit_family("deepseek_v41", &exports, None, unit, waves)?;
                assert_eq!(executions.decode.len(), 2);
                let bytes = owner_bytes(&input, &working, "deepseek_v41", 0)?;
                let prefill = bytes(executions.owners().last().unwrap())?;
                let decode = bytes(executions.owners().next().unwrap())?;
                charge_executions(&mut device, &input, &working, "deepseek_v41", &executions)?;
                let find = |group| device.items.iter().find(|item| item.group == group).unwrap().bytes;
                assert_eq!(find("execution owners"), prefill.scratch + prefill.io + 2 * (decode.scratch + decode.io));
                assert_eq!(find("execution staging"), 2 * 64 * 6 * 8 + unit * waves * 6 * 8);
            }
        }
        Ok(())
    }

    #[test]
    fn exl3_pro_catalog_charges_one_prefill_owner_and_one_decode_owner() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        crate::plan::testing::write_v4_pro_exl3_snapshot(root.path());
        let input = crate::plan::PlanInput::open(root.path())?;
        let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            prefill_lanes: 2, ..Default::default() };
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 * GIB,
            items: vec![], kv_tokens: 0 };
        let exports = crate::plan::spark_executions::SparkBackend::for_catalog(input.expert_catalog()?)?.exports();
        let row = 7168 + 7168 / 32 + 6 * 8;
        let decode = 51 * (1 << 20) + 64 * row;
        for (unit, waves) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (256, 3)] {
            let executions = crate::plan::spark_executions::admit_family("deepseek_v4", &exports, None, unit, waves)?;
            assert_eq!(executions.decode.len(), 1);
            charge_executions(&mut device, &input, &working, "deepseek_v4", &executions)?;
            let find = |group| device.items.iter().find(|item| item.group == group).unwrap().bytes;
            assert_eq!(find("execution owners"), 810 * (1 << 20) - (4096 - unit) * row + decode);
            assert_eq!(find("execution staging"), 64 * 6 * 8 + unit * waves * 6 * 8);
        }
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
