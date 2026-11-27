//! Conservative installed-package bounds until manifest-backed planning (#776).
use super::*;
use anyhow::Context;
use crate::formats::fp8_experts::ExpertFormat;
use crate::placement::SparkWorkingSet;
use crate::serving_capacity::Exl3Epilogue;

/// Installed-package scratch census (#778, issuecomment-6401148082): every
/// shipped FP8-family package's scratch at each export capacity. A typed owner
/// is charged its compiled export's maximum over the family/format packages.
/// MiMo/Step rows include their BF16-input siblings.
const CENSUS_CAPACITIES: [u32; 8] = cuteafd_core::EXL3_CAPACITIES;

fn group_scratch(family: &str, format: ExpertFormat, width: u32, compiled: u32) -> anyhow::Result<u64> {
    use std::sync::OnceLock;
    static CENSUS: OnceLock<Vec<(String, String, Option<u32>, Vec<u64>)>> = OnceLock::new();
    let census = CENSUS.get_or_init(|| {
        let rows: Vec<serde_json::Value> = serde_json::from_str(include_str!("spark_slot_census.json"))
            .expect("bundled Spark slot census is valid JSON");
        rows.into_iter().map(|row| (row["family"].as_str().unwrap_or_default().to_owned(),
            row["format"].as_str().unwrap_or_default().to_owned(),
            row["width"].as_u64().and_then(|width| width.try_into().ok()),
            row["scratch"].as_array().map(|values| values.iter().filter_map(serde_json::Value::as_u64).collect())
                .unwrap_or_default())).collect()
    });
    let format_name = match format {
        ExpertFormat::Fp8Block128 => "fp8", ExpertFormat::Mxfp4 => "mxfp4",
        ExpertFormat::Nvfp4 => "nvfp4", ExpertFormat::Bf16 => "bf16", ExpertFormat::Mxfp8 => "mxfp8",
    };
    let index = CENSUS_CAPACITIES.iter().position(|&rows| rows == compiled)
        .ok_or_else(|| anyhow::anyhow!("no Spark scratch census at export m{compiled}"))?;
    let has_widths = census.iter().any(|(f, k, width, _)| f == family && k == format_name && width.is_some());
    census.iter().filter(|(f, k, measured_width, _)| f == family && k == format_name
            && (!has_widths || *measured_width == Some(width)))
        .filter_map(|(.., scratch)| scratch.get(index).copied()).max()
        .ok_or_else(|| anyhow::anyhow!("no calibrated Spark slot scratch for {family}/{format:?} width{width}; add the package census or manifest-backed sizing (#776)"))
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
    native_catalog_groups(input.expert_catalog()?, family, working, rank)
}

fn native_catalog_groups(catalog: &crate::OfficialV41Catalog, family: &str,
    working: &SparkWorkingSet, rank: usize)
    -> anyhow::Result<Option<(cuteafd_core::ExpertGeometry, Vec<(ExpertFormat, u32)>)>> {
    if !working.selected_ranks().contains(&rank) { return Ok(None); }
    let Some(tensors) = catalog.fp8() else { return Ok(None); };
    let shape = tensors.shape();
    let geometry = shape.geometry()?;
    let mut groups = Vec::new();
    let mut formats = Vec::new();
    for &layer in &working.spark_layers {
        if !tensors.has_layer(layer) { continue; }
        let view = tensors.for_layer_metadata(layer)?;
        let format = view.format();
        let package_family = match family {
            "glm5" => "glm", "glm5_flash" => "glmf", "minimax_m3" => "minimax",
            "mimo_v2" => if format == ExpertFormat::Mxfp4 {
                if shape.hidden == 6144 { "mimop" } else { "mimof" }
            } else { "mimo" },
            other => other,
        };
        let format_name = match format {
            ExpertFormat::Nvfp4 => "nvfp4", ExpertFormat::Bf16 => "bf16",
            ExpertFormat::Mxfp8 => "mxfp8", _ => "fp8",
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
/// A prefill owner whose packages have a row-addressed program also charges
/// its merged-wave tables and arena at the widest merged wave
/// (`segment_rows`) and its staged input slices for every buffered unit.
fn fp8_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry, formats: &[(ExpertFormat, u32)],
    extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    use cuteafd_core::spark_executions::{segment_input_bytes, segment_scratch_bytes, SegmentArithmetic, SEGMENT_MAX_ROWS};
    let mut scratch = 0;
    let mut io = 0u64;
    let mut address_staging = 0;
    let hidden = u64::from(geometry.hidden);
    for &(format, width) in formats {
        scratch = scratch.max(group_scratch(family, format, width, extent.compiled)?);
        io += group_io_bytes(family, format, geometry, extent.capacity);
        // Inkling's NVFP4 is dynamic-amax, not the static NVFP4 row-addressed
        // recipe. Its wave staging stays parked until a served card exists.
        if family == "inkling2" { continue; }
        // Installed row-addressed recipes and their smallest member units.
        let (minimum, arithmetic) = match format {
            // Every installed form is retained; W8A16 is the largest stream arena.
            ExpertFormat::Fp8Block128 => (769, SegmentArithmetic::Stream { act_row_bytes: width * 2 }),
            ExpertFormat::Mxfp4 => (641, SegmentArithmetic::Stream { act_row_bytes: width * 2 }),
            ExpertFormat::Nvfp4 => (1, SegmentArithmetic::Nvfp4A4 { slice: width }),
            _ => continue,
        };
        let (Some(rows), Some(staged)) = (extent.segment_rows().filter(|_| extent.staging_rows >= minimum),
            extent.staged_rows()) else { continue };
        let bytes = segment_scratch_bytes(geometry.hidden, geometry.experts, geometry.topk, arithmetic, rows,
            SEGMENT_MAX_ROWS).context("segment scratch overflow")?;
        scratch = scratch.max(bytes);
        io += u64::from(rows) * (16 + u64::from(geometry.topk) * 8);
        let row = if format == ExpertFormat::Nvfp4 { 2 * hidden } else { hidden + hidden / 32 };
        let sibling = (format == ExpertFormat::Fp8Block128 && matches!(family, "mimo_v2" | "step5")).then_some(2 * hidden);
        let old_input = u64::from(extent.capacity) * sibling.unwrap_or(row);
        io += segment_input_bytes(extent.capacity, staged, row.try_into()?, sibling.map(u32::try_from).transpose()?)
            .context("segmented input extent overflow")? - old_input;
        address_staging += u64::from(extent.staging_rows) * 16;
    }
    Ok(cuteafd_core::OwnerBytes { scratch, io,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 * formats.len() as u64 + address_staging })
}

/// Catalog adapter for callers that already resolved immutable checkpoint
/// selection. Uses exactly the shared group census and physical owner sum.
pub(crate) fn native_catalog_owner_bytes(catalog: &crate::OfficialV41Catalog, family: &str,
    working: &SparkWorkingSet, rank: usize, extent: cuteafd_core::OwnerExtent)
    -> anyhow::Result<cuteafd_core::OwnerBytes> {
    let Some((geometry, formats)) = native_catalog_groups(catalog, family, working, rank)?
        else { return Ok(Default::default()) };
    fp8_owner_bytes(family, geometry, &formats, extent)
}

/// Same catalog-owner dispatch for the EXL3 families. Missing recorded scratch
/// is an explicit shared census error, never a family-local extrapolation.
pub(crate) fn catalog_owner_bytes(catalog: &crate::OfficialV41Catalog, family: &str,
    working: &SparkWorkingSet, rank: usize, extent: cuteafd_core::OwnerExtent)
    -> anyhow::Result<cuteafd_core::OwnerBytes> {
    if catalog.fp8().is_some() { return native_catalog_owner_bytes(catalog, family, working, rank, extent); }
    let geometry = catalog.routed_experts().geometry()?;
    static RECORDED: std::sync::OnceLock<Vec<serde_json::Value>> = std::sync::OnceLock::new();
    let rows = RECORDED.get_or_init(|| serde_json::from_str(include_str!("spark_slot_census.json"))
        .expect("bundled Spark owner census is valid JSON"));
    if !rows.iter().any(|row| row["family"] == family && row["format"] == "exl3" && row["width"].is_number()) {
        let segment_width = working.selected_ranks().contains(&rank)
            .then(|| exl3_segment_width(family, geometry, working.spark_ranks))
            .flatten();
        return exl3_owner_bytes(family, geometry, extent, segment_width);
    }
    anyhow::ensure!(!working.spark_layers.is_empty(), "EXL3 worker has no retained layers");
    let package_family = match family {
        "glm5" => "glm", "glm5_flash" => "glmf", "kimi_k3" | "kimi_k31" => "kimik3",
        "muse_spark" => "muse", "mistral_large4" => "ml4", "deepseek_v4" => "dsv4p",
        "hy4_flash" => "hy4f",
        other => other,
    };
    let exl3 = catalog.exl3().context("EXL3 worker needs its catalog")?;
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let recipe = super::super::experts::spark_recipe(
        &format!("{package_family}:exl3-k{tiers}{}", exl3.codebook().package_suffix()),
        geometry.intermediate as usize, working.spark_ranks)?;
    let width = *recipe.stored_widths.get(rank).context("EXL3 worker rank outside recipe")?;
    let compact = recipe.rule == cuteafd_core::tp_units::UnitRule::Rotated;
    recorded_exl3_owner_bytes(rows, family, geometry, width.try_into()?, compact, extent)
}

fn recorded_exl3_owner_bytes(rows: &[serde_json::Value], family: &str,
    geometry: cuteafd_core::ExpertGeometry, width: u32, compact: bool, extent: cuteafd_core::OwnerExtent)
    -> anyhow::Result<cuteafd_core::OwnerBytes> {
    let row = rows.iter().find(|row| row["family"] == family && row["format"] == "exl3"
        && row["width"].as_u64() == Some(u64::from(width))
        && row["compact"].as_bool().unwrap_or(false) == compact)
        .with_context(|| format!("no EXL3 owner census for {family} width{width} compact={compact}"))?;
    let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(geometry).capacities(extent.capacity as usize)?;
    anyhow::ensure!(capacities.last() == Some(&extent.compiled), "EXL3 retained exports differ from owner extent");
    let recorded = row["manifests"].as_array().context("EXL3 census manifests absent")?;
    let manifests = capacities.iter().map(|&capacity| recorded.iter()
        .find(|manifest| manifest["capacity"].as_u64() == Some(u64::from(capacity)))
        .cloned().with_context(|| format!("EXL3 census lacks width{width}/m{capacity}")))
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(manifests.iter().all(|manifest| manifest["hidden"].as_u64() == Some(u64::from(geometry.hidden))
        && manifest["experts"].as_u64() == Some(u64::from(geometry.experts))
        && manifest["top_k"].as_u64() == Some(u64::from(geometry.topk))), "EXL3 census/checkpoint geometry mismatch");
    let scratch = crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::Sum)?;
    let io = u64::from(extent.capacity).checked_mul(u64::from(geometry.hidden) * 33 / 32
        + u64::from(geometry.topk) * 8).context("EXL3 owner I/O overflow")?;
    let mut bytes = cuteafd_core::OwnerBytes { scratch, io,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 };
    if !compact && family != "muse_spark" && geometry.hidden % 512 == 0 {
        let waves = exl3_wave_bytes(geometry, width, extent).context("EXL3 merged-wave extent overflow")?;
        bytes.io += waves.0;
        bytes.pinned_staging += waves.1;
    }
    Ok(bytes)
}

/// One native NVFP4 owner on the shared FP8-family worker: the installed
/// package's measured scratch at the compiled export (census), BF16 ingress
/// I/O at the owner's capacity, and packed route staging per queued unit.
pub(crate) fn nvfp4_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry, width: u32,
    extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    fp8_owner_bytes(family, geometry, &[(ExpertFormat::Nvfp4, width)], extent)
}

/// [`nvfp4_owner_bytes`] with the exact NVFP4 wire ingress: the same package
/// scratch (the NVFP4-wire sibling's regions equal the BF16-input ones), and
/// the device input plane, including every staged unit's slice of a merged
/// wave, holds wire rows instead of BF16 rows (the worker sizes both from the
/// package's `input_row_bytes`).
pub(crate) fn nvfp4_owner_bytes_for(family: &str, geometry: cuteafd_core::ExpertGeometry, width: u32,
    extent: cuteafd_core::OwnerExtent, nvfp4_wire: bool) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    let mut bytes = nvfp4_owner_bytes(family, geometry, width, extent)?;
    if nvfp4_wire {
        let hidden = geometry.hidden as usize;
        // Input rows the BF16 charge holds: the ordinary plane, or the staged
        // slices when the owner merges waves (they cover the plane).
        let rows = match (extent.segment_rows(), extent.staged_rows()) {
            (Some(_), Some(staged)) => staged.max(extent.capacity),
            _ => extent.capacity,
        };
        let saved = (2 * hidden - cuteafd_core::nvfp4_wire_row_bytes(hidden)) as u64 * u64::from(rows);
        bytes.io = bytes.io.checked_sub(saved).context("NVFP4 wire owner input below BF16")?;
    }
    Ok(bytes)
}

/// EXL3 owner bytes. Prefill-scale exports charge the calibrated retained
/// arena maxima plus private capacity state (#762 census totals, which include
/// 4096 rows of K32 wire/routes; those rows are re-charged as the owner's I/O
/// at its own capacity). Decode exports (m80 and below) charge the m1/m16/m80
/// retained set (installed SM121 exports, 2026-10-24: largest GLM tp1
/// 50,445,548 B). Packed routes stage per queued unit.
pub(crate) fn exl3_owner_bytes(family: &str, geometry: cuteafd_core::ExpertGeometry, extent: cuteafd_core::OwnerExtent,
    segment_width: Option<u32>) -> anyhow::Result<cuteafd_core::OwnerBytes> {
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
    let mut bytes = cuteafd_core::OwnerBytes { scratch, io: u64::from(extent.capacity) * row,
        pinned_staging: u64::from(extent.staging_rows) * topk * 8 };
    if let Some(width) = segment_width {
        let waves = exl3_wave_bytes(geometry, width, extent).context("EXL3 merged-wave extent overflow")?;
        bytes.io += waves.0;
        bytes.pinned_staging += waves.1;
    }
    Ok(bytes)
}

/// Stored intermediate width of an EXL3 Spark layout that ships a merged-wave
/// segments program (`package_exl3_aot.segment_export_options`): noncompact
/// worlds (whole 128-row blocks split evenly over `world`) with `hidden % 512 == 0`
/// and the weighted-sum epilogue (Muse replies per-expert partials and never merges).
pub(crate) fn exl3_segment_width(family: &str, geometry: cuteafd_core::ExpertGeometry, world: usize) -> Option<u32> {
    if family == "muse_spark" || world == 0 || geometry.hidden % 512 != 0 || geometry.intermediate % 128 != 0 {
        return None;
    }
    ((geometry.intermediate / 128) % world as u32 == 0).then(|| geometry.intermediate / world as u32)
}

/// A prefill owner's merged EXL3 waves on a layout that ships a segments
/// program: (device bytes, pinned descriptor bytes per queued unit). The
/// worker's `Exl3Worker::wave_bytes` charges the same terms from the same
/// core layout: the arena at the widest merged wave, every staged unit's wire
/// slice, the address and route tables.
pub(crate) fn exl3_wave_bytes(geometry: cuteafd_core::ExpertGeometry, width: u32,
    extent: cuteafd_core::OwnerExtent) -> Option<(u64, u64)> {
    let (Some(rows), Some(staged)) = (extent.segment_rows(), extent.staged_rows()) else { return Some((0, 0)) };
    let segment = cuteafd_core::spark_executions::Exl3SegmentGeometry::for_spark(geometry.hidden, width,
        geometry.topk, geometry.experts);
    let arena = segment.layout(rows)?.1;
    let hidden = u64::from(geometry.hidden);
    let tables = u64::from(rows) * (16 + 8 * u64::from(geometry.topk));
    Some((arena + u64::from(staged) * (hidden + hidden / 32) + tables, u64::from(extent.staging_rows) * 16))
}

/// Muse retains the widest installed TP slice of this world, never a refused
/// world's wider package. Keep each capacity's allocation owners/aliases so
/// admission uses the same arena census as the EXL3 worker.
pub(crate) fn muse_owner_bytes(world: usize, extent: cuteafd_core::OwnerExtent)
    -> anyhow::Result<cuteafd_core::OwnerBytes> {
    use std::sync::OnceLock;
    static CENSUS: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    let census = CENSUS.get_or_init(|| serde_json::from_str(include_str!("spark_slot_census.json"))
        .expect("bundled Spark slot census is valid JSON"));
    let row = census.iter().find(|row| row["family"] == "muse_spark" && row["format"] == "exl3"
        && row["world"].as_u64() == Some(world as u64))
        .with_context(|| format!("no Muse Spark TP{world} EXL3 allocation census"))?;
    let capacities = cuteafd_core::Exl3RowPolicy::Nearest.capacities(extent.capacity as usize)?;
    anyhow::ensure!(capacities.last() == Some(&extent.compiled), "Muse owner export and retained capacities disagree");
    let installed = row["manifests"].as_array().context("Muse census has no capacity witnesses")?;
    let manifests = capacities.into_iter().map(|capacity| installed.iter()
        .find(|manifest| manifest["capacity"].as_u64() == Some(u64::from(capacity)))
        .cloned().with_context(|| format!("Muse Spark TP{world} census has no m{capacity}")))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let retained = crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::PartialReplies)?;
    let rows = u64::from(extent.capacity);
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    let wire_routes = u64::from(geometry.hidden) * 33 / 32 + u64::from(geometry.topk) * 8;
    // One device and one pinned set of three u32 maps per owner. The pinned maps
    // are singleton storage, not per-unit staging: do not multiply by waves.
    Ok(cuteafd_core::OwnerBytes { scratch: retained + rows * 24, io: rows * wire_routes,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 })
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
    let retained = crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::Sum)?;
    let wire_routes = u64::from(geometry.hidden) * 33 / 32 + u64::from(geometry.topk) * 8;
    let mut bytes = cuteafd_core::OwnerBytes { scratch: retained, io: u64::from(extent.capacity) * wire_routes,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 };
    if let Some(width) = exl3_segment_width("mistral_large4", geometry, world) {
        let waves = exl3_wave_bytes(geometry, width, extent).context("EXL3 merged-wave extent overflow")?;
        bytes.io += waves.0;
        bytes.pinned_staging += waves.1;
    }
    Ok(bytes)
}

/// Native width states coexist, while their stream and live I/O are shared.
fn mxfp4_owner_bytes(geometry: cuteafd_core::ExpertGeometry,
    artifacts: &[cuteafd_core::tp_units::NativeArtifact], coded: bool,
    extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    anyhow::ensure!(!artifacts.is_empty() && artifacts.len() <= 2,
        "native owner needs one or two retained width states");
    let mut scratch = 0u64;
    for artifact in artifacts {
        let exports = cuteafd_core::mxfp4_layouts::native_export_capacities(
            geometry.family().context("unknown native expert family")?, &artifact.name);
        let grouped = exports.grouped_decode(extent.compiled).map_err(anyhow::Error::msg)?;
        let mut capacities = vec![extent.compiled];
        if extent.compiled > 1 { capacities.push(1); }
        if extent.compiled > grouped { capacities.push(grouped); }
        for &capacity in &capacities {
            let bytes = cuteafd_core::mxfp4_layouts::native_scratch_bytes(geometry.hidden as usize,
                geometry.experts as usize, geometry.topk as usize, artifact.logical,
                artifact.role, capacity, coded).context("invalid native owner scratch geometry")?;
            scratch = scratch.checked_add(bytes).context("native width scratch sum overflow")?;
        }
    }
    let waves = crate::serving_capacity::spark_waves::staging(extent, geometry)?;
    Ok(cuteafd_core::OwnerBytes { scratch,
        io: u64::from(extent.capacity) * (u64::from(geometry.hidden) * 97 / 32
            + u64::from(geometry.topk) * 8) + waves.plane_bytes + waves.table_bytes,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 + waves.pinned_per_unit })
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
    let artifacts = if backend.is_native_census() && catalog.nvfp4().is_none() {
        Some(mxfp4_rank_artifacts(catalog, working, rank)?)
    } else { None };
    let coded = catalog.mxfp4_scale_codes();
    // Every sum-epilogue EXL3 Spark family ships a merged-wave segments program
    // on every noncompact layout (`package_exl3_aot.segment_export_options`).
    let segment_width = match backend {
        SparkBackend::Exl3(_) if working.selected_ranks().contains(&rank) =>
            exl3_segment_width(family, geometry, world),
        _ => None,
    };
    let family = family.to_owned();
    Ok(move |extent| {
        if let Some(artifacts) = &artifacts {
            return mxfp4_owner_bytes(geometry, artifacts, coded, extent);
        }
        match (&groups, backend) {
            (Some((geometry, formats)), SparkBackend::Fp8) => fp8_owner_bytes(&family, *geometry, formats, extent),
            (_, SparkBackend::Exl3(_)) if family == "muse_spark" => muse_owner_bytes(world, extent),
            (_, SparkBackend::Exl3(_)) if family == "mistral_large4" => ml4_owner_bytes(world, extent),
            (_, SparkBackend::Exl3(_)) => exl3_owner_bytes(&family, geometry, extent, segment_width),
            (_, SparkBackend::Exlr8) => Ok(legacy_owner_bytes(&family, geometry, extent)),
            (_, SparkBackend::Native | SparkBackend::V41 | SparkBackend::Fp8) =>
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
    if family == "kimi_k3" {
        return crate::families::kimi_k3::spark::charge(device, working.spark_ranks, executions);
    }
    charge_owners(device, executions, owner_bytes(input, working, family, device.index as usize)?)
}

/// Installed wave programs are part of the same owner, not an extra arena charge.
pub(crate) fn charge_installed_executions(device: &mut DeviceLayout, input: &super::super::PlanInput,
    working: &SparkWorkingSet, family: &str, executions: &cuteafd_core::SparkExecutions,
    options: &LayoutOptions) -> anyhow::Result<()> {
    if family == "kimi_k3" { return charge_executions(device, input, working, family, executions); }
    let catalog = input.expert_catalog()?;
    let backend = crate::plan::spark_executions::SparkBackend::for_catalog(catalog)?;
    let geometry = catalog.routed_experts().geometry()?;
    let native = backend.is_native_census() && catalog.nvfp4().is_none();
    if !native { return charge_executions(device, input, working, family, executions); }
    let path = options.workspace_manifest.as_deref().unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let programs = input.manifest(path)?;
    let rank = device.index as usize;
    let widths = if native { mxfp4_rank_artifacts(catalog, working, rank)?.into_iter().map(|a| a.logical).collect::<Vec<_>>() }
        else { Vec::new() };
    let bytes = owner_bytes(input, working, family, rank)?;
    charge_owners(device, executions, |extent| {
        let mut owner = bytes(extent)?;
        let scratch = crate::serving_capacity::spark_waves::native_scratch(&programs, extent, geometry, widths.iter().copied())?;
        owner.scratch = owner.scratch.checked_add(scratch).context("installed wave scratch overflow")?;
        Ok(owner)
    })
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
    fn inkling_native_census_uses_exporter_bytes_and_shared_mixed_owner_sum() -> anyhow::Result<()> {
        let geometry = cuteafd_core::ExpertGeometry::INKLING2;
        for (format, scratch) in [(ExpertFormat::Mxfp8, 400_695_296), (ExpertFormat::Nvfp4, 780_804_096)] {
            assert_eq!(group_scratch("inkling2", format, 1152, 4096)?, scratch);
            let one = fp8_owner_bytes("inkling2", geometry, &[(format, 1152)], extent(2752, 4096))?;
            assert_eq!(one.scratch, scratch);
            let mut queued = extent(2752, 4096);
            queued.buffered_units = 3;
            assert_eq!(fp8_owner_bytes("inkling2", geometry, &[(format, 1152)], queued)?, one);
        }
        let both = fp8_owner_bytes("inkling2", geometry,
            &[(ExpertFormat::Mxfp8, 1152), (ExpertFormat::Nvfp4, 1152)], extent(2752, 4096))?;
        assert_eq!(both.scratch, 780_804_096);
        assert_eq!(both.io, group_io_bytes("inkling2", ExpertFormat::Mxfp8, geometry, 2752)
            + group_io_bytes("inkling2", ExpertFormat::Nvfp4, geometry, 2752));
        assert_eq!(both.pinned_staging, 2752 * 8 * 8 * 2);
        for format in [ExpertFormat::Mxfp8, ExpertFormat::Nvfp4] {
            let narrow = group_scratch("inkling2", format, 256, 4096)?;
            let wide = group_scratch("inkling2", format, 1152, 4096)?;
            assert!(narrow < wide, "narrow owner must not borrow widest scratch");
            let exact = group_scratch("inkling2", format, 576, 4096)?;
            assert_eq!(exact, if format == ExpertFormat::Mxfp8 { 352_198_656 } else { 648_093_696 });
            assert!(narrow < exact && exact < wide);
            assert!(group_scratch("inkling2", format, 320, 4096).is_err(),
                "an unrecorded width cannot borrow a neighbour or a margin");
        }
        Ok(())
    }

    #[test]
    fn recorded_exl3_widths_keep_aliases_and_private_capacity_owners_exact() -> anyhow::Result<()> {
        let rows: Vec<serde_json::Value> = serde_json::from_str(include_str!("spark_slot_census.json"))?;
        let geometry = cuteafd_core::ExpertGeometry::INKLING2;
        for (width, expected) in [(256, 516_083_772), (384, 541_249_596), (512, 566_415_420),
            (640, 591_581_244), (768, 616_747_068), (1152, 692_244_540), (2304, 918_736_956)] {
            let owner = recorded_exl3_owner_bytes(&rows, "inkling2", geometry, width, false, extent(4096,4096))?;
            assert_eq!(owner.scratch, expected);
            assert_eq!(owner.io, 4096*(4608+144+64));
        }
        for (width, expected) in [(384, 541_249_596), (512, 566_415_420), (640, 591_581_244)] {
            assert_eq!(recorded_exl3_owner_bytes(&rows, "inkling2", geometry, width, true,
                extent(4096,4096))?.scratch, expected);
        }
        assert!(recorded_exl3_owner_bytes(&rows, "inkling2", geometry, 256, true, extent(4096,4096)).is_err(),
            "same width's noncompact export is not compact ABI evidence");
        assert!(recorded_exl3_owner_bytes(&rows, "inkling2", geometry, 320, false, extent(4096,4096)).is_err());
        Ok(())
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
                let exports = cuteafd_core::mxfp4_layouts::native_export_capacities(geometry.family().unwrap(), "spark");
                let decode = *exports.decode.last().unwrap();
                for compiled in exports.capacities {
                    let live = compiled.min(decode);
                    let shape = extent(live, compiled);
                    let total = mxfp4_owner_bytes(geometry, &artifacts, coded, shape)?;
                    let singles = artifacts.iter().map(|artifact|
                        mxfp4_owner_bytes(geometry, std::slice::from_ref(artifact), coded, shape))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    assert_eq!(total.scratch, singles.iter().map(|owner| owner.scratch).sum::<u64>());
                    assert_eq!(total.io, singles[0].io);
                    assert_eq!(total.pinned_staging, singles[0].pinned_staging);
                    let capacities = [Some(compiled), (compiled > 1).then_some(1), (compiled > decode).then_some(decode)];
                    let expected = artifacts.iter().map(|a| capacities.into_iter().flatten().map(|rows|
                        cuteafd_core::mxfp4_layouts::native_scratch_bytes(geometry.hidden as usize,
                            geometry.experts as usize, geometry.topk as usize, a.logical, a.role, rows, coded).unwrap())
                        .sum::<u64>()).sum::<u64>();
                    assert_eq!(total.scratch, expected);
                    let waves = crate::serving_capacity::spark_waves::staging(shape, geometry).unwrap();
                    assert_eq!(total.io, u64::from(live) * (u64::from(geometry.hidden) * 97 / 32 + u64::from(geometry.topk) * 8)
                        + waves.plane_bytes + waves.table_bytes);
                    let buffered = mxfp4_owner_bytes(geometry, &artifacts, coded,
                        cuteafd_core::OwnerExtent { buffered_units: 3, ..shape })?;
                    assert_eq!(buffered.scratch, total.scratch);
                    let extra = crate::serving_capacity::spark_waves::staging(
                        cuteafd_core::OwnerExtent { buffered_units: 3, ..shape }, geometry).unwrap();
                    assert_eq!(buffered.io, total.io + extra.plane_bytes + extra.table_bytes - waves.plane_bytes - waves.table_bytes);
                    assert_eq!(buffered.pinned_staging, total.pinned_staging + extra.pinned_per_unit - waves.pinned_per_unit);
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
                "nvfp4" => ExpertFormat::Nvfp4, "bf16" => ExpertFormat::Bf16,
                "mxfp8" => ExpertFormat::Mxfp8, _ => unreachable!(),
            };
            let hidden = row["hidden"].as_u64().unwrap();
            let topk = row["topk"].as_u64().unwrap();
            let input = if row["bf16"].as_bool().unwrap() { 2 * hidden } else { hidden + hidden / 32 };
            let geometry = cuteafd_core::ExpertGeometry::from_wire_dims(hidden as u32, 1, topk as u32, 1, 1);
            for (index, compiled) in CENSUS_CAPACITIES.into_iter().enumerate() {
                let rows = u64::from(compiled);
                let exact = row["scratch"][index].as_u64().unwrap() + rows * (input + 8 * topk + 2 * hidden);
                let width = row["width"].as_u64().unwrap_or(128) as u32;
                let owner = fp8_owner_bytes(family, geometry, &[(format, width)], extent(compiled, compiled)).unwrap();
                if row["width"].is_number() {
                    assert_eq!(owner.scratch + owner.io, exact, "{package} m{compiled}: width-qualified owner differs");
                } else {
                    assert!(owner.scratch + owner.io >= exact, "{package} m{compiled}: planner < census {exact}");
                }
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
                let retained = crate::serving_capacity::exl3_workspace_bytes(&selected, true, Exl3Epilogue::Sum)?;
                let without_wire = crate::serving_capacity::exl3_workspace_bytes(&selected, false, Exl3Epilogue::Sum)?;
                assert_eq!(retained - without_wire,
                    retained_set.iter().map(|&rows| u64::from(rows)).sum::<u64>() * 6144 * 2);
                let owner = ml4_owner_bytes(world, extent(capacity, compiled))?;
                assert_eq!(owner.scratch, retained, "TP{world} m{compiled}");
                assert_eq!(owner.io, u64::from(capacity) * (6144 + 192 + 32));
                assert_eq!(owner.pinned_staging, u64::from(capacity) * 32);
                let buffered_extent = cuteafd_core::OwnerExtent { buffered_units: 3, ..extent(capacity, compiled) };
                let buffered = ml4_owner_bytes(world, buffered_extent)?;
                assert_eq!(buffered.scratch, owner.scratch, "retained arenas are not per wave");
                let geometry = cuteafd_core::ExpertGeometry::MISTRAL_LARGE4;
                let (wave_dev, wave_pin) = exl3_segment_width("mistral_large4", geometry, world)
                    .map_or((0, 0), |w| exl3_wave_bytes(geometry, w, buffered_extent).unwrap());
                assert_eq!(buffered.io, owner.io + wave_dev);
                assert_eq!(buffered.pinned_staging, owner.pinned_staging + wave_pin);
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
                        capacity_bytes: 121 * GIB, worker_budget_bytes: 108 * GIB, kv_tokens: 0,
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
    fn nvfp4_wire_owner_charges_wire_rows_for_the_plane_and_staged_slices() -> anyhow::Result<()> {
        let geometry = cuteafd_core::ExpertGeometry::MINIMAX_M3;
        let hidden = u64::from(geometry.hidden);
        let delta = 2 * hidden - cuteafd_core::nvfp4_wire_row_bytes(geometry.hidden as usize) as u64;
        let unit = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill, capacity: 2752,
            compiled: 4096, staging_rows: 2752, buffered_units: 3 };
        let decode = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Decode(0), capacity: 80,
            compiled: 80, staging_rows: 80, buffered_units: 1 };
        for (extent, rows) in [(unit, u64::from(unit.staged_rows().unwrap())), (decode, 80)] {
            let bf16 = nvfp4_owner_bytes_for("minimax_m3", geometry, 768, extent, false)?;
            let wire = nvfp4_owner_bytes_for("minimax_m3", geometry, 768, extent, true)?;
            // The worker sizes the ordinary plane and every staged unit's slice
            // from the package's input row (fp8 worker `input_row_bytes`).
            assert_eq!(bf16.io - wire.io, rows * delta, "{extent:?}");
            assert_eq!((bf16.scratch, bf16.pinned_staging), (wire.scratch, wire.pinned_staging));
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
        assert!(group_scratch("glm5", ExpertFormat::Nvfp4, 1024, 64).is_err());
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
            assert_eq!(actual.scratch, group_scratch("step5", ExpertFormat::Fp8Block128, 384, compiled)?
                .max(group_scratch("step5", ExpertFormat::Bf16, 384, compiled)?));
            assert_eq!(actual.io, maximum.io / 4096 * unit);
            scratch.push(actual.scratch);
        }
        assert_eq!(scratch[0], scratch[1], "default shapes still cover the 4096 export");
        assert!(scratch[2] < scratch[0], "narrow pin must not retain maximum-unit scratch");
        Ok(())
    }

    #[test]
    fn segmented_prefill_charges_merged_waves_staged_units_and_each_actual_rank_width() -> anyhow::Result<()> {
        use cuteafd_core::spark_executions::{segment_input_bytes, segment_scratch_bytes, SegmentArithmetic};
        let geometry = cuteafd_core::ExpertGeometry::STEP5;
        let singleton = extent(2752, 4096);
        let grouped = cuteafd_core::OwnerExtent { buffered_units: 3, ..singleton };
        let hidden = u64::from(geometry.hidden);
        for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4, ExpertFormat::Nvfp4] {
            for width in [320, 384, 1024] {
                let family = match format { ExpertFormat::Mxfp4 => "mimo_v2", _ => "step5" };
                let formats = [(format, width)];
                let one = fp8_owner_bytes(family, geometry, &formats, singleton)?;
                let three = fp8_owner_bytes(family, geometry, &formats, grouped)?;
                let arithmetic = if format == ExpertFormat::Nvfp4 { SegmentArithmetic::Nvfp4A4 { slice: width } }
                    else { SegmentArithmetic::Stream { act_row_bytes: width * 2 } };
                // One merged launch holds two units (5504); all three stage (8256).
                let exact = segment_scratch_bytes(geometry.hidden, geometry.experts, geometry.topk, arithmetic,
                    5504, 8192).unwrap();
                assert_eq!(three.scratch, one.scratch.max(exact));
                let row = if format == ExpertFormat::Nvfp4 { 2 * hidden } else { hidden * 33 / 32 };
                let sibling = (format == ExpertFormat::Fp8Block128).then_some(2 * hidden);
                let old_input = u64::from(singleton.capacity) * sibling.unwrap_or(row);
                let input_delta = segment_input_bytes(singleton.capacity, 8256, row as u32,
                    sibling.map(|b| b as u32)).unwrap() - old_input;
                assert_eq!(three.io - one.io, 5504 * (16 + 8 * u64::from(geometry.topk)) + input_delta);
                assert_eq!(three.pinned_staging - one.pinned_staging, 2752 * 16);
                // Negative controls: neither every buffered unit nor the code
                // bound is one launch's allocation.
                for rows in [8192] {
                    assert!(exact < segment_scratch_bytes(geometry.hidden, geometry.experts, geometry.topk,
                        arithmetic, rows, 8192).unwrap());
                }
                let decode = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Decode(0), ..grouped };
                assert_eq!(fp8_owner_bytes(family, geometry, &formats, decode)?, one);
            }
        }
        {
            let format = ExpertFormat::Bf16;
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
            let retained = crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::Sum).unwrap();
            let wire_routes = 4096 * (hidden + hidden / 32 + 8 * topk);
            assert_eq!(retained + wire_routes, expected, "{family}");
            let geometry = cuteafd_core::ExpertGeometry::from_wire_dims(hidden as u32, 1, topk as u32, 1, 1);
            let owner = exl3_owner_bytes(family, geometry, extent(4096, 4096), None).unwrap();
            assert!(owner.scratch + owner.io >= expected, "{family}");
        }
    }

    #[test]
    fn exl3_decode_owner_covers_every_installed_narrow_export_set() {
        // Largest retained m1/m16/m80 set among installed SM121 exports.
        let geometry = cuteafd_core::ExpertGeometry::GLM5;
        let decode = exl3_owner_bytes("glm5_flash", geometry, extent(64, 80), Some(1024)).unwrap();
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
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 * GIB, worker_budget_bytes: 108 * GIB,
                items: vec![], kv_tokens: 0 };
            let exports = crate::plan::spark_executions::SparkBackend::for_catalog(input.expert_catalog()?)?.exports();
            for (unit, waves) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (256, 3)] {
                let executions = crate::plan::spark_executions::admit_family_lanes("deepseek_v41", &exports, None, unit, waves, 2)?;
                assert_eq!(executions.decode.len(), 2);
                let bytes = owner_bytes(&input, &working, "deepseek_v41", 0)?;
                let prefill = bytes(executions.owners().last().unwrap())?;
                let decode = bytes(executions.owners().next().unwrap())?;
                charge_executions(&mut device, &input, &working, "deepseek_v41", &executions)?;
                let find = |group| device.items.iter().find(|item| item.group == group).unwrap().bytes;
                assert_eq!(find("execution owners"), prefill.scratch + prefill.io + 2 * (decode.scratch + decode.io));
                assert_eq!(find("execution staging"), 2 * decode.pinned_staging + waves * prefill.pinned_staging);
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
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 * GIB, worker_budget_bytes: 108 * GIB,
            items: vec![], kv_tokens: 0 };
        let exports = crate::plan::spark_executions::SparkBackend::for_catalog(input.expert_catalog()?)?.exports();
        let row = 7168 + 7168 / 32 + 6 * 8;
        let decode = 51 * (1 << 20) + 64 * row;
        let geometry = cuteafd_core::ExpertGeometry::DEEPSEEK_V4_PRO;
        for (unit, waves) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (256, 3)] {
            let executions = crate::plan::spark_executions::admit_family("deepseek_v4", &exports, None, unit, waves)?;
            assert_eq!(executions.decode.len(), 1);
            charge_executions(&mut device, &input, &working, "deepseek_v4", &executions)?;
            let prefill = executions.owners().last().unwrap();
            let (wave_dev, wave_pin) = exl3_wave_bytes(geometry, 768, prefill).unwrap();
            let find = |group| device.items.iter().find(|item| item.group == group).unwrap().bytes;
            assert_eq!(find("execution owners"), 810 * (1 << 20) - (4096 - unit) * row + decode + wave_dev);
            assert_eq!(find("execution staging"), 64 * 6 * 8 + unit * waves * 6 * 8 + waves * wave_pin);
        }
        Ok(())
    }

    #[test]
    fn owners_credit_only_the_existing_prefill_residual() {
        let executions = cuteafd_core::SparkExecutions::admit(&[64], 4096, 2048, 4, &cuteafd_core::EXL3_CAPACITIES).unwrap();
        let bytes = |extent: cuteafd_core::OwnerExtent| Ok(cuteafd_core::OwnerBytes {
            scratch: if extent.compiled == 4096 { 200 } else { 50 }, io: 0, pinned_staging: 1 });
        for residual in [0, 100, 1000] {
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: GIB, worker_budget_bytes: GIB,
                kv_tokens: 0, items: vec![Item::new(Category::Workspace, "expert waves", "", residual, Basis::Calibrated)] };
            charge_owners(&mut device, &executions, bytes).unwrap();
            let charged = device.items.iter().find(|item| item.group == "execution owners").unwrap().bytes;
            assert_eq!(charged, 250 - residual.min(200));
            assert_eq!(device.items.iter().find(|item| item.group == "execution staging").unwrap().bytes, 1 + 4);
        }
    }
    #[test]
    fn muse_census_keeps_each_worlds_own_retained_capacity_set() -> anyhow::Result<()> {
        // SM121 allocation witnesses, K4/K5 with the empty K5 tier. TP2/TP3
        // remain honest while refused; admitted worlds never pay their widths.
        // Muse replies per-expert partials, so these are the witnesses without
        // the weighted sum's output owner (m80 -655,360 B, m2752 -22,544,384 B).
        for (world, decode, prefill) in [(2, 25_465_516, 532_133_180),
            (3, 23_007_916, 447_591_740), (4, 21_533_356, 396_866_876),
            (5, 21_041_836, 379_958_588), (6, 20_550_316, 363_050_300),
            (7, 20_058_796, 346_142_012), (8, 19_567_276, 329_233_724)] {
            for (capacity, compiled, retained) in [(64, 80, decode), (2752, 2752, prefill)] {
                let owner = muse_owner_bytes(world, extent(capacity, compiled))?;
                assert_eq!(owner.scratch, retained + u64::from(capacity) * 24, "TP{world} m{compiled}");
                assert_eq!(owner.io, u64::from(capacity) * (4096 + 128 + 64));
                assert_eq!(owner.pinned_staging, u64::from(capacity) * 64);
            }
            let one = muse_owner_bytes(world, extent(2752, 2752))?;
            let three = muse_owner_bytes(world, cuteafd_core::OwnerExtent {
                buffered_units: 3, ..extent(2752, 2752) })?;
            assert_eq!(one, three, "buffering never multiplies retained arenas or row maps");
            for capacity in CENSUS_CAPACITIES {
                let compiled = *cuteafd_core::Exl3RowPolicy::Nearest.capacities(capacity as usize)?.last().unwrap();
                let owner = muse_owner_bytes(world, extent(capacity, compiled))?;
                assert!(owner.scratch > u64::from(capacity) * 16);
            }
        }
        assert!(muse_owner_bytes(1, extent(64, 80)).is_err());
        assert!(muse_owner_bytes(9, extent(64, 80)).is_err());
        assert!(muse_owner_bytes(4, extent(64, 256)).is_err());
        Ok(())
    }

    #[test]
    fn muse_census_preserves_private_state_aliases_and_wire_decode() -> anyhow::Result<()> {
        let census: Vec<serde_json::Value> = serde_json::from_str(include_str!("spark_slot_census.json"))?;
        for row in census.iter().filter(|row| row["family"] == "muse_spark" && row["format"] == "exl3") {
            let world = row["world"].as_u64().unwrap() as usize;
            let manifests = row["manifests"].as_array().unwrap();
            assert_eq!(manifests.len(), CENSUS_CAPACITIES.len());
            for (index, &capacity) in CENSUS_CAPACITIES.iter().enumerate() {
                assert_eq!(manifests[index]["capacity"], capacity);
                let capacities = cuteafd_core::Exl3RowPolicy::Nearest.capacities(capacity as usize)?;
                let compiled = *capacities.last().unwrap();
                let retained_index = CENSUS_CAPACITIES.iter().position(|&rows| rows == compiled).unwrap();
                let retained = crate::serving_capacity::exl3_workspace_bytes(&manifests[..=retained_index], true,
                    Exl3Epilogue::PartialReplies)?;
                assert_eq!(muse_owner_bytes(world, extent(capacity, compiled))?.scratch,
                    retained + u64::from(capacity) * 24);
                let without_wire = crate::serving_capacity::exl3_workspace_bytes(&manifests[..=retained_index], false,
                    Exl3Epilogue::PartialReplies)?;
                let rows: u64 = capacities.iter().map(|&rows| u64::from(rows)).sum();
                assert_eq!(retained - without_wire, rows * 4096 * 2);
            }
        }
        Ok(())
    }

    #[test]
    fn every_sum_epilogue_exl3_family_charges_merged_waves_on_noncompact_worlds_only() -> anyhow::Result<()> {
        use cuteafd_core::{ExecutionOwner, ExpertGeometry, OwnerExtent};
        let prefill = OwnerExtent { owner: ExecutionOwner::Prefill, capacity: 2752, compiled: 2752,
            staging_rows: 2752, buffered_units: 3 };
        let singleton = OwnerExtent { buffered_units: 1, ..prefill };
        let decode = OwnerExtent { owner: ExecutionOwner::Decode(0), capacity: 64, compiled: 80,
            staging_rows: 64, buffered_units: 1 };
        let rows: Vec<serde_json::Value> = serde_json::from_str(include_str!("spark_slot_census.json"))?;

        for (family, geometry, noncompact_world, compact_world) in [
            ("glm5", ExpertGeometry::GLM5, 4usize, 6usize),
            ("glm5_flash", ExpertGeometry::GLM5_FLASH, 4, 3),
            ("deepseek_v4", ExpertGeometry::DEEPSEEK_V4_PRO, 4, 5),
            ("qwen4", ExpertGeometry::QWEN4, 1, 2),
            ("kimi_k3", ExpertGeometry::KIMI_K3, 6, 7),
            ("mistral_large4", ExpertGeometry::MISTRAL_LARGE4, 6, 5),
            ("step5", ExpertGeometry::STEP5, 4, 5),
            ("inkling2", ExpertGeometry::INKLING2, 6, 4),
            ("hy4_flash", ExpertGeometry::HY4_FLASH, 4, 5),
            ("hy4", ExpertGeometry::HY4, 4, 6),
        ] {
            let width = exl3_segment_width(family, geometry, noncompact_world)
                .with_context(|| format!("{family} TP{noncompact_world} should ship segments"))?;
            assert_eq!(width, geometry.intermediate / noncompact_world as u32);
            assert_eq!(exl3_segment_width(family, geometry, compact_world), None,
                "{family} TP{compact_world} is compact and must not charge segments");
            let (dev, pin) = exl3_wave_bytes(geometry, width, prefill).unwrap();
            assert!(dev > 0 && pin == 2752 * 16, "{family}: positive merged-wave charge");
            assert_eq!(exl3_wave_bytes(geometry, width, singleton), Some((0, 0)));
            assert_eq!(exl3_wave_bytes(geometry, width, decode), Some((0, 0)));
            if matches!(family, "inkling2" | "hy4_flash") {
                let nc_single = recorded_exl3_owner_bytes(&rows, family, geometry, width, false, singleton)?;
                let nc_waves = recorded_exl3_owner_bytes(&rows, family, geometry, width, false, prefill)?;
                assert_eq!(nc_waves.scratch, nc_single.scratch);
                assert_eq!(nc_waves.io - nc_single.io, dev);
                assert_eq!(nc_waves.pinned_staging - nc_single.pinned_staging, pin);
                let compact_width = (geometry.intermediate / 128).div_ceil(compact_world as u32) * 128;
                let c_single = recorded_exl3_owner_bytes(&rows, family, geometry, compact_width, true, singleton)?;
                let c_waves = recorded_exl3_owner_bytes(&rows, family, geometry, compact_width, true, prefill)?;
                assert_eq!(c_waves, c_single, "{family} compact layout must not add segments");
            }
        }
        // Muse Spark uses PartialReplies and never loads m8192-segments.
        assert_eq!(exl3_segment_width("muse_spark", ExpertGeometry::MUSE_SPARK, 4), None);
        assert_eq!(exl3_segment_width("muse_spark", ExpertGeometry::MUSE_SPARK, 8), None);
        Ok(())
    }

}
