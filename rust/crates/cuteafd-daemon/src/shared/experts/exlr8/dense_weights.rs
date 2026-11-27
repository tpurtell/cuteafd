//! Shared native EXLR8 dense operands. Admission and loading consume the same
//! rank slices, absorption layouts and lossless conversion recipes. Source
//! staging is load-only; no converted operand retains a raw resident copy.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::plan::exlr8::dense::{DenseComponent, DenseEncoding, DenseRecord,
    DenseTransform, KvBLayout};
use std::collections::HashMap;
use crate::shared::memory::Tag;

// Records and their exact allocation requests are loader recipes, so
// admission charges the arena this loader binds. This file binds them only.
use cuteafd_loader::plan::exlr8::dense::requests::{absorbed_bytes, key, plain_pack_layouts, LayerRecords,
    PlainPack, ABSORBED_NAMES};

use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES};
use crate::shared::memory::{DeviceAllocation, device::{Allocation, Device}};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};

/// One checkpoint byte range as a single-row rectangle landing whole.
fn rect(name: &str, range: &cuteafd_loader::plan::exlr8::dense::DenseRange) -> Rect {
    Rect { name: name.into(), path: range.path.clone(), base: range.offset, row_bytes: range.bytes,
        rows: 0..1, cols: 0..range.bytes }
}

fn source_rect(name: &str, component: &DenseComponent, transform: DenseTransform) -> Result<Rect> {
    let window = transform.read_window(component)?;
    Ok(Rect { name: name.into(), path: component.source.path.clone(), base: component.source.offset,
        row_bytes: window.row_bytes, rows: window.rows, cols: window.cols })
}

/// Build the very steps the mount and push readers execute, without allocating
/// a raw BF16 device operand. This recipe is CPU-testable with a buffer descriptor.
fn widen_steps(name: &str, component: &DenseComponent, device: i32, dst: CuteafdDeviceBuffer)
    -> Result<Vec<Step>> {
    let bytes = DenseTransform::WidenBf16.resident_bytes(component)?;
    ensure!(bytes <= dst.bytes && !dst.ptr.is_null(), "native widening destination too small");
    let source = source_rect(name, component, DenseTransform::WidenBf16)?;
    let width = source.width();
    crate::shared::experts::rtx::dense::windows(&[source], 1)?.into_iter().map(|(rects, rows)| {
        let offset = rows.start.checked_mul(width).and_then(|n| n.checked_mul(2)).context("native widening offset overflow")?;
        let source_bytes = rows.len().checked_mul(width).context("native widening step overflow")?;
        let bytes = source_bytes.checked_mul(2).context("native widening step overflow")?;
        ensure!(offset.checked_add(bytes).is_some_and(|end| end <= dst.bytes), "native widening step outside operand");
        let dst = CuteafdDeviceBuffer {
            // SAFETY: the checked byte extent is within the caller's live operand.
            ptr: unsafe { dst.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..dst };
        Ok(Step { device, rects, jobs: vec![Job::Widen { at: 0, bytes: source_bytes, dst }] })
    }).collect()
}

/// Allocate only the admitted resident representation; source bytes are either
/// landed unchanged or staged load-only for the existing lossless Widen job.
pub(crate) fn landed<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, name: &str, component: &DenseComponent,
    transform: DenseTransform, stream: *mut std::ffi::c_void) -> Result<DeviceAllocation<'a>> {
    let device = Device { library, id: library.cuda_get_device()? };
    landed_with(library, load, name, component, transform, stream,
        |_, bytes| Allocation::new(device, bytes, Tag::weights("shared/exlr8-dense")))
}

pub(crate) fn landed_with<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, name: &str,
    component: &DenseComponent, transform: DenseTransform, stream: *mut std::ffi::c_void,
    mut allocate: impl FnMut(&str, usize) -> Result<Allocation<'a>>) -> Result<DeviceAllocation<'a>> {
    let bytes = transform.allocation_bytes(component)?;
    let device = library.cuda_get_device()?;
    let allocation = allocate(name, bytes)?;
    ensure!(std::ptr::eq(allocation.device.library, library), "native landing library mismatch");
    let allocation = DeviceAllocation::from_allocation(allocation);
    ensure!(allocation.buffer.bytes == bytes && allocation.buffer.device_id == device && !allocation.buffer.ptr.is_null(),
        "{name}: native actual allocation differs from request");
    land_into(library, load, name, component, transform, allocation.buffer, stream)?;
    Ok(allocation)
}

fn land_into(library: &NativeLibrary, load: &DenseLoad<'_>, name: &str, component: &DenseComponent,
    transform: DenseTransform, destination: CuteafdDeviceBuffer, stream: *mut std::ffi::c_void) -> Result<()> {
    let bytes = transform.resident_bytes(component)?;
    ensure!(!destination.ptr.is_null() && destination.bytes >= bytes, "native landing destination extent");
    let device = library.cuda_get_device()?;
    ensure!(destination.device_id == device, "native landing destination device");
    match transform {
        DenseTransform::Identity => load.operand(Operand { device, buffer: destination,
            lands: vec![(source_rect(name, component, transform)?, 0)], bytes })?,
        DenseTransform::WidenBf16 => {
            for step in widen_steps(name, component, device, destination)? { load.step(step, stream)?; }
        }
    }
    Ok(())
}

fn allocate_absorbed<T>(layout: &KvBLayout,
    mut allocate: impl FnMut(usize) -> Result<T>) -> Result<[T; 4]> {
    let [uk, uk_scale, uv, uv_scale] = absorbed_bytes(layout);
    Ok([allocate(uk)?, allocate(uk_scale)?, allocate(uv)?, allocate(uv_scale)?])
}

/// A layer's native operands on the current device (this rank's slices), keyed
/// [`key`]; `kv_b` split per head for heads `h0..h0 + heads` into the FP8
/// programs' `w_uk_fp8`/`w_uk_scale`/`w_uv_fp8`/`w_uv_scale`.
pub(crate) fn load_layer<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, records: &LayerRecords,
    stream: *mut std::ffi::c_void) -> Result<HashMap<String, DeviceAllocation<'a>>> {
    load_layer_packed(library, load, records, &[], stream)
}

pub(crate) enum LayerOperation<'a> {
    Land { name: String, component: &'a DenseComponent, transform: DenseTransform },
    Pack { pack: &'a PlainPack, offsets: Vec<usize> },
    Absorb { record: &'a DenseRecord, layout: &'a KvBLayout },
}

pub(crate) struct LayerRequest<'a> {
    pub owners: Vec<(String, usize)>,
    pub operation: LayerOperation<'a>,
}

/// The native reader and CPU replay traverse the identical allocation dispatch.
/// Owners and bytes equal the loader's `requests::layer_requests`, which admission charges.
pub(crate) fn layer_requests<'a>(records: &'a LayerRecords, packs: &'a [PlainPack],
    mut receive: impl FnMut(LayerRequest<'a>) -> Result<()>) -> Result<()> {
    let (layouts, consumed) = plain_pack_layouts(records, packs)?;
    for (stem, (projection, slice)) in &records.projections {
        ensure!(!matches!(projection.encoding, DenseEncoding::F32 | DenseEncoding::F16 | DenseEncoding::RowFp8),
            "{}: no native dense kernel for {:?}", slice.tensor, projection.encoding);
        for component in &slice.components {
            receive(LayerRequest { owners: vec![(key(stem, &component.suffix), slice.transform.allocation_bytes(component)?)],
                operation: LayerOperation::Land { name: format!("{}.{}", slice.tensor, component.suffix), component,
                    transform: slice.transform } })?;
        }
    }
    for (tensor, (component, transform)) in &records.plain {
        if !consumed.contains(tensor.as_str()) {
            receive(LayerRequest { owners: vec![(tensor.clone(), transform.allocation_bytes(component)?)],
                operation: LayerOperation::Land { name: tensor.clone(), component, transform: *transform } })?;
        }
    }
    for (pack, (bytes, offsets)) in layouts {
        receive(LayerRequest { owners: vec![(pack.name.clone(), bytes)], operation: LayerOperation::Pack { pack, offsets } })?;
    }
    if let Some((record, layout)) = &records.kv_b {
        let owners = ABSORBED_NAMES.into_iter().zip(absorbed_bytes(layout))
            .map(|(name, bytes)| (name.into(), bytes)).collect();
        receive(LayerRequest { owners, operation: LayerOperation::Absorb { record, layout } })?;
    }
    Ok(())
}

pub(crate) fn load_layer_packed<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, records: &LayerRecords,
    packs: &[PlainPack], stream: *mut std::ffi::c_void) -> Result<HashMap<String, DeviceAllocation<'a>>> {
    let device = Device { library, id: library.cuda_get_device()? };
    load_layer_packed_with(library, load, records, packs, stream, |_, bytes| Allocation::new(device, bytes, Tag::weights("shared/exlr8-dense")))
}

pub(crate) fn load_layer_packed_with<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, records: &LayerRecords,
    packs: &[PlainPack], stream: *mut std::ffi::c_void,
    mut allocate: impl FnMut(&str, usize) -> Result<Allocation<'a>>) -> Result<HashMap<String, DeviceAllocation<'a>>> {
    let mut out = HashMap::new();
    let device = library.cuda_get_device()?;
    layer_requests(records, packs, |request| {
        let buffers = match request.operation {
            LayerOperation::Land { name, component, transform } => vec![landed_with(library, load, &name, component,
                transform, stream, |_, bytes| allocate(&format!("layers.{}.{}", records.layer, request.owners[0].0), bytes))?],
            LayerOperation::Pack { pack, offsets } => {
                let allocation = allocate(&format!("layers.{}.{}", records.layer, request.owners[0].0), request.owners[0].1)?;
                ensure!(std::ptr::eq(allocation.device.library, library) && allocation.buffer.device_id == device
                    && allocation.device.id == device && allocation.buffer.bytes == request.owners[0].1
                    && !allocation.buffer.ptr.is_null(), "native pack allocation differs from request");
                let allocation = DeviceAllocation::from_allocation(allocation);
                for (name, offset) in pack.components.iter().zip(offsets) {
                    let (component, transform) = &records.plain[name];
                    let extent = transform.resident_bytes(component)?;
                    ensure!(offset.checked_add(extent).is_some_and(|end| end <= allocation.buffer.bytes),
                        "native packed component outside allocation");
                    let destination = CuteafdDeviceBuffer {
                        // SAFETY: checked disjoint component subrange in this live allocation.
                        ptr: unsafe { allocation.buffer.ptr.cast::<u8>().add(offset) }.cast(),
                        bytes: extent, ..allocation.buffer };
                    land_into(library, load, name, component, *transform, destination, stream)?;
                }
                vec![allocation]
            }
            LayerOperation::Absorb { record, layout } => absorb_with(library, load, record, layout, stream,
                |name, bytes| allocate(&format!("layers.{}.{name}", records.layer), bytes))?.into_iter().collect(),
        };
        ensure!(buffers.len() == request.owners.len(), "native request owner count differs");
        for ((name, bytes), buffer) in request.owners.into_iter().zip(buffers) {
            ensure!(buffer.buffer.bytes == bytes && buffer.buffer.device_id == device && !buffer.buffer.ptr.is_null(),
                "{name}: native actual allocation differs from request");
            ensure!(out.insert(name, buffer).is_none(), "duplicate native owner");
        }
        Ok(())
    })?;
    Ok(out)
}

pub(crate) fn absorb_with<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, kv_b: &DenseRecord,
    layout: &KvBLayout, stream: *mut std::ffi::c_void,
    mut allocate: impl FnMut(&str, usize) -> Result<Allocation<'a>>) -> Result<[DeviceAllocation<'a>; 4]> {
    let geometry = layout.geometry;
    let (heads_total, nope, v, lat) = (geometry.heads, geometry.nope, geometry.values, geometry.latent);
    let (h0, heads) = (layout.first_head, layout.heads);
    ensure!(kv_b.shape == [heads_total * (nope + v), lat] && h0 + heads <= heads_total
        && kv_b.components.len() == 2 && kv_b.components[0].suffix == "fp8" && kv_b.components[1].suffix == "scale",
        "{}: native kv_b layout", kv_b.tensor);
    let (values, grid) = (&kv_b.components[0].source, &kv_b.components[1].source);
    let mut names = ["w_uk_fp8", "w_uk_scale", "w_uv_fp8", "w_uv_scale"].into_iter();
    let [uk, uk_scale, uv, uv_scale] = allocate_absorbed(layout, |bytes| {
        let owner = allocate(names.next().expect("four absorption owners"), bytes)?;
        ensure!(std::ptr::eq(owner.device.library, library) && owner.device.id == library.cuda_get_device()?
            && owner.buffer.bytes == bytes && !owner.buffer.ptr.is_null(), "native absorption allocation differs from request");
        Ok(DeviceAllocation::from_allocation(owner))
    })?;
    ensure!(STEP_BYTES == cuteafd_loader::plan::exlr8::dense::DENSE_STEP_BYTES,
        "native dense plan/load staging bound mismatch");
    let per = ((STEP_BYTES - grid.bytes) / ((nope + v) * lat)).max(1);
    let device = library.cuda_get_device()?;
    let at = |buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize| CuteafdDeviceBuffer {
        // SAFETY: offset + bytes lie inside the head-major operand (checked by the job).
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..buffer };
    let mut h = 0;
    while h < heads {
        let count = per.min(heads - h);
        let row0 = (h0 + h) * (nope + v);
        let rows = count * (nope + v);
        let staged = Rect { name: format!("{}.fp8", kv_b.tensor), path: values.path.clone(), base: values.offset,
            row_bytes: lat, rows: row0..row0 + rows, cols: 0..lat };
        let whole_grid = rect(&format!("{}.scale", kv_b.tensor), grid);
        let job = Job::AbsorbFp8 { at: 0, grid_at: rows * lat, grid_bytes: grid.bytes, row0, heads: count, nope, v, lat,
            uk: at(uk.buffer, h * lat * nope, count * lat * nope),
            uk_scale: at(uk_scale.buffer, h * lat * (nope / 64) * 4, count * lat * (nope / 64) * 4),
            uv: at(uv.buffer, h * v * lat, count * v * lat),
            uv_scale: at(uv_scale.buffer, h * v * (lat / 64) * 4, count * v * (lat / 64) * 4) };
        load.step(Step { device, rects: vec![staged, whole_grid], jobs: vec![job] }, stream)?;
        h += count;
    }
    Ok([uk, uk_scale, uv, uv_scale])
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cuteafd_loader::plan::exlr8::dense::{DenseExecution, DenseOwnership, KvBGeometry,
        DensePlan, DenseSelection, DenseSlice, ShardSide};
    use cuteafd_loader::plan::exlr8::dense::requests::{packed_plain_layout, NativeDenseRecords, NativeProjection};
    use std::collections::BTreeMap;

    fn configs() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../../../cuteafd-loader/tests/fixtures/exlr8-dense-configs.json")).unwrap()
    }

    /// Native header-only fixture: published shapes, sparse payload, tiny record
    /// tables. No checkpoint payload or GPU is needed for loader selection.
    pub(crate) fn fixture(family: &str) -> tempfile::TempDir {
        use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
        use serde_json::json;
        use std::io::{Seek, SeekFrom, Write};
        let root = tempfile::tempdir().unwrap();
        let config = configs()[family].clone();
        std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
        std::fs::create_dir(root.path().join("dense")).unwrap();
        let selection = DenseSelection::from_hf(&config).unwrap();
        let (hidden, heads, nope, values, latent, mla, kda) = if family == "flash" {
            let cfg = GlmNextConfig::from_hf(&config).unwrap();
            let mla = cfg.attention.iter().position(|a| *a == GlmNextAttention::Mla).unwrap();
            let kda = cfg.attention.iter().position(|a| *a == GlmNextAttention::Kda).unwrap();
            (cfg.hidden, cfg.heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank, mla, Some((kda, cfg)))
        } else {
            let cfg = cuteafd_loader::families::glm5::GlmDsaConfig::from_hf(&config).unwrap();
            (cfg.hidden, cfg.heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank, 0, None)
        };
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), json!({"format":"exlr8-v2", "scheme":"exlr8-e4m3-v2"}));
        let mut offset = 0usize;
        let mut tiny = Vec::new();
        let mut tensor = |name: String, dtype: &str, shape: Vec<usize>, record: Option<[i64; 8]>| {
            let bytes = shape.iter().product::<usize>() * match dtype { "U8" => 1, "BF16" => 2, "F32" => 4, "I64" | "U64" => 8, _ => panic!() };
            header.insert(name, json!({"dtype":dtype, "shape":shape, "data_offsets":[offset, offset + bytes]}));
            if let Some(record) = record { tiny.push((offset, record)); }
            offset += bytes;
        };
        tensor(selection.norm_name(), "BF16", vec![hidden], None);
        let p = selection.layer_prefix(mla).unwrap();
        let n = heads * (nope + values);
        // Native block FP8, replicated storage, no rotation.
        tensor(format!("{p}.self_attn.kv_b_proj.record"), "I64", vec![8], Some([4, n as i64, latent as i64, 0, 0, 0, 0, 0]));
        tensor(format!("{p}.self_attn.kv_b_proj.fp8"), "U8", vec![n, latent], None);
        tensor(format!("{p}.self_attn.kv_b_proj.scale"), "F32", vec![n / 128, latent / 128], None);
        tensor(format!("{p}.self_attn.kv_b_proj.checksum"), "U64", vec![n / 128], None);
        if let Some((layer, cfg)) = kda {
            let p = selection.layer_prefix(layer).unwrap();
            let width = cfg.kda_heads * cfg.kda_head_dim;
            let conv = config["text_config"]["linear_attn_config"]["short_conv_kernel_size"].as_u64().unwrap() as usize;
            for leaf in ["hc_attn_fn", "hc_ffn_fn"] {
                tensor(format!("{p}.{leaf}"), "BF16", vec![2 * cfg.hc_mult + cfg.hc_mult * cfg.hc_mult, cfg.hc_mult * hidden], None);
            }
            for leaf in ["q", "k", "v"] {
                tensor(format!("{p}.self_attn.{leaf}_conv1d.weight"), "BF16", vec![width, 1, conv], None);
            }
            tensor(format!("{p}.self_attn.b_proj.weight"), "BF16", vec![cfg.kda_heads, hidden], None);
            let routed = selection.layer_prefix(cfg.dense.iter().position(|dense| !dense).unwrap()).unwrap();
            tensor(format!("{routed}.mlp.gate.e_score_correction_bias"), "F32", vec![cfg.experts], None);
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(root.path().join("dense/fixture.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        let start = 8 + header.len();
        file.set_len((start + offset) as u64).unwrap();
        for (offset, record) in tiny {
            file.seek(SeekFrom::Start((start + offset) as u64)).unwrap();
            for value in record { file.write_all(&value.to_le_bytes()).unwrap(); }
        }
        root
    }

    #[test]
    fn glm_whole_attention_records_and_scratch_follow_owners() -> Result<()> {
        use cuteafd_loader::families::glm5::{GlmDsaConfig, native};
        let root = fixture("glm");
        let config = serde_json::from_slice(&std::fs::read(root.path().join("config.json"))?)?;
        let cfg = GlmDsaConfig::from_hf(&config)?;
        let records = DenseRecord::inventory(root.path())?;
        let whole_mla = BTreeMap::from([(0, 1)]);
        for rank in 0..2 {
            let owned = NativeDenseRecords::new(root.path(), 1, 2, rank, &whole_mla)?;
            assert_eq!(owned.layers[0].kv_b.is_some(), rank == 1);
            if rank == 0 {
                assert!(owned.plan.tensors.iter().all(|slice| !slice.tensor.contains(".self_attn.")));
            } else {
                assert_eq!(owned.layers[0].kv_b.as_ref().unwrap().1.heads, cfg.heads);
                let layouts = native::scratch_layouts(&cfg, &records, &owned.plan, 64, 4096, false)?;
                for (rows, layout) in [64, 4096].into_iter().zip(layouts) {
                    assert_eq!(layout.stages["q"], rows * cfg.heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) * 2);
                    assert_eq!(layout.stages["values"], rows * cfg.heads * cfg.v_head_dim * 2);
                    assert_eq!(layout.stages["gate"], rows * (cfg.dense_intermediate / 2) * 2);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn flash_alternating_mla_records_and_residency_follow_both_first_owners() -> Result<()> {
        use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig, resident};
        use cuteafd_loader::plan::exlr8::native::DenseFacts;
        let root = fixture("flash");
        let cfg = GlmNextConfig::read(root.path())?;
        let config = serde_json::from_slice(&std::fs::read(root.path().join("config.json"))?)?;
        let selection = DenseSelection::from_hf(&config)?;
        let mut records = DenseRecord::inventory(root.path())?;
        let original = records.iter().find(|record| record.tensor.ends_with(".self_attn.kv_b_proj.weight"))
            .context("fixture MLA record")?.clone();
        let mla = cfg.attention.iter().enumerate().filter_map(|(layer, kind)|
            (*kind == GlmNextAttention::Mla).then_some(layer)).collect::<Vec<_>>();
        assert!(mla.len() > 1);
        for &layer in mla.iter().skip(1) {
            let mut record = original.clone();
            record.tensor = format!("{}.self_attn.kv_b_proj.weight", selection.layer_prefix(layer)?);
            record.ownership.whole_mla = Some(layer);
            records.push(record);
        }
        let facts = DenseFacts::new(records.clone(), selection.clone(), cfg.layers, vec![0; 2])?;
        for first in [0, 1] {
            let owners = mla.iter().enumerate().map(|(index, &layer)| (layer, first ^ (index % 2)))
                .collect::<BTreeMap<_, _>>();
            let plans = facts.plans(&owners)?;
            let admitted = resident::native_resident(&selection, cfg.layers, &plans, &[0; 2])?;
            for rank in 0..2 {
                let runtime = NativeDenseRecords::from_records(&records, &selection, cfg.layers, 2, rank, &owners)?;
                assert_eq!(runtime.plan.resident_bytes as u64, admitted[rank].weights + admitted[rank].embedding);
                for (&layer, &owner) in &owners {
                    assert_eq!(runtime.layers[layer].kv_b.is_some(), rank == owner);
                    let name = format!("{}.self_attn.kv_b_proj.weight", selection.layer_prefix(layer)?);
                    assert_eq!(runtime.plan.tensors.iter().filter(|slice| slice.tensor == name).count(), usize::from(rank == owner));
                    if rank == owner { assert_eq!(runtime.layers[layer].kv_b.as_ref().unwrap().1.heads, cfg.heads); }
                }
            }
            for &layer in &mla {
                let name = format!("{}.self_attn.kv_b_proj.weight", selection.layer_prefix(layer)?);
                assert_eq!(plans.iter().flat_map(|plan| &plan.tensors).filter(|slice| slice.tensor == name).count(), 1);
            }
        }
        Ok(())
    }

    #[test]
    fn compact_native_reader_dispatch_matches_real_allocations() -> Result<()> {
        use cuteafd_loader::plan::exlr8::dense::{DenseAxis, DenseWindow};
        let root = std::path::PathBuf::from(std::env::var_os("HOME").context("HOME")?)
            .join(".cache/cuteafd/builds/glm-weight-inventory-fixtures");
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let component = |name: &str, suffix: &str, dtype: &str, shape: Vec<usize>, bytes: usize| -> Result<DenseComponent> {
            let path = root.path().join(name);
            let seed = name.bytes().fold(0u8, |value, byte| value.wrapping_add(byte));
            let data = match dtype {
                "BF16" => (0..bytes / 2).flat_map(|i| (0x3f00u16 + (i % 127) as u16).to_le_bytes()).collect(),
                "F16" => (0..bytes / 2).flat_map(|i| (0x3800u16 + (i % 127) as u16).to_le_bytes()).collect(),
                "F32" => (0..bytes / 4).flat_map(|i| (0.5f32 + i as f32 / 16.0).to_le_bytes()).collect(),
                _ => (0..bytes).map(|i| 1 + ((i + usize::from(seed)) % 119) as u8).collect::<Vec<_>>(),
            };
            std::fs::write(&path, data)?;
            Ok(DenseComponent { suffix: suffix.into(), dtype: dtype.into(), shape,
                source: cuteafd_loader::plan::exlr8::dense::DenseRange { path, offset: 0, bytes }, window: None })
        };
        let mut projections = HashMap::new();
        for (i, (encoding, components)) in [
            (DenseEncoding::Trellis { bits: 4 }, vec![("trellis", "U8", 8192), ("suh", "F16", 256), ("svh", "F16", 256)]),
            (DenseEncoding::Int4G32, vec![("q4", "U8", 8192), ("s4", "F16", 1024)]),
            (DenseEncoding::Mxfp8, vec![("fp8", "U8", 16384), ("e8m0", "U8", 512)]),
            (DenseEncoding::Bf16, vec![("bf16", "BF16", 32768)]),
        ].into_iter().enumerate() {
            let stem = ["q_a", "kv_a", "q_b", "o"][i];
            let components = components.into_iter().map(|(suffix, dtype, bytes)| component(
                &format!("{stem}.{suffix}"), suffix, dtype, vec![bytes], bytes)).collect::<Result<Vec<_>>>()?;
            let slice = DenseSlice { tensor: format!("model.layers.0.{stem}.weight"), absorbed: None,
                transform: DenseTransform::Identity, axis: DenseAxis::Output, first_channel: 0, channels: 128,
                side: ShardSide::Output, components, units: Vec::new() };
            projections.insert(stem, (NativeProjection { encoding, shape: [128, 128], side: ShardSide::Output,
                rotation_id: 0 }, slice));
        }
        let mut plain = HashMap::new();
        let mut widened = component("wide", "weight", "BF16", vec![128], 256)?;
        widened.window = Some(DenseWindow { row_bytes: 2, rows: 0..128, cols: 0..2 });
        plain.insert("wide".into(), (widened, DenseTransform::WidenBf16));
        plain.insert("norm".into(), (component("norm", "weight", "BF16", vec![128], 256)?, DenseTransform::Identity));
        for name in ["packed_a", "packed_b"] {
            plain.insert(name.into(), (component(name, "weight", "BF16", vec![128], 256)?, DenseTransform::Identity));
        }
        let geometry = KvBGeometry { heads: 2, nope: 64, values: 128, latent: 128 };
        let kv = DenseRecord { tensor: "model.layers.0.self_attn.kv_b_proj.weight".into(),
            encoding: DenseEncoding::BlockFp8, shape: vec![384, 128], side: ShardSide::Replicated,
            ownership: DenseOwnership::new(DenseExecution::AbsorbHeads(geometry)), channels_per_shard: 128,
            rotation_id: 0, rotation: None, hessian_solved: false, native_blocks: 0,
            components: vec![component("kv.fp8", "fp8", "U8", vec![384, 128], 384 * 128)?,
                component("kv.scale", "scale", "F32", vec![3, 1], 12)?] };
        let records = LayerRecords { layer: 0, projections, plain, kv_b: Some((kv, geometry.layout(1, 0)?)) };
        let packs = [PlainPack { name: "packed".into(), components: vec!["packed_a".into(), "packed_b".into()] }];
        let mut requests = BTreeMap::new();
        layer_requests(&records, &packs, |r| { for (name, bytes) in r.owners { assert!(requests.insert(name, bytes).is_none()); } Ok(()) })?;
        let load = DenseLoad::mount(&library);
        let actual = load_layer_packed(&library, &load, &records, &packs, std::ptr::null_mut())?;
        assert_eq!(requests, actual.iter().map(|(name, owner)| (name.clone(), owner.buffer.bytes)).collect());
        let read = |owner: &DeviceAllocation<'_>| -> Result<Vec<u8>> {
            let mut data = vec![0; owner.buffer.bytes];
            library.copy_d2h_quiescent(&mut data, owner.buffer)?;
            assert!(data.iter().any(|&byte| byte != 0));
            Ok(data)
        };
        let expected = actual.iter().map(|(name, owner)| Ok((name.clone(), read(owner)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let named: BTreeMap<_, _> = requests.iter().map(|(name, bytes)| (format!("layers.0.{name}"), *bytes)).collect();
        // Admission charges the loader recipe; it must equal what the binder allocates.
        let admitted = cuteafd_loader::plan::exlr8::dense::requests::layer_requests(&records, &packs)?;
        assert_eq!(admitted.len(), named.len(), "loader request list has duplicate or missing owners");
        assert_eq!(admitted.into_iter().collect::<BTreeMap<_, _>>(), named);
        for recording in [false, true] {
            use crate::shared::memory::{weight_arena::DeviceWeightArena, receipts::Inventory};
            use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, Phase};
            let arena = DeviceWeightArena::new(Device { library: &library, id: 0 },
                cuteafd_loader::serving_capacity::WeightArenaLayout::new("test/exlr8", named.clone())?, crate::shared::memory::TEST_ARENA)?;
            let load = if recording { DenseLoad::push(&library) } else { DenseLoad::mount(&library) };
            let mut called = BTreeMap::new();
            let packed = load_layer_packed_with(&library, &load, &records, &packs, std::ptr::null_mut(), |name, bytes| {
                assert!(called.insert(name.to_owned(), bytes).is_none());
                arena.view(name, bytes)
            })?;
            assert_eq!(called, named, "binder allocations differ from the admitted loader requests");
            if let Some(sink) = &load.sink {
                let recorded = sink.take_all();
                assert!(!recorded.operands.is_empty() && !recorded.steps.is_empty());
                for buffer in recorded.operands.iter().map(|operand| operand.buffer)
                    .chain(recorded.steps.iter().flat_map(|step| step.jobs.iter().flat_map(|job| match job {
                        Job::Widen { dst, .. } => vec![*dst],
                        Job::AbsorbFp8 { uk, uk_scale, uv, uv_scale, .. } => vec![*uk, *uk_scale, *uv, *uv_scale],
                        _ => panic!("unexpected EXLR8 fixture transform"),
                    }))) {
                    assert!(packed.values().any(|owner| {
                        let offset = (buffer.ptr as usize).checked_sub(owner.buffer.ptr as usize);
                        offset.and_then(|at| at.checked_add(buffer.bytes)).is_some_and(|end| end <= owner.buffer.bytes)
                    }));
                }
                crate::shared::experts::rtx::dense::fill_records_by_mount(&library, &recorded, |_| std::ptr::null_mut())?;
            }
            load.release();
            drop(arena);
            assert_eq!(expected, packed.iter().map(|(name, owner)| Ok((name.clone(), read(owner)?)))
                .collect::<Result<BTreeMap<_, _>>>()?);
            let physical = [0];
            let mut inventory = Inventory::new(CoordinatorDevices::new(&physical)?, "dense",
                Lifetime { first: Phase::Load, last: Phase::Growth },
                cuteafd_core::memory_layout::Category::Weights, "dense weights");
            for (name, owner) in &packed { owner.inventory(&mut inventory, name)?; }
            assert_eq!(inventory.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).count(), 1);
            assert_eq!(inventory.receipts.iter().filter(|receipt| receipt.allocation.view.is_some()).count(), packed.len());
            assert!(inventory.receipts.iter().filter_map(|receipt| receipt.allocation.view.as_ref())
                .all(|view| view.backing == "test/exlr8"));
        }
        Ok(())
    }

    #[test]
    fn shared_loader_uses_config_paths_planner_owners_and_excludes_mtp() {
        for family in ["glm", "flash"] {
            let root = fixture(family);
            let config = configs()[family].clone();
            let selection = DenseSelection::from_hf(&config).unwrap();
            let text = config.get("text_config").unwrap_or(&config);
            let layers = text["num_hidden_layers"].as_u64().unwrap() as usize;
            let mut records = DenseRecord::inventory(root.path()).unwrap();
            let before = NativeDenseRecords::from_records(&records, &selection, layers, 1, 0, &BTreeMap::new()).unwrap();
            let mut mtp = records.iter().find(|r| r.tensor.contains("layers.")).unwrap().clone();
            let last = selection.layer_prefix(layers - 1).unwrap();
            mtp.tensor = format!("{}.{layers}.eh_proj.weight", last.rsplit_once('.').unwrap().0);
            let mtp_name = mtp.tensor.clone();
            records.push(mtp);
            for world in [1, 2] { for rank in 0..world {
                let shared = NativeDenseRecords::from_records(&records, &selection, layers, world, rank, &BTreeMap::new()).unwrap();
                assert!(!shared.plan.tensors.iter().any(|s| s.tensor == mtp_name));
                assert_eq!(shared.layers.len(), layers);
                let allocated: usize = shared.layers.iter().map(|layer| {
                    let absorbed = layer.kv_b.as_ref().map_or(0, |(_, layout)| allocate_absorbed(layout, Ok).unwrap().iter().sum());
                    let plain: usize = layer.plain.values().map(|(c, t)| t.allocation_bytes(c).unwrap()).sum();
                    let projections: usize = layer.projections.values().flat_map(|(_, s)| s.components.iter().map(move |c|
                        s.transform.allocation_bytes(c).unwrap())).sum();
                    absorbed + plain + projections
                }).sum();
                let norm = if rank == 0 { DenseTransform::Identity.allocation_bytes(&shared.norm).unwrap() } else { 0 };
                assert_eq!(allocated + norm, shared.plan.resident_bytes);
                if world == 1 { assert_eq!(shared.plan.resident_bytes, before.plan.resident_bytes); }
            } }
            if family == "flash" {
                let mla = before.layers.iter().position(|l| l.kv_b.is_some()).unwrap();
                let owners = BTreeMap::from([(mla, 1)]);
                let rank0 = NativeDenseRecords::from_records(&records, &selection, layers, 2, 0, &owners).unwrap();
                let rank1 = NativeDenseRecords::from_records(&records, &selection, layers, 2, 1, &owners).unwrap();
                assert!(rank0.layers[mla].kv_b.is_none());
                assert_eq!(rank1.layers[mla].kv_b.as_ref().unwrap().1, before.layers[mla].kv_b.as_ref().unwrap().1);
                assert!(rank0.layers[0].kv_b.is_none());
                assert!(rank0.layers[0].projections.contains_key("beta"));
                assert!(!rank0.layers[0].plain.keys().any(|n| n.ends_with("b_proj.weight")));
                let bias = rank0.layers.iter().flat_map(|l| &l.plain)
                    .find(|(name, _)| name.ends_with("e_score_correction_bias")).unwrap().1;
                assert_eq!(bias.0.source.bytes, 288 * 4);
                assert_eq!(bias.1.allocation_bytes(&bias.0).unwrap(), 1152);
                assert_eq!(rank0.layers[0].plain.len(), 5);
                assert!(rank0.layers[0].plain.values().all(|(_, t)| *t == DenseTransform::WidenBf16));
            } else {
                assert!(before.layers.iter().flat_map(|l| l.plain.values()).all(|(_, t)| *t == DenseTransform::Identity));
            }
        }
    }

    #[test]
    #[ignore = "requires CUTEAFD_EXLR8_GLM_SNAPSHOT; headers only, no GPU"]
    fn published_glm_native_owner_totals_match_whole_and_tp2_admission() -> Result<()> {
        use cuteafd_loader::placement::{families::glm5::weights::WeightInventory, planner::Location};
        use cuteafd_loader::plan::exlr8::residency::{ExpertSet, MetadataInput};
        use cuteafd_loader::shared::token_table::native_embedding_owner_bytes;
        let root = std::path::PathBuf::from(std::env::var_os("CUTEAFD_EXLR8_GLM_SNAPSHOT")
            .context("CUTEAFD_EXLR8_GLM_SNAPSHOT")?);
        let config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        let cfg = cuteafd_loader::families::glm5::GlmDsaConfig::from_hf(&config)?;
        let selection = DenseSelection::from_hf(&config)?;
        let records = DenseRecord::inventory(&root)?.into_iter().filter_map(|record|
            match selection.role(&record.tensor, cfg.layers) {
                Ok(cuteafd_loader::plan::exlr8::dense::DenseRole::Excluded) => None,
                Ok(_) => Some(Ok(record)),
                Err(error) => Some(Err(error)),
            }).collect::<Result<Vec<_>>>()?;
        let metadata = MetadataInput::open(&root, "up1")?;
        for world in [1, 2] {
            let plans = (0..world).map(|rank| DensePlan::for_rank(&records, world, rank))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut admitted = WeightInventory::from_native(&selection, cfg.layers, &plans)?;
            let shared = (0..world).map(|rank| (cfg.first_moe_layer..cfg.layers)
                .map(|layer| metadata.plan(layer, world, rank, ExpertSet::Shared))
                .collect::<std::result::Result<Vec<_>, _>>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            admitted.extend_shared(&shared)?;
            for rank in 0..world {
                let loaded = NativeDenseRecords::new(&root, cfg.layers, world, rank, &BTreeMap::new())?;
                let mut actual = if rank == 0 { DenseTransform::Identity.allocation_bytes(&loaded.norm)? } else { 0 };
                for layer in &loaded.layers {
                    layer_requests(layer, &[], |request| {
                        actual += request.owners.iter().map(|(_, bytes)| bytes).sum::<usize>();
                        Ok(())
                    })?;
                }
                for name in selection.vocabulary_names() {
                    let slice = plans[rank].tensors.iter().find(|slice| slice.tensor == name);
                    if let Some(slice) = slice {
                        let bytes = slice.components.iter().map(|c| slice.transform.allocation_bytes(c))
                            .collect::<std::result::Result<Vec<_>, _>>()?.into_iter().sum::<usize>();
                        actual += if selection.resident_group(&slice.tensor, cfg.layers)? == "embedding" {
                            native_embedding_owner_bytes(bytes)
                        } else { bytes };
                    }
                }
                actual += shared[rank].iter().map(|plan| plan.resident_bytes).sum::<usize>();
                let expected = admitted.owners.iter().filter(|owner|
                    owner.allocation.location == Location::coordinator(rank as u8))
                    .map(|owner| owner.allocation.bytes as usize).sum::<usize>();
                assert_eq!(actual, expected, "TP{world} rank{rank}");
                eprintln!("TP{world} rank{rank}: actual={actual}, admitted={expected}; native placeholder={}",
                    if rank == 0 { native_embedding_owner_bytes(0) } else { 0 });
                for onboard in 1..=2 {
                    let layers = cfg.first_moe_layer..cfg.first_moe_layer + onboard;
                    let loaded_experts = layers.clone().map(|layer| metadata.plan(layer, world, rank, ExpertSet::Routed))
                        .collect::<std::result::Result<Vec<_>, _>>()?.iter().map(|p| p.resident_bytes).sum::<usize>();
                    let admitted_experts = cuteafd_loader::plan::exlr8::routed_expert_costs(
                        &metadata, &layers.collect::<Vec<_>>())?.iter().map(|cost|
                            if world == 1 { cost.whole.resident } else { cost.half[rank].resident })
                        .sum::<u64>() as usize;
                    assert_eq!(actual + loaded_experts, expected + admitted_experts);
                    eprintln!("TP{world} rank{rank} onboard={onboard}: expert owners={loaded_experts}");
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires both native EXLR8 checkpoints; headers only, no GPU"]
    fn published_glm_and_flash_shared_loader_matches_admission() {
        for (variable, family) in [("CUTEAFD_EXLR8_GLM_SNAPSHOT", "glm"), ("CUTEAFD_EXLR8_FLASH_SNAPSHOT", "flash")] {
            let root = std::env::var_os(variable).expect(variable);
            let config = configs()[family].clone();
            let text = config.get("text_config").unwrap_or(&config);
            let layers = text["num_hidden_layers"].as_u64().unwrap() as usize;
            for world in [1, 2] { for rank in 0..world {
                let loaded = NativeDenseRecords::new(std::path::Path::new(&root), layers, world, rank, &BTreeMap::new()).unwrap();
                let mut actual = if rank == 0 { DenseTransform::Identity.allocation_bytes(&loaded.norm).unwrap() } else { 0 };
                let mut widened = 0;
                for layer in &loaded.layers {
                    if let Some((_, layout)) = &layer.kv_b {
                        actual += allocate_absorbed(layout, Ok).unwrap().iter().sum::<usize>();
                    }
                    for (component, transform) in layer.plain.values() {
                        actual += transform.allocation_bytes(component).unwrap();
                        widened += usize::from(*transform == DenseTransform::WidenBf16);
                    }
                    for (_, slice) in layer.projections.values() { for component in &slice.components {
                        actual += slice.transform.allocation_bytes(component).unwrap();
                    } }
                }
                assert_eq!(actual, loaded.plan.resident_bytes);
                let kda_layers = text["layer_types"].as_array().map_or(0,
                    |types| types.iter().filter(|t| **t == "linear_attention").count());
                assert_eq!(widened, if family == "flash" { 2 * layers + 3 * kda_layers } else { 0 });
                eprintln!("{family} TP{world} rank{rank}: resident={actual}, widened={widened}, load_host={}", loaded.plan.read_staging_bytes);
            } }
        }
    }

    #[test]
    fn packed_plain_records_keep_logical_offsets_and_exact_admitted_bytes() {
        let root = fixture("flash");
        let config = configs()["flash"].clone();
        let cfg = cuteafd_loader::families::glm5_flash::GlmNextConfig::from_hf(&config).unwrap();
        for world in [1, 2] { for rank in 0..world {
            let records = NativeDenseRecords::new(root.path(), cfg.layers, world, rank, &BTreeMap::new()).unwrap();
            let layer = &records.layers[0];
            let prefix = DenseSelection::from_hf(&config).unwrap().layer_prefix(0).unwrap();
            let pack = PlainPack { name: "conv_w".into(), components: ["q", "k", "v"].iter()
                .map(|kind| format!("{prefix}.self_attn.{kind}_conv1d.weight")).collect() };
            let (bytes, offsets) = packed_plain_layout(layer, &pack).unwrap();
            let component_bytes = cfg.kda_heads / world * cfg.kda_head_dim * 4 * 4;
            assert_eq!(offsets, vec![0, component_bytes, 2 * component_bytes]);
            assert_eq!(bytes, 3 * component_bytes);
            assert_eq!(bytes, pack.components.iter().map(|name| {
                let (component, transform) = &layer.plain[name];
                assert_eq!(*transform, DenseTransform::WidenBf16);
                transform.allocation_bytes(component).unwrap()
            }).sum::<usize>());
            let mut invalid = PlainPack { name: "conv_w".into(), components: pack.components.clone() };
            invalid.components[1] = invalid.components[0].clone();
            assert!(packed_plain_layout(layer, &invalid).is_err());
            invalid.components[1] = "absent".into();
            assert!(packed_plain_layout(layer, &invalid).is_err());
            let mut small = LayerRecords { layer: 0, projections: HashMap::new(), kv_b: None, plain: HashMap::new() };
            for name in ["a", "b"] {
                let mut component = layer.plain[&pack.components[0]].0.clone();
                component.shape = vec![1]; component.source.bytes = 2; component.window = None;
                small.plain.insert(name.into(), (component, DenseTransform::Identity));
            }
            assert!(packed_plain_layout(&small, &PlainPack { name: "packed".into(), components: vec!["a".into(), "b".into()] }).is_err(),
                "packing must not silently change per-record allocator padding");
        } }
    }

    #[test]
    fn widening_jobs_cover_only_the_admitted_f32_allocation_with_bounded_source_staging() {
        use cuteafd_loader::plan::exlr8::dense::DenseRange;
        let source_bytes = STEP_BYTES + 8;
        let component = DenseComponent { suffix: "plain".into(), dtype: "BF16".into(), shape: vec![source_bytes / 2],
            source: DenseRange { path: "not-read".into(), offset: 16, bytes: source_bytes }, window: None };
        let bytes = DenseTransform::WidenBf16.resident_bytes(&component).unwrap();
        let mut resident = vec![0u8; bytes];
        let buffer = CuteafdDeviceBuffer { ptr: resident.as_mut_ptr().cast(), bytes, device_id: 0, flags: 0 };
        let steps = widen_steps("wide", &component, 0, buffer).unwrap();
        assert!(!steps.is_empty());
        let mut written = 0;
        let mut read = 0;
        for step in steps {
            assert!(step.bytes() <= crate::shared::experts::rtx::dense::TRANSFORM_BYTES);
            assert_eq!(step.rects.len(), 1);
            assert_eq!(step.rects[0].base, 16);
            let [Job::Widen { at, bytes, dst }] = step.jobs.as_slice() else { panic!("only a widening job may write the resident operand") };
            assert_eq!(*at, 0);
            assert_eq!(*bytes, step.bytes());
            assert_eq!(dst.bytes, bytes * 2);
            assert_eq!(dst.ptr as usize, buffer.ptr as usize + written);
            written += dst.bytes;
            read += bytes;
        }
        assert_eq!(written, buffer.bytes);
        assert_eq!(read, source_bytes);
        assert!(widen_steps("wide", &component, 0, CuteafdDeviceBuffer { bytes: bytes - 1, ..buffer }).is_err());
    }

    #[test]
    fn replicated_k_slice_admits_actual_reader_batch_and_cache_only_at_load() {
        use cuteafd_loader::plan::exlr8::dense::{DenseAxis, DenseRange};
        use cuteafd_loader::shared::dense_read::{ReadBatch, STEP_BYTES};
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("native.bin");
        std::fs::write(&path, vec![17u8; 8192]).unwrap();
        let record = DenseRecord { tensor: "model.layers.0.self_attn.o_proj.weight".into(),
            encoding: DenseEncoding::Mxfp8, shape: vec![64, 128], side: ShardSide::Replicated,
            ownership: DenseOwnership::new(DenseExecution::Split(DenseAxis::Input)), channels_per_shard: 0,
            rotation_id: 0, rotation: None, hessian_solved: false, native_blocks: 1,
            components: vec![DenseComponent { suffix: "fp8".into(), dtype: "U8".into(), shape: vec![64,128],
                source: DenseRange { path, offset: 0, bytes: 8192 }, window: None }] };
        let plan = DensePlan::for_rank(&[record], 2, 1).unwrap();
        let component = &plan.tensors[0].components[0];
        let w = component.window.as_ref().unwrap();
        let rect = Rect { name: "native".into(), path: component.source.path.clone(), base: 0,
            row_bytes: w.row_bytes, rows: w.rows.clone(), cols: w.cols.clone() };
        let batch = ReadBatch::new(rect.rows.len(), rect.row_bytes, rect.width(), STEP_BYTES).unwrap();
        let part = rect.sub(0..batch.rows);
        let mut packed = vec![0; part.bytes()];
        let mut cache = None;
        part.read_cached(&mut packed, &mut cache).unwrap();
        let cached_bytes = cache.as_ref().unwrap().1.len();
        assert_eq!(cached_bytes, 8192);
        assert_eq!(batch.cached_bytes, cached_bytes);
        assert!(plan.admit_host(plan.read_staging_bytes, 0).unwrap() >= batch.pinned_bytes() + cached_bytes);
        assert_eq!(plan.resident_bytes, 4096);
        assert_eq!(plan.admit(plan.resident_bytes + 256, 1, 1, 0).unwrap(), 4352);
    }

    #[test]
    fn tp2_kv_b_plan_equals_actual_allocation_requests() {
        let geometry = KvBGeometry { heads: 64, nope: 192, values: 256, latent: 512 };
        let record = DenseRecord { tensor: "model.layers.0.self_attn.kv_b_proj.weight".into(),
            encoding: DenseEncoding::BlockFp8, shape: vec![64 * 448, 512], side: ShardSide::Output,
            ownership: DenseOwnership::new(DenseExecution::AbsorbHeads(geometry)),
            channels_per_shard: 128, rotation_id: 0, rotation: None, hessian_solved: false,
            components: vec![], native_blocks: 224 };
        for rank in 0..2 {
            let plan = DensePlan::for_rank(&[record.clone()], 2, rank).unwrap();
            let layout = plan.tensors[0].absorbed.as_ref().unwrap();
            let mut requests = Vec::new();
            let allocations = allocate_absorbed(layout, |bytes| { requests.push(bytes); Ok(bytes) }).unwrap();
            assert_eq!(requests, [3_145_728, 196_608, 4_194_304, 262_144]);
            assert_eq!(allocations, layout.allocations);
            assert_eq!(requests.iter().sum::<usize>(), plan.resident_bytes);
        }
    }
}
