//! Shared native EXLR8 dense operands. Admission and loading consume the same
//! rank slices, absorption layouts and lossless conversion recipes. Source
//! staging is load-only; no converted operand retains a raw resident copy.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::plan::exlr8::dense::{DenseComponent, DenseEncoding, DensePlan, DenseRecord,
    allocation, DenseSelection, DenseSlice, DenseTransform, KvBLayout, ShardSide};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Shared MLA/KDA projection roles, independent of the checkpoint's root prefix.
const PROJECTIONS: [(&str, &str); 19] = [
    ("q_a", "self_attn.q_a_proj"),
    ("kv_a", "self_attn.kv_a_proj_with_mqa"),
    ("q_b", "self_attn.q_b_proj"),
    ("o", "self_attn.o_proj"),
    ("index_q", "self_attn.indexer.wq_b"),
    ("index_k", "self_attn.indexer.wk"),
    ("index_w", "self_attn.indexer.weights_proj"),
    ("gate", "mlp.gate_proj"),
    ("up", "mlp.up_proj"),
    ("down", "mlp.down_proj"),
    ("router", "mlp.gate"),
    ("q", "self_attn.q_proj"),
    ("k", "self_attn.k_proj"),
    ("v", "self_attn.v_proj"),
    ("beta", "self_attn.b_proj"),
    ("f_a", "self_attn.f_a_proj"),
    ("g_a", "self_attn.g_a_proj"),
    ("f_b", "self_attn.f_b_proj"),
    ("g_b", "self_attn.g_b_proj"),
];

/// How one native projection runs: its encoding and this rank's slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeProjection {
    pub encoding: DenseEncoding,
    /// Logical `[output, input]` of this rank's slice.
    pub shape: [usize; 2],
    pub side: ShardSide,
    pub rotation_id: usize,
}

/// One layer's native dense slices on one rank, in record order.
pub(crate) struct LayerRecords {
    #[allow(dead_code)]
    pub layer: usize,
    pub projections: HashMap<&'static str, (NativeProjection, DenseSlice)>,
    /// Absent for KDA and for attention assigned to the other coordinator.
    pub kv_b: Option<(DenseRecord, KvBLayout)>,
    /// Non-projection operands, with the same conversion recipe admission used.
    pub plain: HashMap<String, (DenseComponent, DenseTransform)>,
}

/// The EXLR8 checkpoint's dense records for one rank of `world` coordinators.
pub(crate) struct NativeDenseRecords {
    pub layers: Vec<LayerRecords>,
    pub norm: DenseComponent,
    pub plan: DensePlan,
}

impl NativeDenseRecords {
    pub(crate) fn new(snapshot: &std::path::Path, layers: usize, world: usize, rank: usize,
        whole_mla: &BTreeMap<usize, usize>) -> Result<Self> {
        let records = DenseRecord::inventory(snapshot)?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let selection = DenseSelection::from_hf(&config)?;
        Self::from_records(&records, &selection, layers, world, rank, whole_mla)
    }

    fn from_records(records: &[DenseRecord], selection: &DenseSelection, layers: usize, world: usize,
        rank: usize, whole_mla: &BTreeMap<usize, usize>) -> Result<Self> {
        // Selection precedes both admission and loading: external DFlash2 never
        // makes a checkpoint's native MTP operands resident.
        let kept = selection.backbone(records, layers)?;
        let plan = DensePlan::for_rank_owned(&kept, world, rank, whole_mla)?;
        let by_name: HashMap<&str, &DenseSlice> = plan.tensors.iter().map(|s| (s.tensor.as_str(), s)).collect();
        let record = |name: &str| records.iter().find(|r| r.tensor == name);
        let mut out = Vec::with_capacity(layers);
        for layer in 0..layers {
            let prefix = selection.layer_prefix(layer)?;
            let mut projections = HashMap::new();
            let mut projected = HashSet::new();
            for (stem, path) in PROJECTIONS {
                let tensor = format!("{prefix}.{path}.weight");
                let Some(full) = record(&tensor) else { continue };
                // The shared plan already excludes rank-zero and whole-MLA
                // operands owned by the other rank.
                let Some(slice) = by_name.get(tensor.as_str()) else { continue };
                projected.insert(tensor.clone());
                let channels = slice.channels;
                let shape = match slice.axis {
                    cuteafd_loader::plan::exlr8::dense::DenseAxis::Input => [full.shape[0], channels],
                    _ => [channels, full.shape[1]],
                };
                projections.insert(stem, (NativeProjection { encoding: full.encoding, shape, side: full.side,
                    rotation_id: full.rotation_id }, (*slice).clone()));
            }
            let kv_b_name = format!("{prefix}.self_attn.kv_b_proj.weight");
            let kv_b = by_name.get(kv_b_name.as_str()).map(|slice| -> Result<_> {
                let full = record(&kv_b_name).context("native kv_b source")?;
                ensure!(full.encoding == DenseEncoding::BlockFp8, "{kv_b_name}: native kv_b is block FP8");
                Ok((full.clone(), slice.absorbed.clone().context("native kv_b has no planned absorption layout")?))
            }).transpose()?;
            let mut plain = HashMap::new();
            for slice in plan.tensors.iter().filter(|s| s.tensor.starts_with(&format!("{prefix}."))
                && s.absorbed.is_none() && !projected.contains(&s.tensor)) {
                let full = record(&slice.tensor).context("native dense source record")?;
                ensure!(full.native_blocks == 0 && slice.components.len() == 1,
                    "{}: no native projection role", slice.tensor);
                plain.insert(slice.tensor.clone(), (slice.components[0].clone(), slice.transform));
            }
            out.push(LayerRecords { layer, projections, kv_b, plain });
        }
        let norm_name = selection.norm_name();
        // Keeping a source descriptor on rank 1 allocates no device memory;
        // only rank 0 loads the final norm and runs the head.
        let norm = record(&norm_name).with_context(|| format!("{norm_name}: missing final norm"))?;
        ensure!(norm.components.len() == 1 && norm.ownership.transform == DenseTransform::Identity,
            "{norm_name}: native final norm recipe");
        let norm = norm.components[0].clone();
        Ok(Self { layers: out, norm, plan })
    }
}

use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES};
use crate::shared::memory::DeviceAllocation;
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
    let allocation = DeviceAllocation::new(library, transform.allocation_bytes(component)?)?;
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

/// Named plain records may land directly into one kernel's packed operand.
/// Total allocator bytes must equal the original separately admitted records;
/// logical bytes are contiguous and no unpacked resident copies are allocated.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PlainPack {
    pub name: String,
    pub components: Vec<String>,
}

fn packed_plain_layout(records: &LayerRecords, pack: &PlainPack) -> Result<(usize, Vec<usize>)> {
    ensure!(!pack.name.is_empty() && !pack.components.is_empty(), "empty native plain pack");
    let mut unique = HashSet::new();
    let (mut logical, mut admitted) = (0usize, 0usize);
    let mut offsets = Vec::new();
    for name in &pack.components {
        ensure!(unique.insert(name), "duplicate native plain pack component {name}");
        let (component, transform) = records.plain.get(name).context("native plain pack component missing")?;
        offsets.push(logical);
        logical = logical.checked_add(transform.resident_bytes(component)?).context("native plain pack extent")?;
        admitted = admitted.checked_add(transform.allocation_bytes(component)?).context("native plain pack admission")?;
    }
    ensure!(allocation(logical) == admitted, "native plain pack changes admitted allocation bytes");
    Ok((admitted, offsets))
}

fn allocate_absorbed<T>(layout: &cuteafd_loader::plan::exlr8::dense::KvBLayout,
    mut allocate: impl FnMut(usize) -> Result<T>) -> Result<[T; 4]> {
    let [uk, uk_scale, uv, uv_scale] = layout.allocations;
    Ok([allocate(allocation(uk))?, allocate(allocation(uk_scale))?,
        allocate(allocation(uv))?, allocate(allocation(uv_scale))?])
}

/// Operand key of a native projection component.
pub(crate) fn key(projection: &str, component: &str) -> String { format!("n.{projection}.{component}") }

/// A layer's native operands on the current device (this rank's slices), keyed
/// [`key`]; `kv_b` split per head for heads `h0..h0 + heads` into the FP8
/// programs' `w_uk_fp8`/`w_uk_scale`/`w_uv_fp8`/`w_uv_scale`.
pub(crate) fn load_layer<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, records: &LayerRecords,
    stream: *mut std::ffi::c_void) -> Result<HashMap<String, DeviceAllocation<'a>>> {
    load_layer_packed(library, load, records, &[], stream)
}

pub(crate) fn load_layer_packed<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, records: &LayerRecords,
    packs: &[PlainPack], stream: *mut std::ffi::c_void) -> Result<HashMap<String, DeviceAllocation<'a>>> {
    let mut consumed = HashSet::new();
    let mut outputs = HashSet::new();
    let layouts = packs.iter().map(|pack| {
        ensure!(outputs.insert(&pack.name) && !records.plain.contains_key(&pack.name)
            && !pack.name.starts_with("n.") && !pack.name.starts_with("w_"), "native plain pack output collision");
        for name in &pack.components { ensure!(consumed.insert(name.as_str()), "native plain record packed twice"); }
        packed_plain_layout(records, pack)
    }).collect::<Result<Vec<_>>>()?;
    let mut out = HashMap::new();
    for (stem, (projection, slice)) in &records.projections {
        ensure!(!matches!(projection.encoding, DenseEncoding::F32 | DenseEncoding::F16 | DenseEncoding::RowFp8),
            "{}: no native dense kernel for {:?}", slice.tensor, projection.encoding);
        for component in &slice.components {
            let name = format!("{}.{}", slice.tensor, component.suffix);
            out.insert(key(stem, &component.suffix), landed(library, load, &name, component, slice.transform, stream)?);
        }
    }
    for (tensor, (component, transform)) in &records.plain {
        if !consumed.contains(tensor.as_str()) {
            out.insert(tensor.clone(), landed(library, load, tensor, component, *transform, stream)?);
        }
    }
    for (pack, (bytes, offsets)) in packs.iter().zip(layouts) {
        let allocation = DeviceAllocation::new(library, bytes)?;
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
        out.insert(pack.name.clone(), allocation);
    }
    // kv_b: whole heads per step, staged rows and the whole grid.
    let Some((kv_b, layout)) = &records.kv_b else { return Ok(out) };
    let geometry = layout.geometry;
    let (heads_total, nope, v, lat) = (geometry.heads, geometry.nope, geometry.values, geometry.latent);
    let (h0, heads) = (layout.first_head, layout.heads);
    ensure!(kv_b.shape == [heads_total * (nope + v), lat] && h0 + heads <= heads_total
        && kv_b.components.len() == 2 && kv_b.components[0].suffix == "fp8" && kv_b.components[1].suffix == "scale",
        "{}: native kv_b layout", kv_b.tensor);
    let (values, grid) = (&kv_b.components[0].source, &kv_b.components[1].source);
    let [uk, uk_scale, uv, uv_scale] = allocate_absorbed(layout, |bytes| DeviceAllocation::new(library, bytes))?;
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
    out.insert("w_uk_fp8".into(), uk);
    out.insert("w_uk_scale".into(), uk_scale);
    out.insert("w_uv_fp8".into(), uv);
    out.insert("w_uv_scale".into(), uv_scale);
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cuteafd_loader::plan::exlr8::dense::{DenseExecution, DenseOwnership, KvBGeometry};

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
        assert_eq!(steps.len(), 2);
        let mut written = 0;
        let mut read = 0;
        for step in steps {
            assert!(step.bytes() <= STEP_BYTES);
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
