//! GLM 5.3 coordinator dense weights from an EXLR8 checkpoint's own records
//! (`dense/`, `exlr8-vocab.safetensors`): every operand is the checkpoint's
//! bytes landing unchanged on its rank (trellis payload and suh/svh, MXFP8
//! bytes and UE8M0 scales, int4 nibbles and F16 group scales, BF16 norms and
//! router), except `kv_b`, which is staged and split per head into the FP8
//! programs' `w_uk`/`w_uv` operands on the GPU. Placement follows the official
//! head split: q_b and gate/up by output rows, o and down by input columns,
//! q_a/kv_a/indexer replicated, router and vocabulary on rank 0.
//!
//! Operand keys are `n.{projection}.{component}` (`n.q_b.trellis`,
//! `n.q_a.e8m0`, ...); the native executor binds them.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::plan::exlr8::dense::{DenseComponent, DenseEncoding, DensePlan, DenseRecord, DenseSlice, ShardSide};
use std::collections::HashMap;

/// The checkpoint projection of each native operand stem.
pub(crate) const PROJECTIONS: [(&str, &str); 11] = [
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
    pub layer: usize,
    pub projections: HashMap<&'static str, (NativeProjection, DenseSlice)>,
    /// `kv_b` (block FP8, whole tensor; split per head at load).
    pub kv_b: DenseRecord,
    /// Plain tensors (norms, router bias) by logical tensor name.
    pub plain: HashMap<String, DenseComponent>,
}

/// The EXLR8 checkpoint's dense records for one rank of `world` coordinators.
pub(crate) struct NativeDenseRecords {
    pub layers: Vec<LayerRecords>,
    pub norm: DenseComponent,
    pub plan: DensePlan,
}

impl NativeDenseRecords {
    pub(crate) fn new(snapshot: &std::path::Path, layers: usize, world: usize, rank: usize) -> Result<Self> {
        let records = DenseRecord::inventory(snapshot)?;
        // The vocabulary owner (#569) holds embedding and head; the official
        // head split decides every other projection's slice here.
        let kept: Vec<DenseRecord> = records.iter().filter(|r| !matches!(r.tensor.as_str(),
            "model.embed_tokens.weight" | "lm_head.weight")).cloned().collect();
        let plan = DensePlan::for_rank(&kept, world, rank)?;
        let by_name: HashMap<&str, &DenseSlice> = plan.tensors.iter().map(|s| (s.tensor.as_str(), s)).collect();
        let record = |name: &str| records.iter().find(|r| r.tensor == name);
        let mut out = Vec::with_capacity(layers);
        for layer in 0..layers {
            let prefix = format!("model.layers.{layer}");
            let mut projections = HashMap::new();
            for (stem, path) in PROJECTIONS {
                let tensor = format!("{prefix}.{path}.weight");
                let Some(full) = record(&tensor) else { continue };
                // Rank-0-only records (the router) have no slice on other ranks.
                let Some(slice) = by_name.get(tensor.as_str()) else {
                    ensure!(rank != 0, "{tensor}: no rank-0 slice");
                    continue;
                };
                let channels = slice.channels;
                let shape = match full.side {
                    ShardSide::Input => [full.shape[0], channels],
                    _ => [channels, full.shape[1]],
                };
                projections.insert(stem, (NativeProjection { encoding: full.encoding, shape, side: full.side,
                    rotation_id: full.rotation_id }, (*slice).clone()));
            }
            let kv_b_name = format!("{prefix}.self_attn.kv_b_proj.weight");
            let kv_b = record(&kv_b_name).with_context(|| format!("{kv_b_name}: missing"))?.clone();
            ensure!(kv_b.encoding == DenseEncoding::BlockFp8, "{kv_b_name}: native GLM kv_b is block FP8");
            let plain = plan.tensors.iter().filter(|s| s.tensor.starts_with(&format!("{prefix}."))
                && s.units.is_empty()).map(|s| (s.tensor.clone(), s.components[0].clone())).collect();
            out.push(LayerRecords { layer, projections, kv_b, plain });
        }
        let norm = plan.tensors.iter().find(|s| s.tensor == "model.norm.weight")
            .map(|s| s.components[0].clone());
        // Rank 1 holds no final norm (rank 0 runs the head).
        let norm = match (norm, rank) {
            (Some(n), _) => n,
            (None, 0) => anyhow::bail!("EXLR8 dense records have no model.norm.weight"),
            (None, _) => records.iter().find(|r| r.tensor == "model.norm.weight").context("final norm")?
                .components[0].clone(),
        };
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

/// The component's bytes on the current device: its own allocation, filled
/// through the mount or recorded for the push group.
fn landed<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, name: &str, component: &DenseComponent)
    -> Result<DeviceAllocation<'a>> {
    let bytes = component.source.bytes;
    let allocation = DeviceAllocation::new(library, bytes.max(256))?;
    load.operand(Operand { device: library.cuda_get_device()?, buffer: allocation.buffer,
        lands: vec![(rect(name, &component.source), 0)], bytes })?;
    Ok(allocation)
}

/// Operand key of a native projection component.
pub(crate) fn key(projection: &str, component: &str) -> String { format!("n.{projection}.{component}") }

/// A layer's native operands on the current device (this rank's slices), keyed
/// [`key`]; `kv_b` split per head for heads `h0..h0 + heads` into the FP8
/// programs' `w_uk_fp8`/`w_uk_scale`/`w_uv_fp8`/`w_uv_scale`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_layer<'a>(library: &'a NativeLibrary, load: &DenseLoad<'a>, records: &LayerRecords,
    heads_total: usize, h0: usize, heads: usize, nope: usize, v: usize, lat: usize, stream: *mut std::ffi::c_void)
    -> Result<HashMap<String, DeviceAllocation<'a>>> {
    let mut out = HashMap::new();
    for (stem, (projection, slice)) in &records.projections {
        ensure!(!matches!(projection.encoding, DenseEncoding::F32 | DenseEncoding::F16 | DenseEncoding::RowFp8),
            "{}: no native GLM dense kernel for {:?}", slice.tensor, projection.encoding);
        for component in &slice.components {
            let name = format!("{}.{}", slice.tensor, component.suffix);
            out.insert(key(stem, &component.suffix), landed(library, load, &name, component)?);
        }
    }
    for (tensor, component) in &records.plain {
        out.insert(tensor.clone(), landed(library, load, tensor, component)?);
    }
    // kv_b: whole heads per step, staged rows and the whole grid.
    let kv_b = &records.kv_b;
    ensure!(kv_b.shape == [heads_total * (nope + v), lat] && h0 + heads <= heads_total
        && kv_b.components.len() == 2 && kv_b.components[0].suffix == "fp8" && kv_b.components[1].suffix == "scale",
        "{}: native kv_b layout", kv_b.tensor);
    let (values, grid) = (&kv_b.components[0].source, &kv_b.components[1].source);
    let alloc = |bytes: usize| DeviceAllocation::new(library, bytes.max(256));
    let (uk, uk_scale) = (alloc(heads * lat * nope)?, alloc(heads * lat * (nope / 64) * 4)?);
    let (uv, uv_scale) = (alloc(heads * v * lat)?, alloc(heads * v * (lat / 64) * 4)?);
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
