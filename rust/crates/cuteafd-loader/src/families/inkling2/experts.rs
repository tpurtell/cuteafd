//! Header-exact routed-expert residency for Inkling 2's three checkpoint
//! storages. The planner charges experts by the checkpoint's own tensor
//! headers: fused `w13_weight`/`w2_weight` MXFP8 or native NVFP4 tensors
//! (`[E, N, K]` with their scale, amax and denominator members), or per-expert
//! EXL3 `gate_proj`/`up_proj`/`down_proj` trellis modules. Whole layers are
//! byte sums; TP2 halves split the intermediate rows of gate/up and the
//! intermediate columns of down over whole 128-row blocks, the shared H128
//! coordinator slicing every RTX expert package uses.
//!
//! This reader is stage-1 planning only. Stage 3 lands `ExpertFormat::Mxfp8`
//! on the shared FP8-family catalog (#1934 S1) and the fused-tensor reader in
//! `Fp8ExpertTensors`; the per-layer byte facts stay identical.
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

use crate::plan::checkpoint::Checkpoint;
use crate::plan::families::inkling2::{expert_stem, expert_storage, ExpertStorage, TEXT};
use super::Inkling2Config;

/// One routed layer's header-exact coordinator residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayerExperts {
    pub storage: Option<ExpertStorage>,
    /// Every expert, whole, on one GPU.
    pub whole: u64,
    /// Each GPU's TP2 half (intermediate rows/columns split over 128-row blocks).
    pub half: [u64; 2],
    /// Checkpoint bytes of the layer's expert tensors (the Spark slice source).
    pub source: u64,
}

/// Every routed layer's residency, in layer order.
pub fn layer_experts(checkpoint: &Checkpoint, cfg: &Inkling2Config) -> Result<Vec<LayerExperts>> {
    (0..cfg.layers).map(|layer| layer_bytes(checkpoint, cfg, layer)).collect()
}

/// The intermediate rows rank `rank` of a TP2 pair owns, over whole 128-row blocks.
fn half_rows(intermediate: usize, rank: usize) -> usize {
    let blocks = intermediate / 128;
    let first = blocks.div_ceil(2);
    128 * if rank == 0 { first } else { blocks - first }
}

fn header<'a>(checkpoint: &'a Checkpoint, name: &str) -> Result<&'a crate::SafetensorsTensorMetadata> {
    checkpoint.header(name).with_context(|| format!("Inkling 2 expert inventory lacks {name}"))
}

fn layer_bytes(checkpoint: &Checkpoint, cfg: &Inkling2Config, layer: usize) -> Result<LayerExperts> {
    ensure!(cfg.intermediate % 256 == 0, "Inkling 2 intermediate {} does not split into TP2 halves of whole 128-row blocks",
        cfg.intermediate);
    let storage = expert_storage(checkpoint, layer)?;
    let (e, h, i) = (cfg.experts as u64, cfg.hidden as u64, cfg.intermediate as u64);
    match storage {
        ExpertStorage::Mxfp8 | ExpertStorage::Nvfp4 => {
            let w13 = expert_stem(layer, "w13");
            let w2 = expert_stem(layer, "w2");
            let weight13 = header(checkpoint, &format!("{w13}_weight"))?;
            let weight2 = header(checkpoint, &format!("{w2}_weight"))?;
            let scale13 = header(checkpoint, &format!("{w13}_weight_scale"))?;
            let scale2 = header(checkpoint, &format!("{w2}_weight_scale"))?;
            let (pack, group, dtype, scale_dtype) = match storage {
                ExpertStorage::Mxfp8 => (1, 32, DType::F8E4M3, DType::U8),
                _ => (2, 16, DType::U8, DType::F8E4M3),
            };
            let expect = |meta: &crate::SafetensorsTensorMetadata, dtype: &DType, shape: [u64; 3]| -> Result<()> {
                ensure!(meta.dtype == *dtype && meta.shape.iter().map(|&d| d as u64).collect::<Vec<_>>() == shape,
                    "{}: expected {dtype:?} {shape:?}, found {:?} {:?}", meta.name, meta.dtype, meta.shape);
                let elements = shape.iter().product::<u64>();
                let bytes = match dtype { DType::F32 => 4, DType::Bf16 => 2, _ => 1 } * elements;
                ensure!(meta.byte_length == bytes, "{}: header byte extent {} differs from {bytes}", meta.name, meta.byte_length);
                Ok(())
            };
            expect(weight13, &dtype, [e, 2 * i, h / pack])?;
            expect(weight2, &dtype, [e, h, i / pack])?;
            expect(scale13, &scale_dtype, [e, 2 * i, h / group])?;
            expect(scale2, &scale_dtype, [e, h, i / group])?;
            let mut extras = 0u64;
            if storage == ExpertStorage::Nvfp4 {
                for stem in [&w13, &w2] {
                    let amax = header(checkpoint, &format!("{stem}_weight_amax"))?;
                    let denominator = header(checkpoint, &format!("{stem}_weight_outer_scale_denominator"))?;
                    expect(amax, &DType::F32, [e, 1, 1]).or_else(|_| -> Result<()> {
                        ensure!(amax.dtype == DType::F32 && amax.shape == [cfg.experts] && amax.byte_length == 4 * e,
                            "{}: expected F32 [{e}], found {:?} {:?}", amax.name, amax.dtype, amax.shape);
                        Ok(())
                    })?;
                    ensure!(denominator.dtype == DType::F32 && denominator.shape.iter().product::<usize>() == 1
                        && denominator.byte_length == 4, "{}: expected one F32 value, found {:?} {:?}",
                        denominator.name, denominator.dtype, denominator.shape);
                    extras += amax.byte_length + denominator.byte_length;
                }
            }
            let source = weight13.byte_length + weight2.byte_length + scale13.byte_length + scale2.byte_length + extras;
            // Per-expert bytes of a slice of `rows` intermediate rows: gate+up
            // rows (2 x rows x H) and down columns (H x rows), codes plus scales.
            let per_expert = |rows: u64| -> u64 {
                let codes = 2 * rows * h / pack + h * rows / pack;
                let scales = 2 * rows * (h / group) + h * (rows / group);
                codes + scales
            };
            let whole = e * per_expert(i) + extras;
            let half = [0usize, 1].map(|rank| e * per_expert(half_rows(cfg.intermediate, rank) as u64) + extras);
            Ok(LayerExperts { storage: Some(storage), whole, half, source })
        }
        ExpertStorage::Exl3 { bits } => {
            // Per-expert MCG trellis: `trellis` I16 [K/16, N/16, 16 bits] plus
            // F16 `suh` [K], `svh` [N] and one marker; headers are authoritative.
            let mut whole = 0u64;
            let mut half = [0u64; 2];
            let mut source = 0u64;
            for expert in 0..cfg.experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    let stem = format!("{TEXT}layers.{layer}.mlp.experts.{expert}.{projection}");
                    let trellis = header(checkpoint, &format!("{stem}.trellis"))?;
                    let (k, n) = if projection == "down_proj" { (i, h) } else { (h, i) };
                    ensure!(trellis.dtype == DType::I16 && trellis.shape.iter().map(|&d| d as u64).collect::<Vec<_>>()
                        == [k / 16, n / 16, 16 * bits as u64], "{}: expected I16 [{}, {}, {}] K{bits} trellis, found {:?} {:?}",
                        trellis.name, k / 16, n / 16, 16 * bits, trellis.dtype, trellis.shape);
                    let suh = header(checkpoint, &format!("{stem}.suh"))?;
                    let svh = header(checkpoint, &format!("{stem}.svh"))?;
                    let module = trellis.byte_length + suh.byte_length + svh.byte_length;
                    source += module + checkpoint.header(&format!("{stem}.mcg")).map_or(0, |m| m.byte_length);
                    whole += module;
                    // A TP2 half keeps the whole hidden axis and half the intermediate
                    // rows (gate/up outputs, down inputs): trellis and the sliced vector.
                    for (rank, bytes) in half.iter_mut().enumerate() {
                        let rows = half_rows(cfg.intermediate, rank) as u64;
                        let share = rows * 2 * bits as u64 * (if projection == "down_proj" { h } else { h }) / 16;
                        let vectors = if projection == "down_proj" { rows * 2 + h * 2 } else { h * 2 + rows * 2 };
                        *bytes += share + vectors;
                    }
                }
            }
            Ok(LayerExperts { storage: Some(storage), whole, half, source })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp2_halves_split_whole_blocks_and_cover_the_intermediate() {
        assert_eq!(half_rows(2304, 0) + half_rows(2304, 1), 2304);
        assert_eq!(half_rows(2304, 0), 1152);
        assert_eq!(half_rows(2304, 0) % 128, 0);
    }
}
